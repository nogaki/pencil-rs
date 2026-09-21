//! Distributed complex-to-complex FFTs behind the `distributed` feature.
//!
//! [`C2cPlan`] construction and transform calls are collective: every rank
//! must use the same Cartesian communicator context, API, order, scalar type,
//! [`TransposeMethod`], and matching layouts. In-place array/workspace
//! allocation and array views are noncollective. Callers must coordinate a
//! local allocation failure before entering the next collective call. The
//! legacy constructors select Alltoallv; the `_with_method` constructors can
//! select the checked point-to-point transport as well. The out-of-place
//! forward transform consumes the canonical input layout and produces the
//! reversed output layout by default; [`DistributedLayout::permute_dims`] can
//! keep every stage in identity memory order and use strided line kernels.
//! [`C2cPlan::inverse`] and [`C2cPlan::backward`] consume the selected output
//! layout in reverse, with `inverse` normalized and `backward` left unnormalized. `AxisSelection` chooses which local stages perform FFTs,
//! but the canonical route still visits every axis in descending order so an
//! unselected stage is an identity rather than a skipped transition. In-place
//! execution uses the same route and state-checks its single buffer.
//!
//! The fixed C2C descriptor includes the selected transport before native FFT,
//! output, workspace, or in-place state changes. Descriptor and initial
//! preflight errors are collectively reported before any source, destination,
//! or workspace write. After execution begins, the selected checked transition
//! may prepare metadata and return a collectively agreed preparation or
//! allocation error; out-of-place sources remain preserved, while in-place
//! state is poisoned before the first write and workspace contents are not
//! transactional on that path. Point-to-point request metadata is reserved by
//! its existing transport implementation; its fixed context/tag and
//! no-overlap MPI failure contract are inherited rather than duplicated here.
//! Out-of-place overlap rejects Alltoallv transitions; identity and local
//! transitions run synchronously, while point-to-point transitions run the
//! next local stage after receive/unpack and before send completion. Partial
//! final sends are completed before return, so callers must not assume pending
//! network work.
//!
//! [`R2cPlan`] reuses the same private route and transition machinery for an
//! original real endpoint followed by homogeneous complex tail stages. Its
//! reduced output shape and post-tail constrained-plane validation are
//! implemented in the child module, which also provides a single-allocation
//! real in-place API.
//!
//! [`R2rPlan`] keeps the original shape for real or complex DCT/DST data. Its
//! public constructors accept only the legacy FFTW kind array; Hartley and
//! mixed-axis construction stays separate from this legacy API and
//! is handled by the private sibling core used by [`DhtPlan`].

#![allow(clippy::too_many_arguments)]

use std::{
    borrow::Borrow,
    mem::size_of,
    sync::Arc,
    time::{Duration, Instant},
};

use mpi::{
    Count,
    collective::{CommunicatorCollectives, SystemOperation},
    datatype::Equivalence,
};
use pencil_array::{
    AllToAllvTransposePlan, ArrayError, AxisPermutation, ExtraShape, LocalTransposeError,
    LocalTransposePlan, ManyPencilArray, MpiTopology, OverlapError, OverwriteError, Pencil,
    PencilArray, PencilArrayView, PencilArrayViewMut, PencilConfig, PencilError,
    PointToPointTransposePlan, SpatialAxis, TransposeError, TransposeWorkspace,
    TransposeWorkspaceRequirements,
};
use thiserror::Error;

#[cfg(test)]
#[derive(Clone, Copy)]
enum C2cCallbackInjection {
    Error,
    Panic,
}

#[cfg(test)]
#[derive(Clone, Copy)]
enum R2cCallbackInjection {
    Error,
    Panic,
}

#[cfg(test)]
thread_local! {
    static C2C_CALLBACK_INJECTION: std::cell::Cell<Option<C2cCallbackInjection>> =
        const { std::cell::Cell::new(None) };
    static R2C_CALLBACK_INJECTION: std::cell::Cell<Option<R2cCallbackInjection>> =
        const { std::cell::Cell::new(None) };
}

#[cfg(test)]
fn consume_c2c_callback_injection() -> Result<(), FftError> {
    let injection = C2C_CALLBACK_INJECTION.with(|slot| slot.take());
    match injection {
        Some(C2cCallbackInjection::Error) => Err(FftError::PreparationFailed),
        Some(C2cCallbackInjection::Panic) => panic!("injected C2C overlap callback panic"),
        None => Ok(()),
    }
}

#[cfg(test)]
fn consume_r2c_callback_injection() -> Result<(), FftError> {
    let injection = R2C_CALLBACK_INJECTION.with(|slot| slot.take());
    match injection {
        Some(R2cCallbackInjection::Error) => {
            Err(FftError::LocalC2c(LocalC2cError::ScratchTooSmall {
                required: 7,
                actual: 3,
            }))
        }
        Some(R2cCallbackInjection::Panic) => panic!("injected R2C overlap callback panic"),
        None => Ok(()),
    }
}

use crate::{
    Complex, FftReal, LocalC2cError, LocalC2cPlan, LocalR2cError, LocalR2cPlan, LocalR2rError,
};

const DESCRIPTOR_SCHEMA: u64 = 3;
const OPERATION_PLAN: u64 = 7;
const OPERATION_FORWARD: u64 = 8;
const OPERATION_INVERSE: u64 = 9;
const OPERATION_FORWARD_IN_PLACE: u64 = 10;
const OPERATION_INVERSE_IN_PLACE: u64 = 11;
const OPERATION_R2C_PLAN: u64 = 12;
const OPERATION_R2C_FORWARD: u64 = 13;
const OPERATION_R2C_INVERSE: u64 = 14;
const OPERATION_BACKWARD: u64 = 15;
const OPERATION_BACKWARD_IN_PLACE: u64 = 16;
const OPERATION_R2C_BACKWARD: u64 = 17;
const OPERATION_R2R_PLAN: u64 = 18;
const OPERATION_R2R_FORWARD: u64 = 19;
const OPERATION_R2R_INVERSE: u64 = 20;
const OPERATION_R2R_BACKWARD: u64 = 21;
const OPERATION_R2R_FORWARD_IN_PLACE: u64 = 22;
const OPERATION_R2R_INVERSE_IN_PLACE: u64 = 23;
const OPERATION_R2R_BACKWARD_IN_PLACE: u64 = 24;
// Operation words 18..24 are reserved for the parallel distributed R2R API.
const OPERATION_R2C_FORWARD_IN_PLACE: u64 = 25;
const OPERATION_R2C_INVERSE_IN_PLACE: u64 = 26;
const OPERATION_R2C_BACKWARD_IN_PLACE: u64 = 27;
const OPERATION_DHT_PLAN: u64 = 28;
const OPERATION_DHT_FORWARD: u64 = 29;
const OPERATION_DHT_INVERSE: u64 = 30;
const OPERATION_DHT_BACKWARD: u64 = 31;
const OPERATION_DHT_FORWARD_IN_PLACE: u64 = 32;
const OPERATION_DHT_INVERSE_IN_PLACE: u64 = 33;
const OPERATION_DHT_BACKWARD_IN_PLACE: u64 = 34;
const OPERATION_MIXED_C2C_PLAN: u64 = 35;
const OPERATION_MIXED_C2C_FORWARD: u64 = 36;
const OPERATION_MIXED_C2C_INVERSE: u64 = 37;
const OPERATION_MIXED_C2C_BACKWARD: u64 = 38;
const OPERATION_MIXED_C2C_FORWARD_IN_PLACE: u64 = 39;
const OPERATION_MIXED_C2C_INVERSE_IN_PLACE: u64 = 40;
const OPERATION_MIXED_C2C_BACKWARD_IN_PLACE: u64 = 41;
const OPERATION_MIXED_R2C_PLAN: u64 = 42;
const OPERATION_MIXED_R2C_FORWARD: u64 = 43;
const OPERATION_MIXED_R2C_INVERSE: u64 = 44;
const OPERATION_MIXED_R2C_BACKWARD: u64 = 45;
const OPERATION_MIXED_R2C_FORWARD_IN_PLACE: u64 = 46;
const OPERATION_MIXED_R2C_INVERSE_IN_PLACE: u64 = 47;
const OPERATION_MIXED_R2C_BACKWARD_IN_PLACE: u64 = 48;
// Profiled calls use distinct collective operation words; the fixed five-word
// descriptor header therefore rejects old/profiled mismatches before payload.
const OPERATION_FORWARD_TIMED: u64 = 49;
const OPERATION_INVERSE_TIMED: u64 = 50;
const OPERATION_BACKWARD_TIMED: u64 = 51;
const OPERATION_FORWARD_IN_PLACE_TIMED: u64 = 52;
const OPERATION_INVERSE_IN_PLACE_TIMED: u64 = 53;
const OPERATION_BACKWARD_IN_PLACE_TIMED: u64 = 54;
const OPERATION_FORWARD_OVERLAP: u64 = 55;
const OPERATION_INVERSE_OVERLAP: u64 = 56;
const OPERATION_BACKWARD_OVERLAP: u64 = 57;
const INVALID_WORD: u64 = u64::MAX;
const METHOD_ALL_TO_ALLV: u64 = 0;
const METHOD_POINT_TO_POINT: u64 = 1;
const VALUE_KIND_C2C: u64 = 1;
const VALUE_KIND_R2C: u64 = 2;
const HEADER_WORDS: usize = 5;

/// Timing for one fixed route stage.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StageTiming {
    /// Time spent in the local transform.
    pub fft: Duration,
    /// Time spent in the following route transition.
    pub transpose: Duration,
    /// Number of local FFT phases included in `fft`.
    pub fft_calls: u64,
    /// Number of route transitions included in `transpose`.
    pub transition_calls: u64,
    /// Pack, unpack, and transport waits for this stage's transition.
    /// Unused phases (including phases of local-only transitions) are zero.
    pub communication: pencil_array::TransposeTiming,
    /// Sum of this stage's measured work (`fft + transpose`).
    pub total: Duration,
}

/// Fixed-size timing returned by profiled distributed transforms.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransformTiming<const N: usize> {
    /// One entry for each logical stage index (`0..N`), in route order;
    /// identity stages are included.
    pub stages: [StageTiming; N],
    /// Wall-clock duration of the complete call.
    pub total: Duration,
}

impl<const N: usize> Default for TransformTiming<N> {
    fn default() -> Self {
        Self {
            stages: [StageTiming::default(); N],
            total: Duration::ZERO,
        }
    }
}

/// Completion state of a distributed C2C in-place array.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum C2cState {
    /// The active buffer contains canonical spatial input data.
    Input,
    /// The active buffer contains the plan's output-layout spectral data.
    Output,
    /// An in-place execution started but did not complete.
    Poisoned,
}

/// Errors returned by the feature-gated distributed FFT APIs.
#[derive(Debug, Error)]
pub enum FftError {
    /// `N` must be at least two and `M` must satisfy `1 <= M < N`.
    #[error("distributed FFT requires N >= 2 and 1 <= M < N")]
    InvalidDimensions,

    /// The input pencil was not the canonical identity layout.
    #[error("input pencil must use identity permutation and decomposition [0..M)")]
    InvalidInputLayout,

    /// The supplied source array does not match the plan's input layout.
    #[error("source layout does not match the distributed FFT plan")]
    InputLayoutMismatch,

    /// The supplied destination array does not match the plan's output layout.
    #[error("destination layout does not match the distributed FFT plan")]
    OutputLayoutMismatch,

    /// An array's extra shape does not match the plan or its peer array.
    #[error("extra shape does not match the distributed FFT plan")]
    ExtraShapeMismatch,

    /// The workspace was created for another plan or has invalid registered layouts.
    #[error("workspace does not belong to this distributed FFT plan")]
    WorkspaceMismatch,

    /// A workspace prefix or native FFT scratch slice is too short.
    #[error("{kind} workspace length {actual} is less than required {required}")]
    WorkspaceTooSmall {
        /// The checked resource name.
        kind: &'static str,
        /// The required initialized length.
        required: usize,
        /// The supplied initialized length.
        actual: usize,
    },

    /// Fixed headers or exact descriptors differed between ranks.
    #[error("distributed FFT collective descriptors differ between ranks")]
    CollectiveDescriptorMismatch,

    /// At least one rank rejected a collective precondition.
    #[error("a distributed FFT collective precondition failed on another rank")]
    CollectivePreconditionFailed,

    /// Checked route or descriptor preparation failed.
    #[error("distributed FFT preparation failed")]
    PreparationFailed,

    /// A requested initialized allocation could not be made.
    #[error("failed to allocate {required} elements")]
    AllocationFailed {
        /// The requested element count.
        required: usize,
    },

    /// The single in-place allocation cannot be safely recast for its next phase.
    #[error("distributed in-place storage has an incompatible scalar layout")]
    StorageLayoutMismatch,

    /// The local native FFT plan or operation rejected checked input.
    #[error(transparent)]
    LocalC2c(#[from] LocalC2cError),

    /// The array crate rejected a pencil construction or topology-derived layout.
    #[error(transparent)]
    Pencil(#[from] PencilError),

    /// The array crate rejected local storage or a shared-array state.
    #[error(transparent)]
    Array(#[from] ArrayError),

    /// The array crate rejected a distributed transpose transition.
    #[error(transparent)]
    Transpose(#[from] TransposeError),

    /// The array crate rejected a process-local transition.
    #[error(transparent)]
    LocalTranspose(#[from] LocalTransposeError),
}

/// Errors returned only by the out-of-place `*_with_overlap` APIs.
#[derive(Debug, Error)]
pub enum FftOverlapError<E: std::fmt::Debug> {
    /// The base transform operation failed validation or execution.
    #[error(transparent)]
    Operation(E),
    /// Alltoallv transitions do not support overlap.
    #[error("distributed FFT overlap does not support Alltoallv transpose")]
    UnsupportedTransport,
    /// A transpose or local callback failed, or a peer callback failed or panicked.
    #[error(transparent)]
    Overlap(OverlapError<E>),
}

impl<E: std::fmt::Debug> From<E> for FftOverlapError<E> {
    fn from(error: E) -> Self {
        Self::Operation(error)
    }
}

/// Errors returned by the distributed real-to-half-complex API.
#[derive(Debug, Error)]
pub enum R2cError {
    /// An existing distributed FFT or array/transpose validation failed.
    #[error(transparent)]
    Fft(#[from] FftError),

    /// The local real FFT plan or operation rejected checked input.
    #[error(transparent)]
    LocalR2c(#[from] LocalR2cError),

    /// A constrained boundary plane was not sufficiently real after the
    /// transverse inverse stages.
    #[error("distributed inverse spectrum has an invalid constrained boundary plane")]
    InvalidSpectrum,
}

/// Errors returned by the distributed DCT/DST/DHT real-to-real APIs.
#[derive(Debug, Error)]
pub enum R2rError {
    /// An existing distributed FFT or array/transpose validation failed.
    #[error(transparent)]
    Fft(#[from] FftError),

    /// The local DCT/DST plan or operation rejected checked input.
    #[error(transparent)]
    LocalR2r(#[from] LocalR2rError),
}

pub mod mixed;
mod r2c;
mod r2r;

#[cfg(test)]
pub(crate) static MPI_TEST_LOCK: std::sync::OnceLock<std::sync::Mutex<()>> =
    std::sync::OnceLock::new();

pub use r2c::{R2cInPlaceArray, R2cInPlaceWorkspace, R2cPlan, R2cWorkspace};
pub use r2r::{DhtPlan, R2rInPlaceArray, R2rInPlaceWorkspace, R2rPlan, R2rWorkspace};

/// Selects the distributed transition transport used by [`C2cPlan`], [`R2cPlan`], and [`R2rPlan`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransposeMethod {
    /// Use one checked `MPI_Alltoallv` for each distributed transition.
    AllToAllv,
    /// Use the checked receive-before-send point-to-point transport.
    PointToPoint,
}

impl TransposeMethod {
    fn descriptor_word(self) -> u64 {
        match self {
            Self::AllToAllv => METHOD_ALL_TO_ALLV,
            Self::PointToPoint => METHOD_POINT_TO_POINT,
        }
    }
}

/// The transport and memory-layout policy for a distributed transform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DistributedLayout {
    /// The checked MPI transport used for distributed transitions.
    pub transpose_method: TransposeMethod,
    /// Whether each canonical route stage moves its transform axis to the
    /// last memory dimension. `false` keeps every stage in identity memory
    /// order and uses strided local line kernels where required.
    pub permute_dims: bool,
}

/// The sign convention configured for a Fourier axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FourierDirection {
    /// The negative-exponent (forward) sign.
    Forward,
    /// The positive-exponent (backward) sign.
    Backward,
}

/// Per-axis Fourier sign configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FourierDirections<const N: usize>([FourierDirection; N]);

impl<const N: usize> FourierDirections<N> {
    /// Configures every axis with the forward sign.
    pub fn forward() -> Self {
        Self([FourierDirection::Forward; N])
    }

    /// Creates an explicit per-axis configuration.
    pub const fn new(directions: [FourierDirection; N]) -> Self {
        Self(directions)
    }

    /// Returns the configured sign for an axis.
    pub fn get(&self, axis: usize) -> Option<FourierDirection> {
        self.0.get(axis).copied()
    }
}

impl<const N: usize> Default for FourierDirections<N> {
    fn default() -> Self {
        Self::forward()
    }
}

/// Immutable geometry of one planned distributed FFT stage.
#[derive(Clone, Debug)]
pub struct StageGeometry<const N: usize, const M: usize> {
    /// Logical transform axis.
    pub axis: usize,
    /// Immutable source pencil.
    pub source: Arc<Pencil<N, M>>,
    /// Immutable output pencil.
    pub output: Arc<Pencil<N, M>>,
}

impl Default for DistributedLayout {
    fn default() -> Self {
        Self {
            transpose_method: TransposeMethod::AllToAllv,
            permute_dims: true,
        }
    }
}

/// A validated set of spatial axes to transform.
///
/// The selection stores a private boolean mask. Indices supplied to
/// [`Self::from_indices`] are only validation input; execution always uses the
/// canonical descending route, not caller order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AxisSelection<const N: usize> {
    mask: [bool; N],
}

/// An invalid axis-selection index list.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum AxisSelectionError {
    /// An index is outside `0..N`.
    #[error("selected axis {axis} is out of bounds for N={dimensions}")]
    OutOfBounds {
        /// The invalid axis index.
        axis: usize,
        /// The number of spatial axes.
        dimensions: usize,
    },
    /// An index occurs more than once.
    #[error("selected axis {axis} occurs more than once")]
    Duplicate {
        /// The repeated axis index.
        axis: usize,
    },
}

impl<const N: usize> AxisSelection<N> {
    /// Selects every spatial axis.
    pub fn all() -> Self {
        Self { mask: [true; N] }
    }

    /// Selects no spatial axes.
    pub fn empty() -> Self {
        Self { mask: [false; N] }
    }

    /// Validates and creates a selection from axis indices.
    ///
    /// Duplicate and out-of-bounds indices are rejected. The input order does
    /// not affect the resulting selection or transform route.
    pub fn from_indices<I>(indices: I) -> Result<Self, AxisSelectionError>
    where
        I: IntoIterator,
        I::Item: Borrow<usize>,
    {
        let mut mask = [false; N];
        for axis in indices {
            let axis = *axis.borrow();
            if axis >= N {
                return Err(AxisSelectionError::OutOfBounds {
                    axis,
                    dimensions: N,
                });
            }
            if mask[axis] {
                return Err(AxisSelectionError::Duplicate { axis });
            }
            mask[axis] = true;
        }
        Ok(Self { mask })
    }

    /// Returns whether `axis` is selected.
    pub fn contains(&self, axis: usize) -> bool {
        self.mask.get(axis).copied().unwrap_or(false)
    }

    /// Returns whether this selection is empty.
    pub fn is_empty(&self) -> bool {
        !self.mask.iter().any(|&selected| selected)
    }

    /// Returns the number of selected axes.
    pub fn len(&self) -> usize {
        self.mask.iter().filter(|&&selected| selected).count()
    }

    /// Returns whether this selection contains every spatial axis.
    pub fn is_all(&self) -> bool {
        self.mask.iter().all(|&selected| selected)
    }

    pub(crate) fn mask(&self) -> &[bool; N] {
        &self.mask
    }
}

/// An immutable, checked distributed complex-to-complex FFT plan.
///
/// Construction and transform calls are collective on the topology's
/// Cartesian communicator. Every rank must use the same communicator context,
/// API, order, scalar type, and [`TransposeMethod`]. The legacy
/// `from_pencil`, `from_array`, and `from_shape` constructors select
/// [`TransposeMethod::AllToAllv`]; their `_with_method` counterparts select
/// either supported transport. The selected method is part of the exact
/// collective descriptor, so a rank-local method choice is rejected before
/// native planning or execution state changes. Point-to-point transitions use
/// the same topology-owned context for their changed axis and the fixed
/// internal `0x5054` tag; do not overlap unfinished transposes on that
/// context. A successful call returns only after all native MPI requests have
/// completed. MPI failures, arbitrary panics, and process loss do not promise
/// global recovery, and an unfinished request scope may abort. In-place
/// array/workspace allocation and views are noncollective; callers must
/// coordinate an allocation failure before the next collective call. The
/// input must use the identity permutation and decomposition `[0, ..., M)`;
/// this feature supports `N >= 2` and `1 <= M < N`. A validated
/// [`AxisSelection`] controls which axes are transformed. The route always
/// contains all `N` stages (`N - 1` transitions), from axis `N - 1` down to
/// `0`; unselected stages are identities with no native FFT or scaling. The
/// derived output uses decomposition `[1, ..., M]`; its spatial memory order
/// is reversed when `layout.permute_dims` is true and identity otherwise,
/// including for an empty selection. The [`Self::forward`],
/// [`Self::inverse`], and [`Self::backward`] methods are
/// out of place and input preserving; their source is respectively the
/// canonical input and the selected output layout. `inverse` is normalized,
/// while `backward` is the positive-sign raw transform and scales
/// a forward result by the product of the selected spatial extents. Identity
/// axes do not contribute. The [`Self::forward_in_place`], [`Self::inverse_in_place`], and
/// [`Self::backward_in_place`] methods use one state-checked buffer.
///
/// # Example
///
/// Initialize MPI once and drop all topology, plan, array, and workspace
/// values before the MPI universe.
///
/// ```
/// use mpi::traits::*;
/// use pencil_array::{ExtraShape, MpiTopology};
/// use pencil_fft::{C2cPlan, Complex, TransposeMethod};
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let universe = mpi::initialize().expect("MPI must not already be initialized");
///     let world = universe.world();
///     let result = {
///         let world_size = usize::try_from(world.size())?;
///         let topology = MpiTopology::<1>::new(&world, [world_size])?;
///         let plan = C2cPlan::<f64, 2, 1>::from_shape_with_method(
///             topology,
///             [2 * world_size, 3],
///             ExtraShape::scalar(),
///             TransposeMethod::PointToPoint,
///         )?;
///         let mut input = plan.allocate_input()?;
///         for (index, value) in input.as_mut_slice().iter_mut().enumerate() {
///             *value = Complex::new(index as f64 + 1.0, -(index as f64));
///         }
///         let original = input.as_slice().to_vec();
///         let mut transformed = plan.allocate_output()?;
///         let mut recovered = plan.allocate_input()?;
///         let mut workspace = plan.allocate_out_of_place_workspace()?;
///
///         plan.forward(&input, &mut transformed, &mut workspace)?;
///         plan.inverse(&transformed, &mut recovered, &mut workspace)?;
///         assert_eq!(input.as_slice(), original.as_slice());
///         for (actual, expected) in recovered.as_slice().iter().zip(&original) {
///             assert!((actual.re - expected.re).abs() < 1e-9);
///             assert!((actual.im - expected.im).abs() < 1e-9);
///         }
///         let mut raw = plan.allocate_input()?;
///         plan.backward(&transformed, &mut raw, &mut workspace)?;
///         let scale = (2 * world_size * 3) as f64;
///         for (actual, expected) in raw.as_slice().iter().zip(&original) {
///             assert!((actual.re - expected.re * scale).abs() < 1e-7 * scale);
///             assert!((actual.im - expected.im * scale).abs() < 1e-7 * scale);
///         }
///         Ok::<(), Box<dyn std::error::Error>>(())
///     };
///     result
/// }
/// ```
#[derive(Debug)]
pub struct C2cPlan<R: FftReal, const N: usize, const M: usize> {
    core: Arc<TransformPlanCore<R, N, M>>,
}

/// Reusable storage for [`C2cPlan`] out-of-place execution.
///
/// The workspace is private to the exact plan that allocated it. It contains
/// one registered [`ManyPencilArray`], shared checked transpose buffers, and
/// native FFT scratch. Constructing multiple workspaces from one plan is
/// supported; execution mutates only the selected workspace and destination.
/// A point-to-point transition also reserves its request metadata during the
/// call; execution is not promised to be allocation-free.
#[derive(Debug)]
pub struct C2cOutOfPlaceWorkspace<R: FftReal, const N: usize, const M: usize> {
    core: Arc<TransformPlanCore<R, N, M>>,
    intermediate: ManyPencilArray<Complex<R>, N, M>,
    transpose: TransposeWorkspace<Complex<R>>,
    fft_scratch: Vec<Complex<R>>,
    /// Initialized line storage used only by non-unit-stride stages.
    line_buffer: Vec<Complex<R>>,
}

/// An opaque single-buffer array for distributed C2C in-place execution.
///
/// The active view is available only in [`C2cState::Input`] or
/// [`C2cState::Output`]. An execution changes the state to
/// [`C2cState::Poisoned`] before its first write and commits the target state
/// only after every local FFT and transition succeeds. Reallocate after a
/// poisoned execution; this initial API has no recovery operation.
///
/// # Example
///
/// ```
/// use mpi::traits::*;
/// use pencil_array::{ExtraShape, MpiTopology};
/// use pencil_fft::{C2cPlan, C2cState, Complex};
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let universe = mpi::initialize().expect("MPI must not already be initialized");
///     let world = universe.world();
///     let result = {
///         let world_size = usize::try_from(world.size())?;
///         let topology = MpiTopology::<1>::new(&world, [world_size])?;
///         let plan = C2cPlan::<f64, 2, 1>::from_shape(
///             topology,
///             [2 * world_size, 3],
///             ExtraShape::scalar(),
///         )?;
///         let mut array = plan.allocate_in_place()?;
///         {
///             let mut input = array.view_mut()?;
///             for (index, value) in input.as_mut_slice().iter_mut().enumerate() {
///                 *value = Complex::new(index as f64 + 1.0, -(index as f64));
///             }
///         }
///         let original = array.view()?.as_slice().to_vec();
///         let mut workspace = plan.allocate_in_place_workspace()?;
///         plan.forward_in_place(&mut array, &mut workspace)?;
///         assert_eq!(array.state(), C2cState::Output);
///         plan.inverse_in_place(&mut array, &mut workspace)?;
///         assert_eq!(array.state(), C2cState::Input);
///         for (actual, expected) in array.view()?.as_slice().iter().zip(&original) {
///             assert!((actual.re - expected.re).abs() < 1e-9);
///             assert!((actual.im - expected.im).abs() < 1e-9);
///         }
///         Ok::<(), Box<dyn std::error::Error>>(())
///     };
///     result
/// }
/// ```
///
/// Both [`C2cPlan::inverse_in_place`] and [`C2cPlan::backward_in_place`]
/// consume `Output` and return `Input`; only the former applies normalization.
/// The backing `ManyPencilArray` and raw slices are intentionally private:
///
/// ```compile_fail
/// use pencil_fft::{C2cInPlaceArray, FftReal};
///
/// fn no_raw_storage<R: FftReal, const N: usize, const M: usize>(
///     array: &mut C2cInPlaceArray<R, N, M>,
/// ) {
///     let _ = &array.array;
/// }
/// ```
#[derive(Debug)]
pub struct C2cInPlaceArray<R: FftReal, const N: usize, const M: usize> {
    core: Arc<TransformPlanCore<R, N, M>>,
    array: ManyPencilArray<Complex<R>, N, M>,
    state: C2cState,
}

/// Reusable scratch for distributed C2C in-place execution.
///
/// The workspace is private to the exact plan that allocated it. It contains
/// only shared checked transpose buffers and native FFT scratch; the transform
/// data is owned by [`C2cInPlaceArray`].
#[derive(Debug)]
pub struct C2cInPlaceWorkspace<R: FftReal, const N: usize, const M: usize> {
    core: Arc<TransformPlanCore<R, N, M>>,
    transpose: TransposeWorkspace<Complex<R>>,
    fft_scratch: Vec<Complex<R>>,
    line_buffer: Vec<Complex<R>>,
}

/// The distributed R2R in-place API reuses the C2C completion states.
pub type R2rState = C2cState;

#[derive(Debug)]
enum LocalTransform<R: FftReal> {
    Identity,
    Complex(LocalC2cPlan<R>),
    RealComplex(LocalR2cPlan<R>),
}

impl<R: FftReal> LocalTransform<R> {
    fn real_complex(&self) -> &LocalR2cPlan<R> {
        match self {
            Self::RealComplex(plan) => plan,
            Self::Identity => panic!("real stage requested from an identity stage"),
            Self::Complex(_) => panic!("real stage requested from a complex stage"),
        }
    }

    fn scratch_len(&self) -> usize {
        match self {
            Self::Identity => 0,
            Self::Complex(plan) => plan.scratch_len(),
            Self::RealComplex(plan) => plan.scratch_len(),
        }
    }
}

#[derive(Debug)]
struct TransformStage<R: FftReal, const N: usize, const M: usize> {
    axis: usize,
    input: Arc<Pencil<N, M>>,
    output: Arc<Pencil<N, M>>,
    local: LocalTransform<R>,
}

#[derive(Debug)]
enum C2cTransition<const N: usize, const M: usize> {
    /// The two adjacent stages already have the same registered layout.
    Identity,
    Local(LocalTransposePlan<N, M>),
    AllToAllv(AllToAllvTransposePlan<N, M>),
    PointToPoint(PointToPointTransposePlan<N, M>),
}

#[derive(Debug)]
struct C2cStageTransition<const N: usize, const M: usize> {
    forward: C2cTransition<N, M>,
    backward: C2cTransition<N, M>,
}

#[derive(Debug)]
struct TransformPlanCore<R: FftReal, const N: usize, const M: usize> {
    stages: Box<[TransformStage<R, N, M>]>,
    transitions: Box<[C2cStageTransition<N, M>]>,
    extra_shape: ExtraShape,
    selection: AxisSelection<N>,
    layout: DistributedLayout,
    real_stage_index: Option<usize>,
    descriptor: Box<[u64]>,
    fft_scratch_len: usize,
    /// The largest local complex line needed by a non-unit-stride stage.
    /// This is zero for the legacy permuted route.
    strided_line_len: usize,
    transpose_send_len: usize,
    transpose_receive_len: usize,
    real_transpose_send_len: usize,
    real_transpose_receive_len: usize,
    directions: FourierDirections<N>,
    // Reconfigured plans reject arrays belonging to the prior core; legacy
    // constructors retain their existing layout-compatible array contract.
    strict_array_identity: bool,
}

#[derive(Debug)]
struct RouteCandidate<const N: usize, const M: usize> {
    stages: Box<[Arc<Pencil<N, M>>]>,
    distributed: Box<[bool]>,
}

#[derive(Debug)]
struct StagePreparation<R: FftReal, const N: usize, const M: usize> {
    stages: Box<[TransformStage<R, N, M>]>,
    fft_scratch_len: usize,
}

#[derive(Clone, Copy, Debug)]
enum Direction {
    Forward,
    Inverse,
    Backward,
}

impl<R: FftReal, const N: usize, const M: usize> C2cPlan<R, N, M>
where
    Complex<R>: Equivalence,
{
    /// Collectively builds an Alltoallv plan from a canonical input pencil.
    pub fn from_pencil(
        input: Arc<Pencil<N, M>>,
        extra_shape: ExtraShape,
    ) -> Result<Self, FftError> {
        Self::from_pencil_with_selection_and_layout(
            input,
            extra_shape,
            AxisSelection::all(),
            DistributedLayout::default(),
        )
    }

    /// Collectively builds an Alltoallv plan for a validated axis selection.
    pub fn from_pencil_with_selection(
        input: Arc<Pencil<N, M>>,
        extra_shape: ExtraShape,
        selection: AxisSelection<N>,
    ) -> Result<Self, FftError> {
        Self::from_pencil_with_selection_and_layout(
            input,
            extra_shape,
            selection,
            DistributedLayout::default(),
        )
    }

    /// Collectively builds a plan from a canonical input pencil and selection.
    pub fn from_pencil_with_selection_and_method(
        input: Arc<Pencil<N, M>>,
        extra_shape: ExtraShape,
        selection: AxisSelection<N>,
        method: TransposeMethod,
    ) -> Result<Self, FftError> {
        Self::from_pencil_with_selection_and_layout(
            input,
            extra_shape,
            selection,
            DistributedLayout {
                transpose_method: method,
                ..DistributedLayout::default()
            },
        )
    }

    /// Collectively builds a plan with an explicit transport and layout policy.
    pub fn from_pencil_with_layout(
        input: Arc<Pencil<N, M>>,
        extra_shape: ExtraShape,
        layout: DistributedLayout,
    ) -> Result<Self, FftError> {
        Self::from_pencil_with_selection_and_layout(
            input,
            extra_shape,
            AxisSelection::all(),
            layout,
        )
    }

    /// Collectively builds a plan for a validated selection and layout policy.
    pub fn from_pencil_with_selection_and_layout(
        input: Arc<Pencil<N, M>>,
        extra_shape: ExtraShape,
        selection: AxisSelection<N>,
        layout: DistributedLayout,
    ) -> Result<Self, FftError> {
        let topology = Arc::clone(input.topology());
        let global_shape = *input.global_shape();
        Self::construct(
            topology,
            global_shape,
            extra_shape,
            Ok(input),
            selection,
            layout,
            FourierDirections::default(),
        )
    }

    /// Collectively builds a plan from a canonical input pencil and exact extra shape.
    pub fn from_pencil_with_method(
        input: Arc<Pencil<N, M>>,
        extra_shape: ExtraShape,
        method: TransposeMethod,
    ) -> Result<Self, FftError> {
        Self::from_pencil_with_selection_and_layout(
            input,
            extra_shape,
            AxisSelection::all(),
            DistributedLayout {
                transpose_method: method,
                ..DistributedLayout::default()
            },
        )
    }

    /// Collectively builds an Alltoallv plan from a canonical input array's layout.
    pub fn from_array(input: &PencilArray<Complex<R>, N, M>) -> Result<Self, FftError> {
        Self::from_array_with_selection_and_layout(
            input,
            AxisSelection::all(),
            DistributedLayout::default(),
        )
    }

    /// Collectively builds an Alltoallv plan for a validated axis selection.
    pub fn from_array_with_selection(
        input: &PencilArray<Complex<R>, N, M>,
        selection: AxisSelection<N>,
    ) -> Result<Self, FftError> {
        Self::from_array_with_selection_and_layout(input, selection, DistributedLayout::default())
    }

    /// Collectively builds a plan from a canonical input array's layout.
    pub fn from_array_with_selection_and_method(
        input: &PencilArray<Complex<R>, N, M>,
        selection: AxisSelection<N>,
        method: TransposeMethod,
    ) -> Result<Self, FftError> {
        Self::from_array_with_selection_and_layout(
            input,
            selection,
            DistributedLayout {
                transpose_method: method,
                ..DistributedLayout::default()
            },
        )
    }

    /// Collectively builds a plan with an explicit transport and layout policy.
    pub fn from_array_with_layout(
        input: &PencilArray<Complex<R>, N, M>,
        layout: DistributedLayout,
    ) -> Result<Self, FftError> {
        Self::from_array_with_selection_and_layout(input, AxisSelection::all(), layout)
    }

    /// Collectively builds a plan for a validated selection and layout policy.
    pub fn from_array_with_selection_and_layout(
        input: &PencilArray<Complex<R>, N, M>,
        selection: AxisSelection<N>,
        layout: DistributedLayout,
    ) -> Result<Self, FftError> {
        let topology = Arc::clone(input.pencil().topology());
        let global_shape = *input.pencil().global_shape();
        Self::construct(
            topology,
            global_shape,
            input.extra_shape().clone(),
            Ok(Arc::clone(input.pencil())),
            selection,
            layout,
            FourierDirections::default(),
        )
    }

    /// Collectively builds a plan from a canonical input array's layout.
    pub fn from_array_with_method(
        input: &PencilArray<Complex<R>, N, M>,
        method: TransposeMethod,
    ) -> Result<Self, FftError> {
        Self::from_array_with_selection_and_layout(
            input,
            AxisSelection::all(),
            DistributedLayout {
                transpose_method: method,
                ..DistributedLayout::default()
            },
        )
    }

    /// Collectively builds an Alltoallv plan from topology, global shape, and extra shape.
    pub fn from_shape(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
    ) -> Result<Self, FftError> {
        Self::from_shape_with_selection_and_layout(
            topology,
            global_shape,
            extra_shape,
            AxisSelection::all(),
            DistributedLayout::default(),
        )
    }

    /// Collectively builds an Alltoallv plan for a validated axis selection.
    pub fn from_shape_with_selection(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
        selection: AxisSelection<N>,
    ) -> Result<Self, FftError> {
        Self::from_shape_with_selection_and_layout(
            topology,
            global_shape,
            extra_shape,
            selection,
            DistributedLayout::default(),
        )
    }

    /// Collectively builds a plan from topology, shape, selection, and method.
    pub fn from_shape_with_selection_and_method(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
        selection: AxisSelection<N>,
        method: TransposeMethod,
    ) -> Result<Self, FftError> {
        Self::from_shape_with_selection_and_layout(
            topology,
            global_shape,
            extra_shape,
            selection,
            DistributedLayout {
                transpose_method: method,
                ..DistributedLayout::default()
            },
        )
    }

    /// Collectively builds a plan with an explicit transport and layout policy.
    pub fn from_shape_with_layout(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
        layout: DistributedLayout,
    ) -> Result<Self, FftError> {
        Self::from_shape_with_selection_and_layout(
            topology,
            global_shape,
            extra_shape,
            AxisSelection::all(),
            layout,
        )
    }

    /// Collectively builds a plan for a validated selection and layout policy.
    pub fn from_shape_with_selection_and_layout(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
        selection: AxisSelection<N>,
        layout: DistributedLayout,
    ) -> Result<Self, FftError> {
        let input = Pencil::new(
            Arc::clone(&topology),
            global_shape,
            std::array::from_fn(|axis| axis),
        )
        .map_err(FftError::Pencil);
        Self::construct(
            topology,
            global_shape,
            extra_shape,
            input,
            selection,
            layout,
            FourierDirections::default(),
        )
    }

    /// Collectively builds a plan from topology, global shape, and extra shape.
    pub fn from_shape_with_method(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
        method: TransposeMethod,
    ) -> Result<Self, FftError> {
        Self::from_shape_with_selection_and_layout(
            topology,
            global_shape,
            extra_shape,
            AxisSelection::all(),
            DistributedLayout {
                transpose_method: method,
                ..DistributedLayout::default()
            },
        )
    }

    /// Returns the canonical input pencil.
    pub fn input_pencil(&self) -> &Arc<Pencil<N, M>> {
        &self.core.stages[0].input
    }

    /// Returns the derived output pencil.
    pub fn output_pencil(&self) -> &Arc<Pencil<N, M>> {
        &self
            .core
            .stages
            .last()
            .expect("distributed C2C has at least two stages")
            .output
    }

    /// Returns the exact extra shape required by this plan.
    pub fn extra_shape(&self) -> &ExtraShape {
        &self.core.extra_shape
    }

    /// Returns the transport and memory-layout policy used by this plan.
    pub fn layout(&self) -> DistributedLayout {
        self.core.layout
    }

    /// Returns the immutable checked geometry used by every route stage.
    pub fn stage_geometry(&self) -> Box<[StageGeometry<N, M>]> {
        self.core
            .stages
            .iter()
            .map(|stage| StageGeometry {
                axis: stage.axis,
                source: Arc::clone(&stage.input),
                output: Arc::clone(&stage.output),
            })
            .collect()
    }

    /// Allocates a zero-initialized local input array for this plan.
    pub fn allocate_input(&self) -> Result<PencilArray<Complex<R>, N, M>, FftError> {
        PencilArray::from_fn(
            Arc::clone(self.input_pencil()),
            self.core.extra_shape.clone(),
            zero_complex::<R>,
        )
        .map_err(FftError::Array)
    }

    /// Allocates a zero-initialized local output array for this plan.
    pub fn allocate_output(&self) -> Result<PencilArray<Complex<R>, N, M>, FftError> {
        PencilArray::from_fn(
            Arc::clone(self.output_pencil()),
            self.core.extra_shape.clone(),
            zero_complex::<R>,
        )
        .map_err(FftError::Array)
    }

    /// Allocates a zero-initialized canonical input for in-place execution.
    ///
    /// This method is noncollective. If allocation fails on one rank, callers
    /// must coordinate that failure before the next collective call.
    pub fn allocate_in_place(&self) -> Result<C2cInPlaceArray<R, N, M>, FftError> {
        let array = ManyPencilArray::from_elem(
            registered_stage_pencils(&self.core.stages)?,
            0,
            self.core.extra_shape.clone(),
            zero_complex::<R>(),
        )
        .map_err(map_array_allocation)?;
        Ok(C2cInPlaceArray {
            core: Arc::clone(&self.core),
            array,
            state: C2cState::Input,
        })
    }

    /// Allocates reusable scratch for in-place forward, inverse, and backward execution.
    ///
    /// This method is noncollective. If allocation fails on one rank, callers
    /// must coordinate that failure before the next collective call.
    pub fn allocate_in_place_workspace(&self) -> Result<C2cInPlaceWorkspace<R, N, M>, FftError> {
        Ok(C2cInPlaceWorkspace {
            core: Arc::clone(&self.core),
            transpose: TransposeWorkspace::from_vecs(
                initialized_vec(self.core.transpose_send_len, zero_complex::<R>())?,
                initialized_vec(self.core.transpose_receive_len, zero_complex::<R>())?,
            ),
            fft_scratch: initialized_vec(self.core.fft_scratch_len, zero_complex::<R>())?,
            line_buffer: initialized_vec(self.core.strided_line_len, zero_complex::<R>())?,
        })
    }

    /// Allocates a reusable workspace for out-of-place forward, inverse, and backward execution.
    pub fn allocate_out_of_place_workspace(
        &self,
    ) -> Result<C2cOutOfPlaceWorkspace<R, N, M>, FftError> {
        let intermediate = ManyPencilArray::from_elem(
            registered_stage_pencils(&self.core.stages)?,
            0,
            self.core.extra_shape.clone(),
            zero_complex::<R>(),
        )
        .map_err(map_array_allocation)?;

        let transpose = TransposeWorkspace::from_vecs(
            initialized_vec(self.core.transpose_send_len, zero_complex::<R>())?,
            initialized_vec(self.core.transpose_receive_len, zero_complex::<R>())?,
        );
        let fft_scratch = initialized_vec(self.core.fft_scratch_len, zero_complex::<R>())?;
        let line_buffer = initialized_vec(self.core.strided_line_len, zero_complex::<R>())?;
        Ok(C2cOutOfPlaceWorkspace {
            core: Arc::clone(&self.core),
            intermediate,
            transpose,
            fft_scratch,
            line_buffer,
        })
    }

    /// Computes an unnormalized forward distributed C2C transform.
    ///
    /// The source uses the input layout and the destination uses the output
    /// layout. Ordinary descriptor and initial preflight errors are collective
    /// and leave source, destination, and workspace unchanged. After the first
    /// FFT stage, the selected checked transition may return a collectively
    /// agreed metadata or allocation error; the source remains preserved, but
    /// the workspace may already be changed. Native FFT, MPI failures,
    /// arbitrary panics, and process loss have no global recovery guarantee.
    pub fn forward(
        &self,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &mut PencilArray<Complex<R>, N, M>,
        workspace: &mut C2cOutOfPlaceWorkspace<R, N, M>,
    ) -> Result<(), FftError> {
        self.execute(Direction::Forward, source, destination, workspace, None)
    }

    /// Computes a normalized inverse distributed C2C transform.
    ///
    /// The source uses the forward output layout and the destination uses the
    /// canonical input layout. It has the same collective and initial-error
    /// guarantees as [`Self::forward`]: initial descriptor/preflight errors
    /// leave all buffers unchanged, while a post-start checked transition
    /// preparation or allocation error may change workspace contents but never
    /// the source. The local inverse plan divides once by each selected spatial axis
    /// length, so the complete inverse is normalized by the product of selected
    /// spatial lengths. Identity axes and extra batch dimensions are not
    /// included.
    pub fn inverse(
        &self,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &mut PencilArray<Complex<R>, N, M>,
        workspace: &mut C2cOutOfPlaceWorkspace<R, N, M>,
    ) -> Result<(), FftError> {
        self.execute(Direction::Inverse, source, destination, workspace, None)
    }

    /// Computes an unnormalized positive-sign backward distributed C2C transform.
    ///
    /// The source uses the forward output layout and the destination uses the
    /// canonical input layout. Unlike [`Self::inverse`], this raw backward
    /// transform does not divide by any selected spatial extent. A forward
    /// transform followed by this method therefore scales each value by the
    /// product of the selected spatial extents; identity axes and extra batch
    /// dimensions are not included.
    /// Initial descriptor/preflight errors preserve source, destination, and
    /// workspace, with the same post-start guarantees as [`Self::inverse`].
    pub fn backward(
        &self,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &mut PencilArray<Complex<R>, N, M>,
        workspace: &mut C2cOutOfPlaceWorkspace<R, N, M>,
    ) -> Result<(), FftError> {
        self.execute(Direction::Backward, source, destination, workspace, None)
    }

    /// Computes forward, running the next local FFT after receive/unpack
    /// completion and before the P2P send wait.
    /// A successful return has no pending network requests, including partial
    /// final sends.
    pub fn forward_with_overlap(
        &self,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &mut PencilArray<Complex<R>, N, M>,
        workspace: &mut C2cOutOfPlaceWorkspace<R, N, M>,
    ) -> Result<(), FftOverlapError<FftError>> {
        self.execute_overlap(Direction::Forward, source, destination, workspace)
    }

    /// Computes inverse, running the next local FFT after receive/unpack
    /// completion and before the P2P send wait.
    /// A successful return has no pending network requests, including partial
    /// final sends.
    pub fn inverse_with_overlap(
        &self,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &mut PencilArray<Complex<R>, N, M>,
        workspace: &mut C2cOutOfPlaceWorkspace<R, N, M>,
    ) -> Result<(), FftOverlapError<FftError>> {
        self.execute_overlap(Direction::Inverse, source, destination, workspace)
    }

    /// Computes backward, running the next local FFT after receive/unpack
    /// completion and before the P2P send wait.
    /// A successful return has no pending network requests, including partial
    /// final sends.
    pub fn backward_with_overlap(
        &self,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &mut PencilArray<Complex<R>, N, M>,
        workspace: &mut C2cOutOfPlaceWorkspace<R, N, M>,
    ) -> Result<(), FftOverlapError<FftError>> {
        self.execute_overlap(Direction::Backward, source, destination, workspace)
    }

    /// Runs [`Self::forward`] and returns fixed-size route timing.
    pub fn forward_with_timing(
        &self,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &mut PencilArray<Complex<R>, N, M>,
        workspace: &mut C2cOutOfPlaceWorkspace<R, N, M>,
    ) -> Result<TransformTiming<N>, FftError> {
        let mut timing = TransformTiming::<N>::default();
        self.execute(
            Direction::Forward,
            source,
            destination,
            workspace,
            Some(&mut timing),
        )?;
        Ok(timing)
    }

    /// Runs [`Self::inverse`] and returns fixed-size route timing.
    pub fn inverse_with_timing(
        &self,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &mut PencilArray<Complex<R>, N, M>,
        workspace: &mut C2cOutOfPlaceWorkspace<R, N, M>,
    ) -> Result<TransformTiming<N>, FftError> {
        let mut timing = TransformTiming::<N>::default();
        self.execute(
            Direction::Inverse,
            source,
            destination,
            workspace,
            Some(&mut timing),
        )?;
        Ok(timing)
    }

    /// Runs [`Self::backward`] and returns fixed-size route timing.
    pub fn backward_with_timing(
        &self,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &mut PencilArray<Complex<R>, N, M>,
        workspace: &mut C2cOutOfPlaceWorkspace<R, N, M>,
    ) -> Result<TransformTiming<N>, FftError> {
        let mut timing = TransformTiming::<N>::default();
        self.execute(
            Direction::Backward,
            source,
            destination,
            workspace,
            Some(&mut timing),
        )?;
        Ok(timing)
    }

    /// Computes an unnormalized forward transform in the array's single buffer.
    ///
    /// The array must be in [`C2cState::Input`]. Initial collective
    /// preflight errors preserve its state, data, and workspace. Once
    /// execution begins, the state is set to [`C2cState::Poisoned`] before the
    /// first write and remains so until the complete route succeeds; a later
    /// error or panic does not roll back the buffer. A foreign-plan array maps to
    /// `FftError::Array(ArrayError::IncompatiblePencils)`, a poisoned array to
    /// `FftError::Array(ArrayError::Poisoned)`, and a wrong state or active
    /// endpoint layout to `FftError::InputLayoutMismatch`; inspect
    /// [`C2cInPlaceArray::state`] to distinguish state outcomes. A foreign
    /// workspace remains `FftError::WorkspaceMismatch`.
    pub fn forward_in_place(
        &self,
        array: &mut C2cInPlaceArray<R, N, M>,
        workspace: &mut C2cInPlaceWorkspace<R, N, M>,
    ) -> Result<(), FftError> {
        self.execute_in_place(Direction::Forward, array, workspace, None)
    }

    /// Computes a normalized inverse transform in the array's single buffer.
    ///
    /// The array must be in [`C2cState::Output`] and may be edited through
    /// [`C2cInPlaceArray::view_mut`] before this call. The inverse normalizes
    /// once per selected spatial axis through the existing local inverse plans;
    /// identity axes and extra batches are not included. It has the same collective preflight,
    /// poisoning, and MPI failure contract as
    /// [`Self::forward_in_place`], including its error mappings.
    pub fn inverse_in_place(
        &self,
        array: &mut C2cInPlaceArray<R, N, M>,
        workspace: &mut C2cInPlaceWorkspace<R, N, M>,
    ) -> Result<(), FftError> {
        self.execute_in_place(Direction::Inverse, array, workspace, None)
    }

    /// Computes an unnormalized positive-sign backward transform in the array's
    /// single buffer.
    ///
    /// The array must be in [`C2cState::Output`]. The route ends in the
    /// canonical input layout and does not divide by selected spatial extents,
    /// so a forward/backward pair scales by their product. Identity axes do not
    /// contribute. It has the same
    /// collective preflight, poisoning, and MPI failure contract as
    /// [`Self::inverse_in_place`].
    pub fn backward_in_place(
        &self,
        array: &mut C2cInPlaceArray<R, N, M>,
        workspace: &mut C2cInPlaceWorkspace<R, N, M>,
    ) -> Result<(), FftError> {
        self.execute_in_place(Direction::Backward, array, workspace, None)
    }

    /// Runs [`Self::forward_in_place`] and returns fixed-size route timing.
    pub fn forward_in_place_with_timing(
        &self,
        array: &mut C2cInPlaceArray<R, N, M>,
        workspace: &mut C2cInPlaceWorkspace<R, N, M>,
    ) -> Result<TransformTiming<N>, FftError> {
        let mut timing = TransformTiming::<N>::default();
        self.execute_in_place(Direction::Forward, array, workspace, Some(&mut timing))?;
        Ok(timing)
    }

    /// Runs [`Self::inverse_in_place`] and returns fixed-size route timing.
    pub fn inverse_in_place_with_timing(
        &self,
        array: &mut C2cInPlaceArray<R, N, M>,
        workspace: &mut C2cInPlaceWorkspace<R, N, M>,
    ) -> Result<TransformTiming<N>, FftError> {
        let mut timing = TransformTiming::<N>::default();
        self.execute_in_place(Direction::Inverse, array, workspace, Some(&mut timing))?;
        Ok(timing)
    }

    /// Runs [`Self::backward_in_place`] and returns fixed-size route timing.
    pub fn backward_in_place_with_timing(
        &self,
        array: &mut C2cInPlaceArray<R, N, M>,
        workspace: &mut C2cInPlaceWorkspace<R, N, M>,
    ) -> Result<TransformTiming<N>, FftError> {
        let mut timing = TransformTiming::<N>::default();
        self.execute_in_place(Direction::Backward, array, workspace, Some(&mut timing))?;
        Ok(timing)
    }

    /// Builds a plan with explicit per-axis Fourier signs.
    ///
    /// Forward uses the configured signs; inverse and backward use their opposites.
    pub fn from_shape_with_fft_directions(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
        directions: FourierDirections<N>,
    ) -> Result<Self, FftError> {
        let input = Pencil::new(
            Arc::clone(&topology),
            global_shape,
            std::array::from_fn(|axis| axis),
        )
        .map_err(FftError::Pencil);
        Self::construct(
            topology,
            global_shape,
            extra_shape,
            input,
            AxisSelection::all(),
            DistributedLayout::default(),
            directions,
        )
    }

    /// Rebuilds this plan with new Fourier signs and fresh array/workspace identities.
    /// Forward is unscaled with these signs; inverse uses opposite signs with
    /// normalization, and backward uses opposite signs without normalization.
    /// Non-selected axes must use `Forward` (checked collectively).
    pub fn with_fft_directions(&self, directions: FourierDirections<N>) -> Result<Self, FftError> {
        let topology = Arc::clone(self.input_pencil().topology());
        let shape = *self.input_pencil().global_shape();
        let input = Pencil::new(
            Arc::clone(&topology),
            shape,
            std::array::from_fn(|axis| axis),
        )
        .map_err(FftError::Pencil);
        let mut plan = Self::construct(
            topology,
            shape,
            self.core.extra_shape.clone(),
            input,
            self.core.selection,
            self.core.layout,
            directions,
        )?;
        Arc::get_mut(&mut plan.core)
            .expect("fresh core")
            .strict_array_identity = true;
        Ok(plan)
    }

    /// Returns the signs used by the forward operation.
    pub fn fft_directions(&self) -> FourierDirections<N> {
        self.core.directions
    }

    fn construct(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
        input: Result<Arc<Pencil<N, M>>, FftError>,
        selection: AxisSelection<N>,
        layout: DistributedLayout,
        directions: FourierDirections<N>,
    ) -> Result<Self, FftError> {
        let communicator = topology.communicator();
        let expected_len =
            descriptor_len::<N, M>(&extra_shape).and_then(|length| length.checked_add(N));
        let descriptor = expected_len.and_then(|_| {
            let mut descriptor = build_descriptor::<R, N, M>(
                &topology,
                global_shape,
                &extra_shape,
                selection,
                VALUE_KIND_C2C,
                layout,
            )
            .ok()?;
            descriptor.extend(directions.0.iter().map(|direction| match direction {
                FourierDirection::Forward => 0,
                FourierDirection::Backward => 1,
            }));
            Some(descriptor)
        });
        let descriptor_len_word = expected_len
            .and_then(|length| u64::try_from(length).ok())
            .unwrap_or(INVALID_WORD);
        let header = [
            DESCRIPTOR_SCHEMA,
            OPERATION_PLAN,
            u64::try_from(N).unwrap_or(INVALID_WORD),
            u64::try_from(M).unwrap_or(INVALID_WORD),
            descriptor_len_word,
        ];
        if !agree_header(communicator, header) {
            return Err(FftError::CollectiveDescriptorMismatch);
        }
        // The exact mask is agreed before route construction or native FFT
        // planning. A rank-local selection mismatch therefore cannot enter a
        // different route; callers coordinate any local constructor result.
        let descriptor = collective_descriptor(communicator, descriptor, expected_len)?;
        let route = agree_result(
            communicator,
            build_route(input, &topology, global_shape, layout.permute_dims),
        )?;
        agree_result(
            communicator,
            if (0..N).any(|axis| {
                !selection.contains(axis)
                    && directions.get(axis) == Some(FourierDirection::Backward)
            }) {
                Err(FftError::PreparationFailed)
            } else {
                Ok(())
            },
        )?;
        let stages = agree_result(
            communicator,
            prepare_stages::<R, N, M>(&route, global_shape, selection, directions),
        )?;

        let (
            transitions,
            transpose_send_len,
            transpose_receive_len,
            real_transpose_send_len,
            real_transpose_receive_len,
        ) = build_transitions::<R, N, M>(
            communicator,
            &stages,
            &route.distributed,
            &extra_shape,
            layout.transpose_method,
            0,
        )?;

        let strided_line_len = strided_complex_line_len(&stages.stages)?;
        let core = TransformPlanCore {
            stages: stages.stages,
            transitions: transitions.into_boxed_slice(),
            extra_shape,
            selection,
            layout,
            real_stage_index: None,
            descriptor: descriptor.into_boxed_slice(),
            fft_scratch_len: stages.fft_scratch_len,
            strided_line_len,
            transpose_send_len,
            transpose_receive_len,
            real_transpose_send_len,
            real_transpose_receive_len,
            directions,
            strict_array_identity: false,
        };
        Ok(Self {
            core: Arc::new(core),
        })
    }

    fn execute(
        &self,
        direction: Direction,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &mut PencilArray<Complex<R>, N, M>,
        workspace: &mut C2cOutOfPlaceWorkspace<R, N, M>,
        mut report: Option<&mut TransformTiming<N>>,
    ) -> Result<(), FftError> {
        let started = Instant::now();
        let communicator = self.input_pencil().topology().communicator();
        let operation = match direction {
            Direction::Forward => {
                if report.is_some() {
                    OPERATION_FORWARD_TIMED
                } else {
                    OPERATION_FORWARD
                }
            }
            Direction::Inverse => {
                if report.is_some() {
                    OPERATION_INVERSE_TIMED
                } else {
                    OPERATION_INVERSE
                }
            }
            Direction::Backward => {
                if report.is_some() {
                    OPERATION_BACKWARD_TIMED
                } else {
                    OPERATION_BACKWARD
                }
            }
        };
        agree_execution_descriptor(communicator, operation, &self.core)?;

        let local_preflight = self.preflight(direction, source, destination, workspace);
        if !collective_valid(communicator, local_preflight.is_ok()) {
            return Err(local_preflight
                .err()
                .unwrap_or(FftError::CollectivePreconditionFailed));
        }
        local_preflight.expect("collective distributed C2C preflight succeeded");

        let normalize_inverse = matches!(direction, Direction::Inverse);
        let result = match direction {
            Direction::Forward => execute_forward(
                &self.core,
                source,
                destination,
                workspace,
                report.as_deref_mut(),
            ),
            Direction::Inverse | Direction::Backward => execute_inverse(
                &self.core,
                source,
                destination,
                workspace,
                normalize_inverse,
                report.as_deref_mut(),
            ),
        };
        if let Some(timing) = report {
            timing.total = started.elapsed();
        }
        result
    }

    fn execute_overlap(
        &self,
        direction: Direction,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &mut PencilArray<Complex<R>, N, M>,
        workspace: &mut C2cOutOfPlaceWorkspace<R, N, M>,
    ) -> Result<(), FftOverlapError<FftError>> {
        let communicator = self.input_pencil().topology().communicator();
        let operation = match direction {
            Direction::Forward => OPERATION_FORWARD_OVERLAP,
            Direction::Inverse => OPERATION_INVERSE_OVERLAP,
            Direction::Backward => OPERATION_BACKWARD_OVERLAP,
        };
        agree_execution_descriptor(communicator, operation, &self.core)?;
        let local = self.preflight(direction, source, destination, workspace);
        if !collective_valid(communicator, local.is_ok()) {
            return Err(local
                .err()
                .unwrap_or(FftError::CollectivePreconditionFailed)
                .into());
        }
        let overlap_supported = self.core.transitions.iter().all(|transition| {
            !matches!(transition.forward, C2cTransition::AllToAllv(_))
                && !matches!(transition.backward, C2cTransition::AllToAllv(_))
        });
        // This agreement is deliberately before stage-zero writes or payload.
        if !collective_valid(communicator, overlap_supported) {
            return Err(FftOverlapError::UnsupportedTransport);
        }
        match direction {
            Direction::Forward => {
                execute_forward_overlap(&self.core, source, destination, workspace)
            }
            Direction::Inverse => {
                execute_inverse_overlap(&self.core, source, destination, workspace, true)
            }
            Direction::Backward => {
                execute_inverse_overlap(&self.core, source, destination, workspace, false)
            }
        }
    }

    fn preflight(
        &self,
        direction: Direction,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &PencilArray<Complex<R>, N, M>,
        workspace: &C2cOutOfPlaceWorkspace<R, N, M>,
    ) -> Result<(), FftError> {
        validate_out_of_place(
            &self.core,
            &workspace.core,
            direction,
            source,
            destination,
            &workspace.intermediate,
            (
                workspace.fft_scratch.len(),
                workspace.transpose.send_len(),
                workspace.transpose.receive_len(),
            ),
        )?;
        validate_strided_line_buffer(workspace.line_buffer.len(), self.core.strided_line_len)
    }

    fn execute_in_place(
        &self,
        direction: Direction,
        array: &mut C2cInPlaceArray<R, N, M>,
        workspace: &mut C2cInPlaceWorkspace<R, N, M>,
        mut report: Option<&mut TransformTiming<N>>,
    ) -> Result<(), FftError> {
        let started = Instant::now();
        let communicator = self.input_pencil().topology().communicator();
        let operation = match direction {
            Direction::Forward => {
                if report.is_some() {
                    OPERATION_FORWARD_IN_PLACE_TIMED
                } else {
                    OPERATION_FORWARD_IN_PLACE
                }
            }
            Direction::Inverse => {
                if report.is_some() {
                    OPERATION_INVERSE_IN_PLACE_TIMED
                } else {
                    OPERATION_INVERSE_IN_PLACE
                }
            }
            Direction::Backward => {
                if report.is_some() {
                    OPERATION_BACKWARD_IN_PLACE_TIMED
                } else {
                    OPERATION_BACKWARD_IN_PLACE
                }
            }
        };
        agree_execution_descriptor(communicator, operation, &self.core)?;

        let local_preflight = self.preflight_in_place(direction, array, workspace);
        if !collective_valid(communicator, local_preflight.is_ok()) {
            return Err(local_preflight
                .err()
                .unwrap_or(FftError::CollectivePreconditionFailed));
        }
        local_preflight.expect("collective distributed C2C in-place preflight succeeded");

        let normalize_inverse = matches!(direction, Direction::Inverse);
        let target = match direction {
            Direction::Forward => C2cState::Output,
            Direction::Inverse | Direction::Backward => C2cState::Input,
        };
        let result =
            run_in_place_transaction(
                array,
                workspace,
                target,
                |array, workspace| match direction {
                    Direction::Forward => execute_forward_in_place(
                        &self.core,
                        &mut array.array,
                        &mut workspace.transpose,
                        &mut workspace.fft_scratch,
                        &mut workspace.line_buffer,
                        report.as_deref_mut(),
                    ),
                    Direction::Inverse | Direction::Backward => execute_inverse_in_place(
                        &self.core,
                        &mut array.array,
                        &mut workspace.transpose,
                        &mut workspace.fft_scratch,
                        &mut workspace.line_buffer,
                        normalize_inverse,
                        report.as_deref_mut(),
                    ),
                },
            );
        if let Some(timing) = report {
            timing.total = started.elapsed();
        }
        result
    }

    fn preflight_in_place(
        &self,
        direction: Direction,
        array: &C2cInPlaceArray<R, N, M>,
        workspace: &C2cInPlaceWorkspace<R, N, M>,
    ) -> Result<(), FftError> {
        if !Arc::ptr_eq(&array.core, &self.core) {
            return Err(FftError::Array(ArrayError::IncompatiblePencils));
        }
        if !Arc::ptr_eq(&workspace.core, &self.core) {
            return Err(FftError::WorkspaceMismatch);
        }

        let expected_state = match direction {
            Direction::Forward => C2cState::Input,
            Direction::Inverse | Direction::Backward => C2cState::Output,
        };
        match array.state {
            C2cState::Poisoned => return Err(FftError::Array(ArrayError::Poisoned)),
            state if state != expected_state => return Err(FftError::InputLayoutMismatch),
            _ => {}
        }

        if array.array.extra_shape() != &self.core.extra_shape {
            return Err(FftError::ExtraShapeMismatch);
        }

        let expected_pencil = match direction {
            Direction::Forward => self.input_pencil(),
            Direction::Inverse | Direction::Backward => self.output_pencil(),
        };
        let active = array.array.active_pencil().map_err(FftError::Array)?;
        if !active.same_layout(expected_pencil.as_ref()) {
            return Err(FftError::InputLayoutMismatch);
        }

        validate_workspace_lengths(
            &self.core,
            workspace.fft_scratch.len(),
            workspace.transpose.send_len(),
            workspace.transpose.receive_len(),
        )?;
        validate_strided_line_buffer(workspace.line_buffer.len(), self.core.strided_line_len)?;
        Ok(())
    }
}

impl<R: FftReal, const N: usize, const M: usize> C2cInPlaceArray<R, N, M> {
    /// Returns the array's FFT completion state.
    pub fn state(&self) -> C2cState {
        self.state
    }

    /// Borrows the active array view when the state is valid.
    ///
    /// This borrow is noncollective.
    pub fn view(&self) -> Result<PencilArrayView<'_, Complex<R>, N, M>, FftError> {
        self.ensure_viewable()?;
        self.array.active_view().map_err(FftError::Array)
    }

    /// Borrows the active mutable array view when the state is valid.
    ///
    /// This borrow is noncollective.
    pub fn view_mut(&mut self) -> Result<PencilArrayViewMut<'_, Complex<R>, N, M>, FftError> {
        self.ensure_viewable()?;
        self.array.active_view_mut().map_err(FftError::Array)
    }

    fn ensure_viewable(&self) -> Result<(), FftError> {
        match self.state {
            C2cState::Input | C2cState::Output => Ok(()),
            C2cState::Poisoned => Err(FftError::Array(ArrayError::Poisoned)),
        }
    }
}

fn registered_stage_pencils<R: FftReal, const N: usize, const M: usize>(
    stages: &[TransformStage<R, N, M>],
) -> Result<Box<[Arc<Pencil<N, M>>]>, FftError> {
    let mut pencils = Vec::new();
    pencils
        .try_reserve_exact(stages.len())
        .map_err(|_| FftError::AllocationFailed {
            required: stages.len(),
        })?;
    for stage in stages {
        if !pencils
            .iter()
            .any(|registered: &Arc<Pencil<N, M>>| registered.same_layout(stage.output.as_ref()))
        {
            pencils.push(Arc::clone(&stage.output));
        }
    }
    Ok(pencils.into_boxed_slice())
}

fn validate_input<const N: usize, const M: usize>(
    input: Result<Arc<Pencil<N, M>>, FftError>,
    topology: &Arc<MpiTopology<M>>,
    global_shape: [usize; N],
) -> Result<Arc<Pencil<N, M>>, FftError> {
    if N < 2 || M == 0 || M >= N {
        return Err(FftError::InvalidDimensions);
    }
    if let Some(axis) = global_shape.iter().position(|&extent| extent == 0) {
        return Err(FftError::Pencil(PencilError::ZeroGlobalExtent { axis }));
    }
    let input = input?;
    if !Arc::ptr_eq(input.topology(), topology) {
        return Err(FftError::PreparationFailed);
    }
    if input.global_shape() != &global_shape {
        return Err(FftError::PreparationFailed);
    }
    let identity = std::array::from_fn(|axis| axis);
    if input.permutation().axes().map(SpatialAxis::index) != identity {
        return Err(FftError::InvalidInputLayout);
    }
    if input.decomposition().map(SpatialAxis::index) != std::array::from_fn(|axis| axis) {
        return Err(FftError::InvalidInputLayout);
    }
    Ok(input)
}

fn build_route<const N: usize, const M: usize>(
    input: Result<Arc<Pencil<N, M>>, FftError>,
    topology: &Arc<MpiTopology<M>>,
    global_shape: [usize; N],
    permute_dims: bool,
) -> Result<RouteCandidate<N, M>, FftError> {
    let input = validate_input(input, topology, global_shape)?;
    let mut stages = Vec::new();
    stages
        .try_reserve_exact(N)
        .map_err(|_| FftError::AllocationFailed { required: N })?;
    stages.push(Arc::clone(&input));
    let mut distributed = Vec::new();
    distributed
        .try_reserve_exact(N - 1)
        .map_err(|_| FftError::AllocationFailed { required: N - 1 })?;

    let mut current = input;
    for axis in (0..N - 1).rev() {
        let current_axes = current.permutation().axes().map(SpatialAxis::index);
        let mut next_axes = [0usize; N];
        let mut position = 0;
        for value in current_axes {
            if value != axis {
                next_axes[position] = value;
                position += 1;
            }
        }
        next_axes[N - 1] = axis;

        let mut next_decomposition = current.decomposition().map(SpatialAxis::index);
        let is_distributed = advance_decomposition(&mut next_decomposition, axis);
        let next_permutation = if permute_dims {
            AxisPermutation::new(next_axes)
                .map_err(PencilError::InvalidPermutation)
                .map_err(FftError::Pencil)?
        } else {
            AxisPermutation::identity()
        };
        let next = current
            .reconfigured(PencilConfig {
                global_shape,
                decomposition: next_decomposition,
                permutation: next_permutation,
            })
            .map_err(FftError::Pencil)?;
        debug_assert_eq!(is_distributed, !current.same_distribution(next.as_ref()));
        distributed.push(is_distributed);
        stages.push(Arc::clone(&next));
        current = next;
    }

    let expected_output = std::array::from_fn(|axis| axis + 1);
    let expected_permutation = if permute_dims {
        std::array::from_fn(|position| N - 1 - position)
    } else {
        std::array::from_fn(|position| position)
    };
    if current.decomposition().map(SpatialAxis::index) != expected_output
        || current.permutation().axes().map(SpatialAxis::index) != expected_permutation
    {
        return Err(FftError::PreparationFailed);
    }
    Ok(RouteCandidate {
        stages: stages.into_boxed_slice(),
        distributed: distributed.into_boxed_slice(),
    })
}

fn advance_decomposition<const M: usize>(decomposition: &mut [usize; M], axis: usize) -> bool {
    if let Some(position) = decomposition.iter().position(|&value| value == axis) {
        decomposition[position] = axis + 1;
        true
    } else {
        false
    }
}

fn prepare_complex_stage<R: FftReal, const N: usize, const M: usize>(
    pencil: &Arc<Pencil<N, M>>,
    global_shape: [usize; N],
    axis_index: usize,
    selected: bool,
    directions: FourierDirections<N>,
) -> Result<TransformStage<R, N, M>, FftError> {
    if pencil
        .decomposition()
        .iter()
        .any(|distributed| distributed.index() == axis_index)
        || pencil.local_shape_logical()[axis_index] != global_shape[axis_index]
    {
        return Err(FftError::PreparationFailed);
    }
    let local = if selected {
        LocalTransform::Complex(LocalC2cPlan::new_with_sign(
            global_shape[axis_index],
            directions.get(axis_index) == Some(FourierDirection::Backward),
        )?)
    } else {
        LocalTransform::Identity
    };
    Ok(TransformStage {
        axis: axis_index,
        input: Arc::clone(pencil),
        output: Arc::clone(pencil),
        local,
    })
}

fn prepare_stages<R: FftReal, const N: usize, const M: usize>(
    route: &RouteCandidate<N, M>,
    global_shape: [usize; N],
    selection: AxisSelection<N>,
    directions: FourierDirections<N>,
) -> Result<StagePreparation<R, N, M>, FftError> {
    let required = route.stages.len();
    let mut stages = Vec::new();
    stages
        .try_reserve_exact(required)
        .map_err(|_| FftError::AllocationFailed { required })?;
    let mut fft_scratch_len = 0usize;
    for (index, pencil) in route.stages.iter().enumerate() {
        let axis = N - 1 - index;
        let stage = prepare_complex_stage(
            pencil,
            global_shape,
            axis,
            selection.contains(axis),
            directions,
        )?;
        fft_scratch_len = fft_scratch_len.max(stage.local.scratch_len());
        stages.push(stage);
    }
    Ok(StagePreparation {
        stages: stages.into_boxed_slice(),
        fft_scratch_len,
    })
}

fn build_transitions<R: FftReal, const N: usize, const M: usize>(
    communicator: &mpi::topology::CartesianCommunicator,
    stages: &StagePreparation<R, N, M>,
    distributed: &[bool],
    extra_shape: &ExtraShape,
    method: TransposeMethod,
    real_transition_count: usize,
) -> Result<(Vec<C2cStageTransition<N, M>>, usize, usize, usize, usize), FftError> {
    let mut transitions = Vec::new();
    agree_result(
        communicator,
        transitions
            .try_reserve_exact(distributed.len())
            .map_err(|_| FftError::AllocationFailed {
                required: distributed.len(),
            }),
    )?;

    let mut transpose_send_len = 0usize;
    let mut transpose_receive_len = 0usize;
    let mut real_transpose_send_len = 0usize;
    let mut real_transpose_receive_len = 0usize;
    let extra_count = extra_shape.element_count();

    for (index, &is_distributed) in distributed.iter().enumerate() {
        let source = Arc::clone(&stages.stages[index].output);
        let destination = Arc::clone(&stages.stages[index + 1].input);
        let is_real = index < real_transition_count;
        if is_distributed {
            let forward = agree_result(
                communicator,
                build_distributed_transition(Arc::clone(&source), Arc::clone(&destination), method),
            )?;
            let forward_requirements = agree_result(
                communicator,
                transition_workspace_requirements(&forward, extra_shape),
            )?;
            let backward = agree_result(
                communicator,
                build_distributed_transition(Arc::clone(&destination), Arc::clone(&source), method),
            )?;
            let backward_requirements = agree_result(
                communicator,
                transition_workspace_requirements(&backward, extra_shape),
            )?;
            let send_len = forward_requirements
                .send_len
                .max(backward_requirements.send_len);
            let receive_len = forward_requirements
                .receive_len
                .max(backward_requirements.receive_len);
            if is_real {
                real_transpose_send_len = real_transpose_send_len.max(send_len);
                real_transpose_receive_len = real_transpose_receive_len.max(receive_len);
            } else {
                transpose_send_len = transpose_send_len.max(send_len);
                transpose_receive_len = transpose_receive_len.max(receive_len);
            }
            transitions.push(C2cStageTransition { forward, backward });
        } else if source.same_layout(destination.as_ref()) {
            transitions.push(C2cStageTransition {
                forward: C2cTransition::Identity,
                backward: C2cTransition::Identity,
            });
            continue;
        } else {
            let local = (|| {
                let forward =
                    LocalTransposePlan::new(Arc::clone(&source), Arc::clone(&destination))?;
                let backward =
                    LocalTransposePlan::new(Arc::clone(&destination), Arc::clone(&source))?;
                let forward_send_len = source
                    .local_len()
                    .checked_mul(extra_count)
                    .ok_or(FftError::PreparationFailed)?;
                let backward_send_len = destination
                    .local_len()
                    .checked_mul(extra_count)
                    .ok_or(FftError::PreparationFailed)?;
                Ok::<_, FftError>((
                    C2cStageTransition {
                        forward: C2cTransition::Local(forward),
                        backward: C2cTransition::Local(backward),
                    },
                    forward_send_len.max(backward_send_len),
                ))
            })();
            let (transition, send_len) = agree_result(communicator, local)?;
            if is_real {
                real_transpose_send_len = real_transpose_send_len.max(send_len);
            } else {
                transpose_send_len = transpose_send_len.max(send_len);
            }
            transitions.push(transition);
        }
    }

    Ok((
        transitions,
        transpose_send_len,
        transpose_receive_len,
        real_transpose_send_len,
        real_transpose_receive_len,
    ))
}

fn build_distributed_transition<const N: usize, const M: usize>(
    source: Arc<Pencil<N, M>>,
    destination: Arc<Pencil<N, M>>,
    method: TransposeMethod,
) -> Result<C2cTransition<N, M>, FftError> {
    match method {
        TransposeMethod::AllToAllv => AllToAllvTransposePlan::new(source, destination)
            .map(C2cTransition::AllToAllv)
            .map_err(FftError::Transpose),
        TransposeMethod::PointToPoint => PointToPointTransposePlan::new(source, destination)
            .map(C2cTransition::PointToPoint)
            .map_err(FftError::Transpose),
    }
}

fn transition_workspace_requirements<const N: usize, const M: usize>(
    transition: &C2cTransition<N, M>,
    extra_shape: &ExtraShape,
) -> Result<TransposeWorkspaceRequirements, FftError> {
    match transition {
        C2cTransition::Identity => Ok(TransposeWorkspaceRequirements {
            send_len: 0,
            receive_len: 0,
        }),
        C2cTransition::AllToAllv(plan) => plan
            .workspace_requirements(extra_shape)
            .map_err(FftError::Transpose),
        C2cTransition::PointToPoint(plan) => plan
            .workspace_requirements(extra_shape)
            .map_err(FftError::Transpose),
        C2cTransition::Local(_) => Err(FftError::PreparationFailed),
    }
}

fn memory_stride<const N: usize, const M: usize>(
    pencil: &Pencil<N, M>,
    axis: usize,
) -> Result<usize, FftError> {
    let position = pencil
        .permutation()
        .axes()
        .iter()
        .position(|value| value.index() == axis)
        .ok_or(FftError::PreparationFailed)?;
    let shape = pencil.local_shape_memory();
    let mut stride = 1usize;
    for &extent in shape.iter().skip(position + 1) {
        stride = stride
            .checked_mul(extent)
            .ok_or(FftError::PreparationFailed)?;
    }
    Ok(if pencil.local_len() == 0 { 0 } else { stride })
}

fn strided_line_count(data_len: usize, line_len: usize, stride: usize) -> Result<usize, FftError> {
    if stride == 0 {
        return if data_len == 0 {
            Ok(0)
        } else {
            Err(FftError::PreparationFailed)
        };
    }
    let block = line_len
        .checked_mul(stride)
        .ok_or(FftError::PreparationFailed)?;
    if data_len % block != 0 {
        return Err(LocalC2cError::NonIntegralBatch.into());
    }
    Ok(data_len / block)
}

fn strided_complex_line_len<R: FftReal, const N: usize, const M: usize>(
    stages: &[TransformStage<R, N, M>],
) -> Result<usize, FftError> {
    let mut required = 0usize;
    for stage in stages {
        let LocalTransform::Complex(plan) = &stage.local else {
            continue;
        };
        if memory_stride(stage.output.as_ref(), stage.axis)? > 1 {
            required = required.max(plan.line_len());
        }
    }
    Ok(required)
}

fn validate_strided_line_buffer(actual: usize, required: usize) -> Result<(), FftError> {
    if actual < required {
        return Err(FftError::WorkspaceTooSmall {
            kind: "complex line",
            required,
            actual,
        });
    }
    Ok(())
}

fn execute_strided_complex_forward<R: FftReal, const N: usize, const M: usize>(
    plan: &LocalC2cPlan<R>,
    pencil: &Pencil<N, M>,
    axis: usize,
    source: &[Complex<R>],
    destination: &mut [Complex<R>],
    scratch: &mut [Complex<R>],
    line_buffer: &mut [Complex<R>],
) -> Result<(), FftError> {
    let stride = memory_stride(pencil, axis)?;
    if stride <= 1 {
        return plan
            .forward(source, destination, scratch)
            .map_err(FftError::LocalC2c);
    }
    validate_strided_line_buffer(line_buffer.len(), plan.line_len())?;
    let count = strided_line_count(source.len(), plan.line_len(), stride)?;
    if destination.len() != source.len() {
        return Err(FftError::LocalC2c(LocalC2cError::BufferLengthMismatch));
    }
    if source.is_empty() {
        return plan
            .forward(source, destination, scratch)
            .map_err(FftError::LocalC2c);
    }
    let block = plan
        .line_len()
        .checked_mul(stride)
        .ok_or(FftError::PreparationFailed)?;
    for outer in 0..count {
        let base = outer
            .checked_mul(block)
            .ok_or(FftError::PreparationFailed)?;
        for inner in 0..stride {
            for k in 0..plan.line_len() {
                line_buffer[k] = source[base + k * stride + inner];
            }
            plan.forward_in_place(&mut line_buffer[..plan.line_len()], scratch)
                .map_err(FftError::LocalC2c)?;
            for k in 0..plan.line_len() {
                destination[base + k * stride + inner] = line_buffer[k];
            }
        }
    }
    Ok(())
}

fn execute_strided_complex_forward_in_place<R: FftReal, const N: usize, const M: usize>(
    plan: &LocalC2cPlan<R>,
    pencil: &Pencil<N, M>,
    axis: usize,
    data: &mut [Complex<R>],
    scratch: &mut [Complex<R>],
    line_buffer: &mut [Complex<R>],
) -> Result<(), FftError> {
    let stride = memory_stride(pencil, axis)?;
    if stride <= 1 {
        return plan
            .forward_in_place(data, scratch)
            .map_err(FftError::LocalC2c);
    }
    validate_strided_line_buffer(line_buffer.len(), plan.line_len())?;
    let count = strided_line_count(data.len(), plan.line_len(), stride)?;
    if data.is_empty() {
        return plan
            .forward_in_place(data, scratch)
            .map_err(FftError::LocalC2c);
    }
    let block = plan
        .line_len()
        .checked_mul(stride)
        .ok_or(FftError::PreparationFailed)?;
    for outer in 0..count {
        let base = outer
            .checked_mul(block)
            .ok_or(FftError::PreparationFailed)?;
        for inner in 0..stride {
            for k in 0..plan.line_len() {
                line_buffer[k] = data[base + k * stride + inner];
            }
            plan.forward_in_place(&mut line_buffer[..plan.line_len()], scratch)
                .map_err(FftError::LocalC2c)?;
            for k in 0..plan.line_len() {
                data[base + k * stride + inner] = line_buffer[k];
            }
        }
    }
    Ok(())
}

fn execute_strided_complex_reverse<R: FftReal, const N: usize, const M: usize>(
    plan: &LocalC2cPlan<R>,
    pencil: &Pencil<N, M>,
    axis: usize,
    source: &[Complex<R>],
    destination: &mut [Complex<R>],
    scratch: &mut [Complex<R>],
    line_buffer: &mut [Complex<R>],
    normalize: bool,
) -> Result<(), FftError> {
    let stride = memory_stride(pencil, axis)?;
    if stride <= 1 {
        return if normalize {
            plan.inverse(source, destination, scratch)
        } else {
            plan.backward(source, destination, scratch)
        }
        .map_err(FftError::LocalC2c);
    }
    validate_strided_line_buffer(line_buffer.len(), plan.line_len())?;
    let count = strided_line_count(source.len(), plan.line_len(), stride)?;
    if destination.len() != source.len() {
        return Err(FftError::LocalC2c(LocalC2cError::BufferLengthMismatch));
    }
    if source.is_empty() {
        return if normalize {
            plan.inverse(source, destination, scratch)
        } else {
            plan.backward(source, destination, scratch)
        }
        .map_err(FftError::LocalC2c);
    }
    let block = plan
        .line_len()
        .checked_mul(stride)
        .ok_or(FftError::PreparationFailed)?;
    for outer in 0..count {
        let base = outer
            .checked_mul(block)
            .ok_or(FftError::PreparationFailed)?;
        for inner in 0..stride {
            for k in 0..plan.line_len() {
                line_buffer[k] = source[base + k * stride + inner];
            }
            if normalize {
                plan.inverse_in_place(&mut line_buffer[..plan.line_len()], scratch)
            } else {
                plan.backward_in_place(&mut line_buffer[..plan.line_len()], scratch)
            }
            .map_err(FftError::LocalC2c)?;
            for k in 0..plan.line_len() {
                destination[base + k * stride + inner] = line_buffer[k];
            }
        }
    }
    Ok(())
}

fn execute_strided_complex_reverse_in_place<R: FftReal, const N: usize, const M: usize>(
    plan: &LocalC2cPlan<R>,
    pencil: &Pencil<N, M>,
    axis: usize,
    data: &mut [Complex<R>],
    scratch: &mut [Complex<R>],
    line_buffer: &mut [Complex<R>],
    normalize: bool,
) -> Result<(), FftError> {
    let stride = memory_stride(pencil, axis)?;
    if stride <= 1 {
        return if normalize {
            plan.inverse_in_place(data, scratch)
        } else {
            plan.backward_in_place(data, scratch)
        }
        .map_err(FftError::LocalC2c);
    }
    validate_strided_line_buffer(line_buffer.len(), plan.line_len())?;
    let count = strided_line_count(data.len(), plan.line_len(), stride)?;
    if data.is_empty() {
        return if normalize {
            plan.inverse_in_place(data, scratch)
        } else {
            plan.backward_in_place(data, scratch)
        }
        .map_err(FftError::LocalC2c);
    }
    let block = plan
        .line_len()
        .checked_mul(stride)
        .ok_or(FftError::PreparationFailed)?;
    for outer in 0..count {
        let base = outer
            .checked_mul(block)
            .ok_or(FftError::PreparationFailed)?;
        for inner in 0..stride {
            for k in 0..plan.line_len() {
                line_buffer[k] = data[base + k * stride + inner];
            }
            if normalize {
                plan.inverse_in_place(&mut line_buffer[..plan.line_len()], scratch)
            } else {
                plan.backward_in_place(&mut line_buffer[..plan.line_len()], scratch)
            }
            .map_err(FftError::LocalC2c)?;
            for k in 0..plan.line_len() {
                data[base + k * stride + inner] = line_buffer[k];
            }
        }
    }
    Ok(())
}

fn execute_complex_forward<R: FftReal, const N: usize, const M: usize>(
    local: &LocalTransform<R>,
    pencil: &Pencil<N, M>,
    axis: usize,
    source: &[Complex<R>],
    destination: &mut [Complex<R>],
    scratch: &mut [Complex<R>],
    line_buffer: &mut [Complex<R>],
) -> Result<(), FftError> {
    match local {
        LocalTransform::Identity => {
            if source.len() != destination.len() {
                return Err(FftError::PreparationFailed);
            }
            destination.copy_from_slice(source);
            Ok(())
        }
        LocalTransform::Complex(plan) => execute_strided_complex_forward(
            plan,
            pencil,
            axis,
            source,
            destination,
            scratch,
            line_buffer,
        ),
        LocalTransform::RealComplex(_) => Err(FftError::PreparationFailed),
    }
}

fn execute_complex_forward_in_place<R: FftReal, const N: usize, const M: usize>(
    local: &LocalTransform<R>,
    pencil: &Pencil<N, M>,
    axis: usize,
    data: &mut [Complex<R>],
    scratch: &mut [Complex<R>],
    line_buffer: &mut [Complex<R>],
) -> Result<(), FftError> {
    match local {
        LocalTransform::Identity => Ok(()),
        LocalTransform::Complex(plan) => {
            execute_strided_complex_forward_in_place(plan, pencil, axis, data, scratch, line_buffer)
        }
        LocalTransform::RealComplex(_) => Err(FftError::PreparationFailed),
    }
}

fn execute_complex_reverse<R: FftReal, const N: usize, const M: usize>(
    local: &LocalTransform<R>,
    pencil: &Pencil<N, M>,
    axis: usize,
    source: &[Complex<R>],
    destination: &mut [Complex<R>],
    scratch: &mut [Complex<R>],
    line_buffer: &mut [Complex<R>],
    normalize: bool,
) -> Result<(), FftError> {
    match local {
        LocalTransform::Identity => {
            if source.len() != destination.len() {
                return Err(FftError::PreparationFailed);
            }
            destination.copy_from_slice(source);
            Ok(())
        }
        LocalTransform::Complex(plan) => execute_strided_complex_reverse(
            plan,
            pencil,
            axis,
            source,
            destination,
            scratch,
            line_buffer,
            normalize,
        ),
        LocalTransform::RealComplex(_) => Err(FftError::PreparationFailed),
    }
}

fn execute_complex_reverse_in_place<R: FftReal, const N: usize, const M: usize>(
    local: &LocalTransform<R>,
    pencil: &Pencil<N, M>,
    axis: usize,
    data: &mut [Complex<R>],
    scratch: &mut [Complex<R>],
    line_buffer: &mut [Complex<R>],
    normalize: bool,
) -> Result<(), FftError> {
    match local {
        LocalTransform::Identity => Ok(()),
        LocalTransform::Complex(plan) => execute_strided_complex_reverse_in_place(
            plan,
            pencil,
            axis,
            data,
            scratch,
            line_buffer,
            normalize,
        ),
        LocalTransform::RealComplex(_) => Err(FftError::PreparationFailed),
    }
}

fn execute_forward<R: FftReal, const N: usize, const M: usize>(
    core: &Arc<TransformPlanCore<R, N, M>>,
    source: &PencilArray<Complex<R>, N, M>,
    destination: &mut PencilArray<Complex<R>, N, M>,
    workspace: &mut C2cOutOfPlaceWorkspace<R, N, M>,
    mut report: Option<&mut TransformTiming<N>>,
) -> Result<(), FftError>
where
    Complex<R>: Equivalence,
{
    let source_view = source.view();
    {
        let stage = &core.stages[0];
        let fft_scratch = &mut workspace.fft_scratch;
        let line_buffer = &mut workspace.line_buffer;
        let fft_started = Instant::now();
        workspace
            .intermediate
            .overwrite_with(stage.output.as_ref(), |mut target| {
                execute_complex_forward(
                    &stage.local,
                    stage.output.as_ref(),
                    stage.axis,
                    source_view.as_slice(),
                    target.as_mut_slice(),
                    fft_scratch,
                    line_buffer,
                )
                .expect("distributed C2C forward preflight validated stage zero");
                Ok::<_, ()>(())
            })
            .expect("distributed C2C stage-zero overwrite was preflighted");
        record_fft_timing(&mut report, 0, fft_started);
    }

    execute_forward_complex_tail_timed(
        &core.stages,
        &core.transitions,
        &mut workspace.intermediate,
        &mut workspace.transpose,
        destination,
        &mut workspace.fft_scratch,
        &mut workspace.line_buffer,
        report,
    )
}

fn execute_forward_overlap<R: FftReal, const N: usize, const M: usize>(
    core: &Arc<TransformPlanCore<R, N, M>>,
    source: &PencilArray<Complex<R>, N, M>,
    destination: &mut PencilArray<Complex<R>, N, M>,
    workspace: &mut C2cOutOfPlaceWorkspace<R, N, M>,
) -> Result<(), FftOverlapError<FftError>>
where
    Complex<R>: Equivalence,
{
    let source_view = source.view();
    let stage = &core.stages[0];
    workspace
        .intermediate
        .overwrite_with(stage.output.as_ref(), |mut target| {
            execute_complex_forward(
                &stage.local,
                stage.output.as_ref(),
                stage.axis,
                source_view.as_slice(),
                target.as_mut_slice(),
                &mut workspace.fft_scratch,
                &mut workspace.line_buffer,
            )
        })
        .map_err(|error| match error {
            OverwriteError::Writer(error) => error,
            OverwriteError::Array(error) => FftError::Array(error),
        })?;
    for (index, transition) in core.transitions.iter().enumerate() {
        let stage = &core.stages[index + 1];
        match &transition.forward {
            C2cTransition::PointToPoint(plan) => {
                let callback = |data: &mut [Complex<R>]| {
                    #[cfg(test)]
                    consume_c2c_callback_injection()?;
                    execute_complex_forward_in_place(
                        &stage.local,
                        stage.output.as_ref(),
                        stage.axis,
                        data,
                        &mut workspace.fft_scratch,
                        &mut workspace.line_buffer,
                    )
                };
                plan.execute_in_place_with_callback(
                    &mut workspace.intermediate,
                    &mut workspace.transpose,
                    callback,
                )
                .map_err(map_overlap)?;
            }
            C2cTransition::Identity | C2cTransition::Local(_) => {
                execute_transition(
                    &transition.forward,
                    &mut workspace.intermediate,
                    &mut workspace.transpose,
                )?;
                let mut active = workspace
                    .intermediate
                    .active_view_mut()
                    .map_err(FftError::Array)?;
                execute_complex_forward_in_place(
                    &stage.local,
                    stage.output.as_ref(),
                    stage.axis,
                    active.as_mut_slice(),
                    &mut workspace.fft_scratch,
                    &mut workspace.line_buffer,
                )?;
            }
            C2cTransition::AllToAllv(_) => return Err(FftOverlapError::UnsupportedTransport),
        }
    }
    let active = workspace
        .intermediate
        .active_view()
        .map_err(FftError::Array)?;
    destination
        .view_mut()
        .as_mut_slice()
        .copy_from_slice(active.as_slice());
    Ok(())
}

fn execute_inverse_overlap<R: FftReal, const N: usize, const M: usize>(
    core: &Arc<TransformPlanCore<R, N, M>>,
    source: &PencilArray<Complex<R>, N, M>,
    destination: &mut PencilArray<Complex<R>, N, M>,
    workspace: &mut C2cOutOfPlaceWorkspace<R, N, M>,
    normalize_inverse: bool,
) -> Result<(), FftOverlapError<FftError>>
where
    Complex<R>: Equivalence,
{
    let source_view = source.view();
    let last = core.stages.len() - 1;
    let stage = &core.stages[last];
    workspace
        .intermediate
        .overwrite_with(stage.output.as_ref(), |mut target| {
            execute_complex_reverse(
                &stage.local,
                stage.input.as_ref(),
                stage.axis,
                source_view.as_slice(),
                target.as_mut_slice(),
                &mut workspace.fft_scratch,
                &mut workspace.line_buffer,
                normalize_inverse,
            )
        })
        .map_err(|error| match error {
            OverwriteError::Writer(error) => error,
            OverwriteError::Array(error) => FftError::Array(error),
        })?;
    for (index, transition) in core.transitions.iter().enumerate().rev() {
        let stage = &core.stages[index];
        match &transition.backward {
            C2cTransition::PointToPoint(plan) => {
                let callback = |data: &mut [Complex<R>]| {
                    #[cfg(test)]
                    consume_c2c_callback_injection()?;
                    execute_complex_reverse_in_place(
                        &stage.local,
                        stage.input.as_ref(),
                        stage.axis,
                        data,
                        &mut workspace.fft_scratch,
                        &mut workspace.line_buffer,
                        normalize_inverse,
                    )
                };
                plan.execute_in_place_with_callback(
                    &mut workspace.intermediate,
                    &mut workspace.transpose,
                    callback,
                )
                .map_err(map_overlap)?;
            }
            C2cTransition::Identity | C2cTransition::Local(_) => {
                execute_transition(
                    &transition.backward,
                    &mut workspace.intermediate,
                    &mut workspace.transpose,
                )?;
                let mut active = workspace
                    .intermediate
                    .active_view_mut()
                    .map_err(FftError::Array)?;
                execute_complex_reverse_in_place(
                    &stage.local,
                    stage.input.as_ref(),
                    stage.axis,
                    active.as_mut_slice(),
                    &mut workspace.fft_scratch,
                    &mut workspace.line_buffer,
                    normalize_inverse,
                )?;
            }
            C2cTransition::AllToAllv(_) => return Err(FftOverlapError::UnsupportedTransport),
        }
    }
    let active = workspace
        .intermediate
        .active_view()
        .map_err(FftError::Array)?;
    destination
        .view_mut()
        .as_mut_slice()
        .copy_from_slice(active.as_slice());
    Ok(())
}

fn map_overlap<E: Into<FftError>>(error: OverlapError<E>) -> FftOverlapError<FftError> {
    FftOverlapError::Overlap(match error {
        OverlapError::Transpose(error) => OverlapError::Transpose(error),
        OverlapError::Callback(error) => OverlapError::Callback(error.into()),
        OverlapError::PeerPanicked => OverlapError::PeerPanicked,
        OverlapError::PeerCallbackFailed => OverlapError::PeerCallbackFailed,
        OverlapError::CollectivePreconditionFailed => OverlapError::CollectivePreconditionFailed,
    })
}

fn execute_inverse<R: FftReal, const N: usize, const M: usize>(
    core: &Arc<TransformPlanCore<R, N, M>>,
    source: &PencilArray<Complex<R>, N, M>,
    destination: &mut PencilArray<Complex<R>, N, M>,
    workspace: &mut C2cOutOfPlaceWorkspace<R, N, M>,
    normalize_inverse: bool,
    mut report: Option<&mut TransformTiming<N>>,
) -> Result<(), FftError>
where
    Complex<R>: Equivalence,
{
    execute_inverse_complex_tail_timed(
        &core.stages,
        &core.transitions,
        source,
        &mut workspace.intermediate,
        &mut workspace.transpose,
        &mut workspace.fft_scratch,
        &mut workspace.line_buffer,
        normalize_inverse,
        report.as_deref_mut(),
    )?;
    let stage = &core.stages[0];
    let active = workspace
        .intermediate
        .active_view()
        .expect("distributed C2C inverse final active layout was preflighted");
    let mut destination_view = destination.view_mut();
    let fft_started = Instant::now();
    execute_complex_reverse(
        &stage.local,
        stage.input.as_ref(),
        stage.axis,
        active.as_slice(),
        destination_view.as_mut_slice(),
        &mut workspace.fft_scratch,
        &mut workspace.line_buffer,
        normalize_inverse,
    )
    .expect("distributed C2C final inverse stage was preflighted");
    record_fft_timing(&mut report, 0, fft_started);
    Ok(())
}

fn execute_forward_complex_tail_timed<R: FftReal, const N: usize, const M: usize>(
    stages: &[TransformStage<R, N, M>],
    transitions: &[C2cStageTransition<N, M>],
    intermediate: &mut ManyPencilArray<Complex<R>, N, M>,
    transpose: &mut TransposeWorkspace<Complex<R>>,
    destination: &mut PencilArray<Complex<R>, N, M>,
    fft_scratch: &mut [Complex<R>],
    line_buffer: &mut [Complex<R>],
    mut report: Option<&mut TransformTiming<N>>,
) -> Result<(), FftError>
where
    Complex<R>: Equivalence,
{
    let stage_count = stages.len();
    if stage_count == 0 || transitions.len() + 1 != stage_count {
        return Err(FftError::PreparationFailed);
    }
    if stage_count == 1 {
        let active = intermediate
            .active_view()
            .expect("single complex-tail active layout was preflighted");
        let mut destination_view = destination.view_mut();
        if active.as_slice().len() != destination_view.as_slice().len() {
            return Err(FftError::PreparationFailed);
        }
        destination_view
            .as_mut_slice()
            .copy_from_slice(active.as_slice());
        return Ok(());
    }
    for index in 0..transitions.len() {
        execute_transition_timed(
            &transitions[index].forward,
            intermediate,
            transpose,
            report.as_deref_mut(),
            index,
        )?;
        let stage = &stages[index + 1];
        if index + 1 == stage_count - 1 {
            let active = intermediate
                .active_view()
                .expect("complex tail final active layout was preflighted");
            let mut destination_view = destination.view_mut();
            let fft_started = Instant::now();
            execute_complex_forward(
                &stage.local,
                stage.output.as_ref(),
                stage.axis,
                active.as_slice(),
                destination_view.as_mut_slice(),
                fft_scratch,
                line_buffer,
            )
            .expect("complex tail final forward stage was preflighted");
            record_fft_timing(&mut report, index + 1, fft_started);
        } else {
            let mut active = intermediate
                .active_view_mut()
                .expect("complex tail middle active layout was preflighted");
            let fft_started = Instant::now();
            execute_complex_forward_in_place(
                &stage.local,
                stage.output.as_ref(),
                stage.axis,
                active.as_mut_slice(),
                fft_scratch,
                line_buffer,
            )
            .expect("complex tail middle forward stage was preflighted");
            record_fft_timing(&mut report, index + 1, fft_started);
        }
    }
    Ok(())
}

fn execute_inverse_complex_tail_timed<R: FftReal, const N: usize, const M: usize>(
    stages: &[TransformStage<R, N, M>],
    transitions: &[C2cStageTransition<N, M>],
    source: &PencilArray<Complex<R>, N, M>,
    intermediate: &mut ManyPencilArray<Complex<R>, N, M>,
    transpose: &mut TransposeWorkspace<Complex<R>>,
    fft_scratch: &mut [Complex<R>],
    line_buffer: &mut [Complex<R>],
    normalize_inverse: bool,
    mut report: Option<&mut TransformTiming<N>>,
) -> Result<(), FftError>
where
    Complex<R>: Equivalence,
{
    let stage_count = stages.len();
    if stage_count == 0 || transitions.len() + 1 != stage_count {
        return Err(FftError::PreparationFailed);
    }
    let source_view = source.view();
    let last = stage_count - 1;
    {
        let stage = &stages[last];
        let fft_started = Instant::now();
        intermediate
            .overwrite_with(stage.output.as_ref(), |mut target| {
                if stage_count == 1 {
                    if target.as_mut_slice().len() != source_view.as_slice().len() {
                        return Err(());
                    }
                    target
                        .as_mut_slice()
                        .copy_from_slice(source_view.as_slice());
                    return Ok(());
                }
                execute_complex_reverse(
                    &stage.local,
                    stage.input.as_ref(),
                    stage.axis,
                    source_view.as_slice(),
                    target.as_mut_slice(),
                    fft_scratch,
                    line_buffer,
                    normalize_inverse,
                )
                .expect("complex tail first inverse stage was preflighted");
                Ok::<_, ()>(())
            })
            .expect("complex tail inverse overwrite was preflighted");
        if stage_count > 1 {
            record_fft_timing(&mut report, last, fft_started);
        }
    }
    for index in (0..transitions.len()).rev() {
        execute_transition_timed(
            &transitions[index].backward,
            intermediate,
            transpose,
            report.as_deref_mut(),
            index,
        )?;
        if index != 0 {
            let stage = &stages[index];
            let mut active = intermediate
                .active_view_mut()
                .expect("complex tail inverse middle layout was preflighted");
            let fft_started = Instant::now();
            execute_complex_reverse_in_place(
                &stage.local,
                stage.input.as_ref(),
                stage.axis,
                active.as_mut_slice(),
                fft_scratch,
                line_buffer,
                normalize_inverse,
            )
            .expect("complex tail middle inverse stage was preflighted");
            record_fft_timing(&mut report, index, fft_started);
        }
    }
    Ok(())
}

fn execute_forward_in_place<R: FftReal, const N: usize, const M: usize>(
    core: &Arc<TransformPlanCore<R, N, M>>,
    array: &mut ManyPencilArray<Complex<R>, N, M>,
    transpose: &mut TransposeWorkspace<Complex<R>>,
    fft_scratch: &mut [Complex<R>],
    line_buffer: &mut [Complex<R>],
    mut report: Option<&mut TransformTiming<N>>,
) -> Result<(), FftError>
where
    Complex<R>: Equivalence,
{
    {
        let stage = &core.stages[0];
        let mut active = array.active_view_mut().map_err(FftError::Array)?;
        let fft_started = Instant::now();
        execute_complex_forward_in_place(
            &stage.local,
            stage.output.as_ref(),
            stage.axis,
            active.as_mut_slice(),
            fft_scratch,
            line_buffer,
        )?;
        record_fft_timing(&mut report, 0, fft_started);
    }

    for (index, transition) in core.transitions.iter().enumerate() {
        execute_transition_timed(
            &transition.forward,
            array,
            transpose,
            report.as_deref_mut(),
            index,
        )?;
        let stage = &core.stages[index + 1];
        let mut active = array.active_view_mut().map_err(FftError::Array)?;
        let fft_started = Instant::now();
        execute_complex_forward_in_place(
            &stage.local,
            stage.output.as_ref(),
            stage.axis,
            active.as_mut_slice(),
            fft_scratch,
            line_buffer,
        )?;
        record_fft_timing(&mut report, index + 1, fft_started);
    }
    Ok(())
}

fn execute_inverse_in_place<R: FftReal, const N: usize, const M: usize>(
    core: &Arc<TransformPlanCore<R, N, M>>,
    array: &mut ManyPencilArray<Complex<R>, N, M>,
    transpose: &mut TransposeWorkspace<Complex<R>>,
    fft_scratch: &mut [Complex<R>],
    line_buffer: &mut [Complex<R>],
    normalize_inverse: bool,
    mut report: Option<&mut TransformTiming<N>>,
) -> Result<(), FftError>
where
    Complex<R>: Equivalence,
{
    {
        let last = core.stages.len() - 1;
        let stage = &core.stages[last];
        let mut active = array.active_view_mut().map_err(FftError::Array)?;
        let fft_started = Instant::now();
        execute_complex_reverse_in_place(
            &stage.local,
            stage.input.as_ref(),
            stage.axis,
            active.as_mut_slice(),
            fft_scratch,
            line_buffer,
            normalize_inverse,
        )?;
        record_fft_timing(&mut report, core.stages.len() - 1, fft_started);
    }

    for (index, transition) in core.transitions.iter().enumerate().rev() {
        execute_transition_timed(
            &transition.backward,
            array,
            transpose,
            report.as_deref_mut(),
            index,
        )?;
        let stage = &core.stages[index];
        let mut active = array.active_view_mut().map_err(FftError::Array)?;
        let fft_started = Instant::now();
        execute_complex_reverse_in_place(
            &stage.local,
            stage.input.as_ref(),
            stage.axis,
            active.as_mut_slice(),
            fft_scratch,
            line_buffer,
            normalize_inverse,
        )?;
        record_fft_timing(&mut report, index, fft_started);
    }
    Ok(())
}

fn run_in_place_transaction<R: FftReal, const N: usize, const M: usize, F>(
    array: &mut C2cInPlaceArray<R, N, M>,
    workspace: &mut C2cInPlaceWorkspace<R, N, M>,
    target: C2cState,
    body: F,
) -> Result<(), FftError>
where
    F: FnOnce(
        &mut C2cInPlaceArray<R, N, M>,
        &mut C2cInPlaceWorkspace<R, N, M>,
    ) -> Result<(), FftError>,
{
    array.state = C2cState::Poisoned;
    let result = body(array, workspace);
    if result.is_ok() {
        array.state = target;
    }
    result
}

fn record_fft_timing<const N: usize>(
    report: &mut Option<&mut TransformTiming<N>>,
    index: usize,
    started: Instant,
) {
    if let Some(timing) = report.as_deref_mut() {
        timing.stages[index].fft = started.elapsed();
        timing.stages[index].fft_calls += 1;
        timing.stages[index].total = timing.stages[index].fft + timing.stages[index].transpose;
    }
}

fn execute_transition<T: Equivalence + Copy + Clone, const N: usize, const M: usize>(
    transition: &C2cTransition<N, M>,
    intermediate: &mut ManyPencilArray<T, N, M>,
    workspace: &mut TransposeWorkspace<T>,
) -> Result<(), FftError> {
    execute_transition_timed(transition, intermediate, workspace, None, 0)
}

fn execute_transition_timed<T: Equivalence + Copy + Clone, const N: usize, const M: usize>(
    transition: &C2cTransition<N, M>,
    intermediate: &mut ManyPencilArray<T, N, M>,
    workspace: &mut TransposeWorkspace<T>,
    mut report: Option<&mut TransformTiming<N>>,
    stage_index: usize,
) -> Result<(), FftError> {
    let started = Instant::now();
    let result = match transition {
        C2cTransition::Identity => Ok(()),
        C2cTransition::Local(plan) => {
            plan.execute_in_place_with_transpose_workspace(intermediate, workspace)
                .expect("distributed C2C local transition was preflighted");
            Ok(())
        }
        C2cTransition::AllToAllv(plan) => {
            if report.is_some() {
                plan.execute_in_place_with_timing(intermediate, workspace)
                    .map(|phases| {
                        report.as_deref_mut().expect("timed transition").stages[stage_index]
                            .communication = phases;
                    })
                    .map_err(FftError::Transpose)
            } else {
                plan.execute_in_place(intermediate, workspace)
                    .map(|_| ())
                    .map_err(FftError::Transpose)
            }
        }
        C2cTransition::PointToPoint(plan) => {
            if report.is_some() {
                plan.execute_in_place_with_timing(intermediate, workspace)
                    .map(|phases| {
                        report.as_deref_mut().expect("timed transition").stages[stage_index]
                            .communication = phases;
                    })
                    .map_err(FftError::Transpose)
            } else {
                plan.execute_in_place(intermediate, workspace)
                    .map(|_| ())
                    .map_err(FftError::Transpose)
            }
        }
    };
    if let Some(timing) = report {
        timing.stages[stage_index].transpose = started.elapsed();
        timing.stages[stage_index].transition_calls += 1;
        timing.stages[stage_index].total =
            timing.stages[stage_index].fft + timing.stages[stage_index].transpose;
    }
    result
}

fn descriptor_len<const N: usize, const M: usize>(extra_shape: &ExtraShape) -> Option<usize> {
    N.checked_add(M)?
        .checked_add(3)?
        .checked_add(extra_shape.dimensions().len())?
        .checked_add(N)?
        .checked_add(2)
}

fn build_descriptor<R: FftReal, const N: usize, const M: usize>(
    topology: &MpiTopology<M>,
    global_shape: [usize; N],
    extra_shape: &ExtraShape,
    selection: AxisSelection<N>,
    value_kind: u64,
    layout: DistributedLayout,
) -> Result<Vec<u64>, ()> {
    let length = descriptor_len::<N, M>(extra_shape).ok_or(())?;
    let mut descriptor = Vec::new();
    descriptor.try_reserve_exact(length).map_err(|_| ())?;
    append_usizes(&mut descriptor, &global_shape)?;
    append_usizes(&mut descriptor, topology.process_grid())?;
    append_shape(&mut descriptor, extra_shape)?;
    descriptor.extend(selection.mask().iter().map(|&selected| u64::from(selected)));
    descriptor.push(value_kind);
    descriptor.push(u64::try_from(size_of::<R>()).map_err(|_| ())?);
    descriptor.push(layout.transpose_method.descriptor_word());
    descriptor.push(u64::from(layout.permute_dims));
    if descriptor.len() != length {
        return Err(());
    }
    Ok(descriptor)
}

fn append_shape(descriptor: &mut Vec<u64>, shape: &ExtraShape) -> Result<(), ()> {
    append_usizes(descriptor, &[shape.dimensions().len()])?;
    append_usizes(descriptor, shape.dimensions())
}

fn append_usizes(descriptor: &mut Vec<u64>, values: &[usize]) -> Result<(), ()> {
    for &value in values {
        descriptor.push(u64::try_from(value).map_err(|_| ())?);
    }
    Ok(())
}

fn agree_header<C: CommunicatorCollectives>(comm: &C, header: [u64; HEADER_WORDS]) -> bool {
    let mut minimum = [0u64; HEADER_WORDS];
    let mut maximum = [0u64; HEADER_WORDS];
    comm.all_reduce_into(&header[..], &mut minimum[..], SystemOperation::min());
    comm.all_reduce_into(&header[..], &mut maximum[..], SystemOperation::max());
    minimum == maximum
}

fn agree_execution_descriptor_ref<const N: usize, const M: usize>(
    communicator: &mpi::topology::CartesianCommunicator,
    operation: u64,
    descriptor: &[u64],
) -> Result<(), FftError> {
    let header = [
        DESCRIPTOR_SCHEMA,
        operation,
        u64::try_from(N).unwrap_or(INVALID_WORD),
        u64::try_from(M).unwrap_or(INVALID_WORD),
        u64::try_from(descriptor.len()).unwrap_or(INVALID_WORD),
    ];
    if !agree_header(communicator, header) {
        return Err(FftError::CollectiveDescriptorMismatch);
    }
    collective_descriptor_ref(communicator, Some(descriptor), Some(descriptor.len()))
}

fn agree_execution_descriptor<R: FftReal, const N: usize, const M: usize>(
    communicator: &mpi::topology::CartesianCommunicator,
    operation: u64,
    core: &TransformPlanCore<R, N, M>,
) -> Result<(), FftError> {
    agree_execution_descriptor_ref::<N, M>(communicator, operation, &core.descriptor)
}

fn validate_workspace_lengths_values(
    fft_scratch_len: usize,
    required_fft_scratch_len: usize,
    transpose_send_len: usize,
    required_transpose_send_len: usize,
    transpose_receive_len: usize,
    required_transpose_receive_len: usize,
) -> Result<(), FftError> {
    for (actual, required, kind) in [
        (fft_scratch_len, required_fft_scratch_len, "FFT scratch"),
        (
            transpose_send_len,
            required_transpose_send_len,
            "transpose send",
        ),
        (
            transpose_receive_len,
            required_transpose_receive_len,
            "transpose receive",
        ),
    ] {
        if actual < required {
            return Err(FftError::WorkspaceTooSmall {
                kind,
                required,
                actual,
            });
        }
    }
    Ok(())
}

fn validate_workspace_lengths<R: FftReal, const N: usize, const M: usize>(
    core: &TransformPlanCore<R, N, M>,
    fft_scratch_len: usize,
    transpose_send_len: usize,
    transpose_receive_len: usize,
) -> Result<(), FftError> {
    validate_workspace_lengths_values(
        fft_scratch_len,
        core.fft_scratch_len,
        transpose_send_len,
        core.transpose_send_len,
        transpose_receive_len,
        core.transpose_receive_len,
    )
}

fn validate_out_of_place<R: FftReal, T, U, V, const N: usize, const M: usize>(
    core: &Arc<TransformPlanCore<R, N, M>>,
    workspace_core: &Arc<TransformPlanCore<R, N, M>>,
    direction: Direction,
    source: &PencilArray<T, N, M>,
    destination: &PencilArray<U, N, M>,
    intermediate: &ManyPencilArray<V, N, M>,
    lengths: (usize, usize, usize),
) -> Result<(), FftError> {
    if !Arc::ptr_eq(workspace_core, core) {
        return Err(FftError::WorkspaceMismatch);
    }
    let input = &core.stages[0].input;
    let output = &core
        .stages
        .last()
        .expect("distributed FFT has at least two stages")
        .output;
    let (expected_source, expected_destination) = match direction {
        Direction::Forward => (input, output),
        Direction::Inverse | Direction::Backward => (output, input),
    };
    if (core.strict_array_identity && !Arc::ptr_eq(source.pencil(), expected_source))
        || !source.pencil().same_layout(expected_source.as_ref())
    {
        return Err(FftError::InputLayoutMismatch);
    }
    if (core.strict_array_identity && !Arc::ptr_eq(destination.pencil(), expected_destination))
        || !destination
            .pencil()
            .same_layout(expected_destination.as_ref())
    {
        return Err(FftError::OutputLayoutMismatch);
    }
    if source.extra_shape() != &core.extra_shape || destination.extra_shape() != &core.extra_shape {
        return Err(FftError::ExtraShapeMismatch);
    }
    let (fft_scratch_len, transpose_send_len, transpose_receive_len) = lengths;
    validate_workspace_lengths_values(
        fft_scratch_len,
        core.fft_scratch_len,
        transpose_send_len,
        core.transpose_send_len,
        transpose_receive_len,
        core.transpose_receive_len,
    )?;
    if intermediate.extra_shape() != &core.extra_shape {
        return Err(FftError::WorkspaceMismatch);
    }
    let active = intermediate.active_pencil()?;
    if !core
        .stages
        .iter()
        .any(|stage| active.same_layout(stage.output.as_ref()))
    {
        return Err(FftError::WorkspaceMismatch);
    }
    Ok(())
}

fn collective_valid<C: CommunicatorCollectives>(comm: &C, valid: bool) -> bool {
    let value = i32::from(valid);
    let mut result = 0i32;
    comm.all_reduce_into(&value, &mut result, SystemOperation::min());
    result != 0
}

fn agree_result<C: CommunicatorCollectives, T, E>(comm: &C, result: Result<T, E>) -> Result<T, E>
where
    E: From<FftError>,
{
    if !collective_valid(comm, result.is_ok()) {
        return Err(result
            .err()
            .unwrap_or_else(|| E::from(FftError::CollectivePreconditionFailed)));
    }
    result
}

fn collective_descriptor<C: CommunicatorCollectives>(
    comm: &C,
    descriptor: Option<Vec<u64>>,
    expected_len: Option<usize>,
) -> Result<Vec<u64>, FftError> {
    let descriptor_ref = descriptor.as_deref();
    collective_descriptor_ref(comm, descriptor_ref, expected_len)?;
    descriptor.ok_or(FftError::PreparationFailed)
}

fn collective_descriptor_ref<C: CommunicatorCollectives>(
    comm: &C,
    descriptor: Option<&[u64]>,
    expected_len: Option<usize>,
) -> Result<(), FftError> {
    let mut minimum = None;
    let mut maximum = None;
    let ready = match (descriptor, expected_len) {
        (Some(descriptor), Some(expected_len)) if descriptor.len() == expected_len => {
            if Count::try_from(expected_len).is_err() {
                false
            } else {
                let mut local_minimum = Vec::new();
                let mut local_maximum = Vec::new();
                let allocated = local_minimum.try_reserve_exact(expected_len).is_ok()
                    && local_maximum.try_reserve_exact(expected_len).is_ok();
                if allocated {
                    local_minimum.extend(std::iter::repeat_n(0u64, expected_len));
                    local_maximum.extend(std::iter::repeat_n(0u64, expected_len));
                    minimum = Some(local_minimum);
                    maximum = Some(local_maximum);
                }
                allocated
            }
        }
        _ => false,
    };
    if !collective_valid(comm, ready) {
        return Err(FftError::PreparationFailed);
    }
    let descriptor = descriptor.expect("collective descriptor readiness succeeded");
    let minimum = minimum
        .as_mut()
        .expect("collective descriptor minimum was allocated");
    let maximum = maximum
        .as_mut()
        .expect("collective descriptor maximum was allocated");
    comm.all_reduce_into(descriptor, minimum.as_mut_slice(), SystemOperation::min());
    comm.all_reduce_into(descriptor, maximum.as_mut_slice(), SystemOperation::max());
    if minimum != maximum {
        return Err(FftError::CollectiveDescriptorMismatch);
    }
    Ok(())
}

fn initialized_vec<T: Clone>(length: usize, value: T) -> Result<Vec<T>, FftError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| FftError::AllocationFailed { required: length })?;
    values.extend(std::iter::repeat_n(value, length));
    Ok(values)
}

fn map_array_allocation(error: ArrayError) -> FftError {
    match error {
        ArrayError::AllocationFailed { required } => FftError::AllocationFailed { required },
        other => FftError::Array(other),
    }
}

fn zero_complex<R: FftReal>() -> Complex<R> {
    Complex::new(R::zero(), R::zero())
}

#[cfg(test)]
mod tests {
    use std::{panic::AssertUnwindSafe, sync::Arc};

    use super::{
        C2C_CALLBACK_INJECTION, C2cCallbackInjection, C2cInPlaceArray, C2cInPlaceWorkspace,
        C2cPlan, C2cState, C2cTransition, Complex, Direction, DistributedLayout, ExtraShape,
        FftError, FftOverlapError, LocalC2cError, LocalC2cPlan, OPERATION_BACKWARD,
        OPERATION_BACKWARD_IN_PLACE, OPERATION_DHT_BACKWARD, OPERATION_DHT_BACKWARD_IN_PLACE,
        OPERATION_DHT_FORWARD, OPERATION_DHT_FORWARD_IN_PLACE, OPERATION_DHT_INVERSE,
        OPERATION_DHT_INVERSE_IN_PLACE, OPERATION_DHT_PLAN, OPERATION_FORWARD,
        OPERATION_FORWARD_IN_PLACE, OPERATION_INVERSE, OPERATION_INVERSE_IN_PLACE, OPERATION_PLAN,
        OPERATION_R2C_BACKWARD, OPERATION_R2C_BACKWARD_IN_PLACE, OPERATION_R2C_FORWARD,
        OPERATION_R2C_FORWARD_IN_PLACE, OPERATION_R2C_INVERSE, OPERATION_R2C_INVERSE_IN_PLACE,
        OPERATION_R2C_PLAN, OPERATION_R2R_BACKWARD, OPERATION_R2R_BACKWARD_IN_PLACE,
        OPERATION_R2R_FORWARD, OPERATION_R2R_FORWARD_IN_PLACE, OPERATION_R2R_INVERSE,
        OPERATION_R2R_INVERSE_IN_PLACE, OPERATION_R2R_PLAN, OverlapError, R2C_CALLBACK_INJECTION,
        R2cCallbackInjection, R2cError, R2cPlan, TransposeMethod, descriptor_len,
        run_in_place_transaction,
    };
    use crate::{
        AxisR2rKind, AxisSelection, AxisTransform, MixedC2cPlan, MixedError, MixedR2cPlan, R2cState,
    };
    use mpi::{collective::CommunicatorCollectives, topology::Communicator};
    use pencil_array::{ArrayError, MpiTopology, TransposeWorkspace};

    #[test]
    fn axis_selection_validates_and_ignores_input_order() {
        let selection = AxisSelection::<4>::from_indices([3, 0]).unwrap();
        assert!(selection.contains(0));
        assert!(selection.contains(3));
        assert!(!selection.contains(1));
        assert_eq!(selection, AxisSelection::<4>::from_indices([0, 3]).unwrap());
        assert!(AxisSelection::<4>::from_indices([3, 3]).is_err());
        assert!(AxisSelection::<4>::from_indices([4]).is_err());
        assert!(AxisSelection::<4>::empty().is_empty());
        assert!(AxisSelection::<4>::all().is_all());
    }

    #[test]
    fn protocol_words_and_minimal_descriptor_length_are_stable() {
        assert_eq!(
            (OPERATION_PLAN, OPERATION_FORWARD, OPERATION_INVERSE),
            (7, 8, 9)
        );
        let scalar_len = descriptor_len::<2, 1>(&ExtraShape::scalar()).unwrap();
        let batched_len = descriptor_len::<4, 2>(&ExtraShape::new([2, 3]).unwrap()).unwrap();
        assert_eq!(scalar_len, 10);
        assert_eq!(batched_len, 17);
        assert_eq!(
            batched_len - descriptor_len::<4, 2>(&ExtraShape::scalar()).unwrap(),
            2,
        );
        assert_eq!(
            (
                TransposeMethod::AllToAllv.descriptor_word(),
                TransposeMethod::PointToPoint.descriptor_word(),
            ),
            (0, 1),
        );
    }

    #[test]
    fn in_place_operation_words_are_reserved_after_out_of_place_words() {
        assert_eq!(
            (
                OPERATION_PLAN,
                OPERATION_FORWARD,
                OPERATION_INVERSE,
                OPERATION_FORWARD_IN_PLACE,
                OPERATION_INVERSE_IN_PLACE,
            ),
            (7, 8, 9, 10, 11)
        );
        assert_eq!(
            (
                OPERATION_R2C_PLAN,
                OPERATION_R2C_FORWARD,
                OPERATION_R2C_INVERSE,
                OPERATION_BACKWARD,
                OPERATION_BACKWARD_IN_PLACE,
                OPERATION_R2C_BACKWARD,
            ),
            (12, 13, 14, 15, 16, 17)
        );
        assert_eq!(
            (
                OPERATION_R2R_PLAN,
                OPERATION_R2R_FORWARD,
                OPERATION_R2R_INVERSE,
                OPERATION_R2R_BACKWARD,
                OPERATION_R2R_FORWARD_IN_PLACE,
                OPERATION_R2R_INVERSE_IN_PLACE,
                OPERATION_R2R_BACKWARD_IN_PLACE,
            ),
            (18, 19, 20, 21, 22, 23, 24)
        );
        assert_eq!(
            (
                OPERATION_R2C_FORWARD_IN_PLACE,
                OPERATION_R2C_INVERSE_IN_PLACE,
                OPERATION_R2C_BACKWARD_IN_PLACE,
            ),
            (25, 26, 27)
        );
        assert_eq!(
            (
                OPERATION_DHT_PLAN,
                OPERATION_DHT_FORWARD,
                OPERATION_DHT_INVERSE,
                OPERATION_DHT_BACKWARD,
                OPERATION_DHT_FORWARD_IN_PLACE,
                OPERATION_DHT_INVERSE_IN_PLACE,
                OPERATION_DHT_BACKWARD_IN_PLACE,
            ),
            (28, 29, 30, 31, 32, 33, 34)
        );
    }

    #[test]
    fn in_place_transaction_poison_survives_error_and_panic() {
        let _mpi_test_lock = super::MPI_TEST_LOCK
            .get_or_init(|| std::sync::Mutex::new(()))
            .lock()
            .unwrap();
        let universe = mpi::initialize().expect("MPI initialization failed");
        let world = universe.world();
        {
            let topology = MpiTopology::<1>::new(&world, [world.size() as usize]).unwrap();
            let plan = C2cPlan::<f64, 2, 1>::from_shape_with_method(
                Arc::clone(&topology),
                [2, 3],
                ExtraShape::scalar(),
                TransposeMethod::PointToPoint,
            )
            .unwrap();
            let source = plan.allocate_input().unwrap();
            let before = source.as_slice().to_vec();
            for injection in [C2cCallbackInjection::Error, C2cCallbackInjection::Panic] {
                let mut destination = plan.allocate_output().unwrap();
                let mut workspace = plan.allocate_out_of_place_workspace().unwrap();
                if world.rank() == 0 {
                    C2C_CALLBACK_INJECTION.with(|slot| slot.set(Some(injection)));
                }
                let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                    plan.forward_with_overlap(&source, &mut destination, &mut workspace)
                }));
                if matches!(injection, C2cCallbackInjection::Panic) && world.rank() == 0 {
                    assert!(result.is_err());
                } else {
                    let error = result.unwrap().unwrap_err();
                    match (injection, world.rank() == 0, error) {
                        (
                            C2cCallbackInjection::Error,
                            true,
                            FftOverlapError::Overlap(OverlapError::Callback(
                                FftError::PreparationFailed,
                            )),
                        ) => {}
                        (C2cCallbackInjection::Error, false, FftOverlapError::Overlap(error)) => {
                            assert!(matches!(error, OverlapError::PeerCallbackFailed))
                        }
                        (C2cCallbackInjection::Panic, false, FftOverlapError::Overlap(error)) => {
                            assert!(matches!(error, OverlapError::PeerPanicked))
                        }
                        _ => panic!("unexpected callback error"),
                    }
                }
                assert!(matches!(
                    workspace.intermediate.active_view(),
                    Err(ArrayError::Poisoned)
                ));
                assert_eq!(source.as_slice(), before);
                world.barrier(); // A peer must not be stuck in another outer FFT agreement.
                let mut fresh = plan.allocate_out_of_place_workspace().unwrap();
                plan.forward_with_overlap(&source, &mut destination, &mut fresh)
                    .unwrap();
            }
            let plan = C2cPlan::<f64, 3, 1>::from_shape_with_layout(
                Arc::clone(&topology),
                [2, 3, 4],
                ExtraShape::scalar(),
                DistributedLayout {
                    transpose_method: TransposeMethod::PointToPoint,
                    permute_dims: false,
                },
            )
            .unwrap();
            let mut source = plan.allocate_input().unwrap();
            source.as_mut_slice().fill(Complex::new(1.0, 0.0));
            let mut destination = plan.allocate_output().unwrap();
            let mut workspace = plan.allocate_out_of_place_workspace().unwrap();
            plan.forward_with_overlap(&source, &mut destination, &mut workspace)
                .unwrap();
            let ranges = destination.pencil().local_ranges();
            let shape = destination.local_spatial_shape();
            for i in 0..shape[0] {
                for j in 0..shape[1] {
                    for k in 0..shape[2] {
                        let dc = i + ranges[0].start == 0
                            && j + ranges[1].start == 0
                            && k + ranges[2].start == 0;
                        let expected = if dc { 24.0 } else { 0.0 };
                        let actual = destination.get_local(&[], [i, j, k]).unwrap();
                        assert!((actual.re - expected).abs() < 1e-10 && actual.im.abs() < 1e-10);
                    }
                }
            }
            let mut recovered = plan.allocate_input().unwrap();
            plan.inverse_with_overlap(&destination, &mut recovered, &mut workspace)
                .unwrap();
            for value in recovered.as_slice() {
                assert!((value.re - 1.0).abs() < 1e-10 && value.im.abs() < 1e-10);
            }
        }
        if world.size() != 1 {
            super::r2c::run_rank_specific_conversion_failure(&world);
            return;
        }
        let result = {
            let topology = MpiTopology::<1>::new(&world, [1]).unwrap();
            let legacy_shape = C2cPlan::<f64, 2, 1>::from_shape(
                Arc::clone(&topology),
                [2, 3],
                ExtraShape::scalar(),
            )
            .unwrap();
            let legacy_source = legacy_shape.allocate_input().unwrap();
            let legacy_pencil = C2cPlan::<f64, 2, 1>::from_pencil(
                Arc::clone(legacy_shape.input_pencil()),
                ExtraShape::scalar(),
            )
            .unwrap();
            let legacy_array = C2cPlan::<f64, 2, 1>::from_array(&legacy_source).unwrap();
            for legacy_plan in [&legacy_shape, &legacy_pencil, &legacy_array] {
                assert_transition_method(legacy_plan, TransposeMethod::AllToAllv);
            }
            for method in [TransposeMethod::AllToAllv, TransposeMethod::PointToPoint] {
                let plan = C2cPlan::<f64, 2, 1>::from_shape_with_method(
                    Arc::clone(&topology),
                    [2, 3],
                    ExtraShape::scalar(),
                    method,
                )
                .unwrap();
                let explicit_pencil = C2cPlan::<f64, 2, 1>::from_pencil_with_method(
                    Arc::clone(plan.input_pencil()),
                    ExtraShape::scalar(),
                    method,
                )
                .unwrap();
                let explicit_array =
                    C2cPlan::<f64, 2, 1>::from_array_with_method(&legacy_source, method).unwrap();
                for explicit_plan in [&plan, &explicit_pencil, &explicit_array] {
                    assert_transition_method(explicit_plan, method);
                }

                let empty_plan = C2cPlan::<f64, 2, 1>::from_shape_with_selection_and_method(
                    Arc::clone(&topology),
                    [2, 3],
                    ExtraShape::scalar(),
                    AxisSelection::empty(),
                    method,
                )
                .unwrap();
                let mut empty_array = empty_plan.allocate_in_place().unwrap();
                let mut empty_workspace = empty_plan.allocate_in_place_workspace().unwrap();
                let empty_result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                    run_in_place_transaction(
                        &mut empty_array,
                        &mut empty_workspace,
                        C2cState::Output,
                        |array, workspace| {
                            assert_eq!(array.state(), C2cState::Poisoned);
                            array.array.active_view_mut().unwrap().as_mut_slice()[0].re = 7.0;
                            if let Some(value) = workspace.fft_scratch.first_mut() {
                                value.re = 9.0;
                            }
                            Err(FftError::PreparationFailed)
                        },
                    )
                }));
                assert!(matches!(empty_result, Ok(Err(FftError::PreparationFailed))));
                assert_poisoned_views_and_retries(
                    &empty_plan,
                    &mut empty_array,
                    &mut empty_workspace,
                );

                let mut short_array = plan.allocate_in_place().unwrap();
                let mut short_workspace = C2cInPlaceWorkspace {
                    core: Arc::clone(&plan.core),
                    transpose: TransposeWorkspace::from_vecs(Vec::new(), Vec::new()),
                    fft_scratch: Vec::new(),
                    line_buffer: Vec::new(),
                };
                let short_array_before = format!("{short_array:?}");
                let short_workspace_before = format!("{short_workspace:?}");
                assert!(matches!(
                    plan.forward_in_place(&mut short_array, &mut short_workspace),
                    Err(FftError::WorkspaceTooSmall { .. })
                ));
                assert_eq!(short_array.state(), C2cState::Input);
                assert_eq!(format!("{short_array:?}"), short_array_before);
                assert_eq!(format!("{short_workspace:?}"), short_workspace_before);

                for direction in [Direction::Forward, Direction::Inverse, Direction::Backward] {
                    for panic_failure in [false, true] {
                        let mut array = plan.allocate_in_place().unwrap();
                        let mut workspace = plan.allocate_in_place_workspace().unwrap();
                        if matches!(direction, Direction::Inverse | Direction::Backward) {
                            plan.forward_in_place(&mut array, &mut workspace).unwrap();
                            assert_eq!(array.state(), C2cState::Output);
                        }
                        let target = match direction {
                            Direction::Forward => C2cState::Output,
                            Direction::Inverse | Direction::Backward => C2cState::Input,
                        };
                        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                            run_in_place_transaction(
                                &mut array,
                                &mut workspace,
                                target,
                                |array, workspace| {
                                    assert_eq!(array.state(), C2cState::Poisoned);
                                    let mut view = array.array.active_view_mut().unwrap();
                                    view.as_mut_slice()[0].re = 7.0;
                                    if let Some(value) = workspace.fft_scratch.first_mut() {
                                        value.re = 9.0;
                                    }
                                    if panic_failure {
                                        panic!("in-place test panic after start");
                                    }
                                    Err(FftError::PreparationFailed)
                                },
                            )
                        }));
                        if panic_failure {
                            assert!(result.is_err());
                        } else {
                            assert!(matches!(result, Ok(Err(FftError::PreparationFailed))));
                        }
                        assert_poisoned_views_and_retries(&plan, &mut array, &mut workspace);
                    }
                }

                // Corrupt only the last immutable backend plan. The six-element
                // active buffer passes the first stage and transition, then the
                // replacement length four plan returns NonIntegralBatch.
                let mut corrupted_plan = C2cPlan::<f64, 2, 1>::from_shape_with_method(
                    Arc::clone(&topology),
                    [2, 3],
                    ExtraShape::scalar(),
                    method,
                )
                .unwrap();
                Arc::get_mut(&mut corrupted_plan.core)
                    .expect("the separate plan core has no other owners")
                    .stages
                    .last_mut()
                    .expect("the C2C route has a final stage")
                    .local = super::LocalTransform::Complex(LocalC2cPlan::new(4).unwrap());
                let mut array = corrupted_plan.allocate_in_place().unwrap();
                {
                    let mut view = array.view_mut().unwrap();
                    for (index, value) in view.as_mut_slice().iter_mut().enumerate() {
                        *value = Complex::new(index as f64 + 1.0, -(index as f64));
                    }
                }
                let mut workspace = corrupted_plan.allocate_in_place_workspace().unwrap();
                let result = corrupted_plan.forward_in_place(&mut array, &mut workspace);
                assert!(matches!(
                    result,
                    Err(FftError::LocalC2c(LocalC2cError::NonIntegralBatch))
                ));
                assert_eq!(array.state(), C2cState::Poisoned);
                assert_poisoned_views_and_retries(&corrupted_plan, &mut array, &mut workspace);

                // For raw backward, corrupt only the final stage in the
                // reverse route. The first raw stage and backward transition
                // run before stage zero returns NonIntegralBatch.
                let mut corrupted_backward_plan = C2cPlan::<f64, 2, 1>::from_shape_with_method(
                    Arc::clone(&topology),
                    [2, 3],
                    ExtraShape::scalar(),
                    method,
                )
                .unwrap();
                Arc::get_mut(&mut corrupted_backward_plan.core)
                    .expect("the separate backward plan core has no other owners")
                    .stages
                    .first_mut()
                    .expect("the C2C route has an initial stage")
                    .local = super::LocalTransform::Complex(LocalC2cPlan::new(4).unwrap());
                let mut backward_array = corrupted_backward_plan.allocate_in_place().unwrap();
                backward_array
                    .array
                    .overwrite_with(
                        corrupted_backward_plan.output_pencil().as_ref(),
                        |mut view| {
                            for (index, value) in view.as_mut_slice().iter_mut().enumerate() {
                                *value = Complex::new(index as f64 + 1.0, -(index as f64));
                            }
                            Ok::<_, ()>(())
                        },
                    )
                    .unwrap();
                backward_array.state = C2cState::Output;
                let mut backward_workspace = corrupted_backward_plan
                    .allocate_in_place_workspace()
                    .unwrap();
                let result = corrupted_backward_plan
                    .backward_in_place(&mut backward_array, &mut backward_workspace);
                assert!(matches!(
                    result,
                    Err(FftError::LocalC2c(LocalC2cError::NonIntegralBatch))
                ));
                assert_eq!(backward_array.state(), C2cState::Poisoned);
                assert_poisoned_views_and_retries(
                    &corrupted_backward_plan,
                    &mut backward_array,
                    &mut backward_workspace,
                );
            }

            let capacity_plan = R2cPlan::<f64, 2, 1>::from_shape_with_method(
                Arc::clone(&topology),
                [3, 4],
                ExtraShape::scalar(),
                TransposeMethod::AllToAllv,
            )
            .unwrap();
            let mut capacity_array = capacity_plan.allocate_in_place().unwrap();
            capacity_array
                .real_view_mut()
                .unwrap()
                .as_mut_slice()
                .fill(7.0);
            let capacity_snapshot = capacity_array.real_view().unwrap().as_slice().to_vec();
            super::r2c::force_odd_real_capacity(&mut capacity_array);
            let capacity_pointer = capacity_array.real_view().unwrap().as_slice().as_ptr();
            let mut capacity_workspace = capacity_plan.allocate_in_place_workspace().unwrap();
            let workspace_snapshot = format!("{capacity_workspace:?}");
            assert!(matches!(
                capacity_plan.forward_in_place(&mut capacity_array, &mut capacity_workspace),
                Err(R2cError::Fft(FftError::StorageLayoutMismatch))
            ));
            assert_eq!(capacity_array.state(), R2cState::RealInput);
            assert_eq!(
                capacity_array.real_view().unwrap().as_slice(),
                capacity_snapshot.as_slice()
            );
            assert_eq!(
                capacity_array.real_view().unwrap().as_slice().as_ptr(),
                capacity_pointer
            );
            assert_eq!(format!("{capacity_workspace:?}"), workspace_snapshot);

            let mut short_array = capacity_plan.allocate_in_place().unwrap();
            short_array
                .real_view_mut()
                .unwrap()
                .as_mut_slice()
                .fill(8.0);
            let short_data = short_array.real_view().unwrap().as_slice().to_vec();
            let short_pointer = short_array.real_view().unwrap().as_slice().as_ptr();
            let mut short_workspace = capacity_plan.allocate_in_place_workspace().unwrap();
            super::r2c::empty_in_place_workspace_for_test(&mut short_workspace);
            let short_workspace_snapshot = format!("{short_workspace:?}");
            assert!(matches!(
                capacity_plan.forward_in_place(&mut short_array, &mut short_workspace),
                Err(R2cError::Fft(FftError::WorkspaceTooSmall { .. }))
            ));
            assert_eq!(short_array.state(), R2cState::RealInput);
            assert_eq!(short_array.real_view().unwrap().as_slice(), short_data);
            assert_eq!(
                short_array.real_view().unwrap().as_slice().as_ptr(),
                short_pointer
            );
            assert_eq!(format!("{short_workspace:?}"), short_workspace_snapshot);

            r2c_conversion_panic_cases(&topology);
            r2c_conversion_owner_failure_cases(&topology);
            r2c_preflight_corruption_cases(&topology);
            mixed_in_place_error_cases(&topology);

            Ok::<(), ()>(())
        };
        result.unwrap();
    }

    #[test]
    #[ignore = "MPI can be initialized only once per unit-test process; run this test explicitly"]
    fn r2c_overlap_callback_error_and_panic_map_per_rank() {
        let _mpi_test_lock = super::MPI_TEST_LOCK
            .get_or_init(|| std::sync::Mutex::new(()))
            .lock()
            .unwrap();
        let universe = mpi::initialize().expect("MPI initialization failed");
        let world = universe.world();
        let topology = MpiTopology::<1>::new(&world, [world.size() as usize]).unwrap();
        let plan = R2cPlan::<f64, 2, 1>::from_shape_with_method(
            Arc::clone(&topology),
            [2, 3],
            ExtraShape::scalar(),
            TransposeMethod::PointToPoint,
        )
        .unwrap();
        let source = plan.allocate_input().unwrap();
        for injection in [R2cCallbackInjection::Error, R2cCallbackInjection::Panic] {
            let mut destination = plan.allocate_output().unwrap();
            let mut workspace = plan.allocate_workspace().unwrap();
            if world.rank() == 0 {
                R2C_CALLBACK_INJECTION.with(|slot| slot.set(Some(injection)));
            }
            let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                plan.forward_with_overlap(&source, &mut destination, &mut workspace)
            }));
            if matches!(injection, R2cCallbackInjection::Panic) && world.rank() == 0 {
                assert!(result.is_err());
            } else {
                let error = result.unwrap().unwrap_err();
                match (injection, world.rank() == 0, error) {
                    (
                        R2cCallbackInjection::Error,
                        true,
                        FftOverlapError::Overlap(OverlapError::Callback(R2cError::Fft(
                            FftError::LocalC2c(crate::LocalC2cError::ScratchTooSmall {
                                required: 7,
                                actual: 3,
                            }),
                        ))),
                    ) => {}
                    (R2cCallbackInjection::Error, false, FftOverlapError::Overlap(error)) => {
                        assert!(matches!(error, OverlapError::PeerCallbackFailed))
                    }
                    (R2cCallbackInjection::Panic, false, FftOverlapError::Overlap(error)) => {
                        assert!(matches!(error, OverlapError::PeerPanicked))
                    }
                    _ => panic!("unexpected R2C callback error"),
                }
            }
            world.barrier();
            let mut fresh = plan.allocate_workspace().unwrap();
            plan.forward_with_overlap(&source, &mut destination, &mut fresh)
                .unwrap();
        }
    }

    fn r2c_conversion_panic_cases(topology: &Arc<MpiTopology<1>>) {
        let plan = R2cPlan::<f64, 2, 1>::from_shape_with_method(
            Arc::clone(topology),
            [3, 4],
            ExtraShape::scalar(),
            TransposeMethod::AllToAllv,
        )
        .unwrap();
        let mut forward_array = plan.allocate_in_place().unwrap();
        forward_array
            .real_view_mut()
            .unwrap()
            .as_mut_slice()
            .fill(1.0);
        super::r2c::panic_after_forward_detach_for_test(&mut forward_array);
        let mut forward_workspace = plan.allocate_in_place_workspace().unwrap();
        let panic_result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            plan.forward_in_place(&mut forward_array, &mut forward_workspace)
        }));
        assert!(panic_result.is_err());
        assert!(super::r2c::storage_capacity_bytes_for_test(&forward_array).is_none());
        super::r2c::assert_poisoned_views_and_retries(
            &plan,
            &mut forward_array,
            &mut forward_workspace,
        );

        let mut reverse_array = plan.allocate_in_place().unwrap();
        let mut reverse_workspace = plan.allocate_in_place_workspace().unwrap();
        plan.forward_in_place(&mut reverse_array, &mut reverse_workspace)
            .unwrap();
        super::r2c::panic_after_reverse_detach_for_test(&mut reverse_array);
        let panic_result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            plan.inverse_in_place(&mut reverse_array, &mut reverse_workspace)
        }));
        assert!(panic_result.is_err());
        assert!(super::r2c::storage_capacity_bytes_for_test(&reverse_array).is_none());
        super::r2c::assert_poisoned_views_and_retries(
            &plan,
            &mut reverse_array,
            &mut reverse_workspace,
        );
    }

    fn r2c_conversion_owner_failure_cases(topology: &Arc<MpiTopology<1>>) {
        let plan = R2cPlan::<f64, 2, 1>::from_shape_with_method(
            Arc::clone(topology),
            [3, 4],
            ExtraShape::scalar(),
            TransposeMethod::AllToAllv,
        )
        .unwrap();
        let mut forward_array = plan.allocate_in_place().unwrap();
        forward_array
            .real_view_mut()
            .unwrap()
            .as_mut_slice()
            .fill(2.0);
        let capacity = super::r2c::storage_capacity_bytes_for_test(&forward_array).unwrap();
        super::r2c::force_odd_forward_cast_for_test(&mut forward_array);
        let mut forward_workspace = plan.allocate_in_place_workspace().unwrap();
        let result = plan.forward_in_place(&mut forward_array, &mut forward_workspace);
        assert!(matches!(
            result,
            Err(R2cError::Fft(FftError::StorageLayoutMismatch))
        ));
        assert_eq!(forward_array.state(), R2cState::Poisoned);
        assert!(super::r2c::storage_is_poisoned_for_test(&forward_array));
        assert_eq!(
            super::r2c::storage_capacity_bytes_for_test(&forward_array),
            Some(capacity)
        );
        super::r2c::assert_poisoned_views_and_retries(
            &plan,
            &mut forward_array,
            &mut forward_workspace,
        );

        let mut reverse_array = plan.allocate_in_place().unwrap();
        let mut reverse_workspace = plan.allocate_in_place_workspace().unwrap();
        plan.forward_in_place(&mut reverse_array, &mut reverse_workspace)
            .unwrap();
        let capacity = super::r2c::storage_capacity_bytes_for_test(&reverse_array).unwrap();
        super::r2c::force_reverse_restore_rejection_for_test(&mut reverse_array);
        let result = plan.inverse_in_place(&mut reverse_array, &mut reverse_workspace);
        assert!(matches!(
            result,
            Err(R2cError::Fft(FftError::Array(
                ArrayError::IncompatiblePencils
            )))
        ));
        assert_eq!(reverse_array.state(), R2cState::Poisoned);
        assert!(super::r2c::storage_is_poisoned_for_test(&reverse_array));
        assert_eq!(
            super::r2c::storage_capacity_bytes_for_test(&reverse_array),
            Some(capacity)
        );
        super::r2c::assert_poisoned_views_and_retries(
            &plan,
            &mut reverse_array,
            &mut reverse_workspace,
        );
    }

    fn r2c_preflight_corruption_cases(topology: &Arc<MpiTopology<1>>) {
        let extra = ExtraShape::new([2]).unwrap();
        let wrong_extra = ExtraShape::new([1, 2]).unwrap();
        let plan = R2cPlan::<f64, 2, 1>::from_shape_with_method(
            Arc::clone(topology),
            [3, 4],
            extra.clone(),
            TransposeMethod::AllToAllv,
        )
        .unwrap();
        let mut real_extra_array = plan.allocate_in_place().unwrap();
        real_extra_array
            .real_view_mut()
            .unwrap()
            .as_mut_slice()
            .fill(3.0);
        super::r2c::corrupt_real_extra_shape_for_test(&mut real_extra_array, wrong_extra.clone());
        let real_before = real_extra_array.real_view().unwrap().as_slice().to_vec();
        let real_array_before = format!("{real_extra_array:?}");
        let mut real_workspace = plan.allocate_in_place_workspace().unwrap();
        let real_workspace_before = format!("{real_workspace:?}");
        assert!(matches!(
            plan.forward_in_place(&mut real_extra_array, &mut real_workspace),
            Err(R2cError::Fft(FftError::StorageLayoutMismatch))
        ));
        assert_eq!(real_extra_array.state(), R2cState::RealInput);
        assert_eq!(
            real_extra_array.real_view().unwrap().as_slice(),
            real_before.as_slice()
        );
        assert_eq!(format!("{real_extra_array:?}"), real_array_before);
        assert_eq!(format!("{real_workspace:?}"), real_workspace_before);

        let registry_plan = R2cPlan::<f64, 2, 1>::from_shape_with_method(
            Arc::clone(topology),
            [3, 4],
            ExtraShape::scalar(),
            TransposeMethod::AllToAllv,
        )
        .unwrap();
        let mut real_registry_array = registry_plan.allocate_in_place().unwrap();
        real_registry_array
            .real_view_mut()
            .unwrap()
            .as_mut_slice()
            .fill(4.0);
        super::r2c::corrupt_forward_registry_for_test(&mut real_registry_array);
        let real_before = real_registry_array.real_view().unwrap().as_slice().to_vec();
        let real_array_before = format!("{real_registry_array:?}");
        let mut real_registry_workspace = registry_plan.allocate_in_place_workspace().unwrap();
        let real_workspace_before = format!("{real_registry_workspace:?}");
        assert!(matches!(
            registry_plan.forward_in_place(&mut real_registry_array, &mut real_registry_workspace),
            Err(R2cError::Fft(FftError::WorkspaceMismatch))
        ));
        assert_eq!(real_registry_array.state(), R2cState::RealInput);
        assert_eq!(
            real_registry_array.real_view().unwrap().as_slice(),
            real_before.as_slice()
        );
        assert_eq!(format!("{real_registry_array:?}"), real_array_before);
        assert_eq!(
            format!("{real_registry_workspace:?}"),
            real_workspace_before
        );

        let mut complex_extra_array = plan.allocate_in_place().unwrap();
        let mut complex_extra_workspace = plan.allocate_in_place_workspace().unwrap();
        plan.forward_in_place(&mut complex_extra_array, &mut complex_extra_workspace)
            .unwrap();
        super::r2c::corrupt_complex_extra_shape_for_test(&mut complex_extra_array, wrong_extra);
        let complex_before = complex_extra_array
            .complex_view()
            .unwrap()
            .as_slice()
            .to_vec();
        let complex_array_before = format!("{complex_extra_array:?}");
        let complex_workspace_before = format!("{complex_extra_workspace:?}");
        assert!(matches!(
            plan.inverse_in_place(&mut complex_extra_array, &mut complex_extra_workspace),
            Err(R2cError::Fft(FftError::StorageLayoutMismatch))
        ));
        assert_eq!(complex_extra_array.state(), R2cState::ComplexOutput);
        assert_eq!(
            complex_extra_array.complex_view().unwrap().as_slice(),
            complex_before.as_slice()
        );
        assert_eq!(format!("{complex_extra_array:?}"), complex_array_before);
        assert_eq!(
            format!("{complex_extra_workspace:?}"),
            complex_workspace_before
        );

        let mut complex_registry_array = registry_plan.allocate_in_place().unwrap();
        let mut complex_registry_workspace = registry_plan.allocate_in_place_workspace().unwrap();
        registry_plan
            .forward_in_place(&mut complex_registry_array, &mut complex_registry_workspace)
            .unwrap();
        super::r2c::corrupt_reverse_registry_for_test(&mut complex_registry_array);
        let complex_before = complex_registry_array
            .complex_view()
            .unwrap()
            .as_slice()
            .to_vec();
        let complex_array_before = format!("{complex_registry_array:?}");
        let complex_workspace_before = format!("{complex_registry_workspace:?}");
        assert!(matches!(
            registry_plan
                .inverse_in_place(&mut complex_registry_array, &mut complex_registry_workspace),
            Err(R2cError::Fft(FftError::WorkspaceMismatch))
        ));
        assert_eq!(complex_registry_array.state(), R2cState::ComplexOutput);
        assert_eq!(
            complex_registry_array.complex_view().unwrap().as_slice(),
            complex_before.as_slice()
        );
        assert_eq!(format!("{complex_registry_array:?}"), complex_array_before);
        assert_eq!(
            format!("{complex_registry_workspace:?}"),
            complex_workspace_before
        );
    }

    fn mixed_in_place_error_cases(topology: &Arc<MpiTopology<1>>) {
        let c2c_transforms = [AxisTransform::Fft, AxisTransform::R2r(AxisR2rKind::Dht)];
        let mut c2c_allocation_plan = MixedC2cPlan::<f64, 2, 1>::from_shape(
            Arc::clone(topology),
            [2, 3],
            ExtraShape::scalar(),
            c2c_transforms,
        )
        .unwrap();
        super::mixed::force_c2c_workspace_allocation_failure_for_test(&mut c2c_allocation_plan);
        assert!(matches!(
            c2c_allocation_plan.allocate_workspace(),
            Err(MixedError::Fft(FftError::AllocationFailed { .. }))
        ));
        assert!(matches!(
            c2c_allocation_plan.allocate_in_place_workspace(),
            Err(MixedError::Fft(FftError::AllocationFailed { .. }))
        ));

        let c2c_plan = MixedC2cPlan::<f64, 2, 1>::from_shape(
            Arc::clone(topology),
            [2, 3],
            ExtraShape::scalar(),
            c2c_transforms,
        )
        .unwrap();
        let mut c2c_source = c2c_plan.allocate_input().unwrap();
        c2c_source.as_mut_slice().fill(1.0.into());
        let c2c_source_before = c2c_source.as_slice().to_vec();
        let mut c2c_output = c2c_plan.allocate_output().unwrap();
        c2c_output.as_mut_slice().fill(2.0.into());
        let c2c_output_before = c2c_output.as_slice().to_vec();
        let mut c2c_workspace = c2c_plan.allocate_workspace().unwrap();
        super::mixed::empty_c2c_workspace_for_test(&mut c2c_workspace);
        let c2c_workspace_before = format!("{c2c_workspace:?}");
        assert!(matches!(
            c2c_plan.forward(&c2c_source, &mut c2c_output, &mut c2c_workspace),
            Err(MixedError::Fft(FftError::WorkspaceTooSmall { .. }))
        ));
        assert_eq!(c2c_source.as_slice(), c2c_source_before.as_slice());
        assert_eq!(c2c_output.as_slice(), c2c_output_before.as_slice());
        assert_eq!(format!("{c2c_workspace:?}"), c2c_workspace_before);

        let mut c2c_array = c2c_plan.allocate_in_place().unwrap();
        let mut c2c_ip_workspace = c2c_plan.allocate_in_place_workspace().unwrap();
        super::mixed::empty_c2c_in_place_workspace_for_test(&mut c2c_ip_workspace);
        let c2c_array_before = format!("{c2c_array:?}");
        let c2c_ip_workspace_before = format!("{c2c_ip_workspace:?}");
        assert!(matches!(
            c2c_plan.forward_in_place(&mut c2c_array, &mut c2c_ip_workspace),
            Err(MixedError::Fft(FftError::WorkspaceTooSmall { .. }))
        ));
        assert_eq!(c2c_array.state(), C2cState::Input);
        assert_eq!(format!("{c2c_array:?}"), c2c_array_before);
        assert_eq!(format!("{c2c_ip_workspace:?}"), c2c_ip_workspace_before);

        let mut c2c_panic_array = c2c_plan.allocate_in_place().unwrap();
        let mut c2c_panic_workspace = c2c_plan.allocate_in_place_workspace().unwrap();
        super::mixed::panic_after_start_for_c2c_test(&mut c2c_panic_array);
        assert!(
            std::panic::catch_unwind(AssertUnwindSafe(|| {
                c2c_plan.forward_in_place(&mut c2c_panic_array, &mut c2c_panic_workspace)
            }))
            .is_err()
        );
        assert_eq!(c2c_panic_array.state(), C2cState::Poisoned);
        assert!(matches!(
            c2c_panic_array.view(),
            Err(MixedError::Fft(FftError::Array(ArrayError::Poisoned)))
        ));
        assert!(matches!(
            c2c_plan.forward_in_place(&mut c2c_panic_array, &mut c2c_panic_workspace),
            Err(MixedError::Fft(FftError::Array(ArrayError::Poisoned)))
        ));

        let r2c_transforms = [AxisTransform::Fft, AxisTransform::Rfft];
        let mut r2c_allocation_plan = MixedR2cPlan::<f64, 2, 1>::from_shape(
            Arc::clone(topology),
            [2, 3],
            ExtraShape::scalar(),
            r2c_transforms,
        )
        .unwrap();
        super::mixed::force_r2c_workspace_allocation_failure_for_test(&mut r2c_allocation_plan);
        assert!(matches!(
            r2c_allocation_plan.allocate_workspace(),
            Err(MixedError::Fft(FftError::AllocationFailed { .. }))
        ));
        assert!(matches!(
            r2c_allocation_plan.allocate_in_place_workspace(),
            Err(MixedError::Fft(FftError::AllocationFailed { .. }))
        ));

        let r2c_plan = MixedR2cPlan::<f64, 2, 1>::from_shape(
            Arc::clone(topology),
            [2, 3],
            ExtraShape::scalar(),
            r2c_transforms,
        )
        .unwrap();
        let mut r2c_source = r2c_plan.allocate_input().unwrap();
        r2c_source.as_mut_slice().fill(1.0);
        let r2c_source_before = r2c_source.as_slice().to_vec();
        let mut r2c_output = r2c_plan.allocate_output().unwrap();
        r2c_output.as_mut_slice().fill(Complex::new(2.0, -1.0));
        let r2c_output_before = r2c_output.as_slice().to_vec();
        let mut r2c_workspace = r2c_plan.allocate_workspace().unwrap();
        super::mixed::empty_r2c_workspace_for_test(&mut r2c_workspace);
        let r2c_workspace_before = format!("{r2c_workspace:?}");
        assert!(matches!(
            r2c_plan.forward(&r2c_source, &mut r2c_output, &mut r2c_workspace),
            Err(MixedError::Fft(FftError::WorkspaceTooSmall { .. }))
        ));
        assert_eq!(r2c_source.as_slice(), r2c_source_before.as_slice());
        assert_eq!(r2c_output.as_slice(), r2c_output_before.as_slice());
        assert_eq!(format!("{r2c_workspace:?}"), r2c_workspace_before);

        let mut r2c_array = r2c_plan.allocate_in_place().unwrap();
        let mut r2c_ip_workspace = r2c_plan.allocate_in_place_workspace().unwrap();
        super::mixed::empty_r2c_in_place_workspace_for_test(&mut r2c_ip_workspace);
        let r2c_array_before = format!("{r2c_array:?}");
        let r2c_ip_workspace_before = format!("{r2c_ip_workspace:?}");
        assert!(matches!(
            r2c_plan.forward_in_place(&mut r2c_array, &mut r2c_ip_workspace),
            Err(MixedError::Fft(FftError::WorkspaceTooSmall { .. }))
        ));
        assert_eq!(r2c_array.state(), R2cState::RealInput);
        assert_eq!(format!("{r2c_array:?}"), r2c_array_before);
        assert_eq!(format!("{r2c_ip_workspace:?}"), r2c_ip_workspace_before);

        let mut r2c_panic_array = r2c_plan.allocate_in_place().unwrap();
        let mut r2c_panic_workspace = r2c_plan.allocate_in_place_workspace().unwrap();
        super::mixed::panic_after_start_for_r2c_test(&mut r2c_panic_array);
        assert!(
            std::panic::catch_unwind(AssertUnwindSafe(|| {
                r2c_plan.forward_in_place(&mut r2c_panic_array, &mut r2c_panic_workspace)
            }))
            .is_err()
        );
        assert_eq!(r2c_panic_array.state(), R2cState::Poisoned);
        assert!(matches!(
            r2c_panic_array.real_view(),
            Err(MixedError::Fft(FftError::Array(ArrayError::Poisoned)))
        ));
        assert!(matches!(
            r2c_plan.forward_in_place(&mut r2c_panic_array, &mut r2c_panic_workspace),
            Err(MixedError::Fft(FftError::Array(ArrayError::Poisoned)))
        ));

        let mut forward_detach_array = r2c_plan.allocate_in_place().unwrap();
        let mut forward_detach_workspace = r2c_plan.allocate_in_place_workspace().unwrap();
        super::mixed::panic_after_forward_detach_for_r2c_test(&mut forward_detach_array);
        assert!(
            std::panic::catch_unwind(AssertUnwindSafe(|| {
                r2c_plan.forward_in_place(&mut forward_detach_array, &mut forward_detach_workspace)
            }))
            .is_err()
        );
        assert_eq!(forward_detach_array.state(), R2cState::Poisoned);
        assert!(forward_detach_array.storage_capacity_bytes().is_none());
        assert!(matches!(
            forward_detach_array.real_view(),
            Err(MixedError::Fft(FftError::Array(ArrayError::Poisoned)))
        ));

        let mut reverse_detach_array = r2c_plan.allocate_in_place().unwrap();
        let mut reverse_detach_workspace = r2c_plan.allocate_in_place_workspace().unwrap();
        r2c_plan
            .forward_in_place(&mut reverse_detach_array, &mut reverse_detach_workspace)
            .unwrap();
        super::mixed::panic_after_reverse_detach_for_r2c_test(&mut reverse_detach_array);
        assert!(
            std::panic::catch_unwind(AssertUnwindSafe(|| {
                r2c_plan.inverse_in_place(&mut reverse_detach_array, &mut reverse_detach_workspace)
            }))
            .is_err()
        );
        assert_eq!(reverse_detach_array.state(), R2cState::Poisoned);
        assert!(reverse_detach_array.storage_capacity_bytes().is_none());
        assert!(matches!(
            reverse_detach_array.complex_view(),
            Err(MixedError::Fft(FftError::Array(ArrayError::Poisoned)))
        ));
    }

    fn assert_transition_method(plan: &C2cPlan<f64, 2, 1>, method: TransposeMethod) {
        match method {
            TransposeMethod::AllToAllv => {
                assert!(matches!(
                    &plan.core.transitions[0].forward,
                    C2cTransition::AllToAllv(_)
                ));
                assert!(matches!(
                    &plan.core.transitions[0].backward,
                    C2cTransition::AllToAllv(_)
                ));
            }
            TransposeMethod::PointToPoint => {
                assert!(matches!(
                    &plan.core.transitions[0].forward,
                    C2cTransition::PointToPoint(_)
                ));
                assert!(matches!(
                    &plan.core.transitions[0].backward,
                    C2cTransition::PointToPoint(_)
                ));
            }
        }
    }

    fn assert_poisoned_views_and_retries(
        plan: &C2cPlan<f64, 2, 1>,
        array: &mut C2cInPlaceArray<f64, 2, 1>,
        workspace: &mut C2cInPlaceWorkspace<f64, 2, 1>,
    ) {
        assert_eq!(array.state(), C2cState::Poisoned);
        assert!(matches!(
            array.view(),
            Err(FftError::Array(pencil_array::ArrayError::Poisoned))
        ));
        assert!(matches!(
            array.view_mut(),
            Err(FftError::Array(pencil_array::ArrayError::Poisoned))
        ));
        for direction in [Direction::Forward, Direction::Inverse, Direction::Backward] {
            let array_before = format!("{array:?}");
            let workspace_before = format!("{workspace:?}");
            let result = match direction {
                Direction::Forward => plan.forward_in_place(array, workspace),
                Direction::Inverse => plan.inverse_in_place(array, workspace),
                Direction::Backward => plan.backward_in_place(array, workspace),
            };
            assert!(matches!(
                result,
                Err(FftError::Array(pencil_array::ArrayError::Poisoned))
            ));
            assert_eq!(array.state(), C2cState::Poisoned);
            assert_eq!(format!("{array:?}"), array_before);
            assert_eq!(format!("{workspace:?}"), workspace_before);
        }
    }

    #[test]
    fn canonical_routes_have_exact_local_and_distributed_transition_counts() {
        assert_eq!(route_counts::<2, 1>(), (1, 0));
        assert_eq!(route_counts::<3, 1>(), (1, 1));
        assert_eq!(route_counts::<4, 1>(), (1, 2));
        assert_eq!(route_counts::<3, 2>(), (2, 0));
        assert_eq!(route_counts::<4, 2>(), (2, 1));
    }

    fn route_counts<const N: usize, const M: usize>() -> (usize, usize) {
        let mut decomposition: [usize; M] = std::array::from_fn(|axis| axis);
        let mut distributed = 0;
        for axis in (0..N - 1).rev() {
            distributed += usize::from(super::advance_decomposition(&mut decomposition, axis));
        }
        (distributed, (N - 1) - distributed)
    }
}
