//! Distributed complex-to-complex FFTs behind the `distributed` feature.
//!
//! [`C2cPlan`] construction and transform calls are collective: every rank
//! must use the same Cartesian communicator context, API, order, scalar type,
//! [`TransposeMethod`], and matching layouts. In-place array/workspace
//! allocation and array views are noncollective. Callers must coordinate a
//! local allocation failure before entering the next collective call. The
//! legacy constructors select Alltoallv; the `_with_method` constructors can
//! select the checked point-to-point transport as well. The out-of-place
//! forward transform consumes identity memory order with trailing-axis
//! decomposition and produces the endpoint of the selected-axis route.
//! Selected axes run in ascending logical order; when a selected axis is
//! distributed, its grid slot moves to the preceding local transform axis (or
//! the smallest local axis initially). By default the selected axis becomes
//! the first physical memory axis; [`DistributedLayout::permute_dims`] can
//! retain identity memory order and use column-major prefix-stride line kernels.
//! Inverse and backward traverse only selected stages in reverse, with inverse
//! normalized and backward left unnormalized. Unselected axes have no stage or
//! transition; an empty selection is a validated no-op. In-place execution
//! uses the same route and state-checks its single buffer.
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
//! real in-place API. [`C2rPlan`] instead accepts canonical complex input and
//! uses a descending C2R route with its lowest selected axis reduced. It
//! provides raw positive-sign forward, raw negative-sign backward, and
//! normalized negative-sign inverse, out-of-place only.
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

/// One observed step of the endpoint statistics protocol.
///
/// Only the statistics reductions of `validate_boundary` and
/// `validate_mixed_boundary_data` are recorded; constructor agreements and
/// point-to-point status reductions are not part of this log.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EndpointReductionEvent {
    PreparationAgreement,
    Allocation { slot: usize, length: usize },
    MaxReduction { elements: usize },
    SumReduction { elements: usize },
    FinalAgreement,
}

#[cfg(test)]
thread_local! {
    static ENDPOINT_REDUCTION_LOG: std::cell::RefCell<Vec<EndpointReductionEvent>> =
        const { std::cell::RefCell::new(Vec::new()) };
    static ENDPOINT_ALLOCATION_FAILURE: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
}

#[cfg(test)]
fn record_endpoint_reduction(event: EndpointReductionEvent) {
    ENDPOINT_REDUCTION_LOG.with(|log| log.borrow_mut().push(event));
}

#[cfg(test)]
fn endpoint_reduction_log() -> Vec<EndpointReductionEvent> {
    ENDPOINT_REDUCTION_LOG.with(|log| log.borrow().clone())
}

#[cfg(test)]
fn reset_endpoint_reduction_log() {
    ENDPOINT_REDUCTION_LOG.with(|log| log.borrow_mut().clear());
    ENDPOINT_ALLOCATION_FAILURE.with(|failure| failure.set(None));
}

#[cfg(test)]
fn force_endpoint_allocation_failure(slot: usize) {
    ENDPOINT_ALLOCATION_FAILURE.with(|failure| failure.set(Some(slot)));
}

use crate::{
    Complex, FftReal, LocalC2cError, LocalC2cPlan, LocalR2cError, LocalR2cPlan, LocalR2rError,
};

use crate::BackendInitError;
#[cfg(feature = "fftw")]
use crate::PlanOptions;

#[cfg(test)]
#[path = "distributed/tests/mod.rs"]
mod regression_tests;

#[derive(Clone, Copy, Debug)]
enum BackendChoice {
    RustFft,
    #[cfg(feature = "fftw")]
    Fftw(PlanOptions),
}

impl BackendChoice {
    fn kind(self) -> crate::BackendKind {
        match self {
            Self::RustFft => crate::BackendKind::RustFft,
            #[cfg(feature = "fftw")]
            Self::Fftw(_) => crate::BackendKind::Fftw,
        }
    }

    #[allow(clippy::extra_unused_type_parameters)]
    fn descriptor_words<R: FftReal>(self) -> [u64; BACKEND_DESCRIPTOR_WORDS] {
        match self {
            Self::RustFft => [0; BACKEND_DESCRIPTOR_WORDS],
            #[cfg(feature = "fftw")]
            Self::Fftw(options) => {
                let (present, secs, nanos) = options.time_limit().map_or((0, 0, 0), |limit| {
                    (1, limit.as_secs(), u64::from(limit.subsec_nanos()))
                });
                [
                    1,
                    size_of::<R>() as u64,
                    options.rigor() as u64,
                    present,
                    secs,
                    nanos,
                    options.requested_threads() as u64,
                    u64::from(options.wisdom_only()) | (u64::from(options.conserve_memory()) << 1),
                ]
            }
        }
    }
}

const BACKEND_DESCRIPTOR_WORDS: usize = 8;

/// Extra-shape batches fused into one endpoint statistics reduction.
///
/// Each statistics buffer holds at most this many batches per rank, and
/// endpoint validation runs one MAX and one SUM reduction per chunk. For `B`
/// batches that is `2 * B.div_ceil(ENDPOINT_BATCH_CHUNK)` all-reduces instead
/// of two per batch.
const ENDPOINT_BATCH_CHUNK: usize = 1024;
const DESCRIPTOR_SCHEMA: u64 = 5;
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
#[cfg(feature = "fftw")]
const OPERATION_NATIVE_C2C_PLAN: u64 = 127;
#[cfg(feature = "fftw")]
const OPERATION_NATIVE_R2C_PLAN: u64 = 128;
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

/// Timing aggregated into one fixed route-stage slot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StageTiming {
    /// Time spent in the local transform.
    pub fft: Duration,
    /// Sum of wall-clock times for route transitions assigned to this slot.
    /// The initial transition, when present, is assigned to slot 0.
    pub transpose: Duration,
    /// Number of local FFT phases included in `fft`.
    pub fft_calls: u64,
    /// Number of route transitions included in this slot.
    pub transition_calls: u64,
    /// Communication phases summed across this slot's route transitions.
    /// Unused phases (including phases of local-only transitions) are zero.
    pub communication: pencil_array::TransposeTiming,
    /// Sum of this slot's measured work (`fft + transpose`).
    pub total: Duration,
}

/// Timing entries for a distributed transform.
///
/// The active prefix is exposed through indexing, iteration, and `len`; the
/// fixed-size `map` method retains array-style access to all `N` slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageTimings<const N: usize> {
    entries: [StageTiming; N],
    len: usize,
}

impl<const N: usize> StageTimings<N> {
    /// Returns the number of executed route stages.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether no route stages were executed.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Maps all `N` storage slots, matching fixed-array `map` behavior.
    pub fn map<T>(self, f: impl FnMut(StageTiming) -> T) -> [T; N] {
        self.entries.map(f)
    }

    pub(crate) fn set_active_len(&mut self, len: usize) {
        assert!(len <= N, "active timing stage count exceeds capacity");
        self.len = len;
    }
}

impl<const N: usize> Default for StageTimings<N> {
    fn default() -> Self {
        Self {
            entries: [StageTiming::default(); N],
            len: N,
        }
    }
}

impl<const N: usize> std::ops::Deref for StageTimings<N> {
    type Target = [StageTiming];

    fn deref(&self) -> &Self::Target {
        &self.entries[..self.len]
    }
}

impl<const N: usize> std::ops::DerefMut for StageTimings<N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.entries[..self.len]
    }
}

impl<const N: usize> IntoIterator for StageTimings<N> {
    type Item = StageTiming;
    type IntoIter = std::iter::Take<std::array::IntoIter<StageTiming, N>>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter().take(self.len)
    }
}

impl<'a, const N: usize> IntoIterator for &'a StageTimings<N> {
    type Item = &'a StageTiming;
    type IntoIter = std::slice::Iter<'a, StageTiming>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Timing returned by profiled distributed transforms.
///
/// For C2C, entries follow selected-axis order. A transition between route
/// stages `i` and `i + 1` is assigned to slot `i`; the initial transition is
/// also assigned to slot 0. Consequently a partial route can accumulate more
/// than one transition in a slot, in both forward and inverse execution.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransformTiming<const N: usize> {
    /// One entry for each executed route stage, in route order.
    pub stages: StageTimings<N>,
    /// Wall-clock duration of the complete call.
    pub total: Duration,
}

impl<const N: usize> Default for TransformTiming<N> {
    fn default() -> Self {
        Self {
            stages: StageTimings::default(),
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

    /// The input pencil was not the default identity/trailing-decomposition layout.
    #[error("input pencil must use identity permutation and trailing decomposition [N-M..N)")]
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

impl From<FftError> for BackendInitError<R2rError> {
    fn from(error: FftError) -> Self {
        Self::Local(R2rError::Fft(error))
    }
}

pub mod collections;
pub use collections::CollectionError;
#[cfg(test)]
#[path = "distributed/r2c/tests/boundary.rs"]
mod boundary_regression_tests;
mod c2r;
#[cfg(all(test, feature = "fftw"))]
pub(crate) mod fftw_tests;
pub mod mixed;
mod r2c;
#[cfg(test)]
use r2c::for_each_boundary_endpoint;
mod r2r;

#[cfg(test)]
pub(crate) static MPI_TEST_LOCK: std::sync::OnceLock<std::sync::Mutex<()>> =
    std::sync::OnceLock::new();

pub use c2r::{C2rError, C2rPlan, C2rWorkspace};
pub use r2c::{R2cInPlaceArray, R2cInPlaceWorkspace, R2cPlan, R2cWorkspace};
pub use r2r::{DhtPlan, R2rInPlaceArray, R2rInPlaceWorkspace, R2rPlan, R2rWorkspace};

/// Selects the checked transition transport used by distributed transform plans.
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
    /// Whether each C2C route stage moves its transform axis to the
    /// first memory dimension. `false` keeps every stage in identity memory
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
/// [`Self::from_indices`] are only validation input; C2C execution visits
/// selected logical axes in ascending order, not caller order.
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
/// See the [crate-level compatibility and failure contract](crate#distributed-compatibility-and-failure-contract)
/// for endpoint identity, workspace ownership, and error boundaries.
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
/// input must use identity memory permutation and default trailing decomposition
/// `[N-M, ..., N)`; this feature supports `N >= 2` and `1 <= M < N`. A validated
/// [`AxisSelection`] controls which axes are transformed. The route always
/// contains one stage per selected axis, in ascending logical-axis order;
/// unselected axes have no transform stage or route transition. The default
/// input uses identity memory order and trailing-axis decomposition. A
/// selected distributed axis is localized by transferring its grid slot to
/// the preceding local transform axis (or the smallest local axis initially).
/// With `layout.permute_dims`, each selected axis becomes the first physical
/// memory axis; otherwise all stages retain identity memory order. An empty
/// selection is a validated no-op with unchanged endpoint layout. The [`Self::forward`],
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

impl<const N: usize, const M: usize> C2cTransition<N, M> {
    fn supports_overlap(&self) -> bool {
        !matches!(self, Self::AllToAllv(_))
    }
}

pub(super) struct ComplexWorkspaceBuffers<R: FftReal> {
    pub(super) transpose: TransposeWorkspace<Complex<R>>,
    pub(super) fft_scratch: Vec<Complex<R>>,
    pub(super) line_buffer: Vec<Complex<R>>,
}

pub(super) fn allocate_complex_workspace_buffers<R: FftReal>(
    send_len: usize,
    receive_len: usize,
    scratch_len: usize,
    line_len: usize,
) -> Result<ComplexWorkspaceBuffers<R>, FftError> {
    Ok(ComplexWorkspaceBuffers {
        transpose: TransposeWorkspace::from_vecs(
            initialized_vec(send_len, zero_complex::<R>())?,
            initialized_vec(receive_len, zero_complex::<R>())?,
        ),
        fft_scratch: initialized_vec(scratch_len, zero_complex::<R>())?,
        line_buffer: initialized_vec(line_len, zero_complex::<R>())?,
    })
}

pub(super) struct R2rTransposeBuffers<T> {
    pub(super) send_buffer: Vec<T>,
    pub(super) receive_buffer: Vec<T>,
}

pub(super) struct R2rWorkspaceBuffers<T: crate::R2rScalar> {
    pub(super) transpose: R2rTransposeBuffers<T>,
    pub(super) embedding_line: Vec<Complex<T::Real>>,
    pub(super) fft_scratch: Vec<Complex<T::Real>>,
    pub(super) line_buffer: Vec<T>,
}

pub(super) fn allocate_r2r_workspace_buffers<T: crate::R2rScalar>(
    send_len: usize,
    receive_len: usize,
    embedding_len: usize,
    scratch_len: usize,
    line_len: usize,
) -> Result<R2rWorkspaceBuffers<T>, FftError> {
    Ok(R2rWorkspaceBuffers {
        transpose: R2rTransposeBuffers {
            send_buffer: initialized_vec(send_len, crate::r2r::r2r_zero::<T>())?,
            receive_buffer: initialized_vec(receive_len, crate::r2r::r2r_zero::<T>())?,
        },
        embedding_line: initialized_vec(embedding_len, zero_complex::<T::Real>())?,
        fft_scratch: initialized_vec(scratch_len, zero_complex::<T::Real>())?,
        line_buffer: initialized_vec(line_len, crate::r2r::r2r_zero::<T>())?,
    })
}

#[derive(Debug)]
struct C2cStageTransition<const N: usize, const M: usize> {
    forward: C2cTransition<N, M>,
    backward: C2cTransition<N, M>,
}

#[derive(Debug)]
struct TransformPlanCore<R: FftReal, const N: usize, const M: usize> {
    stages: Box<[TransformStage<R, N, M>]>,
    initial_transition: C2cStageTransition<N, M>,
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
    backend: BackendChoice,
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
struct C2cRouteStage<const N: usize, const M: usize> {
    axis: usize,
    source: Arc<Pencil<N, M>>,
    output: Arc<Pencil<N, M>>,
}

#[derive(Debug)]
struct C2cRouteCandidate<const N: usize, const M: usize> {
    stages: Box<[C2cRouteStage<N, M>]>,
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
        let input =
            Pencil::new_default(Arc::clone(&topology), global_shape).map_err(FftError::Pencil);
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

    /// Returns the backend used by this plan.
    pub fn backend_kind(&self) -> crate::BackendKind {
        self.core.backend.kind()
    }

    /// Returns native planning options, or `None` for RustFFT.
    #[cfg(feature = "fftw")]
    pub fn options(&self) -> Option<PlanOptions> {
        match self.core.backend {
            BackendChoice::RustFft => None,
            BackendChoice::Fftw(options) => Some(options),
        }
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
            .expect("distributed C2C has an internal endpoint stage")
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
        if self.core.selection.is_empty() {
            return Box::new([]);
        }
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
            registered_pencils(
                std::iter::once(self.input_pencil())
                    .chain(self.core.stages.iter().map(|stage| &stage.output)),
            )?,
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
        let buffers = allocate_complex_workspace_buffers::<R>(
            self.core.transpose_send_len,
            self.core.transpose_receive_len,
            self.core.fft_scratch_len,
            self.core.strided_line_len,
        )?;
        Ok(C2cInPlaceWorkspace {
            core: Arc::clone(&self.core),
            transpose: buffers.transpose,
            fft_scratch: buffers.fft_scratch,
            line_buffer: buffers.line_buffer,
        })
    }

    /// Allocates a reusable workspace for out-of-place forward, inverse, and backward execution.
    pub fn allocate_out_of_place_workspace(
        &self,
    ) -> Result<C2cOutOfPlaceWorkspace<R, N, M>, FftError> {
        let intermediate = ManyPencilArray::from_elem(
            registered_pencils(
                std::iter::once(self.input_pencil())
                    .chain(self.core.stages.iter().map(|stage| &stage.output)),
            )?,
            0,
            self.core.extra_shape.clone(),
            zero_complex::<R>(),
        )
        .map_err(map_array_allocation)?;

        let buffers = allocate_complex_workspace_buffers::<R>(
            self.core.transpose_send_len,
            self.core.transpose_receive_len,
            self.core.fft_scratch_len,
            self.core.strided_line_len,
        )?;
        Ok(C2cOutOfPlaceWorkspace {
            core: Arc::clone(&self.core),
            intermediate,
            transpose: buffers.transpose,
            fft_scratch: buffers.fft_scratch,
            line_buffer: buffers.line_buffer,
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
    ///
    /// Preflight failures leave arrays and workspace unchanged. After execution
    /// starts, sources remain unchanged but destinations may be modified. A
    /// poisoned opaque FFT workspace must be reallocated; rerunning cannot
    /// recover it. Changed-but-valid storage is reusable only where the
    /// underlying operation explicitly guarantees validity, not as a general
    /// error-recovery rule. Low-level transpose recovery contracts are separate.
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
    /// final sends. State and recovery guarantees are those of
    /// [`Self::forward_with_overlap`], not the low-level transpose contract.
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
    /// final sends. State and recovery guarantees are those of
    /// [`Self::forward_with_overlap`], not the low-level transpose contract.
    pub fn backward_with_overlap(
        &self,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &mut PencilArray<Complex<R>, N, M>,
        workspace: &mut C2cOutOfPlaceWorkspace<R, N, M>,
    ) -> Result<(), FftOverlapError<FftError>> {
        self.execute_overlap(Direction::Backward, source, destination, workspace)
    }

    /// Runs [`Self::forward`] and returns timing for its executed route stages.
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

    /// Runs [`Self::inverse`] and returns timing for its executed route stages.
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

    /// Runs [`Self::backward`] and returns timing for its executed route stages.
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

    /// Runs [`Self::forward_in_place`] and returns timing for its executed route stages.
    pub fn forward_in_place_with_timing(
        &self,
        array: &mut C2cInPlaceArray<R, N, M>,
        workspace: &mut C2cInPlaceWorkspace<R, N, M>,
    ) -> Result<TransformTiming<N>, FftError> {
        let mut timing = TransformTiming::<N>::default();
        self.execute_in_place(Direction::Forward, array, workspace, Some(&mut timing))?;
        Ok(timing)
    }

    /// Runs [`Self::inverse_in_place`] and returns timing for its executed route stages.
    pub fn inverse_in_place_with_timing(
        &self,
        array: &mut C2cInPlaceArray<R, N, M>,
        workspace: &mut C2cInPlaceWorkspace<R, N, M>,
    ) -> Result<TransformTiming<N>, FftError> {
        let mut timing = TransformTiming::<N>::default();
        self.execute_in_place(Direction::Inverse, array, workspace, Some(&mut timing))?;
        Ok(timing)
    }

    /// Runs [`Self::backward_in_place`] and returns timing for its executed route stages.
    pub fn backward_in_place_with_timing(
        &self,
        array: &mut C2cInPlaceArray<R, N, M>,
        workspace: &mut C2cInPlaceWorkspace<R, N, M>,
    ) -> Result<TransformTiming<N>, FftError> {
        let mut timing = TransformTiming::<N>::default();
        self.execute_in_place(Direction::Backward, array, workspace, Some(&mut timing))?;
        Ok(timing)
    }

    /// Builds a RustFFT plan with explicit per-axis Fourier signs.
    ///
    /// Unlike [`Self::with_fft_directions`], this direct constructor uses the
    /// non-strict endpoint rules in the [crate-level contract](crate#distributed-compatibility-and-failure-contract).
    /// Forward uses the configured signs; inverse and backward use their opposites.
    pub fn from_shape_with_fft_directions(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
        directions: FourierDirections<N>,
    ) -> Result<Self, FftError> {
        let input =
            Pencil::new_default(Arc::clone(&topology), global_shape).map_err(FftError::Pencil);
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

    /// Collectively rebuilds with new Fourier signs, preserving the backend.
    ///
    /// Use the returned plan's endpoint-pencil `Arc`s and allocate new workspaces
    /// and in-place arrays, even with RustFFT or unchanged signs. See the
    /// [crate-level contract](crate#distributed-compatibility-and-failure-contract).
    /// Forward is unscaled with these signs; inverse uses opposite signs with
    /// normalization, and backward uses opposite signs without normalization.
    /// Non-selected axes must use `Forward` (checked collectively).
    pub fn with_fft_directions(&self, directions: FourierDirections<N>) -> Result<Self, FftError> {
        let topology = Arc::clone(self.input_pencil().topology());
        let shape = *self.input_pencil().global_shape();
        let input = Pencil::new_default(Arc::clone(&topology), shape).map_err(FftError::Pencil);
        let mut plan = Self::construct_with_backend(
            topology,
            shape,
            self.core.extra_shape.clone(),
            input,
            self.core.selection,
            self.core.layout,
            directions,
            self.core.backend,
        )
        .map_err(|error| match error {
            BackendInitError::Local(error) => error,
            #[cfg(feature = "fftw")]
            BackendInitError::Native(_) | BackendInitError::PeerPreflight => {
                FftError::PreparationFailed
            }
            #[cfg(not(feature = "fftw"))]
            BackendInitError::PeerPreflight => FftError::PreparationFailed,
        })?;
        Arc::get_mut(&mut plan.core)
            .expect("fresh core")
            .strict_array_identity = true;
        Ok(plan)
    }

    /// Returns the signs used by the forward operation.
    pub fn fft_directions(&self) -> FourierDirections<N> {
        self.core.directions
    }

    #[cfg(feature = "fftw")]
    /// Collectively builds a distributed C2C plan using FFTW on every rank.
    pub fn from_shape_with_fftw(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
        options: PlanOptions,
    ) -> Result<Self, BackendInitError<FftError>> {
        let input =
            Pencil::new_default(Arc::clone(&topology), global_shape).map_err(FftError::Pencil);
        Self::construct_with_backend(
            topology,
            global_shape,
            extra_shape,
            input,
            AxisSelection::all(),
            DistributedLayout::default(),
            FourierDirections::default(),
            BackendChoice::Fftw(options),
        )
    }

    /// Collectively rebuilds with FFTW while preserving its shape, layout,
    /// selected axes, and Fourier signs.
    ///
    /// Use the returned plan's endpoint-pencil `Arc`s and allocate new workspaces
    /// and in-place arrays. See the [crate-level contract](crate#distributed-compatibility-and-failure-contract).
    #[cfg(feature = "fftw")]
    pub fn with_fftw(&self, options: PlanOptions) -> Result<Self, BackendInitError<FftError>> {
        let topology = Arc::clone(self.input_pencil().topology());
        let shape = *self.input_pencil().global_shape();
        let input = Pencil::new_default(Arc::clone(&topology), shape).map_err(FftError::Pencil);
        let mut plan = Self::construct_with_backend(
            topology,
            shape,
            self.core.extra_shape.clone(),
            input,
            self.core.selection,
            self.core.layout,
            self.core.directions,
            BackendChoice::Fftw(options),
        )?;
        Arc::get_mut(&mut plan.core)
            .expect("fresh core")
            .strict_array_identity = true;
        Ok(plan)
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
        Self::construct_with_backend(
            topology,
            global_shape,
            extra_shape,
            input,
            selection,
            layout,
            directions,
            BackendChoice::RustFft,
        )
        .map_err(|error| match error {
            BackendInitError::Local(error) => error,
            #[cfg(feature = "fftw")]
            BackendInitError::Native(_) | BackendInitError::PeerPreflight => {
                FftError::PreparationFailed
            }
            #[cfg(not(feature = "fftw"))]
            BackendInitError::PeerPreflight => FftError::PreparationFailed,
        })
    }

    fn construct_with_backend(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
        input: Result<Arc<Pencil<N, M>>, FftError>,
        selection: AxisSelection<N>,
        layout: DistributedLayout,
        directions: FourierDirections<N>,
        backend: BackendChoice,
    ) -> Result<Self, BackendInitError<FftError>> {
        let communicator = topology.communicator();
        let expected_len = descriptor_len::<N, M>(&extra_shape)
            .and_then(|length| length.checked_add(N))
            .and_then(|length| length.checked_add(BACKEND_DESCRIPTOR_WORDS));
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
            descriptor
                .try_reserve_exact(N + BACKEND_DESCRIPTOR_WORDS)
                .ok()?;
            descriptor.extend(directions.0.iter().map(|direction| match direction {
                FourierDirection::Forward => 0,
                FourierDirection::Backward => 1,
            }));
            descriptor.extend(backend.descriptor_words::<R>());
            Some(descriptor)
        });
        let descriptor_len_word = expected_len
            .and_then(|length| u64::try_from(length).ok())
            .unwrap_or(INVALID_WORD);
        let header = [
            DESCRIPTOR_SCHEMA,
            match backend {
                BackendChoice::RustFft => OPERATION_PLAN,
                #[cfg(feature = "fftw")]
                BackendChoice::Fftw(_) => OPERATION_NATIVE_C2C_PLAN,
            },
            u64::try_from(N).unwrap_or(INVALID_WORD),
            u64::try_from(M).unwrap_or(INVALID_WORD),
            descriptor_len_word,
        ];
        if !agree_header(communicator, header) {
            return Err(BackendInitError::Local(
                FftError::CollectiveDescriptorMismatch,
            ));
        }
        // The exact mask is agreed before route construction or native FFT
        // planning. A rank-local selection mismatch therefore cannot enter a
        // different route; callers coordinate any local constructor result.
        let descriptor = collective_descriptor(communicator, descriptor, expected_len)?;
        let route = agree_result(
            communicator,
            build_c2c_route(
                input,
                &topology,
                global_shape,
                selection,
                layout.permute_dims,
            ),
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
        let prepared = prepare_c2c_stages_backend::<R, N, M>(
            &route,
            global_shape,
            selection,
            directions,
            backend,
        );
        if !collective_valid(communicator, prepared.is_ok()) {
            return Err(match prepared {
                Err(error) => error,
                Ok(_) => BackendInitError::PeerPreflight,
            });
        }
        let stages = prepared.expect("collective backend preflight accepted");

        let first_layouts = [
            Arc::clone(&route.stages[0].source),
            Arc::clone(&route.stages[0].output),
        ];
        let (initial_inputs, initial_outputs) = agree_result(
            communicator,
            pencil_columns(first_layouts.iter(), first_layouts.iter()),
        )?;
        let (mut initial_transitions, initial_send, initial_receive, _, _) =
            build_transitions::<N, M>(
                communicator,
                &initial_inputs,
                &initial_outputs,
                &[route.distributed[0]],
                &extra_shape,
                layout.transpose_method,
                0,
            )?;
        let initial_transition = initial_transitions
            .pop()
            .ok_or(FftError::PreparationFailed)?;

        let (transition_layouts, _) = agree_result(
            communicator,
            pencil_columns(
                route.stages.iter().map(|stage| &stage.output),
                route.stages.iter().map(|stage| &stage.output),
            ),
        )?;
        let (
            transitions,
            transpose_send_len,
            transpose_receive_len,
            real_transpose_send_len,
            real_transpose_receive_len,
        ) = build_transitions::<N, M>(
            communicator,
            &transition_layouts,
            &transition_layouts,
            &route.distributed[1..],
            &extra_shape,
            layout.transpose_method,
            0,
        )?;
        let transpose_send_len = transpose_send_len.max(initial_send);
        let transpose_receive_len = transpose_receive_len.max(initial_receive);

        let strided_line_len = strided_complex_line_len(&stages.stages)?;
        let core = TransformPlanCore {
            stages: stages.stages,
            initial_transition,
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
            backend,
            strict_array_identity: backend.kind() == crate::BackendKind::Fftw,
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
        if let Some(timing) = report.as_deref_mut() {
            timing.stages.set_active_len(self.core.selection.len());
        }
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

        if self.core.selection.is_empty() {
            destination
                .as_mut_slice()
                .copy_from_slice(source.as_slice());
            if let Some(timing) = report {
                timing.total = started.elapsed();
            }
            return Ok(());
        }

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
        let overlap_supported = self.core.initial_transition.forward.supports_overlap()
            && self.core.initial_transition.backward.supports_overlap()
            && self.core.transitions.iter().all(|transition| {
                transition.forward.supports_overlap() && transition.backward.supports_overlap()
            });
        // This agreement is deliberately before stage-zero writes or payload.
        if !collective_valid(communicator, overlap_supported) {
            return Err(FftOverlapError::UnsupportedTransport);
        }
        if self.core.selection.is_empty() {
            destination
                .as_mut_slice()
                .copy_from_slice(source.as_slice());
            return Ok(());
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
        if let Some(timing) = report.as_deref_mut() {
            timing.stages.set_active_len(self.core.selection.len());
        }
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

        let target = match direction {
            Direction::Forward => C2cState::Output,
            Direction::Inverse | Direction::Backward => C2cState::Input,
        };
        if self.core.selection.is_empty() {
            array.state = target;
            if let Some(timing) = report {
                timing.total = started.elapsed();
            }
            return Ok(());
        }

        let normalize_inverse = matches!(direction, Direction::Inverse);
        let result = run_in_place_transaction(array, workspace, target, |array, workspace| {
            #[cfg(test)]
            collections::inject_in_place_failure()?;
            match direction {
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
            }
        });
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

fn registered_pencils<'a, const N: usize, const M: usize>(
    pencils: impl Iterator<Item = &'a Arc<Pencil<N, M>>>,
) -> Result<Box<[Arc<Pencil<N, M>>]>, FftError> {
    let (minimum, maximum) = pencils.size_hint();
    let required = maximum.unwrap_or(minimum);
    let mut registered = Vec::new();
    registered
        .try_reserve_exact(required)
        .map_err(|_| FftError::AllocationFailed { required })?;
    for pencil in pencils {
        if !registered
            .iter()
            .any(|other: &Arc<Pencil<N, M>>| other.same_layout(pencil))
        {
            registered.push(Arc::clone(pencil));
        }
    }
    Ok(registered.into_boxed_slice())
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

fn validate_c2c_input<const N: usize, const M: usize>(
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
    if !Arc::ptr_eq(input.topology(), topology) || input.global_shape() != &global_shape {
        return Err(FftError::PreparationFailed);
    }
    let identity = std::array::from_fn(|axis| axis);
    let default_decomposition = std::array::from_fn(|slot| N - M + slot);
    if input.permutation().axes().map(SpatialAxis::index) != identity
        || input.decomposition().map(SpatialAxis::index) != default_decomposition
    {
        return Err(FftError::InvalidInputLayout);
    }
    Ok(input)
}

fn build_c2c_route<const N: usize, const M: usize>(
    input: Result<Arc<Pencil<N, M>>, FftError>,
    topology: &Arc<MpiTopology<M>>,
    global_shape: [usize; N],
    selection: AxisSelection<N>,
    permute_dims: bool,
) -> Result<C2cRouteCandidate<N, M>, FftError> {
    let input = validate_c2c_input(input, topology, global_shape)?;
    let mut axes = Vec::new();
    axes.try_reserve_exact(selection.len())
        .map_err(|_| FftError::AllocationFailed {
            required: selection.len(),
        })?;
    axes.extend((0..N).filter(|&axis| selection.contains(axis)));

    let route_len = axes.len().max(1);
    let mut stages = Vec::new();
    stages
        .try_reserve_exact(route_len)
        .map_err(|_| FftError::AllocationFailed {
            required: route_len,
        })?;
    let mut distributed = Vec::new();
    distributed
        .try_reserve_exact(route_len)
        .map_err(|_| FftError::AllocationFailed {
            required: route_len,
        })?;

    let mut current = input;
    if axes.is_empty() {
        stages.push(C2cRouteStage {
            axis: 0,
            source: Arc::clone(&current),
            output: Arc::clone(&current),
        });
        distributed.push(false);
    } else {
        for (index, &axis) in axes.iter().enumerate() {
            let source = Arc::clone(&current);
            let mut decomposition = current.decomposition().map(SpatialAxis::index);
            let is_distributed = if let Some(slot) = decomposition.iter().position(|&a| a == axis) {
                let previous_local = if index == 0 {
                    (0..N)
                        .find(|candidate| !decomposition.contains(candidate))
                        .ok_or(FftError::PreparationFailed)?
                } else {
                    axes[index - 1]
                };
                if decomposition.contains(&previous_local) {
                    return Err(FftError::PreparationFailed);
                }
                decomposition[slot] = previous_local;
                true
            } else {
                false
            };
            let permutation = if permute_dims {
                let mut next_axes = [0; N];
                next_axes[0] = axis;
                let mut position = 1;
                for old_axis in current.permutation().axes().map(SpatialAxis::index) {
                    if old_axis != axis {
                        next_axes[position] = old_axis;
                        position += 1;
                    }
                }
                AxisPermutation::new(next_axes)
                    .map_err(PencilError::InvalidPermutation)
                    .map_err(FftError::Pencil)?
            } else {
                AxisPermutation::identity()
            };
            let next = current
                .reconfigured(PencilConfig {
                    global_shape,
                    decomposition,
                    permutation,
                })
                .map_err(FftError::Pencil)?;
            stages.push(C2cRouteStage {
                axis,
                source,
                output: Arc::clone(&next),
            });
            distributed.push(is_distributed);
            current = next;
        }
    }
    Ok(C2cRouteCandidate {
        stages: stages.into_boxed_slice(),
        distributed: distributed.into_boxed_slice(),
    })
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

#[cfg(feature = "fftw")]
fn prepare_complex_stage_for_backend<R: FftReal, const N: usize, const M: usize>(
    pencil: &Arc<Pencil<N, M>>,
    global_shape: [usize; N],
    axis_index: usize,
    selected: bool,
    directions: FourierDirections<N>,
    backend: BackendChoice,
) -> Result<TransformStage<R, N, M>, BackendInitError<FftError>> {
    if matches!(backend, BackendChoice::RustFft) {
        return prepare_complex_stage(pencil, global_shape, axis_index, selected, directions)
            .map_err(BackendInitError::Local);
    }
    if pencil
        .decomposition()
        .iter()
        .any(|d| d.index() == axis_index)
        || pencil.local_shape_logical()[axis_index] != global_shape[axis_index]
    {
        return Err(BackendInitError::Local(FftError::PreparationFailed));
    }
    if !selected {
        return Ok(TransformStage {
            axis: axis_index,
            input: Arc::clone(pencil),
            output: Arc::clone(pencil),
            local: LocalTransform::Identity,
        });
    }
    let sign = directions.get(axis_index) == Some(FourierDirection::Backward);
    #[cfg(feature = "fftw")]
    let plan = match backend {
        BackendChoice::Fftw(options) => {
            LocalC2cPlan::new_fftw_with_sign(global_shape[axis_index], sign, options)
                .map_err(|error| error.map_local(FftError::LocalC2c))?
        }
        BackendChoice::RustFft => unreachable!(),
    };
    Ok(TransformStage {
        axis: axis_index,
        input: Arc::clone(pencil),
        output: Arc::clone(pencil),
        local: LocalTransform::Complex(plan),
    })
}

#[cfg(not(feature = "fftw"))]
fn prepare_complex_stage_for_backend<R: FftReal, const N: usize, const M: usize>(
    pencil: &Arc<Pencil<N, M>>,
    global_shape: [usize; N],
    axis_index: usize,
    selected: bool,
    directions: FourierDirections<N>,
    _backend: BackendChoice,
) -> Result<TransformStage<R, N, M>, BackendInitError<FftError>> {
    prepare_complex_stage(pencil, global_shape, axis_index, selected, directions)
        .map_err(BackendInitError::Local)
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

fn prepare_c2c_stages_backend<R: FftReal, const N: usize, const M: usize>(
    route: &C2cRouteCandidate<N, M>,
    global_shape: [usize; N],
    selection: AxisSelection<N>,
    directions: FourierDirections<N>,
    backend: BackendChoice,
) -> Result<StagePreparation<R, N, M>, BackendInitError<FftError>> {
    let required = route.stages.len();
    let mut stages = Vec::new();
    stages
        .try_reserve_exact(required)
        .map_err(|_| BackendInitError::Local(FftError::AllocationFailed { required }))?;
    let mut fft_scratch_len = 0;
    for route_stage in route.stages.iter() {
        let mut stage = prepare_complex_stage_for_backend(
            &route_stage.output,
            global_shape,
            route_stage.axis,
            selection.contains(route_stage.axis),
            directions,
            backend,
        )?;
        stage.input = Arc::clone(&route_stage.source);
        stage.output = Arc::clone(&route_stage.output);
        fft_scratch_len = fft_scratch_len.max(stage.local.scratch_len());
        stages.push(stage);
    }
    Ok(StagePreparation {
        stages: stages.into_boxed_slice(),
        fft_scratch_len,
    })
}

/// Copies the stage input and output pencils into the two transition columns.
///
/// Transition `index` runs from `outputs[index]` to `inputs[index + 1]`; the
/// two columns differ at the real/complex boundary that a single pencil column
/// cannot express.
#[allow(clippy::type_complexity)]
fn pencil_columns<'a, I, J, const N: usize, const M: usize>(
    inputs: I,
    outputs: J,
) -> Result<(Box<[Arc<Pencil<N, M>>]>, Box<[Arc<Pencil<N, M>>]>), FftError>
where
    I: ExactSizeIterator<Item = &'a Arc<Pencil<N, M>>>,
    J: ExactSizeIterator<Item = &'a Arc<Pencil<N, M>>>,
{
    let required = inputs.len();
    if outputs.len() != required {
        return Err(FftError::PreparationFailed);
    }
    let mut input_column = Vec::new();
    input_column
        .try_reserve_exact(required)
        .map_err(|_| FftError::AllocationFailed { required })?;
    input_column.extend(inputs.map(Arc::clone));
    let mut output_column = Vec::new();
    output_column
        .try_reserve_exact(required)
        .map_err(|_| FftError::AllocationFailed { required })?;
    output_column.extend(outputs.map(Arc::clone));
    Ok((
        input_column.into_boxed_slice(),
        output_column.into_boxed_slice(),
    ))
}

fn build_transitions<const N: usize, const M: usize>(
    communicator: &mpi::topology::CartesianCommunicator,
    inputs: &[Arc<Pencil<N, M>>],
    outputs: &[Arc<Pencil<N, M>>],
    distributed: &[bool],
    extra_shape: &ExtraShape,
    method: TransposeMethod,
    real_transition_count: usize,
) -> Result<(Vec<C2cStageTransition<N, M>>, usize, usize, usize, usize), FftError> {
    let required_columns = distributed
        .len()
        .checked_add(1)
        .ok_or(FftError::PreparationFailed)?;
    if inputs.len() < required_columns || outputs.len() < required_columns {
        return Err(FftError::PreparationFailed);
    }
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
        let source = Arc::clone(&outputs[index]);
        let destination = Arc::clone(&inputs[index + 1]);
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
    for &extent in &shape[..position] {
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
    if !data_len.is_multiple_of(block) {
        return Err(LocalC2cError::NonIntegralBatch.into());
    }
    Ok(data_len / block)
}

/// Runs `transform` on each length-`line_len` strided line of `source` and writes
/// the transformed line back to the same strided position in `destination`.
///
/// The caller owns the fast paths: `line_buffer`, `count` (from
/// [`strided_line_count`]), the length checks, and the contiguous and empty
/// cases. Nothing is allocated here.
fn for_each_strided_line<T: Copy, E: From<FftError>>(
    source: &[T],
    destination: &mut [T],
    line_buffer: &mut [T],
    count: usize,
    line_len: usize,
    stride: usize,
    mut transform: impl FnMut(&mut [T]) -> Result<(), E>,
) -> Result<(), E> {
    let block = line_len
        .checked_mul(stride)
        .ok_or_else(|| E::from(FftError::PreparationFailed))?;
    for outer in 0..count {
        let base = outer
            .checked_mul(block)
            .ok_or_else(|| E::from(FftError::PreparationFailed))?;
        for inner in 0..stride {
            for k in 0..line_len {
                line_buffer[k] = source[base + k * stride + inner];
            }
            transform(&mut line_buffer[..line_len])?;
            for k in 0..line_len {
                destination[base + k * stride + inner] = line_buffer[k];
            }
        }
    }
    Ok(())
}

/// In-place variant of [`for_each_strided_line`]: the strided lines are gathered
/// from `data` and the transformed line is written back to `data`.
fn for_each_strided_line_in_place<T: Copy, E: From<FftError>>(
    data: &mut [T],
    line_buffer: &mut [T],
    count: usize,
    line_len: usize,
    stride: usize,
    mut transform: impl FnMut(&mut [T]) -> Result<(), E>,
) -> Result<(), E> {
    let block = line_len
        .checked_mul(stride)
        .ok_or_else(|| E::from(FftError::PreparationFailed))?;
    for outer in 0..count {
        let base = outer
            .checked_mul(block)
            .ok_or_else(|| E::from(FftError::PreparationFailed))?;
        for inner in 0..stride {
            for k in 0..line_len {
                line_buffer[k] = data[base + k * stride + inner];
            }
            transform(&mut line_buffer[..line_len])?;
            for k in 0..line_len {
                data[base + k * stride + inner] = line_buffer[k];
            }
        }
    }
    Ok(())
}

/// Runs `transform` on each strided line of `source` and writes the result to
/// the matching strided line of `destination`, for source and destination line
/// lengths that differ (the lengths of `source_line` and `destination_line`).
///
/// [`for_each_strided_line`] covers the equal-length case. The caller owns the
/// fast paths: `stride <= 1`, `count` (from [`strided_line_count`]), the line
/// buffer checks, and the length checks. Nothing is allocated here.
fn for_each_strided_line_differing<S: Copy, D: Copy, E: From<FftError>>(
    source: &[S],
    destination: &mut [D],
    source_line: &mut [S],
    destination_line: &mut [D],
    count: usize,
    stride: usize,
    mut transform: impl FnMut(&[S], &mut [D]) -> Result<(), E>,
) -> Result<(), E> {
    let source_line_len = source_line.len();
    let destination_line_len = destination_line.len();
    let source_block = source_line_len
        .checked_mul(stride)
        .ok_or_else(|| E::from(FftError::PreparationFailed))?;
    let destination_block = destination_line_len
        .checked_mul(stride)
        .ok_or_else(|| E::from(FftError::PreparationFailed))?;
    for outer in 0..count {
        let source_base = outer
            .checked_mul(source_block)
            .ok_or_else(|| E::from(FftError::PreparationFailed))?;
        let destination_base = outer
            .checked_mul(destination_block)
            .ok_or_else(|| E::from(FftError::PreparationFailed))?;
        for inner in 0..stride {
            for k in 0..source_line_len {
                source_line[k] = source[source_base + k * stride + inner];
            }
            transform(source_line, destination_line)?;
            for k in 0..destination_line_len {
                destination[destination_base + k * stride + inner] = destination_line[k];
            }
        }
    }
    Ok(())
}

/// In-place variant of [`for_each_strided_line_differing`] for storage whose
/// element type differs from the line element types: `read` and `write`
/// project between `storage` and the lines.
///
/// `source_block` and `destination_block` are the already checked element
/// counts of one outer group of `storage`; the line lengths are the lengths of
/// the line buffers.
#[allow(clippy::too_many_arguments)]
fn for_each_strided_line_differing_in_place<S, T: Copy, U: Copy, E: From<FftError>>(
    storage: &mut [S],
    source_line: &mut [T],
    destination_line: &mut [U],
    count: usize,
    source_block: usize,
    destination_block: usize,
    stride: usize,
    read: impl Fn(&S) -> T,
    write: impl Fn(&mut S, U),
    mut transform: impl FnMut(&[T], &mut [U]) -> Result<(), E>,
) -> Result<(), E> {
    let expected_source_block = source_line
        .len()
        .checked_mul(stride)
        .ok_or_else(|| E::from(FftError::PreparationFailed))?;
    let expected_destination_block = destination_line
        .len()
        .checked_mul(stride)
        .ok_or_else(|| E::from(FftError::PreparationFailed))?;
    let source_span = count
        .checked_mul(source_block)
        .ok_or_else(|| E::from(FftError::PreparationFailed))?;
    let destination_span = count
        .checked_mul(destination_block)
        .ok_or_else(|| E::from(FftError::PreparationFailed))?;
    if stride == 0
        || source_block < expected_source_block
        || destination_block < expected_destination_block
        || source_span > storage.len()
        || destination_span > storage.len()
    {
        return Err(E::from(FftError::PreparationFailed));
    }
    for outer in 0..count {
        let source_base = outer
            .checked_mul(source_block)
            .ok_or_else(|| E::from(FftError::PreparationFailed))?;
        let source_end = source_base
            .checked_add(source_block)
            .ok_or_else(|| E::from(FftError::PreparationFailed))?;
        let destination_base = outer
            .checked_mul(destination_block)
            .ok_or_else(|| E::from(FftError::PreparationFailed))?;
        let destination_end = destination_base
            .checked_add(destination_block)
            .ok_or_else(|| E::from(FftError::PreparationFailed))?;
        for inner in 0..stride {
            let source_group = storage
                .get(source_base..source_end)
                .ok_or_else(|| E::from(FftError::PreparationFailed))?;
            for (slot, chunk) in source_line
                .iter_mut()
                .zip(source_group.chunks_exact(stride))
            {
                *slot = read(&chunk[inner]);
            }
            transform(source_line, destination_line)?;
            let destination_group = storage
                .get_mut(destination_base..destination_end)
                .ok_or_else(|| E::from(FftError::PreparationFailed))?;
            for (element, chunk) in destination_line
                .iter()
                .zip(destination_group.chunks_exact_mut(stride))
            {
                write(&mut chunk[inner], *element);
            }
        }
    }
    Ok(())
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

fn prepare_strided_complex_lines<const N: usize, const M: usize>(
    pencil: &Pencil<N, M>,
    axis: usize,
    data_len: usize,
    line_len: usize,
    line_buffer_len: usize,
) -> Result<Option<(usize, usize)>, FftError> {
    let stride = memory_stride(pencil, axis)?;
    if stride <= 1 {
        return Ok(None);
    }
    validate_strided_line_buffer(line_buffer_len, line_len)?;
    let count = strided_line_count(data_len, line_len, stride)?;
    Ok(Some((stride, count)))
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
    let Some((stride, count)) = prepare_strided_complex_lines(
        pencil,
        axis,
        source.len(),
        plan.line_len(),
        line_buffer.len(),
    )?
    else {
        return plan
            .forward(source, destination, scratch)
            .map_err(FftError::LocalC2c);
    };
    if destination.len() != source.len() {
        return Err(FftError::LocalC2c(LocalC2cError::BufferLengthMismatch));
    }
    if source.is_empty() {
        return plan
            .forward(source, destination, scratch)
            .map_err(FftError::LocalC2c);
    }
    for_each_strided_line(
        source,
        destination,
        line_buffer,
        count,
        plan.line_len(),
        stride,
        |line| {
            plan.forward_in_place(line, scratch)
                .map_err(FftError::LocalC2c)
        },
    )
}

fn execute_strided_complex_forward_in_place<R: FftReal, const N: usize, const M: usize>(
    plan: &LocalC2cPlan<R>,
    pencil: &Pencil<N, M>,
    axis: usize,
    data: &mut [Complex<R>],
    scratch: &mut [Complex<R>],
    line_buffer: &mut [Complex<R>],
) -> Result<(), FftError> {
    let Some((stride, count)) = prepare_strided_complex_lines(
        pencil,
        axis,
        data.len(),
        plan.line_len(),
        line_buffer.len(),
    )?
    else {
        return plan
            .forward_in_place(data, scratch)
            .map_err(FftError::LocalC2c);
    };
    if data.is_empty() {
        return plan
            .forward_in_place(data, scratch)
            .map_err(FftError::LocalC2c);
    }
    for_each_strided_line_in_place(data, line_buffer, count, plan.line_len(), stride, |line| {
        plan.forward_in_place(line, scratch)
            .map_err(FftError::LocalC2c)
    })
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
    let Some((stride, count)) = prepare_strided_complex_lines(
        pencil,
        axis,
        source.len(),
        plan.line_len(),
        line_buffer.len(),
    )?
    else {
        return if normalize {
            plan.inverse(source, destination, scratch)
        } else {
            plan.backward(source, destination, scratch)
        }
        .map_err(FftError::LocalC2c);
    };
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
    for_each_strided_line(
        source,
        destination,
        line_buffer,
        count,
        plan.line_len(),
        stride,
        |line| {
            if normalize {
                plan.inverse_in_place(line, scratch)
            } else {
                plan.backward_in_place(line, scratch)
            }
            .map_err(FftError::LocalC2c)
        },
    )
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
    let Some((stride, count)) = prepare_strided_complex_lines(
        pencil,
        axis,
        data.len(),
        plan.line_len(),
        line_buffer.len(),
    )?
    else {
        return if normalize {
            plan.inverse_in_place(data, scratch)
        } else {
            plan.backward_in_place(data, scratch)
        }
        .map_err(FftError::LocalC2c);
    };
    if data.is_empty() {
        return if normalize {
            plan.inverse_in_place(data, scratch)
        } else {
            plan.backward_in_place(data, scratch)
        }
        .map_err(FftError::LocalC2c);
    }
    for_each_strided_line_in_place(data, line_buffer, count, plan.line_len(), stride, |line| {
        if normalize {
            plan.inverse_in_place(line, scratch)
        } else {
            plan.backward_in_place(line, scratch)
        }
        .map_err(FftError::LocalC2c)
    })
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
        workspace
            .intermediate
            .overwrite_with(stage.input.as_ref(), |mut target| {
                if target.as_mut_slice().len() != source_view.as_slice().len() {
                    return Err(());
                }
                target
                    .as_mut_slice()
                    .copy_from_slice(source_view.as_slice());
                Ok::<_, ()>(())
            })
            .expect("distributed C2C input overwrite was preflighted");
        if !matches!(&core.initial_transition.forward, C2cTransition::Identity) {
            execute_transition_timed(
                &core.initial_transition.forward,
                &mut workspace.intermediate,
                &mut workspace.transpose,
                report.as_deref_mut(),
                0,
            )?;
        }
        let mut active = workspace
            .intermediate
            .active_view_mut()
            .expect("initial C2C route layout was preflighted");
        let fft_started = Instant::now();
        execute_complex_forward_in_place(
            &stage.local,
            stage.output.as_ref(),
            stage.axis,
            active.as_mut_slice(),
            &mut workspace.fft_scratch,
            &mut workspace.line_buffer,
        )?;
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
        .overwrite_with(stage.input.as_ref(), |mut target| {
            if target.as_mut_slice().len() != source_view.as_slice().len() {
                return Err(FftError::PreparationFailed);
            }
            target
                .as_mut_slice()
                .copy_from_slice(source_view.as_slice());
            Ok::<_, FftError>(())
        })
        .map_err(|error| map_overwrite(error, FftError::Array))?;
    match &core.initial_transition.forward {
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
                &core.initial_transition.forward,
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
                stage.output.as_ref(),
                stage.axis,
                source_view.as_slice(),
                target.as_mut_slice(),
                &mut workspace.fft_scratch,
                &mut workspace.line_buffer,
                normalize_inverse,
            )
        })
        .map_err(|error| map_overwrite(error, FftError::Array))?;
    for (index, transition) in core.transitions.iter().enumerate().rev() {
        let stage = &core.stages[index];
        match &transition.backward {
            C2cTransition::PointToPoint(plan) => {
                let callback = |data: &mut [Complex<R>]| {
                    #[cfg(test)]
                    consume_c2c_callback_injection()?;
                    execute_complex_reverse_in_place(
                        &stage.local,
                        stage.output.as_ref(),
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
                    stage.output.as_ref(),
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
    execute_transition(
        &core.initial_transition.backward,
        &mut workspace.intermediate,
        &mut workspace.transpose,
    )?;
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

/// Repacks every `OverlapError` variant into a new callback error type.
fn map_overlap<E, F>(error: OverlapError<E>) -> FftOverlapError<F>
where
    E: Into<F>,
    F: std::fmt::Debug,
{
    FftOverlapError::Overlap(match error {
        OverlapError::Transpose(error) => OverlapError::Transpose(error),
        OverlapError::Callback(error) => OverlapError::Callback(error.into()),
        OverlapError::PeerPanicked => OverlapError::PeerPanicked,
        OverlapError::PeerCallbackFailed => OverlapError::PeerCallbackFailed,
        OverlapError::CollectivePreconditionFailed => OverlapError::CollectivePreconditionFailed,
    })
}

/// Maps an overwrite failure to any writer/array error pair reachable from the
/// caller, including the overlap wrapper whose array arm is an operation error.
fn map_overwrite<E, F>(error: OverwriteError<E>, array: impl FnOnce(ArrayError) -> F) -> F
where
    E: Into<F>,
{
    match error {
        OverwriteError::Array(error) => array(error),
        OverwriteError::Writer(error) => error.into(),
    }
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
    let source_view = source.view();
    let last = core.stages.len() - 1;
    let last_stage = &core.stages[last];
    let fft_started = Instant::now();
    workspace
        .intermediate
        .overwrite_with(last_stage.output.as_ref(), |mut target| {
            execute_complex_reverse(
                &last_stage.local,
                last_stage.output.as_ref(),
                last_stage.axis,
                source_view.as_slice(),
                target.as_mut_slice(),
                &mut workspace.fft_scratch,
                &mut workspace.line_buffer,
                normalize_inverse,
            )
        })
        .expect("distributed C2C inverse output overwrite was preflighted");
    record_fft_timing(&mut report, last, fft_started);

    for index in (0..core.transitions.len()).rev() {
        execute_transition_timed(
            &core.transitions[index].backward,
            &mut workspace.intermediate,
            &mut workspace.transpose,
            report.as_deref_mut(),
            index,
        )?;
        let stage = &core.stages[index];
        let mut active = workspace
            .intermediate
            .active_view_mut()
            .expect("C2C reverse route layout was preflighted");
        let fft_started = Instant::now();
        execute_complex_reverse_in_place(
            &stage.local,
            stage.output.as_ref(),
            stage.axis,
            active.as_mut_slice(),
            &mut workspace.fft_scratch,
            &mut workspace.line_buffer,
            normalize_inverse,
        )?;
        record_fft_timing(&mut report, index, fft_started);
    }
    if !matches!(&core.initial_transition.backward, C2cTransition::Identity) {
        execute_transition_timed(
            &core.initial_transition.backward,
            &mut workspace.intermediate,
            &mut workspace.transpose,
            report,
            0,
        )?;
    }
    let active = workspace
        .intermediate
        .active_view()
        .expect("distributed C2C inverse input layout was preflighted");
    destination
        .view_mut()
        .as_mut_slice()
        .copy_from_slice(active.as_slice());
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
    if !matches!(&core.initial_transition.forward, C2cTransition::Identity) {
        execute_transition_timed(
            &core.initial_transition.forward,
            array,
            transpose,
            report.as_deref_mut(),
            0,
        )?;
    }
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
            stage.output.as_ref(),
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
            stage.output.as_ref(),
            stage.axis,
            active.as_mut_slice(),
            fft_scratch,
            line_buffer,
            normalize_inverse,
        )?;
        record_fft_timing(&mut report, index, fft_started);
    }
    if !matches!(&core.initial_transition.backward, C2cTransition::Identity) {
        execute_transition_timed(
            &core.initial_transition.backward,
            array,
            transpose,
            report,
            0,
        )?;
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
        C2cTransition::Identity => Ok(None),
        C2cTransition::Local(plan) => {
            plan.execute_in_place_with_transpose_workspace(intermediate, workspace)
                .expect("distributed C2C local transition was preflighted");
            Ok(None)
        }
        C2cTransition::AllToAllv(plan) => {
            if report.is_some() {
                plan.execute_in_place_with_timing(intermediate, workspace)
                    .map(Some)
                    .map_err(FftError::Transpose)
            } else {
                plan.execute_in_place(intermediate, workspace)
                    .map(|_| None)
                    .map_err(FftError::Transpose)
            }
        }
        C2cTransition::PointToPoint(plan) => {
            if report.is_some() {
                plan.execute_in_place_with_timing(intermediate, workspace)
                    .map(Some)
                    .map_err(FftError::Transpose)
            } else {
                plan.execute_in_place(intermediate, workspace)
                    .map(|_| None)
                    .map_err(FftError::Transpose)
            }
        }
    };
    if let (Some(timing), Ok(Some(phases))) = (report.as_deref_mut(), result.as_ref()) {
        timing.stages[stage_index].communication.pack += phases.pack;
        timing.stages[stage_index].communication.post_receive += phases.post_receive;
        timing.stages[stage_index].communication.receive_wait += phases.receive_wait;
        timing.stages[stage_index].communication.collective_wait += phases.collective_wait;
        timing.stages[stage_index].communication.unpack += phases.unpack;
        timing.stages[stage_index].communication.send_wait += phases.send_wait;
        timing.stages[stage_index].communication.total += phases.total;
    }
    let result = result.map(|_| ());
    if let Some(timing) = report {
        timing.stages[stage_index].transpose += started.elapsed();
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
    let mut paired = [0u64; 2 * HEADER_WORDS];
    let mut reduced = [0u64; 2 * HEADER_WORDS];
    let (top, bottom) = paired.split_at_mut(HEADER_WORDS);
    top.copy_from_slice(&header);
    for (destination, &value) in bottom.iter_mut().zip(header.iter()) {
        *destination = !value;
    }
    comm.all_reduce_into(&paired[..], &mut reduced[..], SystemOperation::max());
    // The second half is the complement of the global minimum, so comparing it
    // with the maximum half is the same exact agreement as min == max.
    let (maximum, complemented_minimum) = reduced.split_at(HEADER_WORDS);
    maximum
        .iter()
        .zip(complemented_minimum)
        .all(|(&maximum, &complemented_minimum)| maximum == !complemented_minimum)
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
    if !core.stages.iter().any(|stage| {
        active.same_layout(stage.input.as_ref()) || active.same_layout(stage.output.as_ref())
    }) {
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

fn endpoint_chunk_count(batch_count: usize) -> usize {
    batch_count.div_ceil(ENDPOINT_BATCH_CHUNK)
}

fn endpoint_chunk_len(batch_count: usize, chunk_start: usize) -> usize {
    (batch_count - chunk_start).min(ENDPOINT_BATCH_CHUNK)
}

/// The ULP neighbors of `threshold` in `R`, with the exact value in the middle.
///
/// Threshold fixtures use exactly representable values so that the middle
/// neighbor pins the analytic threshold and MPI's reduction order cannot
/// change its decision.
#[cfg(test)]
fn endpoint_ulp_neighbors<R: FftReal>(threshold: f64) -> [f64; 3] {
    let neighbors = if size_of::<R>() == size_of::<f32>() {
        let bits = (threshold as f32).to_bits();
        [bits - 1, bits, bits + 1].map(|bits| f32::from_bits(bits) as f64)
    } else {
        let bits = threshold.to_bits();
        [bits - 1, bits, bits + 1].map(f64::from_bits)
    };
    assert_eq!(neighbors[1], threshold, "fixture threshold must be exact");
    neighbors
}

/// Allocates the three chunk statistics buffers before any collective call.
///
/// The slot order is local statistics, global maxima, and global sums. A
/// failure here must be agreed before the first statistics reduction, so the
/// caller allocates all three inside its preparation result.
fn endpoint_chunk_buffers(length: usize) -> Result<[Vec<[f64; 4]>; 3], FftError> {
    let mut buffers: [Vec<[f64; 4]>; 3] = std::array::from_fn(|_| Vec::new());
    for (slot, buffer) in buffers.iter_mut().enumerate() {
        #[cfg(test)]
        {
            record_endpoint_reduction(EndpointReductionEvent::Allocation { slot, length });
            let failure = ENDPOINT_ALLOCATION_FAILURE.with(std::cell::Cell::get);
            if failure == Some(slot) {
                return Err(FftError::AllocationFailed { required: length });
            }
        }
        #[cfg(not(test))]
        let _ = slot;
        *buffer = initialized_vec(length, [0.0_f64; 4])?;
    }
    Ok(buffers)
}

/// The chunked endpoint statistics shared by the distributed R2C and mixed
/// boundary checks.
///
/// Each implementation keeps its own coordinate walk, threshold values, and
/// preparation checks. The accumulation, both reductions, the acceptance
/// check, and the final agreement stay here so the collective sequence cannot
/// diverge between the two plans.
struct EndpointStatistics {
    plane_count: usize,
    relative: f64,
    absolute: f64,
    invalid: bool,
    local: Vec<[f64; 4]>,
    maxima: Vec<[f64; 4]>,
    sums: Vec<[f64; 4]>,
}

impl EndpointStatistics {
    /// Allocates every chunk buffer before the first collective call.
    ///
    /// A failure here belongs to the caller's agreed preparation result.
    fn new(
        batch_count: usize,
        plane_count: usize,
        relative: f64,
        absolute: f64,
    ) -> Result<Self, FftError> {
        let [local, maxima, sums] = endpoint_chunk_buffers(batch_count.min(ENDPOINT_BATCH_CHUNK))?;
        Ok(Self {
            plane_count,
            relative,
            absolute,
            invalid: false,
            local,
            maxima,
            sums,
        })
    }

    /// Reduces and checks every statistics chunk of one boundary.
    ///
    /// `walk` visits one batch and reports each constrained endpoint as
    /// `(2 * plane, value)` through its visitor; the slot choice therefore
    /// stays with each implementation. Every chunk runs the MAX reduction, the
    /// scaled-square SUM reduction, and the acceptance check on every rank, and
    /// a local numeric failure only marks the result invalid.
    fn reduce_chunks<C, R, W>(
        &mut self,
        communicator: &C,
        batch_count: usize,
        data: &[Complex<R>],
        local_len: usize,
        mut walk: W,
    ) where
        C: CommunicatorCollectives,
        R: FftReal,
        W: FnMut(&[Complex<R>], &mut dyn FnMut(usize, Complex<R>)),
    {
        let plane_count = self.plane_count;
        let relative = self.relative;
        let absolute = self.absolute;
        let Self {
            invalid,
            local: local_stats,
            maxima: global_maxima,
            sums: global_sums,
            ..
        } = self;
        for chunk_index in 0..endpoint_chunk_count(batch_count) {
            let chunk_start = chunk_index * ENDPOINT_BATCH_CHUNK;
            let chunk = endpoint_chunk_len(batch_count, chunk_start);
            let local = &mut local_stats[..chunk];
            let maxima = &mut global_maxima[..chunk];
            let sums = &mut global_sums[..chunk];

            local.fill([0.0_f64; 4]);
            for (batch_max, batch) in local.iter_mut().zip(chunk_start..chunk_start + chunk) {
                let start = batch
                    .checked_mul(local_len)
                    .expect("validated boundary batch offset");
                walk(&data[start..start + local_len], &mut |slot, value| {
                    let re = <R as crate::private::Sealed>::pencil_fft_as_f64(value.re);
                    let im = <R as crate::private::Sealed>::pencil_fft_as_f64(value.im);
                    if !re.is_finite() || !im.is_finite() {
                        *invalid = true;
                        return;
                    }
                    batch_max[slot] = batch_max[slot].max(re.abs().max(im.abs()));
                    batch_max[slot + 1] = batch_max[slot + 1].max(im.abs());
                });
            }
            #[cfg(test)]
            record_endpoint_reduction(EndpointReductionEvent::MaxReduction {
                elements: 4 * chunk,
            });
            communicator.all_reduce_into(
                local.as_flattened(),
                maxima.as_flattened_mut(),
                SystemOperation::max(),
            );

            local.fill([0.0_f64; 4]);
            for (batch, (local_sum, global_max)) in local.iter_mut().zip(maxima.iter()).enumerate()
            {
                let start = (chunk_start + batch)
                    .checked_mul(local_len)
                    .expect("validated boundary batch offset");
                walk(&data[start..start + local_len], &mut |slot, value| {
                    let re = <R as crate::private::Sealed>::pencil_fft_as_f64(value.re);
                    let im = <R as crate::private::Sealed>::pencil_fft_as_f64(value.im);
                    if !re.is_finite() || !im.is_finite() {
                        return;
                    }
                    let scale = global_max[slot];
                    if scale != 0.0 && scale.is_finite() {
                        let normalized_re = re / scale;
                        let normalized_im = im / scale;
                        local_sum[slot] += normalized_re * normalized_re;
                        local_sum[slot + 1] += normalized_im * normalized_im;
                    }
                });
            }
            #[cfg(test)]
            record_endpoint_reduction(EndpointReductionEvent::SumReduction {
                elements: 4 * chunk,
            });
            communicator.all_reduce_into(
                local.as_flattened(),
                sums.as_flattened_mut(),
                SystemOperation::sum(),
            );

            for batch in 0..chunk {
                let global_max = maxima[batch];
                let global_sum = sums[batch];
                if global_max[..plane_count * 2]
                    .iter()
                    .any(|value| !value.is_finite())
                    || global_sum[..plane_count * 2]
                        .iter()
                        .any(|value| !value.is_finite())
                    || !relative.is_finite()
                    || !absolute.is_finite()
                {
                    *invalid = true;
                }
                for plane in 0..plane_count {
                    let slot = plane * 2;
                    let absolute_ok = global_max[slot + 1] <= absolute;
                    let relative_ok =
                        global_sum[slot + 1].sqrt() <= relative * global_sum[slot].sqrt();
                    if !(absolute_ok || relative_ok) {
                        *invalid = true;
                    }
                }
            }
        }
    }

    /// Returns whether every rank accepted the spectrum.
    fn accepted<C: CommunicatorCollectives>(&self, communicator: &C) -> bool {
        #[cfg(test)]
        record_endpoint_reduction(EndpointReductionEvent::FinalAgreement);
        collective_valid(communicator, !self.invalid)
    }
}

/// Repacks strided real boundary lines in place into the compressed complex
/// layout.
///
/// `real_source_line`, `complex_line`, `real_line`, and `scratch` are the
/// plan's line and local transform buffers. The caller keeps array ownership,
/// poison handling, and the no-rows case.
fn convert_strided_real_forward<R, E>(
    storage: &mut [Complex<R>],
    plan: &LocalR2cPlan<R>,
    rows: usize,
    real_len: usize,
    complex_len: usize,
    stride: usize,
    real_source_line: &mut [R],
    complex_line: &mut [Complex<R>],
    real_line: &mut [R],
    scratch: &mut [Complex<R>],
) -> Result<(), E>
where
    R: FftReal,
    Complex<R>: Equivalence,
    E: From<FftError> + From<LocalR2cError>,
{
    if stride <= 1 || !rows.is_multiple_of(stride) {
        return Err(FftError::PreparationFailed.into());
    }
    let groups = rows / stride;
    let source_block = real_len
        .checked_mul(stride)
        .ok_or(FftError::PreparationFailed)?;
    let destination_block = complex_len
        .checked_mul(stride)
        .ok_or(FftError::PreparationFailed)?;
    let source_span = groups
        .checked_mul(source_block)
        .ok_or(FftError::PreparationFailed)?;
    let destination_span = groups
        .checked_mul(destination_block)
        .ok_or(FftError::PreparationFailed)?;
    if source_span > storage.len() || destination_span > storage.len() {
        return Err(FftError::StorageLayoutMismatch.into());
    }
    if real_source_line.len() < real_len
        || complex_line.len() < complex_len
        || real_line.len() < real_len
    {
        return Err(FftError::WorkspaceTooSmall {
            kind: "real/complex line",
            required: real_len.max(complex_len),
            actual: real_source_line
                .len()
                .min(complex_line.len())
                .min(real_line.len()),
        }
        .into());
    }
    for_each_strided_line_differing_in_place(
        storage,
        &mut real_source_line[..real_len],
        &mut complex_line[..complex_len],
        groups,
        source_block,
        destination_block,
        stride,
        |value: &Complex<R>| value.re,
        |slot: &mut Complex<R>, value: Complex<R>| *slot = value,
        |source, destination| {
            plan.forward(source, destination, real_line, scratch)
                .map_err(E::from)
        },
    )
}

/// Repacks compressed complex boundary lines in place into the strided real
/// layout.
///
/// `complex_source_line`, `real_line`, `complex_line`, and `scratch` are the
/// plan's line and local transform buffers. The caller keeps array ownership,
/// poison handling, and the no-rows case.
fn convert_strided_complex_reverse<R, E>(
    storage: &mut [Complex<R>],
    plan: &LocalR2cPlan<R>,
    rows: usize,
    real_len: usize,
    complex_len: usize,
    stride: usize,
    complex_source_line: &mut [Complex<R>],
    real_line: &mut [R],
    complex_line: &mut [Complex<R>],
    scratch: &mut [Complex<R>],
    normalize_inverse: bool,
) -> Result<(), E>
where
    R: FftReal,
    Complex<R>: Equivalence,
    E: From<FftError> + From<LocalR2cError>,
{
    if stride <= 1 || !rows.is_multiple_of(stride) {
        return Err(FftError::PreparationFailed.into());
    }
    let groups = rows / stride;
    let source_block = complex_len
        .checked_mul(stride)
        .ok_or(FftError::PreparationFailed)?;
    let destination_block = real_len
        .checked_mul(stride)
        .ok_or(FftError::PreparationFailed)?;
    let source_span = groups
        .checked_mul(source_block)
        .ok_or(FftError::PreparationFailed)?;
    let destination_span = groups
        .checked_mul(destination_block)
        .ok_or(FftError::PreparationFailed)?;
    if source_span > storage.len() || destination_span > storage.len() {
        return Err(FftError::StorageLayoutMismatch.into());
    }

    // Expand compressed outer blocks from the back.  Each move is overlap
    // safe, and a later (smaller) q never overwrites a source block still to
    // be moved.
    for outer in (0..groups).rev() {
        let source_start = outer
            .checked_mul(source_block)
            .ok_or(FftError::PreparationFailed)?;
        let source_end = source_start
            .checked_add(source_block)
            .ok_or(FftError::PreparationFailed)?;
        let destination_start = outer
            .checked_mul(destination_block)
            .ok_or(FftError::PreparationFailed)?;
        storage.copy_within(source_start..source_end, destination_start);
    }

    if complex_source_line.len() < complex_len
        || complex_line.len() < complex_len
        || real_line.len() < real_len
    {
        return Err(FftError::WorkspaceTooSmall {
            kind: "complex/real line",
            required: real_len.max(complex_len),
            actual: complex_source_line
                .len()
                .min(complex_line.len())
                .min(real_line.len()),
        }
        .into());
    }
    // Store the C2R result in Complex::re slots.  Residues modulo S are
    // disjoint, so every other inner line remains intact until it has been
    // gathered.  The scalar projection out of Complex::re is left to the
    // caller, which performs it once this function has consumed all source
    // lines.  Both directions index the expanded real block.
    for_each_strided_line_differing_in_place(
        storage,
        &mut complex_source_line[..complex_len],
        &mut real_line[..real_len],
        groups,
        destination_block,
        destination_block,
        stride,
        |value: &Complex<R>| *value,
        |slot: &mut Complex<R>, value: R| slot.re = value,
        |source, destination| {
            if normalize_inverse {
                plan.inverse(source, destination, complex_line, scratch)
            } else {
                plan.backward(source, destination, complex_line, scratch)
            }
            .map_err(E::from)
        },
    )
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
#[path = "distributed/tests/layout_tests.rs"]
mod tests;
