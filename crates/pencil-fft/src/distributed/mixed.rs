//! Heterogeneous distributed FFT, R2R, and real-to-complex plans.
//!
//! The mixed plans deliberately have their own cores and collective operation
//! words.  They share the canonical route and checked transition builders with
//! the homogeneous plans, but their stage opcodes and endpoint types are not
//! interchangeable with legacy plans.

#![allow(clippy::too_many_arguments)]

use std::{
    mem::{align_of, size_of},
    sync::Arc,
    time::Instant,
};

use bytemuck::{try_cast_slice, try_cast_vec};
use mpi::{
    collective::{CommunicatorCollectives, SystemOperation},
    datatype::Equivalence,
};
use pencil_array::{
    ArrayError, ExtraShape, ManyPencilArray, MpiTopology, OverlapError, OverwriteError, Pencil,
    PencilArray, PencilArrayView, PencilArrayViewMut, TransposeWorkspace,
};
use thiserror::Error;

use crate::BackendInitError;
#[cfg(feature = "fftw")]
use crate::PlanOptions;
use crate::{
    Complex, FftReal, LocalC2cError, LocalC2cPlan, LocalR2cError, LocalR2cPlan, LocalR2rError,
    LocalR2rPlan, R2cState, R2rScalar,
};

impl From<FftError> for FftOverlapError<MixedError> {
    fn from(error: FftError) -> Self {
        Self::Operation(error.into())
    }
}

use super::{
    BackendChoice, C2cStageTransition, C2cState, Direction, DistributedLayout, FftError,
    FftOverlapError, FourierDirection, FourierDirections, INVALID_WORD, LocalTransform,
    OPERATION_MIXED_C2C_BACKWARD, OPERATION_MIXED_C2C_BACKWARD_IN_PLACE,
    OPERATION_MIXED_C2C_FORWARD, OPERATION_MIXED_C2C_FORWARD_IN_PLACE, OPERATION_MIXED_C2C_INVERSE,
    OPERATION_MIXED_C2C_INVERSE_IN_PLACE, OPERATION_MIXED_R2C_BACKWARD,
    OPERATION_MIXED_R2C_BACKWARD_IN_PLACE, OPERATION_MIXED_R2C_FORWARD,
    OPERATION_MIXED_R2C_FORWARD_IN_PLACE, OPERATION_MIXED_R2C_INVERSE,
    OPERATION_MIXED_R2C_INVERSE_IN_PLACE, RouteCandidate, StagePreparation, TransformStage,
    TransformTiming, TransposeMethod, VALUE_KIND_C2C, VALUE_KIND_R2C,
    agree_execution_descriptor_ref, agree_header, agree_result, build_route, build_transitions,
    collective_descriptor, collective_valid, initialized_vec, map_array_allocation, memory_stride,
    strided_line_count, validate_input, validate_workspace_lengths_values, zero_complex,
};

use crate::r2r::AxisR2rKind;

// Kept separate from the homogeneous overlap words: the five-word outer
// descriptor agreement must reject a mixed/non-mixed call before P2P starts.
#[cfg(feature = "fftw")]
const OPERATION_MIXED_C2C_PLAN_NATIVE: u64 = 131;
#[cfg(feature = "fftw")]
const OPERATION_MIXED_R2C_PLAN_NATIVE: u64 = 132;
const OPERATION_MIXED_C2C_FORWARD_OVERLAP: u64 = 85;
const OPERATION_MIXED_C2C_INVERSE_OVERLAP: u64 = 86;
const OPERATION_MIXED_C2C_BACKWARD_OVERLAP: u64 = 87;
const OPERATION_MIXED_R2C_FORWARD_OVERLAP: u64 = 88;
const OPERATION_MIXED_R2C_INVERSE_OVERLAP: u64 = 89;
const OPERATION_MIXED_R2C_BACKWARD_OVERLAP: u64 = 90;
// Mixed timing calls use their own collective descriptor words.
const OPERATION_MIXED_FORWARD_TIMED: u64 = 91;
const OPERATION_MIXED_INVERSE_TIMED: u64 = 92;
const OPERATION_MIXED_BACKWARD_TIMED: u64 = 93;
const OPERATION_MIXED_FORWARD_IN_PLACE_TIMED: u64 = 94;
const OPERATION_MIXED_INVERSE_IN_PLACE_TIMED: u64 = 95;
const OPERATION_MIXED_BACKWARD_IN_PLACE_TIMED: u64 = 96;
const OPERATION_MIXED_R2C_FORWARD_TIMED: u64 = 115;
const OPERATION_MIXED_R2C_INVERSE_TIMED: u64 = 116;
const OPERATION_MIXED_R2C_BACKWARD_TIMED: u64 = 117;
const OPERATION_MIXED_R2C_FORWARD_IN_PLACE_TIMED: u64 = 118;
const OPERATION_MIXED_R2C_INVERSE_IN_PLACE_TIMED: u64 = 119;
const OPERATION_MIXED_R2C_BACKWARD_IN_PLACE_TIMED: u64 = 120;

/// A concrete transform assigned to one logical spatial axis of a mixed plan.
///
/// `None` is an identity stage. `Fft` is a complex-to-complex FFT. `Rfft` is
/// the one real-to-half-complex boundary accepted by [`MixedR2cPlan`]. `R2r`
/// selects one of the eight FFTW DCT/DST kinds or the discrete Hartley
/// transform. `Rfft` is rejected by [`MixedC2cPlan`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AxisTransform {
    /// Leave this axis unchanged.
    None,
    /// Apply a complex FFT on this axis.
    Fft,
    /// Apply the real-to-half-complex boundary on this axis.
    Rfft,
    /// Apply a DCT, DST, or Hartley transform on this axis.
    R2r(AxisR2rKind),
}

/// Errors returned by heterogeneous distributed plans.
#[derive(Debug, Error)]
pub enum MixedError {
    /// A shared distributed validation or transition failed.
    #[error(transparent)]
    Fft(#[from] FftError),
    /// A complex FFT stage rejected checked input.
    #[error(transparent)]
    LocalC2c(#[from] LocalC2cError),
    /// A DCT, DST, or Hartley stage rejected checked input.
    #[error(transparent)]
    LocalR2r(#[from] LocalR2rError),
    /// The real-to-complex boundary rejected checked input.
    #[error(transparent)]
    LocalR2c(#[from] LocalR2cError),
    /// The axis graph is not valid for the selected mixed plan family.
    #[error("invalid mixed transform axis graph")]
    InvalidGraph,
    /// The constrained Fourier boundary was not sufficiently real.
    #[error("distributed mixed inverse spectrum has an invalid constrained boundary plane")]
    InvalidSpectrum,
}

impl From<FftError> for BackendInitError<MixedError> {
    fn from(error: FftError) -> Self {
        Self::Local(MixedError::Fft(error))
    }
}

/// Completion state for a [`MixedC2cInPlaceArray`].
pub type MixedC2cState = C2cState;

/// Completion state for a [`MixedR2cInPlaceArray`].
pub type MixedR2cState = R2cState;

#[derive(Debug)]
enum MixedR2rLocal<T: R2rScalar> {
    Transform(LocalR2rPlan<T>),
    Hartley(crate::LocalDhtPlan<T>),
}

impl<T: R2rScalar> MixedR2rLocal<T> {
    fn embedding_len(&self) -> usize {
        match self {
            Self::Transform(plan) => plan.embedding_len(),
            Self::Hartley(plan) => plan.embedding_len(),
        }
    }

    fn scratch_len(&self) -> usize {
        match self {
            Self::Transform(plan) => plan.scratch_len(),
            Self::Hartley(plan) => plan.scratch_len(),
        }
    }

    fn line_len(&self) -> usize {
        match self {
            Self::Transform(plan) => plan.line_len(),
            Self::Hartley(plan) => plan.line_len(),
        }
    }
}

#[derive(Debug)]
enum MixedComplexLocal<R: FftReal> {
    Identity,
    Fft(LocalC2cPlan<R>),
    R2r(MixedR2rLocal<Complex<R>>),
}

impl<R: FftReal> MixedComplexLocal<R> {
    fn embedding_len(&self) -> usize {
        match self {
            Self::Identity | Self::Fft(_) => 0,
            Self::R2r(plan) => plan.embedding_len(),
        }
    }

    fn scratch_len(&self) -> usize {
        match self {
            Self::Identity => 0,
            Self::Fft(plan) => plan.scratch_len(),
            Self::R2r(plan) => plan.scratch_len(),
        }
    }

    fn line_len(&self) -> usize {
        match self {
            Self::Identity => 0,
            Self::Fft(plan) => plan.line_len(),
            Self::R2r(plan) => plan.line_len(),
        }
    }
}

#[derive(Debug)]
enum MixedRealLocal<R: FftReal> {
    Identity,
    Rfft(LocalR2cPlan<R>),
    R2r(MixedR2rLocal<R>),
}

impl<R: FftReal> MixedRealLocal<R> {
    fn embedding_len(&self) -> usize {
        match self {
            Self::Identity | Self::Rfft(_) => 0,
            Self::R2r(plan) => plan.embedding_len(),
        }
    }

    fn scratch_len(&self) -> usize {
        match self {
            Self::Identity => 0,
            Self::Rfft(plan) => plan.scratch_len(),
            Self::R2r(plan) => plan.scratch_len(),
        }
    }

    fn line_len(&self) -> usize {
        match self {
            Self::Identity => 0,
            Self::Rfft(plan) => plan.real_len(),
            Self::R2r(plan) => plan.line_len(),
        }
    }
}

#[derive(Debug)]
struct MixedC2cStage<R: FftReal, const N: usize, const M: usize> {
    axis: usize,
    input: Arc<Pencil<N, M>>,
    output: Arc<Pencil<N, M>>,
    local: MixedComplexLocal<R>,
}

#[derive(Debug)]
struct MixedC2cCore<R: FftReal, const N: usize, const M: usize> {
    stages: Box<[MixedC2cStage<R, N, M>]>,
    transitions: Box<[C2cStageTransition<N, M>]>,
    registered_pencils: Box<[Arc<Pencil<N, M>>]>,
    extra_shape: ExtraShape,
    transforms: [AxisTransform; N],
    layout: DistributedLayout,
    descriptor: Box<[u64]>,
    embedding_len: usize,
    fft_scratch_len: usize,
    strided_line_len: usize,
    transpose_send_len: usize,
    transpose_receive_len: usize,
    directions: FourierDirections<N>,
    backend: BackendChoice,
    // Reconfigured plans reject arrays belonging to the prior core; legacy
    // constructors retain their existing layout-compatible array contract.
    strict_array_identity: bool,
}

#[derive(Debug)]
enum MixedR2cStageLocal<R: FftReal> {
    Real(MixedRealLocal<R>),
    Complex(MixedComplexLocal<R>),
}

#[derive(Debug)]
struct MixedR2cStage<R: FftReal, const N: usize, const M: usize> {
    axis: usize,
    input: Arc<Pencil<N, M>>,
    output: Arc<Pencil<N, M>>,
    local: MixedR2cStageLocal<R>,
}

#[derive(Debug)]
struct MixedR2cCore<R: FftReal, const N: usize, const M: usize> {
    stages: Box<[MixedR2cStage<R, N, M>]>,
    transitions: Box<[C2cStageTransition<N, M>]>,
    real_pencils: Box<[Arc<Pencil<N, M>>]>,
    complex_pencils: Box<[Arc<Pencil<N, M>>]>,
    extra_shape: ExtraShape,
    transforms: [AxisTransform; N],
    layout: DistributedLayout,
    descriptor: Box<[u64]>,
    real_stage_index: usize,
    real_len: usize,
    complex_len: usize,
    embedding_len: usize,
    fft_scratch_len: usize,
    real_line_len: usize,
    complex_line_len: usize,
    transpose_send_len: usize,
    transpose_receive_len: usize,
    real_transpose_send_len: usize,
    real_transpose_receive_len: usize,
    raw_absolute_threshold: f64,
    directions: FourierDirections<N>,
    backend: BackendChoice,
    // Reconfigured plans reject arrays belonging to the prior core; legacy
    // constructors retain their existing layout-compatible array contract.
    strict_array_identity: bool,
}

/// Reusable out-of-place workspace for [`MixedC2cPlan`].
#[derive(Debug)]
pub struct MixedC2cWorkspace<R: FftReal, const N: usize, const M: usize> {
    core: Arc<MixedC2cCore<R, N, M>>,
    intermediate: ManyPencilArray<Complex<R>, N, M>,
    transpose: TransposeWorkspace<Complex<R>>,
    embedding_line: Vec<Complex<R>>,
    fft_scratch: Vec<Complex<R>>,
    line_buffer: Vec<Complex<R>>,
}

/// Compatibility alias for the homogeneous-plan workspace spelling.
pub type MixedC2cOutOfPlaceWorkspace<R, const N: usize, const M: usize> =
    MixedC2cWorkspace<R, N, M>;

/// A state-checked one-buffer array for [`MixedC2cPlan`].
#[derive(Debug)]
pub struct MixedC2cInPlaceArray<R: FftReal, const N: usize, const M: usize> {
    core: Arc<MixedC2cCore<R, N, M>>,
    array: ManyPencilArray<Complex<R>, N, M>,
    state: C2cState,
    #[cfg(test)]
    test_hook: Option<MixedInPlaceTestHook>,
}

/// Reusable in-place workspace for [`MixedC2cPlan`].
#[derive(Debug)]
pub struct MixedC2cInPlaceWorkspace<R: FftReal, const N: usize, const M: usize> {
    core: Arc<MixedC2cCore<R, N, M>>,
    transpose: TransposeWorkspace<Complex<R>>,
    embedding_line: Vec<Complex<R>>,
    fft_scratch: Vec<Complex<R>>,
    line_buffer: Vec<Complex<R>>,
}

/// Reusable out-of-place workspace for [`MixedR2cPlan`].
#[derive(Debug)]
pub struct MixedR2cWorkspace<R: FftReal, const N: usize, const M: usize> {
    core: Arc<MixedR2cCore<R, N, M>>,
    intermediate: ManyPencilArray<Complex<R>, N, M>,
    real_intermediate: Option<ManyPencilArray<R, N, M>>,
    transpose: TransposeWorkspace<Complex<R>>,
    real_transpose: Option<TransposeWorkspace<R>>,
    embedding_line: Vec<Complex<R>>,
    fft_scratch: Vec<Complex<R>>,
    real_source_line: Vec<R>,
    real_line: Vec<R>,
    complex_source_line: Vec<Complex<R>>,
    complex_line: Vec<Complex<R>>,
    real_strided_line: Vec<R>,
    complex_strided_line: Vec<Complex<R>>,
}

/// Compatibility alias for the homogeneous-plan workspace spelling.
pub type MixedR2cOutOfPlaceWorkspace<R, const N: usize, const M: usize> =
    MixedR2cWorkspace<R, N, M>;

#[derive(Debug)]
enum MixedR2cStorage<R: FftReal, const N: usize, const M: usize> {
    Real(ManyPencilArray<R, N, M>),
    Complex(ManyPencilArray<Complex<R>, N, M>),
    PoisonedReal(Vec<R>),
    PoisonedComplex(Vec<Complex<R>>),
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MixedInPlaceTestHook {
    Start,
    ForwardDetach,
    ReverseDetach,
}

/// A one-allocation state-checked array for [`MixedR2cPlan`].
#[derive(Debug)]
pub struct MixedR2cInPlaceArray<R: FftReal, const N: usize, const M: usize> {
    core: Arc<MixedR2cCore<R, N, M>>,
    real_pencils: Option<Box<[Arc<Pencil<N, M>>]>>,
    complex_pencils: Option<Box<[Arc<Pencil<N, M>>]>>,
    real_storage_len: usize,
    complex_storage_len: usize,
    storage_bytes: usize,
    storage: Option<MixedR2cStorage<R, N, M>>,
    state: R2cState,
    #[cfg(test)]
    test_hook: Option<MixedInPlaceTestHook>,
}

/// Reusable in-place workspace for [`MixedR2cPlan`].
#[derive(Debug)]
pub struct MixedR2cInPlaceWorkspace<R: FftReal, const N: usize, const M: usize> {
    core: Arc<MixedR2cCore<R, N, M>>,
    transpose: TransposeWorkspace<Complex<R>>,
    real_transpose: Option<TransposeWorkspace<R>>,
    embedding_line: Vec<Complex<R>>,
    fft_scratch: Vec<Complex<R>>,
    real_source_line: Vec<R>,
    real_line: Vec<R>,
    complex_source_line: Vec<Complex<R>>,
    complex_line: Vec<Complex<R>>,
    real_strided_line: Vec<R>,
    complex_strided_line: Vec<Complex<R>>,
}

/// A heterogeneous complex-to-complex distributed transform.
#[derive(Debug)]
pub struct MixedC2cPlan<R: FftReal, const N: usize, const M: usize> {
    core: Arc<MixedC2cCore<R, N, M>>,
}

/// A heterogeneous real-to-complex distributed transform.
#[derive(Debug)]
pub struct MixedR2cPlan<R: FftReal, const N: usize, const M: usize> {
    core: Arc<MixedR2cCore<R, N, M>>,
}

impl<R: FftReal, const N: usize, const M: usize> MixedC2cPlan<R, N, M>
where
    Complex<R>: Equivalence,
{
    /// Builds an Alltoallv mixed complex plan from a canonical pencil.
    pub fn from_pencil(
        input: Arc<Pencil<N, M>>,
        extra_shape: ExtraShape,
        transforms: [AxisTransform; N],
    ) -> Result<Self, MixedError> {
        Self::from_pencil_with_layout(input, extra_shape, transforms, DistributedLayout::default())
    }

    /// Builds a mixed complex plan with the selected transition method.
    pub fn from_pencil_with_method(
        input: Arc<Pencil<N, M>>,
        extra_shape: ExtraShape,
        transforms: [AxisTransform; N],
        method: TransposeMethod,
    ) -> Result<Self, MixedError> {
        Self::from_pencil_with_layout(
            input,
            extra_shape,
            transforms,
            DistributedLayout {
                transpose_method: method,
                ..DistributedLayout::default()
            },
        )
    }

    /// Builds a mixed complex plan with an explicit layout policy.
    pub fn from_pencil_with_layout(
        input: Arc<Pencil<N, M>>,
        extra_shape: ExtraShape,
        transforms: [AxisTransform; N],
        layout: DistributedLayout,
    ) -> Result<Self, MixedError> {
        let topology = Arc::clone(input.topology());
        Self::construct(
            topology,
            *input.global_shape(),
            extra_shape,
            Ok(input),
            transforms,
            layout,
            FourierDirections::default(),
        )
    }

    /// Builds an Alltoallv mixed complex plan from a canonical array.
    pub fn from_array(
        input: &PencilArray<Complex<R>, N, M>,
        transforms: [AxisTransform; N],
    ) -> Result<Self, MixedError> {
        Self::from_array_with_layout(input, transforms, DistributedLayout::default())
    }

    /// Builds a mixed complex plan from an array and transition method.
    pub fn from_array_with_method(
        input: &PencilArray<Complex<R>, N, M>,
        transforms: [AxisTransform; N],
        method: TransposeMethod,
    ) -> Result<Self, MixedError> {
        Self::from_array_with_layout(
            input,
            transforms,
            DistributedLayout {
                transpose_method: method,
                ..DistributedLayout::default()
            },
        )
    }

    /// Builds a mixed complex plan from an array and explicit layout.
    pub fn from_array_with_layout(
        input: &PencilArray<Complex<R>, N, M>,
        transforms: [AxisTransform; N],
        layout: DistributedLayout,
    ) -> Result<Self, MixedError> {
        let topology = Arc::clone(input.pencil().topology());
        Self::construct(
            topology,
            *input.pencil().global_shape(),
            input.extra_shape().clone(),
            Ok(Arc::clone(input.pencil())),
            transforms,
            layout,
            FourierDirections::default(),
        )
    }

    /// Builds a mixed complex plan with explicit Fourier signs.
    pub fn from_shape_with_fft_directions(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
        transforms: [AxisTransform; N],
        directions: FourierDirections<N>,
    ) -> Result<Self, MixedError> {
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
            transforms,
            DistributedLayout::default(),
            directions,
        )
    }

    /// Rebuilds this plan with fresh array/workspace identities and Fourier signs.
    ///
    /// Forward is unscaled with these signs; inverse uses opposite signs and
    /// normalization, and backward uses opposite signs without normalization.
    /// Non-FFT axes must use `Forward`; other signs are rejected collectively.
    pub fn with_fft_directions(
        &self,
        directions: FourierDirections<N>,
    ) -> Result<Self, MixedError> {
        let mut plan = Self::construct_with_backend(
            Arc::clone(self.input_pencil().topology()),
            *self.input_pencil().global_shape(),
            self.core.extra_shape.clone(),
            Pencil::new(
                Arc::clone(self.input_pencil().topology()),
                *self.input_pencil().global_shape(),
                std::array::from_fn(|axis| axis),
            )
            .map_err(FftError::Pencil),
            self.core.transforms,
            self.core.layout,
            directions,
            self.core.backend,
        )
        .map_err(|error| match error {
            BackendInitError::Local(error) => error,
            #[cfg(feature = "fftw")]
            BackendInitError::Native(_) | BackendInitError::PeerPreflight => {
                MixedError::Fft(FftError::PreparationFailed)
            }
            #[cfg(not(feature = "fftw"))]
            BackendInitError::PeerPreflight => MixedError::Fft(FftError::PreparationFailed),
        })?;
        Arc::get_mut(&mut plan.core)
            .expect("fresh core")
            .strict_array_identity = true;
        Ok(plan)
    }

    /// Builds an Alltoallv mixed complex plan from a shape.
    pub fn from_shape(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
        transforms: [AxisTransform; N],
    ) -> Result<Self, MixedError> {
        Self::from_shape_with_layout(
            topology,
            global_shape,
            extra_shape,
            transforms,
            DistributedLayout::default(),
        )
    }

    /// Builds a mixed complex plan from a shape and transition method.
    pub fn from_shape_with_method(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
        transforms: [AxisTransform; N],
        method: TransposeMethod,
    ) -> Result<Self, MixedError> {
        Self::from_shape_with_layout(
            topology,
            global_shape,
            extra_shape,
            transforms,
            DistributedLayout {
                transpose_method: method,
                ..DistributedLayout::default()
            },
        )
    }

    /// Builds a mixed complex plan from a shape and explicit layout.
    pub fn from_shape_with_layout(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
        transforms: [AxisTransform; N],
        layout: DistributedLayout,
    ) -> Result<Self, MixedError> {
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
            transforms,
            layout,
            FourierDirections::default(),
        )
    }

    /// Returns the concrete transform assigned to each logical axis.
    pub fn transforms(&self) -> [AxisTransform; N] {
        self.core.transforms
    }

    /// Alias for [`Self::transforms`].
    pub fn axis_transforms(&self) -> [AxisTransform; N] {
        self.transforms()
    }

    /// Returns the canonical complex input pencil.
    pub fn input_pencil(&self) -> &Arc<Pencil<N, M>> {
        &self.core.stages[0].input
    }

    /// Returns the output pencil selected by the layout policy.
    pub fn output_pencil(&self) -> &Arc<Pencil<N, M>> {
        &self.core.stages[self.core.stages.len() - 1].output
    }

    /// Returns the exact extra shape required by this plan.
    pub fn extra_shape(&self) -> &ExtraShape {
        &self.core.extra_shape
    }

    /// Returns the transport and memory-layout policy used by this plan.
    pub fn layout(&self) -> DistributedLayout {
        self.core.layout
    }

    /// Returns the configured Fourier signs.
    pub fn fft_directions(&self) -> FourierDirections<N> {
        self.core.directions
    }

    /// Returns the selected local backend.
    pub fn backend_kind(&self) -> crate::BackendKind {
        self.core.backend.kind()
    }

    pub(super) fn collection_descriptor(&self) -> &[u64] {
        &self.core.descriptor
    }

    #[cfg(feature = "fftw")]
    /// Returns native planning options, or `None` for RustFFT.
    pub fn options(&self) -> Option<PlanOptions> {
        match self.core.backend {
            BackendChoice::Fftw(options) => Some(options),
            BackendChoice::RustFft => None,
        }
    }

    pub(super) fn collection_preflight_forward(
        &self,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &PencilArray<Complex<R>, N, M>,
        workspace: &MixedC2cWorkspace<R, N, M>,
    ) -> Result<(), MixedError> {
        validate_mixed_c2c_oop(
            &self.core,
            Direction::Forward,
            source,
            destination,
            workspace,
        )
    }

    pub(super) fn collection_preflight_inverse(
        &self,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &PencilArray<Complex<R>, N, M>,
        workspace: &MixedC2cWorkspace<R, N, M>,
    ) -> Result<(), MixedError> {
        validate_mixed_c2c_oop(
            &self.core,
            Direction::Inverse,
            source,
            destination,
            workspace,
        )
    }

    pub(super) fn collection_preflight_in_place(
        &self,
        direction: super::Direction,
        array: &MixedC2cInPlaceArray<R, N, M>,
        workspace: &MixedC2cInPlaceWorkspace<R, N, M>,
    ) -> Result<(), MixedError> {
        validate_mixed_c2c_ip(&self.core, direction, array, workspace)
    }

    /// Allocates a zero-initialized complex input array.
    pub fn allocate_input(&self) -> Result<PencilArray<Complex<R>, N, M>, MixedError> {
        PencilArray::from_fn(
            Arc::clone(self.input_pencil()),
            self.core.extra_shape.clone(),
            zero_complex::<R>,
        )
        .map_err(FftError::Array)
        .map_err(Into::into)
    }

    /// Allocates a zero-initialized complex output array.
    pub fn allocate_output(&self) -> Result<PencilArray<Complex<R>, N, M>, MixedError> {
        PencilArray::from_fn(
            Arc::clone(self.output_pencil()),
            self.core.extra_shape.clone(),
            zero_complex::<R>,
        )
        .map_err(FftError::Array)
        .map_err(Into::into)
    }

    /// Allocates reusable out-of-place workspace.
    pub fn allocate_workspace(&self) -> Result<MixedC2cWorkspace<R, N, M>, MixedError> {
        let intermediate = ManyPencilArray::from_elem(
            self.core.registered_pencils.clone(),
            0,
            self.core.extra_shape.clone(),
            zero_complex::<R>(),
        )
        .map_err(map_array_allocation)?;
        Ok(MixedC2cWorkspace {
            core: Arc::clone(&self.core),
            intermediate,
            transpose: TransposeWorkspace::from_vecs(
                initialized_vec(self.core.transpose_send_len, zero_complex::<R>())?,
                initialized_vec(self.core.transpose_receive_len, zero_complex::<R>())?,
            ),
            embedding_line: initialized_vec(self.core.embedding_len, zero_complex::<R>())?,
            fft_scratch: initialized_vec(self.core.fft_scratch_len, zero_complex::<R>())?,
            line_buffer: initialized_vec(self.core.strided_line_len, zero_complex::<R>())?,
        })
    }

    /// Compatibility spelling used by the homogeneous C2C plan.
    pub fn allocate_out_of_place_workspace(
        &self,
    ) -> Result<MixedC2cWorkspace<R, N, M>, MixedError> {
        self.allocate_workspace()
    }

    /// Allocates the single complex buffer used by in-place execution.
    pub fn allocate_in_place(&self) -> Result<MixedC2cInPlaceArray<R, N, M>, MixedError> {
        let array = ManyPencilArray::from_elem(
            self.core.registered_pencils.clone(),
            0,
            self.core.extra_shape.clone(),
            zero_complex::<R>(),
        )
        .map_err(map_array_allocation)?;
        Ok(MixedC2cInPlaceArray {
            core: Arc::clone(&self.core),
            array,
            state: C2cState::Input,
            #[cfg(test)]
            test_hook: None,
        })
    }

    /// Allocates reusable in-place workspace.
    pub fn allocate_in_place_workspace(
        &self,
    ) -> Result<MixedC2cInPlaceWorkspace<R, N, M>, MixedError> {
        Ok(MixedC2cInPlaceWorkspace {
            core: Arc::clone(&self.core),
            transpose: TransposeWorkspace::from_vecs(
                initialized_vec(self.core.transpose_send_len, zero_complex::<R>())?,
                initialized_vec(self.core.transpose_receive_len, zero_complex::<R>())?,
            ),
            embedding_line: initialized_vec(self.core.embedding_len, zero_complex::<R>())?,
            fft_scratch: initialized_vec(self.core.fft_scratch_len, zero_complex::<R>())?,
            line_buffer: initialized_vec(self.core.strided_line_len, zero_complex::<R>())?,
        })
    }

    /// Computes the unnormalized mixed forward transform.
    pub fn forward(
        &self,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &mut PencilArray<Complex<R>, N, M>,
        workspace: &mut MixedC2cWorkspace<R, N, M>,
    ) -> Result<(), MixedError> {
        self.execute(Direction::Forward, source, destination, workspace, None)
    }

    /// Computes the per-axis normalized mixed inverse transform.
    pub fn inverse(
        &self,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &mut PencilArray<Complex<R>, N, M>,
        workspace: &mut MixedC2cWorkspace<R, N, M>,
    ) -> Result<(), MixedError> {
        self.execute(Direction::Inverse, source, destination, workspace, None)
    }

    /// Computes the raw paired mixed backward transform.
    pub fn backward(
        &self,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &mut PencilArray<Complex<R>, N, M>,
        workspace: &mut MixedC2cWorkspace<R, N, M>,
    ) -> Result<(), MixedError> {
        self.execute(Direction::Backward, source, destination, workspace, None)
    }

    /// Computes forward with the next local kernel after receive/unpack completion and before P2P send waits.
    pub fn forward_with_overlap(
        &self,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &mut PencilArray<Complex<R>, N, M>,
        workspace: &mut MixedC2cWorkspace<R, N, M>,
    ) -> Result<(), FftOverlapError<MixedError>> {
        self.execute_overlap(Direction::Forward, source, destination, workspace)
    }

    /// Computes inverse with the next local kernel after receive/unpack completion and before P2P send waits.
    pub fn inverse_with_overlap(
        &self,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &mut PencilArray<Complex<R>, N, M>,
        workspace: &mut MixedC2cWorkspace<R, N, M>,
    ) -> Result<(), FftOverlapError<MixedError>> {
        self.execute_overlap(Direction::Inverse, source, destination, workspace)
    }

    /// Computes backward with the next local kernel after receive/unpack completion and before P2P send waits.
    pub fn backward_with_overlap(
        &self,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &mut PencilArray<Complex<R>, N, M>,
        workspace: &mut MixedC2cWorkspace<R, N, M>,
    ) -> Result<(), FftOverlapError<MixedError>> {
        self.execute_overlap(Direction::Backward, source, destination, workspace)
    }

    /// Computes the mixed forward transform in place.
    pub fn forward_in_place(
        &self,
        array: &mut MixedC2cInPlaceArray<R, N, M>,
        workspace: &mut MixedC2cInPlaceWorkspace<R, N, M>,
    ) -> Result<(), MixedError> {
        self.execute_in_place(Direction::Forward, array, workspace, None)
    }

    /// Computes the per-axis normalized mixed inverse in place.
    pub fn inverse_in_place(
        &self,
        array: &mut MixedC2cInPlaceArray<R, N, M>,
        workspace: &mut MixedC2cInPlaceWorkspace<R, N, M>,
    ) -> Result<(), MixedError> {
        self.execute_in_place(Direction::Inverse, array, workspace, None)
    }

    /// Computes the raw paired mixed backward transform in place.
    pub fn backward_in_place(
        &self,
        array: &mut MixedC2cInPlaceArray<R, N, M>,
        workspace: &mut MixedC2cInPlaceWorkspace<R, N, M>,
    ) -> Result<(), MixedError> {
        self.execute_in_place(Direction::Backward, array, workspace, None)
    }

    /// Runs [`Self::forward`] and returns measured per-stage timing.
    /// Computes the mixed forward transform and records per-stage timing.
    pub fn forward_with_timing(
        &self,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &mut PencilArray<Complex<R>, N, M>,
        workspace: &mut MixedC2cWorkspace<R, N, M>,
    ) -> Result<TransformTiming<N>, MixedError> {
        let mut t = TransformTiming::default();
        self.execute(
            Direction::Forward,
            source,
            destination,
            workspace,
            Some(&mut t),
        )?;
        Ok(t)
    }
    /// Runs [`Self::inverse`] and returns measured per-stage timing.
    /// Computes the mixed inverse transform and records per-stage timing.
    pub fn inverse_with_timing(
        &self,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &mut PencilArray<Complex<R>, N, M>,
        workspace: &mut MixedC2cWorkspace<R, N, M>,
    ) -> Result<TransformTiming<N>, MixedError> {
        let mut t = TransformTiming::default();
        self.execute(
            Direction::Inverse,
            source,
            destination,
            workspace,
            Some(&mut t),
        )?;
        Ok(t)
    }
    /// Runs [`Self::backward`] and returns measured per-stage timing.
    /// Computes the mixed backward transform and records per-stage timing.
    pub fn backward_with_timing(
        &self,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &mut PencilArray<Complex<R>, N, M>,
        workspace: &mut MixedC2cWorkspace<R, N, M>,
    ) -> Result<TransformTiming<N>, MixedError> {
        let mut t = TransformTiming::default();
        self.execute(
            Direction::Backward,
            source,
            destination,
            workspace,
            Some(&mut t),
        )?;
        Ok(t)
    }
    /// Runs [`Self::forward_in_place`] and returns measured per-stage timing.
    /// Computes the in-place mixed forward transform and records timing.
    pub fn forward_in_place_with_timing(
        &self,
        array: &mut MixedC2cInPlaceArray<R, N, M>,
        workspace: &mut MixedC2cInPlaceWorkspace<R, N, M>,
    ) -> Result<TransformTiming<N>, MixedError> {
        let mut t = TransformTiming::default();
        self.execute_in_place(Direction::Forward, array, workspace, Some(&mut t))?;
        Ok(t)
    }
    /// Runs [`Self::inverse_in_place`] and returns measured per-stage timing.
    /// Computes the in-place mixed inverse transform and records timing.
    pub fn inverse_in_place_with_timing(
        &self,
        array: &mut MixedC2cInPlaceArray<R, N, M>,
        workspace: &mut MixedC2cInPlaceWorkspace<R, N, M>,
    ) -> Result<TransformTiming<N>, MixedError> {
        let mut t = TransformTiming::default();
        self.execute_in_place(Direction::Inverse, array, workspace, Some(&mut t))?;
        Ok(t)
    }
    /// Runs [`Self::backward_in_place`] and returns measured per-stage timing.
    /// Computes the in-place mixed backward transform and records timing.
    pub fn backward_in_place_with_timing(
        &self,
        array: &mut MixedC2cInPlaceArray<R, N, M>,
        workspace: &mut MixedC2cInPlaceWorkspace<R, N, M>,
    ) -> Result<TransformTiming<N>, MixedError> {
        let mut t = TransformTiming::default();
        self.execute_in_place(Direction::Backward, array, workspace, Some(&mut t))?;
        Ok(t)
    }

    fn construct(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
        input: Result<Arc<Pencil<N, M>>, FftError>,
        transforms: [AxisTransform; N],
        layout: DistributedLayout,
        directions: FourierDirections<N>,
    ) -> Result<Self, MixedError> {
        Self::construct_with_backend(
            topology,
            global_shape,
            extra_shape,
            input,
            transforms,
            layout,
            directions,
            BackendChoice::RustFft,
        )
        .map_err(|error| match error {
            BackendInitError::Local(error) => error,
            _ => MixedError::Fft(FftError::PreparationFailed),
        })
    }

    #[cfg(feature = "fftw")]
    #[allow(private_bounds)]
    /// Builds a mixed plan from a shape using FFTW.
    pub fn from_shape_with_fftw(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
        transforms: [AxisTransform; N],
        options: PlanOptions,
    ) -> Result<Self, BackendInitError<MixedError>>
    where
        R: crate::backend::FftwReal,
    {
        let input = Pencil::new(
            Arc::clone(&topology),
            global_shape,
            std::array::from_fn(|axis| axis),
        )
        .map_err(FftError::Pencil);
        Self::construct_with_backend(
            topology,
            global_shape,
            extra_shape,
            input,
            transforms,
            DistributedLayout::default(),
            FourierDirections::default(),
            BackendChoice::Fftw(options),
        )
    }

    #[cfg(feature = "fftw")]
    #[allow(private_bounds)]
    /// Rebuilds this mixed plan using FFTW.
    pub fn with_fftw(&self, options: PlanOptions) -> Result<Self, BackendInitError<MixedError>>
    where
        R: crate::backend::FftwReal,
    {
        let topology = Arc::clone(self.input_pencil().topology());
        let shape = *self.input_pencil().global_shape();
        let input = Pencil::new(
            Arc::clone(&topology),
            shape,
            std::array::from_fn(|axis| axis),
        )
        .map_err(FftError::Pencil);
        let mut plan = Self::construct_with_backend(
            topology,
            shape,
            self.core.extra_shape.clone(),
            input,
            self.core.transforms,
            self.core.layout,
            self.core.directions,
            BackendChoice::Fftw(options),
        )?;
        Arc::get_mut(&mut plan.core)
            .expect("fresh core")
            .strict_array_identity = true;
        Ok(plan)
    }

    fn construct_with_backend(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
        input: Result<Arc<Pencil<N, M>>, FftError>,
        transforms: [AxisTransform; N],
        layout: DistributedLayout,
        directions: FourierDirections<N>,
        backend: BackendChoice,
    ) -> Result<Self, BackendInitError<MixedError>> {
        let communicator = topology.communicator();
        let expected_len =
            mixed_descriptor_len::<N, M>(&extra_shape).and_then(|length| length.checked_add(N + 7));
        let descriptor = expected_len.and_then(|_| {
            build_mixed_descriptor::<R, N, M>(
                &topology,
                global_shape,
                global_shape,
                &extra_shape,
                transforms,
                1,
                usize::MAX,
                0,
                layout,
            )
            .ok()
            .and_then(|mut descriptor| {
                descriptor.try_reserve_exact(N + 7).ok()?;
                descriptor.extend(backend.descriptor_words::<R>());
                descriptor.extend(directions.0.iter().map(|direction| match direction {
                    FourierDirection::Forward => 0,
                    FourierDirection::Backward => 1,
                }));
                Some(descriptor)
            })
        });
        let operation = match backend {
            BackendChoice::RustFft => super::OPERATION_MIXED_C2C_PLAN,
            #[cfg(feature = "fftw")]
            BackendChoice::Fftw(_) => OPERATION_MIXED_C2C_PLAN_NATIVE,
        };
        let header = mixed_header::<N, M>(operation, N, M, expected_len);
        if !agree_header(communicator, header) {
            return Err(BackendInitError::Local(MixedError::Fft(
                FftError::CollectiveDescriptorMismatch,
            )));
        }
        let descriptor = collective_descriptor(communicator, descriptor, expected_len)?;
        agree_result(communicator, validate_c2c_graph(transforms))?;
        agree_result(
            communicator,
            validate_r2c_directions(transforms, directions),
        )?;
        let input = agree_result(communicator, validate_input(input, &topology, global_shape))?;
        let route = agree_result(
            communicator,
            build_route(Ok(input), &topology, global_shape, layout.permute_dims),
        )?;
        let prepared =
            prepare_mixed_c2c_stages(&route, global_shape, transforms, directions, backend);
        if !collective_valid(communicator, prepared.is_ok()) {
            return Err(match prepared {
                Err(error) => error,
                Ok(_) => BackendInitError::PeerPreflight,
            });
        }
        let (stages, embedding_len, fft_scratch_len, strided_line_len) =
            prepared.expect("collective backend preflight accepted");
        let registered_pencils = agree_result(communicator, mixed_c2c_stage_pencils(&stages))?;
        let layout_stages = agree_result(communicator, mixed_layout_stages(&stages))?;
        let stage_prep = StagePreparation {
            stages: layout_stages,
            fft_scratch_len: 0,
        };
        let (transitions, transpose_send_len, transpose_receive_len, _real_send, _real_receive) =
            agree_result(
                communicator,
                build_transitions::<R, N, M>(
                    communicator,
                    &stage_prep,
                    &route.distributed,
                    &extra_shape,
                    layout.transpose_method,
                    0,
                ),
            )?;
        Ok(Self {
            core: Arc::new(MixedC2cCore {
                stages,
                transitions: transitions.into_boxed_slice(),
                registered_pencils,
                extra_shape,
                transforms,
                layout,
                descriptor: descriptor.into_boxed_slice(),
                embedding_len,
                fft_scratch_len,
                strided_line_len,
                transpose_send_len,
                transpose_receive_len,
                directions,
                backend,
                strict_array_identity: backend.kind() == crate::BackendKind::Fftw,
            }),
        })
    }

    fn execute(
        &self,
        direction: Direction,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &mut PencilArray<Complex<R>, N, M>,
        workspace: &mut MixedC2cWorkspace<R, N, M>,
        mut report: Option<&mut TransformTiming<N>>,
    ) -> Result<(), MixedError> {
        let started = Instant::now();
        let communicator = self.input_pencil().topology().communicator();
        let operation = if report.is_some() {
            match direction {
                Direction::Forward => OPERATION_MIXED_FORWARD_TIMED,
                Direction::Inverse => OPERATION_MIXED_INVERSE_TIMED,
                Direction::Backward => OPERATION_MIXED_BACKWARD_TIMED,
            }
        } else {
            mixed_c2c_operation(direction, false)
        };
        agree_execution_descriptor_ref::<N, M>(communicator, operation, &self.core.descriptor)?;
        let preflight =
            validate_mixed_c2c_oop(&self.core, direction, source, destination, workspace);
        if !collective_valid(communicator, preflight.is_ok()) {
            return Err(preflight
                .err()
                .unwrap_or(MixedError::Fft(FftError::CollectivePreconditionFailed)));
        }
        preflight.expect("mixed C2C out-of-place preflight succeeded");
        let result = match direction {
            Direction::Forward => execute_mixed_c2c_forward(
                &self.core,
                source,
                destination,
                workspace,
                report.as_deref_mut(),
            ),
            Direction::Inverse | Direction::Backward => execute_mixed_c2c_reverse(
                &self.core,
                source,
                destination,
                workspace,
                matches!(direction, Direction::Inverse),
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
        workspace: &mut MixedC2cWorkspace<R, N, M>,
    ) -> Result<(), FftOverlapError<MixedError>> {
        let communicator = self.input_pencil().topology().communicator();
        let operation = match direction {
            Direction::Forward => OPERATION_MIXED_C2C_FORWARD_OVERLAP,
            Direction::Inverse => OPERATION_MIXED_C2C_INVERSE_OVERLAP,
            Direction::Backward => OPERATION_MIXED_C2C_BACKWARD_OVERLAP,
        };
        agree_execution_descriptor_ref::<N, M>(communicator, operation, &self.core.descriptor)?;
        if self.core.layout.transpose_method != TransposeMethod::PointToPoint {
            return Err(FftOverlapError::UnsupportedTransport);
        }
        let preflight =
            validate_mixed_c2c_oop(&self.core, direction, source, destination, workspace);
        if !collective_valid(communicator, preflight.is_ok()) {
            return Err(preflight
                .err()
                .unwrap_or(MixedError::Fft(FftError::CollectivePreconditionFailed))
                .into());
        }
        preflight?;
        if !collective_valid(
            communicator,
            self.core.transitions.iter().all(|t| {
                matches!(
                    if matches!(direction, Direction::Forward) {
                        &t.forward
                    } else {
                        &t.backward
                    },
                    super::C2cTransition::Identity
                        | super::C2cTransition::Local(_)
                        | super::C2cTransition::PointToPoint(_)
                )
            }),
        ) {
            return Err(FftOverlapError::UnsupportedTransport);
        }
        execute_mixed_c2c_overlap(&self.core, direction, source, destination, workspace)
    }

    fn execute_in_place(
        &self,
        direction: Direction,
        array: &mut MixedC2cInPlaceArray<R, N, M>,
        workspace: &mut MixedC2cInPlaceWorkspace<R, N, M>,
        mut report: Option<&mut TransformTiming<N>>,
    ) -> Result<(), MixedError> {
        let started = Instant::now();
        let communicator = self.input_pencil().topology().communicator();
        agree_execution_descriptor_ref::<N, M>(
            communicator,
            if report.is_some() {
                match direction {
                    Direction::Forward => OPERATION_MIXED_FORWARD_IN_PLACE_TIMED,
                    Direction::Inverse => OPERATION_MIXED_INVERSE_IN_PLACE_TIMED,
                    Direction::Backward => OPERATION_MIXED_BACKWARD_IN_PLACE_TIMED,
                }
            } else {
                mixed_c2c_operation(direction, true)
            },
            &self.core.descriptor,
        )?;
        let preflight = validate_mixed_c2c_ip(&self.core, direction, array, workspace);
        if !collective_valid(communicator, preflight.is_ok()) {
            return Err(preflight
                .err()
                .unwrap_or(MixedError::Fft(FftError::CollectivePreconditionFailed)));
        }
        preflight.expect("mixed C2C in-place preflight succeeded");
        array.state = C2cState::Poisoned;
        #[cfg(test)]
        if array.test_hook == Some(MixedInPlaceTestHook::Start) {
            panic!("injected mixed C2C in-place panic after start");
        }
        let result = match direction {
            Direction::Forward => execute_mixed_c2c_forward_ip(
                &self.core,
                &mut array.array,
                &mut workspace.transpose,
                &mut workspace.embedding_line,
                &mut workspace.fft_scratch,
                &mut workspace.line_buffer,
                report.as_deref_mut(),
            ),
            Direction::Inverse | Direction::Backward => execute_mixed_c2c_reverse_ip(
                &self.core,
                &mut array.array,
                &mut workspace.transpose,
                &mut workspace.embedding_line,
                &mut workspace.fft_scratch,
                &mut workspace.line_buffer,
                matches!(direction, Direction::Inverse),
                report.as_deref_mut(),
            ),
        };
        if result.is_ok() {
            array.state = match direction {
                Direction::Forward => C2cState::Output,
                Direction::Inverse | Direction::Backward => C2cState::Input,
            };
        }
        if let Some(timing) = report {
            timing.total = started.elapsed();
        }
        result
    }
}

impl<R: FftReal, const N: usize, const M: usize> MixedC2cInPlaceArray<R, N, M> {
    /// Returns the completion state.
    pub fn state(&self) -> C2cState {
        self.state
    }

    /// Borrows the active complex view.
    pub fn view(&self) -> Result<PencilArrayView<'_, Complex<R>, N, M>, MixedError> {
        match self.state {
            C2cState::Poisoned => Err(MixedError::Fft(FftError::Array(ArrayError::Poisoned))),
            C2cState::Input | C2cState::Output => self
                .array
                .active_view()
                .map_err(FftError::Array)
                .map_err(Into::into),
        }
    }

    /// Borrows the active mutable complex view.
    pub fn view_mut(&mut self) -> Result<PencilArrayViewMut<'_, Complex<R>, N, M>, MixedError> {
        match self.state {
            C2cState::Poisoned => Err(MixedError::Fft(FftError::Array(ArrayError::Poisoned))),
            C2cState::Input | C2cState::Output => self
                .array
                .active_view_mut()
                .map_err(FftError::Array)
                .map_err(Into::into),
        }
    }
}

#[cfg(test)]
pub(super) fn empty_c2c_workspace_for_test<R: FftReal, const N: usize, const M: usize>(
    workspace: &mut MixedC2cWorkspace<R, N, M>,
) {
    workspace.embedding_line.clear();
}

#[cfg(test)]
pub(super) fn empty_c2c_in_place_workspace_for_test<R: FftReal, const N: usize, const M: usize>(
    workspace: &mut MixedC2cInPlaceWorkspace<R, N, M>,
) {
    workspace.embedding_line.clear();
}

#[cfg(test)]
pub(super) fn empty_r2c_workspace_for_test<R: FftReal, const N: usize, const M: usize>(
    workspace: &mut MixedR2cWorkspace<R, N, M>,
) {
    workspace.real_source_line.clear();
}

#[cfg(test)]
pub(super) fn empty_r2c_in_place_workspace_for_test<R: FftReal, const N: usize, const M: usize>(
    workspace: &mut MixedR2cInPlaceWorkspace<R, N, M>,
) {
    workspace.real_source_line.clear();
}

#[cfg(test)]
pub(super) fn force_c2c_workspace_allocation_failure_for_test<
    R: FftReal,
    const N: usize,
    const M: usize,
>(
    plan: &mut MixedC2cPlan<R, N, M>,
) {
    Arc::get_mut(&mut plan.core)
        .expect("test plan core has no other owners")
        .embedding_len = usize::MAX;
}

#[cfg(test)]
pub(super) fn force_r2c_workspace_allocation_failure_for_test<
    R: FftReal,
    const N: usize,
    const M: usize,
>(
    plan: &mut MixedR2cPlan<R, N, M>,
) {
    Arc::get_mut(&mut plan.core)
        .expect("test plan core has no other owners")
        .embedding_len = usize::MAX;
}

#[cfg(test)]
pub(super) fn panic_after_start_for_c2c_test<R: FftReal, const N: usize, const M: usize>(
    array: &mut MixedC2cInPlaceArray<R, N, M>,
) {
    array.test_hook = Some(MixedInPlaceTestHook::Start);
}

#[cfg(test)]
pub(super) fn panic_after_start_for_r2c_test<R: FftReal, const N: usize, const M: usize>(
    array: &mut MixedR2cInPlaceArray<R, N, M>,
) {
    array.test_hook = Some(MixedInPlaceTestHook::Start);
}

#[cfg(test)]
pub(super) fn panic_after_forward_detach_for_r2c_test<
    R: FftReal,
    const N: usize,
    const M: usize,
>(
    array: &mut MixedR2cInPlaceArray<R, N, M>,
) {
    array.test_hook = Some(MixedInPlaceTestHook::ForwardDetach);
}

#[cfg(test)]
pub(super) fn panic_after_reverse_detach_for_r2c_test<
    R: FftReal,
    const N: usize,
    const M: usize,
>(
    array: &mut MixedR2cInPlaceArray<R, N, M>,
) {
    array.test_hook = Some(MixedInPlaceTestHook::ReverseDetach);
}

fn mixed_header<const N: usize, const M: usize>(
    operation: u64,
    _n: usize,
    _m: usize,
    descriptor_len: Option<usize>,
) -> [u64; super::HEADER_WORDS] {
    [
        super::DESCRIPTOR_SCHEMA,
        operation,
        u64::try_from(N).unwrap_or(INVALID_WORD),
        u64::try_from(M).unwrap_or(INVALID_WORD),
        descriptor_len
            .and_then(|length| u64::try_from(length).ok())
            .unwrap_or(INVALID_WORD),
    ]
}

fn mixed_c2c_operation(direction: Direction, in_place: bool) -> u64 {
    match (direction, in_place) {
        (Direction::Forward, false) => OPERATION_MIXED_C2C_FORWARD,
        (Direction::Inverse, false) => OPERATION_MIXED_C2C_INVERSE,
        (Direction::Backward, false) => OPERATION_MIXED_C2C_BACKWARD,
        (Direction::Forward, true) => OPERATION_MIXED_C2C_FORWARD_IN_PLACE,
        (Direction::Inverse, true) => OPERATION_MIXED_C2C_INVERSE_IN_PLACE,
        (Direction::Backward, true) => OPERATION_MIXED_C2C_BACKWARD_IN_PLACE,
    }
}

fn mixed_r2c_operation(direction: Direction, in_place: bool) -> u64 {
    match (direction, in_place) {
        (Direction::Forward, false) => OPERATION_MIXED_R2C_FORWARD,
        (Direction::Inverse, false) => OPERATION_MIXED_R2C_INVERSE,
        (Direction::Backward, false) => OPERATION_MIXED_R2C_BACKWARD,
        (Direction::Forward, true) => OPERATION_MIXED_R2C_FORWARD_IN_PLACE,
        (Direction::Inverse, true) => OPERATION_MIXED_R2C_INVERSE_IN_PLACE,
        (Direction::Backward, true) => OPERATION_MIXED_R2C_BACKWARD_IN_PLACE,
    }
}

fn mixed_descriptor_len<const N: usize, const M: usize>(extra: &ExtraShape) -> Option<usize> {
    N.checked_mul(4)?
        .checked_add(M)?
        .checked_add(extra.dimensions().len())?
        .checked_add(9)
}

fn build_mixed_descriptor<R: FftReal, const N: usize, const M: usize>(
    topology: &MpiTopology<M>,
    input_shape: [usize; N],
    output_shape: [usize; N],
    extra: &ExtraShape,
    transforms: [AxisTransform; N],
    family: u64,
    reduction_axis: usize,
    original_len: usize,
    layout: DistributedLayout,
) -> Result<Vec<u64>, ()> {
    let length = mixed_descriptor_len::<N, M>(extra).ok_or(())?;
    let mut descriptor = Vec::new();
    descriptor.try_reserve_exact(length).map_err(|_| ())?;
    super::append_usizes(&mut descriptor, &input_shape)?;
    super::append_usizes(&mut descriptor, &output_shape)?;
    super::append_usizes(&mut descriptor, topology.process_grid())?;
    super::append_shape(&mut descriptor, extra)?;
    descriptor.push(family);
    descriptor.push(if family == 1 {
        VALUE_KIND_C2C
    } else {
        VALUE_KIND_R2C
    });
    descriptor.push(u64::try_from(size_of::<R>()).map_err(|_| ())?);
    descriptor.push(u64::try_from(size_of::<Complex<R>>()).map_err(|_| ())?);
    descriptor.push(u64::try_from(reduction_axis).unwrap_or(INVALID_WORD));
    descriptor.push(u64::try_from(original_len).unwrap_or(INVALID_WORD));
    descriptor.push(layout.transpose_method.descriptor_word());
    descriptor.push(u64::from(layout.permute_dims));
    for transform in transforms {
        let (tag, kind) = axis_transform_words(transform);
        descriptor.push(tag);
        descriptor.push(kind);
    }
    if descriptor.len() != length {
        return Err(());
    }
    Ok(descriptor)
}

fn axis_transform_words(transform: AxisTransform) -> (u64, u64) {
    match transform {
        AxisTransform::None => (0, 0),
        AxisTransform::Fft => (1, 0),
        AxisTransform::Rfft => (2, 0),
        AxisTransform::R2r(kind) => (3, kind.descriptor_code()),
    }
}

fn validate_c2c_graph<const N: usize>(transforms: [AxisTransform; N]) -> Result<(), MixedError> {
    if transforms
        .iter()
        .any(|transform| matches!(transform, AxisTransform::Rfft))
    {
        return Err(MixedError::InvalidGraph);
    }
    Ok(())
}

fn validate_r2c_graph<const N: usize>(
    global_shape: [usize; N],
    transforms: [AxisTransform; N],
) -> Result<usize, MixedError> {
    let mut boundary = None;
    for (axis, transform) in transforms.into_iter().enumerate() {
        if matches!(transform, AxisTransform::Rfft) && boundary.replace(axis).is_some() {
            return Err(MixedError::InvalidGraph);
        }
    }
    let boundary = boundary.ok_or(MixedError::InvalidGraph)?;
    for (axis, transform) in transforms.into_iter().enumerate() {
        let valid = if axis == boundary {
            matches!(transform, AxisTransform::Rfft)
        } else if axis > boundary {
            matches!(transform, AxisTransform::None | AxisTransform::R2r(_))
        } else {
            matches!(
                transform,
                AxisTransform::None | AxisTransform::Fft | AxisTransform::R2r(_)
            )
        };
        if !valid || global_shape[axis] == 0 {
            return Err(MixedError::InvalidGraph);
        }
    }
    Ok(boundary)
}

fn validate_r2c_directions<const N: usize>(
    transforms: [AxisTransform; N],
    directions: FourierDirections<N>,
) -> Result<(), MixedError> {
    if transforms.into_iter().enumerate().any(|(axis, transform)| {
        matches!(directions.get(axis), Some(FourierDirection::Backward))
            && !matches!(transform, AxisTransform::Fft)
    }) {
        return Err(MixedError::Fft(FftError::PreparationFailed));
    }
    Ok(())
}

fn map_overwrite<E: Into<MixedError>>(error: OverwriteError<E>) -> MixedError {
    match error {
        OverwriteError::Array(error) => MixedError::Fft(FftError::Array(error)),
        OverwriteError::Writer(error) => error.into(),
    }
}

fn mixed_r2r_local<T: R2rScalar>(
    kind: AxisR2rKind,
    length: usize,
    backend: BackendChoice,
) -> Result<MixedR2rLocal<T>, BackendInitError<MixedError>> {
    match (kind, backend) {
        (AxisR2rKind::Fftw(kind), BackendChoice::RustFft) => Ok(MixedR2rLocal::Transform(
            LocalR2rPlan::new(length, kind)
                .map_err(MixedError::LocalR2r)
                .map_err(BackendInitError::Local)?,
        )),
        (AxisR2rKind::Dht, BackendChoice::RustFft) => Ok(MixedR2rLocal::Hartley(
            crate::LocalDhtPlan::new(length)
                .map_err(MixedError::LocalR2r)
                .map_err(BackendInitError::Local)?,
        )),
        #[cfg(feature = "fftw")]
        (AxisR2rKind::Fftw(kind), BackendChoice::Fftw(options)) => Ok(MixedR2rLocal::Transform(
            LocalR2rPlan::new_fftw(length, kind, options).map_err(|error| match error {
                BackendInitError::Local(error) => {
                    BackendInitError::Local(MixedError::LocalR2r(error))
                }
                BackendInitError::Native(error) => BackendInitError::Native(error),
                BackendInitError::PeerPreflight => BackendInitError::PeerPreflight,
            })?,
        )),
        #[cfg(feature = "fftw")]
        (AxisR2rKind::Dht, BackendChoice::Fftw(options)) => Ok(MixedR2rLocal::Hartley(
            crate::LocalDhtPlan::new_fftw(length, options).map_err(|error| match error {
                BackendInitError::Local(error) => {
                    BackendInitError::Local(MixedError::LocalR2r(error))
                }
                BackendInitError::Native(error) => BackendInitError::Native(error),
                BackendInitError::PeerPreflight => BackendInitError::PeerPreflight,
            })?,
        )),
    }
}

fn mixed_c2c_stage_pencils<R: FftReal, const N: usize, const M: usize>(
    stages: &[MixedC2cStage<R, N, M>],
) -> Result<Box<[Arc<Pencil<N, M>>]>, MixedError> {
    let mut pencils = Vec::new();
    pencils.try_reserve_exact(stages.len()).map_err(|_| {
        MixedError::Fft(FftError::AllocationFailed {
            required: stages.len(),
        })
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

fn validate_exact_pencil_list<const N: usize, const M: usize>(
    actual: &[Arc<Pencil<N, M>>],
    required: &[Arc<Pencil<N, M>>],
) -> Result<(), FftError> {
    if actual.len() != required.len()
        || actual.is_empty()
        || actual.iter().enumerate().any(|(index, pencil)| {
            actual[..index]
                .iter()
                .any(|other| other.same_layout(pencil))
        })
        || required.iter().enumerate().any(|(index, pencil)| {
            required[..index]
                .iter()
                .any(|other| other.same_layout(pencil))
        })
        || actual
            .iter()
            .any(|pencil| !required.iter().any(|other| other.same_layout(pencil)))
        || required
            .iter()
            .any(|pencil| !actual.iter().any(|other| other.same_layout(pencil)))
    {
        return Err(FftError::WorkspaceMismatch);
    }
    Ok(())
}

fn validate_registry_storage<const N: usize, const M: usize>(
    storage_len: usize,
    required: &[Arc<Pencil<N, M>>],
    extra_shape: &ExtraShape,
) -> Result<(), FftError> {
    let local_len = required
        .iter()
        .map(|pencil| pencil.local_len())
        .max()
        .ok_or(FftError::WorkspaceMismatch)?;
    let required_len = local_len
        .checked_mul(extra_shape.element_count())
        .ok_or(FftError::PreparationFailed)?;
    if storage_len != required_len {
        return Err(FftError::WorkspaceMismatch);
    }
    Ok(())
}

fn validate_exact_pencil_registry<const N: usize, const M: usize>(
    actual: &[Arc<Pencil<N, M>>],
    required: &[Arc<Pencil<N, M>>],
    active: &Pencil<N, M>,
    storage_len: usize,
    extra_shape: &ExtraShape,
) -> Result<(), FftError> {
    validate_exact_pencil_list(actual, required)?;
    validate_registry_storage(storage_len, required, extra_shape)?;
    if !actual.iter().any(|pencil| pencil.same_layout(active)) {
        return Err(FftError::WorkspaceMismatch);
    }
    Ok(())
}

fn mixed_layout_stages<R: FftReal, const N: usize, const M: usize>(
    stages: &[MixedC2cStage<R, N, M>],
) -> Result<Box<[TransformStage<R, N, M>]>, MixedError> {
    let mut result = Vec::new();
    result.try_reserve_exact(stages.len()).map_err(|_| {
        MixedError::Fft(FftError::AllocationFailed {
            required: stages.len(),
        })
    })?;
    result.extend(stages.iter().map(|stage| TransformStage {
        axis: stage.axis,
        input: Arc::clone(&stage.input),
        output: Arc::clone(&stage.output),
        local: LocalTransform::Identity,
    }));
    Ok(result.into_boxed_slice())
}

#[allow(clippy::type_complexity)]
fn prepare_mixed_c2c_stages<R: FftReal, const N: usize, const M: usize>(
    route: &RouteCandidate<N, M>,
    shape: [usize; N],
    transforms: [AxisTransform; N],
    directions: FourierDirections<N>,
    backend: BackendChoice,
) -> Result<(Box<[MixedC2cStage<R, N, M>]>, usize, usize, usize), BackendInitError<MixedError>> {
    if route.stages.len() != N {
        return Err(BackendInitError::Local(MixedError::Fft(
            FftError::PreparationFailed,
        )));
    }
    let mut stages = Vec::new();
    stages
        .try_reserve_exact(N)
        .map_err(|_| MixedError::Fft(FftError::AllocationFailed { required: N }))?;
    let mut embedding_len = 0;
    let mut scratch_len = 0;
    let mut strided_line_len = 0;
    for (index, pencil) in route.stages.iter().enumerate() {
        let axis = N - 1 - index;
        validate_stage_pencil(pencil, axis, shape[axis])?;
        let local = match transforms[axis] {
            AxisTransform::None => MixedComplexLocal::Identity,
            AxisTransform::Fft => MixedComplexLocal::Fft(match backend {
                BackendChoice::RustFft => LocalC2cPlan::new_with_sign(
                    shape[axis],
                    directions.get(axis) == Some(FourierDirection::Backward),
                )
                .map_err(MixedError::from)
                .map_err(BackendInitError::Local)?,
                #[cfg(feature = "fftw")]
                BackendChoice::Fftw(options) => LocalC2cPlan::new_fftw_with_sign(
                    shape[axis],
                    directions.get(axis) == Some(FourierDirection::Backward),
                    options,
                )
                .map_err(|error| match error {
                    BackendInitError::Local(e) => BackendInitError::Local(MixedError::LocalC2c(e)),
                    BackendInitError::Native(e) => BackendInitError::Native(e),
                    BackendInitError::PeerPreflight => BackendInitError::PeerPreflight,
                })?,
            }),
            AxisTransform::Rfft => return Err(BackendInitError::Local(MixedError::InvalidGraph)),
            AxisTransform::R2r(kind) => {
                MixedComplexLocal::R2r(mixed_r2r_local::<Complex<R>>(kind, shape[axis], backend)?)
            }
        };
        embedding_len = embedding_len.max(local.embedding_len());
        scratch_len = scratch_len.max(local.scratch_len());
        if memory_stride(pencil.as_ref(), axis)? > 1 {
            strided_line_len = strided_line_len.max(local.line_len());
        }
        stages.push(MixedC2cStage {
            axis,
            input: Arc::clone(pencil),
            output: Arc::clone(pencil),
            local,
        });
    }
    Ok((
        stages.into_boxed_slice(),
        embedding_len,
        scratch_len,
        strided_line_len,
    ))
}

fn validate_stage_pencil<const N: usize, const M: usize>(
    pencil: &Arc<Pencil<N, M>>,
    axis: usize,
    expected_len: usize,
) -> Result<(), MixedError> {
    if pencil
        .decomposition()
        .iter()
        .any(|distributed| distributed.index() == axis)
        || pencil.local_shape_logical()[axis] != expected_len
    {
        return Err(MixedError::Fft(FftError::PreparationFailed));
    }
    Ok(())
}

fn mixed_r2r_forward<T: R2rScalar, const N: usize, const M: usize>(
    plan: &MixedR2rLocal<T>,
    pencil: &Pencil<N, M>,
    axis: usize,
    source: &[T],
    destination: &mut [T],
    embedding: &mut [Complex<T::Real>],
    scratch: &mut [Complex<T::Real>],
    line: &mut [T],
) -> Result<(), MixedError> {
    let line_len = match plan {
        MixedR2rLocal::Transform(plan) => plan.line_len(),
        MixedR2rLocal::Hartley(plan) => plan.line_len(),
    };
    let stride = memory_stride(pencil, axis)?;
    if stride <= 1 {
        return match plan {
            MixedR2rLocal::Transform(plan) => plan
                .forward(source, destination, embedding, scratch)
                .map_err(MixedError::LocalR2r),
            MixedR2rLocal::Hartley(plan) => plan
                .forward(source, destination, embedding, scratch)
                .map_err(MixedError::LocalR2r),
        };
    }
    if line.len() < line_len {
        return Err(MixedError::Fft(FftError::WorkspaceTooSmall {
            kind: "real line",
            required: line_len,
            actual: line.len(),
        }));
    }
    let count = strided_line_count(source.len(), line_len, stride)?;
    if destination.len() != source.len() {
        return Err(MixedError::Fft(FftError::PreparationFailed));
    }
    let block = line_len
        .checked_mul(stride)
        .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
    for outer in 0..count {
        let base = outer
            .checked_mul(block)
            .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
        for inner in 0..stride {
            for k in 0..line_len {
                line[k] = source[base + k * stride + inner];
            }
            match plan {
                MixedR2rLocal::Transform(plan) => plan
                    .forward_in_place(&mut line[..line_len], embedding, scratch)
                    .map_err(MixedError::LocalR2r)?,
                MixedR2rLocal::Hartley(plan) => plan
                    .forward_in_place(&mut line[..line_len], embedding, scratch)
                    .map_err(MixedError::LocalR2r)?,
            }
            for k in 0..line_len {
                destination[base + k * stride + inner] = line[k];
            }
        }
    }
    Ok(())
}

fn mixed_r2r_forward_in_place<T: R2rScalar, const N: usize, const M: usize>(
    plan: &MixedR2rLocal<T>,
    pencil: &Pencil<N, M>,
    axis: usize,
    data: &mut [T],
    embedding: &mut [Complex<T::Real>],
    scratch: &mut [Complex<T::Real>],
    line: &mut [T],
) -> Result<(), MixedError> {
    let line_len = match plan {
        MixedR2rLocal::Transform(plan) => plan.line_len(),
        MixedR2rLocal::Hartley(plan) => plan.line_len(),
    };
    let stride = memory_stride(pencil, axis)?;
    if stride <= 1 {
        return match plan {
            MixedR2rLocal::Transform(plan) => plan
                .forward_in_place(data, embedding, scratch)
                .map_err(MixedError::LocalR2r),
            MixedR2rLocal::Hartley(plan) => plan
                .forward_in_place(data, embedding, scratch)
                .map_err(MixedError::LocalR2r),
        };
    }
    if line.len() < line_len {
        return Err(MixedError::Fft(FftError::WorkspaceTooSmall {
            kind: "real line",
            required: line_len,
            actual: line.len(),
        }));
    }
    let count = strided_line_count(data.len(), line_len, stride)?;
    let block = line_len
        .checked_mul(stride)
        .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
    for outer in 0..count {
        let base = outer
            .checked_mul(block)
            .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
        for inner in 0..stride {
            for k in 0..line_len {
                line[k] = data[base + k * stride + inner];
            }
            match plan {
                MixedR2rLocal::Transform(plan) => plan
                    .forward_in_place(&mut line[..line_len], embedding, scratch)
                    .map_err(MixedError::LocalR2r)?,
                MixedR2rLocal::Hartley(plan) => plan
                    .forward_in_place(&mut line[..line_len], embedding, scratch)
                    .map_err(MixedError::LocalR2r)?,
            }
            for k in 0..line_len {
                data[base + k * stride + inner] = line[k];
            }
        }
    }
    Ok(())
}

fn mixed_r2r_reverse<T: R2rScalar, const N: usize, const M: usize>(
    plan: &MixedR2rLocal<T>,
    pencil: &Pencil<N, M>,
    axis: usize,
    source: &[T],
    destination: &mut [T],
    embedding: &mut [Complex<T::Real>],
    scratch: &mut [Complex<T::Real>],
    line: &mut [T],
    normalize: bool,
) -> Result<(), MixedError> {
    let line_len = match plan {
        MixedR2rLocal::Transform(plan) => plan.line_len(),
        MixedR2rLocal::Hartley(plan) => plan.line_len(),
    };
    let stride = memory_stride(pencil, axis)?;
    if stride <= 1 {
        return match plan {
            MixedR2rLocal::Transform(plan) if normalize => plan
                .inverse(source, destination, embedding, scratch)
                .map_err(MixedError::LocalR2r),
            MixedR2rLocal::Transform(plan) => plan
                .backward(source, destination, embedding, scratch)
                .map_err(MixedError::LocalR2r),
            MixedR2rLocal::Hartley(plan) if normalize => plan
                .inverse(source, destination, embedding, scratch)
                .map_err(MixedError::LocalR2r),
            MixedR2rLocal::Hartley(plan) => plan
                .backward(source, destination, embedding, scratch)
                .map_err(MixedError::LocalR2r),
        };
    }
    if line.len() < line_len {
        return Err(MixedError::Fft(FftError::WorkspaceTooSmall {
            kind: "real line",
            required: line_len,
            actual: line.len(),
        }));
    }
    let count = strided_line_count(source.len(), line_len, stride)?;
    if destination.len() != source.len() {
        return Err(MixedError::Fft(FftError::PreparationFailed));
    }
    let block = line_len
        .checked_mul(stride)
        .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
    for outer in 0..count {
        let base = outer
            .checked_mul(block)
            .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
        for inner in 0..stride {
            for k in 0..line_len {
                line[k] = source[base + k * stride + inner];
            }
            match plan {
                MixedR2rLocal::Transform(plan) if normalize => plan
                    .inverse_in_place(&mut line[..line_len], embedding, scratch)
                    .map_err(MixedError::LocalR2r)?,
                MixedR2rLocal::Transform(plan) => plan
                    .backward_in_place(&mut line[..line_len], embedding, scratch)
                    .map_err(MixedError::LocalR2r)?,
                MixedR2rLocal::Hartley(plan) if normalize => plan
                    .inverse_in_place(&mut line[..line_len], embedding, scratch)
                    .map_err(MixedError::LocalR2r)?,
                MixedR2rLocal::Hartley(plan) => plan
                    .backward_in_place(&mut line[..line_len], embedding, scratch)
                    .map_err(MixedError::LocalR2r)?,
            }
            for k in 0..line_len {
                destination[base + k * stride + inner] = line[k];
            }
        }
    }
    Ok(())
}

fn mixed_r2r_reverse_in_place<T: R2rScalar, const N: usize, const M: usize>(
    plan: &MixedR2rLocal<T>,
    pencil: &Pencil<N, M>,
    axis: usize,
    data: &mut [T],
    embedding: &mut [Complex<T::Real>],
    scratch: &mut [Complex<T::Real>],
    line: &mut [T],
    normalize: bool,
) -> Result<(), MixedError> {
    let line_len = match plan {
        MixedR2rLocal::Transform(plan) => plan.line_len(),
        MixedR2rLocal::Hartley(plan) => plan.line_len(),
    };
    let stride = memory_stride(pencil, axis)?;
    if stride <= 1 {
        return match plan {
            MixedR2rLocal::Transform(plan) if normalize => plan
                .inverse_in_place(data, embedding, scratch)
                .map_err(MixedError::LocalR2r),
            MixedR2rLocal::Transform(plan) => plan
                .backward_in_place(data, embedding, scratch)
                .map_err(MixedError::LocalR2r),
            MixedR2rLocal::Hartley(plan) if normalize => plan
                .inverse_in_place(data, embedding, scratch)
                .map_err(MixedError::LocalR2r),
            MixedR2rLocal::Hartley(plan) => plan
                .backward_in_place(data, embedding, scratch)
                .map_err(MixedError::LocalR2r),
        };
    }
    if line.len() < line_len {
        return Err(MixedError::Fft(FftError::WorkspaceTooSmall {
            kind: "real line",
            required: line_len,
            actual: line.len(),
        }));
    }
    let count = strided_line_count(data.len(), line_len, stride)?;
    let block = line_len
        .checked_mul(stride)
        .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
    for outer in 0..count {
        let base = outer
            .checked_mul(block)
            .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
        for inner in 0..stride {
            for k in 0..line_len {
                line[k] = data[base + k * stride + inner];
            }
            match plan {
                MixedR2rLocal::Transform(plan) if normalize => plan
                    .inverse_in_place(&mut line[..line_len], embedding, scratch)
                    .map_err(MixedError::LocalR2r)?,
                MixedR2rLocal::Transform(plan) => plan
                    .backward_in_place(&mut line[..line_len], embedding, scratch)
                    .map_err(MixedError::LocalR2r)?,
                MixedR2rLocal::Hartley(plan) if normalize => plan
                    .inverse_in_place(&mut line[..line_len], embedding, scratch)
                    .map_err(MixedError::LocalR2r)?,
                MixedR2rLocal::Hartley(plan) => plan
                    .backward_in_place(&mut line[..line_len], embedding, scratch)
                    .map_err(MixedError::LocalR2r)?,
            }
            for k in 0..line_len {
                data[base + k * stride + inner] = line[k];
            }
        }
    }
    Ok(())
}

fn mixed_complex_forward<R: FftReal, const N: usize, const M: usize>(
    local: &MixedComplexLocal<R>,
    pencil: &Pencil<N, M>,
    axis: usize,
    source: &[Complex<R>],
    destination: &mut [Complex<R>],
    embedding: &mut [Complex<R>],
    scratch: &mut [Complex<R>],
    line: &mut [Complex<R>],
) -> Result<(), MixedError>
where
    Complex<R>: Equivalence,
{
    match local {
        MixedComplexLocal::Identity => {
            if source.len() != destination.len() {
                return Err(MixedError::Fft(FftError::PreparationFailed));
            }
            destination.copy_from_slice(source);
            Ok(())
        }
        MixedComplexLocal::Fft(plan) => super::execute_strided_complex_forward(
            plan,
            pencil,
            axis,
            source,
            destination,
            scratch,
            line,
        )
        .map_err(MixedError::Fft),
        MixedComplexLocal::R2r(plan) => mixed_r2r_forward(
            plan,
            pencil,
            axis,
            source,
            destination,
            embedding,
            scratch,
            line,
        ),
    }
}

fn mixed_complex_forward_in_place<R: FftReal, const N: usize, const M: usize>(
    local: &MixedComplexLocal<R>,
    pencil: &Pencil<N, M>,
    axis: usize,
    data: &mut [Complex<R>],
    embedding: &mut [Complex<R>],
    scratch: &mut [Complex<R>],
    line: &mut [Complex<R>],
) -> Result<(), MixedError>
where
    Complex<R>: Equivalence,
{
    match local {
        MixedComplexLocal::Identity => Ok(()),
        MixedComplexLocal::Fft(plan) => {
            super::execute_strided_complex_forward_in_place(plan, pencil, axis, data, scratch, line)
                .map_err(MixedError::Fft)
        }
        MixedComplexLocal::R2r(plan) => {
            mixed_r2r_forward_in_place(plan, pencil, axis, data, embedding, scratch, line)
        }
    }
}

fn mixed_complex_reverse<R: FftReal, const N: usize, const M: usize>(
    local: &MixedComplexLocal<R>,
    pencil: &Pencil<N, M>,
    axis: usize,
    source: &[Complex<R>],
    destination: &mut [Complex<R>],
    embedding: &mut [Complex<R>],
    scratch: &mut [Complex<R>],
    line: &mut [Complex<R>],
    normalize: bool,
) -> Result<(), MixedError>
where
    Complex<R>: Equivalence,
{
    match local {
        MixedComplexLocal::Identity => {
            if source.len() != destination.len() {
                return Err(MixedError::Fft(FftError::PreparationFailed));
            }
            destination.copy_from_slice(source);
            Ok(())
        }
        MixedComplexLocal::Fft(plan) => super::execute_strided_complex_reverse(
            plan,
            pencil,
            axis,
            source,
            destination,
            scratch,
            line,
            normalize,
        )
        .map_err(MixedError::Fft),
        MixedComplexLocal::R2r(plan) => mixed_r2r_reverse(
            plan,
            pencil,
            axis,
            source,
            destination,
            embedding,
            scratch,
            line,
            normalize,
        ),
    }
}

fn mixed_complex_reverse_in_place<R: FftReal, const N: usize, const M: usize>(
    local: &MixedComplexLocal<R>,
    pencil: &Pencil<N, M>,
    axis: usize,
    data: &mut [Complex<R>],
    embedding: &mut [Complex<R>],
    scratch: &mut [Complex<R>],
    line: &mut [Complex<R>],
    normalize: bool,
) -> Result<(), MixedError>
where
    Complex<R>: Equivalence,
{
    match local {
        MixedComplexLocal::Identity => Ok(()),
        MixedComplexLocal::Fft(plan) => super::execute_strided_complex_reverse_in_place(
            plan, pencil, axis, data, scratch, line, normalize,
        )
        .map_err(MixedError::Fft),
        MixedComplexLocal::R2r(plan) => mixed_r2r_reverse_in_place(
            plan, pencil, axis, data, embedding, scratch, line, normalize,
        ),
    }
}

fn execute_mixed_transition_timed<T: Equivalence + Copy + Clone, const N: usize, const M: usize>(
    communicator: &mpi::topology::CartesianCommunicator,
    transition: &super::C2cTransition<N, M>,
    intermediate: &mut ManyPencilArray<T, N, M>,
    workspace: &mut TransposeWorkspace<T>,
    report: &mut Option<&mut TransformTiming<N>>,
    index: usize,
) -> Result<(), MixedError> {
    let result = super::execute_transition_timed(
        transition,
        intermediate,
        workspace,
        report.as_deref_mut(),
        index,
    )
    .map_err(MixedError::Fft);
    agree_result(communicator, result)
}

fn mixed_c2c_forward_tail<R: FftReal, const N: usize, const M: usize>(
    stages: &[MixedC2cStage<R, N, M>],
    transitions: &[C2cStageTransition<N, M>],
    intermediate: &mut ManyPencilArray<Complex<R>, N, M>,
    transpose: &mut TransposeWorkspace<Complex<R>>,
    destination: &mut PencilArray<Complex<R>, N, M>,
    embedding: &mut [Complex<R>],
    scratch: &mut [Complex<R>],
    line: &mut [Complex<R>],
    mut report: Option<&mut TransformTiming<N>>,
) -> Result<(), MixedError>
where
    Complex<R>: Equivalence,
{
    if stages.is_empty() || transitions.len() + 1 != stages.len() {
        return Err(MixedError::Fft(FftError::PreparationFailed));
    }
    let communicator = stages[0].input.topology().communicator();
    if stages.len() == 1 {
        let active = intermediate.active_view().map_err(FftError::Array)?;
        let mut target = destination.view_mut();
        if active.as_slice().len() != target.as_slice().len() {
            return Err(MixedError::Fft(FftError::PreparationFailed));
        }
        target.as_mut_slice().copy_from_slice(active.as_slice());
        return Ok(());
    }
    let last = stages.len() - 1;
    for index in 0..transitions.len() {
        execute_mixed_transition_timed(
            communicator,
            &transitions[index].forward,
            intermediate,
            transpose,
            &mut report,
            index,
        )?;
        let stage = &stages[index + 1];
        let fft_started = Instant::now();
        let stage_result = if index + 1 == last {
            (|| {
                let active = intermediate.active_view().map_err(FftError::Array)?;
                let mut target = destination.view_mut();
                mixed_complex_forward(
                    &stage.local,
                    stage.output.as_ref(),
                    stage.axis,
                    active.as_slice(),
                    target.as_mut_slice(),
                    embedding,
                    scratch,
                    line,
                )
            })()
        } else {
            (|| {
                let mut active = intermediate.active_view_mut().map_err(FftError::Array)?;
                mixed_complex_forward_in_place(
                    &stage.local,
                    stage.output.as_ref(),
                    stage.axis,
                    active.as_mut_slice(),
                    embedding,
                    scratch,
                    line,
                )
            })()
        };
        super::record_fft_timing(&mut report, index + 1, fft_started);
        agree_result(communicator, stage_result)?;
    }
    Ok(())
}

fn mixed_c2c_reverse_tail<R: FftReal, const N: usize, const M: usize>(
    stages: &[MixedC2cStage<R, N, M>],
    transitions: &[C2cStageTransition<N, M>],
    source: &PencilArray<Complex<R>, N, M>,
    intermediate: &mut ManyPencilArray<Complex<R>, N, M>,
    transpose: &mut TransposeWorkspace<Complex<R>>,
    embedding: &mut [Complex<R>],
    scratch: &mut [Complex<R>],
    line: &mut [Complex<R>],
    normalize: bool,
    mut report: Option<&mut TransformTiming<N>>,
) -> Result<(), MixedError>
where
    Complex<R>: Equivalence,
{
    if stages.is_empty() || transitions.len() + 1 != stages.len() {
        return Err(MixedError::Fft(FftError::PreparationFailed));
    }
    let source_view = source.view();
    let communicator = stages[0].input.topology().communicator();
    let last = stages.len() - 1;
    let fft_started = Instant::now();
    let overwrite_result = intermediate
        .overwrite_with(stages[last].output.as_ref(), |mut target| {
            if stages.len() == 1 {
                if target.as_mut_slice().len() != source_view.as_slice().len() {
                    return Err(MixedError::Fft(FftError::PreparationFailed));
                }
                target
                    .as_mut_slice()
                    .copy_from_slice(source_view.as_slice());
                return Ok(());
            }
            mixed_complex_reverse(
                &stages[last].local,
                stages[last].input.as_ref(),
                stages[last].axis,
                source_view.as_slice(),
                target.as_mut_slice(),
                embedding,
                scratch,
                line,
                normalize,
            )
        })
        .map_err(map_overwrite);
    if stages.len() > 1 {
        super::record_fft_timing(&mut report, last, fft_started);
    }
    agree_result(communicator, overwrite_result)?;
    for index in (0..transitions.len()).rev() {
        execute_mixed_transition_timed(
            communicator,
            &transitions[index].backward,
            intermediate,
            transpose,
            &mut report,
            index,
        )?;
        if index != 0 {
            let stage = &stages[index];
            let fft_started = Instant::now();
            let stage_result = (|| {
                let mut active = intermediate.active_view_mut().map_err(FftError::Array)?;
                mixed_complex_reverse_in_place(
                    &stage.local,
                    stage.input.as_ref(),
                    stage.axis,
                    active.as_mut_slice(),
                    embedding,
                    scratch,
                    line,
                    normalize,
                )
            })();
            super::record_fft_timing(&mut report, index, fft_started);
            agree_result(communicator, stage_result)?;
        }
    }
    Ok(())
}

fn execute_mixed_c2c_forward<R: FftReal, const N: usize, const M: usize>(
    core: &Arc<MixedC2cCore<R, N, M>>,
    source: &PencilArray<Complex<R>, N, M>,
    destination: &mut PencilArray<Complex<R>, N, M>,
    workspace: &mut MixedC2cWorkspace<R, N, M>,
    mut report: Option<&mut TransformTiming<N>>,
) -> Result<(), MixedError>
where
    Complex<R>: Equivalence,
{
    let communicator = core.stages[0].input.topology().communicator();
    let source_view = source.view();
    let stage = &core.stages[0];
    let fft_started = Instant::now();
    let overwrite_result = workspace
        .intermediate
        .overwrite_with(stage.output.as_ref(), |mut target| {
            mixed_complex_forward(
                &stage.local,
                stage.output.as_ref(),
                stage.axis,
                source_view.as_slice(),
                target.as_mut_slice(),
                &mut workspace.embedding_line,
                &mut workspace.fft_scratch,
                &mut workspace.line_buffer,
            )
        })
        .map_err(map_overwrite);
    super::record_fft_timing(&mut report, 0, fft_started);
    agree_result(communicator, overwrite_result)?;
    mixed_c2c_forward_tail(
        &core.stages,
        &core.transitions,
        &mut workspace.intermediate,
        &mut workspace.transpose,
        destination,
        &mut workspace.embedding_line,
        &mut workspace.fft_scratch,
        &mut workspace.line_buffer,
        report,
    )
}

fn execute_mixed_c2c_reverse<R: FftReal, const N: usize, const M: usize>(
    core: &Arc<MixedC2cCore<R, N, M>>,
    source: &PencilArray<Complex<R>, N, M>,
    destination: &mut PencilArray<Complex<R>, N, M>,
    workspace: &mut MixedC2cWorkspace<R, N, M>,
    normalize: bool,
    mut report: Option<&mut TransformTiming<N>>,
) -> Result<(), MixedError>
where
    Complex<R>: Equivalence,
{
    mixed_c2c_reverse_tail(
        &core.stages,
        &core.transitions,
        source,
        &mut workspace.intermediate,
        &mut workspace.transpose,
        &mut workspace.embedding_line,
        &mut workspace.fft_scratch,
        &mut workspace.line_buffer,
        normalize,
        report.as_deref_mut(),
    )?;
    let stage = &core.stages[0];
    let fft_started = Instant::now();
    let active = workspace
        .intermediate
        .active_view()
        .map_err(FftError::Array)?;
    let mut target = destination.view_mut();
    let result = mixed_complex_reverse(
        &stage.local,
        stage.input.as_ref(),
        stage.axis,
        active.as_slice(),
        target.as_mut_slice(),
        &mut workspace.embedding_line,
        &mut workspace.fft_scratch,
        &mut workspace.line_buffer,
        normalize,
    );
    super::record_fft_timing(&mut report, 0, fft_started);
    agree_result(core.stages[0].input.topology().communicator(), result)
}

fn execute_mixed_c2c_forward_ip<R: FftReal, const N: usize, const M: usize>(
    core: &Arc<MixedC2cCore<R, N, M>>,
    array: &mut ManyPencilArray<Complex<R>, N, M>,
    transpose: &mut TransposeWorkspace<Complex<R>>,
    embedding: &mut [Complex<R>],
    scratch: &mut [Complex<R>],
    line: &mut [Complex<R>],
    mut report: Option<&mut TransformTiming<N>>,
) -> Result<(), MixedError>
where
    Complex<R>: Equivalence,
{
    let communicator = core.stages[0].input.topology().communicator();
    {
        let fft_started = Instant::now();
        let stage_result = (|| {
            let stage = &core.stages[0];
            let mut active = array.active_view_mut().map_err(FftError::Array)?;
            mixed_complex_forward_in_place(
                &stage.local,
                stage.output.as_ref(),
                stage.axis,
                active.as_mut_slice(),
                embedding,
                scratch,
                line,
            )
        })();
        super::record_fft_timing(&mut report, 0, fft_started);
        agree_result(communicator, stage_result)?;
    }
    for (index, transition) in core.transitions.iter().enumerate() {
        execute_mixed_transition_timed(
            communicator,
            &transition.forward,
            array,
            transpose,
            &mut report,
            index,
        )?;
        let fft_started = Instant::now();
        let stage_result = (|| {
            let stage = &core.stages[index + 1];
            let mut active = array.active_view_mut().map_err(FftError::Array)?;
            mixed_complex_forward_in_place(
                &stage.local,
                stage.output.as_ref(),
                stage.axis,
                active.as_mut_slice(),
                embedding,
                scratch,
                line,
            )
        })();
        super::record_fft_timing(&mut report, index + 1, fft_started);
        agree_result(communicator, stage_result)?;
    }
    Ok(())
}

fn execute_mixed_c2c_reverse_ip<R: FftReal, const N: usize, const M: usize>(
    core: &Arc<MixedC2cCore<R, N, M>>,
    array: &mut ManyPencilArray<Complex<R>, N, M>,
    transpose: &mut TransposeWorkspace<Complex<R>>,
    embedding: &mut [Complex<R>],
    scratch: &mut [Complex<R>],
    line: &mut [Complex<R>],
    normalize: bool,
    mut report: Option<&mut TransformTiming<N>>,
) -> Result<(), MixedError>
where
    Complex<R>: Equivalence,
{
    let communicator = core.stages[0].input.topology().communicator();
    let last = core.stages.len() - 1;
    {
        let fft_started = Instant::now();
        let stage_result = (|| {
            let stage = &core.stages[last];
            let mut active = array.active_view_mut().map_err(FftError::Array)?;
            mixed_complex_reverse_in_place(
                &stage.local,
                stage.input.as_ref(),
                stage.axis,
                active.as_mut_slice(),
                embedding,
                scratch,
                line,
                normalize,
            )
        })();
        super::record_fft_timing(&mut report, last, fft_started);
        agree_result(communicator, stage_result)?;
    }
    for (index, transition) in core.transitions.iter().enumerate().rev() {
        execute_mixed_transition_timed(
            communicator,
            &transition.backward,
            array,
            transpose,
            &mut report,
            index,
        )?;
        let fft_started = Instant::now();
        let stage_result = (|| {
            let stage = &core.stages[index];
            let mut active = array.active_view_mut().map_err(FftError::Array)?;
            mixed_complex_reverse_in_place(
                &stage.local,
                stage.input.as_ref(),
                stage.axis,
                active.as_mut_slice(),
                embedding,
                scratch,
                line,
                normalize,
            )
        })();
        super::record_fft_timing(&mut report, index, fft_started);
        agree_result(communicator, stage_result)?;
    }
    Ok(())
}

fn execute_mixed_c2c_overlap<R: FftReal, const N: usize, const M: usize>(
    core: &Arc<MixedC2cCore<R, N, M>>,
    direction: Direction,
    source: &PencilArray<Complex<R>, N, M>,
    destination: &mut PencilArray<Complex<R>, N, M>,
    workspace: &mut MixedC2cWorkspace<R, N, M>,
) -> Result<(), FftOverlapError<MixedError>>
where
    Complex<R>: Equivalence,
{
    let communicator = core.stages[0].input.topology().communicator();
    let forward = matches!(direction, Direction::Forward);
    if forward {
        let stage = &core.stages[0];
        let source_view = source.view();
        let result = workspace
            .intermediate
            .overwrite_with(stage.output.as_ref(), |mut target| {
                mixed_complex_forward(
                    &stage.local,
                    stage.output.as_ref(),
                    stage.axis,
                    source_view.as_slice(),
                    target.as_mut_slice(),
                    &mut workspace.embedding_line,
                    &mut workspace.fft_scratch,
                    &mut workspace.line_buffer,
                )
            })
            .map_err(map_overwrite);
        agree_result(communicator, result)?;
        for (index, transition) in core.transitions.iter().enumerate() {
            let stage = &core.stages[index + 1];
            match &transition.forward {
                super::C2cTransition::PointToPoint(plan) => {
                    let callback = |data: &mut [Complex<R>]| {
                        mixed_complex_forward_in_place(
                            &stage.local,
                            stage.output.as_ref(),
                            stage.axis,
                            data,
                            &mut workspace.embedding_line,
                            &mut workspace.fft_scratch,
                            &mut workspace.line_buffer,
                        )
                    };
                    plan.execute_in_place_with_callback(
                        &mut workspace.intermediate,
                        &mut workspace.transpose,
                        callback,
                    )
                    .map_err(map_mixed_overlap)?;
                }
                super::C2cTransition::Identity | super::C2cTransition::Local(_) => {
                    super::execute_transition(
                        &transition.forward,
                        &mut workspace.intermediate,
                        &mut workspace.transpose,
                    )
                    .map_err(MixedError::Fft)?;
                    let mut active = workspace
                        .intermediate
                        .active_view_mut()
                        .map_err(FftError::Array)?;
                    mixed_complex_forward_in_place(
                        &stage.local,
                        stage.output.as_ref(),
                        stage.axis,
                        active.as_mut_slice(),
                        &mut workspace.embedding_line,
                        &mut workspace.fft_scratch,
                        &mut workspace.line_buffer,
                    )?;
                }
                super::C2cTransition::AllToAllv(_) => unreachable!(),
            }
        }
    } else {
        let last = core.stages.len() - 1;
        let source_view = source.view();
        let result = workspace
            .intermediate
            .overwrite_with(core.stages[last].output.as_ref(), |mut target| {
                mixed_complex_reverse(
                    &core.stages[last].local,
                    core.stages[last].input.as_ref(),
                    core.stages[last].axis,
                    source_view.as_slice(),
                    target.as_mut_slice(),
                    &mut workspace.embedding_line,
                    &mut workspace.fft_scratch,
                    &mut workspace.line_buffer,
                    matches!(direction, Direction::Inverse),
                )
            })
            .map_err(map_overwrite);
        agree_result(communicator, result)?;
        for (index, transition) in core.transitions.iter().enumerate().rev() {
            let stage = &core.stages[index];
            match &transition.backward {
                super::C2cTransition::PointToPoint(plan) => {
                    let callback = |data: &mut [Complex<R>]| {
                        mixed_complex_reverse_in_place(
                            &stage.local,
                            stage.input.as_ref(),
                            stage.axis,
                            data,
                            &mut workspace.embedding_line,
                            &mut workspace.fft_scratch,
                            &mut workspace.line_buffer,
                            matches!(direction, Direction::Inverse),
                        )
                    };
                    plan.execute_in_place_with_callback(
                        &mut workspace.intermediate,
                        &mut workspace.transpose,
                        callback,
                    )
                    .map_err(map_mixed_overlap)?;
                }
                super::C2cTransition::Identity | super::C2cTransition::Local(_) => {
                    super::execute_transition(
                        &transition.backward,
                        &mut workspace.intermediate,
                        &mut workspace.transpose,
                    )
                    .map_err(MixedError::Fft)?;
                    let mut active = workspace
                        .intermediate
                        .active_view_mut()
                        .map_err(FftError::Array)?;
                    mixed_complex_reverse_in_place(
                        &stage.local,
                        stage.input.as_ref(),
                        stage.axis,
                        active.as_mut_slice(),
                        &mut workspace.embedding_line,
                        &mut workspace.fft_scratch,
                        &mut workspace.line_buffer,
                        matches!(direction, Direction::Inverse),
                    )?;
                }
                super::C2cTransition::AllToAllv(_) => unreachable!(),
            }
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

fn map_mixed_overlap<E: Into<MixedError>>(error: OverlapError<E>) -> FftOverlapError<MixedError> {
    FftOverlapError::Overlap(match error {
        OverlapError::Transpose(error) => OverlapError::Transpose(error),
        OverlapError::Callback(error) => OverlapError::Callback(error.into()),
        OverlapError::PeerPanicked => OverlapError::PeerPanicked,
        OverlapError::PeerCallbackFailed => OverlapError::PeerCallbackFailed,
        OverlapError::CollectivePreconditionFailed => OverlapError::CollectivePreconditionFailed,
    })
}

fn validate_mixed_c2c_oop<R: FftReal, const N: usize, const M: usize>(
    core: &Arc<MixedC2cCore<R, N, M>>,
    direction: Direction,
    source: &PencilArray<Complex<R>, N, M>,
    destination: &PencilArray<Complex<R>, N, M>,
    workspace: &MixedC2cWorkspace<R, N, M>,
) -> Result<(), MixedError> {
    if !Arc::ptr_eq(&workspace.core, core) {
        return Err(MixedError::Fft(FftError::WorkspaceMismatch));
    }
    let input = &core.stages[0].input;
    let output = &core.stages[core.stages.len() - 1].output;
    let (expected_source, expected_destination) = match direction {
        Direction::Forward => (input, output),
        Direction::Inverse | Direction::Backward => (output, input),
    };
    if (core.strict_array_identity && !Arc::ptr_eq(source.pencil(), expected_source))
        || !source.pencil().same_layout(expected_source.as_ref())
    {
        return Err(MixedError::Fft(FftError::InputLayoutMismatch));
    }
    if (core.strict_array_identity && !Arc::ptr_eq(destination.pencil(), expected_destination))
        || !destination
            .pencil()
            .same_layout(expected_destination.as_ref())
    {
        return Err(MixedError::Fft(FftError::OutputLayoutMismatch));
    }
    if source.extra_shape() != &core.extra_shape || destination.extra_shape() != &core.extra_shape {
        return Err(MixedError::Fft(FftError::ExtraShapeMismatch));
    }
    if workspace.intermediate.extra_shape() != &core.extra_shape {
        return Err(MixedError::Fft(FftError::WorkspaceMismatch));
    }
    let active = workspace
        .intermediate
        .active_pencil()
        .map_err(FftError::Array)?;
    validate_exact_pencil_registry(
        workspace.intermediate.pencils(),
        &core.registered_pencils,
        active,
        workspace.intermediate.storage_len(),
        workspace.intermediate.extra_shape(),
    )
    .map_err(MixedError::Fft)?;
    validate_workspace_lengths_values(
        workspace.fft_scratch.len(),
        core.fft_scratch_len,
        workspace.transpose.send_len(),
        core.transpose_send_len,
        workspace.transpose.receive_len(),
        core.transpose_receive_len,
    )?;
    if workspace.embedding_line.len() < core.embedding_len {
        return Err(MixedError::Fft(FftError::WorkspaceTooSmall {
            kind: "complex embedding",
            required: core.embedding_len,
            actual: workspace.embedding_line.len(),
        }));
    }
    if workspace.line_buffer.len() < core.strided_line_len {
        return Err(MixedError::Fft(FftError::WorkspaceTooSmall {
            kind: "complex line",
            required: core.strided_line_len,
            actual: workspace.line_buffer.len(),
        }));
    }
    Ok(())
}

fn validate_mixed_c2c_ip<R: FftReal, const N: usize, const M: usize>(
    core: &Arc<MixedC2cCore<R, N, M>>,
    direction: Direction,
    array: &MixedC2cInPlaceArray<R, N, M>,
    workspace: &MixedC2cInPlaceWorkspace<R, N, M>,
) -> Result<(), MixedError> {
    if !Arc::ptr_eq(&array.core, core) {
        return Err(MixedError::Fft(FftError::Array(
            ArrayError::IncompatiblePencils,
        )));
    }
    if !Arc::ptr_eq(&workspace.core, core) {
        return Err(MixedError::Fft(FftError::WorkspaceMismatch));
    }
    let expected_state = match direction {
        Direction::Forward => C2cState::Input,
        Direction::Inverse | Direction::Backward => C2cState::Output,
    };
    if array.state == C2cState::Poisoned {
        return Err(MixedError::Fft(FftError::Array(ArrayError::Poisoned)));
    }
    if array.state != expected_state {
        return Err(MixedError::Fft(FftError::InputLayoutMismatch));
    }
    if array.array.extra_shape() != &core.extra_shape {
        return Err(MixedError::Fft(FftError::ExtraShapeMismatch));
    }
    let active = array.array.active_pencil().map_err(FftError::Array)?;
    validate_exact_pencil_registry(
        array.array.pencils(),
        &core.registered_pencils,
        active,
        array.array.storage_len(),
        array.array.extra_shape(),
    )
    .map_err(MixedError::Fft)?;
    let expected = match direction {
        Direction::Forward => &core.stages[0].input,
        Direction::Inverse | Direction::Backward => &core.stages[core.stages.len() - 1].output,
    };
    if !array
        .array
        .active_pencil()
        .map_err(FftError::Array)?
        .same_layout(expected.as_ref())
    {
        return Err(MixedError::Fft(FftError::InputLayoutMismatch));
    }
    validate_workspace_lengths_values(
        workspace.fft_scratch.len(),
        core.fft_scratch_len,
        workspace.transpose.send_len(),
        core.transpose_send_len,
        workspace.transpose.receive_len(),
        core.transpose_receive_len,
    )?;
    if workspace.embedding_line.len() < core.embedding_len {
        return Err(MixedError::Fft(FftError::WorkspaceTooSmall {
            kind: "complex embedding",
            required: core.embedding_len,
            actual: workspace.embedding_line.len(),
        }));
    }
    if workspace.line_buffer.len() < core.strided_line_len {
        return Err(MixedError::Fft(FftError::WorkspaceTooSmall {
            kind: "complex line",
            required: core.strided_line_len,
            actual: workspace.line_buffer.len(),
        }));
    }
    Ok(())
}

impl<R: FftReal, const N: usize, const M: usize> MixedR2cPlan<R, N, M>
where
    Complex<R>: Equivalence,
{
    /// Builds an Alltoallv mixed real-to-complex plan from a canonical pencil.
    pub fn from_pencil(
        input: Arc<Pencil<N, M>>,
        extra_shape: ExtraShape,
        transforms: [AxisTransform; N],
    ) -> Result<Self, MixedError> {
        Self::from_pencil_with_layout(input, extra_shape, transforms, DistributedLayout::default())
    }

    /// Builds a mixed real-to-complex plan with the selected transition method.
    pub fn from_pencil_with_method(
        input: Arc<Pencil<N, M>>,
        extra_shape: ExtraShape,
        transforms: [AxisTransform; N],
        method: TransposeMethod,
    ) -> Result<Self, MixedError> {
        Self::from_pencil_with_layout(
            input,
            extra_shape,
            transforms,
            DistributedLayout {
                transpose_method: method,
                ..DistributedLayout::default()
            },
        )
    }

    /// Builds a mixed real-to-complex plan with an explicit layout policy.
    pub fn from_pencil_with_layout(
        input: Arc<Pencil<N, M>>,
        extra_shape: ExtraShape,
        transforms: [AxisTransform; N],
        layout: DistributedLayout,
    ) -> Result<Self, MixedError> {
        let topology = Arc::clone(input.topology());
        Self::construct(
            topology,
            *input.global_shape(),
            extra_shape,
            Ok(input),
            transforms,
            layout,
            FourierDirections::default(),
        )
    }

    /// Builds an Alltoallv mixed real-to-complex plan from a canonical array.
    pub fn from_array(
        input: &PencilArray<R, N, M>,
        transforms: [AxisTransform; N],
    ) -> Result<Self, MixedError> {
        Self::from_array_with_layout(input, transforms, DistributedLayout::default())
    }

    /// Builds a mixed real-to-complex plan from an array and method.
    pub fn from_array_with_method(
        input: &PencilArray<R, N, M>,
        transforms: [AxisTransform; N],
        method: TransposeMethod,
    ) -> Result<Self, MixedError> {
        Self::from_array_with_layout(
            input,
            transforms,
            DistributedLayout {
                transpose_method: method,
                ..DistributedLayout::default()
            },
        )
    }

    /// Builds a mixed real-to-complex plan from an array and explicit layout.
    pub fn from_array_with_layout(
        input: &PencilArray<R, N, M>,
        transforms: [AxisTransform; N],
        layout: DistributedLayout,
    ) -> Result<Self, MixedError> {
        let topology = Arc::clone(input.pencil().topology());
        Self::construct(
            topology,
            *input.pencil().global_shape(),
            input.extra_shape().clone(),
            Ok(Arc::clone(input.pencil())),
            transforms,
            layout,
            FourierDirections::default(),
        )
    }

    /// Builds a mixed real-to-complex plan with explicit Fourier signs.
    pub fn from_shape_with_fft_directions(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
        transforms: [AxisTransform; N],
        directions: FourierDirections<N>,
    ) -> Result<Self, MixedError> {
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
            transforms,
            DistributedLayout::default(),
            directions,
        )
    }

    /// Builds an Alltoallv mixed real-to-complex plan from a shape.
    pub fn from_shape(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
        transforms: [AxisTransform; N],
    ) -> Result<Self, MixedError> {
        Self::from_shape_with_layout(
            topology,
            global_shape,
            extra_shape,
            transforms,
            DistributedLayout::default(),
        )
    }

    /// Builds a mixed real-to-complex plan from a shape and method.
    pub fn from_shape_with_method(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
        transforms: [AxisTransform; N],
        method: TransposeMethod,
    ) -> Result<Self, MixedError> {
        Self::from_shape_with_layout(
            topology,
            global_shape,
            extra_shape,
            transforms,
            DistributedLayout {
                transpose_method: method,
                ..DistributedLayout::default()
            },
        )
    }

    /// Builds a mixed real-to-complex plan from a shape and explicit layout.
    pub fn from_shape_with_layout(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
        transforms: [AxisTransform; N],
        layout: DistributedLayout,
    ) -> Result<Self, MixedError> {
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
            transforms,
            layout,
            FourierDirections::default(),
        )
    }

    /// Rebuilds this plan with fresh array/workspace identities and Fourier signs.
    ///
    /// Forward is unscaled with these signs; inverse uses opposite signs and
    /// normalization, and backward uses opposite signs without normalization.
    /// Non-FFT axes must use `Forward`; other signs are rejected collectively.
    pub fn with_fft_directions(
        &self,
        directions: FourierDirections<N>,
    ) -> Result<Self, MixedError> {
        let mut plan = Self::construct_with_backend(
            Arc::clone(self.input_pencil().topology()),
            *self.input_pencil().global_shape(),
            self.core.extra_shape.clone(),
            Pencil::new(
                Arc::clone(self.input_pencil().topology()),
                *self.input_pencil().global_shape(),
                std::array::from_fn(|axis| axis),
            )
            .map_err(FftError::Pencil),
            self.core.transforms,
            self.core.layout,
            directions,
            self.core.backend,
        )
        .map_err(|error| match error {
            BackendInitError::Local(error) => error,
            #[cfg(feature = "fftw")]
            BackendInitError::Native(_) | BackendInitError::PeerPreflight => {
                MixedError::Fft(FftError::PreparationFailed)
            }
            #[cfg(not(feature = "fftw"))]
            BackendInitError::PeerPreflight => MixedError::Fft(FftError::PreparationFailed),
        })?;
        Arc::get_mut(&mut plan.core)
            .expect("fresh core")
            .strict_array_identity = true;
        Ok(plan)
    }

    /// Returns the concrete transform assigned to each logical axis.
    pub fn transforms(&self) -> [AxisTransform; N] {
        self.core.transforms
    }

    /// Alias for [`Self::transforms`].
    pub fn axis_transforms(&self) -> [AxisTransform; N] {
        self.transforms()
    }

    /// Returns the canonical real input pencil.
    pub fn input_pencil(&self) -> &Arc<Pencil<N, M>> {
        &self.core.stages[0].input
    }

    /// Returns the reduced complex output pencil.
    pub fn output_pencil(&self) -> &Arc<Pencil<N, M>> {
        &self.core.stages[self.core.stages.len() - 1].output
    }

    /// Returns the exact extra shape required by this plan.
    pub fn extra_shape(&self) -> &ExtraShape {
        &self.core.extra_shape
    }

    /// Returns the logical RFFT axis.
    pub fn reduction_axis(&self) -> usize {
        N - 1 - self.core.real_stage_index
    }

    /// Returns the original real extent on the RFFT axis.
    pub fn original_n(&self) -> usize {
        self.core.real_len
    }

    /// Returns the transport and memory-layout policy used by this plan.
    pub fn layout(&self) -> DistributedLayout {
        self.core.layout
    }

    /// Returns the configured Fourier signs.
    pub fn fft_directions(&self) -> FourierDirections<N> {
        self.core.directions
    }

    /// Returns the selected local backend.
    pub fn backend_kind(&self) -> crate::BackendKind {
        self.core.backend.kind()
    }

    pub(super) fn collection_descriptor(&self) -> &[u64] {
        &self.core.descriptor
    }

    #[cfg(feature = "fftw")]
    /// Returns native planning options, or `None` for RustFFT.
    pub fn options(&self) -> Option<PlanOptions> {
        match self.core.backend {
            BackendChoice::Fftw(options) => Some(options),
            BackendChoice::RustFft => None,
        }
    }

    pub(super) fn collection_preflight_forward(
        &self,
        source: &PencilArray<R, N, M>,
        destination: &PencilArray<Complex<R>, N, M>,
        workspace: &MixedR2cWorkspace<R, N, M>,
    ) -> Result<(), MixedError> {
        validate_mixed_r2c_forward(&self.core, source, destination, workspace)
    }

    pub(super) fn collection_preflight_inverse(
        &self,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &PencilArray<R, N, M>,
        workspace: &MixedR2cWorkspace<R, N, M>,
    ) -> Result<(), MixedError> {
        validate_mixed_r2c_reverse(&self.core, source, destination, workspace)
    }

    pub(super) fn collection_preflight_in_place(
        &self,
        direction: super::Direction,
        array: &MixedR2cInPlaceArray<R, N, M>,
        workspace: &MixedR2cInPlaceWorkspace<R, N, M>,
    ) -> Result<(), MixedError> {
        validate_mixed_r2c_ip(&self.core, direction, array, workspace)
    }

    /// Allocates a zero-initialized real input array.
    pub fn allocate_input(&self) -> Result<PencilArray<R, N, M>, MixedError> {
        PencilArray::from_fn(
            Arc::clone(self.input_pencil()),
            self.core.extra_shape.clone(),
            R::zero,
        )
        .map_err(FftError::Array)
        .map_err(Into::into)
    }

    /// Allocates a zero-initialized reduced complex output array.
    pub fn allocate_output(&self) -> Result<PencilArray<Complex<R>, N, M>, MixedError> {
        PencilArray::from_fn(
            Arc::clone(self.output_pencil()),
            self.core.extra_shape.clone(),
            zero_complex::<R>,
        )
        .map_err(FftError::Array)
        .map_err(Into::into)
    }

    /// Allocates reusable out-of-place workspace.
    pub fn allocate_workspace(&self) -> Result<MixedR2cWorkspace<R, N, M>, MixedError> {
        let boundary = self.core.real_stage_index;
        let intermediate = ManyPencilArray::from_elem(
            self.core.complex_pencils.clone(),
            0,
            self.core.extra_shape.clone(),
            zero_complex::<R>(),
        )
        .map_err(map_array_allocation)?;
        let real_intermediate = if boundary == 0 {
            None
        } else {
            Some(
                ManyPencilArray::from_elem(
                    self.core.real_pencils.clone(),
                    0,
                    self.core.extra_shape.clone(),
                    R::zero(),
                )
                .map_err(map_array_allocation)?,
            )
        };
        let real_transpose = if boundary == 0 {
            None
        } else {
            Some(TransposeWorkspace::from_vecs(
                initialized_vec(self.core.real_transpose_send_len, R::zero())?,
                initialized_vec(self.core.real_transpose_receive_len, R::zero())?,
            ))
        };
        Ok(MixedR2cWorkspace {
            core: Arc::clone(&self.core),
            intermediate,
            real_intermediate,
            transpose: TransposeWorkspace::from_vecs(
                initialized_vec(self.core.transpose_send_len, zero_complex::<R>())?,
                initialized_vec(self.core.transpose_receive_len, zero_complex::<R>())?,
            ),
            real_transpose,
            embedding_line: initialized_vec(self.core.embedding_len, zero_complex::<R>())?,
            fft_scratch: initialized_vec(self.core.fft_scratch_len, zero_complex::<R>())?,
            real_source_line: initialized_vec(self.core.real_len, R::zero())?,
            real_line: initialized_vec(self.core.real_line_len.max(self.core.real_len), R::zero())?,
            complex_source_line: initialized_vec(self.core.complex_len, zero_complex::<R>())?,
            complex_line: initialized_vec(
                self.core.complex_line_len.max(self.core.complex_len),
                zero_complex::<R>(),
            )?,
            real_strided_line: initialized_vec(self.core.real_line_len, R::zero())?,
            complex_strided_line: initialized_vec(self.core.complex_line_len, zero_complex::<R>())?,
        })
    }

    /// Allocates the one-allocation real/complex in-place array.
    pub fn allocate_in_place(&self) -> Result<MixedR2cInPlaceArray<R, N, M>, MixedError> {
        let real_pencils = self.core.real_pencils.clone();
        let complex_pencils = self.core.complex_pencils.clone();
        let (real_storage_len, complex_storage_len, storage_bytes, complex_capacity) =
            mixed_r2c_storage_requirements(&self.core)?;
        let mut storage = Vec::new();
        storage.try_reserve_exact(complex_capacity).map_err(|_| {
            MixedError::Fft(FftError::AllocationFailed {
                required: complex_capacity,
            })
        })?;
        storage.resize(complex_capacity, zero_complex::<R>());
        let mut real_storage = try_cast_vec(storage)
            .map_err(|(_, _)| MixedError::Fft(FftError::StorageLayoutMismatch))?;
        if real_storage.len() < real_storage_len {
            return Err(MixedError::Fft(FftError::StorageLayoutMismatch));
        }
        real_storage.truncate(real_storage_len);
        let storage =
            ManyPencilArray::from_vec(real_pencils, 0, self.core.extra_shape.clone(), real_storage)
                .map_err(map_array_allocation)?;
        Ok(MixedR2cInPlaceArray {
            core: Arc::clone(&self.core),
            real_pencils: None,
            complex_pencils: Some(complex_pencils),
            real_storage_len,
            complex_storage_len,
            storage_bytes,
            storage: Some(MixedR2cStorage::Real(storage)),
            state: R2cState::RealInput,
            #[cfg(test)]
            test_hook: None,
        })
    }

    /// Allocates reusable in-place workspace.
    pub fn allocate_in_place_workspace(
        &self,
    ) -> Result<MixedR2cInPlaceWorkspace<R, N, M>, MixedError> {
        Ok(MixedR2cInPlaceWorkspace {
            core: Arc::clone(&self.core),
            transpose: TransposeWorkspace::from_vecs(
                initialized_vec(self.core.transpose_send_len, zero_complex::<R>())?,
                initialized_vec(self.core.transpose_receive_len, zero_complex::<R>())?,
            ),
            real_transpose: if self.core.real_stage_index == 0 {
                None
            } else {
                Some(TransposeWorkspace::from_vecs(
                    initialized_vec(self.core.real_transpose_send_len, R::zero())?,
                    initialized_vec(self.core.real_transpose_receive_len, R::zero())?,
                ))
            },
            embedding_line: initialized_vec(self.core.embedding_len, zero_complex::<R>())?,
            fft_scratch: initialized_vec(self.core.fft_scratch_len, zero_complex::<R>())?,
            real_source_line: initialized_vec(self.core.real_len, R::zero())?,
            real_line: initialized_vec(self.core.real_line_len.max(self.core.real_len), R::zero())?,
            complex_source_line: initialized_vec(self.core.complex_len, zero_complex::<R>())?,
            complex_line: initialized_vec(
                self.core.complex_line_len.max(self.core.complex_len),
                zero_complex::<R>(),
            )?,
            real_strided_line: initialized_vec(self.core.real_line_len, R::zero())?,
            complex_strided_line: initialized_vec(self.core.complex_line_len, zero_complex::<R>())?,
        })
    }

    /// Computes the unnormalized mixed forward transform.
    pub fn forward(
        &self,
        source: &PencilArray<R, N, M>,
        destination: &mut PencilArray<Complex<R>, N, M>,
        workspace: &mut MixedR2cWorkspace<R, N, M>,
    ) -> Result<(), MixedError> {
        self.execute_forward_public(source, destination, workspace, None)
    }

    /// Computes the mixed forward transform and records per-stage timing.
    pub fn forward_with_timing(
        &self,
        source: &PencilArray<R, N, M>,
        destination: &mut PencilArray<Complex<R>, N, M>,
        workspace: &mut MixedR2cWorkspace<R, N, M>,
    ) -> Result<TransformTiming<N>, MixedError> {
        let mut timing = TransformTiming::default();
        self.execute_forward_public(source, destination, workspace, Some(&mut timing))?;
        Ok(timing)
    }

    /// Computes the per-axis normalized mixed inverse transform.
    pub fn inverse(
        &self,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &mut PencilArray<R, N, M>,
        workspace: &mut MixedR2cWorkspace<R, N, M>,
    ) -> Result<(), MixedError> {
        self.execute_reverse_public(source, destination, workspace, true, None)
    }

    /// Computes the raw paired mixed backward transform.
    pub fn backward(
        &self,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &mut PencilArray<R, N, M>,
        workspace: &mut MixedR2cWorkspace<R, N, M>,
    ) -> Result<(), MixedError> {
        self.execute_reverse_public(source, destination, workspace, false, None)
    }

    /// Computes forward with the next local kernel after receive/unpack completion and before P2P send waits.
    pub fn forward_with_overlap(
        &self,
        source: &PencilArray<R, N, M>,
        destination: &mut PencilArray<Complex<R>, N, M>,
        workspace: &mut MixedR2cWorkspace<R, N, M>,
    ) -> Result<(), FftOverlapError<MixedError>> {
        self.execute_forward_overlap(source, destination, workspace)
    }
    /// Computes inverse with the next local kernel after receive/unpack completion and before P2P send waits.
    pub fn inverse_with_overlap(
        &self,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &mut PencilArray<R, N, M>,
        workspace: &mut MixedR2cWorkspace<R, N, M>,
    ) -> Result<(), FftOverlapError<MixedError>> {
        self.execute_reverse_overlap(source, destination, workspace, true)
    }
    /// Computes backward with the next local kernel after receive/unpack completion and before P2P send waits.
    pub fn backward_with_overlap(
        &self,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &mut PencilArray<R, N, M>,
        workspace: &mut MixedR2cWorkspace<R, N, M>,
    ) -> Result<(), FftOverlapError<MixedError>> {
        self.execute_reverse_overlap(source, destination, workspace, false)
    }

    /// Computes the mixed forward transform in place.
    pub fn forward_in_place(
        &self,
        array: &mut MixedR2cInPlaceArray<R, N, M>,
        workspace: &mut MixedR2cInPlaceWorkspace<R, N, M>,
    ) -> Result<(), MixedError> {
        self.execute_in_place(Direction::Forward, array, workspace, None)
    }

    /// Computes the mixed inverse transform and records per-stage timing.
    pub fn inverse_with_timing(
        &self,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &mut PencilArray<R, N, M>,
        workspace: &mut MixedR2cWorkspace<R, N, M>,
    ) -> Result<TransformTiming<N>, MixedError> {
        let mut timing = TransformTiming::default();
        self.execute_reverse_public(source, destination, workspace, true, Some(&mut timing))?;
        Ok(timing)
    }
    /// Computes the mixed backward transform and records per-stage timing.
    pub fn backward_with_timing(
        &self,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &mut PencilArray<R, N, M>,
        workspace: &mut MixedR2cWorkspace<R, N, M>,
    ) -> Result<TransformTiming<N>, MixedError> {
        let mut timing = TransformTiming::default();
        self.execute_reverse_public(source, destination, workspace, false, Some(&mut timing))?;
        Ok(timing)
    }
    /// Computes the in-place mixed forward transform and records timing.
    pub fn forward_in_place_with_timing(
        &self,
        array: &mut MixedR2cInPlaceArray<R, N, M>,
        workspace: &mut MixedR2cInPlaceWorkspace<R, N, M>,
    ) -> Result<TransformTiming<N>, MixedError> {
        let mut timing = TransformTiming::default();
        self.execute_in_place(Direction::Forward, array, workspace, Some(&mut timing))?;
        Ok(timing)
    }
    /// Computes the in-place mixed inverse transform and records timing.
    pub fn inverse_in_place_with_timing(
        &self,
        array: &mut MixedR2cInPlaceArray<R, N, M>,
        workspace: &mut MixedR2cInPlaceWorkspace<R, N, M>,
    ) -> Result<TransformTiming<N>, MixedError> {
        let mut timing = TransformTiming::default();
        self.execute_in_place(Direction::Inverse, array, workspace, Some(&mut timing))?;
        Ok(timing)
    }
    /// Computes the in-place mixed backward transform and records timing.
    pub fn backward_in_place_with_timing(
        &self,
        array: &mut MixedR2cInPlaceArray<R, N, M>,
        workspace: &mut MixedR2cInPlaceWorkspace<R, N, M>,
    ) -> Result<TransformTiming<N>, MixedError> {
        let mut timing = TransformTiming::default();
        self.execute_in_place(Direction::Backward, array, workspace, Some(&mut timing))?;
        Ok(timing)
    }

    /// Computes the per-axis normalized mixed inverse in place.
    pub fn inverse_in_place(
        &self,
        array: &mut MixedR2cInPlaceArray<R, N, M>,
        workspace: &mut MixedR2cInPlaceWorkspace<R, N, M>,
    ) -> Result<(), MixedError> {
        self.execute_in_place(Direction::Inverse, array, workspace, None)
    }

    /// Computes the raw paired mixed backward transform in place.
    pub fn backward_in_place(
        &self,
        array: &mut MixedR2cInPlaceArray<R, N, M>,
        workspace: &mut MixedR2cInPlaceWorkspace<R, N, M>,
    ) -> Result<(), MixedError> {
        self.execute_in_place(Direction::Backward, array, workspace, None)
    }

    fn construct(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
        input: Result<Arc<Pencil<N, M>>, FftError>,
        transforms: [AxisTransform; N],
        layout: DistributedLayout,
        directions: FourierDirections<N>,
    ) -> Result<Self, MixedError> {
        Self::construct_with_backend(
            topology,
            global_shape,
            extra_shape,
            input,
            transforms,
            layout,
            directions,
            BackendChoice::RustFft,
        )
        .map_err(|error| match error {
            BackendInitError::Local(error) => error,
            _ => MixedError::Fft(FftError::PreparationFailed),
        })
    }

    #[cfg(feature = "fftw")]
    #[allow(private_bounds)]
    /// Builds a mixed plan from a shape using FFTW.
    pub fn from_shape_with_fftw(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
        transforms: [AxisTransform; N],
        options: PlanOptions,
    ) -> Result<Self, BackendInitError<MixedError>>
    where
        R: crate::backend::FftwReal,
    {
        let input = Pencil::new(
            Arc::clone(&topology),
            global_shape,
            std::array::from_fn(|axis| axis),
        )
        .map_err(FftError::Pencil);
        Self::construct_with_backend(
            topology,
            global_shape,
            extra_shape,
            input,
            transforms,
            DistributedLayout::default(),
            FourierDirections::default(),
            BackendChoice::Fftw(options),
        )
    }

    #[cfg(feature = "fftw")]
    #[allow(private_bounds)]
    /// Rebuilds this mixed plan using FFTW.
    pub fn with_fftw(&self, options: PlanOptions) -> Result<Self, BackendInitError<MixedError>>
    where
        R: crate::backend::FftwReal,
    {
        let topology = Arc::clone(self.input_pencil().topology());
        let shape = *self.input_pencil().global_shape();
        let input = Pencil::new(
            Arc::clone(&topology),
            shape,
            std::array::from_fn(|axis| axis),
        )
        .map_err(FftError::Pencil);
        let mut plan = Self::construct_with_backend(
            topology,
            shape,
            self.core.extra_shape.clone(),
            input,
            self.core.transforms,
            self.core.layout,
            self.core.directions,
            BackendChoice::Fftw(options),
        )?;
        Arc::get_mut(&mut plan.core)
            .expect("fresh core")
            .strict_array_identity = true;
        Ok(plan)
    }

    fn construct_with_backend(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
        input: Result<Arc<Pencil<N, M>>, FftError>,
        transforms: [AxisTransform; N],
        layout: DistributedLayout,
        directions: FourierDirections<N>,
        backend: BackendChoice,
    ) -> Result<Self, BackendInitError<MixedError>> {
        let communicator = topology.communicator();
        let local_boundary = validate_r2c_graph(global_shape, transforms).ok();
        let mut reduced_shape = global_shape;
        let original_len = local_boundary.map(|axis| global_shape[axis]).unwrap_or(0);
        if let Some(axis) = local_boundary {
            reduced_shape[axis] = global_shape[axis] / 2 + 1;
        }
        let expected_len =
            mixed_descriptor_len::<N, M>(&extra_shape).and_then(|length| length.checked_add(N + 7));
        let descriptor = expected_len.and_then(|_| {
            build_mixed_descriptor::<R, N, M>(
                &topology,
                global_shape,
                reduced_shape,
                &extra_shape,
                transforms,
                2,
                local_boundary.unwrap_or(usize::MAX),
                original_len,
                layout,
            )
            .ok()
            .and_then(|mut descriptor| {
                descriptor.try_reserve_exact(N + 7).ok()?;
                descriptor.extend(backend.descriptor_words::<R>());
                descriptor.extend(directions.0.iter().map(|direction| match direction {
                    FourierDirection::Forward => 0,
                    FourierDirection::Backward => 1,
                }));
                Some(descriptor)
            })
        });
        let operation = match backend {
            BackendChoice::RustFft => super::OPERATION_MIXED_R2C_PLAN,
            #[cfg(feature = "fftw")]
            BackendChoice::Fftw(_) => OPERATION_MIXED_R2C_PLAN_NATIVE,
        };
        let header = mixed_header::<N, M>(operation, N, M, expected_len);
        if !agree_header(communicator, header) {
            return Err(BackendInitError::Local(MixedError::Fft(
                FftError::CollectiveDescriptorMismatch,
            )));
        }
        let descriptor = collective_descriptor(communicator, descriptor, expected_len)?;
        agree_result(
            communicator,
            validate_r2c_directions(transforms, directions),
        )?;
        let reduction_axis =
            agree_result(communicator, validate_r2c_graph(global_shape, transforms))?;
        let boundary = N
            .checked_sub(1 + reduction_axis)
            .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
        let input = agree_result(communicator, validate_input(input, &topology, global_shape))?;
        let mut reduced_shape = global_shape;
        let real_len = global_shape[reduction_axis];
        reduced_shape[reduction_axis] = real_len / 2 + 1;
        let original_route = agree_result(
            communicator,
            build_route(
                Ok(Arc::clone(&input)),
                &topology,
                global_shape,
                layout.permute_dims,
            ),
        )?;
        let reduced_input = Pencil::new(
            Arc::clone(&topology),
            reduced_shape,
            std::array::from_fn(|axis| axis),
        )
        .map_err(FftError::Pencil);
        let reduced_input = agree_result(communicator, reduced_input)?;
        let reduced_route = agree_result(
            communicator,
            build_route(
                Ok(reduced_input),
                &topology,
                reduced_shape,
                layout.permute_dims,
            ),
        )?;
        let prepared = prepare_mixed_r2c_stages(
            &original_route,
            &reduced_route,
            global_shape,
            reduced_shape,
            reduction_axis,
            transforms,
            directions,
            backend,
        );
        if !collective_valid(communicator, prepared.is_ok()) {
            return Err(match prepared {
                Err(error) => error,
                Ok(_) => BackendInitError::PeerPreflight,
            });
        }
        let (stages, embedding_len, fft_scratch_len, real_line_len, complex_line_len) =
            prepared.expect("collective backend preflight accepted");
        let raw_absolute_threshold = agree_result(
            communicator,
            mixed_raw_threshold::<R, N, M>(&stages, &extra_shape, boundary),
        )?;
        let layout_stages = agree_result(communicator, mixed_r2c_layout_stages(&stages))?;
        let stage_prep = StagePreparation {
            stages: layout_stages,
            fft_scratch_len: 0,
        };
        let mut distributed = Vec::new();
        agree_result(
            communicator,
            distributed
                .try_reserve_exact(N - 1)
                .map_err(|_| MixedError::Fft(FftError::AllocationFailed { required: N - 1 })),
        )?;
        for index in 0..N - 1 {
            distributed.push(if index < boundary {
                original_route.distributed[index]
            } else {
                reduced_route.distributed[index]
            });
        }
        let (
            transitions,
            transpose_send_len,
            transpose_receive_len,
            real_transpose_send_len,
            real_transpose_receive_len,
        ) = agree_result(
            communicator,
            build_transitions::<R, N, M>(
                communicator,
                &stage_prep,
                &distributed,
                &extra_shape,
                layout.transpose_method,
                boundary,
            ),
        )?;
        let real_pencils = agree_result(communicator, mixed_r2c_real_pencils(&stages, boundary))?;
        let complex_pencils =
            agree_result(communicator, mixed_r2c_complex_pencils(&stages, boundary))?;
        let (real_line_required, complex_line_required) = agree_result(
            communicator,
            (|| {
                let mut real_line_required = 0;
                let mut complex_line_required = 0;
                for stage in &stages {
                    let stride = memory_stride(stage.input.as_ref(), stage.axis)?;
                    match &stage.local {
                        MixedR2cStageLocal::Real(local) if stride > 1 => {
                            real_line_required = real_line_required.max(local_line_len_real(local));
                        }
                        MixedR2cStageLocal::Complex(local) if stride > 1 => {
                            complex_line_required =
                                complex_line_required.max(local_line_len_complex(local));
                        }
                        _ => {}
                    }
                }
                if memory_stride(stages[boundary].input.as_ref(), stages[boundary].axis)? > 1 {
                    complex_line_required = complex_line_required.max(real_len / 2 + 1);
                }
                Ok::<_, MixedError>((real_line_required, complex_line_required))
            })(),
        )?;
        Ok(Self {
            core: Arc::new(MixedR2cCore {
                stages,
                transitions: transitions.into_boxed_slice(),
                real_pencils,
                complex_pencils,
                extra_shape,
                transforms,
                layout,
                descriptor: descriptor.into_boxed_slice(),
                real_stage_index: boundary,
                real_len,
                complex_len: real_len / 2 + 1,
                embedding_len,
                fft_scratch_len,
                real_line_len: real_line_len.max(real_line_required),
                complex_line_len: complex_line_len.max(complex_line_required),
                transpose_send_len,
                transpose_receive_len,
                real_transpose_send_len,
                real_transpose_receive_len,
                raw_absolute_threshold,
                directions,
                backend,
                strict_array_identity: backend.kind() == crate::BackendKind::Fftw,
            }),
        })
    }

    fn execute_forward_public(
        &self,
        source: &PencilArray<R, N, M>,
        destination: &mut PencilArray<Complex<R>, N, M>,
        workspace: &mut MixedR2cWorkspace<R, N, M>,
        mut report: Option<&mut TransformTiming<N>>,
    ) -> Result<(), MixedError> {
        let started = Instant::now();
        let communicator = self.input_pencil().topology().communicator();
        agree_execution_descriptor_ref::<N, M>(
            communicator,
            if report.is_some() {
                OPERATION_MIXED_R2C_FORWARD_TIMED
            } else {
                OPERATION_MIXED_R2C_FORWARD
            },
            &self.core.descriptor,
        )?;
        let preflight = validate_mixed_r2c_forward(&self.core, source, destination, workspace);
        if !collective_valid(communicator, preflight.is_ok()) {
            return Err(preflight
                .err()
                .unwrap_or(MixedError::Fft(FftError::CollectivePreconditionFailed)));
        }
        preflight.expect("mixed R2C forward preflight succeeded");
        let result = execute_mixed_r2c_forward(
            &self.core,
            source,
            destination,
            workspace,
            report.as_deref_mut(),
        );
        if let Some(timing) = report {
            timing.total = started.elapsed();
        }
        result
    }

    fn execute_reverse_public(
        &self,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &mut PencilArray<R, N, M>,
        workspace: &mut MixedR2cWorkspace<R, N, M>,
        normalize: bool,
        mut report: Option<&mut TransformTiming<N>>,
    ) -> Result<(), MixedError> {
        let started = Instant::now();
        let communicator = self.input_pencil().topology().communicator();
        agree_execution_descriptor_ref::<N, M>(
            communicator,
            if report.is_some() {
                if normalize {
                    OPERATION_MIXED_R2C_INVERSE_TIMED
                } else {
                    OPERATION_MIXED_R2C_BACKWARD_TIMED
                }
            } else if normalize {
                OPERATION_MIXED_R2C_INVERSE
            } else {
                OPERATION_MIXED_R2C_BACKWARD
            },
            &self.core.descriptor,
        )?;
        let preflight = validate_mixed_r2c_reverse(&self.core, source, destination, workspace);
        if !collective_valid(communicator, preflight.is_ok()) {
            return Err(preflight
                .err()
                .unwrap_or(MixedError::Fft(FftError::CollectivePreconditionFailed)));
        }
        preflight.expect("mixed R2C reverse preflight succeeded");
        let result = execute_mixed_r2c_reverse(
            &self.core,
            source,
            destination,
            workspace,
            normalize,
            report.as_deref_mut(),
        );
        if let Some(timing) = report {
            timing.total = started.elapsed();
        }
        result
    }

    fn execute_forward_overlap(
        &self,
        source_real: &PencilArray<R, N, M>,
        destination_complex: &mut PencilArray<Complex<R>, N, M>,
        workspace: &mut MixedR2cWorkspace<R, N, M>,
    ) -> Result<(), FftOverlapError<MixedError>> {
        let communicator = self.input_pencil().topology().communicator();
        agree_execution_descriptor_ref::<N, M>(
            communicator,
            OPERATION_MIXED_R2C_FORWARD_OVERLAP,
            &self.core.descriptor,
        )?;
        if self.core.layout.transpose_method != TransposeMethod::PointToPoint {
            return Err(FftOverlapError::UnsupportedTransport);
        }
        let result =
            validate_mixed_r2c_forward(&self.core, source_real, destination_complex, workspace);
        if !collective_valid(communicator, result.is_ok()) {
            return Err(result
                .err()
                .unwrap_or(MixedError::Fft(FftError::CollectivePreconditionFailed))
                .into());
        }
        result?;
        if !collective_valid(
            communicator,
            self.core.transitions.iter().all(|t| {
                matches!(
                    t.forward,
                    super::C2cTransition::Identity
                        | super::C2cTransition::Local(_)
                        | super::C2cTransition::PointToPoint(_)
                )
            }),
        ) {
            return Err(FftOverlapError::UnsupportedTransport);
        }
        execute_mixed_r2c_forward_overlap(&self.core, source_real, destination_complex, workspace)
    }

    fn execute_reverse_overlap(
        &self,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &mut PencilArray<R, N, M>,
        workspace: &mut MixedR2cWorkspace<R, N, M>,
        normalize: bool,
    ) -> Result<(), FftOverlapError<MixedError>> {
        let communicator = self.input_pencil().topology().communicator();
        agree_execution_descriptor_ref::<N, M>(
            communicator,
            if normalize {
                OPERATION_MIXED_R2C_INVERSE_OVERLAP
            } else {
                OPERATION_MIXED_R2C_BACKWARD_OVERLAP
            },
            &self.core.descriptor,
        )?;
        if self.core.layout.transpose_method != TransposeMethod::PointToPoint {
            return Err(FftOverlapError::UnsupportedTransport);
        }
        let result = validate_mixed_r2c_reverse(&self.core, source, destination, workspace);
        if !collective_valid(communicator, result.is_ok()) {
            return Err(result
                .err()
                .unwrap_or(MixedError::Fft(FftError::CollectivePreconditionFailed))
                .into());
        }
        result?;
        if !collective_valid(
            communicator,
            self.core.transitions.iter().all(|t| {
                matches!(
                    t.backward,
                    super::C2cTransition::Identity
                        | super::C2cTransition::Local(_)
                        | super::C2cTransition::PointToPoint(_)
                )
            }),
        ) {
            return Err(FftOverlapError::UnsupportedTransport);
        }
        execute_mixed_r2c_reverse_overlap(&self.core, source, destination, workspace, normalize)
    }

    fn execute_in_place(
        &self,
        direction: Direction,
        array: &mut MixedR2cInPlaceArray<R, N, M>,
        workspace: &mut MixedR2cInPlaceWorkspace<R, N, M>,
        mut report: Option<&mut TransformTiming<N>>,
    ) -> Result<(), MixedError> {
        let started = Instant::now();
        let communicator = self.input_pencil().topology().communicator();
        agree_execution_descriptor_ref::<N, M>(
            communicator,
            if report.is_some() {
                match direction {
                    Direction::Forward => OPERATION_MIXED_R2C_FORWARD_IN_PLACE_TIMED,
                    Direction::Inverse => OPERATION_MIXED_R2C_INVERSE_IN_PLACE_TIMED,
                    Direction::Backward => OPERATION_MIXED_R2C_BACKWARD_IN_PLACE_TIMED,
                }
            } else {
                mixed_r2c_operation(direction, true)
            },
            &self.core.descriptor,
        )?;
        let preflight = validate_mixed_r2c_ip(&self.core, direction, array, workspace);
        if !collective_valid(communicator, preflight.is_ok()) {
            return Err(preflight
                .err()
                .unwrap_or(MixedError::Fft(FftError::CollectivePreconditionFailed)));
        }
        preflight.expect("mixed R2C in-place preflight succeeded");
        array.state = R2cState::Poisoned;
        #[cfg(test)]
        if array.test_hook == Some(MixedInPlaceTestHook::Start) {
            panic!("injected mixed R2C in-place panic after start");
        }
        let result = match direction {
            Direction::Forward => {
                execute_mixed_r2c_forward_ip(&self.core, array, workspace, report.as_deref_mut())
            }
            Direction::Inverse | Direction::Backward => execute_mixed_r2c_reverse_ip(
                &self.core,
                array,
                workspace,
                matches!(direction, Direction::Inverse),
                report.as_deref_mut(),
            ),
        };
        if result.is_ok() {
            array.state = match direction {
                Direction::Forward => R2cState::ComplexOutput,
                Direction::Inverse | Direction::Backward => R2cState::RealInput,
            };
        }
        if let Some(timing) = report {
            timing.total = started.elapsed();
        }
        result
    }
}

#[allow(clippy::type_complexity)]
fn prepare_mixed_r2c_stages<R: FftReal, const N: usize, const M: usize>(
    original_route: &RouteCandidate<N, M>,
    reduced_route: &RouteCandidate<N, M>,
    original_shape: [usize; N],
    reduced_shape: [usize; N],
    boundary: usize,
    transforms: [AxisTransform; N],
    directions: FourierDirections<N>,
    backend: BackendChoice,
) -> Result<(Box<[MixedR2cStage<R, N, M>]>, usize, usize, usize, usize), BackendInitError<MixedError>>
{
    if original_route.stages.len() != N || reduced_route.stages.len() != N {
        return Err(BackendInitError::Local(MixedError::Fft(
            FftError::PreparationFailed,
        )));
    }
    let mut stages = Vec::new();
    stages
        .try_reserve_exact(N)
        .map_err(|_| MixedError::Fft(FftError::AllocationFailed { required: N }))?;
    let mut embedding_len = 0;
    let mut scratch_len = 0;
    let mut real_line_len = 0;
    let mut complex_line_len = 0;
    for index in 0..N {
        let axis = N - 1 - index;
        let (input, output, local) = if axis > boundary {
            validate_stage_pencil(&original_route.stages[index], axis, original_shape[axis])?;
            let local = match transforms[axis] {
                AxisTransform::None => MixedR2cStageLocal::Real(MixedRealLocal::Identity),
                AxisTransform::R2r(kind) => MixedR2cStageLocal::Real(MixedRealLocal::R2r(
                    mixed_r2r_local::<R>(kind, original_shape[axis], backend)?,
                )),
                _ => return Err(BackendInitError::Local(MixedError::InvalidGraph)),
            };
            (
                Arc::clone(&original_route.stages[index]),
                Arc::clone(&original_route.stages[index]),
                local,
            )
        } else if axis == boundary {
            validate_stage_pencil(&original_route.stages[index], axis, original_shape[axis])?;
            validate_stage_pencil(&reduced_route.stages[index], axis, reduced_shape[axis])?;
            (
                Arc::clone(&original_route.stages[index]),
                Arc::clone(&reduced_route.stages[index]),
                MixedR2cStageLocal::Real(MixedRealLocal::Rfft(match backend {
                    BackendChoice::RustFft => LocalR2cPlan::new(original_shape[axis])
                        .map_err(MixedError::from)
                        .map_err(BackendInitError::Local)?,
                    #[cfg(feature = "fftw")]
                    BackendChoice::Fftw(options) => {
                        LocalR2cPlan::new_fftw(original_shape[axis], options).map_err(|error| {
                            match error {
                                BackendInitError::Local(e) => {
                                    BackendInitError::Local(MixedError::LocalR2c(e))
                                }
                                BackendInitError::Native(e) => BackendInitError::Native(e),
                                BackendInitError::PeerPreflight => BackendInitError::PeerPreflight,
                            }
                        })?
                    }
                })),
            )
        } else {
            validate_stage_pencil(&reduced_route.stages[index], axis, reduced_shape[axis])?;
            let local = match transforms[axis] {
                AxisTransform::None => MixedR2cStageLocal::Complex(MixedComplexLocal::Identity),
                AxisTransform::Fft => {
                    MixedR2cStageLocal::Complex(MixedComplexLocal::Fft(match backend {
                        BackendChoice::RustFft => LocalC2cPlan::new_with_sign(
                            reduced_shape[axis],
                            directions.get(axis) == Some(FourierDirection::Backward),
                        )
                        .map_err(MixedError::from)
                        .map_err(BackendInitError::Local)?,
                        #[cfg(feature = "fftw")]
                        BackendChoice::Fftw(options) => LocalC2cPlan::new_fftw_with_sign(
                            reduced_shape[axis],
                            directions.get(axis) == Some(FourierDirection::Backward),
                            options,
                        )
                        .map_err(|error| match error {
                            BackendInitError::Local(e) => {
                                BackendInitError::Local(MixedError::LocalC2c(e))
                            }
                            BackendInitError::Native(e) => BackendInitError::Native(e),
                            BackendInitError::PeerPreflight => BackendInitError::PeerPreflight,
                        })?,
                    }))
                }
                AxisTransform::R2r(kind) => {
                    MixedR2cStageLocal::Complex(MixedComplexLocal::R2r(mixed_r2r_local::<
                        Complex<R>,
                    >(
                        kind,
                        reduced_shape[axis],
                        backend,
                    )?))
                }
                AxisTransform::Rfft => {
                    return Err(BackendInitError::Local(MixedError::InvalidGraph));
                }
            };
            (
                Arc::clone(&reduced_route.stages[index]),
                Arc::clone(&reduced_route.stages[index]),
                local,
            )
        };
        match &local {
            MixedR2cStageLocal::Real(local) => {
                embedding_len = embedding_len.max(local.embedding_len());
                scratch_len = scratch_len.max(local.scratch_len());
                real_line_len = real_line_len.max(local.line_len());
            }
            MixedR2cStageLocal::Complex(local) => {
                embedding_len = embedding_len.max(local.embedding_len());
                scratch_len = scratch_len.max(local.scratch_len());
                complex_line_len = complex_line_len.max(local.line_len());
            }
        }
        stages.push(MixedR2cStage {
            axis,
            input,
            output,
            local,
        });
    }
    Ok((
        stages.into_boxed_slice(),
        embedding_len,
        scratch_len,
        real_line_len,
        complex_line_len,
    ))
}

fn mixed_r2c_layout_stages<R: FftReal, const N: usize, const M: usize>(
    stages: &[MixedR2cStage<R, N, M>],
) -> Result<Box<[TransformStage<R, N, M>]>, MixedError> {
    let mut result = Vec::new();
    result.try_reserve_exact(stages.len()).map_err(|_| {
        MixedError::Fft(FftError::AllocationFailed {
            required: stages.len(),
        })
    })?;
    result.extend(stages.iter().map(|stage| TransformStage {
        axis: stage.axis,
        input: Arc::clone(&stage.input),
        output: Arc::clone(&stage.output),
        local: LocalTransform::Identity,
    }));
    Ok(result.into_boxed_slice())
}

fn local_line_len_real<R: FftReal>(local: &MixedRealLocal<R>) -> usize {
    local.line_len()
}

fn local_line_len_complex<R: FftReal>(local: &MixedComplexLocal<R>) -> usize {
    local.line_len()
}

fn mixed_r2c_real_pencils<R: FftReal, const N: usize, const M: usize>(
    stages: &[MixedR2cStage<R, N, M>],
    boundary: usize,
) -> Result<Box<[Arc<Pencil<N, M>>]>, MixedError> {
    let mut pencils = Vec::new();
    pencils.try_reserve_exact(boundary + 1).map_err(|_| {
        MixedError::Fft(FftError::AllocationFailed {
            required: boundary + 1,
        })
    })?;
    for stage in &stages[..=boundary] {
        if !pencils
            .iter()
            .any(|p: &Arc<Pencil<N, M>>| p.same_layout(stage.input.as_ref()))
        {
            pencils.push(Arc::clone(&stage.input));
        }
    }
    Ok(pencils.into_boxed_slice())
}

fn mixed_r2c_complex_pencils<R: FftReal, const N: usize, const M: usize>(
    stages: &[MixedR2cStage<R, N, M>],
    boundary: usize,
) -> Result<Box<[Arc<Pencil<N, M>>]>, MixedError> {
    let mut pencils = Vec::new();
    pencils
        .try_reserve_exact(stages.len() - boundary)
        .map_err(|_| {
            MixedError::Fft(FftError::AllocationFailed {
                required: stages.len() - boundary,
            })
        })?;
    for stage in &stages[boundary..] {
        if !pencils
            .iter()
            .any(|p: &Arc<Pencil<N, M>>| p.same_layout(stage.output.as_ref()))
        {
            pencils.push(Arc::clone(&stage.output));
        }
    }
    Ok(pencils.into_boxed_slice())
}

fn mixed_r2c_storage_requirements<R: FftReal, const N: usize, const M: usize>(
    core: &MixedR2cCore<R, N, M>,
) -> Result<(usize, usize, usize, usize), MixedError> {
    let extra = core.extra_shape.element_count();
    let boundary = core.real_stage_index;
    let mut real_len = 0;
    for stage in &core.stages[..=boundary] {
        real_len = real_len.max(
            stage
                .input
                .local_len()
                .checked_mul(extra)
                .ok_or(MixedError::Fft(FftError::PreparationFailed))?,
        );
    }
    let mut complex_len = 0;
    for stage in &core.stages[boundary..] {
        complex_len = complex_len.max(
            stage
                .output
                .local_len()
                .checked_mul(extra)
                .ok_or(MixedError::Fft(FftError::PreparationFailed))?,
        );
    }
    let real_bytes = real_len
        .checked_mul(size_of::<R>())
        .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
    let complex_size = size_of::<Complex<R>>();
    let complex_bytes = complex_len
        .checked_mul(complex_size)
        .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
    let boundary_stride = memory_stride(
        core.stages[boundary].input.as_ref(),
        core.stages[boundary].axis,
    )?;
    let storage_bytes = if boundary_stride > 1 {
        real_len
            .max(complex_len)
            .checked_mul(complex_size)
            .ok_or(MixedError::Fft(FftError::PreparationFailed))?
    } else {
        real_bytes.max(complex_bytes)
    };
    if storage_bytes > isize::MAX as usize {
        return Err(MixedError::Fft(FftError::PreparationFailed));
    }
    let capacity = storage_bytes
        .checked_add(complex_size - 1)
        .ok_or(MixedError::Fft(FftError::PreparationFailed))?
        / complex_size;
    Ok((real_len, complex_len, storage_bytes, capacity))
}

fn ceil_log2_len(length: usize) -> f64 {
    if length <= 1 {
        0.0
    } else {
        (usize::BITS - (length - 1).leading_zeros()) as f64
    }
}

fn mixed_raw_threshold<R: FftReal, const N: usize, const M: usize>(
    stages: &[MixedR2cStage<R, N, M>],
    _extra: &ExtraShape,
    boundary: usize,
) -> Result<f64, MixedError> {
    let mut depth = 1.0;
    let mut normalization = 1.0;
    for stage in &stages[boundary + 1..] {
        let (native_len, factor, allowance) = match &stage.local {
            MixedR2cStageLocal::Complex(MixedComplexLocal::Identity) => continue,
            MixedR2cStageLocal::Complex(MixedComplexLocal::Fft(plan)) => {
                (plan.line_len(), plan.line_len() as f64, 0.0)
            }
            MixedR2cStageLocal::Complex(MixedComplexLocal::R2r(MixedR2rLocal::Transform(plan))) => {
                (
                    plan.embedding_len(),
                    plan.normalization_factor() as f64,
                    2.0,
                )
            }
            MixedR2cStageLocal::Complex(MixedComplexLocal::R2r(MixedR2rLocal::Hartley(plan))) => (
                plan.embedding_len(),
                plan.normalization_factor() as f64,
                2.0,
            ),
            MixedR2cStageLocal::Real(_) => continue,
        };
        depth += ceil_log2_len(native_len) + allowance;
        normalization *= factor;
        if !depth.is_finite() || depth <= 0.0 || !normalization.is_finite() || normalization <= 0.0
        {
            return Err(MixedError::Fft(FftError::PreparationFailed));
        }
    }
    let absolute = 128.0
        * <R as crate::private::Sealed>::pencil_fft_min_subnormal_f64()
        * depth
        * normalization;
    if absolute.is_finite() && absolute > 0.0 {
        Ok(absolute)
    } else {
        Err(MixedError::Fft(FftError::PreparationFailed))
    }
}

fn mixed_real_prefix_forward<R: FftReal, const N: usize, const M: usize>(
    local: &MixedRealLocal<R>,
    pencil: &Pencil<N, M>,
    axis: usize,
    source: &[R],
    destination: &mut [R],
    embedding: &mut [Complex<R>],
    scratch: &mut [Complex<R>],
    line: &mut [R],
) -> Result<(), MixedError> {
    match local {
        MixedRealLocal::Identity => {
            if source.len() != destination.len() {
                return Err(MixedError::Fft(FftError::PreparationFailed));
            }
            destination.copy_from_slice(source);
            Ok(())
        }
        MixedRealLocal::R2r(plan) => mixed_r2r_forward(
            plan,
            pencil,
            axis,
            source,
            destination,
            embedding,
            scratch,
            line,
        ),
        MixedRealLocal::Rfft(_) => Err(MixedError::InvalidGraph),
    }
}

fn mixed_real_prefix_forward_in_place<R: FftReal, const N: usize, const M: usize>(
    local: &MixedRealLocal<R>,
    pencil: &Pencil<N, M>,
    axis: usize,
    data: &mut [R],
    embedding: &mut [Complex<R>],
    scratch: &mut [Complex<R>],
    line: &mut [R],
) -> Result<(), MixedError> {
    match local {
        MixedRealLocal::Identity => Ok(()),
        MixedRealLocal::R2r(plan) => {
            mixed_r2r_forward_in_place(plan, pencil, axis, data, embedding, scratch, line)
        }
        MixedRealLocal::Rfft(_) => Err(MixedError::InvalidGraph),
    }
}

fn mixed_real_prefix_reverse<R: FftReal, const N: usize, const M: usize>(
    local: &MixedRealLocal<R>,
    pencil: &Pencil<N, M>,
    axis: usize,
    source: &[R],
    destination: &mut [R],
    embedding: &mut [Complex<R>],
    scratch: &mut [Complex<R>],
    line: &mut [R],
    normalize: bool,
) -> Result<(), MixedError> {
    match local {
        MixedRealLocal::Identity => {
            if source.len() != destination.len() {
                return Err(MixedError::Fft(FftError::PreparationFailed));
            }
            destination.copy_from_slice(source);
            Ok(())
        }
        MixedRealLocal::R2r(plan) => mixed_r2r_reverse(
            plan,
            pencil,
            axis,
            source,
            destination,
            embedding,
            scratch,
            line,
            normalize,
        ),
        MixedRealLocal::Rfft(_) => Err(MixedError::InvalidGraph),
    }
}

fn mixed_real_prefix_reverse_in_place<R: FftReal, const N: usize, const M: usize>(
    local: &MixedRealLocal<R>,
    pencil: &Pencil<N, M>,
    axis: usize,
    data: &mut [R],
    embedding: &mut [Complex<R>],
    scratch: &mut [Complex<R>],
    line: &mut [R],
    normalize: bool,
) -> Result<(), MixedError> {
    match local {
        MixedRealLocal::Identity => Ok(()),
        MixedRealLocal::R2r(plan) => mixed_r2r_reverse_in_place(
            plan, pencil, axis, data, embedding, scratch, line, normalize,
        ),
        MixedRealLocal::Rfft(_) => Err(MixedError::InvalidGraph),
    }
}

fn mixed_rfft_forward<R: FftReal, const N: usize, const M: usize>(
    plan: &LocalR2cPlan<R>,
    pencil: &Pencil<N, M>,
    axis: usize,
    source: &[R],
    destination: &mut [Complex<R>],
    real_source_line: &mut [R],
    real_line: &mut [R],
    complex_line: &mut [Complex<R>],
    scratch: &mut [Complex<R>],
) -> Result<(), MixedError>
where
    Complex<R>: Equivalence,
{
    let stride = memory_stride(pencil, axis)?;
    if stride <= 1 {
        return plan
            .forward(source, destination, real_line, scratch)
            .map_err(MixedError::LocalR2c);
    }
    if real_source_line.len() < plan.real_len() || complex_line.len() < plan.complex_len() {
        return Err(MixedError::Fft(FftError::WorkspaceTooSmall {
            kind: "real/complex line",
            required: plan.real_len().max(plan.complex_len()),
            actual: real_source_line.len().min(complex_line.len()),
        }));
    }
    let count = strided_line_count(source.len(), plan.real_len(), stride)?;
    let destination_len = count
        .checked_mul(plan.complex_len())
        .and_then(|value| value.checked_mul(stride))
        .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
    if destination.len() != destination_len {
        return Err(MixedError::Fft(FftError::PreparationFailed));
    }
    let source_block = plan
        .real_len()
        .checked_mul(stride)
        .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
    let destination_block = plan
        .complex_len()
        .checked_mul(stride)
        .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
    for outer in 0..count {
        let source_base = outer
            .checked_mul(source_block)
            .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
        let destination_base = outer
            .checked_mul(destination_block)
            .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
        for inner in 0..stride {
            for k in 0..plan.real_len() {
                real_source_line[k] = source[source_base + k * stride + inner];
            }
            plan.forward(
                &real_source_line[..plan.real_len()],
                &mut complex_line[..plan.complex_len()],
                real_line,
                scratch,
            )
            .map_err(MixedError::LocalR2c)?;
            for k in 0..plan.complex_len() {
                destination[destination_base + k * stride + inner] = complex_line[k];
            }
        }
    }
    Ok(())
}

fn mixed_rfft_reverse<R: FftReal, const N: usize, const M: usize>(
    plan: &LocalR2cPlan<R>,
    pencil: &Pencil<N, M>,
    axis: usize,
    source: &[Complex<R>],
    destination: &mut [R],
    complex_source_line: &mut [Complex<R>],
    complex_line: &mut [Complex<R>],
    real_line: &mut [R],
    scratch: &mut [Complex<R>],
    normalize: bool,
) -> Result<(), MixedError>
where
    Complex<R>: Equivalence,
{
    let stride = memory_stride(pencil, axis)?;
    if stride <= 1 {
        return if normalize {
            plan.inverse(source, destination, complex_line, scratch)
        } else {
            plan.backward(source, destination, complex_line, scratch)
        }
        .map_err(MixedError::LocalR2c);
    }
    if complex_source_line.len() < plan.complex_len() || real_line.len() < plan.real_len() {
        return Err(MixedError::Fft(FftError::WorkspaceTooSmall {
            kind: "complex/real line",
            required: plan.real_len().max(plan.complex_len()),
            actual: complex_source_line.len().min(real_line.len()),
        }));
    }
    let count = strided_line_count(source.len(), plan.complex_len(), stride)?;
    let source_block = plan
        .complex_len()
        .checked_mul(stride)
        .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
    let destination_block = plan
        .real_len()
        .checked_mul(stride)
        .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
    let expected = count
        .checked_mul(destination_block)
        .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
    if destination.len() != expected {
        return Err(MixedError::Fft(FftError::PreparationFailed));
    }
    for outer in 0..count {
        let source_base = outer
            .checked_mul(source_block)
            .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
        let destination_base = outer
            .checked_mul(destination_block)
            .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
        for inner in 0..stride {
            for k in 0..plan.complex_len() {
                complex_source_line[k] = source[source_base + k * stride + inner];
            }
            let result = if normalize {
                plan.inverse(
                    &complex_source_line[..plan.complex_len()],
                    &mut real_line[..plan.real_len()],
                    complex_line,
                    scratch,
                )
            } else {
                plan.backward(
                    &complex_source_line[..plan.complex_len()],
                    &mut real_line[..plan.real_len()],
                    complex_line,
                    scratch,
                )
            };
            result.map_err(MixedError::LocalR2c)?;
            for k in 0..plan.real_len() {
                destination[destination_base + k * stride + inner] = real_line[k];
            }
        }
    }
    Ok(())
}

fn mixed_r2c_complex_forward_tail<R: FftReal, const N: usize, const M: usize>(
    core: &MixedR2cCore<R, N, M>,
    intermediate: &mut ManyPencilArray<Complex<R>, N, M>,
    transpose: &mut TransposeWorkspace<Complex<R>>,
    destination: &mut PencilArray<Complex<R>, N, M>,
    embedding: &mut [Complex<R>],
    scratch: &mut [Complex<R>],
    line: &mut [Complex<R>],
    mut report: Option<&mut TransformTiming<N>>,
) -> Result<(), MixedError>
where
    Complex<R>: Equivalence,
{
    let communicator = core.stages[0].input.topology().communicator();
    let boundary = core.real_stage_index;
    let last = core.stages.len() - 1;
    if boundary == last {
        let active = intermediate.active_view().map_err(FftError::Array)?;
        let mut target = destination.view_mut();
        if active.as_slice().len() != target.as_slice().len() {
            return Err(MixedError::Fft(FftError::PreparationFailed));
        }
        target.as_mut_slice().copy_from_slice(active.as_slice());
        return Ok(());
    }
    for index in boundary..core.transitions.len() {
        execute_mixed_transition_timed(
            communicator,
            &core.transitions[index].forward,
            intermediate,
            transpose,
            &mut report,
            index,
        )?;
        let stage = &core.stages[index + 1];
        let fft_started = Instant::now();
        let stage_result = if index + 1 == last {
            (|| {
                let active = intermediate.active_view().map_err(FftError::Array)?;
                let mut target = destination.view_mut();
                let MixedR2cStageLocal::Complex(local) = &stage.local else {
                    return Err(MixedError::Fft(FftError::PreparationFailed));
                };
                mixed_complex_forward(
                    local,
                    stage.output.as_ref(),
                    stage.axis,
                    active.as_slice(),
                    target.as_mut_slice(),
                    embedding,
                    scratch,
                    line,
                )
            })()
        } else {
            (|| {
                let mut active = intermediate.active_view_mut().map_err(FftError::Array)?;
                let MixedR2cStageLocal::Complex(local) = &stage.local else {
                    return Err(MixedError::Fft(FftError::PreparationFailed));
                };
                mixed_complex_forward_in_place(
                    local,
                    stage.output.as_ref(),
                    stage.axis,
                    active.as_mut_slice(),
                    embedding,
                    scratch,
                    line,
                )
            })()
        };
        super::record_fft_timing(&mut report, index + 1, fft_started);
        agree_result(communicator, stage_result)?;
    }
    Ok(())
}

fn mixed_r2c_complex_reverse_tail<R: FftReal, const N: usize, const M: usize>(
    core: &MixedR2cCore<R, N, M>,
    source: &PencilArray<Complex<R>, N, M>,
    intermediate: &mut ManyPencilArray<Complex<R>, N, M>,
    transpose: &mut TransposeWorkspace<Complex<R>>,
    embedding: &mut [Complex<R>],
    scratch: &mut [Complex<R>],
    line: &mut [Complex<R>],
    normalize: bool,
    mut report: Option<&mut TransformTiming<N>>,
) -> Result<(), MixedError>
where
    Complex<R>: Equivalence,
{
    let communicator = core.stages[0].input.topology().communicator();
    let boundary = core.real_stage_index;
    let last = core.stages.len() - 1;
    let source_view = source.view();
    let fft_started = Instant::now();
    let overwrite_result = intermediate
        .overwrite_with(core.stages[last].output.as_ref(), |mut target| {
            if boundary == last {
                if target.as_mut_slice().len() != source_view.as_slice().len() {
                    return Err(MixedError::Fft(FftError::PreparationFailed));
                }
                target
                    .as_mut_slice()
                    .copy_from_slice(source_view.as_slice());
                return Ok(());
            }
            let MixedR2cStageLocal::Complex(local) = &core.stages[last].local else {
                return Err(MixedError::Fft(FftError::PreparationFailed));
            };
            mixed_complex_reverse(
                local,
                core.stages[last].input.as_ref(),
                core.stages[last].axis,
                source_view.as_slice(),
                target.as_mut_slice(),
                embedding,
                scratch,
                line,
                normalize,
            )
        })
        .map_err(map_overwrite);
    if boundary != last {
        super::record_fft_timing(&mut report, last, fft_started);
    }
    agree_result(communicator, overwrite_result)?;
    if boundary == last {
        return Ok(());
    }
    for index in (boundary..core.transitions.len()).rev() {
        execute_mixed_transition_timed(
            communicator,
            &core.transitions[index].backward,
            intermediate,
            transpose,
            &mut report,
            index,
        )?;
        if index != boundary {
            let fft_started = Instant::now();
            let stage_result = (|| {
                let stage = &core.stages[index];
                let MixedR2cStageLocal::Complex(local) = &stage.local else {
                    return Err(MixedError::Fft(FftError::PreparationFailed));
                };
                let mut active = intermediate.active_view_mut().map_err(FftError::Array)?;
                mixed_complex_reverse_in_place(
                    local,
                    stage.input.as_ref(),
                    stage.axis,
                    active.as_mut_slice(),
                    embedding,
                    scratch,
                    line,
                    normalize,
                )
            })();
            super::record_fft_timing(&mut report, index, fft_started);
            agree_result(communicator, stage_result)?;
        }
    }
    Ok(())
}

fn execute_mixed_r2c_forward<R: FftReal, const N: usize, const M: usize>(
    core: &Arc<MixedR2cCore<R, N, M>>,
    source: &PencilArray<R, N, M>,
    destination: &mut PencilArray<Complex<R>, N, M>,
    workspace: &mut MixedR2cWorkspace<R, N, M>,
    mut report: Option<&mut TransformTiming<N>>,
) -> Result<(), MixedError>
where
    Complex<R>: Equivalence,
{
    let communicator = core.stages[0].input.topology().communicator();
    let boundary = core.real_stage_index;
    let source_view = source.view();
    if boundary == 0 {
        let stage = &core.stages[0];
        let MixedR2cStageLocal::Real(MixedRealLocal::Rfft(plan)) = &stage.local else {
            return Err(MixedError::Fft(FftError::PreparationFailed));
        };
        let fft_started = Instant::now();
        let overwrite_result = workspace
            .intermediate
            .overwrite_with(stage.output.as_ref(), |mut target| {
                mixed_rfft_forward(
                    plan,
                    stage.input.as_ref(),
                    stage.axis,
                    source_view.as_slice(),
                    target.as_mut_slice(),
                    &mut workspace.real_source_line,
                    &mut workspace.real_line,
                    &mut workspace.complex_line,
                    &mut workspace.fft_scratch,
                )
            })
            .map_err(map_overwrite);
        super::record_fft_timing(&mut report, 0, fft_started);
        agree_result(communicator, overwrite_result)?;
    } else {
        let real = workspace
            .real_intermediate
            .as_mut()
            .ok_or(MixedError::Fft(FftError::WorkspaceMismatch))?;
        let stage0 = &core.stages[0];
        let MixedR2cStageLocal::Real(local0) = &stage0.local else {
            return Err(MixedError::Fft(FftError::PreparationFailed));
        };
        let fft_started = Instant::now();
        let overwrite_result = real
            .overwrite_with(stage0.output.as_ref(), |mut target| {
                mixed_real_prefix_forward(
                    local0,
                    stage0.output.as_ref(),
                    stage0.axis,
                    source_view.as_slice(),
                    target.as_mut_slice(),
                    &mut workspace.embedding_line,
                    &mut workspace.fft_scratch,
                    &mut workspace.real_strided_line,
                )
            })
            .map_err(map_overwrite);
        super::record_fft_timing(&mut report, 0, fft_started);
        agree_result(communicator, overwrite_result)?;
        for index in 0..boundary {
            let real_transpose = workspace
                .real_transpose
                .as_mut()
                .ok_or(MixedError::Fft(FftError::WorkspaceMismatch))?;
            execute_mixed_transition_timed(
                communicator,
                &core.transitions[index].forward,
                real,
                real_transpose,
                &mut report,
                index,
            )?;
            if index + 1 == boundary {
                let active = real.active_view().map_err(FftError::Array)?;
                let stage = &core.stages[boundary];
                let MixedR2cStageLocal::Real(MixedRealLocal::Rfft(plan)) = &stage.local else {
                    return Err(MixedError::Fft(FftError::PreparationFailed));
                };
                let fft_started = Instant::now();
                let overwrite_result = workspace
                    .intermediate
                    .overwrite_with(stage.output.as_ref(), |mut target| {
                        mixed_rfft_forward(
                            plan,
                            stage.input.as_ref(),
                            stage.axis,
                            active.as_slice(),
                            target.as_mut_slice(),
                            &mut workspace.real_source_line,
                            &mut workspace.real_line,
                            &mut workspace.complex_line,
                            &mut workspace.fft_scratch,
                        )
                    })
                    .map_err(map_overwrite);
                super::record_fft_timing(&mut report, boundary, fft_started);
                agree_result(communicator, overwrite_result)?;
            } else {
                let stage = &core.stages[index + 1];
                let MixedR2cStageLocal::Real(local) = &stage.local else {
                    return Err(MixedError::Fft(FftError::PreparationFailed));
                };
                let fft_started = Instant::now();
                let stage_result = (|| {
                    let mut active = real.active_view_mut().map_err(FftError::Array)?;
                    mixed_real_prefix_forward_in_place(
                        local,
                        stage.output.as_ref(),
                        stage.axis,
                        active.as_mut_slice(),
                        &mut workspace.embedding_line,
                        &mut workspace.fft_scratch,
                        &mut workspace.real_strided_line,
                    )
                })();
                super::record_fft_timing(&mut report, index + 1, fft_started);
                agree_result(communicator, stage_result)?;
            }
        }
    }
    mixed_r2c_complex_forward_tail(
        core,
        &mut workspace.intermediate,
        &mut workspace.transpose,
        destination,
        &mut workspace.embedding_line,
        &mut workspace.fft_scratch,
        &mut workspace.complex_strided_line,
        report,
    )
}

fn execute_mixed_r2c_reverse<R: FftReal, const N: usize, const M: usize>(
    core: &Arc<MixedR2cCore<R, N, M>>,
    source: &PencilArray<Complex<R>, N, M>,
    destination: &mut PencilArray<R, N, M>,
    workspace: &mut MixedR2cWorkspace<R, N, M>,
    normalize: bool,
    mut report: Option<&mut TransformTiming<N>>,
) -> Result<(), MixedError>
where
    Complex<R>: Equivalence,
{
    let communicator = core.stages[0].input.topology().communicator();
    let boundary = core.real_stage_index;
    mixed_r2c_complex_reverse_tail(
        core,
        source,
        &mut workspace.intermediate,
        &mut workspace.transpose,
        &mut workspace.embedding_line,
        &mut workspace.fft_scratch,
        &mut workspace.complex_strided_line,
        normalize,
        report.as_deref_mut(),
    )?;
    validate_mixed_boundary(core, &workspace.intermediate, normalize)?;
    agree_result(
        communicator,
        zero_mixed_boundary(core, &mut workspace.intermediate),
    )?;
    let stage = &core.stages[boundary];
    let MixedR2cStageLocal::Real(MixedRealLocal::Rfft(plan)) = &stage.local else {
        return Err(MixedError::Fft(FftError::PreparationFailed));
    };
    let active = workspace
        .intermediate
        .active_view()
        .map_err(FftError::Array)?;
    if boundary == 0 {
        let fft_started = Instant::now();
        let result = {
            let mut target = destination.view_mut();
            mixed_rfft_reverse(
                plan,
                stage.output.as_ref(),
                stage.axis,
                active.as_slice(),
                target.as_mut_slice(),
                &mut workspace.complex_source_line,
                &mut workspace.complex_line,
                &mut workspace.real_line,
                &mut workspace.fft_scratch,
                normalize,
            )
        };
        super::record_fft_timing(&mut report, boundary, fft_started);
        let result = agree_result(communicator, result);
        return result;
    }
    let real = workspace
        .real_intermediate
        .as_mut()
        .ok_or(MixedError::Fft(FftError::WorkspaceMismatch))?;
    let fft_started = Instant::now();
    let overwrite_result = real
        .overwrite_with(stage.input.as_ref(), |mut target| {
            mixed_rfft_reverse(
                plan,
                stage.output.as_ref(),
                stage.axis,
                active.as_slice(),
                target.as_mut_slice(),
                &mut workspace.complex_source_line,
                &mut workspace.complex_line,
                &mut workspace.real_line,
                &mut workspace.fft_scratch,
                normalize,
            )
        })
        .map_err(map_overwrite);
    super::record_fft_timing(&mut report, boundary, fft_started);
    agree_result(communicator, overwrite_result)?;
    for index in (0..boundary).rev() {
        let real_transpose = workspace
            .real_transpose
            .as_mut()
            .ok_or(MixedError::Fft(FftError::WorkspaceMismatch))?;
        execute_mixed_transition_timed(
            communicator,
            &core.transitions[index].backward,
            real,
            real_transpose,
            &mut report,
            index,
        )?;
        if index != 0 {
            let stage = &core.stages[index];
            let MixedR2cStageLocal::Real(local) = &stage.local else {
                return Err(MixedError::Fft(FftError::PreparationFailed));
            };
            let fft_started = Instant::now();
            let stage_result = (|| {
                let mut active = real.active_view_mut().map_err(FftError::Array)?;
                mixed_real_prefix_reverse_in_place(
                    local,
                    stage.input.as_ref(),
                    stage.axis,
                    active.as_mut_slice(),
                    &mut workspace.embedding_line,
                    &mut workspace.fft_scratch,
                    &mut workspace.real_strided_line,
                    normalize,
                )
            })();
            super::record_fft_timing(&mut report, index, fft_started);
            agree_result(communicator, stage_result)?;
        }
    }
    let active = real.active_view().map_err(FftError::Array)?;
    let mut target = destination.view_mut();
    let stage0 = &core.stages[0];
    let MixedR2cStageLocal::Real(local0) = &stage0.local else {
        return Err(MixedError::Fft(FftError::PreparationFailed));
    };
    let fft_started = Instant::now();
    let result = mixed_real_prefix_reverse(
        local0,
        stage0.input.as_ref(),
        stage0.axis,
        active.as_slice(),
        target.as_mut_slice(),
        &mut workspace.embedding_line,
        &mut workspace.fft_scratch,
        &mut workspace.real_strided_line,
        normalize,
    );
    super::record_fft_timing(&mut report, 0, fft_started);
    agree_result(communicator, result)
}

fn execute_mixed_r2c_forward_overlap<R: FftReal, const N: usize, const M: usize>(
    core: &Arc<MixedR2cCore<R, N, M>>,
    source: &PencilArray<R, N, M>,
    destination: &mut PencilArray<Complex<R>, N, M>,
    workspace: &mut MixedR2cWorkspace<R, N, M>,
) -> Result<(), FftOverlapError<MixedError>>
where
    Complex<R>: Equivalence,
{
    let comm = core.stages[0].input.topology().communicator();
    let boundary = core.real_stage_index;
    if boundary == 0 {
        let stage = &core.stages[0];
        let MixedR2cStageLocal::Real(MixedRealLocal::Rfft(plan)) = &stage.local else {
            return Err(MixedError::Fft(FftError::PreparationFailed).into());
        };
        let source_view = source.view();
        let result = workspace
            .intermediate
            .overwrite_with(stage.output.as_ref(), |mut target| {
                mixed_rfft_forward(
                    plan,
                    stage.input.as_ref(),
                    stage.axis,
                    source_view.as_slice(),
                    target.as_mut_slice(),
                    &mut workspace.real_source_line,
                    &mut workspace.real_line,
                    &mut workspace.complex_line,
                    &mut workspace.fft_scratch,
                )
            })
            .map_err(map_overwrite);
        agree_result(comm, result)?;
    } else {
        let real = workspace
            .real_intermediate
            .as_mut()
            .ok_or(MixedError::Fft(FftError::WorkspaceMismatch))?;
        let stage = &core.stages[0];
        let MixedR2cStageLocal::Real(local) = &stage.local else {
            return Err(MixedError::Fft(FftError::PreparationFailed).into());
        };
        let source_view = source.view();
        let result = real
            .overwrite_with(stage.output.as_ref(), |mut target| {
                mixed_real_prefix_forward(
                    local,
                    stage.output.as_ref(),
                    stage.axis,
                    source_view.as_slice(),
                    target.as_mut_slice(),
                    &mut workspace.embedding_line,
                    &mut workspace.fft_scratch,
                    &mut workspace.real_strided_line,
                )
            })
            .map_err(map_overwrite);
        agree_result(comm, result)?;
        for index in 0..boundary {
            if matches!(
                core.transitions[index].forward,
                super::C2cTransition::Identity | super::C2cTransition::Local(_)
            ) {
                super::execute_transition(
                    &core.transitions[index].forward,
                    real,
                    workspace
                        .real_transpose
                        .as_mut()
                        .ok_or(MixedError::Fft(FftError::WorkspaceMismatch))?,
                )
                .map_err(MixedError::Fft)?;
                let stage = &core.stages[index + 1];
                if index + 1 == boundary {
                    let MixedR2cStageLocal::Real(MixedRealLocal::Rfft(rfft)) = &stage.local else {
                        return Err(MixedError::Fft(FftError::PreparationFailed).into());
                    };
                    let active = real.active_view().map_err(FftError::Array)?;
                    workspace
                        .intermediate
                        .overwrite_with(stage.output.as_ref(), |mut out| {
                            mixed_rfft_forward(
                                rfft,
                                stage.input.as_ref(),
                                stage.axis,
                                active.as_slice(),
                                out.as_mut_slice(),
                                &mut workspace.real_source_line,
                                &mut workspace.real_line,
                                &mut workspace.complex_line,
                                &mut workspace.fft_scratch,
                            )
                        })
                        .map_err(map_overwrite)?;
                } else {
                    let MixedR2cStageLocal::Real(local) = &stage.local else {
                        return Err(MixedError::Fft(FftError::PreparationFailed).into());
                    };
                    let mut active = real.active_view_mut().map_err(FftError::Array)?;
                    mixed_real_prefix_forward_in_place(
                        local,
                        stage.output.as_ref(),
                        stage.axis,
                        active.as_mut_slice(),
                        &mut workspace.embedding_line,
                        &mut workspace.fft_scratch,
                        &mut workspace.real_strided_line,
                    )?;
                }
                continue;
            }
            let super::C2cTransition::PointToPoint(plan) = &core.transitions[index].forward else {
                return Err(FftOverlapError::UnsupportedTransport);
            };
            if index + 1 == boundary {
                let stage = &core.stages[boundary];
                let MixedR2cStageLocal::Real(MixedRealLocal::Rfft(rfft)) = &stage.local else {
                    return Err(MixedError::Fft(FftError::PreparationFailed).into());
                };
                let callback = |data: &mut [R]| {
                    workspace
                        .intermediate
                        .overwrite_with(stage.output.as_ref(), |mut out| {
                            mixed_rfft_forward(
                                rfft,
                                stage.input.as_ref(),
                                stage.axis,
                                data,
                                out.as_mut_slice(),
                                &mut workspace.real_source_line,
                                &mut workspace.real_line,
                                &mut workspace.complex_line,
                                &mut workspace.fft_scratch,
                            )
                        })
                        .map_err(map_overwrite)?;
                    Ok::<(), MixedError>(())
                };
                plan.execute_in_place_with_callback(
                    real,
                    workspace
                        .real_transpose
                        .as_mut()
                        .ok_or(MixedError::Fft(FftError::WorkspaceMismatch))?,
                    callback,
                )
                .map_err(map_mixed_overlap)?;
            } else {
                let stage = &core.stages[index + 1];
                let MixedR2cStageLocal::Real(local) = &stage.local else {
                    return Err(MixedError::Fft(FftError::PreparationFailed).into());
                };
                let callback = |data: &mut [R]| {
                    mixed_real_prefix_forward_in_place(
                        local,
                        stage.output.as_ref(),
                        stage.axis,
                        data,
                        &mut workspace.embedding_line,
                        &mut workspace.fft_scratch,
                        &mut workspace.real_strided_line,
                    )
                };
                plan.execute_in_place_with_callback(
                    real,
                    workspace
                        .real_transpose
                        .as_mut()
                        .ok_or(MixedError::Fft(FftError::WorkspaceMismatch))?,
                    callback,
                )
                .map_err(map_mixed_overlap)?;
            }
        }
    }
    for index in boundary..core.transitions.len() {
        let stage = &core.stages[index + 1];
        if matches!(
            core.transitions[index].forward,
            super::C2cTransition::Identity | super::C2cTransition::Local(_)
        ) {
            super::execute_transition(
                &core.transitions[index].forward,
                &mut workspace.intermediate,
                &mut workspace.transpose,
            )
            .map_err(MixedError::Fft)?;
            let mut active = workspace
                .intermediate
                .active_view_mut()
                .map_err(FftError::Array)?;
            let MixedR2cStageLocal::Complex(local) = &stage.local else {
                return Err(MixedError::Fft(FftError::PreparationFailed).into());
            };
            mixed_complex_forward_in_place(
                local,
                stage.output.as_ref(),
                stage.axis,
                active.as_mut_slice(),
                &mut workspace.embedding_line,
                &mut workspace.fft_scratch,
                &mut workspace.complex_strided_line,
            )?;
            continue;
        }
        let super::C2cTransition::PointToPoint(plan) = &core.transitions[index].forward else {
            return Err(FftOverlapError::UnsupportedTransport);
        };
        let MixedR2cStageLocal::Complex(local) = &stage.local else {
            return Err(MixedError::Fft(FftError::PreparationFailed).into());
        };
        let callback = |data: &mut [Complex<R>]| {
            mixed_complex_forward_in_place(
                local,
                stage.output.as_ref(),
                stage.axis,
                data,
                &mut workspace.embedding_line,
                &mut workspace.fft_scratch,
                &mut workspace.complex_strided_line,
            )
        };
        plan.execute_in_place_with_callback(
            &mut workspace.intermediate,
            &mut workspace.transpose,
            callback,
        )
        .map_err(map_mixed_overlap)?;
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

fn execute_mixed_r2c_reverse_overlap<R: FftReal, const N: usize, const M: usize>(
    core: &Arc<MixedR2cCore<R, N, M>>,
    source: &PencilArray<Complex<R>, N, M>,
    destination: &mut PencilArray<R, N, M>,
    workspace: &mut MixedR2cWorkspace<R, N, M>,
    normalize: bool,
) -> Result<(), FftOverlapError<MixedError>>
where
    Complex<R>: Equivalence,
{
    let comm = core.stages[0].input.topology().communicator();
    let boundary = core.real_stage_index;
    let last = core.stages.len() - 1;
    let stage = &core.stages[last];
    let result = workspace
        .intermediate
        .overwrite_with(stage.output.as_ref(), |mut target| {
            if boundary == last {
                target.as_mut_slice().copy_from_slice(source.as_slice());
                Ok(())
            } else {
                let MixedR2cStageLocal::Complex(local) = &stage.local else {
                    return Err(MixedError::Fft(FftError::PreparationFailed));
                };
                mixed_complex_reverse(
                    local,
                    stage.input.as_ref(),
                    stage.axis,
                    source.as_slice(),
                    target.as_mut_slice(),
                    &mut workspace.embedding_line,
                    &mut workspace.fft_scratch,
                    &mut workspace.complex_strided_line,
                    normalize,
                )
            }
        })
        .map_err(map_overwrite);
    agree_result(comm, result)?;
    for index in (boundary + 1..core.transitions.len()).rev() {
        let stage = &core.stages[index];
        let MixedR2cStageLocal::Complex(local) = &stage.local else {
            return Err(MixedError::Fft(FftError::PreparationFailed).into());
        };
        let mut transform = |data: &mut [Complex<R>]| {
            mixed_complex_reverse_in_place(
                local,
                stage.input.as_ref(),
                stage.axis,
                data,
                &mut workspace.embedding_line,
                &mut workspace.fft_scratch,
                &mut workspace.complex_strided_line,
                normalize,
            )
        };
        match &core.transitions[index].backward {
            super::C2cTransition::PointToPoint(plan) => {
                plan.execute_in_place_with_callback(
                    &mut workspace.intermediate,
                    &mut workspace.transpose,
                    |data| {
                        let result = transform(data);
                        #[cfg(test)]
                        reverse_overlap_tests::callback()?;
                        result
                    },
                )
                .map_err(map_mixed_overlap)?;
            }
            transition => {
                super::execute_transition(
                    transition,
                    &mut workspace.intermediate,
                    &mut workspace.transpose,
                )
                .map_err(MixedError::Fft)?;
                let result = transform(
                    workspace
                        .intermediate
                        .active_view_mut()
                        .map_err(FftError::Array)?
                        .as_mut_slice(),
                );
                agree_result(comm, result)?;
            }
        }
    }
    let stage = &core.stages[boundary];
    let MixedR2cStageLocal::Real(MixedRealLocal::Rfft(rfft)) = &stage.local else {
        return Err(MixedError::Fft(FftError::PreparationFailed).into());
    };
    let mut convert = |data: &mut [Complex<R>]| {
        validate_mixed_boundary_data(core, Ok((stage.output.as_ref(), data)), normalize)?;
        zero_mixed_boundary_data(
            core,
            stage.output.local_len(),
            memory_stride(stage.output.as_ref(), stage.axis)?,
            data,
        )?;
        let mut transform = |out: &mut [R]| {
            mixed_rfft_reverse(
                rfft,
                stage.output.as_ref(),
                stage.axis,
                data,
                out,
                &mut workspace.complex_source_line,
                &mut workspace.complex_line,
                &mut workspace.real_line,
                &mut workspace.fft_scratch,
                normalize,
            )
        };
        if boundary == 0 {
            transform(destination.view_mut().as_mut_slice())
        } else {
            workspace
                .real_intermediate
                .as_mut()
                .ok_or(MixedError::Fft(FftError::WorkspaceMismatch))?
                .overwrite_with(stage.input.as_ref(), |mut out| {
                    transform(out.as_mut_slice())
                })
                .map_err(map_overwrite)
        }
    };
    if boundary < last {
        match &core.transitions[boundary].backward {
            super::C2cTransition::PointToPoint(plan) => {
                plan.execute_in_place_with_callback(
                    &mut workspace.intermediate,
                    &mut workspace.transpose,
                    |data| {
                        let result = convert(data);
                        #[cfg(test)]
                        reverse_overlap_tests::callback()?;
                        result
                    },
                )
                .map_err(map_mixed_overlap)?;
            }
            transition => {
                super::execute_transition(
                    transition,
                    &mut workspace.intermediate,
                    &mut workspace.transpose,
                )
                .map_err(MixedError::Fft)?;
                let result = convert(
                    workspace
                        .intermediate
                        .active_view_mut()
                        .map_err(FftError::Array)?
                        .as_mut_slice(),
                );
                agree_result(comm, result)?;
            }
        }
    } else {
        let result = convert(
            workspace
                .intermediate
                .active_view_mut()
                .map_err(FftError::Array)?
                .as_mut_slice(),
        );
        agree_result(comm, result)?;
    }
    if boundary == 0 {
        return Ok(());
    }
    let real = workspace
        .real_intermediate
        .as_mut()
        .ok_or(MixedError::Fft(FftError::WorkspaceMismatch))?;
    let transpose = workspace
        .real_transpose
        .as_mut()
        .ok_or(MixedError::Fft(FftError::WorkspaceMismatch))?;
    for index in (0..boundary).rev() {
        let stage = &core.stages[index];
        let MixedR2cStageLocal::Real(local) = &stage.local else {
            return Err(MixedError::Fft(FftError::PreparationFailed).into());
        };
        let mut transform = |data: &mut [R]| {
            mixed_real_prefix_reverse_in_place(
                local,
                stage.input.as_ref(),
                stage.axis,
                data,
                &mut workspace.embedding_line,
                &mut workspace.fft_scratch,
                &mut workspace.real_strided_line,
                normalize,
            )
        };
        match &core.transitions[index].backward {
            super::C2cTransition::PointToPoint(plan) => {
                plan.execute_in_place_with_callback(real, transpose, |data| {
                    let result = transform(data);
                    #[cfg(test)]
                    reverse_overlap_tests::callback()?;
                    result
                })
                .map_err(map_mixed_overlap)?;
            }
            transition => {
                super::execute_transition(transition, real, transpose).map_err(MixedError::Fft)?;
                let result = transform(
                    real.active_view_mut()
                        .map_err(FftError::Array)?
                        .as_mut_slice(),
                );
                agree_result(comm, result)?;
            }
        }
    }
    destination
        .view_mut()
        .as_mut_slice()
        .copy_from_slice(real.active_view().map_err(FftError::Array)?.as_slice());
    Ok(())
}

fn validate_mixed_r2c_common<R: FftReal, const N: usize, const M: usize>(
    core: &Arc<MixedR2cCore<R, N, M>>,
    workspace: &MixedR2cWorkspace<R, N, M>,
) -> Result<(), MixedError> {
    if !Arc::ptr_eq(&workspace.core, core) {
        return Err(MixedError::Fft(FftError::WorkspaceMismatch));
    }
    validate_workspace_lengths_values(
        workspace.fft_scratch.len(),
        core.fft_scratch_len,
        workspace.transpose.send_len(),
        core.transpose_send_len,
        workspace.transpose.receive_len(),
        core.transpose_receive_len,
    )?;
    if workspace.intermediate.extra_shape() != &core.extra_shape {
        return Err(MixedError::Fft(FftError::WorkspaceMismatch));
    }
    let complex_active = workspace
        .intermediate
        .active_pencil()
        .map_err(FftError::Array)?;
    validate_exact_pencil_registry(
        workspace.intermediate.pencils(),
        &core.complex_pencils,
        complex_active,
        workspace.intermediate.storage_len(),
        workspace.intermediate.extra_shape(),
    )
    .map_err(MixedError::Fft)?;
    if core.real_stage_index > 0 {
        let real_intermediate = workspace
            .real_intermediate
            .as_ref()
            .ok_or(MixedError::Fft(FftError::WorkspaceMismatch))?;
        if real_intermediate.extra_shape() != &core.extra_shape {
            return Err(MixedError::Fft(FftError::WorkspaceMismatch));
        }
        let real_active = real_intermediate.active_pencil().map_err(FftError::Array)?;
        validate_exact_pencil_registry(
            real_intermediate.pencils(),
            &core.real_pencils,
            real_active,
            real_intermediate.storage_len(),
            real_intermediate.extra_shape(),
        )
        .map_err(MixedError::Fft)?;
        let real_transpose = workspace
            .real_transpose
            .as_ref()
            .ok_or(MixedError::Fft(FftError::WorkspaceMismatch))?;
        validate_workspace_lengths_values(
            workspace.fft_scratch.len(),
            core.fft_scratch_len,
            real_transpose.send_len(),
            core.real_transpose_send_len,
            real_transpose.receive_len(),
            core.real_transpose_receive_len,
        )?;
    } else if workspace.real_intermediate.is_some() || workspace.real_transpose.is_some() {
        return Err(MixedError::Fft(FftError::WorkspaceMismatch));
    }
    if workspace.embedding_line.len() < core.embedding_len {
        return Err(MixedError::Fft(FftError::WorkspaceTooSmall {
            kind: "complex embedding",
            required: core.embedding_len,
            actual: workspace.embedding_line.len(),
        }));
    }
    for (actual, required, kind) in [
        (
            workspace.real_source_line.len(),
            core.real_len,
            "real source line",
        ),
        (
            workspace.real_line.len(),
            core.real_line_len.max(core.real_len),
            "real line",
        ),
        (
            workspace.real_strided_line.len(),
            core.real_line_len,
            "real line",
        ),
        (
            workspace.complex_source_line.len(),
            core.complex_len,
            "complex source line",
        ),
        (
            workspace.complex_line.len(),
            core.complex_line_len.max(core.complex_len),
            "complex line",
        ),
        (
            workspace.complex_strided_line.len(),
            core.complex_line_len,
            "complex line",
        ),
    ] {
        if actual < required {
            return Err(MixedError::Fft(FftError::WorkspaceTooSmall {
                kind,
                required,
                actual,
            }));
        }
    }
    Ok(())
}

fn validate_mixed_r2c_forward<R: FftReal, const N: usize, const M: usize>(
    core: &Arc<MixedR2cCore<R, N, M>>,
    source: &PencilArray<R, N, M>,
    destination: &PencilArray<Complex<R>, N, M>,
    workspace: &MixedR2cWorkspace<R, N, M>,
) -> Result<(), MixedError> {
    if (core.strict_array_identity && !Arc::ptr_eq(source.pencil(), &core.stages[0].input))
        || !source.pencil().same_layout(core.stages[0].input.as_ref())
    {
        return Err(MixedError::Fft(FftError::InputLayoutMismatch));
    }
    if (core.strict_array_identity
        && !Arc::ptr_eq(
            destination.pencil(),
            &core.stages[core.stages.len() - 1].output,
        ))
        || !destination
            .pencil()
            .same_layout(core.stages[core.stages.len() - 1].output.as_ref())
    {
        return Err(MixedError::Fft(FftError::OutputLayoutMismatch));
    }
    if source.extra_shape() != &core.extra_shape || destination.extra_shape() != &core.extra_shape {
        return Err(MixedError::Fft(FftError::ExtraShapeMismatch));
    }
    validate_mixed_r2c_common(core, workspace)
}

fn validate_mixed_r2c_reverse<R: FftReal, const N: usize, const M: usize>(
    core: &Arc<MixedR2cCore<R, N, M>>,
    source: &PencilArray<Complex<R>, N, M>,
    destination: &PencilArray<R, N, M>,
    workspace: &MixedR2cWorkspace<R, N, M>,
) -> Result<(), MixedError> {
    if (core.strict_array_identity
        && !Arc::ptr_eq(source.pencil(), &core.stages[core.stages.len() - 1].output))
        || !source
            .pencil()
            .same_layout(core.stages[core.stages.len() - 1].output.as_ref())
    {
        return Err(MixedError::Fft(FftError::InputLayoutMismatch));
    }
    if (core.strict_array_identity && !Arc::ptr_eq(destination.pencil(), &core.stages[0].input))
        || !destination
            .pencil()
            .same_layout(core.stages[0].input.as_ref())
    {
        return Err(MixedError::Fft(FftError::OutputLayoutMismatch));
    }
    if source.extra_shape() != &core.extra_shape || destination.extra_shape() != &core.extra_shape {
        return Err(MixedError::Fft(FftError::ExtraShapeMismatch));
    }
    validate_mixed_r2c_common(core, workspace)
}

fn mixed_endpoint_policy<R: FftReal, const N: usize, const M: usize>(
    core: &MixedR2cCore<R, N, M>,
) -> Result<(f64, f64), MixedError> {
    let mut depth = 1.0;
    let mut normalization = 1.0;
    for stage in &core.stages[core.real_stage_index + 1..] {
        let (native_len, factor, allowance) = match &stage.local {
            MixedR2cStageLocal::Complex(MixedComplexLocal::Identity) => continue,
            MixedR2cStageLocal::Complex(MixedComplexLocal::Fft(plan)) => {
                (plan.line_len(), plan.line_len() as f64, 0.0)
            }
            MixedR2cStageLocal::Complex(MixedComplexLocal::R2r(MixedR2rLocal::Transform(plan))) => {
                (
                    plan.embedding_len(),
                    plan.normalization_factor() as f64,
                    2.0,
                )
            }
            MixedR2cStageLocal::Complex(MixedComplexLocal::R2r(MixedR2rLocal::Hartley(plan))) => (
                plan.embedding_len(),
                plan.normalization_factor() as f64,
                2.0,
            ),
            MixedR2cStageLocal::Real(_) => continue,
        };
        depth += ceil_log2_len(native_len) + allowance;
        normalization *= factor;
    }
    if !depth.is_finite() || depth <= 0.0 || !normalization.is_finite() || normalization <= 0.0 {
        return Err(MixedError::Fft(FftError::PreparationFailed));
    }
    Ok((depth, normalization))
}

fn validate_mixed_boundary<R: FftReal, const N: usize, const M: usize>(
    core: &MixedR2cCore<R, N, M>,
    intermediate: &ManyPencilArray<Complex<R>, N, M>,
    normalize: bool,
) -> Result<(), MixedError>
where
    Complex<R>: Equivalence,
{
    match intermediate.active_view() {
        Ok(view) => {
            validate_mixed_boundary_data(core, Ok((view.pencil(), view.as_slice())), normalize)
        }
        Err(error) => validate_mixed_boundary_data(
            core,
            Err(MixedError::Fft(FftError::Array(error))),
            normalize,
        ),
    }
}

fn validate_mixed_boundary_data<R: FftReal, const N: usize, const M: usize>(
    core: &MixedR2cCore<R, N, M>,
    input: Result<(&Pencil<N, M>, &[Complex<R>]), MixedError>,
    normalize: bool,
) -> Result<(), MixedError>
where
    Complex<R>: Equivalence,
{
    let communicator = core.stages[0].input.topology().communicator();
    let preparation = (|| {
        let (pencil, data) = input?;
        let local_len = pencil.local_len();
        let stride = memory_stride(pencil, core.stages[core.real_stage_index].axis)?;
        let line_count = strided_line_count(local_len, core.complex_len, stride)?;
        let line_block = core
            .complex_len
            .checked_mul(stride)
            .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
        let (depth, _normalization) = mixed_endpoint_policy(core)?;
        let relative = 128.0 * <R as crate::private::Sealed>::pencil_fft_epsilon_f64() * depth;
        let absolute = if normalize {
            128.0 * <R as crate::private::Sealed>::pencil_fft_min_subnormal_f64() * depth
        } else {
            core.raw_absolute_threshold
        };
        let plane_count = if core.real_len % 2 == 0 { 2 } else { 1 };
        let batch_count = core.extra_shape.element_count();
        let mut local_maxima = Vec::new();
        local_maxima.try_reserve_exact(batch_count).map_err(|_| {
            MixedError::Fft(FftError::AllocationFailed {
                required: batch_count,
            })
        })?;
        let mut invalid = !relative.is_finite() || !absolute.is_finite();
        for batch in 0..batch_count {
            let start = batch
                .checked_mul(local_len)
                .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
            let end = start
                .checked_add(local_len)
                .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
            let values = data
                .get(start..end)
                .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
            let mut local_max = [0.0f64; 4];
            for outer in 0..line_count {
                let base = outer
                    .checked_mul(line_block)
                    .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
                for inner in 0..stride {
                    for plane in 0..plane_count {
                        let k = if plane == 0 { 0 } else { core.complex_len - 1 };
                        let index = base
                            .checked_add(
                                k.checked_mul(stride)
                                    .ok_or(MixedError::Fft(FftError::PreparationFailed))?,
                            )
                            .and_then(|value| value.checked_add(inner))
                            .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
                        let value = values
                            .get(index)
                            .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
                        let re = <R as crate::private::Sealed>::pencil_fft_as_f64(value.re);
                        let im = <R as crate::private::Sealed>::pencil_fft_as_f64(value.im);
                        if !re.is_finite() || !im.is_finite() {
                            invalid = true;
                            continue;
                        }
                        let slot = plane * 2;
                        local_max[slot] = local_max[slot].max(re.abs().max(im.abs()));
                        local_max[slot + 1] = local_max[slot + 1].max(im.abs());
                    }
                }
            }
            local_maxima.push(local_max);
        }
        Ok::<_, MixedError>((
            data,
            local_len,
            stride,
            line_count,
            line_block,
            plane_count,
            relative,
            absolute,
            local_maxima,
            invalid,
        ))
    })();
    let (
        data,
        local_len,
        stride,
        line_count,
        line_block,
        plane_count,
        relative,
        absolute,
        local_maxima,
        mut invalid,
    ) = agree_result(communicator, preparation)?;

    for (batch, local_max) in local_maxima.into_iter().enumerate() {
        let mut global_max = [0.0f64; 4];
        communicator.all_reduce_into(&local_max, &mut global_max, SystemOperation::max());
        let mut local_sum = [0.0f64; 4];
        let start = batch
            .checked_mul(local_len)
            .expect("validated mixed boundary batch offset");
        let end = start
            .checked_add(local_len)
            .expect("validated mixed boundary batch end");
        let values = &data[start..end];
        for outer in 0..line_count {
            let base = outer
                .checked_mul(line_block)
                .expect("validated mixed boundary line offset");
            for inner in 0..stride {
                for plane in 0..plane_count {
                    let k = if plane == 0 { 0 } else { core.complex_len - 1 };
                    let index = base
                        .checked_add(
                            k.checked_mul(stride)
                                .expect("validated mixed boundary plane offset"),
                        )
                        .and_then(|value| value.checked_add(inner))
                        .expect("validated mixed boundary endpoint index");
                    let value = values[index];
                    let re = <R as crate::private::Sealed>::pencil_fft_as_f64(value.re);
                    let im = <R as crate::private::Sealed>::pencil_fft_as_f64(value.im);
                    if !re.is_finite() || !im.is_finite() {
                        continue;
                    }
                    let slot = plane * 2;
                    let scale = global_max[slot];
                    if scale != 0.0 && scale.is_finite() {
                        let re = re / scale;
                        let im = im / scale;
                        local_sum[slot] += re * re;
                        local_sum[slot + 1] += im * im;
                    }
                }
            }
        }
        let mut global_sum = [0.0f64; 4];
        communicator.all_reduce_into(&local_sum, &mut global_sum, SystemOperation::sum());
        if global_max[..plane_count * 2]
            .iter()
            .any(|value| !value.is_finite())
            || global_sum[..plane_count * 2]
                .iter()
                .any(|value| !value.is_finite())
        {
            invalid = true;
        }
        for plane in 0..plane_count {
            let slot = plane * 2;
            let absolute_ok = global_max[slot + 1] <= absolute;
            let relative_ok = global_sum[slot + 1].sqrt() <= relative * global_sum[slot].sqrt();
            if !(absolute_ok || relative_ok) {
                invalid = true;
            }
        }
    }
    if !collective_valid(communicator, !invalid) {
        return Err(MixedError::InvalidSpectrum);
    }
    Ok(())
}

fn zero_mixed_boundary<R: FftReal, const N: usize, const M: usize>(
    core: &MixedR2cCore<R, N, M>,
    intermediate: &mut ManyPencilArray<Complex<R>, N, M>,
) -> Result<(), MixedError> {
    let mut view = intermediate.active_view_mut().map_err(FftError::Array)?;
    let local_len = view.pencil().local_len();
    let stride = memory_stride(view.pencil(), core.stages[core.real_stage_index].axis)?;
    zero_mixed_boundary_data(core, local_len, stride, view.as_mut_slice())
}

fn zero_mixed_boundary_data<R: FftReal, const N: usize, const M: usize>(
    core: &MixedR2cCore<R, N, M>,
    local_len: usize,
    stride: usize,
    data: &mut [Complex<R>],
) -> Result<(), MixedError> {
    let line_count = strided_line_count(local_len, core.complex_len, stride)?;
    let line_block = core
        .complex_len
        .checked_mul(stride)
        .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
    let plane_count = if core.real_len % 2 == 0 { 2 } else { 1 };
    for batch in 0..core.extra_shape.element_count() {
        let start = batch
            .checked_mul(local_len)
            .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
        let end = start
            .checked_add(local_len)
            .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
        let values = &mut data[start..end];
        for outer in 0..line_count {
            let base = outer
                .checked_mul(line_block)
                .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
            for inner in 0..stride {
                values[base + inner].im = R::zero();
                if plane_count == 2 {
                    values[base + (core.complex_len - 1) * stride + inner].im = R::zero();
                }
            }
        }
    }
    Ok(())
}

fn validate_mixed_r2c_ip<R: FftReal, const N: usize, const M: usize>(
    core: &Arc<MixedR2cCore<R, N, M>>,
    direction: Direction,
    array: &MixedR2cInPlaceArray<R, N, M>,
    workspace: &MixedR2cInPlaceWorkspace<R, N, M>,
) -> Result<(), MixedError> {
    if !Arc::ptr_eq(&array.core, core) {
        return Err(MixedError::Fft(FftError::Array(
            ArrayError::IncompatiblePencils,
        )));
    }
    if !Arc::ptr_eq(&workspace.core, core) {
        return Err(MixedError::Fft(FftError::WorkspaceMismatch));
    }
    let expected_state = match direction {
        Direction::Forward => R2cState::RealInput,
        Direction::Inverse | Direction::Backward => R2cState::ComplexOutput,
    };
    if array.state == R2cState::Poisoned {
        return Err(MixedError::Fft(FftError::Array(ArrayError::Poisoned)));
    }
    if array.state != expected_state {
        return Err(MixedError::Fft(FftError::InputLayoutMismatch));
    }
    let expected = match direction {
        Direction::Forward => core.stages[0].input.as_ref(),
        Direction::Inverse | Direction::Backward => {
            core.stages[core.stages.len() - 1].output.as_ref()
        }
    };
    let Some(storage) = array.storage.as_ref() else {
        return Err(MixedError::Fft(FftError::StorageLayoutMismatch));
    };
    match (direction, storage) {
        (Direction::Forward, MixedR2cStorage::Real(real)) => {
            let active = real.active_pencil().map_err(FftError::Array)?;
            if real.extra_shape() != &core.extra_shape
                || real.storage_len() != array.real_storage_len
            {
                return Err(MixedError::Fft(FftError::StorageLayoutMismatch));
            }
            validate_exact_pencil_registry(
                real.pencils(),
                &core.real_pencils,
                active,
                real.storage_len(),
                real.extra_shape(),
            )
            .map_err(MixedError::Fft)?;
            if !active.same_layout(expected) {
                return Err(MixedError::Fft(FftError::InputLayoutMismatch));
            }
            let complex_pencils = array
                .complex_pencils
                .as_ref()
                .ok_or(MixedError::Fft(FftError::StorageLayoutMismatch))?;
            validate_exact_pencil_list(complex_pencils, &core.complex_pencils)
                .map_err(MixedError::Fft)?;
        }
        (Direction::Inverse | Direction::Backward, MixedR2cStorage::Complex(complex)) => {
            let active = complex.active_pencil().map_err(FftError::Array)?;
            if complex.extra_shape() != &core.extra_shape
                || complex.storage_len() != array.complex_storage_len
            {
                return Err(MixedError::Fft(FftError::StorageLayoutMismatch));
            }
            validate_exact_pencil_registry(
                complex.pencils(),
                &core.complex_pencils,
                active,
                complex.storage_len(),
                complex.extra_shape(),
            )
            .map_err(MixedError::Fft)?;
            if !active.same_layout(expected) {
                return Err(MixedError::Fft(FftError::InputLayoutMismatch));
            }
            let real_pencils = array
                .real_pencils
                .as_ref()
                .ok_or(MixedError::Fft(FftError::StorageLayoutMismatch))?;
            validate_exact_pencil_list(real_pencils, &core.real_pencils)
                .map_err(MixedError::Fft)?;
        }
        _ => return Err(MixedError::Fft(FftError::StorageLayoutMismatch)),
    }
    validate_workspace_lengths_values(
        workspace.fft_scratch.len(),
        core.fft_scratch_len,
        workspace.transpose.send_len(),
        core.transpose_send_len,
        workspace.transpose.receive_len(),
        core.transpose_receive_len,
    )?;
    if core.real_stage_index > 0 {
        let transpose = workspace
            .real_transpose
            .as_ref()
            .ok_or(MixedError::Fft(FftError::WorkspaceMismatch))?;
        validate_workspace_lengths_values(
            workspace.fft_scratch.len(),
            core.fft_scratch_len,
            transpose.send_len(),
            core.real_transpose_send_len,
            transpose.receive_len(),
            core.real_transpose_receive_len,
        )?;
    } else if workspace.real_transpose.is_some() {
        return Err(MixedError::Fft(FftError::WorkspaceMismatch));
    }
    for (actual, required, kind) in [
        (
            workspace.embedding_line.len(),
            core.embedding_len,
            "complex embedding",
        ),
        (
            workspace.real_source_line.len(),
            core.real_len,
            "real source line",
        ),
        (
            workspace.real_line.len(),
            core.real_line_len.max(core.real_len),
            "real line",
        ),
        (
            workspace.real_strided_line.len(),
            core.real_line_len,
            "real line",
        ),
        (
            workspace.complex_source_line.len(),
            core.complex_len,
            "complex source line",
        ),
        (
            workspace.complex_line.len(),
            core.complex_line_len.max(core.complex_len),
            "complex line",
        ),
        (
            workspace.complex_strided_line.len(),
            core.complex_line_len,
            "complex line",
        ),
    ] {
        if actual < required {
            return Err(MixedError::Fft(FftError::WorkspaceTooSmall {
                kind,
                required,
                actual,
            }));
        }
    }
    let (real_len, complex_len, storage_bytes, storage_capacity) =
        mixed_r2c_storage_requirements(core)?;
    if real_len != array.real_storage_len
        || complex_len != array.complex_storage_len
        || storage_bytes != array.storage_bytes
    {
        return Err(MixedError::Fft(FftError::StorageLayoutMismatch));
    }
    match (direction, storage) {
        (Direction::Forward, MixedR2cStorage::Real(real)) => {
            if real.storage_len() != real_len {
                return Err(MixedError::Fft(FftError::StorageLayoutMismatch));
            }
            validate_mixed_recast_capacity::<R, Complex<R>>(
                real.storage_capacity(),
                storage_bytes,
            )?;
        }
        (Direction::Inverse | Direction::Backward, MixedR2cStorage::Complex(complex)) => {
            if complex.storage_len() != complex_len {
                return Err(MixedError::Fft(FftError::StorageLayoutMismatch));
            }
            validate_mixed_recast_capacity::<Complex<R>, R>(
                complex.storage_capacity(),
                storage_bytes,
            )?;
        }
        _ => return Err(MixedError::Fft(FftError::StorageLayoutMismatch)),
    }
    let (pencil, rows, source_line_len, destination_line_len) = match direction {
        Direction::Forward => (
            core.stages[core.real_stage_index].input.as_ref(),
            mixed_boundary_rows(core, false)?,
            core.real_len,
            core.complex_len,
        ),
        Direction::Inverse | Direction::Backward => (
            core.stages[core.real_stage_index].output.as_ref(),
            mixed_boundary_rows(core, true)?,
            core.complex_len,
            core.real_len,
        ),
    };
    let stride = memory_stride(pencil, core.stages[core.real_stage_index].axis)?;
    if stride > 1 {
        if rows % stride != 0 {
            return Err(MixedError::Fft(FftError::PreparationFailed));
        }
        let groups = rows / stride;
        let source_span = groups
            .checked_mul(source_line_len)
            .and_then(|value| value.checked_mul(stride))
            .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
        let destination_span = groups
            .checked_mul(destination_line_len)
            .and_then(|value| value.checked_mul(stride))
            .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
        if source_span > storage_capacity || destination_span > storage_capacity {
            return Err(MixedError::Fft(FftError::PreparationFailed));
        }
    }
    Ok(())
}

fn mixed_boundary_rows<R: FftReal, const N: usize, const M: usize>(
    core: &MixedR2cCore<R, N, M>,
    complex: bool,
) -> Result<usize, MixedError> {
    let boundary = core.real_stage_index;
    let line_len = if complex {
        core.complex_len
    } else {
        core.real_len
    };
    let local_len = if complex {
        core.stages[boundary].output.local_len()
    } else {
        core.stages[boundary].input.local_len()
    };
    if local_len % line_len != 0 {
        return Err(MixedError::Fft(FftError::PreparationFailed));
    }
    (local_len / line_len)
        .checked_mul(core.extra_shape.element_count())
        .ok_or(MixedError::Fft(FftError::PreparationFailed))
}

fn validate_mixed_recast_capacity<T, U>(
    capacity: usize,
    required_bytes: usize,
) -> Result<(), MixedError> {
    let source_size = size_of::<T>();
    let target_size = size_of::<U>();
    if source_size == 0
        || target_size == 0
        || align_of::<T>() != align_of::<U>()
        || capacity
            .checked_mul(source_size)
            .is_none_or(|bytes| bytes > isize::MAX as usize || bytes < required_bytes)
    {
        return Err(MixedError::Fft(FftError::StorageLayoutMismatch));
    }
    let capacity_bytes = capacity
        .checked_mul(source_size)
        .ok_or(MixedError::Fft(FftError::StorageLayoutMismatch))?;
    if capacity_bytes % target_size != 0 {
        return Err(MixedError::Fft(FftError::StorageLayoutMismatch));
    }
    Ok(())
}

fn mixed_fail_real_forward<R: FftReal, const N: usize, const M: usize>(
    array: &mut MixedR2cInPlaceArray<R, N, M>,
    source_pencils: Box<[Arc<Pencil<N, M>>]>,
    destination_pencils: Box<[Arc<Pencil<N, M>>]>,
    storage: Vec<R>,
    cause: MixedError,
) -> MixedError {
    array.storage = Some(MixedR2cStorage::PoisonedReal(storage));
    array.real_pencils = Some(source_pencils);
    array.complex_pencils = Some(destination_pencils);
    cause
}

fn mixed_fail_complex_forward<R: FftReal, const N: usize, const M: usize>(
    array: &mut MixedR2cInPlaceArray<R, N, M>,
    source_pencils: Box<[Arc<Pencil<N, M>>]>,
    destination_pencils: Box<[Arc<Pencil<N, M>>]>,
    storage: Vec<Complex<R>>,
    cause: MixedError,
) -> MixedError {
    array.storage = Some(MixedR2cStorage::PoisonedComplex(storage));
    array.real_pencils = Some(source_pencils);
    array.complex_pencils = Some(destination_pencils);
    cause
}

fn mixed_fail_complex_reverse<R: FftReal, const N: usize, const M: usize>(
    array: &mut MixedR2cInPlaceArray<R, N, M>,
    source_pencils: Box<[Arc<Pencil<N, M>>]>,
    destination_pencils: Box<[Arc<Pencil<N, M>>]>,
    storage: Vec<Complex<R>>,
    cause: MixedError,
) -> MixedError {
    array.storage = Some(MixedR2cStorage::PoisonedComplex(storage));
    array.real_pencils = Some(destination_pencils);
    array.complex_pencils = Some(source_pencils);
    cause
}

fn mixed_fail_real_reverse<R: FftReal, const N: usize, const M: usize>(
    array: &mut MixedR2cInPlaceArray<R, N, M>,
    source_pencils: Box<[Arc<Pencil<N, M>>]>,
    destination_pencils: Box<[Arc<Pencil<N, M>>]>,
    storage: Vec<R>,
    cause: MixedError,
) -> MixedError {
    array.storage = Some(MixedR2cStorage::PoisonedReal(storage));
    array.real_pencils = Some(destination_pencils);
    array.complex_pencils = Some(source_pencils);
    cause
}

fn mixed_convert_strided_real_forward<R: FftReal, const N: usize, const M: usize>(
    core: &MixedR2cCore<R, N, M>,
    plan: &LocalR2cPlan<R>,
    rows: usize,
    stride: usize,
    storage: &mut [Complex<R>],
    workspace: &mut MixedR2cInPlaceWorkspace<R, N, M>,
) -> Result<(), MixedError>
where
    Complex<R>: Equivalence,
{
    if stride <= 1 || rows % stride != 0 {
        return Err(MixedError::Fft(FftError::PreparationFailed));
    }
    let groups = rows / stride;
    let source_span = groups
        .checked_mul(core.real_len)
        .and_then(|value| value.checked_mul(stride))
        .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
    let destination_span = groups
        .checked_mul(core.complex_len)
        .and_then(|value| value.checked_mul(stride))
        .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
    if source_span > storage.len() || destination_span > storage.len() {
        return Err(MixedError::Fft(FftError::StorageLayoutMismatch));
    }
    if workspace.real_source_line.len() < core.real_len
        || workspace.complex_line.len() < core.complex_len
        || workspace.real_line.len() < core.real_len
    {
        return Err(MixedError::Fft(FftError::WorkspaceTooSmall {
            kind: "real/complex line",
            required: core.real_len.max(core.complex_len),
            actual: workspace
                .real_source_line
                .len()
                .min(workspace.complex_line.len())
                .min(workspace.real_line.len()),
        }));
    }
    for outer in 0..groups {
        let source_base = outer
            .checked_mul(core.real_len)
            .and_then(|value| value.checked_mul(stride))
            .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
        let destination_base = outer
            .checked_mul(core.complex_len)
            .and_then(|value| value.checked_mul(stride))
            .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
        for inner in 0..stride {
            for k in 0..core.real_len {
                let index = source_base
                    .checked_add(
                        k.checked_mul(stride)
                            .ok_or(MixedError::Fft(FftError::PreparationFailed))?,
                    )
                    .and_then(|value| value.checked_add(inner))
                    .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
                workspace.real_source_line[k] = storage[index].re;
            }
            plan.forward(
                &workspace.real_source_line[..core.real_len],
                &mut workspace.complex_line[..core.complex_len],
                &mut workspace.real_line,
                &mut workspace.fft_scratch,
            )
            .map_err(MixedError::LocalR2c)?;
            for k in 0..core.complex_len {
                let index = destination_base
                    .checked_add(
                        k.checked_mul(stride)
                            .ok_or(MixedError::Fft(FftError::PreparationFailed))?,
                    )
                    .and_then(|value| value.checked_add(inner))
                    .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
                storage[index] = workspace.complex_line[k];
            }
        }
    }
    Ok(())
}

fn mixed_convert_real_to_complex<R: FftReal, const N: usize, const M: usize>(
    core: &Arc<MixedR2cCore<R, N, M>>,
    array: &mut MixedR2cInPlaceArray<R, N, M>,
    workspace: &mut MixedR2cInPlaceWorkspace<R, N, M>,
) -> Result<(), MixedError>
where
    Complex<R>: Equivalence,
{
    let boundary = core.real_stage_index;
    let stage = &core.stages[boundary];
    let stride = memory_stride(stage.input.as_ref(), stage.axis)?;
    let rows = mixed_boundary_rows(core, false)?;
    let active = array
        .complex_pencils
        .as_ref()
        .and_then(|pencils| {
            pencils
                .iter()
                .position(|p| p.same_layout(stage.output.as_ref()))
        })
        .ok_or(MixedError::Fft(FftError::StorageLayoutMismatch))?;
    let destination_pencils = array
        .complex_pencils
        .take()
        .ok_or(MixedError::Fft(FftError::StorageLayoutMismatch))?;
    let real = match array.storage.take() {
        Some(MixedR2cStorage::Real(real)) => real,
        _ => {
            array.complex_pencils = Some(destination_pencils);
            return Err(MixedError::Fft(FftError::StorageLayoutMismatch));
        }
    };
    let (source_pencils, _source_active, _extra, mut storage) = match real.into_parts_preserving() {
        Ok(parts) => parts,
        Err((error, real)) => {
            array.storage = Some(MixedR2cStorage::PoisonedReal(real.into_storage()));
            array.complex_pencils = Some(destination_pencils);
            return Err(MixedError::Fft(FftError::Array(error)));
        }
    };
    #[cfg(test)]
    if array.test_hook == Some(MixedInPlaceTestHook::ForwardDetach) {
        panic!("injected mixed R2C forward panic after owner detachment");
    }
    let full_bytes = match storage.capacity().checked_mul(size_of::<R>()) {
        Some(value) => value,
        None => {
            return Err(mixed_fail_real_forward(
                array,
                source_pencils,
                destination_pencils,
                storage,
                MixedError::Fft(FftError::StorageLayoutMismatch),
            ));
        }
    };
    if full_bytes % size_of::<Complex<R>>() != 0 {
        return Err(mixed_fail_real_forward(
            array,
            source_pencils,
            destination_pencils,
            storage,
            MixedError::Fft(FftError::StorageLayoutMismatch),
        ));
    }
    storage.resize(full_bytes / size_of::<R>(), R::zero());
    let mut complex_storage = match try_cast_vec(storage) {
        Ok(storage) => storage,
        Err((_, storage)) => {
            array.storage = Some(MixedR2cStorage::PoisonedReal(storage));
            array.real_pencils = Some(source_pencils);
            array.complex_pencils = Some(destination_pencils);
            return Err(MixedError::Fft(FftError::StorageLayoutMismatch));
        }
    };
    let MixedR2cStageLocal::Real(MixedRealLocal::Rfft(plan)) = &stage.local else {
        return Err(mixed_fail_complex_forward(
            array,
            source_pencils,
            destination_pencils,
            complex_storage,
            MixedError::Fft(FftError::PreparationFailed),
        ));
    };
    if stride > 1 {
        if array.real_storage_len > complex_storage.len() {
            return Err(mixed_fail_complex_forward(
                array,
                source_pencils,
                destination_pencils,
                complex_storage,
                MixedError::Fft(FftError::StorageLayoutMismatch),
            ));
        }
        for i in (0..array.real_storage_len).rev() {
            let value = match try_cast_slice::<Complex<R>, R>(&complex_storage) {
                Ok(view) => match view.get(i).copied() {
                    Some(value) => value,
                    None => {
                        return Err(mixed_fail_complex_forward(
                            array,
                            source_pencils,
                            destination_pencils,
                            complex_storage,
                            MixedError::Fft(FftError::StorageLayoutMismatch),
                        ));
                    }
                },
                Err(_) => {
                    return Err(mixed_fail_complex_forward(
                        array,
                        source_pencils,
                        destination_pencils,
                        complex_storage,
                        MixedError::Fft(FftError::StorageLayoutMismatch),
                    ));
                }
            };
            complex_storage[i] = Complex::new(value, R::zero());
        }
        if let Err(error) = mixed_convert_strided_real_forward(
            core,
            plan,
            rows,
            stride,
            &mut complex_storage,
            workspace,
        ) {
            return Err(mixed_fail_complex_forward(
                array,
                source_pencils,
                destination_pencils,
                complex_storage,
                error,
            ));
        }
    } else {
        for row in (0..rows).rev() {
            let source_start = match row.checked_mul(core.real_len) {
                Some(value) => value,
                None => {
                    return Err(mixed_fail_complex_forward(
                        array,
                        source_pencils,
                        destination_pencils,
                        complex_storage,
                        MixedError::Fft(FftError::PreparationFailed),
                    ));
                }
            };
            let source_end = match source_start.checked_add(core.real_len) {
                Some(value) => value,
                None => {
                    return Err(mixed_fail_complex_forward(
                        array,
                        source_pencils,
                        destination_pencils,
                        complex_storage,
                        MixedError::Fft(FftError::PreparationFailed),
                    ));
                }
            };
            let scalar_len = match try_cast_slice::<Complex<R>, R>(&complex_storage) {
                Ok(view) => {
                    if source_end > view.len() {
                        return Err(mixed_fail_complex_forward(
                            array,
                            source_pencils,
                            destination_pencils,
                            complex_storage,
                            MixedError::Fft(FftError::PreparationFailed),
                        ));
                    }
                    workspace.real_source_line[..core.real_len]
                        .copy_from_slice(&view[source_start..source_end]);
                    view.len()
                }
                Err(_) => {
                    return Err(mixed_fail_complex_forward(
                        array,
                        source_pencils,
                        destination_pencils,
                        complex_storage,
                        MixedError::Fft(FftError::StorageLayoutMismatch),
                    ));
                }
            };
            let _ = scalar_len;
            let destination_start = match row.checked_mul(core.complex_len) {
                Some(value) => value,
                None => {
                    return Err(mixed_fail_complex_forward(
                        array,
                        source_pencils,
                        destination_pencils,
                        complex_storage,
                        MixedError::Fft(FftError::PreparationFailed),
                    ));
                }
            };
            let destination_end = match destination_start.checked_add(core.complex_len) {
                Some(value) => value,
                None => {
                    return Err(mixed_fail_complex_forward(
                        array,
                        source_pencils,
                        destination_pencils,
                        complex_storage,
                        MixedError::Fft(FftError::PreparationFailed),
                    ));
                }
            };
            if destination_end > complex_storage.len() {
                return Err(mixed_fail_complex_forward(
                    array,
                    source_pencils,
                    destination_pencils,
                    complex_storage,
                    MixedError::Fft(FftError::PreparationFailed),
                ));
            }
            if let Err(error) = plan.forward(
                &workspace.real_source_line[..core.real_len],
                &mut complex_storage[destination_start..destination_end],
                &mut workspace.real_line,
                &mut workspace.fft_scratch,
            ) {
                return Err(mixed_fail_complex_forward(
                    array,
                    source_pencils,
                    destination_pencils,
                    complex_storage,
                    MixedError::LocalR2c(error),
                ));
            }
        }
    }
    complex_storage.truncate(array.complex_storage_len);
    match ManyPencilArray::from_vec_preserving(
        destination_pencils,
        active,
        core.extra_shape.clone(),
        complex_storage,
    ) {
        Ok(complex) => {
            array.storage = Some(MixedR2cStorage::Complex(complex));
            array.real_pencils = Some(source_pencils);
            Ok(())
        }
        Err((error, storage)) => {
            array.storage = Some(MixedR2cStorage::PoisonedComplex(storage));
            array.real_pencils = Some(source_pencils);
            Err(MixedError::Fft(FftError::Array(error)))
        }
    }
}

fn mixed_convert_strided_complex_reverse<R: FftReal, const N: usize, const M: usize>(
    core: &MixedR2cCore<R, N, M>,
    plan: &LocalR2cPlan<R>,
    rows: usize,
    stride: usize,
    storage: &mut [Complex<R>],
    workspace: &mut MixedR2cInPlaceWorkspace<R, N, M>,
    normalize: bool,
) -> Result<(), MixedError>
where
    Complex<R>: Equivalence,
{
    if stride <= 1 || rows % stride != 0 {
        return Err(MixedError::Fft(FftError::PreparationFailed));
    }
    let groups = rows / stride;
    let source_span = groups
        .checked_mul(core.complex_len)
        .and_then(|value| value.checked_mul(stride))
        .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
    let destination_span = groups
        .checked_mul(core.real_len)
        .and_then(|value| value.checked_mul(stride))
        .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
    if source_span > storage.len() || destination_span > storage.len() {
        return Err(MixedError::Fft(FftError::StorageLayoutMismatch));
    }
    for outer in (0..groups).rev() {
        let source_start = outer
            .checked_mul(core.complex_len)
            .and_then(|value| value.checked_mul(stride))
            .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
        let source_end = source_start
            .checked_add(
                core.complex_len
                    .checked_mul(stride)
                    .ok_or(MixedError::Fft(FftError::PreparationFailed))?,
            )
            .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
        let destination_start = outer
            .checked_mul(core.real_len)
            .and_then(|value| value.checked_mul(stride))
            .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
        storage.copy_within(source_start..source_end, destination_start);
    }
    if workspace.complex_source_line.len() < core.complex_len
        || workspace.complex_line.len() < core.complex_len
        || workspace.real_line.len() < core.real_len
    {
        return Err(MixedError::Fft(FftError::WorkspaceTooSmall {
            kind: "complex/real line",
            required: core.real_len.max(core.complex_len),
            actual: workspace
                .complex_source_line
                .len()
                .min(workspace.complex_line.len())
                .min(workspace.real_line.len()),
        }));
    }
    for outer in 0..groups {
        let base = outer
            .checked_mul(core.real_len)
            .and_then(|value| value.checked_mul(stride))
            .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
        for inner in 0..stride {
            for k in 0..core.complex_len {
                let index = base
                    .checked_add(
                        k.checked_mul(stride)
                            .ok_or(MixedError::Fft(FftError::PreparationFailed))?,
                    )
                    .and_then(|value| value.checked_add(inner))
                    .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
                workspace.complex_source_line[k] = storage[index];
            }
            let result = if normalize {
                plan.inverse(
                    &workspace.complex_source_line[..core.complex_len],
                    &mut workspace.real_line[..core.real_len],
                    &mut workspace.complex_line,
                    &mut workspace.fft_scratch,
                )
            } else {
                plan.backward(
                    &workspace.complex_source_line[..core.complex_len],
                    &mut workspace.real_line[..core.real_len],
                    &mut workspace.complex_line,
                    &mut workspace.fft_scratch,
                )
            };
            result.map_err(MixedError::LocalR2c)?;
            for k in 0..core.real_len {
                let index = base
                    .checked_add(
                        k.checked_mul(stride)
                            .ok_or(MixedError::Fft(FftError::PreparationFailed))?,
                    )
                    .and_then(|value| value.checked_add(inner))
                    .ok_or(MixedError::Fft(FftError::PreparationFailed))?;
                storage[index].re = workspace.real_line[k];
            }
        }
    }
    Ok(())
}

fn mixed_convert_complex_to_real<R: FftReal, const N: usize, const M: usize>(
    core: &Arc<MixedR2cCore<R, N, M>>,
    array: &mut MixedR2cInPlaceArray<R, N, M>,
    workspace: &mut MixedR2cInPlaceWorkspace<R, N, M>,
    normalize: bool,
) -> Result<(), MixedError>
where
    Complex<R>: Equivalence,
{
    let boundary = core.real_stage_index;
    let stage = &core.stages[boundary];
    let stride = memory_stride(stage.output.as_ref(), stage.axis)?;
    let rows = mixed_boundary_rows(core, true)?;
    let active = array
        .real_pencils
        .as_ref()
        .and_then(|pencils| {
            pencils
                .iter()
                .position(|p| p.same_layout(stage.input.as_ref()))
        })
        .ok_or(MixedError::Fft(FftError::StorageLayoutMismatch))?;
    let destination_pencils = array
        .real_pencils
        .take()
        .ok_or(MixedError::Fft(FftError::StorageLayoutMismatch))?;
    let complex = match array.storage.take() {
        Some(MixedR2cStorage::Complex(complex)) => complex,
        _ => {
            array.real_pencils = Some(destination_pencils);
            return Err(MixedError::Fft(FftError::StorageLayoutMismatch));
        }
    };
    let (source_pencils, _source_active, _extra, mut storage) =
        match complex.into_parts_preserving() {
            Ok(parts) => parts,
            Err((error, complex)) => {
                array.storage = Some(MixedR2cStorage::PoisonedComplex(complex.into_storage()));
                array.real_pencils = Some(destination_pencils);
                return Err(MixedError::Fft(FftError::Array(error)));
            }
        };
    #[cfg(test)]
    if array.test_hook == Some(MixedInPlaceTestHook::ReverseDetach) {
        panic!("injected mixed R2C reverse panic after owner detachment");
    }
    if storage.capacity() < array.complex_storage_len {
        return Err(mixed_fail_complex_reverse(
            array,
            source_pencils,
            destination_pencils,
            storage,
            MixedError::Fft(FftError::StorageLayoutMismatch),
        ));
    }
    storage.resize(storage.capacity(), zero_complex::<R>());
    let MixedR2cStageLocal::Real(MixedRealLocal::Rfft(plan)) = &stage.local else {
        return Err(mixed_fail_complex_reverse(
            array,
            source_pencils,
            destination_pencils,
            storage,
            MixedError::Fft(FftError::PreparationFailed),
        ));
    };
    if stride > 1 {
        if let Err(error) = mixed_convert_strided_complex_reverse(
            core,
            plan,
            rows,
            stride,
            &mut storage,
            workspace,
            normalize,
        ) {
            return Err(mixed_fail_complex_reverse(
                array,
                source_pencils,
                destination_pencils,
                storage,
                error,
            ));
        }
        let mut real_storage = match try_cast_vec(storage) {
            Ok(storage) => storage,
            Err((_, storage)) => {
                return Err(mixed_fail_complex_reverse(
                    array,
                    source_pencils,
                    destination_pencils,
                    storage,
                    MixedError::Fft(FftError::StorageLayoutMismatch),
                ));
            }
        };
        if array.real_storage_len > real_storage.len() {
            return Err(mixed_fail_real_reverse(
                array,
                source_pencils,
                destination_pencils,
                real_storage,
                MixedError::Fft(FftError::StorageLayoutMismatch),
            ));
        }
        for i in 0..array.real_storage_len {
            let value = match try_cast_slice::<R, Complex<R>>(&real_storage) {
                Ok(view) => match view.get(i).copied() {
                    Some(value) => value.re,
                    None => {
                        return Err(mixed_fail_real_reverse(
                            array,
                            source_pencils,
                            destination_pencils,
                            real_storage,
                            MixedError::Fft(FftError::StorageLayoutMismatch),
                        ));
                    }
                },
                Err(_) => {
                    return Err(mixed_fail_real_reverse(
                        array,
                        source_pencils,
                        destination_pencils,
                        real_storage,
                        MixedError::Fft(FftError::StorageLayoutMismatch),
                    ));
                }
            };
            real_storage[i] = value;
        }
        real_storage.truncate(array.real_storage_len);
        return match ManyPencilArray::from_vec_preserving(
            destination_pencils,
            active,
            core.extra_shape.clone(),
            real_storage,
        ) {
            Ok(real) => {
                array.storage = Some(MixedR2cStorage::Real(real));
                array.complex_pencils = Some(source_pencils);
                Ok(())
            }
            Err((error, storage)) => {
                array.storage = Some(MixedR2cStorage::PoisonedReal(storage));
                array.complex_pencils = Some(source_pencils);
                Err(MixedError::Fft(FftError::Array(error)))
            }
        };
    }
    let mut real_storage = match try_cast_vec(storage) {
        Ok(storage) => storage,
        Err((_, storage)) => {
            return Err(mixed_fail_complex_reverse(
                array,
                source_pencils,
                destination_pencils,
                storage,
                MixedError::Fft(FftError::StorageLayoutMismatch),
            ));
        }
    };
    for row in 0..rows {
        let source_start = match row.checked_mul(core.complex_len) {
            Some(value) => value,
            None => {
                return Err(mixed_fail_real_reverse(
                    array,
                    source_pencils,
                    destination_pencils,
                    real_storage,
                    MixedError::Fft(FftError::PreparationFailed),
                ));
            }
        };
        let source_end = match source_start.checked_add(core.complex_len) {
            Some(value) => value,
            None => {
                return Err(mixed_fail_real_reverse(
                    array,
                    source_pencils,
                    destination_pencils,
                    real_storage,
                    MixedError::Fft(FftError::PreparationFailed),
                ));
            }
        };
        let source = match try_cast_slice::<R, Complex<R>>(&real_storage) {
            Ok(view) => {
                if source_end > view.len() {
                    return Err(mixed_fail_real_reverse(
                        array,
                        source_pencils,
                        destination_pencils,
                        real_storage,
                        MixedError::Fft(FftError::PreparationFailed),
                    ));
                }
                workspace.complex_source_line[..core.complex_len]
                    .copy_from_slice(&view[source_start..source_end]);
                true
            }
            Err(_) => {
                return Err(mixed_fail_real_reverse(
                    array,
                    source_pencils,
                    destination_pencils,
                    real_storage,
                    MixedError::Fft(FftError::StorageLayoutMismatch),
                ));
            }
        };
        let _ = source;
        let destination_start = match row.checked_mul(core.real_len) {
            Some(value) => value,
            None => {
                return Err(mixed_fail_real_reverse(
                    array,
                    source_pencils,
                    destination_pencils,
                    real_storage,
                    MixedError::Fft(FftError::PreparationFailed),
                ));
            }
        };
        let destination_end = match destination_start.checked_add(core.real_len) {
            Some(value) => value,
            None => {
                return Err(mixed_fail_real_reverse(
                    array,
                    source_pencils,
                    destination_pencils,
                    real_storage,
                    MixedError::Fft(FftError::PreparationFailed),
                ));
            }
        };
        if destination_end > real_storage.len() {
            return Err(mixed_fail_real_reverse(
                array,
                source_pencils,
                destination_pencils,
                real_storage,
                MixedError::Fft(FftError::PreparationFailed),
            ));
        }
        let result = if normalize {
            plan.inverse(
                &workspace.complex_source_line[..core.complex_len],
                &mut real_storage[destination_start..destination_end],
                &mut workspace.complex_line,
                &mut workspace.fft_scratch,
            )
        } else {
            plan.backward(
                &workspace.complex_source_line[..core.complex_len],
                &mut real_storage[destination_start..destination_end],
                &mut workspace.complex_line,
                &mut workspace.fft_scratch,
            )
        };
        if let Err(error) = result {
            return Err(mixed_fail_real_reverse(
                array,
                source_pencils,
                destination_pencils,
                real_storage,
                MixedError::LocalR2c(error),
            ));
        }
    }
    real_storage.truncate(array.real_storage_len);
    match ManyPencilArray::from_vec_preserving(
        destination_pencils,
        active,
        core.extra_shape.clone(),
        real_storage,
    ) {
        Ok(real) => {
            array.storage = Some(MixedR2cStorage::Real(real));
            array.complex_pencils = Some(source_pencils);
            Ok(())
        }
        Err((error, storage)) => {
            array.storage = Some(MixedR2cStorage::PoisonedReal(storage));
            array.complex_pencils = Some(source_pencils);
            Err(MixedError::Fft(FftError::Array(error)))
        }
    }
}

impl<R: FftReal, const N: usize, const M: usize> MixedR2cInPlaceArray<R, N, M> {
    /// Returns the current in-place state.
    pub fn state(&self) -> R2cState {
        self.state
    }

    /// Returns the backing allocation size in bytes while retained.
    pub fn storage_capacity_bytes(&self) -> Option<usize> {
        let (capacity, element_size) = match self.storage.as_ref()? {
            MixedR2cStorage::Real(real) => (real.storage_capacity(), size_of::<R>()),
            MixedR2cStorage::Complex(complex) => {
                (complex.storage_capacity(), size_of::<Complex<R>>())
            }
            MixedR2cStorage::PoisonedReal(storage) => (storage.capacity(), size_of::<R>()),
            MixedR2cStorage::PoisonedComplex(storage) => {
                (storage.capacity(), size_of::<Complex<R>>())
            }
        };
        capacity.checked_mul(element_size)
    }

    /// Borrows the active real view.
    pub fn real_view(&self) -> Result<PencilArrayView<'_, R, N, M>, MixedError> {
        match self.state {
            R2cState::Poisoned => Err(MixedError::Fft(FftError::Array(ArrayError::Poisoned))),
            R2cState::ComplexOutput => Err(MixedError::Fft(FftError::InputLayoutMismatch)),
            R2cState::RealInput => match self.storage.as_ref() {
                Some(MixedR2cStorage::Real(real)) => real
                    .active_view()
                    .map_err(FftError::Array)
                    .map_err(Into::into),
                _ => Err(MixedError::Fft(FftError::StorageLayoutMismatch)),
            },
        }
    }

    /// Borrows the active mutable real view.
    pub fn real_view_mut(&mut self) -> Result<PencilArrayViewMut<'_, R, N, M>, MixedError> {
        match self.state {
            R2cState::Poisoned => Err(MixedError::Fft(FftError::Array(ArrayError::Poisoned))),
            R2cState::ComplexOutput => Err(MixedError::Fft(FftError::InputLayoutMismatch)),
            R2cState::RealInput => match self.storage.as_mut() {
                Some(MixedR2cStorage::Real(real)) => real
                    .active_view_mut()
                    .map_err(FftError::Array)
                    .map_err(Into::into),
                _ => Err(MixedError::Fft(FftError::StorageLayoutMismatch)),
            },
        }
    }

    /// Borrows the active reduced-complex view.
    pub fn complex_view(&self) -> Result<PencilArrayView<'_, Complex<R>, N, M>, MixedError> {
        match self.state {
            R2cState::Poisoned => Err(MixedError::Fft(FftError::Array(ArrayError::Poisoned))),
            R2cState::RealInput => Err(MixedError::Fft(FftError::OutputLayoutMismatch)),
            R2cState::ComplexOutput => match self.storage.as_ref() {
                Some(MixedR2cStorage::Complex(complex)) => complex
                    .active_view()
                    .map_err(FftError::Array)
                    .map_err(Into::into),
                _ => Err(MixedError::Fft(FftError::StorageLayoutMismatch)),
            },
        }
    }

    /// Borrows the active mutable reduced-complex view.
    pub fn complex_view_mut(
        &mut self,
    ) -> Result<PencilArrayViewMut<'_, Complex<R>, N, M>, MixedError> {
        match self.state {
            R2cState::Poisoned => Err(MixedError::Fft(FftError::Array(ArrayError::Poisoned))),
            R2cState::RealInput => Err(MixedError::Fft(FftError::OutputLayoutMismatch)),
            R2cState::ComplexOutput => match self.storage.as_mut() {
                Some(MixedR2cStorage::Complex(complex)) => complex
                    .active_view_mut()
                    .map_err(FftError::Array)
                    .map_err(Into::into),
                _ => Err(MixedError::Fft(FftError::StorageLayoutMismatch)),
            },
        }
    }
}

fn execute_mixed_r2c_forward_ip<R: FftReal, const N: usize, const M: usize>(
    core: &Arc<MixedR2cCore<R, N, M>>,
    array: &mut MixedR2cInPlaceArray<R, N, M>,
    workspace: &mut MixedR2cInPlaceWorkspace<R, N, M>,
    mut report: Option<&mut TransformTiming<N>>,
) -> Result<(), MixedError>
where
    Complex<R>: Equivalence,
{
    let communicator = core.stages[0].input.topology().communicator();
    let boundary = core.real_stage_index;
    if boundary > 0 {
        let real = match array.storage.as_mut() {
            Some(MixedR2cStorage::Real(real)) => real,
            _ => return Err(MixedError::Fft(FftError::StorageLayoutMismatch)),
        };
        let stage0 = &core.stages[0];
        let MixedR2cStageLocal::Real(local0) = &stage0.local else {
            return Err(MixedError::Fft(FftError::PreparationFailed));
        };
        let fft_started = Instant::now();
        let stage_result = (|| {
            mixed_real_prefix_forward_in_place(
                local0,
                stage0.input.as_ref(),
                stage0.axis,
                real.active_view_mut()
                    .map_err(FftError::Array)?
                    .as_mut_slice(),
                &mut workspace.embedding_line,
                &mut workspace.fft_scratch,
                &mut workspace.real_strided_line,
            )
        })();
        super::record_fft_timing(&mut report, 0, fft_started);
        agree_result(communicator, stage_result)?;
        for index in 0..boundary {
            let transpose = workspace
                .real_transpose
                .as_mut()
                .ok_or(MixedError::Fft(FftError::WorkspaceMismatch))?;
            execute_mixed_transition_timed(
                communicator,
                &core.transitions[index].forward,
                real,
                transpose,
                &mut report,
                index,
            )?;
            if index + 1 < boundary {
                let stage = &core.stages[index + 1];
                let MixedR2cStageLocal::Real(local) = &stage.local else {
                    return Err(MixedError::Fft(FftError::PreparationFailed));
                };
                let fft_started = Instant::now();
                let stage_result = (|| {
                    mixed_real_prefix_forward_in_place(
                        local,
                        stage.output.as_ref(),
                        stage.axis,
                        real.active_view_mut()
                            .map_err(FftError::Array)?
                            .as_mut_slice(),
                        &mut workspace.embedding_line,
                        &mut workspace.fft_scratch,
                        &mut workspace.real_strided_line,
                    )
                })();
                super::record_fft_timing(&mut report, index + 1, fft_started);
                agree_result(communicator, stage_result)?;
            }
        }
    }
    let fft_started = Instant::now();
    let conversion = mixed_convert_real_to_complex(core, array, workspace);
    super::record_fft_timing(&mut report, boundary, fft_started);
    agree_result(communicator, conversion)?;
    let complex = match array.storage.as_mut() {
        Some(MixedR2cStorage::Complex(complex)) => complex,
        _ => return Err(MixedError::Fft(FftError::StorageLayoutMismatch)),
    };
    for index in boundary + 1..core.stages.len() {
        execute_mixed_transition_timed(
            communicator,
            &core.transitions[index - 1].forward,
            complex,
            &mut workspace.transpose,
            &mut report,
            index - 1,
        )?;
        let stage = &core.stages[index];
        let MixedR2cStageLocal::Complex(local) = &stage.local else {
            return Err(MixedError::Fft(FftError::PreparationFailed));
        };
        let fft_started = Instant::now();
        let stage_result = (|| {
            mixed_complex_forward_in_place(
                local,
                stage.input.as_ref(),
                stage.axis,
                complex
                    .active_view_mut()
                    .map_err(FftError::Array)?
                    .as_mut_slice(),
                &mut workspace.embedding_line,
                &mut workspace.fft_scratch,
                &mut workspace.complex_strided_line,
            )
        })();
        super::record_fft_timing(&mut report, index, fft_started);
        agree_result(communicator, stage_result)?;
    }
    Ok(())
}

fn execute_mixed_r2c_reverse_ip<R: FftReal, const N: usize, const M: usize>(
    core: &Arc<MixedR2cCore<R, N, M>>,
    array: &mut MixedR2cInPlaceArray<R, N, M>,
    workspace: &mut MixedR2cInPlaceWorkspace<R, N, M>,
    normalize: bool,
    mut report: Option<&mut TransformTiming<N>>,
) -> Result<(), MixedError>
where
    Complex<R>: Equivalence,
{
    let communicator = core.stages[0].input.topology().communicator();
    let boundary = core.real_stage_index;
    let complex = match array.storage.as_mut() {
        Some(MixedR2cStorage::Complex(complex)) => complex,
        _ => return Err(MixedError::Fft(FftError::StorageLayoutMismatch)),
    };
    for index in (boundary + 1..core.stages.len()).rev() {
        let stage = &core.stages[index];
        let MixedR2cStageLocal::Complex(local) = &stage.local else {
            return Err(MixedError::Fft(FftError::PreparationFailed));
        };
        let fft_started = Instant::now();
        let stage_result = (|| {
            mixed_complex_reverse_in_place(
                local,
                stage.output.as_ref(),
                stage.axis,
                complex
                    .active_view_mut()
                    .map_err(FftError::Array)?
                    .as_mut_slice(),
                &mut workspace.embedding_line,
                &mut workspace.fft_scratch,
                &mut workspace.complex_strided_line,
                normalize,
            )
        })();
        super::record_fft_timing(&mut report, index, fft_started);
        agree_result(communicator, stage_result)?;
        execute_mixed_transition_timed(
            communicator,
            &core.transitions[index - 1].backward,
            complex,
            &mut workspace.transpose,
            &mut report,
            index - 1,
        )?;
    }
    validate_mixed_boundary(core, complex, normalize)?;
    agree_result(communicator, zero_mixed_boundary(core, complex))?;
    let fft_started = Instant::now();
    let conversion = mixed_convert_complex_to_real(core, array, workspace, normalize);
    super::record_fft_timing(&mut report, boundary, fft_started);
    agree_result(communicator, conversion)?;
    if boundary > 0 {
        let real = match array.storage.as_mut() {
            Some(MixedR2cStorage::Real(real)) => real,
            _ => return Err(MixedError::Fft(FftError::StorageLayoutMismatch)),
        };
        for index in (0..boundary).rev() {
            let transpose = workspace
                .real_transpose
                .as_mut()
                .ok_or(MixedError::Fft(FftError::WorkspaceMismatch))?;
            execute_mixed_transition_timed(
                communicator,
                &core.transitions[index].backward,
                real,
                transpose,
                &mut report,
                index,
            )?;
            let stage = &core.stages[index];
            let MixedR2cStageLocal::Real(local) = &stage.local else {
                return Err(MixedError::Fft(FftError::PreparationFailed));
            };
            let fft_started = Instant::now();
            let stage_result = (|| {
                mixed_real_prefix_reverse_in_place(
                    local,
                    stage.output.as_ref(),
                    stage.axis,
                    real.active_view_mut()
                        .map_err(FftError::Array)?
                        .as_mut_slice(),
                    &mut workspace.embedding_line,
                    &mut workspace.fft_scratch,
                    &mut workspace.real_strided_line,
                    normalize,
                )
            })();
            super::record_fft_timing(&mut report, index, fft_started);
            agree_result(communicator, stage_result)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod reverse_overlap_tests {
    use super::*;
    use mpi::topology::Communicator;
    use std::cell::Cell;
    thread_local! { static CALLBACKS: Cell<usize> = const { Cell::new(0) }; }

    pub(super) fn callback() -> Result<(), MixedError> {
        CALLBACKS.with(|count| count.set(count.get() + 1));
        super::super::consume_c2c_callback_injection().map_err(MixedError::Fft)
    }

    #[test]
    #[ignore = "MPI must be initialized in a separate test process"]
    fn reverse_local_routes_run_callbacks_and_independent_spectrum() {
        let universe = mpi::initialize().unwrap();
        let world = universe.world();
        let size = world.size() as usize;
        assert!(matches!(size, 1 | 4 | 6));
        let topology = pencil_array::MpiTopology::<2>::new(&world, [size, 1]).unwrap();
        for permute_dims in [false, true] {
            for boundary in 0..4 {
                let mut transforms = [AxisTransform::R2r(AxisR2rKind::Dht); 4];
                transforms[..boundary].fill(AxisTransform::Fft);
                transforms[boundary] = AxisTransform::Rfft;
                let plan = MixedR2cPlan::<f64, 4, 2>::from_shape_with_layout(
                    Arc::clone(&topology),
                    [8, 9, 10, 12],
                    ExtraShape::scalar(),
                    transforms,
                    DistributedLayout {
                        permute_dims,
                        transpose_method: TransposeMethod::PointToPoint,
                    },
                )
                .unwrap();
                let callbacks = plan
                    .core
                    .transitions
                    .iter()
                    .filter(|t| matches!(t.backward, super::super::C2cTransition::PointToPoint(_)))
                    .count();
                assert!(callbacks > 0);
                assert!(
                    plan.core.transitions.iter().any(|t| if permute_dims {
                        matches!(t.backward, super::super::C2cTransition::Local(_))
                    } else {
                        matches!(t.backward, super::super::C2cTransition::Identity)
                    }),
                    "must exercise the requested local/identity segment as well"
                );
                let mut source = plan.allocate_output().unwrap();
                source.as_mut_slice().fill(Complex::new(1.0, 0.0));
                let mut destination = plan.allocate_input().unwrap();
                let mut workspace = plan.allocate_workspace().unwrap();
                for raw in [false, true] {
                    CALLBACKS.with(|count| count.set(0));
                    if raw {
                        plan.backward_with_overlap(&source, &mut destination, &mut workspace)
                            .unwrap();
                    } else {
                        plan.inverse_with_overlap(&source, &mut destination, &mut workspace)
                            .unwrap();
                    }
                    assert_eq!(
                        CALLBACKS.with(Cell::get),
                        callbacks,
                        "each P2P FFT must run inside the actual transpose callback"
                    );
                    // An independently specified constant half-spectrum is a unit impulse.
                    // Raw FFT/DHT normalization is the product of all four extents.
                    for i in 0..8 {
                        for j in 0..9 {
                            for k in 0..10 {
                                for l in 0..12 {
                                    if let Some(&actual) = destination.get_global(&[], [i, j, k, l])
                                    {
                                        let expected = if [i, j, k, l] == [0; 4] {
                                            if raw { 8640.0 } else { 1.0 }
                                        } else {
                                            0.0
                                        };
                                        assert!(
                                            (actual - expected).abs() < 1e-9,
                                            "boundary={boundary} raw={raw} at {:?}: {actual} != {expected}",
                                            [i, j, k, l]
                                        );
                                    }
                                }
                            }
                        }
                    }
                    assert!(
                        source
                            .as_slice()
                            .iter()
                            .all(|&x| x == Complex::new(1.0, 0.0))
                    );
                    use super::super::{C2C_CALLBACK_INJECTION, C2cCallbackInjection};
                    for injection in [C2cCallbackInjection::Error, C2cCallbackInjection::Panic] {
                        let mut failed_workspace = plan.allocate_workspace().unwrap();
                        if world.rank() == 0 {
                            C2C_CALLBACK_INJECTION.with(|slot| slot.set(Some(injection)));
                        }
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            if raw {
                                plan.backward_with_overlap(
                                    &source,
                                    &mut destination,
                                    &mut failed_workspace,
                                )
                            } else {
                                plan.inverse_with_overlap(
                                    &source,
                                    &mut destination,
                                    &mut failed_workspace,
                                )
                            }
                        }));
                        if matches!(injection, C2cCallbackInjection::Panic) && world.rank() == 0 {
                            assert!(result.is_err());
                        } else {
                            match (injection, world.rank() == 0, result.unwrap().unwrap_err()) {
                                (
                                    C2cCallbackInjection::Error,
                                    true,
                                    FftOverlapError::Overlap(OverlapError::Callback(
                                        MixedError::Fft(FftError::PreparationFailed),
                                    )),
                                ) => {}
                                (
                                    C2cCallbackInjection::Error,
                                    false,
                                    FftOverlapError::Overlap(OverlapError::PeerCallbackFailed),
                                ) => {}
                                (
                                    C2cCallbackInjection::Panic,
                                    false,
                                    FftOverlapError::Overlap(OverlapError::PeerPanicked),
                                ) => {}
                                (_, _, error) => {
                                    panic!("unexpected mixed callback result: {error:?}")
                                }
                            }
                        }
                        assert!(
                            source
                                .as_slice()
                                .iter()
                                .all(|&x| x == Complex::new(1.0, 0.0))
                        );
                        assert!(
                            matches!(
                                failed_workspace.intermediate.active_view(),
                                Err(ArrayError::Poisoned)
                            ) || failed_workspace.real_intermediate.as_ref().is_some_and(
                                |real| matches!(real.active_view(), Err(ArrayError::Poisoned))
                            )
                        );
                        world.barrier(); // No extra FFT agreement after callback Err/panic.
                        plan.inverse(&source, &mut destination, &mut workspace)
                            .unwrap();
                    }
                }
            }
        }
    }
}
