//! Distributed complex-to-complex FFTs behind the `distributed` feature.
//!
//! [`C2cPlan`] construction and execution are collective: every rank must use
//! the same Cartesian communicator context, API, order, scalar type, and
//! matching layouts. Forward transforms consume the canonical input layout and
//! produce the reversed output layout; [`C2cPlan::inverse`] consumes those
//! layouts in reverse and applies the complete normalization.
//!
//! Descriptor and initial preflight errors are collectively reported before any
//! source, destination, or workspace write. After the first FFT stage starts,
//! a checked Alltoallv transition may prepare its own metadata and return a
//! collectively agreed preparation or allocation error; the source remains
//! preserved, but workspace contents are not transactional on that path.

use std::{mem::size_of, sync::Arc};

use mpi::{
    Count,
    collective::{CommunicatorCollectives, SystemOperation},
    datatype::Equivalence,
};
use pencil_array::{
    AllToAllvTransposePlan, ArrayError, AxisPermutation, ExtraShape, LocalTransposeError,
    LocalTransposePlan, ManyPencilArray, MpiTopology, Pencil, PencilArray, PencilConfig,
    PencilError, SpatialAxis, TransposeError, TransposeWorkspace,
};
use thiserror::Error;

use crate::{Complex, FftReal, LocalC2cError, LocalC2cPlan};

const DESCRIPTOR_SCHEMA: u64 = 1;
const OPERATION_PLAN: u64 = 7;
const OPERATION_FORWARD: u64 = 8;
const OPERATION_INVERSE: u64 = 9;
const INVALID_WORD: u64 = u64::MAX;
const HEADER_WORDS: usize = 5;

/// Errors returned by the feature-gated distributed C2C API.
#[derive(Debug, Error)]
pub enum FftError {
    /// `N` must be at least two and `M` must satisfy `1 <= M < N`.
    #[error("distributed C2C requires N >= 2 and 1 <= M < N")]
    InvalidDimensions,

    /// The input pencil was not the canonical identity layout.
    #[error("input pencil must use identity permutation and decomposition [0..M)")]
    InvalidInputLayout,

    /// The supplied source array does not match the plan's input layout.
    #[error("source layout does not match the distributed C2C plan")]
    InputLayoutMismatch,

    /// The supplied destination array does not match the plan's output layout.
    #[error("destination layout does not match the distributed C2C plan")]
    OutputLayoutMismatch,

    /// An array's extra shape does not match the plan or its peer array.
    #[error("extra shape does not match the distributed C2C plan")]
    ExtraShapeMismatch,

    /// The workspace was created for another plan or has invalid registered layouts.
    #[error("workspace does not belong to this distributed C2C plan")]
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
    #[error("distributed C2C collective descriptors differ between ranks")]
    CollectiveDescriptorMismatch,

    /// At least one rank rejected a collective precondition.
    #[error("a distributed C2C collective precondition failed on another rank")]
    CollectivePreconditionFailed,

    /// Checked route or descriptor preparation failed.
    #[error("distributed C2C preparation failed")]
    PreparationFailed,

    /// A requested initialized allocation could not be made.
    #[error("failed to allocate {required} elements")]
    AllocationFailed {
        /// The requested element count.
        required: usize,
    },

    /// The local native FFT plan or operation rejected checked input.
    #[error(transparent)]
    LocalC2c(#[from] LocalC2cError),

    /// The array crate rejected a pencil construction or topology-derived layout.
    #[error(transparent)]
    Pencil(#[from] PencilError),

    /// The array crate rejected local storage or a shared-array state.
    #[error(transparent)]
    Array(#[from] ArrayError),

    /// The array crate rejected a distributed Alltoallv transition.
    #[error(transparent)]
    Transpose(#[from] TransposeError),

    /// The array crate rejected a process-local transition.
    #[error(transparent)]
    LocalTranspose(#[from] LocalTransposeError),
}

/// An immutable, checked Alltoallv distributed complex-to-complex FFT plan.
///
/// Construction and execution are collective on the topology's Cartesian
/// communicator. Every rank must use the same communicator context and call
/// the same API in the same order. The input must use the identity permutation
/// and decomposition `[0, ..., M)`; this feature supports `N >= 2` and
/// `1 <= M < N`. The derived output uses decomposition `[1, ..., M]` and
/// reversed spatial memory order. Forward and inverse execution are out of
/// place and input preserving.
///
/// # Example
///
/// Initialize MPI once and drop all topology, plan, array, and workspace
/// values before the MPI universe.
///
/// ```
/// use mpi::traits::*;
/// use pencil_array::{ExtraShape, MpiTopology};
/// use pencil_fft::{C2cPlan, Complex};
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
///         Ok::<(), Box<dyn std::error::Error>>(())
///     };
///     result
/// }
/// ```
#[derive(Debug)]
pub struct C2cPlan<R: FftReal, const N: usize, const M: usize> {
    core: Arc<C2cPlanCore<R, N, M>>,
}

/// Reusable storage for [`C2cPlan`] out-of-place execution.
///
/// The workspace is private to the exact plan that allocated it. It contains
/// one registered [`ManyPencilArray`], shared checked transpose buffers, and
/// native FFT scratch. Constructing multiple workspaces from one plan is
/// supported; execution mutates only the selected workspace and destination.
#[derive(Debug)]
pub struct C2cOutOfPlaceWorkspace<R: FftReal, const N: usize, const M: usize> {
    core: Arc<C2cPlanCore<R, N, M>>,
    intermediate: ManyPencilArray<Complex<R>, N, M>,
    transpose: TransposeWorkspace<Complex<R>>,
    fft_scratch: Vec<Complex<R>>,
}

#[derive(Debug)]
struct C2cStage<R: FftReal, const N: usize, const M: usize> {
    pencil: Arc<Pencil<N, M>>,
    local: LocalC2cPlan<R>,
}

#[derive(Debug)]
enum C2cTransition<const N: usize, const M: usize> {
    Local(LocalTransposePlan<N, M>),
    AllToAllv(AllToAllvTransposePlan<N, M>),
}

#[derive(Debug)]
struct C2cStageTransition<const N: usize, const M: usize> {
    forward: C2cTransition<N, M>,
    backward: C2cTransition<N, M>,
}

#[derive(Debug)]
struct C2cPlanCore<R: FftReal, const N: usize, const M: usize> {
    stages: Box<[C2cStage<R, N, M>]>,
    transitions: Box<[C2cStageTransition<N, M>]>,
    extra_shape: ExtraShape,
    descriptor: Box<[u64]>,
    fft_scratch_len: usize,
    transpose_send_len: usize,
    transpose_receive_len: usize,
}

#[derive(Debug)]
struct RouteCandidate<const N: usize, const M: usize> {
    stages: Box<[Arc<Pencil<N, M>>]>,
    distributed: Box<[bool]>,
}

#[derive(Debug)]
struct StagePreparation<R: FftReal, const N: usize, const M: usize> {
    stages: Box<[C2cStage<R, N, M>]>,
    fft_scratch_len: usize,
}

#[derive(Clone, Copy, Debug)]
enum Direction {
    Forward,
    Inverse,
}

impl<R: FftReal, const N: usize, const M: usize> C2cPlan<R, N, M>
where
    Complex<R>: Equivalence,
{
    /// Collectively builds a plan from a canonical input pencil and exact extra shape.
    pub fn from_pencil(
        input: Arc<Pencil<N, M>>,
        extra_shape: ExtraShape,
    ) -> Result<Self, FftError> {
        let topology = Arc::clone(input.topology());
        let global_shape = *input.global_shape();
        Self::construct(topology, global_shape, extra_shape, Ok(input))
    }

    /// Collectively builds a plan from a canonical input array's layout.
    pub fn from_array(input: &PencilArray<Complex<R>, N, M>) -> Result<Self, FftError> {
        let topology = Arc::clone(input.pencil().topology());
        let global_shape = *input.pencil().global_shape();
        Self::construct(
            topology,
            global_shape,
            input.extra_shape().clone(),
            Ok(Arc::clone(input.pencil())),
        )
    }

    /// Collectively builds a plan from topology, global shape, and extra shape.
    ///
    /// Zero global spatial extents are rejected collectively before any native
    /// FFT backend plan is created. Zero extra batches remain valid.
    pub fn from_shape(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
    ) -> Result<Self, FftError> {
        let input = Pencil::new(
            Arc::clone(&topology),
            global_shape,
            std::array::from_fn(|axis| axis),
        )
        .map_err(FftError::Pencil);
        Self::construct(topology, global_shape, extra_shape, input)
    }

    /// Returns the canonical input pencil.
    pub fn input_pencil(&self) -> &Arc<Pencil<N, M>> {
        &self.core.stages[0].pencil
    }

    /// Returns the derived output pencil.
    pub fn output_pencil(&self) -> &Arc<Pencil<N, M>> {
        &self
            .core
            .stages
            .last()
            .expect("distributed C2C has at least two stages")
            .pencil
    }

    /// Returns the exact extra shape required by this plan.
    pub fn extra_shape(&self) -> &ExtraShape {
        &self.core.extra_shape
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

    /// Allocates a reusable workspace for out-of-place forward and inverse execution.
    pub fn allocate_out_of_place_workspace(
        &self,
    ) -> Result<C2cOutOfPlaceWorkspace<R, N, M>, FftError> {
        let mut pencils = Vec::new();
        pencils
            .try_reserve_exact(self.core.stages.len())
            .map_err(|_| FftError::AllocationFailed {
                required: self.core.stages.len(),
            })?;
        pencils.extend(
            self.core
                .stages
                .iter()
                .map(|stage| Arc::clone(&stage.pencil)),
        );
        let intermediate = ManyPencilArray::from_elem(
            pencils.into_boxed_slice(),
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
        Ok(C2cOutOfPlaceWorkspace {
            core: Arc::clone(&self.core),
            intermediate,
            transpose,
            fft_scratch,
        })
    }

    /// Computes an unnormalized forward distributed C2C transform.
    ///
    /// The source uses the input layout and the destination uses the output
    /// layout. Ordinary descriptor and initial preflight errors are collective
    /// and leave source, destination, and workspace unchanged. After the first
    /// FFT stage, a checked Alltoallv may return a collectively agreed metadata
    /// preparation or allocation error; the source remains preserved, but the
    /// workspace may already be changed. Native FFT, MPI failures, arbitrary
    /// panics, and process loss have no global recovery guarantee.
    pub fn forward(
        &self,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &mut PencilArray<Complex<R>, N, M>,
        workspace: &mut C2cOutOfPlaceWorkspace<R, N, M>,
    ) -> Result<(), FftError> {
        self.execute(Direction::Forward, source, destination, workspace)
    }

    /// Computes a normalized inverse distributed C2C transform.
    ///
    /// The source uses the forward output layout and the destination uses the
    /// canonical input layout. It has the same collective and initial-error
    /// guarantees as [`Self::forward`]: initial descriptor/preflight errors
    /// leave all buffers unchanged, while a post-start checked Alltoallv
    /// preparation or allocation error may change workspace contents but never
    /// the source. The local inverse plan divides once by each spatial axis
    /// length, so the complete inverse is normalized by the product of global
    /// spatial lengths. Extra batch dimensions are not included.
    pub fn inverse(
        &self,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &mut PencilArray<Complex<R>, N, M>,
        workspace: &mut C2cOutOfPlaceWorkspace<R, N, M>,
    ) -> Result<(), FftError> {
        self.execute(Direction::Inverse, source, destination, workspace)
    }

    fn construct(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
        input: Result<Arc<Pencil<N, M>>, FftError>,
    ) -> Result<Self, FftError> {
        let communicator = topology.communicator();
        // This route stage deliberately performs no native FFT planning. In
        // particular, a zero global extent is rejected here on every rank
        // before any LocalC2cPlan reaches RustFFT.
        let route = build_route(input, &topology, global_shape);
        let expected_len = descriptor_len::<N, M>(&extra_shape);
        let descriptor = expected_len
            .and_then(|_| build_descriptor::<R, N, M>(&topology, global_shape, &extra_shape).ok());
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
        let descriptor = collective_descriptor(communicator, descriptor, expected_len)?;
        let route = agree_result(communicator, route)?;
        let stages = prepare_stages::<R, N, M>(&route, global_shape);
        let stages = agree_result(communicator, stages)?;

        let (transitions, transpose_send_len, transpose_receive_len) =
            build_transitions::<R, N, M>(communicator, &stages, &route.distributed, &extra_shape)?;

        let core = C2cPlanCore {
            stages: stages.stages,
            transitions: transitions.into_boxed_slice(),
            extra_shape,
            descriptor: descriptor.into_boxed_slice(),
            fft_scratch_len: stages.fft_scratch_len,
            transpose_send_len,
            transpose_receive_len,
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
    ) -> Result<(), FftError> {
        let communicator = self.input_pencil().topology().communicator();
        let operation = match direction {
            Direction::Forward => OPERATION_FORWARD,
            Direction::Inverse => OPERATION_INVERSE,
        };
        let descriptor = self.core.descriptor.as_ref();
        let descriptor_len_word = u64::try_from(descriptor.len()).unwrap_or(INVALID_WORD);
        let header = [
            DESCRIPTOR_SCHEMA,
            operation,
            u64::try_from(N).unwrap_or(INVALID_WORD),
            u64::try_from(M).unwrap_or(INVALID_WORD),
            descriptor_len_word,
        ];
        if !agree_header(communicator, header) {
            return Err(FftError::CollectiveDescriptorMismatch);
        }
        collective_descriptor_ref(communicator, Some(descriptor), Some(descriptor.len()))?;

        let local_preflight = self.preflight(direction, source, destination, workspace);
        if !collective_valid(communicator, local_preflight.is_ok()) {
            return Err(local_preflight
                .err()
                .unwrap_or(FftError::CollectivePreconditionFailed));
        }
        local_preflight.expect("collective distributed C2C preflight succeeded");

        match direction {
            Direction::Forward => execute_forward(&self.core, source, destination, workspace),
            Direction::Inverse => execute_inverse(&self.core, source, destination, workspace),
        }
    }

    fn preflight(
        &self,
        direction: Direction,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &PencilArray<Complex<R>, N, M>,
        workspace: &C2cOutOfPlaceWorkspace<R, N, M>,
    ) -> Result<(), FftError> {
        if !Arc::ptr_eq(&workspace.core, &self.core) {
            return Err(FftError::WorkspaceMismatch);
        }
        let (expected_source, expected_destination) = match direction {
            Direction::Forward => (self.input_pencil(), self.output_pencil()),
            Direction::Inverse => (self.output_pencil(), self.input_pencil()),
        };
        if !source.pencil().same_layout(expected_source.as_ref()) {
            return Err(FftError::InputLayoutMismatch);
        }
        if !destination
            .pencil()
            .same_layout(expected_destination.as_ref())
        {
            return Err(FftError::OutputLayoutMismatch);
        }
        if source.extra_shape() != &self.core.extra_shape
            || destination.extra_shape() != &self.core.extra_shape
        {
            return Err(FftError::ExtraShapeMismatch);
        }

        if workspace.fft_scratch.len() < self.core.fft_scratch_len {
            return Err(FftError::WorkspaceTooSmall {
                kind: "FFT scratch",
                required: self.core.fft_scratch_len,
                actual: workspace.fft_scratch.len(),
            });
        }
        if workspace.transpose.send_len() < self.core.transpose_send_len {
            return Err(FftError::WorkspaceTooSmall {
                kind: "transpose send",
                required: self.core.transpose_send_len,
                actual: workspace.transpose.send_len(),
            });
        }
        if workspace.transpose.receive_len() < self.core.transpose_receive_len {
            return Err(FftError::WorkspaceTooSmall {
                kind: "transpose receive",
                required: self.core.transpose_receive_len,
                actual: workspace.transpose.receive_len(),
            });
        }

        if workspace.intermediate.extra_shape() != &self.core.extra_shape {
            return Err(FftError::WorkspaceMismatch);
        }
        let active = workspace.intermediate.active_pencil()?;
        if !self
            .core
            .stages
            .iter()
            .any(|stage| active.same_layout(stage.pencil.as_ref()))
        {
            return Err(FftError::WorkspaceMismatch);
        }
        Ok(())
    }
}

fn build_route<const N: usize, const M: usize>(
    input: Result<Arc<Pencil<N, M>>, FftError>,
    topology: &Arc<MpiTopology<M>>,
    global_shape: [usize; N],
) -> Result<RouteCandidate<N, M>, FftError> {
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
        let next_permutation = AxisPermutation::new(next_axes)
            .map_err(PencilError::InvalidPermutation)
            .map_err(FftError::Pencil)?;
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
    let expected_permutation = std::array::from_fn(|position| N - 1 - position);
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

fn prepare_stages<R: FftReal, const N: usize, const M: usize>(
    route: &RouteCandidate<N, M>,
    global_shape: [usize; N],
) -> Result<StagePreparation<R, N, M>, FftError> {
    let mut stages = Vec::new();
    stages
        .try_reserve_exact(route.stages.len())
        .map_err(|_| FftError::AllocationFailed {
            required: route.stages.len(),
        })?;
    let mut fft_scratch_len = 0usize;
    for (index, pencil) in route.stages.iter().enumerate() {
        let axis_index = N - 1 - index;
        if pencil.permutation().axes()[N - 1].index() != axis_index
            || pencil
                .decomposition()
                .iter()
                .any(|distributed| distributed.index() == axis_index)
            || pencil.local_shape_logical()[axis_index] != global_shape[axis_index]
        {
            return Err(FftError::PreparationFailed);
        }
        let local = LocalC2cPlan::new(global_shape[axis_index])?;
        fft_scratch_len = fft_scratch_len.max(local.scratch_len());
        stages.push(C2cStage {
            pencil: Arc::clone(pencil),
            local,
        });
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
) -> Result<(Vec<C2cStageTransition<N, M>>, usize, usize), FftError> {
    let mut transitions = Vec::new();
    agree_result(
        communicator,
        transitions
            .try_reserve_exact(N - 1)
            .map_err(|_| FftError::AllocationFailed { required: N - 1 }),
    )?;

    let mut transpose_send_len = 0usize;
    let mut transpose_receive_len = 0usize;
    let extra_count = extra_shape.element_count();

    for (index, &is_distributed) in distributed.iter().enumerate() {
        let source = Arc::clone(&stages.stages[index].pencil);
        let destination = Arc::clone(&stages.stages[index + 1].pencil);
        if is_distributed {
            let forward =
                AllToAllvTransposePlan::new(Arc::clone(&source), Arc::clone(&destination))
                    .map_err(FftError::Transpose)?;
            let forward_requirements = agree_result(
                communicator,
                forward
                    .workspace_requirements(extra_shape)
                    .map_err(FftError::Transpose),
            )?;
            let backward =
                AllToAllvTransposePlan::new(Arc::clone(&destination), Arc::clone(&source))
                    .map_err(FftError::Transpose)?;
            let backward_requirements = agree_result(
                communicator,
                backward
                    .workspace_requirements(extra_shape)
                    .map_err(FftError::Transpose),
            )?;
            transpose_send_len = transpose_send_len
                .max(forward_requirements.send_len)
                .max(backward_requirements.send_len);
            transpose_receive_len = transpose_receive_len
                .max(forward_requirements.receive_len)
                .max(backward_requirements.receive_len);
            transitions.push(C2cStageTransition {
                forward: C2cTransition::AllToAllv(forward),
                backward: C2cTransition::AllToAllv(backward),
            });
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
            transpose_send_len = transpose_send_len.max(send_len);
            transitions.push(transition);
        }
    }

    Ok((transitions, transpose_send_len, transpose_receive_len))
}

fn execute_forward<R: FftReal, const N: usize, const M: usize>(
    core: &Arc<C2cPlanCore<R, N, M>>,
    source: &PencilArray<Complex<R>, N, M>,
    destination: &mut PencilArray<Complex<R>, N, M>,
    workspace: &mut C2cOutOfPlaceWorkspace<R, N, M>,
) -> Result<(), FftError>
where
    Complex<R>: Equivalence,
{
    let source_view = source.view();
    {
        let stage = &core.stages[0];
        let fft_scratch = &mut workspace.fft_scratch;
        workspace
            .intermediate
            .overwrite_with(stage.pencil.as_ref(), |mut target| {
                stage
                    .local
                    .forward(source_view.as_slice(), target.as_mut_slice(), fft_scratch)
                    .expect("distributed C2C forward preflight validated stage zero");
                Ok::<_, ()>(())
            })
            .expect("distributed C2C stage-zero overwrite was preflighted");
    }

    for index in 0..N - 1 {
        execute_transition(
            &core.transitions[index].forward,
            &mut workspace.intermediate,
            &mut workspace.transpose,
        )?;
        if index + 1 == N - 1 {
            let stage = &core.stages[index + 1];
            let active = workspace
                .intermediate
                .active_view()
                .expect("distributed C2C final active layout was preflighted");
            let mut destination_view = destination.view_mut();
            stage
                .local
                .forward(
                    active.as_slice(),
                    destination_view.as_mut_slice(),
                    &mut workspace.fft_scratch,
                )
                .expect("distributed C2C final forward stage was preflighted");
        } else {
            let stage = &core.stages[index + 1];
            let mut active = workspace
                .intermediate
                .active_view_mut()
                .expect("distributed C2C middle active layout was preflighted");
            stage
                .local
                .forward_in_place(active.as_mut_slice(), &mut workspace.fft_scratch)
                .expect("distributed C2C middle forward stage was preflighted");
        }
    }
    Ok(())
}

fn execute_inverse<R: FftReal, const N: usize, const M: usize>(
    core: &Arc<C2cPlanCore<R, N, M>>,
    source: &PencilArray<Complex<R>, N, M>,
    destination: &mut PencilArray<Complex<R>, N, M>,
    workspace: &mut C2cOutOfPlaceWorkspace<R, N, M>,
) -> Result<(), FftError>
where
    Complex<R>: Equivalence,
{
    let source_view = source.view();
    {
        let stage = &core.stages[N - 1];
        let fft_scratch = &mut workspace.fft_scratch;
        workspace
            .intermediate
            .overwrite_with(stage.pencil.as_ref(), |mut target| {
                stage
                    .local
                    .inverse(source_view.as_slice(), target.as_mut_slice(), fft_scratch)
                    .expect("distributed C2C inverse preflight validated final stage");
                Ok::<_, ()>(())
            })
            .expect("distributed C2C inverse stage-zero overwrite was preflighted");
    }

    for index in (0..N - 1).rev() {
        execute_transition(
            &core.transitions[index].backward,
            &mut workspace.intermediate,
            &mut workspace.transpose,
        )?;
        if index == 0 {
            let stage = &core.stages[0];
            let active = workspace
                .intermediate
                .active_view()
                .expect("distributed C2C inverse final active layout was preflighted");
            let mut destination_view = destination.view_mut();
            stage
                .local
                .inverse(
                    active.as_slice(),
                    destination_view.as_mut_slice(),
                    &mut workspace.fft_scratch,
                )
                .expect("distributed C2C final inverse stage was preflighted");
        } else {
            let stage = &core.stages[index];
            let mut active = workspace
                .intermediate
                .active_view_mut()
                .expect("distributed C2C inverse middle active layout was preflighted");
            stage
                .local
                .inverse_in_place(active.as_mut_slice(), &mut workspace.fft_scratch)
                .expect("distributed C2C middle inverse stage was preflighted");
        }
    }
    Ok(())
}

fn execute_transition<T: Equivalence + Copy + Clone, const N: usize, const M: usize>(
    transition: &C2cTransition<N, M>,
    intermediate: &mut ManyPencilArray<T, N, M>,
    workspace: &mut TransposeWorkspace<T>,
) -> Result<(), FftError> {
    match transition {
        C2cTransition::Local(plan) => {
            plan.execute_in_place_with_transpose_workspace(intermediate, workspace)
                .expect("distributed C2C local transition was preflighted");
            Ok(())
        }
        C2cTransition::AllToAllv(plan) => plan
            .execute_in_place(intermediate, workspace)
            .map_err(FftError::Transpose),
    }
}

fn descriptor_len<const N: usize, const M: usize>(extra_shape: &ExtraShape) -> Option<usize> {
    N.checked_add(M)?
        .checked_add(2)?
        .checked_add(extra_shape.dimensions().len())
}

fn build_descriptor<R: FftReal, const N: usize, const M: usize>(
    topology: &MpiTopology<M>,
    global_shape: [usize; N],
    extra_shape: &ExtraShape,
) -> Result<Vec<u64>, ()> {
    let length = descriptor_len::<N, M>(extra_shape).ok_or(())?;
    let mut descriptor = Vec::new();
    descriptor.try_reserve_exact(length).map_err(|_| ())?;
    append_usizes(&mut descriptor, &global_shape)?;
    append_usizes(&mut descriptor, topology.process_grid())?;
    append_shape(&mut descriptor, extra_shape)?;
    descriptor.push(u64::try_from(size_of::<R>()).map_err(|_| ())?);
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

fn collective_valid<C: CommunicatorCollectives>(comm: &C, valid: bool) -> bool {
    let value = i32::from(valid);
    let mut result = 0i32;
    comm.all_reduce_into(&value, &mut result, SystemOperation::min());
    result != 0
}

fn agree_result<C: CommunicatorCollectives, T>(
    comm: &C,
    result: Result<T, FftError>,
) -> Result<T, FftError> {
    if !collective_valid(comm, result.is_ok()) {
        return Err(result
            .err()
            .unwrap_or(FftError::CollectivePreconditionFailed));
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
    use super::{OPERATION_FORWARD, OPERATION_INVERSE, OPERATION_PLAN, descriptor_len};
    use pencil_array::ExtraShape;

    #[test]
    fn protocol_words_and_minimal_descriptor_length_are_stable() {
        assert_eq!(
            (OPERATION_PLAN, OPERATION_FORWARD, OPERATION_INVERSE),
            (7, 8, 9)
        );
        let scalar_len = descriptor_len::<2, 1>(&ExtraShape::scalar()).unwrap();
        let batched_len = descriptor_len::<4, 2>(&ExtraShape::new([2, 3]).unwrap()).unwrap();
        assert_eq!(scalar_len, 5);
        assert_eq!(batched_len, 10);
        assert_eq!(
            batched_len - descriptor_len::<4, 2>(&ExtraShape::scalar()).unwrap(),
            2,
        );
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
