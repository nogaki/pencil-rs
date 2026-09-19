//! Distributed real-to-half-complex and half-complex-to-real transforms.

use std::{cmp::Ordering, sync::Arc};

use mpi::{
    collective::{CommunicatorCollectives, SystemOperation},
    datatype::Equivalence,
};
use pencil_array::{
    ExtraShape, ManyPencilArray, MpiTopology, OverwriteError, Pencil, PencilArray,
    TransposeWorkspace,
};

use super::{
    AxisSelection, DESCRIPTOR_SCHEMA, Direction, FftError, INVALID_WORD, LocalTransform,
    OPERATION_R2C_BACKWARD, OPERATION_R2C_FORWARD, OPERATION_R2C_INVERSE, OPERATION_R2C_PLAN,
    R2cError, StagePreparation, TransformPlanCore, TransformStage, TransposeMethod, VALUE_KIND_R2C,
    agree_execution_descriptor_ref, agree_header, agree_result, build_descriptor, build_route,
    build_transitions, collective_valid, descriptor_len, initialized_vec, map_array_allocation,
    prepare_complex_stage, registered_stage_pencils, validate_out_of_place,
    validate_workspace_lengths_values, zero_complex,
};
use crate::{Complex, FftReal, LocalR2cError, LocalR2cPlan};

/// An immutable, checked distributed real-to-half-complex FFT plan.
///
/// This API requires `N >= 2` and `1 <= M < N`. The input uses identity
/// permutation and decomposition `[0..M)` with the original real global shape.
/// A non-empty [`AxisSelection`] chooses the largest selected axis as the
/// real-to-half-complex boundary. Its extent `n` becomes `n / 2 + 1`; selected
/// lower axes are complex stages and unselected axes are identity stages. The
/// route still contains all `N` stages and `N - 1` transitions, including
/// real-prefix transposes before a non-last boundary. The final output uses
/// decomposition `[1..=M]` and reversed spatial memory order. Empty selections
/// are collectively rejected before native planning.
///
/// Constructors and `forward`/`inverse`/`backward` are collective. Every rank must use
/// the same communicator context, API and call order, scalar type, transport
/// method, and matching layouts. The legacy constructors select
/// [`TransposeMethod::AllToAllv`]; the `_with_method` constructors can select
/// [`TransposeMethod::PointToPoint`]. Point-to-point uses the existing fixed
/// `0x5054` tag and changed-axis context, so unfinished transposes must not
/// overlap on that context. Allocation methods are noncollective; callers
/// must coordinate a local allocation failure before the next collective.
/// There is intentionally no distributed real in-place API.
///
/// `inverse` is normalized by the product of the selected spatial extents;
/// `backward` uses the same positive-sign C2R transform without that
/// normalization. Both reverse operations preserve their complex source.
/// Only selected spatial axes contribute to normalization. Backward endpoint
/// acceptance uses the same relative threshold as inverse and an absolute
/// threshold multiplied by the product of selected complex-tail extents;
/// extra dimensions and identity axes do not contribute.
///
/// # Example
///
/// MPI is initialized once, and all topology, plan, array, and workspace
/// values are dropped before the universe is dropped.
///
/// ```
/// use mpi::traits::*;
/// use pencil_array::{ExtraShape, MpiTopology};
/// use pencil_fft::R2cPlan;
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let universe = mpi::initialize().expect("MPI must not already be initialized");
///     let world = universe.world();
///     let result = {
///         let world_size = usize::try_from(world.size())?;
///         let topology = MpiTopology::<1>::new(&world, [world_size])?;
///         let plan = R2cPlan::<f64, 2, 1>::from_shape(
///             topology,
///             [2 * world_size, 5],
///             ExtraShape::scalar(),
///         )?;
///         assert_eq!(plan.input_pencil().global_shape(), &[2 * world_size, 5]);
///         assert_eq!(plan.output_pencil().global_shape(), &[2 * world_size, 3]);
///
///         let mut input = plan.allocate_input()?;
///         for (index, value) in input.as_mut_slice().iter_mut().enumerate() {
///             *value = index as f64 + 1.0;
///         }
///         let original = input.as_slice().to_vec();
///         let mut spectrum = plan.allocate_output()?;
///         let mut recovered = plan.allocate_input()?;
///         let mut workspace = plan.allocate_workspace()?;
///
///         plan.forward(&input, &mut spectrum, &mut workspace)?;
///         assert_eq!(input.as_slice(), original.as_slice());
///         let spectrum_before = spectrum.as_slice().to_vec();
///         plan.inverse(&spectrum, &mut recovered, &mut workspace)?;
///         assert_eq!(spectrum.as_slice(), spectrum_before.as_slice());
///         for (actual, expected) in recovered.as_slice().iter().zip(&original) {
///             assert!((actual - expected).abs() < 1e-9);
///         }
///         Ok::<(), Box<dyn std::error::Error>>(())
///     };
///     result
/// }
/// ```
///
/// `inverse` first performs the transverse complex inverse stages and then
/// validates each extra batch and each constrained endpoint plane. For a
/// plane `z(x)`, all endpoint values must be finite, and it is accepted when
/// `max_x abs(Im z(x)) <= absolute_R` or
/// `||Im z||_2 <= relative_R * ||Re z||_2`, with the exact fixed policy
///
/// ```text
/// D = 1 + sum over selected complex-tail axes a of ceil(log2(n_a))
/// relative_R = 128 * epsilon_R * D
/// absolute_R = 128 * min_subnormal_R * D
/// ```
///
/// Here `n_a` are the original extents of selected spatial axes below the
/// real axis, and a length-one axis contributes zero. DC is constrained for every `n`; the
/// Nyquist plane is additionally constrained only for even `n`. For odd
/// `n > 1`, the final complex bin is not constrained; for `n == 1` it is DC
/// and is constrained. Signed zero is accepted. This is an explicit
/// normwise-relative/componentwise-absolute policy, not a formal RustFFT
/// error bound. Accepted endpoint imaginary values are projected to zero only
/// in the private intermediate workspace before the strict local C2R call;
/// the complex source is never projected. Interior bins have no blanket
/// finite-or-real requirement.
///
/// `backward` uses the same endpoint policy, except that its absolute
/// threshold is `absolute_inverse * product(selected complex-tail n_a)`; the
/// factor is finite and positive-validated collectively while constructing the
/// plan.
///
/// Descriptor and initial preflight errors are collectively returned before
/// source, destination, or workspace writes and preserve all three. A
/// materially invalid endpoint returns [`R2cError::InvalidSpectrum`] after
/// the transverse stages have used the workspace: the source and real
/// destination remain unchanged, but workspace mutation is permitted. The
/// inverse is normalized by the product of selected spatial extents;
/// identity axes, extra dimensions, and the reduced complex extent are not
/// normalization factors.
///
/// ```compile_fail
/// use pencil_fft::R2cPlan;
///
/// let _ = R2cPlan::<f64, 2, 1>::forward_in_place;
/// ```
///
/// Existing generic callers only need their original bounds:
///
/// ```
/// use std::sync::Arc;
/// use mpi::datatype::Equivalence;
/// use pencil_array::{ExtraShape, MpiTopology};
/// use pencil_fft::{Complex, FftReal, R2cPlan};
///
/// fn old_generic<R, const N: usize, const M: usize>(
///     topology: Arc<MpiTopology<M>>,
///     shape: [usize; N],
/// ) where
///     R: FftReal,
///     Complex<R>: Equivalence,
/// {
///     let _ = R2cPlan::<R, N, M>::from_shape(topology, shape, ExtraShape::scalar());
/// }
/// ```
#[derive(Debug)]
pub struct R2cPlan<R: FftReal, const N: usize, const M: usize> {
    core: Arc<TransformPlanCore<R, N, M>>,
    raw_absolute_threshold: f64,
}

/// Reusable storage for [`R2cPlan`] forward, inverse, and backward execution.
///
/// The workspace is private to the exact plan that allocated it. It contains
/// one registered reduced-complex intermediate, shared transpose buffers,
/// native FFT scratch, and one line buffer for each local real/complex native
/// operation.
#[derive(Debug)]
pub struct R2cWorkspace<R: FftReal, const N: usize, const M: usize> {
    core: Arc<TransformPlanCore<R, N, M>>,
    intermediate: ManyPencilArray<Complex<R>, N, M>,
    transpose: TransposeWorkspace<Complex<R>>,
    real_intermediate: Option<ManyPencilArray<R, N, M>>,
    real_transpose: Option<TransposeWorkspace<R>>,
    fft_scratch: Vec<Complex<R>>,
    real_line: Vec<R>,
    complex_line: Vec<Complex<R>>,
}

impl<R: FftReal, const N: usize, const M: usize> R2cPlan<R, N, M>
where
    Complex<R>: Equivalence,
{
    /// Collectively builds an Alltoallv plan from a canonical real pencil.
    pub fn from_pencil(
        input: Arc<Pencil<N, M>>,
        extra_shape: ExtraShape,
    ) -> Result<Self, R2cError> {
        Self::from_pencil_with_selection_and_method(
            input,
            extra_shape,
            AxisSelection::all(),
            TransposeMethod::AllToAllv,
        )
    }

    /// Collectively builds an Alltoallv plan for a validated axis selection.
    pub fn from_pencil_with_selection(
        input: Arc<Pencil<N, M>>,
        extra_shape: ExtraShape,
        selection: AxisSelection<N>,
    ) -> Result<Self, R2cError> {
        Self::from_pencil_with_selection_and_method(
            input,
            extra_shape,
            selection,
            TransposeMethod::AllToAllv,
        )
    }

    /// Collectively builds a plan from a canonical real pencil and selection.
    pub fn from_pencil_with_selection_and_method(
        input: Arc<Pencil<N, M>>,
        extra_shape: ExtraShape,
        selection: AxisSelection<N>,
        method: TransposeMethod,
    ) -> Result<Self, R2cError> {
        let topology = Arc::clone(input.topology());
        let global_shape = *input.global_shape();
        Self::construct(
            topology,
            global_shape,
            extra_shape,
            Ok(input),
            selection,
            method,
        )
    }

    /// Collectively builds a plan from a canonical real pencil and transport.
    pub fn from_pencil_with_method(
        input: Arc<Pencil<N, M>>,
        extra_shape: ExtraShape,
        method: TransposeMethod,
    ) -> Result<Self, R2cError> {
        Self::from_pencil_with_selection_and_method(
            input,
            extra_shape,
            AxisSelection::all(),
            method,
        )
    }

    /// Collectively builds an Alltoallv plan from a canonical real array.
    pub fn from_array(input: &PencilArray<R, N, M>) -> Result<Self, R2cError> {
        Self::from_array_with_selection_and_method(
            input,
            AxisSelection::all(),
            TransposeMethod::AllToAllv,
        )
    }

    /// Collectively builds an Alltoallv plan for a validated axis selection.
    pub fn from_array_with_selection(
        input: &PencilArray<R, N, M>,
        selection: AxisSelection<N>,
    ) -> Result<Self, R2cError> {
        Self::from_array_with_selection_and_method(input, selection, TransposeMethod::AllToAllv)
    }

    /// Collectively builds a plan from a canonical real array and selection.
    pub fn from_array_with_selection_and_method(
        input: &PencilArray<R, N, M>,
        selection: AxisSelection<N>,
        method: TransposeMethod,
    ) -> Result<Self, R2cError> {
        let topology = Arc::clone(input.pencil().topology());
        let global_shape = *input.pencil().global_shape();
        Self::construct(
            topology,
            global_shape,
            input.extra_shape().clone(),
            Ok(Arc::clone(input.pencil())),
            selection,
            method,
        )
    }

    /// Collectively builds a plan from a canonical real array and transport.
    pub fn from_array_with_method(
        input: &PencilArray<R, N, M>,
        method: TransposeMethod,
    ) -> Result<Self, R2cError> {
        Self::from_array_with_selection_and_method(input, AxisSelection::all(), method)
    }

    /// Collectively builds an Alltoallv plan from topology and shape.
    pub fn from_shape(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
    ) -> Result<Self, R2cError> {
        Self::from_shape_with_selection_and_method(
            topology,
            global_shape,
            extra_shape,
            AxisSelection::all(),
            TransposeMethod::AllToAllv,
        )
    }

    /// Collectively builds a plan from topology, shape, extra dimensions, and
    /// the selected checked transition transport.
    pub fn from_shape_with_method(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
        method: TransposeMethod,
    ) -> Result<Self, R2cError> {
        Self::from_shape_with_selection_and_method(
            topology,
            global_shape,
            extra_shape,
            AxisSelection::all(),
            method,
        )
    }

    /// Collectively builds an Alltoallv plan for a validated axis selection.
    pub fn from_shape_with_selection(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
        selection: AxisSelection<N>,
    ) -> Result<Self, R2cError> {
        Self::from_shape_with_selection_and_method(
            topology,
            global_shape,
            extra_shape,
            selection,
            TransposeMethod::AllToAllv,
        )
    }

    /// Collectively builds a plan from topology, shape, selection, and method.
    pub fn from_shape_with_selection_and_method(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
        selection: AxisSelection<N>,
        method: TransposeMethod,
    ) -> Result<Self, R2cError> {
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
            method,
        )
    }

    /// Returns the canonical real input pencil.
    pub fn input_pencil(&self) -> &Arc<Pencil<N, M>> {
        &self.core.stages[0].input
    }

    /// Returns the reduced complex output pencil.
    pub fn output_pencil(&self) -> &Arc<Pencil<N, M>> {
        &self
            .core
            .stages
            .last()
            .expect("distributed R2C has at least two stages")
            .output
    }

    /// Returns the exact extra shape required by this plan.
    pub fn extra_shape(&self) -> &ExtraShape {
        &self.core.extra_shape
    }

    /// Allocates a zero-initialized local real input array.
    pub fn allocate_input(&self) -> Result<PencilArray<R, N, M>, R2cError> {
        Ok(PencilArray::from_fn(
            Arc::clone(self.input_pencil()),
            self.core.extra_shape.clone(),
            R::zero,
        )
        .map_err(FftError::Array)?)
    }

    /// Allocates a zero-initialized local reduced-complex output array.
    pub fn allocate_output(&self) -> Result<PencilArray<Complex<R>, N, M>, R2cError> {
        Ok(PencilArray::from_fn(
            Arc::clone(self.output_pencil()),
            self.core.extra_shape.clone(),
            zero_complex::<R>,
        )
        .map_err(FftError::Array)?)
    }

    /// Allocates reusable noncollective forward/inverse/backward workspace.
    pub fn allocate_workspace(&self) -> Result<R2cWorkspace<R, N, M>, R2cError> {
        let boundary = self
            .core
            .real_stage_index
            .expect("R2C core has a real boundary stage");
        let (real_len, complex_len) = r2c_lengths(&self.core);
        let intermediate = ManyPencilArray::from_elem(
            registered_stage_pencils(&self.core.stages[boundary..])?,
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
                    registered_real_stage_pencils(&self.core.stages, boundary)?,
                    0,
                    self.core.extra_shape.clone(),
                    R::zero(),
                )
                .map_err(map_array_allocation)?,
            )
        };
        let transpose = TransposeWorkspace::from_vecs(
            initialized_vec(self.core.transpose_send_len, zero_complex::<R>())?,
            initialized_vec(self.core.transpose_receive_len, zero_complex::<R>())?,
        );
        let real_transpose = if boundary == 0 {
            None
        } else {
            Some(TransposeWorkspace::from_vecs(
                initialized_vec(self.core.real_transpose_send_len, R::zero())?,
                initialized_vec(self.core.real_transpose_receive_len, R::zero())?,
            ))
        };
        let fft_scratch = initialized_vec(self.core.fft_scratch_len, zero_complex::<R>())?;
        let real_line = initialized_vec(real_len, R::zero())?;
        let complex_line = initialized_vec(complex_len, zero_complex::<R>())?;
        Ok(R2cWorkspace {
            core: Arc::clone(&self.core),
            intermediate,
            transpose,
            real_intermediate,
            real_transpose,
            fft_scratch,
            real_line,
            complex_line,
        })
    }

    /// Computes an unnormalized forward real-to-half-complex transform.
    pub fn forward(
        &self,
        source: &PencilArray<R, N, M>,
        destination: &mut PencilArray<Complex<R>, N, M>,
        workspace: &mut R2cWorkspace<R, N, M>,
    ) -> Result<(), R2cError> {
        self.execute_forward(source, destination, workspace)
    }

    /// Computes a normalized inverse half-complex-to-real transform.
    ///
    /// After the transverse inverse stages, each extra batch and constrained
    /// DC/Nyquist plane is checked. Every endpoint value must be finite. With
    /// original shape `n_a`,
    ///
    /// ```text
    /// D = 1 + sum over selected complex-tail axes a of ceil(log2(n_a))
    /// relative_R = 128 * epsilon_R * D
    /// absolute_R = 128 * min_subnormal_R * D
    /// ```
    ///
    /// A plane `z(x)` is accepted when
    /// `max_x abs(Im z(x)) <= absolute_R` or
    /// `||Im z||_2 <= relative_R * ||Re z||_2`. DC is always constrained;
    /// Nyquist is constrained only for even `n`. The final odd bin is
    /// unconstrained for odd `n > 1`, while `n == 1` has only constrained DC.
    /// This is an explicit normwise-relative/componentwise-absolute policy,
    /// not a formal RustFFT error bound. Accepted endpoint imaginary values
    /// are zeroed only in the private intermediate workspace; interior bins
    /// have no blanket finite policy. Initial descriptor/preflight errors
    /// preserve source, destination, and workspace. Invalid boundaries return
    /// [`R2cError::InvalidSpectrum`] after tail workspace use, preserving the
    /// source and destination but not promising unchanged workspace.
    pub fn inverse(
        &self,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &mut PencilArray<R, N, M>,
        workspace: &mut R2cWorkspace<R, N, M>,
    ) -> Result<(), R2cError> {
        self.execute_reverse(source, destination, workspace, true)
    }

    /// Computes an unnormalized positive-sign backward half-complex-to-real
    /// transform.
    ///
    /// The source uses the reduced complex output layout and the destination
    /// uses the canonical real input layout. Unlike [`Self::inverse`], this
    /// method does not normalize by the product of the selected spatial
    /// extents, so a forward/backward pair scales by that product. Identity
    /// axes, extra dimensions, and the reduced real-axis extent are not
    /// factors. Initial
    /// descriptor and preflight errors preserve source, destination, and
    /// workspace; a post-tail invalid boundary preserves source and
    /// destination while the workspace may already have changed.
    pub fn backward(
        &self,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &mut PencilArray<R, N, M>,
        workspace: &mut R2cWorkspace<R, N, M>,
    ) -> Result<(), R2cError> {
        self.execute_reverse(source, destination, workspace, false)
    }

    fn construct(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
        input: Result<Arc<Pencil<N, M>>, FftError>,
        selection: AxisSelection<N>,
        method: TransposeMethod,
    ) -> Result<Self, R2cError> {
        let communicator = topology.communicator();
        let expected_len = descriptor_len::<N, M>(&extra_shape);
        let descriptor = expected_len.and_then(|_| {
            build_descriptor::<R, N, M>(
                &topology,
                global_shape,
                &extra_shape,
                selection,
                VALUE_KIND_R2C,
                method,
            )
            .ok()
        });
        let descriptor_len_word = expected_len
            .and_then(|length| u64::try_from(length).ok())
            .unwrap_or(INVALID_WORD);
        let header = [
            DESCRIPTOR_SCHEMA,
            OPERATION_R2C_PLAN,
            u64::try_from(N).unwrap_or(INVALID_WORD),
            u64::try_from(M).unwrap_or(INVALID_WORD),
            descriptor_len_word,
        ];
        if !agree_header(communicator, header) {
            return Err(R2cError::Fft(FftError::CollectiveDescriptorMismatch));
        }
        let descriptor = super::collective_descriptor(communicator, descriptor, expected_len)?;
        if !collective_valid(communicator, !selection.is_empty()) {
            return Err(R2cError::Fft(FftError::InvalidDimensions));
        }
        let reduction_axis = selection
            .mask()
            .iter()
            .rposition(|&selected| selected)
            .expect("collectively accepted R2C selection is nonempty");
        let boundary = N - 1 - reduction_axis;
        let input_pencil = agree_result(
            communicator,
            super::validate_input(input, &topology, global_shape),
        )?;
        let raw_absolute_threshold = agree_result(
            communicator,
            raw_absolute_threshold_for_shape::<R, N>(global_shape, selection, reduction_axis)
                .map_err(R2cError::Fft),
        )?;
        let real_len = global_shape[reduction_axis];
        let complex_len = real_len / 2 + 1;
        let mut reduced_shape = global_shape;
        reduced_shape[reduction_axis] = complex_len;

        let original_route = agree_result(
            communicator,
            build_route(Ok(Arc::clone(&input_pencil)), &topology, global_shape),
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
            build_route(Ok(Arc::clone(&reduced_input)), &topology, reduced_shape),
        )?;
        let stages = agree_result(
            communicator,
            prepare_r2c_stages::<R, N, M>(
                &original_route,
                &reduced_route,
                global_shape,
                reduced_shape,
                reduction_axis,
                selection,
                real_len,
            ),
        )?;
        if original_route.distributed.len() != reduced_route.distributed.len()
            || original_route.distributed.len() != N.saturating_sub(1)
        {
            return Err(R2cError::Fft(FftError::PreparationFailed));
        }
        let transition_count = original_route.distributed.len();
        let mut distributed = Vec::new();
        agree_result(
            communicator,
            distributed
                .try_reserve_exact(transition_count)
                .map_err(|_| FftError::AllocationFailed {
                    required: transition_count,
                }),
        )?;
        for index in 0..transition_count {
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
        ) = build_transitions::<R, N, M>(
            communicator,
            &stages,
            &distributed,
            &extra_shape,
            method,
            boundary,
        )?;
        let core = TransformPlanCore {
            stages: stages.stages,
            transitions: transitions.into_boxed_slice(),
            extra_shape,
            selection,
            real_stage_index: Some(boundary),
            descriptor: descriptor.into_boxed_slice(),
            fft_scratch_len: stages.fft_scratch_len,
            transpose_send_len,
            transpose_receive_len,
            real_transpose_send_len,
            real_transpose_receive_len,
        };
        Ok(Self {
            core: Arc::new(core),
            raw_absolute_threshold,
        })
    }
}

fn prepare_r2c_stages<R: FftReal, const N: usize, const M: usize>(
    original_route: &super::RouteCandidate<N, M>,
    reduced_route: &super::RouteCandidate<N, M>,
    original_shape: [usize; N],
    reduced_shape: [usize; N],
    reduction_axis: usize,
    selection: AxisSelection<N>,
    real_len: usize,
) -> Result<StagePreparation<R, N, M>, R2cError> {
    let boundary = N - 1 - reduction_axis;
    if original_route.stages.len() != N || reduced_route.stages.len() != N {
        return Err(R2cError::Fft(FftError::PreparationFailed));
    }
    let mut stages = Vec::new();
    stages
        .try_reserve_exact(N)
        .map_err(|_| FftError::AllocationFailed { required: N })?;
    let mut fft_scratch_len = 0usize;
    for index in 0..N {
        let axis = N - 1 - index;
        let stage = match index.cmp(&boundary) {
            Ordering::Less => {
                prepare_complex_stage(&original_route.stages[index], original_shape, axis, false)?
            }
            Ordering::Equal => {
                let pencil = &original_route.stages[index];
                if pencil.permutation().axes()[N - 1].index() != reduction_axis
                    || pencil
                        .decomposition()
                        .iter()
                        .any(|distributed| distributed.index() == reduction_axis)
                    || pencil.local_shape_logical()[reduction_axis] != real_len
                {
                    return Err(R2cError::Fft(FftError::PreparationFailed));
                }
                let real = LocalR2cPlan::new(real_len).map_err(R2cError::LocalR2c)?;
                TransformStage {
                    input: Arc::clone(&original_route.stages[index]),
                    output: Arc::clone(&reduced_route.stages[index]),
                    local: LocalTransform::RealComplex(real),
                }
            }
            Ordering::Greater => prepare_complex_stage(
                &reduced_route.stages[index],
                reduced_shape,
                axis,
                selection.contains(axis),
            )?,
        };
        fft_scratch_len = fft_scratch_len.max(stage.local.scratch_len());
        stages.push(stage);
    }
    Ok(StagePreparation {
        stages: stages.into_boxed_slice(),
        fft_scratch_len,
    })
}

fn registered_real_stage_pencils<R: FftReal, const N: usize, const M: usize>(
    stages: &[TransformStage<R, N, M>],
    boundary: usize,
) -> Result<Box<[Arc<Pencil<N, M>>]>, FftError> {
    let count = boundary.checked_add(1).ok_or(FftError::PreparationFailed)?;
    if count > stages.len() {
        return Err(FftError::PreparationFailed);
    }
    let mut pencils = Vec::new();
    pencils
        .try_reserve_exact(count)
        .map_err(|_| FftError::AllocationFailed { required: count })?;
    pencils.extend(stages[..count].iter().map(|stage| Arc::clone(&stage.input)));
    Ok(pencils.into_boxed_slice())
}

impl<R: FftReal, const N: usize, const M: usize> R2cPlan<R, N, M>
where
    Complex<R>: Equivalence,
{
    fn execute_forward(
        &self,
        source: &PencilArray<R, N, M>,
        destination: &mut PencilArray<Complex<R>, N, M>,
        workspace: &mut R2cWorkspace<R, N, M>,
    ) -> Result<(), R2cError> {
        let communicator = self.input_pencil().topology().communicator();
        agree_execution_descriptor_ref::<N, M>(
            communicator,
            OPERATION_R2C_FORWARD,
            &self.core.descriptor,
        )?;
        let preflight = self.preflight_forward(source, destination, workspace);
        if !collective_valid(communicator, preflight.is_ok()) {
            return Err(preflight
                .err()
                .unwrap_or(R2cError::Fft(FftError::CollectivePreconditionFailed)));
        }
        preflight.expect("distributed R2C forward preflight succeeded");
        execute_forward(&self.core, source, destination, workspace)
    }

    fn execute_reverse(
        &self,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &mut PencilArray<R, N, M>,
        workspace: &mut R2cWorkspace<R, N, M>,
        normalize_inverse: bool,
    ) -> Result<(), R2cError> {
        let communicator = self.input_pencil().topology().communicator();
        let operation = if normalize_inverse {
            OPERATION_R2C_INVERSE
        } else {
            OPERATION_R2C_BACKWARD
        };
        agree_execution_descriptor_ref::<N, M>(communicator, operation, &self.core.descriptor)?;
        let preflight = self.preflight_inverse(source, destination, workspace);
        if !collective_valid(communicator, preflight.is_ok()) {
            return Err(preflight
                .err()
                .unwrap_or(R2cError::Fft(FftError::CollectivePreconditionFailed)));
        }
        preflight.expect("distributed R2C reverse preflight succeeded");
        execute_inverse(
            &self.core,
            source,
            destination,
            workspace,
            normalize_inverse,
            self.raw_absolute_threshold,
        )
    }

    fn preflight_forward(
        &self,
        source: &PencilArray<R, N, M>,
        destination: &PencilArray<Complex<R>, N, M>,
        workspace: &R2cWorkspace<R, N, M>,
    ) -> Result<(), R2cError> {
        self.preflight_common(Direction::Forward, source, destination, workspace)
    }

    fn preflight_inverse(
        &self,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &PencilArray<R, N, M>,
        workspace: &R2cWorkspace<R, N, M>,
    ) -> Result<(), R2cError> {
        self.preflight_common(Direction::Inverse, source, destination, workspace)
    }

    fn preflight_common<T, U>(
        &self,
        direction: Direction,
        source: &PencilArray<T, N, M>,
        destination: &PencilArray<U, N, M>,
        workspace: &R2cWorkspace<R, N, M>,
    ) -> Result<(), R2cError> {
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
        let boundary = self
            .core
            .real_stage_index
            .ok_or(FftError::PreparationFailed)?;
        if boundary > 0 {
            let real_intermediate = workspace
                .real_intermediate
                .as_ref()
                .ok_or(FftError::WorkspaceMismatch)?;
            if real_intermediate.extra_shape() != &self.core.extra_shape
                || !real_intermediate
                    .pencils()
                    .iter()
                    .any(|pencil| pencil.same_layout(self.core.stages[boundary].input.as_ref()))
            {
                return Err(FftError::WorkspaceMismatch.into());
            }
            let real_transpose = workspace
                .real_transpose
                .as_ref()
                .ok_or(FftError::WorkspaceMismatch)?;
            validate_workspace_lengths_values(
                workspace.fft_scratch.len(),
                self.core.fft_scratch_len,
                real_transpose.send_len(),
                self.core.real_transpose_send_len,
                real_transpose.receive_len(),
                self.core.real_transpose_receive_len,
            )?;
        }
        let (real_len, complex_len) = r2c_lengths(&self.core);
        if workspace.real_line.len() < real_len {
            return Err(FftError::WorkspaceTooSmall {
                kind: "real line",
                required: real_len,
                actual: workspace.real_line.len(),
            }
            .into());
        }
        if workspace.complex_line.len() < complex_len {
            return Err(FftError::WorkspaceTooSmall {
                kind: "complex line",
                required: complex_len,
                actual: workspace.complex_line.len(),
            }
            .into());
        }
        Ok(())
    }
}

fn execute_forward<R: FftReal, const N: usize, const M: usize>(
    core: &Arc<TransformPlanCore<R, N, M>>,
    source: &PencilArray<R, N, M>,
    destination: &mut PencilArray<Complex<R>, N, M>,
    workspace: &mut R2cWorkspace<R, N, M>,
) -> Result<(), R2cError>
where
    Complex<R>: Equivalence,
{
    let boundary = core.real_stage_index.ok_or(FftError::PreparationFailed)?;
    let source_view = source.view();
    if boundary == 0 {
        let stage = &core.stages[boundary];
        let real_line = &mut workspace.real_line;
        let fft_scratch = &mut workspace.fft_scratch;
        workspace
            .intermediate
            .overwrite_with(stage.output.as_ref(), |mut target| {
                stage
                    .local
                    .real_complex()
                    .forward(
                        source_view.as_slice(),
                        target.as_mut_slice(),
                        real_line,
                        fft_scratch,
                    )
                    .expect("distributed R2C forward preflight validated real stage");
                Ok::<_, ()>(())
            })
            .expect("distributed R2C stage-zero overwrite was preflighted");
    } else {
        let real_intermediate = workspace
            .real_intermediate
            .as_mut()
            .ok_or(FftError::WorkspaceMismatch)?;
        real_intermediate
            .overwrite_with(core.stages[0].output.as_ref(), |mut target| {
                if target.as_mut_slice().len() != source_view.as_slice().len() {
                    return Err(());
                }
                target
                    .as_mut_slice()
                    .copy_from_slice(source_view.as_slice());
                Ok::<_, ()>(())
            })
            .expect("distributed R2C real-prefix overwrite was preflighted");
        {
            let real_transpose = workspace
                .real_transpose
                .as_mut()
                .ok_or(FftError::WorkspaceMismatch)?;
            for index in 0..boundary {
                super::execute_transition(
                    &core.transitions[index].forward,
                    real_intermediate,
                    real_transpose,
                )?;
            }
        }
        let active = real_intermediate.active_view().map_err(FftError::Array)?;
        let stage = &core.stages[boundary];
        let real_line = &mut workspace.real_line;
        let fft_scratch = &mut workspace.fft_scratch;
        workspace
            .intermediate
            .overwrite_with(stage.output.as_ref(), |mut target| {
                stage
                    .local
                    .real_complex()
                    .forward(
                        active.as_slice(),
                        target.as_mut_slice(),
                        real_line,
                        fft_scratch,
                    )
                    .expect("distributed R2C forward real boundary was preflighted");
                Ok::<_, ()>(())
            })
            .expect("distributed R2C boundary overwrite was preflighted");
    }

    let tail = &core.stages[boundary..];
    let tail_transitions = &core.transitions[boundary..];
    super::execute_forward_complex_tail(
        tail,
        tail_transitions,
        &mut workspace.intermediate,
        &mut workspace.transpose,
        destination,
        &mut workspace.fft_scratch,
    )?;
    Ok(())
}

fn execute_inverse<R: FftReal, const N: usize, const M: usize>(
    core: &Arc<TransformPlanCore<R, N, M>>,
    source: &PencilArray<Complex<R>, N, M>,
    destination: &mut PencilArray<R, N, M>,
    workspace: &mut R2cWorkspace<R, N, M>,
    normalize_inverse: bool,
    raw_absolute_threshold: f64,
) -> Result<(), R2cError>
where
    Complex<R>: Equivalence,
{
    let boundary = core.real_stage_index.ok_or(FftError::PreparationFailed)?;
    super::execute_inverse_complex_tail(
        &core.stages[boundary..],
        &core.transitions[boundary..],
        source,
        &mut workspace.intermediate,
        &mut workspace.transpose,
        &mut workspace.fft_scratch,
        normalize_inverse,
    )?;
    validate_boundary(
        core,
        &workspace.intermediate,
        normalize_inverse,
        raw_absolute_threshold,
    )?;
    zero_accepted_boundary(core, &mut workspace.intermediate)?;
    let stage = &core.stages[boundary];
    let active = workspace
        .intermediate
        .active_view()
        .expect("distributed R2C boundary active layout was validated");
    if boundary == 0 {
        let mut destination_view = destination.view_mut();
        if normalize_inverse {
            stage.local.real_complex().inverse(
                active.as_slice(),
                destination_view.as_mut_slice(),
                &mut workspace.complex_line,
                &mut workspace.fft_scratch,
            )?;
        } else {
            stage.local.real_complex().backward(
                active.as_slice(),
                destination_view.as_mut_slice(),
                &mut workspace.complex_line,
                &mut workspace.fft_scratch,
            )?;
        }
        return Ok(());
    }

    let real_intermediate = workspace
        .real_intermediate
        .as_mut()
        .ok_or(FftError::WorkspaceMismatch)?;
    real_intermediate
        .overwrite_with(stage.input.as_ref(), |mut target| {
            if normalize_inverse {
                stage.local.real_complex().inverse(
                    active.as_slice(),
                    target.as_mut_slice(),
                    &mut workspace.complex_line,
                    &mut workspace.fft_scratch,
                )?;
            } else {
                stage.local.real_complex().backward(
                    active.as_slice(),
                    target.as_mut_slice(),
                    &mut workspace.complex_line,
                    &mut workspace.fft_scratch,
                )?;
            }
            Ok::<_, LocalR2cError>(())
        })
        .map_err(|error| match error {
            OverwriteError::Array(error) => R2cError::Fft(FftError::Array(error)),
            OverwriteError::Writer(error) => R2cError::LocalR2c(error),
        })?;
    {
        let real_transpose = workspace
            .real_transpose
            .as_mut()
            .ok_or(FftError::WorkspaceMismatch)?;
        for index in (0..boundary).rev() {
            super::execute_transition(
                &core.transitions[index].backward,
                real_intermediate,
                real_transpose,
            )?;
        }
    }
    let active = real_intermediate.active_view().map_err(FftError::Array)?;
    let mut destination_view = destination.view_mut();
    if active.as_slice().len() != destination_view.as_slice().len() {
        return Err(FftError::PreparationFailed.into());
    }
    destination_view
        .as_mut_slice()
        .copy_from_slice(active.as_slice());
    Ok(())
}

fn r2c_lengths<R: FftReal, const N: usize, const M: usize>(
    core: &TransformPlanCore<R, N, M>,
) -> (usize, usize) {
    let boundary = core
        .real_stage_index
        .expect("R2C core has a real boundary stage");
    let plan = core.stages[boundary].local.real_complex();
    (plan.real_len(), plan.complex_len())
}

fn validate_boundary<R: FftReal, const N: usize, const M: usize>(
    core: &TransformPlanCore<R, N, M>,
    intermediate: &ManyPencilArray<Complex<R>, N, M>,
    normalize_inverse: bool,
    raw_absolute_threshold: f64,
) -> Result<(), R2cError>
where
    Complex<R>: Equivalence,
{
    let communicator = core.stages[0].input.topology().communicator();
    let view = intermediate.active_view().map_err(FftError::Array)?;
    let local_len = view.pencil().local_len();
    let (real_len, complex_len) = r2c_lengths(core);
    let reduction_axis = core
        .selection
        .mask()
        .iter()
        .rposition(|&selected| selected)
        .expect("R2C core has a selected reduction axis");
    let depth = normalization_depth(
        *core.stages[0].input.global_shape(),
        core.selection,
        reduction_axis,
    );
    let plane_count = if real_len % 2 == 0 { 2 } else { 1 };
    let relative = 128.0 * <R as crate::private::Sealed>::pencil_fft_epsilon_f64() * depth;
    let absolute = if normalize_inverse {
        128.0 * <R as crate::private::Sealed>::pencil_fft_min_subnormal_f64() * depth
    } else {
        raw_absolute_threshold
    };
    let mut invalid = false;

    // ponytail: two full-Cartesian reductions per batch keep the four-word
    // statistics bounded; batch reductions can be fused only if latency is
    // measured to justify the added buffering.
    for batch in 0..core.extra_shape.element_count() {
        let start = batch
            .checked_mul(local_len)
            .expect("validated extra and local lengths fit the intermediate");
        let end = start
            .checked_add(local_len)
            .expect("validated extra and local lengths fit the intermediate");
        let values = &view.as_slice()[start..end];
        let mut local_max = [0.0_f64; 4];
        for line in values.chunks_exact(complex_len) {
            for plane in 0..plane_count {
                let value = line[if plane == 0 { 0 } else { complex_len - 1 }];
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
        let mut global_max = [0.0_f64; 4];
        communicator.all_reduce_into(&local_max, &mut global_max, SystemOperation::max());

        let mut local_sum = [0.0_f64; 4];
        for line in values.chunks_exact(complex_len) {
            for plane in 0..plane_count {
                let value = line[if plane == 0 { 0 } else { complex_len - 1 }];
                let re = <R as crate::private::Sealed>::pencil_fft_as_f64(value.re);
                let im = <R as crate::private::Sealed>::pencil_fft_as_f64(value.im);
                if !re.is_finite() || !im.is_finite() {
                    continue;
                }
                let slot = plane * 2;
                let scale = global_max[slot];
                if scale != 0.0 && scale.is_finite() {
                    let normalized_re = re / scale;
                    let normalized_im = im / scale;
                    local_sum[slot] += normalized_re * normalized_re;
                    local_sum[slot + 1] += normalized_im * normalized_im;
                }
            }
        }
        let mut global_sum = [0.0_f64; 4];
        communicator.all_reduce_into(&local_sum, &mut global_sum, SystemOperation::sum());
        if global_max[..plane_count * 2]
            .iter()
            .any(|value| !value.is_finite())
            || global_sum[..plane_count * 2]
                .iter()
                .any(|value| !value.is_finite())
            || !relative.is_finite()
            || !absolute.is_finite()
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
        return Err(R2cError::InvalidSpectrum);
    }
    Ok(())
}

fn zero_accepted_boundary<R: FftReal, const N: usize, const M: usize>(
    core: &TransformPlanCore<R, N, M>,
    intermediate: &mut ManyPencilArray<Complex<R>, N, M>,
) -> Result<(), R2cError> {
    let mut view = intermediate.active_view_mut().map_err(FftError::Array)?;
    let local_len = view.pencil().local_len();
    let (real_len, complex_len) = r2c_lengths(core);
    let plane_count = if real_len % 2 == 0 { 2 } else { 1 };
    for batch in 0..core.extra_shape.element_count() {
        let start = batch
            .checked_mul(local_len)
            .expect("validated extra and local lengths fit the intermediate");
        let end = start
            .checked_add(local_len)
            .expect("validated extra and local lengths fit the intermediate");
        for line in view.as_mut_slice()[start..end].chunks_exact_mut(complex_len) {
            line[0].im = R::zero();
            if plane_count == 2 {
                line[complex_len - 1].im = R::zero();
            }
        }
    }
    Ok(())
}

fn normalization_depth<const N: usize>(
    shape: [usize; N],
    selection: AxisSelection<N>,
    reduction_axis: usize,
) -> f64 {
    let mut depth = 1.0_f64;
    for (axis, length) in shape.into_iter().enumerate() {
        if axis < reduction_axis && selection.contains(axis) && length > 1 {
            depth += (usize::BITS - (length - 1).leading_zeros()) as f64;
        }
    }
    depth
}

fn raw_absolute_threshold_for_shape<R: FftReal, const N: usize>(
    shape: [usize; N],
    selection: AxisSelection<N>,
    reduction_axis: usize,
) -> Result<f64, FftError> {
    let mut transverse = 1.0_f64;
    for (axis, length) in shape.into_iter().enumerate() {
        if axis < reduction_axis && selection.contains(axis) {
            transverse *= length as f64;
            if !transverse.is_finite() || transverse <= 0.0 {
                return Err(FftError::PreparationFailed);
            }
        }
    }
    let inverse_absolute = 128.0
        * <R as crate::private::Sealed>::pencil_fft_min_subnormal_f64()
        * normalization_depth(shape, selection, reduction_axis);
    let raw_absolute = inverse_absolute * transverse;
    if raw_absolute.is_finite() && raw_absolute > 0.0 {
        Ok(raw_absolute)
    } else {
        Err(FftError::PreparationFailed)
    }
}
