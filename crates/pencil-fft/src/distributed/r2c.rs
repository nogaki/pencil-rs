//! Distributed real-to-half-complex and half-complex-to-real transforms.

#![allow(clippy::too_many_arguments)]

use std::{
    cmp::Ordering,
    mem::{align_of, size_of},
    sync::Arc,
    time::Instant,
};

use bytemuck::{allocation::try_cast_vec, try_cast_slice, try_cast_slice_mut};
#[cfg(test)]
use mpi::topology::Communicator;
use mpi::{
    collective::{CommunicatorCollectives, SystemOperation},
    datatype::Equivalence,
};
use pencil_array::{
    ArrayError, ExtraShape, ManyPencilArray, MpiTopology, OverlapError, OverwriteError, Pencil,
    PencilArray, PencilArrayView, PencilArrayViewMut, TransposeWorkspace,
};

impl From<FftError> for FftOverlapError<R2cError> {
    fn from(error: FftError) -> Self {
        Self::Operation(error.into())
    }
}

use super::{
    AxisSelection, C2cTransition, DESCRIPTOR_SCHEMA, Direction, DistributedLayout, FftError,
    FftOverlapError, FourierDirections, INVALID_WORD, LocalTransform, OPERATION_R2C_BACKWARD,
    OPERATION_R2C_BACKWARD_IN_PLACE, OPERATION_R2C_FORWARD, OPERATION_R2C_FORWARD_IN_PLACE,
    OPERATION_R2C_INVERSE, OPERATION_R2C_INVERSE_IN_PLACE, OPERATION_R2C_PLAN, R2cError,
    StagePreparation, TransformPlanCore, TransformStage, TransformTiming, TransposeMethod,
    VALUE_KIND_R2C, agree_execution_descriptor_ref, agree_header, agree_result, build_descriptor,
    build_route, build_transitions, collective_valid, descriptor_len, initialized_vec,
    map_array_allocation, prepare_complex_stage, registered_stage_pencils,
    strided_complex_line_len, validate_out_of_place, validate_workspace_lengths_values,
    zero_complex,
};
#[cfg(test)]
use crate::LocalR2cError;
use crate::{Complex, FftReal, LocalR2cPlan, R2cState};

// 55..57 are C2C overlap words. Keep every operation descriptor unique.
const OPERATION_R2C_FORWARD_OVERLAP: u64 = 73;
const OPERATION_R2C_INVERSE_OVERLAP: u64 = 74;
const OPERATION_R2C_BACKWARD_OVERLAP: u64 = 75;
const OPERATION_R2C_FORWARD_TIMED: u64 = 79;
const OPERATION_R2C_INVERSE_TIMED: u64 = 80;
const OPERATION_R2C_BACKWARD_TIMED: u64 = 81;
const OPERATION_R2C_FORWARD_IN_PLACE_TIMED: u64 = 82;
const OPERATION_R2C_INVERSE_IN_PLACE_TIMED: u64 = 83;
const OPERATION_R2C_BACKWARD_IN_PLACE_TIMED: u64 = 84;

/// An immutable, checked distributed real-to-half-complex FFT plan.
///
/// This API requires `N >= 2` and `1 <= M < N`. The input uses identity
/// permutation and decomposition `[0..M)` with the original real global shape.
/// A non-empty [`AxisSelection`] chooses the largest selected axis as the
/// real-to-half-complex boundary. Its extent `n` becomes `n / 2 + 1`; selected
/// lower axes are complex stages and unselected axes are identity stages. The
/// route still contains all `N` stages and `N - 1` transitions, including
/// real-prefix transposes before a non-last boundary. The final output uses
/// decomposition `[1..=M]`; memory order is reversed by default and remains
/// identity when `layout.permute_dims` is false. Empty selections are
/// collectively rejected before native planning.
///
/// Constructors and `forward`/`inverse`/`backward` are collective. Every rank must use
/// the same communicator context, API and call order, scalar type, transport
/// method, and matching layouts. The legacy constructors select
/// [`TransposeMethod::AllToAllv`]; the `_with_method` constructors can select
/// [`TransposeMethod::PointToPoint`]. Point-to-point uses the existing fixed
/// `0x5054` tag and changed-axis context, so unfinished transposes must not
/// overlap on that context. Allocation methods are noncollective; callers
/// must coordinate a local allocation failure before the next collective.
/// The `*_in_place` methods use one state-checked allocation shared by the
/// real prefix and reduced-complex suffix.
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
/// use pencil_fft::{R2cPlan, R2cState};
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
///
///         let mut in_place = plan.allocate_in_place()?;
///         in_place.real_view_mut()?.as_mut_slice().copy_from_slice(&original);
///         let mut in_place_workspace = plan.allocate_in_place_workspace()?;
///         plan.forward_in_place(&mut in_place, &mut in_place_workspace)?;
///         assert_eq!(in_place.state(), R2cState::ComplexOutput);
///         plan.inverse_in_place(&mut in_place, &mut in_place_workspace)?;
///         assert_eq!(in_place.state(), R2cState::RealInput);
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
/// The real/complex handoff is a rank-local checked phase. Its complete
/// conversion result is agreed collectively before the next transpose; a
/// conversion error on any rank therefore returns on every rank and leaves
/// every in-place array poisoned. The failed owner is retained privately when
/// ordinary reconstruction is rejected; a panic may abandon the typed owner.
/// The typed views remain state-checked, and Rust also prevents two mutable
/// representations from being borrowed at once:
///
/// ```compile_fail
/// use pencil_fft::{Complex, FftReal, R2cInPlaceArray};
///
/// fn alias<R: FftReal, const N: usize, const M: usize>(
///     array: &mut R2cInPlaceArray<R, N, M>,
/// ) {
///     let mut real = array.real_view_mut().unwrap();
///     let _complex = array.complex_view_mut();
///     real.as_mut_slice();
///     let _: Option<Complex<R>> = None;
/// }
/// ```
///
/// Existing generic callers only need their original bounds:
///
/// ```
/// use std::sync::Arc;
/// use mpi::{datatype::Equivalence, topology::Communicator};
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
    real_source_line: Option<Vec<R>>,
    real_line: Vec<R>,
    complex_line: Vec<Complex<R>>,
    complex_strided_line: Vec<Complex<R>>,
}

#[allow(dead_code)]
#[derive(Debug)]
enum R2cInPlaceStorage<R: FftReal, const N: usize, const M: usize> {
    Real(ManyPencilArray<R, N, M>),
    Complex(ManyPencilArray<Complex<R>, N, M>),
    /// An owner retained after a failed representation handoff. The outer
    /// array is poisoned, so this variant is never exposed as a typed view.
    PoisonedReal(Vec<R>),
    PoisonedComplex(Vec<Complex<R>>),
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InPlaceTestHook {
    PanicAfterForwardDetach,
    PanicAfterReverseDetach,
    OddForwardCast,
    ReverseRestoreRejection,
}

/// An opaque single-allocation distributed real-to-half-complex array.
///
/// The allocation is represented as real scalars while the array is in
/// [`R2cState::RealInput`] and as complex values while it is in
/// [`R2cState::ComplexOutput`]. The public typed views expose only the active
/// representation. A failed or panicking operation leaves the array poisoned;
/// the backing owner may be absent after an interrupted representation handoff.
#[derive(Debug)]
pub struct R2cInPlaceArray<R: FftReal, const N: usize, const M: usize> {
    core: Arc<TransformPlanCore<R, N, M>>,
    real_pencils: Option<Box<[Arc<Pencil<N, M>>]>>,
    complex_pencils: Option<Box<[Arc<Pencil<N, M>>]>>,
    real_storage_len: usize,
    complex_storage_len: usize,
    storage_bytes: usize,
    storage: Option<R2cInPlaceStorage<R, N, M>>,
    state: R2cState,
    #[cfg(test)]
    test_hook: Option<InPlaceTestHook>,
}

/// Reusable workspace for distributed real in-place R2C/C2R execution.
///
/// It contains only transpose buffers, native FFT scratch, and per-line
/// buffers. The transform data remains in [`R2cInPlaceArray`].
#[derive(Debug)]
pub struct R2cInPlaceWorkspace<R: FftReal, const N: usize, const M: usize> {
    core: Arc<TransformPlanCore<R, N, M>>,
    transpose: TransposeWorkspace<Complex<R>>,
    real_transpose: Option<TransposeWorkspace<R>>,
    fft_scratch: Vec<Complex<R>>,
    real_source_line: Option<Vec<R>>,
    real_line: Vec<R>,
    complex_source_line: Vec<Complex<R>>,
    complex_line: Vec<Complex<R>>,
    complex_strided_line: Vec<Complex<R>>,
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
    ) -> Result<Self, R2cError> {
        Self::from_pencil_with_selection_and_layout(
            input,
            extra_shape,
            selection,
            DistributedLayout::default(),
        )
    }

    /// Collectively builds a plan from a canonical real pencil and selection.
    pub fn from_pencil_with_selection_and_method(
        input: Arc<Pencil<N, M>>,
        extra_shape: ExtraShape,
        selection: AxisSelection<N>,
        method: TransposeMethod,
    ) -> Result<Self, R2cError> {
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
    ) -> Result<Self, R2cError> {
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
    ) -> Result<Self, R2cError> {
        let topology = Arc::clone(input.topology());
        let global_shape = *input.global_shape();
        Self::construct(
            topology,
            global_shape,
            extra_shape,
            Ok(input),
            selection,
            layout,
        )
    }

    /// Collectively builds a plan from a canonical real pencil and transport.
    pub fn from_pencil_with_method(
        input: Arc<Pencil<N, M>>,
        extra_shape: ExtraShape,
        method: TransposeMethod,
    ) -> Result<Self, R2cError> {
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

    /// Collectively builds an Alltoallv plan from a canonical real array.
    pub fn from_array(input: &PencilArray<R, N, M>) -> Result<Self, R2cError> {
        Self::from_array_with_selection_and_layout(
            input,
            AxisSelection::all(),
            DistributedLayout::default(),
        )
    }

    /// Collectively builds an Alltoallv plan for a validated axis selection.
    pub fn from_array_with_selection(
        input: &PencilArray<R, N, M>,
        selection: AxisSelection<N>,
    ) -> Result<Self, R2cError> {
        Self::from_array_with_selection_and_layout(input, selection, DistributedLayout::default())
    }

    /// Collectively builds a plan from a canonical real array and selection.
    pub fn from_array_with_selection_and_method(
        input: &PencilArray<R, N, M>,
        selection: AxisSelection<N>,
        method: TransposeMethod,
    ) -> Result<Self, R2cError> {
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
        input: &PencilArray<R, N, M>,
        layout: DistributedLayout,
    ) -> Result<Self, R2cError> {
        Self::from_array_with_selection_and_layout(input, AxisSelection::all(), layout)
    }

    /// Collectively builds a plan for a validated selection and layout policy.
    pub fn from_array_with_selection_and_layout(
        input: &PencilArray<R, N, M>,
        selection: AxisSelection<N>,
        layout: DistributedLayout,
    ) -> Result<Self, R2cError> {
        let topology = Arc::clone(input.pencil().topology());
        let global_shape = *input.pencil().global_shape();
        Self::construct(
            topology,
            global_shape,
            input.extra_shape().clone(),
            Ok(Arc::clone(input.pencil())),
            selection,
            layout,
        )
    }

    /// Collectively builds a plan from a canonical real array and transport.
    pub fn from_array_with_method(
        input: &PencilArray<R, N, M>,
        method: TransposeMethod,
    ) -> Result<Self, R2cError> {
        Self::from_array_with_selection_and_layout(
            input,
            AxisSelection::all(),
            DistributedLayout {
                transpose_method: method,
                ..DistributedLayout::default()
            },
        )
    }

    /// Collectively builds an Alltoallv plan from topology and shape.
    pub fn from_shape(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
    ) -> Result<Self, R2cError> {
        Self::from_shape_with_selection_and_layout(
            topology,
            global_shape,
            extra_shape,
            AxisSelection::all(),
            DistributedLayout::default(),
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

    /// Collectively builds an Alltoallv plan for a validated axis selection.
    pub fn from_shape_with_selection(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
        selection: AxisSelection<N>,
    ) -> Result<Self, R2cError> {
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
    ) -> Result<Self, R2cError> {
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
    ) -> Result<Self, R2cError> {
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
            layout,
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

    /// Returns the transport and memory-layout policy used by this plan.
    pub fn layout(&self) -> DistributedLayout {
        self.core.layout
    }

    /// Returns the immutable checked geometry used by every route stage.
    pub fn stage_geometry(&self) -> Box<[super::StageGeometry<N, M>]> {
        self.core
            .stages
            .iter()
            .map(|stage| super::StageGeometry {
                axis: stage.axis,
                source: Arc::clone(&stage.input),
                output: Arc::clone(&stage.output),
            })
            .collect()
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
        let boundary_stride = super::memory_stride(
            self.core.stages[boundary].input.as_ref(),
            self.core.stages[boundary].axis,
        )?;
        let real_source_line = if boundary_stride > 1 {
            Some(initialized_vec(real_len, R::zero())?)
        } else {
            None
        };
        let real_line = initialized_vec(real_len, R::zero())?;
        let complex_line = initialized_vec(complex_len, zero_complex::<R>())?;
        let complex_strided_line =
            initialized_vec(self.core.strided_line_len, zero_complex::<R>())?;
        Ok(R2cWorkspace {
            core: Arc::clone(&self.core),
            intermediate,
            transpose,
            real_intermediate,
            real_transpose,
            fft_scratch,
            real_source_line,
            real_line,
            complex_line,
            complex_strided_line,
        })
    }

    /// Allocates one zero-initialized data allocation for distributed in-place
    /// forward, inverse, and backward execution.
    ///
    /// The returned array starts in [`R2cState::RealInput`]. Its backing
    /// allocation is large enough in bytes for every registered real-prefix
    /// and reduced-complex-suffix layout. No data reallocation is performed by
    /// any later representation transition.
    pub fn allocate_in_place(&self) -> Result<R2cInPlaceArray<R, N, M>, R2cError> {
        let boundary = self
            .core
            .real_stage_index
            .ok_or(FftError::PreparationFailed)?;
        let real_pencils = registered_real_stage_pencils(&self.core.stages, boundary)?;
        let complex_pencils = registered_stage_pencils(&self.core.stages[boundary..])?;
        let (real_storage_len, complex_storage_len, storage_bytes, complex_capacity) =
            in_place_storage_requirements(&self.core)?;

        let mut storage = Vec::new();
        storage
            .try_reserve_exact(complex_capacity)
            .map_err(|_| FftError::AllocationFailed {
                required: complex_capacity,
            })?;
        storage.resize(complex_capacity, zero_complex::<R>());
        let mut real_storage = cast_complex_vec_to_real(storage)?;
        if real_storage.len() < real_storage_len {
            return Err(FftError::StorageLayoutMismatch.into());
        }
        real_storage.truncate(real_storage_len);
        let storage =
            ManyPencilArray::from_vec(real_pencils, 0, self.core.extra_shape.clone(), real_storage)
                .map_err(map_array_allocation)?;

        Ok(R2cInPlaceArray {
            core: Arc::clone(&self.core),
            real_pencils: None,
            complex_pencils: Some(complex_pencils),
            real_storage_len,
            complex_storage_len,
            storage_bytes,
            storage: Some(R2cInPlaceStorage::Real(storage)),
            state: R2cState::RealInput,
            #[cfg(test)]
            test_hook: None,
        })
    }

    /// Allocates reusable scratch for distributed real in-place execution.
    pub fn allocate_in_place_workspace(&self) -> Result<R2cInPlaceWorkspace<R, N, M>, R2cError> {
        let boundary = self
            .core
            .real_stage_index
            .ok_or(FftError::PreparationFailed)?;
        let (real_len, complex_len) = r2c_lengths(&self.core);
        let real_transpose = if boundary == 0 {
            None
        } else {
            Some(TransposeWorkspace::from_vecs(
                initialized_vec(self.core.real_transpose_send_len, R::zero())?,
                initialized_vec(self.core.real_transpose_receive_len, R::zero())?,
            ))
        };
        let boundary_stride = super::memory_stride(
            self.core.stages[boundary].input.as_ref(),
            self.core.stages[boundary].axis,
        )?;
        Ok(R2cInPlaceWorkspace {
            core: Arc::clone(&self.core),
            transpose: TransposeWorkspace::from_vecs(
                initialized_vec(self.core.transpose_send_len, zero_complex::<R>())?,
                initialized_vec(self.core.transpose_receive_len, zero_complex::<R>())?,
            ),
            real_transpose,
            fft_scratch: initialized_vec(self.core.fft_scratch_len, zero_complex::<R>())?,
            real_source_line: if boundary_stride > 1 {
                Some(initialized_vec(real_len, R::zero())?)
            } else {
                None
            },
            real_line: initialized_vec(real_len, R::zero())?,
            complex_source_line: initialized_vec(complex_len, zero_complex::<R>())?,
            complex_line: initialized_vec(complex_len, zero_complex::<R>())?,
            complex_strided_line: initialized_vec(self.core.strided_line_len, zero_complex::<R>())?,
        })
    }

    /// Computes an unnormalized forward real-to-half-complex transform.
    pub fn forward(
        &self,
        source: &PencilArray<R, N, M>,
        destination: &mut PencilArray<Complex<R>, N, M>,
        workspace: &mut R2cWorkspace<R, N, M>,
    ) -> Result<(), R2cError> {
        self.execute_forward(source, destination, workspace, None)
    }

    /// Runs [`Self::forward`] and returns per-stage timing.
    pub fn forward_with_timing(
        &self,
        source: &PencilArray<R, N, M>,
        destination: &mut PencilArray<Complex<R>, N, M>,
        workspace: &mut R2cWorkspace<R, N, M>,
    ) -> Result<TransformTiming<N>, R2cError> {
        let mut timing = TransformTiming::default();
        self.execute_forward(source, destination, workspace, Some(&mut timing))?;
        Ok(timing)
    }

    /// Computes forward while running each next complex FFT after receive/unpack completion and before the P2P send wait.
    pub fn forward_with_overlap(
        &self,
        source: &PencilArray<R, N, M>,
        destination: &mut PencilArray<Complex<R>, N, M>,
        workspace: &mut R2cWorkspace<R, N, M>,
    ) -> Result<(), FftOverlapError<R2cError>> {
        self.execute_overlap_forward(source, destination, workspace)
    }

    /// Computes a normalized inverse while running each next complex FFT after receive/unpack completion and before the P2P send wait.
    pub fn inverse_with_overlap(
        &self,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &mut PencilArray<R, N, M>,
        workspace: &mut R2cWorkspace<R, N, M>,
    ) -> Result<(), FftOverlapError<R2cError>> {
        self.execute_overlap_reverse(Direction::Inverse, source, destination, workspace)
    }

    /// Computes an unnormalized backward transform with local FFTs after
    /// receive/unpack completion and before point-to-point send waits.
    pub fn backward_with_overlap(
        &self,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &mut PencilArray<R, N, M>,
        workspace: &mut R2cWorkspace<R, N, M>,
    ) -> Result<(), FftOverlapError<R2cError>> {
        self.execute_overlap_reverse(Direction::Backward, source, destination, workspace)
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
        self.execute_reverse(source, destination, workspace, true, None)
    }

    /// Runs [`Self::inverse`] and returns per-stage timing.
    pub fn inverse_with_timing(
        &self,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &mut PencilArray<R, N, M>,
        workspace: &mut R2cWorkspace<R, N, M>,
    ) -> Result<TransformTiming<N>, R2cError> {
        let mut timing = TransformTiming::default();
        self.execute_reverse(source, destination, workspace, true, Some(&mut timing))?;
        Ok(timing)
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
        self.execute_reverse(source, destination, workspace, false, None)
    }

    /// Runs [`Self::backward`] and returns per-stage timing.
    pub fn backward_with_timing(
        &self,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &mut PencilArray<R, N, M>,
        workspace: &mut R2cWorkspace<R, N, M>,
    ) -> Result<TransformTiming<N>, R2cError> {
        let mut timing = TransformTiming::default();
        self.execute_reverse(source, destination, workspace, false, Some(&mut timing))?;
        Ok(timing)
    }

    /// Computes an unnormalized forward transform in the single data
    /// allocation. The array must be in [`R2cState::RealInput`].
    pub fn forward_in_place(
        &self,
        array: &mut R2cInPlaceArray<R, N, M>,
        workspace: &mut R2cInPlaceWorkspace<R, N, M>,
    ) -> Result<(), R2cError> {
        self.execute_in_place(Direction::Forward, array, workspace, None)
    }

    /// Runs [`Self::forward_in_place`] and returns per-stage timing.
    pub fn forward_in_place_with_timing(
        &self,
        array: &mut R2cInPlaceArray<R, N, M>,
        workspace: &mut R2cInPlaceWorkspace<R, N, M>,
    ) -> Result<TransformTiming<N>, R2cError> {
        let mut timing = TransformTiming::default();
        self.execute_in_place(Direction::Forward, array, workspace, Some(&mut timing))?;
        Ok(timing)
    }

    /// Computes a normalized inverse transform in the single data allocation.
    /// The array must be in [`R2cState::ComplexOutput`].
    pub fn inverse_in_place(
        &self,
        array: &mut R2cInPlaceArray<R, N, M>,
        workspace: &mut R2cInPlaceWorkspace<R, N, M>,
    ) -> Result<(), R2cError> {
        self.execute_in_place(Direction::Inverse, array, workspace, None)
    }

    /// Runs [`Self::inverse_in_place`] and returns per-stage timing.
    pub fn inverse_in_place_with_timing(
        &self,
        array: &mut R2cInPlaceArray<R, N, M>,
        workspace: &mut R2cInPlaceWorkspace<R, N, M>,
    ) -> Result<TransformTiming<N>, R2cError> {
        let mut timing = TransformTiming::default();
        self.execute_in_place(Direction::Inverse, array, workspace, Some(&mut timing))?;
        Ok(timing)
    }

    /// Computes an unnormalized positive-sign backward transform in the
    /// single data allocation. The array must be in
    /// [`R2cState::ComplexOutput`].
    pub fn backward_in_place(
        &self,
        array: &mut R2cInPlaceArray<R, N, M>,
        workspace: &mut R2cInPlaceWorkspace<R, N, M>,
    ) -> Result<(), R2cError> {
        self.execute_in_place(Direction::Backward, array, workspace, None)
    }

    /// Runs [`Self::backward_in_place`] and returns per-stage timing.
    pub fn backward_in_place_with_timing(
        &self,
        array: &mut R2cInPlaceArray<R, N, M>,
        workspace: &mut R2cInPlaceWorkspace<R, N, M>,
    ) -> Result<TransformTiming<N>, R2cError> {
        let mut timing = TransformTiming::default();
        self.execute_in_place(Direction::Backward, array, workspace, Some(&mut timing))?;
        Ok(timing)
    }

    fn construct(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
        input: Result<Arc<Pencil<N, M>>, FftError>,
        selection: AxisSelection<N>,
        layout: DistributedLayout,
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
                layout,
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
            build_route(
                Ok(Arc::clone(&input_pencil)),
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
                Ok(Arc::clone(&reduced_input)),
                &topology,
                reduced_shape,
                layout.permute_dims,
            ),
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
            layout.transpose_method,
            boundary,
        )?;
        let mut strided_line_len = strided_complex_line_len(&stages.stages)?;
        let boundary_stride = super::memory_stride(
            stages.stages[boundary].input.as_ref(),
            stages.stages[boundary].axis,
        )?;
        if boundary_stride > 1 {
            strided_line_len = strided_line_len.max(complex_len);
        }
        let core = TransformPlanCore {
            stages: stages.stages,
            transitions: transitions.into_boxed_slice(),
            extra_shape,
            selection,
            layout,
            real_stage_index: Some(boundary),
            descriptor: descriptor.into_boxed_slice(),
            fft_scratch_len: stages.fft_scratch_len,
            strided_line_len,
            transpose_send_len,
            transpose_receive_len,
            real_transpose_send_len,
            real_transpose_receive_len,
            directions: FourierDirections::default(),
            strict_array_identity: false,
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
            Ordering::Less => prepare_complex_stage(
                &original_route.stages[index],
                original_shape,
                axis,
                false,
                FourierDirections::default(),
            )?,
            Ordering::Equal => {
                let pencil = &original_route.stages[index];
                if pencil
                    .decomposition()
                    .iter()
                    .any(|distributed| distributed.index() == reduction_axis)
                    || pencil.local_shape_logical()[reduction_axis] != real_len
                {
                    return Err(R2cError::Fft(FftError::PreparationFailed));
                }
                let real = LocalR2cPlan::new(real_len).map_err(R2cError::LocalR2c)?;
                TransformStage {
                    axis: reduction_axis,
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
                FourierDirections::default(),
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

#[allow(clippy::too_many_arguments)]
fn execute_strided_real_forward<R: FftReal, const N: usize, const M: usize>(
    plan: &LocalR2cPlan<R>,
    pencil: &Pencil<N, M>,
    axis: usize,
    source: &[R],
    destination: &mut [Complex<R>],
    real_source_line: &mut [R],
    real_line: &mut [R],
    complex_line: &mut [Complex<R>],
    scratch: &mut [Complex<R>],
) -> Result<(), R2cError>
where
    Complex<R>: Equivalence,
{
    let stride = super::memory_stride(pencil, axis)?;
    if stride <= 1 {
        return plan
            .forward(source, destination, real_line, scratch)
            .map_err(R2cError::LocalR2c);
    }
    if real_source_line.len() < plan.real_len() || complex_line.len() < plan.complex_len() {
        return Err(FftError::WorkspaceTooSmall {
            kind: "real/complex line",
            required: plan.real_len().max(plan.complex_len()),
            actual: real_source_line.len().min(complex_line.len()),
        }
        .into());
    }
    let count =
        super::strided_line_count(source.len(), plan.real_len(), stride).map_err(R2cError::Fft)?;
    let destination_len = count
        .checked_mul(plan.complex_len())
        .and_then(|value| value.checked_mul(stride))
        .ok_or(FftError::PreparationFailed)?;
    if destination.len() != destination_len {
        return Err(FftError::PreparationFailed.into());
    }
    let source_block = plan
        .real_len()
        .checked_mul(stride)
        .ok_or(FftError::PreparationFailed)?;
    let destination_block = plan
        .complex_len()
        .checked_mul(stride)
        .ok_or(FftError::PreparationFailed)?;
    for outer in 0..count {
        let source_base = outer
            .checked_mul(source_block)
            .ok_or(FftError::PreparationFailed)?;
        let destination_base = outer
            .checked_mul(destination_block)
            .ok_or(FftError::PreparationFailed)?;
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
            .map_err(R2cError::LocalR2c)?;
            for k in 0..plan.complex_len() {
                destination[destination_base + k * stride + inner] = complex_line[k];
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn execute_strided_real_reverse<R: FftReal, const N: usize, const M: usize>(
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
) -> Result<(), R2cError>
where
    Complex<R>: Equivalence,
{
    let stride = super::memory_stride(pencil, axis)?;
    if stride <= 1 {
        return if normalize {
            plan.inverse(source, destination, complex_line, scratch)
        } else {
            plan.backward(source, destination, complex_line, scratch)
        }
        .map_err(R2cError::LocalR2c);
    }
    if complex_source_line.len() < plan.complex_len() || real_line.len() < plan.real_len() {
        return Err(FftError::WorkspaceTooSmall {
            kind: "complex/real line",
            required: plan.real_len().max(plan.complex_len()),
            actual: complex_source_line.len().min(real_line.len()),
        }
        .into());
    }
    let count = super::strided_line_count(source.len(), plan.complex_len(), stride)
        .map_err(R2cError::Fft)?;
    let destination_len = count
        .checked_mul(plan.real_len())
        .and_then(|value| value.checked_mul(stride))
        .ok_or(FftError::PreparationFailed)?;
    if destination.len() != destination_len {
        return Err(FftError::PreparationFailed.into());
    }
    let source_block = plan
        .complex_len()
        .checked_mul(stride)
        .ok_or(FftError::PreparationFailed)?;
    let destination_block = plan
        .real_len()
        .checked_mul(stride)
        .ok_or(FftError::PreparationFailed)?;
    for outer in 0..count {
        let source_base = outer
            .checked_mul(source_block)
            .ok_or(FftError::PreparationFailed)?;
        let destination_base = outer
            .checked_mul(destination_block)
            .ok_or(FftError::PreparationFailed)?;
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
            result.map_err(R2cError::LocalR2c)?;
            for k in 0..plan.real_len() {
                destination[destination_base + k * stride + inner] = real_line[k];
            }
        }
    }
    Ok(())
}

fn registered_layout_matches<R: FftReal, const N: usize, const M: usize>(
    registered: &[Arc<Pencil<N, M>>],
    stages: &[TransformStage<R, N, M>],
    use_inputs: bool,
) -> bool {
    stages.iter().all(|stage| {
        registered.iter().any(|registered| {
            if use_inputs {
                registered.same_layout(stage.input.as_ref())
            } else {
                registered.same_layout(stage.output.as_ref())
            }
        })
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
    for stage in &stages[..count] {
        if !pencils
            .iter()
            .any(|registered: &Arc<Pencil<N, M>>| registered.same_layout(stage.input.as_ref()))
        {
            pencils.push(Arc::clone(&stage.input));
        }
    }
    Ok(pencils.into_boxed_slice())
}

fn in_place_storage_requirements<R: FftReal, const N: usize, const M: usize>(
    core: &TransformPlanCore<R, N, M>,
) -> Result<(usize, usize, usize, usize), FftError> {
    let boundary = core.real_stage_index.ok_or(FftError::PreparationFailed)?;
    let extra = core.extra_shape.element_count();
    let mut real_len = 0usize;
    for stage in &core.stages[..=boundary] {
        let length = stage
            .input
            .local_len()
            .checked_mul(extra)
            .ok_or(FftError::PreparationFailed)?;
        real_len = real_len.max(length);
    }
    let mut complex_len = 0usize;
    for stage in &core.stages[boundary..] {
        let length = stage
            .output
            .local_len()
            .checked_mul(extra)
            .ok_or(FftError::PreparationFailed)?;
        complex_len = complex_len.max(length);
    }
    let real_bytes = real_len
        .checked_mul(size_of::<R>())
        .ok_or(FftError::PreparationFailed)?;
    let complex_bytes = complex_len
        .checked_mul(size_of::<Complex<R>>())
        .ok_or(FftError::PreparationFailed)?;
    let boundary_stride = super::memory_stride(
        core.stages[boundary].input.as_ref(),
        core.stages[boundary].axis,
    )?;
    let complex_size = size_of::<Complex<R>>();
    let storage_bytes = if boundary_stride > 1 {
        // The strided handoff promotes each real scalar to one Complex slot
        // before outer compression.  This is the only route that needs the
        // extra roughly-one-real-array allocation.
        real_len
            .max(complex_len)
            .checked_mul(complex_size)
            .ok_or(FftError::PreparationFailed)?
    } else {
        real_bytes.max(complex_bytes)
    };
    if storage_bytes > isize::MAX as usize {
        return Err(FftError::PreparationFailed);
    }
    let rounded = storage_bytes
        .checked_add(
            complex_size
                .checked_sub(1)
                .ok_or(FftError::PreparationFailed)?,
        )
        .ok_or(FftError::PreparationFailed)?;
    let complex_capacity = rounded / complex_size;
    Ok((real_len, complex_len, storage_bytes, complex_capacity))
}

fn promoted_complex_len<R: FftReal, const N: usize, const M: usize>(
    core: &TransformPlanCore<R, N, M>,
    real_len: usize,
    complex_len: usize,
) -> Result<usize, FftError> {
    let boundary = core.real_stage_index.ok_or(FftError::PreparationFailed)?;
    let stride = super::memory_stride(
        core.stages[boundary].input.as_ref(),
        core.stages[boundary].axis,
    )?;
    if stride > 1 {
        Ok(real_len.max(complex_len))
    } else {
        let real_bytes = real_len
            .checked_mul(size_of::<R>())
            .ok_or(FftError::PreparationFailed)?;
        let complex_size = size_of::<Complex<R>>();
        let rounded = real_bytes
            .checked_add(
                complex_size
                    .checked_sub(1)
                    .ok_or(FftError::PreparationFailed)?,
            )
            .ok_or(FftError::PreparationFailed)?
            / complex_size;
        Ok(rounded.max(complex_len))
    }
}

fn strided_phase_shape<R: FftReal, const N: usize, const M: usize>(
    core: &TransformPlanCore<R, N, M>,
) -> Result<Option<(usize, usize, usize, usize)>, FftError> {
    let boundary = core.real_stage_index.ok_or(FftError::PreparationFailed)?;
    let plan = core.stages[boundary].local.real_complex();
    let stride = super::memory_stride(
        core.stages[boundary].input.as_ref(),
        core.stages[boundary].axis,
    )?;
    if stride <= 1 {
        return Ok(None);
    }
    let rows = boundary_row_count(core).map_err(|error| match error {
        R2cError::Fft(error) => error,
        R2cError::InvalidSpectrum => FftError::PreparationFailed,
        R2cError::LocalR2c(_) => FftError::PreparationFailed,
    })?;
    if rows == 0 {
        return Ok(Some((0, 0, 0, 0)));
    }
    if rows % stride != 0 {
        return Err(FftError::PreparationFailed);
    }
    let groups = rows / stride;
    let real_span = groups
        .checked_mul(plan.real_len())
        .and_then(|value| value.checked_mul(stride))
        .ok_or(FftError::PreparationFailed)?;
    let complex_span = groups
        .checked_mul(plan.complex_len())
        .and_then(|value| value.checked_mul(stride))
        .ok_or(FftError::PreparationFailed)?;
    let (real_storage_len, complex_storage_len, _, _) = in_place_storage_requirements(core)?;
    if real_span > real_storage_len || complex_span > complex_storage_len {
        return Err(FftError::PreparationFailed);
    }
    Ok(Some((rows, groups, real_span, complex_span)))
}

fn cast_complex_vec_to_real<R: FftReal>(storage: Vec<Complex<R>>) -> Result<Vec<R>, R2cError> {
    try_cast_vec(storage).map_err(|(_, _)| R2cError::Fft(FftError::StorageLayoutMismatch))
}

fn validate_recast_capacity<T, U>(capacity: usize, required_bytes: usize) -> Result<(), FftError> {
    let source_size = size_of::<T>();
    let target_size = size_of::<U>();
    if source_size == 0
        || target_size == 0
        || align_of::<T>() != align_of::<U>()
        || capacity
            .checked_mul(source_size)
            .is_none_or(|bytes| bytes > isize::MAX as usize || bytes < required_bytes)
    {
        return Err(FftError::StorageLayoutMismatch);
    }
    let capacity_bytes = capacity
        .checked_mul(source_size)
        .ok_or(FftError::StorageLayoutMismatch)?;
    if capacity_bytes % target_size != 0 {
        return Err(FftError::StorageLayoutMismatch);
    }
    Ok(())
}

fn install_real_storage<R: FftReal, const N: usize, const M: usize>(
    array: &mut R2cInPlaceArray<R, N, M>,
    pencils: Box<[Arc<Pencil<N, M>>]>,
    extra_shape: ExtraShape,
    mut storage: Vec<R>,
    active: usize,
) -> Result<(), R2cError> {
    storage.truncate(array.real_storage_len);
    match ManyPencilArray::from_vec_preserving(pencils, active, extra_shape, storage) {
        Ok(real) => {
            array.storage = Some(R2cInPlaceStorage::Real(real));
            Ok(())
        }
        Err((error, storage)) => {
            array.storage = Some(R2cInPlaceStorage::PoisonedReal(storage));
            Err(map_array_allocation(error).into())
        }
    }
}

fn install_complex_storage<R: FftReal, const N: usize, const M: usize>(
    array: &mut R2cInPlaceArray<R, N, M>,
    pencils: Box<[Arc<Pencil<N, M>>]>,
    extra_shape: ExtraShape,
    mut storage: Vec<Complex<R>>,
    active: usize,
) -> Result<(), R2cError> {
    storage.truncate(array.complex_storage_len);
    match ManyPencilArray::from_vec_preserving(pencils, active, extra_shape, storage) {
        Ok(complex) => {
            array.storage = Some(R2cInPlaceStorage::Complex(complex));
            Ok(())
        }
        Err((error, storage)) => {
            array.storage = Some(R2cInPlaceStorage::PoisonedComplex(storage));
            Err(map_array_allocation(error).into())
        }
    }
}

fn retain_real_error<R: FftReal, const N: usize, const M: usize>(
    array: &mut R2cInPlaceArray<R, N, M>,
    pencils: Box<[Arc<Pencil<N, M>>]>,
    extra_shape: ExtraShape,
    storage: Vec<R>,
    active: usize,
    cause: R2cError,
) -> R2cError {
    install_real_storage(array, pencils, extra_shape, storage, active)
        .err()
        .unwrap_or(cause)
}

fn retain_complex_error<R: FftReal, const N: usize, const M: usize>(
    array: &mut R2cInPlaceArray<R, N, M>,
    pencils: Box<[Arc<Pencil<N, M>>]>,
    extra_shape: ExtraShape,
    storage: Vec<Complex<R>>,
    active: usize,
    cause: R2cError,
) -> R2cError {
    install_complex_storage(array, pencils, extra_shape, storage, active)
        .err()
        .unwrap_or(cause)
}

fn fail_real_to_complex<R: FftReal, const N: usize, const M: usize>(
    array: &mut R2cInPlaceArray<R, N, M>,
    source_pencils: Box<[Arc<Pencil<N, M>>]>,
    destination_pencils: Box<[Arc<Pencil<N, M>>]>,
    extra_shape: ExtraShape,
    storage: Vec<R>,
    active: usize,
    cause: R2cError,
) -> R2cError {
    let error = retain_real_error(array, source_pencils, extra_shape, storage, active, cause);
    array.complex_pencils = Some(destination_pencils);
    error
}

fn fail_complex_to_real<R: FftReal, const N: usize, const M: usize>(
    array: &mut R2cInPlaceArray<R, N, M>,
    source_pencils: Box<[Arc<Pencil<N, M>>]>,
    destination_pencils: Box<[Arc<Pencil<N, M>>]>,
    extra_shape: ExtraShape,
    storage: Vec<Complex<R>>,
    active: usize,
    cause: R2cError,
) -> R2cError {
    let error = retain_complex_error(array, source_pencils, extra_shape, storage, active, cause);
    array.real_pencils = Some(destination_pencils);
    error
}

fn fail_forward_complex<R: FftReal, const N: usize, const M: usize>(
    array: &mut R2cInPlaceArray<R, N, M>,
    source_pencils: Box<[Arc<Pencil<N, M>>]>,
    destination_pencils: Box<[Arc<Pencil<N, M>>]>,
    extra_shape: ExtraShape,
    storage: Vec<Complex<R>>,
    active: usize,
    cause: R2cError,
) -> R2cError {
    let error = retain_complex_error(
        array,
        destination_pencils,
        extra_shape,
        storage,
        active,
        cause,
    );
    array.real_pencils = Some(source_pencils);
    error
}

fn fail_reverse_real<R: FftReal, const N: usize, const M: usize>(
    array: &mut R2cInPlaceArray<R, N, M>,
    source_pencils: Box<[Arc<Pencil<N, M>>]>,
    destination_pencils: Box<[Arc<Pencil<N, M>>]>,
    extra_shape: ExtraShape,
    storage: Vec<R>,
    active: usize,
    cause: R2cError,
) -> R2cError {
    let error = retain_real_error(
        array,
        destination_pencils,
        extra_shape,
        storage,
        active,
        cause,
    );
    array.complex_pencils = Some(source_pencils);
    error
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
        mut report: Option<&mut TransformTiming<N>>,
    ) -> Result<(), R2cError> {
        let started = Instant::now();
        let communicator = self.input_pencil().topology().communicator();
        agree_execution_descriptor_ref::<N, M>(
            communicator,
            if report.is_some() {
                OPERATION_R2C_FORWARD_TIMED
            } else {
                OPERATION_R2C_FORWARD
            },
            &self.core.descriptor,
        )?;
        let preflight = self.preflight_forward(source, destination, workspace);
        if !collective_valid(communicator, preflight.is_ok()) {
            return Err(preflight
                .err()
                .unwrap_or(R2cError::Fft(FftError::CollectivePreconditionFailed)));
        }
        preflight.expect("distributed R2C forward preflight succeeded");
        let result = execute_forward(
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

    fn execute_overlap_forward(
        &self,
        source: &PencilArray<R, N, M>,
        destination: &mut PencilArray<Complex<R>, N, M>,
        workspace: &mut R2cWorkspace<R, N, M>,
    ) -> Result<(), FftOverlapError<R2cError>> {
        let communicator = self.input_pencil().topology().communicator();
        agree_execution_descriptor_ref::<N, M>(
            communicator,
            OPERATION_R2C_FORWARD_OVERLAP,
            &self.core.descriptor,
        )?;
        let preflight = self.preflight_forward(source, destination, workspace);
        if !collective_valid(communicator, preflight.is_ok()) {
            return Err(preflight
                .err()
                .unwrap_or(R2cError::Fft(FftError::CollectivePreconditionFailed))
                .into());
        }
        if !collective_valid(
            communicator,
            overlap_supported(&self.core, Direction::Forward),
        ) {
            return Err(FftOverlapError::UnsupportedTransport);
        }
        execute_forward_overlap(&self.core, source, destination, workspace)
    }

    fn execute_overlap_reverse(
        &self,
        direction: Direction,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &mut PencilArray<R, N, M>,
        workspace: &mut R2cWorkspace<R, N, M>,
    ) -> Result<(), FftOverlapError<R2cError>> {
        let communicator = self.input_pencil().topology().communicator();
        let operation = if matches!(direction, Direction::Inverse) {
            OPERATION_R2C_INVERSE_OVERLAP
        } else {
            OPERATION_R2C_BACKWARD_OVERLAP
        };
        agree_execution_descriptor_ref::<N, M>(communicator, operation, &self.core.descriptor)?;
        let preflight = self.preflight_inverse(source, destination, workspace);
        if !collective_valid(communicator, preflight.is_ok()) {
            return Err(preflight
                .err()
                .unwrap_or(R2cError::Fft(FftError::CollectivePreconditionFailed))
                .into());
        }
        if !collective_valid(communicator, overlap_supported(&self.core, direction)) {
            return Err(FftOverlapError::UnsupportedTransport);
        }
        execute_reverse_overlap(
            &self.core,
            source,
            destination,
            workspace,
            matches!(direction, Direction::Inverse),
            self.raw_absolute_threshold,
        )
    }

    fn execute_reverse(
        &self,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &mut PencilArray<R, N, M>,
        workspace: &mut R2cWorkspace<R, N, M>,
        normalize_inverse: bool,
        mut report: Option<&mut TransformTiming<N>>,
    ) -> Result<(), R2cError> {
        let started = Instant::now();
        let communicator = self.input_pencil().topology().communicator();
        let operation = if normalize_inverse {
            if report.is_some() {
                OPERATION_R2C_INVERSE_TIMED
            } else {
                OPERATION_R2C_INVERSE
            }
        } else {
            if report.is_some() {
                OPERATION_R2C_BACKWARD_TIMED
            } else {
                OPERATION_R2C_BACKWARD
            }
        };
        agree_execution_descriptor_ref::<N, M>(communicator, operation, &self.core.descriptor)?;
        let preflight = self.preflight_inverse(source, destination, workspace);
        if !collective_valid(communicator, preflight.is_ok()) {
            return Err(preflight
                .err()
                .unwrap_or(R2cError::Fft(FftError::CollectivePreconditionFailed)));
        }
        preflight.expect("distributed R2C reverse preflight succeeded");
        let result = execute_inverse(
            &self.core,
            source,
            destination,
            workspace,
            normalize_inverse,
            self.raw_absolute_threshold,
            report.as_deref_mut(),
        );
        if let Some(timing) = report {
            timing.total = started.elapsed();
        }
        result
    }

    fn execute_in_place(
        &self,
        direction: Direction,
        array: &mut R2cInPlaceArray<R, N, M>,
        workspace: &mut R2cInPlaceWorkspace<R, N, M>,
        mut report: Option<&mut TransformTiming<N>>,
    ) -> Result<(), R2cError> {
        let started = Instant::now();
        let communicator = self.input_pencil().topology().communicator();
        let operation = match direction {
            Direction::Forward => {
                if report.is_some() {
                    OPERATION_R2C_FORWARD_IN_PLACE_TIMED
                } else {
                    OPERATION_R2C_FORWARD_IN_PLACE
                }
            }
            Direction::Inverse => {
                if report.is_some() {
                    OPERATION_R2C_INVERSE_IN_PLACE_TIMED
                } else {
                    OPERATION_R2C_INVERSE_IN_PLACE
                }
            }
            Direction::Backward => {
                if report.is_some() {
                    OPERATION_R2C_BACKWARD_IN_PLACE_TIMED
                } else {
                    OPERATION_R2C_BACKWARD_IN_PLACE
                }
            }
        };
        agree_execution_descriptor_ref::<N, M>(communicator, operation, &self.core.descriptor)?;
        let preflight = self.preflight_in_place(direction, array, workspace);
        if !collective_valid(communicator, preflight.is_ok()) {
            return Err(preflight
                .err()
                .unwrap_or(R2cError::Fft(FftError::CollectivePreconditionFailed)));
        }
        preflight.expect("distributed R2C in-place preflight succeeded");

        // The owner is deliberately detached by the phase helpers. A panic or
        // a post-start error therefore cannot expose a partly typed buffer.
        array.state = R2cState::Poisoned;
        let result = match direction {
            Direction::Forward => {
                execute_in_place_forward(&self.core, array, workspace, report.as_deref_mut())
            }
            Direction::Inverse | Direction::Backward => execute_in_place_reverse(
                &self.core,
                array,
                workspace,
                matches!(direction, Direction::Inverse),
                self.raw_absolute_threshold,
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

    fn preflight_in_place(
        &self,
        direction: Direction,
        array: &R2cInPlaceArray<R, N, M>,
        workspace: &R2cInPlaceWorkspace<R, N, M>,
    ) -> Result<(), R2cError> {
        if !Arc::ptr_eq(&array.core, &self.core) {
            return Err(FftError::Array(ArrayError::IncompatiblePencils).into());
        }
        if !Arc::ptr_eq(&workspace.core, &self.core) {
            return Err(FftError::WorkspaceMismatch.into());
        }
        let expected_state = match direction {
            Direction::Forward => R2cState::RealInput,
            Direction::Inverse | Direction::Backward => R2cState::ComplexOutput,
        };
        match array.state {
            R2cState::Poisoned => return Err(FftError::Array(ArrayError::Poisoned).into()),
            state if state != expected_state => return Err(FftError::InputLayoutMismatch.into()),
            _ => {}
        }

        let boundary = self
            .core
            .real_stage_index
            .ok_or(FftError::PreparationFailed)?;
        if array.storage.is_none() {
            return Err(FftError::StorageLayoutMismatch.into());
        }
        let expected_pencil = match direction {
            Direction::Forward => self.input_pencil(),
            Direction::Inverse | Direction::Backward => self.output_pencil(),
        };
        let inactive_registry_valid = match direction {
            Direction::Forward => array.complex_pencils.as_ref().is_some_and(|pencils| {
                registered_layout_matches(pencils, &self.core.stages[boundary..], false)
            }),
            Direction::Inverse | Direction::Backward => {
                array.real_pencils.as_ref().is_some_and(|pencils| {
                    registered_layout_matches(pencils, &self.core.stages[..=boundary], true)
                })
            }
        };
        if !inactive_registry_valid {
            return Err(FftError::WorkspaceMismatch.into());
        }
        match (direction, array.storage.as_ref().expect("checked above")) {
            (Direction::Forward, R2cInPlaceStorage::Real(real)) => {
                if real.extra_shape() != &self.core.extra_shape
                    || !registered_layout_matches(
                        real.pencils(),
                        &self.core.stages[..=boundary],
                        true,
                    )
                {
                    return Err(FftError::StorageLayoutMismatch.into());
                }
                if !real
                    .active_pencil()
                    .map_err(FftError::Array)?
                    .same_layout(expected_pencil.as_ref())
                {
                    return Err(FftError::InputLayoutMismatch.into());
                }
            }
            (Direction::Inverse | Direction::Backward, R2cInPlaceStorage::Complex(complex)) => {
                if complex.extra_shape() != &self.core.extra_shape
                    || !registered_layout_matches(
                        complex.pencils(),
                        &self.core.stages[boundary..],
                        false,
                    )
                {
                    return Err(FftError::StorageLayoutMismatch.into());
                }
                if !complex
                    .active_pencil()
                    .map_err(FftError::Array)?
                    .same_layout(expected_pencil.as_ref())
                {
                    return Err(FftError::InputLayoutMismatch.into());
                }
            }
            _ => return Err(FftError::StorageLayoutMismatch.into()),
        }

        let (required_real, required_complex, required_bytes, _) =
            in_place_storage_requirements(&self.core)?;
        if required_real != array.real_storage_len
            || required_complex != array.complex_storage_len
            || required_bytes != array.storage_bytes
        {
            return Err(FftError::StorageLayoutMismatch.into());
        }
        // Prove every strided phase span and divisibility condition before the
        // array is poisoned. The conversion routines contain no data-sized
        // allocation after this point.
        let boundary_stride = super::memory_stride(
            self.core.stages[boundary].input.as_ref(),
            self.core.stages[boundary].axis,
        )?;
        let strided = strided_phase_shape(&self.core)?;
        if boundary_stride > 1 && strided.is_none() {
            return Err(FftError::PreparationFailed.into());
        }
        // Check the actual Vec length and capacity, not the requested
        // allocation size. A real Vec with an odd scalar capacity cannot be
        // recast to Complex even when its current length and total byte count
        // look sufficient.
        match array.storage.as_ref().expect("checked above") {
            R2cInPlaceStorage::Real(real) => {
                if real.storage_len() != required_real {
                    return Err(FftError::StorageLayoutMismatch.into());
                }
                validate_recast_capacity::<R, Complex<R>>(real.storage_capacity(), required_bytes)?
            }
            R2cInPlaceStorage::Complex(complex) => {
                if complex.storage_len() != required_complex {
                    return Err(FftError::StorageLayoutMismatch.into());
                }
                validate_recast_capacity::<Complex<R>, R>(
                    complex.storage_capacity(),
                    required_bytes,
                )?
            }
            R2cInPlaceStorage::PoisonedReal(_) | R2cInPlaceStorage::PoisonedComplex(_) => {
                return Err(FftError::StorageLayoutMismatch.into());
            }
        }

        validate_workspace_lengths_values(
            workspace.fft_scratch.len(),
            self.core.fft_scratch_len,
            workspace.transpose.send_len(),
            self.core.transpose_send_len,
            workspace.transpose.receive_len(),
            self.core.transpose_receive_len,
        )?;
        if boundary > 0 {
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
        // The inactive registry is moved into the next typed Many array at
        // the boundary. Check that it is available while collective
        // preflight is still active, before any payload or data write starts.
        let registry_ready = match direction {
            Direction::Forward => array.complex_pencils.is_some(),
            Direction::Inverse | Direction::Backward => array.real_pencils.is_some(),
        };
        if !registry_ready {
            return Err(FftError::WorkspaceMismatch.into());
        }

        let (real_len, complex_len) = r2c_lengths(&self.core);
        if boundary_stride > 1 {
            let actual = workspace.real_source_line.as_ref().map_or(0, Vec::len);
            if actual < real_len {
                return Err(FftError::WorkspaceTooSmall {
                    kind: "real source line",
                    required: real_len,
                    actual,
                }
                .into());
            }
        }
        for (actual, required, kind) in [
            (workspace.real_line.len(), real_len, "real line"),
            (
                workspace.complex_source_line.len(),
                complex_len,
                "complex source line",
            ),
            (workspace.complex_line.len(), complex_len, "complex line"),
            (
                workspace.complex_strided_line.len(),
                self.core.strided_line_len,
                "complex line",
            ),
        ] {
            if actual < required {
                return Err(FftError::WorkspaceTooSmall {
                    kind,
                    required,
                    actual,
                }
                .into());
            }
        }
        Ok(())
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
        let boundary_stride = super::memory_stride(
            self.core.stages[boundary].input.as_ref(),
            self.core.stages[boundary].axis,
        )?;
        if boundary_stride > 1 {
            let actual = workspace.real_source_line.as_ref().map_or(0, Vec::len);
            if actual < real_len {
                return Err(FftError::WorkspaceTooSmall {
                    kind: "real source line",
                    required: real_len,
                    actual,
                }
                .into());
            }
        }
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
        if workspace.complex_strided_line.len() < self.core.strided_line_len {
            return Err(FftError::WorkspaceTooSmall {
                kind: "complex line",
                required: self.core.strided_line_len,
                actual: workspace.complex_strided_line.len(),
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
    mut report: Option<&mut TransformTiming<N>>,
) -> Result<(), R2cError>
where
    Complex<R>: Equivalence,
{
    let boundary = core.real_stage_index.ok_or(FftError::PreparationFailed)?;
    let source_view = source.view();
    if boundary == 0 {
        let stage = &core.stages[boundary];
        let stride = super::memory_stride(stage.input.as_ref(), stage.axis)?;
        let real_line = &mut workspace.real_line;
        let real_source_line = if stride > 1 {
            workspace
                .real_source_line
                .as_mut()
                .expect("strided R2C source line was preflighted")
                .as_mut_slice()
        } else {
            &mut []
        };
        let complex_line = &mut workspace.complex_line;
        let fft_scratch = &mut workspace.fft_scratch;
        let fft_started = Instant::now();
        workspace
            .intermediate
            .overwrite_with(stage.output.as_ref(), |mut target| {
                if stride > 1 {
                    execute_strided_real_forward(
                        stage.local.real_complex(),
                        stage.input.as_ref(),
                        stage.axis,
                        source_view.as_slice(),
                        target.as_mut_slice(),
                        real_source_line,
                        real_line,
                        complex_line,
                        fft_scratch,
                    )
                    .expect("distributed R2C strided forward was preflighted");
                } else {
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
                }
                Ok::<_, ()>(())
            })
            .expect("distributed R2C stage-zero overwrite was preflighted");
        super::record_fft_timing(&mut report, boundary, fft_started);
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
                super::execute_transition_timed(
                    &core.transitions[index].forward,
                    real_intermediate,
                    real_transpose,
                    report.as_deref_mut(),
                    index,
                )?;
            }
        }
        let active = real_intermediate.active_view().map_err(FftError::Array)?;
        let stage = &core.stages[boundary];
        let stride = super::memory_stride(stage.input.as_ref(), stage.axis)?;
        let real_line = &mut workspace.real_line;
        let real_source_line = if stride > 1 {
            workspace
                .real_source_line
                .as_mut()
                .expect("strided R2C source line was preflighted")
                .as_mut_slice()
        } else {
            &mut []
        };
        let complex_line = &mut workspace.complex_line;
        let fft_scratch = &mut workspace.fft_scratch;
        let fft_started = Instant::now();
        workspace
            .intermediate
            .overwrite_with(stage.output.as_ref(), |mut target| {
                if stride > 1 {
                    execute_strided_real_forward(
                        stage.local.real_complex(),
                        stage.input.as_ref(),
                        stage.axis,
                        active.as_slice(),
                        target.as_mut_slice(),
                        real_source_line,
                        real_line,
                        complex_line,
                        fft_scratch,
                    )
                    .expect("distributed R2C strided forward was preflighted");
                } else {
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
                }
                Ok::<_, ()>(())
            })
            .expect("distributed R2C boundary overwrite was preflighted");
        super::record_fft_timing(&mut report, boundary, fft_started);
    }

    let tail = &core.stages[boundary..];
    let tail_transitions = &core.transitions[boundary..];
    if let Some(timing) = report {
        // The shared tail helper numbers stages from zero.  Keep the real
        // prefix timings intact while relocating both FFT and communication
        // (pack/unpack/wait) fields to their route indices.
        let mut tail_timing = TransformTiming::default();
        super::execute_forward_complex_tail_timed(
            tail,
            tail_transitions,
            &mut workspace.intermediate,
            &mut workspace.transpose,
            destination,
            &mut workspace.fft_scratch,
            &mut workspace.complex_strided_line,
            Some(&mut tail_timing),
        )?;
        // Stage zero of this helper is the already-computed real boundary:
        // retain its kernel measurement and attach only its outgoing transition.
        let boundary_fft = timing.stages[boundary].fft;
        let boundary_calls = timing.stages[boundary].fft_calls;
        for index in 0..tail.len() {
            timing.stages[boundary + index] = tail_timing.stages[index];
        }
        timing.stages[boundary].fft = boundary_fft;
        timing.stages[boundary].fft_calls = boundary_calls;
        timing.stages[boundary].total = boundary_fft + timing.stages[boundary].transpose;
    } else {
        super::execute_forward_complex_tail_timed(
            tail,
            tail_transitions,
            &mut workspace.intermediate,
            &mut workspace.transpose,
            destination,
            &mut workspace.fft_scratch,
            &mut workspace.complex_strided_line,
            None,
        )?;
    }
    Ok(())
}

fn overlap_supported<R: FftReal, const N: usize, const M: usize>(
    core: &TransformPlanCore<R, N, M>,
    direction: Direction,
) -> bool {
    core.transitions.iter().all(|transition| {
        let item = match direction {
            Direction::Forward => &transition.forward,
            Direction::Inverse | Direction::Backward => &transition.backward,
        };
        matches!(
            item,
            C2cTransition::Identity | C2cTransition::Local(_) | C2cTransition::PointToPoint(_)
        )
    })
}

fn execute_forward_overlap<R: FftReal, const N: usize, const M: usize>(
    core: &Arc<TransformPlanCore<R, N, M>>,
    source: &PencilArray<R, N, M>,
    destination: &mut PencilArray<Complex<R>, N, M>,
    workspace: &mut R2cWorkspace<R, N, M>,
) -> Result<(), FftOverlapError<R2cError>>
where
    Complex<R>: Equivalence,
{
    let boundary = core.real_stage_index.ok_or(FftError::PreparationFailed)?;
    if boundary > 0 {
        let real = workspace
            .real_intermediate
            .as_mut()
            .ok_or(FftError::WorkspaceMismatch)?;
        let view = source.view();
        real.overwrite_with(core.stages[0].output.as_ref(), |mut target| {
            target.as_mut_slice().copy_from_slice(view.as_slice());
            Ok::<_, ()>(())
        })
        .expect("preflighted real prefix");

        // The boundary RFFT belongs to the receive side of the last real
        // transition.  This is the only way the real prefix can overlap its
        // final redistribution (and also handles a boundary at the last stage).
        let prefix = boundary - 1;
        let rt = workspace
            .real_transpose
            .as_mut()
            .ok_or(FftError::WorkspaceMismatch)?;
        for index in 0..prefix {
            super::execute_transition(&core.transitions[index].forward, real, rt)?;
        }
        let transition = &core.transitions[prefix].forward;
        let stage = &core.stages[boundary];
        match transition {
            C2cTransition::PointToPoint(plan) => {
                let stride = super::memory_stride(stage.input.as_ref(), stage.axis)?;
                let real_source_line = if stride > 1 {
                    workspace
                        .real_source_line
                        .as_mut()
                        .expect("strided R2C source line was preflighted")
                        .as_mut_slice()
                } else {
                    &mut []
                };
                let real_line = &mut workspace.real_line;
                let complex_line = &mut workspace.complex_line;
                let fft_scratch = &mut workspace.fft_scratch;
                let intermediate = &mut workspace.intermediate;
                intermediate
                    .overwrite_with(stage.output.as_ref(), |mut target| {
                        let callback = |data: &mut [R]| {
                            if stride > 1 {
                                execute_strided_real_forward(
                                    stage.local.real_complex(),
                                    stage.input.as_ref(),
                                    stage.axis,
                                    data,
                                    target.as_mut_slice(),
                                    real_source_line,
                                    real_line,
                                    complex_line,
                                    fft_scratch,
                                )
                            } else {
                                stage
                                    .local
                                    .real_complex()
                                    .forward(data, target.as_mut_slice(), real_line, fft_scratch)
                                    .map_err(R2cError::LocalR2c)
                            }
                        };
                        plan.execute_in_place_with_callback(real, rt, callback)
                            .map_err(map_r2c_overlap)
                    })
                    .map_err(|error| match error {
                        OverwriteError::Array(error) => {
                            FftOverlapError::Operation(R2cError::Fft(FftError::Array(error)))
                        }
                        OverwriteError::Writer(error) => error,
                    })?;
            }
            C2cTransition::Identity | C2cTransition::Local(_) => {
                super::execute_transition(transition, real, rt)?;
                let active = real.active_view().map_err(FftError::Array)?;
                let stride = super::memory_stride(stage.input.as_ref(), stage.axis)?;
                let real_source_line = if stride > 1 {
                    workspace
                        .real_source_line
                        .as_mut()
                        .expect("strided R2C source line was preflighted")
                        .as_mut_slice()
                } else {
                    &mut []
                };
                let real_line = &mut workspace.real_line;
                let complex_line = &mut workspace.complex_line;
                let fft_scratch = &mut workspace.fft_scratch;
                workspace
                    .intermediate
                    .overwrite_with(stage.output.as_ref(), |mut target| {
                        if stride > 1 {
                            execute_strided_real_forward(
                                stage.local.real_complex(),
                                stage.input.as_ref(),
                                stage.axis,
                                active.as_slice(),
                                target.as_mut_slice(),
                                real_source_line,
                                real_line,
                                complex_line,
                                fft_scratch,
                            )
                        } else {
                            stage
                                .local
                                .real_complex()
                                .forward(
                                    active.as_slice(),
                                    target.as_mut_slice(),
                                    real_line,
                                    fft_scratch,
                                )
                                .map_err(R2cError::LocalR2c)
                        }
                    })
                    .map_err(|error| match error {
                        OverwriteError::Array(error) => R2cError::Fft(FftError::Array(error)),
                        OverwriteError::Writer(error) => error,
                    })?;
            }
            C2cTransition::AllToAllv(_) => {
                return Err(FftOverlapError::UnsupportedTransport);
            }
        }
    } else {
        let stage = &core.stages[boundary];
        let view = source.view();
        let stride = super::memory_stride(stage.input.as_ref(), stage.axis)?;
        let real_source_line = if stride > 1 {
            workspace
                .real_source_line
                .as_mut()
                .expect("strided R2C source line was preflighted")
                .as_mut_slice()
        } else {
            &mut []
        };
        workspace
            .intermediate
            .overwrite_with(stage.output.as_ref(), |mut target| {
                execute_strided_real_forward(
                    stage.local.real_complex(),
                    stage.input.as_ref(),
                    stage.axis,
                    view.as_slice(),
                    target.as_mut_slice(),
                    real_source_line,
                    &mut workspace.real_line,
                    &mut workspace.complex_line,
                    &mut workspace.fft_scratch,
                )
            })
            .map_err(|error| match error {
                OverwriteError::Array(error) => R2cError::Fft(FftError::Array(error)),
                OverwriteError::Writer(error) => error,
            })?;
    }
    for index in boundary..core.transitions.len() {
        let stage = &core.stages[index + 1];
        match &core.transitions[index].forward {
            C2cTransition::Identity => {
                super::execute_complex_forward_in_place(
                    &stage.local,
                    stage.output.as_ref(),
                    stage.axis,
                    workspace
                        .intermediate
                        .active_view_mut()
                        .map_err(FftError::Array)?
                        .as_mut_slice(),
                    &mut workspace.fft_scratch,
                    &mut workspace.complex_strided_line,
                )?;
            }
            C2cTransition::Local(plan) => {
                plan.execute_in_place_with_transpose_workspace(
                    &mut workspace.intermediate,
                    &mut workspace.transpose,
                )
                .expect("distributed R2C local transition was preflighted");
                super::execute_complex_forward_in_place(
                    &stage.local,
                    stage.output.as_ref(),
                    stage.axis,
                    workspace
                        .intermediate
                        .active_view_mut()
                        .map_err(FftError::Array)?
                        .as_mut_slice(),
                    &mut workspace.fft_scratch,
                    &mut workspace.complex_strided_line,
                )?;
            }
            C2cTransition::PointToPoint(plan) => {
                let callback = |data: &mut [Complex<R>]| {
                    #[cfg(test)]
                    super::consume_r2c_callback_injection()?;
                    super::execute_complex_forward_in_place(
                        &stage.local,
                        stage.output.as_ref(),
                        stage.axis,
                        data,
                        &mut workspace.fft_scratch,
                        &mut workspace.complex_strided_line,
                    )
                };
                plan.execute_in_place_with_callback(
                    &mut workspace.intermediate,
                    &mut workspace.transpose,
                    callback,
                )
                .map_err(map_r2c_overlap)?;
            }
            C2cTransition::AllToAllv(_) => {
                return Err(FftOverlapError::UnsupportedTransport);
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

fn map_r2c_overlap<E: Into<R2cError>>(error: OverlapError<E>) -> FftOverlapError<R2cError> {
    FftOverlapError::Overlap(match error {
        OverlapError::Transpose(error) => OverlapError::Transpose(error),
        OverlapError::Callback(error) => OverlapError::Callback(error.into()),
        OverlapError::PeerPanicked => OverlapError::PeerPanicked,
        OverlapError::PeerCallbackFailed => OverlapError::PeerCallbackFailed,
        OverlapError::CollectivePreconditionFailed => OverlapError::CollectivePreconditionFailed,
    })
}

fn execute_reverse_overlap<R: FftReal, const N: usize, const M: usize>(
    core: &Arc<TransformPlanCore<R, N, M>>,
    source: &PencilArray<Complex<R>, N, M>,
    destination: &mut PencilArray<R, N, M>,
    workspace: &mut R2cWorkspace<R, N, M>,
    normalize: bool,
    threshold: f64,
) -> Result<(), FftOverlapError<R2cError>>
where
    Complex<R>: Equivalence,
{
    let boundary = core.real_stage_index.ok_or(FftError::PreparationFailed)?;
    let last = core.stages.len() - 1;
    let source_view = source.view();
    let last_stage = &core.stages[last];
    workspace
        .intermediate
        .overwrite_with(last_stage.output.as_ref(), |mut target| {
            if boundary == last {
                if target.as_mut_slice().len() != source_view.as_slice().len() {
                    return Err(FftError::PreparationFailed);
                }
                target
                    .as_mut_slice()
                    .copy_from_slice(source_view.as_slice());
                return Ok(());
            }
            super::execute_complex_reverse(
                &last_stage.local,
                last_stage.input.as_ref(),
                last_stage.axis,
                source_view.as_slice(),
                target.as_mut_slice(),
                &mut workspace.fft_scratch,
                &mut workspace.complex_strided_line,
                normalize,
            )
        })
        .map_err(|error| match error {
            OverwriteError::Array(error) => R2cError::Fft(FftError::Array(error)),
            OverwriteError::Writer(error) => error.into(),
        })?;

    // Every complex transition except the handoff to C2R has the next local
    // FFT in its receive callback.  Identity/local routes have no receive to
    // overlap, but still execute the same next stage instead of rejecting the
    // valid permute_dims=false layout.
    for index in (boundary + 1..last).rev() {
        let stage = &core.stages[index];
        match &core.transitions[index].backward {
            C2cTransition::Identity => {
                super::execute_complex_reverse_in_place(
                    &stage.local,
                    stage.input.as_ref(),
                    stage.axis,
                    workspace
                        .intermediate
                        .active_view_mut()
                        .map_err(FftError::Array)?
                        .as_mut_slice(),
                    &mut workspace.fft_scratch,
                    &mut workspace.complex_strided_line,
                    normalize,
                )?;
            }
            C2cTransition::Local(plan) => {
                plan.execute_in_place_with_transpose_workspace(
                    &mut workspace.intermediate,
                    &mut workspace.transpose,
                )
                .expect("distributed R2C local transition was preflighted");
                super::execute_complex_reverse_in_place(
                    &stage.local,
                    stage.input.as_ref(),
                    stage.axis,
                    workspace
                        .intermediate
                        .active_view_mut()
                        .map_err(FftError::Array)?
                        .as_mut_slice(),
                    &mut workspace.fft_scratch,
                    &mut workspace.complex_strided_line,
                    normalize,
                )?;
            }
            C2cTransition::PointToPoint(plan) => {
                let callback = |data: &mut [Complex<R>]| {
                    super::execute_complex_reverse_in_place(
                        &stage.local,
                        stage.input.as_ref(),
                        stage.axis,
                        data,
                        &mut workspace.fft_scratch,
                        &mut workspace.complex_strided_line,
                        normalize,
                    )
                };
                plan.execute_in_place_with_callback(
                    &mut workspace.intermediate,
                    &mut workspace.transpose,
                    callback,
                )
                .map_err(map_r2c_overlap)?;
            }
            C2cTransition::AllToAllv(_) => {
                return Err(FftOverlapError::UnsupportedTransport);
            }
        }
    }

    validate_boundary(core, &workspace.intermediate, normalize, threshold)?;
    zero_accepted_boundary(core, &mut workspace.intermediate)?;
    let stage = &core.stages[boundary];

    // Keep the boundary C2R inside the receive callback.  In particular, do
    // not wait for the complex send before starting the real work.
    if boundary < last {
        match &core.transitions[boundary].backward {
            C2cTransition::PointToPoint(plan) => {
                if boundary == 0 {
                    let mut target = destination.view_mut();
                    let callback = |data: &mut [Complex<R>]| {
                        execute_boundary_real_reverse(
                            stage,
                            data,
                            target.as_mut_slice(),
                            &mut workspace.complex_strided_line,
                            &mut workspace.complex_line,
                            &mut workspace.real_line,
                            &mut workspace.fft_scratch,
                            normalize,
                        )
                    };
                    plan.execute_in_place_with_callback(
                        &mut workspace.intermediate,
                        &mut workspace.transpose,
                        callback,
                    )
                    .map_err(map_r2c_overlap)?;
                    return Ok(());
                }
                let real = workspace
                    .real_intermediate
                    .as_mut()
                    .ok_or(FftError::WorkspaceMismatch)?;
                real.overwrite_with(stage.input.as_ref(), |mut target| {
                    let callback = |data: &mut [Complex<R>]| {
                        execute_boundary_real_reverse(
                            stage,
                            data,
                            target.as_mut_slice(),
                            &mut workspace.complex_strided_line,
                            &mut workspace.complex_line,
                            &mut workspace.real_line,
                            &mut workspace.fft_scratch,
                            normalize,
                        )
                    };
                    plan.execute_in_place_with_callback(
                        &mut workspace.intermediate,
                        &mut workspace.transpose,
                        callback,
                    )
                    .map_err(map_r2c_overlap)
                })
                .map_err(|error| match error {
                    OverwriteError::Array(error) => {
                        FftOverlapError::Operation(R2cError::Fft(FftError::Array(error)))
                    }
                    OverwriteError::Writer(error) => error,
                })?;
            }
            transition => {
                super::execute_transition(
                    transition,
                    &mut workspace.intermediate,
                    &mut workspace.transpose,
                )?;
                execute_boundary_c2r_after_transition(
                    core,
                    stage,
                    destination,
                    workspace,
                    boundary,
                    normalize,
                )?;
            }
        }
    } else {
        execute_boundary_c2r_after_transition(
            core,
            stage,
            destination,
            workspace,
            boundary,
            normalize,
        )?;
    }

    if boundary == 0 {
        return Ok(());
    }
    let real = workspace
        .real_intermediate
        .as_mut()
        .ok_or(FftError::WorkspaceMismatch)?;
    let real_transpose = workspace
        .real_transpose
        .as_mut()
        .ok_or(FftError::WorkspaceMismatch)?;
    for index in (0..boundary).rev() {
        super::execute_transition(&core.transitions[index].backward, real, real_transpose)?;
    }
    let active = real.active_view().map_err(FftError::Array)?;
    destination
        .view_mut()
        .as_mut_slice()
        .copy_from_slice(active.as_slice());
    Ok(())
}

fn execute_boundary_real_reverse<R: FftReal, const N: usize, const M: usize>(
    stage: &TransformStage<R, N, M>,
    source: &[Complex<R>],
    destination: &mut [R],
    complex_source_line: &mut [Complex<R>],
    complex_line: &mut [Complex<R>],
    real_line: &mut [R],
    scratch: &mut [Complex<R>],
    normalize: bool,
) -> Result<(), R2cError>
where
    Complex<R>: Equivalence,
{
    execute_strided_real_reverse(
        stage.local.real_complex(),
        stage.output.as_ref(),
        stage.axis,
        source,
        destination,
        complex_source_line,
        complex_line,
        real_line,
        scratch,
        normalize,
    )
}

fn execute_boundary_c2r_after_transition<R: FftReal, const N: usize, const M: usize>(
    _core: &TransformPlanCore<R, N, M>,
    stage: &TransformStage<R, N, M>,
    destination: &mut PencilArray<R, N, M>,
    workspace: &mut R2cWorkspace<R, N, M>,
    boundary: usize,
    normalize: bool,
) -> Result<(), R2cError>
where
    Complex<R>: Equivalence,
{
    let active = workspace
        .intermediate
        .active_view()
        .map_err(FftError::Array)?;
    if boundary == 0 {
        return execute_boundary_real_reverse(
            stage,
            active.as_slice(),
            destination.view_mut().as_mut_slice(),
            &mut workspace.complex_strided_line,
            &mut workspace.complex_line,
            &mut workspace.real_line,
            &mut workspace.fft_scratch,
            normalize,
        );
    }
    let real = workspace
        .real_intermediate
        .as_mut()
        .ok_or(FftError::WorkspaceMismatch)?;
    real.overwrite_with(stage.input.as_ref(), |mut target| {
        execute_boundary_real_reverse(
            stage,
            active.as_slice(),
            target.as_mut_slice(),
            &mut workspace.complex_strided_line,
            &mut workspace.complex_line,
            &mut workspace.real_line,
            &mut workspace.fft_scratch,
            normalize,
        )
    })
    .map_err(|error| match error {
        OverwriteError::Array(error) => R2cError::Fft(FftError::Array(error)),
        OverwriteError::Writer(error) => error,
    })
}

fn shift_tail_timing<const N: usize>(
    report: &mut Option<&mut TransformTiming<N>>,
    offset: usize,
    count: usize,
) {
    if let Some(timing) = report.as_deref_mut() {
        for index in (0..count).rev() {
            timing.stages[offset + index] = timing.stages[index];
        }
        timing.stages[..offset].fill(super::StageTiming::default());
    }
}

fn execute_inverse<R: FftReal, const N: usize, const M: usize>(
    core: &Arc<TransformPlanCore<R, N, M>>,
    source: &PencilArray<Complex<R>, N, M>,
    destination: &mut PencilArray<R, N, M>,
    workspace: &mut R2cWorkspace<R, N, M>,
    normalize_inverse: bool,
    raw_absolute_threshold: f64,
    mut report: Option<&mut TransformTiming<N>>,
) -> Result<(), R2cError>
where
    Complex<R>: Equivalence,
{
    let boundary = core.real_stage_index.ok_or(FftError::PreparationFailed)?;
    super::execute_inverse_complex_tail_timed(
        &core.stages[boundary..],
        &core.transitions[boundary..],
        source,
        &mut workspace.intermediate,
        &mut workspace.transpose,
        &mut workspace.fft_scratch,
        &mut workspace.complex_strided_line,
        normalize_inverse,
        report.as_deref_mut(),
    )?;
    shift_tail_timing(&mut report, boundary, core.stages.len() - boundary);
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
    let stride = super::memory_stride(stage.output.as_ref(), stage.axis)?;
    if boundary == 0 {
        let mut destination_view = destination.view_mut();
        let fft_started = Instant::now();
        if stride > 1 {
            execute_strided_real_reverse(
                stage.local.real_complex(),
                stage.output.as_ref(),
                stage.axis,
                active.as_slice(),
                destination_view.as_mut_slice(),
                &mut workspace.complex_strided_line,
                &mut workspace.complex_line,
                &mut workspace.real_line,
                &mut workspace.fft_scratch,
                normalize_inverse,
            )?;
        } else if normalize_inverse {
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
        super::record_fft_timing(&mut report, boundary, fft_started);
        return Ok(());
    }

    let real_intermediate = workspace
        .real_intermediate
        .as_mut()
        .ok_or(FftError::WorkspaceMismatch)?;
    let fft_started = Instant::now();
    real_intermediate
        .overwrite_with(stage.input.as_ref(), |mut target| {
            if stride > 1 {
                execute_strided_real_reverse(
                    stage.local.real_complex(),
                    stage.output.as_ref(),
                    stage.axis,
                    active.as_slice(),
                    target.as_mut_slice(),
                    &mut workspace.complex_strided_line,
                    &mut workspace.complex_line,
                    &mut workspace.real_line,
                    &mut workspace.fft_scratch,
                    normalize_inverse,
                )?;
            } else if normalize_inverse {
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
            Ok::<_, R2cError>(())
        })
        .map_err(|error| match error {
            OverwriteError::Array(error) => R2cError::Fft(FftError::Array(error)),
            OverwriteError::Writer(error) => error,
        })?;
    super::record_fft_timing(&mut report, boundary, fft_started);
    {
        let real_transpose = workspace
            .real_transpose
            .as_mut()
            .ok_or(FftError::WorkspaceMismatch)?;
        for index in (0..boundary).rev() {
            super::execute_transition_timed(
                &core.transitions[index].backward,
                real_intermediate,
                real_transpose,
                report.as_deref_mut(),
                index,
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

impl<R: FftReal, const N: usize, const M: usize> R2cInPlaceArray<R, N, M> {
    /// Returns the current in-place completion state.
    pub fn state(&self) -> R2cState {
        self.state
    }

    /// Returns the backing allocation capacity in bytes while it is present.
    pub fn storage_capacity_bytes(&self) -> Option<usize> {
        let (capacity, element_size) = match self.storage.as_ref()? {
            R2cInPlaceStorage::Real(real) => (real.storage_capacity(), size_of::<R>()),
            R2cInPlaceStorage::Complex(complex) => {
                (complex.storage_capacity(), size_of::<Complex<R>>())
            }
            R2cInPlaceStorage::PoisonedReal(storage) => (storage.capacity(), size_of::<R>()),
            R2cInPlaceStorage::PoisonedComplex(storage) => {
                (storage.capacity(), size_of::<Complex<R>>())
            }
        };
        capacity.checked_mul(element_size)
    }

    /// Borrows the active real input view.
    pub fn real_view(&self) -> Result<PencilArrayView<'_, R, N, M>, R2cError> {
        match self.state {
            R2cState::Poisoned => Err(FftError::Array(ArrayError::Poisoned).into()),
            R2cState::ComplexOutput => Err(FftError::InputLayoutMismatch.into()),
            R2cState::RealInput => match self.storage.as_ref() {
                Some(R2cInPlaceStorage::Real(real)) => real
                    .active_view()
                    .map_err(FftError::Array)
                    .map_err(Into::into),
                _ => Err(FftError::StorageLayoutMismatch.into()),
            },
        }
    }

    /// Borrows the active mutable real input view.
    pub fn real_view_mut(&mut self) -> Result<PencilArrayViewMut<'_, R, N, M>, R2cError> {
        match self.state {
            R2cState::Poisoned => Err(FftError::Array(ArrayError::Poisoned).into()),
            R2cState::ComplexOutput => Err(FftError::InputLayoutMismatch.into()),
            R2cState::RealInput => match self.storage.as_mut() {
                Some(R2cInPlaceStorage::Real(real)) => real
                    .active_view_mut()
                    .map_err(FftError::Array)
                    .map_err(Into::into),
                _ => Err(FftError::StorageLayoutMismatch.into()),
            },
        }
    }

    /// Borrows the active reduced-complex output view.
    pub fn complex_view(&self) -> Result<PencilArrayView<'_, Complex<R>, N, M>, R2cError> {
        match self.state {
            R2cState::Poisoned => Err(FftError::Array(ArrayError::Poisoned).into()),
            R2cState::RealInput => Err(FftError::OutputLayoutMismatch.into()),
            R2cState::ComplexOutput => match self.storage.as_ref() {
                Some(R2cInPlaceStorage::Complex(complex)) => complex
                    .active_view()
                    .map_err(FftError::Array)
                    .map_err(Into::into),
                _ => Err(FftError::StorageLayoutMismatch.into()),
            },
        }
    }

    /// Borrows the active mutable reduced-complex output view.
    pub fn complex_view_mut(
        &mut self,
    ) -> Result<PencilArrayViewMut<'_, Complex<R>, N, M>, R2cError> {
        match self.state {
            R2cState::Poisoned => Err(FftError::Array(ArrayError::Poisoned).into()),
            R2cState::RealInput => Err(FftError::OutputLayoutMismatch.into()),
            R2cState::ComplexOutput => match self.storage.as_mut() {
                Some(R2cInPlaceStorage::Complex(complex)) => complex
                    .active_view_mut()
                    .map_err(FftError::Array)
                    .map_err(Into::into),
                _ => Err(FftError::StorageLayoutMismatch.into()),
            },
        }
    }
}

fn boundary_row_count<R: FftReal, const N: usize, const M: usize>(
    core: &TransformPlanCore<R, N, M>,
) -> Result<usize, R2cError> {
    let boundary = core.real_stage_index.ok_or(FftError::PreparationFailed)?;
    let real_len = core.stages[boundary].local.real_complex().real_len();
    let local_len = core.stages[boundary].input.local_len();
    if local_len % real_len != 0 {
        return Err(FftError::PreparationFailed.into());
    }
    (local_len / real_len)
        .checked_mul(core.extra_shape.element_count())
        .ok_or_else(|| FftError::PreparationFailed.into())
}

fn execute_in_place_forward<R: FftReal, const N: usize, const M: usize>(
    core: &Arc<TransformPlanCore<R, N, M>>,
    array: &mut R2cInPlaceArray<R, N, M>,
    workspace: &mut R2cInPlaceWorkspace<R, N, M>,
    mut report: Option<&mut TransformTiming<N>>,
) -> Result<(), R2cError>
where
    Complex<R>: Equivalence,
{
    let boundary = core.real_stage_index.ok_or(FftError::PreparationFailed)?;
    let mut real = match array.storage.take() {
        Some(R2cInPlaceStorage::Real(real)) => real,
        _ => return Err(FftError::StorageLayoutMismatch.into()),
    };
    if boundary > 0 {
        let real_transpose = workspace
            .real_transpose
            .as_mut()
            .ok_or(FftError::WorkspaceMismatch)?;
        for index in 0..boundary {
            super::execute_transition_timed(
                &core.transitions[index].forward,
                &mut real,
                real_transpose,
                report.as_deref_mut(),
                index,
            )?;
        }
    }
    array.storage = Some(R2cInPlaceStorage::Real(real));
    // Conversion is rank-local. Agree once for the complete handoff before
    // any rank can enter the first complex transpose.
    let fft_started = Instant::now();
    agree_result(
        core.stages[0].input.topology().communicator(),
        convert_real_to_complex(core, array, workspace),
    )?;
    super::record_fft_timing(&mut report, boundary, fft_started);

    let complex = match array.storage.as_mut() {
        Some(R2cInPlaceStorage::Complex(complex)) => complex,
        _ => return Err(FftError::StorageLayoutMismatch.into()),
    };
    for index in boundary..core.transitions.len() {
        super::execute_transition_timed(
            &core.transitions[index].forward,
            complex,
            &mut workspace.transpose,
            report.as_deref_mut(),
            index,
        )?;
        let stage = &core.stages[index + 1];
        let mut active = complex.active_view_mut().map_err(FftError::Array)?;
        let fft_started = Instant::now();
        super::execute_complex_forward_in_place(
            &stage.local,
            stage.output.as_ref(),
            stage.axis,
            active.as_mut_slice(),
            &mut workspace.fft_scratch,
            &mut workspace.complex_strided_line,
        )?;
        super::record_fft_timing(&mut report, index + 1, fft_started);
    }
    let output = core
        .stages
        .last()
        .ok_or(FftError::PreparationFailed)?
        .output
        .as_ref();
    if !complex
        .active_pencil()
        .map_err(FftError::Array)?
        .same_layout(output)
        || array.real_pencils.is_none()
        || array.complex_pencils.is_some()
    {
        return Err(FftError::PreparationFailed.into());
    }
    Ok(())
}

fn execute_in_place_reverse<R: FftReal, const N: usize, const M: usize>(
    core: &Arc<TransformPlanCore<R, N, M>>,
    array: &mut R2cInPlaceArray<R, N, M>,
    workspace: &mut R2cInPlaceWorkspace<R, N, M>,
    normalize_inverse: bool,
    raw_absolute_threshold: f64,
    mut report: Option<&mut TransformTiming<N>>,
) -> Result<(), R2cError>
where
    Complex<R>: Equivalence,
{
    let boundary = core.real_stage_index.ok_or(FftError::PreparationFailed)?;
    {
        let complex = match array.storage.as_mut() {
            Some(R2cInPlaceStorage::Complex(complex)) => complex,
            _ => return Err(FftError::StorageLayoutMismatch.into()),
        };
        let last = core
            .stages
            .len()
            .checked_sub(1)
            .ok_or(FftError::PreparationFailed)?;
        if last > boundary {
            let stage = &core.stages[last];
            let mut active = complex.active_view_mut().map_err(FftError::Array)?;
            let fft_started = Instant::now();
            super::execute_complex_reverse_in_place(
                &stage.local,
                stage.input.as_ref(),
                stage.axis,
                active.as_mut_slice(),
                &mut workspace.fft_scratch,
                &mut workspace.complex_strided_line,
                normalize_inverse,
            )?;
            super::record_fft_timing(&mut report, last, fft_started);
        }
        for index in (boundary..core.transitions.len()).rev() {
            super::execute_transition_timed(
                &core.transitions[index].backward,
                complex,
                &mut workspace.transpose,
                report.as_deref_mut(),
                index,
            )?;
            if index != boundary {
                let stage = &core.stages[index];
                let mut active = complex.active_view_mut().map_err(FftError::Array)?;
                let fft_started = Instant::now();
                super::execute_complex_reverse_in_place(
                    &stage.local,
                    stage.input.as_ref(),
                    stage.axis,
                    active.as_mut_slice(),
                    &mut workspace.fft_scratch,
                    &mut workspace.complex_strided_line,
                    normalize_inverse,
                )?;
                super::record_fft_timing(&mut report, index, fft_started);
            }
        }
    }

    {
        let complex = match array.storage.as_ref() {
            Some(R2cInPlaceStorage::Complex(complex)) => complex,
            _ => return Err(FftError::StorageLayoutMismatch.into()),
        };
        validate_boundary(core, complex, normalize_inverse, raw_absolute_threshold)?;
    }
    {
        let complex = match array.storage.as_mut() {
            Some(R2cInPlaceStorage::Complex(complex)) => complex,
            _ => return Err(FftError::StorageLayoutMismatch.into()),
        };
        zero_accepted_boundary(core, complex)?;
    }
    // The real representation handoff is the one rank-local phase in the
    // reverse route. All ranks must agree before the real-prefix transpose.
    let fft_started = Instant::now();
    agree_result(
        core.stages[0].input.topology().communicator(),
        convert_complex_to_real(core, array, workspace, normalize_inverse),
    )?;
    super::record_fft_timing(&mut report, boundary, fft_started);

    let real = match array.storage.as_mut() {
        Some(R2cInPlaceStorage::Real(real)) => real,
        _ => return Err(FftError::StorageLayoutMismatch.into()),
    };
    if boundary > 0 {
        let real_transpose = workspace
            .real_transpose
            .as_mut()
            .ok_or(FftError::WorkspaceMismatch)?;
        for index in (0..boundary).rev() {
            super::execute_transition_timed(
                &core.transitions[index].backward,
                real,
                real_transpose,
                report.as_deref_mut(),
                index,
            )?;
        }
    }
    if !real
        .active_pencil()
        .map_err(FftError::Array)?
        .same_layout(core.stages[0].input.as_ref())
        || array.real_pencils.is_some()
        || array.complex_pencils.is_none()
    {
        return Err(FftError::PreparationFailed.into());
    }
    Ok(())
}

#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
fn convert_strided_real_storage<R: FftReal, const N: usize, const M: usize>(
    storage: &mut [Complex<R>],
    plan: &LocalR2cPlan<R>,
    rows: usize,
    real_len: usize,
    complex_len: usize,
    stride: usize,
    workspace: &mut R2cInPlaceWorkspace<R, N, M>,
) -> Result<(), R2cError>
where
    Complex<R>: Equivalence,
{
    if stride <= 1 {
        return Err(FftError::PreparationFailed.into());
    }
    if rows == 0 {
        return Ok(());
    }
    if rows % stride != 0 {
        return Err(FftError::PreparationFailed.into());
    }
    let groups = rows / stride;
    let source_span = groups
        .checked_mul(real_len)
        .and_then(|value| value.checked_mul(stride))
        .ok_or(FftError::PreparationFailed)?;
    let destination_span = groups
        .checked_mul(complex_len)
        .and_then(|value| value.checked_mul(stride))
        .ok_or(FftError::PreparationFailed)?;
    if source_span > storage.len() || destination_span > storage.len() {
        return Err(FftError::StorageLayoutMismatch.into());
    }
    let real_source_line = workspace
        .real_source_line
        .as_mut()
        .ok_or(FftError::WorkspaceMismatch)?;
    if real_source_line.len() < real_len
        || workspace.real_line.len() < real_len
        || workspace.complex_line.len() < complex_len
    {
        return Err(FftError::WorkspaceTooSmall {
            kind: "real/complex line",
            required: real_len.max(complex_len),
            actual: real_source_line
                .len()
                .min(workspace.real_line.len())
                .min(workspace.complex_line.len()),
        }
        .into());
    }
    let real_line = &mut workspace.real_line;
    let complex_line = &mut workspace.complex_line;
    for outer in 0..groups {
        let source_base = outer
            .checked_mul(real_len)
            .and_then(|value| value.checked_mul(stride))
            .ok_or(FftError::PreparationFailed)?;
        let destination_base = outer
            .checked_mul(complex_len)
            .and_then(|value| value.checked_mul(stride))
            .ok_or(FftError::PreparationFailed)?;
        for inner in 0..stride {
            for j in 0..real_len {
                let index = source_base
                    .checked_add(j.checked_mul(stride).ok_or(FftError::PreparationFailed)?)
                    .and_then(|value| value.checked_add(inner))
                    .ok_or(FftError::PreparationFailed)?;
                real_source_line[j] = storage[index].re;
            }
            plan.forward(
                &real_source_line[..real_len],
                &mut complex_line[..complex_len],
                real_line,
                &mut workspace.fft_scratch,
            )
            .map_err(R2cError::LocalR2c)?;
            for k in 0..complex_len {
                let index = destination_base
                    .checked_add(k.checked_mul(stride).ok_or(FftError::PreparationFailed)?)
                    .and_then(|value| value.checked_add(inner))
                    .ok_or(FftError::PreparationFailed)?;
                storage[index] = complex_line[k];
            }
        }
    }
    Ok(())
}

fn convert_real_to_complex<R: FftReal, const N: usize, const M: usize>(
    core: &Arc<TransformPlanCore<R, N, M>>,
    array: &mut R2cInPlaceArray<R, N, M>,
    workspace: &mut R2cInPlaceWorkspace<R, N, M>,
) -> Result<(), R2cError>
where
    Complex<R>: Equivalence,
{
    let boundary = core.real_stage_index.ok_or(FftError::PreparationFailed)?;
    let (real_len, complex_len) = r2c_lengths(core);
    let rows = boundary_row_count(core)?;
    let scalar_size = size_of::<R>();
    let complex_size = size_of::<Complex<R>>();
    let full_complex_len =
        promoted_complex_len(core, array.real_storage_len, array.complex_storage_len)?;
    let full_bytes = full_complex_len
        .checked_mul(complex_size)
        .ok_or(FftError::PreparationFailed)?;
    if full_bytes % scalar_size != 0 {
        return Err(FftError::StorageLayoutMismatch.into());
    }
    let full_real_len = full_bytes / scalar_size;
    let destination_pencils = array
        .complex_pencils
        .take()
        .ok_or(FftError::StorageLayoutMismatch)?;
    let real = match array.storage.take() {
        Some(R2cInPlaceStorage::Real(real)) => real,
        _ => {
            array.complex_pencils = Some(destination_pencils);
            return Err(FftError::StorageLayoutMismatch.into());
        }
    };
    let (source_pencils, source_active, extra_shape, mut storage) =
        match real.into_parts_preserving() {
            Ok(parts) => parts,
            Err((error, real)) => {
                array.storage = Some(R2cInPlaceStorage::PoisonedReal(real.into_storage()));
                array.complex_pencils = Some(destination_pencils);
                return Err(map_array_allocation(error).into());
            }
        };
    #[cfg(test)]
    if array.test_hook == Some(InPlaceTestHook::PanicAfterForwardDetach) {
        panic!("injected R2C forward panic after owner detachment");
    }
    if full_real_len > storage.capacity() {
        return Err(fail_real_to_complex(
            array,
            source_pencils,
            destination_pencils,
            extra_shape,
            storage,
            source_active,
            FftError::StorageLayoutMismatch.into(),
        ));
    }
    storage.resize(full_real_len, R::zero());
    #[cfg(test)]
    if array.test_hook == Some(InPlaceTestHook::OddForwardCast) {
        assert!(storage.pop().is_some());
    }
    let mut complex_storage = match try_cast_vec(storage) {
        Ok(storage) => storage,
        Err((_, storage)) => {
            // `try_cast_vec` returns the original owner on failure. Keep that
            // raw owner in the poisoned object instead of dropping or trying
            // to expose it through a typed registry.
            array.storage = Some(R2cInPlaceStorage::PoisonedReal(storage));
            array.real_pencils = Some(source_pencils);
            array.complex_pencils = Some(destination_pencils);
            return Err(FftError::StorageLayoutMismatch.into());
        }
    };

    let plan = core.stages[boundary].local.real_complex();
    let stride = super::memory_stride(
        core.stages[boundary].input.as_ref(),
        core.stages[boundary].axis,
    )?;
    if stride > 1 {
        if array.real_storage_len > complex_storage.len() {
            return Err(fail_forward_complex(
                array,
                source_pencils,
                destination_pencils,
                extra_shape,
                complex_storage,
                0,
                FftError::StorageLayoutMismatch.into(),
            ));
        }
        // Promote the original packed real prefix before projecting any
        // result into the compressed complex layout. Read through a checked
        // scalar view before each write; reading `Complex::re` here would
        // observe a slot already overwritten by the reverse promotion order.
        for i in (0..array.real_storage_len).rev() {
            let value = match try_cast_slice::<Complex<R>, R>(&complex_storage) {
                Ok(scalar_view) => match scalar_view.get(i).copied() {
                    Some(value) => value,
                    None => {
                        return Err(fail_forward_complex(
                            array,
                            source_pencils,
                            destination_pencils,
                            extra_shape,
                            complex_storage,
                            0,
                            FftError::StorageLayoutMismatch.into(),
                        ));
                    }
                },
                Err(_) => {
                    return Err(fail_forward_complex(
                        array,
                        source_pencils,
                        destination_pencils,
                        extra_shape,
                        complex_storage,
                        0,
                        FftError::StorageLayoutMismatch.into(),
                    ));
                }
            };
            complex_storage[i] = Complex::new(value, R::zero());
        }
        if let Err(error) = convert_strided_real_storage(
            &mut complex_storage,
            plan,
            rows,
            real_len,
            complex_len,
            stride,
            workspace,
        ) {
            return Err(fail_forward_complex(
                array,
                source_pencils,
                destination_pencils,
                extra_shape,
                complex_storage,
                0,
                error,
            ));
        }
        let result =
            install_complex_storage(array, destination_pencils, extra_shape, complex_storage, 0);
        array.real_pencils = Some(source_pencils);
        return result;
    }
    let real_source_line =
        match try_cast_slice_mut::<Complex<R>, R>(&mut workspace.complex_source_line) {
            Ok(line) if line.len() >= real_len => line,
            _ => {
                return Err(fail_forward_complex(
                    array,
                    source_pencils,
                    destination_pencils,
                    extra_shape,
                    complex_storage,
                    0,
                    FftError::StorageLayoutMismatch.into(),
                ));
            }
        };
    for row in (0..rows).rev() {
        let source_start = match row.checked_mul(real_len) {
            Some(value) => value,
            None => {
                return Err(fail_forward_complex(
                    array,
                    source_pencils,
                    destination_pencils,
                    extra_shape,
                    complex_storage,
                    0,
                    FftError::PreparationFailed.into(),
                ));
            }
        };
        let source_end = match source_start.checked_add(real_len) {
            Some(value) => value,
            None => {
                return Err(fail_forward_complex(
                    array,
                    source_pencils,
                    destination_pencils,
                    extra_shape,
                    complex_storage,
                    0,
                    FftError::PreparationFailed.into(),
                ));
            }
        };
        let source = match try_cast_slice::<Complex<R>, R>(&complex_storage) {
            Ok(source) => source,
            Err(_) => {
                return Err(fail_forward_complex(
                    array,
                    source_pencils,
                    destination_pencils,
                    extra_shape,
                    complex_storage,
                    0,
                    FftError::StorageLayoutMismatch.into(),
                ));
            }
        };
        if source_end > source.len() {
            return Err(fail_forward_complex(
                array,
                source_pencils,
                destination_pencils,
                extra_shape,
                complex_storage,
                0,
                FftError::PreparationFailed.into(),
            ));
        }
        real_source_line[..real_len].copy_from_slice(&source[source_start..source_end]);
        let destination_start = match row.checked_mul(complex_len) {
            Some(value) => value,
            None => {
                return Err(fail_forward_complex(
                    array,
                    source_pencils,
                    destination_pencils,
                    extra_shape,
                    complex_storage,
                    0,
                    FftError::PreparationFailed.into(),
                ));
            }
        };
        let destination_end = match destination_start.checked_add(complex_len) {
            Some(value) => value,
            None => {
                return Err(fail_forward_complex(
                    array,
                    source_pencils,
                    destination_pencils,
                    extra_shape,
                    complex_storage,
                    0,
                    FftError::PreparationFailed.into(),
                ));
            }
        };
        if destination_end > complex_storage.len() {
            return Err(fail_forward_complex(
                array,
                source_pencils,
                destination_pencils,
                extra_shape,
                complex_storage,
                0,
                FftError::PreparationFailed.into(),
            ));
        }
        if let Err(error) = plan.forward(
            &real_source_line[..real_len],
            &mut complex_storage[destination_start..destination_end],
            &mut workspace.real_line,
            &mut workspace.fft_scratch,
        ) {
            return Err(fail_forward_complex(
                array,
                source_pencils,
                destination_pencils,
                extra_shape,
                complex_storage,
                0,
                error.into(),
            ));
        }
    }
    let result =
        install_complex_storage(array, destination_pencils, extra_shape, complex_storage, 0);
    array.real_pencils = Some(source_pencils);
    result
}

#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
fn convert_strided_complex_storage<R: FftReal, const N: usize, const M: usize>(
    storage: &mut [Complex<R>],
    plan: &LocalR2cPlan<R>,
    rows: usize,
    real_len: usize,
    complex_len: usize,
    stride: usize,
    workspace: &mut R2cInPlaceWorkspace<R, N, M>,
    normalize_inverse: bool,
) -> Result<(), R2cError>
where
    Complex<R>: Equivalence,
{
    if stride <= 1 {
        return Err(FftError::PreparationFailed.into());
    }
    if rows == 0 {
        return Ok(());
    }
    if rows % stride != 0 {
        return Err(FftError::PreparationFailed.into());
    }
    let groups = rows / stride;
    let source_span = groups
        .checked_mul(complex_len)
        .and_then(|value| value.checked_mul(stride))
        .ok_or(FftError::PreparationFailed)?;
    let destination_span = groups
        .checked_mul(real_len)
        .and_then(|value| value.checked_mul(stride))
        .ok_or(FftError::PreparationFailed)?;
    if source_span > storage.len() || destination_span > storage.len() {
        return Err(FftError::StorageLayoutMismatch.into());
    }

    // Expand compressed outer blocks from the back.  Each move is overlap
    // safe, and a later (smaller) q never overwrites a source block still to
    // be moved.
    for outer in (0..groups).rev() {
        let source_start = outer
            .checked_mul(complex_len)
            .and_then(|value| value.checked_mul(stride))
            .ok_or(FftError::PreparationFailed)?;
        let source_end = source_start
            .checked_add(
                complex_len
                    .checked_mul(stride)
                    .ok_or(FftError::PreparationFailed)?,
            )
            .ok_or(FftError::PreparationFailed)?;
        let destination_start = outer
            .checked_mul(real_len)
            .and_then(|value| value.checked_mul(stride))
            .ok_or(FftError::PreparationFailed)?;
        storage.copy_within(source_start..source_end, destination_start);
    }

    let complex_source_line = &mut workspace.complex_source_line;
    let complex_line = &mut workspace.complex_line;
    let real_line = &mut workspace.real_line;
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
    for outer in 0..groups {
        let base = outer
            .checked_mul(real_len)
            .and_then(|value| value.checked_mul(stride))
            .ok_or(FftError::PreparationFailed)?;
        for inner in 0..stride {
            for k in 0..complex_len {
                let index = base
                    .checked_add(k.checked_mul(stride).ok_or(FftError::PreparationFailed)?)
                    .and_then(|value| value.checked_add(inner))
                    .ok_or(FftError::PreparationFailed)?;
                complex_source_line[k] = storage[index];
            }
            let result = if normalize_inverse {
                plan.inverse(
                    &complex_source_line[..complex_len],
                    &mut real_line[..real_len],
                    complex_line,
                    &mut workspace.fft_scratch,
                )
            } else {
                plan.backward(
                    &complex_source_line[..complex_len],
                    &mut real_line[..real_len],
                    complex_line,
                    &mut workspace.fft_scratch,
                )
            };
            result.map_err(R2cError::LocalR2c)?;
            // Store the C2R result in Complex::re slots.  Residues modulo S
            // are disjoint, so every other inner line remains intact until it
            // has been gathered.  The final scalar projection below is done
            // only after all source lines have been consumed.
            for j in 0..real_len {
                let index = base
                    .checked_add(j.checked_mul(stride).ok_or(FftError::PreparationFailed)?)
                    .and_then(|value| value.checked_add(inner))
                    .ok_or(FftError::PreparationFailed)?;
                storage[index].re = real_line[j];
            }
        }
    }
    Ok(())
}

fn convert_complex_to_real<R: FftReal, const N: usize, const M: usize>(
    core: &Arc<TransformPlanCore<R, N, M>>,
    array: &mut R2cInPlaceArray<R, N, M>,
    workspace: &mut R2cInPlaceWorkspace<R, N, M>,
    normalize_inverse: bool,
) -> Result<(), R2cError>
where
    Complex<R>: Equivalence,
{
    let boundary = core.real_stage_index.ok_or(FftError::PreparationFailed)?;
    let (real_len, complex_len) = r2c_lengths(core);
    let rows = boundary_row_count(core)?;
    let full_complex_len =
        promoted_complex_len(core, array.real_storage_len, array.complex_storage_len)?;
    let stride = super::memory_stride(
        core.stages[boundary].output.as_ref(),
        core.stages[boundary].axis,
    )?;
    let destination_pencils = array
        .real_pencils
        .take()
        .ok_or(FftError::StorageLayoutMismatch)?;
    let destination_active = match destination_pencils
        .iter()
        .position(|pencil| pencil.same_layout(core.stages[boundary].input.as_ref()))
    {
        Some(active) => active,
        None => {
            array.real_pencils = Some(destination_pencils);
            return Err(FftError::StorageLayoutMismatch.into());
        }
    };
    let complex = match array.storage.take() {
        Some(R2cInPlaceStorage::Complex(complex)) => complex,
        _ => {
            array.real_pencils = Some(destination_pencils);
            return Err(FftError::StorageLayoutMismatch.into());
        }
    };
    let (source_pencils, source_active, extra_shape, mut storage) =
        match complex.into_parts_preserving() {
            Ok(parts) => parts,
            Err((error, complex)) => {
                array.storage = Some(R2cInPlaceStorage::PoisonedComplex(complex.into_storage()));
                array.real_pencils = Some(destination_pencils);
                return Err(map_array_allocation(error).into());
            }
        };
    #[cfg(test)]
    if array.test_hook == Some(InPlaceTestHook::PanicAfterReverseDetach) {
        panic!("injected R2C reverse panic after owner detachment");
    }
    if full_complex_len > storage.capacity() {
        return Err(fail_complex_to_real(
            array,
            source_pencils,
            destination_pencils,
            extra_shape,
            storage,
            source_active,
            FftError::StorageLayoutMismatch.into(),
        ));
    }
    storage.resize(full_complex_len, zero_complex::<R>());
    let plan = core.stages[boundary].local.real_complex();
    if stride > 1 {
        if let Err(error) = convert_strided_complex_storage(
            &mut storage,
            plan,
            rows,
            real_len,
            complex_len,
            stride,
            workspace,
            normalize_inverse,
        ) {
            return Err(fail_complex_to_real(
                array,
                source_pencils,
                destination_pencils,
                extra_shape,
                storage,
                source_active,
                error,
            ));
        }
        let mut real_storage = match try_cast_vec(storage) {
            Ok(storage) => storage,
            Err((_, storage)) => {
                array.storage = Some(R2cInPlaceStorage::PoisonedComplex(storage));
                array.complex_pencils = Some(source_pencils);
                array.real_pencils = Some(destination_pencils);
                return Err(FftError::StorageLayoutMismatch.into());
            }
        };
        if array.real_storage_len > real_storage.len() {
            return Err(fail_reverse_real(
                array,
                source_pencils,
                destination_pencils,
                extra_shape,
                real_storage,
                destination_active,
                FftError::StorageLayoutMismatch.into(),
            ));
        }
        for i in 0..array.real_storage_len {
            let value = match try_cast_slice::<R, Complex<R>>(&real_storage) {
                Ok(complex_view) => match complex_view.get(i).copied() {
                    Some(value) => value.re,
                    None => {
                        return Err(fail_reverse_real(
                            array,
                            source_pencils,
                            destination_pencils,
                            extra_shape,
                            real_storage,
                            boundary,
                            FftError::StorageLayoutMismatch.into(),
                        ));
                    }
                },
                Err(_) => {
                    return Err(fail_reverse_real(
                        array,
                        source_pencils,
                        destination_pencils,
                        extra_shape,
                        real_storage,
                        boundary,
                        FftError::StorageLayoutMismatch.into(),
                    ));
                }
            };
            real_storage[i] = value;
        }
        #[cfg(test)]
        let mut destination_pencils = destination_pencils;
        #[cfg(test)]
        if array.test_hook == Some(InPlaceTestHook::ReverseRestoreRejection) {
            let mut pencils = destination_pencils.into_vec();
            pencils.pop();
            destination_pencils = pencils.into_boxed_slice();
        }
        let result = install_real_storage(
            array,
            destination_pencils,
            extra_shape,
            real_storage,
            destination_active,
        );
        array.complex_pencils = Some(source_pencils);
        return result;
    }
    let mut real_storage = match try_cast_vec(storage) {
        Ok(storage) => storage,
        Err((_, storage)) => {
            array.storage = Some(R2cInPlaceStorage::PoisonedComplex(storage));
            array.complex_pencils = Some(source_pencils);
            array.real_pencils = Some(destination_pencils);
            return Err(FftError::StorageLayoutMismatch.into());
        }
    };
    if array.real_storage_len > real_storage.len() {
        return Err(fail_reverse_real(
            array,
            source_pencils,
            destination_pencils,
            extra_shape,
            real_storage,
            destination_active,
            FftError::StorageLayoutMismatch.into(),
        ));
    }
    for row in 0..rows {
        let source_start = match row.checked_mul(complex_len) {
            Some(value) => value,
            None => {
                return Err(fail_reverse_real(
                    array,
                    source_pencils,
                    destination_pencils,
                    extra_shape,
                    real_storage,
                    boundary,
                    FftError::PreparationFailed.into(),
                ));
            }
        };
        let source_end = match source_start.checked_add(complex_len) {
            Some(value) => value,
            None => {
                return Err(fail_reverse_real(
                    array,
                    source_pencils,
                    destination_pencils,
                    extra_shape,
                    real_storage,
                    boundary,
                    FftError::PreparationFailed.into(),
                ));
            }
        };
        let source = match try_cast_slice::<R, Complex<R>>(&real_storage) {
            Ok(source) => source,
            Err(_) => {
                return Err(fail_reverse_real(
                    array,
                    source_pencils,
                    destination_pencils,
                    extra_shape,
                    real_storage,
                    boundary,
                    FftError::StorageLayoutMismatch.into(),
                ));
            }
        };
        if source_end > source.len() {
            return Err(fail_reverse_real(
                array,
                source_pencils,
                destination_pencils,
                extra_shape,
                real_storage,
                destination_active,
                FftError::PreparationFailed.into(),
            ));
        }
        workspace.complex_source_line[..complex_len]
            .copy_from_slice(&source[source_start..source_end]);
        let destination_start = match row.checked_mul(real_len) {
            Some(value) => value,
            None => {
                return Err(fail_reverse_real(
                    array,
                    source_pencils,
                    destination_pencils,
                    extra_shape,
                    real_storage,
                    boundary,
                    FftError::PreparationFailed.into(),
                ));
            }
        };
        let destination_end = match destination_start.checked_add(real_len) {
            Some(value) => value,
            None => {
                return Err(fail_reverse_real(
                    array,
                    source_pencils,
                    destination_pencils,
                    extra_shape,
                    real_storage,
                    boundary,
                    FftError::PreparationFailed.into(),
                ));
            }
        };
        if destination_end > real_storage.len() {
            return Err(fail_reverse_real(
                array,
                source_pencils,
                destination_pencils,
                extra_shape,
                real_storage,
                destination_active,
                FftError::PreparationFailed.into(),
            ));
        }
        let result = if normalize_inverse {
            plan.inverse(
                &workspace.complex_source_line[..complex_len],
                &mut real_storage[destination_start..destination_end],
                &mut workspace.complex_line,
                &mut workspace.fft_scratch,
            )
        } else {
            plan.backward(
                &workspace.complex_source_line[..complex_len],
                &mut real_storage[destination_start..destination_end],
                &mut workspace.complex_line,
                &mut workspace.fft_scratch,
            )
        };
        if let Err(error) = result {
            return Err(fail_reverse_real(
                array,
                source_pencils,
                destination_pencils,
                extra_shape,
                real_storage,
                destination_active,
                error.into(),
            ));
        }
    }
    #[cfg(test)]
    let mut destination_pencils = destination_pencils;
    #[cfg(test)]
    if array.test_hook == Some(InPlaceTestHook::ReverseRestoreRejection) {
        let mut pencils = destination_pencils.into_vec();
        pencils.pop();
        destination_pencils = pencils.into_boxed_slice();
    }
    let result = install_real_storage(
        array,
        destination_pencils,
        extra_shape,
        real_storage,
        destination_active,
    );
    array.complex_pencils = Some(source_pencils);
    result
}

#[cfg(all(test, feature = "distributed"))]
pub(super) fn run_rank_specific_conversion_failure(world: &mpi::topology::SimpleCommunicator) {
    let size = usize::try_from(world.size()).unwrap();
    assert!(matches!(size, 4 | 6));

    // Keep the existing full-axis forward case: the conversion is followed by
    // a complex distributed payload, so peers must not enter it after rank 1
    // rejects the local conversion.
    let topology = MpiTopology::<1>::new(world, [size]).unwrap();
    let mut plan = R2cPlan::<f64, 2, 1>::from_shape_with_method(
        Arc::clone(&topology),
        [3, 4],
        ExtraShape::scalar(),
        TransposeMethod::AllToAllv,
    )
    .unwrap();
    inject_conversion_scratch_failure(&mut plan, topology.rank() == 1);
    let mut array = plan.allocate_in_place().unwrap();
    array.real_view_mut().unwrap().as_mut_slice().fill(1.0);
    let mut workspace = plan.allocate_in_place_workspace().unwrap();
    let result = plan.forward_in_place(&mut array, &mut workspace);
    assert_conversion_failure_result(result, topology.rank() == 1);
    assert_poisoned_views_and_retries(&plan, &mut array, &mut workspace);

    // A non-last reduction leaves real-prefix transposes after the
    // representation handoff on reverse execution. Run both reverse APIs;
    // the collective conversion agreement must finish before either peer can
    // enter those payload collectives.
    let grid = match size {
        4 => [2, 2],
        6 => [2, 3],
        _ => unreachable!(),
    };
    let topology_2d = MpiTopology::<2>::new(world, grid).unwrap();
    let selection = AxisSelection::<3>::from_indices([0]).unwrap();
    for normalize in [true, false] {
        let mut reverse_plan = R2cPlan::<f64, 3, 2>::from_shape_with_selection_and_method(
            Arc::clone(&topology_2d),
            [size, size / 2, 3],
            ExtraShape::scalar(),
            selection,
            TransposeMethod::AllToAllv,
        )
        .unwrap();
        inject_conversion_scratch_failure(&mut reverse_plan, topology_2d.rank() == 1);
        let mut reverse_array = reverse_plan.allocate_in_place().unwrap();
        force_complex_input(&mut reverse_array);
        let mut reverse_workspace = reverse_plan.allocate_in_place_workspace().unwrap();
        let result = if normalize {
            reverse_plan.inverse_in_place(&mut reverse_array, &mut reverse_workspace)
        } else {
            reverse_plan.backward_in_place(&mut reverse_array, &mut reverse_workspace)
        };
        assert_conversion_failure_result(result, topology_2d.rank() == 1);
        assert_poisoned_views_and_retries(
            &reverse_plan,
            &mut reverse_array,
            &mut reverse_workspace,
        );
        topology_2d.communicator().barrier();
    }

    world.barrier();
}

#[cfg(all(test, feature = "distributed"))]
fn inject_conversion_scratch_failure<R: FftReal, const N: usize, const M: usize>(
    plan: &mut R2cPlan<R, N, M>,
    enabled: bool,
) {
    if enabled {
        let core = Arc::get_mut(&mut plan.core).unwrap();
        let boundary = core.real_stage_index.unwrap();
        match &mut core.stages[boundary].local {
            LocalTransform::RealComplex(local) => local.inject_scratch_shortage_for_test(),
            _ => unreachable!("test plan has a real boundary"),
        }
    }
}

#[cfg(all(test, feature = "distributed"))]
fn assert_conversion_failure_result(result: Result<(), R2cError>, local: bool) {
    if local {
        assert!(matches!(
            result,
            Err(R2cError::LocalR2c(LocalR2cError::ScratchTooSmall { .. }))
        ));
    } else {
        assert!(matches!(
            result,
            Err(R2cError::Fft(FftError::CollectivePreconditionFailed))
        ));
    }
}

#[cfg(all(test, feature = "distributed"))]
pub(super) fn assert_poisoned_views_and_retries<const N: usize, const M: usize>(
    plan: &R2cPlan<f64, N, M>,
    array: &mut R2cInPlaceArray<f64, N, M>,
    workspace: &mut R2cInPlaceWorkspace<f64, N, M>,
) where
    Complex<f64>: Equivalence,
{
    assert_eq!(array.state(), R2cState::Poisoned);
    assert!(matches!(
        array.real_view(),
        Err(R2cError::Fft(FftError::Array(ArrayError::Poisoned)))
    ));
    assert!(matches!(
        array.real_view_mut(),
        Err(R2cError::Fft(FftError::Array(ArrayError::Poisoned)))
    ));
    assert!(matches!(
        array.complex_view(),
        Err(R2cError::Fft(FftError::Array(ArrayError::Poisoned)))
    ));
    assert!(matches!(
        array.complex_view_mut(),
        Err(R2cError::Fft(FftError::Array(ArrayError::Poisoned)))
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
            Err(R2cError::Fft(FftError::Array(ArrayError::Poisoned)))
        ));
        assert_eq!(array.state(), R2cState::Poisoned);
        assert_eq!(format!("{array:?}"), array_before);
        assert_eq!(format!("{workspace:?}"), workspace_before);
    }
}

#[cfg(all(test, feature = "distributed"))]
pub(super) fn force_odd_real_capacity<R: FftReal, const N: usize, const M: usize>(
    array: &mut R2cInPlaceArray<R, N, M>,
) {
    let storage = array.storage.take().expect("test array has storage");
    let R2cInPlaceStorage::Real(real) = storage else {
        panic!("test array starts in the real representation");
    };
    let (pencils, active, extra_shape, storage) = real.into_parts().unwrap();
    let requested = if storage.capacity() % 2 == 0 {
        storage.capacity().checked_add(1).unwrap()
    } else {
        storage.capacity()
    };
    let mut replacement = Vec::with_capacity(requested);
    replacement.extend(storage);
    assert_eq!(replacement.capacity() % 2, 1);
    array.storage = Some(R2cInPlaceStorage::Real(
        ManyPencilArray::from_vec(pencils, active, extra_shape, replacement).unwrap(),
    ));
}

#[cfg(all(test, feature = "distributed"))]
pub(super) fn empty_in_place_workspace_for_test<R: FftReal, const N: usize, const M: usize>(
    workspace: &mut R2cInPlaceWorkspace<R, N, M>,
) {
    workspace.fft_scratch.clear();
    if let Some(real_source_line) = workspace.real_source_line.as_mut() {
        real_source_line.clear();
    }
    workspace.real_line.clear();
    workspace.complex_source_line.clear();
    workspace.complex_line.clear();
}

#[cfg(all(test, feature = "distributed"))]
pub(super) fn force_complex_input<R: FftReal, const N: usize, const M: usize>(
    array: &mut R2cInPlaceArray<R, N, M>,
) where
    Complex<R>: Equivalence,
{
    let storage = array.storage.take().expect("test array has storage");
    let R2cInPlaceStorage::Real(real) = storage else {
        panic!("test array starts in the real representation");
    };
    let (real_pencils, _active, extra_shape, mut storage) = real.into_parts().unwrap();
    let complex_pencils = array
        .complex_pencils
        .take()
        .expect("test array has complex layouts");
    let complex_size = size_of::<Complex<R>>();
    let scalar_size = size_of::<R>();
    let target_bytes = array.complex_storage_len.checked_mul(complex_size).unwrap();
    let source_bytes = storage.len().checked_mul(scalar_size).unwrap();
    let bytes = source_bytes.max(target_bytes);
    let full_bytes = bytes.checked_add(complex_size - 1).unwrap() / complex_size * complex_size;
    storage.resize(full_bytes / scalar_size, R::zero());
    let mut storage = try_cast_vec(storage).unwrap();
    storage.truncate(array.complex_storage_len);
    let active = complex_pencils.len().checked_sub(1).unwrap();
    array.real_pencils = Some(real_pencils);
    array.storage = Some(R2cInPlaceStorage::Complex(
        ManyPencilArray::from_vec(complex_pencils, active, extra_shape, storage).unwrap(),
    ));
    array.state = R2cState::ComplexOutput;
}

#[cfg(all(test, feature = "distributed"))]
pub(super) fn panic_after_forward_detach_for_test<R: FftReal, const N: usize, const M: usize>(
    array: &mut R2cInPlaceArray<R, N, M>,
) {
    array.test_hook = Some(InPlaceTestHook::PanicAfterForwardDetach);
}

#[cfg(all(test, feature = "distributed"))]
pub(super) fn panic_after_reverse_detach_for_test<R: FftReal, const N: usize, const M: usize>(
    array: &mut R2cInPlaceArray<R, N, M>,
) {
    array.test_hook = Some(InPlaceTestHook::PanicAfterReverseDetach);
}

#[cfg(all(test, feature = "distributed"))]
pub(super) fn force_odd_forward_cast_for_test<R: FftReal, const N: usize, const M: usize>(
    array: &mut R2cInPlaceArray<R, N, M>,
) {
    array.test_hook = Some(InPlaceTestHook::OddForwardCast);
}

#[cfg(all(test, feature = "distributed"))]
pub(super) fn force_reverse_restore_rejection_for_test<
    R: FftReal,
    const N: usize,
    const M: usize,
>(
    array: &mut R2cInPlaceArray<R, N, M>,
) {
    array.test_hook = Some(InPlaceTestHook::ReverseRestoreRejection);
}

#[cfg(all(test, feature = "distributed"))]
pub(super) fn corrupt_real_extra_shape_for_test<R: FftReal, const N: usize, const M: usize>(
    array: &mut R2cInPlaceArray<R, N, M>,
    extra_shape: ExtraShape,
) {
    let storage = array.storage.take().expect("test array has storage");
    let R2cInPlaceStorage::Real(real) = storage else {
        panic!("test array is not in the real representation");
    };
    let (pencils, active, _old_extra_shape, storage) = real.into_parts().unwrap();
    array.storage = Some(R2cInPlaceStorage::Real(
        ManyPencilArray::from_vec(pencils, active, extra_shape, storage).unwrap(),
    ));
}

#[cfg(all(test, feature = "distributed"))]
pub(super) fn corrupt_complex_extra_shape_for_test<R: FftReal, const N: usize, const M: usize>(
    array: &mut R2cInPlaceArray<R, N, M>,
    extra_shape: ExtraShape,
) {
    let storage = array.storage.take().expect("test array has storage");
    let R2cInPlaceStorage::Complex(complex) = storage else {
        panic!("test array is not in the complex representation");
    };
    let (pencils, active, _old_extra_shape, storage) = complex.into_parts().unwrap();
    array.storage = Some(R2cInPlaceStorage::Complex(
        ManyPencilArray::from_vec(pencils, active, extra_shape, storage).unwrap(),
    ));
}

#[cfg(all(test, feature = "distributed"))]
pub(super) fn corrupt_forward_registry_for_test<R: FftReal, const N: usize, const M: usize>(
    array: &mut R2cInPlaceArray<R, N, M>,
) {
    let boundary = array.core.real_stage_index.unwrap();
    let registry = array
        .complex_pencils
        .as_mut()
        .expect("test array has a complex registry");
    assert!(!registry.is_empty());
    registry[0] = Arc::clone(&array.core.stages[boundary].input);
}

#[cfg(all(test, feature = "distributed"))]
pub(super) fn corrupt_reverse_registry_for_test<R: FftReal, const N: usize, const M: usize>(
    array: &mut R2cInPlaceArray<R, N, M>,
) {
    let boundary = array.core.real_stage_index.unwrap();
    let registry = array
        .real_pencils
        .as_mut()
        .expect("test array has a real registry");
    assert!(!registry.is_empty());
    registry[0] = Arc::clone(&array.core.stages[boundary].output);
}

#[cfg(all(test, feature = "distributed"))]
pub(super) fn storage_is_poisoned_for_test<R: FftReal, const N: usize, const M: usize>(
    array: &R2cInPlaceArray<R, N, M>,
) -> bool {
    matches!(
        array.storage.as_ref(),
        Some(R2cInPlaceStorage::PoisonedReal(_)) | Some(R2cInPlaceStorage::PoisonedComplex(_))
    )
}

#[cfg(all(test, feature = "distributed"))]
pub(super) fn storage_capacity_bytes_for_test<R: FftReal, const N: usize, const M: usize>(
    array: &R2cInPlaceArray<R, N, M>,
) -> Option<usize> {
    array.storage_capacity_bytes()
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
    let stride = super::memory_stride(view.pencil(), reduction_axis)?;
    let local_axis_len = view.pencil().local_shape_logical()[reduction_axis];
    let global_axis_start = view.pencil().local_ranges()[reduction_axis].start;
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
        if local_len != 0 {
            for (offset, value) in values.iter().enumerate() {
                let coord = (offset / stride) % local_axis_len;
                let k = global_axis_start + coord;
                let plane = if k == 0 {
                    Some(0)
                } else if plane_count == 2 && k == complex_len - 1 {
                    Some(1)
                } else {
                    None
                };
                let Some(plane) = plane else { continue };
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
        if local_len != 0 {
            for (offset, value) in values.iter().enumerate() {
                let coord = (offset / stride) % local_axis_len;
                let k = global_axis_start + coord;
                let plane = if k == 0 {
                    Some(0)
                } else if plane_count == 2 && k == complex_len - 1 {
                    Some(1)
                } else {
                    None
                };
                let Some(plane) = plane else { continue };
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
    let reduction_axis = core
        .selection
        .mask()
        .iter()
        .rposition(|&selected| selected)
        .expect("R2C core has a selected reduction axis");
    let stride = super::memory_stride(view.pencil(), reduction_axis)?;
    let local_axis_len = view.pencil().local_shape_logical()[reduction_axis];
    let global_axis_start = view.pencil().local_ranges()[reduction_axis].start;
    let plane_count = if real_len % 2 == 0 { 2 } else { 1 };
    for batch in 0..core.extra_shape.element_count() {
        let start = batch
            .checked_mul(local_len)
            .expect("validated extra and local lengths fit the intermediate");
        let end = start
            .checked_add(local_len)
            .expect("validated extra and local lengths fit the intermediate");
        let values = &mut view.as_mut_slice()[start..end];
        if local_len != 0 {
            for (offset, value) in values.iter_mut().enumerate() {
                let coord = (offset / stride) % local_axis_len;
                let k = global_axis_start + coord;
                if k == 0 || (plane_count == 2 && k == complex_len - 1) {
                    value.im = R::zero();
                }
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
