//! Distributed FFTW-style DCT/DST transforms.

#![allow(clippy::too_many_arguments)]

use std::{mem::size_of, sync::Arc};

use super::{
    AxisSelection, C2cStageTransition, Direction, DistributedLayout, FftError, INVALID_WORD,
    OPERATION_R2R_BACKWARD, OPERATION_R2R_BACKWARD_IN_PLACE, OPERATION_R2R_FORWARD,
    OPERATION_R2R_FORWARD_IN_PLACE, OPERATION_R2R_INVERSE, OPERATION_R2R_INVERSE_IN_PLACE,
    R2rError, RouteCandidate, StagePreparation, TransformStage, TransposeMethod,
    agree_execution_descriptor_ref, agree_header, agree_result, build_route, build_transitions,
    collective_descriptor, collective_valid, descriptor_len, execute_transition, initialized_vec,
    map_array_allocation, validate_input, validate_workspace_lengths_values,
};
use crate::{
    Complex, LocalDhtPlan, LocalR2rError, LocalR2rPlan, R2rKind, R2rScalar, r2r::AxisR2rKind,
};
use mpi::datatype::Equivalence;
use pencil_array::{
    ExtraShape, ManyPencilArray, MpiTopology, OverwriteError, Pencil, PencilArray, PencilArrayView,
    PencilArrayViewMut, TransposeWorkspace,
};

/// An immutable, checked distributed FFTW-compatible DCT/DST plan.
///
/// `None` entries are identity stages. Discrete Hartley and mixed-axis
/// construction is separate from this legacy API.
#[derive(Debug)]
pub struct R2rPlan<T: R2rScalar, const N: usize, const M: usize> {
    core: Arc<R2rCore<T, N, M>>,
}

/// A distributed separable discrete Hartley transform plan.
#[derive(Debug)]
pub struct DhtPlan<T: R2rScalar, const N: usize, const M: usize> {
    core: Arc<R2rCore<T, N, M>>,
    selection: AxisSelection<N>,
}

impl<T: R2rScalar, const N: usize, const M: usize> DhtPlan<T, N, M>
where
    T: Equivalence,
{
    /// Collectively builds an Alltoallv plan for every spatial axis.
    pub fn from_pencil(
        input: Arc<Pencil<N, M>>,
        extra_shape: ExtraShape,
    ) -> Result<Self, R2rError> {
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
    ) -> Result<Self, R2rError> {
        Self::from_pencil_with_selection_and_layout(
            input,
            extra_shape,
            selection,
            DistributedLayout::default(),
        )
    }

    /// Collectively builds a plan from a canonical pencil and transport.
    pub fn from_pencil_with_method(
        input: Arc<Pencil<N, M>>,
        extra_shape: ExtraShape,
        method: TransposeMethod,
    ) -> Result<Self, R2rError> {
        Self::from_pencil_with_selection_and_method(
            input,
            extra_shape,
            AxisSelection::all(),
            method,
        )
    }

    /// Collectively builds a plan from a validated selection and transport.
    pub fn from_pencil_with_selection_and_method(
        input: Arc<Pencil<N, M>>,
        extra_shape: ExtraShape,
        selection: AxisSelection<N>,
        method: TransposeMethod,
    ) -> Result<Self, R2rError> {
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

    /// Collectively builds a plan with an explicit transport and layout.
    pub fn from_pencil_with_layout(
        input: Arc<Pencil<N, M>>,
        extra_shape: ExtraShape,
        layout: DistributedLayout,
    ) -> Result<Self, R2rError> {
        Self::from_pencil_with_selection_and_layout(
            input,
            extra_shape,
            AxisSelection::all(),
            layout,
        )
    }

    /// Collectively builds a plan for a validated selection and layout.
    pub fn from_pencil_with_selection_and_layout(
        input: Arc<Pencil<N, M>>,
        extra_shape: ExtraShape,
        selection: AxisSelection<N>,
        layout: DistributedLayout,
    ) -> Result<Self, R2rError> {
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

    /// Collectively builds an Alltoallv plan from a real/complex array layout.
    pub fn from_array(input: &PencilArray<T, N, M>) -> Result<Self, R2rError> {
        Self::from_array_with_selection_and_layout(
            input,
            AxisSelection::all(),
            DistributedLayout::default(),
        )
    }

    /// Collectively builds a plan from an array and validated selection.
    pub fn from_array_with_selection(
        input: &PencilArray<T, N, M>,
        selection: AxisSelection<N>,
    ) -> Result<Self, R2rError> {
        Self::from_array_with_selection_and_layout(input, selection, DistributedLayout::default())
    }

    /// Collectively builds a plan from an array and transport.
    pub fn from_array_with_method(
        input: &PencilArray<T, N, M>,
        method: TransposeMethod,
    ) -> Result<Self, R2rError> {
        Self::from_array_with_selection_and_method(input, AxisSelection::all(), method)
    }

    /// Collectively builds a plan from an array, selection, and transport.
    pub fn from_array_with_selection_and_method(
        input: &PencilArray<T, N, M>,
        selection: AxisSelection<N>,
        method: TransposeMethod,
    ) -> Result<Self, R2rError> {
        Self::from_array_with_selection_and_layout(
            input,
            selection,
            DistributedLayout {
                transpose_method: method,
                ..DistributedLayout::default()
            },
        )
    }

    /// Collectively builds a plan from an array and explicit layout.
    pub fn from_array_with_layout(
        input: &PencilArray<T, N, M>,
        layout: DistributedLayout,
    ) -> Result<Self, R2rError> {
        Self::from_array_with_selection_and_layout(input, AxisSelection::all(), layout)
    }

    /// Collectively builds a plan from an array, selection, and explicit layout.
    pub fn from_array_with_selection_and_layout(
        input: &PencilArray<T, N, M>,
        selection: AxisSelection<N>,
        layout: DistributedLayout,
    ) -> Result<Self, R2rError> {
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

    /// Collectively builds an Alltoallv plan from topology and shape.
    pub fn from_shape(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
    ) -> Result<Self, R2rError> {
        Self::from_shape_with_selection_and_layout(
            topology,
            global_shape,
            extra_shape,
            AxisSelection::all(),
            DistributedLayout::default(),
        )
    }

    /// Collectively builds a plan from topology, shape, and transport.
    pub fn from_shape_with_method(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
        method: TransposeMethod,
    ) -> Result<Self, R2rError> {
        Self::from_shape_with_selection_and_method(
            topology,
            global_shape,
            extra_shape,
            AxisSelection::all(),
            method,
        )
    }

    /// Collectively builds a plan from topology, shape, selection, and transport.
    pub fn from_shape_with_selection_and_method(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
        selection: AxisSelection<N>,
        method: TransposeMethod,
    ) -> Result<Self, R2rError> {
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

    /// Collectively builds an Alltoallv plan from topology, shape, and selection.
    pub fn from_shape_with_selection(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
        selection: AxisSelection<N>,
    ) -> Result<Self, R2rError> {
        Self::from_shape_with_selection_and_layout(
            topology,
            global_shape,
            extra_shape,
            selection,
            DistributedLayout::default(),
        )
    }

    /// Collectively builds a plan from topology, shape, and explicit layout.
    pub fn from_shape_with_layout(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
        layout: DistributedLayout,
    ) -> Result<Self, R2rError> {
        Self::from_shape_with_selection_and_layout(
            topology,
            global_shape,
            extra_shape,
            AxisSelection::all(),
            layout,
        )
    }

    /// Collectively builds a plan from topology, shape, selection, and layout.
    pub fn from_shape_with_selection_and_layout(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
        selection: AxisSelection<N>,
        layout: DistributedLayout,
    ) -> Result<Self, R2rError> {
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

    fn construct(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
        input: Result<Arc<Pencil<N, M>>, FftError>,
        selection: AxisSelection<N>,
        layout: DistributedLayout,
    ) -> Result<Self, R2rError> {
        let axis_kinds =
            std::array::from_fn(|axis| selection.contains(axis).then_some(AxisR2rKind::Dht));
        let core = R2rPlan::<T, N, M>::construct_core(
            topology,
            global_shape,
            extra_shape,
            input,
            [None; N],
            axis_kinds,
            layout,
            super::OPERATION_DHT_PLAN,
        )?;
        Ok(Self { core, selection })
    }

    /// Returns the canonical input pencil.
    pub fn input_pencil(&self) -> &Arc<Pencil<N, M>> {
        &self.core.stages[0].input
    }

    /// Returns the output pencil. Its permutation follows this plan's layout.
    pub fn output_pencil(&self) -> &Arc<Pencil<N, M>> {
        &self.core.stages[self.core.stages.len() - 1].output
    }

    /// Returns the exact extra shape required by this plan.
    pub fn extra_shape(&self) -> &ExtraShape {
        &self.core.extra_shape
    }

    /// Returns the selected axes.
    pub fn selection(&self) -> AxisSelection<N> {
        self.selection
    }

    /// Returns the transport and memory-layout policy used by this plan.
    pub fn layout(&self) -> DistributedLayout {
        self.core.layout
    }

    /// Allocates a zero-initialized local input array.
    pub fn allocate_input(&self) -> Result<PencilArray<T, N, M>, R2rError> {
        PencilArray::from_fn(
            Arc::clone(self.input_pencil()),
            self.core.extra_shape.clone(),
            crate::r2r::r2r_zero::<T>,
        )
        .map_err(FftError::Array)
        .map_err(Into::into)
    }

    /// Allocates a zero-initialized local output array.
    pub fn allocate_output(&self) -> Result<PencilArray<T, N, M>, R2rError> {
        PencilArray::from_fn(
            Arc::clone(self.output_pencil()),
            self.core.extra_shape.clone(),
            crate::r2r::r2r_zero::<T>,
        )
        .map_err(FftError::Array)
        .map_err(Into::into)
    }

    /// Allocates reusable out-of-place workspace.
    pub fn allocate_workspace(&self) -> Result<R2rWorkspace<T, N, M>, R2rError> {
        let intermediate = ManyPencilArray::from_elem(
            registered_stage_pencils(&self.core.stages)?,
            0,
            self.core.extra_shape.clone(),
            crate::r2r::r2r_zero::<T>(),
        )
        .map_err(map_array_allocation)
        .map_err(R2rError::Fft)?;
        Ok(R2rWorkspace {
            core: Arc::clone(&self.core),
            intermediate,
            transpose: TransposeWorkspace::from_vecs(
                initialized_vec(self.core.transpose_send_len, crate::r2r::r2r_zero::<T>())?,
                initialized_vec(self.core.transpose_receive_len, crate::r2r::r2r_zero::<T>())?,
            ),
            embedding_line: initialized_vec(self.core.embedding_len, zero_complex::<T::Real>())?,
            fft_scratch: initialized_vec(self.core.fft_scratch_len, zero_complex::<T::Real>())?,
            line_buffer: initialized_vec(self.core.strided_line_len, crate::r2r::r2r_zero::<T>())?,
        })
    }

    /// Allocates an opaque canonical input array for in-place execution.
    pub fn allocate_in_place(&self) -> Result<R2rInPlaceArray<T, N, M>, R2rError> {
        let array = ManyPencilArray::from_elem(
            registered_stage_pencils(&self.core.stages)?,
            0,
            self.core.extra_shape.clone(),
            crate::r2r::r2r_zero::<T>(),
        )
        .map_err(map_array_allocation)
        .map_err(R2rError::Fft)?;
        Ok(R2rInPlaceArray {
            core: Arc::clone(&self.core),
            array,
            state: super::R2rState::Input,
        })
    }

    /// Allocates reusable in-place workspace.
    pub fn allocate_in_place_workspace(&self) -> Result<R2rInPlaceWorkspace<T, N, M>, R2rError> {
        Ok(R2rInPlaceWorkspace {
            core: Arc::clone(&self.core),
            transpose: TransposeWorkspace::from_vecs(
                initialized_vec(self.core.transpose_send_len, crate::r2r::r2r_zero::<T>())?,
                initialized_vec(self.core.transpose_receive_len, crate::r2r::r2r_zero::<T>())?,
            ),
            embedding_line: initialized_vec(self.core.embedding_len, zero_complex::<T::Real>())?,
            fft_scratch: initialized_vec(self.core.fft_scratch_len, zero_complex::<T::Real>())?,
            line_buffer: initialized_vec(self.core.strided_line_len, crate::r2r::r2r_zero::<T>())?,
        })
    }

    /// Computes the selected unnormalized Hartley transforms.
    pub fn forward(
        &self,
        source: &PencilArray<T, N, M>,
        destination: &mut PencilArray<T, N, M>,
        workspace: &mut R2rWorkspace<T, N, M>,
    ) -> Result<(), R2rError> {
        self.execute(Direction::Forward, source, destination, workspace)
    }

    /// Computes the normalized self-paired inverse transforms.
    pub fn inverse(
        &self,
        source: &PencilArray<T, N, M>,
        destination: &mut PencilArray<T, N, M>,
        workspace: &mut R2rWorkspace<T, N, M>,
    ) -> Result<(), R2rError> {
        self.execute(Direction::Inverse, source, destination, workspace)
    }

    /// Computes the raw self-paired backward transforms.
    pub fn backward(
        &self,
        source: &PencilArray<T, N, M>,
        destination: &mut PencilArray<T, N, M>,
        workspace: &mut R2rWorkspace<T, N, M>,
    ) -> Result<(), R2rError> {
        self.execute(Direction::Backward, source, destination, workspace)
    }

    /// Computes the selected Hartley transforms in place.
    pub fn forward_in_place(
        &self,
        array: &mut R2rInPlaceArray<T, N, M>,
        workspace: &mut R2rInPlaceWorkspace<T, N, M>,
    ) -> Result<(), R2rError> {
        self.execute_in_place(Direction::Forward, array, workspace)
    }

    /// Computes the normalized inverse in place.
    pub fn inverse_in_place(
        &self,
        array: &mut R2rInPlaceArray<T, N, M>,
        workspace: &mut R2rInPlaceWorkspace<T, N, M>,
    ) -> Result<(), R2rError> {
        self.execute_in_place(Direction::Inverse, array, workspace)
    }

    /// Computes the raw backward transform in place.
    pub fn backward_in_place(
        &self,
        array: &mut R2rInPlaceArray<T, N, M>,
        workspace: &mut R2rInPlaceWorkspace<T, N, M>,
    ) -> Result<(), R2rError> {
        self.execute_in_place(Direction::Backward, array, workspace)
    }

    fn execute(
        &self,
        direction: Direction,
        source: &PencilArray<T, N, M>,
        destination: &mut PencilArray<T, N, M>,
        workspace: &mut R2rWorkspace<T, N, M>,
    ) -> Result<(), R2rError> {
        let communicator = self.input_pencil().topology().communicator();
        let operation = match direction {
            Direction::Forward => super::OPERATION_DHT_FORWARD,
            Direction::Inverse => super::OPERATION_DHT_INVERSE,
            Direction::Backward => super::OPERATION_DHT_BACKWARD,
        };
        agree_execution_descriptor_ref::<N, M>(communicator, operation, &self.core.descriptor)?;
        let preflight = validate_r2r_out_of_place(
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
            workspace.embedding_line.len(),
        );
        let preflight = preflight.and_then(|()| {
            if workspace.line_buffer.len() < self.core.strided_line_len {
                Err(FftError::WorkspaceTooSmall {
                    kind: "real line",
                    required: self.core.strided_line_len,
                    actual: workspace.line_buffer.len(),
                }
                .into())
            } else {
                Ok(())
            }
        });
        if !collective_valid(communicator, preflight.is_ok()) {
            return Err(preflight
                .err()
                .unwrap_or(R2rError::Fft(FftError::CollectivePreconditionFailed)));
        }
        preflight.expect("distributed DHT out-of-place preflight succeeded");
        match direction {
            Direction::Forward => execute_forward(&self.core, source, destination, workspace),
            Direction::Inverse | Direction::Backward => execute_reverse(
                &self.core,
                source,
                destination,
                workspace,
                matches!(direction, Direction::Inverse),
            ),
        }
    }

    fn execute_in_place(
        &self,
        direction: Direction,
        array: &mut R2rInPlaceArray<T, N, M>,
        workspace: &mut R2rInPlaceWorkspace<T, N, M>,
    ) -> Result<(), R2rError> {
        let communicator = self.input_pencil().topology().communicator();
        let operation = match direction {
            Direction::Forward => super::OPERATION_DHT_FORWARD_IN_PLACE,
            Direction::Inverse => super::OPERATION_DHT_INVERSE_IN_PLACE,
            Direction::Backward => super::OPERATION_DHT_BACKWARD_IN_PLACE,
        };
        agree_execution_descriptor_ref::<N, M>(communicator, operation, &self.core.descriptor)?;
        let expected_state = match direction {
            Direction::Forward => super::R2rState::Input,
            Direction::Inverse | Direction::Backward => super::R2rState::Output,
        };
        let expected = match direction {
            Direction::Forward => self.input_pencil(),
            Direction::Inverse | Direction::Backward => self.output_pencil(),
        };
        let preflight = if !Arc::ptr_eq(&array.core, &self.core) {
            Err(FftError::Array(pencil_array::ArrayError::IncompatiblePencils).into())
        } else if !Arc::ptr_eq(&workspace.core, &self.core) {
            Err(FftError::WorkspaceMismatch.into())
        } else if array.state == super::R2rState::Poisoned {
            Err(FftError::Array(pencil_array::ArrayError::Poisoned).into())
        } else if array.state != expected_state {
            Err(FftError::InputLayoutMismatch.into())
        } else if array.array.extra_shape() != &self.core.extra_shape {
            Err(FftError::ExtraShapeMismatch.into())
        } else if !array
            .array
            .active_pencil()
            .map_err(FftError::Array)?
            .same_layout(expected.as_ref())
        {
            Err(FftError::InputLayoutMismatch.into())
        } else {
            validate_workspace_lengths_values(
                workspace.fft_scratch.len(),
                self.core.fft_scratch_len,
                workspace.transpose.send_len(),
                self.core.transpose_send_len,
                workspace.transpose.receive_len(),
                self.core.transpose_receive_len,
            )
            .map_err(R2rError::Fft)
            .and_then(|()| {
                validate_embedding_len(workspace.embedding_line.len(), self.core.embedding_len)
            })
            .and_then(|()| {
                if workspace.line_buffer.len() < self.core.strided_line_len {
                    Err(FftError::WorkspaceTooSmall {
                        kind: "real line",
                        required: self.core.strided_line_len,
                        actual: workspace.line_buffer.len(),
                    }
                    .into())
                } else {
                    Ok(())
                }
            })
        };
        if !collective_valid(communicator, preflight.is_ok()) {
            return Err(preflight
                .err()
                .unwrap_or(R2rError::Fft(FftError::CollectivePreconditionFailed)));
        }
        preflight.expect("distributed DHT in-place preflight succeeded");
        let target = match direction {
            Direction::Forward => super::R2rState::Output,
            Direction::Inverse | Direction::Backward => super::R2rState::Input,
        };
        run_r2r_in_place_transaction(
            array,
            workspace,
            target,
            |array, workspace| match direction {
                Direction::Forward => execute_forward_in_place(&self.core, array, workspace),
                Direction::Inverse | Direction::Backward => execute_reverse_in_place(
                    &self.core,
                    array,
                    workspace,
                    matches!(direction, Direction::Inverse),
                ),
            },
        )
    }
}

/// Reusable storage for distributed [`R2rPlan`] out-of-place execution.
#[derive(Debug)]
pub struct R2rWorkspace<T: R2rScalar, const N: usize, const M: usize> {
    core: Arc<R2rCore<T, N, M>>,
    intermediate: ManyPencilArray<T, N, M>,
    transpose: TransposeWorkspace<T>,
    embedding_line: Vec<Complex<T::Real>>,
    fft_scratch: Vec<Complex<T::Real>>,
    line_buffer: Vec<T>,
}

/// An opaque single-buffer array for distributed R2R execution.
#[derive(Debug)]
pub struct R2rInPlaceArray<T: R2rScalar, const N: usize, const M: usize> {
    core: Arc<R2rCore<T, N, M>>,
    array: ManyPencilArray<T, N, M>,
    state: super::R2rState,
}

/// Reusable storage for distributed R2R in-place execution.
#[derive(Debug)]
pub struct R2rInPlaceWorkspace<T: R2rScalar, const N: usize, const M: usize> {
    core: Arc<R2rCore<T, N, M>>,
    transpose: TransposeWorkspace<T>,
    embedding_line: Vec<Complex<T::Real>>,
    fft_scratch: Vec<Complex<T::Real>>,
    line_buffer: Vec<T>,
}

#[derive(Debug)]
enum R2rLocal<T: R2rScalar> {
    Identity,
    Transform(LocalR2rPlan<T>),
    Hartley(LocalDhtPlan<T>),
}

impl<T: R2rScalar> R2rLocal<T> {
    fn embedding_len(&self) -> usize {
        match self {
            Self::Identity => 0,
            Self::Transform(plan) => plan.embedding_len(),
            Self::Hartley(plan) => plan.embedding_len(),
        }
    }

    fn scratch_len(&self) -> usize {
        match self {
            Self::Identity => 0,
            Self::Transform(plan) => plan.scratch_len(),
            Self::Hartley(plan) => plan.scratch_len(),
        }
    }
}

#[derive(Debug)]
struct R2rStage<T: R2rScalar, const N: usize, const M: usize> {
    axis: usize,
    input: Arc<Pencil<N, M>>,
    output: Arc<Pencil<N, M>>,
    local: R2rLocal<T>,
}

#[derive(Debug)]
struct R2rCore<T: R2rScalar, const N: usize, const M: usize> {
    stages: Box<[R2rStage<T, N, M>]>,
    transitions: Box<[C2cStageTransition<N, M>]>,
    extra_shape: ExtraShape,
    kinds: [Option<R2rKind>; N],
    descriptor: Box<[u64]>,
    layout: DistributedLayout,
    embedding_len: usize,
    fft_scratch_len: usize,
    strided_line_len: usize,
    transpose_send_len: usize,
    transpose_receive_len: usize,
}

impl<T: R2rScalar, const N: usize, const M: usize> R2rPlan<T, N, M>
where
    T: Equivalence,
{
    /// Collectively builds an Alltoallv plan from a canonical input pencil.
    pub fn from_pencil(
        input: Arc<Pencil<N, M>>,
        extra_shape: ExtraShape,
        kinds: [Option<R2rKind>; N],
    ) -> Result<Self, R2rError> {
        Self::from_pencil_with_layout(input, extra_shape, kinds, DistributedLayout::default())
    }

    /// Collectively builds a plan from a canonical input pencil and transport.
    pub fn from_pencil_with_method(
        input: Arc<Pencil<N, M>>,
        extra_shape: ExtraShape,
        kinds: [Option<R2rKind>; N],
        method: TransposeMethod,
    ) -> Result<Self, R2rError> {
        Self::from_pencil_with_layout(
            input,
            extra_shape,
            kinds,
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
        kinds: [Option<R2rKind>; N],
        layout: DistributedLayout,
    ) -> Result<Self, R2rError> {
        let topology = Arc::clone(input.topology());
        let global_shape = *input.global_shape();
        Self::construct(
            topology,
            global_shape,
            extra_shape,
            Ok(input),
            kinds,
            layout,
        )
    }

    /// Collectively builds an Alltoallv plan from a canonical input array.
    pub fn from_array(
        input: &PencilArray<T, N, M>,
        kinds: [Option<R2rKind>; N],
    ) -> Result<Self, R2rError> {
        Self::from_array_with_layout(input, kinds, DistributedLayout::default())
    }

    /// Collectively builds a plan from a canonical input array and transport.
    pub fn from_array_with_method(
        input: &PencilArray<T, N, M>,
        kinds: [Option<R2rKind>; N],
        method: TransposeMethod,
    ) -> Result<Self, R2rError> {
        Self::from_array_with_layout(
            input,
            kinds,
            DistributedLayout {
                transpose_method: method,
                ..DistributedLayout::default()
            },
        )
    }

    /// Collectively builds a plan with an explicit transport and layout policy.
    pub fn from_array_with_layout(
        input: &PencilArray<T, N, M>,
        kinds: [Option<R2rKind>; N],
        layout: DistributedLayout,
    ) -> Result<Self, R2rError> {
        let topology = Arc::clone(input.pencil().topology());
        let global_shape = *input.pencil().global_shape();
        Self::construct(
            topology,
            global_shape,
            input.extra_shape().clone(),
            Ok(Arc::clone(input.pencil())),
            kinds,
            layout,
        )
    }

    /// Collectively builds an Alltoallv plan from topology, shape, and kinds.
    pub fn from_shape(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
        kinds: [Option<R2rKind>; N],
    ) -> Result<Self, R2rError> {
        Self::from_shape_with_layout(
            topology,
            global_shape,
            extra_shape,
            kinds,
            DistributedLayout::default(),
        )
    }

    /// Collectively builds a plan from topology, shape, kinds, and transport.
    pub fn from_shape_with_method(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
        kinds: [Option<R2rKind>; N],
        method: TransposeMethod,
    ) -> Result<Self, R2rError> {
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
            kinds,
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
        kinds: [Option<R2rKind>; N],
        layout: DistributedLayout,
    ) -> Result<Self, R2rError> {
        let input = Pencil::new(
            Arc::clone(&topology),
            global_shape,
            std::array::from_fn(|axis| axis),
        )
        .map_err(FftError::Pencil);
        Self::construct(topology, global_shape, extra_shape, input, kinds, layout)
    }

    /// Returns the canonical input pencil.
    pub fn input_pencil(&self) -> &Arc<Pencil<N, M>> {
        &self.core.stages[0].input
    }

    /// Returns the output pencil selected by this plan's layout policy.
    pub fn output_pencil(&self) -> &Arc<Pencil<N, M>> {
        &self
            .core
            .stages
            .last()
            .expect("distributed R2R has at least two stages")
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

    /// Returns the per-logical-axis legacy FFTW transform kinds.
    pub fn kinds(&self) -> [Option<R2rKind>; N] {
        self.core.kinds
    }

    /// Allocates a zero-initialized local input array.
    pub fn allocate_input(&self) -> Result<PencilArray<T, N, M>, R2rError> {
        PencilArray::from_fn(
            Arc::clone(self.input_pencil()),
            self.core.extra_shape.clone(),
            crate::r2r::r2r_zero::<T>,
        )
        .map_err(FftError::Array)
        .map_err(R2rError::from)
    }

    /// Allocates a zero-initialized local output array.
    pub fn allocate_output(&self) -> Result<PencilArray<T, N, M>, R2rError> {
        PencilArray::from_fn(
            Arc::clone(self.output_pencil()),
            self.core.extra_shape.clone(),
            crate::r2r::r2r_zero::<T>,
        )
        .map_err(FftError::Array)
        .map_err(R2rError::from)
    }

    /// Allocates reusable out-of-place workspace.
    pub fn allocate_workspace(&self) -> Result<R2rWorkspace<T, N, M>, R2rError> {
        let intermediate = ManyPencilArray::from_elem(
            registered_stage_pencils(&self.core.stages)?,
            0,
            self.core.extra_shape.clone(),
            crate::r2r::r2r_zero::<T>(),
        )
        .map_err(map_array_allocation)
        .map_err(R2rError::from)?;
        Ok(R2rWorkspace {
            core: Arc::clone(&self.core),
            intermediate,
            transpose: TransposeWorkspace::from_vecs(
                initialized_vec(self.core.transpose_send_len, crate::r2r::r2r_zero::<T>())?,
                initialized_vec(self.core.transpose_receive_len, crate::r2r::r2r_zero::<T>())?,
            ),
            embedding_line: initialized_vec(self.core.embedding_len, zero_complex::<T::Real>())?,
            fft_scratch: initialized_vec(self.core.fft_scratch_len, zero_complex::<T::Real>())?,
            line_buffer: initialized_vec(self.core.strided_line_len, crate::r2r::r2r_zero::<T>())?,
        })
    }

    /// Allocates an opaque canonical input array for in-place execution.
    pub fn allocate_in_place(&self) -> Result<R2rInPlaceArray<T, N, M>, R2rError> {
        let array = ManyPencilArray::from_elem(
            registered_stage_pencils(&self.core.stages)?,
            0,
            self.core.extra_shape.clone(),
            crate::r2r::r2r_zero::<T>(),
        )
        .map_err(map_array_allocation)
        .map_err(R2rError::from)?;
        Ok(R2rInPlaceArray {
            core: Arc::clone(&self.core),
            array,
            state: super::R2rState::Input,
        })
    }

    /// Allocates reusable in-place workspace.
    pub fn allocate_in_place_workspace(&self) -> Result<R2rInPlaceWorkspace<T, N, M>, R2rError> {
        Ok(R2rInPlaceWorkspace {
            core: Arc::clone(&self.core),
            transpose: TransposeWorkspace::from_vecs(
                initialized_vec(self.core.transpose_send_len, crate::r2r::r2r_zero::<T>())?,
                initialized_vec(self.core.transpose_receive_len, crate::r2r::r2r_zero::<T>())?,
            ),
            embedding_line: initialized_vec(self.core.embedding_len, zero_complex::<T::Real>())?,
            fft_scratch: initialized_vec(self.core.fft_scratch_len, zero_complex::<T::Real>())?,
            line_buffer: initialized_vec(self.core.strided_line_len, crate::r2r::r2r_zero::<T>())?,
        })
    }

    /// Computes the selected raw forward transforms, preserving `source`.
    pub fn forward(
        &self,
        source: &PencilArray<T, N, M>,
        destination: &mut PencilArray<T, N, M>,
        workspace: &mut R2rWorkspace<T, N, M>,
    ) -> Result<(), R2rError> {
        self.execute(Direction::Forward, source, destination, workspace)
    }

    /// Computes the paired normalized inverse transforms, preserving `source`.
    pub fn inverse(
        &self,
        source: &PencilArray<T, N, M>,
        destination: &mut PencilArray<T, N, M>,
        workspace: &mut R2rWorkspace<T, N, M>,
    ) -> Result<(), R2rError> {
        self.execute(Direction::Inverse, source, destination, workspace)
    }

    /// Computes the paired raw backward transforms, preserving `source`.
    pub fn backward(
        &self,
        source: &PencilArray<T, N, M>,
        destination: &mut PencilArray<T, N, M>,
        workspace: &mut R2rWorkspace<T, N, M>,
    ) -> Result<(), R2rError> {
        self.execute(Direction::Backward, source, destination, workspace)
    }

    /// Computes the selected raw forward transforms in one state-checked buffer.
    pub fn forward_in_place(
        &self,
        array: &mut R2rInPlaceArray<T, N, M>,
        workspace: &mut R2rInPlaceWorkspace<T, N, M>,
    ) -> Result<(), R2rError> {
        self.execute_in_place(Direction::Forward, array, workspace)
    }

    /// Computes the paired normalized inverse in one state-checked buffer.
    pub fn inverse_in_place(
        &self,
        array: &mut R2rInPlaceArray<T, N, M>,
        workspace: &mut R2rInPlaceWorkspace<T, N, M>,
    ) -> Result<(), R2rError> {
        self.execute_in_place(Direction::Inverse, array, workspace)
    }

    /// Computes the paired raw backward transform in one state-checked buffer.
    pub fn backward_in_place(
        &self,
        array: &mut R2rInPlaceArray<T, N, M>,
        workspace: &mut R2rInPlaceWorkspace<T, N, M>,
    ) -> Result<(), R2rError> {
        self.execute_in_place(Direction::Backward, array, workspace)
    }

    fn construct(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
        input: Result<Arc<Pencil<N, M>>, FftError>,
        kinds: [Option<R2rKind>; N],
        layout: DistributedLayout,
    ) -> Result<Self, R2rError> {
        let axis_kinds = std::array::from_fn(|axis| kinds[axis].map(AxisR2rKind::Fftw));
        let core = Self::construct_core(
            topology,
            global_shape,
            extra_shape,
            input,
            kinds,
            axis_kinds,
            layout,
            super::OPERATION_R2R_PLAN,
        )?;
        Ok(Self { core })
    }

    fn construct_core(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
        input: Result<Arc<Pencil<N, M>>, FftError>,
        kinds: [Option<R2rKind>; N],
        axis_kinds: [Option<AxisR2rKind>; N],
        layout: DistributedLayout,
        operation: u64,
    ) -> Result<Arc<R2rCore<T, N, M>>, R2rError> {
        let communicator = topology.communicator();
        let expected_len = descriptor_len::<N, M>(&extra_shape);
        let descriptor = expected_len.and_then(|_| {
            build_r2r_descriptor::<T, N, M>(
                &topology,
                global_shape,
                &extra_shape,
                axis_kinds,
                layout,
            )
            .ok()
        });
        let descriptor_len_word = expected_len
            .and_then(|length| u64::try_from(length).ok())
            .unwrap_or(INVALID_WORD);
        let header = [
            super::DESCRIPTOR_SCHEMA,
            operation,
            u64::try_from(N).unwrap_or(INVALID_WORD),
            u64::try_from(M).unwrap_or(INVALID_WORD),
            descriptor_len_word,
        ];
        if !agree_header(communicator, header) {
            return Err(R2rError::Fft(FftError::CollectiveDescriptorMismatch));
        }
        let descriptor = collective_descriptor(communicator, descriptor, expected_len)?;
        let input = agree_result(communicator, validate_input(input, &topology, global_shape))?;
        agree_result(communicator, validate_kinds(global_shape, kinds))?;
        let route = agree_result(
            communicator,
            build_route(Ok(input), &topology, global_shape, layout.permute_dims),
        )?;
        let stages = agree_result(
            communicator,
            prepare_r2r_stages::<T, N, M>(&route, global_shape, axis_kinds),
        )?;

        let layout_stages = agree_result(
            communicator,
            (|| {
                let mut layout_stages = Vec::new();
                layout_stages
                    .try_reserve_exact(route.stages.len())
                    .map_err(|_| {
                        R2rError::Fft(FftError::AllocationFailed {
                            required: route.stages.len(),
                        })
                    })?;
                for (index, pencil) in route.stages.iter().enumerate() {
                    layout_stages.push(TransformStage {
                        axis: N - 1 - index,
                        input: Arc::clone(pencil),
                        output: Arc::clone(pencil),
                        local: super::LocalTransform::Identity,
                    });
                }
                Ok::<_, R2rError>(StagePreparation {
                    stages: layout_stages.into_boxed_slice(),
                    fft_scratch_len: 0,
                })
            })(),
        )?;
        let (
            transitions,
            transpose_send_len,
            transpose_receive_len,
            _real_send_len,
            _real_receive_len,
        ) = build_transitions::<T::Real, N, M>(
            communicator,
            &layout_stages,
            &route.distributed,
            &extra_shape,
            layout.transpose_method,
            0,
        )?;
        let strided_line_len = strided_r2r_line_len(&stages.stages)?;
        let core = R2rCore {
            stages: stages.stages,
            transitions: transitions.into_boxed_slice(),
            extra_shape,
            kinds,
            descriptor: descriptor.into_boxed_slice(),
            layout,
            embedding_len: stages.embedding_len,
            fft_scratch_len: stages.fft_scratch_len,
            strided_line_len,
            transpose_send_len,
            transpose_receive_len,
        };
        Ok(Arc::new(core))
    }

    fn execute(
        &self,
        direction: Direction,
        source: &PencilArray<T, N, M>,
        destination: &mut PencilArray<T, N, M>,
        workspace: &mut R2rWorkspace<T, N, M>,
    ) -> Result<(), R2rError> {
        let communicator = self.input_pencil().topology().communicator();
        let operation = match direction {
            Direction::Forward => OPERATION_R2R_FORWARD,
            Direction::Inverse => OPERATION_R2R_INVERSE,
            Direction::Backward => OPERATION_R2R_BACKWARD,
        };
        agree_execution_descriptor_ref::<N, M>(communicator, operation, &self.core.descriptor)?;
        let preflight = self.preflight(direction, source, destination, workspace);
        if !collective_valid(communicator, preflight.is_ok()) {
            return Err(preflight
                .err()
                .unwrap_or(R2rError::Fft(FftError::CollectivePreconditionFailed)));
        }
        preflight.expect("distributed R2R out-of-place preflight succeeded");
        match direction {
            Direction::Forward => execute_forward(&self.core, source, destination, workspace),
            Direction::Inverse | Direction::Backward => execute_reverse(
                &self.core,
                source,
                destination,
                workspace,
                matches!(direction, Direction::Inverse),
            ),
        }
    }

    fn preflight(
        &self,
        direction: Direction,
        source: &PencilArray<T, N, M>,
        destination: &PencilArray<T, N, M>,
        workspace: &R2rWorkspace<T, N, M>,
    ) -> Result<(), R2rError> {
        validate_r2r_out_of_place(
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
            workspace.embedding_line.len(),
        )?;
        if workspace.line_buffer.len() < self.core.strided_line_len {
            return Err(FftError::WorkspaceTooSmall {
                kind: "real line",
                required: self.core.strided_line_len,
                actual: workspace.line_buffer.len(),
            }
            .into());
        }
        Ok(())
    }

    fn execute_in_place(
        &self,
        direction: Direction,
        array: &mut R2rInPlaceArray<T, N, M>,
        workspace: &mut R2rInPlaceWorkspace<T, N, M>,
    ) -> Result<(), R2rError> {
        let communicator = self.input_pencil().topology().communicator();
        let operation = match direction {
            Direction::Forward => OPERATION_R2R_FORWARD_IN_PLACE,
            Direction::Inverse => OPERATION_R2R_INVERSE_IN_PLACE,
            Direction::Backward => OPERATION_R2R_BACKWARD_IN_PLACE,
        };
        agree_execution_descriptor_ref::<N, M>(communicator, operation, &self.core.descriptor)?;
        let preflight = self.preflight_in_place(direction, array, workspace);
        if !collective_valid(communicator, preflight.is_ok()) {
            return Err(preflight
                .err()
                .unwrap_or(R2rError::Fft(FftError::CollectivePreconditionFailed)));
        }
        preflight.expect("distributed R2R in-place preflight succeeded");
        let target = match direction {
            Direction::Forward => super::R2rState::Output,
            Direction::Inverse | Direction::Backward => super::R2rState::Input,
        };
        run_r2r_in_place_transaction(
            array,
            workspace,
            target,
            |array, workspace| match direction {
                Direction::Forward => execute_forward_in_place(&self.core, array, workspace),
                Direction::Inverse | Direction::Backward => execute_reverse_in_place(
                    &self.core,
                    array,
                    workspace,
                    matches!(direction, Direction::Inverse),
                ),
            },
        )
    }

    fn preflight_in_place(
        &self,
        direction: Direction,
        array: &R2rInPlaceArray<T, N, M>,
        workspace: &R2rInPlaceWorkspace<T, N, M>,
    ) -> Result<(), R2rError> {
        if !Arc::ptr_eq(&array.core, &self.core) {
            return Err(FftError::Array(pencil_array::ArrayError::IncompatiblePencils).into());
        }
        if !Arc::ptr_eq(&workspace.core, &self.core) {
            return Err(FftError::WorkspaceMismatch.into());
        }
        let expected_state = match direction {
            Direction::Forward => super::R2rState::Input,
            Direction::Inverse | Direction::Backward => super::R2rState::Output,
        };
        match array.state {
            super::R2rState::Poisoned => {
                return Err(FftError::Array(pencil_array::ArrayError::Poisoned).into());
            }
            state if state != expected_state => return Err(FftError::InputLayoutMismatch.into()),
            _ => {}
        }
        if array.array.extra_shape() != &self.core.extra_shape {
            return Err(FftError::ExtraShapeMismatch.into());
        }
        let expected = match direction {
            Direction::Forward => self.input_pencil(),
            Direction::Inverse | Direction::Backward => self.output_pencil(),
        };
        if !array
            .array
            .active_pencil()
            .map_err(FftError::Array)?
            .same_layout(expected.as_ref())
        {
            return Err(FftError::InputLayoutMismatch.into());
        }
        validate_workspace_lengths_values(
            workspace.fft_scratch.len(),
            self.core.fft_scratch_len,
            workspace.transpose.send_len(),
            self.core.transpose_send_len,
            workspace.transpose.receive_len(),
            self.core.transpose_receive_len,
        )?;
        validate_embedding_len(workspace.embedding_line.len(), self.core.embedding_len)?;
        if workspace.line_buffer.len() < self.core.strided_line_len {
            return Err(FftError::WorkspaceTooSmall {
                kind: "real line",
                required: self.core.strided_line_len,
                actual: workspace.line_buffer.len(),
            }
            .into());
        }
        Ok(())
    }
}

impl<T: R2rScalar, const N: usize, const M: usize> R2rInPlaceArray<T, N, M> {
    /// Returns the completion state of this in-place array.
    pub fn state(&self) -> super::R2rState {
        self.state
    }

    /// Borrows the active view while the array is not poisoned.
    pub fn view(&self) -> Result<PencilArrayView<'_, T, N, M>, R2rError> {
        self.ensure_viewable()?;
        self.array
            .active_view()
            .map_err(FftError::Array)
            .map_err(Into::into)
    }

    /// Borrows the active mutable view while the array is not poisoned.
    pub fn view_mut(&mut self) -> Result<PencilArrayViewMut<'_, T, N, M>, R2rError> {
        self.ensure_viewable()?;
        self.array
            .active_view_mut()
            .map_err(FftError::Array)
            .map_err(Into::into)
    }

    fn ensure_viewable(&self) -> Result<(), R2rError> {
        match self.state {
            super::R2rState::Input | super::R2rState::Output => Ok(()),
            super::R2rState::Poisoned => {
                Err(FftError::Array(pencil_array::ArrayError::Poisoned).into())
            }
        }
    }
}

fn run_r2r_in_place_transaction<T: R2rScalar, const N: usize, const M: usize, F>(
    array: &mut R2rInPlaceArray<T, N, M>,
    workspace: &mut R2rInPlaceWorkspace<T, N, M>,
    target: super::R2rState,
    body: F,
) -> Result<(), R2rError>
where
    F: FnOnce(
        &mut R2rInPlaceArray<T, N, M>,
        &mut R2rInPlaceWorkspace<T, N, M>,
    ) -> Result<(), R2rError>,
{
    array.state = super::R2rState::Poisoned;
    let result = body(array, workspace);
    if result.is_ok() {
        array.state = target;
    }
    result
}

fn validate_kinds<const N: usize>(
    global_shape: [usize; N],
    kinds: [Option<R2rKind>; N],
) -> Result<(), R2rError> {
    for (axis, kind) in kinds.into_iter().enumerate() {
        if kind == Some(R2rKind::DctI) && global_shape[axis] == 1 {
            return Err(LocalR2rError::InvalidLength.into());
        }
    }
    Ok(())
}

fn prepare_r2r_stages<T: R2rScalar, const N: usize, const M: usize>(
    route: &RouteCandidate<N, M>,
    global_shape: [usize; N],
    kinds: [Option<AxisR2rKind>; N],
) -> Result<R2rStagePreparation<T, N, M>, R2rError> {
    if route.stages.len() != N {
        return Err(FftError::PreparationFailed.into());
    }
    let mut stages = Vec::new();
    stages
        .try_reserve_exact(N)
        .map_err(|_| R2rError::Fft(FftError::AllocationFailed { required: N }))?;
    let mut embedding_len = 0;
    let mut fft_scratch_len = 0;
    for (index, pencil) in route.stages.iter().enumerate() {
        let axis = N - 1 - index;
        if pencil
            .decomposition()
            .iter()
            .any(|distributed| distributed.index() == axis)
            || pencil.local_shape_logical()[axis] != global_shape[axis]
        {
            return Err(FftError::PreparationFailed.into());
        }
        let local = match kinds[axis] {
            None => R2rLocal::Identity,
            Some(AxisR2rKind::Fftw(kind)) => {
                R2rLocal::Transform(LocalR2rPlan::new(global_shape[axis], kind)?)
            }
            Some(AxisR2rKind::Dht) => R2rLocal::Hartley(LocalDhtPlan::new(global_shape[axis])?),
        };
        embedding_len = embedding_len.max(local.embedding_len());
        fft_scratch_len = fft_scratch_len.max(local.scratch_len());
        stages.push(R2rStage {
            axis,
            input: Arc::clone(pencil),
            output: Arc::clone(pencil),
            local,
        });
    }
    Ok(R2rStagePreparation {
        stages: stages.into_boxed_slice(),
        embedding_len,
        fft_scratch_len,
    })
}

#[derive(Debug)]
struct R2rStagePreparation<T: R2rScalar, const N: usize, const M: usize> {
    stages: Box<[R2rStage<T, N, M>]>,
    embedding_len: usize,
    fft_scratch_len: usize,
}

fn registered_stage_pencils<T: R2rScalar, const N: usize, const M: usize>(
    stages: &[R2rStage<T, N, M>],
) -> Result<Box<[Arc<Pencil<N, M>>]>, R2rError> {
    let mut pencils = Vec::new();
    pencils.try_reserve_exact(stages.len()).map_err(|_| {
        R2rError::Fft(FftError::AllocationFailed {
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

fn build_r2r_descriptor<T: R2rScalar, const N: usize, const M: usize>(
    topology: &MpiTopology<M>,
    global_shape: [usize; N],
    extra_shape: &ExtraShape,
    kinds: [Option<AxisR2rKind>; N],
    layout: DistributedLayout,
) -> Result<Vec<u64>, ()> {
    let length = descriptor_len::<N, M>(extra_shape).ok_or(())?;
    let mut descriptor = Vec::new();
    descriptor.try_reserve_exact(length).map_err(|_| ())?;
    super::append_usizes(&mut descriptor, &global_shape)?;
    super::append_usizes(&mut descriptor, topology.process_grid())?;
    super::append_shape(&mut descriptor, extra_shape)?;
    descriptor.push(crate::r2r::r2r_value_kind::<T>());
    descriptor.push(u64::try_from(size_of::<T::Real>()).map_err(|_| ())?);
    descriptor.extend(
        kinds
            .into_iter()
            .map(|kind| kind.map_or(0, AxisR2rKind::descriptor_code)),
    );
    descriptor.push(layout.transpose_method.descriptor_word());
    descriptor.push(u64::from(layout.permute_dims));
    if descriptor.len() != length {
        return Err(());
    }
    Ok(descriptor)
}

fn execute_forward<T: R2rScalar + Equivalence, const N: usize, const M: usize>(
    core: &Arc<R2rCore<T, N, M>>,
    source: &PencilArray<T, N, M>,
    destination: &mut PencilArray<T, N, M>,
    workspace: &mut R2rWorkspace<T, N, M>,
) -> Result<(), R2rError> {
    let source_view = source.view();
    let stage = &core.stages[0];
    {
        let intermediate = &mut workspace.intermediate;
        let embedding_line = &mut workspace.embedding_line;
        let fft_scratch = &mut workspace.fft_scratch;
        intermediate
            .overwrite_with(stage.output.as_ref(), |mut target| {
                execute_local_forward(
                    &stage.local,
                    stage.output.as_ref(),
                    stage.axis,
                    source_view.as_slice(),
                    target.as_mut_slice(),
                    embedding_line,
                    fft_scratch,
                    &mut workspace.line_buffer,
                )
            })
            .map_err(map_overwrite_error)?;
    }
    let last = core.stages.len() - 1;
    for index in 0..core.transitions.len() {
        execute_transition(
            &core.transitions[index].forward,
            &mut workspace.intermediate,
            &mut workspace.transpose,
        )?;
        let stage = &core.stages[index + 1];
        if index + 1 == last {
            let active = workspace
                .intermediate
                .active_view()
                .map_err(FftError::Array)?;
            let mut destination_view = destination.view_mut();
            execute_local_forward(
                &stage.local,
                stage.output.as_ref(),
                stage.axis,
                active.as_slice(),
                destination_view.as_mut_slice(),
                &mut workspace.embedding_line,
                &mut workspace.fft_scratch,
                &mut workspace.line_buffer,
            )?;
        } else {
            let mut active = workspace
                .intermediate
                .active_view_mut()
                .map_err(FftError::Array)?;
            execute_local_forward_in_place(
                &stage.local,
                stage.output.as_ref(),
                stage.axis,
                active.as_mut_slice(),
                &mut workspace.embedding_line,
                &mut workspace.fft_scratch,
                &mut workspace.line_buffer,
            )?;
        }
    }
    Ok(())
}

fn execute_reverse<T: R2rScalar + Equivalence, const N: usize, const M: usize>(
    core: &Arc<R2rCore<T, N, M>>,
    source: &PencilArray<T, N, M>,
    destination: &mut PencilArray<T, N, M>,
    workspace: &mut R2rWorkspace<T, N, M>,
    normalize: bool,
) -> Result<(), R2rError> {
    let source_view = source.view();
    let last = core.stages.len() - 1;
    let stage = &core.stages[last];
    {
        let intermediate = &mut workspace.intermediate;
        let embedding_line = &mut workspace.embedding_line;
        let fft_scratch = &mut workspace.fft_scratch;
        intermediate
            .overwrite_with(stage.output.as_ref(), |mut target| {
                execute_local_reverse(
                    &stage.local,
                    stage.input.as_ref(),
                    stage.axis,
                    source_view.as_slice(),
                    target.as_mut_slice(),
                    embedding_line,
                    fft_scratch,
                    &mut workspace.line_buffer,
                    normalize,
                )
            })
            .map_err(map_overwrite_error)?;
    }
    for index in (0..core.transitions.len()).rev() {
        execute_transition(
            &core.transitions[index].backward,
            &mut workspace.intermediate,
            &mut workspace.transpose,
        )?;
        if index != 0 {
            let stage = &core.stages[index];
            let mut active = workspace
                .intermediate
                .active_view_mut()
                .map_err(FftError::Array)?;
            execute_local_reverse_in_place(
                &stage.local,
                stage.input.as_ref(),
                stage.axis,
                active.as_mut_slice(),
                &mut workspace.embedding_line,
                &mut workspace.fft_scratch,
                &mut workspace.line_buffer,
                normalize,
            )?;
        }
    }
    let stage = &core.stages[0];
    let active = workspace
        .intermediate
        .active_view()
        .map_err(FftError::Array)?;
    let mut destination_view = destination.view_mut();
    execute_local_reverse(
        &stage.local,
        stage.input.as_ref(),
        stage.axis,
        active.as_slice(),
        destination_view.as_mut_slice(),
        &mut workspace.embedding_line,
        &mut workspace.fft_scratch,
        &mut workspace.line_buffer,
        normalize,
    )
}

fn execute_forward_in_place<T: R2rScalar + Equivalence, const N: usize, const M: usize>(
    core: &Arc<R2rCore<T, N, M>>,
    array: &mut R2rInPlaceArray<T, N, M>,
    workspace: &mut R2rInPlaceWorkspace<T, N, M>,
) -> Result<(), R2rError> {
    {
        let stage = &core.stages[0];
        let mut active = array.array.active_view_mut().map_err(FftError::Array)?;
        execute_local_forward_in_place(
            &stage.local,
            stage.output.as_ref(),
            stage.axis,
            active.as_mut_slice(),
            &mut workspace.embedding_line,
            &mut workspace.fft_scratch,
            &mut workspace.line_buffer,
        )?;
    }
    for (index, transition) in core.transitions.iter().enumerate() {
        execute_transition(
            &transition.forward,
            &mut array.array,
            &mut workspace.transpose,
        )?;
        let stage = &core.stages[index + 1];
        let mut active = array.array.active_view_mut().map_err(FftError::Array)?;
        execute_local_forward_in_place(
            &stage.local,
            stage.output.as_ref(),
            stage.axis,
            active.as_mut_slice(),
            &mut workspace.embedding_line,
            &mut workspace.fft_scratch,
            &mut workspace.line_buffer,
        )?;
    }
    Ok(())
}

fn execute_reverse_in_place<T: R2rScalar + Equivalence, const N: usize, const M: usize>(
    core: &Arc<R2rCore<T, N, M>>,
    array: &mut R2rInPlaceArray<T, N, M>,
    workspace: &mut R2rInPlaceWorkspace<T, N, M>,
    normalize: bool,
) -> Result<(), R2rError> {
    let last = core.stages.len() - 1;
    {
        let stage = &core.stages[last];
        let mut active = array.array.active_view_mut().map_err(FftError::Array)?;
        execute_local_reverse_in_place(
            &stage.local,
            stage.input.as_ref(),
            stage.axis,
            active.as_mut_slice(),
            &mut workspace.embedding_line,
            &mut workspace.fft_scratch,
            &mut workspace.line_buffer,
            normalize,
        )?;
    }
    for (index, transition) in core.transitions.iter().enumerate().rev() {
        execute_transition(
            &transition.backward,
            &mut array.array,
            &mut workspace.transpose,
        )?;
        let stage = &core.stages[index];
        let mut active = array.array.active_view_mut().map_err(FftError::Array)?;
        execute_local_reverse_in_place(
            &stage.local,
            stage.input.as_ref(),
            stage.axis,
            active.as_mut_slice(),
            &mut workspace.embedding_line,
            &mut workspace.fft_scratch,
            &mut workspace.line_buffer,
            normalize,
        )?;
    }
    Ok(())
}

fn strided_r2r_line_len<T: R2rScalar, const N: usize, const M: usize>(
    stages: &[R2rStage<T, N, M>],
) -> Result<usize, R2rError> {
    let mut required = 0usize;
    for stage in stages {
        let line_len = match &stage.local {
            R2rLocal::Transform(plan) => plan.line_len(),
            R2rLocal::Hartley(plan) => plan.line_len(),
            R2rLocal::Identity => continue,
        };
        if super::memory_stride(stage.output.as_ref(), stage.axis)? > 1 {
            required = required.max(line_len);
        }
    }
    Ok(required)
}

fn execute_strided_r2r_forward<T: R2rScalar, const N: usize, const M: usize>(
    plan: &LocalR2rPlan<T>,
    pencil: &Pencil<N, M>,
    axis: usize,
    source: &[T],
    destination: &mut [T],
    embedding_line: &mut [Complex<T::Real>],
    fft_scratch: &mut [Complex<T::Real>],
    line_buffer: &mut [T],
) -> Result<(), R2rError> {
    let stride = super::memory_stride(pencil, axis)?;
    if stride <= 1 {
        return plan
            .forward(source, destination, embedding_line, fft_scratch)
            .map_err(R2rError::LocalR2r);
    }
    if line_buffer.len() < plan.line_len() {
        return Err(FftError::WorkspaceTooSmall {
            kind: "real line",
            required: plan.line_len(),
            actual: line_buffer.len(),
        }
        .into());
    }
    let count =
        super::strided_line_count(source.len(), plan.line_len(), stride).map_err(R2rError::Fft)?;
    if destination.len() != source.len() {
        return Err(FftError::PreparationFailed.into());
    }
    if source.is_empty() {
        return plan
            .forward(source, destination, embedding_line, fft_scratch)
            .map_err(R2rError::LocalR2r);
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
            plan.forward_in_place(
                &mut line_buffer[..plan.line_len()],
                embedding_line,
                fft_scratch,
            )
            .map_err(R2rError::LocalR2r)?;
            for k in 0..plan.line_len() {
                destination[base + k * stride + inner] = line_buffer[k];
            }
        }
    }
    Ok(())
}

fn execute_strided_r2r_forward_in_place<T: R2rScalar, const N: usize, const M: usize>(
    plan: &LocalR2rPlan<T>,
    pencil: &Pencil<N, M>,
    axis: usize,
    data: &mut [T],
    embedding_line: &mut [Complex<T::Real>],
    fft_scratch: &mut [Complex<T::Real>],
    line_buffer: &mut [T],
) -> Result<(), R2rError> {
    let stride = super::memory_stride(pencil, axis)?;
    if stride <= 1 {
        return plan
            .forward_in_place(data, embedding_line, fft_scratch)
            .map_err(R2rError::LocalR2r);
    }
    if line_buffer.len() < plan.line_len() {
        return Err(FftError::WorkspaceTooSmall {
            kind: "real line",
            required: plan.line_len(),
            actual: line_buffer.len(),
        }
        .into());
    }
    let count =
        super::strided_line_count(data.len(), plan.line_len(), stride).map_err(R2rError::Fft)?;
    if data.is_empty() {
        return plan
            .forward_in_place(data, embedding_line, fft_scratch)
            .map_err(R2rError::LocalR2r);
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
            plan.forward_in_place(
                &mut line_buffer[..plan.line_len()],
                embedding_line,
                fft_scratch,
            )
            .map_err(R2rError::LocalR2r)?;
            for k in 0..plan.line_len() {
                data[base + k * stride + inner] = line_buffer[k];
            }
        }
    }
    Ok(())
}

fn execute_strided_r2r_reverse<T: R2rScalar, const N: usize, const M: usize>(
    plan: &LocalR2rPlan<T>,
    pencil: &Pencil<N, M>,
    axis: usize,
    source: &[T],
    destination: &mut [T],
    embedding_line: &mut [Complex<T::Real>],
    fft_scratch: &mut [Complex<T::Real>],
    line_buffer: &mut [T],
    normalize: bool,
) -> Result<(), R2rError> {
    let stride = super::memory_stride(pencil, axis)?;
    if stride <= 1 {
        return if normalize {
            plan.inverse(source, destination, embedding_line, fft_scratch)
        } else {
            plan.backward(source, destination, embedding_line, fft_scratch)
        }
        .map_err(R2rError::LocalR2r);
    }
    if line_buffer.len() < plan.line_len() {
        return Err(FftError::WorkspaceTooSmall {
            kind: "real line",
            required: plan.line_len(),
            actual: line_buffer.len(),
        }
        .into());
    }
    let count =
        super::strided_line_count(source.len(), plan.line_len(), stride).map_err(R2rError::Fft)?;
    if destination.len() != source.len() {
        return Err(FftError::PreparationFailed.into());
    }
    if source.is_empty() {
        return if normalize {
            plan.inverse(source, destination, embedding_line, fft_scratch)
        } else {
            plan.backward(source, destination, embedding_line, fft_scratch)
        }
        .map_err(R2rError::LocalR2r);
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
                plan.inverse_in_place(
                    &mut line_buffer[..plan.line_len()],
                    embedding_line,
                    fft_scratch,
                )
            } else {
                plan.backward_in_place(
                    &mut line_buffer[..plan.line_len()],
                    embedding_line,
                    fft_scratch,
                )
            }
            .map_err(R2rError::LocalR2r)?;
            for k in 0..plan.line_len() {
                destination[base + k * stride + inner] = line_buffer[k];
            }
        }
    }
    Ok(())
}

fn execute_strided_r2r_reverse_in_place<T: R2rScalar, const N: usize, const M: usize>(
    plan: &LocalR2rPlan<T>,
    pencil: &Pencil<N, M>,
    axis: usize,
    data: &mut [T],
    embedding_line: &mut [Complex<T::Real>],
    fft_scratch: &mut [Complex<T::Real>],
    line_buffer: &mut [T],
    normalize: bool,
) -> Result<(), R2rError> {
    let stride = super::memory_stride(pencil, axis)?;
    if stride <= 1 {
        return if normalize {
            plan.inverse_in_place(data, embedding_line, fft_scratch)
        } else {
            plan.backward_in_place(data, embedding_line, fft_scratch)
        }
        .map_err(R2rError::LocalR2r);
    }
    if line_buffer.len() < plan.line_len() {
        return Err(FftError::WorkspaceTooSmall {
            kind: "real line",
            required: plan.line_len(),
            actual: line_buffer.len(),
        }
        .into());
    }
    let count =
        super::strided_line_count(data.len(), plan.line_len(), stride).map_err(R2rError::Fft)?;
    if data.is_empty() {
        return if normalize {
            plan.inverse_in_place(data, embedding_line, fft_scratch)
        } else {
            plan.backward_in_place(data, embedding_line, fft_scratch)
        }
        .map_err(R2rError::LocalR2r);
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
                plan.inverse_in_place(
                    &mut line_buffer[..plan.line_len()],
                    embedding_line,
                    fft_scratch,
                )
            } else {
                plan.backward_in_place(
                    &mut line_buffer[..plan.line_len()],
                    embedding_line,
                    fft_scratch,
                )
            }
            .map_err(R2rError::LocalR2r)?;
            for k in 0..plan.line_len() {
                data[base + k * stride + inner] = line_buffer[k];
            }
        }
    }
    Ok(())
}

fn execute_strided_dht_forward<T: R2rScalar, const N: usize, const M: usize>(
    plan: &LocalDhtPlan<T>,
    pencil: &Pencil<N, M>,
    axis: usize,
    source: &[T],
    destination: &mut [T],
    embedding_line: &mut [Complex<T::Real>],
    fft_scratch: &mut [Complex<T::Real>],
    line_buffer: &mut [T],
) -> Result<(), R2rError> {
    let stride = super::memory_stride(pencil, axis)?;
    if stride <= 1 {
        return plan
            .forward(source, destination, embedding_line, fft_scratch)
            .map_err(R2rError::LocalR2r);
    }
    if line_buffer.len() < plan.line_len() {
        return Err(FftError::WorkspaceTooSmall {
            kind: "real line",
            required: plan.line_len(),
            actual: line_buffer.len(),
        }
        .into());
    }
    let count =
        super::strided_line_count(source.len(), plan.line_len(), stride).map_err(R2rError::Fft)?;
    if destination.len() != source.len() {
        return Err(FftError::PreparationFailed.into());
    }
    if source.is_empty() {
        return plan
            .forward(source, destination, embedding_line, fft_scratch)
            .map_err(R2rError::LocalR2r);
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
            plan.forward_in_place(
                &mut line_buffer[..plan.line_len()],
                embedding_line,
                fft_scratch,
            )
            .map_err(R2rError::LocalR2r)?;
            for k in 0..plan.line_len() {
                destination[base + k * stride + inner] = line_buffer[k];
            }
        }
    }
    Ok(())
}

fn execute_strided_dht_forward_in_place<T: R2rScalar, const N: usize, const M: usize>(
    plan: &LocalDhtPlan<T>,
    pencil: &Pencil<N, M>,
    axis: usize,
    data: &mut [T],
    embedding_line: &mut [Complex<T::Real>],
    fft_scratch: &mut [Complex<T::Real>],
    line_buffer: &mut [T],
) -> Result<(), R2rError> {
    let stride = super::memory_stride(pencil, axis)?;
    if stride <= 1 {
        return plan
            .forward_in_place(data, embedding_line, fft_scratch)
            .map_err(R2rError::LocalR2r);
    }
    if line_buffer.len() < plan.line_len() {
        return Err(FftError::WorkspaceTooSmall {
            kind: "real line",
            required: plan.line_len(),
            actual: line_buffer.len(),
        }
        .into());
    }
    let count =
        super::strided_line_count(data.len(), plan.line_len(), stride).map_err(R2rError::Fft)?;
    if data.is_empty() {
        return plan
            .forward_in_place(data, embedding_line, fft_scratch)
            .map_err(R2rError::LocalR2r);
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
            plan.forward_in_place(
                &mut line_buffer[..plan.line_len()],
                embedding_line,
                fft_scratch,
            )
            .map_err(R2rError::LocalR2r)?;
            for k in 0..plan.line_len() {
                data[base + k * stride + inner] = line_buffer[k];
            }
        }
    }
    Ok(())
}

fn execute_strided_dht_reverse<T: R2rScalar, const N: usize, const M: usize>(
    plan: &LocalDhtPlan<T>,
    pencil: &Pencil<N, M>,
    axis: usize,
    source: &[T],
    destination: &mut [T],
    embedding_line: &mut [Complex<T::Real>],
    fft_scratch: &mut [Complex<T::Real>],
    line_buffer: &mut [T],
    normalize: bool,
) -> Result<(), R2rError> {
    let stride = super::memory_stride(pencil, axis)?;
    if stride <= 1 {
        return if normalize {
            plan.inverse(source, destination, embedding_line, fft_scratch)
        } else {
            plan.backward(source, destination, embedding_line, fft_scratch)
        }
        .map_err(R2rError::LocalR2r);
    }
    if line_buffer.len() < plan.line_len() {
        return Err(FftError::WorkspaceTooSmall {
            kind: "real line",
            required: plan.line_len(),
            actual: line_buffer.len(),
        }
        .into());
    }
    let count =
        super::strided_line_count(source.len(), plan.line_len(), stride).map_err(R2rError::Fft)?;
    if destination.len() != source.len() {
        return Err(FftError::PreparationFailed.into());
    }
    if source.is_empty() {
        return if normalize {
            plan.inverse(source, destination, embedding_line, fft_scratch)
        } else {
            plan.backward(source, destination, embedding_line, fft_scratch)
        }
        .map_err(R2rError::LocalR2r);
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
                plan.inverse_in_place(
                    &mut line_buffer[..plan.line_len()],
                    embedding_line,
                    fft_scratch,
                )
            } else {
                plan.backward_in_place(
                    &mut line_buffer[..plan.line_len()],
                    embedding_line,
                    fft_scratch,
                )
            }
            .map_err(R2rError::LocalR2r)?;
            for k in 0..plan.line_len() {
                destination[base + k * stride + inner] = line_buffer[k];
            }
        }
    }
    Ok(())
}

fn execute_strided_dht_reverse_in_place<T: R2rScalar, const N: usize, const M: usize>(
    plan: &LocalDhtPlan<T>,
    pencil: &Pencil<N, M>,
    axis: usize,
    data: &mut [T],
    embedding_line: &mut [Complex<T::Real>],
    fft_scratch: &mut [Complex<T::Real>],
    line_buffer: &mut [T],
    normalize: bool,
) -> Result<(), R2rError> {
    let stride = super::memory_stride(pencil, axis)?;
    if stride <= 1 {
        return if normalize {
            plan.inverse_in_place(data, embedding_line, fft_scratch)
        } else {
            plan.backward_in_place(data, embedding_line, fft_scratch)
        }
        .map_err(R2rError::LocalR2r);
    }
    if line_buffer.len() < plan.line_len() {
        return Err(FftError::WorkspaceTooSmall {
            kind: "real line",
            required: plan.line_len(),
            actual: line_buffer.len(),
        }
        .into());
    }
    let count =
        super::strided_line_count(data.len(), plan.line_len(), stride).map_err(R2rError::Fft)?;
    if data.is_empty() {
        return if normalize {
            plan.inverse_in_place(data, embedding_line, fft_scratch)
        } else {
            plan.backward_in_place(data, embedding_line, fft_scratch)
        }
        .map_err(R2rError::LocalR2r);
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
                plan.inverse_in_place(
                    &mut line_buffer[..plan.line_len()],
                    embedding_line,
                    fft_scratch,
                )
            } else {
                plan.backward_in_place(
                    &mut line_buffer[..plan.line_len()],
                    embedding_line,
                    fft_scratch,
                )
            }
            .map_err(R2rError::LocalR2r)?;
            for k in 0..plan.line_len() {
                data[base + k * stride + inner] = line_buffer[k];
            }
        }
    }
    Ok(())
}

fn execute_local_forward<T: R2rScalar, const N: usize, const M: usize>(
    local: &R2rLocal<T>,
    pencil: &Pencil<N, M>,
    axis: usize,
    source: &[T],
    destination: &mut [T],
    embedding_line: &mut [Complex<T::Real>],
    fft_scratch: &mut [Complex<T::Real>],
    line_buffer: &mut [T],
) -> Result<(), R2rError> {
    match local {
        R2rLocal::Identity => {
            if source.len() != destination.len() {
                return Err(FftError::PreparationFailed.into());
            }
            destination.copy_from_slice(source);
            Ok(())
        }
        R2rLocal::Transform(plan) => execute_strided_r2r_forward(
            plan,
            pencil,
            axis,
            source,
            destination,
            embedding_line,
            fft_scratch,
            line_buffer,
        ),
        R2rLocal::Hartley(plan) => execute_strided_dht_forward(
            plan,
            pencil,
            axis,
            source,
            destination,
            embedding_line,
            fft_scratch,
            line_buffer,
        ),
    }
}

fn execute_local_forward_in_place<T: R2rScalar, const N: usize, const M: usize>(
    local: &R2rLocal<T>,
    pencil: &Pencil<N, M>,
    axis: usize,
    data: &mut [T],
    embedding_line: &mut [Complex<T::Real>],
    fft_scratch: &mut [Complex<T::Real>],
    line_buffer: &mut [T],
) -> Result<(), R2rError> {
    match local {
        R2rLocal::Identity => Ok(()),
        R2rLocal::Transform(plan) => execute_strided_r2r_forward_in_place(
            plan,
            pencil,
            axis,
            data,
            embedding_line,
            fft_scratch,
            line_buffer,
        ),
        R2rLocal::Hartley(plan) => execute_strided_dht_forward_in_place(
            plan,
            pencil,
            axis,
            data,
            embedding_line,
            fft_scratch,
            line_buffer,
        ),
    }
}

fn execute_local_reverse<T: R2rScalar, const N: usize, const M: usize>(
    local: &R2rLocal<T>,
    pencil: &Pencil<N, M>,
    axis: usize,
    source: &[T],
    destination: &mut [T],
    embedding_line: &mut [Complex<T::Real>],
    fft_scratch: &mut [Complex<T::Real>],
    line_buffer: &mut [T],
    normalize: bool,
) -> Result<(), R2rError> {
    match local {
        R2rLocal::Identity => {
            if source.len() != destination.len() {
                return Err(FftError::PreparationFailed.into());
            }
            destination.copy_from_slice(source);
            Ok(())
        }
        R2rLocal::Transform(plan) => execute_strided_r2r_reverse(
            plan,
            pencil,
            axis,
            source,
            destination,
            embedding_line,
            fft_scratch,
            line_buffer,
            normalize,
        ),
        R2rLocal::Hartley(plan) => execute_strided_dht_reverse(
            plan,
            pencil,
            axis,
            source,
            destination,
            embedding_line,
            fft_scratch,
            line_buffer,
            normalize,
        ),
    }
}

fn execute_local_reverse_in_place<T: R2rScalar, const N: usize, const M: usize>(
    local: &R2rLocal<T>,
    pencil: &Pencil<N, M>,
    axis: usize,
    data: &mut [T],
    embedding_line: &mut [Complex<T::Real>],
    fft_scratch: &mut [Complex<T::Real>],
    line_buffer: &mut [T],
    normalize: bool,
) -> Result<(), R2rError> {
    match local {
        R2rLocal::Identity => Ok(()),
        R2rLocal::Transform(plan) => execute_strided_r2r_reverse_in_place(
            plan,
            pencil,
            axis,
            data,
            embedding_line,
            fft_scratch,
            line_buffer,
            normalize,
        ),
        R2rLocal::Hartley(plan) => execute_strided_dht_reverse_in_place(
            plan,
            pencil,
            axis,
            data,
            embedding_line,
            fft_scratch,
            line_buffer,
            normalize,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_r2r_out_of_place<T: R2rScalar, const N: usize, const M: usize>(
    core: &Arc<R2rCore<T, N, M>>,
    workspace_core: &Arc<R2rCore<T, N, M>>,
    direction: Direction,
    source: &PencilArray<T, N, M>,
    destination: &PencilArray<T, N, M>,
    intermediate: &ManyPencilArray<T, N, M>,
    lengths: (usize, usize, usize),
    embedding_len: usize,
) -> Result<(), R2rError> {
    if !Arc::ptr_eq(workspace_core, core) {
        return Err(FftError::WorkspaceMismatch.into());
    }
    let input = &core.stages[0].input;
    let output = &core
        .stages
        .last()
        .expect("distributed R2R has at least two stages")
        .output;
    let (expected_source, expected_destination) = match direction {
        Direction::Forward => (input, output),
        Direction::Inverse | Direction::Backward => (output, input),
    };
    if !source.pencil().same_layout(expected_source.as_ref()) {
        return Err(FftError::InputLayoutMismatch.into());
    }
    if !destination
        .pencil()
        .same_layout(expected_destination.as_ref())
    {
        return Err(FftError::OutputLayoutMismatch.into());
    }
    if source.extra_shape() != &core.extra_shape || destination.extra_shape() != &core.extra_shape {
        return Err(FftError::ExtraShapeMismatch.into());
    }
    validate_workspace_lengths_values(
        lengths.0,
        core.fft_scratch_len,
        lengths.1,
        core.transpose_send_len,
        lengths.2,
        core.transpose_receive_len,
    )?;
    validate_embedding_len(embedding_len, core.embedding_len)?;
    if intermediate.extra_shape() != &core.extra_shape {
        return Err(FftError::WorkspaceMismatch.into());
    }
    let active = intermediate.active_pencil().map_err(FftError::Array)?;
    if !core
        .stages
        .iter()
        .any(|stage| active.same_layout(stage.output.as_ref()))
    {
        return Err(FftError::WorkspaceMismatch.into());
    }
    Ok(())
}

fn validate_embedding_len(actual: usize, required: usize) -> Result<(), R2rError> {
    if actual < required {
        return Err(FftError::WorkspaceTooSmall {
            kind: "complex embedding",
            required,
            actual,
        }
        .into());
    }
    Ok(())
}

fn map_overwrite_error<E>(error: OverwriteError<E>) -> R2rError
where
    E: Into<R2rError>,
{
    match error {
        OverwriteError::Array(error) => FftError::Array(error).into(),
        OverwriteError::Writer(error) => error.into(),
    }
}

fn zero_complex<R: super::FftReal>() -> Complex<R> {
    Complex::new(R::zero(), R::zero())
}

#[cfg(test)]
mod tests {
    use std::{panic::AssertUnwindSafe, sync::Arc};

    use mpi::traits::*;
    use pencil_array::{ExtraShape, MpiTopology};

    use super::*;

    fn assert_poisoned(
        plan: &R2rPlan<f64, 2, 1>,
        array: &mut R2rInPlaceArray<f64, 2, 1>,
        workspace: &mut R2rInPlaceWorkspace<f64, 2, 1>,
    ) {
        assert_eq!(array.state(), super::super::R2rState::Poisoned);
        assert!(matches!(
            array.view(),
            Err(R2rError::Fft(FftError::Array(
                pencil_array::ArrayError::Poisoned
            )))
        ));
        assert!(matches!(
            array.view_mut(),
            Err(R2rError::Fft(FftError::Array(
                pencil_array::ArrayError::Poisoned
            )))
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
                Err(R2rError::Fft(FftError::Array(
                    pencil_array::ArrayError::Poisoned
                )))
            ));
            assert_eq!(format!("{array:?}"), array_before);
            assert_eq!(format!("{workspace:?}"), workspace_before);
        }
    }

    fn assert_dht_poisoned(
        plan: &DhtPlan<f64, 2, 1>,
        array: &mut R2rInPlaceArray<f64, 2, 1>,
        workspace: &mut R2rInPlaceWorkspace<f64, 2, 1>,
    ) {
        assert_eq!(array.state(), super::super::R2rState::Poisoned);
        assert!(matches!(
            array.view(),
            Err(R2rError::Fft(FftError::Array(
                pencil_array::ArrayError::Poisoned
            )))
        ));
        assert!(matches!(
            array.view_mut(),
            Err(R2rError::Fft(FftError::Array(
                pencil_array::ArrayError::Poisoned
            )))
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
                Err(R2rError::Fft(FftError::Array(
                    pencil_array::ArrayError::Poisoned
                )))
            ));
            assert_eq!(format!("{array:?}"), array_before);
            assert_eq!(format!("{workspace:?}"), workspace_before);
        }
    }

    fn dht_public_in_place_error_contracts(
        topology: &Arc<MpiTopology<1>>,
        method: TransposeMethod,
    ) {
        let plan = DhtPlan::<f64, 2, 1>::from_shape_with_method(
            Arc::clone(topology),
            [2, 3],
            ExtraShape::scalar(),
            method,
        )
        .unwrap();

        let mut short_array = plan.allocate_in_place().unwrap();
        let mut short_workspace = plan.allocate_in_place_workspace().unwrap();
        short_workspace.embedding_line.clear();
        let array_before = format!("{short_array:?}");
        let workspace_before = format!("{short_workspace:?}");
        let short_result = plan.forward_in_place(&mut short_array, &mut short_workspace);
        assert!(matches!(
            short_result,
            Err(R2rError::Fft(FftError::WorkspaceTooSmall { .. }))
        ));
        assert_eq!(short_array.state(), super::super::R2rState::Input);
        assert_eq!(format!("{short_array:?}"), array_before);
        assert_eq!(format!("{short_workspace:?}"), workspace_before);

        let mut wrong_state = plan.allocate_in_place().unwrap();
        let mut wrong_workspace = plan.allocate_in_place_workspace().unwrap();
        assert!(matches!(
            plan.inverse_in_place(&mut wrong_state, &mut wrong_workspace),
            Err(R2rError::Fft(FftError::InputLayoutMismatch))
        ));
        plan.forward_in_place(&mut wrong_state, &mut wrong_workspace)
            .unwrap();
        assert!(matches!(
            plan.forward_in_place(&mut wrong_state, &mut wrong_workspace),
            Err(R2rError::Fft(FftError::InputLayoutMismatch))
        ));
        plan.inverse_in_place(&mut wrong_state, &mut wrong_workspace)
            .unwrap();
        plan.forward_in_place(&mut wrong_state, &mut wrong_workspace)
            .unwrap();
        plan.backward_in_place(&mut wrong_state, &mut wrong_workspace)
            .unwrap();

        let foreign_plan = DhtPlan::<f64, 2, 1>::from_shape_with_method(
            Arc::clone(topology),
            [2, 3],
            ExtraShape::scalar(),
            method,
        )
        .unwrap();
        let mut foreign_array = foreign_plan.allocate_in_place().unwrap();
        let mut own_workspace = plan.allocate_in_place_workspace().unwrap();
        assert!(matches!(
            plan.forward_in_place(&mut foreign_array, &mut own_workspace),
            Err(R2rError::Fft(FftError::Array(
                pencil_array::ArrayError::IncompatiblePencils
            )))
        ));
        let mut own_array = plan.allocate_in_place().unwrap();
        let mut foreign_workspace = foreign_plan.allocate_in_place_workspace().unwrap();
        assert!(matches!(
            plan.forward_in_place(&mut own_array, &mut foreign_workspace),
            Err(R2rError::Fft(FftError::WorkspaceMismatch))
        ));

        for panic_failure in [false, true] {
            let mut array = plan.allocate_in_place().unwrap();
            let mut workspace = plan.allocate_in_place_workspace().unwrap();
            let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                run_r2r_in_place_transaction(
                    &mut array,
                    &mut workspace,
                    super::super::R2rState::Output,
                    |array, workspace| {
                        assert_eq!(array.state(), super::super::R2rState::Poisoned);
                        array.array.active_view_mut().unwrap().as_mut_slice()[0] = 7.0;
                        workspace.embedding_line[0].re = 9.0;
                        if panic_failure {
                            panic!("DHT in-place test panic after start");
                        }
                        Err(FftError::PreparationFailed.into())
                    },
                )
            }));
            if panic_failure {
                assert!(result.is_err());
            } else {
                assert!(matches!(
                    result,
                    Ok(Err(R2rError::Fft(FftError::PreparationFailed)))
                ));
            }
            assert_dht_poisoned(&plan, &mut array, &mut workspace);
        }

        let mut corrupted_plan = DhtPlan::<f64, 2, 1>::from_shape_with_method(
            Arc::clone(topology),
            [2, 3],
            ExtraShape::scalar(),
            method,
        )
        .unwrap();
        Arc::get_mut(&mut corrupted_plan.core)
            .unwrap()
            .stages
            .last_mut()
            .unwrap()
            .local = R2rLocal::Hartley(LocalDhtPlan::new(4).unwrap());
        let mut corrupted_array = corrupted_plan.allocate_in_place().unwrap();
        let mut corrupted_workspace = corrupted_plan.allocate_in_place_workspace().unwrap();
        assert!(matches!(
            corrupted_plan.forward_in_place(&mut corrupted_array, &mut corrupted_workspace),
            Err(R2rError::LocalR2r(LocalR2rError::NonIntegralBatch))
        ));
        assert_dht_poisoned(
            &corrupted_plan,
            &mut corrupted_array,
            &mut corrupted_workspace,
        );
    }

    #[test]
    #[ignore = "MPI can be initialized only once per unit-test process; run this test explicitly"]
    fn in_place_error_panic_backend_and_short_workspace_poison_contracts() {
        let _mpi_test_lock = super::super::MPI_TEST_LOCK
            .get_or_init(|| std::sync::Mutex::new(()))
            .lock()
            .unwrap();
        let universe = mpi::initialize().expect("MPI initialization failed");
        let world = universe.world();
        assert_eq!(world.size(), 1, "run this unit test with one MPI rank");
        let topology = MpiTopology::<1>::new(&world, [1]).unwrap();

        for method in [TransposeMethod::AllToAllv, TransposeMethod::PointToPoint] {
            let plan = R2rPlan::<f64, 2, 1>::from_shape_with_method(
                Arc::clone(&topology),
                [2, 3],
                ExtraShape::scalar(),
                [Some(R2rKind::DctII), Some(R2rKind::DstIII)],
                method,
            )
            .unwrap();

            let mut short_array = plan.allocate_in_place().unwrap();
            let mut short_workspace = plan.allocate_in_place_workspace().unwrap();
            short_workspace.embedding_line.clear();
            let array_before = format!("{short_array:?}");
            let workspace_before = format!("{short_workspace:?}");
            let short_result = plan.forward_in_place(&mut short_array, &mut short_workspace);
            assert!(
                matches!(
                    &short_result,
                    Err(R2rError::Fft(FftError::WorkspaceTooSmall { .. }))
                ),
                "short workspace result: {short_result:?}"
            );
            assert_eq!(short_array.state(), super::super::R2rState::Input);
            assert_eq!(format!("{short_array:?}"), array_before);
            assert_eq!(format!("{short_workspace:?}"), workspace_before);

            for panic_failure in [false, true] {
                let mut array = plan.allocate_in_place().unwrap();
                let mut workspace = plan.allocate_in_place_workspace().unwrap();
                let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                    run_r2r_in_place_transaction(
                        &mut array,
                        &mut workspace,
                        super::super::R2rState::Output,
                        |array, workspace| {
                            assert_eq!(array.state(), super::super::R2rState::Poisoned);
                            array.array.active_view_mut().unwrap().as_mut_slice()[0] = 7.0;
                            workspace.embedding_line[0].re = 9.0;
                            if panic_failure {
                                panic!("R2R in-place test panic after start");
                            }
                            Err(FftError::PreparationFailed.into())
                        },
                    )
                }));
                if panic_failure {
                    assert!(result.is_err());
                } else {
                    assert!(matches!(
                        result,
                        Ok(Err(R2rError::Fft(FftError::PreparationFailed)))
                    ));
                }
                assert_poisoned(&plan, &mut array, &mut workspace);
            }

            let mut forward_plan = R2rPlan::<f64, 2, 1>::from_shape_with_method(
                Arc::clone(&topology),
                [2, 3],
                ExtraShape::scalar(),
                [Some(R2rKind::DctII), Some(R2rKind::DstIII)],
                method,
            )
            .unwrap();
            Arc::get_mut(&mut forward_plan.core)
                .unwrap()
                .stages
                .last_mut()
                .unwrap()
                .local = R2rLocal::Transform(LocalR2rPlan::new(4, R2rKind::DctII).unwrap());
            let mut forward_array = forward_plan.allocate_in_place().unwrap();
            let mut forward_workspace = forward_plan.allocate_in_place_workspace().unwrap();
            assert!(matches!(
                forward_plan.forward_in_place(&mut forward_array, &mut forward_workspace),
                Err(R2rError::LocalR2r(LocalR2rError::NonIntegralBatch))
            ));
            assert_poisoned(&forward_plan, &mut forward_array, &mut forward_workspace);

            let mut backward_plan = R2rPlan::<f64, 2, 1>::from_shape_with_method(
                Arc::clone(&topology),
                [2, 3],
                ExtraShape::scalar(),
                [Some(R2rKind::DctII), Some(R2rKind::DstIII)],
                method,
            )
            .unwrap();
            Arc::get_mut(&mut backward_plan.core)
                .unwrap()
                .stages
                .first_mut()
                .unwrap()
                .local = R2rLocal::Transform(LocalR2rPlan::new(4, R2rKind::DctII).unwrap());
            let mut backward_array = backward_plan.allocate_in_place().unwrap();
            backward_array
                .array
                .overwrite_with(backward_plan.output_pencil().as_ref(), |mut view| {
                    view.as_mut_slice().fill(1.0);
                    Ok::<_, ()>(())
                })
                .unwrap();
            backward_array.state = super::super::R2rState::Output;
            let mut backward_workspace = backward_plan.allocate_in_place_workspace().unwrap();
            assert!(matches!(
                backward_plan.backward_in_place(&mut backward_array, &mut backward_workspace),
                Err(R2rError::LocalR2r(LocalR2rError::NonIntegralBatch))
            ));
            assert_poisoned(&backward_plan, &mut backward_array, &mut backward_workspace);

            dht_public_in_place_error_contracts(&topology, method);
        }
    }
}
