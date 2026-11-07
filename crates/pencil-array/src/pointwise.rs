use crate::{
    ArrayError, PencilArray, PencilArrayView, PencilArrayViewMut, checked::checked_product,
};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
/// Validation errors for checked pointwise operations.
pub enum PointwiseError {
    /// Spatial layouts differ.
    #[error("pointwise arrays must have identical spatial layouts")]
    IncompatiblePencils,
    /// Extra ranks differ.
    #[error("pointwise extra rank mismatch: expected {expected}, got {actual}")]
    ExtraRankMismatch {
        /// Expected rank.
        expected: usize,
        /// Actual rank.
        actual: usize,
    },
    /// An input extent is neither output extent nor singleton.
    #[error("pointwise extra extent mismatch at axis {axis}: output {output}, input {input}")]
    ExtraExtentMismatch {
        /// Axis.
        axis: usize,
        /// Output extent.
        output: usize,
        /// Input extent.
        input: usize,
    },
    /// A checked storage calculation failed.
    #[error(transparent)]
    Array(#[from] ArrayError),
}

/// Applies `f` to two views with identical spatial layouts and equal extra rank.
/// Extra dimensions broadcast from extent one (including one to zero).
/// Scalar operands can be captured by `f`. Uses O(extra rank) metadata, no input
/// clones or full-sized temporary. Visits extra indices then spatial memory order.
/// Validation errors leave output unchanged; callback panics may partially write.
pub fn pointwise2_views<A, B, U, F, const N: usize, const M: usize>(
    left: PencilArrayView<'_, A, N, M>,
    right: PencilArrayView<'_, B, N, M>,
    mut output: PencilArrayViewMut<'_, U, N, M>,
    mut f: F,
) -> Result<(), PointwiseError>
where
    F: FnMut(&A, &B) -> U,
{
    validate_two_inputs(
        &left,
        &right,
        output.pencil(),
        output.extra_shape().dimensions(),
    )?;
    let (plan, (left_dims, left_strides)) =
        OutputPlan::new(&left, output.pencil(), output.extra_shape().dimensions())?;
    let (right_dims, right_strides) = plan.input(output.pencil(), &right)?;
    if plan.spatial == 0 {
        return Ok(());
    }
    let ls = left.as_slice();
    let rs = right.as_slice();
    let os = output.as_mut_slice();
    for n in 0..plan.extra_count {
        let li = broadcast_linear(n, &plan.dims, &left_dims, &left_strides, &plan.strides);
        let ri = broadcast_linear(n, &plan.dims, &right_dims, &right_strides, &plan.strides);
        for k in 0..plan.spatial {
            let i = n * plan.spatial + k;
            os[i] = f(&ls[li * plan.spatial + k], &rs[ri * plan.spatial + k]);
        }
    }
    Ok(())
}

/// Applies `f` to two arrays, writing the output array.
pub fn pointwise2<A, B, U, F, const N: usize, const M: usize>(
    left: &PencilArray<A, N, M>,
    right: &PencilArray<B, N, M>,
    output: &mut PencilArray<U, N, M>,
    f: F,
) -> Result<(), PointwiseError>
where
    F: FnMut(&A, &B) -> U,
{
    pointwise2_views(left.view(), right.view(), output.view_mut(), f)
}

/// Applies `f` in place without cloning the left input.
pub fn pointwise2_in_place<A, B, F, const N: usize, const M: usize>(
    left: &mut PencilArray<A, N, M>,
    right: &PencilArray<B, N, M>,
    f: F,
) -> Result<(), PointwiseError>
where
    F: FnMut(&A, &B) -> A,
{
    pointwise2_in_place_views(left.view_mut(), right.view(), f)
}

/// In-place borrowed-view counterpart of [`pointwise2_in_place`].
/// Only the right extras may expand; validation completes before any callback.
pub fn pointwise2_in_place_views<A, B, F, const N: usize, const M: usize>(
    mut left: PencilArrayViewMut<'_, A, N, M>,
    right: PencilArrayView<'_, B, N, M>,
    mut f: F,
) -> Result<(), PointwiseError>
where
    F: FnMut(&A, &B) -> A,
{
    let left_view = left.as_view();
    validate_two_inputs(
        &left_view,
        &right,
        left.pencil(),
        left.extra_shape().dimensions(),
    )?;
    let (plan, (left_dims, left_strides)) =
        OutputPlan::new(&left_view, left.pencil(), left.extra_shape().dimensions())?;
    let (right_dims, right_strides) = plan.input(left.pencil(), &right)?;
    if plan.spatial == 0 {
        return Ok(());
    }
    let rs = right.as_slice();
    let os = left.as_mut_slice();
    for n in 0..plan.extra_count {
        let li = broadcast_linear(n, &plan.dims, &left_dims, &left_strides, &plan.strides);
        let ri = broadcast_linear(n, &plan.dims, &right_dims, &right_strides, &plan.strides);
        for k in 0..plan.spatial {
            let i = n * plan.spatial + k;
            os[i] = f(&os[li * plan.spatial + k], &rs[ri * plan.spatial + k]);
        }
    }
    Ok(())
}
fn strides(dims: &[usize]) -> Result<Vec<usize>, PointwiseError> {
    let mut out = Vec::new();
    out.try_reserve_exact(dims.len())
        .map_err(|_| ArrayError::AllocationFailed {
            required: dims.len(),
        })?;
    out.resize(dims.len(), 1);
    let mut stride = 1usize;
    for i in (0..dims.len()).rev() {
        out[i] = stride;
        stride = stride
            .checked_mul(dims[i])
            .ok_or(ArrayError::AllocationFailed {
                required: usize::MAX,
            })?;
    }
    Ok(out)
}

/// Repeats the two-input checks in their historical order so shape errors keep
/// priority over the plan's allocations. `OutputPlan` then reuses the same
/// validation helpers while preparing the broadcast metadata.
fn validate_two_inputs<A, B, const N: usize, const M: usize>(
    left: &PencilArrayView<'_, A, N, M>,
    right: &PencilArrayView<'_, B, N, M>,
    output_pencil: &crate::Pencil<N, M>,
    output_dimensions: &[usize],
) -> Result<(), PointwiseError> {
    if !left.pencil().same_layout(right.pencil()) || !left.pencil().same_layout(output_pencil) {
        return Err(PointwiseError::IncompatiblePencils);
    }
    for dimensions in [
        left.extra_shape().dimensions(),
        right.extra_shape().dimensions(),
    ] {
        validate_extents(dimensions, output_dimensions)?;
    }
    Ok(())
}

/// Validates one input's extra dimensions against the output dimensions.
fn validate_extents(input: &[usize], output: &[usize]) -> Result<(), PointwiseError> {
    if input.len() != output.len() {
        return Err(PointwiseError::ExtraRankMismatch {
            expected: output.len(),
            actual: input.len(),
        });
    }
    for (axis, (&extent, &output_extent)) in input.iter().zip(output).enumerate() {
        if extent != output_extent && extent != 1 {
            return Err(PointwiseError::ExtraExtentMismatch {
                axis,
                output: output_extent,
                input: extent,
            });
        }
    }
    Ok(())
}

fn broadcast_linear(
    out: usize,
    output_dims: &[usize],
    dims: &[usize],
    strides: &[usize],
    output_strides: &[usize],
) -> usize {
    output_dims
        .iter()
        .enumerate()
        .map(|(axis, &extent)| {
            let index = (out / output_strides[axis]) % extent;
            if dims[axis] == 1 { 0 } else { index }
        })
        .zip(strides)
        .map(|(i, s)| i * s)
        .sum()
}

/// Validation errors for three-input and runtime-sized pointwise operations.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum MultiInputError {
    /// A many-input operation needs at least one input.
    #[error("pointwise many operations require at least one input")]
    EmptyInputs,
    /// One input failed the ordinary pointwise shape checks.
    #[error("pointwise input {index}: {source}")]
    Input {
        /// Zero-based input position.
        index: usize,
        /// The input's validation failure.
        source: PointwiseError,
    },
    /// A checked temporary allocation failed.
    #[error(transparent)]
    Array(#[from] ArrayError),
}

fn copy_dims(dims: &[usize]) -> Result<Vec<usize>, ArrayError> {
    let mut out = Vec::new();
    out.try_reserve_exact(dims.len())
        .map_err(|_| ArrayError::AllocationFailed {
            required: dims.len(),
        })?;
    out.extend_from_slice(dims);
    Ok(out)
}

type InputStrides = (Vec<usize>, Vec<usize>);

/// The output side of a broadcast pointwise operation.
///
/// Every input shares the output dimensions and strides, so multi-input
/// operations prepare them once; [`Self::input`] then only validates one input
/// and returns its broadcast mapping.
#[derive(Debug)]
struct OutputPlan {
    dims: Vec<usize>,
    strides: Vec<usize>,
    spatial: usize,
    extra_count: usize,
}

impl OutputPlan {
    /// Validates `first` before allocating the shared output metadata, so its
    /// validation error keeps priority over an allocation failure. Returns the
    /// prepared output side together with the first input's broadcast mapping.
    fn new<A, const N: usize, const M: usize>(
        first: &PencilArrayView<'_, A, N, M>,
        output_pencil: &crate::Pencil<N, M>,
        output_dimensions: &[usize],
    ) -> Result<(Self, InputStrides), PointwiseError> {
        if !first.pencil().same_layout(output_pencil) {
            return Err(PointwiseError::IncompatiblePencils);
        }
        validate_extents(first.extra_shape().dimensions(), output_dimensions)?;
        let dims = copy_dims(output_dimensions)?;
        let first_dims = copy_dims(first.extra_shape().dimensions())?;
        let spatial = output_pencil.local_len();
        let extra_count = checked_product(&dims).map_err(ArrayError::from)?;
        spatial
            .checked_mul(extra_count)
            .ok_or(ArrayError::AllocationFailed {
                required: usize::MAX,
            })?;
        // No element is visited for an empty extra shape, so the stride
        // products may overflow without affecting the operation.
        let (strides, first_strides) = if extra_count == 0 {
            (Vec::new(), Vec::new())
        } else {
            (strides(&dims)?, strides(&first_dims)?)
        };
        Ok((
            Self {
                dims,
                strides,
                spatial,
                extra_count,
            },
            (first_dims, first_strides),
        ))
    }

    /// Validates one input against the prepared output side and returns its
    /// broadcast dimensions and strides.
    fn input<A, const N: usize, const M: usize>(
        &self,
        output_pencil: &crate::Pencil<N, M>,
        input: &PencilArrayView<'_, A, N, M>,
    ) -> Result<InputStrides, PointwiseError> {
        if !input.pencil().same_layout(output_pencil) {
            return Err(PointwiseError::IncompatiblePencils);
        }
        let dimensions = input.extra_shape().dimensions();
        validate_extents(dimensions, &self.dims)?;
        let dims = copy_dims(dimensions)?;
        let strides = if self.extra_count == 0 {
            Vec::new()
        } else {
            strides(&dims)?
        };
        Ok((dims, strides))
    }
}

fn many_plan<A, U, const N: usize, const M: usize>(
    inputs: &[PencilArrayView<'_, A, N, M>],
    output: &PencilArrayViewMut<'_, U, N, M>,
) -> Result<(OutputPlan, Vec<InputStrides>), MultiInputError> {
    if inputs.is_empty() {
        return Err(MultiInputError::EmptyInputs);
    }
    let (plan, first_strides) = OutputPlan::new(
        &inputs[0],
        output.pencil(),
        output.extra_shape().dimensions(),
    )
    .map_err(|source| MultiInputError::Input { index: 0, source })?;
    let mut strides = Vec::new();
    strides.try_reserve_exact(inputs.len()).map_err(|_| {
        MultiInputError::Array(ArrayError::AllocationFailed {
            required: inputs.len(),
        })
    })?;
    strides.push(first_strides);
    for (index, input) in inputs.iter().enumerate().skip(1) {
        strides.push(
            plan.input(output.pencil(), input)
                .map_err(|source| MultiInputError::Input { index, source })?,
        );
    }
    Ok((plan, strides))
}

/// Applies a closure to each point of a non-empty homogeneous input slice.
/// Inputs may broadcast singleton extra dimensions; the output is caller-provided.
/// All validation completes before callbacks run, so validation errors leave
/// output unchanged. Callback panics may leave partial writes in physical
/// storage order: extra indices first, then spatial memory order.
pub fn pointwise_many_views<A, U, F, const N: usize, const M: usize>(
    inputs: &[PencilArrayView<'_, A, N, M>],
    mut output: PencilArrayViewMut<'_, U, N, M>,
    mut f: F,
) -> Result<(), MultiInputError>
where
    F: FnMut(&[&A]) -> U,
{
    let (plan, input_strides) = many_plan(inputs, &output)?;
    if plan.spatial == 0 {
        return Ok(());
    }
    let mut refs = Vec::new();
    refs.try_reserve_exact(inputs.len()).map_err(|_| {
        MultiInputError::Array(ArrayError::AllocationFailed {
            required: inputs.len(),
        })
    })?;
    let os = output.as_mut_slice();
    for out_linear in 0..plan.extra_count {
        for k in 0..plan.spatial {
            refs.clear();
            for (input, (dims, strides)) in inputs.iter().zip(&input_strides) {
                let linear = broadcast_linear(out_linear, &plan.dims, dims, strides, &plan.strides);
                refs.push(&input.as_slice()[linear * plan.spatial + k]);
            }
            os[out_linear * plan.spatial + k] = f(&refs);
        }
    }
    Ok(())
}

/// Owned-array convenience wrapper for [`pointwise_many_views`].
/// Validation is atomic; callback panics may leave partial writes in physical
/// storage order.
pub fn pointwise_many<A, U, F, const N: usize, const M: usize>(
    inputs: &[&PencilArray<A, N, M>],
    output: &mut PencilArray<U, N, M>,
    f: F,
) -> Result<(), MultiInputError>
where
    F: FnMut(&[&A]) -> U,
{
    let mut views = Vec::new();
    views.try_reserve_exact(inputs.len()).map_err(|_| {
        MultiInputError::Array(ArrayError::AllocationFailed {
            required: inputs.len(),
        })
    })?;
    views.extend(inputs.iter().map(|input| input.view()));
    pointwise_many_views(&views, output.view_mut(), f)
}

/// In-place many-input pointwise operation. The first input is the destination.
/// All validation completes before callbacks run. Callback panics may leave
/// partial writes in physical storage order: extra indices first, then spatial
/// memory order.
pub fn pointwise_many_in_place_views<A, F, const N: usize, const M: usize>(
    mut left: PencilArrayViewMut<'_, A, N, M>,
    inputs: &[PencilArrayView<'_, A, N, M>],
    mut f: F,
) -> Result<(), MultiInputError>
where
    F: FnMut(&A, &[&A]) -> A,
{
    if inputs.is_empty() {
        return Err(MultiInputError::EmptyInputs);
    }
    let (plan, first_plan) =
        OutputPlan::new(&inputs[0], left.pencil(), left.extra_shape().dimensions())
            .map_err(|source| MultiInputError::Input { index: 0, source })?;
    let mut plans = Vec::new();
    plans.try_reserve_exact(inputs.len()).map_err(|_| {
        MultiInputError::Array(ArrayError::AllocationFailed {
            required: inputs.len(),
        })
    })?;
    plans.push(first_plan);
    for (index, input) in inputs.iter().enumerate().skip(1) {
        plans.push(
            plan.input(left.pencil(), input)
                .map_err(|source| MultiInputError::Input { index, source })?,
        );
    }
    let mut refs = Vec::new();
    refs.try_reserve_exact(inputs.len()).map_err(|_| {
        MultiInputError::Array(ArrayError::AllocationFailed {
            required: inputs.len(),
        })
    })?;
    let os = left.as_mut_slice();
    for out_linear in 0..plan.extra_count {
        for k in 0..plan.spatial {
            refs.clear();
            for (input, (dims, strides)) in inputs.iter().zip(&plans) {
                let i = broadcast_linear(out_linear, &plan.dims, dims, strides, &plan.strides);
                refs.push(&input.as_slice()[i * plan.spatial + k]);
            }
            let i = out_linear * plan.spatial + k;
            let value = f(&os[i], &refs);
            os[i] = value;
        }
    }
    Ok(())
}

/// Owned-array convenience wrapper for [`pointwise_many_in_place_views`].
/// Validation is atomic; callback panics may leave partial writes in physical
/// storage order.
pub fn pointwise_many_in_place<A, F, const N: usize, const M: usize>(
    left: &mut PencilArray<A, N, M>,
    inputs: &[&PencilArray<A, N, M>],
    f: F,
) -> Result<(), MultiInputError>
where
    F: FnMut(&A, &[&A]) -> A,
{
    let mut views = Vec::new();
    views.try_reserve_exact(inputs.len()).map_err(|_| {
        MultiInputError::Array(ArrayError::AllocationFailed {
            required: inputs.len(),
        })
    })?;
    views.extend(inputs.iter().map(|input| input.view()));
    pointwise_many_in_place_views(left.view_mut(), &views, f)
}

/// Three-input heterogeneous pointwise operation writing a supplied output.
pub fn pointwise3_views<A, B, C, U, F, const N: usize, const M: usize>(
    a: PencilArrayView<'_, A, N, M>,
    b: PencilArrayView<'_, B, N, M>,
    c: PencilArrayView<'_, C, N, M>,
    mut output: PencilArrayViewMut<'_, U, N, M>,
    mut f: F,
) -> Result<(), MultiInputError>
where
    F: FnMut(&A, &B, &C) -> U,
{
    let (plan, first_plan) =
        OutputPlan::new(&a, output.pencil(), output.extra_shape().dimensions())
            .map_err(|source| MultiInputError::Input { index: 0, source })?;
    let (a_dims, a_strides) = first_plan;
    let (b_dims, b_strides) = plan
        .input(output.pencil(), &b)
        .map_err(|source| MultiInputError::Input { index: 1, source })?;
    let (c_dims, c_strides) = plan
        .input(output.pencil(), &c)
        .map_err(|source| MultiInputError::Input { index: 2, source })?;
    if plan.spatial == 0 {
        return Ok(());
    }
    let (as_, bs_, cs_, os) = (
        a.as_slice(),
        b.as_slice(),
        c.as_slice(),
        output.as_mut_slice(),
    );
    for n in 0..plan.extra_count {
        let ai = broadcast_linear(n, &plan.dims, &a_dims, &a_strides, &plan.strides);
        let bi = broadcast_linear(n, &plan.dims, &b_dims, &b_strides, &plan.strides);
        let ci = broadcast_linear(n, &plan.dims, &c_dims, &c_strides, &plan.strides);
        for k in 0..plan.spatial {
            os[n * plan.spatial + k] = f(
                &as_[ai * plan.spatial + k],
                &bs_[bi * plan.spatial + k],
                &cs_[ci * plan.spatial + k],
            );
        }
    }
    Ok(())
}

/// Owned-array convenience wrapper for [`pointwise3_views`].
pub fn pointwise3<A, B, C, U, F, const N: usize, const M: usize>(
    a: &PencilArray<A, N, M>,
    b: &PencilArray<B, N, M>,
    c: &PencilArray<C, N, M>,
    out: &mut PencilArray<U, N, M>,
    f: F,
) -> Result<(), MultiInputError>
where
    F: FnMut(&A, &B, &C) -> U,
{
    pointwise3_views(a.view(), b.view(), c.view(), out.view_mut(), f)
}

/// Three-input in-place operation; `left` is also the first callback argument.
pub fn pointwise3_in_place_views<A, B, C, F, const N: usize, const M: usize>(
    mut left: PencilArrayViewMut<'_, A, N, M>,
    b: PencilArrayView<'_, B, N, M>,
    c: PencilArrayView<'_, C, N, M>,
    mut f: F,
) -> Result<(), MultiInputError>
where
    F: FnMut(&A, &B, &C) -> A,
{
    let (plan, first_plan) = OutputPlan::new(&b, left.pencil(), left.extra_shape().dimensions())
        .map_err(|source| MultiInputError::Input { index: 1, source })?;
    let (b_dims, b_strides) = first_plan;
    let (c_dims, c_strides) = plan
        .input(left.pencil(), &c)
        .map_err(|source| MultiInputError::Input { index: 2, source })?;
    if plan.spatial == 0 {
        return Ok(());
    }
    let bs = b.as_slice();
    let cs = c.as_slice();
    let os = left.as_mut_slice();
    for n in 0..plan.extra_count {
        let bi = broadcast_linear(n, &plan.dims, &b_dims, &b_strides, &plan.strides);
        let ci = broadcast_linear(n, &plan.dims, &c_dims, &c_strides, &plan.strides);
        for k in 0..plan.spatial {
            let i = n * plan.spatial + k;
            os[i] = f(
                &os[i],
                &bs[bi * plan.spatial + k],
                &cs[ci * plan.spatial + k],
            );
        }
    }
    Ok(())
}

/// Owned-array convenience wrapper for [`pointwise3_in_place_views`].
pub fn pointwise3_in_place<A, B, C, F, const N: usize, const M: usize>(
    a: &mut PencilArray<A, N, M>,
    b: &PencilArray<B, N, M>,
    c: &PencilArray<C, N, M>,
    f: F,
) -> Result<(), MultiInputError>
where
    F: FnMut(&A, &B, &C) -> A,
{
    pointwise3_in_place_views(a.view_mut(), b.view(), c.view(), f)
}
