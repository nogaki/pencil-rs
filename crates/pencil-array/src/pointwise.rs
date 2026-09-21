use crate::{ArrayError, PencilArray, PencilArrayView, PencilArrayViewMut};
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
    let plan = Plan::new(&left, &right, &output)?;
    if plan.spatial == 0 {
        return Ok(());
    }
    let ls = left.as_slice();
    let rs = right.as_slice();
    let os = output.as_mut_slice();
    for out_linear in 0..plan.extra_count {
        let left_linear = broadcast_linear(
            out_linear,
            &plan.output_dims,
            &plan.left_dims,
            &plan.left_strides,
            &plan.output_strides,
        );
        let right_linear = broadcast_linear(
            out_linear,
            &plan.output_dims,
            &plan.right_dims,
            &plan.right_strides,
            &plan.output_strides,
        );
        let oo = out_linear * plan.spatial;
        let lo = left_linear * plan.spatial;
        let ro = right_linear * plan.spatial;
        for k in 0..plan.spatial {
            os[oo + k] = f(&ls[lo + k], &rs[ro + k]);
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
    let plan = Plan::build_shapes(
        left.pencil(),
        left.extra_shape().dimensions(),
        right.pencil(),
        right.extra_shape().dimensions(),
        left.pencil(),
        left.extra_shape().dimensions(),
    )?;
    if plan.spatial == 0 {
        return Ok(());
    }
    let rs = right.as_slice();
    let os = left.as_mut_slice();
    for out_linear in 0..plan.extra_count {
        let right_linear = broadcast_linear(
            out_linear,
            &plan.output_dims,
            &plan.right_dims,
            &plan.right_strides,
            &plan.output_strides,
        );
        let oo = out_linear * plan.spatial;
        let ro = right_linear * plan.spatial;
        for k in 0..plan.spatial {
            let value = f(&os[oo + k], &rs[ro + k]);
            os[oo + k] = value;
        }
    }
    Ok(())
}

#[derive(Debug)]
struct Plan {
    output_dims: Vec<usize>,
    left_dims: Vec<usize>,
    right_dims: Vec<usize>,
    output_strides: Vec<usize>,
    left_strides: Vec<usize>,
    right_strides: Vec<usize>,
    spatial: usize,
    extra_count: usize,
}
impl Plan {
    fn new<A, B, U, const N: usize, const M: usize>(
        left: &PencilArrayView<'_, A, N, M>,
        right: &PencilArrayView<'_, B, N, M>,
        output: &PencilArrayViewMut<'_, U, N, M>,
    ) -> Result<Self, PointwiseError> {
        Self::build(
            left,
            right,
            output.pencil(),
            output.extra_shape().dimensions(),
        )
    }
    fn build<A, B, const N: usize, const M: usize>(
        left: &PencilArrayView<'_, A, N, M>,
        right: &PencilArrayView<'_, B, N, M>,
        output_pencil: &crate::Pencil<N, M>,
        output_dimensions: &[usize],
    ) -> Result<Self, PointwiseError> {
        Self::build_shapes(
            left.pencil(),
            left.extra_shape().dimensions(),
            right.pencil(),
            right.extra_shape().dimensions(),
            output_pencil,
            output_dimensions,
        )
    }
    fn build_shapes<const N: usize, const M: usize>(
        left_pencil: &crate::Pencil<N, M>,
        left_dimensions: &[usize],
        right_pencil: &crate::Pencil<N, M>,
        right_dimensions: &[usize],
        output_pencil: &crate::Pencil<N, M>,
        output_dimensions: &[usize],
    ) -> Result<Self, PointwiseError> {
        if !left_pencil.same_layout(right_pencil) || !left_pencil.same_layout(output_pencil) {
            return Err(PointwiseError::IncompatiblePencils);
        }
        let rank = output_dimensions.len();
        let dims = [left_dimensions, right_dimensions];
        for shape in dims {
            if shape.len() != rank {
                return Err(PointwiseError::ExtraRankMismatch {
                    expected: rank,
                    actual: shape.len(),
                });
            }
            for (axis, (&input, &out)) in shape.iter().zip(output_dimensions).enumerate() {
                if input != out && input != 1 {
                    return Err(PointwiseError::ExtraExtentMismatch {
                        axis,
                        output: out,
                        input,
                    });
                }
            }
        }
        let copy = |dims: &[usize]| -> Result<Vec<usize>, PointwiseError> {
            let mut out = Vec::new();
            out.try_reserve_exact(dims.len())
                .map_err(|_| ArrayError::AllocationFailed {
                    required: dims.len(),
                })?;
            out.extend_from_slice(dims);
            Ok(out)
        };
        let output_dims = copy(output_dimensions)?;
        let left_dims = copy(left_dimensions)?;
        let right_dims = copy(right_dimensions)?;
        let spatial = left_pencil.local_len();
        let output_strides = strides(&output_dims)?;
        let extra_count = output_dims
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b))
            .ok_or(ArrayError::AllocationFailed {
                required: usize::MAX,
            })?;
        spatial
            .checked_mul(extra_count)
            .ok_or(ArrayError::AllocationFailed {
                required: usize::MAX,
            })?;
        Ok(Self {
            output_strides,
            left_strides: strides(&left_dims)?,
            right_strides: strides(&right_dims)?,
            output_dims,
            left_dims,
            right_dims,
            spatial,
            extra_count,
        })
    }
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
