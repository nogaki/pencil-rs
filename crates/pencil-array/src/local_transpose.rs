use std::sync::Arc;

use thiserror::Error;

use crate::{
    ArrayError, ManyPencilArray, Pencil, PencilArrayView, PencilArrayViewMut,
    checked::checked_product, geometry::row_major_offset,
};

#[derive(Debug, Error)]
/// Errors returned by a local transpose operation.
pub enum LocalTransposeError {
    /// The source and destination pencils do not share one distribution.
    #[error("source and destination distributions are incompatible")]
    IncompatibleDistribution,

    /// The supplied source view does not match the plan's source layout.
    #[error("source view or active layout does not match the plan")]
    SourceLayoutMismatch,

    /// The supplied destination view does not match the plan's destination layout.
    #[error("destination view or registered layout does not match the plan")]
    DestinationLayoutMismatch,

    /// The source and destination have different extra dimensions.
    #[error("source and destination extra shapes differ")]
    ExtraShapeMismatch,

    /// The supplied scratch storage is too small for an in-place operation.
    #[error("scratch capacity {actual} is less than required {required}")]
    ScratchTooSmall {
        /// The number of elements required by the operation.
        required: usize,
        /// The scratch vector's capacity.
        actual: usize,
    },

    /// An array layout or checked-arithmetic validation failed.
    #[error(transparent)]
    Array(#[from] ArrayError),
}

#[derive(Debug)]
/// A process-local permutation between two compatible pencil layouts.
pub struct LocalTransposePlan<const N: usize, const M: usize> {
    source: Arc<Pencil<N, M>>,
    destination: Arc<Pencil<N, M>>,
}

impl<const N: usize, const M: usize> LocalTransposePlan<N, M> {
    /// Creates a plan for two layouts with the same topology, shape, and
    /// ordered decomposition.
    ///
    /// The permutations may differ. Construction and execution are
    /// process-local and noncollective; the topology itself still has its
    /// existing collective-construction contract.
    pub fn new(
        source: Arc<Pencil<N, M>>,
        destination: Arc<Pencil<N, M>>,
    ) -> Result<Self, LocalTransposeError> {
        if !source.same_distribution(destination.as_ref()) {
            return Err(LocalTransposeError::IncompatibleDistribution);
        }

        Ok(Self {
            source,
            destination,
        })
    }

    /// Copies logical values from `source` into the destination memory order.
    ///
    /// The arguments must be views whose pencils match the plan's source and
    /// destination layouts, and their [`crate::ExtraShape`] values must be exactly
    /// equal. The view constructors validate their storage lengths. All
    /// ordinary validation, including checked offset validation, completes
    /// before the first destination write and returns a
    /// [`LocalTransposeError`] on failure.
    ///
    /// This operation is process-local and noncollective: it does not
    /// communicate with or require any other MPI rank. A panic from `T::clone`
    /// leaves `source` unchanged, but may leave the separate destination
    /// partially written.
    pub fn execute_views<T: Clone>(
        &self,
        source: PencilArrayView<'_, T, N, M>,
        mut destination: PencilArrayViewMut<'_, T, N, M>,
    ) -> Result<(), LocalTransposeError> {
        if !source.pencil().same_layout(self.source.as_ref()) {
            return Err(LocalTransposeError::SourceLayoutMismatch);
        }
        if !destination.pencil().same_layout(self.destination.as_ref()) {
            return Err(LocalTransposeError::DestinationLayoutMismatch);
        }
        if source.extra_shape() != destination.extra_shape() {
            return Err(LocalTransposeError::ExtraShapeMismatch);
        }

        let required = checked_product(&[
            self.source.local_len(),
            source.extra_shape().element_count(),
        ])
        .map_err(ArrayError::from)?;
        if source.len() != required {
            return Err(ArrayError::StorageLengthMismatch {
                required,
                actual: source.len(),
            }
            .into());
        }
        if destination.len() != required {
            return Err(ArrayError::StorageLengthMismatch {
                required,
                actual: destination.len(),
            }
            .into());
        }

        // Resolve every checked offset before borrowing the destination for
        // writes, so an ordinary validation failure cannot partially write it.
        for linear in 0..required {
            let (source_offset, destination_offset) =
                logical_offsets(self.source.as_ref(), self.destination.as_ref(), linear)?;
            if source_offset >= source.len() || destination_offset >= destination.len() {
                return Err(ArrayError::StorageLengthMismatch {
                    required,
                    actual: source.len().min(destination.len()),
                }
                .into());
            }
        }

        // ponytail: the two-stage checked element-wise path is the baseline; optimize only after measurement.
        let source_storage = source.as_slice();
        let destination_storage = destination.as_mut_slice();
        for linear in 0..required {
            let (source_offset, destination_offset) =
                logical_offsets(self.source.as_ref(), self.destination.as_ref(), linear)?;
            destination_storage[destination_offset] = source_storage[source_offset].clone();
        }

        Ok(())
    }

    /// Permutes the active layout in the array's existing storage.
    ///
    /// This operation is process-local and noncollective: it does not
    /// communicate with or require any other MPI rank. The array's active
    /// layout must match the plan source, and the destination must be
    /// registered. Public [`ManyPencilArray`] constructors validate the
    /// checked storage length, so an array with an overflowing required size
    /// cannot be passed here.
    ///
    /// The required scratch length is
    /// `source.local_len() * array.extra_shape().element_count()` with checked
    /// arithmetic. `scratch.capacity()` must be at least that value. The
    /// existing allocation is reused without growing it: `scratch` is cleared
    /// before staging and, on success, contains the source's physical storage
    /// order with length `required`.
    ///
    /// Ordinary validation errors occur before the array or scratch is
    /// changed. Source values are cloned into scratch before the array is
    /// poisoned. A panic from `scratch.clear()` (including `T::drop`) or from
    /// staging leaves the source layout valid and its data unchanged. After a
    /// clear panic scratch may be partially cleared; after a staging panic it
    /// may contain a partial staging result. Once writing begins the
    /// array is poisoned, and any panic from `clone_from` or dropping its old
    /// values leaves it poisoned for recovery by `overwrite_with`.
    pub fn execute_in_place<T: Clone>(
        &self,
        array: &mut ManyPencilArray<T, N, M>,
        scratch: &mut Vec<T>,
    ) -> Result<(), LocalTransposeError> {
        let source_index = array.active_index()?;
        if !array.pencils()[source_index].same_layout(self.source.as_ref()) {
            return Err(LocalTransposeError::SourceLayoutMismatch);
        }
        let destination_index = array
            .find_layout(self.destination.as_ref())
            .ok_or(LocalTransposeError::DestinationLayoutMismatch)?;
        let required =
            checked_product(&[self.source.local_len(), array.extra_shape().element_count()])
                .map_err(ArrayError::from)?;
        if scratch.capacity() < required {
            return Err(LocalTransposeError::ScratchTooSmall {
                required,
                actual: scratch.capacity(),
            });
        }

        {
            let source = array.active_view()?;
            if source.len() != required {
                return Err(ArrayError::StorageLengthMismatch {
                    required,
                    actual: source.len(),
                }
                .into());
            }
            for linear in 0..required {
                let (source_offset, destination_offset) =
                    logical_offsets(self.source.as_ref(), self.destination.as_ref(), linear)?;
                if source_offset >= source.len() || destination_offset >= required {
                    return Err(ArrayError::StorageLengthMismatch {
                        required,
                        actual: source.len().min(required),
                    }
                    .into());
                }
            }

            scratch.clear();
            for value in source.as_slice() {
                scratch.push(value.clone());
            }
        }

        let mut guard = array.begin_in_place_write()?;
        let storage = guard.storage_mut();
        for linear in 0..required {
            let (source_offset, destination_offset) =
                logical_offsets(self.source.as_ref(), self.destination.as_ref(), linear)?;
            storage[destination_offset].clone_from(&scratch[source_offset]);
        }
        guard.commit(destination_index)?;
        Ok(())
    }
}

fn logical_offsets<const N: usize, const M: usize>(
    source: &Pencil<N, M>,
    destination: &Pencil<N, M>,
    linear: usize,
) -> Result<(usize, usize), ArrayError> {
    let local_len = source.local_len();
    let spatial_linear = linear % local_len;
    let extra_linear = linear / local_len;
    let mut spatial_indices = [0; N];
    let mut remainder = spatial_linear;
    let logical_shape = source.local_shape_logical();

    for axis in (0..N).rev() {
        let extent = logical_shape[axis];
        spatial_indices[axis] = remainder % extent;
        remainder /= extent;
    }

    let source_memory_indices = source.permutation().permute(spatial_indices);
    let destination_memory_indices = destination.permutation().permute(spatial_indices);
    let source_spatial_offset =
        row_major_offset(&source.local_shape_memory(), &source_memory_indices)
            .map_err(ArrayError::from)?;
    let destination_spatial_offset = row_major_offset(
        &destination.local_shape_memory(),
        &destination_memory_indices,
    )
    .map_err(ArrayError::from)?;

    let source_offset = extra_linear
        .checked_mul(source.local_len())
        .and_then(|base| base.checked_add(source_spatial_offset))
        .ok_or(ArrayError::Geometry(crate::GeometryError::SizeOverflow))?;
    let destination_offset = extra_linear
        .checked_mul(destination.local_len())
        .and_then(|base| base.checked_add(destination_spatial_offset))
        .ok_or(ArrayError::Geometry(crate::GeometryError::SizeOverflow))?;

    Ok((source_offset, destination_offset))
}
