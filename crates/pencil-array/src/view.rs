use crate::{
    ArrayError, ExtraShape, GeometryError, LocalGrid, LocalGridError, Pencil,
    checked::{check_storage_len, local_storage_len},
    geometry::column_major_offset,
};

pub(crate) trait LocalArrayLayout<T, const N: usize, const M: usize> {
    fn pencil(&self) -> &Pencil<N, M>;
    fn extra_shape(&self) -> &ExtraShape;
    fn as_slice(&self) -> &[T];

    fn local_spatial_shape(&self) -> [usize; N] {
        self.pencil().local_shape_logical()
    }

    fn local_spatial_memory_shape(&self) -> [usize; N] {
        self.pencil().local_shape_memory()
    }

    fn logical_shape(&self) -> Vec<usize> {
        let mut shape = Vec::with_capacity(self.extra_shape().dimensions().len() + N);
        shape.extend_from_slice(&self.local_spatial_shape());
        shape.extend_from_slice(self.extra_shape().dimensions());
        shape
    }

    fn memory_shape(&self) -> Vec<usize> {
        let mut shape = Vec::with_capacity(self.extra_shape().dimensions().len() + N);
        shape.extend_from_slice(&self.local_spatial_memory_shape());
        shape.extend_from_slice(self.extra_shape().dimensions());
        shape
    }

    fn local_offset(
        &self,
        extra_indices: &[usize],
        spatial_indices: [usize; N],
    ) -> Result<usize, ArrayError> {
        let extra_rank = self.extra_shape().dimensions().len();
        if extra_indices.len() != extra_rank {
            return Err(ArrayError::ExtraIndexRankMismatch {
                required: extra_rank,
                actual: extra_indices.len(),
            });
        }

        let extra_offset = column_major_offset(self.extra_shape().dimensions(), extra_indices)?;
        let memory_shape = self.pencil().local_shape_memory();
        let memory_indices = self.pencil().permutation().permute(spatial_indices);
        let spatial_offset = column_major_offset(&memory_shape, &memory_indices)?;
        extra_offset
            .checked_mul(self.pencil().local_len())
            .and_then(|base| base.checked_add(spatial_offset))
            .ok_or_else(|| GeometryError::SizeOverflow.into())
    }

    fn global_offset(
        &self,
        extra_indices: &[usize],
        global_spatial_indices: [usize; N],
    ) -> Option<usize> {
        let mut local_spatial_indices = [0; N];
        for axis in 0..N {
            let range = &self.pencil().local_ranges()[axis];
            if !range.contains(&global_spatial_indices[axis]) {
                return None;
            }
            local_spatial_indices[axis] = global_spatial_indices[axis] - range.start;
        }
        self.local_offset(extra_indices, local_spatial_indices).ok()
    }

    fn local_grid<'a, C>(
        &self,
        coordinates: [&'a [C]; N],
    ) -> Result<LocalGrid<'a, C, N>, LocalGridError> {
        self.pencil().local_grid(coordinates)
    }
}

#[derive(Debug)]
/// A read-only local array view borrowing its layout metadata and storage.
pub struct PencilArrayView<'a, T, const N: usize, const M: usize> {
    pencil: &'a Pencil<N, M>,
    extra_shape: &'a ExtraShape,
    storage: &'a [T],
}

impl<'a, T, const N: usize, const M: usize> PencilArrayView<'a, T, N, M> {
    /// Borrows external contiguous storage without copying or allocating.
    ///
    /// Storage must be compact column-major physical order: `[permuted
    /// spatial..., extra...]`, with the first axis varying fastest. Spatial axes
    /// follow `pencil.permutation()`; extra axes follow `extra_shape.dimensions()`.
    /// The slice must contain the full local buffer, exactly
    /// `extra_shape.element_count() * pencil.local_len()` elements, not an
    /// arbitrary subset. A wrong length returns `ArrayError::StorageLengthMismatch`;
    /// an overflowing length returns the existing checked-size error.
    ///
    /// The pencil, extra shape, and storage are borrowed for the view's lifetime.
    /// No ownership is transferred and no MPI operation is performed.
    pub fn from_slice(
        pencil: &'a Pencil<N, M>,
        extra_shape: &'a ExtraShape,
        storage: &'a [T],
    ) -> Result<Self, ArrayError> {
        Self::new(pencil, extra_shape, storage)
    }

    pub(crate) fn new(
        pencil: &'a Pencil<N, M>,
        extra_shape: &'a ExtraShape,
        storage: &'a [T],
    ) -> Result<Self, ArrayError> {
        validate_storage_len(pencil, extra_shape, storage.len())?;
        Ok(Self {
            pencil,
            extra_shape,
            storage,
        })
    }

    /// Returns the view's pencil layout.
    pub fn pencil(&self) -> &Pencil<N, M> {
        LocalArrayLayout::pencil(self)
    }

    /// Returns the view's extra shape.
    pub fn extra_shape(&self) -> &ExtraShape {
        LocalArrayLayout::extra_shape(self)
    }

    /// Returns the local spatial shape in logical-axis order.
    pub fn local_spatial_shape(&self) -> [usize; N] {
        LocalArrayLayout::local_spatial_shape(self)
    }

    /// Returns the local spatial shape in memory-axis order.
    pub fn local_spatial_memory_shape(&self) -> [usize; N] {
        LocalArrayLayout::local_spatial_memory_shape(self)
    }

    /// Returns `[spatial..., extra...]` in logical-axis order.
    pub fn logical_shape(&self) -> Vec<usize> {
        LocalArrayLayout::logical_shape(self)
    }

    /// Returns `[permuted spatial..., extra...]` in column-major memory order.
    pub fn memory_shape(&self) -> Vec<usize> {
        LocalArrayLayout::memory_shape(self)
    }

    /// Returns the number of elements in the borrowed buffer.
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    /// Returns whether the borrowed buffer has no elements.
    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    /// Returns the borrowed column-major buffer.
    pub fn as_slice(&self) -> &[T] {
        LocalArrayLayout::as_slice(self)
    }

    /// Returns an element at local logical indices, or `None` if invalid.
    pub fn get_local(&self, extra_indices: &[usize], spatial_indices: [usize; N]) -> Option<&T> {
        let offset = LocalArrayLayout::local_offset(self, extra_indices, spatial_indices).ok()?;
        self.storage.get(offset)
    }

    /// Returns a shared reference at locally owned global spatial indices.
    ///
    /// Global indices owned by another rank, outside the global shape, or
    /// outside this rank's local range return `None`. This accessor is local
    /// and performs no MPI operation.
    pub fn get_global(
        &self,
        extra_indices: &[usize],
        global_spatial_indices: [usize; N],
    ) -> Option<&T> {
        let offset = LocalArrayLayout::global_offset(self, extra_indices, global_spatial_indices)?;
        self.storage.get(offset)
    }

    /// Borrows caller-provided global coordinate axes for this view's local grid.
    pub fn local_grid<'c, C>(
        &self,
        coordinates: [&'c [C]; N],
    ) -> Result<LocalGrid<'c, C, N>, LocalGridError> {
        LocalArrayLayout::local_grid(self, coordinates)
    }
}

impl<T, const N: usize, const M: usize> LocalArrayLayout<T, N, M> for PencilArrayView<'_, T, N, M> {
    fn pencil(&self) -> &Pencil<N, M> {
        self.pencil
    }

    fn extra_shape(&self) -> &ExtraShape {
        self.extra_shape
    }

    fn as_slice(&self) -> &[T] {
        self.storage
    }
}

#[derive(Debug)]
/// An exclusive local array view borrowing its layout metadata and storage.
pub struct PencilArrayViewMut<'a, T, const N: usize, const M: usize> {
    pencil: &'a Pencil<N, M>,
    extra_shape: &'a ExtraShape,
    storage: &'a mut [T],
}

impl<'a, T, const N: usize, const M: usize> PencilArrayViewMut<'a, T, N, M> {
    /// Exclusively borrows external contiguous storage without copying or allocating.
    ///
    /// Storage must be compact column-major physical order: `[permuted
    /// spatial..., extra...]`, with the first axis varying fastest. Spatial axes
    /// follow `pencil.permutation()`; extra axes follow `extra_shape.dimensions()`.
    /// The slice must contain the full local buffer, exactly
    /// `extra_shape.element_count() * pencil.local_len()` elements, not an
    /// arbitrary subset. A wrong length returns `ArrayError::StorageLengthMismatch`;
    /// an overflowing length returns the existing checked-size error.
    ///
    /// The pencil and extra shape are shared borrows; storage is exclusively
    /// borrowed for the view's lifetime. No ownership is transferred and no MPI
    /// operation is performed. Writes update the caller's storage directly.
    pub fn from_slice_mut(
        pencil: &'a Pencil<N, M>,
        extra_shape: &'a ExtraShape,
        storage: &'a mut [T],
    ) -> Result<Self, ArrayError> {
        Self::new(pencil, extra_shape, storage)
    }

    pub(crate) fn new(
        pencil: &'a Pencil<N, M>,
        extra_shape: &'a ExtraShape,
        storage: &'a mut [T],
    ) -> Result<Self, ArrayError> {
        validate_storage_len(pencil, extra_shape, storage.len())?;
        Ok(Self {
            pencil,
            extra_shape,
            storage,
        })
    }

    /// Returns the view's pencil layout.
    pub fn pencil(&self) -> &Pencil<N, M> {
        LocalArrayLayout::pencil(self)
    }

    /// Returns the view's extra shape.
    pub fn extra_shape(&self) -> &ExtraShape {
        LocalArrayLayout::extra_shape(self)
    }

    /// Returns the local spatial shape in logical-axis order.
    pub fn local_spatial_shape(&self) -> [usize; N] {
        LocalArrayLayout::local_spatial_shape(self)
    }

    /// Returns the local spatial shape in memory-axis order.
    pub fn local_spatial_memory_shape(&self) -> [usize; N] {
        LocalArrayLayout::local_spatial_memory_shape(self)
    }

    /// Returns `[spatial..., extra...]` in logical-axis order.
    pub fn logical_shape(&self) -> Vec<usize> {
        LocalArrayLayout::logical_shape(self)
    }

    /// Returns `[permuted spatial..., extra...]` in column-major memory order.
    pub fn memory_shape(&self) -> Vec<usize> {
        LocalArrayLayout::memory_shape(self)
    }

    /// Returns the number of elements in the borrowed buffer.
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    /// Returns whether the borrowed buffer has no elements.
    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    /// Returns a shared borrow of the column-major buffer.
    pub fn as_slice(&self) -> &[T] {
        LocalArrayLayout::as_slice(self)
    }

    /// Reborrows this view immutably, wrapping the existing fields.
    pub(crate) fn as_view(&self) -> PencilArrayView<'_, T, N, M> {
        PencilArrayView {
            pencil: self.pencil,
            extra_shape: self.extra_shape,
            storage: &*self.storage,
        }
    }

    /// Returns an exclusive borrow of the column-major buffer.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.storage
    }

    /// Returns an element at local logical indices, or `None` if invalid.
    pub fn get_local(&self, extra_indices: &[usize], spatial_indices: [usize; N]) -> Option<&T> {
        let offset = LocalArrayLayout::local_offset(self, extra_indices, spatial_indices).ok()?;
        self.storage.get(offset)
    }

    /// Returns a mutable element at local logical indices, or `None` if invalid.
    pub fn get_local_mut(
        &mut self,
        extra_indices: &[usize],
        spatial_indices: [usize; N],
    ) -> Option<&mut T> {
        let offset = LocalArrayLayout::local_offset(self, extra_indices, spatial_indices).ok()?;
        self.storage.get_mut(offset)
    }

    /// Returns a shared reference at locally owned global spatial indices.
    ///
    /// Global indices owned by another rank, outside the global shape, or
    /// outside this rank's local range return `None`. This accessor is local
    /// and performs no MPI operation.
    pub fn get_global(
        &self,
        extra_indices: &[usize],
        global_spatial_indices: [usize; N],
    ) -> Option<&T> {
        let offset = LocalArrayLayout::global_offset(self, extra_indices, global_spatial_indices)?;
        self.storage.get(offset)
    }

    /// Returns a mutable reference at locally owned global spatial indices.
    ///
    /// The lookup is local-only and performs no MPI operation.
    pub fn get_global_mut(
        &mut self,
        extra_indices: &[usize],
        global_spatial_indices: [usize; N],
    ) -> Option<&mut T> {
        let offset = LocalArrayLayout::global_offset(self, extra_indices, global_spatial_indices)?;
        self.storage.get_mut(offset)
    }

    /// Borrows caller-provided global coordinate axes for this view's local grid.
    pub fn local_grid<'c, C>(
        &self,
        coordinates: [&'c [C]; N],
    ) -> Result<LocalGrid<'c, C, N>, LocalGridError> {
        LocalArrayLayout::local_grid(self, coordinates)
    }
}

impl<T, const N: usize, const M: usize> LocalArrayLayout<T, N, M>
    for PencilArrayViewMut<'_, T, N, M>
{
    fn pencil(&self) -> &Pencil<N, M> {
        self.pencil
    }

    fn extra_shape(&self) -> &ExtraShape {
        self.extra_shape
    }

    fn as_slice(&self) -> &[T] {
        self.storage
    }
}

fn validate_storage_len<const N: usize, const M: usize>(
    pencil: &Pencil<N, M>,
    extra_shape: &ExtraShape,
    actual: usize,
) -> Result<(), ArrayError> {
    let required = local_storage_len(pencil.local_len(), extra_shape.element_count())?;
    check_storage_len(required, actual)?;
    Ok(())
}

#[cfg(test)]
#[path = "tests/view.rs"]
mod tests;
