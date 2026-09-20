use std::sync::Arc;

use crate::{
    ArrayError, ExtraShape, LocalGrid, LocalGridError, Pencil, PencilArrayView, PencilArrayViewMut,
    checked::checked_product, view::LocalArrayLayout,
};

#[derive(Debug)]
/// An owning local buffer associated with one immutable [`Pencil`] layout.
pub struct PencilArray<T, const N: usize, const M: usize> {
    pencil: Arc<Pencil<N, M>>,
    extra_shape: ExtraShape,
    storage: Vec<T>,
}

impl<T, const N: usize, const M: usize> PencilArray<T, N, M> {
    /// Takes ownership of `storage` after validating its exact required length.
    pub fn from_vec(
        pencil: Arc<Pencil<N, M>>,
        extra_shape: ExtraShape,
        storage: Vec<T>,
    ) -> Result<Self, ArrayError> {
        let required = required_len(&pencil, &extra_shape)?;
        if storage.len() != required {
            return Err(ArrayError::StorageLengthMismatch {
                required,
                actual: storage.len(),
            });
        }
        Ok(Self {
            pencil,
            extra_shape,
            storage,
        })
    }

    /// Allocates the local buffer and fills every element with a clone of `value`.
    pub fn from_elem(
        pencil: Arc<Pencil<N, M>>,
        extra_shape: ExtraShape,
        value: T,
    ) -> Result<Self, ArrayError>
    where
        T: Clone,
    {
        let required = required_len(&pencil, &extra_shape)?;
        let storage = allocate_with(required, || value.clone())?;
        Self::from_vec(pencil, extra_shape, storage)
    }

    /// Allocates the local buffer and initializes its elements in row-major order.
    pub fn from_fn(
        pencil: Arc<Pencil<N, M>>,
        extra_shape: ExtraShape,
        f: impl FnMut() -> T,
    ) -> Result<Self, ArrayError> {
        let required = required_len(&pencil, &extra_shape)?;
        let storage = allocate_with(required, f)?;
        Self::from_vec(pencil, extra_shape, storage)
    }

    /// Returns the complete local row-major buffer.
    pub fn as_slice(&self) -> &[T] {
        LocalArrayLayout::as_slice(self)
    }

    /// Returns the array's shared pencil layout.
    pub fn pencil(&self) -> &Arc<Pencil<N, M>> {
        &self.pencil
    }

    /// Returns the undistributed dimensions preceding the spatial dimensions.
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

    /// Returns `[extra..., spatial...]` in logical-axis order.
    pub fn logical_shape(&self) -> Vec<usize> {
        LocalArrayLayout::logical_shape(self)
    }

    /// Returns `[extra..., permuted spatial...]` in row-major memory order.
    pub fn memory_shape(&self) -> Vec<usize> {
        LocalArrayLayout::memory_shape(self)
    }

    /// Returns the number of elements in the local buffer.
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    /// Returns whether the local buffer has no elements.
    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    /// Returns the complete mutable local row-major buffer.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.storage
    }

    /// Returns a shared reference at local logical indices, or `None` if invalid.
    pub fn get_local(&self, extra_indices: &[usize], spatial_indices: [usize; N]) -> Option<&T> {
        let offset = LocalArrayLayout::local_offset(self, extra_indices, spatial_indices).ok()?;
        self.storage.get(offset)
    }

    /// Returns a mutable reference at local logical indices, or `None` if invalid.
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

    /// Borrows caller-provided global coordinate axes for this array's local grid.
    pub fn local_grid<'a, C>(
        &self,
        coordinates: [&'a [C]; N],
    ) -> Result<LocalGrid<'a, C, N>, LocalGridError> {
        LocalArrayLayout::local_grid(self, coordinates)
    }

    /// Borrows the array as a read-only view tied to this owner.
    pub fn view(&self) -> PencilArrayView<'_, T, N, M> {
        PencilArrayView::new(&self.pencil, &self.extra_shape, &self.storage)
            .expect("PencilArray storage length was validated at construction")
    }

    /// Borrows the array as an exclusive view tied to this owner.
    pub fn view_mut(&mut self) -> PencilArrayViewMut<'_, T, N, M> {
        PencilArrayViewMut::new(&self.pencil, &self.extra_shape, &mut self.storage)
            .expect("PencilArray storage length was validated at construction")
    }
}

impl<T, const N: usize, const M: usize> LocalArrayLayout<T, N, M> for PencilArray<T, N, M> {
    fn pencil(&self) -> &Pencil<N, M> {
        &self.pencil
    }

    fn extra_shape(&self) -> &ExtraShape {
        &self.extra_shape
    }

    fn as_slice(&self) -> &[T] {
        &self.storage
    }
}

fn allocate_with<T>(required: usize, f: impl FnMut() -> T) -> Result<Vec<T>, ArrayError> {
    let mut storage = Vec::new();
    storage
        .try_reserve_exact(required)
        .map_err(|_| ArrayError::AllocationFailed { required })?;
    storage.extend(std::iter::repeat_with(f).take(required));
    Ok(storage)
}

fn required_len<const N: usize, const M: usize>(
    pencil: &Pencil<N, M>,
    extra_shape: &ExtraShape,
) -> Result<usize, ArrayError> {
    Ok(checked_product(&[
        pencil.local_len(),
        extra_shape.element_count(),
    ])?)
}
