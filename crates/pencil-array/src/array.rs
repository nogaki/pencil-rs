use std::sync::Arc;

use crate::{
    ArrayError, ExtraShape, Pencil, PencilArrayView, PencilArrayViewMut, checked::checked_product,
    view::LocalArrayLayout,
};

#[derive(Debug)]
pub struct PencilArray<T, const N: usize, const M: usize> {
    pencil: Arc<Pencil<N, M>>,
    extra_shape: ExtraShape,
    storage: Vec<T>,
}

impl<T, const N: usize, const M: usize> PencilArray<T, N, M> {
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

    pub fn from_fn(
        pencil: Arc<Pencil<N, M>>,
        extra_shape: ExtraShape,
        f: impl FnMut() -> T,
    ) -> Result<Self, ArrayError> {
        let required = required_len(&pencil, &extra_shape)?;
        let storage = allocate_with(required, f)?;
        Self::from_vec(pencil, extra_shape, storage)
    }

    pub fn as_slice(&self) -> &[T] {
        LocalArrayLayout::as_slice(self)
    }

    pub fn pencil(&self) -> &Arc<Pencil<N, M>> {
        &self.pencil
    }

    pub fn extra_shape(&self) -> &ExtraShape {
        LocalArrayLayout::extra_shape(self)
    }

    pub fn local_spatial_shape(&self) -> [usize; N] {
        LocalArrayLayout::local_spatial_shape(self)
    }

    pub fn local_spatial_memory_shape(&self) -> [usize; N] {
        LocalArrayLayout::local_spatial_memory_shape(self)
    }

    pub fn logical_shape(&self) -> Vec<usize> {
        LocalArrayLayout::logical_shape(self)
    }

    pub fn memory_shape(&self) -> Vec<usize> {
        LocalArrayLayout::memory_shape(self)
    }

    pub fn len(&self) -> usize {
        self.storage.len()
    }

    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.storage
    }

    pub fn get_local(&self, extra_indices: &[usize], spatial_indices: [usize; N]) -> Option<&T> {
        let offset = LocalArrayLayout::local_offset(self, extra_indices, spatial_indices).ok()?;
        self.storage.get(offset)
    }

    pub fn get_local_mut(
        &mut self,
        extra_indices: &[usize],
        spatial_indices: [usize; N],
    ) -> Option<&mut T> {
        let offset = LocalArrayLayout::local_offset(self, extra_indices, spatial_indices).ok()?;
        self.storage.get_mut(offset)
    }

    pub fn view(&self) -> PencilArrayView<'_, T, N, M> {
        PencilArrayView::new(&self.pencil, &self.extra_shape, &self.storage)
            .expect("PencilArray storage length was validated at construction")
    }

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
