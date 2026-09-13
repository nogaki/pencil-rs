use crate::{ArrayError, ExtraShape, Pencil, checked::checked_product, geometry::row_major_offset};

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
        shape.extend_from_slice(self.extra_shape().dimensions());
        shape.extend_from_slice(&self.local_spatial_shape());
        shape
    }

    fn memory_shape(&self) -> Vec<usize> {
        let mut shape = Vec::with_capacity(self.extra_shape().dimensions().len() + N);
        shape.extend_from_slice(self.extra_shape().dimensions());
        shape.extend_from_slice(&self.local_spatial_memory_shape());
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

        let mut memory_indices = Vec::with_capacity(extra_rank + N);
        memory_indices.extend_from_slice(extra_indices);
        memory_indices.extend(
            self.pencil()
                .permutation()
                .axes()
                .iter()
                .map(|axis| spatial_indices[axis.index()]),
        );
        Ok(row_major_offset(&self.memory_shape(), &memory_indices)?)
    }
}

#[derive(Debug)]
pub struct PencilArrayView<'a, T, const N: usize, const M: usize> {
    pencil: &'a Pencil<N, M>,
    extra_shape: &'a ExtraShape,
    storage: &'a [T],
}

impl<'a, T, const N: usize, const M: usize> PencilArrayView<'a, T, N, M> {
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

    pub fn pencil(&self) -> &Pencil<N, M> {
        LocalArrayLayout::pencil(self)
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

    pub fn as_slice(&self) -> &[T] {
        LocalArrayLayout::as_slice(self)
    }

    pub fn get_local(&self, extra_indices: &[usize], spatial_indices: [usize; N]) -> Option<&T> {
        let offset = LocalArrayLayout::local_offset(self, extra_indices, spatial_indices).ok()?;
        self.storage.get(offset)
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
pub struct PencilArrayViewMut<'a, T, const N: usize, const M: usize> {
    pencil: &'a Pencil<N, M>,
    extra_shape: &'a ExtraShape,
    storage: &'a mut [T],
}

impl<'a, T, const N: usize, const M: usize> PencilArrayViewMut<'a, T, N, M> {
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

    pub fn pencil(&self) -> &Pencil<N, M> {
        LocalArrayLayout::pencil(self)
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

    pub fn as_slice(&self) -> &[T] {
        LocalArrayLayout::as_slice(self)
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.storage
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
    let required = checked_product(&[extra_shape.element_count(), pencil.local_len()])?;
    if actual != required {
        return Err(ArrayError::StorageLengthMismatch { required, actual });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use mpi::traits::*;

    use super::{PencilArrayView, PencilArrayViewMut};
    use crate::{ArrayError, AxisPermutation, ExtraShape, MpiTopology, Pencil};

    #[test]
    fn views_validate_storage_and_expose_permuted_local_layout() {
        let universe = mpi::initialize().expect("MPI initialization failed");
        let world = universe.world();
        assert_eq!(world.size(), 1, "run this unit test with one MPI rank");

        let topology = MpiTopology::<1>::new(&world, [1]).unwrap();
        let pencil = Pencil::<2, 1>::new_permuted(
            Arc::clone(&topology),
            [2, 3],
            [0],
            AxisPermutation::new([1, 0]).unwrap(),
        )
        .unwrap();
        let extra_shape = ExtraShape::new([2]).unwrap();

        assert_eq!(
            PencilArrayView::new(&pencil, &extra_shape, &[0; 11]).unwrap_err(),
            ArrayError::StorageLengthMismatch {
                required: 12,
                actual: 11,
            },
        );
        assert_eq!(
            PencilArrayView::new(&pencil, &extra_shape, &[0; 13]).unwrap_err(),
            ArrayError::StorageLengthMismatch {
                required: 12,
                actual: 13,
            },
        );
        let mut short_storage = [0; 11];
        assert_eq!(
            PencilArrayViewMut::new(&pencil, &extra_shape, &mut short_storage).unwrap_err(),
            ArrayError::StorageLengthMismatch {
                required: 12,
                actual: 11,
            },
        );

        let storage: Vec<_> = (0..12).collect();
        let view = PencilArrayView::new(&pencil, &extra_shape, &storage).unwrap();
        assert!(std::ptr::eq(view.pencil(), pencil.as_ref()));
        assert_eq!(view.extra_shape(), &extra_shape);
        assert_eq!(view.local_spatial_shape(), [2, 3]);
        assert_eq!(view.local_spatial_memory_shape(), [3, 2]);
        assert_eq!(view.logical_shape(), [2, 2, 3]);
        assert_eq!(view.memory_shape(), [2, 3, 2]);
        assert_eq!(view.len(), 12);
        assert!(!view.is_empty());
        assert_eq!(view.as_slice(), storage.as_slice());
        assert_eq!(view.get_local(&[1], [1, 2]), Some(&11));
        assert_eq!(view.get_local(&[0], [1, 0]), Some(&1));
        assert_eq!(view.get_local(&[0], [0, 1]), Some(&2));
        assert_eq!(view.get_local(&[], [1, 2]), None);
        assert_eq!(view.get_local(&[2], [1, 2]), None);
        assert_eq!(view.get_local(&[0], [2, 0]), None);
        assert_eq!(view.get_local(&[0], [0, 3]), None);

        let mut storage: Vec<_> = (0..12).collect();
        {
            let mut view = PencilArrayViewMut::new(&pencil, &extra_shape, &mut storage).unwrap();
            assert_eq!(view.logical_shape(), [2, 2, 3]);
            assert_eq!(view.memory_shape(), [2, 3, 2]);
            assert_eq!(view.get_local(&[0], [1, 2]), Some(&5));
            assert_eq!(view.get_local(&[0], [1, 0]), Some(&1));
            assert_eq!(view.get_local(&[0], [0, 1]), Some(&2));
            *view.get_local_mut(&[0], [1, 2]).unwrap() = 99;
            view.as_mut_slice()[0] = 77;
        }
        assert_eq!(storage[5], 99);
        assert_eq!(storage[0], 77);
    }
}
