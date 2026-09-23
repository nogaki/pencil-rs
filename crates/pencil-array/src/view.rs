use crate::{
    ArrayError, ExtraShape, LocalGrid, LocalGridError, Pencil, checked::checked_product,
    geometry::row_major_offset,
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
    /// Storage must be in row-major physical order: `[extra..., permuted
    /// spatial...]`, with the last axis varying fastest. Extra axes follow
    /// `extra_shape.dimensions()`; spatial axes follow `pencil.permutation()`.
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

    /// Returns `[extra..., spatial...]` in logical-axis order.
    pub fn logical_shape(&self) -> Vec<usize> {
        LocalArrayLayout::logical_shape(self)
    }

    /// Returns `[extra..., permuted spatial...]` in row-major memory order.
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

    /// Returns the borrowed row-major buffer.
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
    /// Storage must be in row-major physical order: `[extra..., permuted
    /// spatial...]`, with the last axis varying fastest. Extra axes follow
    /// `extra_shape.dimensions()`; spatial axes follow `pencil.permutation()`.
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

    /// Returns `[extra..., spatial...]` in logical-axis order.
    pub fn logical_shape(&self) -> Vec<usize> {
        LocalArrayLayout::logical_shape(self)
    }

    /// Returns `[extra..., permuted spatial...]` in row-major memory order.
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

    /// Returns a shared borrow of the row-major buffer.
    pub fn as_slice(&self) -> &[T] {
        LocalArrayLayout::as_slice(self)
    }

    /// Returns an exclusive borrow of the row-major buffer.
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
    use crate::{
        ArrayError, AxisPermutation, ExtraShape, LocalTransposeError, LocalTransposePlan,
        ManyPencilArray, MpiTopology, Pencil, PointToPointTransposePlan, TransposeWorkspace,
        point_to_point_transpose::{test_trace_event, test_trace_finish, test_trace_start},
    };

    #[test]
    fn views_validate_storage_and_expose_permuted_local_layout() {
        let universe = mpi::initialize().expect("MPI initialization failed");
        let world = universe.world();
        if world.size() != 1 {
            let topology = MpiTopology::<1>::new(&world, [world.size() as usize]).unwrap();
            check_callback_schedule(&topology);
            return;
        }

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

        let destination = pencil
            .with_permutation(AxisPermutation::new([0, 1]).unwrap())
            .unwrap();
        let transpose_plan =
            LocalTransposePlan::new(Arc::clone(&pencil), Arc::clone(&destination)).unwrap();
        let mut many = ManyPencilArray::from_vec(
            vec![Arc::clone(&pencil), Arc::clone(&destination)],
            0,
            extra_shape.clone(),
            (0..12).collect(),
        )
        .unwrap();
        let expected_logical = {
            let view = many.active_view().unwrap();
            let mut values = Vec::new();
            for extra_index in 0..2 {
                for x in 0..2 {
                    for y in 0..3 {
                        values.push(*view.get_local(&[extra_index], [x, y]).unwrap());
                    }
                }
            }
            values
        };
        let tail = [0xfeed_i32, 0xcafe_i32, 0xbeef_i32];
        let mut send = Vec::with_capacity(32);
        send.resize(12, 0);
        send.extend_from_slice(&tail);
        let send_ptr = send.as_ptr();
        let send_len = send.len();
        let send_capacity = send.capacity();
        let receive = vec![0x1234_i32, 0x5678_i32];
        let receive_before = receive.clone();
        let receive_ptr = receive.as_ptr();
        let receive_len = receive.len();
        let receive_capacity = receive.capacity();
        let mut workspace = TransposeWorkspace::from_vecs(send, receive);
        transpose_plan
            .execute_in_place_with_transpose_workspace(&mut many, &mut workspace)
            .unwrap();
        assert!(
            many.active_pencil()
                .unwrap()
                .same_layout(destination.as_ref())
        );
        {
            let view = many.active_view().unwrap();
            let mut index = 0;
            for extra_index in 0..2 {
                for x in 0..2 {
                    for y in 0..3 {
                        assert_eq!(
                            view.get_local(&[extra_index], [x, y]),
                            Some(&expected_logical[index])
                        );
                        index += 1;
                    }
                }
            }
            assert_eq!(index, expected_logical.len());
        }
        let expected_staged: Vec<_> = (0..12).collect();
        assert_eq!(&workspace.send_buffer[..12], expected_staged.as_slice());
        assert_eq!(&workspace.send_buffer[12..], &tail);
        assert_eq!(workspace.send_buffer.as_ptr(), send_ptr);
        assert_eq!(workspace.send_buffer.len(), send_len);
        assert_eq!(workspace.send_buffer.capacity(), send_capacity);
        assert_eq!(
            workspace.receive_buffer.as_slice(),
            receive_before.as_slice()
        );
        assert_eq!(workspace.receive_buffer.as_ptr(), receive_ptr);
        assert_eq!(workspace.receive_buffer.len(), receive_len);
        assert_eq!(workspace.receive_buffer.capacity(), receive_capacity);

        let mut short_many = ManyPencilArray::from_vec(
            vec![Arc::clone(&pencil), Arc::clone(&destination)],
            0,
            extra_shape,
            (0..12).collect(),
        )
        .unwrap();
        let short_values = short_many.active_view().unwrap().as_slice().to_vec();
        let mut short_send = Vec::with_capacity(12);
        short_send.resize(11, 0);
        let short_send_before = short_send.clone();
        let short_send_ptr = short_send.as_ptr();
        let short_send_capacity = short_send.capacity();
        let short_receive = vec![0x9abc_i32, 0xdef0_i32];
        let short_receive_before = short_receive.clone();
        let short_receive_ptr = short_receive.as_ptr();
        let short_receive_capacity = short_receive.capacity();
        let mut short_workspace = TransposeWorkspace::from_vecs(short_send, short_receive);
        assert!(matches!(
            transpose_plan
                .execute_in_place_with_transpose_workspace(&mut short_many, &mut short_workspace,),
            Err(LocalTransposeError::ScratchTooSmall {
                required: 12,
                actual: 11,
            })
        ));
        assert!(
            short_many
                .active_pencil()
                .unwrap()
                .same_layout(pencil.as_ref())
        );
        assert_eq!(short_many.active_view().unwrap().as_slice(), short_values);
        assert_eq!(
            short_workspace.send_buffer.as_slice(),
            short_send_before.as_slice()
        );
        assert_eq!(short_workspace.send_buffer.as_ptr(), short_send_ptr);
        assert_eq!(short_workspace.send_buffer.capacity(), short_send_capacity);
        assert_eq!(
            short_workspace.receive_buffer.as_slice(),
            short_receive_before.as_slice()
        );
        assert_eq!(short_workspace.receive_buffer.as_ptr(), short_receive_ptr);
        assert_eq!(
            short_workspace.receive_buffer.capacity(),
            short_receive_capacity
        );

        check_callback_schedule(&topology);
    }

    fn check_callback_schedule(topology: &Arc<MpiTopology<1>>) {
        let callback_source_pencil =
            Pencil::<2, 1>::new(Arc::clone(topology), [2, 3], [0]).unwrap();
        let callback_destination_pencil = Pencil::<2, 1>::new_permuted(
            Arc::clone(topology),
            [2, 3],
            [1],
            AxisPermutation::new([1, 0]).unwrap(),
        )
        .unwrap();
        let callback_plan = PointToPointTransposePlan::new(
            Arc::clone(&callback_source_pencil),
            Arc::clone(&callback_destination_pencil),
        )
        .unwrap();
        let callback_requirements = callback_plan
            .workspace_requirements(&ExtraShape::new([2]).unwrap())
            .unwrap();
        let mut callback_workspace = TransposeWorkspace::from_vecs(
            vec![0; callback_requirements.send_len],
            vec![0; callback_requirements.receive_len],
        );
        let callback_source = vec![0_i32; callback_source_pencil.local_len() * 2];
        let mut callback_destination = vec![0_i32; callback_destination_pencil.local_len() * 2];
        test_trace_start();
        callback_plan
            .execute_views_with_callback(
                PencilArrayView::new(
                    &callback_source_pencil,
                    &ExtraShape::new([2]).unwrap(),
                    &callback_source,
                )
                .unwrap(),
                PencilArrayViewMut::new(
                    &callback_destination_pencil,
                    &ExtraShape::new([2]).unwrap(),
                    &mut callback_destination,
                )
                .unwrap(),
                &mut callback_workspace,
                |_values| {
                    test_trace_event("callback");
                    Ok::<_, ()>(())
                },
            )
            .unwrap();
        assert_eq!(test_trace_finish(), ["unpack", "callback", "send_wait"]);
    }
}
