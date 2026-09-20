use std::{iter::FusedIterator, ops::Range};

use crate::{LocalGridError, Pencil, checked::checked_product};

/// Borrowed global coordinate axes for one pencil's local spatial domain.
///
/// `LocalGrid` borrows the supplied coordinate slices, not the [`Pencil`] or
/// any MPI resource. Each supplied slice must describe one complete global
/// spatial axis. The stored axis slices are restricted to this rank's local
/// ranges and remain in logical-axis order.
///
/// The grid contains spatial coordinates only. Extra array dimensions are not
/// transformed; repeat the grid for each extra batch when pairing it with an
/// array's [`crate::PencilArray::as_slice`] storage.
///
/// Ranks that do not own a coordinate in a decomposed range, including ranks
/// with an empty local range, produce no grid points. Global indices owned by
/// another rank are not represented by this grid.
#[derive(Clone, Debug)]
pub struct LocalGrid<'a, C, const N: usize> {
    axes: [&'a [C]; N],
    logical_shape: [usize; N],
    memory_shape: [usize; N],
    logical_to_memory: [usize; N],
    len: usize,
}

impl<'a, C, const N: usize> LocalGrid<'a, C, N> {
    pub(crate) fn from_pencil<const M: usize>(
        pencil: &Pencil<N, M>,
        coordinates: [&'a [C]; N],
    ) -> Result<Self, LocalGridError> {
        for (axis, coordinate_axis) in coordinates.iter().enumerate() {
            let expected = pencil.global_shape()[axis];
            let actual = coordinate_axis.len();
            if actual != expected {
                return Err(LocalGridError::AxisLengthMismatch {
                    axis,
                    expected,
                    actual,
                });
            }
        }

        let logical_shape = pencil.local_shape_logical();
        let memory_shape = pencil.local_shape_memory();
        let len = checked_product(&logical_shape).map_err(|_| LocalGridError::SizeOverflow)?;
        let memory_len =
            checked_product(&memory_shape).map_err(|_| LocalGridError::SizeOverflow)?;
        debug_assert_eq!(len, memory_len);
        debug_assert_eq!(len, pencil.local_len());

        let axes = std::array::from_fn(|axis| {
            let range: Range<usize> = pencil.local_ranges()[axis].clone();
            coordinates[axis]
                .get(range)
                .expect("validated global coordinate axes cover the pencil ranges")
        });
        let axes_in_memory_order = pencil.permutation().axes();
        let mut logical_to_memory = [0; N];
        for (memory_position, axis) in axes_in_memory_order.iter().copied().enumerate() {
            logical_to_memory[axis.index()] = memory_position;
        }

        Ok(Self {
            axes,
            logical_shape,
            memory_shape,
            logical_to_memory,
            len,
        })
    }

    /// Returns the local global-coordinate slice for a logical spatial axis.
    ///
    /// An invalid axis returns `None`. The returned slice is borrowed from the
    /// coordinate slice supplied to [`Pencil::local_grid`].
    pub fn axis(&self, axis: usize) -> Option<&'a [C]> {
        self.axes.get(axis).copied()
    }

    /// Returns the global-coordinate tuple at logical local indices.
    ///
    /// The indices use logical spatial-axis order and are zero-based within
    /// this rank's local ranges. Invalid indices return `None`.
    pub fn get_local(&self, local_indices: [usize; N]) -> Option<[&'a C; N]> {
        if local_indices
            .iter()
            .enumerate()
            .any(|(axis, &index)| index >= self.logical_shape[axis])
        {
            return None;
        }

        Some(std::array::from_fn(|axis| {
            &self.axes[axis][local_indices[axis]]
        }))
    }

    /// Iterates local global-coordinate tuples in physical spatial memory order.
    ///
    /// The tuple itself remains in logical-axis order. The order matches the
    /// spatial suffix of the array's row-major [`crate::PencilArray::as_slice`]
    /// buffer, so one grid iteration can be paired with one extra batch of
    /// spatial values. The iterator performs no allocation and requires no MPI
    /// operation.
    pub fn iter(&self) -> LocalGridIter<'a, C, N> {
        LocalGridIter {
            axes: self.axes,
            memory_shape: self.memory_shape,
            logical_to_memory: self.logical_to_memory,
            memory_indices: [0; N],
            remaining: self.len,
        }
    }
}

/// The allocation-free iterator returned by [`LocalGrid::iter`].
#[derive(Debug)]
pub struct LocalGridIter<'a, C, const N: usize> {
    axes: [&'a [C]; N],
    memory_shape: [usize; N],
    logical_to_memory: [usize; N],
    memory_indices: [usize; N],
    remaining: usize,
}

impl<'a, C, const N: usize> Iterator for LocalGridIter<'a, C, N> {
    type Item = [&'a C; N];

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        let item = std::array::from_fn(|logical_axis| {
            &self.axes[logical_axis][self.memory_indices[self.logical_to_memory[logical_axis]]]
        });
        self.remaining -= 1;
        if self.remaining != 0 {
            self.advance_memory_indices();
        }
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<C, const N: usize> ExactSizeIterator for LocalGridIter<'_, C, N> {
    fn len(&self) -> usize {
        self.remaining
    }
}

impl<C, const N: usize> FusedIterator for LocalGridIter<'_, C, N> {}

impl<'a, C, const N: usize> LocalGridIter<'a, C, N> {
    fn advance_memory_indices(&mut self) {
        for position in (0..N).rev() {
            if self.memory_indices[position] < self.memory_shape[position] - 1 {
                self.memory_indices[position] += 1;
                return;
            }
            self.memory_indices[position] = 0;
        }
        debug_assert_eq!(self.remaining, 0);
    }
}

impl<'coords, C, const N: usize> IntoIterator for &LocalGrid<'coords, C, N> {
    type Item = [&'coords C; N];
    type IntoIter = LocalGridIter<'coords, C, N>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
