//! Partitioned MPI buffers whose segment bounds were checked as `usize`.
//!
//! The standard [`mpi::datatype::Partition`] adapters require the whole buffer
//! count to fit `mpi::Count`. A checked partition keeps the buffer length as
//! `usize`, so segment ends and the total length need not fit `mpi::Count`;
//! only the per-peer counts and displacements handled by MPI do.

use std::ffi::c_void;

use mpi::{
    Count,
    datatype::{
        AsDatatype, DatatypeRef, Equivalence, Partitioned, PartitionedBuffer, PartitionedBufferMut,
        Pointer, PointerMut, UserDatatype,
    },
};

fn assert_partition_bounds(len: usize, counts: &[Count], displacements: &[Count]) {
    assert_eq!(
        counts.len(),
        displacements.len(),
        "partition metadata lengths differ"
    );
    for (&count, &displacement) in counts.iter().zip(displacements) {
        assert!(
            count >= 0 && displacement >= 0,
            "partition count and displacement must be nonnegative"
        );
        let end = (displacement as usize)
            .checked_add(count as usize)
            .expect("partition segment end overflowed usize");
        assert!(end <= len, "partition segment exceeds buffer");
    }
}

pub(crate) struct CheckedPartition<'a, T> {
    buffer: &'a [T],
    counts: &'a [Count],
    displacements: &'a [Count],
}

impl<'a, T> CheckedPartition<'a, T> {
    pub(crate) fn new(buffer: &'a [T], counts: &'a [Count], displacements: &'a [Count]) -> Self {
        assert_partition_bounds(buffer.len(), counts, displacements);
        Self {
            buffer,
            counts,
            displacements,
        }
    }
}

impl<T> Partitioned for CheckedPartition<'_, T> {
    fn counts(&self) -> &[Count] {
        self.counts
    }

    fn displs(&self) -> &[Count] {
        self.displacements
    }
}

// SAFETY: the initialized slice supplies its own pointer and remains shared
// for this partition's lifetime; every segment was checked against its length.
unsafe impl<T: Equivalence> Pointer for CheckedPartition<'_, T> {
    fn pointer(&self) -> *const c_void {
        <[T] as Pointer>::pointer(self.buffer)
    }
}

// SAFETY: the datatype comes from the same T slice that supplies the pointer.
// Correct Equivalence has rsmpi's slice layout contract, including array stride
// and a typemap that does not read uninitialized padding.
unsafe impl<T: Equivalence> AsDatatype for CheckedPartition<'_, T> {
    type Out = <T as Equivalence>::Out;

    fn as_datatype(&self) -> Self::Out {
        <[T] as AsDatatype>::as_datatype(self.buffer)
    }
}

impl<T: Equivalence> PartitionedBuffer for CheckedPartition<'_, T> {}

pub(crate) struct CheckedPartitionMut<'a, T> {
    buffer: &'a mut [T],
    counts: &'a [Count],
    displacements: &'a [Count],
}

impl<'a, T> CheckedPartitionMut<'a, T> {
    pub(crate) fn new(
        buffer: &'a mut [T],
        counts: &'a [Count],
        displacements: &'a [Count],
    ) -> Self {
        assert_partition_bounds(buffer.len(), counts, displacements);
        Self {
            buffer,
            counts,
            displacements,
        }
    }
}

impl<T> Partitioned for CheckedPartitionMut<'_, T> {
    fn counts(&self) -> &[Count] {
        self.counts
    }

    fn displs(&self) -> &[Count] {
        self.displacements
    }
}

// SAFETY: the initialized slice is exclusively borrowed for this partition's
// lifetime; every segment was checked against its length.
unsafe impl<T: Equivalence> PointerMut for CheckedPartitionMut<'_, T> {
    fn pointer_mut(&mut self) -> *mut c_void {
        <[T] as PointerMut>::pointer_mut(self.buffer)
    }
}

// SAFETY: the datatype comes from the same exclusively borrowed T slice.
// Correct Equivalence has rsmpi's slice layout contract, including array stride
// and a typemap that does not read uninitialized padding.
unsafe impl<T: Equivalence> AsDatatype for CheckedPartitionMut<'_, T> {
    type Out = <T as Equivalence>::Out;

    fn as_datatype(&self) -> Self::Out {
        <[T] as AsDatatype>::as_datatype(self.buffer)
    }
}

impl<T: Equivalence> PartitionedBufferMut for CheckedPartitionMut<'_, T> {}

/// A checked send partition carrying an already validated payload datatype.
///
/// MPI interprets displacements in units of the datatype extent, so the
/// adapter needs the datatype itself rather than only the buffer element type.
/// It deliberately implements neither `Collection` nor `Buffer`: the total
/// buffer length must not be converted to `mpi::Count`.
pub(crate) struct DatatypePartition<'a, T> {
    buffer: &'a [T],
    counts: &'a [Count],
    displacements: &'a [Count],
    datatype: &'a UserDatatype,
}

impl<'a, T> DatatypePartition<'a, T> {
    /// Builds a send partition over `buffer` using a validated `datatype`.
    ///
    /// # Safety
    ///
    /// `datatype` must be committed and describe exactly one `T`:
    ///
    /// - its extent equals `size_of::<T>()`,
    /// - its true lower bound and true extent are nonnegative and lie within
    ///   one `T`,
    /// - a typemap touched by count `c` at displacement `d` stays inside the
    ///   initialized prefix of `buffer`; every segment end was checked as
    ///   `usize` against `buffer.len()`, so `d + c <= buffer.len()` suffices
    ///   when the extent is `size_of::<T>()`.
    ///
    /// `T` must have a correct unsafe `Equivalence` implementation, so that
    /// the typemap neither reads uninitialized padding nor misdescribes `T`.
    pub(crate) unsafe fn new(
        buffer: &'a [T],
        counts: &'a [Count],
        displacements: &'a [Count],
        datatype: &'a UserDatatype,
    ) -> Self {
        assert_partition_bounds(buffer.len(), counts, displacements);
        Self {
            buffer,
            counts,
            displacements,
            datatype,
        }
    }
}

impl<T> Partitioned for DatatypePartition<'_, T> {
    fn counts(&self) -> &[Count] {
        self.counts
    }

    fn displs(&self) -> &[Count] {
        self.displacements
    }
}

// SAFETY: the initialized slice supplies its own pointer and remains shared for
// this partition's lifetime; every segment was checked against its length.
unsafe impl<T> Pointer for DatatypePartition<'_, T> {
    fn pointer(&self) -> *const c_void {
        self.buffer.as_ptr().cast::<c_void>()
    }
}

// SAFETY: the constructor requires `datatype` to describe exactly one `T` for
// the same buffer, so the borrowed handle and pointer agree.
unsafe impl<'a, T> AsDatatype for DatatypePartition<'a, T> {
    type Out = DatatypeRef<'a>;

    fn as_datatype(&self) -> Self::Out {
        self.datatype.as_ref()
    }
}

impl<T> PartitionedBuffer for DatatypePartition<'_, T> {}

/// A checked receive partition carrying an already validated payload datatype.
///
/// See [`DatatypePartition`] for the datatype contract.
pub(crate) struct DatatypePartitionMut<'a, T> {
    buffer: &'a mut [T],
    counts: &'a [Count],
    displacements: &'a [Count],
    datatype: &'a UserDatatype,
}

impl<'a, T> DatatypePartitionMut<'a, T> {
    /// Builds a receive partition over `buffer` using a validated `datatype`.
    ///
    /// # Safety
    ///
    /// The requirements are those of [`DatatypePartition::new`], plus that
    /// `buffer` is exclusively borrowed for the whole exchange.
    pub(crate) unsafe fn new(
        buffer: &'a mut [T],
        counts: &'a [Count],
        displacements: &'a [Count],
        datatype: &'a UserDatatype,
    ) -> Self {
        assert_partition_bounds(buffer.len(), counts, displacements);
        Self {
            buffer,
            counts,
            displacements,
            datatype,
        }
    }
}

impl<T> Partitioned for DatatypePartitionMut<'_, T> {
    fn counts(&self) -> &[Count] {
        self.counts
    }

    fn displs(&self) -> &[Count] {
        self.displacements
    }
}

// SAFETY: the initialized slice is exclusively borrowed for this partition's
// lifetime; every segment was checked against its length.
unsafe impl<T> PointerMut for DatatypePartitionMut<'_, T> {
    fn pointer_mut(&mut self) -> *mut c_void {
        self.buffer.as_mut_ptr().cast::<c_void>()
    }
}

// SAFETY: the constructor requires `datatype` to describe exactly one `T` for
// the same buffer, so the borrowed handle and pointer agree.
unsafe impl<'a, T> AsDatatype for DatatypePartitionMut<'a, T> {
    type Out = DatatypeRef<'a>;

    fn as_datatype(&self) -> Self::Out {
        self.datatype.as_ref()
    }
}

impl<T> PartitionedBufferMut for DatatypePartitionMut<'_, T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_partitions_preserve_metadata_and_slice_pointers() {
        let buffer = [0u8; 4];
        let counts = [1, 0, 2];
        let displacements = [0, 4, 2];
        let partition = CheckedPartition::new(&buffer, &counts, &displacements);
        assert_eq!(partition.counts(), &counts);
        assert_eq!(partition.displs(), &displacements);
        assert_eq!(partition.pointer(), buffer.as_ptr().cast::<c_void>());

        let mut buffer = [0u8; 4];
        let expected_pointer = buffer.as_mut_ptr().cast::<c_void>();
        let mut partition = CheckedPartitionMut::new(&mut buffer, &counts, &displacements);
        assert_eq!(partition.counts(), &counts);
        assert_eq!(partition.displs(), &displacements);
        assert_eq!(partition.pointer_mut(), expected_pointer);
    }

    #[test]
    fn checked_partitions_accept_empty_buffers() {
        let buffer: [u8; 0] = [];
        let _ = CheckedPartition::new(&buffer, &[], &[]);
        let _ = CheckedPartition::new(&buffer, &[0], &[0]);

        let mut buffer: [u8; 0] = [];
        let _ = CheckedPartitionMut::new(&mut buffer, &[], &[]);
        let _ = CheckedPartitionMut::new(&mut buffer, &[0], &[0]);
    }

    #[test]
    fn partition_bounds_accept_total_above_count_max_without_payload() {
        let count = 1_500_000_000;
        let total = usize::try_from(3_000_000_000_u64).unwrap();
        assert_partition_bounds(total, &[count, count], &[0, count]);
    }

    #[test]
    fn checked_partitions_reject_invalid_metadata() {
        let cases: &[(&[Count], &[Count])] = &[
            (&[1], &[]),
            (&[], &[0]),
            (&[-1], &[0]),
            (&[0], &[-1]),
            (&[5], &[0]),
            (&[2], &[3]),
            (&[0], &[5]),
            (&[1_500_000_000], &[1_500_000_000]),
        ];

        for &(counts, displacements) in cases {
            assert!(
                std::panic::catch_unwind(|| {
                    let buffer = [0u8; 4];
                    let _ = CheckedPartition::new(&buffer, counts, displacements);
                })
                .is_err()
            );
            assert!(
                std::panic::catch_unwind(|| {
                    let mut buffer = [0u8; 4];
                    let _ = CheckedPartitionMut::new(&mut buffer, counts, displacements);
                })
                .is_err()
            );
        }
    }
}
