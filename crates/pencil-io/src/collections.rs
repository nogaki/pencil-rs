use crate::mpi_io::{agree_phase, descriptor_agreement};
use crate::named_mpi::agree_name;
use crate::options::{MpiIoOptions, agree_options};
use crate::{FORMAT_VERSION, IO_NAMESPACE, IoElement, IoError, NamedIoError};
use mpi::{
    collective::{CommunicatorCollectives, SystemOperation},
    topology::CartesianCommunicator,
    traits::AsRaw,
};
use pencil_array::{ExtraShape, Pencil, PencilArray, PencilArrayView, PencilArrayViewMut};
use std::path::Path;

#[cfg(test)]
thread_local! {
    pub(crate) static STAGING_FAILURE: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
    pub(crate) static STAGING_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

const MPI_WRITE: u64 = 61;
const MPI_READ: u64 = 62;
#[cfg(feature = "parallel-hdf5")]
const HDF5_WRITE: u64 = 63;
#[cfg(feature = "parallel-hdf5")]
const HDF5_READ: u64 = 64;
const MPI_WRITE_OPTIONS: u64 = 0x301;
const MPI_READ_OPTIONS: u64 = 0x302;
#[cfg(feature = "parallel-hdf5")]
const HDF5_WRITE_OPTIONS: u64 = 0x303;
#[cfg(feature = "parallel-hdf5")]
const HDF5_READ_OPTIONS: u64 = 0x304;
const MPI_NAMED_WRITE_OPTIONS: u64 = 0x305;
const MPI_NAMED_APPEND_OPTIONS: u64 = 0x306;
const MPI_NAMED_READ_OPTIONS: u64 = 0x307;
#[cfg(feature = "parallel-hdf5")]
const HDF5_NAMED_WRITE_OPTIONS: u64 = 0x308;
#[cfg(feature = "parallel-hdf5")]
const HDF5_NAMED_APPEND_OPTIONS: u64 = 0x309;
#[cfg(feature = "parallel-hdf5")]
const HDF5_NAMED_READ_OPTIONS: u64 = 0x30a;

/// Collection validation and payload errors. Failed writes are not transactions.
#[non_exhaustive]
#[derive(Debug, Clone, thiserror::Error)]
pub enum CollectionIoError {
    /// No collection members were supplied.
    #[error("collection is empty")]
    Empty,
    /// A member has incompatible metadata or topology.
    #[error("collection member {index} has incompatible metadata")]
    Incompatible {
        /// Index of the first incompatible member.
        index: usize,
    },
    /// A combined collection I/O operation failed.
    #[error("collection I/O: {0}")]
    Io(#[from] IoError),
    /// A named collection I/O operation failed.
    #[error("named collection I/O: {0}")]
    Named(#[from] NamedIoError),
}

fn header(
    comm: &CartesianCommunicator,
    operation: u64,
    count: usize,
) -> Result<(), CollectionIoError> {
    let fixed = [IO_NAMESPACE, operation, FORMAT_VERSION, count as u64, 1];
    let (mut min, mut max) = ([0u64; 5], [0u64; 5]);
    comm.all_reduce_into(&fixed, &mut min, SystemOperation::min());
    comm.all_reduce_into(&fixed, &mut max, SystemOperation::max());
    if min != max {
        return Err(IoError::CollectiveDescriptorMismatch.into());
    }
    if count == 0 {
        return Err(CollectionIoError::Empty);
    }
    Ok(())
}

fn prepare<'a, T: IoElement + 'a, const N: usize, const M: usize>(
    comm: &CartesianCommunicator,
    path: &Path,
    operation: u64,
    count: usize,
    mut members: impl Iterator<Item = (&'a Pencil<N, M>, &'a ExtraShape, &'a [T])> + Clone,
) -> Result<PencilArray<T, N, M>, CollectionIoError> {
    let all = members.clone();
    let (first, extra, data) = members.next().expect("nonempty collection");
    let bad = all
        .clone()
        .enumerate()
        .find(|(_, (p, e, d))| {
            comm.as_raw() != p.topology().communicator().as_raw()
                || !first.same_layout(p)
                || extra != *e
                || data.len() != d.len()
        })
        .map_or(u64::MAX, |(i, _)| i as u64);
    let mut first_bad = u64::MAX;
    comm.all_reduce_into(&bad, &mut first_bad, SystemOperation::min());
    if first_bad != u64::MAX {
        return Err(CollectionIoError::Incompatible {
            index: first_bad as usize,
        });
    }
    let dims = (|| {
        let n = extra
            .dimensions()
            .len()
            .checked_add(1)
            .ok_or(IoError::SizeLimit {
                what: "collection rank",
            })?;
        if n + N > crate::MAX_PROTOCOL_RANK {
            return Err(IoError::SizeLimit {
                what: "collection rank",
            });
        }
        let mut dims = Vec::new();
        dims.try_reserve_exact(n)
            .map_err(|_| IoError::AllocationFailed {
                requested: n * std::mem::size_of::<usize>(),
            })?;
        dims.push(count);
        dims.extend_from_slice(extra.dimensions());
        Ok::<_, IoError>(dims)
    })();
    if let Err(e) = agree_phase(comm, dims.is_ok(), "collection descriptor allocation") {
        return Err(dims.err().unwrap_or(e).into());
    }
    let dims = dims?;
    descriptor_agreement(
        comm,
        path,
        operation,
        first.global_shape(),
        &dims,
        first.topology().process_grid(),
        first.permutation().axes(),
        T::CODE,
        T::WIDTH,
    )?;
    crate::options::agree_decomposition(first)?;
    #[cfg(test)]
    STAGING_CALLS.with(|calls| calls.set(calls.get() + 1));
    let staged = (|| {
        let len = data.len().checked_mul(count).ok_or(IoError::SizeLimit {
            what: "collection storage",
        })?;
        let bytes = len.checked_mul(T::WIDTH).ok_or(IoError::SizeLimit {
            what: "collection storage",
        })?;
        let mut storage = Vec::new();
        storage
            .try_reserve_exact(len)
            .map_err(|_| IoError::AllocationFailed { requested: bytes })?;
        // The index is used only by the per-member allocation fault check.
        #[cfg_attr(not(test), allow(clippy::unused_enumerate_index))]
        for (_index, (_, _, values)) in all.enumerate() {
            #[cfg(test)]
            if STAGING_FAILURE.with(|failure| failure.get() == Some(_index)) {
                return Err(IoError::AllocationFailed { requested: bytes });
            }
            storage.extend_from_slice(values);
        }
        let extra =
            ExtraShape::new(dims).map_err(|_| IoError::InvalidInput("collection extra shape"))?;
        let pencil = Pencil::new_permuted(
            first.topology().clone(),
            *first.global_shape(),
            std::array::from_fn(|i| first.decomposition()[i].index()),
            first.permutation().clone(),
        )
        .map_err(|_| IoError::InvalidInput("collection pencil"))?;
        PencilArray::from_vec(pencil, extra, storage)
            .map_err(|_| IoError::InvalidInput("collection storage"))
    })();
    if let Err(e) = agree_phase(comm, staged.is_ok(), "collection staging allocation") {
        return Err(staged.err().unwrap_or(e).into());
    }
    Ok(staged?)
}

fn mpi_write<P: AsRef<Path>, T: IoElement, const N: usize, const M: usize>(
    path: P,
    comm: &CartesianCommunicator,
    views: &[PencilArrayView<'_, T, N, M>],
    options: Option<&MpiIoOptions>,
    operation: u64,
) -> Result<(), CollectionIoError> {
    header(comm, operation, views.len())?;
    if let Some(o) = options {
        agree_options(comm, o, &[])?;
    }
    let array = prepare(
        comm,
        path.as_ref(),
        operation,
        views.len(),
        views
            .iter()
            .map(|v| (v.pencil(), v.extra_shape(), v.as_slice())),
    )?;
    if let Some(o) = options {
        crate::write_mpi_with_options(path, array.view(), o)?;
    } else {
        crate::write_mpi(path, array.view())?;
    }
    Ok(())
}
/// Write one native combined `[component, extra..., spatial...]` MPI payload.
///
/// All ranks in `comm` participate. Writers using the same file from different
/// jobs or communicators must be externally serialized. A failed write is not
/// rolled back, and this operation does not overwrite or resize an existing
/// dataset.
pub fn write_mpi_collection<P: AsRef<Path>, T: IoElement, const N: usize, const M: usize>(
    path: P,
    comm: &CartesianCommunicator,
    views: &[PencilArrayView<'_, T, N, M>],
) -> Result<(), CollectionIoError> {
    mpi_write(path, comm, views, None, MPI_WRITE)
}
/// Write a collection with explicit MPI-IO controls.
///
/// Writers using the same file from different jobs or communicators must be
/// externally serialized. A failed write is not rolled back, and this
/// operation does not overwrite or resize an existing dataset.
pub fn write_mpi_collection_with_options<
    P: AsRef<Path>,
    T: IoElement,
    const N: usize,
    const M: usize,
>(
    path: P,
    comm: &CartesianCommunicator,
    views: &[PencilArrayView<'_, T, N, M>],
    options: &MpiIoOptions,
) -> Result<(), CollectionIoError> {
    mpi_write(path, comm, views, Some(options), MPI_WRITE_OPTIONS)
}

fn mpi_read<P: AsRef<Path>, T: IoElement, const N: usize, const M: usize>(
    path: P,
    comm: &CartesianCommunicator,
    views: &mut [PencilArrayViewMut<'_, T, N, M>],
    options: Option<&MpiIoOptions>,
    operation: u64,
) -> Result<(), CollectionIoError> {
    header(comm, operation, views.len())?;
    if let Some(o) = options {
        agree_options(comm, o, &[])?;
    }
    let mut array = prepare(
        comm,
        path.as_ref(),
        operation,
        views.len(),
        views
            .iter()
            .map(|v| (v.pencil(), v.extra_shape(), v.as_slice())),
    )?;
    if let Some(o) = options {
        crate::read_mpi_with_options(path, array.view_mut(), o)?;
    } else {
        crate::read_mpi(path, array.view_mut())?;
    }
    let len = views[0].len();
    for (i, v) in views.iter_mut().enumerate() {
        v.as_mut_slice()
            .copy_from_slice(&array.as_slice()[i * len..(i + 1) * len]);
    }
    Ok(())
}
/// Read a combined MPI collection; destinations change only after staging succeeds.
///
/// All ranks in `comm` participate; the file is not overwritten or resized.
pub fn read_mpi_collection<P: AsRef<Path>, T: IoElement, const N: usize, const M: usize>(
    path: P,
    comm: &CartesianCommunicator,
    views: &mut [PencilArrayViewMut<'_, T, N, M>],
) -> Result<(), CollectionIoError> {
    mpi_read(path, comm, views, None, MPI_READ)
}
/// Read a collection with explicit MPI-IO controls.
///
/// All ranks in `comm` participate; the file is not overwritten or resized.
pub fn read_mpi_collection_with_options<
    P: AsRef<Path>,
    T: IoElement,
    const N: usize,
    const M: usize,
>(
    path: P,
    comm: &CartesianCommunicator,
    views: &mut [PencilArrayViewMut<'_, T, N, M>],
    options: &MpiIoOptions,
) -> Result<(), CollectionIoError> {
    mpi_read(path, comm, views, Some(options), MPI_READ_OPTIONS)
}

fn named_prepare<T: IoElement, const N: usize, const M: usize>(
    comm: &CartesianCommunicator,
    path: &Path,
    operation: u64,
    views: &[PencilArrayView<'_, T, N, M>],
) -> Result<PencilArray<T, N, M>, CollectionIoError> {
    prepare(
        comm,
        path,
        operation,
        views.len(),
        views
            .iter()
            .map(|v| (v.pencil(), v.extra_shape(), v.as_slice())),
    )
}
/// Write a named combined MPI collection.
///
/// Writers using the same file from different jobs or communicators must be
/// externally serialized. A failed write is not rolled back, and this
/// operation does not overwrite or resize an existing named dataset.
pub fn write_mpi_named_collection<
    P: AsRef<Path>,
    S: AsRef<str>,
    T: IoElement,
    const N: usize,
    const M: usize,
>(
    path: P,
    name: S,
    comm: &CartesianCommunicator,
    views: &[PencilArrayView<'_, T, N, M>],
    o: &MpiIoOptions,
) -> Result<(), CollectionIoError> {
    header(comm, MPI_NAMED_WRITE_OPTIONS, views.len())?;
    agree_name(comm, name.as_ref())?;
    agree_options(comm, o, &[])?;
    let a = named_prepare(comm, path.as_ref(), MPI_NAMED_WRITE_OPTIONS, views)?;
    crate::write_mpi_named_with_options(path, name, a.view(), o)?;
    Ok(())
}
/// Append a named combined MPI collection.
///
/// Writers using the same file from different jobs or communicators must be
/// externally serialized. Append is not rolled back and does not overwrite or
/// resize an existing named dataset.
pub fn append_mpi_named_collection<
    P: AsRef<Path>,
    S: AsRef<str>,
    T: IoElement,
    const N: usize,
    const M: usize,
>(
    path: P,
    name: S,
    comm: &CartesianCommunicator,
    views: &[PencilArrayView<'_, T, N, M>],
    o: &MpiIoOptions,
) -> Result<(), CollectionIoError> {
    header(comm, MPI_NAMED_APPEND_OPTIONS, views.len())?;
    agree_name(comm, name.as_ref())?;
    agree_options(comm, o, &[])?;
    let a = named_prepare(comm, path.as_ref(), MPI_NAMED_APPEND_OPTIONS, views)?;
    crate::append_mpi_named_with_options(path, name, a.view(), o)?;
    Ok(())
}
/// Read a named combined MPI collection atomically into member destinations.
///
/// The file is not overwritten or resized.
pub fn read_mpi_named_collection<
    P: AsRef<Path>,
    S: AsRef<str>,
    T: IoElement,
    const N: usize,
    const M: usize,
>(
    path: P,
    name: S,
    comm: &CartesianCommunicator,
    views: &mut [PencilArrayViewMut<'_, T, N, M>],
    o: &MpiIoOptions,
) -> Result<(), CollectionIoError> {
    header(comm, MPI_NAMED_READ_OPTIONS, views.len())?;
    agree_name(comm, name.as_ref())?;
    agree_options(comm, o, &[])?;
    let mut a = prepare(
        comm,
        path.as_ref(),
        MPI_NAMED_READ_OPTIONS,
        views.len(),
        views
            .iter()
            .map(|v| (v.pencil(), v.extra_shape(), v.as_slice())),
    )?;
    crate::read_mpi_named_with_options(path, name, a.view_mut(), o)?;
    let len = views[0].len();
    for (i, v) in views.iter_mut().enumerate() {
        v.as_mut_slice()
            .copy_from_slice(&a.as_slice()[i * len..(i + 1) * len]);
    }
    Ok(())
}

#[cfg(feature = "parallel-hdf5")]
/// Write one combined parallel HDF5 collection.
///
/// Writers using the same file from different jobs or communicators must be
/// externally serialized. A failed write is not rolled back, and this
/// operation does not overwrite or resize an existing dataset.
pub fn write_hdf5_collection<P: AsRef<Path>, T: IoElement, const N: usize, const M: usize>(
    path: P,
    comm: &CartesianCommunicator,
    views: &[PencilArrayView<'_, T, N, M>],
) -> Result<(), CollectionIoError> {
    header(comm, HDF5_WRITE, views.len())?;
    let a = prepare(
        comm,
        path.as_ref(),
        HDF5_WRITE,
        views.len(),
        views
            .iter()
            .map(|v| (v.pencil(), v.extra_shape(), v.as_slice())),
    )?;
    crate::write_hdf5(path, a.view())?;
    Ok(())
}
#[cfg(feature = "parallel-hdf5")]
/// Write a combined HDF5 collection with explicit controls.
///
/// Writers using the same file from different jobs or communicators must be
/// externally serialized. A failed write is not rolled back, and this
/// operation does not overwrite or resize an existing dataset.
pub fn write_hdf5_collection_with_options<
    P: AsRef<Path>,
    T: IoElement,
    const N: usize,
    const M: usize,
>(
    path: P,
    comm: &CartesianCommunicator,
    views: &[PencilArrayView<'_, T, N, M>],
    o: &crate::hdf5_options::Hdf5WriteOptions,
) -> Result<(), CollectionIoError> {
    header(comm, HDF5_WRITE_OPTIONS, views.len())?;
    let first = views.first().expect("nonempty collection");
    crate::hdf5_options::agree_collection_write_options(
        comm,
        o,
        N + first.extra_shape().dimensions().len() + 1,
        T::WIDTH,
    )?;
    let a = prepare(
        comm,
        path.as_ref(),
        HDF5_WRITE_OPTIONS,
        views.len(),
        views
            .iter()
            .map(|v| (v.pencil(), v.extra_shape(), v.as_slice())),
    )?;
    crate::write_hdf5_with_options(path, a.view(), o)?;
    Ok(())
}
#[cfg(feature = "parallel-hdf5")]
/// Read one combined parallel HDF5 collection; destinations change only after staging succeeds.
///
/// The file is not overwritten or resized.
pub fn read_hdf5_collection<P: AsRef<Path>, T: IoElement, const N: usize, const M: usize>(
    path: P,
    comm: &CartesianCommunicator,
    views: &mut [PencilArrayViewMut<'_, T, N, M>],
) -> Result<(), CollectionIoError> {
    header(comm, HDF5_READ, views.len())?;
    let mut a = prepare(
        comm,
        path.as_ref(),
        HDF5_READ,
        views.len(),
        views
            .iter()
            .map(|v| (v.pencil(), v.extra_shape(), v.as_slice())),
    )?;
    crate::read_hdf5(path, a.view_mut())?;
    let len = views[0].len();
    for (i, v) in views.iter_mut().enumerate() {
        v.as_mut_slice()
            .copy_from_slice(&a.as_slice()[i * len..(i + 1) * len]);
    }
    Ok(())
}
#[cfg(feature = "parallel-hdf5")]
/// Read a combined HDF5 collection with explicit controls; destinations change only after staging succeeds.
///
/// The file is not overwritten or resized.
pub fn read_hdf5_collection_with_options<
    P: AsRef<Path>,
    T: IoElement,
    const N: usize,
    const M: usize,
>(
    path: P,
    comm: &CartesianCommunicator,
    views: &mut [PencilArrayViewMut<'_, T, N, M>],
    o: &crate::hdf5_options::Hdf5ReadOptions,
) -> Result<(), CollectionIoError> {
    header(comm, HDF5_READ_OPTIONS, views.len())?;
    let first = views.first().expect("nonempty collection");
    crate::hdf5_options::agree_collection_read_options(
        comm,
        o,
        N + first.extra_shape().dimensions().len() + 1,
        T::WIDTH,
    )?;
    let mut a = prepare(
        comm,
        path.as_ref(),
        HDF5_READ_OPTIONS,
        views.len(),
        views
            .iter()
            .map(|v| (v.pencil(), v.extra_shape(), v.as_slice())),
    )?;
    crate::read_hdf5_with_options(path, a.view_mut(), o)?;
    let len = views[0].len();
    for (i, v) in views.iter_mut().enumerate() {
        v.as_mut_slice()
            .copy_from_slice(&a.as_slice()[i * len..(i + 1) * len]);
    }
    Ok(())
}
#[cfg(feature = "parallel-hdf5")]
/// Write a named combined HDF5 collection.
///
/// Writers using the same file from different jobs or communicators must be
/// externally serialized. A failed write is not rolled back, and this
/// operation does not overwrite or resize an existing named dataset.
pub fn write_hdf5_named_collection<
    P: AsRef<Path>,
    S: AsRef<str>,
    T: IoElement,
    const N: usize,
    const M: usize,
>(
    path: P,
    name: S,
    comm: &CartesianCommunicator,
    views: &[PencilArrayView<'_, T, N, M>],
    o: &crate::hdf5_options::Hdf5WriteOptions,
) -> Result<(), CollectionIoError> {
    header(comm, HDF5_NAMED_WRITE_OPTIONS, views.len())?;
    agree_name(comm, name.as_ref())?;
    let first = views.first().expect("nonempty collection");
    crate::hdf5_options::agree_collection_write_options(
        comm,
        o,
        N + first.extra_shape().dimensions().len() + 1,
        T::WIDTH,
    )?;
    let a = named_prepare(comm, path.as_ref(), HDF5_NAMED_WRITE_OPTIONS, views)?;
    crate::write_hdf5_named_with_options(path, name, a.view(), o)?;
    Ok(())
}
#[cfg(feature = "parallel-hdf5")]
/// Append a named combined HDF5 collection.
///
/// Writers using the same file from different jobs or communicators must be
/// externally serialized. Append is not rolled back and does not overwrite or
/// resize an existing named dataset.
pub fn append_hdf5_named_collection<
    P: AsRef<Path>,
    S: AsRef<str>,
    T: IoElement,
    const N: usize,
    const M: usize,
>(
    path: P,
    name: S,
    comm: &CartesianCommunicator,
    views: &[PencilArrayView<'_, T, N, M>],
    o: &crate::hdf5_options::Hdf5WriteOptions,
) -> Result<(), CollectionIoError> {
    header(comm, HDF5_NAMED_APPEND_OPTIONS, views.len())?;
    agree_name(comm, name.as_ref())?;
    let first = views.first().expect("nonempty collection");
    crate::hdf5_options::agree_collection_write_options(
        comm,
        o,
        N + first.extra_shape().dimensions().len() + 1,
        T::WIDTH,
    )?;
    let a = named_prepare(comm, path.as_ref(), HDF5_NAMED_APPEND_OPTIONS, views)?;
    crate::append_hdf5_named_with_options(path, name, a.view(), o)?;
    Ok(())
}
#[cfg(feature = "parallel-hdf5")]
/// Read a named combined HDF5 collection atomically into member destinations.
///
/// The file is not overwritten or resized.
pub fn read_hdf5_named_collection<
    P: AsRef<Path>,
    S: AsRef<str>,
    T: IoElement,
    const N: usize,
    const M: usize,
>(
    path: P,
    name: S,
    comm: &CartesianCommunicator,
    views: &mut [PencilArrayViewMut<'_, T, N, M>],
    o: &crate::hdf5_options::Hdf5ReadOptions,
) -> Result<(), CollectionIoError> {
    header(comm, HDF5_NAMED_READ_OPTIONS, views.len())?;
    agree_name(comm, name.as_ref())?;
    let first = views.first().expect("nonempty collection");
    crate::hdf5_options::agree_collection_read_options(
        comm,
        o,
        N + first.extra_shape().dimensions().len() + 1,
        T::WIDTH,
    )?;
    let mut a = prepare(
        comm,
        path.as_ref(),
        HDF5_NAMED_READ_OPTIONS,
        views.len(),
        views
            .iter()
            .map(|v| (v.pencil(), v.extra_shape(), v.as_slice())),
    )?;
    crate::read_hdf5_named_with_options(path, name, a.view_mut(), o)?;
    let len = views[0].len();
    for (i, v) in views.iter_mut().enumerate() {
        v.as_mut_slice()
            .copy_from_slice(&a.as_slice()[i * len..(i + 1) * len]);
    }
    Ok(())
}
