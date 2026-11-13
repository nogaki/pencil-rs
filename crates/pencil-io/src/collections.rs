use crate::format::ReadTarget;
use crate::mpi_io::{
    MPI_LOCAL_BYTE_LIMIT_WHAT, agree_mpi_preparation, agree_phase, build_layout,
    checked_local_byte_count, descriptor_agreement, mpi_preparation_error,
};
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
    /// Test-only injection of the first combined byte-staging reservation.
    pub(crate) static STAGING_ALLOCATION_FAILURE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
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

fn prepare_metadata<'a, T: IoElement + 'a, const N: usize, const M: usize>(
    comm: &CartesianCommunicator,
    path: &Path,
    operation: u64,
    mpi_preflight: bool,
    count: usize,
    mut members: impl Iterator<Item = (&'a Pencil<N, M>, &'a ExtraShape, &'a [T])> + Clone,
) -> Result<Vec<usize>, CollectionIoError> {
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
        let combined = if mpi_preflight {
            let combined = data.len().checked_mul(count).ok_or(IoError::SizeLimit {
                what: MPI_LOCAL_BYTE_LIMIT_WHAT,
            })?;
            checked_local_byte_count(combined, T::WIDTH)?;
            combined
        } else {
            0
        };
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
        if mpi_preflight {
            build_layout(
                first.global_shape(),
                &dims,
                &first.local_shape_logical(),
                first.local_ranges(),
                combined,
                T::WIDTH,
            )?;
        }
        Ok::<_, IoError>(dims)
    })();
    let agreement = if mpi_preflight {
        agree_mpi_preparation(
            comm,
            dims.as_ref().err(),
            "collection descriptor allocation",
        )
    } else {
        agree_phase(comm, dims.is_ok(), "collection descriptor allocation")
    };
    if let Err(agreement) = agreement {
        let error = if mpi_preflight {
            mpi_preparation_error(dims.err(), agreement)
        } else {
            dims.err().unwrap_or(agreement)
        };
        return Err(error.into());
    }
    let dims = dims?;
    descriptor_agreement(
        comm,
        path,
        operation,
        first.global_shape(),
        &dims,
        first.topology().process_grid(),
        first.decomposition(),
        first.permutation().axes(),
        T::CODE,
        T::WIDTH,
    )?;
    crate::options::agree_decomposition(first)?;
    Ok(dims)
}

fn prepare<'a, T: IoElement + 'a, const N: usize, const M: usize>(
    comm: &CartesianCommunicator,
    path: &Path,
    operation: u64,
    mpi_preflight: bool,
    count: usize,
    members: impl Iterator<Item = (&'a Pencil<N, M>, &'a ExtraShape, &'a [T])> + Clone,
) -> Result<PencilArray<T, N, M>, CollectionIoError> {
    let all = members.clone();
    let (first, _, data) = members.clone().next().expect("nonempty collection");
    let dims = prepare_metadata(comm, path, operation, mpi_preflight, count, members)?;
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

/// Prepares collection read metadata and reserves one combined byte-staging
/// buffer before any destination is touched.
///
/// Returns the combined `[count, extra...]` shape, the combined local element
/// count, and the reserved (empty, `capacity >= bytes`) staging buffer.
fn prepare_read<T: IoElement, const N: usize, const M: usize>(
    comm: &CartesianCommunicator,
    path: &Path,
    operation: u64,
    mpi_preflight: bool,
    views: &[PencilArrayViewMut<'_, T, N, M>],
) -> Result<(Vec<usize>, usize, Vec<u8>), CollectionIoError> {
    let dims = prepare_metadata(
        comm,
        path,
        operation,
        mpi_preflight,
        views.len(),
        views
            .iter()
            .map(|v| (v.pencil(), v.extra_shape(), v.as_slice())),
    )?;
    let first = views.first().expect("nonempty collection");
    #[cfg(test)]
    STAGING_CALLS.with(|calls| calls.set(calls.get() + 1));
    let staged = (|| {
        let len = first
            .len()
            .checked_mul(views.len())
            .ok_or(IoError::SizeLimit {
                what: "collection storage",
            })?;
        let bytes = len.checked_mul(T::WIDTH).ok_or(IoError::SizeLimit {
            what: "collection storage",
        })?;
        #[cfg(test)]
        if STAGING_ALLOCATION_FAILURE.with(std::cell::Cell::get) {
            return Err(IoError::AllocationFailed { requested: bytes });
        }
        #[cfg(test)]
        if STAGING_FAILURE.with(|failure| failure.get().is_some()) {
            return Err(IoError::AllocationFailed { requested: bytes });
        }
        let mut staging = Vec::new();
        staging
            .try_reserve_exact(bytes)
            .map_err(|_| IoError::AllocationFailed { requested: bytes })?;
        Ok::<_, IoError>((len, staging))
    })();
    if let Err(e) = agree_phase(comm, staged.is_ok(), "collection staging allocation") {
        return Err(staged.err().unwrap_or(e).into());
    }
    let (len, staging) = staged?;
    Ok((dims, len, staging))
}

fn mpi_write<P: AsRef<Path>, T: IoElement, const N: usize, const M: usize>(
    path: P,
    comm: &CartesianCommunicator,
    views: &[PencilArrayView<'_, T, N, M>],
    options: Option<&MpiIoOptions>,
    operation: u64,
) -> Result<(), CollectionIoError> {
    header(comm, operation, views.len())?;
    let array = prepare(
        comm,
        path.as_ref(),
        operation,
        true,
        views.len(),
        views
            .iter()
            .map(|v| (v.pencil(), v.extra_shape(), v.as_slice())),
    )?;
    crate::mpi_io::write_mpi_target(path, array.view(), options)?;
    Ok(())
}
/// Write one native combined `[component, extra..., spatial...]` MPI payload.
///
/// Each rank's combined local payload is limited to 2147483647 bytes/rank
/// (`combined_local_elements * scalar_width`; for `f64`, up to 268435455
/// elements), not each member separately or the whole file. If any rank exceeds
/// it, every rank returns `CollectionIoError::Io(IoError::SizeLimit { what: "MPI
/// local byte count" })` before combined typed storage is allocated. Successful
/// paths still allocate and copy the combined typed storage.
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
/// Write a collection with explicit MPI-IO controls. Applies the same
/// combined-payload MPI local byte limit as [`write_mpi_collection`].
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
    let (dims, len, staging) = prepare_read(comm, path.as_ref(), operation, true, views)?;
    let target = ReadTarget::collection(views, dims, len, staging);
    crate::mpi_io::read_mpi_target(path, target, options)?;
    Ok(())
}
/// Read a combined MPI collection into its member destinations.
/// One combined byte-staging payload is reserved and read once; members are
/// decoded directly from it only after native cleanup and collective success,
/// so a returned error leaves every destination unchanged. The same
/// combined-payload 2147483647 bytes/rank limit as [`write_mpi_collection`]
/// applies before staging.
///
/// All ranks in `comm` participate; the file is not overwritten or resized.
pub fn read_mpi_collection<P: AsRef<Path>, T: IoElement, const N: usize, const M: usize>(
    path: P,
    comm: &CartesianCommunicator,
    views: &mut [PencilArrayViewMut<'_, T, N, M>],
) -> Result<(), CollectionIoError> {
    mpi_read(path, comm, views, None, MPI_READ)
}
/// Read a collection with explicit MPI-IO controls. Applies the same
/// combined-payload MPI local byte limit and one-payload direct decode as
/// [`read_mpi_collection`].
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
    mpi_preflight: bool,
    views: &[PencilArrayView<'_, T, N, M>],
) -> Result<PencilArray<T, N, M>, CollectionIoError> {
    prepare(
        comm,
        path,
        operation,
        mpi_preflight,
        views.len(),
        views
            .iter()
            .map(|v| (v.pencil(), v.extra_shape(), v.as_slice())),
    )
}
/// Write a named combined MPI collection. Applies the same combined-payload MPI
/// local byte limit as [`write_mpi_collection`] to each named record, not the
/// whole container.
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
    let a = named_prepare(comm, path.as_ref(), MPI_NAMED_WRITE_OPTIONS, true, views)?;
    crate::named_mpi::write_named_target(path.as_ref(), name.as_ref(), a.view(), false, o)?;
    Ok(())
}
/// Append a named combined MPI collection. Applies the same combined-payload MPI
/// local byte limit as [`write_mpi_collection`] to the appended record.
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
    let a = named_prepare(comm, path.as_ref(), MPI_NAMED_APPEND_OPTIONS, true, views)?;
    crate::named_mpi::write_named_target(path.as_ref(), name.as_ref(), a.view(), true, o)?;
    Ok(())
}
/// Read a named combined MPI collection atomically into member destinations.
/// Applies the same combined-payload MPI local byte limit and direct decode
/// after cleanup as [`read_mpi_collection`]; a rejected read leaves all
/// destinations unchanged.
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
    let (dims, len, staging) =
        prepare_read(comm, path.as_ref(), MPI_NAMED_READ_OPTIONS, true, views)?;
    let target = ReadTarget::collection(views, dims, len, staging);
    crate::named_mpi::read_named_target(path.as_ref(), name.as_ref(), target, Some(o))?;
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
        false,
        views.len(),
        views
            .iter()
            .map(|v| (v.pencil(), v.extra_shape(), v.as_slice())),
    )?;
    crate::hdf5_io::write_hdf5_target(path, a.view())?;
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
    let a = prepare(
        comm,
        path.as_ref(),
        HDF5_WRITE_OPTIONS,
        false,
        views.len(),
        views
            .iter()
            .map(|v| (v.pencil(), v.extra_shape(), v.as_slice())),
    )?;
    crate::hdf5_io::write_hdf5_target_options(path, a.view(), o.settings())?;
    Ok(())
}
#[cfg(feature = "parallel-hdf5")]
/// Read one combined parallel HDF5 collection into its member destinations.
/// One combined byte-staging payload is reserved and read once; members are
/// decoded directly from it only after native cleanup and collective success,
/// so a returned error leaves every destination unchanged. The file is not
/// overwritten or resized.
pub fn read_hdf5_collection<P: AsRef<Path>, T: IoElement, const N: usize, const M: usize>(
    path: P,
    comm: &CartesianCommunicator,
    views: &mut [PencilArrayViewMut<'_, T, N, M>],
) -> Result<(), CollectionIoError> {
    header(comm, HDF5_READ, views.len())?;
    let (dims, len, staging) = prepare_read(comm, path.as_ref(), HDF5_READ, false, views)?;
    let target = ReadTarget::collection(views, dims, len, staging);
    crate::hdf5_io::read_hdf5_target(path, target, true)?;
    Ok(())
}
#[cfg(feature = "parallel-hdf5")]
/// Read a combined HDF5 collection with explicit controls; members are decoded
/// directly from one byte-staging payload after native cleanup, so a returned
/// error leaves every destination unchanged.
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
    let (dims, len, staging) = prepare_read(comm, path.as_ref(), HDF5_READ_OPTIONS, false, views)?;
    let target = ReadTarget::collection(views, dims, len, staging);
    crate::hdf5_io::read_hdf5_target_options(path, target, o.settings(), true)?;
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
    crate::hdf5_io::validate_controls(
        comm,
        o.settings(),
        N + first.extra_shape().dimensions().len() + 1,
        T::WIDTH,
    )?;
    let a = named_prepare(comm, path.as_ref(), HDF5_NAMED_WRITE_OPTIONS, false, views)?;
    crate::hdf5_io::write_hdf5_named_with_options_inner(
        path,
        name,
        a.view(),
        crate::hdf5_io::Hdf5Settings {
            options_agreed: true,
            ..o.settings()
        },
        true,
    )?;
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
    crate::hdf5_io::validate_controls(
        comm,
        o.settings(),
        N + first.extra_shape().dimensions().len() + 1,
        T::WIDTH,
    )?;
    let a = named_prepare(comm, path.as_ref(), HDF5_NAMED_APPEND_OPTIONS, false, views)?;
    crate::hdf5_io::append_hdf5_named_with_options_inner(
        path,
        name,
        a.view(),
        crate::hdf5_io::Hdf5Settings {
            options_agreed: true,
            ..o.settings()
        },
        true,
    )?;
    Ok(())
}
#[cfg(feature = "parallel-hdf5")]
/// Read a named combined HDF5 collection atomically into member destinations,
/// decoding directly from one byte-staging payload after native cleanup.
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
    crate::hdf5_io::validate_controls(
        comm,
        o.settings(),
        N + first.extra_shape().dimensions().len() + 1,
        T::WIDTH,
    )?;
    let (dims, len, staging) =
        prepare_read(comm, path.as_ref(), HDF5_NAMED_READ_OPTIONS, false, views)?;
    let target = ReadTarget::collection(views, dims, len, staging);
    crate::hdf5_io::read_named_with_options(
        path.as_ref(),
        name.as_ref(),
        target,
        crate::hdf5_io::Hdf5Settings {
            options_agreed: true,
            ..o.settings()
        },
        73,
        true,
    )?;
    Ok(())
}
