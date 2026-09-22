use crate::mpi_io::{agree_phase, descriptor_agreement};
use crate::{FORMAT_VERSION, IO_NAMESPACE, IoElement, IoError};
use mpi::{
    collective::{CommunicatorCollectives, SystemOperation},
    topology::CartesianCommunicator,
    traits::AsRaw,
};
use pencil_array::{ExtraShape, Pencil, PencilArray, PencilArrayView, PencilArrayViewMut};
use std::path::Path;

/// Collection validation and payload errors. Failed writes are not transactions.
#[non_exhaustive]
#[derive(Debug, Clone, thiserror::Error)]
pub enum CollectionIoError {
    /// At least one member is required.
    #[error("collection is empty")]
    Empty,
    /// A member has incompatible metadata or topology.
    #[error("collection member {index} has incompatible metadata")]
    Incompatible {
        /// First failing member index across ranks.
        index: usize,
    },
    /// A combined payload operation or its preparation failed.
    #[error("collection I/O: {0}")]
    Io(#[from] IoError),
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
    header(comm, operation, count)?;
    let all = members.clone();
    let (first, extra, data) = members.next().expect("nonempty header");
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
        let mut dims = Vec::new();
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
        for (_, _, values) in all {
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

/// Write one `[component, extra..., spatial...]` MPI payload from separate arrays.
/// Every rank participates. Source arrays are preserved; no rollback of a failed file is promised.
pub fn write_mpi_collection<P: AsRef<Path>, T: IoElement, const N: usize, const M: usize>(
    path: P,
    comm: &CartesianCommunicator,
    views: &[PencilArrayView<'_, T, N, M>],
) -> Result<(), CollectionIoError> {
    let array = prepare(
        comm,
        path.as_ref(),
        61,
        views.len(),
        views
            .iter()
            .map(|v| (v.pencil(), v.extra_shape(), v.as_slice())),
    )?;
    Ok(crate::write_mpi(path, array.view())?)
}
/// Read one MPI collection, staging every member and cleanup before any destination changes.
pub fn read_mpi_collection<P: AsRef<Path>, T: IoElement, const N: usize, const M: usize>(
    path: P,
    comm: &CartesianCommunicator,
    views: &mut [PencilArrayViewMut<'_, T, N, M>],
) -> Result<(), CollectionIoError> {
    let mut array = prepare(
        comm,
        path.as_ref(),
        62,
        views.len(),
        views
            .iter()
            .map(|v| (v.pencil(), v.extra_shape(), v.as_slice())),
    )?;
    crate::read_mpi(path, array.view_mut())?;
    let len = views[0].len();
    for (i, v) in views.iter_mut().enumerate() {
        v.as_mut_slice()
            .copy_from_slice(&array.as_slice()[i * len..(i + 1) * len]);
    }
    Ok(())
}
/// Write one native HDF5 dataset with a leading component axis.
#[cfg(feature = "parallel-hdf5")]
pub fn write_hdf5_collection<P: AsRef<Path>, T: IoElement, const N: usize, const M: usize>(
    path: P,
    comm: &CartesianCommunicator,
    views: &[PencilArrayView<'_, T, N, M>],
) -> Result<(), CollectionIoError> {
    let array = prepare(
        comm,
        path.as_ref(),
        63,
        views.len(),
        views
            .iter()
            .map(|v| (v.pencil(), v.extra_shape(), v.as_slice())),
    )?;
    Ok(crate::write_hdf5(path, array.view())?)
}
/// Read one HDF5 collection without changing any member on error.
#[cfg(feature = "parallel-hdf5")]
pub fn read_hdf5_collection<P: AsRef<Path>, T: IoElement, const N: usize, const M: usize>(
    path: P,
    comm: &CartesianCommunicator,
    views: &mut [PencilArrayViewMut<'_, T, N, M>],
) -> Result<(), CollectionIoError> {
    let mut array = prepare(
        comm,
        path.as_ref(),
        64,
        views.len(),
        views
            .iter()
            .map(|v| (v.pencil(), v.extra_shape(), v.as_slice())),
    )?;
    crate::read_hdf5(path, array.view_mut())?;
    let len = views[0].len();
    for (i, v) in views.iter_mut().enumerate() {
        v.as_mut_slice()
            .copy_from_slice(&array.as_slice()[i * len..(i + 1) * len]);
    }
    Ok(())
}
