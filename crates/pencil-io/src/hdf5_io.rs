use std::ffi::{CStr, CString};
use std::ops::Range;
use std::path::Path;

use hdf5_metno_sys::h5::hsize_t;
use mpi::collective::{CommunicatorCollectives, SystemOperation};
use mpi::raw::AsRaw;
use mpi::topology::Communicator;
use pencil_array::{PencilArrayView, PencilArrayViewMut};

use crate::ffi;
use crate::ffi::hdf5 as native;
use crate::format::{IoElement, pack_view, prepare_physical_values};
use crate::mpi_io::{
    CommGuard, abort_unrecoverable, aggregate_cleanup_result, agree_phase, collective_state,
    descriptor_agreement, duplicate_comm,
};
use crate::{
    COMMIT_MARKER, FORMAT_VERSION, INCOMPLETE_MARKER, IoError, MAX_PROTOCOL_RANK, NamedIoError,
    OP_APPEND_HDF5_NAMED, OP_READ_HDF5, OP_READ_HDF5_NAMED, OP_WRITE_HDF5, OP_WRITE_HDF5_NAMED,
};

const MAX_NAME: usize = 1024;

fn named_link<C: CommunicatorCollectives>(comm: &C, name: &str) -> Result<CString, NamedIoError> {
    let valid = !name.is_empty() && name.len() <= MAX_NAME && !name.as_bytes().contains(&0);
    let n = name.len() as i64;
    let mut lo = 0;
    let mut hi = 0;
    comm.all_reduce_into(&n, &mut lo, SystemOperation::min());
    comm.all_reduce_into(&n, &mut hi, SystemOperation::max());
    if lo != hi {
        return Err(NamedIoError::Io(IoError::CollectiveDescriptorMismatch));
    }
    let ok = i32::from(valid);
    let mut all_ok = 0;
    comm.all_reduce_into(&ok, &mut all_ok, SystemOperation::min());
    if all_ok == 0 {
        return Err(NamedIoError::InvalidName);
    }
    let ranks = usize::try_from(comm.size())
        .map_err(|_| NamedIoError::Io(IoError::SizeLimit { what: "MPI ranks" }))?;
    let bytes_len = name
        .len()
        .checked_mul(ranks)
        .ok_or(NamedIoError::Io(IoError::SizeLimit {
            what: "named descriptor",
        }))?;
    if bytes_len > 64 * 1024 * 1024 {
        return Err(IoError::SizeLimit {
            what: "named descriptor allgather",
        }
        .into());
    }
    let mut bytes = Vec::new();
    let reserve = bytes.try_reserve_exact(bytes_len);
    if let Err(agreement) = agree_phase(
        comm,
        reserve.is_ok(),
        "named descriptor allgather allocation",
    ) {
        return Err(NamedIoError::Io(if reserve.is_err() {
            IoError::AllocationFailed {
                requested: bytes_len,
            }
        } else {
            agreement
        }));
    }
    bytes.resize(bytes_len, 0);
    comm.all_gather_into(name.as_bytes(), &mut bytes);
    if bytes.chunks_exact(name.len()).any(|x| x != name.as_bytes()) {
        return Err(NamedIoError::Io(IoError::CollectiveDescriptorMismatch));
    }
    let hex_len = name
        .len()
        .checked_mul(2)
        .and_then(|len| len.checked_add(1))
        .ok_or(NamedIoError::Io(IoError::SizeLimit {
            what: "named descriptor",
        }))?;
    let mut hex = Vec::new();
    let reserve = hex.try_reserve_exact(hex_len);
    if let Err(agreement) = agree_phase(comm, reserve.is_ok(), "named link allocation") {
        return Err(NamedIoError::Io(if reserve.is_err() {
            IoError::AllocationFailed { requested: hex_len }
        } else {
            agreement
        }));
    }
    for byte in name.bytes() {
        hex.push(b"0123456789abcdef"[(byte >> 4) as usize]);
        hex.push(b"0123456789abcdef"[(byte & 15) as usize]);
    }
    hex.push(0);
    CString::from_vec_with_nul(hex).map_err(|_| NamedIoError::InvalidName)
}

const GROUP_NAME: &[u8] = b"/pencil_io_v1\0";
const DATASET_NAME: &[u8] = b"data\0";
const NAMED_GROUP: &[u8] = b"/pencil_io_named_v1\0";
const NAME_ATTR: &[u8] = b"pencil_io_original_name\0";
const ATTR_VERSION: &[u8] = b"pencil_io_version\0";
const ATTR_COMMIT: &[u8] = b"pencil_io_commit\0";
const ATTR_N: &[u8] = b"pencil_io_n\0";
const ATTR_TYPE: &[u8] = b"pencil_io_type\0";
const ATTR_WIDTH: &[u8] = b"pencil_io_width\0";
const ATTR_EXTRA_RANK: &[u8] = b"pencil_io_extra_rank\0";
const ATTR_EXTRA: &[u8] = b"pencil_io_extra_shape\0";
const ATTR_GLOBAL: &[u8] = b"pencil_io_global_shape\0";
const ATTR_GRID: &[u8] = b"pencil_io_writer_grid\0";
const ATTR_PERM: &[u8] = b"pencil_io_writer_permutation\0";

/// Writes one view collectively with the optional parallel HDF5 backend.
///
/// The dataset is `/pencil_io_v1/data`. Its dimensions are logical
/// `[extra..., spatial...]`, and its HDF5 datatype is explicit little-endian;
/// complex values use a packed compound with fields `r` and `i`.
///
/// The call is collective and must not overlap another operation on the
/// topology communicator.
pub fn write_hdf5<P, T, const N: usize, const M: usize>(
    path: P,
    view: PencilArrayView<'_, T, N, M>,
) -> Result<(), IoError>
where
    P: AsRef<Path>,
    T: IoElement,
{
    write_hdf5_inner(
        path,
        view,
        false,
        cstr(GROUP_NAME),
        cstr(DATASET_NAME),
        false,
        None,
        OP_WRITE_HDF5,
    )
}

#[allow(clippy::too_many_arguments)]
fn write_hdf5_inner<P, T, const N: usize, const M: usize>(
    path: P,
    view: PencilArrayView<'_, T, N, M>,
    inject_post_cleanup_failure: bool,
    group_name: &CStr,
    dataset_name: &CStr,
    update: bool,
    original_name: Option<&[u8]>,
    operation: u64,
) -> Result<(), IoError>
where
    P: AsRef<Path>,
    T: IoElement,
{
    let comm = view.pencil().topology().communicator();
    descriptor_agreement(
        comm,
        path.as_ref(),
        operation,
        view.pencil().global_shape(),
        view.extra_shape().dimensions(),
        view.pencil().topology().process_grid(),
        view.pencil().permutation().axes(),
        T::CODE,
        T::WIDTH,
    )?;
    let path = path_cstring(path.as_ref(), comm)?;
    let packed_result = pack_view(&view);
    let layout_result = build_layout(
        view.pencil().global_shape(),
        view.extra_shape().dimensions(),
        &view.local_spatial_shape(),
        view.pencil().local_ranges(),
        view.len(),
    );
    let prep_error = first_error2(&packed_result, &layout_result);
    if let Err(agreement) = agree_phase(comm, prep_error.is_none(), "HDF5 write preparation") {
        return Err(prep_error.unwrap_or(agreement));
    }
    let mut packed = packed_result.expect("HDF5 preparation agreement established");
    let layout = layout_result.expect("HDF5 preparation agreement established");

    let duplicate = duplicate_comm(comm)?;
    let fapl = match prepare_hdf5_fapl(comm, duplicate.raw) {
        Ok(fapl) => fapl,
        Err(error) => return Err(cleanup_comm_only(comm, duplicate, error)),
    };
    let opened = match open_hdf5_mode(comm, fapl, &path, !update, update) {
        Ok(file) => file,
        Err(error) => return Err(cleanup_comm_only(comm, duplicate, error)),
    };
    let mut resources = Hdf5Resources::default();

    let group = match collective_handle_phase(
        comm,
        if update {
            native::group_open(opened, group_name)
        } else {
            native::group_create(opened, group_name)
        },
        "HDF5 group create",
    ) {
        Ok(group) => group,
        Err(error) => return Err(cleanup_ready(comm, duplicate, opened, resources, error)),
    };
    resources.group = Some(group);

    let (datatype, error) = local_handle_phase(
        comm,
        native::type_from_kind(T::CODE, T::WIDTH, duplicate.raw),
        "HDF5 datatype create",
    );
    resources.datatype = datatype;
    if let Some(error) = error {
        return Err(cleanup_ready(comm, duplicate, opened, resources, error));
    }
    let datatype = resources.datatype.expect("datatype creation agreement");

    let (dataspace, error) = local_handle_phase(
        comm,
        native::dataspace_simple(&layout.global),
        "HDF5 dataspace create",
    );
    resources.dataspace = dataspace;
    if let Some(error) = error {
        return Err(cleanup_ready(comm, duplicate, opened, resources, error));
    }
    let dataspace = resources.dataspace.expect("dataspace creation agreement");

    let dataset = match collective_handle_phase(
        comm,
        native::dataset_create(
            resources.group.expect("group creation agreement"),
            dataset_name,
            datatype,
            dataspace,
        ),
        "HDF5 dataset create",
    ) {
        Ok(dataset) => dataset,
        Err(error) => return Err(cleanup_ready(comm, duplicate, opened, resources, error)),
    };
    resources.dataset = Some(dataset);

    if let Some(name) = original_name {
        if let Err(error) = write_original_name(
            comm,
            duplicate.raw,
            dataset,
            name,
            "HDF5 original-name attribute",
        ) {
            return Err(cleanup_ready(comm, duplicate, opened, resources, error));
        }
    }
    if let Err(error) = write_metadata(
        comm,
        dataset,
        view.pencil().topology().process_grid(),
        view.pencil().permutation().axes(),
        view.pencil().global_shape(),
        view.extra_shape().dimensions(),
        T::CODE,
        T::WIDTH,
    ) {
        return Err(cleanup_ready(comm, duplicate, opened, resources, error));
    }
    if let Err(error) = phase_code(
        comm,
        native::file_flush(opened),
        "HDF5 initial metadata flush",
    ) {
        return Err(cleanup_ready(comm, duplicate, opened, resources, error));
    }

    let file_result = native::dataset_space(dataset);
    let file_space = file_result.as_ref().ok().copied();
    let mem_result = if layout.empty && !layout.global.contains(&0) {
        native::dataspace_simple(&layout.global)
    } else {
        native::dataspace_simple(&layout.local)
    };
    let mem_space = mem_result.as_ref().ok().copied();
    let space_error = file_result
        .as_ref()
        .err()
        .map(|&code| hdf5_error("H5Dget_space", code))
        .or_else(|| {
            mem_result
                .as_ref()
                .err()
                .map(|&code| hdf5_error("H5Screate memory space", code))
        });
    if let Err(agreement) = agree_phase(
        comm,
        file_space.is_some() && mem_space.is_some(),
        "HDF5 hyperslab spaces",
    ) {
        return Err(cleanup_ready(
            comm,
            duplicate,
            opened,
            Hdf5Resources {
                file_space,
                mem_space,
                ..resources
            },
            space_error.unwrap_or(agreement),
        ));
    }
    if let Some(error) = space_error {
        return Err(cleanup_ready(
            comm,
            duplicate,
            opened,
            Hdf5Resources {
                file_space,
                mem_space,
                ..resources
            },
            error,
        ));
    }
    resources.file_space = file_space;
    resources.mem_space = mem_space;
    let file_space = resources.file_space.expect("file-space agreement");
    let mem_space = resources.mem_space.expect("memory-space agreement");
    if let Err(error) = select_spaces(comm, file_space, mem_space, &layout) {
        return Err(cleanup_ready(comm, duplicate, opened, resources, error));
    }
    let (xfer, error) = local_handle_phase(
        comm,
        native::xfer_create_collective(duplicate.raw),
        "HDF5 collective transfer plist",
    );
    resources.xfer = xfer;
    if let Some(error) = error {
        return Err(cleanup_ready(comm, duplicate, opened, resources, error));
    }
    let xfer = resources.xfer.expect("transfer-property agreement");
    // HDF5 1.10's MPIO driver cannot issue a zero-byte collective write to
    // a genuinely zero-extent dataset (it reports an address overflow). All
    // ranks still reach the phase agreement and surrounding collective flush;
    // there is no payload operation to perform in this case.
    let io_code = if layout.global.contains(&0) {
        0
    } else {
        native::dataset_io(
            dataset,
            datatype,
            mem_space,
            file_space,
            xfer,
            &mut packed,
            true,
        )
    };
    if let Err(agreement) = agree_phase(comm, io_code >= 0, "HDF5 collective dataset write") {
        let primary = if io_code < 0 {
            hdf5_error("H5Dwrite", io_code)
        } else {
            agreement
        };
        return Err(cleanup_ready(comm, duplicate, opened, resources, primary));
    }
    if let Err(error) = phase_code(comm, native::file_flush(opened), "HDF5 payload flush") {
        return Err(cleanup_ready(comm, duplicate, opened, resources, error));
    }

    if write_existing_attr(comm, dataset, ATTR_COMMIT, COMMIT_MARKER).is_err() {
        return Err(cleanup_ready(
            comm,
            duplicate,
            opened,
            resources,
            IoError::CommitUncertain {
                stage: "HDF5 commit marker",
            },
        ));
    }
    if phase_code(comm, native::file_flush(opened), "HDF5 commit marker flush").is_err() {
        return Err(cleanup_ready(
            comm,
            duplicate,
            opened,
            resources,
            IoError::CommitUncertain {
                stage: "HDF5 commit marker flush",
            },
        ));
    }

    let cleanup = finish_hdf5(comm, duplicate, opened, resources);
    let injected = if inject_post_cleanup_failure && comm.rank() == 0 {
        Some(IoError::CommitUncertain {
            stage: "test post-marker cleanup result",
        })
    } else {
        None
    };
    if let Some(error) = aggregate_cleanup_result(comm, cleanup.or(injected)) {
        return Err(error);
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn write_hdf5_with_postmarker_uncertainty<P, T, const N: usize, const M: usize>(
    path: P,
    view: PencilArrayView<'_, T, N, M>,
) -> Result<(), IoError>
where
    P: AsRef<Path>,
    T: IoElement,
{
    write_hdf5_inner(
        path,
        view,
        true,
        cstr(GROUP_NAME),
        cstr(DATASET_NAME),
        false,
        None,
        OP_WRITE_HDF5,
    )
}

/// Creates an exclusive named HDF5 container and writes its first dataset.
/// UTF-8 keys (1..=1024 bytes, no NUL) are hex-encoded as link components.
pub fn write_hdf5_named<P, S, T, const N: usize, const M: usize>(
    path: P,
    name: S,
    view: PencilArrayView<'_, T, N, M>,
) -> Result<(), NamedIoError>
where
    P: AsRef<Path>,
    S: AsRef<str>,
    T: IoElement,
{
    write_named(path.as_ref(), name.as_ref(), view, false)
}

pub(crate) fn write_named<T: IoElement, const N: usize, const M: usize>(
    path: &Path,
    name: &str,
    view: PencilArrayView<'_, T, N, M>,
    inject: bool,
) -> Result<(), NamedIoError> {
    let comm = view.pencil().topology().communicator();
    descriptor_agreement(
        comm,
        path,
        OP_WRITE_HDF5_NAMED,
        view.pencil().global_shape(),
        view.extra_shape().dimensions(),
        view.pencil().topology().process_grid(),
        view.pencil().permutation().axes(),
        T::CODE,
        T::WIDTH,
    )
    .map_err(NamedIoError::Io)?;
    let link = named_link(comm, name)?;
    let group = cstr(NAMED_GROUP);
    write_hdf5_inner(
        path,
        view,
        inject,
        group,
        &link,
        false,
        Some(name.as_bytes()),
        OP_WRITE_HDF5_NAMED,
    )
    .map_err(NamedIoError::Io)
}

/// Appends a named dataset to an existing parallel HDF5 container.
///
/// The operation fails with [`NamedIoError::DuplicateName`] without changing
/// the container when `name` already exists.
pub fn append_hdf5_named<P, S, T, const N: usize, const M: usize>(
    path: P,
    name: S,
    view: PencilArrayView<'_, T, N, M>,
) -> Result<(), NamedIoError>
where
    P: AsRef<Path>,
    S: AsRef<str>,
    T: IoElement,
{
    let comm = view.pencil().topology().communicator();
    descriptor_agreement(
        comm,
        path.as_ref(),
        OP_APPEND_HDF5_NAMED,
        view.pencil().global_shape(),
        view.extra_shape().dimensions(),
        view.pencil().topology().process_grid(),
        view.pencil().permutation().axes(),
        T::CODE,
        T::WIDTH,
    )
    .map_err(NamedIoError::Io)?;
    let link = named_link(comm, name.as_ref())?;
    let pc = path_cstring(path.as_ref(), comm).map_err(NamedIoError::Io)?;
    let dup = duplicate_comm(comm).map_err(NamedIoError::Io)?;
    let fapl = match prepare_hdf5_fapl(comm, dup.raw) {
        Ok(fapl) => fapl,
        Err(error) => return Err(NamedIoError::Io(cleanup_comm_only(comm, dup, error))),
    };
    let file = match open_hdf5_mode(comm, fapl, &pc, false, true) {
        Ok(file) => file,
        Err(error) => return Err(NamedIoError::Io(cleanup_comm_only(comm, dup, error))),
    };
    let group = match collective_handle_phase(
        comm,
        native::group_open(file, cstr(NAMED_GROUP)),
        "HDF5 named group open",
    ) {
        Ok(group) => group,
        Err(error) => {
            return Err(NamedIoError::Io(cleanup_ready(
                comm,
                dup,
                file,
                Hdf5Resources::default(),
                error,
            )));
        }
    };
    let exists_result = native::link_exists(group, &link);
    let query_agreement = agree_phase(comm, exists_result.is_ok(), "HDF5 named link query");
    let cleanup = finish_hdf5(
        comm,
        dup,
        file,
        Hdf5Resources {
            group: Some(group),
            ..Default::default()
        },
    );
    let exists = match query_agreement {
        Ok(()) => {
            exists_result.map_err(|c| NamedIoError::Io(hdf5_error("HDF5 named link query", c)))?
        }
        Err(error) => return Err(NamedIoError::Io(error)),
    };
    if let Some(error) = cleanup {
        return Err(NamedIoError::Io(error));
    }
    let (all_exist, mixed) = collective_state(comm, exists);
    if mixed {
        return Err(IoError::CollectiveDescriptorMismatch.into());
    }
    if all_exist {
        return Err(NamedIoError::DuplicateName);
    }
    write_hdf5_inner(
        path,
        view,
        false,
        cstr(NAMED_GROUP),
        &link,
        true,
        Some(name.as_ref().as_bytes()),
        OP_APPEND_HDF5_NAMED,
    )
    .map_err(NamedIoError::Io)
}

/// Reads a named dataset collectively without modifying the destination on failure.
pub fn read_hdf5_named<P, S, T, const N: usize, const M: usize>(
    path: P,
    name: S,
    view: PencilArrayViewMut<'_, T, N, M>,
) -> Result<(), NamedIoError>
where
    P: AsRef<Path>,
    S: AsRef<str>,
    T: IoElement,
{
    read_named(path.as_ref(), name.as_ref(), view, false)
}

pub(crate) fn read_named<T: IoElement, const N: usize, const M: usize>(
    path: &Path,
    name: &str,
    view: PencilArrayViewMut<'_, T, N, M>,
    inject: bool,
) -> Result<(), NamedIoError> {
    let comm = view.pencil().topology().communicator();
    descriptor_agreement(
        comm,
        path,
        OP_READ_HDF5_NAMED,
        view.pencil().global_shape(),
        view.extra_shape().dimensions(),
        view.pencil().topology().process_grid(),
        view.pencil().permutation().axes(),
        T::CODE,
        T::WIDTH,
    )
    .map_err(NamedIoError::Io)?;
    let link = named_link(comm, name)?;
    read_hdf5_inner(
        path,
        view,
        inject,
        cstr(NAMED_GROUP),
        &link,
        Some(name.as_bytes()),
        OP_READ_HDF5_NAMED,
    )
    .map_err(|error| match error {
        IoError::MetadataMismatch {
            field: "named dataset missing",
        } => NamedIoError::NotFound,
        other => NamedIoError::Io(other),
    })
}

/// Reads the version-1 HDF5 dataset collectively without modifying the destination on failure.
pub fn read_hdf5<P, T, const N: usize, const M: usize>(
    path: P,
    view: PencilArrayViewMut<'_, T, N, M>,
) -> Result<(), IoError>
where
    P: AsRef<Path>,
    T: IoElement,
{
    read_hdf5_inner(
        path,
        view,
        false,
        cstr(GROUP_NAME),
        cstr(DATASET_NAME),
        None,
        OP_READ_HDF5,
    )
}

fn read_hdf5_inner<P, T, const N: usize, const M: usize>(
    path: P,
    mut view: PencilArrayViewMut<'_, T, N, M>,
    inject_post_cleanup_failure: bool,
    group_name: &CStr,
    dataset_name: &CStr,
    expected_name: Option<&[u8]>,
    operation: u64,
) -> Result<(), IoError>
where
    P: AsRef<Path>,
    T: IoElement,
{
    let comm = view.pencil().topology().communicator();
    descriptor_agreement(
        comm,
        path.as_ref(),
        operation,
        view.pencil().global_shape(),
        view.extra_shape().dimensions(),
        view.pencil().topology().process_grid(),
        view.pencil().permutation().axes(),
        T::CODE,
        T::WIDTH,
    )?;
    let path = path_cstring(path.as_ref(), comm)?;
    let layout_result = build_layout(
        view.pencil().global_shape(),
        view.extra_shape().dimensions(),
        &view.local_spatial_shape(),
        view.pencil().local_ranges(),
        view.len(),
    );
    let bytes = view.len().checked_mul(T::WIDTH).ok_or(IoError::SizeLimit {
        what: "HDF5 staging bytes",
    });
    let mut staging = Vec::new();
    let staging_result = match bytes {
        Ok(bytes) => staging
            .try_reserve_exact(bytes)
            .map(|_| bytes)
            .map_err(|_| IoError::AllocationFailed { requested: bytes }),
        Err(error) => Err(error),
    };
    let prep_error = first_error2(&layout_result, &staging_result);
    if let Err(agreement) = agree_phase(comm, prep_error.is_none(), "HDF5 read preparation") {
        return Err(prep_error.unwrap_or(agreement));
    }
    let layout = layout_result.expect("HDF5 preparation agreement established");
    staging.resize(
        staging_result.expect("HDF5 preparation agreement established"),
        0,
    );

    let duplicate = duplicate_comm(comm)?;
    let fapl = match prepare_hdf5_fapl(comm, duplicate.raw) {
        Ok(fapl) => fapl,
        Err(error) => return Err(cleanup_comm_only(comm, duplicate, error)),
    };
    let opened = match open_hdf5(comm, fapl, &path, false) {
        Ok(file) => file,
        Err(error) => return Err(cleanup_comm_only(comm, duplicate, error)),
    };
    let mut resources = Hdf5Resources::default();

    let group = match collective_handle_phase(
        comm,
        native::group_open(opened, group_name),
        "HDF5 group open",
    ) {
        Ok(group) => group,
        Err(error) => return Err(cleanup_ready(comm, duplicate, opened, resources, error)),
    };
    resources.group = Some(group);
    if expected_name.is_some() {
        let exists = native::link_exists(group, dataset_name);
        if let Err(error) = agree_phase(comm, exists.is_ok(), "HDF5 named lookup") {
            return Err(cleanup_ready(comm, duplicate, opened, resources, error));
        }
        let present = exists.expect("named lookup agreed");
        if agree_phase(comm, present, "HDF5 named presence").is_err() {
            return Err(cleanup_ready(
                comm,
                duplicate,
                opened,
                resources,
                IoError::MetadataMismatch {
                    field: "named dataset missing",
                },
            ));
        }
    }
    let dataset = match collective_handle_phase(
        comm,
        native::dataset_open(group, dataset_name),
        "HDF5 dataset open",
    ) {
        Ok(dataset) => dataset,
        Err(error) => return Err(cleanup_ready(comm, duplicate, opened, resources, error)),
    };
    resources.dataset = Some(dataset);
    if let Some(expected) = expected_name {
        let actual = match read_original_name(comm, dataset, "HDF5 original-name metadata read") {
            Ok(actual) => actual,
            Err(error) => return Err(cleanup_ready(comm, duplicate, opened, resources, error)),
        };
        let ok = actual == expected;
        if let Err(agreement) = agree_phase(comm, ok, "HDF5 original-name metadata") {
            return Err(cleanup_ready(
                comm,
                duplicate,
                opened,
                resources,
                if ok {
                    agreement
                } else {
                    IoError::MetadataMismatch {
                        field: "original name",
                    }
                },
            ));
        }
    }

    macro_rules! read_attr {
        ($name:expr, $expected:expr, $max:expr, $phase:expr) => {
            match read_attr_phase(comm, duplicate.raw, dataset, $name, $expected, $max, $phase) {
                Ok(values) => values,
                Err(error) => return Err(cleanup_ready(comm, duplicate, opened, resources, error)),
            }
        };
    }
    let version = read_attr!(ATTR_VERSION, Some(1), 1, "HDF5 version metadata");
    let commit = read_attr!(ATTR_COMMIT, Some(1), 1, "HDF5 commit metadata");
    let n = read_attr!(ATTR_N, Some(1), 1, "HDF5 rank metadata");
    let type_code = read_attr!(ATTR_TYPE, Some(1), 1, "HDF5 type metadata");
    let width = read_attr!(ATTR_WIDTH, Some(1), 1, "HDF5 width metadata");
    let extra_rank = read_attr!(ATTR_EXTRA_RANK, Some(1), 1, "HDF5 extra-rank metadata");
    let metadata_ok = version == [FORMAT_VERSION]
        && n == [N as u64]
        && type_code == [T::CODE]
        && width == [T::WIDTH as u64]
        && usize::try_from(extra_rank[0]).ok() == Some(view.extra_shape().dimensions().len())
        && extra_rank[0] <= MAX_PROTOCOL_RANK as u64;
    let commit_state = if commit == [COMMIT_MARKER] {
        Ok(())
    } else if commit == [INCOMPLETE_MARKER] {
        Err(IoError::IncompleteFile)
    } else {
        Err(IoError::InvalidFile {
            reason: "HDF5 commit marker",
        })
    };
    let scalar_ok = metadata_ok && commit_state.is_ok();
    if let Err(agreement) = agree_phase(comm, scalar_ok, "HDF5 scalar metadata validation") {
        let local_error = if !metadata_ok {
            IoError::MetadataMismatch {
                field: "version, rank, or type",
            }
        } else {
            commit_state.err().unwrap_or(agreement)
        };
        return Err(cleanup_ready(
            comm,
            duplicate,
            opened,
            resources,
            local_error,
        ));
    }

    let extra = if extra_rank[0] == 0 {
        match attribute_exists_phase(comm, dataset, ATTR_EXTRA, "HDF5 zero-extra metadata") {
            Ok(false) => Vec::new(),
            Ok(true) => {
                return Err(cleanup_ready(
                    comm,
                    duplicate,
                    opened,
                    resources,
                    IoError::MetadataMismatch {
                        field: "extra shape",
                    },
                ));
            }
            Err(error) => return Err(cleanup_ready(comm, duplicate, opened, resources, error)),
        }
    } else {
        read_attr!(
            ATTR_EXTRA,
            Some(extra_rank[0] as usize),
            extra_rank[0] as usize,
            "HDF5 extra shape metadata"
        )
    };
    let global = read_attr!(ATTR_GLOBAL, Some(N), N, "HDF5 global shape metadata");
    let writer_grid = read_attr!(
        ATTR_GRID,
        None,
        MAX_PROTOCOL_RANK,
        "HDF5 writer grid metadata"
    );
    let writer_perm = read_attr!(ATTR_PERM, Some(N), N, "HDF5 writer permutation metadata");
    let metadata_ok = valid_dimensions(&global)
        && global
            .iter()
            .copied()
            .eq(view.pencil().global_shape().iter().map(|&v| v as u64))
        && extra
            .iter()
            .copied()
            .eq(view.extra_shape().dimensions().iter().map(|&v| v as u64))
        && valid_extents(&writer_grid)
        && is_permutation(&writer_perm, N);
    if let Err(agreement) = agree_phase(comm, metadata_ok, "HDF5 shape metadata validation") {
        return Err(cleanup_ready(
            comm,
            duplicate,
            opened,
            resources,
            if metadata_ok {
                agreement
            } else {
                IoError::MetadataMismatch {
                    field: "shape or writer provenance",
                }
            },
        ));
    }

    let (dataset_type, error) = local_handle_phase(
        comm,
        native::dataset_type(dataset),
        "HDF5 dataset type query",
    );
    resources.datatype = dataset_type;
    if let Some(error) = error {
        return Err(cleanup_ready(comm, duplicate, opened, resources, error));
    }
    let datatype = resources.datatype.expect("dataset type agreement");
    let type_result = native::type_matches(datatype, T::CODE, T::WIDTH, duplicate.raw);
    if let Err(agreement) = agree_phase(comm, type_result.is_ok(), "HDF5 datatype validation query")
    {
        return Err(cleanup_ready(
            comm,
            duplicate,
            opened,
            resources,
            type_result
                .err()
                .map(|code| hdf5_error("HDF5 datatype validation query", code))
                .unwrap_or(agreement),
        ));
    }
    let type_ok = type_result.expect("datatype validation query agreement");
    if let Err(agreement) = collective_bool(comm, type_ok, "HDF5 datatype validation") {
        return Err(cleanup_ready(comm, duplicate, opened, resources, agreement));
    }
    if !type_ok {
        return Err(cleanup_ready(
            comm,
            duplicate,
            opened,
            resources,
            IoError::MetadataMismatch {
                field: "dataset datatype",
            },
        ));
    }

    let (file_space, error) = local_handle_phase(
        comm,
        native::dataset_space(dataset),
        "HDF5 dataset shape query",
    );
    resources.file_space = file_space;
    if let Some(error) = error {
        return Err(cleanup_ready(comm, duplicate, opened, resources, error));
    }
    let file_space = resources.file_space.expect("dataset-space agreement");
    let shape_result = native::space_shape(file_space);
    if let Err(agreement) = agree_phase(comm, shape_result.is_ok(), "HDF5 dataset dimensions query")
    {
        return Err(cleanup_ready(
            comm,
            duplicate,
            opened,
            resources,
            shape_result
                .err()
                .map(|code| hdf5_error("H5Sget_simple_extent_dims", code))
                .unwrap_or(agreement),
        ));
    }
    let shape_ok = shape_result.expect("dataset dimensions agreement") == layout.global;
    if let Err(agreement) = collective_bool(comm, shape_ok, "HDF5 dataset dimensions") {
        return Err(cleanup_ready(comm, duplicate, opened, resources, agreement));
    }
    if !shape_ok {
        return Err(cleanup_ready(
            comm,
            duplicate,
            opened,
            resources,
            IoError::MetadataMismatch {
                field: "dataset dimensions",
            },
        ));
    }

    let mem_result = if layout.empty && !layout.global.contains(&0) {
        native::dataspace_simple(&layout.global)
    } else {
        native::dataspace_simple(&layout.local)
    };
    let (mem_space, error) = local_handle_phase(comm, mem_result, "HDF5 read memory space");
    resources.mem_space = mem_space;
    if let Some(error) = error {
        return Err(cleanup_ready(comm, duplicate, opened, resources, error));
    }
    let mem_space = resources.mem_space.expect("memory-space agreement");
    if let Err(error) = select_spaces(comm, file_space, mem_space, &layout) {
        return Err(cleanup_ready(comm, duplicate, opened, resources, error));
    }

    let (xfer, error) = local_handle_phase(
        comm,
        native::xfer_create_collective(duplicate.raw),
        "HDF5 collective transfer plist",
    );
    resources.xfer = xfer;
    if let Some(error) = error {
        return Err(cleanup_ready(comm, duplicate, opened, resources, error));
    }
    let xfer = resources.xfer.expect("transfer-property agreement");
    // See the write-side zero-extent guard above. The branch is descriptor
    // identical on every rank, while the phase agreement remains collective.
    let io_code = if layout.global.contains(&0) {
        0
    } else {
        native::dataset_io(
            dataset,
            datatype,
            mem_space,
            file_space,
            xfer,
            &mut staging,
            false,
        )
    };
    if let Err(agreement) = agree_phase(comm, io_code >= 0, "HDF5 collective dataset read") {
        let primary = if io_code < 0 {
            hdf5_error("H5Dread", io_code)
        } else {
            agreement
        };
        return Err(cleanup_ready(comm, duplicate, opened, resources, primary));
    }

    // Decode and map before any close.  After the collective prepare agreement
    // and successful cleanup, the destination is changed by one infallible
    // physical-order copy only.
    let values_result = prepare_physical_values(&view, &staging);
    if let Err(agreement) = agree_phase(comm, values_result.is_ok(), "HDF5 read physical staging") {
        return Err(cleanup_ready(
            comm,
            duplicate,
            opened,
            resources,
            values_result.err().unwrap_or(agreement),
        ));
    }
    let values = values_result.expect("HDF5 read physical staging agreement");
    let cleanup = finish_hdf5(comm, duplicate, opened, resources);
    let injected = if inject_post_cleanup_failure && comm.rank() == 0 {
        Some(IoError::Native {
            operation: "test post-cleanup result",
            code: -1,
        })
    } else {
        None
    };
    if let Some(error) = aggregate_cleanup_result(comm, cleanup.or(injected)) {
        return Err(error);
    }
    view.as_mut_slice().copy_from_slice(&values);
    Ok(())
}

#[cfg(test)]
pub(crate) fn read_hdf5_with_post_cleanup_failure<P, T, const N: usize, const M: usize>(
    path: P,
    view: PencilArrayViewMut<'_, T, N, M>,
) -> Result<(), IoError>
where
    P: AsRef<Path>,
    T: IoElement,
{
    read_hdf5_inner(
        path,
        view,
        true,
        cstr(GROUP_NAME),
        cstr(DATASET_NAME),
        None,
        OP_READ_HDF5,
    )
}

#[derive(Debug, Default)]
struct Hdf5Resources {
    group: Option<native::Hid>,
    dataset: Option<native::Hid>,
    datatype: Option<native::Hid>,
    dataspace: Option<native::Hid>,
    file_space: Option<native::Hid>,
    mem_space: Option<native::Hid>,
    xfer: Option<native::Hid>,
}

fn prepare_hdf5_fapl(
    comm: &mpi::topology::CartesianCommunicator,
    duplicate: ffi::MPI_Comm,
) -> Result<native::Hid, IoError> {
    let (fapl, error) = local_handle_phase(
        comm,
        native::fapl_create(),
        "HDF5 file-access property-list create",
    );
    if let Some(error) = error {
        return Err(cleanup_fapl(comm, fapl, error));
    }
    let fapl = fapl.expect("HDF5 file-access property-list creation agreement");
    let set_code = native::fapl_set_mpio(fapl, duplicate);
    if let Err(agreement) = agree_phase(
        comm,
        set_code >= 0,
        "HDF5 file-access property-list MPI setup",
    ) {
        let primary = if set_code < 0 {
            hdf5_error("H5Pset_fapl_mpio", set_code)
        } else {
            agreement
        };
        return Err(cleanup_fapl(comm, Some(fapl), primary));
    }
    Ok(fapl)
}

fn cleanup_fapl(
    comm: &mpi::topology::CartesianCommunicator,
    mut fapl: Option<native::Hid>,
    primary: IoError,
) -> IoError {
    let close_code = fapl.as_mut().map_or(0, native::plist_close);
    let (all_closed, _mixed_close) = collective_state(comm, close_code >= 0);
    if !all_closed {
        abort_unrecoverable(comm.as_raw(), "H5Pclose unrecoverable native handle");
    }
    primary
}

fn open_hdf5(
    comm: &mpi::topology::CartesianCommunicator,
    fapl: native::Hid,
    path: &CString,
    write: bool,
) -> Result<native::Hid, IoError> {
    open_hdf5_mode(comm, fapl, path, write, false)
}

fn open_hdf5_mode(
    comm: &mpi::topology::CartesianCommunicator,
    mut fapl: native::Hid,
    path: &CString,
    write: bool,
    update: bool,
) -> Result<native::Hid, IoError> {
    let result = if update {
        native::file_open_update(fapl, path.as_c_str())
    } else {
        native::file_open(fapl, path.as_c_str(), write)
    };
    // The FAPL is local and must be closed on every rank before any caller can
    // enter another HDF5 collective. Keep it retained until that agreement.
    let fapl_close = native::plist_close(&mut fapl);
    let (all_fapls_closed, _mixed_fapl_close) = collective_state(comm, fapl_close >= 0);
    if !all_fapls_closed {
        abort_unrecoverable(comm.as_raw(), "H5Pclose unrecoverable native handle");
    }

    let (all_succeeded, mixed) = collective_state(comm, result.is_ok());
    if mixed {
        abort_unrecoverable(comm.as_raw(), "HDF5 file open partial native file handle");
    }
    if !all_succeeded {
        return Err(result
            .err()
            .map(|code| hdf5_error(if write { "H5Fcreate" } else { "H5Fopen" }, code))
            .unwrap_or(IoError::CollectivePrecondition {
                phase: "HDF5 file open",
            }));
    }
    Ok(result.expect("HDF5 file-open agreement"))
}

/// H5G/H5D create/open are collective metadata operations.  Mixed success is
/// not treated as an ordinary Rust error because a subset cannot portably
/// close the resulting native object; the bounded fail-stop path avoids
/// leaking it while still allowing ordinary all-rank file/metadata errors.
fn collective_handle_phase(
    comm: &mpi::topology::CartesianCommunicator,
    result: Result<native::Hid, i32>,
    phase: &'static str,
) -> Result<native::Hid, IoError> {
    let (all_succeeded, mixed) = collective_state(comm, result.is_ok());
    if mixed {
        abort_unrecoverable(comm.as_raw(), phase);
    }
    if !all_succeeded {
        return Err(result
            .err()
            .map(|code| hdf5_error(phase, code))
            .unwrap_or(IoError::CollectivePrecondition { phase }));
    }
    Ok(result.expect("collective HDF5 handle agreement"))
}

/// Type, dataspace, transfer-list, and queried object handles are local
/// creations.  A successful peer handle is retained in the returned slot so
/// cleanup can close it when another rank fails.
fn local_handle_phase(
    comm: &mpi::topology::CartesianCommunicator,
    result: Result<native::Hid, i32>,
    phase: &'static str,
) -> (Option<native::Hid>, Option<IoError>) {
    let (handle, native_error) = match result {
        Ok(handle) => (Some(handle), None),
        Err(code) => (None, Some(hdf5_error(phase, code))),
    };
    let agreement = agree_phase(comm, handle.is_some(), phase);
    let error = agreement.err().or(native_error);
    (handle, error)
}

fn select_spaces(
    comm: &mpi::topology::CartesianCommunicator,
    file_space: native::Hid,
    mem_space: native::Hid,
    layout: &H5Layout,
) -> Result<(), IoError> {
    let rank = layout.global.len();
    let mut count = Vec::new();
    let mut mem_start = Vec::new();
    let mut mem_count = Vec::new();
    let allocation = count
        .try_reserve_exact(rank)
        .and_then(|_| mem_start.try_reserve_exact(rank))
        .and_then(|_| mem_count.try_reserve_exact(rank))
        .map_err(|_| IoError::AllocationFailed {
            requested: rank * std::mem::size_of::<hsize_t>() * 3,
        });
    if let Err(agreement) = agree_phase(
        comm,
        allocation.is_ok(),
        "HDF5 hyperslab selection allocation",
    ) {
        return Err(allocation.err().unwrap_or(agreement));
    }
    count.resize(rank, 1);
    mem_start.resize(rank, 0);
    mem_count.resize(rank, 1);

    // Establish an explicit none selection on every rank before any
    // non-empty rank installs its hyperslab. Empty ranks retain the required
    // none selection on the real file dataspace.
    let file_none = native::dataspace_select_none(file_space);
    let mem_none = native::dataspace_select_none(mem_space);
    let file_code = if layout.empty {
        0
    } else {
        native::dataspace_select_hyperslab(file_space, &layout.starts, &count, &layout.block)
    };
    let mem_code = if layout.empty {
        0
    } else {
        native::dataspace_select_hyperslab(mem_space, &mem_start, &mem_count, &layout.block)
    };
    let code = if file_none < 0 {
        file_none
    } else if mem_none < 0 {
        mem_none
    } else if file_code < 0 {
        file_code
    } else {
        mem_code
    };
    phase_code(comm, code, "HDF5 hyperslab selection")
}

#[allow(clippy::too_many_arguments)]
fn write_metadata<const N: usize, const M: usize>(
    comm: &mpi::topology::CartesianCommunicator,
    dataset: native::Hid,
    grid: &[usize; M],
    permutation: &[pencil_array::SpatialAxis; N],
    global: &[usize; N],
    extra: &[usize],
    type_code: u64,
    width: usize,
) -> Result<(), IoError> {
    let prepared = (|| {
        Ok::<_, IoError>((
            to_u64_values(global)?,
            to_u64_values(extra)?,
            to_u64_values(grid)?,
            to_permutation_values(permutation)?,
        ))
    })();
    if let Err(agreement) = agree_phase(comm, prepared.is_ok(), "HDF5 metadata preparation") {
        return Err(prepared.err().unwrap_or(agreement));
    }
    let (global_values, extra_values, grid_values, perm_values) =
        prepared.expect("HDF5 metadata preparation agreement");

    attr_phase(
        comm,
        dataset,
        ATTR_VERSION,
        &[FORMAT_VERSION],
        "HDF5 version attribute",
    )?;
    attr_phase(
        comm,
        dataset,
        ATTR_COMMIT,
        &[INCOMPLETE_MARKER],
        "HDF5 commit attribute",
    )?;
    attr_phase(comm, dataset, ATTR_N, &[N as u64], "HDF5 rank attribute")?;
    attr_phase(
        comm,
        dataset,
        ATTR_TYPE,
        &[type_code],
        "HDF5 type attribute",
    )?;
    attr_phase(
        comm,
        dataset,
        ATTR_WIDTH,
        &[width as u64],
        "HDF5 width attribute",
    )?;
    attr_phase(
        comm,
        dataset,
        ATTR_EXTRA_RANK,
        &[extra.len() as u64],
        "HDF5 extra-rank attribute",
    )?;
    if !extra.is_empty() {
        attr_phase(
            comm,
            dataset,
            ATTR_EXTRA,
            &extra_values,
            "HDF5 extra-shape attribute",
        )?;
    }
    attr_phase(
        comm,
        dataset,
        ATTR_GLOBAL,
        &global_values,
        "HDF5 global-shape attribute",
    )?;
    attr_phase(
        comm,
        dataset,
        ATTR_GRID,
        &grid_values,
        "HDF5 writer-grid attribute",
    )?;
    attr_phase(
        comm,
        dataset,
        ATTR_PERM,
        &perm_values,
        "HDF5 permutation attribute",
    )?;
    Ok(())
}

fn write_original_name(
    comm: &mpi::topology::CartesianCommunicator,
    abort_comm: ffi::MPI_Comm,
    object: native::Hid,
    name: &[u8],
    phase: &'static str,
) -> Result<(), IoError> {
    let size = name.len().checked_add(1).ok_or(IoError::SizeLimit {
        what: "HDF5 original-name attribute",
    });
    let mut bytes = Vec::new();
    let allocation = size.and_then(|size| {
        bytes
            .try_reserve_exact(size)
            .map_err(|_| IoError::AllocationFailed { requested: size })
    });
    if let Err(agreement) = agree_phase(comm, allocation.is_ok(), "HDF5 original-name allocation") {
        return Err(allocation.err().unwrap_or(agreement));
    }
    bytes.extend_from_slice(name);
    bytes.push(0);
    let size = bytes.len();
    let mut handles = AttrHandles::default();
    let (datatype, error) =
        local_handle_phase(comm, native::type_create_string(size, abort_comm), phase);
    handles.datatype = datatype;
    if let Some(error) = error {
        return Err(attr_error(comm, handles, error));
    }
    let (space, error) = local_handle_phase(comm, native::dataspace_scalar(), phase);
    handles.space = space;
    if let Some(error) = error {
        return Err(attr_error(comm, handles, error));
    }
    let attr = collective_handle_phase(
        comm,
        native::attr_create(
            object,
            cstr(NAME_ATTR),
            handles.datatype.expect("name datatype"),
            handles.space.expect("name space"),
        ),
        phase,
    );
    handles.attr = match attr {
        Ok(attr) => Some(attr),
        Err(error) => return Err(attr_error(comm, handles, error)),
    };
    let code = native::attr_write(
        handles.attr.expect("name attribute"),
        handles.datatype.expect("name datatype"),
        &bytes,
    );
    if let Err(agreement) = agree_phase(comm, code >= 0, phase) {
        return Err(attr_error(
            comm,
            handles,
            if code < 0 {
                hdf5_error("H5Awrite", code)
            } else {
                agreement
            },
        ));
    }
    finish_attr_handles(comm, handles).map_or(Ok(()), Err)
}

fn read_original_name(
    comm: &mpi::topology::CartesianCommunicator,
    object: native::Hid,
    phase: &'static str,
) -> Result<Vec<u8>, IoError> {
    let mut handles = AttrHandles::default();
    let (attr, error) = local_handle_phase(comm, native::attr_open(object, cstr(NAME_ATTR)), phase);
    handles.attr = attr;
    if let Some(error) = error {
        return Err(attr_error(comm, handles, error));
    }
    let attr = handles.attr.expect("name attribute");
    let (datatype, error) = local_handle_phase(comm, native::attr_type(attr), phase);
    handles.datatype = datatype;
    if let Some(error) = error {
        return Err(attr_error(comm, handles, error));
    }
    let datatype = handles.datatype.expect("name datatype");
    let (space, error) = local_handle_phase(comm, native::attr_space(attr), phase);
    handles.space = space;
    if let Some(error) = error {
        return Err(attr_error(comm, handles, error));
    }
    let space = handles.space.expect("name space");
    let shape = native::attr_shape(space);
    let size = native::type_size(datatype);
    let valid = shape.is_ok_and(|shape| shape.is_empty())
        && size.is_ok_and(|size| (1..=MAX_NAME + 1).contains(&size))
        && size.is_ok_and(|size| native::type_matches_string(datatype, size));
    if let Err(agreement) = agree_phase(comm, size.is_ok() && valid, phase) {
        return Err(attr_error(
            comm,
            handles,
            size.err()
                .map(|code| hdf5_error("HDF5 name type", code))
                .unwrap_or(agreement),
        ));
    }
    let size = size.expect("name size agreement");
    let mut bytes = Vec::new();
    let allocation = bytes
        .try_reserve_exact(size)
        .map_err(|_| IoError::AllocationFailed { requested: size });
    if let Err(agreement) = agree_phase(comm, allocation.is_ok(), phase) {
        return Err(attr_error(
            comm,
            handles,
            allocation.err().unwrap_or(agreement),
        ));
    }
    bytes.resize(size, 0);
    let code = native::attr_read(attr, datatype, &mut bytes);
    if let Err(agreement) = agree_phase(comm, code >= 0, phase) {
        return Err(attr_error(
            comm,
            handles,
            if code < 0 {
                hdf5_error("H5Aread", code)
            } else {
                agreement
            },
        ));
    }
    let valid = bytes.last() == Some(&0);
    if let Err(agreement) = agree_phase(comm, valid, phase) {
        return Err(attr_error(
            comm,
            handles,
            if valid {
                agreement
            } else {
                IoError::MetadataMismatch {
                    field: "original name",
                }
            },
        ));
    }
    bytes.pop();
    if let Some(error) = finish_attr_handles(comm, handles) {
        return Err(error);
    }
    Ok(bytes)
}

fn attr_phase(
    comm: &mpi::topology::CartesianCommunicator,
    object: native::Hid,
    name: &[u8],
    values: &[u64],
    phase: &'static str,
) -> Result<(), IoError> {
    write_attr_u64(comm, object, name, values, phase)
}

fn write_attr_u64(
    comm: &mpi::topology::CartesianCommunicator,
    object: native::Hid,
    name: &[u8],
    values: &[u64],
    phase: &'static str,
) -> Result<(), IoError> {
    if values.is_empty() {
        return Err(IoError::InvalidInput("HDF5 attributes cannot be empty"));
    }
    let mut handles = AttrHandles::default();
    let (datatype, error) = local_handle_phase(
        comm,
        native::type_copy(*hdf5_metno::globals::H5T_STD_U64LE),
        phase,
    );
    handles.datatype = datatype;
    if let Some(error) = error {
        return Err(attr_error(comm, handles, error));
    }
    let datatype = handles.datatype.expect("attribute datatype agreement");

    let space_result = if values.len() == 1 {
        native::dataspace_scalar()
    } else {
        match to_hsize(values.len()) {
            Ok(dimension) => native::dataspace_simple(&[dimension]),
            Err(_) => Err(-1),
        }
    };
    let (space, error) = local_handle_phase(comm, space_result, phase);
    handles.space = space;
    if let Some(error) = error {
        return Err(attr_error(comm, handles, error));
    }
    let space = handles.space.expect("attribute dataspace agreement");

    let attr = match collective_handle_phase(
        comm,
        native::attr_create(object, cstr(name), datatype, space),
        phase,
    ) {
        Ok(attr) => attr,
        Err(error) => return Err(attr_error(comm, handles, error)),
    };
    handles.attr = Some(attr);

    let byte_len = values.len().checked_mul(8).ok_or(IoError::SizeLimit {
        what: "HDF5 attribute bytes",
    });
    let mut bytes = Vec::new();
    let allocation = match byte_len {
        Ok(byte_len) => bytes
            .try_reserve_exact(byte_len)
            .map_err(|_| IoError::AllocationFailed {
                requested: byte_len,
            }),
        Err(error) => Err(error),
    };
    if let Err(agreement) = agree_phase(comm, allocation.is_ok(), "HDF5 attribute allocation") {
        return Err(attr_error(
            comm,
            handles,
            allocation.err().unwrap_or(agreement),
        ));
    }
    for &value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    let code = native::attr_write(
        handles.attr.expect("attribute creation agreement"),
        datatype,
        &bytes,
    );
    if let Err(agreement) = agree_phase(comm, code >= 0, "HDF5 attribute write") {
        return Err(attr_error(
            comm,
            handles,
            if code < 0 {
                hdf5_error("H5Awrite", code)
            } else {
                agreement
            },
        ));
    }
    finish_attr_handles(comm, handles).map_or(Ok(()), Err)
}

fn write_existing_attr(
    comm: &mpi::topology::CartesianCommunicator,
    object: native::Hid,
    name: &[u8],
    value: u64,
) -> Result<(), IoError> {
    let mut handles = AttrHandles::default();
    let (attr, error) = local_handle_phase(
        comm,
        native::attr_open(object, cstr(name)),
        "HDF5 commit marker open",
    );
    handles.attr = attr;
    if let Some(error) = error {
        return Err(attr_error(comm, handles, error));
    }
    let (datatype, error) = local_handle_phase(
        comm,
        native::type_copy(*hdf5_metno::globals::H5T_STD_U64LE),
        "HDF5 commit marker datatype",
    );
    handles.datatype = datatype;
    if let Some(error) = error {
        return Err(attr_error(comm, handles, error));
    }
    let code = native::attr_write(
        handles.attr.expect("commit attribute agreement"),
        handles.datatype.expect("commit datatype agreement"),
        &value.to_le_bytes(),
    );
    if let Err(agreement) = agree_phase(comm, code >= 0, "HDF5 commit marker write") {
        return Err(attr_error(
            comm,
            handles,
            if code < 0 {
                hdf5_error("H5Awrite", code)
            } else {
                agreement
            },
        ));
    }
    finish_attr_handles(comm, handles).map_or(Ok(()), Err)
}

#[derive(Default)]
struct AttrHandles {
    attr: Option<native::Hid>,
    datatype: Option<native::Hid>,
    space: Option<native::Hid>,
    memory: Option<native::Hid>,
}

fn attr_error(
    comm: &mpi::topology::CartesianCommunicator,
    handles: AttrHandles,
    primary: IoError,
) -> IoError {
    finish_attr_handles(comm, handles).unwrap_or(primary)
}

fn finish_attr_handles(
    comm: &mpi::topology::CartesianCommunicator,
    mut handles: AttrHandles,
) -> Option<IoError> {
    let mut first = None;
    macro_rules! close {
        ($slot:expr, $function:path, $operation:expr) => {
            if !close_optional(comm, $slot, &mut first, $function, $operation) {
                return first;
            }
        };
    }
    close!(
        &mut handles.memory,
        native::type_close,
        "H5Tclose(attribute memory)"
    );
    close!(
        &mut handles.space,
        native::dataspace_close,
        "H5Sclose(attribute)"
    );
    close!(
        &mut handles.datatype,
        native::type_close,
        "H5Tclose(attribute)"
    );
    close!(&mut handles.attr, native::attr_close, "H5Aclose");
    first
}

fn read_attr_phase(
    comm: &mpi::topology::CartesianCommunicator,
    duplicate: ffi::MPI_Comm,
    object: native::Hid,
    name: &[u8],
    expected_len: Option<usize>,
    max_len: usize,
    phase: &'static str,
) -> Result<Vec<u64>, IoError> {
    let exists_result = native::attr_exists(object, cstr(name));
    if let Err(agreement) = agree_phase(comm, exists_result.is_ok(), phase) {
        return Err(exists_result
            .err()
            .map(|code| hdf5_error("H5Aexists", code))
            .unwrap_or(agreement));
    }
    let exists = exists_result.expect("attribute-exists agreement");
    collective_bool(comm, exists, phase)?;
    if !exists {
        return Err(IoError::MetadataMismatch {
            field: "missing HDF5 attribute",
        });
    }

    let mut handles = AttrHandles::default();
    let (attr, error) = local_handle_phase(comm, native::attr_open(object, cstr(name)), phase);
    handles.attr = attr;
    if let Some(error) = error {
        return Err(attr_error(comm, handles, error));
    }
    let attr = handles.attr.expect("attribute-open agreement");

    let (datatype, error) = local_handle_phase(comm, native::attr_type(attr), phase);
    handles.datatype = datatype;
    if let Some(error) = error {
        return Err(attr_error(comm, handles, error));
    }
    let datatype = handles.datatype.expect("attribute-type agreement");
    let type_result = native::type_matches(datatype, 8, 8, duplicate);
    if let Err(agreement) = agree_phase(comm, type_result.is_ok(), phase) {
        return Err(attr_error(
            comm,
            handles,
            type_result
                .err()
                .map(|code| hdf5_error("HDF5 attribute type validation", code))
                .unwrap_or(agreement),
        ));
    }
    let type_ok = type_result.expect("attribute type validation agreement");
    if let Err(agreement) = collective_bool(comm, type_ok, phase) {
        return Err(attr_error(comm, handles, agreement));
    }
    if !type_ok {
        return Err(attr_error(
            comm,
            handles,
            IoError::MetadataMismatch {
                field: "HDF5 attribute type or shape",
            },
        ));
    }

    let (space, error) = local_handle_phase(comm, native::attr_space(attr), phase);
    handles.space = space;
    if let Some(error) = error {
        return Err(attr_error(comm, handles, error));
    }
    let space = handles.space.expect("attribute-space agreement");
    let shape_result = native::attr_shape(space);
    if let Err(agreement) = agree_phase(comm, shape_result.is_ok(), phase) {
        return Err(attr_error(
            comm,
            handles,
            shape_result
                .err()
                .map(|code| hdf5_error("H5Sget_simple_extent_dims", code))
                .unwrap_or(agreement),
        ));
    }
    let shape = shape_result.expect("attribute shape agreement");
    let length_result = if shape.is_empty() {
        Ok(1usize)
    } else if shape.len() == 1 {
        usize::try_from(shape[0]).map_err(|_| IoError::InvalidFile {
            reason: "HDF5 attribute length",
        })
    } else {
        Err(IoError::MetadataMismatch {
            field: "HDF5 attribute type or shape",
        })
    };
    let length_result = length_result.and_then(|length| {
        if length <= max_len && expected_len.is_none_or(|expected| expected == length) {
            Ok(length)
        } else {
            Err(IoError::MetadataMismatch {
                field: "HDF5 attribute type or shape",
            })
        }
    });
    if let Err(agreement) = agree_phase(comm, length_result.is_ok(), phase) {
        return Err(attr_error(
            comm,
            handles,
            length_result.err().unwrap_or(agreement),
        ));
    }
    let length = length_result.expect("attribute length agreement");
    let byte_len = length.checked_mul(8).ok_or(IoError::SizeLimit {
        what: "HDF5 attribute bytes",
    });
    let mut bytes = Vec::new();
    let mut values = Vec::new();
    let allocation = match &byte_len {
        Ok(byte_len) => bytes
            .try_reserve_exact(*byte_len)
            .and_then(|_| values.try_reserve_exact(length))
            .map_err(|_| IoError::AllocationFailed {
                requested: *byte_len,
            }),
        Err(error) => Err(error.clone()),
    };
    if let Err(agreement) = agree_phase(comm, allocation.is_ok(), "HDF5 attribute allocation") {
        return Err(attr_error(
            comm,
            handles,
            allocation.err().unwrap_or(agreement),
        ));
    }
    bytes.resize(byte_len.expect("attribute byte length agreement"), 0);

    let (memory, error) = local_handle_phase(
        comm,
        native::type_copy(*hdf5_metno::globals::H5T_STD_U64LE),
        phase,
    );
    handles.memory = memory;
    if let Some(error) = error {
        return Err(attr_error(comm, handles, error));
    }
    let memory = handles.memory.expect("attribute memory-type agreement");
    let code = native::attr_read(attr, memory, &mut bytes);
    if let Err(agreement) = agree_phase(comm, code >= 0, phase) {
        return Err(attr_error(
            comm,
            handles,
            if code < 0 {
                hdf5_error("H5Aread", code)
            } else {
                agreement
            },
        ));
    }
    // `values` was reserved and bytes have an exact validated length, so this
    // decode cannot allocate or fail after the read agreement.
    for chunk in bytes.chunks_exact(8) {
        values.push(u64::from_le_bytes(
            chunk.try_into().expect("eight-byte HDF5 attribute"),
        ));
    }
    if let Some(error) = finish_attr_handles(comm, handles) {
        return Err(error);
    }
    Ok(values)
}

fn attribute_exists_phase(
    comm: &mpi::topology::CartesianCommunicator,
    object: native::Hid,
    name: &[u8],
    phase: &'static str,
) -> Result<bool, IoError> {
    let result = native::attr_exists(object, cstr(name));
    if let Err(agreement) = agree_phase(comm, result.is_ok(), phase) {
        return Err(result
            .err()
            .map(|code| hdf5_error("H5Aexists", code))
            .unwrap_or(agreement));
    }
    collective_bool(comm, result.expect("attribute-exists agreement"), phase)
}

fn cleanup_comm_only(
    comm: &mpi::topology::CartesianCommunicator,
    mut duplicate: CommGuard,
    primary: IoError,
) -> IoError {
    let code = ffi::comm_free(&mut duplicate.raw);
    let (all_closed, _mixed_close) = collective_state(comm, code == 0);
    if !all_closed {
        abort_unrecoverable(comm.as_raw(), "MPI_Comm_free unrecoverable native handle");
    }
    primary
}

fn cleanup_result(cleanup: Option<IoError>, primary: IoError) -> IoError {
    if matches!(
        &primary,
        IoError::WriteIncomplete { .. } | IoError::CommitUncertain { .. }
    ) {
        primary
    } else {
        cleanup.unwrap_or(primary)
    }
}

fn cleanup_ready(
    comm: &mpi::topology::CartesianCommunicator,
    duplicate: CommGuard,
    file: native::Hid,
    resources: Hdf5Resources,
    primary: IoError,
) -> IoError {
    cleanup_result(finish_hdf5(comm, duplicate, file, resources), primary)
}

fn finish_hdf5(
    comm: &mpi::topology::CartesianCommunicator,
    mut duplicate: CommGuard,
    file: native::Hid,
    mut resources: Hdf5Resources,
) -> Option<IoError> {
    let mut first = None;
    macro_rules! close {
        ($slot:expr, $function:path, $operation:expr) => {
            if !close_optional(comm, $slot, &mut first, $function, $operation) {
                return first;
            }
        };
    }
    close!(&mut resources.xfer, native::plist_close, "H5Pclose");
    close!(
        &mut resources.mem_space,
        native::dataspace_close,
        "H5Sclose(memory)"
    );
    close!(
        &mut resources.file_space,
        native::dataspace_close,
        "H5Sclose(file)"
    );
    close!(&mut resources.datatype, native::type_close, "H5Tclose");
    close!(
        &mut resources.dataspace,
        native::dataspace_close,
        "H5Sclose(dataset)"
    );
    close!(&mut resources.dataset, native::dataset_close, "H5Dclose");
    close!(&mut resources.group, native::group_close, "H5Gclose");
    let mut file_id = file;
    let file_code = native::file_close(&mut file_id);
    let (all_files_closed, _mixed_file_close) = collective_state(comm, file_code >= 0);
    if !all_files_closed {
        // H5Fclose is collective in the parallel build. Do not free the
        // communicator after a mixed/all-failed close while a file handle
        // may still reference it.
        abort_unrecoverable(comm.as_raw(), "H5Fclose unrecoverable native handle");
    }

    let comm_code = ffi::comm_free(&mut duplicate.raw);
    let (all_closed, _mixed_close) = collective_state(comm, comm_code == 0);
    if !all_closed {
        // The duplicate may be null or indeterminate after MPI_Comm_free;
        // only the original topology communicator is a valid fail-stop path.
        abort_unrecoverable(comm.as_raw(), "MPI_Comm_free unrecoverable native handle");
    }
    first
}

fn close_optional(
    comm: &mpi::topology::CartesianCommunicator,
    slot: &mut Option<native::Hid>,
    _first: &mut Option<IoError>,
    close: fn(&mut native::Hid) -> i32,
    operation: &'static str,
) -> bool {
    let code = slot.as_mut().map_or(0, close);
    let (all_closed, _mixed_close) = collective_state(comm, code >= 0);
    if !all_closed {
        // Retain the failed handle and stop instead of dropping a still-live
        // native object or continuing toward communicator/file cleanup.
        abort_unrecoverable(comm.as_raw(), operation);
    }
    slot.take();
    true
}

fn phase_code(
    comm: &mpi::topology::CartesianCommunicator,
    code: i32,
    phase: &'static str,
) -> Result<(), IoError> {
    if let Err(agreement) = agree_phase(comm, code >= 0, phase) {
        Err(if code < 0 {
            hdf5_error(phase, code)
        } else {
            agreement
        })
    } else if code < 0 {
        Err(hdf5_error(phase, code))
    } else {
        Ok(())
    }
}

fn collective_bool(
    comm: &mpi::topology::CartesianCommunicator,
    value: bool,
    phase: &'static str,
) -> Result<bool, IoError> {
    let (all_true, mixed) = collective_state(comm, value);
    if mixed {
        Err(IoError::CollectivePrecondition { phase })
    } else {
        Ok(all_true)
    }
}

fn to_permutation_values<const N: usize>(
    values: &[pencil_array::SpatialAxis; N],
) -> Result<Vec<u64>, IoError> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(N)
        .map_err(|_| IoError::AllocationFailed {
            requested: N * std::mem::size_of::<u64>(),
        })?;
    for axis in values {
        result.push(u64::try_from(axis.index()).map_err(|_| IoError::SizeLimit {
            what: "HDF5 permutation",
        })?);
    }
    Ok(result)
}

fn to_u64_values(values: &[usize]) -> Result<Vec<u64>, IoError> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(values.len())
        .map_err(|_| IoError::AllocationFailed {
            requested: values.len() * std::mem::size_of::<u64>(),
        })?;
    for &value in values {
        result.push(u64::try_from(value).map_err(|_| IoError::SizeLimit {
            what: "HDF5 metadata value",
        })?);
    }
    Ok(result)
}

fn build_layout<const N: usize>(
    global: &[usize; N],
    extra: &[usize],
    local: &[usize; N],
    ranges: &[Range<usize>; N],
    local_elements: usize,
) -> Result<H5Layout, IoError> {
    let rank = extra
        .len()
        .checked_add(N)
        .ok_or(IoError::SizeLimit { what: "HDF5 rank" })?;
    if rank == 0 || rank > MAX_PROTOCOL_RANK {
        return Err(IoError::SizeLimit { what: "HDF5 rank" });
    }
    let mut global_dims = Vec::new();
    let mut local_dims = Vec::new();
    let mut starts = Vec::new();
    let mut block = Vec::new();
    for vector in [&mut global_dims, &mut local_dims, &mut starts, &mut block] {
        vector
            .try_reserve_exact(rank)
            .map_err(|_| IoError::AllocationFailed {
                requested: rank * std::mem::size_of::<hsize_t>(),
            })?;
    }
    for &extent in extra {
        global_dims.push(to_hsize(extent)?);
        local_dims.push(to_hsize(extent)?);
        starts.push(0);
        block.push(to_hsize(extent)?);
    }
    for axis in 0..N {
        global_dims.push(to_hsize(global[axis])?);
        local_dims.push(to_hsize(local[axis])?);
        starts.push(to_hsize(ranges[axis].start)?);
        block.push(to_hsize(ranges[axis].len())?);
    }
    let empty = local_elements == 0 || extra.contains(&0);
    Ok(H5Layout {
        global: global_dims,
        local: local_dims,
        starts,
        block,
        empty,
    })
}

struct H5Layout {
    global: Vec<hsize_t>,
    local: Vec<hsize_t>,
    starts: Vec<hsize_t>,
    block: Vec<hsize_t>,
    empty: bool,
}

fn path_cstring(
    path: &Path,
    comm: &mpi::topology::CartesianCommunicator,
) -> Result<CString, IoError> {
    #[cfg(unix)]
    use std::os::unix::ffi::OsStrExt;
    let bytes = {
        #[cfg(unix)]
        {
            path.as_os_str().as_bytes()
        }
        #[cfg(not(unix))]
        {
            path.to_str().ok_or(IoError::InvalidPath)?.as_bytes()
        }
    };
    let result = CString::new(bytes).map_err(|_| IoError::InvalidPath);
    if let Err(agreement) = agree_phase(comm, result.is_ok(), "HDF5 path preparation") {
        return Err(result.err().unwrap_or(agreement));
    }
    Ok(result.expect("HDF5 path agreement"))
}

fn to_hsize(value: usize) -> Result<hsize_t, IoError> {
    hsize_t::try_from(value).map_err(|_| IoError::SizeLimit {
        what: "HDF5 dimension",
    })
}

fn cstr(bytes: &[u8]) -> &CStr {
    CStr::from_bytes_with_nul(bytes).expect("HDF5 name is NUL terminated")
}

fn hdf5_error(operation: &'static str, code: i32) -> IoError {
    IoError::Native {
        operation,
        code: i64::from(code),
    }
}

fn first_error2<T, U>(first: &Result<T, IoError>, second: &Result<U, IoError>) -> Option<IoError> {
    first
        .as_ref()
        .err()
        .cloned()
        .or_else(|| second.as_ref().err().cloned())
}

fn valid_dimensions(values: &[u64]) -> bool {
    !values.is_empty()
        && values
            .iter()
            .copied()
            .try_fold(1u64, |product, value| product.checked_mul(value))
            .is_some()
}

fn valid_extents(values: &[u64]) -> bool {
    valid_dimensions(values) && values.iter().copied().all(|value| value > 0)
}

fn is_permutation(values: &[u64], n: usize) -> bool {
    if values.len() != n {
        return false;
    }
    let mut seen = Vec::new();
    if seen.try_reserve_exact(n).is_err() {
        return false;
    }
    seen.resize(n, false);
    for &value in values {
        let Ok(index) = usize::try_from(value) else {
            return false;
        };
        if index >= n || seen[index] {
            return false;
        }
        seen[index] = true;
    }
    true
}
