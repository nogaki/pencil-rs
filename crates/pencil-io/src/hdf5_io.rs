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

mod session;
#[cfg(test)]
pub(crate) use session::native_contracts as test_session_native_contracts;
pub use session::{Hdf5FileSession, Hdf5SessionError};

const MAX_NAME: usize = 1024;
const OP_OPTIONS_WRITE_HDF5: u64 = 13;
const OP_OPTIONS_READ_HDF5: u64 = 14;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Hdf5Settings<'a> {
    pub collective: bool,
    pub chunks: Option<&'a [usize]>,
    pub hints: &'a [(String, String)],
    pub explicit: bool,
    pub shuffle: bool,
    pub deflate: Option<u8>,
}

impl Default for Hdf5Settings<'_> {
    fn default() -> Self {
        Self {
            collective: true,
            chunks: None,
            hints: &[],
            explicit: false,
            shuffle: false,
            deflate: None,
        }
    }
}

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
        Hdf5Settings::default(),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn write_hdf5_inner_options<P, T, const N: usize, const M: usize>(
    path: P,
    view: PencilArrayView<'_, T, N, M>,
    settings: Hdf5Settings<'_>,
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
        OP_OPTIONS_WRITE_HDF5,
        settings,
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
    settings: Hdf5Settings<'_>,
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
    if settings.explicit {
        crate::options::agree_decomposition(view.pencil())?;
    }
    validate_options(comm, settings, layout.global.len(), T::WIDTH)?;

    let duplicate = duplicate_comm(comm)?;
    let fapl = match prepare_hdf5_fapl(comm, duplicate.raw, settings.hints) {
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
        match settings.chunks {
            Some(chunks) => native::dataspace_chunked(&layout.global, chunks),
            None => native::dataspace_simple(&layout.global),
        },
        "HDF5 dataspace create",
    );
    resources.dataspace = dataspace;
    if let Some(error) = error {
        return Err(cleanup_ready(comm, duplicate, opened, resources, error));
    }
    let dataspace = resources.dataspace.expect("dataspace creation agreement");

    if let Some(chunks) = settings.chunks {
        let (dcpl, error) = local_handle_phase(
            comm,
            native::dcpl_create(),
            "HDF5 dataset-create property list",
        );
        resources.dcpl = dcpl;
        if let Some(error) = error {
            return Err(cleanup_ready(comm, duplicate, opened, resources, error));
        }
        let mut native_chunks = [0 as hsize_t; crate::MAX_PROTOCOL_RANK];
        for (dst, &x) in native_chunks.iter_mut().zip(chunks) {
            *dst = x as hsize_t;
        }
        if let Err(error) = phase_code(
            comm,
            native::dcpl_set_chunk(
                resources.dcpl.expect("dcpl agreement"),
                &native_chunks[..chunks.len()],
            ),
            "HDF5 chunk layout",
        ) {
            return Err(cleanup_ready(comm, duplicate, opened, resources, error));
        }
        if settings.shuffle {
            if let Err(error) = phase_code(
                comm,
                native::plist_set_shuffle(resources.dcpl.expect("dcpl agreement")),
                "HDF5 shuffle filter",
            ) {
                return Err(cleanup_ready(comm, duplicate, opened, resources, error));
            }
        }
        if let Some(level) = settings.deflate {
            if let Err(error) = phase_code(
                comm,
                native::plist_set_deflate(
                    resources.dcpl.expect("dcpl agreement"),
                    u32::from(level),
                ),
                "HDF5 deflate filter",
            ) {
                return Err(cleanup_ready(comm, duplicate, opened, resources, error));
            }
        }
    }
    let dataset = match collective_handle_phase(
        comm,
        native::dataset_create(
            resources.group.expect("group creation agreement"),
            dataset_name,
            datatype,
            dataspace,
            resources.dcpl.unwrap_or(0),
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
        native::xfer_create(duplicate.raw, settings.collective),
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
        Hdf5Settings::default(),
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

/// Creates a named HDF5 dataset using explicit native HDF5 controls.
pub fn write_hdf5_named_with_options<P, S, T, const N: usize, const M: usize>(
    path: P,
    name: S,
    view: PencilArrayView<'_, T, N, M>,
    options: &crate::hdf5_options::Hdf5WriteOptions,
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
        71,
        view.pencil().global_shape(),
        view.extra_shape().dimensions(),
        view.pencil().topology().process_grid(),
        view.pencil().permutation().axes(),
        T::CODE,
        T::WIDTH,
    )
    .map_err(NamedIoError::Io)?;
    crate::options::agree_decomposition(view.pencil()).map_err(NamedIoError::Io)?;
    validate_options(
        comm,
        options.settings(),
        N + view.extra_shape().dimensions().len(),
        T::WIDTH,
    )
    .map_err(NamedIoError::Io)?;
    let link = named_link(comm, name.as_ref())?;
    write_hdf5_inner(
        path,
        view,
        false,
        cstr(NAMED_GROUP),
        &link,
        false,
        Some(name.as_ref().as_bytes()),
        71,
        options.settings(),
    )
    .map_err(NamedIoError::Io)
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
        Hdf5Settings::default(),
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
    let fapl = match prepare_hdf5_fapl(comm, dup.raw, &[]) {
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
        Hdf5Settings::default(),
    )
    .map_err(NamedIoError::Io)
}

/// Appends a named HDF5 dataset using explicit native HDF5 controls.
pub fn append_hdf5_named_with_options<P, S, T, const N: usize, const M: usize>(
    path: P,
    name: S,
    view: PencilArrayView<'_, T, N, M>,
    options: &crate::hdf5_options::Hdf5WriteOptions,
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
        72,
        view.pencil().global_shape(),
        view.extra_shape().dimensions(),
        view.pencil().topology().process_grid(),
        view.pencil().permutation().axes(),
        T::CODE,
        T::WIDTH,
    )
    .map_err(NamedIoError::Io)?;
    crate::options::agree_decomposition(view.pencil()).map_err(NamedIoError::Io)?;
    validate_options(
        comm,
        options.settings(),
        N + view.extra_shape().dimensions().len(),
        T::WIDTH,
    )
    .map_err(NamedIoError::Io)?;
    let link = named_link(comm, name.as_ref())?;
    let pc = path_cstring(path.as_ref(), comm).map_err(NamedIoError::Io)?;
    let dup = duplicate_comm(comm).map_err(NamedIoError::Io)?;
    let fapl = match prepare_hdf5_fapl(comm, dup.raw, options.settings().hints) {
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
        72,
        options.settings(),
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
        Hdf5Settings::default(),
    )
    .map_err(|error| match error {
        IoError::MetadataMismatch {
            field: "named dataset missing",
        } => NamedIoError::NotFound,
        other => NamedIoError::Io(other),
    })
}

/// Reads a named HDF5 dataset using explicit native HDF5 controls.
pub fn read_hdf5_named_with_options<P, S, T, const N: usize, const M: usize>(
    path: P,
    name: S,
    view: PencilArrayViewMut<'_, T, N, M>,
    options: &crate::hdf5_options::Hdf5ReadOptions,
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
        73,
        view.pencil().global_shape(),
        view.extra_shape().dimensions(),
        view.pencil().topology().process_grid(),
        view.pencil().permutation().axes(),
        T::CODE,
        T::WIDTH,
    )
    .map_err(NamedIoError::Io)?;
    crate::options::agree_decomposition(view.pencil()).map_err(NamedIoError::Io)?;
    validate_options(
        comm,
        options.settings(),
        N + view.extra_shape().dimensions().len(),
        T::WIDTH,
    )
    .map_err(NamedIoError::Io)?;
    let link = named_link(comm, name.as_ref())?;
    read_hdf5_inner(
        path,
        view,
        false,
        cstr(NAMED_GROUP),
        &link,
        Some(name.as_ref().as_bytes()),
        73,
        options.settings(),
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
        Hdf5Settings::default(),
    )
}

pub(crate) fn read_hdf5_inner_options<P, T, const N: usize, const M: usize>(
    path: P,
    view: PencilArrayViewMut<'_, T, N, M>,
    settings: Hdf5Settings<'_>,
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
        OP_OPTIONS_READ_HDF5,
        settings,
    )
}

#[allow(clippy::too_many_arguments)]
fn read_hdf5_inner<P, T, const N: usize, const M: usize>(
    path: P,
    mut view: PencilArrayViewMut<'_, T, N, M>,
    inject_post_cleanup_failure: bool,
    group_name: &CStr,
    dataset_name: &CStr,
    expected_name: Option<&[u8]>,
    operation: u64,
    settings: Hdf5Settings<'_>,
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
    if settings.explicit {
        crate::options::agree_decomposition(view.pencil())?;
    }
    validate_options(comm, settings, layout.global.len(), T::WIDTH)?;

    let duplicate = duplicate_comm(comm)?;
    let fapl = match prepare_hdf5_fapl(comm, duplicate.raw, settings.hints) {
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
        native::xfer_create(duplicate.raw, settings.collective),
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
        Hdf5Settings::default(),
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
    dcpl: Option<native::Hid>,
}

fn validate_options(
    comm: &mpi::topology::CartesianCommunicator,
    settings: Hdf5Settings<'_>,
    rank: usize,
    width: usize,
) -> Result<(), IoError> {
    // Legacy calls retain their original collective sequence and native defaults.
    if !settings.explicit {
        return Ok(());
    }
    let mut extra = [0u64; crate::MAX_PROTOCOL_RANK + 5];
    let filtered = settings.shuffle || settings.deflate.is_some();
    let valid = if let Some(chunks) = settings.chunks {
        let bytes = chunks
            .iter()
            .try_fold(width as u64, |n, &x| n.checked_mul(x as u64));
        chunks.len() == rank
            && chunks.len() <= hdf5_metno_sys::h5s::H5S_MAX_RANK as usize
            && chunks.iter().all(|&x| x > 0 && x <= u32::MAX as usize)
            && bytes.is_some_and(|n| n < (1u64 << 32))
    } else {
        true
    } && (!filtered || settings.chunks.is_some())
        && settings.deflate.is_none_or(|level| level <= 9)
        && !(filtered && !settings.collective && settings.chunks.is_some());
    agree_phase(comm, valid, "HDF5 filter/chunk validation")?;
    let len = if let Some(chunks) = settings.chunks {
        extra[0] = 2;
        extra[1] = u64::from(settings.shuffle);
        extra[2] = settings.deflate.map_or(0, |level| u64::from(level) + 1);
        extra[3] = chunks.len() as u64;
        for (dst, &x) in extra[4..].iter_mut().zip(chunks) {
            *dst = x as u64;
        }
        chunks.len() + 4
    } else {
        extra[0] = 2;
        extra[1] = u64::from(settings.shuffle);
        extra[2] = settings.deflate.map_or(0, |level| u64::from(level) + 1);
        extra[3] = 0;
        4
    };
    crate::options::agree_controls(
        comm,
        if settings.collective {
            crate::MpiIoMode::Collective
        } else {
            crate::MpiIoMode::Independent
        },
        settings.hints,
        &extra[..len],
    )?;
    // Options agree before filter-dependent collectives. Both directions must
    // be supported so a successfully written dataset remains readable.
    for (requested, filter) in [(settings.shuffle, 2), (settings.deflate.is_some(), 1)] {
        if requested {
            let capable = native::filter_avail(filter)
                && native::filter_info(filter).is_ok_and(|flags| flags & 3 == 3);
            agree_phase(comm, capable, "HDF5 filter capability")?;
        }
    }
    Ok(())
}

fn prepare_hdf5_fapl(
    comm: &mpi::topology::CartesianCommunicator,
    duplicate: ffi::MPI_Comm,
    hints: &[(String, String)],
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
    let info = if hints.is_empty() {
        None
    } else {
        match crate::options::InfoGuard::from_hints(comm, hints) {
            Ok(info) => Some(info),
            Err(error) => return Err(cleanup_fapl(comm, Some(fapl), error)),
        }
    };
    let set_code = native::fapl_set_mpio(
        fapl,
        duplicate,
        info.as_ref().map_or_else(ffi::info_null, |i| i.raw),
    );
    drop(info); // H5Pset_fapl_mpio retains its own copy.
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
    close!(&mut resources.dcpl, native::plist_close, "H5Pclose");
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

fn catalog_agree_bytes(
    comm: &mpi::topology::CartesianCommunicator,
    bytes: &[u8],
) -> Result<(), crate::catalog::CatalogError> {
    crate::catalog::agree_bytes(comm, bytes)
}

fn catalog_valid(
    comm: &mpi::topology::CartesianCommunicator,
    valid: bool,
    phase: &'static str,
) -> Result<(), crate::catalog::CatalogError> {
    agree_phase(comm, valid, phase).map_err(crate::catalog::CatalogError::Io)
}

pub(crate) fn inspect_catalog(
    path: &Path,
    comm: &mpi::topology::CartesianCommunicator,
) -> Result<Vec<crate::catalog::DatasetInfo>, crate::catalog::CatalogError> {
    use crate::catalog::{CatalogError, ScalarType};
    let path = path_cstring(path, comm).map_err(CatalogError::Io)?;
    let duplicate = duplicate_comm(comm).map_err(CatalogError::Io)?;
    let fapl = match prepare_hdf5_fapl(comm, duplicate.raw, &[]) {
        Ok(x) => x,
        Err(e) => return Err(CatalogError::Io(cleanup_comm_only(comm, duplicate, e))),
    };
    let file = match open_hdf5(comm, fapl, &path, false) {
        Ok(x) => x,
        Err(e) => return Err(CatalogError::Io(cleanup_comm_only(comm, duplicate, e))),
    };
    let mut resources = Hdf5Resources::default();
    let result = (|| {
        let mut out = Vec::new();
        let mut groups = Vec::new();
        let reserve = groups.try_reserve_exact(2);
        if let Err(e) = agree_phase(
            comm,
            reserve.is_ok(),
            "HDF5 catalog group collection allocation",
        ) {
            return Err(CatalogError::Io(
                reserve
                    .err()
                    .map(|_| IoError::AllocationFailed { requested: 2 })
                    .unwrap_or(e),
            ));
        }
        for group_name in [cstr(GROUP_NAME), cstr(NAMED_GROUP)] {
            let exists_result = native::link_exists(file, group_name);
            let exists =
                match agree_phase(comm, exists_result.is_ok(), "HDF5 catalog group link query") {
                    Ok(()) => {
                        exists_result.map_err(|x| hdf5_error("HDF5 catalog group link query", x))?
                    }
                    Err(e) => return Err(CatalogError::Io(e)),
                };
            catalog_agree_bytes(comm, &[u8::from(exists)])?;
            if exists {
                let group_result =
                    native::link_is_hard_and_type(file, group_name, native::LinkObjectType::Group);
                let is_group = match agree_phase(
                    comm,
                    group_result.is_ok(),
                    "HDF5 catalog group link type query",
                ) {
                    Ok(()) => group_result
                        .map_err(|x| hdf5_error("HDF5 catalog group link type query", x))?,
                    Err(e) => return Err(CatalogError::Io(e)),
                };
                catalog_valid(comm, is_group, "HDF5 catalog group link type")?;
            }
            groups.push(exists);
        }
        catalog_valid(
            comm,
            groups.iter().any(|&x| x),
            "HDF5 catalog recognized group",
        )?;
        for (group_name, named) in [(cstr(GROUP_NAME), false), (cstr(NAMED_GROUP), true)] {
            if !groups[usize::from(named)] {
                continue;
            }
            let group = match collective_handle_phase(
                comm,
                native::group_open(file, group_name),
                "HDF5 catalog group open",
            ) {
                Ok(x) => x,
                Err(e) => return Err(CatalogError::Io(e)),
            };
            resources.group = Some(group);
            let count_result = native::group_link_count(group);
            let count = match agree_phase(comm, count_result.is_ok(), "HDF5 catalog link count") {
                Ok(()) => count_result.map_err(|x| hdf5_error("H5Gget_info", x))?,
                Err(e) => return Err(CatalogError::Io(e)),
            };
            catalog_agree_bytes(comm, &count.to_le_bytes())?;
            catalog_valid(
                comm,
                count > 0 && count <= 65_536 && (named || count == 1),
                "HDF5 catalog link count validation",
            )?;
            for index in 0..count {
                let buf_len = if named {
                    MAX_NAME * 2 + 1
                } else {
                    MAX_NAME + 1
                };
                let mut buf = Vec::new();
                let reserve = buf.try_reserve_exact(buf_len);
                if let Err(e) =
                    agree_phase(comm, reserve.is_ok(), "HDF5 catalog link-name allocation")
                {
                    return Err(CatalogError::Io(
                        reserve
                            .err()
                            .map(|_| IoError::AllocationFailed { requested: buf_len })
                            .unwrap_or(e),
                    ));
                }
                buf.resize(buf_len, 0);
                let nl = native::link_name_by_idx(group, index, &mut buf);
                let nl = match agree_phase(comm, nl.is_ok(), "HDF5 catalog link name") {
                    Ok(()) => nl.map_err(|x| hdf5_error("H5Lget_name_by_idx", x))?,
                    Err(e) => return Err(CatalogError::Io(e)),
                };
                if let Err(e) = agree_phase(
                    comm,
                    nl <= if named { MAX_NAME * 2 } else { MAX_NAME } && nl < buf.len(),
                    "HDF5 catalog link name validation",
                ) {
                    return Err(CatalogError::Io(e));
                }
                let link = &buf[..nl];
                catalog_agree_bytes(comm, link)?;
                catalog_valid(comm, named || link == b"data", "HDF5 catalog legacy link")?;
                let dname_result = CString::new(link);
                catalog_valid(comm, dname_result.is_ok(), "HDF5 catalog link name")?;
                let dname = dname_result.expect("link name agreement");
                let dataset_link =
                    native::link_is_hard_and_type(group, &dname, native::LinkObjectType::Dataset);
                let dataset_link = match agree_phase(
                    comm,
                    dataset_link.is_ok(),
                    "HDF5 catalog dataset link query",
                ) {
                    Ok(()) => dataset_link
                        .map_err(|x| hdf5_error("HDF5 catalog dataset link query", x))?,
                    Err(e) => return Err(CatalogError::Io(e)),
                };
                catalog_agree_bytes(comm, &[u8::from(dataset_link)])?;
                catalog_valid(comm, dataset_link, "HDF5 catalog dataset link")?;
                let dataset = match collective_handle_phase(
                    comm,
                    native::dataset_open(group, &dname),
                    "HDF5 catalog dataset open",
                ) {
                    Ok(x) => x,
                    Err(e) => return Err(CatalogError::Io(e)),
                };
                resources.dataset = Some(dataset);
                macro_rules! attr {
                    ($n:expr, $l:expr, $m:expr) => {{
                        let value = read_attr_phase(
                            comm,
                            duplicate.raw,
                            dataset,
                            $n,
                            $l,
                            $m,
                            "HDF5 catalog attribute",
                        )
                        .map_err(CatalogError::Io)?;
                        let bytes: Vec<u8> = value.iter().flat_map(|x| x.to_le_bytes()).collect();
                        catalog_agree_bytes(comm, &bytes)?;
                        value
                    }};
                }
                let version = attr!(ATTR_VERSION, Some(1), 1);
                let commit = attr!(ATTR_COMMIT, Some(1), 1);
                let nvals = attr!(ATTR_N, Some(1), 1);
                let typ = attr!(ATTR_TYPE, Some(1), 1);
                let width = attr!(ATTR_WIDTH, Some(1), 1);
                let erank = attr!(ATTR_EXTRA_RANK, Some(1), 1);
                let scalar_result = ScalarType::decode(typ[0], width[0]);
                catalog_valid(comm, scalar_result.is_some(), "HDF5 catalog scalar type")?;
                let scalar = scalar_result.expect("scalar type agreement");
                let n_result = usize::try_from(nvals[0]);
                let er_result = usize::try_from(erank[0]);
                catalog_valid(
                    comm,
                    n_result.is_ok() && er_result.is_ok(),
                    "HDF5 catalog rank conversion",
                )?;
                let n = n_result.expect("rank agreement");
                let er = er_result.expect("extra rank agreement");
                catalog_valid(
                    comm,
                    version == [FORMAT_VERSION]
                        && commit == [COMMIT_MARKER]
                        && n > 0
                        && n <= MAX_PROTOCOL_RANK
                        && er <= MAX_PROTOCOL_RANK
                        && n.checked_add(er).is_some_and(|x| x <= MAX_PROTOCOL_RANK),
                    "HDF5 catalog metadata",
                )?;
                let extra = if er == 0 {
                    let has_extra = attribute_exists_phase(
                        comm,
                        dataset,
                        ATTR_EXTRA,
                        "HDF5 catalog extra shape",
                    )
                    .map_err(CatalogError::Io)?;
                    catalog_valid(comm, !has_extra, "HDF5 catalog extra shape")?;
                    Vec::new()
                } else {
                    attr!(ATTR_EXTRA, Some(er), er)
                };
                let global = attr!(ATTR_GLOBAL, Some(n), n);
                let grid = attr!(ATTR_GRID, None, MAX_PROTOCOL_RANK);
                let perm = attr!(ATTR_PERM, Some(n), n);
                let product = global
                    .iter()
                    .chain(extra.iter())
                    .try_fold(1u64, |p, &x| p.checked_mul(x));
                let bytes_product = product.and_then(|x| x.checked_mul(scalar.width() as u64));
                let grid_product = grid.iter().try_fold(1u64, |p, &x| p.checked_mul(x));
                let shape_ok = !global.is_empty()
                    && !grid.is_empty()
                    && global.len() + extra.len() <= MAX_PROTOCOL_RANK
                    && bytes_product.is_some()
                    && grid.iter().all(|&x| x > 0)
                    && grid_product.is_some()
                    && is_permutation(&perm, n);
                catalog_valid(comm, shape_ok, "HDF5 catalog shape or provenance")?;
                let expected_result: Result<Vec<hsize_t>, _> = extra
                    .iter()
                    .chain(global.iter())
                    .map(|&x| hsize_t::try_from(x))
                    .collect();
                catalog_valid(
                    comm,
                    expected_result.is_ok(),
                    "HDF5 catalog native shape bounds",
                )?;
                let expected = expected_result.expect("native shape agreement");
                let (space, e) = local_handle_phase(
                    comm,
                    native::dataset_space(dataset),
                    "HDF5 catalog dataset space",
                );
                resources.file_space = space;
                if let Some(e) = e {
                    return Err(CatalogError::Io(e));
                }
                let actual_result = native::space_shape(resources.file_space.unwrap());
                catalog_valid(comm, actual_result.is_ok(), "HDF5 catalog dataset shape")?;
                let actual = actual_result
                    .map_err(|x| CatalogError::Io(hdf5_error("H5Sget_simple_extent_dims", x)))?;
                let actual_bytes: Vec<u8> = actual.iter().flat_map(|x| x.to_le_bytes()).collect();
                catalog_agree_bytes(comm, &actual_bytes)?;
                catalog_valid(comm, actual == expected, "HDF5 catalog dataset shape")?;
                let (dtype, e) = local_handle_phase(
                    comm,
                    native::dataset_type(dataset),
                    "HDF5 catalog datatype",
                );
                resources.datatype = dtype;
                if let Some(e) = e {
                    return Err(CatalogError::Io(e));
                }
                let type_result = native::type_matches(
                    resources.datatype.unwrap(),
                    scalar.code(),
                    scalar.width(),
                    duplicate.raw,
                );
                catalog_valid(comm, type_result.is_ok(), "HDF5 catalog datatype query")?;
                let type_ok = type_result
                    .map_err(|x| CatalogError::Io(hdf5_error("HDF5 catalog datatype", x)))?;
                catalog_agree_bytes(comm, &[u8::from(type_ok)])?;
                catalog_valid(comm, type_ok, "HDF5 catalog datatype")?;
                let name = if named {
                    let link_ok = nl > 0
                        && nl % 2 == 0
                        && nl <= MAX_NAME * 2
                        && link
                            .iter()
                            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(b));
                    catalog_valid(comm, link_ok, "HDF5 catalog named link")?;
                    let mut decoded = Vec::new();
                    let reserve = decoded.try_reserve_exact(nl / 2);
                    if let Err(e) =
                        agree_phase(comm, reserve.is_ok(), "HDF5 catalog name allocation")
                    {
                        return Err(CatalogError::Io(
                            reserve
                                .err()
                                .map(|_| IoError::AllocationFailed { requested: nl / 2 })
                                .unwrap_or(e),
                        ));
                    }
                    for p in link.chunks_exact(2) {
                        let hi = if p[0] <= b'9' {
                            p[0] - b'0'
                        } else {
                            p[0] - b'a' + 10
                        };
                        let lo = if p[1] <= b'9' {
                            p[1] - b'0'
                        } else {
                            p[1] - b'a' + 10
                        };
                        decoded.push((hi << 4) | lo);
                    }
                    let original = read_original_name(comm, dataset, "HDF5 catalog original name")
                        .map_err(CatalogError::Io)?;
                    catalog_agree_bytes(comm, &original)?;
                    let original_ok = !original.is_empty()
                        && original.len() <= MAX_NAME
                        && !original.contains(&0)
                        && std::str::from_utf8(&original).is_ok()
                        && original == decoded;
                    catalog_valid(comm, original_ok, "HDF5 catalog original name")?;
                    Some(String::from_utf8(decoded).expect("UTF-8 name agreement"))
                } else {
                    None
                };
                let provenance_len =
                    (grid.len() + perm.len())
                        .checked_mul(8)
                        .ok_or(CatalogError::Io(IoError::SizeLimit {
                            what: "catalog provenance",
                        }))?;
                let mut provenance = Vec::new();
                let reserve = provenance.try_reserve_exact(provenance_len);
                if let Err(e) =
                    agree_phase(comm, reserve.is_ok(), "HDF5 catalog provenance allocation")
                {
                    return Err(CatalogError::Io(
                        reserve
                            .err()
                            .map(|_| IoError::AllocationFailed {
                                requested: provenance_len,
                            })
                            .unwrap_or(e),
                    ));
                }
                for &x in grid.iter().chain(perm.iter()) {
                    provenance.extend_from_slice(&x.to_le_bytes());
                }
                let reserve = out.try_reserve_exact(1);
                if let Err(e) =
                    agree_phase(comm, reserve.is_ok(), "HDF5 catalog collection allocation")
                {
                    return Err(CatalogError::Io(
                        reserve
                            .err()
                            .map(|_| IoError::AllocationFailed { requested: 1 })
                            .unwrap_or(e),
                    ));
                }
                out.push(crate::catalog::DatasetInfo {
                    name,
                    scalar_type: scalar,
                    global_shape: global,
                    extra_shape: extra,
                    provenance,
                });
                let mut first = None;
                close_optional(
                    comm,
                    &mut resources.datatype,
                    &mut first,
                    native::type_close,
                    "H5Tclose(catalog)",
                );
                close_optional(
                    comm,
                    &mut resources.file_space,
                    &mut first,
                    native::dataspace_close,
                    "H5Sclose(catalog)",
                );
                close_optional(
                    comm,
                    &mut resources.dataset,
                    &mut first,
                    native::dataset_close,
                    "H5Dclose(catalog)",
                );
                if let Some(e) = first {
                    return Err(CatalogError::Io(e));
                }
            }
            let mut first = None;
            close_optional(
                comm,
                &mut resources.group,
                &mut first,
                native::group_close,
                "H5Gclose(catalog)",
            );
            if let Some(e) = first {
                return Err(CatalogError::Io(e));
            }
        }
        Ok(out)
    })();
    let agreement = agree_phase(comm, result.is_ok(), "HDF5 catalog validation");
    let cleanup = finish_hdf5(comm, duplicate, file, resources);
    match cleanup {
        Some(e) => Err(CatalogError::Io(e)),
        None => match (result, agreement) {
            (Err(e), _) => Err(e),
            (Ok(_), Err(e)) => Err(CatalogError::Io(e)),
            (Ok(x), Ok(())) => Ok(x),
        },
    }
}
