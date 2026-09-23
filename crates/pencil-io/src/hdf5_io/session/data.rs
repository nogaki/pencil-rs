use super::*;
use crate::IoError;
use crate::ffi;
use crate::ffi::hdf5 as native;
use crate::format::{IoElement, pack_view, prepare_physical_values};
use crate::hdf5_io::{Hdf5Settings, validate_options};
use crate::mpi_io::{agree_phase, descriptor_agreement};
use pencil_array::{PencilArrayView, PencilArrayViewMut};
use std::ffi::CStr;

impl<'c> Hdf5FileSession<'c> {
    /// Append a unique dataset. Preflight failures do not mutate the file;
    /// uncertain native mutations poison the session until collective close.
    pub fn write<T: IoElement, const N: usize, const M: usize>(
        &mut self,
        path: &str,
        view: PencilArrayView<'_, T, N, M>,
        options: &crate::hdf5_options::Hdf5WriteOptions,
    ) -> Result<(), Hdf5SessionError> {
        self.entry(0x705, path)?;
        agree_phase(
            self.comm,
            matches!(
                ffi::comm_is_congruent(
                    self.comm.as_raw(),
                    view.pencil().topology().communicator().as_raw()
                ),
                Ok(true)
            ),
            "session view communicator",
        )?;
        let comm = &self.comm;
        descriptor_agreement(
            comm,
            std::path::Path::new(path),
            OP_OPTIONS_WRITE_HDF5,
            view.pencil().global_shape(),
            view.extra_shape().dimensions(),
            view.pencil().topology().process_grid(),
            view.pencil().permutation().axes(),
            T::CODE,
            T::WIDTH,
        )
        .map_err(Hdf5SessionError::Io)?;
        let settings = options.settings();
        validate_options(
            comm,
            settings,
            N + view.extra_shape().dimensions().len(),
            T::WIDTH,
        )?;
        agree_settings(comm, settings)?;
        agree_member(
            comm,
            view.pencil(),
            view.extra_shape().dimensions(),
            T::CODE,
            T::WIDTH,
        )?;
        self.check_hints(settings)?;
        self.writable()?;
        let (parent, name) = self.parent(path)?;
        if let Err(e) = require_absent(comm, parent, &name) {
            close_group(comm, parent)?;
            return Err(e);
        }
        let result = session_write(
            comm,
            self.file.expect("open"),
            self.duplicate.expect("open"),
            parent,
            &name,
            view,
            settings,
            &mut self.poisoned,
        );
        result.map_err(Into::into)
    }
    /// Stage and validate a dataset, publishing only after operation cleanup.
    pub fn read<T: IoElement, const N: usize, const M: usize>(
        &mut self,
        path: &str,
        view: PencilArrayViewMut<'_, T, N, M>,
        options: &crate::hdf5_options::Hdf5ReadOptions,
    ) -> Result<(), Hdf5SessionError> {
        self.entry(0x706, path)?;
        agree_phase(
            self.comm,
            matches!(
                ffi::comm_is_congruent(
                    self.comm.as_raw(),
                    view.pencil().topology().communicator().as_raw()
                ),
                Ok(true)
            ),
            "session view communicator",
        )?;
        let comm = &self.comm;
        descriptor_agreement(
            comm,
            std::path::Path::new(path),
            OP_OPTIONS_READ_HDF5,
            view.pencil().global_shape(),
            view.extra_shape().dimensions(),
            view.pencil().topology().process_grid(),
            view.pencil().permutation().axes(),
            T::CODE,
            T::WIDTH,
        )
        .map_err(Hdf5SessionError::Io)?;
        let settings = options.settings();
        validate_options(
            comm,
            settings,
            N + view.extra_shape().dimensions().len(),
            T::WIDTH,
        )?;
        agree_settings(comm, settings)?;
        agree_member(
            comm,
            view.pencil(),
            view.extra_shape().dimensions(),
            T::CODE,
            T::WIDTH,
        )?;
        self.check_hints(settings)?;
        let (parent, name) = self.parent(path)?;
        if let Err(e) = check_kind(comm, parent, &name, native::LinkObjectType::Dataset) {
            close_group(comm, parent)?;
            return Err(e);
        }
        session_read(
            comm,
            self.duplicate.expect("open"),
            parent,
            &name,
            view,
            settings,
        )
        .map_err(Into::into)
    }
}
fn cleanup_session(
    comm: &mpi::topology::CartesianCommunicator,
    resources: Hdf5Resources,
    primary: IoError,
) -> IoError {
    finish_session(comm, resources).unwrap_or(primary)
}
pub(super) fn finish_session(
    comm: &mpi::topology::CartesianCommunicator,
    mut r: Hdf5Resources,
) -> Option<IoError> {
    let mut first = None;
    for (slot, close) in [
        (
            &mut r.xfer,
            native::plist_close as fn(&mut native::Hid) -> i32,
        ),
        (&mut r.dcpl, native::plist_close),
        (&mut r.mem_space, native::dataspace_close),
        (&mut r.file_space, native::dataspace_close),
        (&mut r.datatype, native::type_close),
        (&mut r.dataspace, native::dataspace_close),
        (&mut r.dataset, native::dataset_close),
        (&mut r.group, native::group_close),
    ] {
        close_optional(comm, slot, &mut first, close, "HDF5 session resource close");
    }
    aggregate_cleanup_result(comm, first)
}
#[allow(clippy::too_many_arguments)]
fn session_write<T, const N: usize, const M: usize>(
    comm: &mpi::topology::CartesianCommunicator,
    session_file: native::Hid,
    session_duplicate: ffi::MPI_Comm,
    group: native::Hid,
    name: &CStr,
    view: PencilArrayView<'_, T, N, M>,
    settings: Hdf5Settings<'_>,
    poisoned: &mut bool,
) -> Result<(), IoError>
where
    T: IoElement,
{
    let mut resources = Hdf5Resources {
        group: Some(group),
        ..Hdf5Resources::default()
    };
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
        return Err(cleanup_session(
            comm,
            resources,
            prep_error.unwrap_or(agreement),
        ));
    }
    let mut packed = packed_result.expect("HDF5 preparation agreement established");
    let layout = layout_result.expect("HDF5 preparation agreement established");
    let file = session_file;
    let duplicate = session_duplicate;

    let (datatype, error) = local_handle_phase(
        comm,
        native::type_from_kind(T::CODE, T::WIDTH, duplicate),
        "HDF5 datatype create",
    );
    resources.datatype = datatype;
    if let Some(error) = error {
        return Err(cleanup_session(comm, resources, error));
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
        return Err(cleanup_session(comm, resources, error));
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
            return Err(cleanup_session(comm, resources, error));
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
            return Err(cleanup_session(comm, resources, error));
        }
        if settings.shuffle {
            if let Err(error) = phase_code(
                comm,
                native::plist_set_shuffle(resources.dcpl.expect("dcpl agreement")),
                "HDF5 shuffle filter",
            ) {
                return Err(cleanup_session(comm, resources, error));
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
                return Err(cleanup_session(comm, resources, error));
            }
        }
    }
    let (xfer, error) = local_handle_phase(
        comm,
        native::xfer_create(duplicate, settings.collective),
        "HDF5 collective transfer plist",
    );
    resources.xfer = xfer;
    if let Some(error) = error {
        return Err(cleanup_session(comm, resources, error));
    }
    let xfer = resources.xfer.expect("transfer-property agreement");
    *poisoned = true;
    let dataset = match collective_handle_phase(
        comm,
        native::dataset_create(
            resources.group.expect("group creation agreement"),
            name,
            datatype,
            dataspace,
            resources.dcpl.unwrap_or(0),
        ),
        "HDF5 dataset create",
    ) {
        Ok(dataset) => dataset,
        Err(error) => return Err(cleanup_session(comm, resources, error)),
    };
    resources.dataset = Some(dataset);

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
        return Err(cleanup_session(comm, resources, error));
    }
    if let Err(error) = phase_code(
        comm,
        native::file_flush(file),
        "HDF5 initial metadata flush",
    ) {
        return Err(cleanup_session(comm, resources, error));
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
        return Err(cleanup_session(
            comm,
            Hdf5Resources {
                file_space,
                mem_space,
                ..resources
            },
            space_error.unwrap_or(agreement),
        ));
    }
    if let Some(error) = space_error {
        return Err(cleanup_session(
            comm,
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
        return Err(cleanup_session(comm, resources, error));
    }
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
        return Err(cleanup_session(comm, resources, primary));
    }
    if let Err(error) = phase_code(comm, native::file_flush(file), "HDF5 payload flush") {
        return Err(cleanup_session(comm, resources, error));
    }

    if write_existing_attr(comm, dataset, ATTR_COMMIT, COMMIT_MARKER).is_err() {
        return Err(cleanup_session(
            comm,
            resources,
            IoError::CommitUncertain {
                stage: "HDF5 commit marker",
            },
        ));
    }
    if phase_code(comm, native::file_flush(file), "HDF5 commit marker flush").is_err() {
        return Err(cleanup_session(
            comm,
            resources,
            IoError::CommitUncertain {
                stage: "HDF5 commit marker flush",
            },
        ));
    }

    if let Some(error) = finish_session(comm, resources) {
        return Err(error);
    }
    #[cfg(test)]
    test_post_op(comm, 1)?;
    *poisoned = false;
    Ok(())
}

fn session_read<T, const N: usize, const M: usize>(
    comm: &mpi::topology::CartesianCommunicator,
    session_duplicate: ffi::MPI_Comm,
    group: native::Hid,
    name: &CStr,
    mut view: PencilArrayViewMut<'_, T, N, M>,
    settings: Hdf5Settings<'_>,
) -> Result<(), IoError>
where
    T: IoElement,
{
    let mut resources = Hdf5Resources {
        group: Some(group),
        ..Hdf5Resources::default()
    };
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
        return Err(cleanup_session(
            comm,
            resources,
            prep_error.unwrap_or(agreement),
        ));
    }
    let layout = layout_result.expect("HDF5 preparation agreement established");
    staging.resize(
        staging_result.expect("HDF5 preparation agreement established"),
        0,
    );
    let duplicate = session_duplicate;
    let dataset =
        match collective_handle_phase(comm, native::dataset_open(group, name), "HDF5 dataset open")
        {
            Ok(dataset) => dataset,
            Err(error) => return Err(cleanup_session(comm, resources, error)),
        };
    resources.dataset = Some(dataset);

    macro_rules! read_attr {
        ($name:expr, $expected:expr, $max:expr, $phase:expr) => {
            match read_attr_phase(comm, duplicate, dataset, $name, $expected, $max, $phase) {
                Ok(values) => values,
                Err(error) => return Err(cleanup_session(comm, resources, error)),
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
        return Err(cleanup_session(comm, resources, local_error));
    }

    let extra = if extra_rank[0] == 0 {
        match attribute_exists_phase(comm, dataset, ATTR_EXTRA, "HDF5 zero-extra metadata") {
            Ok(false) => Vec::new(),
            Ok(true) => {
                return Err(cleanup_session(
                    comm,
                    resources,
                    IoError::MetadataMismatch {
                        field: "extra shape",
                    },
                ));
            }
            Err(error) => return Err(cleanup_session(comm, resources, error)),
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
        return Err(cleanup_session(
            comm,
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
        return Err(cleanup_session(comm, resources, error));
    }
    let datatype = resources.datatype.expect("dataset type agreement");
    let type_result = native::type_matches(datatype, T::CODE, T::WIDTH, duplicate);
    if let Err(agreement) = agree_phase(comm, type_result.is_ok(), "HDF5 datatype validation query")
    {
        return Err(cleanup_session(
            comm,
            resources,
            type_result
                .err()
                .map(|code| hdf5_error("HDF5 datatype validation query", code))
                .unwrap_or(agreement),
        ));
    }
    let type_ok = type_result.expect("datatype validation query agreement");
    if let Err(agreement) = collective_bool(comm, type_ok, "HDF5 datatype validation") {
        return Err(cleanup_session(comm, resources, agreement));
    }
    if !type_ok {
        return Err(cleanup_session(
            comm,
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
        return Err(cleanup_session(comm, resources, error));
    }
    let file_space = resources.file_space.expect("dataset-space agreement");
    let shape_result = native::space_shape(file_space);
    if let Err(agreement) = agree_phase(comm, shape_result.is_ok(), "HDF5 dataset dimensions query")
    {
        return Err(cleanup_session(
            comm,
            resources,
            shape_result
                .err()
                .map(|code| hdf5_error("H5Sget_simple_extent_dims", code))
                .unwrap_or(agreement),
        ));
    }
    let shape_ok = shape_result.expect("dataset dimensions agreement") == layout.global;
    if let Err(agreement) = collective_bool(comm, shape_ok, "HDF5 dataset dimensions") {
        return Err(cleanup_session(comm, resources, agreement));
    }
    if !shape_ok {
        return Err(cleanup_session(
            comm,
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
        return Err(cleanup_session(comm, resources, error));
    }
    let mem_space = resources.mem_space.expect("memory-space agreement");
    if let Err(error) = select_spaces(comm, file_space, mem_space, &layout) {
        return Err(cleanup_session(comm, resources, error));
    }

    let (xfer, error) = local_handle_phase(
        comm,
        native::xfer_create(duplicate, settings.collective),
        "HDF5 collective transfer plist",
    );
    resources.xfer = xfer;
    if let Some(error) = error {
        return Err(cleanup_session(comm, resources, error));
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
        return Err(cleanup_session(comm, resources, primary));
    }

    // Decode and map before any close.  After the collective prepare agreement
    // and successful cleanup, the destination is changed by one infallible
    // physical-order copy only.
    let values_result = prepare_physical_values(&view, &staging);
    if let Err(agreement) = agree_phase(comm, values_result.is_ok(), "HDF5 read physical staging") {
        return Err(cleanup_session(
            comm,
            resources,
            values_result.err().unwrap_or(agreement),
        ));
    }
    let values = values_result.expect("HDF5 read physical staging agreement");
    if let Some(error) = finish_session(comm, resources) {
        return Err(error);
    }
    #[cfg(test)]
    test_post_op(comm, 2)?;
    view.as_mut_slice().copy_from_slice(&values);
    Ok(())
}
