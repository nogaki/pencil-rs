use super::*;
use crate::IoError;
use crate::ffi;
use crate::ffi::hdf5 as native;
use crate::format::{IoElement, pack_view, prepare_physical_decode};
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
        self.entry_without_path(0x705)?;
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
            view.pencil().decomposition(),
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
        self.entry_without_path(0x706)?;
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
            view.pencil().decomposition(),
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
    finish_session(comm, resources);
    primary
}
pub(super) fn finish_session(comm: &mpi::topology::CartesianCommunicator, mut r: Hdf5Resources) {
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
        close_optional(comm, slot, close, "HDF5 session resource close");
    }
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

    if let Err(error) = create_type_and_space_handles(
        comm,
        duplicate,
        &mut resources,
        &layout,
        settings,
        T::CODE,
        T::WIDTH,
    ) {
        return Err(cleanup_session(comm, resources, error));
    }
    if let Err(error) = create_transfer_plist(comm, duplicate, &mut resources, settings) {
        return Err(cleanup_session(comm, resources, error));
    }
    *poisoned = true;
    if let Err(error) = create_dataset_with_metadata(
        comm,
        duplicate,
        file,
        &mut resources,
        name,
        None,
        view.pencil(),
        view.extra_shape().dimensions(),
        T::CODE,
        T::WIDTH,
    ) {
        return Err(cleanup_session(comm, resources, error));
    }
    if let Err(error) = open_hyperslab_spaces(comm, &mut resources, &layout) {
        return Err(cleanup_session(comm, resources, error));
    }
    if let Err(error) = write_payload_and_commit(comm, file, &mut resources, &layout, &mut packed) {
        return Err(cleanup_session(comm, resources, error));
    }

    finish_session(comm, resources);
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
    if let Err(error) = validate_and_open_dataset::<T, N, M>(
        comm,
        duplicate,
        &mut resources,
        group,
        name,
        None,
        view.pencil(),
        view.extra_shape().dimensions(),
        &layout,
    ) {
        return Err(cleanup_session(comm, resources, error));
    }
    if let Err(error) = read_payload(
        comm,
        duplicate,
        &mut resources,
        &layout,
        &mut staging,
        settings,
    ) {
        return Err(cleanup_session(comm, resources, error));
    }

    // Validate layout and allocate decode scratch before native cleanup.
    let plan_result = prepare_physical_decode(&view, &staging);
    if let Err(agreement) = agree_phase(comm, plan_result.is_ok(), "HDF5 read physical staging") {
        return Err(cleanup_session(
            comm,
            resources,
            plan_result.err().unwrap_or(agreement),
        ));
    }
    let plan = plan_result.expect("HDF5 read physical staging agreement");
    finish_session(comm, resources);
    #[cfg(test)]
    test_post_op(comm, 2)?;
    plan.decode_into(&staging, view.as_mut_slice());
    Ok(())
}
