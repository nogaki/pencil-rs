use std::ffi::CString;
use std::ops::Range;
use std::os::raw::c_int;
use std::path::Path;

use mpi::collective::{CommunicatorCollectives, SystemOperation};
use mpi::topology::Communicator;
use mpi::traits::AsRaw;

use crate::ffi;
use pencil_array::{PencilArrayView, PencilArrayViewMut, SpatialAxis};

use crate::format::{IoElement, element_count, pack_view, prepare_physical_values};
use crate::{
    COMMIT_MARKER, FORMAT_VERSION, INCOMPLETE_MARKER, IO_NAMESPACE, IoError, MAX_DESCRIPTOR_BYTES,
    MAX_HEADER_BYTES, MAX_PROTOCOL_RANK, OP_READ_MPI, OP_WRITE_MPI,
};

const HEADER_PREFIX_BYTES: usize = 96;
const MAGIC: &[u8; 8] = b"PENCILIO";
const COMMIT_OFFSET: i64 = 24;
const ROOT_RANK: i32 = 0;

/// A communicator duplicated for native I/O.  The duplicate owns its native
/// error handler; the caller's topology communicator is never retained.
#[derive(Debug)]
pub(crate) struct CommGuard {
    pub(crate) raw: ffi::MPI_Comm,
}

/// Temporarily changes the caller communicator only while MPI_Comm_dup is
/// entered.  MPI_Comm_get_errhandler returns a reference that must be freed
/// after the original handler has been restored.
struct ScopedOriginalErrhandler {
    comm: ffi::MPI_Comm,
    original: Option<ffi::MPI_Errhandler>,
}

impl ScopedOriginalErrhandler {
    fn begin(comm: ffi::MPI_Comm) -> Result<Self, (&'static str, i32)> {
        let original =
            ffi::comm_get_errhandler(comm).map_err(|code| ("MPI_Comm_get_errhandler", code))?;
        let mut guard = Self {
            comm,
            original: Some(original),
        };
        let set = ffi::comm_set_errors_return(comm);
        if set != ffi::MPI_SUCCESS as i32 {
            let cleanup = guard.finish();
            return Err(("MPI_Comm_set_errhandler", cleanup.err().unwrap_or(set)));
        }
        Ok(guard)
    }

    fn finish(&mut self) -> Result<(), i32> {
        let Some(mut original) = self.original.take() else {
            return Ok(());
        };
        // Always release the get_errhandler reference, even when restoring the
        // caller's handler fails.  The first native failure is reported.
        let restore = ffi::comm_set_errhandler(self.comm, original);
        let free = ffi::errhandler_free(&mut original);
        if restore != ffi::MPI_SUCCESS as i32 {
            Err(restore)
        } else if free != ffi::MPI_SUCCESS as i32 {
            Err(free)
        } else {
            Ok(())
        }
    }
}

impl Drop for ScopedOriginalErrhandler {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

/// Duplicates a topology communicator without allowing its default fatal
/// handler to terminate the process on MPI_Comm_dup.  Callers must not use the
/// topology communicator concurrently or overlap another collective while an
/// I/O operation is in progress.  Native MPI process failure remains outside
/// this recoverable Rust error contract.
pub(crate) fn duplicate_comm(
    comm: &mpi::topology::CartesianCommunicator,
) -> Result<CommGuard, IoError> {
    let mut original = match ScopedOriginalErrhandler::begin(comm.as_raw()) {
        Ok(original) => original,
        Err((_operation, _code)) => {
            // A rank that cannot install MPI_ERRORS_RETURN cannot safely
            // enter the following collective: another rank could still have
            // the caller's fatal handler. Fail-stop is the only bounded path
            // that avoids a rank-divergent MPI_Comm_dup.
            abort_unrecoverable(comm.as_raw(), "MPI communicator error-handler setup");
        }
    };

    // This collective is deliberately called while every rank has
    // MPI_ERRORS_RETURN installed on the original communicator.
    let duplicate_result = ffi::comm_dup(comm.as_raw());
    let (all_succeeded, mixed) = collective_state(comm, duplicate_result.is_ok());
    let restore_result = original.finish();
    let restored = restore_result.is_ok();
    let restore_agreement = agree_phase(
        comm,
        restored,
        "original communicator error-handler restoration",
    );

    if mixed {
        // MPI_Comm_free is not a legal subset operation for an incompletely
        // duplicated communicator.  Abort rather than return while successful
        // ranks still own an uncloseable native communicator.
        abort_unrecoverable(comm.as_raw(), "MPI_Comm_dup partial native communicator");
    }
    if !all_succeeded {
        if restore_agreement.is_err() {
            abort_unrecoverable(
                comm.as_raw(),
                "original communicator error-handler restoration",
            );
        }
        return Err(duplicate_result
            .err()
            .map(|code| IoError::Mpi {
                operation: "MPI_Comm_dup",
                code,
            })
            .unwrap_or(IoError::CollectivePrecondition {
                phase: "communicator duplication",
            }));
    }

    let duplicate = CommGuard {
        raw: duplicate_result.expect("communicator duplication agreement established"),
    };
    if restore_agreement.is_err() {
        let _ = finish_comm(comm, duplicate);
        abort_unrecoverable(
            comm.as_raw(),
            "original communicator error-handler restoration",
        );
    }
    let comm_set = ffi::comm_set_errors_return(duplicate.raw).checked_success();
    if let Err(agreement) = agree_phase(comm, comm_set.is_ok(), "communicator error-handler setup")
    {
        let primary = comm_set.err().map(|code| IoError::Mpi {
            operation: "MPI_Comm_set_errhandler",
            code,
        });
        let cleanup = finish_comm(comm, duplicate);
        return Err(cleanup.unwrap_or(primary.unwrap_or(agreement)));
    }
    Ok(duplicate)
}

/// Returns whether every rank succeeded and whether at least one rank
/// succeeded.  The second bit distinguishes an all-failed ordinary error from
/// a mixed native-handle state that cannot be cleaned up collectively.
pub(crate) fn collective_state<C: CommunicatorCollectives>(
    comm: &C,
    local_ok: bool,
) -> (bool, bool) {
    let local = i32::from(local_ok);
    let mut minimum = 0;
    let mut maximum = 0;
    comm.all_reduce_into(&local, &mut minimum, SystemOperation::min());
    comm.all_reduce_into(&local, &mut maximum, SystemOperation::max());
    (minimum == 1, maximum == 1 && minimum == 0)
}

pub(crate) fn abort_unrecoverable(comm: ffi::MPI_Comm, _operation: &'static str) -> ! {
    // MPI_Abort is the native fail-stop contract. A broken MPI can
    // unexpectedly return; never turn that into a recoverable Rust error.
    let _ = ffi::comm_abort(comm, 1);
    std::process::abort();
}

/// Writes one view collectively using native MPI-IO.
pub fn write_mpi<P, T, const N: usize, const M: usize>(
    path: P,
    view: PencilArrayView<'_, T, N, M>,
) -> Result<(), IoError>
where
    P: AsRef<Path>,
    T: IoElement,
{
    write_mpi_inner(path, view, false)
}

fn write_mpi_inner<P, T, const N: usize, const M: usize>(
    path: P,
    view: PencilArrayView<'_, T, N, M>,
    inject_post_cleanup_failure: bool,
) -> Result<(), IoError>
where
    P: AsRef<Path>,
    T: IoElement,
{
    let comm = view.pencil().topology().communicator();
    let global = view.pencil().global_shape();
    let extra = view.extra_shape().dimensions();
    let grid = view.pencil().topology().process_grid();
    let permutation = view.pencil().permutation().axes();
    descriptor_agreement(
        comm,
        path.as_ref(),
        OP_WRITE_MPI,
        global,
        extra,
        grid,
        permutation,
        T::CODE,
        T::WIDTH,
    )?;

    let path_c = match CString::new(path_bytes(path.as_ref())?) {
        Ok(path) => Ok(path),
        Err(_) => Err(IoError::InvalidPath),
    };
    let path_ok = path_c.is_ok();
    if let Err(agreement) = agree_phase(comm, path_ok, "path preparation") {
        return Err(if path_ok {
            agreement
        } else {
            path_c.unwrap_err()
        });
    }
    let path_c = path_c.expect("path agreement established");

    // These are all local preparation phases.  Every rank completes them before
    // the communicator is duplicated or the next collective is entered.
    let header_result = Header::for_write(&view);
    let packed_result = pack_view(&view);
    let layout_result = build_layout(
        global,
        extra,
        &view.local_spatial_shape(),
        view.pencil().local_ranges(),
        view.len(),
        T::WIDTH,
    );
    let prep_error = first_error3(&header_result, &packed_result, &layout_result);
    if let Err(agreement) = agree_phase(comm, prep_error.is_none(), "write preparation") {
        return Err(prep_error.unwrap_or(agreement));
    }
    let header = header_result.expect("write preparation agreement established");
    let packed = packed_result.expect("write preparation agreement established");
    let layout = layout_result.expect("write preparation agreement established");
    let header_count =
        c_int::try_from(header.bytes.len()).expect("validated header is below MPI count limit");
    let header_offset =
        to_offset(header.payload_offset).expect("validated payload offset fits MPI_Offset");
    let expected_size = header
        .payload_offset
        .checked_add(header.payload_bytes)
        .expect("validated file size does not overflow");

    let duplicate = duplicate_comm(comm)?;

    let file = match open_file_collective(comm, &duplicate, &path_c, true) {
        Ok(file) => file,
        Err(error) => return Err(cleanup_comm_error(comm, duplicate, error)),
    };
    let file_set = set_file_errors_return(&file);
    if let Err(agreement) = agree_phase(comm, file_set.is_ok(), "MPI-IO file error-handler setup") {
        let primary = file_set.err().map(|code| IoError::Mpi {
            operation: "MPI_File_set_errhandler",
            code,
        });
        let cleanup = finish_resources(comm, duplicate, file, None);
        return Err(cleanup.unwrap_or(primary.unwrap_or(agreement)));
    }

    let header_write = if comm.rank() == ROOT_RANK {
        ffi::file_write_at_all(file.raw, 0, &header.bytes)
    } else {
        ffi::file_write_at_all(file.raw, 0, &[])
    };
    let header_ok = match header_write {
        Ok(actual) => {
            (comm.rank() != ROOT_RANK && actual == 0)
                || (comm.rank() == ROOT_RANK && actual == header_count as usize)
        }
        Err(_) => false,
    };
    if let Err(agreement) = agree_phase(comm, header_ok, "MPI-IO header write") {
        let primary = IoError::WriteIncomplete {
            stage: "header write",
        };
        let cleanup = finish_resources(comm, duplicate, file, None);
        return Err(cleanup_result(
            cleanup,
            error_or_agreement(primary, agreement),
        ));
    }
    let header_flush = ffi::file_sync(file.raw).checked_success();
    if let Err(agreement) = agree_phase(comm, header_flush.is_ok(), "MPI-IO header flush") {
        let primary = IoError::WriteIncomplete {
            stage: "header flush",
        };
        let cleanup = finish_resources(comm, duplicate, file, None);
        let primary = header_flush
            .err()
            .map(|code| primary_from_mpi(primary, "MPI_File_sync", code))
            .unwrap_or(agreement);
        return Err(cleanup_result(cleanup, primary));
    }

    let mut datatype = None;
    if !layout.empty {
        if let Ok(raw) =
            ffi::type_create_subarray(&layout.global, &layout.local, &layout.starts, comm.as_raw())
        {
            datatype = Some(DatatypeGuard { raw });
        }
    }
    let datatype_ok = layout.empty || datatype.is_some();
    let (all_datatypes_ready, mixed_datatypes) = collective_state(comm, datatype_ok);
    if mixed_datatypes {
        abort_unrecoverable(comm.as_raw(), "MPI byte-subarray partial native datatype");
    }
    if !all_datatypes_ready {
        let primary = IoError::WriteIncomplete {
            stage: "byte-subarray preparation",
        };
        let cleanup = finish_resources(comm, duplicate, file, None);
        return Err(cleanup_result(
            cleanup,
            error_or_agreement(
                primary,
                IoError::CollectivePrecondition {
                    phase: "MPI byte-subarray preparation",
                },
            ),
        ));
    }
    let filetype = datatype
        .as_ref()
        .map_or_else(ffi::byte_datatype, |datatype| datatype.raw);
    let set_view = ffi::file_set_view(file.raw, header_offset, filetype).checked_success();
    if let Err(agreement) = agree_phase(comm, set_view.is_ok(), "MPI-IO byte-subarray view") {
        let primary = IoError::WriteIncomplete {
            stage: "MPI_File_set_view",
        };
        let cleanup = finish_resources(comm, duplicate, file, datatype);
        let primary = set_view
            .err()
            .map(|code| primary_from_mpi(primary, "MPI_File_set_view", code))
            .unwrap_or(agreement);
        return Err(cleanup_result(cleanup, primary));
    }

    let payload_write = ffi::file_write_all(file.raw, &packed);
    let payload_ok = matches!(payload_write, Ok(actual) if actual == packed.len());
    if let Err(agreement) = agree_phase(comm, payload_ok, "MPI-IO payload write") {
        let primary = IoError::WriteIncomplete {
            stage: "payload write",
        };
        let cleanup = finish_resources(comm, duplicate, file, datatype);
        return Err(cleanup_result(
            cleanup,
            error_or_agreement(primary, agreement),
        ));
    }
    let payload_flush = ffi::file_sync(file.raw).checked_success();
    if let Err(agreement) = agree_phase(comm, payload_flush.is_ok(), "MPI-IO payload flush") {
        let primary = IoError::WriteIncomplete {
            stage: "payload flush",
        };
        let cleanup = finish_resources(comm, duplicate, file, datatype);
        let primary = payload_flush
            .err()
            .map(|code| primary_from_mpi(primary, "MPI_File_sync", code))
            .unwrap_or(agreement);
        return Err(cleanup_result(cleanup, primary));
    }

    let size_ok = match ffi::file_get_size(file.raw) {
        Ok(size) => u64::try_from(size).ok() == Some(expected_size),
        Err(_) => false,
    };
    if let Err(agreement) = agree_phase(comm, size_ok, "MPI-IO payload file size") {
        let primary = IoError::WriteIncomplete {
            stage: "payload size",
        };
        let cleanup = finish_resources(comm, duplicate, file, datatype);
        return Err(cleanup_result(
            cleanup,
            error_or_agreement(primary, agreement),
        ));
    }

    // Explicit-offset MPI-IO offsets are expressed in the current view's
    // etypes.  Restore the byte view so the marker offset is truly at header
    // byte 24 rather than payload_offset + 24.
    let marker_view = ffi::file_set_view(file.raw, 0, ffi::byte_datatype()).checked_success();
    if let Err(agreement) = agree_phase(comm, marker_view.is_ok(), "MPI-IO commit marker view") {
        let primary = IoError::WriteIncomplete {
            stage: "commit marker view",
        };
        let cleanup = finish_resources(comm, duplicate, file, datatype);
        let primary = marker_view
            .err()
            .map(|code| primary_from_mpi(primary, "MPI_File_set_view", code))
            .unwrap_or(agreement);
        return Err(cleanup_result(cleanup, primary));
    }

    let marker = COMMIT_MARKER.to_le_bytes();
    let marker_write = if comm.rank() == ROOT_RANK {
        ffi::file_write_at_all(file.raw, COMMIT_OFFSET, &marker)
    } else {
        ffi::file_write_at_all(file.raw, COMMIT_OFFSET, &[])
    };
    let marker_ok = match marker_write {
        Ok(actual) => {
            (comm.rank() != ROOT_RANK && actual == 0)
                || (comm.rank() == ROOT_RANK && actual == marker.len())
        }
        Err(_) => false,
    };
    if let Err(agreement) = agree_phase(comm, marker_ok, "MPI-IO commit marker write") {
        let primary = IoError::CommitUncertain {
            stage: "commit marker write",
        };
        let cleanup = finish_resources(comm, duplicate, file, datatype);
        return Err(cleanup_result(
            cleanup,
            error_or_agreement(primary, agreement),
        ));
    }
    let marker_flush = ffi::file_sync(file.raw).checked_success();
    if let Err(agreement) = agree_phase(comm, marker_flush.is_ok(), "MPI-IO commit marker flush") {
        let primary = IoError::CommitUncertain {
            stage: "commit marker flush",
        };
        let cleanup = finish_resources(comm, duplicate, file, datatype);
        let primary = marker_flush
            .err()
            .map(|code| primary_from_mpi(primary, "MPI_File_sync", code))
            .unwrap_or(agreement);
        return Err(cleanup_result(cleanup, primary));
    }

    let cleanup = finish_resources(comm, duplicate, file, datatype);
    let injected = if inject_post_cleanup_failure && comm.rank() == ROOT_RANK {
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
pub(crate) fn write_mpi_with_postmarker_uncertainty<P, T, const N: usize, const M: usize>(
    path: P,
    view: PencilArrayView<'_, T, N, M>,
) -> Result<(), IoError>
where
    P: AsRef<Path>,
    T: IoElement,
{
    write_mpi_inner(path, view, true)
}

/// Reads one MPI-IO file collectively into a view, committing the destination
/// only after the complete read and all explicit resource closes succeed.
pub fn read_mpi<P, T, const N: usize, const M: usize>(
    path: P,
    view: PencilArrayViewMut<'_, T, N, M>,
) -> Result<(), IoError>
where
    P: AsRef<Path>,
    T: IoElement,
{
    read_mpi_inner(path, view, false)
}

fn read_mpi_inner<P, T, const N: usize, const M: usize>(
    path: P,
    mut view: PencilArrayViewMut<'_, T, N, M>,
    inject_post_cleanup_failure: bool,
) -> Result<(), IoError>
where
    P: AsRef<Path>,
    T: IoElement,
{
    let comm = view.pencil().topology().communicator();
    let global = view.pencil().global_shape();
    let extra = view.extra_shape().dimensions();
    let grid = view.pencil().topology().process_grid();
    let permutation = view.pencil().permutation().axes();
    descriptor_agreement(
        comm,
        path.as_ref(),
        OP_READ_MPI,
        global,
        extra,
        grid,
        permutation,
        T::CODE,
        T::WIDTH,
    )?;

    let path_c = match CString::new(path_bytes(path.as_ref())?) {
        Ok(path) => Ok(path),
        Err(_) => Err(IoError::InvalidPath),
    };
    let path_ok = path_c.is_ok();
    if let Err(agreement) = agree_phase(comm, path_ok, "path preparation") {
        return Err(if path_ok {
            agreement
        } else {
            path_c.unwrap_err()
        });
    }
    let path_c = path_c.expect("path agreement established");

    let global_shape = view.pencil().global_shape();
    let local_shape = view.local_spatial_shape();
    let layout_result = build_layout(
        global_shape,
        extra,
        &local_shape,
        view.pencil().local_ranges(),
        view.len(),
        T::WIDTH,
    );
    let staged_bytes = view.len().checked_mul(T::WIDTH).ok_or(IoError::SizeLimit {
        what: "local staging bytes",
    });
    let mut staging = Vec::new();
    let staging_result = match staged_bytes {
        Ok(bytes) => staging
            .try_reserve_exact(bytes)
            .map(|_| bytes)
            .map_err(|_| IoError::AllocationFailed { requested: bytes }),
        Err(error) => Err(error),
    };
    let prep_error = first_error2(&layout_result, &staging_result);
    if let Err(agreement) = agree_phase(comm, prep_error.is_none(), "read preparation") {
        return Err(prep_error.unwrap_or(agreement));
    }
    let layout = layout_result.expect("read preparation agreement established");
    let staging_len = staging_result.expect("read preparation agreement established");
    staging.resize(staging_len, 0);

    let duplicate = duplicate_comm(comm)?;

    let file = match open_file_collective(comm, &duplicate, &path_c, false) {
        Ok(file) => file,
        Err(error) => return Err(cleanup_comm_error(comm, duplicate, error)),
    };
    let file_set = set_file_errors_return(&file);
    if let Err(agreement) = agree_phase(comm, file_set.is_ok(), "MPI-IO file setup") {
        let primary = file_set.err().map(|code| IoError::Mpi {
            operation: "MPI_File_set_errhandler",
            code,
        });
        let cleanup = finish_resources(comm, duplicate, file, None);
        return Err(cleanup.unwrap_or(primary.unwrap_or(agreement)));
    }

    let file_size_result = ffi::file_get_size(file.raw);
    let size_ok = matches!(file_size_result, Ok(size) if size >= 0 && u64::try_from(size).is_ok());
    if let Err(agreement) = agree_phase(comm, size_ok, "MPI-IO file size") {
        let cleanup = finish_resources(comm, duplicate, file, None);
        return Err(cleanup.unwrap_or(if size_ok {
            agreement
        } else {
            IoError::InvalidFile {
                reason: "file size",
            }
        }));
    }
    let file_size = u64::try_from(file_size_result.expect("file size agreement established"))
        .expect("file size agreement established");
    let mut minimum_size = 0u64;
    let mut maximum_size = 0u64;
    comm.all_reduce_into(&file_size, &mut minimum_size, SystemOperation::min());
    comm.all_reduce_into(&file_size, &mut maximum_size, SystemOperation::max());
    if minimum_size != maximum_size {
        let cleanup = finish_resources(comm, duplicate, file, None);
        return Err(cleanup.unwrap_or(IoError::InvalidFile {
            reason: "file size differs between ranks",
        }));
    }
    if file_size < HEADER_PREFIX_BYTES as u64 {
        let cleanup = finish_resources(comm, duplicate, file, None);
        return Err(cleanup.unwrap_or(IoError::InvalidFile {
            reason: "short header",
        }));
    }

    let mut prefix = [0u8; HEADER_PREFIX_BYTES];
    let prefix_read = if comm.rank() == ROOT_RANK {
        ffi::file_read_at_all(file.raw, 0, &mut prefix)
    } else {
        ffi::file_read_at_all(file.raw, 0, &mut prefix[..0])
    };
    let prefix_ok = match prefix_read {
        Ok(actual) => {
            (comm.rank() != ROOT_RANK && actual == 0)
                || (comm.rank() == ROOT_RANK && actual == prefix.len())
        }
        Err(_) => false,
    };
    if let Err(agreement) = agree_phase(comm, prefix_ok, "MPI-IO header prefix read") {
        let cleanup = finish_resources(comm, duplicate, file, None);
        return Err(cleanup.unwrap_or(agreement));
    }
    let prefix_bcast = ffi::bcast_bytes(duplicate.raw, ROOT_RANK, &mut prefix).checked_success();
    if let Err(agreement) =
        agree_phase(comm, prefix_bcast.is_ok(), "MPI-IO header prefix broadcast")
    {
        let cleanup = finish_resources(comm, duplicate, file, None);
        let primary = prefix_bcast
            .err()
            .map(|code| IoError::Mpi {
                operation: "MPI_Bcast(header prefix)",
                code,
            })
            .unwrap_or(agreement);
        return Err(cleanup.unwrap_or(primary));
    }

    let prefix_result = HeaderPrefix::parse(&prefix);
    if let Err(agreement) = agree_phase(comm, prefix_result.is_ok(), "header prefix validation") {
        let cleanup = finish_resources(comm, duplicate, file, None);
        return Err(cleanup.unwrap_or(prefix_result.err().unwrap_or(agreement)));
    }
    let prefix_info = prefix_result.expect("header prefix agreement established");
    let header_len =
        usize::try_from(prefix_info.header_len).expect("validated header length fits usize");
    if u64::try_from(header_len).unwrap_or(u64::MAX) > file_size {
        let cleanup = finish_resources(comm, duplicate, file, None);
        return Err(cleanup.unwrap_or(IoError::InvalidFile {
            reason: "truncated header",
        }));
    }
    let mut full_header = Vec::new();
    let allocation_result =
        full_header
            .try_reserve_exact(header_len)
            .map_err(|_| IoError::AllocationFailed {
                requested: header_len,
            });
    if let Err(agreement) = agree_phase(comm, allocation_result.is_ok(), "header allocation") {
        let cleanup = finish_resources(comm, duplicate, file, None);
        return Err(cleanup.unwrap_or(allocation_result.err().unwrap_or(agreement)));
    }
    full_header.resize(header_len, 0);
    let full_read = if comm.rank() == ROOT_RANK {
        ffi::file_read_at_all(file.raw, 0, &mut full_header)
    } else {
        ffi::file_read_at_all(file.raw, 0, &mut full_header[..0])
    };
    let full_ok = match full_read {
        Ok(actual) => {
            (comm.rank() != ROOT_RANK && actual == 0)
                || (comm.rank() == ROOT_RANK && actual == header_len)
        }
        Err(_) => false,
    };
    if let Err(agreement) = agree_phase(comm, full_ok, "MPI-IO complete header read") {
        let cleanup = finish_resources(comm, duplicate, file, None);
        return Err(cleanup.unwrap_or(agreement));
    }
    let full_bcast = ffi::bcast_bytes(duplicate.raw, ROOT_RANK, &mut full_header).checked_success();
    if let Err(agreement) =
        agree_phase(comm, full_bcast.is_ok(), "MPI-IO complete header broadcast")
    {
        let cleanup = finish_resources(comm, duplicate, file, None);
        let primary = full_bcast
            .err()
            .map(|code| IoError::Mpi {
                operation: "MPI_Bcast(header)",
                code,
            })
            .unwrap_or(agreement);
        return Err(cleanup.unwrap_or(primary));
    }

    let header_result = Header::decode(&full_header);
    let header_matches = match &header_result {
        Ok(header) => header
            .matches_view(global_shape, extra, T::CODE, T::WIDTH)
            .and_then(|()| {
                let expected_size = header
                    .payload_offset
                    .checked_add(header.payload_bytes)
                    .ok_or(IoError::InvalidFile {
                        reason: "payload size overflow",
                    })?;
                if expected_size != file_size {
                    return Err(IoError::InvalidFile {
                        reason: "trailing or truncated payload",
                    });
                }
                Ok(())
            }),
        Err(error) => Err(error.clone()),
    };
    if let Err(agreement) = agree_phase(comm, header_matches.is_ok(), "header metadata validation")
    {
        let cleanup = finish_resources(comm, duplicate, file, None);
        return Err(cleanup.unwrap_or(header_matches.err().unwrap_or(agreement)));
    }
    let header = header_result.expect("header metadata agreement established");
    let header_offset = match to_offset(header.payload_offset) {
        Ok(offset) => offset,
        Err(error) => {
            let cleanup = finish_resources(comm, duplicate, file, None);
            return Err(cleanup.unwrap_or(error));
        }
    };

    let mut datatype = None;
    if !layout.empty {
        if let Ok(raw) =
            ffi::type_create_subarray(&layout.global, &layout.local, &layout.starts, comm.as_raw())
        {
            datatype = Some(DatatypeGuard { raw });
        }
    }
    let datatype_ok = layout.empty || datatype.is_some();
    let (all_datatypes_ready, mixed_datatypes) = collective_state(comm, datatype_ok);
    if mixed_datatypes {
        abort_unrecoverable(comm.as_raw(), "MPI byte-subarray partial native datatype");
    }
    if !all_datatypes_ready {
        let cleanup = finish_resources(comm, duplicate, file, None);
        return Err(cleanup.unwrap_or(IoError::CollectivePrecondition {
            phase: "MPI byte-subarray preparation",
        }));
    }
    let filetype = datatype
        .as_ref()
        .map_or_else(ffi::byte_datatype, |datatype| datatype.raw);
    let set_view = ffi::file_set_view(file.raw, header_offset, filetype).checked_success();
    if let Err(agreement) = agree_phase(comm, set_view.is_ok(), "MPI-IO byte-subarray view") {
        let cleanup = finish_resources(comm, duplicate, file, datatype);
        let primary = set_view
            .err()
            .map(|code| IoError::Mpi {
                operation: "MPI_File_set_view",
                code,
            })
            .unwrap_or(agreement);
        return Err(cleanup.unwrap_or(primary));
    }

    let payload_read = ffi::file_read_all(file.raw, &mut staging);
    let payload_ok = matches!(payload_read, Ok(actual) if actual == staging.len());
    if let Err(agreement) = agree_phase(comm, payload_ok, "MPI-IO payload read") {
        let cleanup = finish_resources(comm, duplicate, file, datatype);
        return Err(cleanup.unwrap_or(agreement));
    }

    // Decode, map, and allocate the final physical-order buffer while every
    // native resource is still owned.  No rank can reach cleanup or commit
    // alone after this agreement.
    let values_result = prepare_physical_values(&view, &staging);
    if let Err(agreement) = agree_phase(comm, values_result.is_ok(), "MPI-IO read physical staging")
    {
        let cleanup = finish_resources(comm, duplicate, file, datatype);
        return Err(cleanup.unwrap_or(values_result.err().unwrap_or(agreement)));
    }
    let values = values_result.expect("read physical staging agreement established");
    let cleanup = finish_resources(comm, duplicate, file, datatype);
    let injected = if inject_post_cleanup_failure && comm.rank() == ROOT_RANK {
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
pub(crate) fn read_mpi_with_post_cleanup_failure<P, T, const N: usize, const M: usize>(
    path: P,
    view: PencilArrayViewMut<'_, T, N, M>,
) -> Result<(), IoError>
where
    P: AsRef<Path>,
    T: IoElement,
{
    read_mpi_inner(path, view, true)
}

#[derive(Debug)]
struct FileGuard {
    raw: ffi::MPI_File,
}

#[derive(Debug)]
struct DatatypeGuard {
    raw: ffi::MPI_Datatype,
}

fn open_file_collective(
    comm: &mpi::topology::CartesianCommunicator,
    duplicate: &CommGuard,
    path: &std::ffi::CStr,
    write: bool,
) -> Result<FileGuard, IoError> {
    let result = ffi::file_open(duplicate.raw, path, write);
    let (all_succeeded, mixed) = collective_state(comm, result.is_ok());
    if mixed {
        abort_unrecoverable(comm.as_raw(), "MPI_File_open partial native file handle");
    }
    if !all_succeeded {
        return Err(result
            .err()
            .map(|code| IoError::Mpi {
                operation: "MPI_File_open",
                code,
            })
            .unwrap_or(IoError::CollectivePrecondition {
                phase: "MPI-IO file open",
            }));
    }
    Ok(FileGuard {
        raw: result.expect("MPI-IO file-open agreement established"),
    })
}

fn set_file_errors_return(file: &FileGuard) -> Result<(), i32> {
    ffi::file_set_errors_return(file.raw).checked_success()
}

fn cleanup_comm_error(
    comm: &mpi::topology::CartesianCommunicator,
    duplicate: CommGuard,
    primary: IoError,
) -> IoError {
    finish_comm(comm, duplicate).unwrap_or(primary)
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

/// Aggregates a post-cleanup result before a caller mutates its destination.
/// Native close failures fail-stop in the cleanup functions; this extra
/// agreement is for the remaining synthetic/test result and keeps the commit
/// point collective even when only one rank reports it.
pub(crate) fn aggregate_cleanup_result<C: CommunicatorCollectives>(
    comm: &C,
    local: Option<IoError>,
) -> Option<IoError> {
    let failed = i32::from(local.is_some());
    let mut any_failed = 0;
    comm.all_reduce_into(&failed, &mut any_failed, SystemOperation::max());
    if any_failed == 0 {
        None
    } else {
        local.or(Some(IoError::CollectivePrecondition {
            phase: "post-cleanup result",
        }))
    }
}

fn finish_comm(
    comm: &mpi::topology::CartesianCommunicator,
    mut duplicate: CommGuard,
) -> Option<IoError> {
    let code = ffi::comm_free(&mut duplicate.raw);
    let local_ok = code == ffi::MPI_SUCCESS as i32;
    let (all_closed, _mixed_close) = collective_state(comm, local_ok);
    if !all_closed {
        // MPI_Comm_free may have left this rank's duplicate live, or may have
        // invalidated it. Abort through the still-live original topology
        // communicator, never through the post-free duplicate.
        abort_unrecoverable(comm.as_raw(), "MPI_Comm_free unrecoverable native handle");
    }
    None
}

fn finish_resources(
    comm: &mpi::topology::CartesianCommunicator,
    mut duplicate: CommGuard,
    mut file: FileGuard,
    mut datatype: Option<DatatypeGuard>,
) -> Option<IoError> {
    // Derived datatypes must be released before closing the file that may
    // still reference their file view.  Every rank participates in each
    // agreement even when its local datatype is absent (empty rank).
    let datatype_code = datatype
        .as_mut()
        .map_or(ffi::MPI_SUCCESS as i32, |datatype| {
            ffi::type_free(&mut datatype.raw)
        });
    let (all_datatypes_closed, _mixed_datatype_close) =
        collective_state(comm, datatype_code == ffi::MPI_SUCCESS as i32);
    if !all_datatypes_closed {
        // Keep the failed datatype attached to the operation and stop rather
        // than freeing the communicator while a native handle may remain.
        abort_unrecoverable(comm.as_raw(), "MPI_Type_free unrecoverable native handle");
    }

    let file_code = ffi::file_close(&mut file.raw);
    let file_local_ok = file_code == ffi::MPI_SUCCESS as i32;
    let (all_files_closed, _mixed_file_close) = collective_state(comm, file_local_ok);
    if !all_files_closed {
        // A failed collective close may leave the file attached to the
        // duplicate communicator. Do not free that communicator after a
        // mixed/all-failed close; terminate the bounded native-resource state.
        abort_unrecoverable(comm.as_raw(), "MPI_File_close unrecoverable native handle");
    }

    let comm_code = ffi::comm_free(&mut duplicate.raw);
    let (all_closed, _mixed_close) = collective_state(comm, comm_code == ffi::MPI_SUCCESS as i32);
    if !all_closed {
        // The duplicate was just passed to MPI_Comm_free. It may now be
        // MPI_COMM_NULL or indeterminate, so fail-stop on the original
        // topology communicator instead.
        abort_unrecoverable(comm.as_raw(), "MPI_Comm_free unrecoverable native handle");
    }
    None
}

pub(crate) fn agree_phase<C>(comm: &C, local_ok: bool, phase: &'static str) -> Result<(), IoError>
where
    C: CommunicatorCollectives,
{
    let local = i32::from(local_ok);
    let mut all = 0;
    comm.all_reduce_into(&local, &mut all, SystemOperation::min());
    if all == 1 {
        Ok(())
    } else {
        Err(IoError::CollectivePrecondition { phase })
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn descriptor_agreement<const N: usize, const M: usize>(
    comm: &mpi::topology::CartesianCommunicator,
    path: &Path,
    operation: u64,
    global: &[usize; N],
    extra: &[usize],
    grid: &[usize; M],
    permutation: &[SpatialAxis; N],
    type_code: u64,
    width: usize,
) -> Result<(), IoError> {
    let descriptor = build_descriptor(
        path,
        operation,
        global,
        extra,
        grid,
        permutation,
        type_code,
        width,
    );
    let local_valid = descriptor.is_ok();
    let local_len = descriptor.as_ref().map_or(0, |descriptor| descriptor.len);
    let fixed = [
        IO_NAMESPACE,
        operation,
        FORMAT_VERSION,
        u64::try_from(local_len).unwrap_or(u64::MAX),
        u64::from(local_valid),
    ];
    let mut minimum = [0u64; 5];
    let mut maximum = [0u64; 5];
    comm.all_reduce_into(&fixed, &mut minimum, SystemOperation::min());
    comm.all_reduce_into(&fixed, &mut maximum, SystemOperation::max());
    if minimum != maximum {
        return Err(if local_valid {
            IoError::CollectiveDescriptorMismatch
        } else {
            descriptor
                .err()
                .unwrap_or(IoError::CollectiveDescriptorMismatch)
        });
    }
    if !local_valid {
        return Err(descriptor
            .err()
            .unwrap_or(IoError::CollectiveDescriptorMismatch));
    }
    let descriptor = descriptor.expect("descriptor validity established");
    let ranks = usize::try_from(comm.size()).map_err(|_| IoError::SizeLimit {
        what: "communicator size",
    })?;
    let total = descriptor
        .len
        .checked_mul(ranks)
        .ok_or(IoError::SizeLimit {
            what: "descriptor allgather",
        })?;
    if total > 64 * 1024 * 1024 {
        return Err(IoError::SizeLimit {
            what: "descriptor allgather",
        });
    }
    let mut gathered = Vec::new();
    let reserve = gathered.try_reserve_exact(total);
    if let Err(agreement) = agree_phase(comm, reserve.is_ok(), "descriptor allgather allocation") {
        return Err(if reserve.is_err() {
            IoError::AllocationFailed { requested: total }
        } else {
            agreement
        });
    }
    gathered.resize(total, 0);
    comm.all_gather_into(&descriptor.bytes[..descriptor.len], &mut gathered);
    let matches = gathered
        .chunks_exact(descriptor.len)
        .all(|chunk| chunk == &descriptor.bytes[..descriptor.len]);
    if matches {
        Ok(())
    } else {
        Err(IoError::CollectiveDescriptorMismatch)
    }
}

#[derive(Clone, Copy)]
struct Descriptor {
    bytes: [u8; MAX_DESCRIPTOR_BYTES],
    len: usize,
}

#[allow(clippy::too_many_arguments)]
fn build_descriptor<const N: usize, const M: usize>(
    path: &Path,
    operation: u64,
    global: &[usize; N],
    extra: &[usize],
    grid: &[usize; M],
    permutation: &[SpatialAxis; N],
    type_code: u64,
    width: usize,
) -> Result<Descriptor, IoError> {
    let path = path_bytes(path)?;
    if path.contains(&0) {
        return Err(IoError::InvalidPath);
    }
    let mut descriptor = Descriptor {
        bytes: [0; MAX_DESCRIPTOR_BYTES],
        len: 0,
    };
    descriptor.push_u64(IO_NAMESPACE)?;
    descriptor.push_u64(operation)?;
    descriptor.push_u64(FORMAT_VERSION)?;
    descriptor.push_u64(u64::try_from(N).map_err(|_| IoError::SizeLimit { what: "rank" })?)?;
    descriptor.push_u64(u64::try_from(M).map_err(|_| IoError::SizeLimit {
        what: "topology rank",
    })?)?;
    descriptor.push_u64(type_code)?;
    descriptor
        .push_u64(u64::try_from(width).map_err(|_| IoError::SizeLimit { what: "type width" })?)?;
    descriptor.push_u64(
        u64::try_from(extra.len()).map_err(|_| IoError::SizeLimit { what: "extra rank" })?,
    )?;
    for &extent in global {
        descriptor
            .push_u64(u64::try_from(extent).map_err(|_| IoError::SizeLimit { what: "shape" })?)?;
    }
    for &extent in extra {
        descriptor.push_u64(u64::try_from(extent).map_err(|_| IoError::SizeLimit {
            what: "extra shape",
        })?)?;
    }
    for &extent in grid {
        descriptor.push_u64(u64::try_from(extent).map_err(|_| IoError::SizeLimit {
            what: "process grid",
        })?)?;
    }
    for axis in permutation {
        descriptor.push_u64(u64::try_from(axis.index()).map_err(|_| IoError::SizeLimit {
            what: "permutation",
        })?)?;
    }
    descriptor
        .push_u64(u64::try_from(path.len()).map_err(|_| IoError::SizeLimit { what: "path" })?)?;
    descriptor.push_bytes(path)?;
    Ok(descriptor)
}

impl Descriptor {
    fn push_u64(&mut self, value: u64) -> Result<(), IoError> {
        self.push_bytes(&value.to_le_bytes())
    }

    fn push_bytes(&mut self, bytes: &[u8]) -> Result<(), IoError> {
        let end = self
            .len
            .checked_add(bytes.len())
            .ok_or(IoError::SizeLimit { what: "descriptor" })?;
        if end > self.bytes.len() {
            return Err(IoError::SizeLimit { what: "descriptor" });
        }
        self.bytes[self.len..end].copy_from_slice(bytes);
        self.len = end;
        Ok(())
    }
}

struct Header {
    bytes: Vec<u8>,
    n: usize,
    type_code: u64,
    width: usize,
    extra: Vec<u64>,
    global: Vec<u64>,
    writer_grid: Vec<u64>,
    writer_permutation: Vec<u64>,
    payload_offset: u64,
    payload_bytes: u64,
}

impl Header {
    fn for_write<T, const N: usize, const M: usize>(
        view: &PencilArrayView<'_, T, N, M>,
    ) -> Result<Self, IoError>
    where
        T: IoElement,
    {
        let extra = view.extra_shape().dimensions();
        let grid = view.pencil().topology().process_grid();
        let spatial_elements = element_count(view.pencil().global_shape())?;
        let extra_elements = element_count(extra)?;
        let global_elements =
            spatial_elements
                .checked_mul(extra_elements)
                .ok_or(IoError::SizeLimit {
                    what: "global element count",
                })?;
        let payload_bytes = global_elements
            .checked_mul(T::WIDTH)
            .ok_or(IoError::SizeLimit {
                what: "global payload bytes",
            })?;
        let array_count = N
            .checked_add(extra.len())
            .and_then(|count| count.checked_add(M))
            .and_then(|count| count.checked_add(N))
            .ok_or(IoError::SizeLimit {
                what: "header dimensions",
            })?;
        let header_len = HEADER_PREFIX_BYTES
            .checked_add(array_count.checked_mul(8).ok_or(IoError::SizeLimit {
                what: "header bytes",
            })?)
            .ok_or(IoError::SizeLimit {
                what: "header bytes",
            })?;
        if header_len > MAX_HEADER_BYTES {
            return Err(IoError::SizeLimit {
                what: "header bytes",
            });
        }
        let payload_offset = u64::try_from(header_len).map_err(|_| IoError::SizeLimit {
            what: "payload offset",
        })?;
        let payload_bytes = u64::try_from(payload_bytes).map_err(|_| IoError::SizeLimit {
            what: "payload bytes",
        })?;
        payload_offset
            .checked_add(payload_bytes)
            .ok_or(IoError::SizeLimit { what: "file size" })?;
        let extra_values = try_u64_values(extra, "extra shape")?;
        let global_values = try_u64_values(view.pencil().global_shape(), "global shape")?;
        let grid_values = try_u64_values(grid, "process grid")?;
        let mut permutation_values = Vec::new();
        permutation_values
            .try_reserve_exact(N)
            .map_err(|_| IoError::AllocationFailed {
                requested: N * std::mem::size_of::<u64>(),
            })?;
        for axis in view.pencil().permutation().axes() {
            permutation_values.push(u64::try_from(axis.index()).map_err(|_| {
                IoError::SizeLimit {
                    what: "permutation",
                }
            })?);
        }
        let mut header = Self {
            bytes: Vec::new(),
            n: N,
            type_code: T::CODE,
            width: T::WIDTH,
            extra: extra_values,
            global: global_values,
            writer_grid: grid_values,
            writer_permutation: permutation_values,
            payload_offset,
            payload_bytes,
        };
        header
            .bytes
            .try_reserve_exact(header_len)
            .map_err(|_| IoError::AllocationFailed {
                requested: header_len,
            })?;
        header.bytes.resize(header_len, 0);
        header.encode_into();
        Ok(header)
    }

    fn encode_into(&mut self) {
        self.bytes[0..8].copy_from_slice(MAGIC);
        put_u64(&mut self.bytes, 8, FORMAT_VERSION);
        let header_len = self.bytes.len() as u64;
        put_u64(&mut self.bytes, 16, header_len);
        put_u64(&mut self.bytes, 24, INCOMPLETE_MARKER);
        put_u64(&mut self.bytes, 32, self.n as u64);
        put_u64(&mut self.bytes, 40, self.type_code);
        put_u64(&mut self.bytes, 48, self.width as u64);
        put_u64(&mut self.bytes, 56, self.extra.len() as u64);
        put_u64(&mut self.bytes, 64, self.payload_offset);
        put_u64(&mut self.bytes, 72, self.payload_bytes);
        put_u64(&mut self.bytes, 80, self.writer_grid.len() as u64);
        put_u64(&mut self.bytes, 88, 0);
        let mut offset = HEADER_PREFIX_BYTES;
        for values in [
            &self.global,
            &self.extra,
            &self.writer_grid,
            &self.writer_permutation,
        ] {
            for &value in values.as_slice() {
                put_u64(&mut self.bytes, offset, value);
                offset += 8;
            }
        }
        debug_assert_eq!(offset, self.bytes.len());
    }

    fn parse_prefix(bytes: &[u8; HEADER_PREFIX_BYTES]) -> Result<HeaderPrefix, IoError> {
        if &bytes[0..8] != MAGIC {
            return Err(IoError::InvalidFile { reason: "magic" });
        }
        if get_u64(bytes, 8) != FORMAT_VERSION {
            return Err(IoError::InvalidFile { reason: "version" });
        }
        match get_u64(bytes, COMMIT_OFFSET as usize) {
            COMMIT_MARKER => {}
            INCOMPLETE_MARKER => return Err(IoError::IncompleteFile),
            _ => {
                return Err(IoError::InvalidFile {
                    reason: "commit marker",
                });
            }
        }
        let header_len = get_u64(bytes, 16);
        let n = get_u64(bytes, 32);
        let extra_rank = get_u64(bytes, 56);
        let writer_grid_rank = get_u64(bytes, 80);
        if n > MAX_PROTOCOL_RANK as u64
            || extra_rank > MAX_PROTOCOL_RANK as u64
            || writer_grid_rank > MAX_PROTOCOL_RANK as u64
        {
            return Err(IoError::InvalidFile {
                reason: "rank bound",
            });
        }
        let array_count = n
            .checked_add(extra_rank)
            .and_then(|count| count.checked_add(writer_grid_rank))
            .and_then(|count| count.checked_add(n))
            .ok_or(IoError::InvalidFile {
                reason: "header dimensions",
            })?;
        let expected_len = u64::try_from(HEADER_PREFIX_BYTES)
            .ok()
            .and_then(|prefix| array_count.checked_mul(8)?.checked_add(prefix))
            .ok_or(IoError::InvalidFile {
                reason: "header length",
            })?;
        if header_len != expected_len || header_len as usize > MAX_HEADER_BYTES {
            return Err(IoError::InvalidFile {
                reason: "header length",
            });
        }
        if get_u64(bytes, 64) != header_len {
            return Err(IoError::InvalidFile {
                reason: "payload offset",
            });
        }
        if !valid_type_pair(get_u64(bytes, 40), get_u64(bytes, 48)) {
            return Err(IoError::InvalidFile {
                reason: "type descriptor",
            });
        }
        Ok(HeaderPrefix {
            header_len,
            type_code: get_u64(bytes, 40),
            width: get_u64(bytes, 48),
        })
    }

    fn decode(bytes: &[u8]) -> Result<Self, IoError> {
        if bytes.len() < HEADER_PREFIX_BYTES || bytes.len() > MAX_HEADER_BYTES {
            return Err(IoError::InvalidFile {
                reason: "header size",
            });
        }
        let mut prefix = [0u8; HEADER_PREFIX_BYTES];
        prefix.copy_from_slice(&bytes[..HEADER_PREFIX_BYTES]);
        let info = Self::parse_prefix(&prefix)?;
        if usize::try_from(info.header_len).ok() != Some(bytes.len()) {
            return Err(IoError::InvalidFile {
                reason: "header trailing bytes",
            });
        }
        let n = usize::try_from(get_u64(&prefix, 32))
            .map_err(|_| IoError::InvalidFile { reason: "rank" })?;
        let extra_rank =
            usize::try_from(get_u64(&prefix, 56)).map_err(|_| IoError::InvalidFile {
                reason: "extra rank",
            })?;
        let writer_grid_rank =
            usize::try_from(get_u64(&prefix, 80)).map_err(|_| IoError::InvalidFile {
                reason: "grid rank",
            })?;
        let mut offset = HEADER_PREFIX_BYTES;
        let mut read_values = |count: usize| -> Result<Vec<u64>, IoError> {
            let bytes_count = count.checked_mul(8).ok_or(IoError::InvalidFile {
                reason: "header array",
            })?;
            let end = offset
                .checked_add(bytes_count)
                .ok_or(IoError::InvalidFile {
                    reason: "header array",
                })?;
            if end > bytes.len() {
                return Err(IoError::InvalidFile {
                    reason: "header array",
                });
            }
            let mut values = Vec::new();
            values
                .try_reserve_exact(count)
                .map_err(|_| IoError::AllocationFailed {
                    requested: bytes_count,
                })?;
            for chunk in bytes[offset..end].chunks_exact(8) {
                values.push(u64::from_le_bytes(
                    chunk.try_into().expect("eight-byte chunk"),
                ));
            }
            offset = end;
            Ok(values)
        };
        let global = read_values(n)?;
        let extra = read_values(extra_rank)?;
        let writer_grid = read_values(writer_grid_rank)?;
        let writer_permutation = read_values(n)?;
        if offset != bytes.len()
            || !valid_dimensions(&global)
            || !valid_extents(&writer_grid)
            || !is_permutation(&writer_permutation, n)
        {
            return Err(IoError::InvalidFile {
                reason: "header metadata",
            });
        }
        let expected_elements = global
            .iter()
            .chain(extra.iter())
            .try_fold(1u64, |product, &extent| product.checked_mul(extent))
            .ok_or(IoError::InvalidFile {
                reason: "payload element count",
            })?;
        let expected_payload =
            expected_elements
                .checked_mul(info.width)
                .ok_or(IoError::InvalidFile {
                    reason: "payload bytes",
                })?;
        if expected_payload != get_u64(&prefix, 72) {
            return Err(IoError::InvalidFile {
                reason: "payload bytes",
            });
        }
        let mut stored_bytes = Vec::new();
        stored_bytes
            .try_reserve_exact(bytes.len())
            .map_err(|_| IoError::AllocationFailed {
                requested: bytes.len(),
            })?;
        stored_bytes.extend_from_slice(bytes);
        Ok(Self {
            bytes: stored_bytes,
            n,
            type_code: info.type_code,
            width: usize::try_from(info.width).map_err(|_| IoError::InvalidFile {
                reason: "type width",
            })?,
            extra,
            global,
            writer_grid,
            writer_permutation,
            payload_offset: get_u64(&prefix, 64),
            payload_bytes: get_u64(&prefix, 72),
        })
    }

    fn matches_view<const N: usize>(
        &self,
        global: &[usize; N],
        extra: &[usize],
        type_code: u64,
        width: usize,
    ) -> Result<(), IoError> {
        if self.n != N || self.type_code != type_code || self.width != width {
            return Err(IoError::MetadataMismatch {
                field: "type or rank",
            });
        }
        if self.extra.len() != extra.len()
            || self
                .extra
                .iter()
                .zip(extra)
                .any(|(&stored, &expected)| stored != expected as u64)
        {
            return Err(IoError::MetadataMismatch {
                field: "extra shape",
            });
        }
        if self.global.len() != N
            || self
                .global
                .iter()
                .zip(global)
                .any(|(&stored, &expected)| stored != expected as u64)
        {
            return Err(IoError::MetadataMismatch {
                field: "global shape",
            });
        }
        let expected_elements = global.iter().copied().try_fold(1u64, |product, extent| {
            product
                .checked_mul(extent as u64)
                .and_then(|product| product.checked_mul(1))
        });
        let expected_extra = extra
            .iter()
            .copied()
            .try_fold(1u64, |product, extent| product.checked_mul(extent as u64));
        let expected_payload = expected_elements
            .and_then(|elements| expected_extra.and_then(|extra| elements.checked_mul(extra)))
            .and_then(|elements| elements.checked_mul(width as u64))
            .ok_or(IoError::SizeLimit {
                what: "expected payload",
            })?;
        if expected_payload != self.payload_bytes {
            return Err(IoError::MetadataMismatch {
                field: "payload size",
            });
        }
        Ok(())
    }
}

struct HeaderPrefix {
    header_len: u64,
    type_code: u64,
    width: u64,
}

impl HeaderPrefix {
    fn parse(bytes: &[u8; HEADER_PREFIX_BYTES]) -> Result<Self, IoError> {
        Header::parse_prefix(bytes)
    }
}

fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn get_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("eight-byte header field"),
    )
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

fn valid_type_pair(code: u64, width: u64) -> bool {
    matches!(
        (code, width),
        (1 | 2, 1) | (3 | 4, 2) | (5 | 6 | 9, 4) | (7 | 8 | 10, 8) | (11, 8) | (12, 16)
    )
}

struct MpiLayout {
    global: Vec<c_int>,
    local: Vec<c_int>,
    starts: Vec<c_int>,
    empty: bool,
}

fn build_layout<const N: usize>(
    global: &[usize; N],
    extra: &[usize],
    local_shape: &[usize; N],
    ranges: &[Range<usize>; N],
    local_elements: usize,
    width: usize,
) -> Result<MpiLayout, IoError> {
    let rank = extra
        .len()
        .checked_add(N)
        .and_then(|rank| rank.checked_add(1))
        .ok_or(IoError::SizeLimit {
            what: "MPI datatype rank",
        })?;
    if rank == 0 || rank > MAX_PROTOCOL_RANK {
        return Err(IoError::SizeLimit {
            what: "MPI datatype rank",
        });
    }
    let local_bytes = local_elements
        .checked_mul(width)
        .ok_or(IoError::SizeLimit {
            what: "MPI local byte count",
        })?;
    if local_bytes > c_int::MAX as usize {
        return Err(IoError::SizeLimit {
            what: "MPI local byte count",
        });
    }
    let reserve = rank
        .checked_mul(std::mem::size_of::<c_int>())
        .ok_or(IoError::SizeLimit {
            what: "MPI datatype arrays",
        })?;
    let mut global_dims = Vec::new();
    let mut local_dims = Vec::new();
    let mut starts = Vec::new();
    global_dims
        .try_reserve_exact(rank)
        .map_err(|_| IoError::AllocationFailed { requested: reserve })?;
    local_dims
        .try_reserve_exact(rank)
        .map_err(|_| IoError::AllocationFailed { requested: reserve })?;
    starts
        .try_reserve_exact(rank)
        .map_err(|_| IoError::AllocationFailed { requested: reserve })?;
    for &extent in extra {
        global_dims.push(c_int::try_from(extent).map_err(|_| IoError::SizeLimit {
            what: "MPI global dimension",
        })?);
        local_dims.push(c_int::try_from(extent).map_err(|_| IoError::SizeLimit {
            what: "MPI local dimension",
        })?);
        starts.push(0);
    }
    for axis in 0..N {
        global_dims.push(
            c_int::try_from(global[axis]).map_err(|_| IoError::SizeLimit {
                what: "MPI global dimension",
            })?,
        );
        local_dims.push(
            c_int::try_from(local_shape[axis]).map_err(|_| IoError::SizeLimit {
                what: "MPI local dimension",
            })?,
        );
        starts.push(
            c_int::try_from(ranges[axis].start).map_err(|_| IoError::SizeLimit {
                what: "MPI subarray start",
            })?,
        );
    }
    global_dims.push(c_int::try_from(width).map_err(|_| IoError::SizeLimit {
        what: "MPI element width",
    })?);
    local_dims.push(c_int::try_from(width).map_err(|_| IoError::SizeLimit {
        what: "MPI element width",
    })?);
    starts.push(0);
    let global_extra_empty = extra.contains(&0);
    Ok(MpiLayout {
        global: global_dims,
        local: local_dims,
        starts,
        empty: local_bytes == 0 || global_extra_empty,
    })
}

fn to_offset(value: u64) -> Result<ffi::MPI_Offset, IoError> {
    ffi::MPI_Offset::try_from(value).map_err(|_| IoError::SizeLimit {
        what: "MPI file offset",
    })
}

fn path_bytes(path: &Path) -> Result<&[u8], IoError> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        Ok(path.as_os_str().as_bytes())
    }
    #[cfg(not(unix))]
    {
        path.to_str().map(str::as_bytes).ok_or(IoError::InvalidPath)
    }
}

fn try_u64_values(values: &[usize], what: &'static str) -> Result<Vec<u64>, IoError> {
    let mut converted = Vec::new();
    converted
        .try_reserve_exact(values.len())
        .map_err(|_| IoError::AllocationFailed {
            requested: values.len() * std::mem::size_of::<u64>(),
        })?;
    for &value in values {
        converted.push(u64::try_from(value).map_err(|_| IoError::SizeLimit { what })?);
    }
    Ok(converted)
}

fn first_error2<T, U>(first: &Result<T, IoError>, second: &Result<U, IoError>) -> Option<IoError> {
    first
        .as_ref()
        .err()
        .cloned()
        .or_else(|| second.as_ref().err().cloned())
}

fn first_error3<T, U, V>(
    first: &Result<T, IoError>,
    second: &Result<U, IoError>,
    third: &Result<V, IoError>,
) -> Option<IoError> {
    first_error2(first, second).or_else(|| third.as_ref().err().cloned())
}

fn error_or_agreement(primary: IoError, agreement: IoError) -> IoError {
    match agreement {
        IoError::CollectivePrecondition { .. } => primary,
        other => other,
    }
}

fn primary_from_mpi(primary: IoError, operation: &'static str, code: i32) -> IoError {
    match primary {
        IoError::WriteIncomplete { stage } => IoError::WriteIncomplete { stage },
        IoError::CommitUncertain { stage } => IoError::CommitUncertain { stage },
        _ => IoError::Mpi { operation, code },
    }
}

trait MpiCodeExt {
    fn checked_success(self) -> Result<(), i32>;
}

impl MpiCodeExt for i32 {
    fn checked_success(self) -> Result<(), i32> {
        if self == ffi::MPI_SUCCESS as i32 {
            Ok(())
        } else {
            Err(self)
        }
    }
}
