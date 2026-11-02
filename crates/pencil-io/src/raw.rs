use crate::format::{IoElement, element_count, prepare_physical_values};
use crate::mpi_io::{
    DatatypeGuard, FileGuard, abort_unrecoverable, aggregate_cleanup_result, agree_mpi_preparation,
    agree_phase, build_layout, checked_local_byte_count, collective_state, descriptor_agreement,
    duplicate_comm, finish_comm, finish_resources, mpi_preparation_error, path_bytes,
    set_file_errors_return,
};
use crate::options::{InfoGuard, MpiIoMode, RawByteOrder, RawReadOptions, agree_options};
use crate::{IoError, OP_READ_RAW_MPI, ffi};
use mpi::traits::{AsRaw, Communicator};
use pencil_array::PencilArrayViewMut;
use std::{ffi::CString, path::Path};

/// Reads an explicitly requested raw `[extra..., spatial...]` row-major payload.
///
/// The view supplies its type and global shape. No header, commit marker, format,
/// type or Julia-wire detection is performed. Prefix and trailing records are
/// allowed. Every rank participates even with independent payload transfers. The
/// local payload is limited to 2147483647 bytes/rank (`local_elements *
/// scalar_width`; for `f64`, up to 268435455 elements); every rank returns the
/// same `IoError::SizeLimit` with `what == "MPI local byte count"` before staging
/// when any rank exceeds it. The destination changes only after native cleanup and
/// collective agreement.
pub fn read_mpi_raw<P, T, const N: usize, const M: usize>(
    path: P,
    view: PencilArrayViewMut<'_, T, N, M>,
    options: RawReadOptions,
) -> Result<(), IoError>
where
    P: AsRef<Path>,
    T: IoElement,
{
    read_raw_inner(path, view, options, false)
}

#[cfg(test)]
pub(crate) fn read_raw_with_post_cleanup_failure<
    P: AsRef<Path>,
    T: IoElement,
    const N: usize,
    const M: usize,
>(
    path: P,
    view: PencilArrayViewMut<'_, T, N, M>,
    options: RawReadOptions,
) -> Result<(), IoError> {
    read_raw_inner(path, view, options, true)
}

fn read_raw_inner<P: AsRef<Path>, T: IoElement, const N: usize, const M: usize>(
    path: P,
    mut view: PencilArrayViewMut<'_, T, N, M>,
    options: RawReadOptions,
    inject_cleanup_failure: bool,
) -> Result<(), IoError> {
    let comm = view.pencil().topology().communicator();
    let global = view.pencil().global_shape();
    let extra = view.extra_shape().dimensions();
    descriptor_agreement(
        comm,
        path.as_ref(),
        OP_READ_RAW_MPI,
        global,
        extra,
        view.pencil().topology().process_grid(),
        view.pencil().decomposition(),
        view.pencil().permutation().axes(),
        T::CODE,
        T::WIDTH,
    )?;
    crate::options::agree_decomposition(view.pencil())?;
    agree_options(
        comm,
        &options.mpi,
        &[
            options.offset,
            options.endian as u64,
            options.endian.effective() as u64,
        ],
    )?;
    let preparation = (|| {
        let local_bytes = checked_local_byte_count(view.len(), T::WIDTH)?;
        let layout = build_layout(
            global,
            extra,
            &view.local_spatial_shape(),
            view.pencil().local_ranges(),
            view.len(),
            T::WIDTH,
        )?;
        let path = CString::new(path_bytes(path.as_ref())?).map_err(|_| IoError::InvalidPath)?;
        let payload = (element_count(global)? as u64)
            .checked_mul(element_count(extra)? as u64)
            .and_then(|n| n.checked_mul(T::WIDTH as u64))
            .ok_or(IoError::SizeLimit {
                what: "raw global payload",
            })?;
        let end = options
            .offset
            .checked_add(payload)
            .ok_or(IoError::SizeLimit {
                what: "raw payload end",
            })?;
        let offset = ffi::MPI_Offset::try_from(options.offset)
            .map_err(|_| IoError::SizeLimit { what: "raw offset" })?;
        ffi::MPI_Offset::try_from(end).map_err(|_| IoError::SizeLimit {
            what: "raw payload end",
        })?;
        Ok::<_, IoError>((path, layout, offset, end, local_bytes))
    })();
    if let Err(agreement) =
        agree_mpi_preparation(comm, preparation.as_ref().err(), "raw preparation")
    {
        return Err(mpi_preparation_error(preparation.err(), agreement));
    }
    let (path, layout, offset, end, local_bytes) =
        preparation.expect("raw preparation agreement established");
    let info = InfoGuard::new(comm, &options.mpi)?;
    let duplicate = duplicate_comm(comm)?;
    let opened = ffi::file_open_with_info(duplicate.raw, &path, false, info.raw);
    let (all, mixed) = collective_state(comm, opened.is_ok());
    if mixed {
        abort_unrecoverable(comm.as_raw(), "raw file open");
    }
    if !all {
        return Err(finish_comm(comm, duplicate).unwrap_or(IoError::Mpi {
            operation: "MPI_File_open",
            code: opened.err().unwrap_or(-1),
        }));
    }
    let file = FileGuard {
        raw: opened.expect("open agreement"),
    };
    let mut datatype = None;
    let result = (|| {
        let handler = set_file_errors_return(&file);
        if let Err(e) = agree_phase(comm, handler.is_ok(), "raw file handler") {
            return Err(handler
                .err()
                .map(|code| IoError::Mpi {
                    operation: "MPI_File_set_errhandler",
                    code,
                })
                .unwrap_or(e));
        }
        let allocation = (|| {
            let size = ffi::file_get_size(file.raw).map_err(|code| IoError::Mpi {
                operation: "MPI_File_get_size",
                code,
            })?;
            if !u64::try_from(size).is_ok_and(|n| n >= end) {
                return Err(IoError::InvalidFile {
                    reason: "truncated raw payload",
                });
            }
            let mut staging = Vec::new();
            staging
                .try_reserve_exact(local_bytes)
                .map_err(|_| IoError::AllocationFailed {
                    requested: local_bytes,
                })?;
            Ok::<_, IoError>(staging)
        })();
        if let Err(e) = agree_phase(comm, allocation.is_ok(), "raw file size") {
            return Err(allocation.err().unwrap_or(e));
        }
        let mut staging = allocation.expect("raw file-size agreement established");
        staging.resize(local_bytes, 0);
        if !layout.empty {
            datatype = ffi::type_create_subarray(
                &layout.global,
                &layout.local,
                &layout.starts,
                comm.as_raw(),
            )
            .ok()
            .map(|raw| DatatypeGuard { raw });
        }
        let (all, mixed) = collective_state(comm, layout.empty || datatype.is_some());
        if mixed {
            abort_unrecoverable(comm.as_raw(), "raw datatype partial construction");
        }
        if !all {
            return Err(IoError::CollectivePrecondition {
                phase: "raw datatype",
            });
        }
        let code = ffi::file_set_view_with_info(
            file.raw,
            offset,
            datatype.as_ref().map_or_else(ffi::byte_datatype, |d| d.raw),
            info.raw,
        );
        agree_phase(comm, code == ffi::MPI_SUCCESS as i32, "raw file view")?;
        let read = match options.mpi.mode {
            MpiIoMode::Collective => ffi::file_read_all(file.raw, &mut staging),
            MpiIoMode::Independent => ffi::file_read_independent(file.raw, &mut staging),
        };
        agree_phase(
            comm,
            matches!(read,Ok(n) if n==staging.len()),
            "raw payload read",
        )?;
        // Existing v1 decoding is little endian, not native endian. Reverse each
        // real component independently; never reverse a whole complex number.
        if options.endian.effective() == RawByteOrder::Big {
            let width = if matches!(T::CODE, 11 | 12) {
                T::WIDTH / 2
            } else {
                T::WIDTH
            };
            for component in staging.chunks_exact_mut(width) {
                component.reverse();
            }
        }
        let values = prepare_physical_values(&view, &staging);
        if let Err(e) = agree_phase(comm, values.is_ok(), "raw physical values") {
            return Err(values.err().unwrap_or(e));
        }
        values
    })();
    let cleanup = finish_resources(comm, duplicate, file, datatype);
    drop(info);
    let injected = if inject_cleanup_failure && comm.rank() == 0 {
        Some(IoError::Native {
            operation: "raw post-cleanup result",
            code: -1,
        })
    } else {
        None
    };
    let error = aggregate_cleanup_result(
        comm,
        cleanup
            .or(injected)
            .or_else(|| result.as_ref().err().cloned()),
    );
    if let Some(e) = error {
        return Err(e);
    }
    view.as_mut_slice()
        .copy_from_slice(&result.expect("read and cleanup agreement"));
    Ok(())
}
