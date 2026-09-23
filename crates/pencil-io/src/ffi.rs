//! The only raw native boundary in `pencil-io`.
//!
//! Every function below validates the Rust slice/handle preconditions at its
//! caller and keeps the corresponding FFI safety argument next to the unsafe
//! call.  No raw handle is exposed by the public crate API.

use std::ffi::CStr;
use std::mem::MaybeUninit;
use std::os::raw::{c_int, c_void};
use std::ptr;

use mpi::ffi;

#[cfg(test)]
thread_local! { pub(crate) static DUPLICATE_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }

#[cfg(test)]
thread_local! { pub(crate) static AT_CALLS: std::cell::Cell<[usize; 5]> = const { std::cell::Cell::new([0; 5]) }; }
#[cfg(test)]
fn record_at(write: bool, collective: bool, len: usize) {
    AT_CALLS.with(|c| {
        let mut n = c.get();
        n[usize::from(!write) * 2 + usize::from(!collective)] += 1;
        if !write {
            n[4] += len;
        }
        c.set(n);
    });
}

#[cfg(test)]
thread_local! { static PAYLOAD_CALLS: std::cell::Cell<[usize;6]> = const { std::cell::Cell::new([0;6]) }; }
#[cfg(test)]
fn record_payload(index: usize) {
    PAYLOAD_CALLS.with(|c| {
        let mut n = c.get();
        n[index] += 1;
        c.set(n);
    });
}
#[cfg(test)]
pub(crate) fn test_payload_calls() -> [usize; 6] {
    PAYLOAD_CALLS.with(std::cell::Cell::get)
}

#[cfg(all(test, feature = "parallel-hdf5"))]
thread_local! {
    pub(crate) static FILTER_CAPABILITIES: std::cell::Cell<Option<u32>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
thread_local! { static INFO_ARGUMENTS: std::cell::Cell<[i64;3]> = const { std::cell::Cell::new([-1;3]) }; }
#[cfg(test)]
pub(crate) fn test_info_arguments() -> [i64; 3] {
    INFO_ARGUMENTS.with(std::cell::Cell::get)
}
#[cfg(test)]
fn inspect_info_argument(info: ffi::MPI_Info, index: usize) {
    let mut value = -1;
    // SAFETY: this is the live info argument passed immediately to the native
    // call. MPI_Info_get writes at most the fixed buffer length including NUL.
    unsafe {
        if info != ffi::RSMPI_INFO_NULL {
            let mut bytes = [0 as std::os::raw::c_char; 64];
            let mut flag = 0;
            assert_eq!(
                ffi::MPI_Info_get(
                    info,
                    c"cb_buffer_size".as_ptr(),
                    63,
                    bytes.as_mut_ptr(),
                    &mut flag
                ),
                ffi::MPI_SUCCESS as i32
            );
            if flag != 0 {
                value = CStr::from_ptr(bytes.as_ptr())
                    .to_str()
                    .unwrap()
                    .parse()
                    .unwrap();
            }
        }
    }
    INFO_ARGUMENTS.with(|c| {
        let mut args = c.get();
        args[index] = value;
        c.set(args);
    });
}

pub(crate) use mpi::ffi::{
    MPI_Comm, MPI_Datatype, MPI_Errhandler, MPI_File, MPI_Info, MPI_Offset, MPI_SUCCESS,
};

pub(crate) fn info_from_hints<I, K, V>(hints: I, abort_comm: MPI_Comm) -> Result<MPI_Info, i32>
where
    I: IntoIterator<Item = (K, V)>,
    K: AsRef<str>,
    V: AsRef<str>,
{
    // SAFETY: MPI is initialized. Fixed, NUL-terminated buffers remain alive for
    // each Info_set; all partial native handles are released or fail-stop.
    unsafe {
        let mut info = ffi::RSMPI_INFO_NULL;
        let code = ffi::MPI_Info_create(&mut info);
        if code != ffi::MPI_SUCCESS as c_int {
            if info != ffi::RSMPI_INFO_NULL
                && ffi::MPI_Info_free(&mut info) != ffi::MPI_SUCCESS as c_int
            {
                abort_fail_stop(abort_comm);
            }
            return Err(code);
        }
        for (key, value) in hints {
            let (key, value) = (key.as_ref().as_bytes(), value.as_ref().as_bytes());
            let mut k = [0u8; ffi::MPI_MAX_INFO_KEY as usize];
            let mut v = [0u8; ffi::MPI_MAX_INFO_VAL as usize];
            let code = if key.is_empty()
                || value.is_empty()
                || key.len() >= k.len()
                || value.len() >= v.len()
                || key.contains(&0)
                || value.contains(&0)
            {
                ffi::MPI_ERR_ARG as c_int
            } else {
                k[..key.len()].copy_from_slice(key);
                v[..value.len()].copy_from_slice(value);
                ffi::MPI_Info_set(info, k.as_ptr().cast(), v.as_ptr().cast())
            };
            if code != ffi::MPI_SUCCESS as c_int {
                if ffi::MPI_Info_free(&mut info) != ffi::MPI_SUCCESS as c_int {
                    abort_fail_stop(abort_comm);
                }
                return Err(code);
            }
        }
        Ok(info)
    }
}

pub(crate) fn info_free(info: &mut MPI_Info) -> i32 {
    // SAFETY: info is the uniquely owned live handle created by info_from_hints.
    unsafe { ffi::MPI_Info_free(info) }
}

pub(crate) fn info_null() -> MPI_Info {
    // SAFETY: MPI is initialized at all callers.
    unsafe { ffi::RSMPI_INFO_NULL }
}

pub(crate) fn comm_dup(comm: ffi::MPI_Comm) -> Result<ffi::MPI_Comm, i32> {
    #[cfg(test)]
    DUPLICATE_CALLS.with(|calls| calls.set(calls.get() + 1));
    // SAFETY: `comm` is the live intracommunicator borrowed from a validated
    // pencil topology and all ranks call this collective in the same phase.
    unsafe {
        let mut duplicate = ffi::RSMPI_COMM_NULL;
        let code = ffi::MPI_Comm_dup(comm, &mut duplicate);
        if code == ffi::MPI_SUCCESS as c_int && duplicate != ffi::RSMPI_COMM_NULL {
            Ok(duplicate)
        } else {
            Err(code)
        }
    }
}

pub(crate) fn comm_get_errhandler(comm: ffi::MPI_Comm) -> Result<ffi::MPI_Errhandler, i32> {
    // SAFETY: `comm` is a live communicator and `handler` points to writable
    // storage for the handler returned by MPI.
    unsafe {
        let mut handler = ffi::MPI_Errhandler(std::ptr::null_mut());
        let code = ffi::MPI_Comm_get_errhandler(comm, &mut handler);
        if code == ffi::MPI_SUCCESS as c_int {
            Ok(handler)
        } else {
            Err(code)
        }
    }
}

pub(crate) fn comm_set_errhandler(comm: ffi::MPI_Comm, handler: ffi::MPI_Errhandler) -> i32 {
    // SAFETY: `comm` is live and `handler` is either the saved live handler or
    // the predefined MPI_ERRORS_RETURN handler.
    unsafe { ffi::MPI_Comm_set_errhandler(comm, handler) }
}

pub(crate) fn comm_set_errors_return(comm: ffi::MPI_Comm) -> i32 {
    comm_set_errhandler(
        comm,
        // SAFETY: MPI_ERRORS_RETURN is a predefined handler valid after MPI
        // initialization.
        unsafe { ffi::RSMPI_ERRORS_RETURN },
    )
}

pub(crate) fn errhandler_free(handler: &mut ffi::MPI_Errhandler) -> i32 {
    // SAFETY: `handler` is the reference returned by MPI_Comm_get_errhandler;
    // it is released after the communicator has been restored.
    unsafe { ffi::MPI_Errhandler_free(handler) }
}

#[cfg(test)]
pub(crate) fn test_comm_errhandler_token(comm: ffi::MPI_Comm) -> Result<i32, i32> {
    let handler = comm_get_errhandler(comm)?;
    // SAFETY: `handler` is the live value returned by MPI_Comm_get_errhandler.
    let token = unsafe { ffi::RSMPI_Errhandler_c2f(handler) };
    let mut handler = handler;
    let code = errhandler_free(&mut handler);
    if code == ffi::MPI_SUCCESS as c_int {
        Ok(token)
    } else {
        Err(code)
    }
}

pub(crate) fn comm_abort(comm: ffi::MPI_Comm, code: i32) -> i32 {
    // SAFETY: `comm` is live.  This is used only for an unrecoverable partial
    // native-resource state where returning would leak a collective handle.
    unsafe { ffi::MPI_Abort(comm, code) }
}

pub(crate) fn abort_fail_stop(comm: ffi::MPI_Comm) -> ! {
    // MPI_Abort is not allowed to fall through into a recoverable native
    // error. A broken MPI implementation that returns still gets process
    // termination.
    let _ = comm_abort(comm, 1);
    std::process::abort();
}

pub(crate) fn comm_free(comm: &mut ffi::MPI_Comm) -> i32 {
    // SAFETY: `comm` points to the live duplicate owned by this wrapper.  The
    // caller invokes this collective on every rank before MPI finalization.
    unsafe { ffi::MPI_Comm_free(comm) }
}

pub(crate) fn file_open(
    comm: ffi::MPI_Comm,
    path: &CStr,
    write: bool,
) -> Result<ffi::MPI_File, i32> {
    file_open_mode(comm, path, if write { 1 } else { 0 }, unsafe {
        ffi::RSMPI_INFO_NULL
    })
}

pub(crate) fn file_open_with_info(
    comm: ffi::MPI_Comm,
    path: &CStr,
    write: bool,
    info: MPI_Info,
) -> Result<ffi::MPI_File, i32> {
    file_open_mode(comm, path, if write { 1 } else { 0 }, info)
}

/// Open an existing file for read/write without creating or truncating it.
pub(crate) fn file_open_update(comm: ffi::MPI_Comm, path: &CStr) -> Result<ffi::MPI_File, i32> {
    file_open_mode(comm, path, 2, unsafe { ffi::RSMPI_INFO_NULL })
}

pub(crate) fn file_open_update_with_info(
    comm: ffi::MPI_Comm,
    path: &CStr,
    info: MPI_Info,
) -> Result<ffi::MPI_File, i32> {
    file_open_mode(comm, path, 2, info)
}

fn file_open_mode(
    comm: ffi::MPI_Comm,
    path: &CStr,
    mode: i32,
    info: MPI_Info,
) -> Result<ffi::MPI_File, i32> {
    // SAFETY: `path` is NUL-terminated for the duration of MPI_File_open,
    // `comm` is live, and all ranks enter the same collective call.
    unsafe {
        let mut file = ffi::RSMPI_FILE_NULL;
        let mode = match mode {
            1 => (ffi::MPI_MODE_WRONLY | ffi::MPI_MODE_CREATE | ffi::MPI_MODE_EXCL) as c_int,
            2 => (ffi::MPI_MODE_RDWR) as c_int,
            _ => ffi::MPI_MODE_RDONLY as c_int,
        };
        #[cfg(test)]
        inspect_info_argument(info, 0);
        let code = ffi::MPI_File_open(comm, path.as_ptr(), mode, info, &mut file);
        if code == ffi::MPI_SUCCESS as c_int && file != ffi::RSMPI_FILE_NULL {
            Ok(file)
        } else {
            Err(code)
        }
    }
}

pub(crate) fn file_set_errors_return(file: ffi::MPI_File) -> i32 {
    // SAFETY: `file` is the live handle returned by MPI_File_open and the
    // predefined MPI_ERRORS_RETURN handler is valid for MPI files.
    unsafe { ffi::MPI_File_set_errhandler(file, ffi::RSMPI_ERRORS_RETURN) }
}

pub(crate) fn file_close(file: &mut ffi::MPI_File) -> i32 {
    // SAFETY: `file` points to a live MPI file handle and all ranks call the
    // collective close while their duplicate communicator remains alive.
    unsafe { ffi::MPI_File_close(file) }
}

pub(crate) fn file_get_size(file: ffi::MPI_File) -> Result<ffi::MPI_Offset, i32> {
    // SAFETY: `file` is live and `size` points to writable storage for MPI_Offset.
    unsafe {
        let mut size = 0;
        let code = ffi::MPI_File_get_size(file, &mut size);
        if code == ffi::MPI_SUCCESS as c_int {
            Ok(size)
        } else {
            Err(code)
        }
    }
}

pub(crate) fn file_sync(file: ffi::MPI_File) -> i32 {
    // SAFETY: `file` is live and all ranks call this collective synchronization.
    unsafe { ffi::MPI_File_sync(file) }
}

pub(crate) fn file_set_view_with_info(
    file: ffi::MPI_File,
    displacement: ffi::MPI_Offset,
    filetype: ffi::MPI_Datatype,
    info: MPI_Info,
) -> i32 {
    // SAFETY: `file` and `filetype` are live; the static data representation and
    // null info handle are valid MPI constants.  The filetype was committed
    // by the caller or is the predefined byte type.
    unsafe {
        static DATAREP: &[u8] = b"native\0";
        #[cfg(test)]
        inspect_info_argument(info, 1);
        ffi::MPI_File_set_view(
            file,
            displacement,
            ffi::RSMPI_UINT8_T,
            filetype,
            DATAREP.as_ptr().cast(),
            info,
        )
    }
}

pub(crate) fn file_write_at_all(
    file: ffi::MPI_File,
    offset: ffi::MPI_Offset,
    bytes: &[u8],
) -> Result<usize, i32> {
    file_write_at(file, offset, bytes, true)
}

pub(crate) fn file_write_at(
    file: ffi::MPI_File,
    offset: ffi::MPI_Offset,
    bytes: &[u8],
    collective: bool,
) -> Result<usize, i32> {
    let count = c_int::try_from(bytes.len()).map_err(|_| ffi::MPI_ERR_COUNT as i32)?;
    #[cfg(test)]
    record_at(true, collective, bytes.len());
    // SAFETY: `bytes` remains borrowed for the duration of the MPI
    // call, count is its checked length, and the byte datatype is predefined.
    unsafe {
        let mut status = MaybeUninit::<ffi::MPI_Status>::zeroed().assume_init();
        let write = if collective {
            ffi::MPI_File_write_at_all
        } else {
            ffi::MPI_File_write_at
        };
        let code = write(
            file,
            offset,
            if bytes.is_empty() {
                ptr::null()
            } else {
                bytes.as_ptr().cast::<c_void>()
            },
            count,
            ffi::RSMPI_UINT8_T,
            &mut status,
        );
        if code != ffi::MPI_SUCCESS as c_int {
            return Err(code);
        }
        let mut actual = 0;
        let count_code = ffi::MPI_Get_count(&status, ffi::RSMPI_UINT8_T, &mut actual);
        if count_code != ffi::MPI_SUCCESS as c_int || actual < 0 {
            return Err(if count_code == ffi::MPI_SUCCESS as c_int {
                ffi::MPI_ERR_OTHER as i32
            } else {
                count_code
            });
        }
        Ok(actual as usize)
    }
}

pub(crate) fn file_read_at_all(
    file: ffi::MPI_File,
    offset: ffi::MPI_Offset,
    bytes: &mut [u8],
) -> Result<usize, i32> {
    file_read_at(file, offset, bytes, true)
}

pub(crate) fn file_read_at(
    file: ffi::MPI_File,
    offset: ffi::MPI_Offset,
    bytes: &mut [u8],
    collective: bool,
) -> Result<usize, i32> {
    let count = c_int::try_from(bytes.len()).map_err(|_| ffi::MPI_ERR_COUNT as i32)?;
    #[cfg(test)]
    record_at(false, collective, bytes.len());
    // SAFETY: `bytes` is writable and remains borrowed for the MPI
    // call, count is its checked length, and the byte datatype is predefined.
    unsafe {
        let mut status = MaybeUninit::<ffi::MPI_Status>::zeroed().assume_init();
        let read = if collective {
            ffi::MPI_File_read_at_all
        } else {
            ffi::MPI_File_read_at
        };
        let code = read(
            file,
            offset,
            if bytes.is_empty() {
                ptr::null_mut()
            } else {
                bytes.as_mut_ptr().cast::<c_void>()
            },
            count,
            ffi::RSMPI_UINT8_T,
            &mut status,
        );
        if code != ffi::MPI_SUCCESS as c_int {
            return Err(code);
        }
        let mut actual = 0;
        let count_code = ffi::MPI_Get_count(&status, ffi::RSMPI_UINT8_T, &mut actual);
        if count_code != ffi::MPI_SUCCESS as c_int || actual < 0 {
            return Err(if count_code == ffi::MPI_SUCCESS as c_int {
                ffi::MPI_ERR_OTHER as i32
            } else {
                count_code
            });
        }
        Ok(actual as usize)
    }
}

pub(crate) fn file_write_all(file: ffi::MPI_File, bytes: &[u8]) -> Result<usize, i32> {
    #[cfg(test)]
    record_payload(0);
    let count = c_int::try_from(bytes.len()).map_err(|_| ffi::MPI_ERR_COUNT as i32)?;
    // SAFETY: `bytes` remains borrowed for the collective call and the current
    // file view plus predefined byte datatype describe exactly its elements.
    unsafe {
        let mut status = MaybeUninit::<ffi::MPI_Status>::zeroed().assume_init();
        let code = ffi::MPI_File_write_all(
            file,
            if bytes.is_empty() {
                ptr::null()
            } else {
                bytes.as_ptr().cast::<c_void>()
            },
            count,
            ffi::RSMPI_UINT8_T,
            &mut status,
        );
        if code != ffi::MPI_SUCCESS as c_int {
            return Err(code);
        }
        let mut actual = 0;
        let count_code = ffi::MPI_Get_count(&status, ffi::RSMPI_UINT8_T, &mut actual);
        if count_code != ffi::MPI_SUCCESS as c_int || actual < 0 {
            return Err(if count_code == ffi::MPI_SUCCESS as c_int {
                ffi::MPI_ERR_OTHER as i32
            } else {
                count_code
            });
        }
        Ok(actual as usize)
    }
}

pub(crate) fn file_write_independent(file: ffi::MPI_File, bytes: &[u8]) -> Result<usize, i32> {
    #[cfg(test)]
    record_payload(1);
    let count = c_int::try_from(bytes.len()).map_err(|_| ffi::MPI_ERR_COUNT as i32)?;
    // SAFETY: live file and initialized readable byte slice remain valid through the call.
    unsafe {
        let mut status = MaybeUninit::<ffi::MPI_Status>::zeroed().assume_init();
        let code = ffi::MPI_File_write(
            file,
            bytes.as_ptr().cast(),
            count,
            ffi::RSMPI_UINT8_T,
            &mut status,
        );
        if code != ffi::MPI_SUCCESS as c_int {
            return Err(code);
        }
        let mut actual = 0;
        let code = ffi::MPI_Get_count(&status, ffi::RSMPI_UINT8_T, &mut actual);
        if code != ffi::MPI_SUCCESS as c_int || actual < 0 {
            return Err(if code == ffi::MPI_SUCCESS as c_int {
                ffi::MPI_ERR_OTHER as i32
            } else {
                code
            });
        }
        Ok(actual as usize)
    }
}

pub(crate) fn file_read_independent(file: ffi::MPI_File, bytes: &mut [u8]) -> Result<usize, i32> {
    #[cfg(test)]
    record_payload(3);
    let count = c_int::try_from(bytes.len()).map_err(|_| ffi::MPI_ERR_COUNT as i32)?;
    unsafe {
        let mut status = MaybeUninit::<ffi::MPI_Status>::zeroed().assume_init();
        let code = ffi::MPI_File_read(
            file,
            if bytes.is_empty() {
                ptr::null_mut()
            } else {
                bytes.as_mut_ptr().cast()
            },
            count,
            ffi::RSMPI_UINT8_T,
            &mut status,
        );
        if code != ffi::MPI_SUCCESS as c_int {
            return Err(code);
        }
        let mut actual = 0;
        let code = ffi::MPI_Get_count(&status, ffi::RSMPI_UINT8_T, &mut actual);
        if code != ffi::MPI_SUCCESS as c_int || actual < 0 {
            return Err(if code == ffi::MPI_SUCCESS as c_int {
                ffi::MPI_ERR_OTHER as i32
            } else {
                code
            });
        }
        Ok(actual as usize)
    }
}

pub(crate) fn file_read_all(file: ffi::MPI_File, bytes: &mut [u8]) -> Result<usize, i32> {
    #[cfg(test)]
    record_payload(2);
    let count = c_int::try_from(bytes.len()).map_err(|_| ffi::MPI_ERR_COUNT as i32)?;
    // SAFETY: `bytes` is writable for the collective call and the current file
    // view plus predefined byte datatype describe exactly its elements.
    unsafe {
        let mut status = MaybeUninit::<ffi::MPI_Status>::zeroed().assume_init();
        let code = ffi::MPI_File_read_all(
            file,
            if bytes.is_empty() {
                ptr::null_mut()
            } else {
                bytes.as_mut_ptr().cast::<c_void>()
            },
            count,
            ffi::RSMPI_UINT8_T,
            &mut status,
        );
        if code != ffi::MPI_SUCCESS as c_int {
            return Err(code);
        }
        let mut actual = 0;
        let count_code = ffi::MPI_Get_count(&status, ffi::RSMPI_UINT8_T, &mut actual);
        if count_code != ffi::MPI_SUCCESS as c_int || actual < 0 {
            return Err(if count_code == ffi::MPI_SUCCESS as c_int {
                ffi::MPI_ERR_OTHER as i32
            } else {
                count_code
            });
        }
        Ok(actual as usize)
    }
}

pub(crate) fn byte_datatype() -> ffi::MPI_Datatype {
    // SAFETY: MPI_UINT8_T is a predefined datatype that remains valid until
    // MPI_Finalize, and this function is only used after MPI initialization.
    unsafe { ffi::RSMPI_UINT8_T }
}

pub(crate) fn type_create_subarray(
    global: &[c_int],
    local: &[c_int],
    starts: &[c_int],
    abort_comm: ffi::MPI_Comm,
) -> Result<ffi::MPI_Datatype, i32> {
    debug_assert_eq!(global.len(), local.len());
    debug_assert_eq!(global.len(), starts.len());
    // SAFETY: all slices have the same positive rank and remain borrowed for
    // the call; every entry was checked to fit c_int and the byte datatype is
    // a live predefined MPI datatype.
    unsafe {
        let mut datatype = ffi::RSMPI_DATATYPE_NULL;
        let rank = c_int::try_from(global.len()).map_err(|_| ffi::MPI_ERR_DIMS as i32)?;
        let code = ffi::MPI_Type_create_subarray(
            rank,
            global.as_ptr(),
            local.as_ptr(),
            starts.as_ptr(),
            ffi::MPI_ORDER_C as c_int,
            ffi::RSMPI_UINT8_T,
            &mut datatype,
        );
        if code != ffi::MPI_SUCCESS as c_int || datatype == ffi::RSMPI_DATATYPE_NULL {
            let error = code;
            if datatype != ffi::RSMPI_DATATYPE_NULL {
                let free_code = ffi::MPI_Type_free(&mut datatype);
                if free_code != ffi::MPI_SUCCESS as c_int {
                    abort_fail_stop(abort_comm);
                }
            }
            return Err(error);
        }
        let commit_code = ffi::MPI_Type_commit(&mut datatype);
        if commit_code != ffi::MPI_SUCCESS as c_int {
            let free_code = ffi::MPI_Type_free(&mut datatype);
            if free_code != ffi::MPI_SUCCESS as c_int {
                abort_fail_stop(abort_comm);
            }
            return Err(commit_code);
        }
        Ok(datatype)
    }
}

pub(crate) fn type_free(datatype: &mut ffi::MPI_Datatype) -> i32 {
    // SAFETY: `datatype` points to a live committed derived datatype owned by
    // this rank and no in-flight operation refers to it.
    unsafe { ffi::MPI_Type_free(datatype) }
}

pub(crate) fn bcast_bytes(comm: ffi::MPI_Comm, root: i32, bytes: &mut [u8]) -> i32 {
    let count = match c_int::try_from(bytes.len()) {
        Ok(value) => value,
        Err(_) => return ffi::MPI_ERR_COUNT as i32,
    };
    // SAFETY: `bytes` is writable for the duration of MPI_Bcast and the byte
    // datatype is predefined; all ranks pass the same count and root.
    unsafe {
        ffi::MPI_Bcast(
            if bytes.is_empty() {
                ptr::null_mut()
            } else {
                bytes.as_mut_ptr().cast::<c_void>()
            },
            count,
            ffi::RSMPI_UINT8_T,
            root,
            comm,
        )
    }
}

#[cfg(feature = "parallel-hdf5")]
pub(crate) mod hdf5 {
    use super::*;
    use hdf5_metno_sys::{self as h5, h5::hsize_t, h5i::hid_t};

    pub(crate) type Hid = hid_t;

    fn hdf5_invalid(id: Hid) -> bool {
        id < 0
    }

    pub(crate) fn fapl_create() -> Result<Hid, i32> {
        hdf5_metno::sync::sync(|| {
            // SAFETY: the global file-access property-list class is live.
            unsafe {
                let fapl = h5::h5p::H5Pcreate(*hdf5_metno::globals::H5P_FILE_ACCESS);
                if hdf5_invalid(fapl) {
                    Err(fapl as i32)
                } else {
                    Ok(fapl)
                }
            }
        })
    }

    pub(crate) fn fapl_set_mpio(fapl: Hid, comm: ffi::MPI_Comm, info: ffi::MPI_Info) -> i32 {
        hdf5_metno::sync::sync(|| {
            // SAFETY: `fapl` is live, `comm` is the owned duplicate whose
            // MPI_ERRORS_RETURN handler remains installed through file open,
            // and the null info handle is a valid MPI constant.
            #[cfg(test)]
            super::inspect_info_argument(info, 2);
            unsafe { h5::h5p::H5Pset_fapl_mpio(fapl, comm, info) }
        })
    }

    pub(crate) fn file_open(fapl: Hid, path: &CStr, write: bool) -> Result<Hid, i32> {
        hdf5_metno::sync::sync(|| {
            // SAFETY: HDF5 is serialized by its documented crate-global lock;
            // `fapl` and `path` remain live for this collective call. The
            // caller closes the FAPL only after every rank returns.
            unsafe {
                let flags = if write {
                    h5::h5f::H5F_ACC_EXCL
                } else {
                    h5::h5f::H5F_ACC_RDONLY
                };
                let file = if write {
                    h5::h5f::H5Fcreate(path.as_ptr(), flags, h5::h5p::H5P_DEFAULT, fapl)
                } else {
                    h5::h5f::H5Fopen(path.as_ptr(), flags, fapl)
                };
                if hdf5_invalid(file) {
                    Err(file as i32)
                } else {
                    Ok(file)
                }
            }
        })
    }

    pub(crate) fn file_open_update(fapl: Hid, path: &CStr) -> Result<Hid, i32> {
        hdf5_metno::sync::sync(|| {
            // SAFETY: HDF5 is serialized; fapl and path are live for this call.
            unsafe {
                let file = h5::h5f::H5Fopen(path.as_ptr(), h5::h5f::H5F_ACC_RDWR, fapl);
                if hdf5_invalid(file) {
                    Err(file as i32)
                } else {
                    Ok(file)
                }
            }
        })
    }

    pub(crate) fn file_flush(file: Hid) -> i32 {
        hdf5_metno::sync::sync(|| {
            // SAFETY: `file` is a live HDF5 file identifier and the scope is a
            // valid HDF5 flush scope.
            unsafe { h5::h5f::H5Fflush(file, h5::h5f::H5F_SCOPE_GLOBAL) }
        })
    }

    pub(crate) fn file_close(file: &mut Hid) -> i32 {
        hdf5_metno::sync::sync(|| {
            // SAFETY: `file` is a live identifier owned by this rank and all
            // ranks close their corresponding parallel file collectively.
            unsafe { h5::h5f::H5Fclose(*file) }
        })
    }

    pub(crate) fn group_create(file: Hid, name: &CStr) -> Result<Hid, i32> {
        hdf5_metno::sync::sync(|| {
            // SAFETY: file/name are live and valid for the duration of this
            // collective HDF5 metadata operation.
            unsafe {
                let id = h5::h5g::H5Gcreate2(
                    file,
                    name.as_ptr(),
                    h5::h5p::H5P_DEFAULT,
                    h5::h5p::H5P_DEFAULT,
                    h5::h5p::H5P_DEFAULT,
                );
                if hdf5_invalid(id) {
                    Err(id as i32)
                } else {
                    Ok(id)
                }
            }
        })
    }

    pub(crate) fn group_open(file: Hid, name: &CStr) -> Result<Hid, i32> {
        hdf5_metno::sync::sync(|| {
            // SAFETY: file/name are live and valid for the duration of this
            // collective HDF5 metadata operation.
            unsafe {
                let id = h5::h5g::H5Gopen2(file, name.as_ptr(), h5::h5p::H5P_DEFAULT);
                if hdf5_invalid(id) {
                    Err(id as i32)
                } else {
                    Ok(id)
                }
            }
        })
    }

    pub(crate) fn group_close(group: &mut Hid) -> i32 {
        hdf5_metno::sync::sync(|| {
            // SAFETY: `group` points to a live group identifier owned here.
            unsafe { h5::h5g::H5Gclose(*group) }
        })
    }

    pub(crate) fn dcpl_create() -> Result<Hid, i32> {
        hdf5_metno::sync::sync(|| unsafe {
            let id = h5::h5p::H5Pcreate(*hdf5_metno::globals::H5P_DATASET_CREATE);
            if hdf5_invalid(id) {
                Err(id as i32)
            } else {
                Ok(id)
            }
        })
    }

    pub(crate) fn dcpl_set_chunk(dcpl: Hid, dims: &[hsize_t]) -> i32 {
        let rank = match c_int::try_from(dims.len()) {
            Ok(v) => v,
            Err(_) => return -1,
        };
        hdf5_metno::sync::sync(|| unsafe { h5::h5p::H5Pset_chunk(dcpl, rank, dims.as_ptr()) })
    }

    pub(crate) fn dataset_create(
        group: Hid,
        name: &CStr,
        datatype: Hid,
        space: Hid,
        dcpl: Hid,
    ) -> Result<Hid, i32> {
        hdf5_metno::sync::sync(|| {
            // SAFETY: all identifiers and the NUL-terminated name are live;
            // the dataspace/type remain alive through the create call.
            unsafe {
                let id = h5::h5d::H5Dcreate2(
                    group,
                    name.as_ptr(),
                    datatype,
                    space,
                    h5::h5p::H5P_DEFAULT,
                    dcpl,
                    h5::h5p::H5P_DEFAULT,
                );
                if hdf5_invalid(id) {
                    Err(id as i32)
                } else {
                    Ok(id)
                }
            }
        })
    }

    pub(crate) fn link_exists(group: Hid, name: &CStr) -> Result<bool, i32> {
        hdf5_metno::sync::sync(|| {
            // SAFETY: group and name are live for this metadata query.
            unsafe {
                let result = h5::h5l::H5Lexists(group, name.as_ptr(), h5::h5p::H5P_DEFAULT);
                if result < 0 {
                    Err(result as i32)
                } else {
                    Ok(result > 0)
                }
            }
        })
    }

    /// Checks a link without resolving it until its class is known to be hard.
    /// The sys binding's H5Lget_info1 name aliases the HDF5 1.10 H5Lget_info symbol.
    pub(crate) fn link_is_hard_and_type(
        loc: Hid,
        name: &CStr,
        expected: LinkObjectType,
    ) -> Result<bool, i32> {
        hdf5_metno::sync::sync(|| {
            // SAFETY: loc/name are live; both output structs have the native
            // HDF5 1.10 ABI and remain local for the duration of each call.
            unsafe {
                let mut link = h5::h5l::H5L_info1_t::default();
                let code =
                    // hdf5-metno-sys aliases H5Lget_info1 to H5Lget_info for HDF5 1.10.
                    h5::h5l::H5Lget_info1(loc, name.as_ptr(), &mut link, h5::h5p::H5P_DEFAULT);
                if code < 0 {
                    return Err(code as i32);
                }
                if link.type_ != h5::h5l::H5L_TYPE_HARD {
                    return Ok(false);
                }
                let mut object = h5::h5o::H5O_info1_t::default();
                let code = h5::h5o::H5Oget_info_by_name2(
                    loc,
                    name.as_ptr(),
                    &mut object,
                    h5::h5o::H5O_INFO_BASIC,
                    h5::h5p::H5P_DEFAULT,
                );
                if code < 0 {
                    return Err(code as i32);
                }
                Ok(object.type_ == expected.hdf5())
            }
        })
    }

    #[derive(Clone, Copy)]
    pub(crate) enum LinkObjectType {
        Group,
        Dataset,
    }

    impl LinkObjectType {
        fn hdf5(self) -> h5::h5o::H5O_type_t {
            match self {
                Self::Group => h5::h5o::H5O_TYPE_GROUP,
                Self::Dataset => h5::h5o::H5O_TYPE_DATASET,
            }
        }
    }

    pub(crate) fn dataset_open(group: Hid, name: &CStr) -> Result<Hid, i32> {
        hdf5_metno::sync::sync(|| {
            // SAFETY: group/name are live for the duration of the collective
            // metadata operation.
            unsafe {
                let id = h5::h5d::H5Dopen2(group, name.as_ptr(), h5::h5p::H5P_DEFAULT);
                if hdf5_invalid(id) {
                    Err(id as i32)
                } else {
                    Ok(id)
                }
            }
        })
    }

    pub(crate) fn dataset_close(dataset: &mut Hid) -> i32 {
        hdf5_metno::sync::sync(|| {
            // SAFETY: `dataset` points to a live dataset identifier owned here.
            unsafe { h5::h5d::H5Dclose(*dataset) }
        })
    }

    pub(crate) fn dataset_space(dataset: Hid) -> Result<Hid, i32> {
        hdf5_metno::sync::sync(|| {
            // SAFETY: dataset is live and H5Dget_space returns a new identifier.
            unsafe {
                let id = h5::h5d::H5Dget_space(dataset);
                if hdf5_invalid(id) {
                    Err(id as i32)
                } else {
                    Ok(id)
                }
            }
        })
    }

    pub(crate) fn dataset_type(dataset: Hid) -> Result<Hid, i32> {
        hdf5_metno::sync::sync(|| {
            // SAFETY: dataset is live and H5Dget_type returns a new identifier.
            unsafe {
                let id = h5::h5d::H5Dget_type(dataset);
                if hdf5_invalid(id) {
                    Err(id as i32)
                } else {
                    Ok(id)
                }
            }
        })
    }

    pub(crate) fn dataspace_simple(dims: &[hsize_t]) -> Result<Hid, i32> {
        let rank = c_int::try_from(dims.len()).map_err(|_| -1)?;
        hdf5_metno::sync::sync(|| {
            // SAFETY: dims has `rank` entries and remains borrowed for the
            // call; a null maxdims pointer requests fixed extents.
            unsafe {
                let id = h5::h5s::H5Screate_simple(rank, dims.as_ptr(), ptr::null());
                if hdf5_invalid(id) {
                    Err(id as i32)
                } else {
                    Ok(id)
                }
            }
        })
    }

    pub(crate) fn dataspace_chunked(dims: &[hsize_t], chunks: &[usize]) -> Result<Hid, i32> {
        let mut max = [0 as hsize_t; crate::MAX_PROTOCOL_RANK];
        if dims.len() != chunks.len() || dims.len() > max.len() {
            return Err(-1);
        }
        for ((dst, &dim), &chunk) in max.iter_mut().zip(dims).zip(chunks) {
            *dst = dim.max(chunk as hsize_t);
        }
        hdf5_metno::sync::sync(|| {
            // SAFETY: dimensions and maxima have the checked equal rank. Each
            // maximum accommodates both the current extent and the chunk.
            let id = unsafe {
                h5::h5s::H5Screate_simple(dims.len() as c_int, dims.as_ptr(), max.as_ptr())
            };
            if hdf5_invalid(id) {
                Err(id as i32)
            } else {
                Ok(id)
            }
        })
    }

    pub(crate) fn dataspace_scalar() -> Result<Hid, i32> {
        hdf5_metno::sync::sync(|| {
            // SAFETY: H5S_SCALAR is a valid HDF5 dataspace class.
            unsafe {
                let id = h5::h5s::H5Screate(h5::h5s::H5S_class_t::H5S_SCALAR);
                if hdf5_invalid(id) {
                    Err(id as i32)
                } else {
                    Ok(id)
                }
            }
        })
    }

    pub(crate) fn dataspace_close(space: &mut Hid) -> i32 {
        hdf5_metno::sync::sync(|| {
            // SAFETY: `space` points to a live HDF5 dataspace identifier.
            unsafe { h5::h5s::H5Sclose(*space) }
        })
    }

    pub(crate) fn dataspace_select_none(space: Hid) -> i32 {
        hdf5_metno::sync::sync(|| {
            // SAFETY: space is live and H5Sselect_none accepts all HDF5
            // dataspace classes used by this backend.
            unsafe { h5::h5s::H5Sselect_none(space) }
        })
    }

    pub(crate) fn dataspace_select_hyperslab(
        space: Hid,
        start: &[hsize_t],
        count: &[hsize_t],
        block: &[hsize_t],
    ) -> i32 {
        debug_assert_eq!(start.len(), count.len());
        debug_assert_eq!(start.len(), block.len());
        hdf5_metno::sync::sync(|| {
            // SAFETY: start/block have equal rank and remain borrowed; null
            // stride selects one regular block at the requested start.
            unsafe {
                h5::h5s::H5Sselect_hyperslab(
                    space,
                    h5::h5s::H5S_seloper_t::H5S_SELECT_SET,
                    start.as_ptr(),
                    ptr::null(),
                    count.as_ptr(),
                    block.as_ptr(),
                )
            }
        })
    }

    pub(crate) fn type_copy(type_id: Hid) -> Result<Hid, i32> {
        hdf5_metno::sync::sync(|| {
            // SAFETY: type_id is a live HDF5 datatype identifier.
            unsafe {
                let id = h5::h5t::H5Tcopy(type_id);
                if hdf5_invalid(id) {
                    Err(id as i32)
                } else {
                    Ok(id)
                }
            }
        })
    }

    pub(crate) fn type_close(type_id: &mut Hid) -> i32 {
        hdf5_metno::sync::sync(|| {
            // SAFETY: type_id points to a live datatype identifier owned here.
            unsafe { h5::h5t::H5Tclose(*type_id) }
        })
    }

    pub(crate) fn type_create_compound(
        width: usize,
        scalar: Hid,
        abort_comm: ffi::MPI_Comm,
    ) -> Result<Hid, i32> {
        hdf5_metno::sync::sync(|| {
            // SAFETY: scalar is live; field names are static NUL-terminated
            // strings and offsets/widths were checked by the caller.
            unsafe {
                let compound = h5::h5t::H5Tcreate(h5::h5t::H5T_class_t::H5T_COMPOUND, width);
                if hdf5_invalid(compound) {
                    return Err(compound as i32);
                }
                let half = width / 2;
                let real = h5::h5t::H5Tinsert(compound, c"r".as_ptr().cast(), 0, scalar);
                let imag = h5::h5t::H5Tinsert(compound, c"i".as_ptr().cast(), half, scalar);
                let pack = h5::h5t::H5Tpack(compound);
                if real < 0 || imag < 0 || pack < 0 {
                    let error = if real < 0 {
                        real
                    } else if imag < 0 {
                        imag
                    } else {
                        pack
                    };
                    let close_code = h5::h5t::H5Tclose(compound);
                    if close_code < 0 {
                        super::abort_fail_stop(abort_comm);
                    }
                    return Err(error);
                }
                Ok(compound)
            }
        })
    }

    pub(crate) fn type_from_kind(
        code: u64,
        width: usize,
        abort_comm: ffi::MPI_Comm,
    ) -> Result<Hid, i32> {
        let base = match (code, width) {
            (1, 1) => *hdf5_metno::globals::H5T_STD_I8LE,
            (2, 1) => *hdf5_metno::globals::H5T_STD_U8LE,
            (3, 2) => *hdf5_metno::globals::H5T_STD_I16LE,
            (4, 2) => *hdf5_metno::globals::H5T_STD_U16LE,
            (5, 4) => *hdf5_metno::globals::H5T_STD_I32LE,
            (6, 4) => *hdf5_metno::globals::H5T_STD_U32LE,
            (7, 8) => *hdf5_metno::globals::H5T_STD_I64LE,
            (8, 8) => *hdf5_metno::globals::H5T_STD_U64LE,
            (9, 4) => *hdf5_metno::globals::H5T_IEEE_F32LE,
            (10, 8) => *hdf5_metno::globals::H5T_IEEE_F64LE,
            (11, 8) | (12, 16) => {
                let scalar = if code == 11 {
                    *hdf5_metno::globals::H5T_IEEE_F32LE
                } else {
                    *hdf5_metno::globals::H5T_IEEE_F64LE
                };
                return type_create_compound(width, scalar, abort_comm);
            }
            _ => return Err(-1),
        };
        type_copy(base)
    }

    pub(crate) fn type_matches(
        type_id: Hid,
        code: u64,
        width: usize,
        abort_comm: ffi::MPI_Comm,
    ) -> Result<bool, i32> {
        hdf5_metno::sync::sync(|| {
            // SAFETY: type_id is live; all HDF5 queries write only to local
            // scalar storage and member names are explicitly released below.
            unsafe {
                let class = h5::h5t::H5Tget_class(type_id);
                if h5::h5t::H5Tget_size(type_id) != width {
                    return Ok(false);
                }
                match code {
                    1..=10 => {
                        let canonical = match (code, width) {
                            (1, 1) => *hdf5_metno::globals::H5T_STD_I8LE,
                            (2, 1) => *hdf5_metno::globals::H5T_STD_U8LE,
                            (3, 2) => *hdf5_metno::globals::H5T_STD_I16LE,
                            (4, 2) => *hdf5_metno::globals::H5T_STD_U16LE,
                            (5, 4) => *hdf5_metno::globals::H5T_STD_I32LE,
                            (6, 4) => *hdf5_metno::globals::H5T_STD_U32LE,
                            (7, 8) => *hdf5_metno::globals::H5T_STD_I64LE,
                            (8, 8) => *hdf5_metno::globals::H5T_STD_U64LE,
                            (9, 4) => *hdf5_metno::globals::H5T_IEEE_F32LE,
                            (10, 8) => *hdf5_metno::globals::H5T_IEEE_F64LE,
                            _ => return Ok(false),
                        };
                        Ok(h5::h5t::H5Tequal(type_id, canonical) > 0)
                    }
                    11 | 12 => {
                        if class != h5::h5t::H5T_class_t::H5T_COMPOUND
                            || h5::h5t::H5Tget_nmembers(type_id) != 2
                        {
                            return Ok(false);
                        }
                        let expected_scalar = if code == 11 {
                            *hdf5_metno::globals::H5T_IEEE_F32LE
                        } else {
                            *hdf5_metno::globals::H5T_IEEE_F64LE
                        };
                        let expected_scalar_width = width / 2;
                        for (index, expected_name, expected_offset) in [
                            (0u32, b"r".as_slice(), 0usize),
                            (1u32, b"i".as_slice(), expected_scalar_width),
                        ] {
                            let name = h5::h5t::H5Tget_member_name(type_id, index);
                            if name.is_null() {
                                return Ok(false);
                            }
                            let actual_name = CStr::from_ptr(name).to_bytes();
                            let member_type = h5::h5t::H5Tget_member_type(type_id, index);
                            let valid = actual_name == expected_name
                                && h5::h5t::H5Tget_member_offset(type_id, index) == expected_offset
                                && !hdf5_invalid(member_type)
                                && expected_scalar_width == h5::h5t::H5Tget_size(member_type)
                                && h5::h5t::H5Tequal(member_type, expected_scalar) > 0;
                            let free_code = h5::h5::H5free_memory(name.cast());
                            if free_code < 0 {
                                super::abort_fail_stop(abort_comm);
                            }
                            if !hdf5_invalid(member_type) {
                                let close_code = h5::h5t::H5Tclose(member_type);
                                if close_code < 0 {
                                    super::abort_fail_stop(abort_comm);
                                }
                            }
                            if !valid {
                                return Ok(false);
                            }
                        }
                        Ok(true)
                    }
                    _ => Ok(false),
                }
            }
        })
    }

    pub(crate) fn attr_exists(obj: Hid, name: &CStr) -> Result<bool, i32> {
        hdf5_metno::sync::sync(|| {
            // SAFETY: obj/name are live and valid for this query.
            unsafe {
                let result = h5::h5a::H5Aexists(obj, name.as_ptr());
                if result < 0 {
                    Err(result as i32)
                } else {
                    Ok(result > 0)
                }
            }
        })
    }

    pub(crate) fn attr_create(
        obj: Hid,
        name: &CStr,
        datatype: Hid,
        space: Hid,
    ) -> Result<Hid, i32> {
        hdf5_metno::sync::sync(|| {
            // SAFETY: all identifiers/name are live and the created attribute
            // owns its own HDF5 handle.
            unsafe {
                let id = h5::h5a::H5Acreate2(
                    obj,
                    name.as_ptr(),
                    datatype,
                    space,
                    h5::h5p::H5P_DEFAULT,
                    h5::h5p::H5P_DEFAULT,
                );
                if hdf5_invalid(id) {
                    Err(id as i32)
                } else {
                    Ok(id)
                }
            }
        })
    }

    pub(crate) fn attr_open(obj: Hid, name: &CStr) -> Result<Hid, i32> {
        hdf5_metno::sync::sync(|| {
            // SAFETY: obj/name are live for the duration of the open call.
            unsafe {
                let id = h5::h5a::H5Aopen(obj, name.as_ptr(), h5::h5p::H5P_DEFAULT);
                if hdf5_invalid(id) {
                    Err(id as i32)
                } else {
                    Ok(id)
                }
            }
        })
    }

    pub(crate) fn attr_write(attr: Hid, datatype: Hid, bytes: &[u8]) -> i32 {
        hdf5_metno::sync::sync(|| {
            // SAFETY: bytes remains live for the call and is interpreted using
            // the explicit little-endian datatype supplied by the caller.
            unsafe {
                h5::h5a::H5Awrite(
                    attr,
                    datatype,
                    if bytes.is_empty() {
                        ptr::null()
                    } else {
                        bytes.as_ptr().cast()
                    },
                )
            }
        })
    }

    pub(crate) fn type_create_string(size: usize, abort_comm: ffi::MPI_Comm) -> Result<Hid, i32> {
        hdf5_metno::sync::sync(|| unsafe {
            let datatype = h5::h5t::H5Tcopy(*hdf5_metno::globals::H5T_C_S1);
            if hdf5_invalid(datatype) {
                return Err(datatype as i32);
            }
            let valid = h5::h5t::H5Tset_size(datatype, size) >= 0
                && h5::h5t::H5Tset_cset(datatype, h5::h5t::H5T_cset_t::H5T_CSET_UTF8) >= 0
                && h5::h5t::H5Tset_strpad(datatype, h5::h5t::H5T_str_t::H5T_STR_NULLTERM) >= 0;
            if valid {
                Ok(datatype)
            } else {
                if h5::h5t::H5Tclose(datatype) < 0 {
                    super::abort_fail_stop(abort_comm);
                }
                Err(-1)
            }
        })
    }

    pub(crate) fn type_size(type_id: Hid) -> Result<usize, i32> {
        hdf5_metno::sync::sync(|| unsafe {
            let size = h5::h5t::H5Tget_size(type_id);
            if size == 0 { Err(-1) } else { Ok(size) }
        })
    }

    pub(crate) fn type_matches_string(type_id: Hid, size: usize) -> bool {
        hdf5_metno::sync::sync(|| unsafe {
            h5::h5t::H5Tget_class(type_id) == h5::h5t::H5T_class_t::H5T_STRING
                && h5::h5t::H5Tget_size(type_id) == size
                && h5::h5t::H5Tis_variable_str(type_id) == 0
                && h5::h5t::H5Tget_cset(type_id) == h5::h5t::H5T_cset_t::H5T_CSET_UTF8
                && h5::h5t::H5Tget_strpad(type_id) == h5::h5t::H5T_str_t::H5T_STR_NULLTERM
        })
    }

    pub(crate) fn attr_read(attr: Hid, datatype: Hid, bytes: &mut [u8]) -> i32 {
        hdf5_metno::sync::sync(|| {
            // SAFETY: bytes is writable for the call and has exactly the
            // storage represented by the explicit datatype.
            unsafe {
                h5::h5a::H5Aread(
                    attr,
                    datatype,
                    if bytes.is_empty() {
                        ptr::null_mut()
                    } else {
                        bytes.as_mut_ptr().cast()
                    },
                )
            }
        })
    }

    pub(crate) fn attr_space(attr: Hid) -> Result<Hid, i32> {
        hdf5_metno::sync::sync(|| {
            // SAFETY: attr is live and H5Aget_space returns a new identifier.
            unsafe {
                let id = h5::h5a::H5Aget_space(attr);
                if hdf5_invalid(id) {
                    Err(id as i32)
                } else {
                    Ok(id)
                }
            }
        })
    }

    pub(crate) fn attr_type(attr: Hid) -> Result<Hid, i32> {
        hdf5_metno::sync::sync(|| {
            // SAFETY: attr is live and H5Aget_type returns a new identifier.
            unsafe {
                let id = h5::h5a::H5Aget_type(attr);
                if hdf5_invalid(id) {
                    Err(id as i32)
                } else {
                    Ok(id)
                }
            }
        })
    }

    pub(crate) fn attr_close(attr: &mut Hid) -> i32 {
        hdf5_metno::sync::sync(|| {
            // SAFETY: attr points to a live attribute identifier.
            unsafe { h5::h5a::H5Aclose(*attr) }
        })
    }

    pub(crate) fn space_shape(space: Hid) -> Result<Vec<hsize_t>, i32> {
        hdf5_metno::sync::sync(|| {
            // SAFETY: space is live; the first query returns a bounded rank,
            // then the second writes exactly that many dimensions.
            unsafe {
                let rank = h5::h5s::H5Sget_simple_extent_ndims(space);
                if rank < 0 {
                    return Err(rank);
                }
                let rank = usize::try_from(rank).map_err(|_| -1)?;
                if rank > 1024 {
                    return Err(-1);
                }
                let mut dims = Vec::new();
                dims.try_reserve_exact(rank).map_err(|_| -1)?;
                dims.resize(rank, 0);
                let mut maxdims = Vec::new();
                maxdims.try_reserve_exact(rank).map_err(|_| -1)?;
                maxdims.resize(rank, 0);
                let actual = h5::h5s::H5Sget_simple_extent_dims(
                    space,
                    dims.as_mut_ptr(),
                    maxdims.as_mut_ptr(),
                );
                if actual < 0 { Err(actual) } else { Ok(dims) }
            }
        })
    }

    pub(crate) fn attr_shape(space: Hid) -> Result<Vec<hsize_t>, i32> {
        space_shape(space)
    }

    pub(crate) fn xfer_create(abort_comm: ffi::MPI_Comm, collective: bool) -> Result<Hid, i32> {
        hdf5_metno::sync::sync(|| {
            // SAFETY: the global dataset-transfer property-list class is live.
            unsafe {
                let id = h5::h5p::H5Pcreate(*hdf5_metno::globals::H5P_DATASET_XFER);
                if hdf5_invalid(id) {
                    return Err(id as i32);
                }
                let code = h5::h5p::H5Pset_dxpl_mpio(
                    id,
                    if collective {
                        h5::h5p::H5FD_mpio_xfer_t::H5FD_MPIO_COLLECTIVE
                    } else {
                        h5::h5p::H5FD_mpio_xfer_t::H5FD_MPIO_INDEPENDENT
                    },
                );
                if code < 0 {
                    let close_code = h5::h5p::H5Pclose(id);
                    if close_code < 0 {
                        super::abort_fail_stop(abort_comm);
                    }
                    Err(code)
                } else {
                    Ok(id)
                }
            }
        })
    }

    pub(crate) fn plist_close(plist: &mut Hid) -> i32 {
        hdf5_metno::sync::sync(|| {
            // SAFETY: plist points to a live property-list identifier.
            unsafe { h5::h5p::H5Pclose(*plist) }
        })
    }

    pub(crate) fn group_link_count(group: Hid) -> Result<u64, i32> {
        hdf5_metno::sync::sync(|| unsafe {
            // SAFETY: group is live and the initialized output has the native layout.
            let mut info = h5::h5g::H5G_info_t::default();
            let code = h5::h5g::H5Gget_info(group, &mut info);
            if code < 0 { Err(code) } else { Ok(info.nlinks) }
        })
    }

    pub(crate) fn link_name_by_idx(
        group: Hid,
        index: u64,
        buffer: &mut [u8],
    ) -> Result<usize, i32> {
        let index = index as h5::h5::hsize_t;
        hdf5_metno::sync::sync(|| unsafe {
            let n = h5::h5l::H5Lget_name_by_idx(
                group,
                c".".as_ptr(),
                h5::h5::H5_index_t::H5_INDEX_NAME,
                h5::h5::H5_iter_order_t::H5_ITER_INC,
                index,
                buffer.as_mut_ptr().cast(),
                buffer.len(),
                h5::h5p::H5P_DEFAULT,
            );
            if n < 0 { Err(n as i32) } else { Ok(n as usize) }
        })
    }

    pub(crate) fn filter_avail(filter: h5::h5z::H5Z_filter_t) -> bool {
        hdf5_metno::sync::sync(|| unsafe { h5::h5z::H5Zfilter_avail(filter) > 0 })
    }

    pub(crate) fn filter_info(filter: h5::h5z::H5Z_filter_t) -> Result<u32, i32> {
        #[cfg(test)]
        if let Some(flags) = super::FILTER_CAPABILITIES.with(std::cell::Cell::get) {
            return Ok(flags);
        }
        hdf5_metno::sync::sync(|| unsafe {
            let mut flags = 0u32;
            let code = h5::h5z::H5Zget_filter_info(filter, &mut flags);
            if code < 0 { Err(code) } else { Ok(flags) }
        })
    }

    pub(crate) fn plist_set_shuffle(plist: Hid) -> i32 {
        hdf5_metno::sync::sync(|| unsafe { h5::h5p::H5Pset_shuffle(plist) })
    }

    pub(crate) fn plist_set_deflate(plist: Hid, level: u32) -> i32 {
        hdf5_metno::sync::sync(|| unsafe { h5::h5p::H5Pset_deflate(plist, level) })
    }

    pub(crate) fn dataset_io(
        dataset: Hid,
        mem_type: Hid,
        mem_space: Hid,
        file_space: Hid,
        xfer: Hid,
        buffer: &mut [u8],
        write: bool,
    ) -> i32 {
        hdf5_metno::sync::sync(|| {
            // SAFETY: all identifiers are live, buffer has the selected element
            // count in the explicit packed memory type, and all ranks call the
            // same collective HDF5 transfer.
            unsafe {
                #[cfg(test)]
                {
                    use h5::h5p::H5FD_mpio_xfer_t;
                    let mut actual = H5FD_mpio_xfer_t::H5FD_MPIO_COLLECTIVE;
                    assert!(h5::h5p::H5Pget_dxpl_mpio(xfer, &mut actual) >= 0);
                    super::record_payload(if actual == H5FD_mpio_xfer_t::H5FD_MPIO_INDEPENDENT {
                        5
                    } else {
                        4
                    });
                }
                if write {
                    h5::h5d::H5Dwrite(
                        dataset,
                        mem_type,
                        mem_space,
                        file_space,
                        xfer,
                        if buffer.is_empty() {
                            ptr::null()
                        } else {
                            buffer.as_ptr().cast()
                        },
                    )
                } else {
                    h5::h5d::H5Dread(
                        dataset,
                        mem_type,
                        mem_space,
                        file_space,
                        xfer,
                        if buffer.is_empty() {
                            ptr::null_mut()
                        } else {
                            buffer.as_mut_ptr().cast()
                        },
                    )
                }
            }
        })
    }
}
