//! Append-only v2: all integers are little-endian u64. The 32-byte container
//! header is magic/version/header-size/reserved. Each 72-byte record header is
//! magic/version/name-length/metadata-length/payload-length/total-length/type/
//! width/reserved, followed by UTF-8 name, metadata, canonical payload, and a
//! commit word. Metadata is spatial-rank/extra-rank/spatial-dims/extra-dims.
//! Header and payload are flushed before the independent trailing commit word.
//! No mutable global index is needed to recover the committed prefix.

use std::ffi::CString;
use std::path::Path;

use mpi::collective::{CommunicatorCollectives, Root, SystemOperation};
use mpi::topology::Communicator;
use mpi::traits::AsRaw;
use pencil_array::{PencilArrayView, PencilArrayViewMut};

use crate::ffi;
use crate::format::{IoElement, element_count, pack_view, prepare_physical_values};
use crate::mpi_io::agree_phase;
use crate::mpi_io::{DatatypeGuard, build_layout, duplicate_comm, finish_resources};
use crate::{COMMIT_MARKER, IoError, MAX_PROTOCOL_RANK, NamedIoError};

const MAGIC: &[u8; 8] = b"PIONAM02";
const RECORD_MAGIC: &[u8; 8] = b"PIOREC02";
const VERSION: u64 = 2;
const CONTAINER_BYTES: usize = 32;
const RECORD_HEADER_BYTES: usize = 72;
const MAX_NAME: usize = 1024;
const OP_WRITE: u64 = 0x101;
const OP_READ: u64 = 0x102;
const OP_APPEND: u64 = 0x103;

fn u64at(b: &[u8], p: usize) -> Option<u64> {
    b.get(p..p + 8)
        .and_then(|x| x.try_into().ok())
        .map(u64::from_le_bytes)
}
fn put(out: &mut Vec<u8>, x: u64) {
    out.extend_from_slice(&x.to_le_bytes());
}

fn name_bytes<C: CommunicatorCollectives>(comm: &C, name: &str) -> Result<(), NamedIoError> {
    let valid = !name.is_empty() && name.len() <= MAX_NAME && !name.as_bytes().contains(&0);
    if agree_phase(comm, valid, "named key validation").is_err() {
        return Err(NamedIoError::InvalidName);
    }
    let n = name.len() as u64;
    let mut lo = 0;
    let mut hi = 0;
    comm.all_reduce_into(&n, &mut lo, SystemOperation::min());
    comm.all_reduce_into(&n, &mut hi, SystemOperation::max());
    if lo != hi {
        return Err(IoError::CollectiveDescriptorMismatch.into());
    }
    let mut root = allocate(comm, name.len())?;
    if comm.rank() == 0 {
        root.copy_from_slice(name.as_bytes());
    }
    comm.process_at_rank(0).broadcast_into(&mut root[..]);
    agree_phase(comm, root == name.as_bytes(), "named key agreement")?;
    Ok(())
}

fn allocate<C: CommunicatorCollectives>(comm: &C, len: usize) -> Result<Vec<u8>, NamedIoError> {
    let mut bytes = Vec::new();
    let prepared = bytes.try_reserve_exact(len);
    agree_phase(comm, prepared.is_ok(), "named allocation")?;
    bytes.resize(len, 0);
    Ok(bytes)
}

fn prepared<C: CommunicatorCollectives, T>(
    comm: &C,
    value: Result<T, IoError>,
    phase: &'static str,
) -> Result<T, NamedIoError> {
    agree_phase(comm, value.is_ok(), phase)?;
    value.map_err(Into::into)
}

fn path_c<P: AsRef<Path>>(p: P) -> Result<CString, IoError> {
    let path = p.as_ref().to_str().ok_or(IoError::InvalidPath)?;
    let len = path
        .len()
        .checked_add(1)
        .ok_or(IoError::SizeLimit { what: "path" })?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(len)
        .map_err(|_| IoError::AllocationFailed { requested: len })?;
    bytes.extend_from_slice(path.as_bytes());
    bytes.push(0);
    CString::from_vec_with_nul(bytes).map_err(|_| IoError::InvalidPath)
}
fn open<C: CommunicatorCollectives + Communicator>(
    comm: &C,
    raw: ffi::MPI_Comm,
    p: &CString,
    mode: u8,
) -> Result<ffi::MPI_File, NamedIoError> {
    let r = if mode == 0 {
        ffi::file_open(raw, p, true)
    } else if mode == 1 {
        ffi::file_open(raw, p, false)
    } else {
        ffi::file_open_update(raw, p)
    };
    let (ok, mixed) = crate::mpi_io::collective_state(comm, r.is_ok());
    if mixed {
        crate::mpi_io::abort_unrecoverable(comm.as_raw(), "named MPI_File_open");
    }
    if !ok {
        return Err(NamedIoError::Io(
            r.err()
                .map(|code| IoError::Mpi {
                    operation: "MPI_File_open",
                    code,
                })
                .unwrap_or(IoError::CollectivePrecondition {
                    phase: "named file open",
                }),
        ));
    }
    Ok(r.expect("collectively opened file"))
}

#[derive(Clone)]
struct Rec {
    name: Vec<u8>,
    typ: u64,
    width: usize,
    global: Vec<usize>,
    extra: Vec<usize>,
    payload_offset: usize,
}
fn read_segment<C: CommunicatorCollectives>(
    comm: &C,
    file: ffi::MPI_File,
    offset: usize,
    len: usize,
) -> Result<Vec<u8>, NamedIoError> {
    let mut out = allocate(comm, len)?;
    let result = if comm.rank() == 0 {
        ffi::file_read_at_all(file, offset as i64, &mut out)
    } else {
        ffi::file_read_at_all(file, 0, &mut [])
    };
    let ok = if comm.rank() == 0 {
        matches!(result, Ok(n) if n == len)
    } else {
        matches!(result, Ok(0))
    };
    agree_phase(comm, ok, "named metadata read").map_err(NamedIoError::Io)?;
    comm.process_at_rank(0).broadcast_into(&mut out[..]);
    Ok(out)
}

/// Scan only fixed headers, names, metadata, and commit markers. Payload bytes
/// never enter the root's buffer.
fn scan_file<C: CommunicatorCollectives>(
    comm: &C,
    file: ffi::MPI_File,
) -> Result<(Vec<Rec>, usize, usize), NamedIoError> {
    let native_size = ffi::file_get_size(file);
    agree_phase(comm, native_size.is_ok(), "MPI_File_get_size").map_err(NamedIoError::Io)?;
    let local_size = native_size.map_err(|code| {
        NamedIoError::Io(IoError::Mpi {
            operation: "MPI_File_get_size",
            code,
        })
    })?;
    let mut min_size = local_size;
    let mut max_size = local_size;
    comm.all_reduce_into(&local_size, &mut min_size, SystemOperation::min());
    comm.all_reduce_into(&local_size, &mut max_size, SystemOperation::max());
    if min_size != max_size {
        return Err(NamedIoError::Io(IoError::CollectiveDescriptorMismatch));
    }
    let size = usize::try_from(local_size).map_err(|_| {
        NamedIoError::Io(IoError::SizeLimit {
            what: "container size",
        })
    })?;
    // File size itself does not allocate memory. An oversized damaged tail
    // must not hide an earlier bounded, committed record.
    if size < CONTAINER_BYTES {
        return Err(NamedIoError::Io(IoError::InvalidFile {
            reason: "named container header",
        }));
    }
    let h = read_segment(comm, file, 0, CONTAINER_BYTES)?;
    if &h[..8] != MAGIC
        || u64at(&h, 8) != Some(VERSION)
        || u64at(&h, 16) != Some(CONTAINER_BYTES as u64)
        || u64at(&h, 24) != Some(0)
    {
        return Err(NamedIoError::Io(IoError::InvalidFile {
            reason: "named container header",
        }));
    }
    let mut p = CONTAINER_BYTES;
    let mut out = Vec::new();
    while p < size {
        if size - p < RECORD_HEADER_BYTES {
            break;
        }
        let rh = read_segment(comm, file, p, RECORD_HEADER_BYTES)?;
        if &rh[..8] != RECORD_MAGIC || u64at(&rh, 8) != Some(VERSION) || u64at(&rh, 64) != Some(0) {
            break;
        }
        let nl = match u64at(&rh, 16).and_then(|x| usize::try_from(x).ok()) {
            Some(x) if x > 0 && x <= MAX_NAME => x,
            _ => break,
        };
        let ml = match u64at(&rh, 24).and_then(|x| usize::try_from(x).ok()) {
            Some(x) if x <= crate::MAX_HEADER_BYTES => x,
            _ => break,
        };
        let payload = match u64at(&rh, 32).and_then(|x| usize::try_from(x).ok()) {
            Some(x) => x,
            _ => break,
        };
        let total = match u64at(&rh, 40).and_then(|x| usize::try_from(x).ok()) {
            Some(x) => x,
            _ => break,
        };
        let end = match p.checked_add(total) {
            Some(x) if x <= size && i64::try_from(x).is_ok() => x,
            _ => break,
        };
        let minimum = RECORD_HEADER_BYTES
            .checked_add(nl)
            .and_then(|x| x.checked_add(ml))
            .and_then(|x| x.checked_add(payload))
            .and_then(|x| x.checked_add(8));
        if minimum != Some(total) {
            break;
        }
        let body_len = match nl.checked_add(ml) {
            Some(x) => x,
            None => break,
        };
        let body = read_segment(comm, file, p + RECORD_HEADER_BYTES, body_len)?;
        let name = &body[..nl];
        if std::str::from_utf8(name).is_err()
            || name.contains(&0)
            || out.iter().any(|r: &Rec| r.name == name)
        {
            break;
        }
        let meta = &body[nl..];
        let nr = match u64at(meta, 0).and_then(|x| usize::try_from(x).ok()) {
            Some(x) if x > 0 && x <= MAX_PROTOCOL_RANK => x,
            _ => break,
        };
        let er = match u64at(meta, 8).and_then(|x| usize::try_from(x).ok()) {
            Some(x) if x <= MAX_PROTOCOL_RANK => x,
            _ => break,
        };
        let rank = match nr.checked_add(er) {
            Some(x) if x <= MAX_PROTOCOL_RANK => x,
            _ => break,
        };
        let need = match 16usize.checked_add(rank.saturating_mul(8)) {
            Some(x) => x,
            None => break,
        };
        if need != ml {
            break;
        }
        let mut dims = Vec::new();
        agree_phase(
            comm,
            dims.try_reserve_exact(rank).is_ok(),
            "named dimensions allocation",
        )?;
        let mut elements = 1usize;
        for i in 0..rank {
            let d = match u64at(meta, 16 + i * 8).and_then(|x| usize::try_from(x).ok()) {
                Some(x) => x,
                None => {
                    dims.clear();
                    break;
                }
            };
            elements = match elements.checked_mul(d) {
                Some(x) => x,
                None => {
                    dims.clear();
                    break;
                }
            };
            dims.push(d);
        }
        if dims.len() != rank {
            break;
        }
        let typ = match u64at(&rh, 48) {
            Some(x) => x,
            None => break,
        };
        let width = match u64at(&rh, 56).and_then(|x| usize::try_from(x).ok()) {
            Some(x) => x,
            None => break,
        };
        let known = match typ {
            1 | 2 => 1,
            3 | 4 => 2,
            5 | 6 | 9 => 4,
            7 | 8 | 10 | 11 => 8,
            12 => 16,
            _ => 0,
        };
        if known == 0 || width != known || elements.checked_mul(width) != Some(payload) {
            break;
        }
        let marker = read_segment(comm, file, end - 8, 8)?;
        if u64::from_le_bytes(marker.try_into().unwrap()) != COMMIT_MARKER {
            break;
        }
        let mut extra = Vec::new();
        let mut key = Vec::new();
        let allocated = extra.try_reserve_exact(er).is_ok()
            && key.try_reserve_exact(nl).is_ok()
            && out.try_reserve(1).is_ok();
        agree_phase(comm, allocated, "named record index allocation")?;
        extra.extend_from_slice(&dims[nr..]);
        dims.truncate(nr);
        key.extend_from_slice(name);
        out.push(Rec {
            name: key,
            typ,
            width,
            global: dims,
            extra,
            payload_offset: p + RECORD_HEADER_BYTES + nl + ml,
        });
        p = end;
    }
    Ok((out, p, size))
}

fn record<T: IoElement, const N: usize, const M: usize>(
    name: &str,
    view: &PencilArrayView<'_, T, N, M>,
    offset: usize,
) -> Result<(Vec<u8>, usize), NamedIoError> {
    let global = view.pencil().global_shape();
    let extra = view.extra_shape().dimensions();
    let rank = N
        .checked_add(extra.len())
        .ok_or(NamedIoError::Io(IoError::SizeLimit {
            what: "logical rank",
        }))?;
    if name.is_empty()
        || name.len() > MAX_NAME
        || name.as_bytes().contains(&0)
        || N == 0
        || N > MAX_PROTOCOL_RANK
        || extra.len() > MAX_PROTOCOL_RANK
        || rank > MAX_PROTOCOL_RANK
    {
        return Err(NamedIoError::InvalidName);
    }
    let elems = element_count(global)
        .and_then(|x| {
            element_count(extra).and_then(|y| {
                x.checked_mul(y).ok_or(IoError::SizeLimit {
                    what: "global elements",
                })
            })
        })
        .map_err(NamedIoError::Io)?;
    let payload = elems
        .checked_mul(T::WIDTH)
        .ok_or(NamedIoError::Io(IoError::SizeLimit { what: "payload" }))?;
    let meta_len = 16usize
        .checked_add(
            rank.checked_mul(8)
                .ok_or(NamedIoError::Io(IoError::SizeLimit { what: "metadata" }))?,
        )
        .ok_or(NamedIoError::Io(IoError::SizeLimit { what: "metadata" }))?;
    if meta_len > crate::MAX_HEADER_BYTES {
        return Err(NamedIoError::Io(IoError::SizeLimit { what: "metadata" }));
    }
    let mut meta = Vec::new();
    meta.try_reserve_exact(meta_len).map_err(|_| {
        NamedIoError::Io(IoError::AllocationFailed {
            requested: meta_len,
        })
    })?;
    put(&mut meta, N as u64);
    put(&mut meta, extra.len() as u64);
    for &x in global {
        put(&mut meta, x as u64);
    }
    for &x in extra {
        put(&mut meta, x as u64);
    }
    let total = RECORD_HEADER_BYTES
        .checked_add(name.len())
        .and_then(|x| x.checked_add(meta.len()))
        .and_then(|x| x.checked_add(payload))
        .and_then(|x| x.checked_add(8))
        .ok_or(NamedIoError::Io(IoError::SizeLimit { what: "record" }))?;
    if offset
        .checked_add(total)
        .filter(|&x| i64::try_from(x).is_ok())
        .is_none()
    {
        return Err(NamedIoError::Io(IoError::SizeLimit { what: "record" }));
    }
    let header_len = RECORD_HEADER_BYTES
        .checked_add(name.len())
        .and_then(|x| x.checked_add(meta.len()))
        .ok_or(NamedIoError::Io(IoError::SizeLimit { what: "header" }))?;
    let mut h = Vec::new();
    h.try_reserve_exact(header_len).map_err(|_| {
        NamedIoError::Io(IoError::AllocationFailed {
            requested: header_len,
        })
    })?;
    h.extend_from_slice(RECORD_MAGIC);
    put(&mut h, VERSION);
    put(&mut h, name.len() as u64);
    put(&mut h, meta.len() as u64);
    put(&mut h, payload as u64);
    put(&mut h, total as u64);
    put(&mut h, T::CODE);
    put(&mut h, T::WIDTH as u64);
    put(&mut h, 0);
    h.extend_from_slice(name.as_bytes());
    h.extend_from_slice(&meta);
    Ok((h, offset + total))
}

fn container_header() -> [u8; CONTAINER_BYTES] {
    let mut h = [0; CONTAINER_BYTES];
    h[..8].copy_from_slice(MAGIC);
    h[8..16].copy_from_slice(&VERSION.to_le_bytes());
    h[16..24].copy_from_slice(&(CONTAINER_BYTES as u64).to_le_bytes());
    h
}

/// Creates an exclusive v2 named container and collectively writes its first record.
/// Keys are nonempty UTF-8 (at most 1024 bytes), without NUL; slashes are literal.
/// Container offsets must fit MPI_Offset; local transfer counts and subarray
/// dimensions must fit native MPI integers. Global payloads have no fixed byte cap.
pub fn write_mpi_named<
    P: AsRef<Path>,
    S: AsRef<str>,
    T: IoElement,
    const N: usize,
    const M: usize,
>(
    path: P,
    name: S,
    view: PencilArrayView<'_, T, N, M>,
) -> Result<(), NamedIoError> {
    write_named(path.as_ref(), name.as_ref(), view, false, false)
}

/// Appends an independently committed record without modifying the committed prefix.
/// Duplicate keys and malformed or incomplete tails are rejected before writing.
pub fn append_mpi_named<
    P: AsRef<Path>,
    S: AsRef<str>,
    T: IoElement,
    const N: usize,
    const M: usize,
>(
    path: P,
    name: S,
    view: PencilArrayView<'_, T, N, M>,
) -> Result<(), NamedIoError> {
    write_named(path.as_ref(), name.as_ref(), view, true, false)
}

fn write_named<T: IoElement, const N: usize, const M: usize>(
    path: &Path,
    name: &str,
    view: PencilArrayView<'_, T, N, M>,
    append: bool,
    inject_post_cleanup_failure: bool,
) -> Result<(), NamedIoError> {
    let comm = view.pencil().topology().communicator();
    crate::mpi_io::descriptor_agreement(
        comm,
        path,
        if append { OP_APPEND } else { OP_WRITE },
        view.pencil().global_shape(),
        view.extra_shape().dimensions(),
        view.pencil().topology().process_grid(),
        view.pencil().permutation().axes(),
        T::CODE,
        T::WIDTH,
    )?;
    name_bytes(comm, name)?;
    let pc = prepared(comm, path_c(path), "named path preparation")?;
    let packed = prepared(comm, pack_view(&view), "named payload packing")?;
    let layout = prepared(
        comm,
        build_layout(
            view.pencil().global_shape(),
            view.extra_shape().dimensions(),
            &view.local_spatial_shape(),
            view.pencil().local_ranges(),
            view.len(),
            T::WIDTH,
        ),
        "named layout preparation",
    )?;
    // Build bounded metadata before opening. Append adjusts only the absolute end.
    let record_result = record(name, &view, 0);
    agree_phase(comm, record_result.is_ok(), "named record preparation")?;
    let (head, record_len) = record_result?;
    let duplicate = duplicate_comm(comm)?;
    let file = match open(comm, duplicate.raw, &pc, if append { 2 } else { 0 }) {
        Ok(file) => file,
        Err(e) => {
            crate::mpi_io::finish_comm(comm, duplicate);
            return Err(e);
        }
    };
    let mut datatype = None;
    let mut marker_attempted = false;
    let result = (|| {
        agree_phase(
            comm,
            ffi::file_set_errors_return(file) == ffi::MPI_SUCCESS as i32,
            "named file error-handler setup",
        )?;
        let offset = if append {
            let (records, tail, size) = scan_file(comm, file)?;
            if tail != size {
                return Err(NamedIoError::InvalidTail);
            }
            if records.iter().any(|r| r.name == name.as_bytes()) {
                return Err(NamedIoError::DuplicateName);
            }
            size
        } else {
            CONTAINER_BYTES
        };
        let end = offset
            .checked_add(record_len)
            .filter(|&n| i64::try_from(n).is_ok())
            .ok_or(IoError::SizeLimit {
                what: "named container",
            })?;
        let native_type = if layout.empty {
            Ok(None)
        } else {
            ffi::type_create_subarray(&layout.global, &layout.local, &layout.starts, comm.as_raw())
                .map(|raw| Some(DatatypeGuard { raw }))
        };
        let type_ok = native_type.is_ok();
        datatype = native_type.ok().flatten();
        agree_phase(comm, type_ok, "named datatype")?;
        if !append {
            let header = container_header();
            write_metadata(comm, file, 0, &header)?;
        }
        write_metadata(comm, file, offset, &head)?;
        agree_phase(
            comm,
            ffi::file_sync(file) == ffi::MPI_SUCCESS as i32,
            "named header flush",
        )?;
        let ft = datatype
            .as_ref()
            .map_or_else(ffi::byte_datatype, |dt| dt.raw);
        agree_phase(
            comm,
            ffi::file_set_view(file, (offset + head.len()) as i64, ft) == ffi::MPI_SUCCESS as i32,
            "named payload view",
        )?;
        let written = ffi::file_write_all(file, &packed);
        agree_phase(
            comm,
            matches!(written, Ok(n) if n == packed.len()),
            "named payload write",
        )?;
        agree_phase(
            comm,
            ffi::file_sync(file) == ffi::MPI_SUCCESS as i32,
            "named payload flush",
        )?;
        agree_phase(
            comm,
            ffi::file_set_view(file, 0, ffi::byte_datatype()) == ffi::MPI_SUCCESS as i32,
            "named marker view",
        )?;
        marker_attempted = true;
        write_metadata(comm, file, end - 8, &COMMIT_MARKER.to_le_bytes())?;
        agree_phase(
            comm,
            ffi::file_sync(file) == ffi::MPI_SUCCESS as i32,
            "named marker flush",
        )?;
        Ok(())
    })();
    let cleanup = finish_resources(
        comm,
        duplicate,
        crate::mpi_io::FileGuard { raw: file },
        datatype,
    );
    let cleanup = if inject_post_cleanup_failure && comm.rank() == 0 {
        Some(IoError::CommitUncertain {
            stage: "named post-cleanup result",
        })
    } else {
        cleanup
    };
    let cleanup = crate::mpi_io::aggregate_cleanup_result(comm, cleanup);
    if marker_attempted && (result.is_err() || cleanup.is_some()) {
        return Err(IoError::CommitUncertain {
            stage: "named commit or cleanup",
        }
        .into());
    }
    result?;
    if let Some(error) = cleanup {
        return Err(error.into());
    }
    Ok(())
}

fn write_metadata<C: CommunicatorCollectives>(
    comm: &C,
    file: ffi::MPI_File,
    offset: usize,
    bytes: &[u8],
) -> Result<(), NamedIoError> {
    let local = if comm.rank() == 0 { bytes } else { &[] };
    let result = ffi::file_write_at_all(file, offset as i64, local);
    agree_phase(
        comm,
        matches!(result, Ok(n) if n == local.len()),
        "named metadata write",
    )?;
    Ok(())
}

/// Reads a committed named record using collective MPI payload I/O.
/// A damaged tail does not hide earlier records. The destination changes only
/// after staging, native cleanup, and final collective success.
pub fn read_mpi_named<
    P: AsRef<Path>,
    S: AsRef<str>,
    T: IoElement,
    const N: usize,
    const M: usize,
>(
    path: P,
    name: S,
    view: PencilArrayViewMut<'_, T, N, M>,
) -> Result<(), NamedIoError> {
    read_named(path.as_ref(), name.as_ref(), view, false)
}

fn read_named<T: IoElement, const N: usize, const M: usize>(
    path: &Path,
    name: &str,
    mut view: PencilArrayViewMut<'_, T, N, M>,
    inject_post_cleanup_failure: bool,
) -> Result<(), NamedIoError> {
    let comm = view.pencil().topology().communicator();
    crate::mpi_io::descriptor_agreement(
        comm,
        path,
        OP_READ,
        view.pencil().global_shape(),
        view.extra_shape().dimensions(),
        view.pencil().topology().process_grid(),
        view.pencil().permutation().axes(),
        T::CODE,
        T::WIDTH,
    )?;
    name_bytes(comm, name)?;
    let pc = prepared(comm, path_c(path), "named path preparation")?;
    let layout = prepared(
        comm,
        build_layout(
            view.pencil().global_shape(),
            view.extra_shape().dimensions(),
            &view.local_spatial_shape(),
            view.pencil().local_ranges(),
            view.len(),
            T::WIDTH,
        ),
        "named read layout",
    )?;
    let size = view.len().checked_mul(T::WIDTH).ok_or(IoError::SizeLimit {
        what: "named read buffer",
    });
    let size = prepared(comm, size, "named read size")?;
    let mut packed = allocate(comm, size)?;
    let duplicate = duplicate_comm(comm)?;
    let file = match open(comm, duplicate.raw, &pc, 1) {
        Ok(file) => file,
        Err(e) => {
            crate::mpi_io::finish_comm(comm, duplicate);
            return Err(e);
        }
    };
    let mut datatype = None;
    let result = (|| {
        agree_phase(
            comm,
            ffi::file_set_errors_return(file) == ffi::MPI_SUCCESS as i32,
            "named file error-handler setup",
        )?;
        let (records, _, _) = scan_file(comm, file)?;
        let record = records
            .iter()
            .find(|r| r.name == name.as_bytes())
            .ok_or(NamedIoError::NotFound)?;
        if record.typ != T::CODE
            || record.width != T::WIDTH
            || record.global != view.pencil().global_shape()
            || record.extra != view.extra_shape().dimensions()
        {
            return Err(IoError::MetadataMismatch {
                field: "named record",
            }
            .into());
        }
        let native_type = if layout.empty {
            Ok(None)
        } else {
            ffi::type_create_subarray(&layout.global, &layout.local, &layout.starts, comm.as_raw())
                .map(|raw| Some(DatatypeGuard { raw }))
        };
        let type_ok = native_type.is_ok();
        datatype = native_type.ok().flatten();
        agree_phase(comm, type_ok, "named read datatype")?;
        let ft = datatype
            .as_ref()
            .map_or_else(ffi::byte_datatype, |dt| dt.raw);
        agree_phase(
            comm,
            ffi::file_set_view(file, record.payload_offset as i64, ft) == ffi::MPI_SUCCESS as i32,
            "named read view",
        )?;
        let read = ffi::file_read_all(file, &mut packed);
        agree_phase(
            comm,
            matches!(read, Ok(n) if n == packed.len()),
            "named payload read",
        )?;
        prepared(
            comm,
            prepare_physical_values(&view, &packed),
            "named physical staging",
        )
    })();
    let cleanup = finish_resources(
        comm,
        duplicate,
        crate::mpi_io::FileGuard { raw: file },
        datatype,
    );
    let cleanup = if inject_post_cleanup_failure && comm.rank() == 0 {
        Some(IoError::Native {
            operation: "named post-cleanup result",
            code: -1,
        })
    } else {
        cleanup
    };
    let cleanup = crate::mpi_io::aggregate_cleanup_result(comm, cleanup);
    let physical = result?;
    if let Some(error) = cleanup {
        return Err(error.into());
    }
    view.as_mut_slice().copy_from_slice(&physical);
    Ok(())
}

#[cfg(test)]
pub(crate) fn read_with_cleanup_failure<T: IoElement, const N: usize, const M: usize>(
    path: &Path,
    name: &str,
    view: PencilArrayViewMut<'_, T, N, M>,
) -> Result<(), NamedIoError> {
    read_named(path, name, view, true)
}

#[cfg(test)]
pub(crate) fn write_with_commit_uncertainty<T: IoElement, const N: usize, const M: usize>(
    path: &Path,
    name: &str,
    view: PencilArrayView<'_, T, N, M>,
) -> Result<(), NamedIoError> {
    write_named(path, name, view, false, true)
}
