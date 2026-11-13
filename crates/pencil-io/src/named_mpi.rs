//! Append-only v2: all integers are little-endian u64. The 32-byte container
//! header is magic/version/header-size/reserved. Each 72-byte record header is
//! magic/version/name-length/metadata-length/payload-length/total-length/type/
//! width/reserved, followed by UTF-8 name, metadata, canonical payload, and a
//! commit word. Metadata is spatial-rank/extra-rank/spatial-dims/extra-dims.
//! Header and payload are flushed before the independent trailing commit word.
//! No mutable global index is needed to recover the committed prefix.

pub(crate) mod session;

use std::ffi::CString;
use std::path::Path;

use mpi::collective::{CommunicatorCollectives, Root, SystemOperation};
use mpi::topology::Communicator;
use mpi::traits::AsRaw;
use pencil_array::{PencilArrayView, PencilArrayViewMut};

use crate::ffi;
use crate::format::{IoElement, ReadTarget, element_count, pack_view};
use crate::mpi_io::agree_phase;
use crate::mpi_io::{
    DatatypeGuard, agree_mpi_preparation, build_layout, duplicate_comm, finish_resources,
    path_bytes,
};
use crate::options::{InfoGuard, MpiIoMode, MpiIoOptions, agree_options};
use crate::{COMMIT_MARKER, IoError, MAX_PROTOCOL_RANK, NamedIoError, ScalarType};

const MAGIC: &[u8; 8] = b"PIONAM02";
const RECORD_MAGIC: &[u8; 8] = b"PIOREC02";
const VERSION: u64 = 2;
const CONTAINER_BYTES: usize = 32;
const RECORD_HEADER_BYTES: usize = 72;
const MAX_NAME: usize = 1024;
const OP_WRITE: u64 = 0x101;
const OP_READ: u64 = 0x102;
const OP_APPEND: u64 = 0x103;
const OP_WRITE_OPTIONS: u64 = 51;
const OP_APPEND_OPTIONS: u64 = 52;
const OP_READ_OPTIONS: u64 = 53;

fn u64at(b: &[u8], p: usize) -> Option<u64> {
    b.get(p..p + 8)
        .and_then(|x| x.try_into().ok())
        .map(u64::from_le_bytes)
}
fn put(out: &mut Vec<u8>, x: u64) {
    out.extend_from_slice(&x.to_le_bytes());
}

pub(crate) fn agree_name<C: CommunicatorCollectives>(
    comm: &C,
    name: &str,
) -> Result<(), NamedIoError> {
    name_bytes(comm, name)
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
    let path = path_bytes(p.as_ref())?;
    let len = path
        .len()
        .checked_add(1)
        .ok_or(IoError::SizeLimit { what: "path" })?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(len)
        .map_err(|_| IoError::AllocationFailed { requested: len })?;
    bytes.extend_from_slice(path);
    bytes.push(0);
    CString::from_vec_with_nul(bytes).map_err(|_| IoError::InvalidPath)
}
fn open<C: CommunicatorCollectives + Communicator>(
    comm: &C,
    raw: ffi::MPI_Comm,
    p: &CString,
    mode: u8,
    info: Option<&InfoGuard>,
) -> Result<ffi::MPI_File, NamedIoError> {
    let r = if let Some(info) = info {
        if mode == 3 {
            ffi::file_create_readwrite(raw, p, info.raw)
        } else if mode == 0 {
            ffi::file_open_with_info(raw, p, true, info.raw)
        } else if mode == 1 {
            ffi::file_open_with_info(raw, p, false, info.raw)
        } else {
            ffi::file_open_update_with_info(raw, p, info.raw)
        }
    } else if mode == 0 {
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
pub(crate) struct Rec {
    pub(crate) name: Vec<u8>,
    pub(crate) typ: u64,
    pub(crate) width: usize,
    pub(crate) global: Vec<usize>,
    pub(crate) extra: Vec<usize>,
    pub(crate) payload_offset: usize,
}

#[cfg(test)]
thread_local! {
    static SCAN_ALLREDUCES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static SCAN_BROADCASTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn count_scan_allreduce() {
    SCAN_ALLREDUCES.with(|count| count.set(count.get() + 1));
}

#[cfg(test)]
fn count_scan_broadcast() {
    SCAN_BROADCASTS.with(|count| count.set(count.get() + 1));
}

#[cfg(test)]
pub(crate) fn test_scan_collectives() -> (usize, usize) {
    (
        SCAN_ALLREDUCES.with(std::cell::Cell::get),
        SCAN_BROADCASTS.with(std::cell::Cell::get),
    )
}

/// Test-only direct scan of a named container, excluding descriptor, open, and
/// cleanup collectives from the reported counts.
#[cfg(test)]
pub(crate) struct TestScanned {
    pub(crate) records: Vec<Rec>,
    pub(crate) tail: usize,
    pub(crate) size: usize,
}

#[cfg(test)]
pub(crate) struct TestScan {
    pub(crate) scanned: Result<TestScanned, NamedIoError>,
    pub(crate) allreduces: usize,
    pub(crate) broadcasts: usize,
    pub(crate) independent_reads: usize,
    pub(crate) collective_reads: usize,
}

#[cfg(test)]
pub(crate) fn scan_path_for_test(
    comm: &mpi::topology::CartesianCommunicator,
    path: &Path,
    max_records: usize,
) -> Result<TestScan, NamedIoError> {
    let pc = path_c(path)?;
    let duplicate = duplicate_comm(comm)?;
    let file = match open(comm, duplicate.raw, &pc, 1, None) {
        Ok(file) => file,
        Err(error) => {
            crate::mpi_io::finish_comm(comm, duplicate);
            return Err(error);
        }
    };
    let result: Result<TestScan, NamedIoError> = (|| {
        agree_phase(
            comm,
            ffi::file_set_errors_return(file) == ffi::MPI_SUCCESS as i32,
            "named file error-handler setup",
        )?;
        let (allreduces, broadcasts) = test_scan_collectives();
        let at_calls = ffi::test_at_calls();
        let scanned = scan_file_bounded(comm, file, max_records);
        let (allreduces_after, broadcasts_after) = test_scan_collectives();
        let at_calls_after = ffi::test_at_calls();
        Ok(TestScan {
            scanned: scanned.map(|(records, tail, size)| TestScanned {
                records,
                tail,
                size,
            }),
            allreduces: allreduces_after - allreduces,
            broadcasts: broadcasts_after - broadcasts,
            independent_reads: at_calls_after[3] - at_calls[3],
            collective_reads: at_calls_after[2] - at_calls[2],
        })
    })();
    finish_resources(
        comm,
        duplicate,
        crate::mpi_io::FileGuard { raw: file },
        None,
    );
    let scan = result?;
    Ok(scan)
}
const CATALOG_CHUNK_BYTES: usize = 1 << 30;
const REPORT_BYTES: usize = 40;

const STATUS_CONTAINER_HEADER: u64 = 1;
const STATUS_CONTAINER_SIZE: u64 = 2;
const STATUS_RECORD_LIMIT: u64 = 3;
const STATUS_METADATA_READ: u64 = 4;
const STATUS_ALLOCATION: u64 = 5;
const STATUS_DIMENSIONS_ALLOCATION: u64 = 6;
const STATUS_INDEX_ALLOCATION: u64 = 7;
const STATUS_CATALOG_BYTES: u64 = 8;
const STATUS_CATALOG_ENCODING: u64 = 9;

#[cfg(test)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScanFault {
    None,
    BodyAllocation,
    DimensionsAllocation,
    IndexAllocation,
    EncodeAllocation,
    ReportBroadcast,
    ReportValidation,
    ReceiveAllocation,
    Decode,
    ChunkBroadcast(usize),
}

#[cfg(test)]
thread_local! {
    static SCAN_FAULT: std::cell::Cell<ScanFault> = const { std::cell::Cell::new(ScanFault::None) };
    static SCAN_CHUNK_BYTES: std::cell::Cell<usize> =
        const { std::cell::Cell::new(CATALOG_CHUNK_BYTES) };
}

#[cfg(test)]
pub(crate) fn set_scan_fault(fault: ScanFault) {
    SCAN_FAULT.with(|current| current.set(fault));
}

#[cfg(test)]
pub(crate) fn set_catalog_chunk_bytes(bytes: usize) {
    SCAN_CHUNK_BYTES.with(|current| current.set(bytes));
}

#[cfg(test)]
fn scan_fault(fault: ScanFault) -> bool {
    SCAN_FAULT.with(|current| current.get() == fault)
}

fn catalog_chunk_bytes() -> usize {
    #[cfg(test)]
    {
        SCAN_CHUNK_BYTES.with(std::cell::Cell::get)
    }
    #[cfg(not(test))]
    {
        CATALOG_CHUNK_BYTES
    }
}

fn scan_agree<C: CommunicatorCollectives>(
    comm: &C,
    local_ok: bool,
    phase: &'static str,
) -> Result<(), NamedIoError> {
    #[cfg(test)]
    count_scan_allreduce();
    agree_phase(comm, local_ok, phase).map_err(NamedIoError::Io)
}

fn root_read(file: ffi::MPI_File, offset: usize, bytes: &mut [u8]) -> Result<(), u64> {
    let offset = i64::try_from(offset).map_err(|_| STATUS_METADATA_READ)?;
    match ffi::file_read_at(file, offset, bytes, false) {
        Ok(read) if read == bytes.len() => Ok(()),
        _ => Err(STATUS_METADATA_READ),
    }
}

/// Rank zero scans metadata with independent reads. Payload bytes never enter
/// its buffers.
fn scan_root(
    file: ffi::MPI_File,
    size: usize,
    max_records: usize,
) -> Result<(Vec<Rec>, usize), u64> {
    if size < CONTAINER_BYTES {
        return Err(STATUS_CONTAINER_HEADER);
    }
    let mut header = [0u8; CONTAINER_BYTES];
    root_read(file, 0, &mut header)?;
    if &header[..8] != MAGIC
        || u64at(&header, 8) != Some(VERSION)
        || u64at(&header, 16) != Some(CONTAINER_BYTES as u64)
        || u64at(&header, 24) != Some(0)
    {
        return Err(STATUS_CONTAINER_HEADER);
    }
    let mut p = CONTAINER_BYTES;
    let mut out: Vec<Rec> = Vec::new();
    let mut names: std::collections::HashSet<Vec<u8>> = std::collections::HashSet::new();
    while p < size {
        if out.len() == max_records {
            return Err(STATUS_RECORD_LIMIT);
        }
        if size - p < RECORD_HEADER_BYTES {
            break;
        }
        let mut rh = [0u8; RECORD_HEADER_BYTES];
        root_read(file, p, &mut rh)?;
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
        let mut body = Vec::new();
        let body_allocated = body.try_reserve_exact(body_len).is_ok();
        #[cfg(test)]
        let body_allocated = body_allocated && !scan_fault(ScanFault::BodyAllocation);
        if !body_allocated {
            return Err(STATUS_ALLOCATION);
        }
        body.resize(body_len, 0);
        root_read(file, p + RECORD_HEADER_BYTES, &mut body)?;
        let name = &body[..nl];
        if std::str::from_utf8(name).is_err() || name.contains(&0) || names.contains(name) {
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
        let dims_allocated = dims.try_reserve_exact(rank).is_ok();
        #[cfg(test)]
        let dims_allocated = dims_allocated && !scan_fault(ScanFault::DimensionsAllocation);
        if !dims_allocated {
            return Err(STATUS_DIMENSIONS_ALLOCATION);
        }
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
        if ScalarType::decode(typ, width as u64).is_none()
            || elements.checked_mul(width) != Some(payload)
        {
            break;
        }
        let mut marker = [0u8; 8];
        root_read(file, end - 8, &mut marker)?;
        if u64::from_le_bytes(marker) != COMMIT_MARKER {
            break;
        }
        let mut extra = Vec::new();
        let mut key = Vec::new();
        let mut set_key = Vec::new();
        let allocated = extra.try_reserve_exact(er).is_ok()
            && key.try_reserve_exact(nl).is_ok()
            && set_key.try_reserve_exact(nl).is_ok()
            && out.try_reserve(1).is_ok()
            && names.try_reserve(1).is_ok();
        #[cfg(test)]
        let allocated = allocated && !scan_fault(ScanFault::IndexAllocation);
        if !allocated {
            return Err(STATUS_INDEX_ALLOCATION);
        }
        extra.extend_from_slice(&dims[nr..]);
        set_key.extend_from_slice(name);
        dims.truncate(nr);
        key.extend_from_slice(name);
        names.insert(set_key);
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
    Ok((out, p))
}

fn encode_catalog(records: &[Rec]) -> Result<Vec<u8>, u64> {
    let mut catalog_len = 0u64;
    for record in records {
        let rank = record
            .global
            .len()
            .checked_add(record.extra.len())
            .ok_or(STATUS_CATALOG_BYTES)?;
        let fields = 48u64
            .checked_add(record.name.len() as u64)
            .and_then(|x| x.checked_add((rank as u64).checked_mul(8)?))
            .ok_or(STATUS_CATALOG_BYTES)?;
        catalog_len = catalog_len
            .checked_add(fields)
            .ok_or(STATUS_CATALOG_BYTES)?;
    }
    let catalog_len = usize::try_from(catalog_len).map_err(|_| STATUS_CATALOG_BYTES)?;
    let mut out = Vec::new();
    let prepared = out.try_reserve_exact(catalog_len).is_ok();
    #[cfg(test)]
    let prepared = prepared && !scan_fault(ScanFault::EncodeAllocation);
    if !prepared {
        return Err(STATUS_CATALOG_ENCODING);
    }
    for record in records {
        put(&mut out, record.name.len() as u64);
        put(&mut out, record.typ);
        put(&mut out, record.width as u64);
        put(&mut out, record.global.len() as u64);
        put(&mut out, record.extra.len() as u64);
        put(&mut out, record.payload_offset as u64);
        out.extend_from_slice(&record.name);
        for &dim in record.global.iter().chain(record.extra.iter()) {
            put(&mut out, dim as u64);
        }
    }
    debug_assert_eq!(out.len(), catalog_len);
    Ok(out)
}

fn catalog_word(bytes: &[u8], at: &mut usize) -> Option<u64> {
    let word = u64at(bytes, *at)?;
    *at = at.checked_add(8)?;
    Some(word)
}

fn catalog_slice<'a>(bytes: &'a [u8], at: &mut usize, len: usize) -> Option<&'a [u8]> {
    let end = at.checked_add(len)?;
    let slice = bytes.get(*at..end)?;
    *at = end;
    Some(slice)
}

fn decode_catalog(bytes: &[u8], count: usize, tail: usize, size: usize) -> Option<Vec<Rec>> {
    let mut out = Vec::new();
    out.try_reserve_exact(count).ok()?;
    let mut at = 0usize;
    let mut start = CONTAINER_BYTES;
    for _ in 0..count {
        let name_len = usize::try_from(catalog_word(bytes, &mut at)?).ok()?;
        if name_len == 0 || name_len > MAX_NAME {
            return None;
        }
        let typ = catalog_word(bytes, &mut at)?;
        let width = usize::try_from(catalog_word(bytes, &mut at)?).ok()?;
        let spatial = usize::try_from(catalog_word(bytes, &mut at)?).ok()?;
        let extra_rank = usize::try_from(catalog_word(bytes, &mut at)?).ok()?;
        let payload_offset = usize::try_from(catalog_word(bytes, &mut at)?).ok()?;
        if spatial == 0 || spatial > MAX_PROTOCOL_RANK || extra_rank > MAX_PROTOCOL_RANK {
            return None;
        }
        let rank = spatial.checked_add(extra_rank)?;
        if rank > MAX_PROTOCOL_RANK {
            return None;
        }
        let name = catalog_slice(bytes, &mut at, name_len)?;
        if std::str::from_utf8(name).is_err() || name.contains(&0) {
            return None;
        }
        let metadata_len = 16usize.checked_add(rank.checked_mul(8)?)?;
        let header_len = RECORD_HEADER_BYTES
            .checked_add(name_len)?
            .checked_add(metadata_len)?;
        if payload_offset.checked_sub(header_len)? != start {
            return None;
        }
        let mut global = Vec::new();
        global.try_reserve_exact(spatial).ok()?;
        let mut extra = Vec::new();
        extra.try_reserve_exact(extra_rank).ok()?;
        let mut elements = 1usize;
        for i in 0..rank {
            let dim = usize::try_from(catalog_word(bytes, &mut at)?).ok()?;
            elements = elements.checked_mul(dim)?;
            if i < spatial {
                global.push(dim);
            } else {
                extra.push(dim);
            }
        }
        let payload = elements.checked_mul(width)?;
        let end = payload_offset.checked_add(payload)?.checked_add(8)?;
        if end > size || i64::try_from(end).is_err() {
            return None;
        }
        ScalarType::decode(typ, width as u64)?;
        let mut owned = Vec::new();
        owned.try_reserve_exact(name_len).ok()?;
        owned.extend_from_slice(name);
        out.push(Rec {
            name: owned,
            typ,
            width,
            global,
            extra,
            payload_offset,
        });
        start = end;
    }
    if at != bytes.len() || start != tail {
        return None;
    }
    Some(out)
}

#[cfg(test)]
pub(crate) fn encode_catalog_for_test(records: &[Rec]) -> Result<Vec<u8>, u64> {
    encode_catalog(records)
}

#[cfg(test)]
pub(crate) fn decode_catalog_for_test(
    bytes: &[u8],
    count: usize,
    tail: usize,
    size: usize,
) -> Option<Vec<Rec>> {
    decode_catalog(bytes, count, tail, size)
}

fn encode_report(
    status: u64,
    catalog_len: u64,
    count: u64,
    tail: u64,
    size: u64,
) -> [u8; REPORT_BYTES] {
    let mut report = [0u8; REPORT_BYTES];
    for (index, value) in [status, catalog_len, count, tail, size]
        .into_iter()
        .enumerate()
    {
        report[index * 8..index * 8 + 8].copy_from_slice(&value.to_le_bytes());
    }
    report
}

fn report_fields(report: &[u8; REPORT_BYTES]) -> [u64; 5] {
    std::array::from_fn(|index| {
        u64::from_le_bytes(report[index * 8..index * 8 + 8].try_into().unwrap())
    })
}

fn failure_status(status: u64) -> Option<NamedIoError> {
    Some(match status {
        STATUS_CONTAINER_HEADER => NamedIoError::Io(IoError::InvalidFile {
            reason: "named container header",
        }),
        STATUS_CONTAINER_SIZE => NamedIoError::Io(IoError::SizeLimit {
            what: "container size",
        }),
        STATUS_RECORD_LIMIT => NamedIoError::Io(IoError::SizeLimit {
            what: "catalog dataset count",
        }),
        STATUS_METADATA_READ => NamedIoError::Io(IoError::CollectivePrecondition {
            phase: "named metadata read",
        }),
        STATUS_ALLOCATION => NamedIoError::Io(IoError::CollectivePrecondition {
            phase: "named allocation",
        }),
        STATUS_DIMENSIONS_ALLOCATION => NamedIoError::Io(IoError::CollectivePrecondition {
            phase: "named dimensions allocation",
        }),
        STATUS_INDEX_ALLOCATION => NamedIoError::Io(IoError::CollectivePrecondition {
            phase: "named record index allocation",
        }),
        STATUS_CATALOG_BYTES => NamedIoError::Io(IoError::SizeLimit {
            what: "named catalog bytes",
        }),
        STATUS_CATALOG_ENCODING => NamedIoError::Io(IoError::CollectivePrecondition {
            phase: "named catalog encoding",
        }),
        _ => return None,
    })
}

/// Scan only fixed headers, names, metadata, and commit markers. Rank zero
/// reads them independently; the parsed catalog is broadcast once for catalogs
/// up to 1 GiB and once per additional 1 GiB chunk.
fn scan_file<C: CommunicatorCollectives>(
    comm: &C,
    file: ffi::MPI_File,
) -> Result<(Vec<Rec>, usize, usize), NamedIoError> {
    scan_file_bounded(comm, file, usize::MAX)
}

/// Scan metadata with collectives independent of the record count. Rank zero
/// scans metadata with independent reads; payload bytes are skipped. The
/// encoded catalog is broadcast in chunks of at most 1 GiB, with an agreement
/// after every chunk. An empty catalog uses one zero-count broadcast. The scan
/// performs nine collectives for catalogs up to 1 GiB, then two more for each
/// additional chunk. This excludes caller-side collectives such as opening,
/// closing, resetting views, and transferring payloads.
pub(crate) fn scan_file_bounded<C: CommunicatorCollectives>(
    comm: &C,
    file: ffi::MPI_File,
    max_records: usize,
) -> Result<(Vec<Rec>, usize, usize), NamedIoError> {
    let native_size = ffi::file_get_size(file);
    scan_agree(comm, native_size.is_ok(), "MPI_File_get_size")?;
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
    #[cfg(test)]
    {
        count_scan_allreduce();
        count_scan_allreduce();
    }
    if min_size != max_size {
        return Err(NamedIoError::Io(IoError::CollectiveDescriptorMismatch));
    }
    let size = usize::try_from(local_size).ok();

    let mut report = encode_report(0, 0, 0, 0, 0);
    let mut root_catalog: Option<Vec<u8>> = None;
    let mut root_records: Option<Vec<Rec>> = None;
    if comm.rank() == 0 {
        let outcome = match size {
            Some(size) => scan_root(file, size, max_records).and_then(|(records, tail)| {
                encode_catalog(&records).map(|catalog| (records, tail, catalog))
            }),
            None => Err(STATUS_CONTAINER_SIZE),
        };
        match outcome {
            Ok((records, tail, catalog)) => {
                report = encode_report(
                    0,
                    catalog.len() as u64,
                    records.len() as u64,
                    tail as u64,
                    size.unwrap() as u64,
                );
                root_catalog = Some(catalog);
                root_records = Some(records);
            }
            Err(status) => report = encode_report(status, 0, 0, 0, 0),
        }
    }
    let report_code = ffi::bcast_bytes(comm.as_raw(), 0, &mut report);
    #[cfg(test)]
    let report_code = if scan_fault(ScanFault::ReportBroadcast) {
        mpi::ffi::MPI_ERR_OTHER as i32
    } else {
        report_code
    };
    #[cfg(test)]
    count_scan_broadcast();
    scan_agree(
        comm,
        report_code == ffi::MPI_SUCCESS as i32,
        "named catalog report broadcast",
    )?;

    let fields = report_fields(&report);
    let status = fields[0];
    if status != 0 {
        if let Some(error) = failure_status(status) {
            return Err(error);
        }
        scan_agree(comm, false, "named catalog preparation")?;
        unreachable!("failed preparation agreement returned success");
    }

    let catalog_len = usize::try_from(fields[1]).ok();
    let count = usize::try_from(fields[2]).ok();
    let tail = usize::try_from(fields[3]).ok();
    let file_size = usize::try_from(fields[4]).ok();
    let report_valid = matches!(
        (size, catalog_len, count, tail, file_size),
        (Some(size), Some(catalog_len), Some(count), Some(tail), Some(file_size))
            if file_size == size
                && (CONTAINER_BYTES..=size).contains(&tail)
                && count <= max_records
                && if count == 0 {
                    catalog_len == 0 && tail == CONTAINER_BYTES
                } else {
                    count
                        .checked_mul(48)
                        .is_some_and(|minimum| catalog_len >= minimum)
                }
    );
    #[cfg(test)]
    let report_valid = report_valid && !scan_fault(ScanFault::ReportValidation);
    let catalog_len = catalog_len.unwrap_or(0);
    let mut receive = Vec::new();
    let receive_prepared = if comm.rank() == 0 {
        true
    } else {
        receive.try_reserve_exact(catalog_len).is_ok()
    };
    #[cfg(test)]
    let receive_prepared =
        receive_prepared && (comm.rank() == 0 || !scan_fault(ScanFault::ReceiveAllocation));
    scan_agree(
        comm,
        report_valid && receive_prepared,
        "named catalog preparation",
    )?;
    if comm.rank() != 0 {
        receive.resize(catalog_len, 0);
    }

    let count = count.expect("validated catalog count");
    let tail = tail.expect("validated catalog tail");
    let chunk = catalog_chunk_bytes();
    let chunks = if catalog_len == 0 {
        1
    } else {
        (catalog_len - 1) / chunk + 1
    };
    for index in 1..=chunks {
        let start = (index - 1) * chunk;
        let end = start.saturating_add(chunk).min(catalog_len);
        let code = if comm.rank() == 0 {
            let catalog = root_catalog.as_mut().expect("root catalog");
            ffi::bcast_bytes(comm.as_raw(), 0, &mut catalog[start..end])
        } else {
            ffi::bcast_bytes(comm.as_raw(), 0, &mut receive[start..end])
        };
        #[cfg(test)]
        let code = if scan_fault(ScanFault::ChunkBroadcast(index)) {
            mpi::ffi::MPI_ERR_OTHER as i32
        } else {
            code
        };
        #[cfg(test)]
        count_scan_broadcast();
        scan_agree(
            comm,
            code == ffi::MPI_SUCCESS as i32,
            "named catalog broadcast",
        )?;
    }
    drop(root_catalog);

    let decoded = if comm.rank() == 0 {
        None
    } else {
        Some(decode_catalog(&receive, count, tail, size.unwrap_or(0)))
    };
    let decode_ok = decoded.as_ref().is_none_or(|catalog| catalog.is_some());
    #[cfg(test)]
    let decode_ok = decode_ok && (comm.rank() == 0 || !scan_fault(ScanFault::Decode));
    scan_agree(comm, decode_ok, "named catalog decode")?;
    drop(receive);
    let records = match decoded {
        Some(Some(records)) => records,
        Some(None) => unreachable!("failed decode agreement returned success"),
        None => root_records.expect("root scan records"),
    };
    Ok((records, tail, size.unwrap_or(0)))
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
/// dimensions must fit native MPI integers. Each record's local payload is limited
/// to 2147483647 bytes/rank (`local_elements * scalar_width`; for `f64`, up to
/// 268435455 elements), which is not a whole-container cap. If any rank exceeds
/// it, every rank returns `NamedIoError::Io(IoError::SizeLimit { what: "MPI local
/// byte count" })` before packing or staging that record.
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
    write_named(
        path.as_ref(),
        name.as_ref(),
        view,
        false,
        false,
        None,
        false,
    )
}

/// Create a named MPI container with explicit native controls. Applies the same
/// per-record MPI local payload limit as [`write_mpi_named`].
pub fn write_mpi_named_with_options<
    P: AsRef<Path>,
    S: AsRef<str>,
    T: IoElement,
    const N: usize,
    const M: usize,
>(
    path: P,
    name: S,
    view: PencilArrayView<'_, T, N, M>,
    options: &MpiIoOptions,
) -> Result<(), NamedIoError> {
    write_named(
        path.as_ref(),
        name.as_ref(),
        view,
        false,
        false,
        Some(options),
        false,
    )
}

/// Appends an independently committed record without modifying the committed prefix.
/// Each appended record is independently limited to 2147483647 bytes/rank
/// (`local_elements * scalar_width`; for `f64`, up to 268435455 elements), not
/// the whole container; every rank returns the same wrapped `SizeLimit` as
/// [`write_mpi_named`] before packing or staging that record. Duplicate keys and
/// malformed or incomplete tails are rejected before writing.
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
    write_named(path.as_ref(), name.as_ref(), view, true, false, None, false)
}

/// Append a new name with native controls. Same-file writers require external
/// serialization. Applies the same per-record MPI local payload limit as
/// [`append_mpi_named`].
pub fn append_mpi_named_with_options<
    P: AsRef<Path>,
    S: AsRef<str>,
    T: IoElement,
    const N: usize,
    const M: usize,
>(
    path: P,
    name: S,
    view: PencilArrayView<'_, T, N, M>,
    options: &MpiIoOptions,
) -> Result<(), NamedIoError> {
    write_named(
        path.as_ref(),
        name.as_ref(),
        view,
        true,
        false,
        Some(options),
        false,
    )
}

/// Writes a prepared named collection after its caller agreed the options.
pub(crate) fn write_named_target<T: IoElement, const N: usize, const M: usize>(
    path: &Path,
    name: &str,
    view: PencilArrayView<'_, T, N, M>,
    append: bool,
    options: &MpiIoOptions,
) -> Result<(), NamedIoError> {
    write_named(path, name, view, append, false, Some(options), true)
}

fn write_named<T: IoElement, const N: usize, const M: usize>(
    path: &Path,
    name: &str,
    view: PencilArrayView<'_, T, N, M>,
    append: bool,
    inject_post_cleanup_failure: bool,
    options: Option<&MpiIoOptions>,
    agreed: bool,
) -> Result<(), NamedIoError> {
    let comm = view.pencil().topology().communicator();
    if !agreed {
        crate::mpi_io::descriptor_agreement(
            comm,
            path,
            if options.is_some() {
                if append {
                    OP_APPEND_OPTIONS
                } else {
                    OP_WRITE_OPTIONS
                }
            } else if append {
                OP_APPEND
            } else {
                OP_WRITE
            },
            view.pencil().global_shape(),
            view.extra_shape().dimensions(),
            view.pencil().topology().process_grid(),
            view.pencil().decomposition(),
            view.pencil().permutation().axes(),
            T::CODE,
            T::WIDTH,
        )?;
    }
    if !agreed {
        name_bytes(comm, name)?;
    }
    if let Some(options) = options
        && !agreed
    {
        agree_options(comm, options, &[])?;
    }
    let pc = prepared(comm, path_c(path), "named path preparation")?;
    let layout_result = build_layout(
        view.pencil().global_shape(),
        view.extra_shape().dimensions(),
        &view.local_spatial_shape(),
        view.pencil().local_ranges(),
        view.len(),
        T::WIDTH,
    );
    if let Err(agreement) = agree_mpi_preparation(
        comm,
        layout_result.as_ref().err(),
        "named layout preparation",
    ) {
        return Err(NamedIoError::Io(agreement));
    }
    let layout = layout_result.map_err(NamedIoError::Io)?;
    let packed = prepared(comm, pack_view(&view), "named payload packing")?;
    // Build bounded metadata before opening. Append adjusts only the absolute end.
    let record_result = record(name, &view, 0);
    agree_phase(comm, record_result.is_ok(), "named record preparation")?;
    let (head, record_len) = record_result?;
    if options.is_some() && !agreed {
        crate::options::agree_decomposition(view.pencil())?;
    }
    let info = options
        .map(|options| InfoGuard::new(comm, options))
        .transpose()?;
    let duplicate = duplicate_comm(comm)?;
    let file = match open(
        comm,
        duplicate.raw,
        &pc,
        if append { 2 } else { 0 },
        info.as_ref(),
    ) {
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
        if let Err(error) = crate::mpi_io::write_payload_and_commit(
            comm,
            file,
            info.as_ref().map_or_else(ffi::info_null, |i| i.raw),
            options.map_or(MpiIoMode::Collective, |o| o.mode),
            (offset + head.len()) as i64,
            &packed,
            ft,
            None,
            (end - 8) as i64,
            crate::mpi_io::WritePhases {
                payload_view: "named payload view",
                payload_write: "named payload write",
                payload_flush: "named payload flush",
                marker_view: "named marker view",
                marker_write: "named metadata write",
                marker_flush: "named marker flush",
            },
            &mut marker_attempted,
        ) {
            return Err(NamedIoError::Io(error.agreement));
        }
        Ok(())
    })();
    finish_resources(
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
        None
    };
    drop(info);
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
/// after staging, native cleanup, and final collective success. Each record is
/// subject to the 2147483647 bytes/rank local payload limit described by
/// [`write_mpi_named`]; a rejected read leaves the destination unchanged.
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
    read_named(path.as_ref(), name.as_ref(), view, false, None)
}

/// Read a named MPI payload with explicit native controls. Applies the same
/// per-record MPI local payload limit and destination guarantee as
/// [`read_mpi_named`].
pub fn read_mpi_named_with_options<
    P: AsRef<Path>,
    S: AsRef<str>,
    T: IoElement,
    const N: usize,
    const M: usize,
>(
    path: P,
    name: S,
    view: PencilArrayViewMut<'_, T, N, M>,
    options: &MpiIoOptions,
) -> Result<(), NamedIoError> {
    read_named(path.as_ref(), name.as_ref(), view, false, Some(options))
}

/// Reads into a single view or a collection of member views using one byte
/// payload. Collection callers prepare the target and its staging first.
pub(crate) fn read_named_target<T: IoElement, const N: usize, const M: usize>(
    path: &Path,
    name: &str,
    target: ReadTarget<'_, '_, T, N, M>,
    options: Option<&MpiIoOptions>,
) -> Result<(), NamedIoError> {
    read_named_inner(path, name, target, false, options, true)
}

fn read_named<T: IoElement, const N: usize, const M: usize>(
    path: &Path,
    name: &str,
    view: PencilArrayViewMut<'_, T, N, M>,
    inject_post_cleanup_failure: bool,
    options: Option<&MpiIoOptions>,
) -> Result<(), NamedIoError> {
    read_named_inner(
        path,
        name,
        ReadTarget::Single(view),
        inject_post_cleanup_failure,
        options,
        false,
    )
}

fn read_named_inner<T: IoElement, const N: usize, const M: usize>(
    path: &Path,
    name: &str,
    mut target: ReadTarget<'_, '_, T, N, M>,
    inject_post_cleanup_failure: bool,
    options: Option<&MpiIoOptions>,
    agreed: bool,
) -> Result<(), NamedIoError> {
    // Take the reserved staging out before borrowing the target's metadata.
    let mut packed = target.take_staging();
    let comm = target.pencil().topology().communicator();
    if !agreed {
        crate::mpi_io::descriptor_agreement(
            comm,
            path,
            if options.is_some() {
                OP_READ_OPTIONS
            } else {
                OP_READ
            },
            target.pencil().global_shape(),
            target.extra(),
            target.pencil().topology().process_grid(),
            target.pencil().decomposition(),
            target.pencil().permutation().axes(),
            T::CODE,
            T::WIDTH,
        )?;
    }
    if !agreed {
        name_bytes(comm, name)?;
    }
    if let Some(options) = options
        && !agreed
    {
        agree_options(comm, options, &[])?;
    }
    let pc = prepared(comm, path_c(path), "named path preparation")?;
    let layout_result = build_layout(
        target.pencil().global_shape(),
        target.extra(),
        &target.local_spatial_shape(),
        target.pencil().local_ranges(),
        target.len(),
        T::WIDTH,
    );
    if let Err(agreement) =
        agree_mpi_preparation(comm, layout_result.as_ref().err(), "named read layout")
    {
        return Err(NamedIoError::Io(agreement));
    }
    let layout = layout_result.map_err(NamedIoError::Io)?;
    let size = target
        .len()
        .checked_mul(T::WIDTH)
        .ok_or(IoError::SizeLimit {
            what: "named read buffer",
        });
    let size = prepared(comm, size, "named read size")?;
    let allocation = packed.try_reserve_exact(size);
    agree_phase(comm, allocation.is_ok(), "named allocation")?;
    packed.resize(size, 0);
    if options.is_some() && !agreed {
        crate::options::agree_decomposition(target.pencil())?;
    }
    let info = options
        .map(|options| InfoGuard::new(comm, options))
        .transpose()?;
    let duplicate = duplicate_comm(comm)?;
    let file = match open(comm, duplicate.raw, &pc, 1, info.as_ref()) {
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
            || record.global != target.pencil().global_shape()
            || record.extra != target.extra()
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
        if let Err(error) = crate::mpi_io::read_payload(
            comm,
            file,
            info.as_ref().map_or_else(ffi::info_null, |i| i.raw),
            options.map_or(MpiIoMode::Collective, |o| o.mode),
            record.payload_offset as i64,
            ft,
            &mut packed,
            crate::mpi_io::ReadPhases {
                view: "named read view",
                payload: "named payload read",
            },
        ) {
            return Err(NamedIoError::Io(error.agreement));
        }
        prepared(comm, target.prepare(&packed), "named physical staging")
    })();
    finish_resources(
        comm,
        duplicate,
        crate::mpi_io::FileGuard { raw: file },
        datatype,
    );
    let cleanup = if crate::mpi_io::post_cleanup_failure_injected(inject_post_cleanup_failure)
        && comm.rank() == 0
    {
        Some(IoError::Native {
            operation: "named post-cleanup result",
            code: -1,
        })
    } else {
        None
    };
    drop(info);
    let cleanup = crate::mpi_io::aggregate_cleanup_result(comm, cleanup);
    let plan = result?;
    if let Some(error) = cleanup {
        return Err(error.into());
    }
    target.commit(plan, &packed);
    Ok(())
}

#[cfg(test)]
pub(crate) fn read_with_cleanup_failure<T: IoElement, const N: usize, const M: usize>(
    path: &Path,
    name: &str,
    view: PencilArrayViewMut<'_, T, N, M>,
) -> Result<(), NamedIoError> {
    read_named(path, name, view, true, None)
}

#[cfg(test)]
pub(crate) fn write_with_commit_uncertainty<T: IoElement, const N: usize, const M: usize>(
    path: &Path,
    name: &str,
    view: PencilArrayView<'_, T, N, M>,
) -> Result<(), NamedIoError> {
    write_named(path, name, view, false, true, None, false)
}
