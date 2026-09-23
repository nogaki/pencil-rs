//! Rank-ordered physical blocks, not the canonical logical MPI format.
//! Reads require exactly the writer's grid, rank mapping, decomposition and
//! permutation. Files are create-exclusive; concurrent access is unsupported.
use crate::mpi_io::{
    FileGuard, abort_unrecoverable, aggregate_cleanup_result, agree_phase, collective_state,
    descriptor_agreement, duplicate_comm, finish_comm, finish_resources, path_bytes,
};
use crate::options::{InfoGuard, agree_options};
use crate::{
    DatasetInfo, IoElement, IoError, MAX_HEADER_BYTES, MAX_PROTOCOL_RANK, MpiIoMode, MpiIoOptions,
    ScalarType, ffi,
};
use mpi::{collective::SystemOperation, topology::CartesianCommunicator, traits::*};
use pencil_array::{PencilArrayView, PencilArrayViewMut, partition_range};
use std::{ffi::CString, path::Path};

const MAGIC: &[u8; 8] = b"PNCHUNK3";
const VERSION: u64 = 3;
const NAMESPACE: u64 = 0x504e43484e4b0003;
const PREFIX: usize = 96;
fn invalid() -> IoError {
    IoError::InvalidFile {
        reason: "invalid chunked metadata or writer layout mismatch (repartition unsupported)",
    }
}
fn put(b: &mut Vec<u8>, n: usize) {
    b.extend_from_slice(&(n as u64).to_le_bytes());
}
fn word(b: &[u8], i: usize) -> Result<usize, IoError> {
    let v = b.get(i * 8..i * 8 + 8).ok_or_else(invalid)?;
    usize::try_from(u64::from_le_bytes(v.try_into().unwrap())).map_err(|_| invalid())
}
fn product(v: &[usize]) -> Result<usize, IoError> {
    v.iter()
        .try_fold(1usize, |a, &b| a.checked_mul(b).ok_or_else(invalid))
}
fn buffer(n: usize) -> Result<Vec<u8>, IoError> {
    let mut b = Vec::new();
    b.try_reserve_exact(n)
        .map_err(|_| IoError::AllocationFailed { requested: n })?;
    b.resize(n, 0);
    Ok(b)
}
fn prepared<T>(c: &CartesianCommunicator, r: Result<T, IoError>) -> Result<T, IoError> {
    if let Err(e) = agree_phase(c, r.is_ok(), "chunked preparation") {
        return Err(r.err().unwrap_or(e));
    }
    r
}
// The common two five-u64 reductions are always first, on the original comm.
fn entry(c: &CartesianCommunicator, op: u64) -> Result<(), IoError> {
    let h = [NAMESPACE, op, VERSION, 0, 1];
    let (mut lo, mut hi) = ([0u64; 5], [0u64; 5]);
    c.all_reduce_into(&h, &mut lo, SystemOperation::min());
    c.all_reduce_into(&h, &mut hi, SystemOperation::max());
    if lo != hi {
        return Err(IoError::CollectiveDescriptorMismatch);
    }
    Ok(())
}
fn bytes_agree(c: &CartesianCommunicator, b: &[u8]) -> Result<(), IoError> {
    crate::catalog::agree_bytes(c, b).map_err(|_| IoError::CollectiveDescriptorMismatch)
}

struct Header {
    bytes: Vec<u8>,
    offsets: Vec<usize>,
    counts: Vec<usize>,
    end: usize,
    info: DatasetInfo,
}
// Reconstructing the unique canonical table validates *every* rank, including
// empty ranks. Exact offsets imply header exclusion, no gaps/overlap and coverage.
fn header(
    c: &CartesianCommunicator,
    typ: ScalarType,
    global: &[usize],
    extra: &[usize],
    grid: &[usize],
    decomp: &[usize],
    perm: &[usize],
) -> Result<Header, IoError> {
    let (n, e, m, ranks) = (global.len(), extra.len(), grid.len(), c.size() as usize);
    if n == 0
        || n > MAX_PROTOCOL_RANK
        || n.checked_add(e).ok_or_else(invalid)? > MAX_PROTOCOL_RANK
        || m == 0
        || m > n
        || decomp.len() != m
        || perm.len() != n
        || global.contains(&0)
        || grid.contains(&0)
        || product(grid)? != ranks
    {
        return Err(invalid());
    }
    let layout = c.get_layout();
    if layout
        .dims
        .iter()
        .map(|&x| x as usize)
        .ne(grid.iter().copied())
    {
        return Err(invalid());
    }
    for axes in [decomp, perm] {
        for (i, &a) in axes.iter().enumerate() {
            if a >= n || axes[..i].contains(&a) {
                return Err(invalid());
            }
        }
    }
    let record = m
        .checked_add(n.checked_mul(2).ok_or_else(invalid)?)
        .and_then(|x| x.checked_add(2))
        .ok_or_else(invalid)?;
    let words = 12usize
        .checked_add(n * 2)
        .and_then(|x| x.checked_add(e))
        .and_then(|x| x.checked_add(m * 2))
        .and_then(|x| x.checked_add(ranks.checked_mul(record)?))
        .ok_or_else(invalid)?;
    let len = words.checked_mul(8).ok_or_else(invalid)?;
    if len > MAX_HEADER_BYTES {
        return Err(invalid());
    }
    let total = product(global)?
        .checked_mul(product(extra)?)
        .and_then(|x| x.checked_mul(typ.width()))
        .ok_or_else(invalid)?;
    let end = len.checked_add(total).ok_or_else(invalid)?;
    i64::try_from(end).map_err(|_| invalid())?;
    let mut b = buffer(len)?;
    b.clear();
    b.extend_from_slice(MAGIC);
    for v in [
        VERSION as usize,
        len,
        end,
        typ.code() as usize,
        typ.width(),
        n,
        e,
        m,
        ranks,
        1,
        0,
    ] {
        put(&mut b, v);
    }
    for v in [global, extra, grid, decomp, perm] {
        for &x in v {
            put(&mut b, x);
        }
    }
    let mut offsets = Vec::new();
    let mut counts = Vec::new();
    offsets.try_reserve_exact(ranks).map_err(|_| invalid())?;
    counts.try_reserve_exact(ranks).map_err(|_| invalid())?;
    let mut offset = len;
    for rank in 0..c.size() {
        let coords = c.rank_to_coordinates(rank);
        if coords.len() != m {
            return Err(invalid());
        }
        for &x in &coords {
            put(&mut b, x as usize);
        }
        let mut count = product(extra)?;
        for (axis, &g) in global.iter().enumerate() {
            let range = if let Some(j) = decomp.iter().position(|&a| a == axis) {
                partition_range(g, grid[j], coords[j] as usize).map_err(|_| invalid())?
            } else {
                0..g
            };
            put(&mut b, range.start);
            put(&mut b, range.end);
            count = count.checked_mul(range.len()).ok_or_else(invalid)?;
        }
        put(&mut b, count);
        put(&mut b, offset);
        counts.push(count);
        offsets.push(offset);
        offset = offset
            .checked_add(count.checked_mul(typ.width()).ok_or_else(invalid)?)
            .ok_or_else(invalid)?;
    }
    if offset != end || b.len() != len {
        return Err(invalid());
    }
    let mut provenance = Vec::new();
    for &x in grid.iter().chain(perm) {
        put(&mut provenance, x);
    }
    Ok(Header {
        bytes: b,
        offsets,
        counts,
        end,
        info: DatasetInfo {
            name: None,
            scalar_type: typ,
            global_shape: global.iter().map(|&x| x as u64).collect(),
            extra_shape: extra.iter().map(|&x| x as u64).collect(),
            provenance,
        },
    })
}
fn parse(c: &CartesianCommunicator, b: &[u8], size: usize) -> Result<Header, IoError> {
    if b.len() < PREFIX
        || &b[..8] != MAGIC
        || word(b, 1)? != VERSION as usize
        || word(b, 2)? != b.len()
        || word(b, 3)? != size
    {
        return Err(invalid());
    }
    let typ = ScalarType::decode(word(b, 4)? as u64, word(b, 5)? as u64).ok_or_else(invalid)?;
    let (n, e, m) = (word(b, 6)?, word(b, 7)?, word(b, 8)?);
    if n > MAX_PROTOCOL_RANK || e > MAX_PROTOCOL_RANK || m > MAX_PROTOCOL_RANK {
        return Err(invalid());
    }
    let mut pos = 12;
    let mut take = |len| -> Result<Vec<usize>, IoError> {
        let v = (pos..pos + len).map(|i| word(b, i)).collect();
        pos += len;
        v
    };
    let (g, x, grid, d, p) = (take(n)?, take(e)?, take(m)?, take(m)?, take(n)?);
    let h = header(c, typ, &g, &x, &grid, &d, &p)?;
    if h.bytes != b {
        return Err(invalid());
    }
    Ok(h)
}
fn metadata(
    c: &CartesianCommunicator,
    f: ffi::MPI_File,
    off: usize,
    n: usize,
) -> Result<Vec<u8>, IoError> {
    let mut b = prepared(c, buffer(n))?;
    let slice = if c.rank() == 0 { &mut b[..] } else { &mut [] };
    let len = slice.len();
    let r = ffi::file_read_at_all(f, off as i64, slice);
    agree_phase(c, matches!(r,Ok(x) if x==len), "chunked metadata read")?;
    c.process_at_rank(0).broadcast_into(&mut b);
    Ok(b)
}
fn scan(c: &CartesianCommunicator, f: ffi::MPI_File) -> Result<Header, IoError> {
    let size = prepared(
        c,
        ffi::file_get_size(f)
            .map_err(|_| invalid())
            .and_then(|x| usize::try_from(x).map_err(|_| invalid())),
    )?;
    let prefix = metadata(c, f, 0, PREFIX)?;
    let len = prepared(
        c,
        (|| {
            let n = word(&prefix, 2)?;
            if &prefix[..8] != MAGIC
                || word(&prefix, 1)? != VERSION as usize
                || !(PREFIX..=MAX_HEADER_BYTES).contains(&n)
                || n > size
            {
                return Err(invalid());
            }
            Ok(n)
        })(),
    )?;
    let b = metadata(c, f, 0, len)?;
    prepared(c, parse(c, &b, size))
}
fn with_file<T>(
    c: &CartesianCommunicator,
    path: &Path,
    opts: &MpiIoOptions,
    write: bool,
    inject: bool,
    action: impl FnOnce(ffi::MPI_File) -> Result<T, IoError>,
) -> Result<T, IoError> {
    let p = prepared(
        c,
        path_bytes(path).and_then(|b| CString::new(b).map_err(|_| IoError::InvalidPath)),
    )?;
    bytes_agree(c, p.as_bytes())?;
    agree_options(c, opts, &[])?;
    let info = InfoGuard::new(c, opts)?;
    let dup = duplicate_comm(c)?;
    let opened = ffi::file_open_with_info(dup.raw, &p, write, info.raw);
    let (all, mixed) = collective_state(c, opened.is_ok());
    if mixed {
        abort_unrecoverable(c.as_raw(), "chunked partial open");
    }
    if !all {
        return Err(finish_comm(c, dup).unwrap_or(IoError::Mpi {
            operation: "MPI_File_open",
            code: opened.err().unwrap_or(-1),
        }));
    }
    let file = FileGuard {
        raw: opened.unwrap(),
    };
    let result = agree_phase(
        c,
        ffi::file_set_errors_return(file.raw) == ffi::MPI_SUCCESS as i32,
        "chunked file handler",
    )
    .and_then(|()| action(file.raw));
    let cleanup = finish_resources(c, dup, file, None);
    drop(info);
    let fault = if inject && c.rank() == 0 {
        Some(IoError::Native {
            operation: "chunked post-cleanup test",
            code: -1,
        })
    } else {
        None
    };
    if let Some(e) = aggregate_cleanup_result(
        c,
        cleanup.or(fault).or_else(|| result.as_ref().err().cloned()),
    ) {
        return Err(e);
    }
    result
}
fn view_header<T: IoElement, const N: usize, const M: usize>(
    c: &CartesianCommunicator,
    path: &Path,
    op: u64,
    pencil: &pencil_array::Pencil<N, M>,
    extra: &[usize],
) -> Result<Header, IoError> {
    descriptor_agreement(
        c,
        path,
        op,
        pencil.global_shape(),
        extra,
        pencil.topology().process_grid(),
        pencil.permutation().axes(),
        T::CODE,
        T::WIDTH,
    )?;
    crate::options::agree_decomposition(pencil)?;
    prepared(
        c,
        header(
            c,
            ScalarType::decode(T::CODE, T::WIDTH as u64).ok_or_else(invalid)?,
            pencil.global_shape(),
            extra,
            pencil.topology().process_grid(),
            &pencil.decomposition().map(|a| a.index()),
            &pencil.permutation().axes().map(|a| a.index()),
        ),
    )
}
/// Write canonical little-endian elements in each rank's physical slice order.
/// Options are mandatory; existing files are never replaced. Metadata is bounded
/// to 1 MiB and each rank's payload to `i32::MAX` bytes. All ranks must participate,
/// including in independent mode. Repartitioning on read is unsupported.
pub fn write_mpi_chunked<P: AsRef<Path>, T: IoElement, const N: usize, const M: usize>(
    path: P,
    view: PencilArrayView<'_, T, N, M>,
    options: &MpiIoOptions,
) -> Result<(), IoError> {
    let c = view.pencil().topology().communicator();
    entry(c, 1)?;
    let mut h = view_header::<T, N, M>(
        c,
        path.as_ref(),
        1,
        view.pencil(),
        view.extra_shape().dimensions(),
    )?;
    let rank = c.rank() as usize;
    let payload = prepared(
        c,
        (|| {
            let n = view.len().checked_mul(T::WIDTH).ok_or_else(invalid)?;
            if view.len() != h.counts[rank] || n > i32::MAX as usize {
                return Err(invalid());
            }
            let mut b = buffer(n)?;
            b.clear();
            for &v in view.as_slice() {
                v.encode_le(&mut b);
            }
            Ok(b)
        })(),
    )?;
    h.bytes[80..88].fill(0);
    with_file(c, path.as_ref(), options, true, false, |f| {
        let b = if c.rank() == 0 { &h.bytes[..] } else { &[] };
        agree_phase(
            c,
            matches!(ffi::file_write_at_all(f,0,b),Ok(n) if n==b.len()),
            "chunked header write",
        )?;
        agree_phase(
            c,
            matches!(ffi::file_write_at(f,h.offsets[rank] as i64,&payload,options.mode==MpiIoMode::Collective),Ok(n) if n==payload.len()),
            "chunked payload write",
        )?;
        agree_phase(
            c,
            ffi::file_sync(f) == ffi::MPI_SUCCESS as i32,
            "chunked payload sync",
        )?;
        agree_phase(
            c,
            matches!(ffi::file_get_size(f),Ok(n) if n==h.end as i64),
            "chunked file length",
        )?;
        let marker = 1u64.to_le_bytes();
        let b = if c.rank() == 0 { &marker[..] } else { &[] };
        agree_phase(
            c,
            matches!(ffi::file_write_at_all(f,80,b),Ok(n) if n==b.len()),
            "chunked commit",
        )
        .map_err(|_| IoError::CommitUncertain {
            stage: "chunked commit",
        })?;
        agree_phase(
            c,
            ffi::file_sync(f) == ffi::MPI_SUCCESS as i32,
            "chunked commit sync",
        )
        .map_err(|_| IoError::CommitUncertain {
            stage: "chunked commit sync",
        })
    })
}
/// Read into the identical writer grid, rank mapping, decomposition and
/// permutation (no repartitioning). Destination is unchanged on error,
/// including errors reported after actual native cleanup.
pub fn read_mpi_chunked<P: AsRef<Path>, T: IoElement, const N: usize, const M: usize>(
    path: P,
    view: PencilArrayViewMut<'_, T, N, M>,
    options: &MpiIoOptions,
) -> Result<(), IoError> {
    read_inner(path.as_ref(), view, options, false)
}
fn read_inner<T: IoElement, const N: usize, const M: usize>(
    path: &Path,
    mut view: PencilArrayViewMut<'_, T, N, M>,
    options: &MpiIoOptions,
    inject: bool,
) -> Result<(), IoError> {
    let c = view.pencil().topology().communicator();
    entry(c, 2)?;
    let h = view_header::<T, N, M>(c, path, 2, view.pencil(), view.extra_shape().dimensions())?;
    let rank = c.rank() as usize;
    let values = with_file(c, path, options, false, inject, |f| {
        let found = scan(c, f)?;
        agree_phase(c, found.bytes == h.bytes, "chunked writer layout")?;
        let mut b = prepared(
            c,
            (|| {
                let n = view.len().checked_mul(T::WIDTH).ok_or_else(invalid)?;
                if view.len() != h.counts[rank] || n > i32::MAX as usize {
                    return Err(invalid());
                }
                buffer(n)
            })(),
        )?;
        agree_phase(
            c,
            matches!(ffi::file_read_at(f,h.offsets[rank] as i64,&mut b,options.mode==MpiIoMode::Collective),Ok(n) if n==b.len()),
            "chunked payload read",
        )?;
        let mut values = Vec::new();
        prepared(
            c,
            values.try_reserve_exact(view.len()).map_err(|_| invalid()),
        )?;
        values.extend(b.chunks_exact(T::WIDTH).map(T::decode_le));
        Ok(values)
    })?;
    view.as_mut_slice().copy_from_slice(&values);
    Ok(())
}
/// Metadata only: no payload is read. API identity denotes the chunked format.
/// Provenance is little-endian u64 writer grid followed by spatial permutation.
/// The communicator must have the writer's Cartesian grid and rank mapping.
pub fn read_mpi_chunked_catalog<P: AsRef<Path>>(
    path: P,
    c: &CartesianCommunicator,
    options: &MpiIoOptions,
) -> Result<DatasetInfo, IoError> {
    entry(c, 3)?;
    with_file(c, path.as_ref(), options, false, false, |f| {
        Ok(scan(c, f)?.info)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use pencil_array::{ExtraShape, MpiTopology, Pencil, PencilArray};
    mod support {
        include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/mod.rs"));
    }
    #[test]
    fn cleanup_and_native_contracts() {
        let universe = mpi::initialize().unwrap();
        let world = universe.world();
        let dir = support::owned_temp_dir(&world, "chunked-native");
        let t = MpiTopology::new(&world, [world.size() as usize]).unwrap();
        let p = Pencil::new(t.clone(), [3, 5], [0]).unwrap();
        let src = PencilArray::from_elem(p.clone(), ExtraShape::scalar(), 42u64).unwrap();
        let mut dst = PencilArray::from_elem(p, ExtraShape::scalar(), 99u64).unwrap();
        let c = t.communicator();
        for mode in [MpiIoMode::Collective, MpiIoMode::Independent] {
            let path = dir.join(format!("{mode:?}"));
            let options = MpiIoOptions::default()
                .mode(mode)
                .hint("cb_buffer_size", "1048576");
            ffi::AT_CALLS.with(|x| x.set([0; 5]));
            write_mpi_chunked(&path, src.view(), &options).unwrap();
            assert_eq!(ffi::test_info_arguments()[0], 1048576);
            let calls = ffi::AT_CALLS.with(|x| x.get());
            assert_eq!(
                &calls[..2],
                if mode == MpiIoMode::Collective {
                    &[3, 0]
                } else {
                    &[2, 1]
                }
            );
            ffi::AT_CALLS.with(|x| x.set([0; 5]));
            read_mpi_chunked_catalog(&path, c, &options).unwrap();
            let calls = ffi::AT_CALLS.with(|x| x.get());
            assert_eq!(&calls[..4], &[0, 0, 2, 0]);
            let expected = if c.rank() == 0 {
                PREFIX + word(&std::fs::read(&path).unwrap(), 2).unwrap()
            } else {
                0
            };
            assert_eq!(calls[4], expected, "catalog must not read any payload");
            ffi::AT_CALLS.with(|x| x.set([0; 5]));
            dst.as_mut_slice().fill(99);
            assert!(read_inner(&path, dst.view_mut(), &options, true).is_err());
            assert!(dst.as_slice().iter().all(|&x| x == 99));
            let calls = ffi::AT_CALLS.with(|x| x.get());
            assert_eq!(
                &calls[2..4],
                if mode == MpiIoMode::Collective {
                    &[3, 0]
                } else {
                    &[2, 1]
                }
            );
            read_mpi_chunked(&path, dst.view_mut(), &options).unwrap();
            assert_eq!(dst.as_slice(), src.as_slice());
        }
        support::cleanup_owned_temp_dir(&world, &dir);
    }
}
