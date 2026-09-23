//! Collective, metadata-only catalog readers.
use crate::mpi_io::{
    FileGuard, abort_unrecoverable, collective_state, duplicate_comm, finish_resources,
};
use crate::named_mpi::scan_file_bounded;
use crate::{COMMIT_MARKER, FORMAT_VERSION, IoError, MAX_HEADER_BYTES, MAX_PROTOCOL_RANK, ffi};
use mpi::collective::{CommunicatorCollectives, Root, SystemOperation};
use mpi::raw::AsRaw;
use mpi::topology::{CartesianCommunicator, Communicator};
use std::{ffi::CString, path::Path};

const MAGIC: &[u8; 8] = b"PENCILIO";
const CATALOG_NAMESPACE: u64 = 0x494f_0000_0003;
const MPI_CATALOG_OP: u64 = 0x401;
const NAMED_CATALOG_OP: u64 = 0x402;
#[cfg(feature = "parallel-hdf5")]
const HDF5_CATALOG_OP: u64 = 0x403;

/// Scalar element types recorded in a dataset catalog.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ScalarType {
    /// Signed 8-bit integer.
    I8,
    /// Unsigned 8-bit integer.
    U8,
    /// Signed 16-bit integer.
    I16,
    /// Unsigned 16-bit integer.
    U16,
    /// Signed 32-bit integer.
    I32,
    /// Unsigned 32-bit integer.
    U32,
    /// Signed 64-bit integer.
    I64,
    /// Unsigned 64-bit integer.
    U64,
    /// 32-bit floating point.
    F32,
    /// 64-bit floating point.
    F64,
    /// Complex number with 32-bit floating-point components.
    ComplexF32,
    /// Complex number with 64-bit floating-point components.
    ComplexF64,
}
impl ScalarType {
    /// Return the stable catalog code for this scalar type.
    pub fn code(self) -> u64 {
        match self {
            Self::I8 => 1,
            Self::U8 => 2,
            Self::I16 => 3,
            Self::U16 => 4,
            Self::I32 => 5,
            Self::U32 => 6,
            Self::I64 => 7,
            Self::U64 => 8,
            Self::F32 => 9,
            Self::F64 => 10,
            Self::ComplexF32 => 11,
            Self::ComplexF64 => 12,
        }
    }
    /// Return the number of bytes occupied by one scalar value.
    pub fn width(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 => 8,
            Self::ComplexF32 => 8,
            Self::ComplexF64 => 16,
        }
    }
    pub(crate) fn decode(c: u64, w: u64) -> Option<Self> {
        let x = match c {
            1 => Self::I8,
            2 => Self::U8,
            3 => Self::I16,
            4 => Self::U16,
            5 => Self::I32,
            6 => Self::U32,
            7 => Self::I64,
            8 => Self::U64,
            9 => Self::F32,
            10 => Self::F64,
            11 => Self::ComplexF32,
            12 => Self::ComplexF64,
            _ => return None,
        };
        (x.width() as u64 == w).then_some(x)
    }
}
/// Metadata for one dataset returned by a catalog reader.
///
/// The shape vectors are strict metadata: readers do not infer or alter them.
/// Provenance contains the writer process-grid extents followed by the writer
/// axis permutation, encoded as little-endian u64 values. The final
/// `global_shape().len()` values are the permutation. Named MPI catalogs do not
/// store writer provenance and therefore return an empty slice.
/// Collection component axes remain verbatim in `extra_shape`; no component
/// count is inferred. Catalogs are bounded to 65,536 named datasets per
/// container/group, 1,024-byte UTF-8 names, 1 MiB MPI metadata headers,
/// and the existing protocol/native rank limits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatasetInfo {
    pub(crate) name: Option<String>,
    pub(crate) scalar_type: ScalarType,
    pub(crate) global_shape: Vec<u64>,
    pub(crate) extra_shape: Vec<u64>,
    pub(crate) provenance: Vec<u8>,
}
impl DatasetInfo {
    /// Return the dataset name, if the format stores one.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
    /// Return the dataset scalar type.
    pub fn scalar_type(&self) -> ScalarType {
        self.scalar_type
    }
    /// Return the strict global shape metadata.
    pub fn global_shape(&self) -> &[u64] {
        &self.global_shape
    }
    /// Return the strict extra (non-spatial) shape metadata.
    pub fn extra_shape(&self) -> &[u64] {
        &self.extra_shape
    }
    /// Return format-defined writer provenance bytes.
    pub fn provenance(&self) -> &[u8] {
        &self.provenance
    }
}
/// Errors returned while collectively reading a catalog.
#[derive(Debug, thiserror::Error)]
pub enum CatalogError {
    /// The underlying I/O operation failed.
    #[error(transparent)]
    Io(#[from] IoError),
    /// The catalog header or metadata is malformed.
    #[error("malformed catalog: {0}")]
    Malformed(&'static str),
    /// The file contains bytes after its committed records.
    #[error("invalid catalog tail")]
    InvalidTail,
}
fn u(b: &[u8], p: usize) -> Option<u64> {
    b.get(p..p + 8)?.try_into().ok().map(u64::from_le_bytes)
}
fn path<P: AsRef<Path>>(p: P) -> Result<CString, CatalogError> {
    CString::new(p.as_ref().to_str().ok_or(IoError::InvalidPath)?)
        .map_err(|_| IoError::InvalidPath.into())
}
fn catalog_header(c: &CartesianCommunicator, op: u64) -> Result<(), CatalogError> {
    let fixed = [CATALOG_NAMESPACE, op, FORMAT_VERSION, 0, 0];
    let (mut min, mut max) = ([0u64; 5], [0u64; 5]);
    c.all_reduce_into(&fixed, &mut min, SystemOperation::min());
    c.all_reduce_into(&fixed, &mut max, SystemOperation::max());
    if min != max {
        return Err(IoError::CollectiveDescriptorMismatch.into());
    }
    Ok(())
}

fn agree<C: CommunicatorCollectives>(
    c: &C,
    ok: bool,
    why: &'static str,
) -> Result<(), CatalogError> {
    let x = i32::from(ok);
    let mut y = 0;
    c.all_reduce_into(&x, &mut y, SystemOperation::min());
    if y == 0 {
        Err(IoError::CollectivePrecondition { phase: why }.into())
    } else {
        Ok(())
    }
}
fn agree_path(c: &CartesianCommunicator, p: &Path) -> Result<CString, CatalogError> {
    let local = p.to_str();
    let valid = local.is_some_and(|s| {
        !s.is_empty() && s.len() <= MAX_HEADER_BYTES && !s.as_bytes().contains(&0)
    });
    let flag = i32::from(valid);
    let mut all_valid = 0;
    c.all_reduce_into(&flag, &mut all_valid, SystemOperation::min());
    if all_valid == 0 {
        return Err(if valid {
            IoError::CollectiveDescriptorMismatch.into()
        } else {
            IoError::InvalidPath.into()
        });
    }
    let s = local.expect("path validity agreed");
    let n = s.len();
    let mut lo = 0;
    let mut hi = 0;
    c.all_reduce_into(&(n as u64), &mut lo, SystemOperation::min());
    c.all_reduce_into(&(n as u64), &mut hi, SystemOperation::max());
    if lo != hi || n > MAX_HEADER_BYTES {
        return Err(IoError::CollectiveDescriptorMismatch.into());
    }
    agree_bytes(c, s.as_bytes())?;
    path(p)
}
pub(crate) fn agree_bytes(c: &CartesianCommunicator, bytes: &[u8]) -> Result<(), CatalogError> {
    let len = bytes.len() as u64;
    let (mut lo, mut hi) = (0u64, 0u64);
    c.all_reduce_into(&len, &mut lo, SystemOperation::min());
    c.all_reduce_into(&len, &mut hi, SystemOperation::max());
    if lo != hi {
        return Err(IoError::CollectiveDescriptorMismatch.into());
    }
    for chunk in bytes.chunks(512) {
        let (mut low, mut high) = ([0u8; 512], [0u8; 512]);
        c.all_reduce_into(chunk, &mut low[..chunk.len()], SystemOperation::min());
        c.all_reduce_into(chunk, &mut high[..chunk.len()], SystemOperation::max());
        if low != high {
            return Err(IoError::CollectiveDescriptorMismatch.into());
        }
    }
    Ok(())
}

fn read_at(
    c: &CartesianCommunicator,
    f: ffi::MPI_File,
    off: usize,
    n: usize,
) -> Result<Vec<u8>, CatalogError> {
    let mut b = Vec::new();
    agree(
        c,
        n <= MAX_HEADER_BYTES && b.try_reserve_exact(n).is_ok(),
        "catalog metadata allocation",
    )?;
    b.resize(n, 0);
    let r = if c.rank() == 0 {
        ffi::file_read_at_all(f, off as i64, &mut b)
    } else {
        ffi::file_read_at_all(f, 0, &mut [])
    };
    agree(
        c,
        matches!(r,Ok(x) if (c.rank()==0&&x==n)||(c.rank()!=0&&x==0)),
        "catalog metadata read",
    )?;
    c.process_at_rank(0).broadcast_into(&mut b);
    Ok(b)
}
fn open<P: AsRef<Path>>(
    c: &CartesianCommunicator,
    p: P,
) -> Result<(crate::mpi_io::CommGuard, ffi::MPI_File), CatalogError> {
    let p = agree_path(c, p.as_ref())?;
    let d = duplicate_comm(c)?;
    let r = ffi::file_open(d.raw, &p, false);
    let (all_ok, mixed) = collective_state(c, r.is_ok());
    if mixed {
        abort_unrecoverable(c.as_raw(), "catalog MPI_File_open partial handle");
    }
    if !all_ok {
        let _ = crate::mpi_io::finish_comm(c, d);
        return Err(r
            .err()
            .map(|code| IoError::Mpi {
                operation: "MPI_File_open",
                code,
            })
            .unwrap_or(IoError::CollectivePrecondition {
                phase: "catalog file open",
            })
            .into());
    }
    let f = r.expect("collective open");
    if let Err(e) = agree(
        c,
        ffi::file_set_errors_return(f) == ffi::MPI_SUCCESS as i32,
        "catalog file error handler",
    ) {
        close(c, d, f)?;
        return Err(e);
    }
    Ok((d, f))
}
fn close(
    c: &CartesianCommunicator,
    d: crate::mpi_io::CommGuard,
    f: ffi::MPI_File,
) -> Result<(), CatalogError> {
    finish_resources(c, d, FileGuard { raw: f }, None).map_or(Ok(()), |e| Err(CatalogError::Io(e)))
}
fn dims_ok(v: &[u64]) -> bool {
    !v.is_empty() && v.len() <= MAX_PROTOCOL_RANK && v.iter().all(|&x| x > 0)
}

/// Read a strict metadata-only catalog from a native MPI v1 dataset collectively.
///
/// All ranks in `c` must call this function with the same path. The returned
/// vector contains strict metadata only; writers from different jobs or
/// communicators must serialize access externally. This reader does not roll
/// back, overwrite, or resize a file.
pub fn read_mpi_catalog<P: AsRef<Path>>(
    p: P,
    c: &CartesianCommunicator,
) -> Result<Vec<DatasetInfo>, CatalogError> {
    catalog_header(c, MPI_CATALOG_OP)?;
    let (d, f) = open(c, p)?;
    let r = (|| {
        let native_size = ffi::file_get_size(f);
        agree(c, native_size.is_ok(), "MPI_File_get_size")?;
        let size = usize::try_from(native_size.map_err(|x| IoError::Mpi {
            operation: "MPI_File_get_size",
            code: x,
        })?)
        .map_err(|_| CatalogError::Malformed("size"))?;
        agree_bytes(c, &(size as u64).to_le_bytes())?;
        if size < 96 {
            return Err(CatalogError::Malformed("header"));
        }
        let pre = read_at(c, f, 0, 96)?;
        if &pre[..8] != MAGIC
            || u(&pre, 8) != Some(FORMAT_VERSION)
            || u(&pre, 24) != Some(COMMIT_MARKER)
        {
            return Err(CatalogError::Malformed("header"));
        }
        let hl = usize::try_from(u(&pre, 16).ok_or(CatalogError::Malformed("header"))?)
            .map_err(|_| CatalogError::Malformed("header"))?;
        if !(96..=MAX_HEADER_BYTES).contains(&hl) || hl > size {
            return Err(CatalogError::Malformed("header bounds"));
        }
        let h = read_at(c, f, 0, hl)?;
        let n = usize::try_from(u(&h, 32).ok_or(CatalogError::Malformed("rank"))?)
            .map_err(|_| CatalogError::Malformed("rank"))?;
        let er = usize::try_from(u(&h, 56).ok_or(CatalogError::Malformed("rank"))?)
            .map_err(|_| CatalogError::Malformed("rank"))?;
        let gr = usize::try_from(u(&h, 80).ok_or(CatalogError::Malformed("grid"))?)
            .map_err(|_| CatalogError::Malformed("grid"))?;
        if n == 0
            || n > MAX_PROTOCOL_RANK
            || er > MAX_PROTOCOL_RANK
            || n + er > MAX_PROTOCOL_RANK
            || gr == 0
            || gr > MAX_PROTOCOL_RANK
            || hl
                != 96
                    + n.checked_add(er)
                        .and_then(|x| x.checked_add(gr))
                        .and_then(|x| x.checked_add(n))
                        .ok_or(CatalogError::Malformed("header bounds"))?
                        * 8
            || hl > MAX_HEADER_BYTES
        {
            return Err(CatalogError::Malformed("header bounds"));
        }
        let typ = u(&h, 40).ok_or(CatalogError::Malformed("type"))?;
        let st = ScalarType::decode(typ, u(&h, 48).ok_or(CatalogError::Malformed("width"))?)
            .ok_or(CatalogError::Malformed("type"))?;
        let po = usize::try_from(u(&h, 64).ok_or(CatalogError::Malformed("payload"))?)
            .map_err(|_| CatalogError::Malformed("payload"))?;
        let pb = usize::try_from(u(&h, 72).ok_or(CatalogError::Malformed("payload"))?)
            .map_err(|_| CatalogError::Malformed("payload"))?;
        if po != hl || u(&h, 88) != Some(0) || po.checked_add(pb) != Some(size) {
            return Err(CatalogError::Malformed("tail"));
        }
        let mut at = 96;
        let take = |at: &mut usize, k: usize| {
            let mut v = Vec::with_capacity(k);
            for _ in 0..k {
                v.push(u(&h, *at)?);
                *at += 8;
            }
            Some(v)
        };
        let global = take(&mut at, n).ok_or(CatalogError::Malformed("shape"))?;
        let extra = take(&mut at, er).ok_or(CatalogError::Malformed("shape"))?;
        let grid = take(&mut at, gr).ok_or(CatalogError::Malformed("grid"))?;
        let perm = take(&mut at, n).ok_or(CatalogError::Malformed("permutation"))?;
        let gp = grid
            .iter()
            .try_fold(1u64, |p, &x| p.checked_mul(x))
            .ok_or(CatalogError::Malformed("grid"))?;
        let ep = global
            .iter()
            .chain(extra.iter())
            .try_fold(1u64, |p, &x| p.checked_mul(x))
            .ok_or(CatalogError::Malformed("shape"))?;
        if global.is_empty()
            || global.len() > MAX_PROTOCOL_RANK
            || !dims_ok(&grid)
            || gp == 0
            || ep.checked_mul(st.width() as u64) != Some(pb as u64)
            || perm
                .iter()
                .enumerate()
                .any(|(i, &x)| x >= n as u64 || perm[..i].contains(&x))
        {
            return Err(CatalogError::Malformed("shape/grid/permutation"));
        }
        Ok(vec![DatasetInfo {
            name: None,
            scalar_type: st,
            global_shape: global,
            extra_shape: extra,
            provenance: h[96 + (n + er) * 8..].to_vec(),
        }])
    })();
    let agreement = agree(c, r.is_ok(), "catalog result validation");
    close(c, d, f)?;
    if r.is_ok() {
        agreement?;
    }
    r
}

#[cfg(feature = "parallel-hdf5")]
/// List strict metadata from the legacy and named Pencil HDF5 groups collectively.
///
/// All ranks in `c` must call this function with the same path. The returned
/// vector contains strict metadata only; writers from different jobs or
/// communicators must serialize access externally. This reader does not roll
/// back, overwrite, or resize a file.
pub fn read_hdf5_catalog<P: AsRef<Path>>(
    p: P,
    c: &CartesianCommunicator,
) -> Result<Vec<DatasetInfo>, CatalogError> {
    catalog_header(c, HDF5_CATALOG_OP)?;
    let _ = agree_path(c, p.as_ref())?;
    crate::hdf5_io::inspect_catalog(p.as_ref(), c)
}

/// Read metadata for all named MPI datasets collectively.
///
/// All ranks in `c` must call this function with the same path. Named MPI
/// metadata has no writer-provenance field, so each returned
/// [`DatasetInfo::provenance`] is empty. Writers from different jobs or
/// communicators must serialize access externally; this reader does not roll
/// back, overwrite, or resize a file.
pub fn read_mpi_named_catalog<P: AsRef<Path>>(
    p: P,
    c: &CartesianCommunicator,
) -> Result<Vec<DatasetInfo>, CatalogError> {
    catalog_header(c, NAMED_CATALOG_OP)?;
    let (d, f) = open(c, p)?;
    let r = (|| {
        let (records, tail, size) = scan_file_bounded(c, f, 65_536).map_err(|e| match e {
            crate::NamedIoError::InvalidTail => CatalogError::InvalidTail,
            crate::NamedIoError::Io(e) => CatalogError::Io(e),
            _ => CatalogError::InvalidTail,
        })?;
        if tail != size || records.is_empty() {
            return Err(CatalogError::InvalidTail);
        }
        records
            .into_iter()
            .map(|r| {
                let scalar_type =
                    ScalarType::decode(r.typ, r.width as u64).ok_or(CatalogError::InvalidTail)?;
                Ok(DatasetInfo {
                    name: Some(String::from_utf8(r.name).map_err(|_| CatalogError::InvalidTail)?),
                    scalar_type,
                    global_shape: r.global.into_iter().map(|x| x as u64).collect(),
                    extra_shape: r.extra.into_iter().map(|x| x as u64).collect(),
                    provenance: Vec::new(),
                })
            })
            .collect()
    })();
    let agreement = agree(c, r.is_ok(), "catalog result validation");
    close(c, d, f)?;
    if r.is_ok() {
        agreement?;
    }
    r
}
