use super::*;
use mpi::collective::Root;
use mpi::topology::CartesianCommunicator;
use std::sync::atomic::{AtomicU64, Ordering};

mod data;
#[cfg(test)]
mod tests;
mod tree;
#[cfg(test)]
pub(crate) use tests::native_contracts;

#[cfg(test)]
thread_local! { static SESSION_FAULT: std::cell::Cell<u8> = const { std::cell::Cell::new(0) }; }
#[cfg(test)]
fn test_post_op(comm: &CartesianCommunicator, phase: u8) -> Result<(), IoError> {
    let inject = SESSION_FAULT.with(|f| f.get() == phase);
    let error = if inject && comm.rank() == 0 {
        Some(IoError::CommitUncertain {
            stage: "test session operation cleanup",
        })
    } else {
        None
    };
    aggregate_cleanup_result(comm, error).map_or(Ok(()), Err)
}

const TREE: &[u8] = b"/pencil_io_tree_v1\0";
const MAX_DEPTH: usize = 32;
const MAX_PATH: usize = 4096;
const MAX_COMPONENT: usize = 255;
const MAX_OBJECTS: usize = 65_536;
const MAX_TREE_BYTES: usize = 16 * 1024 * 1024;
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// Errors specific to persistent HDF5 sessions; legacy error enums are unchanged.
#[non_exhaustive]
#[derive(Debug)]
pub enum Hdf5SessionError {
    /// Validation, collective agreement or native I/O failure.
    Io(IoError),
    /// Invalid lifecycle state or operation.
    Invalid(&'static str),
}
impl From<IoError> for Hdf5SessionError {
    fn from(error: IoError) -> Self {
        Self::Io(error)
    }
}
impl std::fmt::Display for Hdf5SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => e.fmt(f),
            Self::Invalid(what) => write!(f, "invalid HDF5 session operation: {what}"),
        }
    }
}
impl std::error::Error for Hdf5SessionError {}

/// A native parallel HDF5 file and duplicated communicator retained across calls.
///
/// All methods are collective on the original borrowed communicator. Do not
/// overlap operations on that communicator. Call `close` before MPI finalization:
/// dropping an open session is fail-stop, never an implicit collective close.
/// Same-file writers require external serialization. This Rust-only hierarchical
/// namespace does not implement Julia's format or alter legacy HDF5 paths.
/// Member paths are normalized relative UTF-8 paths: at most 32 components,
/// 255 bytes per component and 4096 bytes total; empty, `.` and `..` components
/// are rejected. Catalogs bound the tree to 65,536 objects (including the root)
/// and 16 MiB of aggregate path/metadata bytes.
/// MPI hints are applied when opening the native file and cannot be changed on
/// an open handle. Per-operation hints must be empty or equal the opening hints.
/// Successful reads publish after per-operation cleanup; a later file-close
/// failure cannot roll back values from an earlier successful read.
pub struct Hdf5FileSession<'c> {
    comm: &'c CartesianCommunicator,
    file: Option<native::Hid>,
    duplicate: Option<ffi::MPI_Comm>,
    id: u64,
    mode: u64,
    poisoned: bool,
    path: CString,
    hints: Vec<(String, String)>,
}

fn entry_header(comm: &CartesianCommunicator, op: u64) -> Result<(), IoError> {
    let fixed = [0x494f_0000_0006, op, 1, 0, 1];
    let mut lo = [0u64; 5];
    let mut hi = [0u64; 5];
    comm.all_reduce_into(&fixed, &mut lo, SystemOperation::min());
    comm.all_reduce_into(&fixed, &mut hi, SystemOperation::max());
    if lo != hi {
        Err(IoError::CollectiveDescriptorMismatch)
    } else {
        Ok(())
    }
}

fn agree_bytes(comm: &CartesianCommunicator, bytes: &[u8]) -> Result<(), IoError> {
    agree_phase(
        comm,
        bytes.len() <= crate::MAX_DESCRIPTOR_BYTES,
        "session descriptor bound",
    )?;
    let mut len = bytes.len() as u64;
    comm.process_at_rank(0).broadcast_into(&mut len);
    // Fixed storage means root-controlled lengths never trigger allocations.
    let mut root = [0u8; crate::MAX_DESCRIPTOR_BYTES];
    if comm.rank() == 0 {
        root[..bytes.len()].copy_from_slice(bytes);
    }
    comm.process_at_rank(0)
        .broadcast_into(&mut root[..len as usize]);
    agree_phase(
        comm,
        bytes == &root[..len as usize],
        "session descriptor agreement",
    )
}

// Controls have already passed the existing bounded validation/agreement.
fn agree_settings(comm: &CartesianCommunicator, settings: Hdf5Settings<'_>) -> Result<(), IoError> {
    let mut bytes = [0u8; (MAX_PROTOCOL_RANK + 5) * 8];
    let chunks = settings.chunks.unwrap_or(&[]);
    let words = [
        u64::from(settings.collective),
        u64::from(settings.shuffle),
        settings.deflate.map_or(0, |v| u64::from(v) + 1),
        chunks.len() as u64,
        settings.hints.len() as u64,
    ];
    for (dst, value) in bytes
        .chunks_exact_mut(8)
        .zip(words.into_iter().chain(chunks.iter().map(|&x| x as u64)))
    {
        dst.copy_from_slice(&value.to_le_bytes());
    }
    agree_bytes(comm, &bytes[..(5 + chunks.len()) * 8])?;
    for (key, value) in settings.hints {
        agree_bytes(comm, key.as_bytes())?;
        agree_bytes(comm, value.as_bytes())?;
    }
    Ok(())
}
fn agree_member<const N: usize, const M: usize>(
    comm: &CartesianCommunicator,
    pencil: &pencil_array::Pencil<N, M>,
    extra: &[usize],
    code: u64,
    width: usize,
) -> Result<(), IoError> {
    let len = 5 + 2 * N + 2 * M + extra.len();
    agree_phase(
        comm,
        len <= crate::MAX_DESCRIPTOR_BYTES / 8,
        "session member bound",
    )?;
    let mut bytes = [0u8; crate::MAX_DESCRIPTOR_BYTES];
    let header = [N as u64, M as u64, extra.len() as u64, code, width as u64];
    let words = header
        .into_iter()
        .chain(pencil.global_shape().iter().map(|&x| x as u64))
        .chain(extra.iter().map(|&x| x as u64))
        .chain(pencil.topology().process_grid().iter().map(|&x| x as u64))
        .chain(pencil.permutation().axes().iter().map(|x| x.index() as u64))
        .chain(pencil.decomposition().iter().map(|x| x.index() as u64));
    for (dst, value) in bytes.chunks_exact_mut(8).zip(words) {
        dst.copy_from_slice(&value.to_le_bytes());
    }
    agree_bytes(comm, &bytes[..len * 8])
}

fn valid_path(path: &str) -> bool {
    !path.is_empty()
        && path.len() <= MAX_PATH
        && !path.as_bytes().contains(&0)
        && path.split('/').count() <= MAX_DEPTH
        && path
            .split('/')
            .all(|c| !c.is_empty() && c != "." && c != ".." && c.len() <= MAX_COMPONENT)
}

impl<'c> Hdf5FileSession<'c> {
    /// Exclusively create a new file and an empty tree namespace.
    /// Existing files are never overwritten.
    pub fn create<P: AsRef<Path>>(
        path: P,
        comm: &'c CartesianCommunicator,
        options: &crate::MpiIoOptions,
    ) -> Result<Self, Hdf5SessionError> {
        Self::open(path.as_ref(), comm, options, 1, 0x701)
    }
    /// Open an existing tree for reads only.
    pub fn open_read<P: AsRef<Path>>(
        path: P,
        comm: &'c CartesianCommunicator,
        options: &crate::MpiIoOptions,
    ) -> Result<Self, Hdf5SessionError> {
        Self::open(path.as_ref(), comm, options, 2, 0x702)
    }
    /// Open an existing tree for append-only writes.
    pub fn open_append<P: AsRef<Path>>(
        path: P,
        comm: &'c CartesianCommunicator,
        options: &crate::MpiIoOptions,
    ) -> Result<Self, Hdf5SessionError> {
        Self::open(path.as_ref(), comm, options, 3, 0x703)
    }
    fn open(
        path: &Path,
        comm: &'c CartesianCommunicator,
        options: &crate::MpiIoOptions,
        mode: u64,
        op: u64,
    ) -> Result<Self, Hdf5SessionError> {
        entry_header(comm, op)?;
        let mut id = if comm.rank() == 0 {
            NEXT_ID
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
                .unwrap_or(0)
        } else {
            0
        };
        comm.process_at_rank(0).broadcast_into(&mut id);
        if id == 0 {
            return Err(Hdf5SessionError::Invalid("open identity exhausted"));
        }
        let mut state = [id, mode, 0, 0];
        comm.process_at_rank(0).broadcast_into(&mut state);
        agree_phase(comm, state == [id, mode, 0, 0], "session open state")?;
        #[cfg(unix)]
        let bytes = {
            use std::os::unix::ffi::OsStrExt;
            path.as_os_str().as_bytes()
        };
        #[cfg(not(unix))]
        let bytes = {
            let text = path.to_str();
            agree_phase(comm, text.is_some(), "session file path encoding")?;
            text.expect("agreed path encoding").as_bytes()
        };
        agree_phase(
            comm,
            !bytes.is_empty() && !bytes.contains(&0),
            "session file path",
        )?;
        agree_bytes(comm, bytes)?;
        let path = path_cstring(path, comm)?;
        crate::options::agree_controls(comm, options.mode, &options.hints, &[])?;
        agree_settings(
            comm,
            Hdf5Settings {
                collective: options.mode == crate::MpiIoMode::Collective,
                hints: &options.hints,
                ..Hdf5Settings::default()
            },
        )?;
        let mut hints = Vec::new();
        agree_phase(
            comm,
            hints.try_reserve_exact(options.hints.len()).is_ok(),
            "session hints allocation",
        )?;
        for (key, value) in &options.hints {
            let (mut k, mut v) = (String::new(), String::new());
            let allocation = k
                .try_reserve_exact(key.len())
                .and_then(|()| v.try_reserve_exact(value.len()));
            agree_phase(comm, allocation.is_ok(), "session hint allocation")?;
            k.push_str(key);
            v.push_str(value);
            hints.push((k, v));
        }
        let duplicate = duplicate_comm(comm)?;
        let fapl = match prepare_hdf5_fapl(comm, duplicate.raw, &options.hints) {
            Ok(x) => x,
            Err(e) => return Err(cleanup_comm_only(comm, duplicate, e).into()),
        };
        let file = match open_hdf5_mode(comm, fapl, &path, mode == 1, mode == 3) {
            Ok(x) => x,
            Err(e) => return Err(cleanup_comm_only(comm, duplicate, e).into()),
        };
        let mut session = Self {
            comm,
            file: Some(file),
            duplicate: Some(duplicate.raw),
            id,
            mode,
            poisoned: false,
            path,
            hints,
        };
        let setup = if mode == 1 {
            collective_handle_phase(
                comm,
                native::group_create(file, cstr(TREE)),
                "session namespace create",
            )
            .and_then(|group| close_group(comm, group))
            .map_err(Hdf5SessionError::from)
        } else {
            session
                .root()
                .and_then(|group| close_group(comm, group).map_err(Into::into))
        };
        if let Err(e) = setup.and_then(|()| {
            if mode == 3 {
                session.inspect_tree().map(|_| ())
            } else {
                Ok(())
            }
        }) {
            session.close_native()?;
            return Err(e);
        }
        Ok(session)
    }
    fn entry(&self, op: u64, path: &str) -> Result<(), Hdf5SessionError> {
        entry_header(self.comm, op)?;
        let local = [
            self.id,
            self.mode,
            u64::from(self.file.is_none()),
            u64::from(self.poisoned),
        ];
        let mut root = local;
        self.comm.process_at_rank(0).broadcast_into(&mut root);
        agree_phase(self.comm, root == local, "session identity and state")?;
        agree_bytes(self.comm, self.path.as_bytes())?;
        agree_bytes(self.comm, path.as_bytes())?;
        if op != 0x708 && (self.file.is_none() || self.poisoned) {
            return Err(Hdf5SessionError::Invalid("closed or poisoned"));
        }
        Ok(())
    }
    fn check_hints(&self, settings: Hdf5Settings<'_>) -> Result<(), IoError> {
        agree_phase(
            self.comm,
            settings.hints.is_empty() || settings.hints == self.hints,
            "session hints fixed at open",
        )
    }
    fn writable(&self) -> Result<(), Hdf5SessionError> {
        if self.mode == 2 {
            Err(Hdf5SessionError::Invalid("read-only"))
        } else {
            Ok(())
        }
    }
    fn root(&self) -> Result<native::Hid, Hdf5SessionError> {
        let file = self.file.ok_or(Hdf5SessionError::Invalid("closed"))?;
        check_kind(self.comm, file, cstr(TREE), native::LinkObjectType::Group)?;
        Ok(collective_handle_phase(
            self.comm,
            native::group_open(file, cstr(TREE)),
            "session root open",
        )?)
    }
    fn parent(&self, path: &str) -> Result<(native::Hid, CString), Hdf5SessionError> {
        agree_phase(self.comm, valid_path(path), "session normalized path")?;
        let mut group = self.root()?;
        let mut components = path.split('/').peekable();
        while let Some(component) = components.next() {
            let name = CString::new(component).expect("validated component");
            if components.peek().is_none() {
                return Ok((group, name));
            }
            let next =
                check_kind(self.comm, group, &name, native::LinkObjectType::Group).and_then(|()| {
                    collective_handle_phase(
                        self.comm,
                        native::group_open(group, &name),
                        "session parent open",
                    )
                    .map_err(Into::into)
                });
            let cleanup = close_group(self.comm, group);
            group = match (next, cleanup) {
                (Ok(next), Ok(())) => next,
                (Ok(next), Err(e)) => {
                    close_group(self.comm, next)?;
                    return Err(e.into());
                }
                (Err(e), _) => return Err(e),
            };
        }
        unreachable!("validated nonempty path")
    }
    /// Create only the final path component. All parents must already exist.
    pub fn create_group(&mut self, path: &str) -> Result<(), Hdf5SessionError> {
        self.entry(0x704, path)?;
        self.writable()?;
        let (parent, name) = self.parent(path)?;
        let result = (|| {
            require_absent(self.comm, parent, &name)?;
            self.poisoned = true;
            let group = collective_handle_phase(
                self.comm,
                native::group_create(parent, &name),
                "session group create",
            )?;
            close_group(self.comm, group)?;
            phase_code(
                self.comm,
                native::file_flush(self.file.expect("open session")),
                "session group flush",
            )?;
            Ok(())
        })();
        let cleanup = close_group(self.comm, parent);
        match (result, cleanup) {
            (Ok(()), Ok(())) => {
                self.poisoned = false;
                Ok(())
            }
            (Err(e), _) => Err(e),
            (_, Err(e)) => Err(e.into()),
        }
    }
    /// Explicit collective close. Preflight mismatches leave the session retryable;
    /// poisoned sessions remain closeable. Repeated agreed closes are harmless.
    pub fn close(&mut self) -> Result<(), Hdf5SessionError> {
        self.entry(0x708, "")?;
        self.close_native()
    }
    fn close_native(&mut self) -> Result<(), Hdf5SessionError> {
        if let Some(mut file) = self.file.take() {
            let code = native::file_close(&mut file);
            if !collective_state(self.comm, code >= 0).0 {
                abort_unrecoverable(self.comm.as_raw(), "session file close");
            }
        }
        if let Some(mut duplicate) = self.duplicate.take() {
            let code = ffi::comm_free(&mut duplicate);
            if !collective_state(self.comm, code == ffi::MPI_SUCCESS as i32).0 {
                abort_unrecoverable(self.comm.as_raw(), "session communicator close");
            }
        }
        Ok(())
    }
}
impl Drop for Hdf5FileSession<'_> {
    fn drop(&mut self) {
        if self.file.is_some() || self.duplicate.is_some() {
            if ffi::mpi_is_finalized().unwrap_or(true) {
                std::process::abort();
            }
            abort_unrecoverable(
                self.comm.as_raw(),
                "open HDF5 session dropped without collective close",
            );
        }
    }
}
fn close_group(comm: &CartesianCommunicator, mut group: native::Hid) -> Result<(), IoError> {
    let code = native::group_close(&mut group);
    if !collective_state(comm, code >= 0).0 {
        abort_unrecoverable(comm.as_raw(), "session group close");
    }
    Ok(())
}
fn check_kind(
    comm: &CartesianCommunicator,
    parent: native::Hid,
    name: &CStr,
    kind: native::LinkObjectType,
) -> Result<(), Hdf5SessionError> {
    let check = native::link_is_hard_and_type(parent, name, kind);
    agree_phase(
        comm,
        matches!(check, Ok(true)),
        "session hard link and object kind",
    )?;
    Ok(())
}
fn require_absent(
    comm: &CartesianCommunicator,
    parent: native::Hid,
    name: &CStr,
) -> Result<(), Hdf5SessionError> {
    let exists = native::link_exists(parent, name);
    agree_phase(comm, matches!(exists, Ok(false)), "session no overwrite")?;
    Ok(())
}
