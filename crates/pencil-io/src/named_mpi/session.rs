//! Retained native handles for the existing named-v2 format.
use super::*;
use crate::mpi_io::{CommGuard, FileGuard, abort_unrecoverable, collective_state};
use crate::{CatalogError, DatasetInfo, ScalarType};
use mpi::topology::CartesianCommunicator;
use std::sync::atomic::{AtomicU64, Ordering};

const NAMESPACE: u64 = 0x494f_0000_0005;
const CREATE: u64 = 0x601;
const OPEN_READ: u64 = 0x602;
const OPEN_APPEND: u64 = 0x603;
const WRITE: u64 = 0x604;
const READ: u64 = 0x605;
const CATALOG: u64 = 0x606;
const FLUSH: u64 = 0x607;
const CLOSE: u64 = 0x608;
const MAX_RECORDS: usize = 65_536;
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// Errors specific to persistent MPI sessions; legacy error enums are unchanged.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum MpiSessionError {
    /// An existing named-format error.
    #[error(transparent)]
    Named(#[from] NamedIoError),
    /// An underlying I/O or collective protocol error.
    #[error(transparent)]
    Io(#[from] IoError),
    /// A catalog metadata error.
    #[error(transparent)]
    Catalog(#[from] CatalogError),
    /// The handle has been explicitly closed.
    #[error("MPI session is closed")]
    Closed,
    /// This session was opened read-only.
    #[error("MPI session is read-only")]
    ReadOnly,
    /// An uncertain native operation prevents further operations except close.
    #[error("MPI session is poisoned; close it collectively")]
    Poisoned,
    /// Root has exhausted the process-wide session identifier space.
    #[error("MPI session identifiers exhausted")]
    IdExhausted,
}

/// One native MPI file and duplicated communicator retained across operations.
///
/// Every method is collective on the original borrowed communicator. All ranks
/// must choose the same session and method. Views may use a different context
/// only if its communicator is locally IDENT or CONGRUENT (same ordered ranks);
/// session methods never communicate implicitly through the view's context.
/// Same-file writers require external serialization, including other sessions.
///
/// Reads commit staged values after operation datatype/view cleanup and collective
/// agreement, **before** the eventual file close. Legacy path reads still close
/// their file before committing. Read-only opens expose the committed prefix;
/// append opens and catalogs require a complete committed tail.
///
/// Call [`Self::close`] collectively before MPI finalization. Dropping an open
/// session is fail-stop, never a collective close: MPI_Abort while MPI is live,
/// process abort after finalization. Closed Drop is benign.
pub struct MpiFileSession<'c> {
    comm: &'c CartesianCommunicator,
    resources: Option<(CommGuard, FileGuard)>,
    info: Option<InfoGuard>,
    options: MpiIoOptions,
    id: u64,
    readonly: bool,
    poisoned: bool,
}

fn entry(comm: &CartesianCommunicator, op: u64) -> Result<(), MpiSessionError> {
    // The first two collectives MUST match all existing MPI I/O entry points.
    let fixed = [NAMESPACE, op, VERSION, 0, 0];
    let (mut min, mut max) = ([0u64; 5], [0u64; 5]);
    comm.all_reduce_into(&fixed, &mut min, SystemOperation::min());
    comm.all_reduce_into(&fixed, &mut max, SystemOperation::max());
    if min != max {
        return Err(IoError::CollectiveDescriptorMismatch.into());
    }
    Ok(())
}

fn reset_byte_view(
    comm: &CartesianCommunicator,
    file: ffi::MPI_File,
    info: ffi::MPI_Info,
) -> Result<(), MpiSessionError> {
    agree_phase(
        comm,
        ffi::file_set_view_with_info(file, 0, ffi::byte_datatype(), info)
            == ffi::MPI_SUCCESS as i32,
        "session byte view reset",
    )?;
    Ok(())
}

#[cfg(test)]
thread_local! {
    static SKEW_OPEN_VIEW: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

impl<'c> MpiFileSession<'c> {
    /// Exclusively create an empty named-v2 container, open for reads and appends.
    pub fn create<P: AsRef<Path>>(
        comm: &'c CartesianCommunicator,
        path: P,
        options: &MpiIoOptions,
    ) -> Result<Self, MpiSessionError> {
        Self::open_session(comm, path.as_ref(), options, CREATE)
    }
    /// Open the committed prefix for reading, without requiring a valid tail.
    pub fn open_read<P: AsRef<Path>>(
        comm: &'c CartesianCommunicator,
        path: P,
        options: &MpiIoOptions,
    ) -> Result<Self, MpiSessionError> {
        Self::open_session(comm, path.as_ref(), options, OPEN_READ)
    }
    /// Open an existing complete container for reads and unique-name appends.
    pub fn open_append<P: AsRef<Path>>(
        comm: &'c CartesianCommunicator,
        path: P,
        options: &MpiIoOptions,
    ) -> Result<Self, MpiSessionError> {
        Self::open_session(comm, path.as_ref(), options, OPEN_APPEND)
    }
    fn open_session(
        comm: &'c CartesianCommunicator,
        path: &Path,
        options: &MpiIoOptions,
        op: u64,
    ) -> Result<Self, MpiSessionError> {
        entry(comm, op)?;
        agree_options(comm, options, &[])?;
        let bytes = prepared(comm, path_bytes(path), "session raw path preparation")?;
        agree_phase(
            comm,
            !bytes.is_empty() && !bytes.contains(&0) && bytes.len() <= crate::MAX_HEADER_BYTES,
            "session path validation",
        )?;
        let pc = prepared(comm, path_c(path), "session path preparation")?;
        crate::catalog::agree_bytes(comm, pc.as_bytes())?;
        // Only root advances the counter; zero is the broadcast exhaustion signal.
        let mut id = 0;
        if comm.rank() == 0 {
            id = NEXT_ID
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
                .unwrap_or(0);
        }
        comm.process_at_rank(0).broadcast_into(&mut id);
        if id == 0 {
            return Err(MpiSessionError::IdExhausted);
        }
        let info = InfoGuard::new(comm, options)?;
        let duplicate = duplicate_comm(comm)?;
        let file = match open(
            comm,
            duplicate.raw,
            &pc,
            match op {
                CREATE => 3,
                OPEN_READ => 1,
                _ => 2,
            },
            Some(&info),
        ) {
            Ok(file) => file,
            Err(e) => {
                crate::mpi_io::finish_comm(comm, duplicate);
                return Err(e.into());
            }
        };
        // No Session exists yet: constructor failures clean up acquired resources,
        // rather than invoking the open-session Drop fail-stop contract.
        let result = (|| -> Result<(), MpiSessionError> {
            agree_phase(
                comm,
                ffi::file_set_errors_return(file) == ffi::MPI_SUCCESS as i32,
                "session file error handler",
            )?;
            if op == CREATE {
                write_metadata(comm, file, 0, &container_header())?;
                agree_phase(
                    comm,
                    ffi::file_sync(file) == ffi::MPI_SUCCESS as i32,
                    "session empty header flush",
                )?;
            } else {
                #[cfg(test)]
                if SKEW_OPEN_VIEW.with(|f| f.replace(false)) {
                    test_skew_view(file, info.raw);
                }
                reset_byte_view(comm, file, info.raw)?;
                let (_, tail, size) = scan_file_bounded(comm, file, MAX_RECORDS)?;
                if op == OPEN_APPEND && tail != size {
                    return Err(NamedIoError::InvalidTail.into());
                }
            }
            Ok(())
        })();
        if let Err(e) = result {
            finish_resources(comm, duplicate, FileGuard { raw: file }, None);
            return Err(e);
        }
        Ok(Self {
            comm,
            resources: Some((duplicate, FileGuard { raw: file })),
            info: Some(info),
            options: options.clone(),
            id,
            readonly: op == OPEN_READ,
            poisoned: false,
        })
    }
    fn begin(&self, op: u64) -> Result<(), MpiSessionError> {
        entry(self.comm, op)?;
        agree_options(
            self.comm,
            &self.options,
            &[
                self.id,
                u64::from(self.readonly),
                u64::from(self.resources.is_none()),
                u64::from(self.poisoned),
            ],
        )?;
        if op != CLOSE {
            if self.resources.is_none() {
                return Err(MpiSessionError::Closed);
            }
            if self.poisoned {
                return Err(MpiSessionError::Poisoned);
            }
        }
        Ok(())
    }
    fn file(&self) -> ffi::MPI_File {
        self.resources.as_ref().expect("session preflight").1.raw
    }
    fn info_raw(&self) -> ffi::MPI_Info {
        self.info.as_ref().expect("session preflight").raw
    }
    fn view_preflight<const N: usize, const M: usize>(
        &self,
        pencil: &pencil_array::Pencil<N, M>,
        extra: &[usize],
        code: u64,
        width: usize,
        op: u64,
    ) -> Result<(), MpiSessionError> {
        agree_phase(
            self.comm,
            ffi::comm_is_congruent(
                self.comm.as_raw(),
                pencil.topology().communicator().as_raw(),
            )
            .unwrap_or(false),
            "session view ordered membership",
        )?;
        crate::mpi_io::descriptor_agreement(
            self.comm,
            Path::new("session"),
            op,
            pencil.global_shape(),
            extra,
            pencil.topology().process_grid(),
            pencil.permutation().axes(),
            code,
            width,
        )?;
        let axes: [u64; M] = std::array::from_fn(|i| pencil.decomposition()[i].index() as u64);
        agree_options(self.comm, &self.options, &axes)?;
        Ok(())
    }
    fn scan(&self) -> Result<(Vec<Rec>, usize, usize), MpiSessionError> {
        self.reset_view()?;
        Ok(scan_file_bounded(self.comm, self.file(), MAX_RECORDS)?)
    }
    fn reset_view(&self) -> Result<(), MpiSessionError> {
        reset_byte_view(self.comm, self.file(), self.info_raw())
    }
    fn finish_operation(
        &mut self,
        mut datatype: Option<DatatypeGuard>,
    ) -> Result<(), MpiSessionError> {
        let reset = self.reset_view();
        let code = datatype
            .as_mut()
            .map_or(ffi::MPI_SUCCESS as i32, |d| ffi::type_free(&mut d.raw));
        if !collective_state(self.comm, code == ffi::MPI_SUCCESS as i32).0 {
            abort_unrecoverable(self.comm.as_raw(), "session MPI_Type_free");
        }
        if reset.is_err() {
            self.poisoned = true;
        }
        reset
    }
    /// Append one uniquely named record using the existing flush/commit protocol.
    pub fn write_named<T: IoElement, const N: usize, const M: usize>(
        &mut self,
        name: impl AsRef<str>,
        view: PencilArrayView<'_, T, N, M>,
    ) -> Result<(), MpiSessionError> {
        self.begin(WRITE)?;
        let name = name.as_ref();
        name_bytes(self.comm, name)?;
        self.view_preflight(
            view.pencil(),
            view.extra_shape().dimensions(),
            T::CODE,
            T::WIDTH,
            WRITE,
        )?;
        if self.readonly {
            return Err(MpiSessionError::ReadOnly);
        }
        let comm = self.comm;
        let packed = prepared(comm, pack_view(&view), "session payload packing")?;
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
            "session write layout",
        )?;
        let (records, tail, size) = self.scan()?;
        if tail != size {
            return Err(NamedIoError::InvalidTail.into());
        }
        if records.iter().any(|r| r.name == name.as_bytes()) {
            return Err(NamedIoError::DuplicateName.into());
        }
        if records.len() == MAX_RECORDS {
            return Err(IoError::SizeLimit {
                what: "session records",
            }
            .into());
        }
        let head = record(name, &view, tail);
        agree_phase(comm, head.is_ok(), "session record preparation")?;
        let (head, end) = head?;
        let file = self.file();
        let mut datatype = None;
        let mut mutated = false;
        let result = (|| -> Result<(), MpiSessionError> {
            let native = if layout.empty {
                Ok(None)
            } else {
                ffi::type_create_subarray(
                    &layout.global,
                    &layout.local,
                    &layout.starts,
                    comm.as_raw(),
                )
                .map(|raw| Some(DatatypeGuard { raw }))
            };
            let ok = native.is_ok();
            datatype = native.ok().flatten();
            agree_phase(comm, ok, "session write datatype")?;
            mutated = true;
            write_metadata(comm, file, tail, &head)?;
            agree_phase(
                comm,
                ffi::file_sync(file) == ffi::MPI_SUCCESS as i32,
                "session header flush",
            )?;
            let ft = datatype.as_ref().map_or_else(ffi::byte_datatype, |d| d.raw);
            agree_phase(
                comm,
                ffi::file_set_view_with_info(file, (tail + head.len()) as i64, ft, self.info_raw())
                    == ffi::MPI_SUCCESS as i32,
                "session write view",
            )?;
            let written = if self.options.mode == MpiIoMode::Independent {
                ffi::file_write_independent(file, &packed)
            } else {
                ffi::file_write_all(file, &packed)
            };
            agree_phase(
                comm,
                matches!(written, Ok(n) if n == packed.len()),
                "session payload write",
            )?;
            agree_phase(
                comm,
                ffi::file_sync(file) == ffi::MPI_SUCCESS as i32,
                "session payload flush",
            )?;
            self.reset_view()?;
            write_metadata(comm, file, end - 8, &COMMIT_MARKER.to_le_bytes())?;
            agree_phase(
                comm,
                ffi::file_sync(file) == ffi::MPI_SUCCESS as i32,
                "session marker flush",
            )?;
            Ok(())
        })();
        let cleanup = self.finish_operation(datatype);
        if mutated && (result.is_err() || cleanup.is_err()) {
            self.poisoned = true;
            return Err(IoError::CommitUncertain {
                stage: "session append or operation cleanup",
            }
            .into());
        }
        result?;
        cleanup
    }
    /// Read a committed record, committing only after operation cleanup/agreement.
    pub fn read_named<T: IoElement, const N: usize, const M: usize>(
        &mut self,
        name: impl AsRef<str>,
        mut view: PencilArrayViewMut<'_, T, N, M>,
    ) -> Result<(), MpiSessionError> {
        self.begin(READ)?;
        let name = name.as_ref();
        name_bytes(self.comm, name)?;
        self.view_preflight(
            view.pencil(),
            view.extra_shape().dimensions(),
            T::CODE,
            T::WIDTH,
            READ,
        )?;
        let comm = self.comm;
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
            "session read layout",
        )?;
        let size = prepared(
            comm,
            view.len().checked_mul(T::WIDTH).ok_or(IoError::SizeLimit {
                what: "session read buffer",
            }),
            "session read size",
        )?;
        let mut packed = allocate(comm, size)?;
        let (records, _, _) = self.scan()?;
        let record = records
            .iter()
            .find(|r| r.name == name.as_bytes())
            .ok_or(NamedIoError::NotFound)?;
        agree_phase(
            comm,
            record.typ == T::CODE
                && record.width == T::WIDTH
                && record.global == view.pencil().global_shape()
                && record.extra == view.extra_shape().dimensions(),
            "session read metadata",
        )?;
        let file = self.file();
        let mut datatype = None;
        let result = (|| -> Result<Vec<T>, MpiSessionError> {
            let native = if layout.empty {
                Ok(None)
            } else {
                ffi::type_create_subarray(
                    &layout.global,
                    &layout.local,
                    &layout.starts,
                    comm.as_raw(),
                )
                .map(|raw| Some(DatatypeGuard { raw }))
            };
            let ok = native.is_ok();
            datatype = native.ok().flatten();
            agree_phase(comm, ok, "session read datatype")?;
            let ft = datatype.as_ref().map_or_else(ffi::byte_datatype, |d| d.raw);
            agree_phase(
                comm,
                ffi::file_set_view_with_info(
                    file,
                    record.payload_offset as i64,
                    ft,
                    self.info_raw(),
                ) == ffi::MPI_SUCCESS as i32,
                "session read view",
            )?;
            let read = if self.options.mode == MpiIoMode::Independent {
                ffi::file_read_independent(file, &mut packed)
            } else {
                ffi::file_read_all(file, &mut packed)
            };
            agree_phase(
                comm,
                matches!(read, Ok(n) if n == packed.len()),
                "session payload read",
            )?;
            Ok(prepared(
                comm,
                prepare_physical_values(&view, &packed),
                "session physical staging",
            )?)
        })();
        let cleanup = self.finish_operation(datatype);
        let physical = result?;
        cleanup?;
        agree_phase(comm, true, "session read commit")?;
        view.as_mut_slice().copy_from_slice(&physical);
        Ok(())
    }
    /// Strict metadata-only inspection on the retained native file handle.
    pub fn catalog(&mut self) -> Result<Vec<DatasetInfo>, MpiSessionError> {
        self.begin(CATALOG)?;
        let (records, tail, size) = self.scan()?;
        if tail != size {
            return Err(NamedIoError::InvalidTail.into());
        }
        let mut out = Vec::new();
        agree_phase(
            self.comm,
            out.try_reserve_exact(records.len()).is_ok(),
            "session catalog allocation",
        )?;
        for r in records {
            out.push(DatasetInfo {
                name: Some(String::from_utf8(r.name).expect("scanner validated UTF-8")),
                scalar_type: ScalarType::decode(r.typ, r.width as u64)
                    .expect("scanner validated scalar"),
                global_shape: r.global.into_iter().map(|n| n as u64).collect(),
                extra_shape: r.extra.into_iter().map(|n| n as u64).collect(),
                provenance: Vec::new(),
            });
        }
        Ok(out)
    }
    /// Synchronize the native file. Writes already flush before and after commit.
    /// A failed explicit flush is retryable and does not poison the session.
    pub fn flush(&mut self) -> Result<(), MpiSessionError> {
        self.begin(FLUSH)?;
        let result = agree_phase(
            self.comm,
            ffi::file_sync(self.file()) == ffi::MPI_SUCCESS as i32,
            "session flush",
        );
        Ok(result?)
    }
    /// Explicit collective close, also valid for poisoned or already closed sessions.
    /// A preflight mismatch leaves the handle intact so this call can be retried.
    pub fn close(&mut self) -> Result<(), MpiSessionError> {
        self.begin(CLOSE)?;
        if let Some((duplicate, file)) = self.resources.take() {
            finish_resources(self.comm, duplicate, file, None);
            self.info.take();
        }
        Ok(())
    }
}
impl Drop for MpiFileSession<'_> {
    fn drop(&mut self) {
        if self.resources.is_some() {
            if ffi::mpi_is_finalized().unwrap_or(true) {
                std::process::abort();
            }
            abort_unrecoverable(
                self.comm.as_raw(),
                "open MPI session dropped; close collectively before finalization",
            );
        }
    }
}

#[cfg(test)]
#[test]
#[ignore = "fresh MPI subprocess harness; run outside mpiexec"]
fn native_cleanup_failstop_children() {
    for (fault, diagnostic) in [
        (1, "MPI_Type_free"),
        (2, "MPI_File_close"),
        (3, "MPI_Comm_free"),
    ] {
        let output = std::process::Command::new("timeout")
            .args(["30", "mpiexec", "--oversubscribe", "-n", "2"])
            .arg(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "tests::post_cleanup_errors_preserve_destination_and_valid_commits",
                "--nocapture",
            ])
            .env("PENCIL_MPI_SESSION_CLEANUP_CHILD", fault.to_string())
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("SESSION_CLEANUP_REACHED"), "{stderr}");
        assert!(stderr.contains(diagnostic), "{stderr}");
        assert!(!output.status.success());
        assert_ne!(output.status.code(), Some(124), "cleanup hung: {stderr}");
    }
}

#[cfg(test)]
pub(crate) fn test_native_cleanup_child(
    directory: &Path,
    source: &pencil_array::PencilArray<i32, 2, 2>,
) -> ! {
    let comm = source.pencil().topology().communicator();
    let path = directory.join("native-cleanup-child.pio");
    let mut s = MpiFileSession::create(comm, &path, &MpiIoOptions::default()).unwrap();
    if comm.rank() == 0 {
        std::fs::remove_file(path).unwrap();
        std::fs::remove_dir(directory).unwrap();
    }
    comm.barrier();
    let fault: u8 = std::env::var("PENCIL_MPI_SESSION_CLEANUP_CHILD")
        .unwrap()
        .parse()
        .unwrap();
    if comm.rank() == 0 {
        ffi::FAIL_SESSION_CLEANUP.with(|f| f.set(fault));
    }
    eprintln!("SESSION_CLEANUP_REACHED");
    if fault == 1 {
        s.write_named("fault", source.view()).unwrap();
    } else {
        s.close().unwrap();
    }
    panic!("native cleanup failure must fail-stop");
}

#[cfg(test)]
fn test_skew_view(file: ffi::MPI_File, info: ffi::MPI_Info) {
    assert_eq!(
        ffi::file_set_view_with_info(file, 7, ffi::byte_datatype(), info),
        ffi::MPI_SUCCESS as i32
    );
    assert_eq!(ffi::test_file_byte_offsets(file), [7, 8]);
}

// Called from the existing single-MPI-initialization private aggregation test.
#[cfg(test)]
pub(crate) fn test_retained_native_handles(
    directory: &Path,
    source: &pencil_array::PencilArray<i32, 2, 2>,
    destination: &mut pencil_array::PencilArray<i32, 2, 2>,
) {
    let comm = source.pencil().topology().communicator();
    let options = MpiIoOptions::default();
    let path = directory.join("session-native.pio");
    let counts = || ffi::FILE_CALLS.with(std::cell::Cell::get);
    let before = counts();
    let duplicates = ffi::DUPLICATE_CALLS.with(std::cell::Cell::get);
    // A single invalid rank must reject before any duplicate or native open.
    for invalid in [
        String::new(),
        "bad\0path".into(),
        "x".repeat(crate::MAX_HEADER_BYTES + 1),
    ] {
        let candidate = if comm.rank() == 0 {
            Path::new(&invalid)
        } else {
            &path
        };
        for op in [CREATE, OPEN_READ, OPEN_APPEND] {
            assert!(MpiFileSession::open_session(comm, candidate, &options, op).is_err());
            assert_eq!(counts(), before);
            assert_eq!(ffi::DUPLICATE_CALLS.with(std::cell::Cell::get), duplicates);
        }
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let raw_path = directory.join(std::ffi::OsStr::from_bytes(b"session-\xff.pio"));
        let mut raw = MpiFileSession::create(comm, &raw_path, &options).unwrap();
        raw.close().unwrap();
        let mut raw = MpiFileSession::open_read(comm, &raw_path, &options).unwrap();
        raw.close().unwrap();
    }
    let before = counts();
    let duplicates = ffi::DUPLICATE_CALLS.with(std::cell::Cell::get);
    let mut s = MpiFileSession::create(comm, &path, &options).unwrap();
    assert_eq!(counts()[0], before[0] + 1);
    assert_eq!(counts()[1], before[1]);
    assert_eq!(
        ffi::DUPLICATE_CALLS.with(std::cell::Cell::get),
        duplicates + 1
    );
    test_skew_view(s.file(), s.info_raw());
    assert!(s.catalog().unwrap().is_empty());
    for name in ["first", "second", "third"] {
        test_skew_view(s.file(), s.info_raw());
        s.write_named(name, source.view()).unwrap();
        assert_eq!(ffi::test_file_byte_offsets(s.file()), [0, 1]);
        test_skew_view(s.file(), s.info_raw());
        s.read_named(name, destination.view_mut()).unwrap();
        assert_eq!(destination.as_slice(), source.as_slice());
        assert_eq!(ffi::test_file_byte_offsets(s.file()), [0, 1]);
        let payloads = ffi::test_payload_calls();
        test_skew_view(s.file(), s.info_raw());
        assert!(!s.catalog().unwrap().is_empty());
        assert_eq!(ffi::test_payload_calls(), payloads);
        assert_eq!(counts()[0], before[0] + 1);
        assert_eq!(counts()[1], before[1]);
    }
    assert_eq!(
        counts()[2] - before[2],
        if source.is_empty() { 0 } else { 6 }
    );
    ffi::FAIL_FILE_SYNC.with(|f| f.set(comm.rank() == 0));
    assert!(s.flush().is_err());
    assert!(!s.poisoned);
    s.flush().unwrap();
    assert_eq!(s.catalog().unwrap().len(), 3);
    s.read_named("first", destination.view_mut()).unwrap();
    assert_eq!(destination.as_slice(), source.as_slice());
    s.write_named("after-flush-retry", source.view()).unwrap();
    let no_payload = ffi::test_payload_calls();
    assert!(s.write_named("first", source.view()).is_err());
    assert!(s.write_named("", source.view()).is_err());
    assert!(s.read_named("missing", destination.view_mut()).is_err());
    assert_eq!(ffi::test_payload_calls(), no_payload);
    assert!(!s.poisoned);
    if comm.size() > 1 {
        s.options.mode = if comm.rank() == 0 {
            MpiIoMode::Independent
        } else {
            MpiIoMode::Collective
        };
        assert!(s.write_named("control-mismatch", source.view()).is_err());
        s.options.mode = MpiIoMode::Collective;
        assert!(!s.poisoned);
        s.poisoned = comm.rank() == 0;
        assert!(s.close().is_err());
        s.poisoned = false;
    }
    s.close().unwrap();
    s.close().unwrap();
    assert_eq!(counts()[1], before[1] + 1);
    drop(s);
    assert_eq!(counts()[1], before[1] + 1);

    let independent = MpiIoOptions::default()
        .mode(MpiIoMode::Independent)
        .hint("cb_buffer_size", "2097152");
    let payloads = ffi::test_payload_calls();
    let mut s = MpiFileSession::create(
        comm,
        directory.join("session-independent.pio"),
        &independent,
    )
    .unwrap();
    s.write_named("independent", source.view()).unwrap();
    s.read_named("independent", destination.view_mut()).unwrap();
    assert_eq!(destination.as_slice(), source.as_slice());
    assert_eq!(&ffi::test_info_arguments()[..2], &[2097152, 2097152]);
    assert_eq!(ffi::test_payload_calls()[0], payloads[0]);
    assert_eq!(ffi::test_payload_calls()[1], payloads[1] + 1);
    assert_eq!(ffi::test_payload_calls()[2], payloads[2]);
    assert_eq!(ffi::test_payload_calls()[3], payloads[3] + 1);
    s.close().unwrap();

    for op in [OPEN_READ, OPEN_APPEND] {
        SKEW_OPEN_VIEW.with(|f| f.set(true));
        let mut opened = MpiFileSession::open_session(comm, &path, &options, op).unwrap();
        assert_eq!(ffi::test_file_byte_offsets(opened.file()), [0, 1]);
        assert_eq!(opened.catalog().unwrap().len(), 4);
        opened.close().unwrap();
    }

    // A real native reset and datatype release run before the injected one-rank
    // reported failure. No destination commit may escape the cleanup agreement.
    let mut s = MpiFileSession::open_read(comm, &path, &options).unwrap();
    destination.as_mut_slice().fill(-876);
    let untouched = destination.as_slice().to_vec();
    ffi::FAIL_BYTE_RESET.with(|f| f.set(if comm.rank() == 0 { 2 } else { 0 }));
    let before_cleanup = counts()[2];
    assert!(s.read_named("first", destination.view_mut()).is_err());
    assert_eq!(destination.as_slice(), untouched);
    assert_eq!(
        counts()[2] - before_cleanup,
        usize::from(!source.is_empty())
    );
    assert!(matches!(s.catalog(), Err(MpiSessionError::Poisoned)));
    s.close().unwrap();

    // Fault after an actual native header write/sync: poison, retain handles for
    // explicit close, and leave the older committed prefix readable.
    let mut s = MpiFileSession::open_append(comm, &path, &options).unwrap();
    ffi::FAIL_FILE_SYNC.with(|f| f.set(comm.rank() == 0));
    assert!(matches!(
        s.write_named("uncertain", source.view()),
        Err(MpiSessionError::Io(IoError::CommitUncertain { .. }))
    ));
    assert!(matches!(s.flush(), Err(MpiSessionError::Poisoned)));
    s.close().unwrap();
    let mut s = MpiFileSession::open_read(comm, &path, &options).unwrap();
    s.read_named("first", destination.view_mut()).unwrap();
    assert_eq!(destination.as_slice(), source.as_slice());
    assert!(s.catalog().is_err());
    s.close().unwrap();
    let before = counts();
    assert!(MpiFileSession::open_append(comm, &path, &options).is_err());
    assert_eq!(counts()[0], before[0] + 1);
    assert_eq!(counts()[1], before[1] + 1);

    // Exhaustion is decided and broadcast by root, before any native open.
    let saved = if comm.rank() == 0 {
        NEXT_ID.swap(u64::MAX, Ordering::Relaxed)
    } else {
        0
    };
    let before = counts();
    assert!(matches!(
        MpiFileSession::open_read(comm, &path, &options),
        Err(MpiSessionError::IdExhausted)
    ));
    assert_eq!(counts(), before);
    if comm.rank() == 0 {
        NEXT_ID.store(saved, Ordering::Relaxed);
    }
    let mut s = MpiFileSession::open_read(comm, &path, &options).unwrap();
    s.close().unwrap();
    let saved_nonroot = if comm.rank() != 0 {
        NEXT_ID.swap(u64::MAX, Ordering::Relaxed)
    } else {
        0
    };
    let mut s = MpiFileSession::open_read(comm, &path, &options).unwrap();
    s.close().unwrap();
    if comm.rank() != 0 {
        NEXT_ID.store(saved_nonroot, Ordering::Relaxed);
    }
    if comm.rank() == 0 {
        println!("MPI_SESSION_NATIVE_CHECKS_OK");
    }
}
