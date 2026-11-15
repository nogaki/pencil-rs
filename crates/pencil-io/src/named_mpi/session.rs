//! Retained native handles for the existing named-v2 format.
use super::*;
use crate::format::prepare_physical_decode;
use crate::mpi_io::{
    CommGuard, FileGuard, ReadPhases, WritePhases, abort_unrecoverable, collective_state,
    read_payload, write_payload_and_commit,
};
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
/// The MPI local payload limit is 2147483647 bytes/rank and applies to each
/// `write_named`/`read_named` record, not the whole container. A violation is
/// returned on every rank as `MpiSessionError::Named(NamedIoError::Io(
/// IoError::SizeLimit { what: "MPI local byte count" }))` before packing or
/// staging, and the session remains usable for a later operation.
///
/// Reads decode staged bytes into the destination after operation datatype/view
/// cleanup and collective agreement, **before** the eventual file close.
/// Legacy path reads still close their file before committing. Read-only opens
/// expose the committed prefix;
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
    fixed_options: MpiIoOptions,
    id: u64,
    readonly: bool,
    poisoned: bool,
}

fn entry(comm: &CartesianCommunicator, op: u64) -> Result<(), MpiSessionError> {
    // The first two collectives MUST match all existing MPI I/O entry points.
    let fixed = [NAMESPACE, op, VERSION, 0, 0];
    crate::options::agree_values(comm, &fixed)?;
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

#[cfg(test)]
fn test_skew_view(file: ffi::MPI_File, info: ffi::MPI_Info) {
    assert_eq!(
        ffi::file_set_view_with_info(file, 7, ffi::byte_datatype(), info),
        ffi::MPI_SUCCESS as i32
    );
    assert_eq!(ffi::test_file_byte_offsets(file), [7, 8]);
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
            !bytes.is_empty() && !bytes.contains(&0) && bytes.len() <= crate::MAX_DESCRIPTOR_BYTES,
            "session path validation",
        )?;
        let pc = prepared(comm, path_c(path), "session path preparation")?;
        crate::catalog::agree_bytes(comm, pc.as_bytes())?;
        // Only root advances the counter; zero is the broadcast exhaustion signal.
        let mut id = 0;
        if comm.rank() == 0 {
            id = NEXT_ID
                .try_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
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
            fixed_options: options.clone(),
            id,
            readonly: op == OPEN_READ,
            poisoned: false,
        })
    }
    fn begin(&self, op: u64) -> Result<(), MpiSessionError> {
        entry(self.comm, op)?;
        crate::options::agree_values(
            self.comm,
            &[
                self.id,
                u64::from(self.readonly),
                u64::from(self.resources.is_none()),
                u64::from(self.poisoned),
                u64::from(self.options == self.fixed_options),
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
            pencil.decomposition(),
            pencil.permutation().axes(),
            code,
            width,
        )?;
        let axes: [u64; M] = std::array::from_fn(|i| pencil.decomposition()[i].index() as u64);
        crate::options::agree_values(self.comm, &axes)?;
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
    /// Applies the same per-record 2147483647 bytes/rank local payload limit as
    /// the session.
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
        let layout_result = build_layout(
            view.pencil().global_shape(),
            view.extra_shape().dimensions(),
            &view.local_spatial_shape(),
            view.pencil().local_ranges(),
            view.len(),
            T::WIDTH,
        );
        if let Err(agreement) =
            agree_mpi_preparation(comm, layout_result.as_ref().err(), "session write layout")
        {
            return Err(NamedIoError::Io(agreement).into());
        }
        let layout = layout_result.map_err(NamedIoError::Io)?;
        let packed = prepared(comm, pack_view(&view), "session payload packing")?;
        // Keep the injected reset fault scoped to this write, not its catalog scan.
        #[cfg(test)]
        let deferred_view_fault = ffi::FAIL_BYTE_RESET.with(|fault| fault.replace(0));
        let scan = self.scan();
        #[cfg(test)]
        ffi::FAIL_BYTE_RESET.with(|fault| fault.set(deferred_view_fault));
        let (records, tail, size) = scan?;
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
        let mut marker_attempted = false;
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
            if let Err(error) = write_payload_and_commit(
                comm,
                file,
                self.info_raw(),
                self.options.mode,
                (tail + head.len()) as i64,
                &packed,
                ft,
                None,
                (end - 8) as i64,
                WritePhases {
                    payload_view: "session write view",
                    payload_write: "session payload write",
                    payload_flush: "session payload flush",
                    marker_view: "session byte view reset",
                    marker_write: "named metadata write",
                    marker_flush: "session marker flush",
                },
                &mut marker_attempted,
            ) {
                return Err(crate::mpi_io::write_payload_error(error).into());
            }
            Ok(())
        })();
        let cleanup = self.finish_operation(datatype);
        if mutated && (result.is_err() || cleanup.is_err()) {
            self.poisoned = true;
        }
        result?;
        if marker_attempted && cleanup.is_err() {
            return Err(IoError::CommitUncertain {
                stage: "session append or operation cleanup",
            }
            .into());
        }
        cleanup
    }
    /// Read a committed record, committing only after operation cleanup/agreement.
    /// The same per-record local payload limit applies before staging.
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
        let layout_result = build_layout(
            view.pencil().global_shape(),
            view.extra_shape().dimensions(),
            &view.local_spatial_shape(),
            view.pencil().local_ranges(),
            view.len(),
            T::WIDTH,
        );
        if let Err(agreement) =
            agree_mpi_preparation(comm, layout_result.as_ref().err(), "session read layout")
        {
            return Err(NamedIoError::Io(agreement).into());
        }
        let layout = layout_result.map_err(NamedIoError::Io)?;
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
        let result = (|| -> Result<_, MpiSessionError> {
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
            if let Err(error) = read_payload(
                comm,
                file,
                self.info_raw(),
                self.options.mode,
                record.payload_offset as i64,
                ft,
                &mut packed,
                ReadPhases {
                    view: "session read view",
                    payload: "session payload read",
                },
            ) {
                return Err(error.agreement.into());
            }
            Ok(prepared(
                comm,
                prepare_physical_decode(&view, &packed),
                "session physical staging",
            )?)
        })();
        let cleanup = self.finish_operation(datatype);
        let plan = result?;
        cleanup?;
        agree_phase(comm, true, "session read commit")?;
        plan.decode_into(&packed, view.as_mut_slice());
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
mod tests;
#[cfg(test)]
pub(crate) use tests::{test_native_cleanup_child, test_retained_native_handles};
