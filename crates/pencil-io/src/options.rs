use crate::mpi_io::{abort_unrecoverable, agree_phase};
use crate::{IoError, ffi};
use mpi::collective::{CommunicatorCollectives, SystemOperation};

/// Payload calls only; all ranks must still participate in metadata and cleanup.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MpiIoMode {
    /// Collective native payload transfer (the legacy default).
    #[default]
    Collective,
    /// Independent native payload transfer.
    Independent,
}

/// Native MPI-IO controls. Same-file writers require external serialization,
/// including writers using different communicators or jobs.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MpiIoOptions {
    pub(crate) mode: MpiIoMode,
    pub(crate) hints: Vec<(String, String)>,
}
impl MpiIoOptions {
    /// Select payload transfer mode.
    pub fn mode(mut self, mode: MpiIoMode) -> Self {
        self.mode = mode;
        self
    }
    /// Add a hint. Duplicate keys and invalid native strings are rejected collectively.
    pub fn hint(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.hints.push((key.into(), value.into()));
        self.hints.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        self
    }
}

/// Byte order of each scalar component in explicitly requested raw input.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RawByteOrder {
    /// Host byte order.
    #[default]
    Native,
    /// Little endian.
    Little,
    /// Big endian.
    Big,
}
impl RawByteOrder {
    pub(crate) fn effective(self) -> Self {
        match self {
            Self::Native if cfg!(target_endian = "little") => Self::Little,
            Self::Native => Self::Big,
            value => value,
        }
    }
}
/// Explicit raw-input controls; no format, type or Julia-wire detection occurs.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RawReadOptions {
    pub(crate) offset: u64,
    pub(crate) endian: RawByteOrder,
    pub(crate) mpi: MpiIoOptions,
}
impl RawReadOptions {
    /// Set the byte displacement from the start of the file.
    pub fn byte_offset(mut self, offset: u64) -> Self {
        self.offset = offset;
        self
    }
    /// Set the byte order of every real or complex component.
    pub fn byte_order(mut self, order: RawByteOrder) -> Self {
        self.endian = order;
        self
    }
    /// Set MPI controls.
    pub fn mpi_options(mut self, options: MpiIoOptions) -> Self {
        self.mpi = options;
        self
    }
    /// Select payload transfer mode.
    pub fn mode(mut self, mode: MpiIoMode) -> Self {
        self.mpi.mode = mode;
        self
    }
    /// Add a native MPI hint.
    pub fn hint(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.mpi = self.mpi.hint(key, value);
        self
    }
}

// Called only after the common five-word header has agreed the operation.
pub(crate) fn agree_options<C: CommunicatorCollectives>(
    comm: &C,
    options: &MpiIoOptions,
    extra: &[u64],
) -> Result<(), IoError> {
    agree_controls(comm, options.mode, &options.hints, extra)
}

pub(crate) fn agree_controls<C: CommunicatorCollectives>(
    comm: &C,
    mode: MpiIoMode,
    hints: &[(String, String)],
    extra: &[u64],
) -> Result<(), IoError> {
    let descriptor = (|| {
        let mut bytes = Vec::new();
        let size = hints
            .iter()
            .try_fold(16 + extra.len() * 8, |n, (k, v)| {
                n.checked_add(16 + k.len() + v.len())
            })
            .ok_or(IoError::SizeLimit {
                what: "options descriptor",
            })?;
        if size > crate::MAX_DESCRIPTOR_BYTES {
            return Err(IoError::SizeLimit {
                what: "options descriptor",
            });
        }
        bytes
            .try_reserve_exact(size)
            .map_err(|_| IoError::AllocationFailed { requested: size })?;
        bytes.extend_from_slice(&(mode as u64).to_le_bytes());
        bytes.extend_from_slice(&(hints.len() as u64).to_le_bytes());
        for &x in extra {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        for (i, (key, value)) in hints.iter().enumerate() {
            if key.is_empty()
                || value.is_empty()
                || key.len() >= mpi::ffi::MPI_MAX_INFO_KEY as usize
                || value.len() >= mpi::ffi::MPI_MAX_INFO_VAL as usize
                || key.contains('\0')
                || value.contains('\0')
                || (i > 0 && hints[i - 1].0 >= *key)
            {
                return Err(IoError::InvalidInput("invalid or duplicate MPI hint"));
            }
            for s in [key, value] {
                bytes.extend_from_slice(&(s.len() as u64).to_le_bytes());
                bytes.extend_from_slice(s.as_bytes());
            }
        }
        Ok(bytes)
    })();
    if let Err(e) = agree_phase(comm, descriptor.is_ok(), "options preparation") {
        return Err(descriptor.err().unwrap_or(e));
    }
    let bytes = descriptor?;
    let length = bytes.len() as u64;
    let (mut min, mut max) = (0u64, 0u64);
    comm.all_reduce_into(&length, &mut min, SystemOperation::min());
    comm.all_reduce_into(&length, &mut max, SystemOperation::max());
    if min != max {
        return Err(IoError::CollectiveDescriptorMismatch);
    }
    let total = bytes
        .len()
        .checked_mul(comm.size() as usize)
        .filter(|&n| n <= 64 * 1024 * 1024)
        .ok_or(IoError::SizeLimit {
            what: "options allgather",
        })?;
    let mut all = Vec::new();
    let reserve = all.try_reserve_exact(total);
    if let Err(e) = agree_phase(comm, reserve.is_ok(), "options allocation") {
        return Err(if reserve.is_err() {
            IoError::AllocationFailed { requested: total }
        } else {
            e
        });
    }
    all.resize(total, 0);
    comm.all_gather_into(&bytes[..], &mut all[..]);
    if all.chunks_exact(bytes.len()).any(|b| b != bytes) {
        return Err(IoError::CollectiveDescriptorMismatch);
    }
    Ok(())
}

// The legacy descriptor omits decomposition axes. New entry points agree them
// separately without changing the legacy header or collective sequence.
pub(crate) fn agree_decomposition<const N: usize, const M: usize>(
    pencil: &pencil_array::Pencil<N, M>,
) -> Result<(), IoError> {
    let axes: [u64; M] = std::array::from_fn(|i| pencil.decomposition()[i].index() as u64);
    agree_controls(
        pencil.topology().communicator(),
        MpiIoMode::Collective,
        &[],
        &axes,
    )
}

pub(crate) struct InfoGuard {
    pub(crate) raw: ffi::MPI_Info,
    comm: ffi::MPI_Comm,
}
impl InfoGuard {
    pub(crate) fn new<C: CommunicatorCollectives>(
        comm: &C,
        options: &MpiIoOptions,
    ) -> Result<Self, IoError> {
        Self::from_hints(comm, &options.hints)
    }
    pub(crate) fn from_hints<C: CommunicatorCollectives>(
        comm: &C,
        hints: &[(String, String)],
    ) -> Result<Self, IoError> {
        let local =
            ffi::info_from_hints(hints.iter().map(|(k, v)| (k, v)), comm.as_raw()).map(|raw| {
                Self {
                    raw,
                    comm: comm.as_raw(),
                }
            });
        if let Err(e) = agree_phase(comm, local.is_ok(), "MPI_Info creation") {
            return Err(local
                .err()
                .map(|code| IoError::Mpi {
                    operation: "MPI_Info",
                    code,
                })
                .unwrap_or(e));
        }
        local.map_err(|code| IoError::Mpi {
            operation: "MPI_Info",
            code,
        })
    }
}
impl Drop for InfoGuard {
    fn drop(&mut self) {
        if ffi::info_free(&mut self.raw) != ffi::MPI_SUCCESS as i32 {
            abort_unrecoverable(self.comm, "MPI_Info_free");
        }
    }
}
