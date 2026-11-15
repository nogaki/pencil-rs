#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

//! Native MPI-IO and HDF5 storage.
//!
//! # Features
//!
//! - `mpi` (default): collective I/O for `pencil_array` views; requires native MPI.
//! - `hdf5`: serial HDF5 over flat slices. Use `default-features = false` with
//!   `features = ["hdf5"]` to exclude MPI and `pencil-array`. An installed HDF5
//!   library and headers are still required; use a serial HDF5 build to avoid
//!   native MPI dependencies too.
//! - `parallel-hdf5`: enables `mpi`, `hdf5`, and collective HDF5; requires a
//!   parallel HDF5 installation built against the same MPI implementation.
//!
//! # Serial HDF5 (`hdf5`)
//!
//! `write_hdf5_serial(path, global_shape, extra_shape, values)` and
//! `read_hdf5_serial(path, global_shape, extra_shape, destination)` accept shape
//! slices and a full flat `IoElement` buffer in row-major `[extra..., spatial...]`
//! order. They use the same single-dataset format at `/pencil_io_v1/data` as
//! parallel HDF5, supporting `i8`/`u8`, `i16`/`u16`, `i32`/`u32`, `i64`/`u64`,
//! `f32`/`f64`, and `num_complex::Complex<f32>`/`Complex<f64>`.
//! Writes exclusively create a new file (no overwrite); a returned read error
//! leaves the destination unchanged. Serial APIs do not provide named datasets,
//! collections, catalogs, sessions, or raw input.
//!
//! ```no_run
//! # #[cfg(feature = "hdf5")]
//! # fn main() -> Result<(), pencil_io::IoError> {
//! use pencil_io::{read_hdf5_serial, write_hdf5_serial};
//!
//! let global_shape = [2, 3];
//! let extra_shape = [2];
//! let values: Vec<f64> = (0..12).map(f64::from).collect();
//! // Choose a path that does not already exist.
//! write_hdf5_serial("field.h5", &global_shape, &extra_shape, &values)?;
//! let mut destination = vec![0.0; values.len()];
//! read_hdf5_serial("field.h5", &global_shape, &extra_shape, &mut destination)?;
//! assert_eq!(destination, values);
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "hdf5"))]
//! # fn main() {}
//! ```
//!
//! # Collective I/O (`mpi`)
//!
//! The original `write_mpi` / `read_mpi` APIs store one versioned
//! little-endian row-major payload per file. Local buffers are packed in logical
//! `[extra..., spatial...]` order;
//! the packing is local and never gathers data at rank zero.  Readers may use
//! a different process grid or memory-axis permutation.
//!
//! MPI local payloads are limited to 2147483647 bytes/rank, checked as
//! `local_elements * scalar_width` (for `f64`, up to 268435455 elements). If
//! any rank exceeds it, every rank returns the same underlying
//! [`IoError::SizeLimit`] with `what == "MPI local byte count"` (wrapped by the
//! API's error type) before packing or staging that payload.
//! Collections apply the limit to their combined payload and sessions to each
//! record; this is not a whole-file or container byte cap. Existing dimension,
//! start and offset representation limits remain. Normal successful paths still
//! pack the full local payload and allocate byte or typed staging. HDF5 and
//! metadata-only catalog operations are not subject to this MPI local byte
//! limit.
//!
//! `write_mpi` and `read_mpi` are collective over the view's Cartesian
//! communicator. Views may borrow an owning `pencil_array::PencilArray` or
//! validated external slices. Every rank must pass the same global shape, extra
//! shape, element type, process grid, ordered decomposition (including axes whose
//! grid extent is one), memory permutation, and path. When every rank builds its
//! agreement descriptor and passes the agreement's size and allocation checks,
//! any such difference makes all ranks return
//! `IoError::CollectiveDescriptorMismatch`; single-call APIs return before the
//! file is opened, and persistent sessions return before the member is accessed.
//! Failures before that comparison, such as an invalid path, the 64 KiB per-rank
//! or 64 MiB total descriptor limit, or an allocation failure, are reported
//! instead and can differ between ranks. Writer and reader may still use
//! different grids or decompositions, and the named and parallel HDF5 APIs apply
//! the same rule.
//!
//! All array, topology, and MPI-IO objects must be
//! dropped before MPI finalization; persistent sessions require explicit close. Do not
//! overlap an I/O call with another operation on that topology communicator;
//! the call temporarily installs `MPI_ERRORS_RETURN` only around
//! `MPI_Comm_dup`, then restores the caller's original handler. Native MPI
//! process failure is not promised to recover as a Rust error. Unrecoverable
//! native cleanup failures that may leave a live or invalid handle are fail-stop
//! paths, distinct from ordinary returned cleanup/post-cleanup errors. Missing,
//! existing, and malformed-file errors return normally.
//! The optional `parallel-hdf5` feature provides the same two operations over
//! a native parallel HDF5 file.  It uses one versioned dataset at
//! `/pencil_io_v1/data`; that is a self-describing HDF5 representation, not a
//! binary-compatible Julia PencilIO custom format.
//!
//! Single-call `read_mpi` and `read_hdf5` reads publish the destination only after
//! staging, validation, explicit native cleanup, and collective success
//! agreement. A returned error leaves that destination unchanged.
//!
//! `write_mpi_named`, `append_mpi_named`, and `read_mpi_named` add a separate
//! append-only named-dataset container. Append creates a new name without
//! overwriting earlier committed records. The optional HDF5 named APIs likewise
//! use per-dataset metadata and commit markers. Names are UTF-8 keys rather than
//! filesystem paths; duplicate names fail before mutation. `NamedIoError`
//! preserves underlying native/commit errors. Neither format promises recovery
//! from process loss or crash-atomic HDF5 journaling.
//!
//! All ranks must participate even when an options API selects independent
//! payload transfers. Writers (including appenders) to the same file require
//! external serialization across jobs and communicators; no whole-file rollback
//! is promised after a write failure. Even a write error can leave a valid
//! commit; [`IoError::CommitUncertain`] is not a safe-retry guarantee. Raw input
//! is explicitly requested through `read_mpi_raw`, never inferred by the strict
//! versioned readers.

#[cfg(feature = "mpi")]
mod catalog;
#[cfg(feature = "mpi")]
mod chunked;
#[cfg(feature = "mpi")]
mod collections;
mod format;
#[cfg(feature = "hdf5")]
mod hdf5_schema;
#[cfg(feature = "mpi")]
mod mpi_io;
#[cfg(feature = "mpi")]
mod named_mpi;
#[cfg(feature = "mpi")]
mod options;
#[cfg(feature = "mpi")]
mod raw;
#[cfg(feature = "hdf5")]
mod serial_hdf5;

#[cfg(feature = "mpi")]
mod ffi;

#[cfg(feature = "parallel-hdf5")]
mod hdf5_io;
#[cfg(feature = "parallel-hdf5")]
mod hdf5_options;

#[cfg(feature = "mpi")]
pub use named_mpi::session::{MpiFileSession, MpiSessionError};

#[cfg(feature = "parallel-hdf5")]
pub use catalog::read_hdf5_catalog;
#[cfg(feature = "mpi")]
pub use catalog::{
    CatalogError, DatasetInfo, ScalarType, read_mpi_catalog, read_mpi_named_catalog,
};
#[cfg(feature = "mpi")]
pub use chunked::{read_mpi_chunked, read_mpi_chunked_catalog, write_mpi_chunked};
#[cfg(feature = "mpi")]
pub use collections::*;
pub use format::IoElement;
#[cfg(feature = "mpi")]
pub use mpi_io::{read_mpi, read_mpi_with_options, write_mpi, write_mpi_with_options};
#[cfg(feature = "mpi")]
pub use named_mpi::{
    append_mpi_named, append_mpi_named_with_options, read_mpi_named, read_mpi_named_with_options,
    write_mpi_named, write_mpi_named_with_options,
};
#[cfg(feature = "mpi")]
pub use options::{MpiIoMode, MpiIoOptions, RawByteOrder, RawReadOptions};
#[cfg(feature = "mpi")]
pub use raw::read_mpi_raw;
#[cfg(feature = "hdf5")]
pub use serial_hdf5::{read_hdf5_serial, write_hdf5_serial};

#[cfg(feature = "parallel-hdf5")]
pub use hdf5_io::{
    Hdf5FileSession, Hdf5SessionError, append_hdf5_named, append_hdf5_named_with_options,
    read_hdf5, read_hdf5_named, read_hdf5_named_with_options, write_hdf5, write_hdf5_named,
    write_hdf5_named_with_options,
};
#[cfg(feature = "parallel-hdf5")]
pub use hdf5_options::{
    Hdf5ReadOptions, Hdf5WriteOptions, read_hdf5_with_options, write_hdf5_with_options,
};

use thiserror::Error;

/// Errors returned by native I/O.
///
/// For collective I/O, variants can differ across ranks: a flush failure on rank
/// zero may report [`IoError::WriteIncomplete`] there and
/// [`IoError::CollectivePrecondition`] on peers.  The MPI local payload byte
/// limit is agreed collectively instead: every rank returns the same
/// [`IoError::SizeLimit`] with `what == "MPI local byte count"` before packing
/// or staging the payload.  That limit is per local payload, not a whole-file or
/// container cap.  All ranks must still follow the
/// same subsequent collective call order,
/// not branch independently on their local error variant.
#[derive(Debug, Clone, Error)]
pub enum IoError {
    /// A rank-local operation argument was invalid.
    #[error("invalid I/O argument: {0}")]
    InvalidInput(&'static str),

    /// A path contained an embedded NUL or could not be represented natively.
    #[error("path cannot be represented as a native path")]
    InvalidPath,

    /// Ranks supplied different operation descriptors.
    #[error("collective I/O descriptor differs between ranks")]
    CollectiveDescriptorMismatch,

    /// At least one rank failed a fallible phase before the next collective.
    #[error("collective I/O phase failed: {phase}")]
    CollectivePrecondition {
        /// The phase whose local validation failed on at least one rank.
        phase: &'static str,
    },

    /// A checked size calculation overflowed or exceeded the native limit.
    /// For MPI I/O, the `"MPI local byte count"` case is agreed by every rank
    /// before payload packing or staging allocation and is not a whole-file or
    /// container cap.
    #[error("I/O size or native count limit exceeded: {what}")]
    SizeLimit {
        /// The checked quantity that exceeded its native representation.
        what: &'static str,
    },

    /// A temporary or packing allocation failed.
    #[error("I/O allocation failed for {requested} bytes")]
    AllocationFailed {
        /// The requested byte count.
        requested: usize,
    },

    /// An MPI call failed.
    #[error("MPI operation {operation} failed with error code {code}")]
    Mpi {
        /// The MPI operation.
        operation: &'static str,
        /// The native MPI return code.
        code: i32,
    },

    /// A completed file did not have the requested metadata.
    #[error("file metadata mismatch: {field}")]
    MetadataMismatch {
        /// The metadata field that differed.
        field: &'static str,
    },

    /// A file was malformed, truncated, or had trailing bytes.
    #[error("invalid I/O file: {reason}")]
    InvalidFile {
        /// The structural file property that was invalid.
        reason: &'static str,
    },

    /// A file was observed before its commit marker.
    #[error("I/O file has no completed commit marker")]
    IncompleteFile,

    /// A native backend operation failed, including opening a file, creating
    /// native resources, or transferring data. This can occur after a write
    /// has started; this variant does not imply that the file is unchanged.
    #[error("native backend operation {operation} failed with code {code}")]
    Native {
        /// The native operation.
        operation: &'static str,
        /// The native library return code.
        code: i64,
    },

    /// A write failed before its marker was durably flushed.
    /// The file may already be modified; no rollback is implied.
    #[error("write is incomplete; the failed file was preserved ({stage})")]
    WriteIncomplete {
        /// The last write stage reached before failure.
        stage: &'static str,
    },

    /// The commit marker write or flush did not reach a known state.
    /// The file may contain a valid commit; this is not a guarantee of rollback
    /// or a safe retry.
    #[error("write commit state is uncertain; the failed file was preserved ({stage})")]
    CommitUncertain {
        /// The marker stage whose durability is unknown.
        stage: &'static str,
    },
}

/// Errors specific to named datasets.
#[cfg(feature = "mpi")]
#[derive(Debug, Clone, Error)]
pub enum NamedIoError {
    /// The name is empty, too long, or contains a NUL byte.
    #[error("invalid dataset name")]
    InvalidName,
    /// The requested name is already present.
    #[error("dataset name already exists")]
    DuplicateName,
    /// The requested name is not present.
    #[error("dataset name was not found")]
    NotFound,
    /// The append container has an incomplete or malformed tail.
    #[error("dataset container has an invalid tail")]
    InvalidTail,
    /// The underlying collective I/O operation failed.
    #[error(transparent)]
    Io(#[from] IoError),
}

/// The on-file format version used by both backends.
#[cfg(any(feature = "mpi", feature = "hdf5"))]
const FORMAT_VERSION: u64 = 1;

/// The I/O protocol namespace; array and FFT operation namespaces are unrelated.
#[cfg(feature = "mpi")]
const IO_NAMESPACE: u64 = 0x494f_0000;

/// Commit marker written only after payload and flush success.
#[cfg(any(feature = "mpi", feature = "hdf5"))]
const COMMIT_MARKER: u64 = 0x434f_4d4d_4954_5445;

/// Initial marker value.  It is deliberately not accepted by readers.
#[cfg(any(feature = "mpi", feature = "hdf5"))]
const INCOMPLETE_MARKER: u64 = 0x494e_434f_4d50_4c45;

/// Maximum descriptor/header size accepted before any file-controlled allocation.
#[cfg(feature = "mpi")]
const MAX_HEADER_BYTES: usize = 1024 * 1024;

/// Maximum collective descriptor size.  Paths and ranks are bounded before allgather.
#[cfg(feature = "mpi")]
const MAX_DESCRIPTOR_BYTES: usize = 64 * 1024;

/// MPI count and HDF5 dimensions use this conservative protocol limit.
#[cfg(any(feature = "mpi", feature = "hdf5"))]
const MAX_PROTOCOL_RANK: usize = 1024;

#[cfg(feature = "mpi")]
pub(crate) const MAX_NAME: usize = 1024;

#[cfg(feature = "mpi")]
pub(crate) fn valid_name_bytes(bytes: &[u8]) -> bool {
    !bytes.is_empty()
        && bytes.len() <= MAX_NAME
        && !bytes.contains(&0)
        && std::str::from_utf8(bytes).is_ok()
}

#[cfg(feature = "mpi")]
const OP_WRITE_MPI: u64 = 1;
#[cfg(feature = "mpi")]
const OP_READ_MPI: u64 = 2;
#[cfg(feature = "mpi")]
const OP_READ_RAW_MPI: u64 = 8;
#[cfg(feature = "parallel-hdf5")]
const OP_WRITE_HDF5: u64 = 3;
#[cfg(feature = "parallel-hdf5")]
const OP_READ_HDF5: u64 = 4;
#[cfg(feature = "parallel-hdf5")]
const OP_WRITE_HDF5_NAMED: u64 = 5;
#[cfg(feature = "parallel-hdf5")]
const OP_APPEND_HDF5_NAMED: u64 = 6;
#[cfg(feature = "parallel-hdf5")]
const OP_READ_HDF5_NAMED: u64 = 7;

#[cfg(all(test, feature = "mpi"))]
mod mpi_preflight_tests;

#[cfg(all(test, feature = "mpi"))]
mod tests;
