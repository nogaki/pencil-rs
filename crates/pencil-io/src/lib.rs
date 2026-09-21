#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

//! Collective native MPI-IO for [`pencil_array`] views.
//!
//! The original [`write_mpi`] / [`read_mpi`] APIs store one versioned
//! little-endian row-major payload per file. Local buffers are packed in logical
//! `[extra..., spatial...]` order;
//! the packing is local and never gathers data at rank zero.  Readers may use
//! a different process grid or memory-axis permutation.
//!
//! `write_mpi` and `read_mpi` are collective over the view's Cartesian
//! communicator.  The view must be borrowed from its owning
//! [`pencil_array::PencilArray`] with `.view()` or `.view_mut()` and all array,
//! topology, and MPI-IO objects must be dropped before MPI finalization. Do not
//! overlap an I/O call with another operation on that topology communicator;
//! the call temporarily installs `MPI_ERRORS_RETURN` only around
//! `MPI_Comm_dup`, then restores the caller's original handler. Native MPI
//! process failure is not promised to recover as a Rust error. Native cleanup
//! failures that may leave a live or invalid handle are documented fail-stop
//! paths; ordinary missing, existing, and malformed-file errors return normally.
//! The optional `parallel-hdf5` feature provides the same two operations over
//! a native parallel HDF5 file.  It uses one versioned dataset at
//! `/pencil_io_v1/data`; that is a self-describing HDF5 representation, not a
//! binary-compatible Julia PencilIO custom format.
//!
//! [`write_mpi_named`], [`append_mpi_named`], and [`read_mpi_named`] add a separate
//! append-only named-dataset container. Append creates a new name without
//! overwriting earlier committed records. The optional HDF5 named APIs likewise
//! use per-dataset metadata and commit markers. Names are UTF-8 keys rather than
//! filesystem paths; duplicate names fail before mutation. [`NamedIoError`]
//! preserves underlying native/commit errors. Neither format promises recovery
//! from process loss or crash-atomic HDF5 journaling.

mod format;
mod mpi_io;
mod named_mpi;

mod ffi;

#[cfg(feature = "parallel-hdf5")]
mod hdf5_io;

pub use format::IoElement;
pub use mpi_io::{read_mpi, write_mpi};
pub use named_mpi::{append_mpi_named, read_mpi_named, write_mpi_named};

#[cfg(feature = "parallel-hdf5")]
pub use hdf5_io::{append_hdf5_named, read_hdf5, read_hdf5_named, write_hdf5, write_hdf5_named};

use thiserror::Error;

/// Errors returned by collective native I/O.
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

    /// The file could not be opened or a required native backend was absent.
    #[error("native backend operation {operation} failed with code {code}")]
    Native {
        /// The native operation.
        operation: &'static str,
        /// The native library return code.
        code: i64,
    },

    /// A write failed before its marker was durably flushed.
    #[error("write is incomplete; the failed file was preserved ({stage})")]
    WriteIncomplete {
        /// The last write stage reached before failure.
        stage: &'static str,
    },

    /// The commit marker write or flush did not reach a known state.
    #[error("write commit state is uncertain; the failed file was preserved ({stage})")]
    CommitUncertain {
        /// The marker stage whose durability is unknown.
        stage: &'static str,
    },
}

/// Errors specific to named datasets.
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
const FORMAT_VERSION: u64 = 1;

/// The I/O protocol namespace; array and FFT operation namespaces are unrelated.
const IO_NAMESPACE: u64 = 0x494f_0000;

/// Commit marker written only after payload and flush success.
const COMMIT_MARKER: u64 = 0x434f_4d4d_4954_5445;

/// Initial marker value.  It is deliberately not accepted by readers.
const INCOMPLETE_MARKER: u64 = 0x494e_434f_4d50_4c45;

/// Maximum descriptor/header size accepted before any file-controlled allocation.
const MAX_HEADER_BYTES: usize = 1024 * 1024;

/// Maximum collective descriptor size.  Paths and ranks are bounded before allgather.
const MAX_DESCRIPTOR_BYTES: usize = 64 * 1024;

/// MPI count and HDF5 dimensions use this conservative protocol limit.
const MAX_PROTOCOL_RANK: usize = 1024;

const OP_WRITE_MPI: u64 = 1;
const OP_READ_MPI: u64 = 2;
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

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    use mpi::collective::{Root, SystemOperation};
    use mpi::raw::AsRaw;
    use mpi::traits::{Communicator, CommunicatorCollectives};
    use pencil_array::{ExtraShape, MpiTopology, Pencil, PencilArray};

    use super::{IoError, read_mpi, write_mpi};

    fn root_status<C, F>(world: &C, operation: F)
    where
        C: CommunicatorCollectives,
        F: FnOnce() -> Result<(), String>,
    {
        let result = if world.rank() == 0 {
            operation()
        } else {
            Ok(())
        };
        let local_ok = i32::from(result.is_ok());
        let mut all_ok = 0;
        world.all_reduce_into(&local_ok, &mut all_ok, SystemOperation::min());
        assert_eq!(all_ok, 1, "root test setup failed: {:?}", result.err());
    }

    fn reset(path: &Path) -> Result<(), String> {
        match std::fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error.to_string()),
        }
    }

    fn owned_temp_dir<C>(world: &C, prefix: &str) -> PathBuf
    where
        C: Communicator + CommunicatorCollectives,
    {
        let setup = if world.rank() == 0 {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|error| error.to_string())
                .and_then(|nanos| {
                    let pid = std::process::id();
                    let mut attempt = 0u64;
                    loop {
                        let path = std::env::temp_dir()
                            .join(format!("{prefix}-{}-{pid}-{attempt}", nanos.as_nanos()));
                        match std::fs::create_dir(&path) {
                            Ok(()) => {
                                return path.to_str().map(str::to_owned).ok_or_else(|| {
                                    let _ = std::fs::remove_dir_all(&path);
                                    "temporary path is not valid UTF-8".to_owned()
                                });
                            }
                            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                                attempt = attempt.saturating_add(1);
                            }
                            Err(error) => return Err(error.to_string()),
                        }
                    }
                })
        } else {
            Ok(String::new())
        };
        let mut status = i32::from(setup.is_ok());
        world.process_at_rank(0).broadcast_into(&mut status);
        let mut all_ok = 0;
        world.all_reduce_into(&status, &mut all_ok, SystemOperation::min());
        if all_ok != 1 {
            panic!(
                "owned private-test directory setup failed: {}",
                setup.err().unwrap_or_default()
            );
        }

        let mut path = if world.rank() == 0 {
            setup
                .expect("directory setup agreement established")
                .into_bytes()
        } else {
            Vec::new()
        };
        let mut length = i32::try_from(path.len()).expect("temporary path fits MPI count");
        world.process_at_rank(0).broadcast_into(&mut length);
        path.resize(
            usize::try_from(length).expect("temporary path length is non-negative"),
            0,
        );
        world.process_at_rank(0).broadcast_into(&mut path[..]);
        PathBuf::from(String::from_utf8(path).expect("temporary path is UTF-8"))
    }

    fn cleanup_owned_temp_dir<C>(world: &C, path: &Path)
    where
        C: Communicator + CommunicatorCollectives,
    {
        world.barrier();
        let cleanup = if world.rank() == 0 {
            std::fs::remove_dir_all(path).map_err(|error| error.to_string())
        } else {
            Ok(())
        };
        let mut status = i32::from(cleanup.is_ok());
        world.process_at_rank(0).broadcast_into(&mut status);
        let mut all_ok = 0;
        world.all_reduce_into(&status, &mut all_ok, SystemOperation::min());
        if all_ok != 1 {
            panic!(
                "owned private-test directory cleanup failed: {}",
                cleanup.err().unwrap_or_default()
            );
        }
        world.barrier();
    }

    #[test]
    fn post_cleanup_errors_preserve_destination_and_valid_commits() {
        let universe = mpi::initialize().expect("MPI must initialize once");
        let world = universe.world();
        let size = usize::try_from(world.size()).unwrap();
        let directory = owned_temp_dir(&world, "pencil-io-private-hook");

        let writer_topology = MpiTopology::<2>::new(&world, [size, 1]).unwrap();
        let writer_pencil = Pencil::<2, 2>::new(writer_topology, [4, 5], [0, 1]).unwrap();
        let mut source =
            PencilArray::from_elem(writer_pencil, ExtraShape::scalar(), -1i32).unwrap();
        {
            let mut view = source.view_mut();
            let ranges = view.pencil().local_ranges().clone();
            for x in 0..ranges[0].len() {
                for y in 0..ranges[1].len() {
                    *view.get_local_mut(&[], [x, y]).unwrap() =
                        (100 * (ranges[0].start + x) + ranges[1].start + y) as i32;
                }
            }
        }

        let valid_path = directory.join("valid.pio");
        root_status(&world, || reset(&valid_path));
        world.barrier();
        let handler_before = crate::ffi::test_comm_errhandler_token(
            source.pencil().topology().communicator().as_raw(),
        )
        .unwrap();
        write_mpi(&valid_path, source.view()).unwrap();
        let handler_after = crate::ffi::test_comm_errhandler_token(
            source.pencil().topology().communicator().as_raw(),
        )
        .unwrap();
        assert_eq!(handler_before, handler_after);
        world.barrier();

        let reader_topology = MpiTopology::<2>::new(&world, [size, 1]).unwrap();
        let reader_pencil = Pencil::<2, 2>::new(reader_topology, [4, 5], [0, 1]).unwrap();
        let mut destination =
            PencilArray::from_elem(reader_pencil, ExtraShape::scalar(), -777i32).unwrap();
        let before = destination.as_slice().to_vec();
        let handler_before = crate::ffi::test_comm_errhandler_token(
            destination.pencil().topology().communicator().as_raw(),
        )
        .unwrap();
        let error =
            crate::mpi_io::read_mpi_with_post_cleanup_failure(&valid_path, destination.view_mut())
                .expect_err("synthetic post-cleanup read failure");
        let handler_after = crate::ffi::test_comm_errhandler_token(
            destination.pencil().topology().communicator().as_raw(),
        )
        .unwrap();
        assert_eq!(handler_before, handler_after);
        if world.rank() == 0 {
            assert!(matches!(
                error,
                IoError::Native {
                    operation: "test post-cleanup result",
                    code: -1
                }
            ));
        } else {
            assert!(matches!(
                error,
                IoError::CollectivePrecondition {
                    phase: "post-cleanup result"
                }
            ));
        }
        assert_eq!(destination.as_slice(), before.as_slice());
        world.barrier();

        let handler_before = crate::ffi::test_comm_errhandler_token(
            destination.pencil().topology().communicator().as_raw(),
        )
        .unwrap();
        read_mpi(&valid_path, destination.view_mut()).unwrap();
        let handler_after = crate::ffi::test_comm_errhandler_token(
            destination.pencil().topology().communicator().as_raw(),
        )
        .unwrap();
        assert_eq!(handler_before, handler_after);
        assert_i32_values(&destination);
        world.barrier();

        let uncertain_path = directory.join("uncertain.pio");
        root_status(&world, || reset(&uncertain_path));
        world.barrier();
        let handler_before = crate::ffi::test_comm_errhandler_token(
            source.pencil().topology().communicator().as_raw(),
        )
        .unwrap();
        let error =
            crate::mpi_io::write_mpi_with_postmarker_uncertainty(&uncertain_path, source.view())
                .expect_err("synthetic post-marker uncertainty");
        let handler_after = crate::ffi::test_comm_errhandler_token(
            source.pencil().topology().communicator().as_raw(),
        )
        .unwrap();
        assert_eq!(handler_before, handler_after);
        let local_ok = i32::from(if world.rank() == 0 {
            matches!(
                &error,
                IoError::CommitUncertain {
                    stage: "test post-marker cleanup result"
                }
            )
        } else {
            matches!(
                &error,
                IoError::CollectivePrecondition {
                    phase: "post-cleanup result"
                }
            )
        });
        let mut all_ok = 0;
        world.all_reduce_into(&local_ok, &mut all_ok, SystemOperation::min());
        assert_eq!(all_ok, 1, "post-marker error was not preserved");
        world.barrier();
        let handler_before = crate::ffi::test_comm_errhandler_token(
            destination.pencil().topology().communicator().as_raw(),
        )
        .unwrap();
        read_mpi(&uncertain_path, destination.view_mut()).unwrap();
        let handler_after = crate::ffi::test_comm_errhandler_token(
            destination.pencil().topology().communicator().as_raw(),
        )
        .unwrap();
        assert_eq!(handler_before, handler_after);
        assert_i32_values(&destination);
        world.barrier();

        #[cfg(feature = "parallel-hdf5")]
        {
            use super::{read_hdf5, write_hdf5};

            let valid_path = directory.join("valid.h5");
            root_status(&world, || reset(&valid_path));
            world.barrier();
            let handler_before = crate::ffi::test_comm_errhandler_token(
                source.pencil().topology().communicator().as_raw(),
            )
            .unwrap();
            write_hdf5(&valid_path, source.view()).unwrap();
            let handler_after = crate::ffi::test_comm_errhandler_token(
                source.pencil().topology().communicator().as_raw(),
            )
            .unwrap();
            assert_eq!(handler_before, handler_after);
            world.barrier();
            let before = destination.as_slice().to_vec();
            let handler_before = crate::ffi::test_comm_errhandler_token(
                destination.pencil().topology().communicator().as_raw(),
            )
            .unwrap();
            let error = crate::hdf5_io::read_hdf5_with_post_cleanup_failure(
                &valid_path,
                destination.view_mut(),
            )
            .expect_err("synthetic HDF5 post-cleanup read failure");
            let handler_after = crate::ffi::test_comm_errhandler_token(
                destination.pencil().topology().communicator().as_raw(),
            )
            .unwrap();
            assert_eq!(handler_before, handler_after);
            if world.rank() == 0 {
                assert!(matches!(
                    error,
                    IoError::Native {
                        operation: "test post-cleanup result",
                        code: -1
                    }
                ));
            } else {
                assert!(matches!(
                    error,
                    IoError::CollectivePrecondition {
                        phase: "post-cleanup result"
                    }
                ));
            }
            assert_eq!(destination.as_slice(), before.as_slice());
            world.barrier();
            let handler_before = crate::ffi::test_comm_errhandler_token(
                destination.pencil().topology().communicator().as_raw(),
            )
            .unwrap();
            read_hdf5(&valid_path, destination.view_mut()).unwrap();
            let handler_after = crate::ffi::test_comm_errhandler_token(
                destination.pencil().topology().communicator().as_raw(),
            )
            .unwrap();
            assert_eq!(handler_before, handler_after);
            assert_i32_values(&destination);
            world.barrier();

            let uncertain_path = directory.join("uncertain.h5");
            root_status(&world, || reset(&uncertain_path));
            world.barrier();
            let handler_before = crate::ffi::test_comm_errhandler_token(
                source.pencil().topology().communicator().as_raw(),
            )
            .unwrap();
            let error = crate::hdf5_io::write_hdf5_with_postmarker_uncertainty(
                &uncertain_path,
                source.view(),
            )
            .expect_err("synthetic HDF5 post-marker uncertainty");
            let handler_after = crate::ffi::test_comm_errhandler_token(
                source.pencil().topology().communicator().as_raw(),
            )
            .unwrap();
            assert_eq!(handler_before, handler_after);
            let local_ok = i32::from(if world.rank() == 0 {
                matches!(
                    &error,
                    IoError::CommitUncertain {
                        stage: "test post-marker cleanup result"
                    }
                )
            } else {
                matches!(
                    &error,
                    IoError::CollectivePrecondition {
                        phase: "post-cleanup result"
                    }
                )
            });
            let mut all_ok = 0;
            world.all_reduce_into(&local_ok, &mut all_ok, SystemOperation::min());
            assert_eq!(all_ok, 1, "HDF5 post-marker error was not preserved");
            world.barrier();
            let handler_before = crate::ffi::test_comm_errhandler_token(
                destination.pencil().topology().communicator().as_raw(),
            )
            .unwrap();
            read_hdf5(&uncertain_path, destination.view_mut()).unwrap();
            let handler_after = crate::ffi::test_comm_errhandler_token(
                destination.pencil().topology().communicator().as_raw(),
            )
            .unwrap();
            assert_eq!(handler_before, handler_after);
            assert_i32_values(&destination);
            world.barrier();
        }

        #[cfg(feature = "parallel-hdf5")]
        {
            let named_path = directory.join("named-uncertain.h5");
            root_status(&world, || reset(&named_path));
            world.barrier();
            let result = crate::hdf5_io::write_named(&named_path, "A/温度", source.view(), true);
            assert!(
                result.is_err(),
                "named HDF5 uncertainty must propagate collectively"
            );
            let before = destination.as_slice().to_vec();
            let handler_before = crate::ffi::test_comm_errhandler_token(
                destination.pencil().topology().communicator().as_raw(),
            )
            .unwrap();
            crate::hdf5_io::read_named(&named_path, "A/温度", destination.view_mut(), true)
                .expect_err("named HDF5 post-cleanup failure");
            assert_eq!(destination.as_slice(), before);
            assert_eq!(
                handler_before,
                crate::ffi::test_comm_errhandler_token(
                    destination.pencil().topology().communicator().as_raw(),
                )
                .unwrap()
            );
            super::read_hdf5_named(&named_path, "A/温度", destination.view_mut()).unwrap();
            assert_i32_values(&destination);
        }

        let named_path = directory.join("named-uncertain.pio");
        root_status(&world, || reset(&named_path));
        world.barrier();
        let named_error =
            crate::named_mpi::write_with_commit_uncertainty(&named_path, "A/温度", source.view())
                .expect_err("named post-marker uncertainty");
        assert!(matches!(
            named_error,
            super::NamedIoError::Io(IoError::CommitUncertain { .. })
        ));
        let before = destination.as_slice().to_vec();
        let handler_before = crate::ffi::test_comm_errhandler_token(
            destination.pencil().topology().communicator().as_raw(),
        )
        .unwrap();
        crate::named_mpi::read_with_cleanup_failure(&named_path, "A/温度", destination.view_mut())
            .expect_err("named post-cleanup failure");
        assert_eq!(destination.as_slice(), before);
        assert_eq!(
            handler_before,
            crate::ffi::test_comm_errhandler_token(
                destination.pencil().topology().communicator().as_raw(),
            )
            .unwrap()
        );
        super::read_mpi_named(&named_path, "A/温度", destination.view_mut()).unwrap();
        assert_i32_values(&destination);

        cleanup_owned_temp_dir(&world, &directory);
    }

    fn assert_i32_values(array: &PencilArray<i32, 2, 2>) {
        let view = array.view();
        let ranges = view.pencil().local_ranges().clone();
        for x in 0..ranges[0].len() {
            for y in 0..ranges[1].len() {
                assert_eq!(
                    view.get_local(&[], [x, y]),
                    Some(&((100 * (ranges[0].start + x) + ranges[1].start + y) as i32))
                );
            }
        }
    }
}
