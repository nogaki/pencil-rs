use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use mpi::collective::{Root, SystemOperation};
use mpi::raw::AsRaw;
use mpi::traits::{Communicator, CommunicatorCollectives};
use pencil_array::{ExtraShape, MpiTopology, Pencil, PencilArray};

use super::{IoError, read_mpi, write_mpi};

#[test]
#[ignore = "requires an MPI launcher and one fresh MPI process"]
fn mpi_payload_limit_precedes_allocation() {
    crate::mpi_preflight_tests::run();
}

#[test]
fn write_payload_error_maps_native_steps_and_preserves_peer_agreements() {
    use crate::mpi_io::{WriteError, WriteStep};

    #[derive(Clone, Copy, Debug)]
    enum Expected {
        Agreement,
        Incomplete(&'static str),
        Uncertain(&'static str),
    }

    let cases = [
        (
            WriteStep::PayloadView,
            Some(-71),
            Expected::Incomplete("MPI_File_set_view"),
        ),
        (WriteStep::PayloadView, None, Expected::Agreement),
        (
            WriteStep::PayloadWrite,
            Some(-71),
            Expected::Incomplete("payload write"),
        ),
        (
            WriteStep::PayloadWrite,
            None,
            Expected::Incomplete("payload write"),
        ),
        (
            WriteStep::PayloadFlush,
            Some(-71),
            Expected::Incomplete("payload flush"),
        ),
        (WriteStep::PayloadFlush, None, Expected::Agreement),
        (
            WriteStep::PayloadSize,
            None,
            Expected::Incomplete("payload size"),
        ),
        (
            WriteStep::MarkerView,
            Some(-71),
            Expected::Incomplete("commit marker view"),
        ),
        (WriteStep::MarkerView, None, Expected::Agreement),
        (
            WriteStep::MarkerWrite,
            Some(-71),
            Expected::Uncertain("commit marker write"),
        ),
        (
            WriteStep::MarkerWrite,
            None,
            Expected::Uncertain("commit marker write"),
        ),
        (
            WriteStep::MarkerFlush,
            Some(-71),
            Expected::Uncertain("commit marker flush"),
        ),
        (WriteStep::MarkerFlush, None, Expected::Agreement),
    ];

    for (step, native, expected) in cases {
        let actual = crate::mpi_io::write_payload_error(WriteError {
            step,
            agreement: IoError::CollectivePrecondition {
                phase: "write error agreement",
            },
            native,
        });
        match (expected, actual) {
            (
                Expected::Agreement,
                IoError::CollectivePrecondition {
                    phase: "write error agreement",
                },
            ) => {}
            (Expected::Incomplete(expected), IoError::WriteIncomplete { stage }) => {
                assert_eq!(stage, expected);
            }
            (Expected::Uncertain(expected), IoError::CommitUncertain { stage }) => {
                assert_eq!(stage, expected);
            }
            (expected, actual) => panic!("unexpected mapping for {expected:?}: {actual:?}"),
        }
    }
}

#[test]
#[ignore = "requires a fresh MPI process; run this case under mpiexec"]
fn session_and_named_write_errors_keep_pre_marker_primary_and_reads_are_atomic() {
    let universe = mpi::initialize().expect("MPI must initialize once");
    let world = universe.world();
    let size = usize::try_from(world.size()).unwrap();
    let topology = MpiTopology::<1>::new(&world, [size]).unwrap();
    let comm = topology.communicator();
    let pencil = Pencil::<1, 1>::new(topology.clone(), [size * 2], [0]).unwrap();
    let source = PencilArray::from_elem(pencil.clone(), ExtraShape::scalar(), 11i32).unwrap();
    let directory = owned_temp_dir(&world, "pencil-io-write-error-primary");

    let named_path = directory.join("named-before-marker.pio");
    crate::ffi::FAIL_BYTE_RESET.with(|fault| fault.set(if world.rank() == 0 { 1 } else { 0 }));
    let named_result = super::write_mpi_named(&named_path, "data", source.view());
    let named_primary = if world.rank() == 0 {
        matches!(
            &named_result,
            Err(NamedIoError::Io(IoError::WriteIncomplete {
                stage: "commit marker view"
            }))
        )
    } else {
        matches!(
            &named_result,
            Err(NamedIoError::Io(IoError::CollectivePrecondition {
                phase: "named marker view"
            }))
        )
    };
    every_rank(&world, || named_primary);

    let session_path = directory.join("session-before-marker.pio");
    let mut writer =
        super::MpiFileSession::create(comm, &session_path, &super::MpiIoOptions::default())
            .unwrap();
    crate::ffi::FAIL_BYTE_RESET.with(|fault| fault.set(if world.rank() == 0 { 1 } else { 0 }));
    let session_result = writer.write_named("data", source.view());
    let session_primary = if world.rank() == 0 {
        matches!(
            &session_result,
            Err(super::MpiSessionError::Io(IoError::WriteIncomplete {
                stage: "commit marker view"
            }))
        )
    } else {
        matches!(
            &session_result,
            Err(super::MpiSessionError::Io(
                IoError::CollectivePrecondition {
                    phase: "session byte view reset"
                }
            ))
        )
    };
    every_rank(&world, || session_primary);
    writer.close().unwrap();

    let header_flush_path = directory.join("session-header-flush.pio");
    let mut writer =
        super::MpiFileSession::create(comm, &header_flush_path, &super::MpiIoOptions::default())
            .unwrap();
    crate::ffi::FAIL_FILE_SYNC.with(|fault| fault.set(world.rank() == 0));
    let header_flush = writer.write_named("data", source.view());
    let header_primary = matches!(
        &header_flush,
        Err(super::MpiSessionError::Io(
            IoError::CollectivePrecondition {
                phase: "session header flush"
            }
        ))
    );
    every_rank(&world, || header_primary);
    writer.close().unwrap();

    let named_marker_path = directory.join("named-marker-flush.pio");
    FAIL_NEXT_MARKER_FLUSH.store(world.rank() == 0, Ordering::SeqCst);
    let named_marker_flush = super::write_mpi_named(&named_marker_path, "data", source.view());
    let named_fault_pending = FAIL_NEXT_MARKER_FLUSH.swap(false, Ordering::SeqCst);
    let named_marker_primary = if world.rank() == 0 {
        matches!(
            &named_marker_flush,
            Err(super::NamedIoError::Io(IoError::CommitUncertain {
                stage: "commit marker flush"
            }))
        )
    } else {
        matches!(
            &named_marker_flush,
            Err(super::NamedIoError::Io(IoError::CollectivePrecondition {
                phase: "named marker flush"
            }))
        )
    };

    let session_marker_path = directory.join("session-marker-flush.pio");
    let mut writer =
        super::MpiFileSession::create(comm, &session_marker_path, &super::MpiIoOptions::default())
            .unwrap();
    FAIL_NEXT_MARKER_FLUSH.store(world.rank() == 0, Ordering::SeqCst);
    let session_marker_flush = writer.write_named("data", source.view());
    let session_fault_pending = FAIL_NEXT_MARKER_FLUSH.swap(false, Ordering::SeqCst);
    writer.close().unwrap();
    let session_marker_primary = if world.rank() == 0 {
        matches!(
            &session_marker_flush,
            Err(super::MpiSessionError::Io(IoError::CommitUncertain {
                stage: "commit marker flush"
            }))
        )
    } else {
        matches!(
            &session_marker_flush,
            Err(super::MpiSessionError::Io(
                IoError::CollectivePrecondition {
                    phase: "session marker flush",
                }
            ))
        )
    };

    let mut all_named_fault_consumed = 0;
    world.all_reduce_into(
        &i32::from(!named_fault_pending),
        &mut all_named_fault_consumed,
        SystemOperation::min(),
    );
    assert_eq!(
        all_named_fault_consumed, 1,
        "named marker-flush fault wiring is not connected"
    );
    let mut all_session_fault_consumed = 0;
    world.all_reduce_into(
        &i32::from(!session_fault_pending),
        &mut all_session_fault_consumed,
        SystemOperation::min(),
    );
    assert_eq!(
        all_session_fault_consumed, 1,
        "session marker-flush fault wiring is not connected"
    );
    every_rank(&world, || named_marker_primary);
    every_rank(&world, || session_marker_primary);

    let post_commit_path = directory.join("session-post-commit-cleanup.pio");
    let mut writer =
        super::MpiFileSession::create(comm, &post_commit_path, &super::MpiIoOptions::default())
            .unwrap();
    crate::ffi::FAIL_BYTE_RESET.with(|fault| fault.set(if world.rank() == 0 { 2 } else { 0 }));
    let cleanup_error = writer.write_named("data", source.view());
    let uncertain = matches!(
        &cleanup_error,
        Err(super::MpiSessionError::Io(IoError::CommitUncertain {
            stage: "session append or operation cleanup"
        }))
    );
    every_rank(&world, || uncertain);
    writer.close().unwrap();
    let mut reader =
        super::MpiFileSession::open_read(comm, &post_commit_path, &super::MpiIoOptions::default())
            .unwrap();
    let mut committed =
        PencilArray::from_elem(pencil.clone(), ExtraShape::scalar(), -1i32).unwrap();
    reader.read_named("data", committed.view_mut()).unwrap();
    every_rank(&world, || committed.as_slice() == source.as_slice());
    reader.close().unwrap();

    let valid_path = directory.join("valid-session.pio");
    let mut writer =
        super::MpiFileSession::create(comm, &valid_path, &super::MpiIoOptions::default()).unwrap();
    writer.write_named("data", source.view()).unwrap();
    writer.close().unwrap();
    let mut reader =
        super::MpiFileSession::open_read(comm, &valid_path, &super::MpiIoOptions::default())
            .unwrap();
    let mut destination = PencilArray::from_elem(pencil, ExtraShape::scalar(), -777i32).unwrap();
    let before = destination.as_slice().to_vec();
    let calls_before = crate::ffi::test_payload_calls();
    crate::ffi::FAIL_BYTE_RESET.with(|fault| fault.set(if world.rank() == 0 { 2 } else { 0 }));
    let read = reader.read_named("data", destination.view_mut());
    let read_primary = matches!(
        &read,
        Err(super::MpiSessionError::Io(
            IoError::CollectivePrecondition {
                phase: "session byte view reset"
            }
        ))
    );
    every_rank(&world, || read_primary);
    every_rank(&world, || destination.as_slice() == before.as_slice());
    assert_eq!(
        crate::ffi::test_payload_calls()[2] - calls_before[2],
        1,
        "the injected reset failure must happen after the native payload read"
    );
    reader.close().unwrap();
    cleanup_owned_temp_dir(&world, &directory);
}

mod support {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/mod.rs"));
}
use support::{cleanup_owned_temp_dir, owned_temp_dir};

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

use super::{CollectionIoError, NamedIoError};

// Test-only handoff for injecting a failure at the actual marker-flush site.
pub(crate) static FAIL_NEXT_MARKER_FLUSH: AtomicBool = AtomicBool::new(false);

fn every_rank<C, F>(world: &C, holds: F)
where
    C: CommunicatorCollectives,
    F: FnOnce() -> bool,
{
    let local_ok = i32::from(holds());
    let mut all_ok = 0;
    world.all_reduce_into(&local_ok, &mut all_ok, SystemOperation::min());
    assert_eq!(all_ok, 1, "per-rank expectation failed");
}

fn inner_io(error: &CollectionIoError) -> Option<&IoError> {
    match error {
        CollectionIoError::Io(error) => Some(error),
        CollectionIoError::Named(NamedIoError::Io(error)) => Some(error),
        _ => None,
    }
}

fn member_views(
    members: &mut [PencilArray<i32, 2, 2>],
) -> Vec<pencil_array::PencilArrayViewMut<'_, i32, 2, 2>> {
    members.iter_mut().map(|array| array.view_mut()).collect()
}

fn poison_members(members: &mut [PencilArray<i32, 2, 2>]) {
    // Distinct negative sentinels never match the non-negative fixture, so a
    // failed read that still publishes file values is detected.
    for (index, member) in members.iter_mut().enumerate() {
        member.as_mut_slice().fill(-10_000 - index as i32);
    }
}

fn snapshot(members: &[PencilArray<i32, 2, 2>]) -> Vec<Vec<i32>> {
    members
        .iter()
        .map(|array| array.as_slice().to_vec())
        .collect()
}

fn assert_members_unchanged(members: &[PencilArray<i32, 2, 2>], before: &[Vec<i32>]) {
    for (member, saved) in members.iter().zip(before) {
        assert_eq!(member.as_slice(), saved.as_slice(), "member changed");
    }
}

fn reset(path: &Path) -> Result<(), String> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.to_string()),
    }
}

#[test]
fn post_cleanup_errors_preserve_destination_and_valid_commits() {
    let universe = mpi::initialize().expect("MPI must initialize once");
    let world = universe.world();
    let size = usize::try_from(world.size()).unwrap();
    assert_eq!(crate::mpi_io::collective_state(&world, true), (true, false));
    assert_eq!(
        crate::mpi_io::collective_state(&world, false),
        (false, false)
    );
    assert_eq!(
        crate::mpi_io::collective_state(&world, world.rank() == 0),
        (size == 1, size > 1)
    );
    let directory = owned_temp_dir(&world, "pencil-io-private-hook");

    let writer_topology = MpiTopology::<2>::new(&world, [size, 1]).unwrap();
    let writer_pencil = Pencil::<2, 2>::new(writer_topology, [4, 5], [0, 1]).unwrap();
    let mut source = PencilArray::from_elem(writer_pencil, ExtraShape::scalar(), -1i32).unwrap();
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

    if std::env::var_os("PENCIL_MPI_SESSION_CLEANUP_CHILD").is_some() {
        crate::named_mpi::session::test_native_cleanup_child(&directory, &source);
    }

    {
        let mut empty = PencilArray::from_elem(
            std::sync::Arc::clone(source.pencil()),
            ExtraShape::new([usize::MAX, 2, 0]).unwrap(),
            0i32,
        )
        .unwrap();
        let view = empty.view_mut();
        assert!(matches!(
            crate::format::prepare_physical_decode(&view, &[]),
            Err(IoError::SizeLimit {
                what: "element count"
            })
        ));
    }

    {
        let mut empty = PencilArray::from_elem(
            std::sync::Arc::clone(source.pencil()),
            ExtraShape::new([0, usize::MAX, 2]).unwrap(),
            0i32,
        )
        .unwrap();
        let mut view = empty.view_mut();
        let plan = crate::format::prepare_physical_decode(&view, &[]).unwrap();
        plan.decode_into(&[], view.as_mut_slice());
        assert!(view.is_empty());
    }
    crate::chunked::tests::cleanup_and_native_contracts(&world);
    let valid_path = directory.join("valid.pio");
    root_status(&world, || reset(&valid_path));
    world.barrier();
    let handler_before =
        crate::ffi::test_comm_errhandler_token(source.pencil().topology().communicator().as_raw())
            .unwrap();
    write_mpi(&valid_path, source.view()).unwrap();
    let handler_after =
        crate::ffi::test_comm_errhandler_token(source.pencil().topology().communicator().as_raw())
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
    let handler_before =
        crate::ffi::test_comm_errhandler_token(source.pencil().topology().communicator().as_raw())
            .unwrap();
    let error =
        crate::mpi_io::write_mpi_with_postmarker_uncertainty(&uncertain_path, source.view())
            .expect_err("synthetic post-marker uncertainty");
    let handler_after =
        crate::ffi::test_comm_errhandler_token(source.pencil().topology().communicator().as_raw())
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

    // A native flush error is local; peers keep the agreement failure.
    let failed_flush = directory.join("failed-flush.pio");
    crate::ffi::FAIL_FILE_SYNC.with(|f| f.set(world.rank() == 0));
    let error = write_mpi(&failed_flush, source.view()).expect_err("injected header flush");
    assert!(
        matches!(
            (&error, world.rank() == 0),
            (
                IoError::WriteIncomplete {
                    stage: "header flush"
                },
                true
            ) | (
                IoError::CollectivePrecondition {
                    phase: "MPI-IO header flush"
                },
                false
            )
        ),
        "{error:?}"
    );
    let before_flush_read = destination.as_slice().to_vec();
    assert!(matches!(
        read_mpi(&failed_flush, destination.view_mut()),
        Err(IoError::IncompleteFile)
    ));
    assert_eq!(destination.as_slice(), before_flush_read);
    world.barrier();

    #[cfg(feature = "parallel-hdf5")]
    {
        use super::{read_hdf5, write_hdf5};

        crate::hdf5_io::test_session_native_contracts(
            &directory.join("session-native.h5"),
            &source,
            &mut destination,
        );

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
        let error =
            crate::hdf5_io::write_hdf5_with_postmarker_uncertainty(&uncertain_path, source.view())
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

    // Verify actual native dispatch, not merely option descriptors or numerics.
    let controls = super::MpiIoOptions::default()
        .mode(super::MpiIoMode::Independent)
        .hint("cb_buffer_size", "2097152");
    let counts = crate::ffi::test_payload_calls();
    let independent = directory.join("native-independent.pio");
    super::write_mpi_with_options(&independent, source.view(), &controls).unwrap();
    super::read_mpi_with_options(&independent, destination.view_mut(), &controls).unwrap();
    let after = crate::ffi::test_payload_calls();
    assert_eq!(&crate::ffi::test_info_arguments()[..2], &[2097152, 2097152]);
    assert_eq!(
        [
            after[0] - counts[0],
            after[1] - counts[1],
            after[2] - counts[2],
            after[3] - counts[3]
        ],
        [0, 1, 0, 1]
    );
    assert_i32_values(&destination);
    let named = directory.join("native-independent-named.pio");
    let counts = crate::ffi::test_payload_calls();
    super::write_mpi_named_with_options(&named, "first", source.view(), &controls).unwrap();
    super::append_mpi_named_with_options(&named, "second", source.view(), &controls).unwrap();
    super::read_mpi_named_with_options(&named, "second", destination.view_mut(), &controls)
        .unwrap();
    let after = crate::ffi::test_payload_calls();
    assert_eq!(
        [
            after[0] - counts[0],
            after[1] - counts[1],
            after[2] - counts[2],
            after[3] - counts[3]
        ],
        [0, 2, 0, 1]
    );
    assert_i32_values(&destination);
    let mut offset = 0u64;
    if world.rank() == 0 {
        let bytes = std::fs::read(&independent).unwrap();
        offset = u64::from_le_bytes(bytes[64..72].try_into().unwrap());
    }
    world.process_at_rank(0).broadcast_into(&mut offset);
    destination.as_mut_slice().fill(-999);
    let raw_before = destination.as_slice().to_vec();
    let counts = crate::ffi::test_payload_calls();
    let raw_controls = super::RawReadOptions::default()
        .mode(super::MpiIoMode::Independent)
        .byte_offset(offset)
        .byte_order(super::RawByteOrder::Little);
    crate::raw::read_raw_with_post_cleanup_failure(
        &independent,
        destination.view_mut(),
        raw_controls.clone(),
    )
    .expect_err("raw cleanup fault");
    assert_eq!(destination.as_slice(), raw_before);
    super::read_mpi_raw(&independent, destination.view_mut(), raw_controls).unwrap();
    let after = crate::ffi::test_payload_calls();
    assert_eq!([after[2] - counts[2], after[3] - counts[3]], [0, 2]);
    assert_i32_values(&destination);
    #[cfg(feature = "parallel-hdf5")]
    {
        let counts = crate::ffi::test_payload_calls();
        let path = directory.join("native-independent.h5");
        super::write_hdf5_with_options(
            &path,
            source.view(),
            &super::Hdf5WriteOptions::default()
                .mode(super::MpiIoMode::Independent)
                .hint("cb_buffer_size", "2097152"),
        )
        .unwrap();
        super::read_hdf5_with_options(
            &path,
            destination.view_mut(),
            &super::Hdf5ReadOptions::default()
                .mode(super::MpiIoMode::Independent)
                .hint("cb_buffer_size", "2097152"),
        )
        .unwrap();
        let after = crate::ffi::test_payload_calls();
        assert_eq!(crate::ffi::test_info_arguments()[2], 2097152);
        assert_eq!([after[4] - counts[4], after[5] - counts[5]], [0, 2]);
        assert_i32_values(&destination);
        let named = directory.join("native-independent-named.h5");
        let write = super::Hdf5WriteOptions::default().mode(super::MpiIoMode::Independent);
        let read = super::Hdf5ReadOptions::default().mode(super::MpiIoMode::Independent);
        let counts = crate::ffi::test_payload_calls();
        super::write_hdf5_named_with_options(&named, "first", source.view(), &write).unwrap();
        super::append_hdf5_named_with_options(&named, "second", source.view(), &write).unwrap();
        super::read_hdf5_named_with_options(&named, "second", destination.view_mut(), &read)
            .unwrap();
        let after = crate::ffi::test_payload_calls();
        assert_eq!([after[4] - counts[4], after[5] - counts[5]], [0, 3]);
        assert_i32_values(&destination);
    }
    // Every member is staged before any native read or destination copy.
    let comm = source.pencil().topology().communicator();
    let collection = directory.join("staging-collection.pio");
    let options = super::MpiIoOptions::default();
    super::write_mpi_named_collection(
        &collection,
        "members",
        comm,
        &[source.view(), source.view()],
        &options,
    )
    .unwrap();
    let mut members: Vec<_> = (0..2)
        .map(|_| {
            PencilArray::from_elem(source.pencil().clone(), ExtraShape::scalar(), -900i32).unwrap()
        })
        .collect();
    for index in 0..members.len() {
        poison_members(&mut members);
        let before: Vec<_> = members.iter().map(|a| a.as_slice().to_vec()).collect();
        let calls = crate::ffi::test_payload_calls();
        crate::collections::STAGING_FAILURE.with(|f| f.set((world.rank() == 0).then_some(index)));
        let result = super::read_mpi_named_collection(
            &collection,
            "members",
            comm,
            &mut members.iter_mut().map(|a| a.view_mut()).collect::<Vec<_>>(),
            &options,
        );
        crate::collections::STAGING_FAILURE.with(|f| f.set(None));
        assert!(result.is_err());
        assert_eq!(calls, crate::ffi::test_payload_calls());
        for (member, saved) in members.iter().zip(before) {
            assert_eq!(member.as_slice(), saved);
        }
    }

    // Collection controls are agreed after staging; the legacy MPI
    // collection entry points still check their named-name precondition
    // first.
    let mut bad_hint = super::MpiIoOptions::default();
    if world.rank() == 0 {
        bad_hint = bad_hint.hint("", "bad");
    }
    crate::collections::STAGING_CALLS.with(|c| c.set(0));
    assert!(
        super::write_mpi_collection_with_options(
            directory.join("bad-hint-collection.pio"),
            comm,
            &[source.view(), source.view()],
            &bad_hint,
        )
        .is_err()
    );
    assert_eq!(
        crate::collections::STAGING_CALLS.with(std::cell::Cell::get),
        1
    );

    let option_collection = directory.join("option-collection.pio");
    super::write_mpi_collection_with_options(
        &option_collection,
        comm,
        &[source.view(), source.view()],
        &options,
    )
    .unwrap();
    let bad_read_hint = if world.rank() == 0 {
        super::MpiIoOptions::default().hint("", "bad")
    } else {
        super::MpiIoOptions::default()
    };
    crate::collections::STAGING_CALLS.with(|c| c.set(0));
    assert!(
        super::read_mpi_collection_with_options(
            &option_collection,
            comm,
            &mut members.iter_mut().map(|a| a.view_mut()).collect::<Vec<_>>(),
            &bad_read_hint,
        )
        .is_err()
    );
    assert_eq!(
        crate::collections::STAGING_CALLS.with(std::cell::Cell::get),
        1
    );

    let bad_name = if size == 1 {
        ""
    } else if world.rank() == 0 {
        "missing"
    } else {
        "members"
    };
    crate::collections::STAGING_FAILURE.with(|f| f.set(Some(0)));
    crate::collections::STAGING_CALLS.with(|c| c.set(0));
    assert!(
        super::read_mpi_named_collection(
            &collection,
            bad_name,
            comm,
            &mut members.iter_mut().map(|a| a.view_mut()).collect::<Vec<_>>(),
            &options,
        )
        .is_err()
    );
    assert_eq!(
        crate::collections::STAGING_CALLS.with(std::cell::Cell::get),
        0
    );
    crate::collections::STAGING_FAILURE.with(|f| f.set(None));

    if size > 1 {
        let mixed_mode = super::MpiIoOptions::default().mode(if world.rank() == 0 {
            super::MpiIoMode::Independent
        } else {
            super::MpiIoMode::Collective
        });
        crate::collections::STAGING_CALLS.with(|c| c.set(0));
        assert!(
            super::write_mpi_collection_with_options(
                directory.join("mixed-mode-collection.pio"),
                comm,
                &[source.view(), source.view()],
                &mixed_mode,
            )
            .is_err()
        );
        assert_eq!(
            crate::collections::STAGING_CALLS.with(std::cell::Cell::get),
            1
        );

        let mixed_hints = super::MpiIoOptions::default()
            .hint("cb_buffer_size", if world.rank() == 0 { "1" } else { "2" });
        crate::collections::STAGING_CALLS.with(|c| c.set(0));
        assert!(
            super::write_mpi_collection_with_options(
                directory.join("mixed-hints-collection.pio"),
                comm,
                &[source.view(), source.view()],
                &mixed_hints,
            )
            .is_err()
        );
        assert_eq!(
            crate::collections::STAGING_CALLS.with(std::cell::Cell::get),
            1
        );
    }

    #[cfg(feature = "parallel-hdf5")]
    {
        let mut bad_hdf_hint = super::Hdf5WriteOptions::default();
        if world.rank() == 0 {
            bad_hdf_hint = bad_hdf_hint.hint("", "bad");
        }
        crate::collections::STAGING_CALLS.with(|c| c.set(0));
        assert!(
            super::write_hdf5_collection_with_options(
                directory.join("bad-hdf-hint-collection.h5"),
                comm,
                &[source.view(), source.view()],
                &bad_hdf_hint,
            )
            .is_err()
        );
        assert_eq!(
            crate::collections::STAGING_CALLS.with(std::cell::Cell::get),
            1
        );
    }

    super::read_mpi_named_collection(
        &collection,
        "members",
        comm,
        &mut members.iter_mut().map(|a| a.view_mut()).collect::<Vec<_>>(),
        &options,
    )
    .unwrap();

    // Collection reads reserve one combined byte payload, read it once, and
    // publish only after native cleanup. Every injected failure must leave
    // all members unchanged, and the following read must recover.
    let plain_collection = directory.join("plain-collection.pio");
    super::write_mpi_collection(&plain_collection, comm, &[source.view(), source.view()]).unwrap();
    #[cfg(feature = "parallel-hdf5")]
    let hdf5_collections = {
        let plain = directory.join("plain-collection.h5");
        let options_path = directory.join("option-collection.h5");
        let named = directory.join("named-collection.h5");
        let write = super::Hdf5WriteOptions::default();
        super::write_hdf5_collection(&plain, comm, &[source.view(), source.view()]).unwrap();
        super::write_hdf5_collection_with_options(
            &options_path,
            comm,
            &[source.view(), source.view()],
            &write,
        )
        .unwrap();
        super::write_hdf5_named_collection(
            &named,
            "members",
            comm,
            &[source.view(), source.view()],
            &write,
        )
        .unwrap();
        (plain, options_path, named)
    };

    macro_rules! collection_failures {
        ($label:literal, $read:expr) => {{
            poison_members(&mut members);
            let before = snapshot(&members);

            crate::collections::STAGING_ALLOCATION_FAILURE.with(|f| f.set(world.rank() == 0));
            let calls = crate::ffi::test_payload_calls();
            let error = $read().expect_err(concat!($label, ": staging allocation failure"));
            crate::collections::STAGING_ALLOCATION_FAILURE.with(|f| f.set(false));
            let expected = if world.rank() == 0 {
                matches!(inner_io(&error), Some(IoError::AllocationFailed { .. }))
            } else {
                matches!(
                    inner_io(&error),
                    Some(IoError::CollectivePrecondition {
                        phase: "collection staging allocation"
                    })
                )
            };
            every_rank(&world, || expected);
            assert_eq!(
                calls,
                crate::ffi::test_payload_calls(),
                concat!($label, ": staging failure must precede the payload read")
            );
            assert_members_unchanged(&members, &before);

            crate::format::PLAN_FAILURE.with(|f| f.set(world.rank() == 0));
            let calls = crate::ffi::test_payload_calls();
            let error = $read().expect_err(concat!($label, ": plan preparation failure"));
            crate::format::PLAN_FAILURE.with(|f| f.set(false));
            // Named readers report the collective staging agreement even on
            // the failing rank, so only the local allocation is required there.
            let expected = if world.rank() == 0 {
                matches!(
                    inner_io(&error),
                    Some(
                        IoError::AllocationFailed { requested: 8 }
                            | IoError::CollectivePrecondition { .. }
                    )
                )
            } else {
                matches!(
                    inner_io(&error),
                    Some(IoError::CollectivePrecondition { .. })
                )
            };
            every_rank(&world, || expected);
            assert_ne!(
                calls,
                crate::ffi::test_payload_calls(),
                concat!($label, ": plan failure must follow the payload read")
            );
            assert_members_unchanged(&members, &before);

            crate::mpi_io::POST_CLEANUP_FAILURE.with(|f| f.set(world.rank() == 0));
            let error = $read().expect_err(concat!($label, ": post-cleanup failure"));
            crate::mpi_io::POST_CLEANUP_FAILURE.with(|f| f.set(false));
            let expected = if world.rank() == 0 {
                matches!(
                    inner_io(&error),
                    Some(IoError::Native {
                        operation: "test post-cleanup result" | "named post-cleanup result",
                        code: -1
                    })
                )
            } else {
                matches!(
                    inner_io(&error),
                    Some(IoError::CollectivePrecondition {
                        phase: "post-cleanup result"
                    })
                )
            };
            every_rank(&world, || expected);
            assert_members_unchanged(&members, &before);

            $read().expect(concat!($label, ": recovery read"));
            assert_i32_values(&members[0]);
            assert_i32_values(&members[1]);
            world.barrier();
        }};
    }

    collection_failures!("read_mpi_collection", || super::read_mpi_collection(
        &plain_collection,
        comm,
        &mut member_views(&mut members)
    ));
    collection_failures!("read_mpi_collection_with_options", || {
        super::read_mpi_collection_with_options(
            &option_collection,
            comm,
            &mut member_views(&mut members),
            &options,
        )
    });
    collection_failures!("read_mpi_named_collection", || {
        super::read_mpi_named_collection(
            &collection,
            "members",
            comm,
            &mut member_views(&mut members),
            &options,
        )
    });

    // Native collection controls reach the payload read, not just the
    // post-staging agreement.
    let native_collection_options = super::MpiIoOptions::default()
        .mode(super::MpiIoMode::Independent)
        .hint("cb_buffer_size", "3145728");
    let counts = crate::ffi::test_payload_calls();
    super::read_mpi_collection_with_options(
        &option_collection,
        comm,
        &mut member_views(&mut members),
        &native_collection_options,
    )
    .unwrap();
    let after = crate::ffi::test_payload_calls();
    assert_eq!(
        [after[2] - counts[2], after[3] - counts[3]],
        [0, 1],
        "collection options read dispatch"
    );
    assert_eq!(crate::ffi::test_info_arguments()[1], 3145728);
    assert_i32_values(&members[0]);
    assert_i32_values(&members[1]);
    let counts = crate::ffi::test_payload_calls();
    super::read_mpi_named_collection(
        &collection,
        "members",
        comm,
        &mut member_views(&mut members),
        &native_collection_options,
    )
    .unwrap();
    let after = crate::ffi::test_payload_calls();
    assert_eq!(
        [after[2] - counts[2], after[3] - counts[3]],
        [0, 1],
        "named collection options read dispatch"
    );
    assert_eq!(crate::ffi::test_info_arguments()[1], 3145728);
    assert_i32_values(&members[0]);
    assert_i32_values(&members[1]);
    world.barrier();

    #[cfg(feature = "parallel-hdf5")]
    {
        let (plain, options_path, named) = &hdf5_collections;
        let read = super::Hdf5ReadOptions::default();
        collection_failures!("read_hdf5_collection", || super::read_hdf5_collection(
            plain,
            comm,
            &mut member_views(&mut members)
        ));
        collection_failures!("read_hdf5_collection_with_options", || {
            super::read_hdf5_collection_with_options(
                options_path,
                comm,
                &mut member_views(&mut members),
                &read,
            )
        });
        collection_failures!("read_hdf5_named_collection", || {
            super::read_hdf5_named_collection(
                named,
                "members",
                comm,
                &mut member_views(&mut members),
                &read,
            )
        });

        let native_hdf5_read = super::Hdf5ReadOptions::default()
            .mode(super::MpiIoMode::Independent)
            .hint("cb_buffer_size", "3145728");
        let counts = crate::ffi::test_payload_calls();
        super::read_hdf5_collection_with_options(
            options_path,
            comm,
            &mut member_views(&mut members),
            &native_hdf5_read,
        )
        .unwrap();
        let after = crate::ffi::test_payload_calls();
        assert_eq!(
            [after[4] - counts[4], after[5] - counts[5]],
            [0, 1],
            "HDF5 collection options read dispatch"
        );
        assert_eq!(crate::ffi::test_info_arguments()[2], 3145728);
        assert_i32_values(&members[0]);
        assert_i32_values(&members[1]);
        let counts = crate::ffi::test_payload_calls();
        super::read_hdf5_named_collection(
            named,
            "members",
            comm,
            &mut member_views(&mut members),
            &native_hdf5_read,
        )
        .unwrap();
        let after = crate::ffi::test_payload_calls();
        assert_eq!(
            [after[4] - counts[4], after[5] - counts[5]],
            [0, 1],
            "named HDF5 collection options read dispatch"
        );
        assert_eq!(crate::ffi::test_info_arguments()[2], 3145728);
        assert_i32_values(&members[0]);
        assert_i32_values(&members[1]);
        world.barrier();
    }

    // Collection preparation validates every member and the staging length
    // before commit, including products that overflow only after a zero extent.
    {
        let extra = ExtraShape::new([2]).unwrap();
        let mut members: Vec<_> = (0..2)
            .map(|_| {
                PencilArray::from_elem(destination.pencil().clone(), extra.clone(), -41i32).unwrap()
            })
            .collect();
        let count = members.len();
        let len = members[0].len() * count;
        let mut views = member_views(&mut members);
        {
            let target =
                crate::format::ReadTarget::collection(&mut views, vec![count, 2], len, Vec::new());
            let wrong = vec![0u8; len * std::mem::size_of::<i32>() + 1];
            assert!(matches!(
                target.prepare(&wrong),
                Err(IoError::InvalidFile {
                    reason: "local staging length"
                })
            ));
        }
        drop(views);
        assert!(
            members
                .iter()
                .all(|member| member.as_slice().iter().all(|&value| value == -41))
        );

        let overflow =
            ExtraShape::new([i32::MAX as usize, i32::MAX as usize, i32::MAX as usize, 0]).unwrap();
        let mut empty: Vec<_> = (0..2)
            .map(|_| {
                PencilArray::from_elem(destination.pencil().clone(), overflow.clone(), -42i32)
                    .unwrap()
            })
            .collect();
        let count = empty.len();
        let mut views = member_views(&mut empty);
        let target = crate::format::ReadTarget::collection(
            &mut views,
            vec![
                count,
                i32::MAX as usize,
                i32::MAX as usize,
                i32::MAX as usize,
                0,
            ],
            0,
            Vec::new(),
        );
        assert!(matches!(
            target.prepare(&[]),
            Err(IoError::SizeLimit {
                what: "element count"
            })
        ));
        drop(target);
        drop(views);
        assert!(empty.iter().all(|member| member.is_empty()));
    }

    let calls = crate::ffi::test_payload_calls();
    assert_eq!(
        super::read_mpi_named_catalog(&collection, comm).unwrap()[0].extra_shape(),
        &[2]
    );
    super::read_mpi_catalog(&valid_path, comm).unwrap();
    assert_eq!(calls, crate::ffi::test_payload_calls());
    #[cfg(feature = "parallel-hdf5")]
    {
        for shuffle in [false, true] {
            for capabilities in [0, 1, 2] {
                let path = directory.join(format!("unsupported-{shuffle}-{capabilities}.h5"));
                let options = super::Hdf5WriteOptions::default().chunks([1, 1]);
                let options = if shuffle {
                    options.shuffle(true)
                } else {
                    options.deflate(0)
                };
                let duplicates = crate::ffi::DUPLICATE_CALLS.with(std::cell::Cell::get);
                crate::ffi::FILTER_CAPABILITIES
                    .with(|f| f.set((world.rank() == 0).then_some(capabilities)));
                let result = super::write_hdf5_with_options(&path, source.view(), &options);
                crate::ffi::FILTER_CAPABILITIES.with(|f| f.set(None));
                assert!(result.is_err());
                assert_eq!(
                    duplicates,
                    crate::ffi::DUPLICATE_CALLS.with(std::cell::Cell::get)
                );
                assert!(!path.exists());
            }
        }
        let path = directory.join("catalog-filtered.h5");
        super::write_hdf5_with_options(
            &path,
            source.view(),
            &super::Hdf5WriteOptions::default().chunks([1, 1]).deflate(0),
        )
        .unwrap();
        let calls = crate::ffi::test_payload_calls();
        super::read_hdf5_catalog(&path, comm).unwrap();
        assert_eq!(calls, crate::ffi::test_payload_calls());
    }
    // Catalog inspection must reject links and wrong object kinds on every
    // rank without dispatching any payload operation, then recover on valid files.
    #[cfg(feature = "parallel-hdf5")]
    {
        let legacy_target = directory.join("catalog-proof-legacy.h5");
        let named_target = directory.join("catalog-proof-named.h5");
        root_status(&world, || reset(&legacy_target));
        root_status(&world, || reset(&named_target));
        world.barrier();
        super::write_hdf5(&legacy_target, source.view()).unwrap();
        super::write_hdf5_named(&named_target, "data", source.view()).unwrap();
        world.barrier();
        for named_ns in [false, true] {
            let ns = if named_ns {
                "pencil_io_named_v1"
            } else {
                "pencil_io_v1"
            };
            let target_group = if named_ns {
                "/pencil_io_v1"
            } else {
                "/pencil_io_named_v1"
            };
            let target_dataset = if named_ns {
                "/pencil_io_v1/data"
            } else {
                "/pencil_io_named_v1/64617461"
            };
            let target_file = if named_ns {
                &legacy_target
            } else {
                &named_target
            };
            let leaf = if named_ns { "64617461" } else { "data" };
            for kind in [
                "soft-root",
                "external-root",
                "soft-leaf",
                "external-leaf",
                "wrong-object-group",
            ] {
                let path = directory.join(format!("catalog-proof-{ns}-{kind}.h5"));
                if named_ns {
                    super::write_hdf5(&path, source.view()).unwrap();
                } else {
                    super::write_hdf5_named(&path, "data", source.view()).unwrap();
                }
                if world.rank() == 0 {
                    let file = hdf5_metno::File::open_rw(&path).unwrap();
                    match kind {
                        "soft-root" => file.link_soft(target_group, ns).unwrap(),
                        "external-root" => file
                            .link_external(target_file.to_str().unwrap(), target_group, ns)
                            .unwrap(),
                        "soft-leaf" => {
                            file.create_group(ns).unwrap();
                            file.link_soft(target_dataset, &format!("{ns}/{leaf}"))
                                .unwrap();
                        }
                        "external-leaf" => {
                            file.create_group(ns).unwrap();
                            file.link_external(
                                target_file.to_str().unwrap(),
                                target_dataset,
                                &format!("{ns}/{leaf}"),
                            )
                            .unwrap();
                        }
                        "wrong-object-group" => {
                            file.create_group(ns).unwrap();
                            file.link_hard(target_group, &format!("{ns}/{leaf}"))
                                .unwrap();
                        }
                        _ => unreachable!(),
                    }
                    file.close().unwrap();
                }
                world.barrier();
                let before = crate::ffi::test_payload_calls();
                let result = super::read_hdf5_catalog(&path, comm);
                let local_ok = i32::from(result.is_err());
                let mut all_ok = 0;
                world.all_reduce_into(&local_ok, &mut all_ok, SystemOperation::min());
                assert_eq!(all_ok, 1, "{ns}/{kind} was accepted on some rank");
                assert_eq!(before, crate::ffi::test_payload_calls());
                world.barrier();
            }
        }
        assert_eq!(
            super::read_hdf5_catalog(&legacy_target, comm)
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            super::read_hdf5_catalog(&named_target, comm).unwrap().len(),
            1
        );
    }
    crate::named_mpi::session::test_retained_native_handles(&directory, &source, &mut destination);
    if world.rank() == 0 {
        println!("IO3_PRIVATE_CHECKS_OK");
    }
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
