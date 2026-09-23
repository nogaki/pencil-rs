use std::{path::Path, process::Command, sync::Arc};

use mpi::collective::CommunicatorCollectives;
use mpi::traits::Communicator;
use pencil_array::{ExtraShape, MpiTopology, Pencil, PencilArray};
use pencil_io::{
    DatasetInfo, MpiFileSession, MpiIoOptions, append_mpi_named, read_mpi_named,
    read_mpi_named_catalog,
};

mod support;
use support::{cleanup_owned_temp_dir, owned_temp_dir};

fn array(topology: &Arc<MpiTopology<2>>, value: i32) -> PencilArray<i32, 2, 2> {
    let pencil = Pencil::<2, 2>::new(topology.clone(), [4, 5], [0, 1]).unwrap();
    PencilArray::from_elem(pencil, ExtraShape::scalar(), value).unwrap()
}

fn names(catalog: &[DatasetInfo]) -> Vec<&str> {
    catalog.iter().map(|d| d.name().unwrap()).collect()
}

// Run directly, outside mpiexec: this harness launches a fresh MPI child.
#[test]
#[ignore = "fresh MPI subprocess harness; run outside mpiexec"]
fn mpi_file_session_active_drop_is_failstop() {
    for mode in ["active", "finalized"] {
        let output = Command::new("timeout")
            .args(["30", "mpiexec", "-n", "1"])
            .arg(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "mpi_file_session_lifecycle_and_collective_errors",
                "--nocapture",
            ])
            .env("PENCIL_MPI_ACTIVE_DROP_CHILD", mode)
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("SESSION_DROP_REACHED"),
            "child did not reach Drop: {stderr}"
        );
        assert!(
            !output.status.success(),
            "an open session Drop must fail-stop"
        );
        assert_ne!(output.status.code(), Some(124), "Drop hung: {stderr}");
        if mode == "finalized" {
            assert!(
                !stderr.contains("MPI_ABORT"),
                "must not call MPI_Abort after finalization: {stderr}"
            );
        }
    }
}

#[test]
fn mpi_file_session_lifecycle_and_collective_errors() {
    let universe = mpi::initialize().expect("MPI initialize once");
    let world = universe.world();
    let topology = MpiTopology::new(&world, [world.size() as usize, 1]).unwrap();
    let comm = topology.communicator();

    if std::env::var_os("PENCIL_MPI_ACTIVE_DROP_CHILD").is_some() {
        let dir = owned_temp_dir(&world, "pencil-mpi-session-child");
        let path = dir.join("persistent.pio");
        let active = MpiFileSession::create(comm, &path, &MpiIoOptions::default()).unwrap();
        // Unlink the test-owned path before fail-stop; the native handle stays open.
        cleanup_owned_temp_dir(&world, &dir);
        if std::env::var("PENCIL_MPI_ACTIVE_DROP_CHILD").unwrap() == "finalized" {
            drop(universe);
        }
        eprintln!("SESSION_DROP_REACHED");
        drop(active);
        unreachable!("active Drop aborts");
    }

    let dir = owned_temp_dir(&world, "pencil-mpi-session");
    let path = dir.join("persistent.pio");
    let options = MpiIoOptions::default();
    assert!(MpiFileSession::open_read(comm, dir.join("missing.pio"), &options).is_err());
    assert!(
        MpiFileSession::create(
            comm,
            dir.join("bad-options.pio"),
            &MpiIoOptions::default().hint("", "invalid")
        )
        .is_err()
    );
    assert!(!dir.join("bad-options.pio").exists());
    let source = array(&topology, 11);
    let second = array(&topology, 22);
    let mut session = MpiFileSession::create(comm, &path, &options).unwrap();
    session.write_named("first", source.view()).unwrap();
    session.write_named("second", second.view()).unwrap();
    session.flush().unwrap();
    assert_eq!(names(&session.catalog().unwrap()), vec!["first", "second"]);

    let mut first = array(&topology, -1);
    let mut second_read = array(&topology, -1);
    session.read_named("first", first.view_mut()).unwrap();
    session
        .read_named("second", second_read.view_mut())
        .unwrap();
    assert!(first.as_slice().iter().all(|&x| x == 11));
    assert!(second_read.as_slice().iter().all(|&x| x == 22));
    session.close().unwrap();
    assert!(MpiFileSession::create(comm, &path, &options).is_err());
    assert!(session.write_named("closed", source.view()).is_err());
    assert!(session.read_named("first", first.view_mut()).is_err());

    // The old path API can read and inspect a session-created container.
    read_mpi_named(&path, "first", first.view_mut()).unwrap();
    assert_eq!(read_mpi_named_catalog(&path, comm).unwrap().len(), 2);

    let mut append = MpiFileSession::open_append(comm, &path, &options).unwrap();
    append.write_named("third", source.view()).unwrap();
    assert_eq!(
        names(&append.catalog().unwrap()),
        vec!["first", "second", "third"]
    );
    append.close().unwrap();
    append_mpi_named(&path, "legacy", source.view()).unwrap();

    let mut readonly = MpiFileSession::open_read(comm, &path, &options).unwrap();
    let before = first.as_slice().to_vec();
    assert!(readonly.write_named("forbidden", source.view()).is_err());
    assert_eq!(first.as_slice(), before.as_slice());
    assert!(readonly.read_named("missing", first.view_mut()).is_err());
    assert_eq!(first.as_slice(), before.as_slice());
    readonly.close().unwrap();

    if world.size() > 1 {
        // Every rank chooses a different operation, then the same handle remains usable.
        let mut mixed = MpiFileSession::open_append(comm, &path, &options).unwrap();
        let mut mixed_target = array(&topology, -9);
        let mixed_result = if world.rank() == 0 {
            mixed.write_named("mixed", source.view()).map(|_| ())
        } else {
            mixed
                .read_named("first", mixed_target.view_mut())
                .map(|_| ())
        };
        assert!(mixed_result.is_err());
        let legacy_result = if world.rank() == 0 {
            mixed.close().map_err(|e| e.to_string())
        } else {
            read_mpi_named(&path, "first", mixed_target.view_mut()).map_err(|e| e.to_string())
        };
        assert!(legacy_result.is_err());
        let name_result = mixed.write_named(
            if world.rank() == 0 {
                "rank-zero"
            } else {
                "other"
            },
            source.view(),
        );
        assert!(name_result.is_err());
        let close_result = if world.rank() == 0 {
            mixed.close()
        } else if world.rank() % 2 == 0 {
            mixed.write_named("mixed-close", source.view())
        } else {
            mixed.read_named("first", mixed_target.view_mut())
        };
        assert!(close_result.is_err());
        assert!(mixed_target.as_slice().iter().all(|&v| v == -9));
        assert!(mixed.write_named("after-mixed", source.view()).is_ok());
        mixed.close().unwrap();

        // Different retained handles must not cross-wire; retry on the same handle.
        let mut left = MpiFileSession::open_append(comm, &path, &options).unwrap();
        let mut right = MpiFileSession::open_append(comm, &path, &options).unwrap();
        let crosswire = if world.rank() == 0 {
            left.write_named("crosswire", source.view()).map(|_| ())
        } else {
            right.write_named("crosswire", source.view()).map(|_| ())
        };
        assert!(crosswire.is_err());
        left.write_named("same-handle-retry", source.view())
            .unwrap();
        left.close().unwrap();
        right.close().unwrap();
    }

    // Invalid names, duplicates, and type mismatches do not mutate the target/file.
    let mut checked = MpiFileSession::open_append(comm, &path, &options).unwrap();
    assert!(checked.write_named("", source.view()).is_err());
    assert!(checked.write_named("first", source.view()).is_err());
    let mut wrong = PencilArray::from_elem(
        Pencil::<2, 2>::new(topology.clone(), [4, 5], [0, 1]).unwrap(),
        ExtraShape::new([2]).unwrap(),
        -77,
    )
    .unwrap();
    let wrong_before = wrong.as_slice().to_vec();
    assert!(checked.read_named("first", wrong.view_mut()).is_err());
    assert_eq!(wrong.as_slice(), wrong_before.as_slice());
    let mut wrong_type =
        PencilArray::from_elem(source.pencil().clone(), ExtraShape::scalar(), -77.0f64).unwrap();
    assert!(checked.read_named("first", wrong_type.view_mut()).is_err());
    assert!(wrong_type.as_slice().iter().all(|&v| v == -77.0));
    let mut target = array(&topology, -77);
    assert!(checked.read_named("first", target.view_mut()).is_ok());
    assert!(target.as_slice().iter().all(|&x| x == 11));
    checked.close().unwrap();

    // A congruent context is accepted, including rank-local choices of context.
    // A differently ordered rank group is rejected without source communication.
    let congruent = MpiTopology::from_cartesian(comm).unwrap();
    let mut congruent_array = array(&congruent, -1);
    let mut view_check = MpiFileSession::open_read(comm, &path, &options).unwrap();
    if world.rank() == 0 {
        view_check.read_named("first", first.view_mut()).unwrap();
    } else {
        view_check
            .read_named("first", congruent_array.view_mut())
            .unwrap();
    }
    view_check.close().unwrap();
    if world.size() > 1 {
        let reversed = world
            .split_by_color_with_key(
                mpi::topology::Color::with_value(0),
                world.size() - world.rank(),
            )
            .unwrap();
        let other = MpiTopology::new(&reversed, [world.size() as usize, 1]).unwrap();
        let mut other_array = array(&other, -1);
        let mut reject = MpiFileSession::open_read(comm, &path, &options).unwrap();
        assert!(reject.read_named("first", other_array.view_mut()).is_err());
        reject.close().unwrap();
    }

    // Read-only recovery exposes the committed prefix, while strict catalog and
    // append reject a trailing partial record.
    world.barrier();
    if world.rank() == 0 {
        use std::io::Write;
        std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"partial-tail")
            .unwrap();
    }
    world.barrier();
    let mut prefix = MpiFileSession::open_read(comm, &path, &options).unwrap();
    let mut recovered = array(&topology, -1);
    prefix.read_named("first", recovered.view_mut()).unwrap();
    assert!(prefix.catalog().is_err());
    prefix.close().unwrap();
    assert!(MpiFileSession::open_append(comm, &path, &options).is_err());

    cleanup_owned_temp_dir(&world, Path::new(&dir));
}
