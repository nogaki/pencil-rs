#![cfg(feature = "parallel-hdf5")]

use mpi::traits::*;
use num_complex::{Complex32, Complex64};
use pencil_array::{ExtraShape, MpiTopology, Pencil, PencilArray};
use pencil_io::{
    Hdf5FileSession, Hdf5ReadOptions, Hdf5WriteOptions, IoElement, MpiIoMode, MpiIoOptions,
};
use std::path::Path;

mod support;
use support::{cleanup_owned_temp_dir, owned_temp_dir};

fn array<T: IoElement>(
    world: &mpi::topology::SimpleCommunicator,
    extra: ExtraShape,
    value: T,
) -> PencilArray<T, 2, 2> {
    let topology = MpiTopology::<2>::new(world, [world.size() as usize, 1]).unwrap();
    let pencil = Pencil::new(topology, [world.size() as usize + 1, 3], [1, 0]).unwrap();
    PencilArray::from_elem(pencil, extra, value).unwrap()
}
fn roundtrip<T: IoElement + PartialEq + std::fmt::Debug>(
    world: &mpi::topology::SimpleCommunicator,
    session: &mut Hdf5FileSession<'_>,
    name: &str,
    value: T,
    zero: T,
    empty: bool,
) {
    let extra = if empty {
        ExtraShape::new([0]).unwrap()
    } else {
        ExtraShape::scalar()
    };
    let source = array(world, extra.clone(), value);
    let mut dest = array(world, extra, zero);
    let chunks = if empty { vec![1, 1, 2] } else { vec![1, 2] };
    let options = Hdf5WriteOptions::default()
        .chunks(chunks)
        .shuffle(true)
        .deflate(1);
    session.write(name, source.view(), &options).unwrap();
    assert!(session.write(name, source.view(), &options).is_err());
    session
        .read(
            name,
            dest.view_mut(),
            &Hdf5ReadOptions::default().mode(MpiIoMode::Independent),
        )
        .unwrap();
    assert_eq!(dest.as_slice(), source.as_slice());
}
fn root_action(world: &mpi::topology::SimpleCommunicator, action: impl FnOnce()) {
    world.barrier();
    if world.rank() == 0 {
        action();
    }
    world.barrier();
}
fn malicious(world: &mpi::topology::SimpleCommunicator, path: &Path, kind: usize) {
    root_action(world, || {
        let file = hdf5_metno::File::create(path).unwrap();
        let group = file.create_group("pencil_io_tree_v1").unwrap();
        match kind {
            0 => group.link_soft("/missing", "soft").unwrap(),
            1 => group
                .link_external("missing-external.h5", "/outside", "external")
                .unwrap(),
            2 => group.link_hard("/pencil_io_tree_v1", "cycle").unwrap(),
            3 => {
                group
                    .new_dataset::<f64>()
                    .shape([2])
                    .create("unknown")
                    .unwrap();
            }
            4 => {
                file.unlink("pencil_io_tree_v1").unwrap();
                file.link_soft("/missing", "pencil_io_tree_v1").unwrap();
            }
            5 => {
                file.unlink("pencil_io_tree_v1").unwrap();
                file.link_external("missing-external.h5", "/outside", "pencil_io_tree_v1")
                    .unwrap();
            }
            _ => unreachable!(),
        }
    });
    let topology = MpiTopology::<2>::new(world, [world.size() as usize, 1]).unwrap();
    let result =
        Hdf5FileSession::open_read(path, topology.communicator(), &MpiIoOptions::default());
    if kind >= 4 {
        assert!(result.is_err());
    } else {
        let mut session = result.unwrap();
        assert!(session.catalog().is_err());
        let mut dest = array(world, ExtraShape::scalar(), -123.0);
        assert!(
            session
                .read("soft", dest.view_mut(), &Hdf5ReadOptions::default())
                .is_err()
        );
        assert!(dest.as_slice().iter().all(|&v| v == -123.0));
        session.close().unwrap();
    }
}

#[test]
fn persistent_native_hierarchy_and_protocol() {
    let universe = mpi::initialize().expect("one MPI initialization per process");
    let world = universe.world();
    if let Ok(mode) = std::env::var("PENCIL_HDF_SESSION_CHILD") {
        let topology = MpiTopology::<2>::new(&world, [world.size() as usize, 1]).unwrap();
        let mut session = Hdf5FileSession::create(
            std::env::var("PENCIL_HDF_SESSION_PATH").unwrap(),
            topology.communicator(),
            &MpiIoOptions::default(),
        )
        .unwrap();
        if mode == "closed" || mode == "closed-finalized" {
            session.close().unwrap();
            if mode == "closed-finalized" {
                drop(universe);
                drop(session);
                std::process::exit(0);
            }
            drop(session);
            drop(topology);
            drop(universe);
            std::process::exit(0);
        }
        if mode == "finalized" {
            drop(universe);
        }
        eprintln!("session child {mode}: entering unclosed Drop");
        drop(session);
        std::process::exit(86);
    }
    let directory = owned_temp_dir(&world, "pencil-hdf-session");
    let path = directory.join("tree.h5");
    let topology = MpiTopology::<2>::new(&world, [world.size() as usize, 1]).unwrap();
    let comm = topology.communicator();
    let mut session = Hdf5FileSession::create(
        &path,
        comm,
        &MpiIoOptions::default().hint("cb_buffer_size", "1048576"),
    )
    .unwrap();
    assert!(session.catalog().unwrap().is_empty());
    for invalid in [
        "",
        "/a",
        "a/",
        "a//b",
        ".",
        "..",
        "a/../b",
        "bad\0name",
        "missing/child",
    ] {
        assert!(session.create_group(invalid).is_err(), "{invalid:?}");
    }
    assert!(session.create_group(&"x".repeat(256)).is_err());
    session.create_group("a").unwrap();
    session.create_group("a/b").unwrap();
    assert!(session.create_group("a/b").is_err());
    roundtrip(&world, &mut session, "a/b/f32", 1.25f32, -1.0, false);
    roundtrip(&world, &mut session, "a/f64", 2.5f64, -1.0, false);
    roundtrip(&world, &mut session, "i32", 123i32, -1, false);
    roundtrip(
        &world,
        &mut session,
        "c32",
        Complex32::new(1.0, -2.0),
        Complex32::new(0.0, 0.0),
        false,
    );
    roundtrip(
        &world,
        &mut session,
        "c64",
        Complex64::new(3.0, -4.0),
        Complex64::new(0.0, 0.0),
        false,
    );
    roundtrip(&world, &mut session, "empty", 1.0f64, 0.0, true);
    assert_eq!(session.catalog().unwrap().len(), 6);
    assert!(session.create_group("i32/child").is_err());
    let mut wrong_kind = array(&world, ExtraShape::scalar(), -77i32);
    assert!(
        session
            .read("a/b", wrong_kind.view_mut(), &Hdf5ReadOptions::default())
            .is_err()
    );
    assert!(wrong_kind.as_slice().iter().all(|&v| v == -77));
    let source = array(&world, ExtraShape::scalar(), 9i32);
    assert!(
        session
            .write(
                "bad_options",
                source.view(),
                &Hdf5WriteOptions::default().deflate(10)
            )
            .is_err()
    );
    assert!(
        session
            .write(
                "independent_filter",
                source.view(),
                &Hdf5WriteOptions::default()
                    .chunks(vec![1, 2])
                    .shuffle(true)
                    .mode(MpiIoMode::Independent)
            )
            .is_err()
    );
    let mut wrong = array(&world, ExtraShape::scalar(), -9f64);
    assert!(
        session
            .read("i32", wrong.view_mut(), &Hdf5ReadOptions::default())
            .is_err()
    );
    assert!(wrong.as_slice().iter().all(|&v| v == -9.0));
    if world.size() > 1 {
        assert!(
            session
                .create_group(if world.rank() == 0 { "one" } else { "two" })
                .is_err()
        );
        let result = if world.rank() == 0 {
            session.close()
        } else {
            session.catalog().map(|_| ())
        };
        assert!(result.is_err());
        let options = Hdf5WriteOptions::default().mode(if world.rank() == 0 {
            MpiIoMode::Independent
        } else {
            MpiIoMode::Collective
        });
        assert!(
            session
                .write("rank_options", source.view(), &options)
                .is_err()
        );
    }
    session
        .write("retry", source.view(), &Hdf5WriteOptions::default())
        .unwrap();
    session.close().unwrap();
    session.close().unwrap();
    assert!(session.catalog().is_err());
    drop(session);
    root_action(&world, || {
        let file = hdf5_metno::File::open(&path).unwrap();
        assert_eq!(
            file.group("pencil_io_tree_v1/a/b")
                .unwrap()
                .member_names()
                .unwrap(),
            ["f32"]
        );
        let dataset = file.dataset("pencil_io_tree_v1/a/b/f32").unwrap();
        assert_eq!(dataset.shape(), [world.size() as usize + 1, 3]);
        assert!(!dataset.filters().is_empty());
        assert!(
            dataset
                .read_raw::<f32>()
                .unwrap()
                .iter()
                .all(|&v| v == 1.25)
        );
        assert!(!file.link_exists("pencil_io_named_v1"));
    });
    let mut a = Hdf5FileSession::open_read(&path, comm, &MpiIoOptions::default()).unwrap();
    let mut b = Hdf5FileSession::open_read(&path, comm, &MpiIoOptions::default()).unwrap();
    if world.size() > 1 {
        let crossed = if world.rank() == 0 { &mut a } else { &mut b };
        assert!(crossed.close().is_err());
        let crossed = if world.rank() == 0 { &mut a } else { &mut b };
        assert!(crossed.catalog().is_err());
    }
    assert!(a.create_group("read_only").is_err());
    assert!(
        a.write("read_only", source.view(), &Hdf5WriteOptions::default())
            .is_err()
    );
    assert_eq!(a.catalog().unwrap().len(), 7);
    assert_eq!(b.catalog().unwrap().len(), 7);
    let mut persisted = array(&world, ExtraShape::scalar(), -1.0f32);
    for reopened in [&mut a, &mut b] {
        reopened
            .read("a/b/f32", persisted.view_mut(), &Hdf5ReadOptions::default())
            .unwrap();
        assert!(persisted.as_slice().iter().all(|&v| v == 1.25));
        persisted.as_mut_slice().fill(-1.0);
    }
    a.close().unwrap();
    b.close().unwrap();
    let mut append = Hdf5FileSession::open_append(&path, comm, &MpiIoOptions::default()).unwrap();
    append.create_group("later").unwrap();
    append
        .write("later/data", source.view(), &Hdf5WriteOptions::default())
        .unwrap();
    append.close().unwrap();
    // A second name for a valid leaf must reject the entire catalog.
    root_action(&world, || {
        let file = hdf5_metno::File::open_rw(&path).unwrap();
        file.link_hard("/pencil_io_tree_v1/i32", "/pencil_io_tree_v1/alias")
            .unwrap();
    });
    let mut invalid = Hdf5FileSession::open_read(&path, comm, &MpiIoOptions::default()).unwrap();
    assert!(invalid.catalog().is_err());
    invalid.close().unwrap();
    root_action(&world, || {
        let file = hdf5_metno::File::open_rw(&path).unwrap();
        file.unlink("pencil_io_tree_v1/alias").unwrap();
        file.dataset("pencil_io_tree_v1/i32")
            .unwrap()
            .attr("pencil_io_commit")
            .unwrap()
            .write_scalar(&0x494e_434f_4d50_4c45u64)
            .unwrap();
    });
    let mut invalid = Hdf5FileSession::open_read(&path, comm, &MpiIoOptions::default()).unwrap();
    assert!(invalid.catalog().is_err());
    invalid.close().unwrap();
    assert!(Hdf5FileSession::open_append(&path, comm, &MpiIoOptions::default()).is_err());
    for kind in 0..6 {
        malicious(
            &world,
            &directory.join(format!("malicious-{kind}.h5")),
            kind,
        );
    }
    root_action(&world, || {
        for mode in ["open", "closed", "finalized", "closed-finalized"] {
            let mpi_tmp = directory.join(format!("child-mpi-{mode}"));
            std::fs::create_dir(&mpi_tmp).unwrap();
            let mut command = std::process::Command::new("/bin/bash");
            command.env_clear().env("PATH", std::env::var("PATH").unwrap()).env("LD_LIBRARY_PATH", std::env::var("LD_LIBRARY_PATH").unwrap_or_default())
                .env("PENCIL_HDF_SESSION_CHILD", mode).env("PENCIL_HDF_SESSION_PATH", directory.join(format!("child-{mode}.h5")))
                .arg("-c").arg("ulimit -c 0; exec mpiexec --mca orte_base_help_aggregate 0 --mca orte_tmpdir_base \"$2\" --oversubscribe -n 1 \"$1\" --exact persistent_native_hierarchy_and_protocol --nocapture")
                .arg("hdf-session-child").arg(std::env::current_exe().unwrap()).arg(mpi_tmp);
            let output = command.output().unwrap();
            assert_eq!(
                output.status.success(),
                mode == "closed" || mode == "closed-finalized",
                "{mode}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            eprintln!(
                "session child {mode}: status={} stderr={}",
                output.status,
                String::from_utf8_lossy(&output.stderr)
            );
            if mode == "open" {
                assert!(
                    String::from_utf8_lossy(&output.stderr)
                        .contains("session child open: entering unclosed Drop")
                );
                // Open MPI's help-message transport can itself fail during abort.
                // Require its exit code, not localized/best-effort help output.
                assert_eq!(output.status.code(), Some(1));
            }
            if mode == "finalized" {
                assert!(
                    String::from_utf8_lossy(&output.stderr)
                        .contains("session child finalized: entering unclosed Drop")
                );
                assert!(!String::from_utf8_lossy(&output.stderr).contains("MPI_ABORT was invoked"));
                assert_eq!(
                    output.status.code(),
                    Some(134),
                    "SIGABRT, not an MPI call after finalize"
                );
            }
        }
    });
    cleanup_owned_temp_dir(&world, &directory);
}
