use super::*;

// Called by the existing single-MPI-init unit test, never initializes MPI itself.
pub(crate) fn native_contracts(
    path: &Path,
    source: &pencil_array::PencilArray<i32, 2, 2>,
    dest: &mut pencil_array::PencilArray<i32, 2, 2>,
) {
    let comm = source.pencil().topology().communicator();
    let before = ffi::HDF_FILE_CALLS.with(std::cell::Cell::get);
    let dup_before = ffi::DUPLICATE_CALLS.with(std::cell::Cell::get);
    assert!(Hdf5FileSession::create("", comm, &crate::MpiIoOptions::default()).is_err());
    if comm.size() > 1 {
        let different = path.with_extension(comm.rank().to_string());
        assert!(Hdf5FileSession::create(different, comm, &crate::MpiIoOptions::default()).is_err());
        let result = if comm.rank() == 0 {
            Hdf5FileSession::open_read(path, comm, &crate::MpiIoOptions::default())
        } else {
            Hdf5FileSession::create(path, comm, &crate::MpiIoOptions::default())
        };
        assert!(result.is_err());
        let hint = if comm.rank() == 0 {
            "1048576"
        } else {
            "2097152"
        };
        assert!(
            Hdf5FileSession::create(
                path,
                comm,
                &crate::MpiIoOptions::default().hint("cb_buffer_size", hint)
            )
            .is_err()
        );
    }
    assert_eq!(ffi::HDF_FILE_CALLS.with(std::cell::Cell::get), before);
    assert_eq!(ffi::DUPLICATE_CALLS.with(std::cell::Cell::get), dup_before);
    let mut session = Hdf5FileSession::create(
        path,
        comm,
        &crate::MpiIoOptions::default().hint("cb_buffer_size", "1048576"),
    )
    .unwrap();
    if std::env::var_os("PENCIL_HDF_NATIVE_CLOSE_CHILD").is_some() {
        // Deliberately invalid native identifier: H5Fclose must fail, not recover.
        // The live file is intentionally left for process termination in this child.
        assert_eq!(comm.size(), 1);
        session.file = Some(-1);
        eprintln!("native-close child entering H5Fclose with invalid identifier");
        let _ = session.close();
        panic!("native-close failure returned instead of fail-stop");
    }
    assert_eq!(ffi::test_info_arguments()[2], 1048576);
    let file = session.file;
    let duplicate = session.duplicate;
    session.create_group("one").unwrap();
    session.create_group("one/two").unwrap();
    let write = crate::Hdf5WriteOptions::default();
    let read = crate::Hdf5ReadOptions::default();
    for mode in [crate::MpiIoMode::Collective, crate::MpiIoMode::Independent] {
        let options = write.clone().chunks(vec![2, 2]).mode(mode);
        let source_before = source.as_slice().to_vec();
        native::FAIL_XFER_MODE.with(|f| f.set(comm.rank() == comm.size() - 1));
        assert!(
            session
                .write("one/two/data", source.view(), &options)
                .is_err()
        );
        native::FAIL_XFER_MODE.with(|f| f.set(false));
        assert!(!session.poisoned);
        assert_eq!(session.file, file);
        assert_eq!(session.duplicate, duplicate);
        assert_eq!(source.as_slice(), source_before);
        let (parent, name) = session.parent("one/two/data").unwrap();
        require_absent(comm, parent, &name).unwrap();
        close_group(comm, parent).unwrap();
        assert!(session.catalog().unwrap().is_empty());
        assert_eq!(
            ffi::HDF_FILE_CALLS.with(std::cell::Cell::get),
            [before[0] + 1, before[1]]
        );
        assert_eq!(
            ffi::DUPLICATE_CALLS.with(std::cell::Cell::get),
            dup_before + 1
        );
    }
    session
        .write("one/two/data", source.view(), &write)
        .unwrap();
    let calls = ffi::test_payload_calls();
    assert_eq!(session.catalog().unwrap().len(), 1);
    assert_eq!(
        ffi::test_payload_calls(),
        calls,
        "catalog must not read payload"
    );
    for marker in [INCOMPLETE_MARKER, COMMIT_MARKER] {
        let (group, name) = session.parent("one/two/data").unwrap();
        check_kind(comm, group, &name, native::LinkObjectType::Dataset).unwrap();
        let dataset = collective_handle_phase(
            comm,
            native::dataset_open(group, &name),
            "test metadata corruption",
        )
        .unwrap();
        write_existing_attr(comm, dataset, ATTR_COMMIT, marker).unwrap();
        assert!(
            data::finish_session(
                comm,
                Hdf5Resources {
                    group: Some(group),
                    dataset: Some(dataset),
                    ..Hdf5Resources::default()
                }
            )
            .is_none()
        );
        if marker == INCOMPLETE_MARKER {
            assert!(session.catalog().is_err());
            assert_eq!(
                ffi::test_payload_calls(),
                calls,
                "invalid catalog must not read payload"
            );
        }
    }
    assert!(
        session
            .write(
                "changed_hint",
                source.view(),
                &write.clone().hint("cb_buffer_size", "2097152")
            )
            .is_err()
    );
    dest.as_mut_slice().fill(-991);
    let old = dest.as_slice().to_vec();
    SESSION_FAULT.with(|f| f.set(2));
    assert!(
        session
            .read("one/two/data", dest.view_mut(), &read)
            .is_err()
    );
    SESSION_FAULT.with(|f| f.set(0));
    assert_eq!(dest.as_slice(), old);
    session
        .read("one/two/data", dest.view_mut(), &read)
        .unwrap();
    assert_eq!(dest.as_slice(), source.as_slice());
    SESSION_FAULT.with(|f| f.set(1));
    assert!(session.write("uncertain", source.view(), &write).is_err());
    SESSION_FAULT.with(|f| f.set(0));
    assert!(session.poisoned);
    assert!(session.catalog().is_err());
    assert!(
        session
            .write("after_poison", source.view(), &write)
            .is_err()
    );
    assert_eq!(session.file, file);
    assert_eq!(session.duplicate, duplicate);
    assert_eq!(
        ffi::HDF_FILE_CALLS.with(std::cell::Cell::get),
        [before[0] + 1, before[1]]
    );
    assert_eq!(
        ffi::DUPLICATE_CALLS.with(std::cell::Cell::get),
        dup_before + 1
    );
    session.close().unwrap();
    assert_eq!(
        ffi::HDF_FILE_CALLS.with(std::cell::Cell::get),
        [before[0] + 1, before[1] + 1]
    );
    let saved = if comm.rank() == 0 {
        NEXT_ID.swap(u64::MAX, Ordering::Relaxed)
    } else {
        0
    };
    assert!(Hdf5FileSession::open_read(path, comm, &crate::MpiIoOptions::default()).is_err());
    if comm.rank() == 0 {
        NEXT_ID.store(saved, Ordering::Relaxed);
    }
    comm.barrier();
    if comm.rank() == 0 {
        let mpi_tmp = path.with_extension("close-child-mpi");
        std::fs::create_dir(&mpi_tmp).unwrap();
        let output = std::process::Command::new("/bin/bash")
            .env_clear()
            .env("PATH", std::env::var("PATH").unwrap())
            .env("LD_LIBRARY_PATH", std::env::var("LD_LIBRARY_PATH").unwrap_or_default())
            .env("PENCIL_HDF_NATIVE_CLOSE_CHILD", "1")
            .arg("-c")
            .arg("ulimit -c 0; exec mpiexec --mca orte_base_help_aggregate 0 --mca orte_tmpdir_base \"$2\" --oversubscribe -n 1 \"$1\" --exact tests::post_cleanup_errors_preserve_destination_and_valid_commits --nocapture")
            .arg("hdf-native-close-child")
            .arg(std::env::current_exe().unwrap())
            .arg(&mpi_tmp)
            .output().unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!(
            "native-close child: status={} stderr={stderr}",
            output.status
        );
        assert_eq!(output.status.code(), Some(1));
        assert!(stderr.contains("native-close child entering H5Fclose"));
        assert!(!stderr.contains("native-close failure returned"));
        // MPI abort exits 1; a returned close/panic would exit 101.
        // Do not depend on Open MPI's best-effort help-message transport.
        std::fs::remove_dir_all(mpi_tmp).unwrap();
    }
    comm.barrier();
    // Counter exhaustion returned before native open or duplication.
    assert_eq!(
        ffi::HDF_FILE_CALLS.with(std::cell::Cell::get),
        [before[0] + 1, before[1] + 1]
    );
    let mut a = Hdf5FileSession::open_read(path, comm, &crate::MpiIoOptions::default()).unwrap();
    let mut b = Hdf5FileSession::open_read(path, comm, &crate::MpiIoOptions::default()).unwrap();
    assert_ne!(a.id, b.id, "same-path sessions require distinct identities");
    a.close().unwrap();
    b.close().unwrap();
}
