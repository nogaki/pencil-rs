use super::*;

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
        Err(MpiSessionError::Io(IoError::CollectivePrecondition {
            phase: "session header flush"
        }))
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
