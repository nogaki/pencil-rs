#![cfg(feature = "hdf5")]

use std::{
    fs,
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

use hdf5_metno::{Dataset, File, H5Type, datatype::ByteOrder};
use num_complex::Complex;
use pencil_io::{IoElement, IoError, read_hdf5_serial, write_hdf5_serial};

#[cfg(target_os = "linux")]
mod serial_lock_tests {
    use super::{File, IoError, read_hdf5_serial, write_hdf5_serial};
    use std::{
        fs,
        io::{self, BufRead, BufReader, Read, Write},
        os::{
            fd::{AsRawFd, RawFd},
            raw::c_int,
            unix::fs::MetadataExt,
        },
        path::{Path, PathBuf},
        process::{Child, Command, ExitStatus, Stdio},
        sync::{Arc, Mutex},
        time::{SystemTime, UNIX_EPOCH},
    };

    const CHILD_MODE: &str = "PENCIL_SERIAL_LOCK_CHILD";
    const CHILD_FD: &str = "PENCIL_SERIAL_LOCK_FD";
    const CHILD_PATH: &str = "PENCIL_SERIAL_LOCK_PATH";
    const F_GETFD: c_int = 1;
    const FD_CLOEXEC: c_int = 1;
    const LOCK_SH: c_int = 1;
    const LOCK_NB: c_int = 4;
    const LOCK_UN: c_int = 8;

    unsafe extern "C" {
        fn fcntl(fd: c_int, command: c_int, ...) -> c_int;
        fn flock(fd: c_int, operation: c_int) -> c_int;
    }

    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            let stamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir()
                .join(format!("pencil-serial-lock-{}-{stamp}", std::process::id()));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn same_inode(left: &fs::Metadata, right: &fs::Metadata) -> bool {
        left.dev() == right.dev() && left.ino() == right.ino()
    }

    fn inheritable_fd(path: &Path) -> io::Result<Option<RawFd>> {
        let expected = fs::metadata(path)?;
        for entry in fs::read_dir("/proc/self/fd")? {
            let entry = entry?;
            let Ok(fd) = entry.file_name().to_string_lossy().parse::<RawFd>() else {
                continue;
            };
            let Ok(actual) = fs::metadata(entry.path()) else {
                continue;
            };
            if !same_inode(&expected, &actual) {
                continue;
            }
            let flags = unsafe { fcntl(fd, F_GETFD) };
            if flags >= 0 && flags & FD_CLOEXEC == 0 {
                return Ok(Some(fd));
            }
        }
        Ok(None)
    }

    fn has_target_fd(path: &Path) -> io::Result<bool> {
        let expected = fs::metadata(path)?;
        for entry in fs::read_dir("/proc/self/fd")? {
            let Ok(entry) = entry else { continue };
            if fs::metadata(entry.path()).is_ok_and(|actual| same_inode(&expected, &actual)) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn exclusive_lock_held(path: &Path) -> io::Result<bool> {
        let probe = fs::File::open(path)?;
        let result = unsafe { flock(probe.as_raw_fd(), LOCK_SH | LOCK_NB) };
        if result == 0 {
            if unsafe { flock(probe.as_raw_fd(), LOCK_UN) } < 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(false)
        } else {
            let error = io::Error::last_os_error();
            if error.kind() == io::ErrorKind::WouldBlock {
                Ok(true)
            } else {
                Err(error)
            }
        }
    }

    struct HeldChild {
        child: Child,
        stdout: Option<BufReader<std::process::ChildStdout>>,
        finished: bool,
    }

    impl HeldChild {
        fn release_and_wait(&mut self) -> io::Result<(ExitStatus, String)> {
            if let Some(stdin) = self.child.stdin.as_mut() {
                let _ = stdin.write_all(b"RELEASE\n");
            }
            let mut tail = String::new();
            if let Some(stdout) = self.stdout.as_mut() {
                stdout.read_to_string(&mut tail)?;
            }
            let status = self.child.wait()?;
            self.finished = true;
            Ok((status, tail))
        }
    }

    impl Drop for HeldChild {
        fn drop(&mut self) {
            if !self.finished {
                let _ = self.child.kill();
                let _ = self.child.wait();
            }
        }
    }

    fn child_command() -> Command {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                "serial_lock_tests::child",
                "--nocapture",
                "--test-threads=1",
            ])
            .stderr(Stdio::inherit());
        command
    }

    fn start_holder(fd: RawFd, path: &Path) -> Result<(HeldChild, String), String> {
        let mut command = child_command();
        let child = command
            .env(CHILD_MODE, "hold")
            .env(CHILD_FD, fd.to_string())
            .env(CHILD_PATH, path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .map_err(|error| format!("spawn holder: {error}"))?;
        let mut held = HeldChild {
            child,
            stdout: None,
            finished: false,
        };
        held.stdout = held.child.stdout.take().map(BufReader::new);
        let mut ready = String::new();
        loop {
            let mut line = String::new();
            let read = held
                .stdout
                .as_mut()
                .expect("piped child stdout")
                .read_line(&mut line)
                .map_err(|error| format!("read holder handshake: {error}"))?;
            if read == 0 {
                return Err(format!("holder exited before fd proof; output: {ready}"));
            }
            ready.push_str(&line);
            if line.contains("PI_CHILD_READY ") {
                return Ok((held, line.trim().to_owned()));
            }
        }
    }

    type HolderState = Arc<Mutex<Option<Result<(HeldChild, String), String>>>>;

    struct HoldFdPath {
        path: PathBuf,
        state: HolderState,
    }

    impl AsRef<Path> for HoldFdPath {
        fn as_ref(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for HoldFdPath {
        fn drop(&mut self) {
            let result = match inheritable_fd(&self.path) {
                Ok(Some(fd)) => start_holder(fd, &self.path),
                Ok(None) => Err("no inheritable descriptor for target inode".into()),
                Err(error) => Err(format!("inspect target descriptor: {error}")),
            };
            *self.state.lock().unwrap() = Some(result);
        }
    }

    #[test]
    fn child() {
        let Ok(mode) = std::env::var(CHILD_MODE) else {
            return;
        };
        match mode.as_str() {
            "hold" => {
                let fd: RawFd = std::env::var(CHILD_FD).unwrap().parse().unwrap();
                let path = PathBuf::from(std::env::var_os(CHILD_PATH).unwrap());
                let actual = fs::metadata(format!("/proc/self/fd/{fd}")).unwrap();
                let expected = fs::metadata(path).unwrap();
                assert!(
                    same_inode(&actual, &expected),
                    "exec child did not retain target fd"
                );
                let flags = unsafe { fcntl(fd, F_GETFD) };
                assert!(
                    flags >= 0 && flags & FD_CLOEXEC == 0,
                    "inherited fd flags={flags}"
                );
                println!(
                    "PI_CHILD_READY fd={fd} dev={} ino={}",
                    actual.dev(),
                    actual.ino()
                );
                io::stdout().flush().unwrap();
                let mut release = String::new();
                io::stdin().lock().read_line(&mut release).unwrap();
                assert_eq!(release.trim(), "RELEASE");
            }
            "lock-error" => {
                assert_eq!(
                    std::env::var("HDF5_USE_FILE_LOCKING").as_deref(),
                    Ok("TRUE")
                );
                let path = PathBuf::from(std::env::var_os(CHILD_PATH).unwrap());
                let error = read_hdf5_serial(&path, &[1], &[], &mut [0u64]).unwrap_err();
                assert!(
                    matches!(
                        error,
                        IoError::Native {
                            operation: "HDF5 serial file lock",
                            ..
                        }
                    ),
                    "native lock failure was not distinguished: {error:?}"
                );
            }
            other => panic!("unknown child mode {other}"),
        }
    }

    #[test]
    fn inherited_writer_fd_does_not_leave_a_lock_or_block_read() {
        let dir = TempDir::new();
        let path = dir.0.join("writer.h5");
        let state = Arc::new(Mutex::new(None));
        write_hdf5_serial(
            HoldFdPath {
                path: path.clone(),
                state: state.clone(),
            },
            &[1],
            &[],
            &[73u64],
        )
        .unwrap();
        let Ok((mut child, proof)) = state.lock().unwrap().take().expect("path Drop handshake")
        else {
            eprintln!("SKIP writer-lock fixture: target fd was not inheritable");
            return;
        };
        eprintln!("writer child proof: {proof}");
        if has_target_fd(&path).unwrap_or(true) {
            eprintln!("SKIP writer-lock fixture: parent still has a target fd after return");
            let (status, output) = child.release_and_wait().unwrap();
            assert!(status.success(), "holder child: {status}; {output}");
            return;
        }
        let before = exclusive_lock_held(&path).expect("independent flock probe");
        eprintln!("writer lock retained after return: {before}");
        let mut actual = [0u64];
        if let Err(error) = read_hdf5_serial(&path, &[1], &[], &mut actual) {
            assert!(
                matches!(
                    error,
                    IoError::Native {
                        operation: "HDF5 serial open" | "HDF5 serial file lock",
                        ..
                    }
                ),
                "unexpected read failure with inherited target fd: {error:?}"
            );
            panic!("inherited writer descriptor blocked read: {error:?}");
        }
        assert_eq!(actual, [73]);
        assert!(
            !exclusive_lock_held(&path).expect("independent flock probe after read"),
            "writer left an exclusive native lock after close/read"
        );
        let (status, output) = child.release_and_wait().unwrap();
        assert!(status.success(), "holder child: {status}; {output}");
    }

    #[test]
    fn reader_ignores_an_inherited_external_writer_lock() {
        let dir = TempDir::new();
        let path = dir.0.join("external-writer.h5");
        write_hdf5_serial(&path, &[1], &[], &[91u64]).unwrap();
        let writer = File::open_rw(&path).unwrap();
        let fd = match inheritable_fd(&path) {
            Ok(Some(fd)) => fd,
            other => {
                let _ = writer.close();
                eprintln!("SKIP external-lock fixture: target fd unavailable: {other:?}");
                return;
            }
        };
        let (mut child, proof) = match start_holder(fd, &path) {
            Ok(holder) => holder,
            Err(reason) => {
                let _ = writer.close();
                eprintln!("SKIP external-lock fixture: {reason}");
                return;
            }
        };
        eprintln!("external writer child proof: {proof}");
        writer.close().unwrap();
        if has_target_fd(&path).unwrap_or(true) {
            eprintln!("SKIP external-lock fixture: parent retained the HDF5 fd after close");
            let (status, output) = child.release_and_wait().unwrap();
            assert!(status.success(), "holder child: {status}; {output}");
            return;
        }
        if !exclusive_lock_held(&path).expect("independent flock probe after close") {
            eprintln!("SKIP external-lock fixture: child did not retain an exclusive lock");
            let (status, output) = child.release_and_wait().unwrap();
            assert!(status.success(), "holder child: {status}; {output}");
            return;
        }
        let mut actual = [0u64];
        if let Err(error) = read_hdf5_serial(&path, &[1], &[], &mut actual) {
            assert!(
                matches!(
                    error,
                    IoError::Native {
                        operation: "HDF5 serial open" | "HDF5 serial file lock",
                        ..
                    }
                ),
                "unexpected read failure under a verified writer lock: {error:?}"
            );
            panic!("reader rejected a valid file while external writer lock remained: {error:?}");
        }
        assert_eq!(actual, [91]);
        assert!(
            exclusive_lock_held(&path).unwrap(),
            "reader released another writer's lock"
        );
        let (status, output) = child.release_and_wait().unwrap();
        assert!(status.success(), "holder child: {status}; {output}");
    }

    #[test]
    fn native_lock_failure_has_a_distinct_operation() {
        let dir = TempDir::new();
        let path = dir.0.join("locked.h5");
        write_hdf5_serial(&path, &[1], &[], &[17u64]).unwrap();
        let writer = File::open_rw(&path).unwrap();
        if !exclusive_lock_held(&path).expect("independent flock probe before child") {
            eprintln!("SKIP lock-diagnostic fixture: HDF5 writer lock unavailable");
            writer.close().unwrap();
            return;
        }
        let output = child_command()
            .env(CHILD_MODE, "lock-error")
            .env(CHILD_PATH, &path)
            .env("HDF5_USE_FILE_LOCKING", "TRUE")
            .output()
            .unwrap();
        writer.close().unwrap();
        assert!(
            output.status.success(),
            "lock-error child status {}; stdout: {}; stderr: {}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn non_lock_open_failure_keeps_the_generic_operation() {
        let dir = TempDir::new();
        let mut destination = [0u64];
        let error =
            read_hdf5_serial(dir.0.join("missing.h5"), &[1], &[], &mut destination).unwrap_err();
        assert!(
            matches!(
                error,
                IoError::Native {
                    operation: "HDF5 serial open",
                    ..
                }
            ),
            "non-lock open error was mislabeled: {error:?}"
        );
    }
}

#[path = "../src/serial_hdf5/global_id.rs"]
mod global_id;

#[test]
fn resolves_static_and_dll_global_id_abis() {
    use hdf5_metno_sys::h5i::hid_t;

    let ids: [hid_t; 2] = [1, 0x1_0000_0001];
    for expected in ids {
        let static_binding: hid_t = expected;
        // SAFETY: the hid_t implementation copies this initialized ID and dereferences no pointer.
        assert_eq!(unsafe { global_id::resolve(&static_binding) }, expected);

        let dll_storage: hid_t = expected;
        let dll_binding: usize = std::ptr::from_ref(&dll_storage) as usize;
        // SAFETY: dll_binding encodes a live, aligned hid_t pointer for this call, exactly as the
        // DLL-import ABI requires; dll_storage remains alive and unchanged through the dereference.
        assert_eq!(unsafe { global_id::resolve(&dll_binding) }, expected);
    }
}

const DATA: &str = "/pencil_io_v1/data";
const COMMIT: u64 = 0x434f_4d4d_4954_5445;

fn bytes<T: IoElement>(values: &[T]) -> Vec<u8> {
    let mut out = Vec::new();
    for &value in values {
        value.encode_le(&mut out);
    }
    out
}

fn attribute<T: H5Type>(dataset: &Dataset, name: &str, shape: &[usize], values: &[T]) {
    if dataset.attr_names().unwrap().iter().any(|n| n == name) {
        dataset.delete_attr(name).unwrap();
    }
    dataset
        .new_attr::<T>()
        .shape(shape)
        .create(name)
        .unwrap()
        .write_raw(values)
        .unwrap();
}

fn roundtrip<T: IoElement>(path: &Path, global: &[usize], extra: &[usize], values: &[T]) {
    write_hdf5_serial(path, global, extra, values).unwrap();
    let original = fs::read(path).unwrap();
    let mut destination = vec![T::decode_le(&vec![0; T::WIDTH]); values.len()];
    read_hdf5_serial(path, global, extra, &mut destination).unwrap();
    assert_eq!(bytes(&destination), bytes(values), "{}", path.display());
    assert_eq!(fs::read(path).unwrap(), original);

    let file = File::open(path).unwrap();
    let dataset = file.dataset(DATA).unwrap();
    assert_eq!(
        dataset.shape(),
        extra.iter().chain(global).copied().collect::<Vec<_>>()
    );
    assert_eq!(dataset.dtype().unwrap().size(), T::WIDTH);
    let mut attrs = vec![
        ("version", vec![1]),
        ("commit", vec![COMMIT]),
        ("n", vec![global.len() as u64]),
        ("type", vec![T::CODE]),
        ("width", vec![T::WIDTH as u64]),
        ("extra_rank", vec![extra.len() as u64]),
        ("global_shape", global.iter().map(|&n| n as u64).collect()),
        ("writer_grid", vec![1]),
        ("writer_permutation", (0..global.len() as u64).collect()),
    ];
    if !extra.is_empty() {
        attrs.push(("extra_shape", extra.iter().map(|&n| n as u64).collect()));
    }
    assert_eq!(dataset.attr_names().unwrap().len(), attrs.len());
    for (name, expected) in attrs {
        let attr = dataset.attr(&format!("pencil_io_{name}")).unwrap();
        assert_eq!(attr.shape(), [expected.len()]);
        assert_eq!(
            attr.dtype().unwrap().to_descriptor().unwrap(),
            u64::type_descriptor()
        );
        assert_eq!(attr.dtype().unwrap().byte_order(), ByteOrder::LittleEndian);
        assert_eq!(attr.read_raw::<u64>().unwrap(), expected, "{name}");
    }
    drop(dataset);
    file.close().unwrap();
}

fn check_type<T: IoElement>(dir: &Path, samples: &[T; 8]) {
    for extra in [&[][..], &[2, 3][..]] {
        // Different batches must not hide an accidental axis permutation.
        let values: Vec<_> = (0..8 * extra.iter().product::<usize>())
            .map(|i| samples[(i + i / 8) % 8])
            .collect();
        let path = dir.join(format!("{}-{}.h5", T::CODE, extra.len()));
        roundtrip(&path, &[2, 4], extra, &values);
    }
}

fn rejected<T: IoElement>(
    path: &Path,
    global: &[usize],
    extra: &[usize],
    dest: &mut [T],
) -> IoError {
    let original = fs::read(path).ok();
    let before = bytes(dest);
    let error = read_hdf5_serial(path, global, extra, dest).expect_err("read must fail");
    assert_eq!(bytes(dest), before, "{error:?}");
    assert_eq!(fs::read(path).ok(), original, "read modified source");
    error
}

#[test]
fn serial_hdf5_contract() {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("pencil-serial-{}-{stamp}", std::process::id()));
    // Exclusive creation: never claim or remove somebody else's directory.
    fs::create_dir(&dir).unwrap();
    macro_rules! integers {
        ($($ty:ty),*) => { $(
            check_type(&dir, &[<$ty>::MIN, <$ty>::MAX, 0, 1, 2, 3, 4, 5]);
        )* };
    }
    integers!(i8, u8, i16, u16, i32, u32, i64, u64);
    // Signed zeros, subnormal, infinities, signalling and signed quiet NaN payloads.
    let f32s = [
        0, 0x80000000, 1, 0x7f800000, 0xff800000, 0x7f800001, 0x7fc12345, 0xffc54321,
    ]
    .map(f32::from_bits);
    let f64s = [
        0,
        0x8000000000000000,
        1,
        0x7ff0000000000000,
        0xfff0000000000000,
        0x7ff0000000000001,
        0x7ff8000000012345,
        0xfff8000000054321,
    ]
    .map(f64::from_bits);
    check_type(&dir, &f32s);
    check_type(&dir, &f64s);
    check_type(
        &dir,
        &std::array::from_fn(|i| Complex::new(f32s[i], f32s[7 - i])),
    );
    check_type(
        &dir,
        &std::array::from_fn(|i| Complex::new(f64s[i], f64s[7 - i])),
    );
    for (i, (global, extra)) in [
        (&[0, 4][..], &[][..]),
        (&[2, 0][..], &[2, 3][..]),
        (&[2, 4][..], &[0, 3][..]),
    ]
    .into_iter()
    .enumerate()
    {
        roundtrip::<u8>(&dir.join(format!("empty-{i}.h5")), global, extra, &[]);
    }
    roundtrip(&dir.join("rank32.h5"), &[1; 16], &[1; 16], &[7u8]);

    let base = dir.join("10-2.h5");
    let original = fs::read(&base).unwrap();
    assert!(write_hdf5_serial(&base, &[2, 4], &[2, 3], &[0f64; 48]).is_err());
    assert_eq!(fs::read(&base).unwrap(), original);
    let file = File::open(&base).unwrap();
    let expected: Vec<_> = (0..48).map(|i| f64s[(i + i / 8) % 8]).collect();
    assert_eq!(
        bytes(&file.dataset(DATA).unwrap().read_raw::<f64>().unwrap()),
        bytes(&expected)
    );
    file.close().unwrap();

    let nan = f64::from_bits(0xfff8000000abcdef);
    let mut destination = vec![nan; 48];
    let invalid = dir.join("invalid.h5");
    for (global, extra) in [
        (&[2, 4][..], &[][..]), // Wrong slice length.
        (&[][..], &[][..]),
        (&[][..], &[1][..]),
        (&[1; 33][..], &[][..]),
        (&[1; 32][..], &[1][..]),
        (&[usize::MAX, 2][..], &[][..]),
        (&[usize::MAX][..], &[2][..]),
        (&[1][..], &[usize::MAX, 2][..]),
    ] {
        let error = write_hdf5_serial(&invalid, global, extra, &[0u8]).unwrap_err();
        assert!(matches!(
            error,
            IoError::InvalidInput(_) | IoError::SizeLimit { .. }
        ));
        assert!(!invalid.exists());
        rejected(&base, global, extra, &mut [nan]);
    }
    rejected(&invalid, &[2, 4], &[2, 3], &mut destination);
    rejected(&base, &[4, 2], &[2, 3], &mut destination);
    rejected(&base, &[2, 4], &[3, 2], &mut destination);
    rejected(&base, &[2, 4], &[2, 3], &mut [0u64; 48]);

    let bad = dir.join("bad.h5");
    for (name, values) in [
        ("version", &[2][..]),
        ("commit", &[0x494e_434f_4d50_4c45][..]),
        ("commit", &[0][..]),
        ("n", &[1][..]),
        ("type", &[8][..]),
        ("width", &[4][..]),
        ("extra_rank", &[1][..]),
        ("extra_shape", &[3, 2][..]),
        ("global_shape", &[4, 2][..]),
        ("writer_grid", &[][..]),
        ("writer_grid", &[0][..]),
        ("writer_grid", &[u64::MAX, 2][..]),
        // The writer grid rank may not exceed the spatial rank.
        ("writer_grid", &[1, 1, 1][..]),
        ("writer_permutation", &[0, 0][..]),
        ("writer_permutation", &[0, 2][..]),
        ("writer_permutation", &[0][..]),
    ] {
        fs::copy(&base, &bad).unwrap();
        let file = File::open_rw(&bad).unwrap();
        attribute(
            &file.dataset(DATA).unwrap(),
            &format!("pencil_io_{name}"),
            &[values.len()],
            values,
        );
        file.close().unwrap();
        let error = rejected(&bad, &[2, 4], &[2, 3], &mut destination);
        match name {
            "commit" if values[0] == 0 => assert!(matches!(error, IoError::InvalidFile { .. })),
            "commit" => assert!(matches!(error, IoError::IncompleteFile)),
            _ => assert!(
                matches!(error, IoError::MetadataMismatch { .. }),
                "{name}: {error}"
            ),
        }
    }
    for case in ["missing", "rank2", "oversized", "u32", "datatype", "shape"] {
        fs::copy(&base, &bad).unwrap();
        let file = File::open_rw(&bad).unwrap();
        let dataset = file.dataset(DATA).unwrap();
        match case {
            "missing" => dataset.delete_attr("pencil_io_width").unwrap(),
            "rank2" => attribute(&dataset, "pencil_io_global_shape", &[1, 2], &[2u64, 4]),
            "oversized" => attribute(&dataset, "pencil_io_writer_grid", &[1025], &[1u64; 1025]),
            "u32" => attribute(&dataset, "pencil_io_type", &[1], &[10u32]),
            _ => {
                file.unlink(DATA).unwrap();
                let replacement = if case == "datatype" {
                    file.new_dataset::<u64>()
                        .shape([2, 3, 2, 4])
                        .create(DATA)
                        .unwrap()
                } else {
                    file.new_dataset::<f64>()
                        .shape([3, 2, 2, 4])
                        .create(DATA)
                        .unwrap()
                };
                for name in dataset.attr_names().unwrap() {
                    let values = dataset.attr(&name).unwrap().read_raw::<u64>().unwrap();
                    attribute(&replacement, &name, &[values.len()], &values);
                }
            }
        }
        drop(dataset);
        file.close().unwrap();
        let error = rejected(&bad, &[2, 4], &[2, 3], &mut destination);
        assert!(
            matches!(error, IoError::MetadataMismatch { .. }),
            "{case}: {error}"
        );
    }
    // Legacy scalar attributes and nonserial writer provenance remain readable.
    fs::copy(&base, &bad).unwrap();
    let file = File::open_rw(&bad).unwrap();
    let dataset = file.dataset(DATA).unwrap();
    attribute(&dataset, "pencil_io_version", &[], &[1u64]);
    attribute(&dataset, "pencil_io_writer_grid", &[2], &[2u64, 3]);
    attribute(&dataset, "pencil_io_writer_permutation", &[2], &[1u64, 0]);
    drop(dataset);
    file.close().unwrap();
    let before = fs::read(&bad).unwrap();
    read_hdf5_serial(&bad, &[2, 4], &[2, 3], &mut destination).unwrap();
    assert_eq!(bytes(&destination), bytes(&expected));
    assert_eq!(fs::read(&bad).unwrap(), before);

    fs::copy(dir.join("10-0.h5"), &bad).unwrap();
    let file = File::open_rw(&bad).unwrap();
    attribute(
        &file.dataset(DATA).unwrap(),
        "pencil_io_extra_shape",
        &[0],
        &[] as &[u64],
    );
    file.close().unwrap();
    rejected(&bad, &[2, 4], &[], &mut [nan; 8]);
    fs::remove_dir_all(&dir).unwrap();
}
