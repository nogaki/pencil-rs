#![cfg(feature = "hdf5")]

use std::{
    fs,
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

use hdf5_metno::{Dataset, File, H5Type, datatype::ByteOrder};
use num_complex::Complex;
use pencil_io::{IoElement, IoError, read_hdf5_serial, write_hdf5_serial};

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
