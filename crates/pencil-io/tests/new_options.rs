//! Option API coverage.  Keep this in one MPI test: MPI may only be initialized once.

use mpi::collective::SystemOperation;
use mpi::traits::*;
use pencil_array::{ExtraShape, MpiTopology, Pencil, PencilArray};
use pencil_io::{
    MpiIoMode, MpiIoOptions, RawByteOrder, RawReadOptions, read_mpi, read_mpi_raw,
    read_mpi_with_options, write_mpi_with_options,
};

mod support;
use support::{cleanup_owned_temp_dir, owned_temp_dir};

fn root_write<C: CommunicatorCollectives>(world: &C, path: &std::path::Path, bytes: &[u8]) {
    let ok = if world.rank() == 0 {
        std::fs::write(path, bytes).is_ok()
    } else {
        true
    };
    let mut all = 0;
    world.all_reduce_into(&i32::from(ok), &mut all, SystemOperation::min());
    assert_eq!(all, 1);
    world.barrier();
}

fn reset<C: CommunicatorCollectives>(world: &C, path: &std::path::Path) {
    if world.rank() == 0 {
        let _ = std::fs::remove_file(path);
    }
    world.barrier();
}

fn array<C: Communicator>(world: &C, grid: [usize; 2], value: i32) -> PencilArray<i32, 2, 2> {
    let topology = MpiTopology::<2>::new(world, grid).unwrap();
    let pencil = Pencil::<2, 2>::new(
        topology,
        [world.size() as usize, world.size() as usize],
        [0, 1],
    )
    .unwrap();
    PencilArray::from_elem(pencil, ExtraShape::scalar(), value).unwrap()
}

fn raw_bytes(n: usize, prefix: usize, trailer: usize) -> Vec<u8> {
    let mut bytes = vec![0xa5; prefix];
    for value in 0..(n * n) {
        bytes.extend_from_slice(&(value as i32).to_be_bytes());
    }
    bytes.extend(std::iter::repeat_n(0x5a, trailer));
    bytes
}

fn raw_case<T: pencil_io::IoElement + PartialEq + std::fmt::Debug>(
    world: &mpi::topology::SimpleCommunicator,
    dir: &std::path::Path,
    value: impl Fn(usize) -> T,
    encode: impl Fn(T, bool) -> Vec<u8>,
) {
    let topology = MpiTopology::<2>::new(world, [world.size() as usize, 1]).unwrap();
    let pencil = Pencil::<2, 2>::new_permuted(
        topology,
        [3, 5],
        [0, 1],
        pencil_array::AxisPermutation::new([1, 0]).unwrap(),
    )
    .unwrap();
    let mut array =
        PencilArray::from_elem(pencil, ExtraShape::new([2]).unwrap(), value(99)).unwrap();
    for order in [
        RawByteOrder::Native,
        RawByteOrder::Little,
        RawByteOrder::Big,
    ] {
        let big = order == RawByteOrder::Big
            || (order == RawByteOrder::Native && cfg!(target_endian = "big"));
        let mut bytes = vec![0xa5; 7];
        for i in 0..30 {
            bytes.extend(encode(value(i), big));
        }
        bytes.extend([0xff; 9]);
        let path = dir.join(format!("scalar-{}-{order:?}.raw", T::CODE));
        root_write(world, &path, &bytes);
        read_mpi_raw(
            &path,
            array.view_mut(),
            RawReadOptions::default().byte_offset(7).byte_order(order),
        )
        .unwrap();
        let v = array.view();
        let r = v.pencil().local_ranges();
        for e in 0..2 {
            for x in 0..r[0].len() {
                for y in 0..r[1].len() {
                    assert_eq!(
                        v.get_local(&[e], [x, y]),
                        Some(&value(e * 15 + (x + r[0].start) * 5 + y + r[1].start))
                    );
                }
            }
        }
    }
}
fn all_raw_scalars(world: &mpi::topology::SimpleCommunicator, dir: &std::path::Path) {
    macro_rules! scalar {
        ($t:ty) => {
            raw_case(
                world,
                dir,
                |i| i as $t,
                |x: $t, big| {
                    if big {
                        x.to_be_bytes().to_vec()
                    } else {
                        x.to_le_bytes().to_vec()
                    }
                },
            );
        };
    }
    scalar!(i8);
    scalar!(u8);
    scalar!(i16);
    scalar!(u16);
    scalar!(i32);
    scalar!(u32);
    scalar!(i64);
    scalar!(u64);
    scalar!(f32);
    scalar!(f64);
    macro_rules! complex {
        ($t:ty) => {
            raw_case(
                world,
                dir,
                |i| num_complex::Complex::<$t>::new(i as $t + 0.25, -(i as $t) - 0.5),
                |x: num_complex::Complex<$t>, big| {
                    let mut bytes = Vec::new();
                    for v in [x.re, x.im] {
                        bytes.extend(if big {
                            v.to_be_bytes()
                        } else {
                            v.to_le_bytes()
                        });
                    }
                    bytes
                },
            );
        };
    }
    complex!(f32);
    complex!(f64);
}

// Every rank must report the descriptor error, not a later native I/O failure.
fn descriptor_rejected<C: CommunicatorCollectives>(
    world: &C,
    result: Result<(), pencil_io::NamedIoError>,
) {
    let rejected = i32::from(matches!(
        result,
        Err(pencil_io::NamedIoError::Io(
            pencil_io::IoError::CollectiveDescriptorMismatch
        ))
    ));
    let mut all = 0;
    world.all_reduce_into(&rejected, &mut all, SystemOperation::min());
    assert_eq!(all, 1, "all ranks must reject the descriptor");
}

fn named_mpi_mismatches<C: CommunicatorCollectives>(
    world: &C,
    directory: &std::path::Path,
    source: &PencilArray<i32, 2, 2>,
) {
    use pencil_io::*;

    let matched = MpiIoOptions::default()
        .mode(MpiIoMode::Independent)
        .hint("access", "write");
    let read_matched = MpiIoOptions::default()
        .mode(MpiIoMode::Independent)
        .hint("access", "write");
    let mut destination = array(world, [1, world.size() as usize], -888);
    let second = array(world, [world.size() as usize, 1], 19);
    // The guard is collective; one-rank runs still exercise matched recovery below.
    for case in 0..3 {
        let path = directory.join(format!("named-mpi-mismatch-{case}"));
        let mut differing = MpiIoOptions::default();
        let mut read_differing = MpiIoOptions::default();
        if world.rank() != 0 {
            match case {
                0 => {
                    differing = differing.mode(MpiIoMode::Independent);
                    read_differing = read_differing.mode(MpiIoMode::Independent);
                }
                1 => {
                    differing = differing.hint("access", "write");
                    read_differing = read_differing.hint("access", "write");
                }
                // Keep defaults: only API selection differs in the legacy case.
                _ => {}
            }
        }
        let legacy = case == 2 && world.rank() == 0;
        if world.size() > 1 {
            let result = if legacy {
                write_mpi_named(&path, "one", source.view())
            } else {
                write_mpi_named_with_options(&path, "one", source.view(), &differing)
            };
            descriptor_rejected(world, result);
            assert!(!path.exists(), "mismatch must precede file creation");
        }
        write_mpi_named_with_options(&path, "one", source.view(), &matched).unwrap();
        if world.size() > 1 {
            let before = std::fs::read(&path).unwrap();
            let result = if legacy {
                append_mpi_named(&path, "two", second.view())
            } else {
                append_mpi_named_with_options(&path, "two", second.view(), &differing)
            };
            descriptor_rejected(world, result);
            assert_eq!(
                std::fs::read(&path).unwrap(),
                before,
                "failed append changed the original file"
            );
            destination.as_mut_slice().fill(-888);
            let result = if legacy {
                read_mpi_named(&path, "one", destination.view_mut())
            } else {
                read_mpi_named_with_options(&path, "one", destination.view_mut(), &read_differing)
            };
            descriptor_rejected(world, result);
            assert!(destination.as_slice().iter().all(|&x| x == -888));
        }
        read_mpi_named_with_options(&path, "one", destination.view_mut(), &read_matched).unwrap();
        assert!(destination.as_slice().iter().all(|&x| x == 7));
        append_mpi_named_with_options(&path, "two", second.view(), &matched).unwrap();
        read_mpi_named_with_options(&path, "two", destination.view_mut(), &read_matched).unwrap();
        assert!(destination.as_slice().iter().all(|&x| x == 19));
        read_mpi_named_with_options(&path, "one", destination.view_mut(), &read_matched).unwrap();
        assert!(destination.as_slice().iter().all(|&x| x == 7));
    }
}

#[cfg(feature = "parallel-hdf5")]
fn named_hdf5_mismatches<C: CommunicatorCollectives>(
    world: &C,
    directory: &std::path::Path,
    source: &PencilArray<i32, 2, 2>,
) {
    use pencil_io::*;

    let matched = Hdf5WriteOptions::default()
        .mode(MpiIoMode::Independent)
        .hint("access", "write");
    let read_matched = Hdf5ReadOptions::default()
        .mode(MpiIoMode::Independent)
        .hint("access", "write");
    let mut destination = array(world, [1, world.size() as usize], -888);
    let second = array(world, [world.size() as usize, 1], 19);
    // The guard is collective; one-rank runs still exercise matched recovery below.
    for case in 0..4 {
        let path = directory.join(format!("named-hdf5-mismatch-{case}"));
        let mut differing = Hdf5WriteOptions::default();
        let mut read_differing = Hdf5ReadOptions::default();
        if world.rank() != 0 {
            match case {
                0 => {
                    differing = differing.mode(MpiIoMode::Independent);
                    read_differing = read_differing.mode(MpiIoMode::Independent);
                }
                1 => {
                    differing = differing.hint("access", "write");
                    read_differing = read_differing.hint("access", "write");
                }
                2 => differing = differing.chunks(vec![1, 1]),
                // Keep defaults: only API selection differs in the legacy case.
                _ => {}
            }
        }
        let legacy = case == 3 && world.rank() == 0;
        if world.size() > 1 {
            let result = if legacy {
                write_hdf5_named(&path, "one", source.view())
            } else {
                write_hdf5_named_with_options(&path, "one", source.view(), &differing)
            };
            descriptor_rejected(world, result);
            assert!(!path.exists(), "mismatch must precede file creation");
        }
        write_hdf5_named_with_options(&path, "one", source.view(), &matched).unwrap();
        if world.size() > 1 {
            let before = std::fs::read(&path).unwrap();
            let result = if legacy {
                append_hdf5_named(&path, "two", second.view())
            } else {
                append_hdf5_named_with_options(&path, "two", second.view(), &differing)
            };
            descriptor_rejected(world, result);
            assert_eq!(
                std::fs::read(&path).unwrap(),
                before,
                "failed append changed the original file"
            );
            // Chunk layout is a write-only option.
            if case != 2 {
                destination.as_mut_slice().fill(-888);
                let result = if legacy {
                    read_hdf5_named(&path, "one", destination.view_mut())
                } else {
                    read_hdf5_named_with_options(
                        &path,
                        "one",
                        destination.view_mut(),
                        &read_differing,
                    )
                };
                descriptor_rejected(world, result);
                assert!(destination.as_slice().iter().all(|&x| x == -888));
            }
        }
        read_hdf5_named_with_options(&path, "one", destination.view_mut(), &read_matched).unwrap();
        assert!(destination.as_slice().iter().all(|&x| x == 7));
        append_hdf5_named_with_options(&path, "two", second.view(), &matched).unwrap();
        read_hdf5_named_with_options(&path, "two", destination.view_mut(), &read_matched).unwrap();
        assert!(destination.as_slice().iter().all(|&x| x == 19));
        read_hdf5_named_with_options(&path, "one", destination.view_mut(), &read_matched).unwrap();
        assert!(destination.as_slice().iter().all(|&x| x == 7));
    }
}

#[test]
fn new_options_and_raw_contracts() {
    let universe = mpi::initialize().expect("MPI must initialize once");
    let world = universe.world();
    let directory = owned_temp_dir(&world, "pencil-io-new-options");
    let n = world.size() as usize;

    // Old writer + new option-bearing reader, including a permuted reader layout.
    let path = directory.join("native.pio");
    reset(&world, &path);
    let source = array(&world, [n, 1], 7);
    write_mpi_with_options(
        &path,
        source.view(),
        &MpiIoOptions::default()
            .mode(MpiIoMode::Independent)
            .hint("access", "write"),
    )
    .unwrap();
    let mut destination = array(&world, [1, n], -1);
    read_mpi(&path, destination.view_mut()).unwrap();
    read_mpi_with_options(
        &path,
        destination.view_mut(),
        &MpiIoOptions::default()
            .mode(MpiIoMode::Independent)
            .hint("access", "random"),
    )
    .unwrap();
    assert!(destination.as_slice().iter().all(|&x| x == 7));

    named_mpi_mismatches(&world, &directory, &source);
    #[cfg(feature = "parallel-hdf5")]
    named_hdf5_mismatches(&world, &directory, &source);

    all_raw_scalars(&world, &directory);

    // A descriptor mismatch must fail collectively, and an identical retry must recover.
    if world.size() > 1 {
        let axes = if world.rank() == 0 { [0, 1] } else { [1, 0] };
        let pencil = Pencil::<2, 2>::new(source.pencil().topology().clone(), [n, n], axes).unwrap();
        let mut wrong = PencilArray::from_elem(pencil, ExtraShape::scalar(), -888i32).unwrap();
        assert!(read_mpi_with_options(&path, wrong.view_mut(), &MpiIoOptions::default()).is_err());
        assert!(wrong.as_slice().iter().all(|&x| x == -888));
        let differing = MpiIoOptions::default().mode(if world.rank() == 0 {
            MpiIoMode::Collective
        } else {
            MpiIoMode::Independent
        });
        assert!(read_mpi_with_options(&path, destination.view_mut(), &differing).is_err());
        world.barrier();
        let old_new = if world.rank() == 0 {
            read_mpi(&path, destination.view_mut())
        } else {
            read_mpi_with_options(&path, destination.view_mut(), &MpiIoOptions::default())
        };
        assert!(old_new.is_err());
        let hints = MpiIoOptions::default()
            .hint("access", if world.rank() == 0 { "read" } else { "write" });
        assert!(read_mpi_with_options(&path, destination.view_mut(), &hints).is_err());
        read_mpi_with_options(&path, destination.view_mut(), &MpiIoOptions::default()).unwrap();
    }

    // Raw: prefix/trailer, big-endian scalar values, and a permuted distribution.
    let raw = directory.join("values.raw");
    let bytes = raw_bytes(n, 3, 9);
    root_write(&world, &raw, &bytes);
    let mut raw_destination = array(&world, [1, n], -1);
    read_mpi_raw(
        &raw,
        raw_destination.view_mut(),
        RawReadOptions::default()
            .byte_offset(3)
            .byte_order(RawByteOrder::Big)
            .mode(MpiIoMode::Independent)
            .hint("access", "sequential"),
    )
    .unwrap();
    let view = raw_destination.view();
    let ranges = view.pencil().local_ranges().clone();
    for x in 0..ranges[0].len() {
        for y in 0..ranges[1].len() {
            let gx = ranges[0].start + x;
            let gy = ranges[1].start + y;
            assert_eq!(view.get_local(&[], [x, y]), Some(&((gx * n + gy) as i32)));
        }
    }

    if world.size() > 1 {
        let axes = if world.rank() == 0 { [0, 1] } else { [1, 0] };
        let pencil =
            Pencil::<2, 2>::new(raw_destination.pencil().topology().clone(), [n, n], axes).unwrap();
        let mut wrong = PencilArray::from_elem(pencil, ExtraShape::scalar(), -888i32).unwrap();
        assert!(
            read_mpi_raw(
                &raw,
                wrong.view_mut(),
                RawReadOptions::default().byte_offset(3)
            )
            .is_err()
        );
        assert!(wrong.as_slice().iter().all(|&x| x == -888));
        let old_new = if world.rank() == 0 {
            read_mpi(&raw, raw_destination.view_mut())
        } else {
            read_mpi_raw(&raw, raw_destination.view_mut(), RawReadOptions::default())
        };
        assert!(old_new.is_err());
        let order = if world.rank() == 0 {
            RawByteOrder::Native
        } else {
            RawByteOrder::Little
        };
        assert!(
            read_mpi_raw(
                &raw,
                raw_destination.view_mut(),
                RawReadOptions::default().byte_order(order)
            )
            .is_err()
        );
        assert!(
            read_mpi_raw(
                &raw,
                raw_destination.view_mut(),
                RawReadOptions::default().byte_offset(world.rank() as u64)
            )
            .is_err()
        );
    }
    // Truncation is rejected without modifying the destination; a huge offset is rejected too.
    let before = raw_destination.as_slice().to_vec();
    root_write(&world, &raw, &bytes[..3 + n * n * 4 - 1]);
    assert!(
        read_mpi_raw(
            &raw,
            raw_destination.view_mut(),
            RawReadOptions::default()
                .byte_offset(3)
                .byte_order(RawByteOrder::Big)
        )
        .is_err()
    );
    assert_eq!(raw_destination.as_slice(), before.as_slice());
    root_write(&world, &raw, &bytes);
    assert!(
        read_mpi_raw(
            &raw,
            raw_destination.view_mut(),
            RawReadOptions::default().byte_offset(u64::MAX)
        )
        .is_err()
    );

    // Invalid hints, including duplicate keys, fail before native creation.
    for (key, value) in [
        (String::new(), "x".to_owned()),
        ("k\0".to_owned(), "x".to_owned()),
        ("k".to_owned(), "x\0y".to_owned()),
        ("k".to_owned(), "x".repeat(1025)),
        ("k".repeat(256), "x".to_owned()),
    ] {
        let options = RawReadOptions::default().hint(key, value);
        assert!(read_mpi_raw(&raw, raw_destination.view_mut(), options).is_err());
    }
    let duplicate = RawReadOptions::default()
        .hint("access", "read")
        .hint("access", "write");
    assert!(read_mpi_raw(&raw, raw_destination.view_mut(), duplicate).is_err());

    // Empty scalar payloads still participate in the collective operation.
    let empty_path = directory.join("empty.raw");
    root_write(&world, &empty_path, &[]);
    let topology = MpiTopology::<2>::new(&world, [n, 1]).unwrap();
    let pencil = Pencil::<2, 2>::new(topology, [n, n], [0, 1]).unwrap();
    let mut empty = PencilArray::from_elem(pencil, ExtraShape::new([0]).unwrap(), 3i32).unwrap();
    read_mpi_raw(&empty_path, empty.view_mut(), RawReadOptions::default()).unwrap();
    assert!(empty.as_slice().is_empty());

    let named = directory.join("named-options.pio");
    let controls = MpiIoOptions::default()
        .mode(MpiIoMode::Independent)
        .hint("access", "write");
    pencil_io::write_mpi_named_with_options(&named, "one", source.view(), &controls).unwrap();
    pencil_io::append_mpi_named_with_options(&named, "two", source.view(), &controls).unwrap();
    pencil_io::read_mpi_named_with_options(&named, "two", destination.view_mut(), &controls)
        .unwrap();
    assert!(destination.as_slice().iter().all(|&x| x == 7));

    #[cfg(feature = "parallel-hdf5")]
    hdf5_options(&world, &directory, &source);

    cleanup_owned_temp_dir(&world, &directory);
}

#[cfg(feature = "parallel-hdf5")]
fn hdf5_options<C: CommunicatorCollectives>(
    world: &C,
    directory: &std::path::Path,
    source: &PencilArray<i32, 2, 2>,
) {
    use hdf5_metno::File;
    use pencil_io::{
        Hdf5ReadOptions, Hdf5WriteOptions, read_hdf5_with_options, write_hdf5_with_options,
    };

    let path = directory.join("options.h5");
    if world.rank() == 0 {
        let _ = std::fs::remove_file(&path);
    }
    world.barrier();
    let write_options = Hdf5WriteOptions::default()
        .chunks(vec![2, 2])
        .mode(MpiIoMode::Independent)
        .hint("fapl", "sec2");
    write_hdf5_with_options(&path, source.view(), &write_options).unwrap();
    world.barrier();
    if world.rank() == 0 {
        let file = File::open(&path).unwrap();
        let dataset = file
            .group("/pencil_io_v1")
            .unwrap()
            .dataset("data")
            .unwrap();
        assert_eq!(dataset.chunk(), Some(vec![2, 2]));
    }
    world.barrier();

    let mut destination = array(world, [1, world.size() as usize], -1);
    let read_options = Hdf5ReadOptions::default()
        .mode(MpiIoMode::Independent)
        .hint("fapl", "sec2");
    read_hdf5_with_options(&path, destination.view_mut(), &read_options).unwrap();
    assert!(destination.as_slice().iter().all(|&x| x == 7));

    // HDF5 options are collective descriptors too; recovery proves no communicator poisoning.
    if world.size() > 1 {
        let differing = Hdf5ReadOptions::default().mode(if world.rank() == 0 {
            MpiIoMode::Collective
        } else {
            MpiIoMode::Independent
        });
        assert!(read_hdf5_with_options(&path, destination.view_mut(), &differing).is_err());
        world.barrier();
        let old_new = if world.rank() == 0 {
            pencil_io::read_hdf5(&path, destination.view_mut())
        } else {
            read_hdf5_with_options(&path, destination.view_mut(), &Hdf5ReadOptions::default())
        };
        assert!(old_new.is_err());
        let hints = Hdf5ReadOptions::default()
            .hint("access", if world.rank() == 0 { "read" } else { "write" });
        assert!(read_hdf5_with_options(&path, destination.view_mut(), &hints).is_err());
        read_hdf5_with_options(&path, destination.view_mut(), &Hdf5ReadOptions::default()).unwrap();
    }
    for (key, value) in [
        (String::new(), "x".to_owned()),
        ("k\0".to_owned(), "x".to_owned()),
        ("k".to_owned(), "x\0y".to_owned()),
        ("k".to_owned(), "x".repeat(1025)),
    ] {
        assert!(
            write_hdf5_with_options(
                directory.join("bad.h5"),
                source.view(),
                &Hdf5WriteOptions::default().hint(key, value)
            )
            .is_err()
        );
    }
    assert!(
        write_hdf5_with_options(
            directory.join("bad-duplicate.h5"),
            source.view(),
            &Hdf5WriteOptions::default().hint("k", "1").hint("k", "2")
        )
        .is_err()
    );
    for chunks in [vec![0, 2], vec![u32::MAX as usize, 2]] {
        assert!(
            write_hdf5_with_options(
                directory.join("invalid-chunk.h5"),
                source.view(),
                &Hdf5WriteOptions::default().chunks(chunks)
            )
            .is_err()
        );
    }
    let high_rank = PencilArray::from_elem(
        source.pencil().clone(),
        ExtraShape::new(vec![1; 31]).unwrap(),
        0i32,
    )
    .unwrap();
    assert!(
        write_hdf5_with_options(
            directory.join("invalid-rank.h5"),
            high_rank.view(),
            &Hdf5WriteOptions::default().chunks(vec![1; 33])
        )
        .is_err()
    );
    if world.size() > 1 {
        let chunks = if world.rank() == 0 {
            vec![1, 1]
        } else {
            vec![2, 2]
        };
        assert!(
            write_hdf5_with_options(
                directory.join("mismatched-chunks.h5"),
                source.view(),
                &Hdf5WriteOptions::default().chunks(chunks)
            )
            .is_err()
        );
    }
    let named = directory.join("named-options.h5");
    pencil_io::write_hdf5_named_with_options(&named, "one", source.view(), &write_options).unwrap();
    pencil_io::append_hdf5_named_with_options(&named, "two", source.view(), &write_options)
        .unwrap();
    pencil_io::read_hdf5_named_with_options(&named, "two", destination.view_mut(), &read_options)
        .unwrap();
    assert!(destination.as_slice().iter().all(|&x| x == 7));
    let empty =
        PencilArray::from_elem(source.pencil().clone(), ExtraShape::new([0]).unwrap(), 0i32)
            .unwrap();
    let empty_path = directory.join("chunked-empty.h5");
    write_hdf5_with_options(
        &empty_path,
        empty.view(),
        &Hdf5WriteOptions::default().chunks(vec![3, 8, 9]),
    )
    .unwrap();
    world.barrier();
    if world.rank() == 0 {
        let f = File::open(&empty_path).unwrap();
        let d = f.dataset("/pencil_io_v1/data").unwrap();
        assert_eq!(d.chunk(), Some(vec![3, 8, 9]));
        assert_eq!(d.shape()[0], 0);
    }
    world.barrier();
    assert!(
        write_hdf5_with_options(
            directory.join("bad-chunk.h5"),
            source.view(),
            &Hdf5WriteOptions::default().chunks(Vec::<usize>::new())
        )
        .is_err()
    );
}
