use mpi::traits::*;
use num_complex::Complex;
use pencil_array::{AxisPermutation, ExtraShape, MpiTopology, Pencil, PencilArray};
use pencil_io::IoElement;
#[cfg(feature = "parallel-hdf5")]
use pencil_io::{
    Hdf5ReadOptions, Hdf5WriteOptions, append_hdf5_named_collection, read_hdf5_named_collection,
    write_hdf5_named_collection,
};
use pencil_io::{
    MpiIoOptions, append_mpi_named_collection, read_mpi_named_collection,
    write_mpi_named_collection,
};
mod support;

#[test]
fn named_collections_mpi_roundtrip_append_and_collective_validation() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    named_collections_cover_all_scalar_types(&world);
    let dir = support::owned_temp_dir(&world, "io-named-collections");
    let topology = MpiTopology::<2>::new(&world, [world.size() as usize, 1]).unwrap();
    let pencil = Pencil::<2, 2>::new(topology.clone(), [3, 5], [0, 1]).unwrap();
    let comm = topology.communicator();
    let options = MpiIoOptions::default();
    let mut source: Vec<_> = (0..2)
        .map(|_| PencilArray::from_elem(pencil.clone(), ExtraShape::new([2]).unwrap(), 0i32))
        .collect::<Result<_, _>>()
        .unwrap();
    for (component, a) in source.iter_mut().enumerate() {
        let ranges = a.pencil().local_ranges().clone();
        for e in 0..2 {
            for x in 0..ranges[0].len() {
                for y in 0..ranges[1].len() {
                    *a.view_mut().get_local_mut(&[e], [x, y]).unwrap() = (component * 10_000
                        + e * 1_000
                        + (ranges[0].start + x) * 100
                        + ranges[1].start
                        + y)
                        as i32;
                }
            }
        }
    }
    let views: Vec<_> = source.iter().map(|a| a.view()).collect();
    let path = dir.join("named.pio");

    write_mpi_named_collection(&path, "first", comm, &views, &options).unwrap();
    append_mpi_named_collection(&path, "second", comm, &views, &options).unwrap();
    let empty_path = dir.join("empty.pio");
    assert!(write_mpi_named_collection(&empty_path, "empty", comm, &views[..0], &options).is_err());
    world.barrier();
    assert!(!empty_path.exists());
    write_mpi_named_collection(&empty_path, "empty", comm, &views, &options).unwrap();
    if world.size() > 1 {
        let n = if world.rank() == 0 { 1 } else { 2 };
        let count_bad = dir.join("count-bad.pio");
        assert!(
            write_mpi_named_collection(&count_bad, "bad", comm, &views[..n], &options).is_err()
        );
        world.barrier();
        assert!(!count_bad.exists());
        world.barrier();
        write_mpi_named_collection(&count_bad, "good", comm, &views, &options).unwrap();
    }
    let bad = PencilArray::from_elem(
        pencil.clone(),
        ExtraShape::new([if world.rank() == 0 { 3 } else { 2 }]).unwrap(),
        0i32,
    )
    .unwrap();
    let member_bad = dir.join("member-bad.pio");
    assert!(
        write_mpi_named_collection(
            &member_bad,
            "bad",
            comm,
            &[source[0].view(), bad.view()],
            &options
        )
        .is_err()
    );
    world.barrier();
    assert!(!member_bad.exists());
    world.barrier();
    write_mpi_named_collection(&member_bad, "good", comm, &views, &options).unwrap();

    let mut dest: Vec<_> = (0..2)
        .map(|_| PencilArray::from_elem(pencil.clone(), ExtraShape::new([2]).unwrap(), -7))
        .collect::<Result<_, _>>()
        .unwrap();
    read_mpi_named_collection(
        &path,
        "second",
        comm,
        &mut dest.iter_mut().map(|a| a.view_mut()).collect::<Vec<_>>(),
        &options,
    )
    .unwrap();
    assert_eq!(dest[0].as_slice(), source[0].as_slice());
    assert_eq!(dest[1].as_slice(), source[1].as_slice());

    let legacy = dir.join("collection-options.pio");
    let independent = MpiIoOptions::default().mode(pencil_io::MpiIoMode::Independent);
    pencil_io::write_mpi_collection_with_options(&legacy, comm, &views, &independent).unwrap();
    for member in &mut dest {
        member.as_mut_slice().fill(-8);
    }
    pencil_io::read_mpi_collection_with_options(
        &legacy,
        comm,
        &mut dest.iter_mut().map(|a| a.view_mut()).collect::<Vec<_>>(),
        &independent,
    )
    .unwrap();
    for (actual, expected) in dest.iter().zip(&source) {
        assert_eq!(actual.as_slice(), expected.as_slice());
    }

    let before: Vec<_> = dest.iter().map(|a| a.as_slice().to_vec()).collect();
    assert!(
        read_mpi_named_collection(
            &path,
            "missing",
            comm,
            &mut dest.iter_mut().map(|a| a.view_mut()).collect::<Vec<_>>(),
            &options,
        )
        .is_err()
    );
    assert!(dest.iter().zip(before).all(|(a, b)| a.as_slice() == b));

    let mut invalid =
        PencilArray::from_elem(pencil.clone(), ExtraShape::new([3]).unwrap(), -11i32).unwrap();
    let invalid_before = invalid.as_slice().to_vec();
    assert!(
        read_mpi_named_collection(
            &path,
            "second",
            comm,
            &mut [dest[0].view_mut(), invalid.view_mut()],
            &options,
        )
        .is_err()
    );
    assert_eq!(invalid.as_slice(), invalid_before.as_slice());
    write_mpi_named_collection(dir.join("read-retry.pio"), "retry", comm, &views, &options)
        .unwrap();

    // Name disagreement and operation disagreement must not create or mutate a file.
    if world.size() > 1 {
        let mismatch = dir.join("name-mismatch.pio");
        assert!(
            write_mpi_named_collection(
                &mismatch,
                format!("name-{}", world.rank()),
                comm,
                &views,
                &options,
            )
            .is_err()
        );
        world.barrier();
        assert!(!mismatch.exists());
        world.barrier();
        write_mpi_named_collection(&mismatch, "retry", comm, &views, &options).unwrap();

        let controls = dir.join("controls-mismatch.pio");
        let mixed = if world.rank() == 0 {
            MpiIoOptions::default().mode(pencil_io::MpiIoMode::Independent)
        } else {
            MpiIoOptions::default().hint("cb_buffer_size", "2097152")
        };
        assert!(write_mpi_named_collection(&controls, "mixed", comm, &views, &mixed).is_err());
        world.barrier();
        assert!(!controls.exists());
        world.barrier();
        write_mpi_named_collection(&controls, "retry", comm, &views, &options).unwrap();

        let opcode = dir.join("opcode-mismatch.pio");
        let result = if world.rank() == 0 {
            write_mpi_named_collection(&opcode, "x", comm, &views, &options)
        } else {
            append_mpi_named_collection(&opcode, "x", comm, &views, &options)
        };
        assert!(result.is_err());
        world.barrier();
        assert!(!opcode.exists());
        world.barrier();
        write_mpi_named_collection(&opcode, "retry", comm, &views, &options).unwrap();
    }

    #[cfg(feature = "parallel-hdf5")]
    {
        let hpath = dir.join("named.h5");
        let wo = Hdf5WriteOptions::default()
            .chunks(vec![2, 1, 3, 5])
            .shuffle(true)
            .deflate(1);
        let ro = Hdf5ReadOptions::default();
        let legacy = dir.join("collection-options.h5");
        pencil_io::write_hdf5_collection_with_options(&legacy, comm, &views, &wo).unwrap();
        for member in &mut dest {
            member.as_mut_slice().fill(-8);
        }
        pencil_io::read_hdf5_collection_with_options(
            &legacy,
            comm,
            &mut dest.iter_mut().map(|a| a.view_mut()).collect::<Vec<_>>(),
            &ro,
        )
        .unwrap();
        for (actual, expected) in dest.iter().zip(&source) {
            assert_eq!(actual.as_slice(), expected.as_slice());
        }
        write_hdf5_named_collection(&hpath, "first", comm, &views, &wo).unwrap();
        append_hdf5_named_collection(&hpath, "second", comm, &views, &wo).unwrap();
        for a in &mut dest {
            a.as_mut_slice().fill(-9);
        }
        read_hdf5_named_collection(
            &hpath,
            "second",
            comm,
            &mut dest.iter_mut().map(|a| a.view_mut()).collect::<Vec<_>>(),
            &ro,
        )
        .unwrap();
        assert_eq!(dest[0].as_slice(), source[0].as_slice());
        assert_eq!(dest[1].as_slice(), source[1].as_slice());

        // Collection members are stored as one dataset and can be read with a
        // different process grid and axis permutation.
        let reader_p = Pencil::<2, 2>::new_permuted(
            MpiTopology::new(&world, [1, world.size() as usize]).unwrap(),
            [3, 5],
            [1, 0],
            AxisPermutation::new([1, 0]).unwrap(),
        )
        .unwrap();
        let mut changed: Vec<_> = (0..2)
            .map(|_| PencilArray::from_elem(reader_p.clone(), ExtraShape::new([2]).unwrap(), -8))
            .collect::<Result<_, _>>()
            .unwrap();
        read_hdf5_named_collection(
            &hpath,
            "first",
            reader_p.topology().communicator(),
            &mut changed.iter_mut().map(|a| a.view_mut()).collect::<Vec<_>>(),
            &ro,
        )
        .unwrap();
        for (component, a) in changed.iter().enumerate() {
            let ranges = a.pencil().local_ranges().clone();
            for e in 0..2 {
                for x in 0..ranges[0].len() {
                    for y in 0..ranges[1].len() {
                        assert_eq!(
                            *a.view().get_local(&[e], [x, y]).unwrap(),
                            (component * 10_000
                                + e * 1_000
                                + (ranges[0].start + x) * 100
                                + ranges[1].start
                                + y) as i32
                        );
                    }
                }
            }
        }
    }

    support::cleanup_owned_temp_dir(&world, &dir);
    if world.rank() == 0 {
        println!("NAMED_COLLECTIONS_OK");
    }
}

fn scalar_named_roundtrip<T: IoElement + PartialEq + std::fmt::Debug>(
    world: &mpi::topology::SimpleCommunicator,
    dir: &std::path::Path,
    name: &str,
    value: T,
) {
    let topology = MpiTopology::<2>::new(world, [world.size() as usize, 1]).unwrap();
    let pencil = Pencil::<2, 2>::new(topology, [2, 2], [0, 1]).unwrap();
    let source = PencilArray::from_elem(pencil, ExtraShape::scalar(), value).unwrap();
    let comm = source.pencil().topology().communicator();
    let path = dir.join(name);
    let views = [source.view()];
    write_mpi_named_collection(&path, "scalar", comm, &views, &MpiIoOptions::default()).unwrap();
    let reader_p = Pencil::<2, 2>::new_permuted(
        MpiTopology::new(world, [1, world.size() as usize]).unwrap(),
        [2, 2],
        [1, 0],
        AxisPermutation::new([1, 0]).unwrap(),
    )
    .unwrap();
    let zero = T::decode_le(&[0; 16][..T::WIDTH]);
    let mut destination = PencilArray::from_elem(reader_p, ExtraShape::scalar(), zero).unwrap();
    let destination_topology = destination.pencil().topology().clone();
    let destination_comm = destination_topology.communicator();
    read_mpi_named_collection(
        &path,
        "scalar",
        destination_comm,
        &mut [destination.view_mut()],
        &MpiIoOptions::default(),
    )
    .unwrap();
    assert!(destination.as_slice().iter().all(|x| *x == value));

    #[cfg(feature = "parallel-hdf5")]
    {
        let hpath = dir.join(format!("{name}.h5"));
        write_hdf5_named_collection(&hpath, "scalar", comm, &views, &Hdf5WriteOptions::default())
            .unwrap();
        let mut hdest = PencilArray::from_elem(
            Pencil::<2, 2>::new_permuted(
                MpiTopology::new(world, [1, world.size() as usize]).unwrap(),
                [2, 2],
                [1, 0],
                AxisPermutation::new([1, 0]).unwrap(),
            )
            .unwrap(),
            ExtraShape::scalar(),
            zero,
        )
        .unwrap();
        let hdest_topology = hdest.pencil().topology().clone();
        let hdest_comm = hdest_topology.communicator();
        read_hdf5_named_collection(
            &hpath,
            "scalar",
            hdest_comm,
            &mut [hdest.view_mut()],
            &Hdf5ReadOptions::default(),
        )
        .unwrap();
        assert!(hdest.as_slice().iter().all(|x| *x == value));
    }
}

fn named_collections_cover_all_scalar_types(world: &mpi::topology::SimpleCommunicator) {
    let dir = support::owned_temp_dir(world, "io-named-scalars");
    scalar_named_roundtrip(world, &dir, "i8", -3i8);
    scalar_named_roundtrip(world, &dir, "u8", 3u8);
    scalar_named_roundtrip(world, &dir, "i16", -300i16);
    scalar_named_roundtrip(world, &dir, "u16", 300u16);
    scalar_named_roundtrip(world, &dir, "i32", -30_000i32);
    scalar_named_roundtrip(world, &dir, "u32", 30_000u32);
    scalar_named_roundtrip(world, &dir, "i64", -3_000_000i64);
    scalar_named_roundtrip(world, &dir, "u64", 3_000_000u64);
    scalar_named_roundtrip(world, &dir, "f32", 1.25f32);
    scalar_named_roundtrip(world, &dir, "f64", -1.25f64);
    scalar_named_roundtrip(world, &dir, "cf32", Complex::new(1.0f32, -2.0));
    scalar_named_roundtrip(world, &dir, "cf64", Complex::new(1.0f64, -2.0));
    support::cleanup_owned_temp_dir(world, &dir);
}
