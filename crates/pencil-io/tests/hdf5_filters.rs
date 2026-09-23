#![cfg(feature = "parallel-hdf5")]

use hdf5_metno::filters::Filter;
use mpi::traits::*;
use num_complex::{Complex, Complex32};
use pencil_array::{ExtraShape, MpiTopology, Pencil, PencilArray};
use pencil_io::{
    Hdf5ReadOptions, Hdf5WriteOptions, MpiIoMode, read_hdf5_with_options, write_hdf5_with_options,
};

mod support;
use support::{cleanup_owned_temp_dir, owned_temp_dir};

fn array<T>(
    world: &mpi::topology::SimpleCommunicator,
    global: [usize; 2],
    value: T,
) -> PencilArray<T, 2, 2>
where
    T: pencil_io::IoElement,
{
    let topology = MpiTopology::<2>::new(world, [world.size() as usize, 1]).unwrap();
    let pencil = Pencil::<2, 2>::new(topology, global, [0, 1]).unwrap();
    PencilArray::from_elem(pencil, ExtraShape::scalar(), value).unwrap()
}

fn empty_extra<T>(world: &mpi::topology::SimpleCommunicator, value: T) -> PencilArray<T, 2, 2>
where
    T: pencil_io::IoElement,
{
    let topology = MpiTopology::<2>::new(world, [world.size() as usize, 1]).unwrap();
    let pencil = Pencil::<2, 2>::new(topology, [world.size() as usize, 2], [0, 1]).unwrap();
    PencilArray::from_elem(pencil, ExtraShape::new([0]).unwrap(), value).unwrap()
}

fn roundtrip<T>(
    world: &mpi::topology::SimpleCommunicator,
    dir: &std::path::Path,
    name: &str,
    value: T,
    make_value: impl Fn(usize) -> T,
) where
    T: pencil_io::IoElement + Copy + PartialEq + std::fmt::Debug,
{
    // For multiple ranks, global[0] is one smaller than the process grid: the
    // last rank has no local elements while the dataset remains non-zero.
    let global0 = (world.size() as usize).saturating_sub(1).max(1);
    roundtrip_global(world, dir, name, [global0, 2], value, make_value);
}

fn roundtrip_global<T>(
    world: &mpi::topology::SimpleCommunicator,
    dir: &std::path::Path,
    name: &str,
    global: [usize; 2],
    value: T,
    make_value: impl Fn(usize) -> T,
) where
    T: pencil_io::IoElement + Copy + PartialEq + std::fmt::Debug,
{
    let mut source = array(world, global, value);
    for (index, item) in source.as_mut_slice().iter_mut().enumerate() {
        *item = make_value(index);
    }
    let mut destination = array(world, global, value);
    let path = dir.join(name);
    let options = Hdf5WriteOptions::default()
        .chunks(vec![1, 2])
        .shuffle(true)
        .deflate(1);
    write_hdf5_with_options(&path, source.view(), &options).unwrap();
    destination.as_mut_slice().fill(value);
    read_hdf5_with_options(
        &path,
        destination.view_mut(),
        &Hdf5ReadOptions::default().mode(MpiIoMode::Independent),
    )
    .unwrap();
    assert_eq!(destination.as_slice(), source.as_slice());
}

#[test]
fn native_filters_and_empty_ranks() {
    let universe = mpi::initialize().expect("one MPI initialization");
    let world = universe.world();
    let dir = owned_temp_dir(&world, "pencil-hdf5-filters");

    roundtrip(&world, &dir, "f32.h5", 0.0_f32, |i| i as f32 + 0.25);
    roundtrip(&world, &dir, "f64.h5", 0.0_f64, |i| i as f64 - 3.5);
    roundtrip(&world, &dir, "complex.h5", Complex::new(0.0, 0.0), |i| {
        Complex::new(i as f64, -(i as f64))
    });
    roundtrip(
        &world,
        &dir,
        "complex32.h5",
        Complex32::new(0.0, 0.0),
        |i| Complex32::new(i as f32, -(i as f32)),
    );
    roundtrip_global(
        &world,
        &dir,
        "nondivisible.h5",
        [world.size() as usize + 1, 3],
        0.0_f32,
        |i| i as f32 + 1.0,
    );

    // A zero extra extent is still a real dataset. Every rank has a none
    // selection, but the global spatial extents remain non-zero.
    let empty_path = dir.join("extra-zero.h5");
    let empty = empty_extra(&world, 7.0_f32);
    let empty_options = Hdf5WriteOptions::default()
        .chunks(vec![1, 1, 2])
        .shuffle(true)
        .deflate(1);
    write_hdf5_with_options(&empty_path, empty.view(), &empty_options).unwrap();
    let mut empty_destination = empty_extra(&world, -1.0_f32);
    read_hdf5_with_options(
        &empty_path,
        empty_destination.view_mut(),
        &Hdf5ReadOptions::default(),
    )
    .unwrap();
    assert!(empty_destination.as_slice().is_empty());

    world.barrier();
    if world.rank() == 0 {
        let file = hdf5_metno::File::open(&empty_path).unwrap();
        let dataset = file.dataset("/pencil_io_v1/data").unwrap();
        assert_eq!(dataset.shape(), vec![0, world.size() as usize, 2]);
        assert_eq!(dataset.chunk(), Some(vec![1, 1, 2]));
        assert!(matches!(
            dataset.filters().as_slice(),
            [Filter::Shuffle, Filter::Deflate(1)]
        ));
    }
    world.barrier();

    for (name, options, expected) in [
        (
            "deflate-zero.h5",
            Hdf5WriteOptions::default().chunks(vec![1, 2]).deflate(0),
            vec![Filter::Deflate(0)],
        ),
        (
            "shuffle-only.h5",
            Hdf5WriteOptions::default().chunks(vec![1, 2]).shuffle(true),
            vec![Filter::Shuffle],
        ),
    ] {
        let path = dir.join(name);
        let values = array(&world, [world.size() as usize, 2], 1.0_f32);
        write_hdf5_with_options(&path, values.view(), &options).unwrap();
        world.barrier();
        if world.rank() == 0 {
            let file = hdf5_metno::File::open(&path).unwrap();
            let dataset = file.dataset("/pencil_io_v1/data").unwrap();
            assert_eq!(dataset.filters(), expected);
        }
        world.barrier();
    }

    // Validation occurs before H5Fcreate, so rejected descriptors must not
    // leave an empty or partial file behind.
    for (name, options) in [
        (
            "bad-level.h5",
            Hdf5WriteOptions::default().chunks(vec![1, 2]).deflate(10),
        ),
        (
            "bad-chunk.h5",
            Hdf5WriteOptions::default().chunks(vec![0, 1, 2]),
        ),
        (
            "bad-rank.h5",
            Hdf5WriteOptions::default().chunks(vec![1, 1, 2]),
        ),
        (
            "bad-missing-chunks.h5",
            Hdf5WriteOptions::default().shuffle(true),
        ),
        (
            "bad-independent-filter.h5",
            Hdf5WriteOptions::default()
                .mode(MpiIoMode::Independent)
                .chunks(vec![1, 2])
                .shuffle(true),
        ),
    ] {
        let path = dir.join(name);
        let invalid = array(&world, [world.size() as usize, 2], 1.0_f32);
        assert!(write_hdf5_with_options(&path, invalid.view(), &options).is_err());
        world.barrier();
        if world.rank() == 0 {
            assert!(!path.exists(), "invalid options created {path:?}");
        }
        world.barrier();
    }

    if world.size() > 1 {
        for (name, options) in [
            (
                "bad-crossrank-filter-flag.h5",
                if world.rank() == 0 {
                    Hdf5WriteOptions::default().chunks(vec![1, 2]).shuffle(true)
                } else {
                    Hdf5WriteOptions::default().chunks(vec![1, 2])
                },
            ),
            (
                "bad-crossrank-filter-level.h5",
                Hdf5WriteOptions::default()
                    .chunks(vec![1, 2])
                    .deflate(if world.rank() == 0 { 1 } else { 2 }),
            ),
        ] {
            let path = dir.join(name);
            let values = array(&world, [world.size() as usize, 2], 1.0_f32);
            assert!(write_hdf5_with_options(&path, values.view(), &options).is_err());
            world.barrier();
            if world.rank() == 0 {
                assert!(!path.exists(), "mismatched options created {path:?}");
            }
            world.barrier();
        }

        let retry_path = dir.join("crossrank-retry.h5");
        let values = array(&world, [world.size() as usize, 2], 1.0_f32);
        let valid = Hdf5WriteOptions::default()
            .chunks(vec![1, 2])
            .shuffle(true)
            .deflate(1);
        write_hdf5_with_options(&retry_path, values.view(), &valid).unwrap();
        world.barrier();
    }

    // Defaults are native contiguous, unfiltered HDF5 controls.
    let default_path = dir.join("defaults.h5");
    let defaults = array(&world, [world.size() as usize, 2], 1.0_f32);
    write_hdf5_with_options(&default_path, defaults.view(), &Hdf5WriteOptions::default()).unwrap();
    world.barrier();
    if world.rank() == 0 {
        let file = hdf5_metno::File::open(&default_path).unwrap();
        let dataset = file.dataset("/pencil_io_v1/data").unwrap();
        assert_eq!(dataset.chunk(), None);
        assert!(dataset.filters().is_empty());
    }
    world.barrier();
    cleanup_owned_temp_dir(&world, &dir);
    if world.rank() == 0 {
        println!("HDF5_FILTERS_OK");
    }
}
