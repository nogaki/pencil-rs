use std::{ops::Range, sync::Arc};

use mpi::traits::*;
use pencil_array::{MpiTopology, Pencil, SpatialAxis};

fn expected_range(global_len: usize, coordinate: usize, process_count: usize) -> Range<usize> {
    global_len * coordinate / process_count..global_len * (coordinate + 1) / process_count
}

#[test]
fn row_major_defaults_and_decomposition_order_define_local_ranges() {
    let universe = mpi::initialize().expect("MPI initialization failed");
    let world = universe.world();
    assert_eq!(world.size(), 4, "run this test with mpiexec -n 4");

    let topology = MpiTopology::<2>::new(&world, [2, 2]).unwrap();
    let global_shape = [10, 14, 6];

    let default = Pencil::<3, 2>::new_default(Arc::clone(&topology), global_shape).unwrap();
    assert_eq!(default.decomposition().map(SpatialAxis::index), [0, 1]);
    assert_eq!(
        default.permutation().axes().map(SpatialAxis::index),
        [0, 1, 2]
    );

    let [c0, c1] = *topology.local_coords();
    assert_eq!(
        default.local_ranges(),
        &[expected_range(10, c0, 2), expected_range(14, c1, 2), 0..6,]
    );

    let swapped = Pencil::<3, 2>::new(Arc::clone(&topology), global_shape, [1, 0]).unwrap();
    assert_eq!(
        swapped.local_ranges(),
        &[expected_range(10, c1, 2), expected_range(14, c0, 2), 0..6,]
    );
    assert!(default.same_topology(&swapped));
    assert!(!default.same_distribution(&swapped));
}
