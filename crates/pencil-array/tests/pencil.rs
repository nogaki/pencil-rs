use std::{ops::Range, sync::Arc};

use mpi::traits::*;
use pencil_array::{
    AxisError, AxisPermutation, MpiTopology, Pencil, PencilConfig, PencilError, SpatialAxis,
};

fn expected_range(global_len: usize, coordinate: usize, process_count: usize) -> Range<usize> {
    global_len * coordinate / process_count..global_len * (coordinate + 1) / process_count
}

#[test]
fn immutable_pencils_define_and_derive_row_major_layouts() {
    let universe = mpi::initialize().expect("MPI initialization failed");
    let world = universe.world();
    assert_eq!(world.size(), 4, "run this test with mpiexec -n 4");

    let topology = MpiTopology::<2>::new(&world, [2, 2]).unwrap();
    let global_shape = [10, 14, 6];
    let [c0, c1] = *topology.local_coords();

    let default = Pencil::<3, 2>::new_default(Arc::clone(&topology), global_shape).unwrap();
    assert!(Arc::ptr_eq(default.topology(), &topology));
    assert_eq!(default.global_shape(), &global_shape);
    assert_eq!(default.decomposition().map(SpatialAxis::index), [0, 1]);
    assert_eq!(
        default.permutation().axes().map(SpatialAxis::index),
        [0, 1, 2]
    );
    assert_eq!(
        default.local_ranges(),
        &[expected_range(10, c0, 2), expected_range(14, c1, 2), 0..6,]
    );
    let logical_shape = [
        expected_range(10, c0, 2).len(),
        expected_range(14, c1, 2).len(),
        6,
    ];
    assert_eq!(default.local_shape_logical(), logical_shape);
    assert_eq!(default.local_shape_memory(), logical_shape);
    assert_eq!(default.local_len(), logical_shape.into_iter().product());
    assert_eq!(default.global_len(), 10 * 14 * 6);
    assert_eq!(
        default.ranges_at([1, 0]).unwrap(),
        [expected_range(10, 1, 2), expected_range(14, 0, 2), 0..6,]
    );

    let swapped = Pencil::<3, 2>::new(Arc::clone(&topology), global_shape, [1, 0]).unwrap();
    assert_eq!(
        swapped.local_ranges(),
        &[expected_range(10, c1, 2), expected_range(14, c0, 2), 0..6,]
    );
    assert!(default.same_topology(&swapped));
    assert!(!default.same_distribution(&swapped));
    assert!(!default.same_layout(&swapped));

    let decomposition_changed = default.with_decomposition([0, 2]).unwrap();
    assert!(default.same_topology(&decomposition_changed));
    assert_eq!(decomposition_changed.global_shape(), &global_shape);
    assert_eq!(
        decomposition_changed
            .decomposition()
            .map(SpatialAxis::index),
        [0, 2]
    );

    let permutation = AxisPermutation::<3>::new([2, 0, 1]).unwrap();
    let permuted = default.with_permutation(permutation.clone()).unwrap();
    assert!(default.same_distribution(&permuted));
    assert!(!default.same_layout(&permuted));
    assert_eq!(
        permuted.local_shape_memory(),
        permutation.permute(logical_shape)
    );

    let reshaped = default.with_global_shape([12, 16, 8]).unwrap();
    assert!(default.same_topology(&reshaped));
    assert_eq!(reshaped.global_shape(), &[12, 16, 8]);
    assert_eq!(reshaped.decomposition().map(SpatialAxis::index), [0, 1]);
    assert_eq!(
        reshaped.permutation().axes().map(SpatialAxis::index),
        [0, 1, 2]
    );

    let copied_config = PencilConfig::from(default.as_ref());
    assert_eq!(copied_config.global_shape, global_shape);
    assert_eq!(copied_config.decomposition, [0, 1]);
    assert_eq!(
        copied_config.permutation.axes().map(SpatialAxis::index),
        [0, 1, 2]
    );

    let reconfigured = default
        .reconfigured(PencilConfig {
            global_shape: [12, 16, 8],
            decomposition: [0, 2],
            permutation: permutation.clone(),
        })
        .unwrap();
    assert_eq!(reconfigured.global_shape(), &[12, 16, 8]);
    assert_eq!(reconfigured.decomposition().map(SpatialAxis::index), [0, 2]);
    assert_eq!(reconfigured.permutation(), &permutation);

    let fully_decomposed = Pencil::<2, 2>::new(Arc::clone(&topology), [1, 1], [0, 1]).unwrap();
    let expected_fully_decomposed = [expected_range(1, c0, 2), expected_range(1, c1, 2)];
    let expected_local_len = expected_fully_decomposed.iter().map(Range::len).product();
    assert_eq!(fully_decomposed.local_ranges(), &expected_fully_decomposed);
    assert_eq!(fully_decomposed.local_len(), expected_local_len);

    assert_eq!(
        Pencil::<3, 2>::new(Arc::clone(&topology), [10, 0, 6], [0, 1]).unwrap_err(),
        PencilError::ZeroGlobalExtent { axis: 1 }
    );
    assert_eq!(
        Pencil::<3, 2>::new(Arc::clone(&topology), global_shape, [0, 0]).unwrap_err(),
        PencilError::InvalidDecomposition(AxisError::Duplicate { axis: 0 })
    );
    assert_eq!(
        Pencil::<3, 2>::new(Arc::clone(&topology), global_shape, [0, 3]).unwrap_err(),
        PencilError::InvalidDecomposition(AxisError::OutOfBounds {
            axis: 3,
            dimensions: 3,
        })
    );
    assert_eq!(
        Pencil::<3, 2>::new(Arc::clone(&topology), [usize::MAX, 2, 1], [0, 1],).unwrap_err(),
        PencilError::SizeOverflow
    );

    let topology_3d = MpiTopology::<3>::new(&world, [2, 2, 1]).unwrap();
    assert_eq!(
        Pencil::<2, 3>::new(topology_3d, [8, 8], [0, 1, 0]).unwrap_err(),
        PencilError::InvalidDimensionRelation {
            spatial: 2,
            topology: 3,
        }
    );
}
