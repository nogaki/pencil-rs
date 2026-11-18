use std::sync::Arc;

use mpi::traits::*;
use pencil_array::{
    AxisPermutation, ExtraShape, MpiTopology, Pencil, PencilArray, PencilArrayView,
};

fn process_grid(size: usize) -> [usize; 2] {
    match size {
        1 => [1, 1],
        4 => [2, 2],
        6 => [2, 3],
        other => panic!("run with 1, 4, or 6 MPI ranks, got {other}"),
    }
}

fn expected_range(length: usize, coordinate: usize, parts: usize) -> std::ops::Range<usize> {
    length * coordinate / parts..length * (coordinate + 1) / parts
}

fn column_major_offset(
    spatial_shape: [usize; 3],
    extra_shape: [usize; 2],
    extra: [usize; 2],
    spatial: [usize; 3],
) -> usize {
    let spatial_index =
        spatial[2] + spatial_shape[2] * (spatial[0] + spatial_shape[0] * spatial[1]);
    let extra_index = extra[0] + extra_shape[0] * extra[1];
    spatial_index + (spatial_shape[0] * spatial_shape[1] * spatial_shape[2]) * extra_index
}

#[test]
fn column_major_layout_contract() {
    let universe = mpi::initialize().expect("MPI initialization failed");
    let world = universe.world();
    let size = usize::try_from(world.size()).unwrap();
    let grid = process_grid(size);
    let topology = MpiTopology::<2>::new(&world, grid).unwrap();

    for c0 in 0..grid[0] {
        for c1 in 0..grid[1] {
            assert_eq!(
                topology.rank_at([c0, c1]).unwrap(),
                i32::try_from(c0 * grid[1] + c1).unwrap(),
                "Cartesian rank/coordinate enumeration must remain unchanged"
            );
        }
    }

    let permutation = AxisPermutation::new([2, 0, 1]).unwrap();
    let pencil =
        Pencil::<3, 2>::new_permuted(Arc::clone(&topology), [3, 5, 7], [1, 2], permutation)
            .unwrap();
    let extra_shape = ExtraShape::new([2, 3]).unwrap();
    let local = pencil.local_shape_logical();
    let memory_shape = vec![local[2], local[0], local[1], 2, 3];
    let local_spatial_len = local[0] * local[1] * local[2];
    let length = local_spatial_len * 2 * 3;
    let array = PencilArray::from_vec(
        Arc::clone(&pencil),
        extra_shape.clone(),
        (0..length).collect::<Vec<usize>>(),
    )
    .unwrap();

    assert_eq!(
        array.memory_shape(),
        memory_shape,
        "CM_LAYOUT_CONTRACT: memory shape must be permuted spatial axes followed by extra axes"
    );

    let default = Pencil::<3, 2>::new_default(Arc::clone(&topology), [3, 5, 7]).unwrap();
    assert_eq!(default.decomposition().map(|axis| axis.index()), [1, 2]);
    let coords = *topology.local_coords();
    assert_eq!(
        default.local_ranges(),
        &[
            0..3,
            expected_range(5, coords[0], grid[0]),
            expected_range(7, coords[1], grid[1]),
        ]
    );

    assert_eq!(array.logical_shape(), [local[0], local[1], local[2], 2, 3]);
    assert_eq!(array.local_spatial_shape(), local);
    assert_eq!(
        array.local_spatial_memory_shape(),
        [local[2], local[0], local[1]]
    );
    assert_eq!(array.len(), length);

    for e1 in 0..3 {
        for e0 in 0..2 {
            for x in 0..local[0] {
                for y in 0..local[1] {
                    for z in 0..local[2] {
                        let spatial = [x, y, z];
                        let extra = [e0, e1];
                        let offset = column_major_offset(local, [2, 3], extra, spatial);
                        assert_eq!(
                            array.get_local(&extra, spatial),
                            Some(&offset),
                            "CM_LAYOUT_CONTRACT: logical lookup must use CM spatial strides and contiguous extra batches"
                        );
                        let global = [
                            pencil.local_ranges()[0].start + x,
                            pencil.local_ranges()[1].start + y,
                            pencil.local_ranges()[2].start + z,
                        ];
                        assert_eq!(array.get_global(&extra, global), Some(&offset));
                    }
                }
            }
        }
    }

    let view = PencilArrayView::from_slice(pencil.as_ref(), &extra_shape, array.as_slice())
        .expect("compact external CM view");
    assert_eq!(view.memory_shape(), memory_shape);
    assert_eq!(view.logical_shape(), [local[0], local[1], local[2], 2, 3]);
    assert_eq!(view.as_slice(), array.as_slice());
    for e1 in 0..3 {
        for e0 in 0..2 {
            for x in 0..local[0] {
                for y in 0..local[1] {
                    for z in 0..local[2] {
                        let expected = column_major_offset(local, [2, 3], [e0, e1], [x, y, z]);
                        assert_eq!(view.get_local(&[e0, e1], [x, y, z]), Some(&expected));
                    }
                }
            }
        }
    }

    let coordinates: [Vec<usize>; 3] = std::array::from_fn(|axis| (0..[3, 5, 7][axis]).collect());
    let coordinate_axes = std::array::from_fn(|axis| coordinates[axis].as_slice());
    let grid_view = array.local_grid(coordinate_axes).unwrap();
    let ranges = pencil.local_ranges();
    let expected_points = (0..local[1])
        .flat_map(|y| {
            (0..local[0]).flat_map(move |x| {
                (0..local[2]).map(move |z| {
                    [
                        ranges[0].start + x,
                        ranges[1].start + y,
                        ranges[2].start + z,
                    ]
                })
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(
        grid_view
            .iter()
            .map(|point| point.map(|coordinate| *coordinate))
            .collect::<Vec<_>>(),
        expected_points,
        "LocalGrid must enumerate physical spatial axes with the first CM axis fastest"
    );

    let mut logical_values =
        PencilArray::from_elem(Arc::clone(&pencil), extra_shape.clone(), 0_i32).unwrap();
    let broadcast_shape = ExtraShape::new([1, 3]).unwrap();
    let mut broadcast =
        PencilArray::from_elem(Arc::clone(&pencil), broadcast_shape, 0_i32).unwrap();
    for e1 in 0..3 {
        for e0 in 0..2 {
            for x in 0..local[0] {
                for y in 0..local[1] {
                    for z in 0..local[2] {
                        let global = [
                            ranges[0].start + x,
                            ranges[1].start + y,
                            ranges[2].start + z,
                        ];
                        *logical_values.get_local_mut(&[e0, e1], [x, y, z]).unwrap() =
                            (e0 * 100_000
                                + e1 * 10_000
                                + global[0] * 100
                                + global[1] * 10
                                + global[2]) as i32;
                        *broadcast.get_local_mut(&[0, e1], [x, y, z]).unwrap() =
                            (1_000 + e1 * 100 + global[0] * 10 + global[1] + global[2]) as i32;
                    }
                }
            }
        }
    }
    let mut pointwise =
        PencilArray::from_elem(Arc::clone(&pencil), extra_shape.clone(), -1_i32).unwrap();
    pencil_array::pointwise2(
        &logical_values,
        &broadcast,
        &mut pointwise,
        |left, right| *left + *right,
    )
    .unwrap();
    for e1 in 0..3 {
        for e0 in 0..2 {
            for x in 0..local[0] {
                for y in 0..local[1] {
                    for z in 0..local[2] {
                        let global = [
                            ranges[0].start + x,
                            ranges[1].start + y,
                            ranges[2].start + z,
                        ];
                        let expected = (e0 * 100_000
                            + e1 * 10_000
                            + global[0] * 100
                            + global[1] * 10
                            + global[2]
                            + 1_000
                            + e1 * 100
                            + global[0] * 10
                            + global[1]
                            + global[2]) as i32;
                        assert_eq!(pointwise.get_local(&[e0, e1], [x, y, z]), Some(&expected));
                    }
                }
            }
        }
    }

    let gathered = pencil_array::gather(&logical_values.view(), 0).unwrap();
    if topology.rank() == 0 {
        let expected = (0..3)
            .flat_map(|e1| {
                (0..2).flat_map(move |e0| {
                    (0..7).flat_map(move |z| {
                        (0..5).flat_map(move |y| {
                            (0..3).map(move |x| e0 * 100_000 + e1 * 10_000 + x * 100 + y * 10 + z)
                        })
                    })
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(gathered.as_deref(), Some(expected.as_slice()));
    } else {
        assert!(gathered.is_none());
    }

    let sparse = Pencil::<3, 2>::new_permuted(
        Arc::clone(&topology),
        [3, 1, 7],
        [1, 2],
        AxisPermutation::new([2, 0, 1]).unwrap(),
    )
    .unwrap();
    if size > 1 && topology.local_coords()[0] == 0 {
        assert_eq!(
            sparse.local_len(),
            0,
            "floor partition must retain empty ranks"
        );
    }
    let empty = PencilArray::<u8, 3, 2>::from_elem(
        Arc::clone(&sparse),
        ExtraShape::new([2, 0, usize::MAX]).unwrap(),
        0,
    )
    .unwrap();
    assert!(empty.is_empty());
    assert_eq!(
        empty.memory_shape(),
        [sparse.local_shape_memory().as_slice(), &[2, 0, usize::MAX]].concat()
    );
}
