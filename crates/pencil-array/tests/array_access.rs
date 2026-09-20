use std::sync::Arc;

use mpi::traits::*;
use pencil_array::{AxisPermutation, ExtraShape, LocalGrid, MpiTopology, Pencil, PencilArray};

#[derive(Debug, PartialEq, Eq)]
struct Coordinate(usize);

fn coordinates<const N: usize>(shape: [usize; N]) -> [Vec<Coordinate>; N] {
    std::array::from_fn(|axis| (0..shape[axis]).map(Coordinate).collect())
}

fn expected_memory_points<const N: usize, const M: usize>(
    pencil: &Pencil<N, M>,
) -> Vec<[usize; N]> {
    let memory_shape = pencil.local_shape_memory();
    let axes_in_memory_order = pencil.permutation().axes().map(|axis| axis.index());
    let mut points = Vec::with_capacity(pencil.local_len());

    for linear in 0..pencil.local_len() {
        let mut remainder = linear;
        let mut memory_indices = [0; N];
        for position in (0..N).rev() {
            memory_indices[position] = remainder % memory_shape[position];
            remainder /= memory_shape[position];
        }
        let mut logical_indices = [0; N];
        for (position, &axis) in axes_in_memory_order.iter().enumerate() {
            logical_indices[axis] = pencil.local_ranges()[axis].start + memory_indices[position];
        }
        points.push(logical_indices);
    }
    points
}

fn assert_grid<const N: usize, const M: usize>(
    pencil: &Pencil<N, M>,
    grid: &LocalGrid<'_, Coordinate, N>,
) {
    assert_eq!(
        grid.iter()
            .map(|point| point.map(|coordinate| coordinate.0))
            .collect::<Vec<_>>(),
        expected_memory_points(pencil),
    );
    assert_eq!(grid.iter().len(), pencil.local_len());
    assert!(grid.axis(N).is_none());
    assert!(grid.axis(usize::MAX).is_none());

    if pencil.local_len() == 0 {
        assert!(grid.get_local([0; N]).is_none());
    } else {
        let point = grid.get_local([0; N]).unwrap();
        for (axis, coordinate) in point.into_iter().enumerate() {
            assert_eq!(coordinate.0, pencil.local_ranges()[axis].start);
        }
    }
    assert!(grid.get_local(pencil.local_shape_logical()).is_none());
}

fn physical_offset<const N: usize>(
    memory_shape: [usize; N],
    axes_in_memory_order: [usize; N],
    logical_indices: [usize; N],
) -> usize {
    let mut offset = 0;
    for position in 0..N {
        offset = offset * memory_shape[position] + logical_indices[axes_in_memory_order[position]];
    }
    offset
}

#[test]
fn global_access_and_local_grid_are_local_and_follow_memory_order() {
    let universe = mpi::initialize().expect("MPI initialization failed");
    let world = universe.world();
    let world_size = usize::try_from(world.size()).unwrap();
    let grid2 = match world_size {
        1 => [1, 1],
        4 => [2, 2],
        6 => [2, 3],
        other => panic!("run this test with mpiexec -n 1, -n 4, or -n 6, got {other}"),
    };

    let topology1 = MpiTopology::<1>::new(&world, [world_size]).unwrap();
    let pencil2 = Pencil::<2, 1>::new_permuted(
        Arc::clone(&topology1),
        [world_size * 2 + 1, 3],
        [0],
        AxisPermutation::new([1, 0]).unwrap(),
    )
    .unwrap();
    let axes2 = coordinates(*pencil2.global_shape());
    let axes2_refs = std::array::from_fn(|axis| axes2[axis].as_slice());
    let grid = pencil2.local_grid(axes2_refs).unwrap();
    assert_grid(pencil2.as_ref(), &grid);
    assert_eq!(grid.axis(0).unwrap().len(), pencil2.local_ranges()[0].len());
    assert_eq!(grid.axis(1).unwrap().len(), pencil2.local_ranges()[1].len());
    let mixed = grid.get_local([1, 2]).unwrap();
    assert_eq!(
        mixed.map(|coordinate| coordinate.0),
        [
            pencil2.local_ranges()[0].start + 1,
            pencil2.local_ranges()[1].start + 2,
        ],
    );

    let extra_shape = ExtraShape::new([2]).unwrap();
    let local_len = pencil2.local_len();
    let mut array = PencilArray::from_vec(
        Arc::clone(&pencil2),
        extra_shape,
        (0..local_len * 2).collect(),
    )
    .unwrap();
    let memory_shape = pencil2.local_shape_memory();
    let axes_in_memory_order = pencil2.permutation().axes().map(|axis| axis.index());
    let mut batches = array.as_slice().chunks_exact(local_len);
    for (extra, batch) in batches.by_ref().enumerate() {
        for (value, point) in batch.iter().zip(grid.iter()) {
            let local =
                std::array::from_fn(|axis| point[axis].0 - pencil2.local_ranges()[axis].start);
            let expected =
                extra * local_len + physical_offset(memory_shape, axes_in_memory_order, local);
            assert_eq!(*value, expected);
        }
    }
    assert!(batches.remainder().is_empty());

    for extra in 0..2 {
        for logical in expected_memory_points(pencil2.as_ref()) {
            let offset = physical_offset(
                memory_shape,
                axes_in_memory_order,
                [
                    logical[0] - pencil2.local_ranges()[0].start,
                    logical[1] - pencil2.local_ranges()[1].start,
                ],
            );
            let expected = extra * local_len + offset;
            assert_eq!(array.get_global(&[extra], logical), Some(&expected));
        }
    }

    let ranges = pencil2.local_ranges();
    let owned_global = [ranges[0].start, ranges[1].start];
    assert!(array.get_global(&[], owned_global).is_none());
    assert!(array.get_global(&[2], owned_global).is_none());
    if ranges[0].start > 0 {
        for extra in 0..2 {
            assert!(
                array
                    .get_global(&[extra], [ranges[0].start - 1, ranges[1].start])
                    .is_none()
            );
        }
    }
    if ranges[0].end < pencil2.global_shape()[0] {
        for extra in 0..2 {
            assert!(
                array
                    .get_global(&[extra], [ranges[0].end, ranges[1].start])
                    .is_none()
            );
        }
    }
    assert!(
        array
            .get_global(&[0], [pencil2.global_shape()[0], 0])
            .is_none()
    );
    if let Some(global) = expected_memory_points(pencil2.as_ref()).first().copied() {
        *array.get_global_mut(&[1], global).unwrap() = 999;
        assert_eq!(array.get_global(&[1], global), Some(&999));
        {
            let view = array.view();
            assert_eq!(view.get_global(&[1], global), Some(&999));
            let view_grid = view.local_grid(axes2_refs).unwrap();
            assert_grid(pencil2.as_ref(), &view_grid);
        }
        {
            let mut view = array.view_mut();
            *view.get_global_mut(&[1], global).unwrap() = 1000;
            assert_eq!(view.get_global(&[1], global), Some(&1000));
            let view_grid = view.local_grid(axes2_refs).unwrap();
            assert_grid(pencil2.as_ref(), &view_grid);
        }
    }
    let array_grid = array.local_grid(axes2_refs).unwrap();
    assert_grid(pencil2.as_ref(), &array_grid);

    let sparse = Pencil::<2, 1>::new(Arc::clone(&topology1), [1, 2], [0]).unwrap();
    let sparse_axes = coordinates(*sparse.global_shape());
    let sparse_refs = std::array::from_fn(|axis| sparse_axes[axis].as_slice());
    let sparse_grid = sparse.local_grid(sparse_refs).unwrap();
    if sparse.local_len() == 0 {
        assert_eq!(sparse_grid.iter().count(), 0);
        let sparse_array = PencilArray::<(), 2, 1>::from_vec(
            Arc::clone(&sparse),
            ExtraShape::scalar(),
            Vec::new(),
        )
        .unwrap();
        assert!(sparse_array.get_global(&[], [0, 0]).is_none());
    }

    let pencil3 = Pencil::<3, 2>::new_permuted(
        Arc::clone(&MpiTopology::<2>::new(&world, grid2).unwrap()),
        [5, 3 * grid2[1], 4 * grid2[0]],
        [2, 0],
        AxisPermutation::new([2, 0, 1]).unwrap(),
    )
    .unwrap();
    let axes3 = coordinates(*pencil3.global_shape());
    let axes3_refs = std::array::from_fn(|axis| axes3[axis].as_slice());
    let grid3 = pencil3.local_grid(axes3_refs).unwrap();
    assert_grid(pencil3.as_ref(), &grid3);
    let mut bad_axes3 = axes3_refs;
    bad_axes3[1] = &axes3[1][..axes3[1].len() - 1];
    assert!(pencil3.local_grid(bad_axes3).is_err());

    let pencil4 = Pencil::<4, 2>::new_permuted(
        Arc::clone(&MpiTopology::<2>::new(&world, grid2).unwrap()),
        [3, 4 * grid2[0], 5, 6 * grid2[1]],
        [1, 3],
        AxisPermutation::new([3, 1, 0, 2]).unwrap(),
    )
    .unwrap();
    let axes4 = coordinates(*pencil4.global_shape());
    let axes4_refs = std::array::from_fn(|axis| axes4[axis].as_slice());
    let grid4 = pencil4.local_grid(axes4_refs).unwrap();
    assert_grid(pencil4.as_ref(), &grid4);

    let maximal = Pencil::<1, 1>::new(topology1, [usize::MAX], [0]).unwrap();
    let maximal_array = PencilArray::<(), 1, 1>::from_vec(
        Arc::clone(&maximal),
        ExtraShape::scalar(),
        vec![(); maximal.local_len()],
    )
    .unwrap();
    assert!(maximal_array.get_global(&[], [usize::MAX]).is_none());
    let local_end = maximal.local_ranges()[0].end;
    assert!(maximal_array.get_global(&[], [local_end - 1]).is_some());
}
