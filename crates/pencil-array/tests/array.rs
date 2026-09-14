use std::{cell::Cell, sync::Arc};

use mpi::traits::*;
use pencil_array::{ArrayError, AxisPermutation, ExtraShape, MpiTopology, Pencil, PencilArray};

#[test]
fn owning_array_contracts() {
    let universe = mpi::initialize().expect("MPI initialization failed");
    let world = universe.world();
    let world_size = usize::try_from(world.size()).unwrap();
    let process_grid = match world_size {
        1 => [1, 1],
        4 => [2, 2],
        other => panic!("run this test with mpiexec -n 1 or -n 4, got {other}"),
    };
    let topology = MpiTopology::<2>::new(&world, process_grid).unwrap();
    let pencil = Pencil::<3, 2>::new_permuted(
        Arc::clone(&topology),
        [2 * process_grid[0], 3 * process_grid[1], 4],
        [0, 1],
        AxisPermutation::new([0, 2, 1]).unwrap(),
    )
    .unwrap();
    let extra_shape = ExtraShape::new([2]).unwrap();

    assert_eq!(pencil.local_shape_logical(), [2, 3, 4]);
    let storage: Vec<_> = (0..48).collect();
    let array =
        PencilArray::from_vec(Arc::clone(&pencil), extra_shape.clone(), storage.clone()).unwrap();
    assert_eq!(array.as_slice(), storage);

    assert_eq!(
        PencilArray::from_vec(Arc::clone(&pencil), extra_shape.clone(), vec![0; 47]).unwrap_err(),
        ArrayError::StorageLengthMismatch {
            required: 48,
            actual: 47,
        }
    );
    assert_eq!(
        PencilArray::from_vec(Arc::clone(&pencil), extra_shape.clone(), vec![0; 49]).unwrap_err(),
        ArrayError::StorageLengthMismatch {
            required: 48,
            actual: 49,
        }
    );

    let filled = PencilArray::from_elem(Arc::clone(&pencil), extra_shape.clone(), 7).unwrap();
    assert_eq!(filled.as_slice(), &[7; 48]);

    let calls = Cell::new(0);
    let generated = PencilArray::from_fn(Arc::clone(&pencil), extra_shape.clone(), || {
        let value = calls.get();
        calls.set(value + 1);
        value
    })
    .unwrap();
    assert_eq!(calls.get(), 48);
    assert_eq!(generated.as_slice(), (0..48).collect::<Vec<_>>());

    let mut indexed =
        PencilArray::from_vec(Arc::clone(&pencil), extra_shape.clone(), (0..48).collect()).unwrap();
    assert!(Arc::ptr_eq(indexed.pencil(), &pencil));
    assert_eq!(indexed.extra_shape(), &extra_shape);
    assert_eq!(indexed.local_spatial_shape(), [2, 3, 4]);
    assert_eq!(indexed.local_spatial_memory_shape(), [2, 4, 3]);
    assert_eq!(indexed.logical_shape(), [2, 2, 3, 4]);
    assert_eq!(indexed.memory_shape(), [2, 2, 4, 3]);
    assert_eq!(indexed.len(), 48);
    assert!(!indexed.is_empty());

    for extra in 0..2 {
        for x in 0..2 {
            for y in 0..3 {
                for z in 0..4 {
                    let expected = extra * 24 + x * 12 + z * 3 + y;
                    assert_eq!(indexed.get_local(&[extra], [x, y, z]), Some(&expected));
                }
            }
        }
    }
    assert_eq!(indexed.get_local(&[], [0, 0, 0]), None);
    assert_eq!(indexed.get_local(&[0, 0], [0, 0, 0]), None);
    assert_eq!(indexed.get_local(&[2], [0, 0, 0]), None);
    assert_eq!(indexed.get_local(&[0], [2, 0, 0]), None);
    assert_eq!(indexed.get_local(&[0], [0, 3, 0]), None);
    assert_eq!(indexed.get_local(&[0], [0, 0, 4]), None);

    {
        let view = indexed.view();
        assert_eq!(view.memory_shape(), [2, 2, 4, 3]);
        assert_eq!(view.get_local(&[1], [0, 2, 1]), Some(&29));
    }
    {
        let mut view = indexed.view_mut();
        *view.get_local_mut(&[0], [1, 1, 2]).unwrap() = 99;
        view.as_mut_slice()[0] = 77;
    }
    assert_eq!(indexed.get_local(&[0], [1, 1, 2]), Some(&99));
    assert_eq!(indexed.as_slice()[0], 77);
    *indexed.get_local_mut(&[1], [0, 2, 1]).unwrap() = 88;
    assert_eq!(indexed.as_slice()[29], 88);
    indexed.as_mut_slice()[1] = 66;
    assert_eq!(indexed.as_slice()[1], 66);

    let empty_extra = ExtraShape::new([2, 0, usize::MAX]).unwrap();
    let empty_calls = Cell::new(0);
    let empty = PencilArray::<u8, 3, 2>::from_fn(Arc::clone(&pencil), empty_extra.clone(), || {
        empty_calls.set(empty_calls.get() + 1);
        1
    })
    .unwrap();
    assert_eq!(empty_calls.get(), 0);
    assert!(empty.is_empty());
    assert_eq!(empty.logical_shape(), [2, 0, usize::MAX, 2, 3, 4]);
    assert_eq!(empty.memory_shape(), [2, 0, usize::MAX, 2, 4, 3]);
    assert!(
        PencilArray::from_elem(pencil, empty_extra, 5_u8)
            .unwrap()
            .is_empty()
    );

    let sparse = Pencil::<3, 2>::new_permuted(
        Arc::clone(&topology),
        [1, 3 * process_grid[1], 4],
        [0, 1],
        AxisPermutation::new([0, 2, 1]).unwrap(),
    )
    .unwrap();
    let sparse_calls = Cell::new(0);
    let sparse_array = PencilArray::<u8, 3, 2>::from_fn(
        Arc::clone(&sparse),
        ExtraShape::new([2]).unwrap(),
        || {
            sparse_calls.set(sparse_calls.get() + 1);
            3
        },
    )
    .unwrap();
    assert_eq!(sparse_calls.get(), sparse.local_len() * 2);
    assert_eq!(sparse_array.len(), sparse.local_len() * 2);
    if world_size == 4 && topology.local_coords()[0] == 0 {
        assert_eq!(sparse.local_len(), 0);
        assert_eq!(sparse_calls.get(), 0);
    }

    let linear_topology = MpiTopology::<1>::new(&world, [world_size]).unwrap();
    let huge_pencil = Pencil::<1, 1>::new(linear_topology, [usize::MAX], [0]).unwrap();
    let overflow_calls = Cell::new(0);
    assert_eq!(
        PencilArray::<u8, 1, 1>::from_fn(
            Arc::clone(&huge_pencil),
            ExtraShape::new([usize::MAX]).unwrap(),
            || {
                overflow_calls.set(overflow_calls.get() + 1);
                0
            },
        )
        .unwrap_err(),
        ArrayError::Geometry(pencil_array::GeometryError::SizeOverflow)
    );
    assert_eq!(overflow_calls.get(), 0);

    let allocation_calls = Cell::new(0);
    assert_eq!(
        PencilArray::<u64, 1, 1>::from_fn(huge_pencil.clone(), ExtraShape::scalar(), || {
            allocation_calls.set(allocation_calls.get() + 1);
            0_u64
        })
        .unwrap_err(),
        ArrayError::AllocationFailed {
            required: huge_pencil.local_len(),
        }
    );
    assert_eq!(allocation_calls.get(), 0);
    assert_eq!(
        PencilArray::from_elem(huge_pencil.clone(), ExtraShape::scalar(), 0_u64).unwrap_err(),
        ArrayError::AllocationFailed {
            required: huge_pencil.local_len(),
        }
    );
}
