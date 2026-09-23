use std::sync::Arc;

use mpi::topology::Communicator;
use pencil_array::{CollectiveError, ExtraShape, MpiTopology, Pencil, PencilArray, map_reduce3};

fn sum_two(x: i64, y: i64) -> i64 {
    x + y
}

/// Exercises every map_reduce3 preflight mismatch that can be selected by rank.
/// Every rejected call is followed by a successful call, so this also covers
/// protocol recovery.  The mismatch callbacks must never be entered.
pub fn run_map_reduce3_mismatch_retry_suite<C: Communicator>(
    world: &C,
    topology: &Arc<MpiTopology<1>>,
    pencil: &Arc<Pencil<1, 1>>,
) {
    let size = usize::try_from(world.size()).unwrap();
    if size == 1 {
        let a = PencilArray::from_elem(Arc::clone(pencil), ExtraShape::scalar(), 1_i64).unwrap();
        let result = map_reduce3(
            topology.communicator(),
            &a.view(),
            &a.view(),
            &a.view(),
            0_i64,
            |x, y, z| x + y + z,
            |x, y| x + y,
        )
        .unwrap();
        assert_eq!(result, 3);
        return;
    }

    let rank = usize::try_from(world.rank()).unwrap();
    let scalar =
        |value| PencilArray::from_elem(Arc::clone(pencil), ExtraShape::scalar(), value).unwrap();

    // Descriptor mismatch: one rank uses a different extra shape.
    let mismatched = if rank == 0 {
        PencilArray::from_elem(Arc::clone(pencil), ExtraShape::new([2]).unwrap(), 1_i64).unwrap()
    } else {
        scalar(1_i64)
    };
    let calls = std::cell::Cell::new(0);
    let result = map_reduce3(
        topology.communicator(),
        &mismatched.view(),
        &mismatched.view(),
        &mismatched.view(),
        0_i64,
        |x, y, z| {
            calls.set(calls.get() + 1);
            x + y + z
        },
        |x, y| x + y,
    );
    assert!(matches!(
        result,
        Err(CollectiveError::CollectiveDescriptorMismatch)
    ));
    assert_eq!(calls.get(), 0);
    let good = scalar(1_i64);
    assert_eq!(
        map_reduce3(
            topology.communicator(),
            &good.view(),
            &good.view(),
            &good.view(),
            0_i64,
            |x, y, z| x + y + z,
            |x, y| x + y,
        )
        .unwrap(),
        size as i64 * 3
    );

    // Input type mismatch.
    let i32_input =
        PencilArray::from_elem(Arc::clone(pencil), ExtraShape::scalar(), 1_i32).unwrap();
    let i64_input = scalar(1_i64);
    let calls = std::cell::Cell::new(0);
    let result = if rank == 0 {
        map_reduce3(
            topology.communicator(),
            &i32_input.view(),
            &i32_input.view(),
            &i32_input.view(),
            0_i64,
            |x, y, z| {
                calls.set(calls.get() + 1);
                i64::from(*x + *y + *z)
            },
            |x, y| x + y,
        )
    } else {
        map_reduce3(
            topology.communicator(),
            &i64_input.view(),
            &i64_input.view(),
            &i64_input.view(),
            0_i64,
            |x, y, z| {
                calls.set(calls.get() + 1);
                x + y + z
            },
            |x, y| x + y,
        )
    };
    assert!(matches!(
        result,
        Err(CollectiveError::CollectiveDescriptorMismatch)
    ));
    assert_eq!(calls.get(), 0);
    assert_eq!(
        map_reduce3(
            topology.communicator(),
            &good.view(),
            &good.view(),
            &good.view(),
            0_i64,
            |x, y, z| x + y + z,
            |x, y| x + y,
        )
        .unwrap(),
        size as i64 * 3
    );

    // Neutral mismatch.
    let calls = std::cell::Cell::new(0);
    let result = map_reduce3(
        topology.communicator(),
        &good.view(),
        &good.view(),
        &good.view(),
        if rank == 0 { 0_i64 } else { 1_i64 },
        |x, y, z| {
            calls.set(calls.get() + 1);
            x + y + z
        },
        |x, y| x + y,
    );
    assert!(matches!(
        result,
        Err(CollectiveError::CollectiveDescriptorMismatch)
    ));
    assert_eq!(calls.get(), 0);
    assert_eq!(
        map_reduce3(
            topology.communicator(),
            &good.view(),
            &good.view(),
            &good.view(),
            0_i64,
            |x, y, z| x + y + z,
            |x, y| x + y,
        )
        .unwrap(),
        size as i64 * 3
    );

    // Layout mismatch: descriptors match, but rank zero uses an equivalent
    // layout backed by a distinct topology object.
    let other_topology = MpiTopology::<1>::new(world, [size]).unwrap();
    let other_pencil = Pencil::<1, 1>::new(other_topology, [size], [0]).unwrap();
    let other =
        PencilArray::from_elem(Arc::clone(&other_pencil), ExtraShape::scalar(), 1_i64).unwrap();
    let calls = std::cell::Cell::new(0);
    let last = if rank == 0 { &other } else { &good };
    let result = map_reduce3(
        topology.communicator(),
        &good.view(),
        &good.view(),
        &last.view(),
        0_i64,
        |x, y, z| {
            calls.set(calls.get() + 1);
            x + y + z
        },
        sum_two,
    );
    assert!(matches!(
        result,
        Err(CollectiveError::CollectivePreconditionFailed)
    ));
    assert_eq!(calls.get(), 0);
    assert_eq!(
        map_reduce3(
            topology.communicator(),
            &good.view(),
            &good.view(),
            &good.view(),
            0_i64,
            |x, y, z| x + y + z,
            |x, y| x + y,
        )
        .unwrap(),
        size as i64 * 3
    );

    // Protocol operation mismatch, with both callbacks observed.
    let map_calls = std::cell::Cell::new(0);
    let reduce_calls = std::cell::Cell::new(0);
    let result = if rank == 0 {
        map_reduce3(
            topology.communicator(),
            &good.view(),
            &good.view(),
            &good.view(),
            0_i64,
            |x, y, z| {
                map_calls.set(map_calls.get() + 1);
                x + y + z
            },
            |x, y| {
                reduce_calls.set(reduce_calls.get() + 1);
                x + y
            },
        )
    } else {
        pencil_array::map_reduce_many(
            topology.communicator(),
            &[good.view()],
            0_i64,
            |xs| {
                map_calls.set(map_calls.get() + 1);
                *xs[0]
            },
            |x, y| {
                reduce_calls.set(reduce_calls.get() + 1);
                x + y
            },
        )
    };
    assert!(matches!(
        result,
        Err(CollectiveError::CollectiveDescriptorMismatch)
    ));
    assert_eq!(map_calls.get(), 0);
    assert_eq!(reduce_calls.get(), 0);
    assert_eq!(
        map_reduce3(
            topology.communicator(),
            &good.view(),
            &good.view(),
            &good.view(),
            0_i64,
            |x, y, z| x + y + z,
            |x, y| x + y,
        )
        .unwrap(),
        size as i64 * 3
    );
}
