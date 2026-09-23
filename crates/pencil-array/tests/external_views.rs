use mpi::traits::*;
use pencil_array::*;
use std::sync::Arc;

#[test]
fn external_views() {
    let universe = mpi::initialize().expect("MPI initialization failed");
    let world = universe.world();
    let size = world.size() as usize;
    let topology = MpiTopology::<1>::new(&world, [size]).unwrap();
    let pencil = Pencil::<2, 1>::new_permuted(
        Arc::clone(&topology),
        [2 * size, 3],
        [0],
        AxisPermutation::new([1, 0]).unwrap(),
    )
    .unwrap();
    let extra = ExtraShape::new([2]).unwrap();
    let mut stack = [0i64; 12];
    let ptr = stack.as_ptr();
    {
        let mut view = PencilArrayViewMut::from_slice_mut(&pencil, &extra, &mut stack).unwrap();
        assert_eq!(view.as_slice().as_ptr(), ptr);
        assert!(std::ptr::eq(view.pencil(), pencil.as_ref()));
        assert!(std::ptr::eq(view.extra_shape(), &extra));
        assert_eq!(view.memory_shape(), [2, 3, 2]);
        for e in 0..2 {
            for y in 0..3 {
                for x in 0..2 {
                    let global_x = pencil.local_ranges()[0].start + x;
                    *view.get_local_mut(&[e], [x, y]).unwrap() =
                        (100 * e + 10 * global_x + y) as i64;
                }
            }
        }
        view.as_mut_slice()[0] += 1;
        view.as_mut_slice()[0] -= 1;
    }
    let view = PencilArrayView::from_slice(&pencil, &extra, &stack).unwrap();
    assert_eq!(view.as_slice().as_ptr(), ptr);
    assert!(std::ptr::eq(view.pencil(), pencil.as_ref()));
    assert!(std::ptr::eq(view.extra_shape(), &extra));
    for e in 0..2 {
        for y in 0..3 {
            for x in 0..2 {
                let gx = pencil.local_ranges()[0].start + x;
                let value = (100 * e + 10 * gx + y) as i64;
                assert_eq!(stack[e * 6 + y * 2 + x], value);
                assert_eq!(view.get_global(&[e], [gx, y]), Some(&value));
            }
        }
    }
    for actual in [11, 13] {
        let mut wrong = vec![0i64; actual];
        let expected = ArrayError::StorageLengthMismatch {
            required: 12,
            actual,
        };
        assert_eq!(
            PencilArrayView::from_slice(&pencil, &extra, &wrong).unwrap_err(),
            expected
        );
        assert_eq!(
            PencilArrayViewMut::from_slice_mut(&pencil, &extra, &mut wrong).unwrap_err(),
            expected
        );
        assert_eq!(wrong, vec![0; actual]);
    }

    struct NonClone(i64);
    let mut values: Vec<_> = (0..12).map(|_| NonClone(1)).collect();
    let values_ptr = values.as_ptr();
    pointwise2_in_place_views(
        PencilArrayViewMut::from_slice_mut(&pencil, &extra, &mut values).unwrap(),
        PencilArrayView::from_slice(&pencil, &extra, &stack).unwrap(),
        |a, b| NonClone(a.0 + b),
    )
    .unwrap();
    let nc = PencilArrayView::from_slice(&pencil, &extra, &values).unwrap();
    assert_eq!(nc.as_slice().as_ptr(), values_ptr);
    assert!(nc.as_slice().iter().zip(stack).all(|(a, b)| a.0 == b + 1));
    let expected_sum: i64 = (0..2)
        .flat_map(|e| {
            (0..2 * size).flat_map(move |x| (0..3).map(move |y| (100 * e + 10 * x + y) as i64))
        })
        .sum();
    assert_eq!(global_sum(&view).unwrap(), expected_sum);
    assert_eq!(
        sum_by(&nc, |v| v.0).unwrap(),
        expected_sum + (12 * size) as i64
    );

    let local = pencil
        .with_permutation(AxisPermutation::identity())
        .unwrap();
    let mut local_storage = vec![0; 12];
    LocalTransposePlan::new(Arc::clone(&pencil), Arc::clone(&local))
        .unwrap()
        .execute_views(
            PencilArrayView::from_slice(&pencil, &extra, &stack).unwrap(),
            PencilArrayViewMut::from_slice_mut(&local, &extra, &mut local_storage).unwrap(),
        )
        .unwrap();
    let destination = Pencil::<2, 1>::new(Arc::clone(&topology), [2 * size, 3], [1]).unwrap();
    let mut output = vec![0; destination.local_len() * 2];
    let plan = AllToAllvTransposePlan::new(Arc::clone(&pencil), Arc::clone(&destination)).unwrap();
    let requirements = plan.workspace_requirements(&extra).unwrap();
    let mut workspace = TransposeWorkspace::from_vecs(
        vec![0; requirements.send_len],
        vec![0; requirements.receive_len],
    );
    plan.execute_views(
        PencilArrayView::from_slice(&pencil, &extra, &stack).unwrap(),
        PencilArrayViewMut::from_slice_mut(&destination, &extra, &mut output).unwrap(),
        &mut workspace,
    )
    .unwrap();
    let mut back = [0; 12];
    let reverse =
        PointToPointTransposePlan::new(Arc::clone(&destination), Arc::clone(&pencil)).unwrap();
    let requirements = reverse.workspace_requirements(&extra).unwrap();
    let mut workspace = TransposeWorkspace::from_vecs(
        vec![0; requirements.send_len],
        vec![0; requirements.receive_len],
    );
    reverse
        .execute_views(
            PencilArrayView::from_slice(&destination, &extra, &output).unwrap(),
            PencilArrayViewMut::from_slice_mut(&pencil, &extra, &mut back).unwrap(),
            &mut workspace,
        )
        .unwrap();
    assert_eq!(back, stack);
    for (p, storage) in [(&local, &local_storage), (&destination, &output)] {
        let v = PencilArrayView::from_slice(p, &extra, storage).unwrap();
        for e in 0..2 {
            for x in p.local_ranges()[0].clone() {
                for y in p.local_ranges()[1].clone() {
                    assert_eq!(
                        v.get_global(&[e], [x, y]),
                        Some(&((100 * e + 10 * x + y) as i64))
                    );
                }
            }
        }
    }

    let zero = ExtraShape::new([0]).unwrap();
    assert!(
        PencilArrayView::from_slice(&pencil, &zero, &[] as &[i64])
            .unwrap()
            .is_empty()
    );
    assert!(
        PencilArrayViewMut::from_slice_mut(&pencil, &zero, &mut [] as &mut [i64])
            .unwrap()
            .is_empty()
    );
    let tiny = Pencil::<2, 1>::new(topology, [1, 1], [0]).unwrap();
    let scalar = ExtraShape::new([]).unwrap();
    let mut storage = vec![1i64; tiny.local_len()];
    assert_eq!(
        PencilArrayViewMut::from_slice_mut(&tiny, &scalar, &mut storage)
            .unwrap()
            .len(),
        tiny.local_len()
    );
    let v = PencilArrayView::from_slice(&tiny, &scalar, &storage).unwrap();
    assert_eq!(v.is_empty(), tiny.local_len() == 0);
    assert_eq!(global_sum(&v).unwrap(), 1);
    let mut empty_ranks = 0;
    world.all_reduce_into(
        &i32::from(v.is_empty()),
        &mut empty_ranks,
        mpi::collective::SystemOperation::sum(),
    );
    assert_eq!(empty_ranks, world.size() - 1);
}
