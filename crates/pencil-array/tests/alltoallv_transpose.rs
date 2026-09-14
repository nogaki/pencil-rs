use std::{fmt::Debug, sync::Arc};

use mpi::traits::*;
use pencil_array::{
    AllToAllvTransposeError, AllToAllvTransposePlan, AllToAllvTransposeWorkspace, AxisPermutation,
    ExtraShape, MpiTopology, Pencil, PencilArray,
};

fn value_2d_u64(extra: [usize; 2], global: [usize; 2]) -> u64 {
    (extra[0] * 1_000_000 + extra[1] * 10_000 + global[0] * 100 + global[1]) as u64
}

fn value_2d_f64(extra: [usize; 2], global: [usize; 2]) -> f64 {
    value_2d_u64(extra, global) as f64
}

fn value_3d(global: [usize; 3]) -> u64 {
    (global[0] * 10_000 + global[1] * 100 + global[2]) as u64
}

fn fill_2d<T: Copy + Clone>(
    pencil: Arc<Pencil<2, 1>>,
    extra_shape: ExtraShape,
    zero: T,
    value: fn([usize; 2], [usize; 2]) -> T,
) -> PencilArray<T, 2, 1> {
    let mut array = PencilArray::from_elem(Arc::clone(&pencil), extra_shape.clone(), zero).unwrap();
    let local_shape = pencil.local_shape_logical();
    let dimensions = extra_shape.dimensions();
    assert_eq!(dimensions.len(), 2);
    let extra = [dimensions[0], dimensions[1]];
    for e0 in 0..extra[0] {
        for e1 in 0..extra[1] {
            for x in 0..local_shape[0] {
                for y in 0..local_shape[1] {
                    let global = [
                        pencil.local_ranges()[0].start + x,
                        pencil.local_ranges()[1].start + y,
                    ];
                    *array.get_local_mut(&[e0, e1], [x, y]).unwrap() = value([e0, e1], global);
                }
            }
        }
    }
    array
}

fn check_2d<T: PartialEq + Debug>(
    array: &PencilArray<T, 2, 1>,
    value: fn([usize; 2], [usize; 2]) -> T,
) {
    let pencil = array.pencil();
    let shape = pencil.local_shape_logical();
    let dimensions = array.extra_shape().dimensions();
    assert_eq!(dimensions.len(), 2);
    let extra = [dimensions[0], dimensions[1]];
    for e0 in 0..extra[0] {
        for e1 in 0..extra[1] {
            for x in 0..shape[0] {
                for y in 0..shape[1] {
                    let global = [
                        pencil.local_ranges()[0].start + x,
                        pencil.local_ranges()[1].start + y,
                    ];
                    assert_eq!(
                        array.get_local(&[e0, e1], [x, y]),
                        Some(&value([e0, e1], global)),
                    );
                }
            }
        }
    }
}

fn run_2d_success<T>(
    topology: &Arc<MpiTopology<1>>,
    value: fn([usize; 2], [usize; 2]) -> T,
    zero: T,
) where
    T: mpi::datatype::Equivalence + Copy + Clone + PartialEq + Debug,
{
    let source_pencil = Pencil::<2, 1>::new_permuted(
        Arc::clone(topology),
        [7, 9],
        [0],
        AxisPermutation::new([0, 1]).unwrap(),
    )
    .unwrap();
    let destination_pencil = Pencil::<2, 1>::new_permuted(
        Arc::clone(topology),
        [7, 9],
        [1],
        AxisPermutation::new([1, 0]).unwrap(),
    )
    .unwrap();
    let extra = ExtraShape::new([2, 3]).unwrap();
    let source = fill_2d(Arc::clone(&source_pencil), extra.clone(), zero, value);
    let source_before = source.as_slice().to_vec();
    let mut destination =
        PencilArray::from_elem(Arc::clone(&destination_pencil), extra.clone(), zero).unwrap();
    let plan =
        AllToAllvTransposePlan::new(Arc::clone(&source_pencil), Arc::clone(&destination_pencil))
            .unwrap();
    let requirements = plan.workspace_requirements(&extra).unwrap();
    let mut workspace = AllToAllvTransposeWorkspace::from_vecs(
        vec![zero; requirements.send_len],
        vec![zero; requirements.receive_len],
    );
    plan.execute_views(source.view(), destination.view_mut(), &mut workspace)
        .unwrap();
    assert_eq!(source.as_slice(), source_before.as_slice());
    check_2d(&destination, value);

    let reverse =
        AllToAllvTransposePlan::new(Arc::clone(&destination_pencil), Arc::clone(&source_pencil))
            .unwrap();
    let reverse_requirements = reverse.workspace_requirements(&extra).unwrap();
    let mut reverse_workspace = AllToAllvTransposeWorkspace::from_vecs(
        vec![zero; reverse_requirements.send_len],
        vec![zero; reverse_requirements.receive_len],
    );
    let mut roundtrip =
        PencilArray::from_elem(Arc::clone(&source_pencil), extra.clone(), zero).unwrap();
    reverse
        .execute_views(
            destination.view(),
            roundtrip.view_mut(),
            &mut reverse_workspace,
        )
        .unwrap();
    assert_eq!(roundtrip.as_slice(), source_before.as_slice());
}

fn fill_3d_physical(pencil: Arc<Pencil<3, 2>>) -> PencilArray<u64, 3, 2> {
    let shape = pencil.local_shape_logical();
    let ranges = pencil.local_ranges();
    let mut physical = Vec::with_capacity(pencil.local_len());
    // The source permutation is [z, x, y], so construct the physical buffer
    // in that order rather than through the library's index helper.
    for z in 0..shape[2] {
        for x in 0..shape[0] {
            for y in 0..shape[1] {
                physical.push(value_3d([
                    ranges[0].start + x,
                    ranges[1].start + y,
                    ranges[2].start + z,
                ]));
            }
        }
    }
    PencilArray::from_vec(pencil, ExtraShape::scalar(), physical).unwrap()
}

fn expected_3d_physical(pencil: &Pencil<3, 2>) -> Vec<u64> {
    let shape = pencil.local_shape_logical();
    let ranges = pencil.local_ranges();
    let mut expected = Vec::with_capacity(pencil.local_len());
    // The destination permutation is [y, z, x], independently spelling out
    // the physical destination order for comparison with as_slice().
    for y in 0..shape[1] {
        for z in 0..shape[2] {
            for x in 0..shape[0] {
                expected.push(value_3d([
                    ranges[0].start + x,
                    ranges[1].start + y,
                    ranges[2].start + z,
                ]));
            }
        }
    }
    expected
}

fn check_3d(array: &PencilArray<u64, 3, 2>) {
    let pencil = array.pencil();
    let shape = pencil.local_shape_logical();
    for x in 0..shape[0] {
        for y in 0..shape[1] {
            for z in 0..shape[2] {
                let global = [
                    pencil.local_ranges()[0].start + x,
                    pencil.local_ranges()[1].start + y,
                    pencil.local_ranges()[2].start + z,
                ];
                assert_eq!(array.get_local(&[], [x, y, z]), Some(&value_3d(global)));
            }
        }
    }
}

fn run_3d_success(
    topology: &Arc<MpiTopology<2>>,
    source_decomposition: [usize; 2],
    destination_decomposition: [usize; 2],
) {
    let source_pencil = Pencil::<3, 2>::new_permuted(
        Arc::clone(topology),
        [5, 2, 7],
        source_decomposition,
        AxisPermutation::new([2, 0, 1]).unwrap(),
    )
    .unwrap();
    let destination_pencil = Pencil::<3, 2>::new_permuted(
        Arc::clone(topology),
        [5, 2, 7],
        destination_decomposition,
        AxisPermutation::new([1, 2, 0]).unwrap(),
    )
    .unwrap();
    let source = fill_3d_physical(Arc::clone(&source_pencil));
    let source_before = source.as_slice().to_vec();
    let expected_destination = expected_3d_physical(&destination_pencil);
    let mut destination = PencilArray::from_elem(
        Arc::clone(&destination_pencil),
        ExtraShape::scalar(),
        u64::MAX,
    )
    .unwrap();
    let plan =
        AllToAllvTransposePlan::new(Arc::clone(&source_pencil), Arc::clone(&destination_pencil))
            .unwrap();
    let requirements = plan.workspace_requirements(&ExtraShape::scalar()).unwrap();
    let mut workspace = AllToAllvTransposeWorkspace::from_vecs(
        vec![0_u64; requirements.send_len],
        vec![0_u64; requirements.receive_len],
    );
    plan.execute_views(source.view(), destination.view_mut(), &mut workspace)
        .unwrap();
    assert_eq!(source.as_slice(), source_before.as_slice());
    assert_eq!(destination.as_slice(), expected_destination.as_slice());
    check_3d(&destination);

    let destination_before = destination.as_slice().to_vec();
    let last_coords: [usize; 2] = std::array::from_fn(|axis| topology.process_grid()[axis] - 1);
    let on_last_coordinate = *topology.local_coords() == last_coords;
    if on_last_coordinate {
        assert!(requirements.send_len > 0);
        assert!(requirements.receive_len > 0);
    }
    for short_send in [true, false] {
        let mut insufficient_workspace = if on_last_coordinate {
            if short_send {
                AllToAllvTransposeWorkspace::from_vecs(
                    Vec::new(),
                    vec![0_u64; requirements.receive_len],
                )
            } else {
                AllToAllvTransposeWorkspace::from_vecs(
                    vec![0_u64; requirements.send_len],
                    Vec::new(),
                )
            }
        } else {
            AllToAllvTransposeWorkspace::from_vecs(
                vec![0_u64; requirements.send_len],
                vec![0_u64; requirements.receive_len],
            )
        };
        let result = plan.execute_views(
            source.view(),
            destination.view_mut(),
            &mut insufficient_workspace,
        );
        assert!(result.is_err());
        if on_last_coordinate {
            assert!(matches!(
                result,
                Err(AllToAllvTransposeError::WorkspaceTooSmall { .. })
            ));
        } else {
            assert_eq!(
                result,
                Err(AllToAllvTransposeError::CollectivePreconditionFailed)
            );
        }
        assert_eq!(source.as_slice(), source_before.as_slice());
        assert_eq!(destination.as_slice(), destination_before.as_slice());
    }

    let reverse =
        AllToAllvTransposePlan::new(Arc::clone(&destination_pencil), Arc::clone(&source_pencil))
            .unwrap();
    let reverse_requirements = reverse
        .workspace_requirements(&ExtraShape::scalar())
        .unwrap();
    let mut reverse_workspace = AllToAllvTransposeWorkspace::from_vecs(
        vec![0_u64; reverse_requirements.send_len],
        vec![0_u64; reverse_requirements.receive_len],
    );
    let mut roundtrip =
        PencilArray::from_elem(Arc::clone(&source_pencil), ExtraShape::scalar(), 0_u64).unwrap();
    reverse
        .execute_views(
            destination.view(),
            roundtrip.view_mut(),
            &mut reverse_workspace,
        )
        .unwrap();
    assert_eq!(roundtrip.as_slice(), source_before.as_slice());
}

fn workspace_for<T: Clone>(
    plan: &AllToAllvTransposePlan<2, 1>,
    extra: &ExtraShape,
    value: T,
) -> AllToAllvTransposeWorkspace<T> {
    let requirements = plan.workspace_requirements(extra).unwrap();
    AllToAllvTransposeWorkspace::from_vecs(
        vec![value.clone(); requirements.send_len],
        vec![value; requirements.receive_len],
    )
}

#[test]
fn alltoallv_transpose_integration() {
    let universe = mpi::initialize().expect("MPI initialization failed");
    let world = universe.world();
    let size = usize::try_from(world.size()).unwrap();
    assert!(matches!(size, 1 | 4 | 6));
    let rank = usize::try_from(world.rank()).unwrap();

    let topology_1d = MpiTopology::<1>::new(&world, [size]).unwrap();
    let grid_2d = if size == 6 {
        [2, 3]
    } else if size == 4 {
        [2, 2]
    } else {
        [1, 1]
    };
    // The non-square case uses a communicator with reversed local rank order.
    // The implementation must use Cartesian rank_to_coordinates, not world
    // ranks; no rank/coordinate inequality is assumed.
    let reversed_world = if size == 6 {
        Some(
            world
                .split_by_color_with_key(
                    mpi::topology::Color::with_value(0),
                    i32::try_from(size - 1 - rank).unwrap(),
                )
                .expect("all ranks join the reversed communicator"),
        )
    } else {
        None
    };
    let topology_2d = match reversed_world.as_ref() {
        Some(comm) => MpiTopology::<2>::new(comm, grid_2d).unwrap(),
        None => MpiTopology::<2>::new(&world, grid_2d).unwrap(),
    };

    run_2d_success(&topology_1d, value_2d_u64, 0_u64);
    run_2d_success(&topology_1d, value_2d_f64, 0.0_f64);
    run_3d_success(&topology_2d, [2, 0], [2, 1]);
    run_3d_success(&topology_2d, [0, 2], [1, 2]);

    let source_pencil = Pencil::<2, 1>::new_permuted(
        Arc::clone(&topology_1d),
        [3, 5],
        [0],
        AxisPermutation::new([0, 1]).unwrap(),
    )
    .unwrap();
    let destination_pencil = Pencil::<2, 1>::new_permuted(
        Arc::clone(&topology_1d),
        [3, 5],
        [1],
        AxisPermutation::new([1, 0]).unwrap(),
    )
    .unwrap();
    let extra = ExtraShape::new([2, 3]).unwrap();
    let source = fill_2d(
        Arc::clone(&source_pencil),
        extra.clone(),
        0_u64,
        value_2d_u64,
    );
    let source_before = source.as_slice().to_vec();
    let plan =
        AllToAllvTransposePlan::new(Arc::clone(&source_pencil), Arc::clone(&destination_pencil))
            .unwrap();
    let requirements = plan.workspace_requirements(&extra).unwrap();

    let mut success_destination =
        PencilArray::from_elem(Arc::clone(&destination_pencil), extra.clone(), 90_u64).unwrap();
    let mut success_workspace = AllToAllvTransposeWorkspace::from_vecs(
        vec![0_u64; requirements.send_len],
        vec![0_u64; requirements.receive_len],
    );
    plan.execute_views(
        source.view(),
        success_destination.view_mut(),
        &mut success_workspace,
    )
    .unwrap();
    assert_eq!(source.as_slice(), source_before.as_slice());
    check_2d(&success_destination, value_2d_u64);

    let reverse =
        AllToAllvTransposePlan::new(Arc::clone(&destination_pencil), Arc::clone(&source_pencil))
            .unwrap();
    let reverse_requirements = reverse.workspace_requirements(&extra).unwrap();
    if size == 4 && rank == 0 {
        assert_eq!(requirements.send_len, 0);
        assert!(requirements.receive_len > 0);
        assert!(reverse_requirements.send_len > 0);
        assert_eq!(reverse_requirements.receive_len, 0);
    }
    if size == 6 {
        match rank {
            0 => {
                assert_eq!(requirements.send_len, 0);
                assert_eq!(requirements.receive_len, 0);
                assert_eq!(reverse_requirements.send_len, 0);
                assert_eq!(reverse_requirements.receive_len, 0);
            }
            2 | 4 => {
                assert_eq!(requirements.send_len, 0);
                assert!(requirements.receive_len > 0);
                assert!(reverse_requirements.send_len > 0);
                assert_eq!(reverse_requirements.receive_len, 0);
            }
            _ => {
                assert!(requirements.send_len > 0);
                assert!(requirements.receive_len > 0);
                assert!(reverse_requirements.send_len > 0);
                assert!(reverse_requirements.receive_len > 0);
            }
        }
    }
    let mut reverse_workspace = AllToAllvTransposeWorkspace::from_vecs(
        vec![0_u64; reverse_requirements.send_len],
        vec![0_u64; reverse_requirements.receive_len],
    );
    let mut roundtrip =
        PencilArray::from_elem(Arc::clone(&source_pencil), extra.clone(), 0_u64).unwrap();
    reverse
        .execute_views(
            success_destination.view(),
            roundtrip.view_mut(),
            &mut reverse_workspace,
        )
        .unwrap();
    assert_eq!(roundtrip.as_slice(), source_before.as_slice());
    world.barrier();

    // An unchanged decomposition is never silently routed to the local API.
    let same_distribution = source_pencil
        .with_permutation(AxisPermutation::new([1, 0]).unwrap())
        .unwrap();
    let same_result = AllToAllvTransposePlan::new(Arc::clone(&source_pencil), same_distribution);
    assert!(matches!(
        same_result,
        Err(AllToAllvTransposeError::UnsupportedDecompositionChange)
    ));
    world.barrier();

    // A rank-local source layout mismatch is agreed before payload traffic.
    let wrong_source =
        PencilArray::from_elem(Arc::clone(&destination_pencil), extra.clone(), 17_u64).unwrap();
    let mut destination =
        PencilArray::from_elem(Arc::clone(&destination_pencil), extra.clone(), 91_u64).unwrap();
    let destination_before = destination.as_slice().to_vec();
    let mut bad_source_workspace = workspace_for(&plan, &extra, 0_u64);
    let result = if rank == 0 {
        plan.execute_views(
            wrong_source.view(),
            destination.view_mut(),
            &mut bad_source_workspace,
        )
    } else {
        plan.execute_views(
            source.view(),
            destination.view_mut(),
            &mut bad_source_workspace,
        )
    };
    assert!(result.is_err());
    assert_eq!(destination.as_slice(), destination_before.as_slice());
    assert_eq!(source.as_slice(), source_before.as_slice());
    world.barrier();

    // A rank-local destination layout mismatch has the same guarantee.
    let mut wrong_destination =
        PencilArray::from_elem(Arc::clone(&source_pencil), extra.clone(), 73_u64).unwrap();
    let mut destination =
        PencilArray::from_elem(Arc::clone(&destination_pencil), extra.clone(), 92_u64).unwrap();
    let destination_before = destination.as_slice().to_vec();
    let mut bad_destination_workspace = workspace_for(&plan, &extra, 0_u64);
    let result = if rank == 0 {
        plan.execute_views(
            source.view(),
            wrong_destination.view_mut(),
            &mut bad_destination_workspace,
        )
    } else {
        plan.execute_views(
            source.view(),
            destination.view_mut(),
            &mut bad_destination_workspace,
        )
    };
    assert!(result.is_err());
    if rank == 0 {
        assert!(
            wrong_destination
                .as_slice()
                .iter()
                .all(|&value| value == 73)
        );
    } else {
        assert_eq!(destination.as_slice(), destination_before.as_slice());
    }
    world.barrier();

    // Same element count but a different extra shape is caught by the exact descriptor.
    let alternate_extra = ExtraShape::new([3, 2]).unwrap();
    let alternate_source =
        PencilArray::from_elem(Arc::clone(&source_pencil), alternate_extra.clone(), 18_u64)
            .unwrap();
    let mut normal_destination =
        PencilArray::from_elem(Arc::clone(&destination_pencil), extra.clone(), 94_u64).unwrap();
    let normal_destination_before = normal_destination.as_slice().to_vec();
    let mut alternate_workspace = AllToAllvTransposeWorkspace::from_vecs(
        vec![0_u64; requirements.send_len],
        vec![0_u64; requirements.receive_len],
    );
    let result = if rank == 0 {
        plan.execute_views(
            alternate_source.view(),
            normal_destination.view_mut(),
            &mut alternate_workspace,
        )
    } else {
        plan.execute_views(
            source.view(),
            normal_destination.view_mut(),
            &mut alternate_workspace,
        )
    };
    assert!(result.is_err());
    assert_eq!(
        normal_destination.as_slice(),
        normal_destination_before.as_slice()
    );
    world.barrier();

    // Each rank's extra shape is locally consistent, but rank 0 disagrees.
    let rank_local_extra = if rank == 0 {
        alternate_extra.clone()
    } else {
        extra.clone()
    };
    let rank_local_source = fill_2d(
        Arc::clone(&source_pencil),
        rank_local_extra.clone(),
        0_u64,
        value_2d_u64,
    );
    let rank_local_source_before = rank_local_source.as_slice().to_vec();
    let mut rank_local_destination = PencilArray::from_elem(
        Arc::clone(&destination_pencil),
        rank_local_extra.clone(),
        94_u64,
    )
    .unwrap();
    let rank_local_destination_before = rank_local_destination.as_slice().to_vec();
    let rank_local_requirements = plan.workspace_requirements(&rank_local_extra).unwrap();
    let mut rank_local_workspace = AllToAllvTransposeWorkspace::from_vecs(
        vec![0_u64; rank_local_requirements.send_len],
        vec![0_u64; rank_local_requirements.receive_len],
    );
    let result = plan.execute_views(
        rank_local_source.view(),
        rank_local_destination.view_mut(),
        &mut rank_local_workspace,
    );
    if size == 1 {
        assert!(result.is_ok());
        check_2d(&rank_local_destination, value_2d_u64);
    } else {
        assert!(result.is_err());
        assert_eq!(
            rank_local_destination.as_slice(),
            rank_local_destination_before.as_slice()
        );
    }
    assert_eq!(
        rank_local_source.as_slice(),
        rank_local_source_before.as_slice()
    );
    world.barrier();

    // A rank-local extra-rank difference is caught by the fixed descriptor length.
    let rank_extra = if rank == 0 {
        ExtraShape::new([6]).unwrap()
    } else {
        extra.clone()
    };
    let rank_extra_source =
        PencilArray::from_elem(Arc::clone(&source_pencil), rank_extra.clone(), 19_u64).unwrap();
    let rank_extra_source_before = rank_extra_source.as_slice().to_vec();
    let mut rank_extra_destination =
        PencilArray::from_elem(Arc::clone(&destination_pencil), rank_extra.clone(), 95_u64)
            .unwrap();
    let rank_extra_destination_before = rank_extra_destination.as_slice().to_vec();
    let rank_extra_requirements = plan.workspace_requirements(&rank_extra).unwrap();
    let mut rank_extra_workspace = AllToAllvTransposeWorkspace::from_vecs(
        vec![0_u64; rank_extra_requirements.send_len],
        vec![0_u64; rank_extra_requirements.receive_len],
    );
    let result = plan.execute_views(
        rank_extra_source.view(),
        rank_extra_destination.view_mut(),
        &mut rank_extra_workspace,
    );
    if size == 1 {
        assert!(result.is_ok());
        assert!(
            rank_extra_destination
                .as_slice()
                .iter()
                .all(|&value| value == 19)
        );
    } else {
        assert!(result.is_err());
        assert_eq!(
            rank_extra_destination.as_slice(),
            rank_extra_destination_before.as_slice()
        );
    }
    assert_eq!(
        rank_extra_source.as_slice(),
        rank_extra_source_before.as_slice()
    );
    world.barrier();

    // Capacity without initialized length is insufficient.
    let mut capacity_only = AllToAllvTransposeWorkspace::from_vecs(
        Vec::with_capacity(requirements.send_len),
        Vec::with_capacity(requirements.receive_len),
    );
    let mut capacity_destination =
        PencilArray::from_elem(Arc::clone(&destination_pencil), extra.clone(), 95_u64).unwrap();
    let capacity_before = capacity_destination.as_slice().to_vec();
    let result = plan.execute_views(
        source.view(),
        capacity_destination.view_mut(),
        &mut capacity_only,
    );
    assert!(matches!(
        result,
        Err(AllToAllvTransposeError::WorkspaceTooSmall { .. })
            | Err(AllToAllvTransposeError::CollectivePreconditionFailed)
    ));
    assert_eq!(capacity_destination.as_slice(), capacity_before.as_slice());
    world.barrier();

    // Only one rank receives a different valid plan at execute time.
    let reverse_plan =
        AllToAllvTransposePlan::new(Arc::clone(&destination_pencil), Arc::clone(&source_pencil))
            .unwrap();
    let reverse_requirements = reverse_plan.workspace_requirements(&extra).unwrap();
    let max_send = requirements.send_len.max(reverse_requirements.send_len);
    let max_receive = requirements
        .receive_len
        .max(reverse_requirements.receive_len);
    let mut mismatched_plan_workspace =
        AllToAllvTransposeWorkspace::from_vecs(vec![0_u64; max_send], vec![0_u64; max_receive]);
    let mut plan_destination =
        PencilArray::from_elem(Arc::clone(&destination_pencil), extra.clone(), 96_u64).unwrap();
    let plan_destination_before = plan_destination.as_slice().to_vec();
    let result = if rank == 0 {
        reverse_plan.execute_views(
            source.view(),
            plan_destination.view_mut(),
            &mut mismatched_plan_workspace,
        )
    } else {
        plan.execute_views(
            source.view(),
            plan_destination.view_mut(),
            &mut mismatched_plan_workspace,
        )
    };
    assert!(result.is_err());
    assert_eq!(
        plan_destination.as_slice(),
        plan_destination_before.as_slice()
    );
    world.barrier();

    // Only one rank uses a different MPI representation type.
    let source_u32 =
        PencilArray::from_elem(Arc::clone(&source_pencil), extra.clone(), 1_u32).unwrap();
    let source_u64 =
        PencilArray::from_elem(Arc::clone(&source_pencil), extra.clone(), 1_u64).unwrap();
    let mut destination_u32 =
        PencilArray::from_elem(Arc::clone(&destination_pencil), extra.clone(), 97_u32).unwrap();
    let mut destination_u64 =
        PencilArray::from_elem(Arc::clone(&destination_pencil), extra.clone(), 98_u64).unwrap();
    let destination_u32_before = destination_u32.as_slice().to_vec();
    let destination_u64_before = destination_u64.as_slice().to_vec();
    let mut type_workspace = AllToAllvTransposeWorkspace::from_vecs(
        vec![0_u64; requirements.send_len],
        vec![0_u64; requirements.receive_len],
    );
    let result = if rank == 0 {
        let req = plan.workspace_requirements(&extra).unwrap();
        let mut workspace = AllToAllvTransposeWorkspace::from_vecs(
            vec![0_u32; req.send_len],
            vec![0_u32; req.receive_len],
        );
        plan.execute_views(
            source_u32.view(),
            destination_u32.view_mut(),
            &mut workspace,
        )
    } else {
        let _ = &mut type_workspace;
        plan.execute_views(
            source_u64.view(),
            destination_u64.view_mut(),
            &mut type_workspace,
        )
    };
    if size == 1 {
        assert!(result.is_ok());
    } else {
        assert!(result.is_err());
        if rank == 0 {
            assert_eq!(
                destination_u32.as_slice(),
                destination_u32_before.as_slice()
            );
        } else {
            assert_eq!(
                destination_u64.as_slice(),
                destination_u64_before.as_slice()
            );
        }
    }
    world.barrier();

    // A construction-time change on one rank is rejected before pattern use.
    let one_rank_same = source_pencil
        .with_permutation(AxisPermutation::new([1, 0]).unwrap())
        .unwrap();
    let result = if rank == 0 {
        AllToAllvTransposePlan::new(Arc::clone(&source_pencil), one_rank_same)
    } else {
        AllToAllvTransposePlan::new(Arc::clone(&source_pencil), Arc::clone(&destination_pencil))
    };
    assert!(result.is_err());
    world.barrier();

    let source_3d = Pencil::<3, 2>::new(Arc::clone(&topology_2d), [5, 2, 7], [2, 0]).unwrap();
    let normal_3d_destination = source_3d.with_decomposition([2, 1]).unwrap();
    let alternate_3d_destination = source_3d.with_decomposition([1, 0]).unwrap();
    let result = if rank == 0 {
        AllToAllvTransposePlan::new(Arc::clone(&source_3d), alternate_3d_destination)
    } else {
        AllToAllvTransposePlan::new(Arc::clone(&source_3d), normal_3d_destination)
    };
    if size == 1 {
        assert!(result.is_ok());
    } else {
        assert!(result.is_err());
    }
    world.barrier();

    // A different const N still completes the common fixed-header reductions.
    if rank == 0 {
        let source_3d = Pencil::<3, 1>::new(Arc::clone(&topology_1d), [7, 9, 5], [0]).unwrap();
        let destination_3d = source_3d.with_decomposition([1]).unwrap();
        let result = AllToAllvTransposePlan::new(source_3d, destination_3d);
        if size == 1 {
            assert!(result.is_ok());
        } else {
            assert!(result.is_err());
        }
    } else {
        assert!(
            AllToAllvTransposePlan::new(
                Arc::clone(&source_pencil),
                Arc::clone(&destination_pencil),
            )
            .is_err()
        );
    }
    world.barrier();

    // A constructor and execute call cannot be mixed on one communicator.
    let mut mixed_destination =
        PencilArray::from_elem(Arc::clone(&destination_pencil), extra.clone(), 99_u64).unwrap();
    let mixed_before = mixed_destination.as_slice().to_vec();
    if size == 1 {
        assert!(
            AllToAllvTransposePlan::new(
                Arc::clone(&source_pencil),
                Arc::clone(&destination_pencil),
            )
            .is_ok()
        );
    } else if rank == 0 {
        assert!(
            AllToAllvTransposePlan::new(
                Arc::clone(&source_pencil),
                Arc::clone(&destination_pencil),
            )
            .is_err()
        );
    } else {
        let mut workspace = workspace_for(&plan, &extra, 0_u64);
        assert!(
            plan.execute_views(source.view(), mixed_destination.view_mut(), &mut workspace)
                .is_err()
        );
    }
    assert_eq!(mixed_destination.as_slice(), mixed_before.as_slice());
    world.barrier();

    // Metadata-only construction reaches CountOverflow before any workspace allocation.
    let giant_x = (i32::MAX as usize + 1).checked_mul(size).unwrap();
    let giant_source = Pencil::<2, 1>::new(Arc::clone(&topology_1d), [giant_x, size], [0]).unwrap();
    let giant_destination = giant_source.with_decomposition([1]).unwrap();
    let giant_plan =
        AllToAllvTransposePlan::new(Arc::clone(&giant_source), Arc::clone(&giant_destination))
            .unwrap();
    assert!(matches!(
        giant_plan.workspace_requirements(&ExtraShape::scalar()),
        Err(AllToAllvTransposeError::CountOverflow)
    ));
    let zero_extra = ExtraShape::new([2, 0]).unwrap();
    let giant_source_array = PencilArray::<u8, 2, 1>::from_vec(
        Arc::clone(&giant_source),
        zero_extra.clone(),
        Vec::new(),
    )
    .unwrap();
    let mut giant_destination_array = PencilArray::<u8, 2, 1>::from_vec(
        Arc::clone(&giant_destination),
        zero_extra.clone(),
        Vec::new(),
    )
    .unwrap();
    let zero_requirements = giant_plan.workspace_requirements(&zero_extra).unwrap();
    assert_eq!(zero_requirements.send_len, 0);
    assert_eq!(zero_requirements.receive_len, 0);
    let mut zero_workspace = AllToAllvTransposeWorkspace::from_vecs(Vec::new(), Vec::new());
    giant_plan
        .execute_views(
            giant_source_array.view(),
            giant_destination_array.view_mut(),
            &mut zero_workspace,
        )
        .unwrap();
    world.barrier();

    assert!(Pencil::<2, 1>::new(Arc::clone(&topology_1d), [usize::MAX, 2], [0],).is_err());
    assert!(ExtraShape::new([usize::MAX, 2]).is_err());
    world.barrier();
}
