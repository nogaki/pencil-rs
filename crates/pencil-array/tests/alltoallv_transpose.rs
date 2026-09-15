use std::{fmt::Debug, sync::Arc};

use mpi::traits::*;
use pencil_array::{
    AllToAllvTransposeError, AllToAllvTransposePlan, AllToAllvTransposeWorkspace, ArrayError,
    AxisPermutation, ExtraShape, ManyPencilArray, MpiTopology, Pencil, PencilArray,
    PointToPointTransposePlan,
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

fn run_2d_point_to_point_success<T>(
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

    let all_plan =
        AllToAllvTransposePlan::new(Arc::clone(&source_pencil), Arc::clone(&destination_pencil))
            .unwrap();
    let p2p_plan =
        PointToPointTransposePlan::new(Arc::clone(&source_pencil), Arc::clone(&destination_pencil))
            .unwrap();
    let all_requirements = all_plan.workspace_requirements(&extra).unwrap();
    let p2p_requirements = p2p_plan.workspace_requirements(&extra).unwrap();
    assert_eq!(all_requirements, p2p_requirements);

    let mut all_destination =
        PencilArray::from_elem(Arc::clone(&destination_pencil), extra.clone(), zero).unwrap();
    let mut p2p_destination =
        PencilArray::from_elem(Arc::clone(&destination_pencil), extra.clone(), zero).unwrap();
    let mut all_workspace = AllToAllvTransposeWorkspace::from_vecs(
        vec![zero; all_requirements.send_len],
        vec![zero; all_requirements.receive_len],
    );
    let mut p2p_workspace = AllToAllvTransposeWorkspace::from_vecs(
        vec![zero; p2p_requirements.send_len],
        vec![zero; p2p_requirements.receive_len],
    );
    all_plan
        .execute_views(
            source.view(),
            all_destination.view_mut(),
            &mut all_workspace,
        )
        .unwrap();
    p2p_plan
        .execute_views(
            source.view(),
            p2p_destination.view_mut(),
            &mut p2p_workspace,
        )
        .unwrap();
    assert_eq!(source.as_slice(), source_before.as_slice());
    assert_eq!(p2p_destination.as_slice(), all_destination.as_slice());
    check_2d(&p2p_destination, value);

    let reverse =
        PointToPointTransposePlan::new(Arc::clone(&destination_pencil), Arc::clone(&source_pencil))
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
            p2p_destination.view(),
            roundtrip.view_mut(),
            &mut reverse_workspace,
        )
        .unwrap();
    assert_eq!(roundtrip.as_slice(), source_before.as_slice());
}

fn run_2d_point_to_point_empty_success(topology: &Arc<MpiTopology<1>>) {
    let source_pencil = Pencil::<2, 1>::new(Arc::clone(topology), [3, 5], [0]).unwrap();
    let destination_pencil = source_pencil.with_decomposition([1]).unwrap();
    let extra = ExtraShape::scalar();
    let source = fill_2d_scalar(Arc::clone(&source_pencil), 0);
    let source_before = source.as_slice().to_vec();
    let all_plan =
        AllToAllvTransposePlan::new(Arc::clone(&source_pencil), Arc::clone(&destination_pencil))
            .unwrap();
    let p2p_plan =
        PointToPointTransposePlan::new(Arc::clone(&source_pencil), Arc::clone(&destination_pencil))
            .unwrap();
    let all_requirements = all_plan.workspace_requirements(&extra).unwrap();
    let p2p_requirements = p2p_plan.workspace_requirements(&extra).unwrap();
    assert_eq!(all_requirements, p2p_requirements);
    let rank = usize::try_from(topology.rank()).unwrap();
    if topology.size() == 4 && rank == 0 {
        assert_eq!(p2p_requirements.send_len, 0);
        assert!(p2p_requirements.receive_len > 0);
    }
    if topology.size() == 6 {
        match rank {
            0 => {
                assert_eq!(p2p_requirements.send_len, 0);
                assert_eq!(p2p_requirements.receive_len, 0);
            }
            2 | 4 => {
                assert_eq!(p2p_requirements.send_len, 0);
                assert!(p2p_requirements.receive_len > 0);
            }
            _ => {
                assert!(p2p_requirements.send_len > 0);
                assert!(p2p_requirements.receive_len > 0);
            }
        }
    }
    let mut all_destination =
        PencilArray::from_elem(Arc::clone(&destination_pencil), extra.clone(), u64::MAX).unwrap();
    let mut p2p_destination =
        PencilArray::from_elem(Arc::clone(&destination_pencil), extra.clone(), u64::MAX).unwrap();
    let mut all_workspace = AllToAllvTransposeWorkspace::from_vecs(
        vec![0_u64; all_requirements.send_len],
        vec![0_u64; all_requirements.receive_len],
    );
    let mut p2p_workspace = AllToAllvTransposeWorkspace::from_vecs(
        vec![0_u64; p2p_requirements.send_len],
        vec![0_u64; p2p_requirements.receive_len],
    );
    all_plan
        .execute_views(
            source.view(),
            all_destination.view_mut(),
            &mut all_workspace,
        )
        .unwrap();
    p2p_plan
        .execute_views(
            source.view(),
            p2p_destination.view_mut(),
            &mut p2p_workspace,
        )
        .unwrap();
    assert_eq!(source.as_slice(), source_before.as_slice());
    assert_eq!(p2p_destination.as_slice(), all_destination.as_slice());
    for x in 0..destination_pencil.local_shape_logical()[0] {
        for y in 0..destination_pencil.local_shape_logical()[1] {
            let global = [
                destination_pencil.local_ranges()[0].start + x,
                destination_pencil.local_ranges()[1].start + y,
            ];
            assert_eq!(
                p2p_destination.get_local(&[], [x, y]),
                Some(&value_2d_u64([0, 0], global))
            );
        }
    }

    let reverse =
        PointToPointTransposePlan::new(Arc::clone(&destination_pencil), Arc::clone(&source_pencil))
            .unwrap();
    let reverse_requirements = reverse.workspace_requirements(&extra).unwrap();
    let mut reusable_workspace = AllToAllvTransposeWorkspace::from_vecs(
        vec![0_u64; p2p_requirements.send_len.max(reverse_requirements.send_len)],
        vec![
            0_u64;
            p2p_requirements
                .receive_len
                .max(reverse_requirements.receive_len)
        ],
    );
    let mut roundtrip =
        PencilArray::from_elem(Arc::clone(&source_pencil), extra.clone(), 0_u64).unwrap();
    reverse
        .execute_views(
            p2p_destination.view(),
            roundtrip.view_mut(),
            &mut reusable_workspace,
        )
        .unwrap();
    assert_eq!(roundtrip.as_slice(), source_before.as_slice());

    // A zero-sized extra dimension keeps the spatial metadata but has no
    // payload, so every point-to-point post is omitted.
    let zero_extra = ExtraShape::new([3, 0]).unwrap();
    let zero_source = PencilArray::<u64, 2, 1>::from_vec(
        Arc::clone(&source_pencil),
        zero_extra.clone(),
        Vec::new(),
    )
    .unwrap();
    let mut zero_all_destination = PencilArray::<u64, 2, 1>::from_vec(
        Arc::clone(&destination_pencil),
        zero_extra.clone(),
        Vec::new(),
    )
    .unwrap();
    let mut zero_p2p_destination = PencilArray::<u64, 2, 1>::from_vec(
        Arc::clone(&destination_pencil),
        zero_extra.clone(),
        Vec::new(),
    )
    .unwrap();
    let zero_all_requirements = all_plan.workspace_requirements(&zero_extra).unwrap();
    let zero_p2p_requirements = p2p_plan.workspace_requirements(&zero_extra).unwrap();
    assert_eq!(zero_all_requirements.send_len, 0);
    assert_eq!(zero_all_requirements.receive_len, 0);
    assert_eq!(zero_all_requirements, zero_p2p_requirements);
    let mut zero_all_workspace = AllToAllvTransposeWorkspace::from_vecs(Vec::new(), Vec::new());
    let mut zero_p2p_workspace = AllToAllvTransposeWorkspace::from_vecs(Vec::new(), Vec::new());
    all_plan
        .execute_views(
            zero_source.view(),
            zero_all_destination.view_mut(),
            &mut zero_all_workspace,
        )
        .unwrap();
    p2p_plan
        .execute_views(
            zero_source.view(),
            zero_p2p_destination.view_mut(),
            &mut zero_p2p_workspace,
        )
        .unwrap();
    assert!(zero_p2p_destination.is_empty());
}

fn run_point_to_point_large_payload(topology: &Arc<MpiTopology<1>>) {
    let size = topology.size();
    // Each nonzero peer segment is 128 * 512 u64 values = 512 KiB.
    let global_shape = [128 * size, 512 * size];
    let source_pencil = Pencil::<2, 1>::new(Arc::clone(topology), global_shape, [0]).unwrap();
    let destination_pencil = source_pencil.with_decomposition([1]).unwrap();
    let shape = source_pencil.local_shape_logical();
    let ranges = source_pencil.local_ranges();
    let mut storage = Vec::with_capacity(source_pencil.local_len());
    for x in 0..shape[0] {
        for y in 0..shape[1] {
            let global_x = ranges[0].start + x;
            let global_y = ranges[1].start + y;
            storage.push((global_x * global_shape[1] + global_y) as u64);
        }
    }
    let source =
        PencilArray::from_vec(Arc::clone(&source_pencil), ExtraShape::scalar(), storage).unwrap();
    let all_plan =
        AllToAllvTransposePlan::new(Arc::clone(&source_pencil), Arc::clone(&destination_pencil))
            .unwrap();
    let p2p_plan =
        PointToPointTransposePlan::new(Arc::clone(&source_pencil), Arc::clone(&destination_pencil))
            .unwrap();
    let all_requirements = all_plan
        .workspace_requirements(&ExtraShape::scalar())
        .unwrap();
    let p2p_requirements = p2p_plan
        .workspace_requirements(&ExtraShape::scalar())
        .unwrap();
    assert_eq!(all_requirements, p2p_requirements);
    let mut all_destination = PencilArray::from_elem(
        Arc::clone(&destination_pencil),
        ExtraShape::scalar(),
        u64::MAX,
    )
    .unwrap();
    let mut p2p_destination = PencilArray::from_elem(
        Arc::clone(&destination_pencil),
        ExtraShape::scalar(),
        u64::MAX,
    )
    .unwrap();
    let mut all_workspace = AllToAllvTransposeWorkspace::from_vecs(
        vec![0_u64; all_requirements.send_len],
        vec![0_u64; all_requirements.receive_len],
    );
    let mut p2p_workspace = AllToAllvTransposeWorkspace::from_vecs(
        vec![0_u64; p2p_requirements.send_len],
        vec![0_u64; p2p_requirements.receive_len],
    );
    all_plan
        .execute_views(
            source.view(),
            all_destination.view_mut(),
            &mut all_workspace,
        )
        .unwrap();
    p2p_plan
        .execute_views(
            source.view(),
            p2p_destination.view_mut(),
            &mut p2p_workspace,
        )
        .unwrap();
    assert_eq!(p2p_destination.as_slice(), all_destination.as_slice());
    let destination_shape = destination_pencil.local_shape_logical();
    for x in 0..destination_shape[0] {
        for y in 0..destination_shape[1] {
            let global = [
                destination_pencil.local_ranges()[0].start + x,
                destination_pencil.local_ranges()[1].start + y,
            ];
            assert_eq!(
                p2p_destination.get_local(&[], [x, y]),
                Some(&((global[0] * global_shape[1] + global[1]) as u64))
            );
        }
    }
}

fn run_point_to_point_preflight_failures(topology: &Arc<MpiTopology<1>>) {
    let source_pencil = Pencil::<2, 1>::new_permuted(
        Arc::clone(topology),
        [7, 9],
        [0],
        AxisPermutation::identity(),
    )
    .unwrap();
    let destination_pencil = source_pencil
        .with_permutation(AxisPermutation::new([1, 0]).unwrap())
        .unwrap()
        .with_decomposition([1])
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
        PointToPointTransposePlan::new(Arc::clone(&source_pencil), Arc::clone(&destination_pencil))
            .unwrap();
    let requirements = plan.workspace_requirements(&extra).unwrap();
    let rank = usize::try_from(topology.rank()).unwrap();
    let size = topology.size();
    let full_workspace = || {
        AllToAllvTransposeWorkspace::from_vecs(
            vec![0_u64; requirements.send_len],
            vec![0_u64; requirements.receive_len],
        )
    };

    // A rank-zero send shortage is agreed before pack or request creation.
    let mut destination =
        PencilArray::from_elem(Arc::clone(&destination_pencil), extra.clone(), 700_u64).unwrap();
    let destination_before = destination.as_slice().to_vec();
    let mut workspace = if rank == 0 {
        AllToAllvTransposeWorkspace::from_vecs(Vec::new(), vec![0_u64; requirements.receive_len])
    } else {
        full_workspace()
    };
    assert!(
        plan.execute_views(source.view(), destination.view_mut(), &mut workspace)
            .is_err()
    );
    assert_eq!(source.as_slice(), source_before.as_slice());
    assert_eq!(destination.as_slice(), destination_before.as_slice());

    // The last coordinate exercises the opposite shortage independently.
    let last = topology.local_coords()[0] + 1 == topology.process_grid()[0];
    let mut destination =
        PencilArray::from_elem(Arc::clone(&destination_pencil), extra.clone(), 701_u64).unwrap();
    let destination_before = destination.as_slice().to_vec();
    let mut workspace = if last {
        AllToAllvTransposeWorkspace::from_vecs(vec![0_u64; requirements.send_len], Vec::new())
    } else {
        full_workspace()
    };
    assert!(
        plan.execute_views(source.view(), destination.view_mut(), &mut workspace)
            .is_err()
    );
    assert_eq!(source.as_slice(), source_before.as_slice());
    assert_eq!(destination.as_slice(), destination_before.as_slice());

    // A rank-local destination layout error does not write its destination.
    let result = if rank == 0 {
        let mut wrong_destination =
            PencilArray::from_elem(Arc::clone(&source_pencil), extra.clone(), 702_u64).unwrap();
        let before = wrong_destination.as_slice().to_vec();
        let mut workspace = full_workspace();
        let result =
            plan.execute_views(source.view(), wrong_destination.view_mut(), &mut workspace);
        assert!(result.is_err());
        assert_eq!(wrong_destination.as_slice(), before.as_slice());
        result
    } else {
        let mut normal_destination =
            PencilArray::from_elem(Arc::clone(&destination_pencil), extra.clone(), 702_u64)
                .unwrap();
        let before = normal_destination.as_slice().to_vec();
        let mut workspace = full_workspace();
        let result =
            plan.execute_views(source.view(), normal_destination.view_mut(), &mut workspace);
        assert!(result.is_err());
        assert_eq!(normal_destination.as_slice(), before.as_slice());
        result
    };
    assert!(result.is_err());

    // A rank-local extra extent disagreement is rejected by the descriptor.
    let local_extra = if rank == 0 {
        ExtraShape::new([3, 2]).unwrap()
    } else {
        extra.clone()
    };
    let local_source = fill_2d(
        Arc::clone(&source_pencil),
        local_extra.clone(),
        0_u64,
        value_2d_u64,
    );
    let mut local_destination = PencilArray::from_elem(
        Arc::clone(&destination_pencil),
        local_extra.clone(),
        703_u64,
    )
    .unwrap();
    let local_before = local_destination.as_slice().to_vec();
    let local_requirements = plan.workspace_requirements(&local_extra).unwrap();
    let mut local_workspace = AllToAllvTransposeWorkspace::from_vecs(
        vec![0_u64; local_requirements.send_len],
        vec![0_u64; local_requirements.receive_len],
    );
    let result = plan.execute_views(
        local_source.view(),
        local_destination.view_mut(),
        &mut local_workspace,
    );
    if size == 1 {
        assert!(result.is_ok());
    } else {
        assert!(result.is_err());
        assert_eq!(local_destination.as_slice(), local_before.as_slice());
    }

    // A rank-count disagreement takes the fixed-header return path.
    let rank_local_extra = if rank == 0 {
        ExtraShape::new([6]).unwrap()
    } else {
        extra.clone()
    };
    let rank_local_source =
        PencilArray::from_elem(Arc::clone(&source_pencil), rank_local_extra.clone(), 0_u64)
            .unwrap();
    let mut rank_local_destination = PencilArray::from_elem(
        Arc::clone(&destination_pencil),
        rank_local_extra.clone(),
        703_u64,
    )
    .unwrap();
    let rank_local_before = rank_local_destination.as_slice().to_vec();
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
    } else {
        assert!(result.is_err());
        assert_eq!(
            rank_local_destination.as_slice(),
            rank_local_before.as_slice()
        );
    }

    let reverse =
        PointToPointTransposePlan::new(Arc::clone(&destination_pencil), Arc::clone(&source_pencil))
            .unwrap();
    let reverse_requirements = reverse.workspace_requirements(&extra).unwrap();
    let mut direction_destination =
        PencilArray::from_elem(Arc::clone(&destination_pencil), extra.clone(), 704_u64).unwrap();
    let direction_before = direction_destination.as_slice().to_vec();
    // This sufficiently sized workspace is used for the direction mismatch
    // and retained for the final successful execution below.
    let mut reusable_workspace = AllToAllvTransposeWorkspace::from_vecs(
        vec![0_u64; requirements.send_len.max(reverse_requirements.send_len)],
        vec![
            0_u64;
            requirements
                .receive_len
                .max(reverse_requirements.receive_len)
        ],
    );
    let result = if rank == 0 {
        reverse.execute_views(
            source.view(),
            direction_destination.view_mut(),
            &mut reusable_workspace,
        )
    } else {
        plan.execute_views(
            source.view(),
            direction_destination.view_mut(),
            &mut reusable_workspace,
        )
    };
    assert!(result.is_err());
    assert_eq!(
        direction_destination.as_slice(),
        direction_before.as_slice()
    );

    // A rank-local element type mismatch is also stopped before payload traffic.
    let source_f64 = fill_2d(
        Arc::clone(&source_pencil),
        extra.clone(),
        0.0_f64,
        value_2d_f64,
    );
    let mut destination_f64 =
        PencilArray::from_elem(Arc::clone(&destination_pencil), extra.clone(), 705.0_f64).unwrap();
    let mut destination_u64 =
        PencilArray::from_elem(Arc::clone(&destination_pencil), extra.clone(), 705_u64).unwrap();
    let f64_before = destination_f64.as_slice().to_vec();
    let u64_before = destination_u64.as_slice().to_vec();
    let result = if rank == 0 {
        let mut workspace = point_to_point_workspace_for(&plan, &extra, 0.0_f64);
        plan.execute_views(
            source_f64.view(),
            destination_f64.view_mut(),
            &mut workspace,
        )
    } else {
        let mut workspace = full_workspace();
        plan.execute_views(source.view(), destination_u64.view_mut(), &mut workspace)
    };
    if size == 1 {
        assert!(result.is_ok());
    } else {
        assert!(result.is_err());
        assert_eq!(destination_f64.as_slice(), f64_before.as_slice());
        assert_eq!(destination_u64.as_slice(), u64_before.as_slice());
    }

    // Construction modes have distinct fixed-header operation codes.
    let mixed_new = if rank == 0 {
        PointToPointTransposePlan::new(Arc::clone(&source_pencil), Arc::clone(&destination_pencil))
            .map(|_| ())
    } else {
        AllToAllvTransposePlan::new(Arc::clone(&source_pencil), Arc::clone(&destination_pencil))
            .map(|_| ())
    };
    if size == 1 {
        assert!(mixed_new.is_ok());
    } else {
        assert!(mixed_new.is_err());
    }

    let all_plan =
        AllToAllvTransposePlan::new(Arc::clone(&source_pencil), Arc::clone(&destination_pencil))
            .unwrap();
    let mut mixed_destination =
        PencilArray::from_elem(Arc::clone(&destination_pencil), extra.clone(), 706_u64).unwrap();
    let mixed_before = mixed_destination.as_slice().to_vec();
    let mut mixed_workspace = full_workspace();
    let result = if rank == 0 {
        plan.execute_views(
            source.view(),
            mixed_destination.view_mut(),
            &mut mixed_workspace,
        )
    } else {
        all_plan.execute_views(
            source.view(),
            mixed_destination.view_mut(),
            &mut mixed_workspace,
        )
    };
    if size == 1 {
        assert!(result.is_ok());
    } else {
        assert!(result.is_err());
        assert_eq!(mixed_destination.as_slice(), mixed_before.as_slice());
    }

    // P2P views and Alltoallv in-place likewise cannot share one collective.
    if rank == 0 {
        let mut destination =
            PencilArray::from_elem(Arc::clone(&destination_pencil), extra.clone(), 707_u64)
                .unwrap();
        let before = destination.as_slice().to_vec();
        let mut workspace = full_workspace();
        let result = plan.execute_views(source.view(), destination.view_mut(), &mut workspace);
        if size == 1 {
            assert!(result.is_ok());
        } else {
            assert!(result.is_err());
            assert_eq!(destination.as_slice(), before.as_slice());
        }
    } else {
        let mut array = many_from_source_2d(
            Arc::clone(&source_pencil),
            Arc::clone(&destination_pencil),
            extra.clone(),
            &source,
            707_u64,
        );
        let before = array.active_view().unwrap().as_slice().to_vec();
        let mut workspace = workspace_for(&all_plan, &extra, 0_u64);
        let result = all_plan.execute_in_place(&mut array, &mut workspace);
        if size == 1 {
            unreachable!("the nonzero rank branch is not used for one rank");
        } else {
            assert!(result.is_err());
            assert_eq!(array.active_view().unwrap().as_slice(), before.as_slice());
        }
    }

    // A constructor and execution call are rejected at the same header.
    let mixed_new_execute = if rank == 0 {
        PointToPointTransposePlan::new(Arc::clone(&source_pencil), Arc::clone(&destination_pencil))
            .map(|_| ())
    } else {
        let mut destination =
            PencilArray::from_elem(Arc::clone(&destination_pencil), extra.clone(), 708_u64)
                .unwrap();
        let mut workspace = full_workspace();
        plan.execute_views(source.view(), destination.view_mut(), &mut workspace)
    };
    if size == 1 {
        assert!(mixed_new_execute.is_ok());
    } else {
        assert!(mixed_new_execute.is_err());
    }

    // The original plan and the workspace used by the direction failure remain
    // reusable after every preflight; no new workspace is made for success.
    let mut destination =
        PencilArray::from_elem(Arc::clone(&destination_pencil), extra.clone(), 709_u64).unwrap();
    plan.execute_views(
        source.view(),
        destination.view_mut(),
        &mut reusable_workspace,
    )
    .unwrap();
    check_2d(&destination, value_2d_u64);
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

fn run_3d_point_to_point_success(
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
    let expected = expected_3d_physical(&destination_pencil);
    let all_plan =
        AllToAllvTransposePlan::new(Arc::clone(&source_pencil), Arc::clone(&destination_pencil))
            .unwrap();
    let p2p_plan =
        PointToPointTransposePlan::new(Arc::clone(&source_pencil), Arc::clone(&destination_pencil))
            .unwrap();
    let extra = ExtraShape::scalar();
    let all_requirements = all_plan.workspace_requirements(&extra).unwrap();
    let p2p_requirements = p2p_plan.workspace_requirements(&extra).unwrap();
    assert_eq!(all_requirements, p2p_requirements);
    let mut all_destination =
        PencilArray::from_elem(Arc::clone(&destination_pencil), extra.clone(), u64::MAX).unwrap();
    let mut p2p_destination =
        PencilArray::from_elem(Arc::clone(&destination_pencil), extra.clone(), u64::MAX).unwrap();
    let mut all_workspace = AllToAllvTransposeWorkspace::from_vecs(
        vec![0_u64; all_requirements.send_len],
        vec![0_u64; all_requirements.receive_len],
    );
    let mut p2p_workspace = AllToAllvTransposeWorkspace::from_vecs(
        vec![0_u64; p2p_requirements.send_len],
        vec![0_u64; p2p_requirements.receive_len],
    );
    all_plan
        .execute_views(
            source.view(),
            all_destination.view_mut(),
            &mut all_workspace,
        )
        .unwrap();
    p2p_plan
        .execute_views(
            source.view(),
            p2p_destination.view_mut(),
            &mut p2p_workspace,
        )
        .unwrap();
    assert_eq!(source.as_slice(), source_before.as_slice());
    assert_eq!(all_destination.as_slice(), expected.as_slice());
    assert_eq!(p2p_destination.as_slice(), expected.as_slice());
    assert_eq!(p2p_destination.as_slice(), all_destination.as_slice());
    check_3d(&p2p_destination);

    let reverse =
        PointToPointTransposePlan::new(Arc::clone(&destination_pencil), Arc::clone(&source_pencil))
            .unwrap();
    let reverse_requirements = reverse.workspace_requirements(&extra).unwrap();
    let max_send = p2p_requirements.send_len.max(reverse_requirements.send_len);
    let max_receive = p2p_requirements
        .receive_len
        .max(reverse_requirements.receive_len);
    let mut reusable_workspace =
        AllToAllvTransposeWorkspace::from_vecs(vec![0_u64; max_send], vec![0_u64; max_receive]);
    let mut roundtrip = PencilArray::from_elem(Arc::clone(&source_pencil), extra, 0_u64).unwrap();
    reverse
        .execute_views(
            p2p_destination.view(),
            roundtrip.view_mut(),
            &mut reusable_workspace,
        )
        .unwrap();
    assert_eq!(roundtrip.as_slice(), source_before.as_slice());
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

fn point_to_point_workspace_for<T: Clone>(
    plan: &PointToPointTransposePlan<2, 1>,
    extra: &ExtraShape,
    value: T,
) -> AllToAllvTransposeWorkspace<T> {
    let requirements = plan.workspace_requirements(extra).unwrap();
    AllToAllvTransposeWorkspace::from_vecs(
        vec![value.clone(); requirements.send_len],
        vec![value; requirements.receive_len],
    )
}

fn fill_2d_scalar(pencil: Arc<Pencil<2, 1>>, value: u64) -> PencilArray<u64, 2, 1> {
    let mut array =
        PencilArray::from_elem(Arc::clone(&pencil), ExtraShape::scalar(), value).unwrap();
    let shape = pencil.local_shape_logical();
    for x in 0..shape[0] {
        for y in 0..shape[1] {
            let global = [
                pencil.local_ranges()[0].start + x,
                pencil.local_ranges()[1].start + y,
            ];
            *array.get_local_mut(&[], [x, y]).unwrap() = value_2d_u64([0, 0], global);
        }
    }
    array
}

fn many_from_source_2d(
    source_pencil: Arc<Pencil<2, 1>>,
    destination_pencil: Arc<Pencil<2, 1>>,
    extra_shape: ExtraShape,
    source: &PencilArray<u64, 2, 1>,
    tail: u64,
) -> ManyPencilArray<u64, 2, 1> {
    let required = source_pencil
        .local_len()
        .max(destination_pencil.local_len())
        * extra_shape.element_count();
    let mut array = ManyPencilArray::from_vec(
        vec![source_pencil, destination_pencil],
        0,
        extra_shape,
        vec![tail; required],
    )
    .unwrap();
    array
        .active_view_mut()
        .unwrap()
        .as_mut_slice()
        .copy_from_slice(source.as_slice());
    array
}

fn many_2d_with_value(
    source_pencil: Arc<Pencil<2, 1>>,
    destination_pencil: Arc<Pencil<2, 1>>,
    active: usize,
    extra_shape: ExtraShape,
    value: u64,
) -> ManyPencilArray<u64, 2, 1> {
    ManyPencilArray::from_elem(
        vec![source_pencil, destination_pencil],
        active,
        extra_shape,
        value,
    )
    .unwrap()
}

fn run_2d_in_place_success(topology: &Arc<MpiTopology<1>>) {
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
    let source = fill_2d(
        Arc::clone(&source_pencil),
        extra.clone(),
        0_u64,
        value_2d_u64,
    );
    let source_before = source.as_slice().to_vec();
    let mut expected =
        PencilArray::from_elem(Arc::clone(&destination_pencil), extra.clone(), 0_u64).unwrap();
    let plan =
        AllToAllvTransposePlan::new(Arc::clone(&source_pencil), Arc::clone(&destination_pencil))
            .unwrap();
    let mut out_workspace = workspace_for(&plan, &extra, 0_u64);
    plan.execute_views(source.view(), expected.view_mut(), &mut out_workspace)
        .unwrap();

    let mut array = many_from_source_2d(
        Arc::clone(&source_pencil),
        Arc::clone(&destination_pencil),
        extra.clone(),
        &source,
        u64::MAX,
    );
    let mut in_workspace = workspace_for(&plan, &extra, 0_u64);
    plan.execute_in_place(&mut array, &mut in_workspace)
        .unwrap();
    assert!(
        array
            .active_pencil()
            .unwrap()
            .same_layout(destination_pencil.as_ref())
    );
    assert_eq!(array.active_view().unwrap().as_slice(), expected.as_slice());

    let reverse =
        AllToAllvTransposePlan::new(Arc::clone(&destination_pencil), Arc::clone(&source_pencil))
            .unwrap();
    let mut reverse_workspace = workspace_for(&reverse, &extra, 0_u64);
    reverse
        .execute_in_place(&mut array, &mut reverse_workspace)
        .unwrap();
    assert!(
        array
            .active_pencil()
            .unwrap()
            .same_layout(source_pencil.as_ref())
    );
    assert_eq!(
        array.active_view().unwrap().as_slice(),
        source_before.as_slice()
    );
}

fn run_2d_empty_partition_success(topology: &Arc<MpiTopology<1>>) {
    let source_pencil = Pencil::<2, 1>::new(Arc::clone(topology), [3, 5], [0]).unwrap();
    let destination_pencil = source_pencil.with_decomposition([1]).unwrap();
    let coordinate = topology.local_coords()[0];
    let empty_source_nonempty_destination = (topology.process_grid()[0] == 4 && coordinate == 0)
        || (topology.process_grid()[0] == 6 && matches!(coordinate, 2 | 4));
    if empty_source_nonempty_destination {
        assert_eq!(source_pencil.local_len(), 0);
        assert!(destination_pencil.local_len() > 0);
    }
    let source = fill_2d_scalar(Arc::clone(&source_pencil), 0);
    let source_before = source.as_slice().to_vec();
    let mut expected =
        PencilArray::from_elem(Arc::clone(&destination_pencil), ExtraShape::scalar(), 0_u64)
            .unwrap();
    let plan =
        AllToAllvTransposePlan::new(Arc::clone(&source_pencil), Arc::clone(&destination_pencil))
            .unwrap();
    let mut out_workspace = AllToAllvTransposeWorkspace::from_vecs(
        vec![
            0_u64;
            plan.workspace_requirements(&ExtraShape::scalar())
                .unwrap()
                .send_len
        ],
        vec![
            0_u64;
            plan.workspace_requirements(&ExtraShape::scalar())
                .unwrap()
                .receive_len
        ],
    );
    plan.execute_views(source.view(), expected.view_mut(), &mut out_workspace)
        .unwrap();

    let mut array = many_from_source_2d(
        Arc::clone(&source_pencil),
        Arc::clone(&destination_pencil),
        ExtraShape::scalar(),
        &source,
        u64::MAX,
    );
    let mut in_workspace = AllToAllvTransposeWorkspace::from_vecs(
        vec![
            0_u64;
            plan.workspace_requirements(&ExtraShape::scalar())
                .unwrap()
                .send_len
        ],
        vec![
            0_u64;
            plan.workspace_requirements(&ExtraShape::scalar())
                .unwrap()
                .receive_len
        ],
    );
    plan.execute_in_place(&mut array, &mut in_workspace)
        .unwrap();
    assert_eq!(array.active_view().unwrap().as_slice(), expected.as_slice());

    let reverse =
        AllToAllvTransposePlan::new(Arc::clone(&destination_pencil), Arc::clone(&source_pencil))
            .unwrap();
    let mut reverse_workspace = AllToAllvTransposeWorkspace::from_vecs(
        vec![
            0_u64;
            reverse
                .workspace_requirements(&ExtraShape::scalar())
                .unwrap()
                .send_len
        ],
        vec![
            0_u64;
            reverse
                .workspace_requirements(&ExtraShape::scalar())
                .unwrap()
                .receive_len
        ],
    );
    reverse
        .execute_in_place(&mut array, &mut reverse_workspace)
        .unwrap();
    assert_eq!(
        array.active_view().unwrap().as_slice(),
        source_before.as_slice()
    );
}

fn run_3d_in_place_success(topology: &Arc<MpiTopology<2>>) {
    let source_pencil = Pencil::<3, 2>::new_permuted(
        Arc::clone(topology),
        [5, 2, 7],
        [2, 0],
        AxisPermutation::new([2, 0, 1]).unwrap(),
    )
    .unwrap();
    let destination_pencil = Pencil::<3, 2>::new_permuted(
        Arc::clone(topology),
        [5, 2, 7],
        [2, 1],
        AxisPermutation::new([1, 2, 0]).unwrap(),
    )
    .unwrap();
    let source = fill_3d_physical(Arc::clone(&source_pencil));
    let source_before = source.as_slice().to_vec();
    let expected_physical = expected_3d_physical(&destination_pencil);
    let mut expected = PencilArray::from_elem(
        Arc::clone(&destination_pencil),
        ExtraShape::scalar(),
        u64::MAX,
    )
    .unwrap();
    let plan =
        AllToAllvTransposePlan::new(Arc::clone(&source_pencil), Arc::clone(&destination_pencil))
            .unwrap();
    let requirements = plan.workspace_requirements(&ExtraShape::scalar()).unwrap();
    let mut out_workspace = AllToAllvTransposeWorkspace::from_vecs(
        vec![0_u64; requirements.send_len],
        vec![0_u64; requirements.receive_len],
    );
    plan.execute_views(source.view(), expected.view_mut(), &mut out_workspace)
        .unwrap();

    let required = source_pencil
        .local_len()
        .max(destination_pencil.local_len());
    let mut array = ManyPencilArray::from_vec(
        vec![Arc::clone(&source_pencil), Arc::clone(&destination_pencil)],
        0,
        ExtraShape::scalar(),
        vec![u64::MAX; required],
    )
    .unwrap();
    array
        .active_view_mut()
        .unwrap()
        .as_mut_slice()
        .copy_from_slice(source.as_slice());
    let mut in_workspace = AllToAllvTransposeWorkspace::from_vecs(
        vec![0_u64; requirements.send_len],
        vec![0_u64; requirements.receive_len],
    );
    plan.execute_in_place(&mut array, &mut in_workspace)
        .unwrap();
    assert_eq!(
        array.active_view().unwrap().as_slice(),
        expected_physical.as_slice()
    );

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
    reverse
        .execute_in_place(&mut array, &mut reverse_workspace)
        .unwrap();
    assert_eq!(
        array.active_view().unwrap().as_slice(),
        source_before.as_slice()
    );
}

fn run_3d_tail_check(topology: &Arc<MpiTopology<2>>) {
    let source_pencil = Pencil::<3, 2>::new_permuted(
        Arc::clone(topology),
        [5, 2, 7],
        [2, 0],
        AxisPermutation::new([2, 0, 1]).unwrap(),
    )
    .unwrap();
    let destination_pencil = Pencil::<3, 2>::new_permuted(
        Arc::clone(topology),
        [5, 2, 7],
        [2, 1],
        AxisPermutation::new([1, 2, 0]).unwrap(),
    )
    .unwrap();
    let third_pencil = Pencil::<3, 2>::new_permuted(
        Arc::clone(topology),
        [5, 2, 7],
        [0, 1],
        AxisPermutation::identity(),
    )
    .unwrap();
    let source = fill_3d_physical(Arc::clone(&source_pencil));
    let mut expected = PencilArray::from_elem(
        Arc::clone(&destination_pencil),
        ExtraShape::scalar(),
        u64::MAX,
    )
    .unwrap();
    let plan =
        AllToAllvTransposePlan::new(Arc::clone(&source_pencil), Arc::clone(&destination_pencil))
            .unwrap();
    let requirements = plan.workspace_requirements(&ExtraShape::scalar()).unwrap();
    let mut out_workspace = AllToAllvTransposeWorkspace::from_vecs(
        vec![0_u64; requirements.send_len],
        vec![0_u64; requirements.receive_len],
    );
    plan.execute_views(source.view(), expected.view_mut(), &mut out_workspace)
        .unwrap();

    let max_len = source_pencil
        .local_len()
        .max(destination_pencil.local_len())
        .max(third_pencil.local_len());
    let mut initial = vec![0xfeed_u64; max_len];
    initial[..source.as_slice().len()].copy_from_slice(source.as_slice());
    let mut array = ManyPencilArray::from_vec(
        vec![
            Arc::clone(&source_pencil),
            Arc::clone(&destination_pencil),
            Arc::clone(&third_pencil),
        ],
        0,
        ExtraShape::scalar(),
        initial.clone(),
    )
    .unwrap();
    let mut in_workspace = AllToAllvTransposeWorkspace::from_vecs(
        vec![0_u64; requirements.send_len],
        vec![0_u64; requirements.receive_len],
    );
    plan.execute_in_place(&mut array, &mut in_workspace)
        .unwrap();
    assert_eq!(array.active_view().unwrap().as_slice(), expected.as_slice());

    let destination_before = array.active_view().unwrap().as_slice().to_vec();
    let mut failed_workspace = AllToAllvTransposeWorkspace::from_vecs(Vec::new(), Vec::new());
    assert!(
        plan.execute_in_place(&mut array, &mut failed_workspace)
            .is_err()
    );
    assert_eq!(
        array.active_view().unwrap().as_slice(),
        destination_before.as_slice()
    );

    if (*topology.process_grid() == [2, 2] && *topology.local_coords() == [1, 0])
        || (*topology.process_grid() == [2, 3] && *topology.local_coords() == [1, 1])
    {
        assert!(third_pencil.local_len() > destination_pencil.local_len());
    }

    if third_pencil.local_len() > destination_pencil.local_len() {
        let destination_len = destination_pencil.local_len();
        // The preflight failure above did not consume the unused storage tail.
        array
            .overwrite_with(third_pencil.as_ref(), |mut view| {
                assert_eq!(
                    &view.as_slice()[destination_len..],
                    &initial[destination_len..third_pencil.local_len()]
                );
                view.as_mut_slice()
                    .copy_from_slice(&initial[..third_pencil.local_len()]);
                Ok::<_, ()>(())
            })
            .unwrap();
    }
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
    run_2d_point_to_point_success(&topology_1d, value_2d_u64, 0_u64);
    run_2d_point_to_point_success(&topology_1d, value_2d_f64, 0.0_f64);
    run_2d_point_to_point_empty_success(&topology_1d);
    run_point_to_point_preflight_failures(&topology_1d);
    run_point_to_point_large_payload(&topology_1d);
    run_3d_success(&topology_2d, [2, 0], [2, 1]);
    run_3d_success(&topology_2d, [0, 2], [1, 2]);
    run_3d_point_to_point_success(&topology_2d, [2, 0], [2, 1]);
    run_3d_point_to_point_success(&topology_2d, [0, 2], [1, 2]);
    run_2d_in_place_success(&topology_1d);
    run_2d_empty_partition_success(&topology_1d);
    run_3d_in_place_success(&topology_2d);
    run_3d_tail_check(&topology_2d);
    world.barrier();

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

    let inplace_source_pencil = Pencil::<2, 1>::new_permuted(
        Arc::clone(&topology_1d),
        [7, 9],
        [0],
        AxisPermutation::new([0, 1]).unwrap(),
    )
    .unwrap();
    let inplace_destination_pencil = Pencil::<2, 1>::new_permuted(
        Arc::clone(&topology_1d),
        [7, 9],
        [1],
        AxisPermutation::new([1, 0]).unwrap(),
    )
    .unwrap();
    let inplace_extra = ExtraShape::new([2, 3]).unwrap();
    let inplace_source = fill_2d(
        Arc::clone(&inplace_source_pencil),
        inplace_extra.clone(),
        0_u64,
        value_2d_u64,
    );
    let inplace_plan = AllToAllvTransposePlan::new(
        Arc::clone(&inplace_source_pencil),
        Arc::clone(&inplace_destination_pencil),
    )
    .unwrap();
    let mut expected_inplace = PencilArray::from_elem(
        Arc::clone(&inplace_destination_pencil),
        inplace_extra.clone(),
        0_u64,
    )
    .unwrap();
    let mut expected_workspace = workspace_for(&inplace_plan, &inplace_extra, 0_u64);
    inplace_plan
        .execute_views(
            inplace_source.view(),
            expected_inplace.view_mut(),
            &mut expected_workspace,
        )
        .unwrap();
    let inplace_requirements = inplace_plan.workspace_requirements(&inplace_extra).unwrap();

    // A source-layout mismatch is collected before the shared array is touched.
    let active = usize::from(rank == 0);
    let mut bad_source = many_2d_with_value(
        Arc::clone(&inplace_source_pencil),
        Arc::clone(&inplace_destination_pencil),
        active,
        inplace_extra.clone(),
        101,
    );
    if active == 0 {
        bad_source
            .active_view_mut()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(inplace_source.as_slice());
    }
    let bad_source_before = bad_source.active_view().unwrap().as_slice().to_vec();
    let mut bad_source_workspace = workspace_for(&inplace_plan, &inplace_extra, 0_u64);
    let result = inplace_plan.execute_in_place(&mut bad_source, &mut bad_source_workspace);
    assert!(result.is_err());
    let expected_active = if rank == 0 {
        &inplace_destination_pencil
    } else {
        &inplace_source_pencil
    };
    assert!(
        bad_source
            .active_pencil()
            .unwrap()
            .same_layout(expected_active.as_ref())
    );
    assert_eq!(
        bad_source.active_view().unwrap().as_slice(),
        bad_source_before
    );
    world.barrier();

    // A destination absent from one registry is also a preflight-only error.
    let mut missing_destination = if rank == 0 {
        ManyPencilArray::from_elem(
            vec![Arc::clone(&inplace_source_pencil)],
            0,
            inplace_extra.clone(),
            102_u64,
        )
        .unwrap()
    } else {
        many_2d_with_value(
            Arc::clone(&inplace_source_pencil),
            Arc::clone(&inplace_destination_pencil),
            0,
            inplace_extra.clone(),
            102,
        )
    };
    missing_destination
        .active_view_mut()
        .unwrap()
        .as_mut_slice()
        .copy_from_slice(inplace_source.as_slice());
    let missing_before = missing_destination
        .active_view()
        .unwrap()
        .as_slice()
        .to_vec();
    let mut missing_workspace = workspace_for(&inplace_plan, &inplace_extra, 0_u64);
    let result = inplace_plan.execute_in_place(&mut missing_destination, &mut missing_workspace);
    assert!(result.is_err());
    assert!(
        missing_destination
            .active_pencil()
            .unwrap()
            .same_layout(inplace_source_pencil.as_ref())
    );
    assert_eq!(
        missing_destination.active_view().unwrap().as_slice(),
        missing_before
    );
    world.barrier();

    // Poisoned is reported collectively without needing an inactive view.
    let mut poisoned = many_from_source_2d(
        Arc::clone(&inplace_source_pencil),
        Arc::clone(&inplace_destination_pencil),
        inplace_extra.clone(),
        &inplace_source,
        103,
    );
    let mut poisoned_expected = poisoned.active_view().unwrap().as_slice().to_vec();
    if let Some(first) = poisoned_expected.first_mut() {
        *first = 777;
    }
    if rank == 0 {
        let writer_result = poisoned.overwrite_with(inplace_source_pencil.as_ref(), |mut view| {
            if let Some(first) = view.as_mut_slice().first_mut() {
                *first = 777;
            }
            Err::<(), _>("poison")
        });
        assert!(matches!(
            writer_result,
            Err(pencil_array::OverwriteError::Writer("poison"))
        ));
    }
    let mut poisoned_workspace = workspace_for(&inplace_plan, &inplace_extra, 0_u64);
    let result = inplace_plan.execute_in_place(&mut poisoned, &mut poisoned_workspace);
    assert!(result.is_err());
    if rank == 0 {
        assert_eq!(poisoned.active_view().unwrap_err(), ArrayError::Poisoned);
        poisoned
            .overwrite_with(inplace_source_pencil.as_ref(), |mut view| {
                assert_eq!(view.as_slice(), poisoned_expected.as_slice());
                view.as_mut_slice().copy_from_slice(&poisoned_expected);
                Ok::<_, ()>(())
            })
            .unwrap();
    } else {
        assert!(
            poisoned
                .active_pencil()
                .unwrap()
                .same_layout(inplace_source_pencil.as_ref())
        );
        assert_eq!(
            poisoned.active_view().unwrap().as_slice(),
            inplace_source.as_slice()
        );
    }
    world.barrier();

    // Both initialized workspace prefixes are checked before communication.
    assert!(inplace_requirements.send_len > 0);
    assert!(inplace_requirements.receive_len > 0);
    for short_send in [true, false] {
        let mut insufficient = many_from_source_2d(
            Arc::clone(&inplace_source_pencil),
            Arc::clone(&inplace_destination_pencil),
            inplace_extra.clone(),
            &inplace_source,
            104,
        );
        let before = insufficient.active_view().unwrap().as_slice().to_vec();
        let mut workspace = if rank == 0 {
            if short_send {
                AllToAllvTransposeWorkspace::from_vecs(
                    Vec::new(),
                    vec![0_u64; inplace_requirements.receive_len],
                )
            } else {
                AllToAllvTransposeWorkspace::from_vecs(
                    vec![0_u64; inplace_requirements.send_len],
                    Vec::new(),
                )
            }
        } else {
            workspace_for(&inplace_plan, &inplace_extra, 0_u64)
        };
        let result = inplace_plan.execute_in_place(&mut insufficient, &mut workspace);
        assert!(result.is_err());
        assert!(
            insufficient
                .active_pencil()
                .unwrap()
                .same_layout(inplace_source_pencil.as_ref())
        );
        assert_eq!(insufficient.active_view().unwrap().as_slice(), before);
        world.barrier();
    }

    // A rank-local extra-shape disagreement is rejected by the exact descriptor.
    let alternate_inplace_extra = ExtraShape::new([3, 2]).unwrap();
    let local_extra = if rank == 0 {
        alternate_inplace_extra.clone()
    } else {
        inplace_extra.clone()
    };
    let mut extra_mismatch = many_2d_with_value(
        Arc::clone(&inplace_source_pencil),
        Arc::clone(&inplace_destination_pencil),
        0,
        local_extra.clone(),
        105,
    );
    let extra_mismatch_before = extra_mismatch.active_view().unwrap().as_slice().to_vec();
    let local_requirements = inplace_plan.workspace_requirements(&local_extra).unwrap();
    let mut extra_workspace = AllToAllvTransposeWorkspace::from_vecs(
        vec![0_u64; local_requirements.send_len],
        vec![0_u64; local_requirements.receive_len],
    );
    let result = inplace_plan.execute_in_place(&mut extra_mismatch, &mut extra_workspace);
    if size == 1 {
        assert!(result.is_ok());
        assert!(
            extra_mismatch
                .active_pencil()
                .unwrap()
                .same_layout(inplace_destination_pencil.as_ref())
        );
    } else {
        assert!(result.is_err());
        assert!(
            extra_mismatch
                .active_pencil()
                .unwrap()
                .same_layout(inplace_source_pencil.as_ref())
        );
        assert_eq!(
            extra_mismatch.active_view().unwrap().as_slice(),
            extra_mismatch_before
        );
    }
    world.barrier();

    // A constructor call and in-place execution cannot share this collective.
    let mut mixed_new = many_from_source_2d(
        Arc::clone(&inplace_source_pencil),
        Arc::clone(&inplace_destination_pencil),
        inplace_extra.clone(),
        &inplace_source,
        106,
    );
    let mixed_new_before = mixed_new.active_view().unwrap().as_slice().to_vec();
    if size == 1 {
        let mut workspace = workspace_for(&inplace_plan, &inplace_extra, 0_u64);
        inplace_plan
            .execute_in_place(&mut mixed_new, &mut workspace)
            .unwrap();
    } else {
        let result = if rank == 0 {
            AllToAllvTransposePlan::new(
                Arc::clone(&inplace_source_pencil),
                Arc::clone(&inplace_destination_pencil),
            )
            .map(|_| ())
        } else {
            let mut workspace = workspace_for(&inplace_plan, &inplace_extra, 0_u64);
            inplace_plan.execute_in_place(&mut mixed_new, &mut workspace)
        };
        assert!(result.is_err());
        assert_eq!(
            mixed_new.active_view().unwrap().as_slice(),
            mixed_new_before
        );
    }
    world.barrier();

    // The other execution API is rejected at the same fixed-header boundary.
    if size == 1 {
        let mut mixed_views = many_from_source_2d(
            Arc::clone(&inplace_source_pencil),
            Arc::clone(&inplace_destination_pencil),
            inplace_extra.clone(),
            &inplace_source,
            107,
        );
        let mut workspace = workspace_for(&inplace_plan, &inplace_extra, 0_u64);
        inplace_plan
            .execute_in_place(&mut mixed_views, &mut workspace)
            .unwrap();
    } else if rank == 0 {
        let mut destination = PencilArray::from_elem(
            Arc::clone(&inplace_destination_pencil),
            inplace_extra.clone(),
            107_u64,
        )
        .unwrap();
        let before = destination.as_slice().to_vec();
        let mut workspace = workspace_for(&inplace_plan, &inplace_extra, 0_u64);
        let result = inplace_plan.execute_views(
            inplace_source.view(),
            destination.view_mut(),
            &mut workspace,
        );
        assert!(result.is_err());
        assert_eq!(destination.as_slice(), before.as_slice());
    } else {
        let mut mixed_views = many_from_source_2d(
            Arc::clone(&inplace_source_pencil),
            Arc::clone(&inplace_destination_pencil),
            inplace_extra.clone(),
            &inplace_source,
            107,
        );
        let before = mixed_views.active_view().unwrap().as_slice().to_vec();
        let mut workspace = workspace_for(&inplace_plan, &inplace_extra, 0_u64);
        let result = inplace_plan.execute_in_place(&mut mixed_views, &mut workspace);
        assert!(result.is_err());
        assert_eq!(mixed_views.active_view().unwrap().as_slice(), before);
    }
    world.barrier();

    // A rank choosing the opposite direction is caught by the descriptor.
    let reverse_inplace_plan = AllToAllvTransposePlan::new(
        Arc::clone(&inplace_destination_pencil),
        Arc::clone(&inplace_source_pencil),
    )
    .unwrap();
    let reverse_requirements = reverse_inplace_plan
        .workspace_requirements(&inplace_extra)
        .unwrap();
    let mut direction_mismatch = many_from_source_2d(
        Arc::clone(&inplace_source_pencil),
        Arc::clone(&inplace_destination_pencil),
        inplace_extra.clone(),
        &inplace_source,
        108,
    );
    let direction_before = direction_mismatch
        .active_view()
        .unwrap()
        .as_slice()
        .to_vec();
    let mut direction_workspace = AllToAllvTransposeWorkspace::from_vecs(
        vec![
            0_u64;
            inplace_requirements
                .send_len
                .max(reverse_requirements.send_len)
        ],
        vec![
            0_u64;
            inplace_requirements
                .receive_len
                .max(reverse_requirements.receive_len)
        ],
    );
    let result = if rank == 0 {
        reverse_inplace_plan.execute_in_place(&mut direction_mismatch, &mut direction_workspace)
    } else {
        inplace_plan.execute_in_place(&mut direction_mismatch, &mut direction_workspace)
    };
    assert!(result.is_err());
    assert!(
        direction_mismatch
            .active_pencil()
            .unwrap()
            .same_layout(inplace_source_pencil.as_ref())
    );
    assert_eq!(
        direction_mismatch.active_view().unwrap().as_slice(),
        direction_before
    );
    world.barrier();

    // A normal preflight failure does not consume the plan or workspace.
    let active = usize::from(rank == 0);
    let mut reusable = many_2d_with_value(
        Arc::clone(&inplace_source_pencil),
        Arc::clone(&inplace_destination_pencil),
        active,
        inplace_extra.clone(),
        109,
    );
    if active == 0 {
        reusable
            .active_view_mut()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(inplace_source.as_slice());
    }
    let reusable_before = reusable.active_view().unwrap().as_slice().to_vec();
    let mut reusable_workspace = workspace_for(&inplace_plan, &inplace_extra, 0_u64);
    assert!(
        inplace_plan
            .execute_in_place(&mut reusable, &mut reusable_workspace)
            .is_err()
    );
    assert!(
        reusable
            .active_pencil()
            .unwrap()
            .same_layout(expected_active.as_ref())
    );
    assert_eq!(reusable.active_view().unwrap().as_slice(), reusable_before);
    if rank == 0 {
        reusable
            .overwrite_with(inplace_source_pencil.as_ref(), |mut view| {
                view.as_mut_slice()
                    .copy_from_slice(inplace_source.as_slice());
                Ok::<_, ()>(())
            })
            .unwrap();
    }
    inplace_plan
        .execute_in_place(&mut reusable, &mut reusable_workspace)
        .unwrap();
    assert!(
        reusable
            .active_pencil()
            .unwrap()
            .same_layout(inplace_destination_pencil.as_ref())
    );
    assert_eq!(
        reusable.active_view().unwrap().as_slice(),
        expected_inplace.as_slice()
    );
    world.barrier();

    // Zero extra elements still complete the collective and commit the layout.
    let zero_inplace = ExtraShape::new([2, 0]).unwrap();
    let mut zero_many = ManyPencilArray::<u64, 2, 1>::from_vec(
        vec![
            Arc::clone(&inplace_source_pencil),
            Arc::clone(&inplace_destination_pencil),
        ],
        0,
        zero_inplace.clone(),
        Vec::new(),
    )
    .unwrap();
    let mut zero_inplace_workspace =
        AllToAllvTransposeWorkspace::from_vecs(Vec::<u64>::new(), Vec::new());
    inplace_plan
        .execute_in_place(&mut zero_many, &mut zero_inplace_workspace)
        .unwrap();
    assert!(
        zero_many
            .active_pencil()
            .unwrap()
            .same_layout(inplace_destination_pencil.as_ref())
    );
    assert!(zero_many.active_view().unwrap().is_empty());

    let mut giant_many = ManyPencilArray::<u8, 2, 1>::from_vec(
        vec![Arc::clone(&giant_source), Arc::clone(&giant_destination)],
        0,
        zero_extra,
        Vec::new(),
    )
    .unwrap();
    let mut giant_inplace_workspace =
        AllToAllvTransposeWorkspace::from_vecs(Vec::new(), Vec::new());
    giant_plan
        .execute_in_place(&mut giant_many, &mut giant_inplace_workspace)
        .unwrap();
    assert!(
        giant_many
            .active_pencil()
            .unwrap()
            .same_layout(giant_destination.as_ref())
    );
    assert!(giant_many.active_view().unwrap().is_empty());
    world.barrier();
}
