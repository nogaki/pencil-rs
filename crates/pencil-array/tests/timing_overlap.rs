use std::{panic::AssertUnwindSafe, sync::Arc};

use mpi::traits::*;
use pencil_array::{
    AllToAllvTransposePlan, AxisPermutation, ExtraShape, ManyPencilArray, MpiTopology, Pencil,
    PencilArray, PointToPointTransposePlan, TransposeWorkspace,
};

#[test]
fn profiled_alltoallv_and_callback_regression() {
    let universe = mpi::initialize().expect("MPI initialization failed");
    let world = universe.world();
    let size = usize::try_from(world.size()).unwrap();
    assert!(matches!(size, 1 | 4 | 6));
    let topology = MpiTopology::<1>::new(&world, [size]).unwrap();
    let source_pencil = Pencil::<2, 1>::new(Arc::clone(&topology), [7, 9], [0]).unwrap();
    let destination_pencil = Pencil::<2, 1>::new_permuted(
        Arc::clone(&topology),
        [7, 9],
        [1],
        AxisPermutation::new([1, 0]).unwrap(),
    )
    .unwrap();
    let extra = ExtraShape::new([2, 3]).unwrap();
    let mut source =
        PencilArray::from_elem(Arc::clone(&source_pencil), extra.clone(), 0_u64).unwrap();
    for (i, value) in source.as_mut_slice().iter_mut().enumerate() {
        *value = (world.rank() as usize * 10_000 + i) as u64;
    }
    let all =
        AllToAllvTransposePlan::new(Arc::clone(&source_pencil), Arc::clone(&destination_pencil))
            .unwrap();
    let p2p =
        PointToPointTransposePlan::new(Arc::clone(&source_pencil), Arc::clone(&destination_pencil))
            .unwrap();
    let req = all.workspace_requirements(&extra).unwrap();
    let mut all_workspace =
        TransposeWorkspace::from_vecs(vec![0; req.send_len], vec![0; req.receive_len]);
    let mut all_destination =
        PencilArray::from_elem(Arc::clone(&destination_pencil), extra.clone(), u64::MAX).unwrap();
    let timing = all
        .execute_views_with_timing(
            source.view(),
            all_destination.view_mut(),
            &mut all_workspace,
        )
        .unwrap();
    assert!(timing.total >= timing.pack + timing.collective_wait + timing.unpack);
    let req = p2p.workspace_requirements(&extra).unwrap();
    let mut workspace =
        TransposeWorkspace::from_vecs(vec![0; req.send_len], vec![0; req.receive_len]);
    let mut destination =
        PencilArray::from_elem(Arc::clone(&destination_pencil), extra.clone(), u64::MAX).unwrap();
    p2p.execute_views_with_callback(
        source.view(),
        destination.view_mut(),
        &mut workspace,
        |values| {
            assert!(!values.is_empty());
            Ok::<_, &'static str>(())
        },
    )
    .unwrap();
    world.barrier();

    assert_eq!(destination.as_slice(), all_destination.as_slice());
    let source_before = source.as_slice().to_vec();
    let rank = world.rank();
    // Timed entries must preserve the same local/peer error and atomicity contract.
    for point_to_point in [false, true] {
        for in_place in [false, true] {
            for short_on_root in [true, false] {
                let send_len = req.send_len - usize::from(short_on_root && rank == 0);
                let mut checked_workspace =
                    TransposeWorkspace::from_vecs(vec![71; send_len], vec![73; req.receive_len]);
                let mut checked_destination = PencilArray::from_elem(
                    Arc::clone(&destination_pencil),
                    extra.clone(),
                    u64::MAX,
                )
                .unwrap();
                let mut checked_array = ManyPencilArray::from_elem(
                    vec![Arc::clone(&source_pencil), Arc::clone(&destination_pencil)],
                    0,
                    extra.clone(),
                    79_u64,
                )
                .unwrap();
                checked_array
                    .active_view_mut()
                    .unwrap()
                    .as_mut_slice()
                    .copy_from_slice(source.as_slice());
                let workspace_before = format!("{checked_workspace:?}");
                let array_before = format!("{checked_array:?}");
                let destination_before = checked_destination.as_slice().to_vec();
                let result = match (point_to_point, in_place) {
                    (false, false) => all.execute_views_with_timing(
                        source.view(),
                        checked_destination.view_mut(),
                        &mut checked_workspace,
                    ),
                    (true, false) => p2p.execute_views_with_timing(
                        source.view(),
                        checked_destination.view_mut(),
                        &mut checked_workspace,
                    ),
                    (false, true) => {
                        all.execute_in_place_with_timing(&mut checked_array, &mut checked_workspace)
                    }
                    (true, true) => {
                        p2p.execute_in_place_with_timing(&mut checked_array, &mut checked_workspace)
                    }
                };
                if short_on_root {
                    let error = if rank == 0 {
                        pencil_array::TransposeError::WorkspaceTooSmall {
                            send_required: req.send_len,
                            send_len,
                            receive_required: req.receive_len,
                            receive_len: req.receive_len,
                        }
                    } else {
                        pencil_array::TransposeError::CollectivePreconditionFailed
                    };
                    assert_eq!(result, Err(error));
                    assert_eq!(format!("{checked_workspace:?}"), workspace_before);
                    assert_eq!(format!("{checked_array:?}"), array_before);
                    assert_eq!(checked_destination.as_slice(), destination_before);
                } else {
                    let timing = result.unwrap();
                    assert!(timing.total >= timing.pack + timing.collective_wait + timing.unpack);
                    if in_place {
                        assert!(
                            checked_array
                                .active_pencil()
                                .unwrap()
                                .same_layout(destination_pencil.as_ref())
                        );
                        assert_eq!(
                            checked_array.active_view().unwrap().as_slice(),
                            all_destination.as_slice()
                        );
                    } else {
                        assert_eq!(checked_destination.as_slice(), all_destination.as_slice());
                    }
                }
                assert_eq!(source.as_slice(), source_before);
            }
        }
    }
    if size > 1 {
        for callback in [false, true] {
            let before = destination.as_slice().to_vec();
            let rejected = if rank == 0 {
                if callback {
                    p2p.execute_views_with_callback(
                        source.view(),
                        destination.view_mut(),
                        &mut workspace,
                        |_| Ok::<_, ()>(()),
                    )
                    .is_err()
                } else {
                    p2p.execute_views(source.view(), destination.view_mut(), &mut workspace)
                        .is_err()
                }
            } else {
                p2p.execute_views_with_timing(source.view(), destination.view_mut(), &mut workspace)
                    .is_err()
            };
            assert!(rejected);
            assert_eq!(destination.as_slice(), before);
            assert_eq!(source.as_slice(), source_before);
        }
    }
    let before = destination.as_slice().to_vec();
    let mut short = TransposeWorkspace::from_vecs(Vec::new(), Vec::new());
    let mut calls = 0;
    let result = p2p.execute_views_with_callback(
        source.view(),
        destination.view_mut(),
        if rank == 0 {
            &mut short
        } else {
            &mut workspace
        },
        |_| {
            calls += 1;
            Ok::<_, ()>(())
        },
    );
    if rank == 0 {
        assert!(matches!(
            result,
            Err(pencil_array::OverlapError::Transpose(
                pencil_array::TransposeError::WorkspaceTooSmall { .. }
            ))
        ));
    } else {
        assert!(matches!(
            result,
            Err(pencil_array::OverlapError::CollectivePreconditionFailed)
        ));
    }
    assert_eq!(calls, 0);
    assert_eq!(destination.as_slice(), before);
    assert_eq!(source.as_slice(), source_before);
    let result = p2p.execute_views_with_callback(
        source.view(),
        destination.view_mut(),
        &mut workspace,
        |_values| {
            if rank == 0 {
                Err("callback error")
            } else {
                Ok(())
            }
        },
    );
    assert!(result.is_err());
    world.barrier();

    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        p2p.execute_views_with_callback(
            source.view(),
            destination.view_mut(),
            &mut workspace,
            |_values| {
                if rank == 0 {
                    panic!("callback panic")
                } else {
                    Ok::<(), &'static str>(())
                }
            },
        )
    }));
    if rank == 0 {
        assert!(result.is_err());
    } else {
        assert!(matches!(
            result,
            Ok(Err(pencil_array::OverlapError::PeerPanicked))
        ));
    }
    world.barrier();

    // The request scopes are drained on failure: the same plan and workspace recover.
    p2p.execute_views_with_callback(
        source.view(),
        destination.view_mut(),
        &mut workspace,
        |_values| Ok::<_, &'static str>(()),
    )
    .unwrap();
    assert_eq!(destination.as_slice(), all_destination.as_slice());
    assert_eq!(source.as_slice(), source_before);
    world.barrier();

    let in_place = ManyPencilArray::from_elem(
        vec![Arc::clone(&source_pencil), Arc::clone(&destination_pencil)],
        0,
        extra.clone(),
        7_u64,
    )
    .unwrap();
    let in_place_tail_start = destination_pencil.local_len() * extra.element_count();
    let initial_storage = in_place.into_storage();
    let tail_before = initial_storage[in_place_tail_start..].to_vec();
    let mut in_place = ManyPencilArray::from_elem(
        vec![Arc::clone(&source_pencil), Arc::clone(&destination_pencil)],
        0,
        extra.clone(),
        7_u64,
    )
    .unwrap();
    let req = p2p.workspace_requirements(&extra).unwrap();
    let mut in_place_workspace =
        TransposeWorkspace::from_vecs(vec![0; req.send_len], vec![0; req.receive_len]);
    let before = in_place.active_view().unwrap().as_slice().to_vec();
    let mut calls = 0;
    let result = p2p.execute_in_place_with_callback(
        &mut in_place,
        if rank == 0 {
            &mut short
        } else {
            &mut in_place_workspace
        },
        |_| {
            calls += 1;
            Ok::<_, ()>(())
        },
    );
    if rank == 0 {
        assert!(matches!(
            result,
            Err(pencil_array::OverlapError::Transpose(
                pencil_array::TransposeError::WorkspaceTooSmall { .. }
            ))
        ));
    } else {
        assert!(matches!(
            result,
            Err(pencil_array::OverlapError::CollectivePreconditionFailed)
        ));
    }
    assert_eq!(calls, 0);
    assert!(
        in_place
            .active_pencil()
            .unwrap()
            .same_layout(source_pencil.as_ref())
    );
    assert_eq!(in_place.active_view().unwrap().as_slice(), before);
    p2p.execute_in_place_with_callback(&mut in_place, &mut in_place_workspace, |_values| {
        Ok::<_, &'static str>(())
    })
    .unwrap();
    assert!(
        in_place
            .active_pencil()
            .unwrap()
            .same_layout(destination_pencil.as_ref())
    );
    let storage = in_place.into_storage();
    assert_eq!(&storage[in_place_tail_start..], tail_before.as_slice());

    let mut in_place = ManyPencilArray::from_elem(
        vec![Arc::clone(&source_pencil), Arc::clone(&destination_pencil)],
        0,
        extra.clone(),
        7_u64,
    )
    .unwrap();
    let result =
        p2p.execute_in_place_with_callback(&mut in_place, &mut in_place_workspace, |_values| {
            if rank == 0 {
                Err("in-place callback error")
            } else {
                Ok(())
            }
        });
    assert!(
        matches!(
            result,
            Err(pencil_array::OverlapError::Callback("in-place callback error"))
                if rank == 0
        ) || matches!(result, Err(pencil_array::OverlapError::PeerCallbackFailed))
    );
    assert!(matches!(
        in_place.active_pencil(),
        Err(pencil_array::ArrayError::Poisoned)
    ));
    world.barrier();

    let mut in_place = ManyPencilArray::from_elem(
        vec![Arc::clone(&source_pencil), Arc::clone(&destination_pencil)],
        0,
        extra,
        7_u64,
    )
    .unwrap();
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        p2p.execute_in_place_with_callback(&mut in_place, &mut in_place_workspace, |_values| {
            if rank == 0 {
                panic!("in-place callback panic")
            } else {
                Ok::<(), &'static str>(())
            }
        })
    }));
    if rank == 0 {
        assert!(result.is_err());
    } else {
        assert!(matches!(
            result,
            Ok(Err(pencil_array::OverlapError::PeerPanicked))
        ));
    }
    assert!(matches!(
        in_place.active_pencil(),
        Err(pencil_array::ArrayError::Poisoned)
    ));
    world.barrier();
}
