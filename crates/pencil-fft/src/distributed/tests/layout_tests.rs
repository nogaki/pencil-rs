use std::{panic::AssertUnwindSafe, sync::Arc};

use super::{
    C2C_CALLBACK_INJECTION, C2cCallbackInjection, C2cInPlaceArray, C2cInPlaceWorkspace, C2cPlan,
    C2cState, C2cTransition, Complex, Direction, DistributedLayout, ENDPOINT_BATCH_CHUNK,
    ExtraShape, FftError, FftOverlapError, LocalC2cError, LocalC2cPlan, OPERATION_BACKWARD,
    OPERATION_BACKWARD_IN_PLACE, OPERATION_DHT_BACKWARD, OPERATION_DHT_BACKWARD_IN_PLACE,
    OPERATION_DHT_FORWARD, OPERATION_DHT_FORWARD_IN_PLACE, OPERATION_DHT_INVERSE,
    OPERATION_DHT_INVERSE_IN_PLACE, OPERATION_DHT_PLAN, OPERATION_FORWARD,
    OPERATION_FORWARD_IN_PLACE, OPERATION_INVERSE, OPERATION_INVERSE_IN_PLACE, OPERATION_PLAN,
    OPERATION_R2C_BACKWARD, OPERATION_R2C_BACKWARD_IN_PLACE, OPERATION_R2C_FORWARD,
    OPERATION_R2C_FORWARD_IN_PLACE, OPERATION_R2C_INVERSE, OPERATION_R2C_INVERSE_IN_PLACE,
    OPERATION_R2C_PLAN, OPERATION_R2R_BACKWARD, OPERATION_R2R_BACKWARD_IN_PLACE,
    OPERATION_R2R_FORWARD, OPERATION_R2R_FORWARD_IN_PLACE, OPERATION_R2R_INVERSE,
    OPERATION_R2R_INVERSE_IN_PLACE, OPERATION_R2R_PLAN, OverlapError, R2C_CALLBACK_INJECTION,
    R2cCallbackInjection, R2cError, R2cPlan, TransposeMethod, descriptor_len, endpoint_chunk_count,
    endpoint_chunk_len, run_in_place_transaction,
};
use crate::{
    AxisR2rKind, AxisSelection, AxisTransform, MixedC2cPlan, MixedError, MixedR2cPlan, R2cState,
};
use mpi::{collective::CommunicatorCollectives, topology::Communicator};
use pencil_array::{ArrayError, MpiTopology, Pencil, TransposeWorkspace};

#[test]
fn endpoint_chunks_bound_batch_statistics() {
    let chunk = ENDPOINT_BATCH_CHUNK;
    assert_eq!(chunk, 1024);
    for (batch_count, chunks, last) in [
        (0, 0, 0),
        (1, 1, 1),
        (chunk - 1, 1, chunk - 1),
        (chunk, 1, chunk),
        (chunk + 1, 2, 1),
        (2 * chunk + 1, 3, 1),
    ] {
        assert_eq!(endpoint_chunk_count(batch_count), chunks);
        if chunks > 0 {
            assert_eq!(endpoint_chunk_len(batch_count, 0), chunk.min(batch_count));
            assert_eq!(endpoint_chunk_len(batch_count, (chunks - 1) * chunk), last);
        }
    }
}

#[test]
fn axis_selection_validates_and_ignores_input_order() {
    let selection = AxisSelection::<4>::from_indices([3, 0]).unwrap();
    assert!(selection.contains(0));
    assert!(selection.contains(3));
    assert!(!selection.contains(1));
    assert_eq!(selection, AxisSelection::<4>::from_indices([0, 3]).unwrap());
    assert!(AxisSelection::<4>::from_indices([3, 3]).is_err());
    assert!(AxisSelection::<4>::from_indices([4]).is_err());
    assert!(AxisSelection::<4>::empty().is_empty());
    assert!(AxisSelection::<4>::all().is_all());
}

#[test]
fn protocol_words_and_minimal_descriptor_length_are_stable() {
    assert_eq!(
        (OPERATION_PLAN, OPERATION_FORWARD, OPERATION_INVERSE),
        (7, 8, 9)
    );
    let scalar_len = descriptor_len::<2, 1>(&ExtraShape::scalar()).unwrap();
    let batched_len = descriptor_len::<4, 2>(&ExtraShape::new([2, 3]).unwrap()).unwrap();
    assert_eq!(scalar_len, 10);
    assert_eq!(batched_len, 17);
    assert_eq!(
        batched_len - descriptor_len::<4, 2>(&ExtraShape::scalar()).unwrap(),
        2,
    );
    assert_eq!(
        (
            TransposeMethod::AllToAllv.descriptor_word(),
            TransposeMethod::PointToPoint.descriptor_word(),
        ),
        (0, 1),
    );
}

#[test]
fn in_place_operation_words_are_reserved_after_out_of_place_words() {
    assert_eq!(
        (
            OPERATION_PLAN,
            OPERATION_FORWARD,
            OPERATION_INVERSE,
            OPERATION_FORWARD_IN_PLACE,
            OPERATION_INVERSE_IN_PLACE,
        ),
        (7, 8, 9, 10, 11)
    );
    assert_eq!(
        (
            OPERATION_R2C_PLAN,
            OPERATION_R2C_FORWARD,
            OPERATION_R2C_INVERSE,
            OPERATION_BACKWARD,
            OPERATION_BACKWARD_IN_PLACE,
            OPERATION_R2C_BACKWARD,
        ),
        (12, 13, 14, 15, 16, 17)
    );
    assert_eq!(
        (
            OPERATION_R2R_PLAN,
            OPERATION_R2R_FORWARD,
            OPERATION_R2R_INVERSE,
            OPERATION_R2R_BACKWARD,
            OPERATION_R2R_FORWARD_IN_PLACE,
            OPERATION_R2R_INVERSE_IN_PLACE,
            OPERATION_R2R_BACKWARD_IN_PLACE,
        ),
        (18, 19, 20, 21, 22, 23, 24)
    );
    assert_eq!(
        (
            OPERATION_R2C_FORWARD_IN_PLACE,
            OPERATION_R2C_INVERSE_IN_PLACE,
            OPERATION_R2C_BACKWARD_IN_PLACE,
        ),
        (25, 26, 27)
    );
    assert_eq!(
        (
            OPERATION_DHT_PLAN,
            OPERATION_DHT_FORWARD,
            OPERATION_DHT_INVERSE,
            OPERATION_DHT_BACKWARD,
            OPERATION_DHT_FORWARD_IN_PLACE,
            OPERATION_DHT_INVERSE_IN_PLACE,
            OPERATION_DHT_BACKWARD_IN_PLACE,
        ),
        (28, 29, 30, 31, 32, 33, 34)
    );
}

#[test]
fn in_place_transaction_poison_survives_error_and_panic() {
    run_in_place_transaction_cases(false);
}

#[test]
#[ignore = "MPI can be initialized only once per unit-test process; run this test explicitly"]
fn c2c_in_place_transaction_poison_survives_error_and_panic() {
    run_in_place_transaction_cases(true);
}

fn run_in_place_transaction_cases(c2c_only: bool) {
    let _mpi_test_lock = super::MPI_TEST_LOCK
        .get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap();
    let universe = mpi::initialize().expect("MPI initialization failed");
    let world = universe.world();
    {
        let topology = MpiTopology::<1>::new(&world, [world.size() as usize]).unwrap();
        if std::env::var_os("PENCIL_COLLECTION_PANIC_SUBPROCESS").is_some() {
            super::collections::panic_regression(&topology);
        }
        super::collections::regression_cases(&topology);
        let plan = C2cPlan::<f64, 2, 1>::from_shape_with_method(
            Arc::clone(&topology),
            [2, 3],
            ExtraShape::scalar(),
            TransposeMethod::PointToPoint,
        )
        .unwrap();
        let source = plan.allocate_input().unwrap();
        let before = source.as_slice().to_vec();
        for injection in [C2cCallbackInjection::Error, C2cCallbackInjection::Panic] {
            let mut destination = plan.allocate_output().unwrap();
            let mut workspace = plan.allocate_out_of_place_workspace().unwrap();
            if world.rank() == 0 {
                C2C_CALLBACK_INJECTION.with(|slot| slot.set(Some(injection)));
            }
            let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                plan.forward_with_overlap(&source, &mut destination, &mut workspace)
            }));
            if matches!(injection, C2cCallbackInjection::Panic) && world.rank() == 0 {
                assert!(result.is_err());
            } else {
                let error = result.unwrap().unwrap_err();
                match (injection, world.rank() == 0, error) {
                    (
                        C2cCallbackInjection::Error,
                        true,
                        FftOverlapError::Overlap(OverlapError::Callback(
                            FftError::PreparationFailed,
                        )),
                    ) => {}
                    (C2cCallbackInjection::Error, false, FftOverlapError::Overlap(error)) => {
                        assert!(matches!(error, OverlapError::PeerCallbackFailed))
                    }
                    (C2cCallbackInjection::Panic, false, FftOverlapError::Overlap(error)) => {
                        assert!(matches!(error, OverlapError::PeerPanicked))
                    }
                    _ => panic!("unexpected callback error"),
                }
            }
            assert!(matches!(
                workspace.intermediate.active_view(),
                Err(ArrayError::Poisoned)
            ));
            assert_eq!(source.as_slice(), before);
            world.barrier(); // A peer must not be stuck in another outer FFT agreement.
            let mut fresh = plan.allocate_out_of_place_workspace().unwrap();
            plan.forward_with_overlap(&source, &mut destination, &mut fresh)
                .unwrap();
        }
        let plan = C2cPlan::<f64, 3, 1>::from_shape_with_layout(
            Arc::clone(&topology),
            [2, 3, 4],
            ExtraShape::scalar(),
            DistributedLayout {
                transpose_method: TransposeMethod::PointToPoint,
                permute_dims: false,
            },
        )
        .unwrap();
        let mut source = plan.allocate_input().unwrap();
        source.as_mut_slice().fill(Complex::new(1.0, 0.0));
        let mut destination = plan.allocate_output().unwrap();
        let mut workspace = plan.allocate_out_of_place_workspace().unwrap();
        plan.forward_with_overlap(&source, &mut destination, &mut workspace)
            .unwrap();
        let ranges = destination.pencil().local_ranges();
        let shape = destination.local_spatial_shape();
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    let dc = i + ranges[0].start == 0
                        && j + ranges[1].start == 0
                        && k + ranges[2].start == 0;
                    let expected = if dc { 24.0 } else { 0.0 };
                    let actual = destination.get_local(&[], [i, j, k]).unwrap();
                    assert!((actual.re - expected).abs() < 1e-10 && actual.im.abs() < 1e-10);
                }
            }
        }
        let mut recovered = plan.allocate_input().unwrap();
        plan.inverse_with_overlap(&destination, &mut recovered, &mut workspace)
            .unwrap();
        for value in recovered.as_slice() {
            assert!((value.re - 1.0).abs() < 1e-10 && value.im.abs() < 1e-10);
        }
    }
    if world.size() != 1 {
        if !c2c_only {
            super::r2c::run_rank_specific_conversion_failure(&world);
        }
        return;
    }
    let result = {
        let topology = MpiTopology::<1>::new(&world, [1]).unwrap();
        let legacy_shape =
            C2cPlan::<f64, 2, 1>::from_shape(Arc::clone(&topology), [2, 3], ExtraShape::scalar())
                .unwrap();
        let legacy_source = legacy_shape.allocate_input().unwrap();
        let legacy_pencil = C2cPlan::<f64, 2, 1>::from_pencil(
            Arc::clone(legacy_shape.input_pencil()),
            ExtraShape::scalar(),
        )
        .unwrap();
        let legacy_array = C2cPlan::<f64, 2, 1>::from_array(&legacy_source).unwrap();
        for legacy_plan in [&legacy_shape, &legacy_pencil, &legacy_array] {
            assert_transition_method(legacy_plan, TransposeMethod::AllToAllv);
        }
        for method in [TransposeMethod::AllToAllv, TransposeMethod::PointToPoint] {
            let plan = C2cPlan::<f64, 2, 1>::from_shape_with_method(
                Arc::clone(&topology),
                [2, 3],
                ExtraShape::scalar(),
                method,
            )
            .unwrap();
            let explicit_pencil = C2cPlan::<f64, 2, 1>::from_pencil_with_method(
                Arc::clone(plan.input_pencil()),
                ExtraShape::scalar(),
                method,
            )
            .unwrap();
            let explicit_array =
                C2cPlan::<f64, 2, 1>::from_array_with_method(&legacy_source, method).unwrap();
            for explicit_plan in [&plan, &explicit_pencil, &explicit_array] {
                assert_transition_method(explicit_plan, method);
            }

            let empty_plan = C2cPlan::<f64, 2, 1>::from_shape_with_selection_and_method(
                Arc::clone(&topology),
                [2, 3],
                ExtraShape::scalar(),
                AxisSelection::empty(),
                method,
            )
            .unwrap();
            let mut empty_array = empty_plan.allocate_in_place().unwrap();
            let mut empty_workspace = empty_plan.allocate_in_place_workspace().unwrap();
            let empty_result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                run_in_place_transaction(
                    &mut empty_array,
                    &mut empty_workspace,
                    C2cState::Output,
                    |array, workspace| {
                        assert_eq!(array.state(), C2cState::Poisoned);
                        array.array.active_view_mut().unwrap().as_mut_slice()[0].re = 7.0;
                        if let Some(value) = workspace.fft_scratch.first_mut() {
                            value.re = 9.0;
                        }
                        Err(FftError::PreparationFailed)
                    },
                )
            }));
            assert!(matches!(empty_result, Ok(Err(FftError::PreparationFailed))));
            assert_poisoned_views_and_retries(&empty_plan, &mut empty_array, &mut empty_workspace);

            let mut short_array = plan.allocate_in_place().unwrap();
            let mut short_workspace = C2cInPlaceWorkspace {
                core: Arc::clone(&plan.core),
                transpose: TransposeWorkspace::from_vecs(Vec::new(), Vec::new()),
                fft_scratch: Vec::new(),
                line_buffer: Vec::new(),
            };
            let short_array_before = format!("{short_array:?}");
            let short_workspace_before = format!("{short_workspace:?}");
            assert!(matches!(
                plan.forward_in_place(&mut short_array, &mut short_workspace),
                Err(FftError::WorkspaceTooSmall { .. })
            ));
            assert_eq!(short_array.state(), C2cState::Input);
            assert_eq!(format!("{short_array:?}"), short_array_before);
            assert_eq!(format!("{short_workspace:?}"), short_workspace_before);

            for direction in [Direction::Forward, Direction::Inverse, Direction::Backward] {
                for panic_failure in [false, true] {
                    let mut array = plan.allocate_in_place().unwrap();
                    let mut workspace = plan.allocate_in_place_workspace().unwrap();
                    if matches!(direction, Direction::Inverse | Direction::Backward) {
                        plan.forward_in_place(&mut array, &mut workspace).unwrap();
                        assert_eq!(array.state(), C2cState::Output);
                    }
                    let target = match direction {
                        Direction::Forward => C2cState::Output,
                        Direction::Inverse | Direction::Backward => C2cState::Input,
                    };
                    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                        run_in_place_transaction(
                            &mut array,
                            &mut workspace,
                            target,
                            |array, workspace| {
                                assert_eq!(array.state(), C2cState::Poisoned);
                                let mut view = array.array.active_view_mut().unwrap();
                                view.as_mut_slice()[0].re = 7.0;
                                if let Some(value) = workspace.fft_scratch.first_mut() {
                                    value.re = 9.0;
                                }
                                if panic_failure {
                                    panic!("in-place test panic after start");
                                }
                                Err(FftError::PreparationFailed)
                            },
                        )
                    }));
                    if panic_failure {
                        assert!(result.is_err());
                    } else {
                        assert!(matches!(result, Ok(Err(FftError::PreparationFailed))));
                    }
                    assert_poisoned_views_and_retries(&plan, &mut array, &mut workspace);
                }
            }

            // Corrupt only the last immutable backend plan. The six-element
            // active buffer passes the first stage and transition, then the
            // replacement length four plan returns NonIntegralBatch.
            let mut corrupted_plan = C2cPlan::<f64, 2, 1>::from_shape_with_method(
                Arc::clone(&topology),
                [2, 3],
                ExtraShape::scalar(),
                method,
            )
            .unwrap();
            Arc::get_mut(&mut corrupted_plan.core)
                .expect("the separate plan core has no other owners")
                .stages
                .last_mut()
                .expect("the C2C route has a final stage")
                .local = super::LocalTransform::Complex(LocalC2cPlan::new(4).unwrap());
            let mut array = corrupted_plan.allocate_in_place().unwrap();
            {
                let mut view = array.view_mut().unwrap();
                for (index, value) in view.as_mut_slice().iter_mut().enumerate() {
                    *value = Complex::new(index as f64 + 1.0, -(index as f64));
                }
            }
            let mut workspace = corrupted_plan.allocate_in_place_workspace().unwrap();
            let result = corrupted_plan.forward_in_place(&mut array, &mut workspace);
            assert!(matches!(
                result,
                Err(FftError::LocalC2c(LocalC2cError::NonIntegralBatch))
            ));
            assert_eq!(array.state(), C2cState::Poisoned);
            assert_poisoned_views_and_retries(&corrupted_plan, &mut array, &mut workspace);

            // For raw backward, corrupt only the final stage in the
            // reverse route. The first raw stage and backward transition
            // run before stage zero returns NonIntegralBatch.
            let mut corrupted_backward_plan = C2cPlan::<f64, 2, 1>::from_shape_with_method(
                Arc::clone(&topology),
                [2, 3],
                ExtraShape::scalar(),
                method,
            )
            .unwrap();
            Arc::get_mut(&mut corrupted_backward_plan.core)
                .expect("the separate backward plan core has no other owners")
                .stages
                .first_mut()
                .expect("the C2C route has an initial stage")
                .local = super::LocalTransform::Complex(LocalC2cPlan::new(4).unwrap());
            let mut backward_array = corrupted_backward_plan.allocate_in_place().unwrap();
            backward_array
                .array
                .overwrite_with(
                    corrupted_backward_plan.output_pencil().as_ref(),
                    |mut view| {
                        for (index, value) in view.as_mut_slice().iter_mut().enumerate() {
                            *value = Complex::new(index as f64 + 1.0, -(index as f64));
                        }
                        Ok::<_, ()>(())
                    },
                )
                .unwrap();
            backward_array.state = C2cState::Output;
            let mut backward_workspace = corrupted_backward_plan
                .allocate_in_place_workspace()
                .unwrap();
            let result = corrupted_backward_plan
                .backward_in_place(&mut backward_array, &mut backward_workspace);
            assert!(matches!(
                result,
                Err(FftError::LocalC2c(LocalC2cError::NonIntegralBatch))
            ));
            assert_eq!(backward_array.state(), C2cState::Poisoned);
            assert_poisoned_views_and_retries(
                &corrupted_backward_plan,
                &mut backward_array,
                &mut backward_workspace,
            );
        }

        if c2c_only {
            return;
        }

        let capacity_plan = R2cPlan::<f64, 2, 1>::from_shape_with_method(
            Arc::clone(&topology),
            [3, 4],
            ExtraShape::scalar(),
            TransposeMethod::AllToAllv,
        )
        .unwrap();
        let mut capacity_array = capacity_plan.allocate_in_place().unwrap();
        capacity_array
            .real_view_mut()
            .unwrap()
            .as_mut_slice()
            .fill(7.0);
        let capacity_snapshot = capacity_array.real_view().unwrap().as_slice().to_vec();
        super::r2c::force_odd_real_capacity(&mut capacity_array);
        let capacity_pointer = capacity_array.real_view().unwrap().as_slice().as_ptr();
        let mut capacity_workspace = capacity_plan.allocate_in_place_workspace().unwrap();
        let workspace_snapshot = format!("{capacity_workspace:?}");
        assert!(matches!(
            capacity_plan.forward_in_place(&mut capacity_array, &mut capacity_workspace),
            Err(R2cError::Fft(FftError::StorageLayoutMismatch))
        ));
        assert_eq!(capacity_array.state(), R2cState::RealInput);
        assert_eq!(
            capacity_array.real_view().unwrap().as_slice(),
            capacity_snapshot.as_slice()
        );
        assert_eq!(
            capacity_array.real_view().unwrap().as_slice().as_ptr(),
            capacity_pointer
        );
        assert_eq!(format!("{capacity_workspace:?}"), workspace_snapshot);

        let mut short_array = capacity_plan.allocate_in_place().unwrap();
        short_array
            .real_view_mut()
            .unwrap()
            .as_mut_slice()
            .fill(8.0);
        let short_data = short_array.real_view().unwrap().as_slice().to_vec();
        let short_pointer = short_array.real_view().unwrap().as_slice().as_ptr();
        let mut short_workspace = capacity_plan.allocate_in_place_workspace().unwrap();
        super::r2c::empty_in_place_workspace_for_test(&mut short_workspace);
        let short_workspace_snapshot = format!("{short_workspace:?}");
        assert!(matches!(
            capacity_plan.forward_in_place(&mut short_array, &mut short_workspace),
            Err(R2cError::Fft(FftError::WorkspaceTooSmall { .. }))
        ));
        assert_eq!(short_array.state(), R2cState::RealInput);
        assert_eq!(short_array.real_view().unwrap().as_slice(), short_data);
        assert_eq!(
            short_array.real_view().unwrap().as_slice().as_ptr(),
            short_pointer
        );
        assert_eq!(format!("{short_workspace:?}"), short_workspace_snapshot);

        r2c_conversion_panic_cases(&topology);
        r2c_conversion_owner_failure_cases(&topology);
        r2c_preflight_corruption_cases(&topology);
        mixed_in_place_error_cases(&topology);

        Ok::<(), ()>(())
    };
    result.unwrap();
}

#[test]
#[ignore = "MPI can be initialized only once per unit-test process; run this test explicitly"]
fn r2c_overlap_callback_error_and_panic_map_per_rank() {
    let _mpi_test_lock = super::MPI_TEST_LOCK
        .get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap();
    let universe = mpi::initialize().expect("MPI initialization failed");
    let world = universe.world();
    let topology = MpiTopology::<1>::new(&world, [world.size() as usize]).unwrap();
    let plan = R2cPlan::<f64, 2, 1>::from_shape_with_method(
        Arc::clone(&topology),
        [2, 3],
        ExtraShape::scalar(),
        TransposeMethod::PointToPoint,
    )
    .unwrap();
    let source = plan.allocate_input().unwrap();
    for injection in [R2cCallbackInjection::Error, R2cCallbackInjection::Panic] {
        let mut destination = plan.allocate_output().unwrap();
        let mut workspace = plan.allocate_workspace().unwrap();
        if world.rank() == 0 {
            R2C_CALLBACK_INJECTION.with(|slot| slot.set(Some(injection)));
        }
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            plan.forward_with_overlap(&source, &mut destination, &mut workspace)
        }));
        if matches!(injection, R2cCallbackInjection::Panic) && world.rank() == 0 {
            assert!(result.is_err());
        } else {
            let error = result.unwrap().unwrap_err();
            match (injection, world.rank() == 0, error) {
                (
                    R2cCallbackInjection::Error,
                    true,
                    FftOverlapError::Overlap(OverlapError::Callback(R2cError::Fft(
                        FftError::LocalC2c(crate::LocalC2cError::ScratchTooSmall {
                            required: 7,
                            actual: 3,
                        }),
                    ))),
                ) => {}
                (R2cCallbackInjection::Error, false, FftOverlapError::Overlap(error)) => {
                    assert!(matches!(error, OverlapError::PeerCallbackFailed))
                }
                (R2cCallbackInjection::Panic, false, FftOverlapError::Overlap(error)) => {
                    assert!(matches!(error, OverlapError::PeerPanicked))
                }
                _ => panic!("unexpected R2C callback error"),
            }
        }
        world.barrier();
        let mut fresh = plan.allocate_workspace().unwrap();
        plan.forward_with_overlap(&source, &mut destination, &mut fresh)
            .unwrap();
    }
}

fn r2c_conversion_panic_cases(topology: &Arc<MpiTopology<1>>) {
    let plan = R2cPlan::<f64, 2, 1>::from_shape_with_method(
        Arc::clone(topology),
        [3, 4],
        ExtraShape::scalar(),
        TransposeMethod::AllToAllv,
    )
    .unwrap();
    let mut forward_array = plan.allocate_in_place().unwrap();
    forward_array
        .real_view_mut()
        .unwrap()
        .as_mut_slice()
        .fill(1.0);
    super::r2c::panic_after_forward_detach_for_test(&mut forward_array);
    let mut forward_workspace = plan.allocate_in_place_workspace().unwrap();
    let panic_result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        plan.forward_in_place(&mut forward_array, &mut forward_workspace)
    }));
    assert!(panic_result.is_err());
    assert!(super::r2c::storage_capacity_bytes_for_test(&forward_array).is_none());
    super::r2c::assert_poisoned_views_and_retries(
        &plan,
        &mut forward_array,
        &mut forward_workspace,
    );

    let mut reverse_array = plan.allocate_in_place().unwrap();
    let mut reverse_workspace = plan.allocate_in_place_workspace().unwrap();
    plan.forward_in_place(&mut reverse_array, &mut reverse_workspace)
        .unwrap();
    super::r2c::panic_after_reverse_detach_for_test(&mut reverse_array);
    let panic_result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        plan.inverse_in_place(&mut reverse_array, &mut reverse_workspace)
    }));
    assert!(panic_result.is_err());
    assert!(super::r2c::storage_capacity_bytes_for_test(&reverse_array).is_none());
    super::r2c::assert_poisoned_views_and_retries(
        &plan,
        &mut reverse_array,
        &mut reverse_workspace,
    );
}

fn r2c_conversion_owner_failure_cases(topology: &Arc<MpiTopology<1>>) {
    let plan = R2cPlan::<f64, 2, 1>::from_shape_with_method(
        Arc::clone(topology),
        [3, 4],
        ExtraShape::scalar(),
        TransposeMethod::AllToAllv,
    )
    .unwrap();
    let mut forward_array = plan.allocate_in_place().unwrap();
    forward_array
        .real_view_mut()
        .unwrap()
        .as_mut_slice()
        .fill(2.0);
    let capacity = super::r2c::storage_capacity_bytes_for_test(&forward_array).unwrap();
    super::r2c::force_odd_forward_cast_for_test(&mut forward_array);
    let mut forward_workspace = plan.allocate_in_place_workspace().unwrap();
    let result = plan.forward_in_place(&mut forward_array, &mut forward_workspace);
    assert!(matches!(
        result,
        Err(R2cError::Fft(FftError::StorageLayoutMismatch))
    ));
    assert_eq!(forward_array.state(), R2cState::Poisoned);
    assert!(super::r2c::storage_is_poisoned_for_test(&forward_array));
    assert_eq!(
        super::r2c::storage_capacity_bytes_for_test(&forward_array),
        Some(capacity)
    );
    super::r2c::assert_poisoned_views_and_retries(
        &plan,
        &mut forward_array,
        &mut forward_workspace,
    );

    let mut reverse_array = plan.allocate_in_place().unwrap();
    let mut reverse_workspace = plan.allocate_in_place_workspace().unwrap();
    plan.forward_in_place(&mut reverse_array, &mut reverse_workspace)
        .unwrap();
    let capacity = super::r2c::storage_capacity_bytes_for_test(&reverse_array).unwrap();
    super::r2c::force_reverse_restore_rejection_for_test(&mut reverse_array);
    let result = plan.inverse_in_place(&mut reverse_array, &mut reverse_workspace);
    assert!(matches!(
        result,
        Err(R2cError::Fft(FftError::Array(
            ArrayError::IncompatiblePencils
        )))
    ));
    assert_eq!(reverse_array.state(), R2cState::Poisoned);
    assert!(super::r2c::storage_is_poisoned_for_test(&reverse_array));
    assert_eq!(
        super::r2c::storage_capacity_bytes_for_test(&reverse_array),
        Some(capacity)
    );
    super::r2c::assert_poisoned_views_and_retries(
        &plan,
        &mut reverse_array,
        &mut reverse_workspace,
    );
}

fn r2c_preflight_corruption_cases(topology: &Arc<MpiTopology<1>>) {
    let extra = ExtraShape::new([2]).unwrap();
    let wrong_extra = ExtraShape::new([1, 2]).unwrap();
    let plan = R2cPlan::<f64, 2, 1>::from_shape_with_method(
        Arc::clone(topology),
        [3, 4],
        extra.clone(),
        TransposeMethod::AllToAllv,
    )
    .unwrap();
    let mut real_extra_array = plan.allocate_in_place().unwrap();
    real_extra_array
        .real_view_mut()
        .unwrap()
        .as_mut_slice()
        .fill(3.0);
    super::r2c::corrupt_real_extra_shape_for_test(&mut real_extra_array, wrong_extra.clone());
    let real_before = real_extra_array.real_view().unwrap().as_slice().to_vec();
    let real_array_before = format!("{real_extra_array:?}");
    let mut real_workspace = plan.allocate_in_place_workspace().unwrap();
    let real_workspace_before = format!("{real_workspace:?}");
    assert!(matches!(
        plan.forward_in_place(&mut real_extra_array, &mut real_workspace),
        Err(R2cError::Fft(FftError::StorageLayoutMismatch))
    ));
    assert_eq!(real_extra_array.state(), R2cState::RealInput);
    assert_eq!(
        real_extra_array.real_view().unwrap().as_slice(),
        real_before.as_slice()
    );
    assert_eq!(format!("{real_extra_array:?}"), real_array_before);
    assert_eq!(format!("{real_workspace:?}"), real_workspace_before);

    let registry_plan = R2cPlan::<f64, 2, 1>::from_shape_with_method(
        Arc::clone(topology),
        [3, 4],
        ExtraShape::scalar(),
        TransposeMethod::AllToAllv,
    )
    .unwrap();
    let mut real_registry_array = registry_plan.allocate_in_place().unwrap();
    real_registry_array
        .real_view_mut()
        .unwrap()
        .as_mut_slice()
        .fill(4.0);
    super::r2c::corrupt_forward_registry_for_test(&mut real_registry_array);
    let real_before = real_registry_array.real_view().unwrap().as_slice().to_vec();
    let real_array_before = format!("{real_registry_array:?}");
    let mut real_registry_workspace = registry_plan.allocate_in_place_workspace().unwrap();
    let real_workspace_before = format!("{real_registry_workspace:?}");
    assert!(matches!(
        registry_plan.forward_in_place(&mut real_registry_array, &mut real_registry_workspace),
        Err(R2cError::Fft(FftError::WorkspaceMismatch))
    ));
    assert_eq!(real_registry_array.state(), R2cState::RealInput);
    assert_eq!(
        real_registry_array.real_view().unwrap().as_slice(),
        real_before.as_slice()
    );
    assert_eq!(format!("{real_registry_array:?}"), real_array_before);
    assert_eq!(
        format!("{real_registry_workspace:?}"),
        real_workspace_before
    );

    let mut complex_extra_array = plan.allocate_in_place().unwrap();
    let mut complex_extra_workspace = plan.allocate_in_place_workspace().unwrap();
    plan.forward_in_place(&mut complex_extra_array, &mut complex_extra_workspace)
        .unwrap();
    super::r2c::corrupt_complex_extra_shape_for_test(&mut complex_extra_array, wrong_extra);
    let complex_before = complex_extra_array
        .complex_view()
        .unwrap()
        .as_slice()
        .to_vec();
    let complex_array_before = format!("{complex_extra_array:?}");
    let complex_workspace_before = format!("{complex_extra_workspace:?}");
    assert!(matches!(
        plan.inverse_in_place(&mut complex_extra_array, &mut complex_extra_workspace),
        Err(R2cError::Fft(FftError::StorageLayoutMismatch))
    ));
    assert_eq!(complex_extra_array.state(), R2cState::ComplexOutput);
    assert_eq!(
        complex_extra_array.complex_view().unwrap().as_slice(),
        complex_before.as_slice()
    );
    assert_eq!(format!("{complex_extra_array:?}"), complex_array_before);
    assert_eq!(
        format!("{complex_extra_workspace:?}"),
        complex_workspace_before
    );

    let mut complex_registry_array = registry_plan.allocate_in_place().unwrap();
    let mut complex_registry_workspace = registry_plan.allocate_in_place_workspace().unwrap();
    registry_plan
        .forward_in_place(&mut complex_registry_array, &mut complex_registry_workspace)
        .unwrap();
    super::r2c::corrupt_reverse_registry_for_test(&mut complex_registry_array);
    let complex_before = complex_registry_array
        .complex_view()
        .unwrap()
        .as_slice()
        .to_vec();
    let complex_array_before = format!("{complex_registry_array:?}");
    let complex_workspace_before = format!("{complex_registry_workspace:?}");
    assert!(matches!(
        registry_plan
            .inverse_in_place(&mut complex_registry_array, &mut complex_registry_workspace),
        Err(R2cError::Fft(FftError::WorkspaceMismatch))
    ));
    assert_eq!(complex_registry_array.state(), R2cState::ComplexOutput);
    assert_eq!(
        complex_registry_array.complex_view().unwrap().as_slice(),
        complex_before.as_slice()
    );
    assert_eq!(format!("{complex_registry_array:?}"), complex_array_before);
    assert_eq!(
        format!("{complex_registry_workspace:?}"),
        complex_workspace_before
    );
}

fn mixed_in_place_error_cases(topology: &Arc<MpiTopology<1>>) {
    let c2c_transforms = [AxisTransform::Fft, AxisTransform::R2r(AxisR2rKind::Dht)];
    let mut c2c_allocation_plan = MixedC2cPlan::<f64, 2, 1>::from_shape(
        Arc::clone(topology),
        [2, 3],
        ExtraShape::scalar(),
        c2c_transforms,
    )
    .unwrap();
    super::mixed::force_c2c_workspace_allocation_failure_for_test(&mut c2c_allocation_plan);
    assert!(matches!(
        c2c_allocation_plan.allocate_workspace(),
        Err(MixedError::Fft(FftError::AllocationFailed { .. }))
    ));
    assert!(matches!(
        c2c_allocation_plan.allocate_in_place_workspace(),
        Err(MixedError::Fft(FftError::AllocationFailed { .. }))
    ));

    let c2c_plan = MixedC2cPlan::<f64, 2, 1>::from_shape(
        Arc::clone(topology),
        [2, 3],
        ExtraShape::scalar(),
        c2c_transforms,
    )
    .unwrap();
    let mut c2c_source = c2c_plan.allocate_input().unwrap();
    c2c_source.as_mut_slice().fill(1.0.into());
    let c2c_source_before = c2c_source.as_slice().to_vec();
    let mut c2c_output = c2c_plan.allocate_output().unwrap();
    c2c_output.as_mut_slice().fill(2.0.into());
    let c2c_output_before = c2c_output.as_slice().to_vec();
    let mut c2c_workspace = c2c_plan.allocate_workspace().unwrap();
    super::mixed::empty_c2c_workspace_for_test(&mut c2c_workspace);
    let c2c_workspace_before = format!("{c2c_workspace:?}");
    assert!(matches!(
        c2c_plan.forward(&c2c_source, &mut c2c_output, &mut c2c_workspace),
        Err(MixedError::Fft(FftError::WorkspaceTooSmall { .. }))
    ));
    assert_eq!(c2c_source.as_slice(), c2c_source_before.as_slice());
    assert_eq!(c2c_output.as_slice(), c2c_output_before.as_slice());
    assert_eq!(format!("{c2c_workspace:?}"), c2c_workspace_before);

    let mut c2c_array = c2c_plan.allocate_in_place().unwrap();
    let mut c2c_ip_workspace = c2c_plan.allocate_in_place_workspace().unwrap();
    super::mixed::empty_c2c_in_place_workspace_for_test(&mut c2c_ip_workspace);
    let c2c_array_before = format!("{c2c_array:?}");
    let c2c_ip_workspace_before = format!("{c2c_ip_workspace:?}");
    assert!(matches!(
        c2c_plan.forward_in_place(&mut c2c_array, &mut c2c_ip_workspace),
        Err(MixedError::Fft(FftError::WorkspaceTooSmall { .. }))
    ));
    assert_eq!(c2c_array.state(), C2cState::Input);
    assert_eq!(format!("{c2c_array:?}"), c2c_array_before);
    assert_eq!(format!("{c2c_ip_workspace:?}"), c2c_ip_workspace_before);

    let mut c2c_panic_array = c2c_plan.allocate_in_place().unwrap();
    let mut c2c_panic_workspace = c2c_plan.allocate_in_place_workspace().unwrap();
    super::mixed::panic_after_start_for_c2c_test(&mut c2c_panic_array);
    assert!(
        std::panic::catch_unwind(AssertUnwindSafe(|| {
            c2c_plan.forward_in_place(&mut c2c_panic_array, &mut c2c_panic_workspace)
        }))
        .is_err()
    );
    assert_eq!(c2c_panic_array.state(), C2cState::Poisoned);
    assert!(matches!(
        c2c_panic_array.view(),
        Err(MixedError::Fft(FftError::Array(ArrayError::Poisoned)))
    ));
    assert!(matches!(
        c2c_plan.forward_in_place(&mut c2c_panic_array, &mut c2c_panic_workspace),
        Err(MixedError::Fft(FftError::Array(ArrayError::Poisoned)))
    ));

    let r2c_transforms = [AxisTransform::Fft, AxisTransform::Rfft];
    let mut r2c_allocation_plan = MixedR2cPlan::<f64, 2, 1>::from_shape(
        Arc::clone(topology),
        [2, 3],
        ExtraShape::scalar(),
        r2c_transforms,
    )
    .unwrap();
    super::mixed::force_r2c_workspace_allocation_failure_for_test(&mut r2c_allocation_plan);
    assert!(matches!(
        r2c_allocation_plan.allocate_workspace(),
        Err(MixedError::Fft(FftError::AllocationFailed { .. }))
    ));
    assert!(matches!(
        r2c_allocation_plan.allocate_in_place_workspace(),
        Err(MixedError::Fft(FftError::AllocationFailed { .. }))
    ));

    let r2c_plan = MixedR2cPlan::<f64, 2, 1>::from_shape(
        Arc::clone(topology),
        [2, 3],
        ExtraShape::scalar(),
        r2c_transforms,
    )
    .unwrap();
    let mut r2c_source = r2c_plan.allocate_input().unwrap();
    r2c_source.as_mut_slice().fill(1.0);
    let r2c_source_before = r2c_source.as_slice().to_vec();
    let mut r2c_output = r2c_plan.allocate_output().unwrap();
    r2c_output.as_mut_slice().fill(Complex::new(2.0, -1.0));
    let r2c_output_before = r2c_output.as_slice().to_vec();
    let mut r2c_workspace = r2c_plan.allocate_workspace().unwrap();
    super::mixed::empty_r2c_workspace_for_test(&mut r2c_workspace);
    let r2c_workspace_before = format!("{r2c_workspace:?}");
    assert!(matches!(
        r2c_plan.forward(&r2c_source, &mut r2c_output, &mut r2c_workspace),
        Err(MixedError::Fft(FftError::WorkspaceTooSmall { .. }))
    ));
    assert_eq!(r2c_source.as_slice(), r2c_source_before.as_slice());
    assert_eq!(r2c_output.as_slice(), r2c_output_before.as_slice());
    assert_eq!(format!("{r2c_workspace:?}"), r2c_workspace_before);

    let mut r2c_array = r2c_plan.allocate_in_place().unwrap();
    let mut r2c_ip_workspace = r2c_plan.allocate_in_place_workspace().unwrap();
    super::mixed::empty_r2c_in_place_workspace_for_test(&mut r2c_ip_workspace);
    let r2c_array_before = format!("{r2c_array:?}");
    let r2c_ip_workspace_before = format!("{r2c_ip_workspace:?}");
    assert!(matches!(
        r2c_plan.forward_in_place(&mut r2c_array, &mut r2c_ip_workspace),
        Err(MixedError::Fft(FftError::WorkspaceTooSmall { .. }))
    ));
    assert_eq!(r2c_array.state(), R2cState::RealInput);
    assert_eq!(format!("{r2c_array:?}"), r2c_array_before);
    assert_eq!(format!("{r2c_ip_workspace:?}"), r2c_ip_workspace_before);

    let mut r2c_panic_array = r2c_plan.allocate_in_place().unwrap();
    let mut r2c_panic_workspace = r2c_plan.allocate_in_place_workspace().unwrap();
    super::mixed::panic_after_start_for_r2c_test(&mut r2c_panic_array);
    assert!(
        std::panic::catch_unwind(AssertUnwindSafe(|| {
            r2c_plan.forward_in_place(&mut r2c_panic_array, &mut r2c_panic_workspace)
        }))
        .is_err()
    );
    assert_eq!(r2c_panic_array.state(), R2cState::Poisoned);
    assert!(matches!(
        r2c_panic_array.real_view(),
        Err(MixedError::Fft(FftError::Array(ArrayError::Poisoned)))
    ));
    assert!(matches!(
        r2c_plan.forward_in_place(&mut r2c_panic_array, &mut r2c_panic_workspace),
        Err(MixedError::Fft(FftError::Array(ArrayError::Poisoned)))
    ));

    let mut forward_detach_array = r2c_plan.allocate_in_place().unwrap();
    let mut forward_detach_workspace = r2c_plan.allocate_in_place_workspace().unwrap();
    super::mixed::panic_after_forward_detach_for_r2c_test(&mut forward_detach_array);
    assert!(
        std::panic::catch_unwind(AssertUnwindSafe(|| {
            r2c_plan.forward_in_place(&mut forward_detach_array, &mut forward_detach_workspace)
        }))
        .is_err()
    );
    assert_eq!(forward_detach_array.state(), R2cState::Poisoned);
    assert!(forward_detach_array.storage_capacity_bytes().is_none());
    assert!(matches!(
        forward_detach_array.real_view(),
        Err(MixedError::Fft(FftError::Array(ArrayError::Poisoned)))
    ));

    let mut reverse_detach_array = r2c_plan.allocate_in_place().unwrap();
    let mut reverse_detach_workspace = r2c_plan.allocate_in_place_workspace().unwrap();
    r2c_plan
        .forward_in_place(&mut reverse_detach_array, &mut reverse_detach_workspace)
        .unwrap();
    super::mixed::panic_after_reverse_detach_for_r2c_test(&mut reverse_detach_array);
    assert!(
        std::panic::catch_unwind(AssertUnwindSafe(|| {
            r2c_plan.inverse_in_place(&mut reverse_detach_array, &mut reverse_detach_workspace)
        }))
        .is_err()
    );
    assert_eq!(reverse_detach_array.state(), R2cState::Poisoned);
    assert!(reverse_detach_array.storage_capacity_bytes().is_none());
    assert!(matches!(
        reverse_detach_array.complex_view(),
        Err(MixedError::Fft(FftError::Array(ArrayError::Poisoned)))
    ));
}

fn assert_transition_method(plan: &C2cPlan<f64, 2, 1>, method: TransposeMethod) {
    match method {
        TransposeMethod::AllToAllv => {
            assert!(matches!(
                &plan.core.transitions[0].forward,
                C2cTransition::AllToAllv(_)
            ));
            assert!(matches!(
                &plan.core.transitions[0].backward,
                C2cTransition::AllToAllv(_)
            ));
        }
        TransposeMethod::PointToPoint => {
            assert!(matches!(
                &plan.core.transitions[0].forward,
                C2cTransition::PointToPoint(_)
            ));
            assert!(matches!(
                &plan.core.transitions[0].backward,
                C2cTransition::PointToPoint(_)
            ));
        }
    }
}

fn assert_poisoned_views_and_retries(
    plan: &C2cPlan<f64, 2, 1>,
    array: &mut C2cInPlaceArray<f64, 2, 1>,
    workspace: &mut C2cInPlaceWorkspace<f64, 2, 1>,
) {
    assert_eq!(array.state(), C2cState::Poisoned);
    assert!(matches!(
        array.view(),
        Err(FftError::Array(pencil_array::ArrayError::Poisoned))
    ));
    assert!(matches!(
        array.view_mut(),
        Err(FftError::Array(pencil_array::ArrayError::Poisoned))
    ));
    for direction in [Direction::Forward, Direction::Inverse, Direction::Backward] {
        let array_before = format!("{array:?}");
        let workspace_before = format!("{workspace:?}");
        let result = match direction {
            Direction::Forward => plan.forward_in_place(array, workspace),
            Direction::Inverse => plan.inverse_in_place(array, workspace),
            Direction::Backward => plan.backward_in_place(array, workspace),
        };
        assert!(matches!(
            result,
            Err(FftError::Array(pencil_array::ArrayError::Poisoned))
        ));
        assert_eq!(array.state(), C2cState::Poisoned);
        assert_eq!(format!("{array:?}"), array_before);
        assert_eq!(format!("{workspace:?}"), workspace_before);
    }
}

#[test]
#[ignore = "MPI route construction must run in its own multi-rank process"]
fn column_major_routes_match_production_stage_geometry_and_transitions() {
    let _mpi_test_lock = super::MPI_TEST_LOCK
        .get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap();
    let universe = mpi::initialize().expect("MPI initialization failed");
    let world = universe.world();
    assert!(matches!(world.size(), 4 | 6), "run with 4 or 6 MPI ranks");
    let topology = MpiTopology::<2>::new(&world, [2, world.size() as usize / 2]).unwrap();
    let layout = DistributedLayout::default();

    let full = C2cPlan::<f64, 3, 2>::from_shape_with_selection_and_layout(
        Arc::clone(&topology),
        [4, 5, 7],
        ExtraShape::scalar(),
        AxisSelection::all(),
        layout,
    )
    .unwrap();
    assert_c2c_route(
        &full,
        &[
            (0, [0, 1, 2], [1, 2], [0, 1, 2], [1, 2]),
            (1, [0, 1, 2], [1, 2], [1, 0, 2], [0, 2]),
            (2, [1, 0, 2], [0, 2], [2, 1, 0], [0, 1]),
        ],
        &[false, true, true],
        2,
        TransposeMethod::AllToAllv,
    );

    let partial_12 = C2cPlan::<f64, 3, 2>::from_shape_with_selection_and_layout(
        Arc::clone(&topology),
        [4, 5, 7],
        ExtraShape::scalar(),
        AxisSelection::from_indices([1, 2]).unwrap(),
        layout,
    )
    .unwrap();
    assert_c2c_route(
        &partial_12,
        &[
            (1, [0, 1, 2], [1, 2], [1, 0, 2], [0, 2]),
            (2, [1, 0, 2], [0, 2], [2, 1, 0], [0, 1]),
        ],
        &[true, true],
        2,
        TransposeMethod::AllToAllv,
    );

    let partial_02 = C2cPlan::<f64, 3, 2>::from_shape_with_selection_and_layout(
        Arc::clone(&topology),
        [4, 5, 7],
        ExtraShape::scalar(),
        AxisSelection::from_indices([0, 2]).unwrap(),
        layout,
    )
    .unwrap();
    assert_c2c_route(
        &partial_02,
        &[
            (0, [0, 1, 2], [1, 2], [0, 1, 2], [1, 2]),
            (2, [0, 1, 2], [1, 2], [2, 0, 1], [1, 0]),
        ],
        &[false, true],
        1,
        TransposeMethod::AllToAllv,
    );

    let single = C2cPlan::<f64, 3, 2>::from_shape_with_selection_and_layout(
        Arc::clone(&topology),
        [4, 5, 7],
        ExtraShape::scalar(),
        AxisSelection::from_indices([2]).unwrap(),
        layout,
    )
    .unwrap();
    assert_c2c_route(
        &single,
        &[(2, [0, 1, 2], [1, 2], [2, 0, 1], [1, 0])],
        &[true],
        1,
        TransposeMethod::AllToAllv,
    );

    let empty = C2cPlan::<f64, 3, 2>::from_shape_with_selection_and_layout(
        Arc::clone(&topology),
        [4, 5, 7],
        ExtraShape::scalar(),
        AxisSelection::empty(),
        layout,
    )
    .unwrap();
    assert_c2c_route(&empty, &[], &[], 0, TransposeMethod::AllToAllv);
    assert!(empty.input_pencil().same_layout(empty.output_pencil()));

    let n4_subset = C2cPlan::<f64, 4, 2>::from_shape_with_selection_and_layout(
        Arc::clone(&topology),
        [4, 5, 6, 7],
        ExtraShape::scalar(),
        AxisSelection::from_indices([0, 2]).unwrap(),
        layout,
    )
    .unwrap();
    assert_c2c_route(
        &n4_subset,
        &[
            (0, [0, 1, 2, 3], [2, 3], [0, 1, 2, 3], [2, 3]),
            (2, [0, 1, 2, 3], [2, 3], [2, 0, 1, 3], [0, 3]),
        ],
        &[false, true],
        1,
        TransposeMethod::AllToAllv,
    );

    for method in [TransposeMethod::AllToAllv, TransposeMethod::PointToPoint] {
        let partial = C2cPlan::<f64, 3, 2>::from_shape_with_selection_and_layout(
            Arc::clone(&topology),
            [4, 5, 7],
            ExtraShape::scalar(),
            AxisSelection::from_indices([1, 2]).unwrap(),
            DistributedLayout {
                transpose_method: method,
                ..layout
            },
        )
        .unwrap();
        assert_c2c_route(
            &partial,
            &[
                (1, [0, 1, 2], [1, 2], [1, 0, 2], [0, 2]),
                (2, [1, 0, 2], [0, 2], [2, 1, 0], [0, 1]),
            ],
            &[true, true],
            2,
            method,
        );
    }
}

type ExpectedC2cStage<const N: usize, const M: usize> =
    (usize, [usize; N], [usize; M], [usize; N], [usize; M]);

fn assert_c2c_route<const N: usize, const M: usize>(
    plan: &C2cPlan<f64, N, M>,
    expected_stages: &[ExpectedC2cStage<N, M>],
    expected_distributed_stages: &[bool],
    expected_distributed_count: usize,
    method: TransposeMethod,
) where
    Complex<f64>: mpi::datatype::Equivalence,
{
    let actual_stages = plan.stage_geometry();
    assert_eq!(actual_stages.len(), expected_stages.len());
    assert_eq!(actual_stages.len(), expected_distributed_stages.len());
    for (index, ((stage, expected), &distributed)) in actual_stages
        .iter()
        .zip(expected_stages)
        .zip(expected_distributed_stages)
        .enumerate()
    {
        assert_eq!(stage.axis, expected.0, "route stage {index} axis");
        assert_pencil_layout(&stage.source, expected.1, expected.2, "source", index);
        assert_pencil_layout(&stage.output, expected.3, expected.4, "output", index);
        assert_eq!(
            !stage.source.same_distribution(&stage.output),
            distributed,
            "route stage {index} actual distribution transition"
        );
    }

    assert_eq!(
        expected_distributed_stages
            .iter()
            .filter(|&&distributed| distributed)
            .count(),
        expected_distributed_count,
        "independent hand-count for the expected route"
    );
    let actual_distributed_count = actual_stages
        .iter()
        .filter(|stage| !stage.source.same_distribution(&stage.output))
        .count();
    assert_eq!(actual_distributed_count, expected_distributed_count);

    let transitions = plan
        .core
        .transitions
        .iter()
        .map(|transition| &transition.forward);
    assert_eq!(
        plan.core.transitions.len(),
        actual_stages.len().saturating_sub(1),
        "one production transition connects each adjacent pair of stages"
    );
    for (index, (transition, &distributed)) in transitions
        .zip(expected_distributed_stages.iter().skip(1))
        .enumerate()
    {
        if distributed {
            match method {
                TransposeMethod::AllToAllv => assert!(
                    matches!(transition, C2cTransition::AllToAllv(_)),
                    "inter-stage transition {index}"
                ),
                TransposeMethod::PointToPoint => assert!(
                    matches!(transition, C2cTransition::PointToPoint(_)),
                    "inter-stage transition {index}"
                ),
            }
        } else {
            assert!(
                matches!(transition, C2cTransition::Identity),
                "inter-stage transition {index}"
            );
        }
    }
}

fn assert_pencil_layout<const N: usize, const M: usize>(
    pencil: &Pencil<N, M>,
    expected_axes: [usize; N],
    expected_decomposition: [usize; M],
    endpoint: &str,
    stage: usize,
) {
    assert_eq!(
        pencil.permutation().axes().map(|axis| axis.index()),
        expected_axes,
        "stage {stage} {endpoint} memory-axis order"
    );
    assert_eq!(
        pencil.decomposition().map(|axis| axis.index()),
        expected_decomposition,
        "stage {stage} {endpoint} decomposition slots"
    );
}
