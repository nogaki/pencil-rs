use std::{cell::Cell, sync::Arc};

use mpi::{
    collective::{CommunicatorCollectives, SystemOperation},
    topology::Communicator,
};
use num_complex::Complex32;
use pencil_array::{
    AllToAllvTransposePlan, AxisPermutation, CollectiveError, ExtraShape, MpiTopology, Pencil,
    PencilArray, TransposeError, all, all_by, any, any_by, gather, global_max, global_min,
    global_sum, l2_norm, map, norm_by, sum_by,
};

fn check_all<C: CommunicatorCollectives>(communicator: &C, local: bool) {
    let local_word = i32::from(local);
    let mut all_valid = 0i32;
    communicator.all_reduce_into(&local_word, &mut all_valid, SystemOperation::min());
    assert_eq!(all_valid, 1, "a rank failed the collective test check");
}

fn descriptor_mismatch<T>(result: &Result<T, CollectiveError>) -> bool {
    matches!(result, Err(CollectiveError::CollectiveDescriptorMismatch))
}

fn regular_array(pencil: Arc<Pencil<2, 2>>, extra: ExtraShape) -> PencilArray<i64, 2, 2> {
    let dimensions = extra.dimensions();
    assert_eq!(dimensions, &[2, 3]);
    let mut array = PencilArray::from_elem(Arc::clone(&pencil), extra, 0_i64).unwrap();
    let shape = pencil.local_shape_logical();
    let ranges = pencil.local_ranges().clone();
    for e0 in 0..2 {
        for e1 in 0..3 {
            for x in 0..shape[0] {
                for y in 0..shape[1] {
                    let global_x = ranges[0].start + x;
                    let global_y = ranges[1].start + y;
                    *array.get_local_mut(&[e0, e1], [x, y]).unwrap() =
                        (e0 * 100_000 + e1 * 10_000 + global_x * 100 + global_y) as i64;
                }
            }
        }
    }
    array
}

fn regular_expected() -> Vec<i64> {
    (0..2)
        .flat_map(|e0| {
            (0..3).flat_map(move |e1| {
                (0..5).flat_map(move |x| {
                    (0..7).map(move |y| (e0 * 100_000 + e1 * 10_000 + x * 100 + y) as i64)
                })
            })
        })
        .collect()
}

fn successful_empty_collective<C: CommunicatorCollectives>(
    communicator: &C,
    array: &PencilArray<i64, 2, 2>,
) {
    let result = global_sum(&array.view());
    check_all(communicator, result == Ok(7_i64));
}

fn run_cross_api_mismatch<C: CommunicatorCollectives>(
    communicator: &C,
    rank: usize,
    size: usize,
    topology: &Arc<MpiTopology<2>>,
    followup: &PencilArray<i64, 2, 2>,
) {
    if size == 1 {
        successful_empty_collective(communicator, followup);
        return;
    }

    // N and M match on both sides. Only the API operation namespace differs,
    // so this exercises the shared five-word header protocol itself.
    let source = Pencil::<3, 2>::new(Arc::clone(topology), [5, 5, 5], [0, 1]).unwrap();
    let destination = Pencil::<3, 2>::new(Arc::clone(topology), [5, 5, 5], [0, 2]).unwrap();
    let local_ok = if rank == 0 {
        let array =
            PencilArray::from_elem(Arc::clone(&source), ExtraShape::scalar(), 1_i64).unwrap();
        descriptor_mismatch(&array.global_sum())
    } else {
        matches!(
            AllToAllvTransposePlan::new(source, destination),
            Err(TransposeError::CollectiveDescriptorMismatch)
        )
    };
    check_all(communicator, local_ok);
    successful_empty_collective(communicator, followup);
}

fn run_descriptor_mismatch_suite<C: CommunicatorCollectives>(
    communicator: &C,
    rank: usize,
    size: usize,
    pencil: &Arc<Pencil<2, 2>>,
    followup: &PencilArray<i64, 2, 2>,
) {
    let scalar_i64 =
        PencilArray::from_elem(Arc::clone(pencil), ExtraShape::scalar(), 1_i64).unwrap();
    let scalar_i64_before = scalar_i64.as_slice().to_vec();

    // Input type mismatch.
    let scalar_f64 =
        PencilArray::from_elem(Arc::clone(pencil), ExtraShape::scalar(), 1.0_f64).unwrap();
    let scalar_f64_before = scalar_f64.as_slice().to_vec();
    let local_ok = if size == 1 {
        global_sum(&scalar_i64.view()) == Ok(pencil.global_len() as i64)
            && scalar_i64.as_slice() == scalar_i64_before.as_slice()
    } else if rank == 0 {
        descriptor_mismatch(&global_sum(&scalar_f64.view()))
            && scalar_f64.as_slice() == scalar_f64_before.as_slice()
    } else {
        descriptor_mismatch(&global_sum(&scalar_i64.view()))
            && scalar_i64.as_slice() == scalar_i64_before.as_slice()
    };
    check_all(communicator, local_ok);
    successful_empty_collective(communicator, followup);

    // Scalar precision mismatch.
    let scalar_i32 =
        PencilArray::from_elem(Arc::clone(pencil), ExtraShape::scalar(), 1_i32).unwrap();
    let scalar_i64_again =
        PencilArray::from_elem(Arc::clone(pencil), ExtraShape::scalar(), 1_i64).unwrap();
    let scalar_i32_before = scalar_i32.as_slice().to_vec();
    let scalar_i64_again_before = scalar_i64_again.as_slice().to_vec();
    let local_ok = if size == 1 {
        global_sum(&scalar_i32.view()) == Ok(pencil.global_len() as i32)
            && scalar_i32.as_slice() == scalar_i32_before.as_slice()
    } else if rank == 0 {
        descriptor_mismatch(&global_sum(&scalar_i32.view()))
            && scalar_i32.as_slice() == scalar_i32_before.as_slice()
    } else {
        descriptor_mismatch(&global_sum(&scalar_i64_again.view()))
            && scalar_i64_again.as_slice() == scalar_i64_again_before.as_slice()
    };
    check_all(communicator, local_ok);
    successful_empty_collective(communicator, followup);

    // Mismatched mapped output types must reject before invoking either
    // callback. The input snapshots also cover the no-write preflight path.
    let output_before = scalar_i64.as_slice().to_vec();
    let calls = Cell::new(0usize);
    let local_ok = if size == 1 {
        let result = sum_by(&scalar_i64.view(), |value| {
            calls.set(calls.get() + 1);
            *value as i32
        });
        result == Ok(pencil.global_len() as i32)
            && calls.get() == scalar_i64.len()
            && scalar_i64.as_slice() == output_before.as_slice()
    } else if rank == 0 {
        let result = sum_by(&scalar_i64.view(), |value| {
            calls.set(calls.get() + 1);
            *value as f32
        });
        descriptor_mismatch(&result)
            && calls.get() == 0
            && scalar_i64.as_slice() == output_before.as_slice()
    } else {
        let result = sum_by(&scalar_i64.view(), |value| {
            calls.set(calls.get() + 1);
            *value as f64
        });
        descriptor_mismatch(&result)
            && calls.get() == 0
            && scalar_i64.as_slice() == output_before.as_slice()
    };
    check_all(communicator, local_ok);
    successful_empty_collective(communicator, followup);

    // Norm output types are also part of the descriptor.
    let calls = Cell::new(0usize);
    let local_ok = if size == 1 {
        let result = norm_by(&scalar_i64.view(), |value| {
            calls.set(calls.get() + 1);
            *value as f32
        });
        result.is_ok() && calls.get() == scalar_i64.len()
    } else if rank == 0 {
        let result = norm_by(&scalar_i64.view(), |value| {
            calls.set(calls.get() + 1);
            *value as f32
        });
        descriptor_mismatch(&result) && calls.get() == 0
    } else {
        let result = norm_by(&scalar_i64.view(), |value| {
            calls.set(calls.get() + 1);
            *value as f64
        });
        descriptor_mismatch(&result) && calls.get() == 0
    };
    check_all(communicator, local_ok);
    successful_empty_collective(communicator, followup);

    // Operation mismatch.
    let local_ok = if size == 1 {
        global_sum(&scalar_i64.view()) == Ok(pencil.global_len() as i64)
    } else if rank == 0 {
        descriptor_mismatch(&global_sum(&scalar_i64.view()))
    } else {
        descriptor_mismatch(&global_min(&scalar_i64.view()))
    };
    check_all(communicator, local_ok);
    successful_empty_collective(communicator, followup);

    // Root mismatch is rejected by the descriptor before point-to-point
    // gather traffic begins.
    let gather_before = scalar_i64.as_slice().to_vec();
    let local_root = if rank == 0 { 0_i32 } else { 1_i32 };
    let local_ok = if size == 1 {
        gather(&scalar_i64.view(), 0_i32).is_ok()
            && scalar_i64.as_slice() == gather_before.as_slice()
    } else {
        descriptor_mismatch(&gather(&scalar_i64.view(), local_root))
            && scalar_i64.as_slice() == gather_before.as_slice()
    };
    check_all(communicator, local_ok);
    successful_empty_collective(communicator, followup);

    // Layout mismatch, including memory permutation, is also preflight-only.
    let wrong_pencil = pencil
        .with_permutation(AxisPermutation::identity())
        .unwrap();
    let wrong_array =
        PencilArray::from_elem(Arc::clone(&wrong_pencil), ExtraShape::scalar(), 1_i64).unwrap();
    let wrong_before = wrong_array.as_slice().to_vec();
    let local_ok = if size == 1 {
        global_sum(&scalar_i64.view()) == Ok(pencil.global_len() as i64)
    } else if rank == 0 {
        descriptor_mismatch(&global_sum(&wrong_array.view()))
            && wrong_array.as_slice() == wrong_before.as_slice()
    } else {
        descriptor_mismatch(&global_sum(&scalar_i64.view()))
            && scalar_i64.as_slice() == scalar_i64_before.as_slice()
    };
    check_all(communicator, local_ok);
    successful_empty_collective(communicator, followup);
}

fn run_integer_overflow_tests<C: CommunicatorCollectives>(
    communicator: &C,
    rank: usize,
    size: usize,
    topology: &Arc<MpiTopology<2>>,
    pencil: &Arc<Pencil<2, 2>>,
    followup: &PencilArray<i64, 2, 2>,
) {
    // Rank zero overflows its local partial; the other ranks do not.
    let local_value = if rank == 0 { i8::MAX } else { 0_i8 };
    let local_array =
        PencilArray::from_elem(Arc::clone(pencil), ExtraShape::scalar(), local_value).unwrap();
    let local_before = local_array.as_slice().to_vec();
    let local_ok = matches!(
        global_sum(&local_array.view()),
        Err(CollectiveError::IntegerOverflow)
    ) && local_array.as_slice() == local_before.as_slice();
    check_all(communicator, local_ok);
    successful_empty_collective(communicator, followup);

    // Every rank's partial fits in u8, but the checked rank-order fold does
    // not. This is distinct from the local partial overflow above.
    let grid = *topology.process_grid();
    let one_value_pencil = Pencil::<2, 2>::new(Arc::clone(topology), grid, [0, 1]).unwrap();
    let global_only_array =
        PencilArray::from_elem(Arc::clone(&one_value_pencil), ExtraShape::scalar(), 200_u8)
            .unwrap();
    let global_only_before = global_only_array.as_slice().to_vec();
    let result = global_sum(&global_only_array.view());
    let local_ok = if size == 1 {
        result == Ok(200_u8)
    } else {
        matches!(result, Err(CollectiveError::IntegerOverflow))
    } && global_only_array.as_slice() == global_only_before.as_slice();
    check_all(communicator, local_ok);
    successful_empty_collective(communicator, followup);
}

fn run_gather_tests<C: CommunicatorCollectives>(
    communicator: &C,
    rank: usize,
    size: usize,
    pencil: &Arc<Pencil<2, 2>>,
    array: &PencilArray<i64, 2, 2>,
    expected: &[i64],
) {
    let root = if size == 1 { 0 } else { size - 1 };
    let before = array.as_slice().to_vec();
    let gathered = gather(&array.view(), i32::try_from(root).unwrap());
    let local_ok = match gathered.as_ref() {
        Ok(Some(values)) => rank == root && values == expected,
        Ok(None) => rank != root,
        Err(_) => false,
    };
    check_all(communicator, local_ok);
    check_all(communicator, array.as_slice() == before.as_slice());

    // A process whose local spatial block is empty can still be the root.
    let empty_pencil = Pencil::<2, 2>::new(Arc::clone(pencil.topology()), [1, 1], [0, 1]).unwrap();
    let empty_array =
        PencilArray::from_elem(Arc::clone(&empty_pencil), ExtraShape::scalar(), 7_i64).unwrap();
    let empty_before = empty_array.as_slice().to_vec();
    let gathered = gather(&empty_array.view(), i32::try_from(root).unwrap());
    let local_ok = match gathered.as_ref() {
        Ok(Some(values)) => rank == root && values == &[7_i64],
        Ok(None) => rank != root,
        Err(_) => false,
    };
    check_all(communicator, local_ok);
    check_all(
        communicator,
        empty_array.as_slice() == empty_before.as_slice(),
    );

    // No payload at all is a separate path from ranks with empty spatial
    // blocks. It must still return an allocated empty root vector.
    let zero_extra = ExtraShape::new([0]).unwrap();
    let zero_array = PencilArray::from_elem(Arc::clone(pencil), zero_extra, 9_i64).unwrap();
    let zero_before = zero_array.as_slice().to_vec();
    let gathered = gather(&zero_array.view(), i32::try_from(root).unwrap());
    let local_ok = match gathered.as_ref() {
        Ok(Some(values)) => rank == root && values.is_empty(),
        Ok(None) => rank != root,
        Err(_) => false,
    };
    check_all(communicator, local_ok);
    check_all(
        communicator,
        zero_array.as_slice() == zero_before.as_slice(),
    );

    let bad_root = i32::try_from(size).unwrap();
    let before = array.as_slice().to_vec();
    let result = gather(&array.view(), bad_root);
    let expected_error = CollectiveError::RootOutOfBounds {
        root: i64::try_from(size).unwrap(),
        size,
    };
    let local_ok =
        result.as_ref().err() == Some(&expected_error) && array.as_slice() == before.as_slice();
    check_all(communicator, local_ok);
}

fn run_complex_sum_tests<C: CommunicatorCollectives>(
    communicator: &C,
    rank: usize,
    size: usize,
    pencil: &Arc<Pencil<2, 2>>,
    followup: &PencilArray<i64, 2, 2>,
) {
    // Rank one has finite partials that overflow to the opposite signs. Rank
    // zero contributes exactly one input infinity per component. Raw input
    // flags, rather than native partial results, determine the answer.
    let mut witness = vec![Complex32::new(0.0, 0.0); pencil.local_len()];
    if rank == 0 {
        witness[0] = Complex32::new(f32::INFINITY, f32::NEG_INFINITY);
    } else if size > 1 && rank == 1 {
        witness.fill(Complex32::new(-f32::MAX, f32::MAX));
    }
    let witness_array =
        PencilArray::from_vec(Arc::clone(pencil), ExtraShape::scalar(), witness).unwrap();
    let witness_result = global_sum(&witness_array.view());
    let witness_ok = match witness_result {
        Ok(value) => {
            value.re.is_infinite()
                && value.re.is_sign_positive()
                && value.im.is_infinite()
                && value.im.is_sign_negative()
        }
        Err(_) => false,
    };
    check_all(communicator, witness_ok);

    // NaN wins independently for each complex component.
    let mut nan_values = vec![Complex32::new(0.0, 0.0); pencil.local_len()];
    if rank == 0 {
        nan_values[0] = Complex32::new(f32::NAN, 3.0);
    }
    let nan_array =
        PencilArray::from_vec(Arc::clone(pencil), ExtraShape::scalar(), nan_values).unwrap();
    let nan_result = global_sum(&nan_array.view());
    let nan_ok = match nan_result {
        Ok(value) => value.re.is_nan() && value.im.is_finite() && value.im == 3.0,
        Err(_) => false,
    };
    check_all(communicator, nan_ok);

    // Opposing input signs produce NaN; with one rank, the lone sign is kept.
    let opposing_value = if rank == 0 {
        Complex32::new(f32::INFINITY, 0.0)
    } else if size > 1 && rank == 1 {
        Complex32::new(f32::NEG_INFINITY, 0.0)
    } else {
        Complex32::new(0.0, 0.0)
    };
    let opposing_array =
        PencilArray::from_elem(Arc::clone(pencil), ExtraShape::scalar(), opposing_value).unwrap();
    let opposing_result = global_sum(&opposing_array.view());
    let opposing_ok = match opposing_result {
        Ok(value) if size == 1 => value.re.is_infinite() && value.re.is_sign_positive(),
        Ok(value) => value.re.is_nan(),
        Err(_) => false,
    };
    check_all(communicator, opposing_ok);

    let finite_large = PencilArray::from_elem(
        Arc::clone(pencil),
        ExtraShape::scalar(),
        Complex32::new(f32::MAX, f32::MAX),
    )
    .unwrap();
    let norm_ok = l2_norm(&finite_large.view())
        .map(|value| value.is_infinite())
        .unwrap_or(false);
    check_all(communicator, norm_ok);
    successful_empty_collective(communicator, followup);
}

#[test]
fn reductions_and_logical_gather_cover_permuted_pencils() {
    let universe = mpi::initialize().expect("MPI initialization failed");
    let world = universe.world();
    let size = usize::try_from(world.size()).unwrap();
    let grid = match size {
        1 => [1, 1],
        4 => [2, 2],
        6 => [2, 3],
        other => panic!("run with one, four, or six MPI ranks, got {other}"),
    };
    let topology = MpiTopology::<2>::new(&world, grid).unwrap();
    let pencil = Pencil::<2, 2>::new_permuted(
        Arc::clone(&topology),
        [5, 7],
        [1, 0],
        AxisPermutation::new([1, 0]).unwrap(),
    )
    .unwrap();
    let extra = ExtraShape::new([2, 3]).unwrap();
    let array = regular_array(Arc::clone(&pencil), extra);
    let expected = regular_expected();
    let expected_sum = expected.iter().sum::<i64>();
    let rank = usize::try_from(world.rank()).unwrap();

    let mixed_sum = match rank % 3 {
        0 => array.global_sum(),
        1 => array.view().global_sum(),
        _ => global_sum(&array.view()),
    };
    check_all(&world, mixed_sum == Ok(expected_sum));
    check_all(
        &world,
        global_min(&array.view()) == expected.iter().copied().min().map(Some).map(Ok).unwrap(),
    );
    check_all(
        &world,
        global_max(&array.view()) == expected.iter().copied().max().map(Some).map(Ok).unwrap(),
    );

    let expected_norm = expected
        .iter()
        .map(|&value| (value as f64) * (value as f64))
        .sum::<f64>()
        .sqrt();
    let norm_result = l2_norm(&array.view());
    check_all(
        &world,
        norm_result
            .map(|value| (value - expected_norm).abs() < 1.0e-6)
            .unwrap_or(false),
    );

    let empty =
        PencilArray::from_elem(Arc::clone(&pencil), ExtraShape::new([0]).unwrap(), 1_i64).unwrap();
    check_all(&world, global_sum(&empty.view()) == Ok(0_i64));
    check_all(&world, global_min(&empty.view()) == Ok(None));
    check_all(&world, global_max(&empty.view()) == Ok(None));
    check_all(&world, l2_norm(&empty.view()) == Ok(0.0));
    check_all(&world, all(&empty.view()) == Ok(true));
    check_all(&world, any(&empty.view()) == Ok(false));

    let any_calls = Cell::new(0usize);
    let any_result = any_by(&array.view(), |value| {
        any_calls.set(any_calls.get() + 1);
        *value < 0
    });
    check_all(
        &world,
        any_result == Ok(false) && any_calls.get() == array.len(),
    );
    let all_calls = Cell::new(0usize);
    let all_result = all_by(&array.view(), |value| {
        all_calls.set(all_calls.get() + 1);
        *value >= 0
    });
    check_all(
        &world,
        all_result == Ok(true) && all_calls.get() == array.len(),
    );

    let mapped = map(&array.view(), |value| *value as i32);
    let expected_mapped: Vec<i32> = array.as_slice().iter().map(|&value| value as i32).collect();
    check_all(
        &world,
        mapped.as_ref().map(|values| values == &expected_mapped) == Ok(true),
    );
    check_all(
        &world,
        sum_by(&array.view(), |value| *value as i32) == Ok(expected_sum as i32),
    );
    let norm_by_result = norm_by(&array.view(), |value| *value as f32);
    let expected_norm_f32 = expected_norm as f32;
    check_all(
        &world,
        norm_by_result
            .map(|value| (value - expected_norm_f32).abs() < 2.0)
            .unwrap_or(false),
    );

    run_gather_tests(&world, rank, size, &pencil, &array, &expected);

    let empty_rank_pencil = Pencil::<2, 2>::new(Arc::clone(&topology), [1, 1], [0, 1]).unwrap();
    let empty_rank_array =
        PencilArray::from_elem(Arc::clone(&empty_rank_pencil), ExtraShape::scalar(), 7_i64)
            .unwrap();
    successful_empty_collective(&world, &empty_rank_array);

    run_cross_api_mismatch(&world, rank, size, &topology, &empty_rank_array);
    run_descriptor_mismatch_suite(&world, rank, size, &pencil, &empty_rank_array);
    run_integer_overflow_tests(&world, rank, size, &topology, &pencil, &empty_rank_array);
    run_complex_sum_tests(&world, rank, size, &pencil, &empty_rank_array);
}
