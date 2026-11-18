use super::super::{
    ENDPOINT_BATCH_CHUNK, EndpointReductionEvent, endpoint_chunk_count, endpoint_chunk_len,
    endpoint_reduction_log, endpoint_ulp_neighbors, force_endpoint_allocation_failure,
    reset_endpoint_reduction_log,
};
use super::*;
use mpi::topology::Communicator;

fn chunk_plan<R: FftReal, const M: usize>(
    topology: &Arc<MpiTopology<M>>,
    batch_count: usize,
    real_len: usize,
) -> R2cPlan<R, 2, M>
where
    Complex<R>: Equivalence,
{
    R2cPlan::from_shape(
        Arc::clone(topology),
        [4, real_len],
        ExtraShape::new(vec![batch_count]).unwrap(),
    )
    .unwrap()
}

// The boundary intermediate's active layout, reconstructed as an ordinary
// array so tests can address constrained planes by global coordinates.
fn boundary_data<R: FftReal, const M: usize>(
    plan: &R2cPlan<R, 2, M>,
    workspace: &R2cWorkspace<R, 2, M>,
) -> PencilArray<Complex<R>, 2, M>
where
    Complex<R>: Equivalence,
{
    let pencil = workspace.intermediate.active_pencil().unwrap();
    let pencil = pencil.with_global_shape(*pencil.global_shape()).unwrap();
    PencilArray::from_elem(pencil, plan.core.extra_shape.clone(), zero_complex::<R>()).unwrap()
}

fn validate_with<R: FftReal, const M: usize>(
    plan: &R2cPlan<R, 2, M>,
    workspace: &mut R2cWorkspace<R, 2, M>,
    data: &PencilArray<Complex<R>, 2, M>,
) -> Result<(), R2cError>
where
    Complex<R>: Equivalence,
{
    validate_with_mode(plan, workspace, data, true)
}

fn validate_with_mode<R: FftReal, const M: usize>(
    plan: &R2cPlan<R, 2, M>,
    workspace: &mut R2cWorkspace<R, 2, M>,
    data: &PencilArray<Complex<R>, 2, M>,
    normalize: bool,
) -> Result<(), R2cError>
where
    Complex<R>: Equivalence,
{
    workspace
        .intermediate
        .active_view_mut()
        .unwrap()
        .as_mut_slice()
        .copy_from_slice(data.as_slice());
    validate_boundary(
        &plan.core,
        &workspace.intermediate,
        normalize,
        plan.raw_absolute_threshold,
    )
}

fn chunk_protocol(batch_count: usize) {
    let chunk = batch_count.min(ENDPOINT_BATCH_CHUNK);
    let log = endpoint_reduction_log();
    let mut allocations = Vec::new();
    let mut maxima = Vec::new();
    let mut sums = Vec::new();
    let mut preparations = 0;
    let mut finals = 0;
    for event in &log {
        match *event {
            EndpointReductionEvent::Allocation { slot, length } => {
                assert_eq!(slot, allocations.len(), "batch_count={batch_count}");
                assert_eq!(length, chunk, "batch_count={batch_count}");
                allocations.push(length);
            }
            EndpointReductionEvent::PreparationAgreement => preparations += 1,
            EndpointReductionEvent::MaxReduction { elements } => maxima.push(elements),
            EndpointReductionEvent::SumReduction { elements } => sums.push(elements),
            EndpointReductionEvent::FinalAgreement => finals += 1,
        }
    }
    assert_eq!(allocations.len(), 3, "batch_count={batch_count}");
    assert_eq!(log[3], EndpointReductionEvent::PreparationAgreement);
    assert_eq!(preparations, 1, "batch_count={batch_count}");
    assert_eq!(finals, 1, "batch_count={batch_count}");
    assert_eq!(
        log.last(),
        Some(&EndpointReductionEvent::FinalAgreement),
        "batch_count={batch_count}"
    );
    let expected: Vec<usize> = (0..endpoint_chunk_count(batch_count))
        .map(|index| 4 * endpoint_chunk_len(batch_count, index * ENDPOINT_BATCH_CHUNK))
        .collect();
    assert_eq!(maxima, expected, "MAX chunks for batch_count={batch_count}");
    assert_eq!(sums, expected, "SUM chunks for batch_count={batch_count}");
}

fn assert_no_statistics_reduction(log: &[EndpointReductionEvent]) {
    assert!(
        log.iter().all(|event| !matches!(
            event,
            EndpointReductionEvent::MaxReduction { .. }
                | EndpointReductionEvent::SumReduction { .. }
                | EndpointReductionEvent::FinalAgreement
        )),
        "statistics reductions ran after a preparation failure: {log:?}"
    );
    assert_eq!(
        log.iter()
            .filter(|event| matches!(event, EndpointReductionEvent::PreparationAgreement))
            .count(),
        1,
        "{log:?}"
    );
}

fn chunked_statistics<R: FftReal, const M: usize>(topology: &Arc<MpiTopology<M>>)
where
    Complex<R>: Equivalence,
{
    let chunk = ENDPOINT_BATCH_CHUNK;
    for real_len in [6, 7, 1] {
        for batch_count in [0, 1, chunk - 1, chunk, chunk + 1, 2 * chunk + 1] {
            let plan = chunk_plan::<R, M>(topology, batch_count, real_len);
            let mut workspace = plan.allocate_workspace().unwrap();
            let data = boundary_data(&plan, &workspace);
            reset_endpoint_reduction_log();
            validate_with(&plan, &mut workspace, &data).unwrap();
            chunk_protocol(batch_count);
        }
    }
    // A permuted boundary layout keeps the same bounded protocol.
    let batch_count = chunk + 1;
    let plan = R2cPlan::<R, 2, M>::from_shape_with_layout(
        Arc::clone(topology),
        [4, 6],
        ExtraShape::new(vec![batch_count]).unwrap(),
        DistributedLayout {
            permute_dims: true,
            ..DistributedLayout::default()
        },
    )
    .unwrap();
    let mut workspace = plan.allocate_workspace().unwrap();
    let data = boundary_data(&plan, &workspace);
    reset_endpoint_reduction_log();
    validate_with(&plan, &mut workspace, &data).unwrap();
    chunk_protocol(batch_count);
}

// Four exactly representable unit real parts make the fused SUM exactly 4,
// so the relative threshold is 2*relative. Replacing the SUM with a MAX
// would see 1 and reject the accepted neighbor, so this fixture pins the
// fused sum across batches and ranks.
fn multi_contribution_probes<R: FftReal, const M: usize>(topology: &Arc<MpiTopology<M>>)
where
    Complex<R>: Equivalence,
{
    let plan = R2cPlan::<R, 2, M>::from_shape_with_selection(
        Arc::clone(topology),
        [4, 2],
        ExtraShape::scalar(),
        AxisSelection::all(),
    )
    .unwrap();
    let mut workspace = plan.allocate_workspace().unwrap();
    let boundary = plan.core.real_stage_index.unwrap();
    let reduction_axis = plan.core.stages[boundary].axis;
    assert_eq!(reduction_axis, 1);
    let (real_len, complex_len) = r2c_lengths(&plan.core);
    assert_eq!((real_len, complex_len), (2, 2));
    let depth = normalization_depth([4, 2], plan.core.selection, reduction_axis);
    assert_eq!(depth, 3.0);
    let relative = 128.0 * <R as crate::private::Sealed>::pencil_fft_epsilon_f64() * depth;
    let inverse_absolute =
        128.0 * <R as crate::private::Sealed>::pencil_fft_min_subnormal_f64() * depth;
    // The unnormalized threshold carries the transverse extent 4.
    assert_eq!(plan.raw_absolute_threshold, 4.0 * inverse_absolute);
    let one = R::from_f64(1.0).unwrap();
    for normalize in [true, false] {
        let absolute = if normalize {
            inverse_absolute
        } else {
            plan.raw_absolute_threshold
        };
        for plane in 0..2 {
            let k = if plane == 0 { 0 } else { complex_len - 1 };
            for (im, accepted) in endpoint_ulp_neighbors::<R>(2.0 * relative)
                .into_iter()
                .zip([true, true, false])
            {
                let mut data = boundary_data(&plan, &workspace);
                for row in 0..4 {
                    if let Some(value) = data.get_global_mut(&[], [row, k]) {
                        *value = Complex::new(one, R::zero());
                    }
                }
                if let Some(value) = data.get_global_mut(&[], [0, k]) {
                    *value = Complex::new(one, R::from_f64(im).unwrap());
                }
                let result = validate_with_mode(&plan, &mut workspace, &data, normalize);
                assert_eq!(
                    result.is_ok(),
                    accepted,
                    "plane={plane} normalize={normalize} im={im}: {result:?}"
                );
                if !accepted {
                    assert!(matches!(result, Err(R2cError::InvalidSpectrum)));
                }
            }
            // The absolute test sees a single imaginary endpoint.
            for (im, accepted) in endpoint_ulp_neighbors::<R>(absolute)
                .into_iter()
                .zip([true, true, false])
            {
                let mut data = boundary_data(&plan, &workspace);
                if let Some(value) = data.get_global_mut(&[], [0, k]) {
                    *value = Complex::new(R::zero(), R::from_f64(im).unwrap());
                }
                let result = validate_with_mode(&plan, &mut workspace, &data, normalize);
                assert_eq!(
                    result.is_ok(),
                    accepted,
                    "absolute plane={plane} normalize={normalize} im={im}: {result:?}"
                );
                if !accepted {
                    assert!(matches!(result, Err(R2cError::InvalidSpectrum)));
                }
            }
        }
    }
}

fn invalid_batches<R: FftReal, const M: usize>(topology: &Arc<MpiTopology<M>>)
where
    Complex<R>: Equivalence,
{
    let chunk = ENDPOINT_BATCH_CHUNK;
    let batch_count = 2 * chunk + 1;
    let plan = chunk_plan::<R, M>(topology, batch_count, 6);
    let mut workspace = plan.allocate_workspace().unwrap();
    let (_, complex_len) = r2c_lengths(&plan.core);
    for (batch, plane, re, im) in [
        (0, 0, 0.0, 1e30),
        (2 * chunk, 0, 0.0, 1e30),
        (chunk, complex_len - 1, 0.0, 1e30),
        (2 * chunk, 0, f64::NAN, 0.0),
        (0, 0, 0.0, f64::INFINITY),
    ] {
        let mut data = boundary_data(&plan, &workspace);
        for i in 0..4 {
            if let Some(value) = data.get_global_mut(&[batch], [i, plane]) {
                *value = Complex::new(R::from_f64(re).unwrap(), R::from_f64(im).unwrap());
            }
        }
        reset_endpoint_reduction_log();
        let result = validate_with(&plan, &mut workspace, &data);
        assert!(
            matches!(result, Err(R2cError::InvalidSpectrum)),
            "batch={batch} plane={plane} re={re} im={im}: {result:?}"
        );
        // The first or last invalid chunk does not stop later chunks.
        chunk_protocol(batch_count);
    }
}

fn plane_and_batch_separation<R: FftReal, const M: usize>(topology: &Arc<MpiTopology<M>>)
where
    Complex<R>: Equivalence,
{
    let chunk = ENDPOINT_BATCH_CHUNK;
    let batch_count = chunk + 1;
    let plan = chunk_plan::<R, M>(topology, batch_count, 6);
    let mut workspace = plan.allocate_workspace().unwrap();
    let (_, complex_len) = r2c_lengths(&plan.core);
    let huge = R::from_f64(1e20).unwrap();
    let small = R::from_f64(1.0).unwrap();
    for (batch, plane) in [
        (0, 0),
        (0, complex_len - 1),
        (chunk, 0),
        (chunk, complex_len - 1),
    ] {
        let mut data = boundary_data(&plan, &workspace);
        data.as_mut_slice().fill(Complex::new(huge, R::zero()));
        for i in 0..4 {
            if let Some(value) = data.get_global_mut(&[batch], [i, plane]) {
                *value = Complex::new(R::zero(), small);
            }
        }
        reset_endpoint_reduction_log();
        let result = validate_with(&plan, &mut workspace, &data);
        assert!(
            matches!(result, Err(R2cError::InvalidSpectrum)),
            "batch={batch} plane={plane}: {result:?}"
        );
        chunk_protocol(batch_count);
    }
    // Uniformly large clean endpoints are accepted.
    let mut data = boundary_data(&plan, &workspace);
    data.as_mut_slice().fill(Complex::new(huge, R::zero()));
    validate_with(&plan, &mut workspace, &data).unwrap();
    // Signed zero is accepted.
    let mut data = boundary_data(&plan, &workspace);
    data.as_mut_slice().fill(Complex::new(
        R::from_f64(-0.0).unwrap(),
        R::from_f64(-0.0).unwrap(),
    ));
    validate_with(&plan, &mut workspace, &data).unwrap();
    // Non-endpoint bins stay unconstrained.
    let mut data = boundary_data(&plan, &workspace);
    for i in 0..4 {
        if let Some(value) = data.get_global_mut(&[chunk], [i, 1]) {
            *value = Complex::new(R::zero(), huge);
        }
    }
    validate_with(&plan, &mut workspace, &data).unwrap();
    // Odd and length-one real axes constrain DC only.
    for real_len in [7, 1] {
        let plan = chunk_plan::<R, M>(topology, batch_count, real_len);
        let mut workspace = plan.allocate_workspace().unwrap();
        let (_, complex_len) = r2c_lengths(&plan.core);
        if real_len % 2 == 1 && complex_len > 1 {
            let mut data = boundary_data(&plan, &workspace);
            for i in 0..4 {
                if let Some(value) = data.get_global_mut(&[batch_count - 1], [i, complex_len - 1]) {
                    *value = Complex::new(R::zero(), huge);
                }
            }
            validate_with(&plan, &mut workspace, &data).unwrap();
        }
        let mut data = boundary_data(&plan, &workspace);
        for i in 0..4 {
            if let Some(value) = data.get_global_mut(&[batch_count - 1], [i, 0]) {
                *value = Complex::new(R::zero(), huge);
            }
        }
        let result = validate_with(&plan, &mut workspace, &data);
        assert!(
            matches!(result, Err(R2cError::InvalidSpectrum)),
            "real_len={real_len}: {result:?}"
        );
    }
}

fn allocation_and_preparation_failures<R: FftReal, const M: usize>(
    topology: &Arc<MpiTopology<M>>,
    rank: usize,
) where
    Complex<R>: Equivalence,
{
    let chunk = ENDPOINT_BATCH_CHUNK;
    let batch_count = chunk + 1;
    let plan = chunk_plan::<R, M>(topology, batch_count, 6);
    let mut workspace = plan.allocate_workspace().unwrap();
    let data = boundary_data(&plan, &workspace);
    for slot in 0..3 {
        reset_endpoint_reduction_log();
        if rank == 0 {
            force_endpoint_allocation_failure(slot);
        }
        let result = validate_with(&plan, &mut workspace, &data);
        match (rank, result) {
            (0, Err(R2cError::Fft(FftError::AllocationFailed { required }))) => {
                assert_eq!(required, chunk, "slot={slot}")
            }
            (_, Err(R2cError::Fft(FftError::CollectivePreconditionFailed))) => {}
            (_, other) => panic!("unexpected allocation failure for slot {slot}: {other:?}"),
        }
        let log = endpoint_reduction_log();
        assert_no_statistics_reduction(&log);
        // The injecting rank stops at its failing slot; every other rank
        // allocated all three buffers before the preparation agreement.
        let allocated = log
            .iter()
            .filter(|event| matches!(event, EndpointReductionEvent::Allocation { .. }))
            .count();
        assert_eq!(
            allocated,
            if rank == 0 { slot + 1 } else { 3 },
            "slot={slot}: {log:?}"
        );
    }
    // A poisoned active layout on one rank is agreed before any reduction.
    reset_endpoint_reduction_log();
    let mut poisoned = plan.allocate_workspace().unwrap();
    if rank == 0 {
        let target = {
            let pencil = poisoned.intermediate.active_pencil().unwrap();
            pencil.with_global_shape(*pencil.global_shape()).unwrap()
        };
        poisoned
            .intermediate
            .overwrite_with(&target, |_| Err::<(), _>(()))
            .unwrap_err();
    }
    let result = validate_boundary(
        &plan.core,
        &poisoned.intermediate,
        true,
        plan.raw_absolute_threshold,
    );
    match (rank, result) {
        (0, Err(R2cError::Fft(FftError::Array(ArrayError::Poisoned)))) => {}
        (_, Err(R2cError::Fft(FftError::CollectivePreconditionFailed))) => {}
        (_, other) => panic!("unexpected poisoned-view result: {other:?}"),
    }
    let log = endpoint_reduction_log();
    assert_no_statistics_reduction(&log);
    // Only the rank with the poisoned layout skips allocation.
    assert_eq!(
        log.iter()
            .filter(|event| matches!(event, EndpointReductionEvent::Allocation { .. }))
            .count(),
        if rank == 0 { 0 } else { 3 },
        "{log:?}"
    );
}

fn public_routes<R: FftReal>(topology: &Arc<MpiTopology<1>>)
where
    Complex<R>: Equivalence,
{
    let chunk = ENDPOINT_BATCH_CHUNK;
    let batch_count = chunk + 1;
    let plan = chunk_plan::<R, 1>(topology, batch_count, 6);
    let mut source = plan.allocate_output().unwrap();
    source.as_mut_slice().fill(zero_complex::<R>());
    let mut destination = plan.allocate_input().unwrap();
    let mut workspace = plan.allocate_workspace().unwrap();
    for raw in [false, true] {
        reset_endpoint_reduction_log();
        destination.as_mut_slice().fill(R::from_f64(7.0).unwrap());
        if raw {
            plan.backward(&source, &mut destination, &mut workspace)
                .unwrap();
        } else {
            plan.inverse(&source, &mut destination, &mut workspace)
                .unwrap();
        }
        chunk_protocol(batch_count);
    }
    reset_endpoint_reduction_log();
    plan.inverse_with_timing(&source, &mut destination, &mut workspace)
        .unwrap();
    chunk_protocol(batch_count);
    reset_endpoint_reduction_log();
    plan.backward_with_timing(&source, &mut destination, &mut workspace)
        .unwrap();
    chunk_protocol(batch_count);

    let mut array = plan.allocate_in_place().unwrap();
    let mut in_place_workspace = plan.allocate_in_place_workspace().unwrap();
    for raw in [false, true] {
        plan.forward_in_place(&mut array, &mut in_place_workspace)
            .unwrap();
        array
            .complex_view_mut()
            .unwrap()
            .as_mut_slice()
            .fill(zero_complex::<R>());
        reset_endpoint_reduction_log();
        if raw {
            plan.backward_in_place(&mut array, &mut in_place_workspace)
                .unwrap();
        } else {
            plan.inverse_in_place(&mut array, &mut in_place_workspace)
                .unwrap();
        }
        chunk_protocol(batch_count);
    }

    let overlap_plan = R2cPlan::<R, 2, 1>::from_shape_with_layout(
        Arc::clone(topology),
        [4, 6],
        ExtraShape::new(vec![batch_count]).unwrap(),
        DistributedLayout {
            transpose_method: TransposeMethod::PointToPoint,
            permute_dims: false,
        },
    )
    .unwrap();
    let mut overlap_source = overlap_plan.allocate_output().unwrap();
    overlap_source.as_mut_slice().fill(zero_complex::<R>());
    let mut overlap_destination = overlap_plan.allocate_input().unwrap();
    let mut overlap_workspace = overlap_plan.allocate_workspace().unwrap();
    reset_endpoint_reduction_log();
    overlap_plan
        .inverse_with_overlap(
            &overlap_source,
            &mut overlap_destination,
            &mut overlap_workspace,
        )
        .unwrap();
    chunk_protocol(batch_count);
    reset_endpoint_reduction_log();
    overlap_plan
        .backward_with_overlap(
            &overlap_source,
            &mut overlap_destination,
            &mut overlap_workspace,
        )
        .unwrap();
    chunk_protocol(batch_count);

    let mut source = plan.allocate_output().unwrap();
    source.as_mut_slice().fill(zero_complex::<R>());
    for i in 0..4 {
        if let Some(value) = source.get_global_mut(&[batch_count - 1], [i, 0]) {
            *value = Complex::new(R::zero(), R::from_f64(1e30).unwrap());
        }
    }
    let source_before = source.as_slice().to_vec();
    destination.as_mut_slice().fill(R::from_f64(7.0).unwrap());
    let destination_before = destination.as_slice().to_vec();
    reset_endpoint_reduction_log();
    let result = plan.inverse(&source, &mut destination, &mut workspace);
    assert!(
        matches!(result, Err(R2cError::InvalidSpectrum)),
        "{result:?}"
    );
    chunk_protocol(batch_count);
    assert_eq!(source.as_slice(), source_before);
    assert_eq!(destination.as_slice(), destination_before);

    let mut array = plan.allocate_in_place().unwrap();
    let mut in_place_workspace = plan.allocate_in_place_workspace().unwrap();
    plan.forward_in_place(&mut array, &mut in_place_workspace)
        .unwrap();
    {
        let mut view = array.complex_view_mut().unwrap();
        view.as_mut_slice().fill(zero_complex::<R>());
        for i in 0..4 {
            if let Some(value) = view.get_global_mut(&[batch_count - 1], [i, 0]) {
                *value = Complex::new(R::zero(), R::from_f64(1e30).unwrap());
            }
        }
    }
    reset_endpoint_reduction_log();
    let result = plan.inverse_in_place(&mut array, &mut in_place_workspace);
    assert!(
        matches!(result, Err(R2cError::InvalidSpectrum)),
        "{result:?}"
    );
    chunk_protocol(batch_count);
    assert_eq!(array.state(), crate::R2cState::Poisoned);
}

// The overlap route can validate an intermediate whose reduced axis is
// split across ranks. Build that layout directly so ranks without any
// endpoint and ranks with an empty local region still join every chunk.
fn distributed_reduced_axis<R: FftReal>(topology: &Arc<MpiTopology<1>>)
where
    Complex<R>: Equivalence,
{
    let chunk = ENDPOINT_BATCH_CHUNK;
    let batch_count = chunk + 1;
    let plan = chunk_plan::<R, 1>(topology, batch_count, 6);
    let pencil = Pencil::<2, 1>::new(Arc::clone(topology), [4, 4], [1]).unwrap();
    let mut array = ManyPencilArray::<Complex<R>, 2, 1>::from_elem(
        vec![pencil],
        0,
        plan.core.extra_shape.clone(),
        zero_complex::<R>(),
    )
    .unwrap();
    reset_endpoint_reduction_log();
    validate_boundary(&plan.core, &array, true, plan.raw_absolute_threshold).unwrap();
    chunk_protocol(batch_count);

    let (local_len, stride, local_axis_len, global_axis_start) = {
        let view = array.active_view().unwrap();
        let pencil = view.pencil();
        let boundary = plan.core.real_stage_index.unwrap();
        let reduction_axis = plan.core.stages[boundary].axis;
        (
            pencil.local_len(),
            super::super::memory_stride(pencil, reduction_axis).unwrap(),
            pencil.local_shape_logical()[reduction_axis],
            pencil.local_ranges()[reduction_axis].start,
        )
    };
    {
        let mut view = array.active_view_mut().unwrap();
        let data = view.as_mut_slice();
        for batch in [0, batch_count - 1] {
            let start = batch * local_len;
            for offset in 0..local_len {
                let coord = (offset / stride) % local_axis_len;
                if global_axis_start + coord == 0 {
                    data[start + offset] = Complex::new(R::zero(), R::from_f64(1e30).unwrap());
                }
            }
        }
    }
    reset_endpoint_reduction_log();
    let result = validate_boundary(&plan.core, &array, true, plan.raw_absolute_threshold);
    assert!(
        matches!(result, Err(R2cError::InvalidSpectrum)),
        "{result:?}"
    );
    chunk_protocol(batch_count);
}

// The overlap route validates outside the receive callback, so a
// preparation failure surfaces as an operation error on every rank: the
// injecting rank keeps its own error and its peers report the collective
// precondition.
fn overlap_preparation_failures<R: FftReal>(topology: &Arc<MpiTopology<1>>, rank: usize)
where
    Complex<R>: Equivalence,
{
    let chunk = ENDPOINT_BATCH_CHUNK;
    let batch_count = chunk + 1;
    let plan = R2cPlan::<R, 2, 1>::from_shape_with_layout(
        Arc::clone(topology),
        [4, 6],
        ExtraShape::new(vec![batch_count]).unwrap(),
        DistributedLayout {
            transpose_method: TransposeMethod::PointToPoint,
            permute_dims: false,
        },
    )
    .unwrap();
    let mut source = plan.allocate_output().unwrap();
    source.as_mut_slice().fill(zero_complex::<R>());
    let source_before = source.as_slice().to_vec();
    let mut destination = plan.allocate_input().unwrap();
    destination.as_mut_slice().fill(R::from_f64(7.0).unwrap());
    let destination_before = destination.as_slice().to_vec();
    for slot in 0..3 {
        let mut workspace = plan.allocate_workspace().unwrap();
        reset_endpoint_reduction_log();
        if rank == 0 {
            force_endpoint_allocation_failure(slot);
        }
        let error = plan
            .inverse_with_overlap(&source, &mut destination, &mut workspace)
            .unwrap_err();
        match (rank, error) {
            (
                0,
                FftOverlapError::Operation(R2cError::Fft(FftError::AllocationFailed { required })),
            ) => assert_eq!(required, chunk, "slot={slot}"),
            (
                _,
                FftOverlapError::Operation(R2cError::Fft(FftError::CollectivePreconditionFailed)),
            ) => {}
            (_, error) => {
                panic!("unexpected overlap allocation failure for slot {slot}: {error:?}")
            }
        }
        let log = endpoint_reduction_log();
        assert_no_statistics_reduction(&log);
        assert_eq!(
            log.iter()
                .filter(|event| matches!(event, EndpointReductionEvent::Allocation { .. }))
                .count(),
            if rank == 0 { slot + 1 } else { 3 },
            "slot={slot}: {log:?}"
        );
        assert_eq!(source.as_slice(), source_before);
    }
    // The last batch's DC plane rejects the whole overlap call and keeps
    // the source and real destination.
    let mut source = plan.allocate_output().unwrap();
    source.as_mut_slice().fill(zero_complex::<R>());
    for i in 0..4 {
        if let Some(value) = source.get_global_mut(&[batch_count - 1], [i, 0]) {
            *value = Complex::new(R::zero(), R::from_f64(1e30).unwrap());
        }
    }
    let source_before = source.as_slice().to_vec();
    for raw in [false, true] {
        let mut workspace = plan.allocate_workspace().unwrap();
        destination.as_mut_slice().fill(R::from_f64(7.0).unwrap());
        reset_endpoint_reduction_log();
        let error = if raw {
            plan.backward_with_overlap(&source, &mut destination, &mut workspace)
                .unwrap_err()
        } else {
            plan.inverse_with_overlap(&source, &mut destination, &mut workspace)
                .unwrap_err()
        };
        assert!(
            matches!(error, FftOverlapError::Operation(R2cError::InvalidSpectrum)),
            "{error:?}"
        );
        chunk_protocol(batch_count);
        assert_eq!(source.as_slice(), source_before);
        assert_eq!(destination.as_slice(), destination_before);
    }
}

#[test]
#[ignore = "MPI must be initialized in a separate test process"]
fn endpoint_reductions_are_chunked() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size() as usize;
    assert!(matches!(size, 1 | 4 | 6));
    let rank = usize::try_from(world.rank()).unwrap();
    let topology = pencil_array::MpiTopology::<1>::new(&world, [size]).unwrap();
    chunked_statistics::<f64, 1>(&topology);
    multi_contribution_probes::<f32, 1>(&topology);
    multi_contribution_probes::<f64, 1>(&topology);
    invalid_batches::<f64, 1>(&topology);
    plane_and_batch_separation::<f64, 1>(&topology);
    allocation_and_preparation_failures::<f64, 1>(&topology, rank);
    public_routes::<f64>(&topology);
    distributed_reduced_axis::<f64>(&topology);
    overlap_preparation_failures::<f64>(&topology, rank);
}
