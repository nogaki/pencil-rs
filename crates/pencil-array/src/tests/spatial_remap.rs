use super::{
    OwnerRoutes, RequestRoute, SpatialRemapError, SpatialRemapPlan, affine_coordinate,
    local_coordinate, partition_owner, payload_layout, receive_layout, validate_partition_layout,
};
use mpi::Count;

fn count_max() -> usize {
    usize::try_from(Count::MAX).unwrap()
}

fn routes_with_counts(counts: &[Count]) -> OwnerRoutes {
    let mut displacements = Vec::new();
    let mut next = 0usize;
    for &count in counts {
        displacements.push(Count::try_from(next).unwrap());
        next += usize::try_from(count).unwrap();
    }
    OwnerRoutes {
        entries: (0..next)
            .map(|index| RequestRoute {
                owner: 0,
                global_index: u64::try_from(index).unwrap(),
                destination_offset: index,
            })
            .collect(),
        counts: counts.to_vec(),
        displacements,
    }
}

#[test]
fn affine_mapping_handles_full_width_offsets_and_signs() {
    let shape = [5usize, 7];
    assert_eq!(
        affine_coordinate([0, 6], shape, &[1, -1], &[i128::MAX, i128::MIN]).unwrap(),
        [2, 6],
    );
    assert_eq!(
        affine_coordinate([4, 3], shape, &[-1, 1], &[i128::MIN, i128::MAX]).unwrap(),
        [3, 4],
    );
}

#[test]
fn floor_partition_owner_handles_empty_partitions() {
    assert_eq!(partition_owner(0, 2, 4).unwrap(), 1);
    assert_eq!(partition_owner(1, 2, 4).unwrap(), 3);
    assert_eq!(partition_owner(8, 10, 3).unwrap(), 2);
    assert!(matches!(
        partition_owner(2, 2, 3),
        Err(SpatialRemapError::StructuralInconsistency),
    ));
}

#[test]
fn count_layout_checks_packed_nonoverlapping_segments() {
    assert_eq!(
        validate_partition_layout(&[2, 0, 3], &[0, 2, 2], 5).unwrap(),
        5
    );
    assert!(matches!(
        validate_partition_layout(&[2, 2], &[0, 1], 4),
        Err(SpatialRemapError::StructuralInconsistency),
    ));
    assert_eq!(
        validate_partition_layout(&[Count::MAX], &[0], count_max()).unwrap(),
        count_max(),
    );
}

#[test]
fn count_layout_accepts_total_above_count_max() {
    let total = count_max() + 1;
    assert_eq!(
        validate_partition_layout(&[Count::MAX, 1], &[0, Count::MAX], total).unwrap(),
        total,
    );
}

#[test]
fn count_layout_rejects_invalid_metadata() {
    let cases: &[(&[Count], &[Count], usize)] = &[
        (&[1], &[], 0),
        (&[], &[0], 0),
        (&[-1], &[0], 0),
        (&[0], &[-1], 0),
        (&[2], &[1], 4),
        (&[2, 1], &[0, 1], 4),
        (&[1, 2], &[0, 4], 5),
        (&[2], &[0], 3),
    ];
    for &(counts, displacements, logical_len) in cases {
        assert!(
            matches!(
                validate_partition_layout(counts, displacements, logical_len),
                Err(SpatialRemapError::StructuralInconsistency)
                    | Err(SpatialRemapError::MpiCountOverflow),
            ),
            "accepted counts {counts:?} displacements {displacements:?}",
        );
    }
}

#[test]
fn receive_layout_accepts_total_above_count_max() {
    let total = count_max() + 1;
    let layout = receive_layout(&[Count::MAX, 1], total).unwrap();
    assert_eq!(layout.displacements, [0, Count::MAX]);
    assert_eq!(layout.total, total);
}

#[test]
fn receive_layout_rejects_displacement_above_count_max() {
    assert!(matches!(
        receive_layout(&[Count::MAX, 1, 0], count_max() + 1),
        Err(SpatialRemapError::MpiCountOverflow),
    ));
}

#[test]
fn payload_layout_accepts_aggregate_above_count_max() {
    let extra = Count::MAX as usize / 2 + 1;
    let extra_count = Count::try_from(extra).unwrap();
    let routes = routes_with_counts(&[1, 1]);
    let payload = payload_layout::<u8>(&routes, &[1, 1], &[0, 1], 2, 2, extra, 2).unwrap();
    assert_eq!(payload.send_len, count_max() + 1);
    assert_eq!(payload.receive_len, count_max() + 1);
    assert_eq!(payload.send_counts, [extra_count, extra_count]);
    assert_eq!(payload.send_displacements, [0, extra_count]);
    assert_eq!(payload.receive_counts, [extra_count, extra_count]);
    assert_eq!(payload.receive_displacements, [0, extra_count]);
}

#[test]
fn payload_layout_rejects_peer_count_above_count_max() {
    let routes = routes_with_counts(&[1]);
    assert!(matches!(
        payload_layout::<u8>(&routes, &[1], &[0], 1, 1, count_max() + 1, 1),
        Err(SpatialRemapError::MpiCountOverflow),
    ));
}

#[test]
fn payload_layout_rejects_peer_displacement_above_count_max() {
    let extra = Count::MAX as usize / 2 + 1;
    let routes = routes_with_counts(&[1, 1, 1]);
    assert!(matches!(
        payload_layout::<u8>(&routes, &[1, 1, 1], &[0, 1, 2], 3, 3, extra, 3),
        Err(SpatialRemapError::MpiCountOverflow),
    ));
}

#[test]
fn payload_layout_rejects_byte_count_overflow() {
    let routes = routes_with_counts(&[1]);
    assert!(matches!(
        payload_layout::<u16>(&routes, &[1], &[0], 1, 1, usize::MAX, 1),
        Err(SpatialRemapError::SizeOverflow),
    ));
}

#[test]
fn payload_layout_rejects_element_count_overflow() {
    let routes = routes_with_counts(&[1, 0]);
    let huge_source_len = count_max().checked_mul(2).unwrap();
    assert!(matches!(
        payload_layout::<u8>(
            &routes,
            &[Count::MAX, Count::MAX],
            &[0, Count::MAX],
            huge_source_len,
            1,
            usize::MAX / 2 + 1,
            2,
        ),
        Err(SpatialRemapError::SizeOverflow),
    ));
}

#[test]
#[ignore = "MPI contract test; run under mpiexec with --ignored"]
fn mpi_preflight_failure_preserves_all_plan_state() {
    use std::sync::Arc;

    use mpi::{raw::traits::AsRaw, traits::*};

    use crate::{
        AxisPermutation, ExtraShape, MpiTopology, Pencil, PencilArrayView, PencilArrayViewMut,
    };

    let universe = mpi::initialize().expect("MPI must not already be initialized");
    let world = universe.world();
    let size = usize::try_from(world.size()).unwrap();
    let topology = MpiTopology::<1>::new(&world, [size]).unwrap();
    let source = Pencil::<2, 1>::new(Arc::clone(&topology), [7, 5], [0]).unwrap();
    let destination = Pencil::<2, 1>::new_permuted(
        Arc::clone(&topology),
        [7, 5],
        [0],
        AxisPermutation::new([1, 0]).unwrap(),
    )
    .unwrap();
    let extra = ExtraShape::new([2, 3]).unwrap();
    let signs = [1, -1];
    let offsets = [2, -3];
    let mut plan = SpatialRemapPlan::<i64, 2, 1>::prepare(
        Arc::clone(&source),
        Arc::clone(&destination),
        extra.clone(),
        signs,
        offsets,
        0,
    )
    .unwrap();
    let mut source_data = Vec::with_capacity(source.local_len() * extra.element_count());
    for extra_index in 0..extra.element_count() {
        for memory_index in 0..source.local_len() {
            let local = local_coordinate(&source, memory_index).unwrap();
            let global = [
                source.local_ranges()[0].start + local[0],
                source.local_ranges()[1].start + local[1],
            ];
            let global_linear = global[0] * source.global_shape()[1] + global[1];
            source_data.push(
                i64::try_from(extra_index * source.global_len() + global_linear + 1).unwrap(),
            );
        }
    }
    let mut destination_data = vec![91i64; destination.local_len() * extra.element_count()];
    let wrong_destination = Pencil::<2, 1>::new(Arc::clone(&topology), [7, 5], [0]).unwrap();
    let verify_oracle = |actual: &[i64]| {
        for extra_index in 0..extra.element_count() {
            for memory_index in 0..destination.local_len() {
                let local = local_coordinate(&destination, memory_index).unwrap();
                let destination_global = [
                    destination.local_ranges()[0].start + local[0],
                    destination.local_ranges()[1].start + local[1],
                ];
                let source_global =
                    affine_coordinate(destination_global, *source.global_shape(), &signs, &offsets)
                        .unwrap();
                let source_linear = source_global[0] * source.global_shape()[1] + source_global[1];
                let expected =
                    i64::try_from(extra_index * source.global_len() + source_linear + 1).unwrap();
                assert_eq!(
                    actual[extra_index * destination.local_len() + memory_index],
                    expected,
                );
            }
        }
    };

    assert!(source_data.iter().all(|value| *value != 0));
    let source_view = PencilArrayView::from_slice(&source, &extra, &source_data).unwrap();
    let destination_view =
        PencilArrayViewMut::from_slice_mut(&destination, &extra, &mut destination_data).unwrap();
    plan.apply_views(source_view, destination_view).unwrap();
    verify_oracle(&destination_data);
    assert!(plan.send_buffer.iter().any(|value| *value != 0));
    assert!(plan.receive_buffer.iter().any(|value| *value != 0));

    let send_before = plan.send_buffer.clone();
    let receive_before = plan.receive_buffer.clone();
    let descriptor_before = plan.descriptor.clone();
    let route_state_before = (
        plan.request_send_counts.clone(),
        plan.request_send_displacements.clone(),
        plan.request_receive_counts.clone(),
        plan.request_receive_displacements.clone(),
        plan.destination_offsets.clone(),
        plan.source_offsets.clone(),
        plan.payload_send_counts.clone(),
        plan.payload_send_displacements.clone(),
        plan.payload_receive_counts.clone(),
        plan.payload_receive_displacements.clone(),
    );
    let metadata_before = (
        plan.extra_shape.clone(),
        plan.signs,
        plan.offsets,
        plan.send_len,
        plan.receive_len,
        plan.datatype.as_raw(),
    );
    let source_before = source_data.clone();
    let destination_before = destination_data.clone();

    let source_view = PencilArrayView::from_slice(&source, &extra, &source_data).unwrap();
    let destination_view = if topology.rank() == 0 {
        PencilArrayViewMut::from_slice_mut(&wrong_destination, &extra, &mut destination_data)
            .unwrap()
    } else {
        PencilArrayViewMut::from_slice_mut(&destination, &extra, &mut destination_data).unwrap()
    };
    let error = plan.apply_views(source_view, destination_view).unwrap_err();
    if topology.rank() == 0 {
        assert!(matches!(error, SpatialRemapError::LayoutMismatch));
    } else {
        assert!(matches!(
            error,
            SpatialRemapError::CollectivePreconditionFailed
        ));
    }
    assert_eq!(plan.send_buffer, send_before);
    assert_eq!(plan.receive_buffer, receive_before);
    assert_eq!(plan.descriptor, descriptor_before);
    assert_eq!(
        (
            plan.request_send_counts.clone(),
            plan.request_send_displacements.clone(),
            plan.request_receive_counts.clone(),
            plan.request_receive_displacements.clone(),
            plan.destination_offsets.clone(),
            plan.source_offsets.clone(),
            plan.payload_send_counts.clone(),
            plan.payload_send_displacements.clone(),
            plan.payload_receive_counts.clone(),
            plan.payload_receive_displacements.clone(),
        ),
        route_state_before,
    );
    assert_eq!(
        (
            plan.extra_shape.clone(),
            plan.signs,
            plan.offsets,
            plan.send_len,
            plan.receive_len,
            plan.datatype.as_raw(),
        ),
        metadata_before,
    );
    assert!(Arc::ptr_eq(&plan.source, &source));
    assert!(Arc::ptr_eq(&plan.destination, &destination));
    assert_eq!(source_data, source_before);
    assert_eq!(destination_data, destination_before);

    let source_view = PencilArrayView::from_slice(&source, &extra, &source_data).unwrap();
    let destination_view =
        PencilArrayViewMut::from_slice_mut(&destination, &extra, &mut destination_data).unwrap();
    plan.apply_views(source_view, destination_view).unwrap();
    verify_oracle(&destination_data);
}
