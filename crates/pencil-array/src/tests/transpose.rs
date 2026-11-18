use super::*;

fn column_major_offset<const N: usize>(shape: &[usize; N], indices: &[usize; N]) -> Option<usize> {
    let mut offset = 0usize;
    let mut stride = 1usize;
    for axis in 0..N {
        if indices[axis] >= shape[axis] {
            return None;
        }
        offset = offset.checked_add(indices[axis].checked_mul(stride)?)?;
        if axis + 1 < N {
            stride = stride.checked_mul(shape[axis])?;
        }
    }
    Some(offset)
}

fn column_major_region_index<const N: usize>(
    region: &[Range<usize>; N],
    mut linear: usize,
) -> [usize; N] {
    let mut index = [0; N];
    for axis in 0..N {
        let extent = region[axis].end - region[axis].start;
        assert_ne!(extent, 0, "reference index requires a nonempty region");
        index[axis] = region[axis].start + linear % extent;
        linear /= extent;
    }
    index
}

/// Local pencil metadata that the offset validator needs.
struct RegionMetadata<const N: usize> {
    local_ranges: [Range<usize>; N],
    local_shape_memory: [usize; N],
    permutation: AxisPermutation<N>,
    local_len: usize,
}

/// Element-by-element copy of the pre-metadata validator. It stays test-only
/// so production preflight never scans spatial_len * extra_count elements.
fn validate_region_offsets_reference<const N: usize>(
    local_ranges: &[Range<usize>; N],
    local_shape_memory: &[usize; N],
    permutation: &AxisPermutation<N>,
    local_len: usize,
    storage_len: usize,
    region: &[Range<usize>; N],
    extra_count: usize,
) -> Result<(), TransposeError> {
    if extra_count == 0 {
        return Ok(());
    }
    let spatial_len = region_spatial_len(region)?;
    if spatial_len == 0 {
        return Ok(());
    }

    for extra_linear in 0..extra_count {
        let extra_offset = extra_linear
            .checked_mul(local_len)
            .ok_or(TransposeError::PreparationFailed)?;
        for linear in 0..spatial_len {
            let global = column_major_region_index(region, linear);
            let mut local = [0usize; N];
            for axis in 0..N {
                let range = &local_ranges[axis];
                if global[axis] < range.start || global[axis] >= range.end {
                    return Err(TransposeError::PreparationFailed);
                }
                local[axis] = global[axis] - range.start;
            }
            let memory = permutation.permute(local);
            let spatial_offset = column_major_offset(local_shape_memory, &memory)
                .ok_or(TransposeError::PreparationFailed)?;
            let offset = extra_offset
                .checked_add(spatial_offset)
                .ok_or(TransposeError::PreparationFailed)?;
            if offset >= storage_len {
                return Err(TransposeError::PreparationFailed);
            }
        }
    }
    Ok(())
}

/// Runs both validators, asserts that they agree, and returns the
/// production result.
fn checked_offsets<const N: usize>(
    metadata: &RegionMetadata<N>,
    region: &[Range<usize>; N],
    extra_count: usize,
    storage_len: usize,
) -> Result<(), TransposeError> {
    let actual = validate_region_offsets(
        &metadata.local_ranges,
        &metadata.local_shape_memory,
        &metadata.permutation,
        metadata.local_len,
        storage_len,
        region,
        extra_count,
    );
    let reference = validate_region_offsets_reference(
        &metadata.local_ranges,
        &metadata.local_shape_memory,
        &metadata.permutation,
        metadata.local_len,
        storage_len,
        region,
        extra_count,
    );
    assert_eq!(
        actual,
        reference,
        "region {region:?}, extra_count {extra_count}, storage_len {storage_len}, permutation {:?}",
        metadata.permutation.axes(),
    );
    actual
}

fn all_permutations<const N: usize>() -> Vec<AxisPermutation<N>> {
    fn fill<const N: usize>(
        position: usize,
        current: &mut [usize; N],
        used: &mut [bool; N],
        out: &mut Vec<[usize; N]>,
    ) {
        if position == N {
            out.push(*current);
            return;
        }
        for axis in 0..N {
            if !used[axis] {
                used[axis] = true;
                current[position] = axis;
                fill(position + 1, current, used, out);
                used[axis] = false;
            }
        }
    }
    let mut axes = Vec::new();
    fill(0, &mut [0usize; N], &mut [false; N], &mut axes);
    axes.into_iter()
        .map(|axes| AxisPermutation::new(axes).expect("valid permutation"))
        .collect()
}

fn region_candidates(local: &Range<usize>) -> [Range<usize>; 7] {
    let (start, end) = (local.start, local.end);
    [
        start..end,
        start + 1..end,
        start..end - 1,
        end + 1..end + 1,
        start - 1..end,
        start..end + 1,
        end..start,
    ]
}

fn check_exhaustive_against_reference<const N: usize>() {
    let local_ranges: [Range<usize>; N] = std::array::from_fn(|axis| {
        let start = 3 + 2 * axis;
        start..start + 2 + axis
    });
    let logical_shape: [usize; N] = std::array::from_fn(|axis| 2 + axis);
    let local_len = logical_shape.iter().product::<usize>();
    let candidates: Vec<[Range<usize>; 7]> = local_ranges.iter().map(region_candidates).collect();

    for permutation in all_permutations::<N>() {
        let metadata = RegionMetadata {
            local_ranges: local_ranges.clone(),
            local_shape_memory: permutation.permute(logical_shape),
            permutation,
            local_len,
        };
        for code in 0..7usize.pow(N as u32) {
            let mut remainder = code;
            let region: [Range<usize>; N] = std::array::from_fn(|axis| {
                let choice = remainder % 7;
                remainder /= 7;
                candidates[axis][choice].clone()
            });
            for extra_count in 0..=3 {
                let max_storage = local_len * extra_count + 1;
                for storage_len in 0..=max_storage {
                    let _ = checked_offsets(&metadata, &region, extra_count, storage_len);
                }
            }
        }
    }
}

#[test]
fn region_offsets_match_elementwise_reference() {
    check_exhaustive_against_reference::<1>();
    check_exhaustive_against_reference::<2>();
    check_exhaustive_against_reference::<3>();
}

#[test]
fn region_offsets_preserve_zero_and_empty_cases() {
    let metadata = RegionMetadata {
        local_ranges: [5..8, 0..4],
        local_shape_memory: [4, 3],
        permutation: AxisPermutation::new([1, 0]).expect("valid permutation"),
        local_len: 12,
    };
    let cases = [
        ([6..8, 1..4], 2, 24, Ok(())),
        ([6..8, 1..4], 2, 23, Err(TransposeError::PreparationFailed)),
        ([4..7, 1..4], 2, 24, Err(TransposeError::PreparationFailed)),
        ([9..9, 0..4], 2, 0, Ok(())),
        ([Range { start: 8, end: 7 }, 0..4], 0, 0, Ok(())),
        (
            [9..9, Range { start: 4, end: 3 }],
            1,
            0,
            Err(TransposeError::PreparationFailed),
        ),
    ];
    for (region, extra_count, storage_len, expected) in cases {
        assert_eq!(
            checked_offsets(&metadata, &region, extra_count, storage_len),
            expected,
            "region {region:?}, extra_count {extra_count}, storage_len {storage_len}",
        );
    }
}

// clippy's single_range_in_vec_init misreads one-axis `[Range; 1]` arrays.
#[allow(clippy::single_range_in_vec_init)]
#[test]
fn region_offsets_check_storage_and_usize_boundaries() {
    let maximum = usize::MAX;
    let half = maximum / 2 + 1;

    let tall = RegionMetadata {
        local_ranges: [0..half],
        local_shape_memory: [half],
        permutation: AxisPermutation::identity(),
        local_len: half,
    };
    assert_eq!(checked_offsets(&tall, &[0..1], 2, half + 1), Ok(()));
    assert_eq!(
        checked_offsets(&tall, &[0..1], 3, maximum),
        Err(TransposeError::PreparationFailed),
    );

    let huge_local = RegionMetadata {
        local_ranges: [0..maximum],
        local_shape_memory: [maximum],
        permutation: AxisPermutation::identity(),
        local_len: maximum,
    };
    assert_eq!(
        checked_offsets(&huge_local, &[1..2], 2, maximum),
        Err(TransposeError::PreparationFailed),
    );

    let single = RegionMetadata {
        local_ranges: [0..1],
        local_shape_memory: [1],
        permutation: AxisPermutation::identity(),
        local_len: 1,
    };
    assert_eq!(checked_offsets(&single, &[0..1], 1, 1), Ok(()));
    assert_eq!(
        checked_offsets(&single, &[0..1], 1, 0),
        Err(TransposeError::PreparationFailed),
    );

    let half = maximum / 2;
    let column_major_multiplication = RegionMetadata {
        local_ranges: [0..half + 1, 0..3],
        local_shape_memory: [half + 1, 3],
        permutation: AxisPermutation::identity(),
        local_len: maximum,
    };
    assert_eq!(
        checked_offsets(&column_major_multiplication, &[0..1, 2..3], 1, maximum),
        Err(TransposeError::PreparationFailed),
    );

    let column_major_addition = RegionMetadata {
        local_ranges: [0..half, 0..3],
        local_shape_memory: [half, 3],
        permutation: AxisPermutation::identity(),
        local_len: maximum,
    };
    assert_eq!(
        checked_offsets(&column_major_addition, &[2..3, 2..3], 1, maximum),
        Err(TransposeError::PreparationFailed),
    );

    let region_product = RegionMetadata {
        local_ranges: [0..maximum, 0..2],
        local_shape_memory: [maximum, 2],
        permutation: AxisPermutation::identity(),
        local_len: 1,
    };
    assert_eq!(
        checked_offsets(&region_product, &[0..maximum, 0..2], 1, maximum),
        Err(TransposeError::PreparationFailed),
    );

    let zero_extent = RegionMetadata {
        local_ranges: [0..2, 0..2],
        local_shape_memory: [0, 2],
        permutation: AxisPermutation::identity(),
        local_len: 1,
    };
    assert_eq!(
        checked_offsets(&zero_extent, &[0..1, 0..1], 1, maximum),
        Err(TransposeError::PreparationFailed),
    );

    let short_memory = RegionMetadata {
        local_ranges: [0..2, 0..2],
        local_shape_memory: [1, 2],
        permutation: AxisPermutation::identity(),
        local_len: 1,
    };
    assert_eq!(
        checked_offsets(&short_memory, &[1..2, 0..1], 1, maximum),
        Err(TransposeError::PreparationFailed),
    );

    let zero_product = RegionMetadata {
        local_ranges: [0..maximum, 0..2, 0..0],
        local_shape_memory: [maximum, 2, 0],
        permutation: AxisPermutation::identity(),
        local_len: 0,
    };
    assert_eq!(
        checked_offsets(&zero_product, &[0..maximum, 0..2, 0..0], 1, 0),
        Ok(()),
    );
}

#[test]
fn region_offsets_large_region_uses_metadata() {
    let shape = [512usize, 512, 512];
    let local_len = 512usize.pow(3);
    let local_ranges: [Range<usize>; 3] = std::array::from_fn(|axis| 0..shape[axis]);
    let permutation = AxisPermutation::<3>::identity();
    let storage_len = 3 * local_len;
    let region = local_ranges.clone();
    assert_eq!(
        validate_region_offsets(
            &local_ranges,
            &shape,
            &permutation,
            local_len,
            storage_len,
            &region,
            3,
        ),
        Ok(()),
    );
}

/// Element-by-element copy of the pre-optimisation pack mapping. It stays
/// test-only so the odometer walk can be checked against the mapping it
/// replaced.
fn reference_region_offsets<const N: usize>(
    local_ranges: &[Range<usize>; N],
    shape_memory: &[usize; N],
    permutation: &AxisPermutation<N>,
    region: &[Range<usize>; N],
) -> Vec<usize> {
    let spatial_len = region_spatial_len(region).expect("valid region");
    (0..spatial_len)
        .map(|linear| {
            let global = column_major_region_index(region, linear);
            let mut local = [0usize; N];
            for axis in 0..N {
                let range = &local_ranges[axis];
                assert!(range.contains(&global[axis]), "region inside the pencil");
                local[axis] = global[axis] - range.start;
            }
            let memory = permutation.permute(local);
            column_major_offset(shape_memory, &memory).expect("valid memory index")
        })
        .collect()
}

#[test]
fn pack_and_unpack_match_elementwise_reference() {
    let local_ranges: [Range<usize>; 3] = [1..4, 1..3, 2..5];
    let logical_shape: [usize; 3] =
        std::array::from_fn(|axis| local_ranges[axis].end - local_ranges[axis].start);
    let local_len = logical_shape.iter().product::<usize>();

    // Regions start at non-zero global offsets and include a length-1 last
    // axis and empty regions, which pack and unpack skip.
    let send_regions = [
        [1..4, 1..3, 2..5],
        [2..4, 1..3, 2..4],
        [1..2, 2..3, 3..4],
        [2..3, 1..2, 2..2],
    ];
    let receive_regions = [
        [1..3, 1..3, 2..5],
        [2..4, 2..3, 3..5],
        [1..4, 1..2, 3..4],
        [1..3, 1..3, 4..4],
    ];

    let mut peers = Vec::new();
    let mut send_displacement = 0usize;
    let mut receive_displacement = 0usize;
    for (index, (send_region, receive_region)) in
        send_regions.iter().zip(&receive_regions).enumerate()
    {
        let send_spatial_len = region_spatial_len(send_region).expect("valid region");
        let receive_spatial_len = region_spatial_len(receive_region).expect("valid region");
        peers.push(PeerMetadata {
            peer_rank: Count::try_from(index).expect("peer rank fits an MPI count"),
            peer_coordinate: index,
            send_region: send_region.clone(),
            receive_region: receive_region.clone(),
            send_spatial_len,
            receive_spatial_len,
            send_spatial_displacement: send_displacement,
            receive_spatial_displacement: receive_displacement,
        });
        send_displacement += send_spatial_len;
        receive_displacement += receive_spatial_len;
    }

    for permutation in all_permutations::<3>() {
        let shape_memory = permutation.permute(logical_shape);
        let layout = LocalLayout {
            local_ranges: &local_ranges,
            shape_memory,
            permutation: &permutation,
            local_len,
        };
        for extra_count in [1usize, 3] {
            let source: Vec<u32> = (0..u32::try_from(local_len * extra_count).unwrap()).collect();
            let send_len = send_displacement * extra_count;
            let mut send_buffer = vec![0u32; send_len];
            let mut expected_send = vec![0u32; send_len];
            for peer in &peers {
                if peer.send_spatial_len == 0 {
                    continue;
                }
                let offsets = reference_region_offsets(
                    &local_ranges,
                    &shape_memory,
                    &permutation,
                    &peer.send_region,
                );
                for extra_linear in 0..extra_count {
                    for (linear, &offset) in offsets.iter().enumerate() {
                        let payload = peer.send_spatial_displacement * extra_count
                            + extra_linear * peer.send_spatial_len
                            + linear;
                        expected_send[payload] = source[extra_linear * local_len + offset];
                    }
                }
            }
            pack_source_regions(
                &peers,
                &layout,
                &source,
                &mut send_buffer,
                send_len,
                extra_count,
            );
            assert_eq!(
                send_buffer,
                expected_send,
                "pack permutation {:?}, extra_count {extra_count}",
                permutation.axes(),
            );

            let receive_len = receive_displacement * extra_count;
            let receive_buffer: Vec<u32> = (0..u32::try_from(receive_len).unwrap())
                .map(|value| 1000 + value)
                .collect();
            let mut storage = vec![0u32; local_len * extra_count];
            let mut expected_storage = vec![0u32; local_len * extra_count];
            for peer in &peers {
                if peer.receive_spatial_len == 0 {
                    continue;
                }
                let offsets = reference_region_offsets(
                    &local_ranges,
                    &shape_memory,
                    &permutation,
                    &peer.receive_region,
                );
                for extra_linear in 0..extra_count {
                    for (linear, &offset) in offsets.iter().enumerate() {
                        let payload = peer.receive_spatial_displacement * extra_count
                            + extra_linear * peer.receive_spatial_len
                            + linear;
                        expected_storage[extra_linear * local_len + offset] =
                            receive_buffer[payload];
                    }
                }
            }
            unpack_destination_regions(&peers, &layout, &mut storage, &receive_buffer, extra_count);
            assert_eq!(
                storage,
                expected_storage,
                "unpack permutation {:?}, extra_count {extra_count}",
                permutation.axes(),
            );
        }
    }
}

#[test]
fn self_region_copy_matches_elementwise_reference() {
    let local_ranges: [Range<usize>; 3] = [1..4, 2..6, 0..3];
    let logical_shape: [usize; 3] =
        std::array::from_fn(|axis| local_ranges[axis].end - local_ranges[axis].start);
    let local_len = logical_shape.iter().product::<usize>();
    // The self peer's send and receive regions are the same global region.
    let region: [Range<usize>; 3] = [2..4, 2..5, 1..3];

    for source_permutation in all_permutations::<3>() {
        let source_shape_memory = source_permutation.permute(logical_shape);
        let source_layout = LocalLayout {
            local_ranges: &local_ranges,
            shape_memory: source_shape_memory,
            permutation: &source_permutation,
            local_len,
        };
        for destination_permutation in all_permutations::<3>() {
            let destination_shape_memory = destination_permutation.permute(logical_shape);
            let destination_layout = LocalLayout {
                local_ranges: &local_ranges,
                shape_memory: destination_shape_memory,
                permutation: &destination_permutation,
                local_len,
            };
            for extra_count in [1usize, 3] {
                let source: Vec<u32> =
                    (0..u32::try_from(local_len * extra_count).unwrap()).collect();
                let mut destination = vec![u32::MAX; local_len * extra_count];
                let mut expected = vec![u32::MAX; local_len * extra_count];
                let source_offsets = reference_region_offsets(
                    &local_ranges,
                    &source_shape_memory,
                    &source_permutation,
                    &region,
                );
                let destination_offsets = reference_region_offsets(
                    &local_ranges,
                    &destination_shape_memory,
                    &destination_permutation,
                    &region,
                );
                for extra_linear in 0..extra_count {
                    for (&source_offset, &destination_offset) in
                        source_offsets.iter().zip(&destination_offsets)
                    {
                        expected[extra_linear * local_len + destination_offset] =
                            source[extra_linear * local_len + source_offset];
                    }
                }
                copy_self_region_regions(
                    &source_layout,
                    &region,
                    &source,
                    &destination_layout,
                    &region,
                    &mut destination,
                    extra_count,
                );
                assert_eq!(
                    destination,
                    expected,
                    "self copy source {:?}, destination {:?}, extra_count {extra_count}",
                    source_permutation.axes(),
                    destination_permutation.axes(),
                );
            }
        }
    }
}

/// A length-1 axis contributes `0 * stride` to the region's span, so its
/// stride is not bounded by `max_offset`. Wrapping that axis must not add
/// the stride to an offset that is already near `usize::MAX`; the old
/// add-then-subtract order overflowed here. A zero-sized element type keeps
/// the huge extent free.
#[test]
fn region_walk_wraps_length_one_axis_without_overflow() {
    let huge = 3 * (1usize << (usize::BITS - 2));
    let local_ranges: [Range<usize>; 3] = [0..huge, 0..1, 0..1];
    let shape_memory = [1, huge, 1];
    let permutation = AxisPermutation::new([1, 0, 2]).expect("valid permutation");
    let layout = LocalLayout {
        local_ranges: &local_ranges,
        shape_memory,
        permutation: &permutation,
        local_len: huge,
    };
    // The region stops one element short of the pencil's end, so
    // max_offset is `huge - 1` while the length-1 axis 1 has stride
    // `huge`. Advancing across that axis adds to `huge - 2`.
    let start = huge - 2;
    let region = [start..huge, 0..1, 0..1];
    let peer = PeerMetadata {
        peer_rank: Count::try_from(0).expect("peer rank fits an MPI count"),
        peer_coordinate: 0,
        send_region: region.clone(),
        receive_region: region,
        send_spatial_len: 2,
        receive_spatial_len: 2,
        send_spatial_displacement: 0,
        receive_spatial_displacement: 0,
    };
    let storage = vec![(); huge];
    let mut send_buffer = vec![(); 2];
    pack_source_regions(&[peer], &layout, &storage, &mut send_buffer, 2, 1);
}

#[test]
fn descriptor_count_boundary_is_checked_without_allocation() {
    let maximum = usize::try_from(Count::MAX).expect("Count::MAX fits usize");
    assert!(checked_count_len(maximum).is_ok());
    assert!(checked_count_len(maximum + 1).is_err());
}

/// One-dimensional peers whose send and receive segments are identical.
/// Each tuple is `(spatial_count, spatial_displacement)`.
#[allow(clippy::single_range_in_vec_init)]
fn count_test_peers(segments: &[(usize, usize)]) -> Vec<PeerMetadata<1>> {
    segments
        .iter()
        .enumerate()
        .map(|(index, &(count, displacement))| PeerMetadata {
            peer_rank: Count::try_from(index).unwrap(),
            peer_coordinate: index,
            send_region: [0..count],
            receive_region: [0..count],
            send_spatial_len: count,
            receive_spatial_len: count,
            send_spatial_displacement: displacement,
            receive_spatial_displacement: displacement,
        })
        .collect()
}

/// Keeps the boundary value on one direction and zeroes the other.
#[allow(clippy::single_range_in_vec_init)]
fn zero_direction(peers: &mut [PeerMetadata<1>], send_side: bool) {
    for peer in peers {
        if send_side {
            peer.receive_region = [0..0];
            peer.receive_spatial_len = 0;
            peer.receive_spatial_displacement = 0;
        } else {
            peer.send_region = [0..0];
            peer.send_spatial_len = 0;
            peer.send_spatial_displacement = 0;
        }
    }
}

fn count_values(values: &[usize]) -> Vec<Count> {
    values
        .iter()
        .map(|&value| Count::try_from(value).expect("Count conversion"))
        .collect()
}

fn both_modes() -> [CommunicationMode; 2] {
    [
        CommunicationMode::AllToAllv,
        CommunicationMode::PointToPoint,
    ]
}

#[test]
fn count_limits_accept_issue_peer_sizes() {
    let count = usize::try_from(1_500_000_000_u64).unwrap();
    let total = usize::try_from(3_000_000_000_u64).unwrap();
    let peers = count_test_peers(&[(count, 0), (count, count)]);
    for mode in both_modes() {
        let requirements = workspace_requirements_for_peers(&peers, 1, mode).unwrap();
        assert_eq!(
            requirements,
            TransposeWorkspaceRequirements {
                send_len: total,
                receive_len: total,
            }
        );

        let prepared = prepare_exchange_counts(&peers, 1, peers.len(), mode).unwrap();
        assert_eq!(prepared.requirements, requirements);
        assert_eq!(prepared.send_counts, count_values(&[count, count]));
        assert_eq!(prepared.receive_counts, count_values(&[count, count]));
        if mode == CommunicationMode::AllToAllv {
            assert_eq!(prepared.send_displacements, count_values(&[0, count]));
            assert_eq!(prepared.receive_displacements, count_values(&[0, count]));
        } else {
            assert!(prepared.send_displacements.is_empty());
            assert!(prepared.receive_displacements.is_empty());
        }
    }
}

#[test]
fn count_limits_keep_point_to_point_displacements_in_usize() {
    let count = usize::try_from(1_100_000_000_u64).unwrap();
    let total = 3 * count;
    let peers = count_test_peers(&[(count, 0), (count, count), (count, 2 * count)]);

    let requirements =
        workspace_requirements_for_peers(&peers, 1, CommunicationMode::PointToPoint).unwrap();
    assert_eq!(
        requirements,
        TransposeWorkspaceRequirements {
            send_len: total,
            receive_len: total,
        }
    );
    let prepared =
        prepare_exchange_counts(&peers, 1, peers.len(), CommunicationMode::PointToPoint).unwrap();
    assert_eq!(prepared.requirements, requirements);
    assert_eq!(prepared.send_counts, count_values(&[count, count, count]));
    assert_eq!(
        prepared.receive_counts,
        count_values(&[count, count, count])
    );
    assert!(prepared.send_displacements.is_empty());
    assert!(prepared.receive_displacements.is_empty());

    assert_eq!(
        workspace_requirements_for_peers(&peers, 1, CommunicationMode::AllToAllv).unwrap_err(),
        TransposeError::CountOverflow,
    );
    assert_eq!(
        prepare_exchange_counts(&peers, 1, peers.len(), CommunicationMode::AllToAllv).unwrap_err(),
        TransposeError::CountOverflow,
    );
}

#[test]
fn count_limits_bound_peer_counts_and_arithmetic() {
    let maximum = usize::try_from(Count::MAX).unwrap();
    let above_maximum = maximum + 1;
    let half = usize::MAX / 2 + 1;

    for mode in both_modes() {
        let peers = count_test_peers(&[(maximum, 0)]);
        let requirements = workspace_requirements_for_peers(&peers, 1, mode).unwrap();
        assert_eq!(
            requirements,
            TransposeWorkspaceRequirements {
                send_len: maximum,
                receive_len: maximum,
            }
        );
        let prepared = prepare_exchange_counts(&peers, 1, peers.len(), mode).unwrap();
        assert_eq!(prepared.requirements, requirements);
        assert_eq!(prepared.send_counts, count_values(&[maximum]));
        assert_eq!(prepared.receive_counts, count_values(&[maximum]));
    }

    let overflow_cases: &[(&[(usize, usize)], usize)] = &[
        (&[(above_maximum, 0)], 1),
        (&[(maximum, 0)], 2),
        (&[(half, 0)], 2),
        (&[(0, half)], 2),
    ];
    for mode in both_modes() {
        for &(segments, batch) in overflow_cases {
            let peers = count_test_peers(segments);
            assert_eq!(
                workspace_requirements_for_peers(&peers, batch, mode).unwrap_err(),
                TransposeError::CountOverflow,
            );
            assert_eq!(
                prepare_exchange_counts(&peers, batch, peers.len(), mode).unwrap_err(),
                TransposeError::CountOverflow,
            );
        }

        let peers = count_test_peers(&[(1, usize::MAX)]);
        assert_eq!(
            workspace_requirements_for_peers(&peers, 1, mode).unwrap_err(),
            TransposeError::CountOverflow,
        );
        assert_eq!(
            prepare_exchange_counts(&peers, 1, peers.len(), mode).unwrap_err(),
            TransposeError::CountOverflow,
        );

        let peers = count_test_peers(&[(1, 1)]);
        assert_eq!(
            workspace_requirements_for_peers(&peers, 1, mode).unwrap(),
            TransposeWorkspaceRequirements {
                send_len: 1,
                receive_len: 1,
            }
        );
        assert_eq!(
            prepare_exchange_counts(&peers, 1, peers.len(), mode).unwrap_err(),
            TransposeError::PreparationFailed,
        );

        let peers = count_test_peers(&[(usize::MAX, usize::MAX)]);
        let requirements = workspace_requirements_for_peers(&peers, 0, mode).unwrap();
        assert_eq!(
            requirements,
            TransposeWorkspaceRequirements {
                send_len: 0,
                receive_len: 0,
            }
        );
        let prepared = prepare_exchange_counts(&peers, 0, peers.len(), mode).unwrap();
        assert_eq!(prepared.requirements, requirements);
        assert_eq!(prepared.send_counts, count_values(&[0]));
        assert_eq!(prepared.receive_counts, count_values(&[0]));
        if mode == CommunicationMode::AllToAllv {
            assert_eq!(prepared.send_displacements, count_values(&[0]));
            assert_eq!(prepared.receive_displacements, count_values(&[0]));
        } else {
            assert!(prepared.send_displacements.is_empty());
            assert!(prepared.receive_displacements.is_empty());
        }
    }
}

#[test]
fn count_limits_check_send_and_receive_independently() {
    let maximum = usize::try_from(Count::MAX).unwrap();
    let above_maximum = maximum + 1;
    let half = usize::MAX / 2 + 1;

    for mode in both_modes() {
        for send_side in [true, false] {
            for (segments, batch) in [(vec![(half, 0)], 2usize), (vec![(above_maximum, 0)], 1)] {
                let mut peers = count_test_peers(&segments);
                zero_direction(&mut peers, send_side);
                assert_eq!(
                    workspace_requirements_for_peers(&peers, batch, mode).unwrap_err(),
                    TransposeError::CountOverflow,
                );
                assert_eq!(
                    prepare_exchange_counts(&peers, batch, peers.len(), mode).unwrap_err(),
                    TransposeError::CountOverflow,
                );
            }

            let mut peers = count_test_peers(&[(1, 1)]);
            zero_direction(&mut peers, send_side);
            let (send_len, receive_len) = if send_side { (1, 0) } else { (0, 1) };
            assert_eq!(
                workspace_requirements_for_peers(&peers, 1, mode).unwrap(),
                TransposeWorkspaceRequirements {
                    send_len,
                    receive_len,
                }
            );
            assert_eq!(
                prepare_exchange_counts(&peers, 1, peers.len(), mode).unwrap_err(),
                TransposeError::PreparationFailed,
            );
        }
    }
}

#[test]
fn count_limits_reject_invalid_peer_indices() {
    let expected = TransposeWorkspaceRequirements {
        send_len: 1,
        receive_len: 1,
    };
    for mode in both_modes() {
        let mut peers = count_test_peers(&[(1, 0)]);
        peers[0].peer_rank = -1;
        assert_eq!(
            workspace_requirements_for_peers(&peers, 1, mode).unwrap(),
            expected,
        );
        assert_eq!(
            prepare_exchange_counts(&peers, 1, peers.len(), mode).unwrap_err(),
            TransposeError::PreparationFailed,
        );

        let mut peers = count_test_peers(&[(1, 0)]);
        peers[0].peer_rank = 1;
        assert_eq!(
            workspace_requirements_for_peers(&peers, 1, mode).unwrap(),
            expected,
        );
        assert_eq!(
            prepare_exchange_counts(&peers, 1, peers.len(), mode).unwrap_err(),
            TransposeError::PreparationFailed,
        );

        let mut peers = count_test_peers(&[(1, 0)]);
        peers[0].peer_coordinate = 1;
        assert_eq!(
            workspace_requirements_for_peers(&peers, 1, mode).unwrap(),
            expected,
        );
        assert_eq!(
            prepare_exchange_counts(&peers, 1, 1, mode).unwrap_err(),
            TransposeError::PreparationFailed,
        );
    }
}

#[test]
fn checked_count_and_partition_count_boundaries() {
    let maximum = usize::try_from(Count::MAX).unwrap();
    let above_maximum = maximum + 1;

    for mode in both_modes() {
        assert_eq!(checked_count(maximum, 0, mode), Ok(Count::MAX));
        assert_eq!(
            checked_count(above_maximum, 0, mode).unwrap_err(),
            TransposeError::CountOverflow,
        );
        let wide_displacement = checked_count(0, above_maximum, mode);
        if mode == CommunicationMode::AllToAllv {
            assert_eq!(
                wide_displacement.unwrap_err(),
                TransposeError::CountOverflow
            );
        } else {
            assert_eq!(wide_displacement, Ok(0));
        }
        assert_eq!(
            checked_count(1, usize::MAX, mode).unwrap_err(),
            TransposeError::CountOverflow,
        );

        assert_eq!(
            checked_partition_count(maximum, 0, maximum, mode),
            Ok(Count::MAX),
        );
        assert_eq!(
            checked_partition_count(1, maximum, above_maximum, mode),
            Ok(1),
        );
        assert_eq!(
            checked_partition_count(1, maximum, maximum, mode).unwrap_err(),
            TransposeError::PreparationFailed,
        );
        assert_eq!(checked_partition_count(0, maximum, maximum, mode), Ok(0));
        assert_eq!(
            checked_partition_count(0, maximum, maximum - 1, mode).unwrap_err(),
            TransposeError::PreparationFailed,
        );
        let wide_count = checked_partition_count(1, above_maximum, maximum + 2, mode);
        if mode == CommunicationMode::AllToAllv {
            assert_eq!(wide_count.unwrap_err(), TransposeError::CountOverflow);
        } else {
            assert_eq!(wide_count, Ok(1));
        }
        assert_eq!(
            checked_partition_count(1, usize::MAX, usize::MAX, mode).unwrap_err(),
            TransposeError::CountOverflow,
        );
        let wide_zero = checked_partition_count(0, usize::MAX, usize::MAX, mode);
        if mode == CommunicationMode::AllToAllv {
            assert_eq!(wide_zero.unwrap_err(), TransposeError::CountOverflow);
        } else {
            assert_eq!(wide_zero, Ok(0));
        }
        assert_eq!(checked_partition_count(0, 0, 0, mode), Ok(0));
    }
}
