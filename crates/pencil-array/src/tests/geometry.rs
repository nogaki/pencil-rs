use std::ops::Range;

use proptest::prelude::*;

use super::{local_ranges_for, partition_range, row_major_offset, shape_from_ranges};
use crate::{GeometryError, SpatialAxis};

#[test]
fn partition_matches_floor_rule_without_multiplication_overflow() {
    assert_eq!(partition_range(10, 3, 0).unwrap(), 0..3);
    assert_eq!(partition_range(10, 3, 1).unwrap(), 3..6);
    assert_eq!(partition_range(10, 3, 2).unwrap(), 6..10);
    assert_eq!(
        partition_range(usize::MAX, 2, 1).unwrap(),
        usize::MAX / 2..usize::MAX,
    );
}

#[test]
fn partition_allows_empty_ranges_and_rejects_invalid_parts() {
    assert_eq!(partition_range(2, 4, 0).unwrap(), 0..0);
    assert_eq!(partition_range(2, 4, 1).unwrap(), 0..1);
    assert_eq!(partition_range(2, 4, 2).unwrap(), 1..1);
    assert_eq!(partition_range(2, 4, 3).unwrap(), 1..2);
    assert_eq!(partition_range(2, 0, 0), Err(GeometryError::ZeroPartitions),);
    assert_eq!(
        partition_range(2, 2, 2),
        Err(GeometryError::ProcessCoordinateOutOfBounds {
            axis: 0,
            coordinate: 2,
            extent: 2,
        }),
    );
}

#[test]
fn partition_handles_maximum_partition_counts_and_empty_domains() {
    assert_eq!(partition_range(0, usize::MAX, usize::MAX - 1), Ok(0..0));
    assert_eq!(
        partition_range(usize::MAX, usize::MAX, usize::MAX - 1),
        Ok(usize::MAX - 1..usize::MAX),
    );
    assert_eq!(
        partition_range(usize::MAX - 2, usize::MAX - 1, usize::MAX - 2),
        Ok(usize::MAX - 3..usize::MAX - 2),
    );
}

#[test]
fn local_ranges_respect_unsorted_decomposition() {
    let axes = [
        SpatialAxis::new::<3>(2).unwrap(),
        SpatialAxis::new::<3>(0).unwrap(),
    ];
    assert_eq!(
        local_ranges_for([9, 5, 8], [2, 3], [1, 2], axes).unwrap(),
        [6..9, 0..5, 4..8],
    );
}

#[test]
fn shape_from_ranges_preserves_axis_order() {
    let ranges: [Range<usize>; 3] = [2..5, 0..7, 9..11];
    assert_eq!(shape_from_ranges(&ranges), [3, 7, 2]);
}

#[test]
fn local_ranges_report_the_failing_topology_axis() {
    let axes = [
        SpatialAxis::new::<3>(2).unwrap(),
        SpatialAxis::new::<3>(0).unwrap(),
    ];
    assert_eq!(
        local_ranges_for([9, 5, 8], [2, 3], [0, 3], axes),
        Err(GeometryError::ProcessCoordinateOutOfBounds {
            axis: 1,
            coordinate: 3,
            extent: 3,
        }),
    );
    assert_eq!(
        local_ranges_for([9, 5, 8], [2, 0], [0, 0], axes),
        Err(GeometryError::ZeroProcessExtent { axis: 1 }),
    );
}

#[test]
fn row_major_offsets_validate_rank_bounds_and_overflow() {
    assert_eq!(row_major_offset(&[], &[]), Ok(0));
    assert_eq!(row_major_offset(&[2, 3, 4], &[1, 2, 3]), Ok(23));
    assert_eq!(row_major_offset(&[2, 3, 4], &[1, 0, 0]), Ok(1));
    assert_eq!(row_major_offset(&[2, 3, 4], &[0, 1, 0]), Ok(2));
    assert_eq!(row_major_offset(&[2, 3, 4], &[0, 0, 1]), Ok(6));
    assert_eq!(
        row_major_offset(&[2, 3], &[1]),
        Err(GeometryError::RankMismatch {
            shape_rank: 2,
            index_rank: 1,
        }),
    );
    assert_eq!(
        row_major_offset(&[2, 3], &[2, 0]),
        Err(GeometryError::LocalIndexOutOfBounds {
            axis: 0,
            index: 2,
            extent: 2,
        }),
    );
    assert_eq!(
        row_major_offset(&[usize::MAX, 2], &[usize::MAX - 1, 1]),
        Err(GeometryError::SizeOverflow),
    );
    assert_eq!(
        row_major_offset(&[usize::MAX, usize::MAX], &[1, 1]),
        Err(GeometryError::SizeOverflow),
    );
    assert_eq!(
        row_major_offset(&[2, 0], &[0, 0]),
        Err(GeometryError::LocalIndexOutOfBounds {
            axis: 1,
            index: 0,
            extent: 0
        }),
    );
}

proptest! {
    #[test]
    fn partitions_cover_the_domain_with_balanced_contiguous_ranges(
        length in any::<usize>(),
        parts in 1usize..32,
    ) {
        let mut boundary = 0;
        let mut total = 0usize;
        for coordinate in 0..parts {
            let range = partition_range(length, parts, coordinate).unwrap();
            prop_assert_eq!(range.start, boundary);
            prop_assert!(range.len() == length / parts || range.len() == length / parts + 1);
            total = total.checked_add(range.len()).unwrap();
            boundary = range.end;
        }
        prop_assert_eq!(total, length);
        prop_assert_eq!(boundary, length);
    }

    #[test]
    fn row_major_offsets_follow_nested_loop_order(shape in proptest::array::uniform3(1usize..8)) {
        let mut expected = 0;
        for k in 0..shape[2] {
            for j in 0..shape[1] {
                for i in 0..shape[0] {
                    prop_assert_eq!(row_major_offset(&shape, &[i, j, k]).unwrap(), expected);
                    expected += 1;
                }
            }
        }
    }
}
