#[cfg(test)]
mod tests {
    use std::ops::Range;

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
        assert_eq!(
            partition_range(2, 0, 0),
            Err(GeometryError::ZeroPartitions),
        );
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
    fn row_major_offsets_validate_rank_bounds_and_overflow() {
        assert_eq!(row_major_offset(&[2, 3, 4], &[1, 2, 3]), Ok(23));
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
    }
}
