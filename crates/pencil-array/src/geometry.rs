use std::ops::Range;

use crate::{GeometryError, SpatialAxis};

/// Partitions `0..length` using the floor rule, allowing empty local ranges.
///
/// Rank `coordinate` owns `floor(length * coordinate / parts)` through
/// `floor(length * (coordinate + 1) / parts)`, with an exclusive end.
pub fn partition_range(
    length: usize,
    parts: usize,
    coordinate: usize,
) -> Result<Range<usize>, GeometryError> {
    if parts == 0 {
        return Err(GeometryError::ZeroPartitions);
    }
    if coordinate >= parts {
        return Err(GeometryError::ProcessCoordinateOutOfBounds {
            axis: 0,
            coordinate,
            extent: parts,
        });
    }

    // Rust's supported pointer widths are at most 64 bits, so both products
    // fit u128. Each quotient is <= length and therefore fits usize again.
    // coordinate < parts also guarantees that coordinate + 1 fits usize.
    let boundary = |position: usize| ((length as u128 * position as u128) / parts as u128) as usize;
    Ok(boundary(coordinate)..boundary(coordinate + 1))
}

/// Computes ranges for an already validated ordered spatial decomposition.
pub(crate) fn local_ranges_for<const N: usize, const M: usize>(
    global_shape: [usize; N],
    process_grid: [usize; M],
    process_coords: [usize; M],
    decomposition: [SpatialAxis; M],
) -> Result<[Range<usize>; N], GeometryError> {
    let mut ranges = std::array::from_fn(|axis| 0..global_shape[axis]);
    for topology_axis in 0..M {
        let extent = process_grid[topology_axis];
        let coordinate = process_coords[topology_axis];
        if extent == 0 {
            return Err(GeometryError::ZeroProcessExtent {
                axis: topology_axis,
            });
        }
        if coordinate >= extent {
            return Err(GeometryError::ProcessCoordinateOutOfBounds {
                axis: topology_axis,
                coordinate,
                extent,
            });
        }
        let spatial_axis = decomposition[topology_axis].index();
        ranges[spatial_axis] = partition_range(global_shape[spatial_axis], extent, coordinate)?;
    }
    Ok(ranges)
}

pub(crate) fn shape_from_ranges<const N: usize>(ranges: &[Range<usize>; N]) -> [usize; N] {
    std::array::from_fn(|axis| ranges[axis].len())
}

pub(crate) fn row_major_offset(shape: &[usize], indices: &[usize]) -> Result<usize, GeometryError> {
    if shape.len() != indices.len() {
        return Err(GeometryError::RankMismatch {
            shape_rank: shape.len(),
            index_rank: indices.len(),
        });
    }
    let mut offset = 0usize;
    for (axis, (&extent, &index)) in shape.iter().zip(indices).enumerate() {
        if index >= extent {
            return Err(GeometryError::LocalIndexOutOfBounds {
                axis,
                index,
                extent,
            });
        }
        offset = offset
            .checked_mul(extent)
            .and_then(|base| base.checked_add(index))
            .ok_or(GeometryError::SizeOverflow)?;
    }
    Ok(offset)
}

#[cfg(test)]
mod tests {
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
            for i in 0..shape[0] {
                for j in 0..shape[1] {
                    for k in 0..shape[2] {
                        prop_assert_eq!(row_major_offset(&shape, &[i, j, k]).unwrap(), expected);
                        expected += 1;
                    }
                }
            }
        }
    }
}
