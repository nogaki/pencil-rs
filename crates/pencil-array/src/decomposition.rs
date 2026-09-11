use std::ops::Range;

use crate::{AxisError, GeometryError, SpatialAxis, checked::checked_product};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decomposition<const N: usize, const M: usize> {
    axes: [SpatialAxis; M],
}

impl<const N: usize, const M: usize> Decomposition<N, M> {
    pub fn new(axes: [usize; M]) -> Result<Self, AxisError> {
        if M == 0 || M > N {
            return Err(AxisError::InvalidDecompositionRank {
                spatial_dimensions: N,
                topology_dimensions: M,
            });
        }

        let mut seen = [false; N];
        for &axis in &axes {
            if axis >= N {
                return Err(AxisError::OutOfBounds {
                    axis,
                    dimensions: N,
                });
            }
            if seen[axis] {
                return Err(AxisError::Duplicate { axis });
            }
            seen[axis] = true;
        }

        Ok(Self {
            axes: std::array::from_fn(|index| {
                SpatialAxis::new::<N>(axes[index]).expect("axis was validated")
            }),
        })
    }

    pub fn axes(&self) -> &[SpatialAxis; M] {
        &self.axes
    }

    pub fn complete_process_grid(&self, process_grid: [usize; M]) -> [usize; N] {
        complete_dims(self.axes.map(SpatialAxis::index), process_grid)
    }

    pub fn complete_process_coords(&self, process_coords: [usize; M]) -> [usize; N] {
        let mut completed = [0; N];
        for (axis, coordinate) in self.axes.iter().copied().zip(process_coords) {
            completed[axis.index()] = coordinate;
        }
        completed
    }
}

pub(crate) fn local_data_range(
    coordinate: usize,
    process_count: usize,
    global_len: usize,
) -> Result<Range<usize>, GeometryError> {
    debug_assert!(process_count > 0);
    debug_assert!(coordinate < process_count);

    let start = global_len
        .checked_mul(coordinate)
        .ok_or(GeometryError::SizeOverflow)?
        / process_count;
    let next_coordinate = coordinate
        .checked_add(1)
        .ok_or(GeometryError::SizeOverflow)?;
    let end = global_len
        .checked_mul(next_coordinate)
        .ok_or(GeometryError::SizeOverflow)?
        / process_count;

    Ok(start..end)
}

pub(crate) fn complete_dims<const N: usize, const M: usize>(
    axes: [usize; M],
    values: [usize; M],
) -> [usize; N] {
    let mut completed = [1; N];
    for (axis, value) in axes.into_iter().zip(values) {
        completed[axis] = value;
    }
    completed
}

// Pencil construction uses this in the next task. Keeping it crate-visible
// lets the geometry tests exercise the exact production implementation.
#[allow(dead_code)]
pub(crate) fn generate_axes<const N: usize, const M: usize>(
    decomposition: &Decomposition<N, M>,
    process_grid: [usize; M],
    global_shape: [usize; N],
) -> Result<Vec<[Range<usize>; N]>, GeometryError> {
    for (axis, extent) in process_grid.iter().copied().enumerate() {
        if extent == 0 {
            return Err(GeometryError::ZeroProcessExtent { axis });
        }
    }

    let region_count = checked_product(&process_grid)?;
    let completed_grid = decomposition.complete_process_grid(process_grid);
    let mut regions = Vec::with_capacity(region_count);

    for flat_index in 0..region_count {
        let mut remainder = flat_index;
        let mut process_coords = [0; M];
        for axis in (0..M).rev() {
            process_coords[axis] = remainder % process_grid[axis];
            remainder /= process_grid[axis];
        }

        let completed_coords = decomposition.complete_process_coords(process_coords);
        let mut region = std::array::from_fn(|_| 0..0);
        for (axis, range) in region.iter_mut().enumerate() {
            *range = local_data_range(
                completed_coords[axis],
                completed_grid[axis],
                global_shape[axis],
            )?;
        }
        regions.push(region);
    }

    Ok(regions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn local_ranges_follow_floor_partitioning() {
        assert_eq!(local_data_range(0, 3, 10).unwrap(), 0..3);
        assert_eq!(local_data_range(1, 3, 10).unwrap(), 3..6);
        assert_eq!(local_data_range(2, 3, 10).unwrap(), 6..10);
    }

    #[test]
    fn local_range_reports_multiplication_overflow() {
        assert_eq!(
            local_data_range(1, 2, usize::MAX),
            Err(GeometryError::SizeOverflow),
        );
    }

    #[test]
    fn complete_dims_respects_topology_axis_order() {
        assert_eq!(
            complete_dims::<5, 2>([2, 1], [42, 12]),
            [1, 12, 42, 1, 1]
        );
    }

    #[test]
    fn decomposition_rejects_invalid_dimension_count() {
        assert_eq!(
            Decomposition::<2, 3>::new([0, 1, 0]),
            Err(AxisError::InvalidDecompositionRank {
                spatial_dimensions: 2,
                topology_dimensions: 3,
            }),
        );
        assert_eq!(
            Decomposition::<3, 0>::new([]),
            Err(AxisError::InvalidDecompositionRank {
                spatial_dimensions: 3,
                topology_dimensions: 0,
            }),
        );
    }

    #[test]
    fn decomposition_rejects_duplicate_and_out_of_bounds_axes() {
        assert_eq!(
            Decomposition::<3, 2>::new([0, 0]),
            Err(AxisError::Duplicate { axis: 0 }),
        );
        assert_eq!(
            Decomposition::<3, 2>::new([0, 3]),
            Err(AxisError::OutOfBounds {
                axis: 3,
                dimensions: 3,
            }),
        );
    }

    #[test]
    fn decomposition_completes_grid_and_coordinates() {
        let decomposition = Decomposition::<3, 2>::new([0, 2]).unwrap();
        assert_eq!(decomposition.axes().map(SpatialAxis::index), [0, 2]);
        assert_eq!(decomposition.complete_process_grid([2, 4]), [2, 1, 4]);
        assert_eq!(decomposition.complete_process_coords([1, 3]), [1, 0, 3]);
    }

    #[test]
    fn generate_axes_rejects_zero_process_extent() {
        let decomposition = Decomposition::<3, 2>::new([0, 2]).unwrap();
        assert_eq!(
            generate_axes(&decomposition, [2, 0], [5, 4, 7]),
            Err(GeometryError::ZeroProcessExtent { axis: 1 }),
        );
    }

    #[test]
    fn generated_regions_cover_global_domain_exactly_once() {
        let decomposition = Decomposition::<3, 2>::new([0, 2]).unwrap();
        let regions = generate_axes(&decomposition, [2, 3], [5, 4, 7]).unwrap();
        assert_eq!(regions.len(), 6);
        assert_eq!(regions[0], [0..2, 0..4, 0..2]);
        assert_eq!(regions[1], [0..2, 0..4, 2..4]);

        let mut visits = vec![0u8; 5 * 4 * 7];
        for region in regions {
            for i in region[0].clone() {
                for j in region[1].clone() {
                    for k in region[2].clone() {
                        visits[(i * 4 + j) * 7 + k] += 1;
                    }
                }
            }
        }
        assert!(visits.into_iter().all(|count| count == 1));
    }

    proptest! {
        #[test]
        fn one_dimensional_ranges_are_contiguous_and_complete(
            global_len in 0usize..128,
            process_count in 1usize..32,
        ) {
            let ranges = (0..process_count)
                .map(|coordinate| local_data_range(coordinate, process_count, global_len).unwrap())
                .collect::<Vec<Range<usize>>>();

            prop_assert_eq!(ranges.first().unwrap().start, 0);
            prop_assert_eq!(ranges.last().unwrap().end, global_len);
            for pair in ranges.windows(2) {
                prop_assert_eq!(pair[0].end, pair[1].start);
            }
            let total: usize = ranges.iter().map(|range| range.len()).sum();
            prop_assert_eq!(total, global_len);
        }
    }
}
