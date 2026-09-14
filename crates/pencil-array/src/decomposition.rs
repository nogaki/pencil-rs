use crate::{AxisError, SpatialAxis};

#[derive(Clone, Debug, PartialEq, Eq)]
/// An ordered mapping from topology axes to distributed spatial axes.
pub struct Decomposition<const N: usize, const M: usize> {
    axes: [SpatialAxis; M],
}

impl<const N: usize, const M: usize> Decomposition<N, M> {
    /// Validates an ordered set of `M` distinct spatial axes in `0..N`.
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

    /// Returns the distributed spatial axes in topology-axis order.
    pub fn axes(&self) -> &[SpatialAxis; M] {
        &self.axes
    }

    /// Embeds topology-grid extents into all spatial dimensions.
    ///
    /// Undistributed spatial axes receive extent one.
    pub fn complete_process_grid(&self, process_grid: [usize; M]) -> [usize; N] {
        complete_dims(self.axes.map(SpatialAxis::index), process_grid)
    }

    /// Embeds topology coordinates into all spatial dimensions.
    ///
    /// Undistributed spatial axes receive coordinate zero.
    pub fn complete_process_coords(&self, process_coords: [usize; M]) -> [usize; N] {
        let mut completed = [0; N];
        for (axis, coordinate) in self.axes.iter().copied().zip(process_coords) {
            completed[axis.index()] = coordinate;
        }
        completed
    }
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

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use super::*;
    use crate::partition_range;
    use proptest::prelude::*;

    #[test]
    fn local_ranges_follow_floor_partitioning() {
        assert_eq!(partition_range(10, 3, 0).unwrap(), 0..3);
        assert_eq!(partition_range(10, 3, 1).unwrap(), 3..6);
        assert_eq!(partition_range(10, 3, 2).unwrap(), 6..10);
    }

    #[test]
    fn local_range_accepts_representable_boundaries_despite_large_intermediate_product() {
        assert_eq!(
            partition_range(usize::MAX, 2, 1),
            Ok(usize::MAX / 2..usize::MAX),
        );
    }

    #[test]
    fn complete_dims_respects_topology_axis_order() {
        assert_eq!(complete_dims::<5, 2>([2, 1], [42, 12]), [1, 12, 42, 1, 1]);
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

    proptest! {
        #[test]
        fn one_dimensional_ranges_are_contiguous_and_complete(
            global_len in 0usize..128,
            process_count in 1usize..32,
        ) {
            let ranges = (0..process_count)
                .map(|coordinate| partition_range(global_len, process_count, coordinate).unwrap())
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
