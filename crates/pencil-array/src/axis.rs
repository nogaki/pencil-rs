use crate::AxisError;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SpatialAxis(usize);

impl SpatialAxis {
    pub fn new<const N: usize>(index: usize) -> Result<Self, AxisError> {
        if index < N {
            Ok(Self(index))
        } else {
            Err(AxisError::OutOfBounds {
                axis: index,
                dimensions: N,
            })
        }
    }

    pub const fn index(self) -> usize {
        self.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AxisPermutation<const N: usize> {
    axes_in_memory_order: [SpatialAxis; N],
    logical_to_memory: [usize; N],
}

impl<const N: usize> AxisPermutation<N> {
    pub fn new(axes: [usize; N]) -> Result<Self, AxisError> {
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

        let axes_in_memory_order = std::array::from_fn(|position| SpatialAxis(axes[position]));
        let mut logical_to_memory = [0; N];
        for (memory_position, axis) in axes_in_memory_order.iter().copied().enumerate() {
            logical_to_memory[axis.index()] = memory_position;
        }

        Ok(Self {
            axes_in_memory_order,
            logical_to_memory,
        })
    }

    pub fn identity() -> Self {
        Self {
            axes_in_memory_order: std::array::from_fn(SpatialAxis),
            logical_to_memory: std::array::from_fn(|index| index),
        }
    }

    pub fn axes(&self) -> &[SpatialAxis; N] {
        &self.axes_in_memory_order
    }

    /// Returns the memory position of a logical axis.
    ///
    /// Axes validated for another dimension are checked against this permutation.
    pub fn inverse_position(&self, axis: SpatialAxis) -> Result<usize, AxisError> {
        self.logical_to_memory
            .get(axis.index())
            .copied()
            .ok_or(AxisError::OutOfBounds {
                axis: axis.index(),
                dimensions: N,
            })
    }

    /// Returns the memory position of a logical axis, as [`Self::inverse_position`] does.
    pub fn position_of(&self, axis: SpatialAxis) -> Result<usize, AxisError> {
        self.inverse_position(axis)
    }

    pub fn permute<T: Copy>(&self, logical: [T; N]) -> [T; N] {
        std::array::from_fn(|memory_position| {
            logical[self.axes_in_memory_order[memory_position].index()]
        })
    }

    pub fn unpermute<T: Copy>(&self, memory: [T; N]) -> [T; N] {
        std::array::from_fn(|logical_axis| memory[self.logical_to_memory[logical_axis]])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn spatial_axis_rejects_out_of_bounds_index() {
        assert_eq!(
            SpatialAxis::new::<3>(3),
            Err(AxisError::OutOfBounds {
                axis: 3,
                dimensions: 3,
            }),
        );
    }

    #[test]
    fn spatial_axis_exposes_valid_index() {
        assert_eq!(SpatialAxis::new::<3>(2).unwrap().index(), 2);
    }

    #[test]
    fn permutation_rejects_duplicate_axes() {
        assert_eq!(
            AxisPermutation::<3>::new([0, 1, 1]),
            Err(AxisError::Duplicate { axis: 1 }),
        );
    }

    #[test]
    fn permutation_rejects_out_of_bounds_axis() {
        assert_eq!(
            AxisPermutation::<3>::new([0, 1, 3]),
            Err(AxisError::OutOfBounds {
                axis: 3,
                dimensions: 3,
            }),
        );
    }

    #[test]
    fn identity_exposes_logical_axes_in_memory_order() {
        let permutation = AxisPermutation::<3>::identity();
        assert_eq!(permutation.axes().map(SpatialAxis::index), [0, 1, 2],);
    }

    #[test]
    fn inverse_lookup_rejects_axis_from_larger_dimension() {
        let permutation = AxisPermutation::<3>::identity();
        let axis = SpatialAxis::new::<4>(3).unwrap();
        let expected = Err(AxisError::OutOfBounds {
            axis: 3,
            dimensions: 3,
        });

        assert_eq!(permutation.position_of(axis), expected);
        assert_eq!(permutation.inverse_position(axis), expected);
    }

    #[test]
    fn inverse_lookup_rejects_axis_for_zero_dimensions() {
        let permutation = AxisPermutation::<0>::identity();
        let axis = SpatialAxis::new::<1>(0).unwrap();
        let expected = Err(AxisError::OutOfBounds {
            axis: 0,
            dimensions: 0,
        });

        assert_eq!(permutation.position_of(axis), expected);
        assert_eq!(permutation.inverse_position(axis), expected);
    }

    #[test]
    fn position_of_returns_memory_position() {
        let permutation = AxisPermutation::<3>::new([2, 0, 1]).unwrap();
        assert_eq!(
            permutation.position_of(SpatialAxis::new::<3>(0).unwrap()),
            Ok(1),
        );
        assert_eq!(
            permutation.position_of(SpatialAxis::new::<3>(1).unwrap()),
            Ok(2),
        );
        assert_eq!(
            permutation.position_of(SpatialAxis::new::<3>(2).unwrap()),
            Ok(0),
        );
    }

    #[test]
    fn inverse_position_returns_memory_position() {
        let permutation = AxisPermutation::<3>::new([2, 0, 1]).unwrap();

        for (logical_axis, memory_position) in [(0, 1), (1, 2), (2, 0)] {
            assert_eq!(
                permutation.inverse_position(SpatialAxis::new::<4>(logical_axis).unwrap()),
                Ok(memory_position),
            );
        }
    }

    #[test]
    fn permutation_maps_logical_values_to_memory_order() {
        let permutation = AxisPermutation::<3>::new([0, 2, 1]).unwrap();
        assert_eq!(permutation.permute([10, 20, 30]), [10, 30, 20]);
        assert_eq!(permutation.unpermute([10, 30, 20]), [10, 20, 30]);
    }

    proptest! {
        #[test]
        fn permute_then_unpermute_is_identity(values in proptest::array::uniform3(any::<u16>())) {
            for axes in [
                [0, 1, 2],
                [0, 2, 1],
                [1, 0, 2],
                [1, 2, 0],
                [2, 0, 1],
                [2, 1, 0],
            ] {
                let permutation = AxisPermutation::<3>::new(axes).unwrap();
                prop_assert_eq!(
                    permutation.unpermute(permutation.permute(values)),
                    values,
                );
            }
        }
    }
}
