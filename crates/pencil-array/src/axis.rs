use crate::AxisError;

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
        assert_eq!(
            permutation.axes().map(SpatialAxis::index),
            [0, 1, 2],
        );
    }

    #[test]
    fn position_of_returns_memory_position() {
        let permutation = AxisPermutation::<3>::new([2, 0, 1]).unwrap();
        assert_eq!(
            permutation.position_of(SpatialAxis::new::<3>(0).unwrap()),
            1,
        );
        assert_eq!(
            permutation.position_of(SpatialAxis::new::<3>(1).unwrap()),
            2,
        );
        assert_eq!(
            permutation.position_of(SpatialAxis::new::<3>(2).unwrap()),
            0,
        );
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
