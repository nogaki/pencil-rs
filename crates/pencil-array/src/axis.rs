use crate::AxisError;

#[cfg(test)]
mod tests {
    use super::*;

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
    fn permutation_rejects_duplicate_axes() {
        assert_eq!(
            AxisPermutation::<3>::new([0, 1, 1]),
            Err(AxisError::Duplicate { axis: 1 }),
        );
    }

    #[test]
    fn permutation_maps_logical_values_to_memory_order() {
        let permutation = AxisPermutation::<3>::new([0, 2, 1]).unwrap();
        assert_eq!(permutation.permute([10, 20, 30]), [10, 30, 20]);
        assert_eq!(permutation.unpermute([10, 30, 20]), [10, 20, 30]);
    }
}
