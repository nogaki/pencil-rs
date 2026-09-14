use crate::GeometryError;

pub(crate) fn checked_product(values: &[usize]) -> Result<usize, GeometryError> {
    if values.contains(&0) {
        return Ok(0);
    }

    values.iter().try_fold(1usize, |acc, &value| {
        acc.checked_mul(value).ok_or(GeometryError::SizeOverflow)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn product_of_empty_shape_is_one() {
        assert_eq!(checked_product(&[]).unwrap(), 1);
    }

    #[test]
    fn product_detects_overflow() {
        assert_eq!(
            checked_product(&[usize::MAX, 2]),
            Err(GeometryError::SizeOverflow),
        );
    }

    #[test]
    fn zero_extent_makes_product_zero_regardless_of_axis_order() {
        assert_eq!(checked_product(&[usize::MAX, 2, 0]), Ok(0));
        assert_eq!(checked_product(&[0, usize::MAX, 2]), Ok(0));
    }
}
