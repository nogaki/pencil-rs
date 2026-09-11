use crate::GeometryError;

pub(crate) fn checked_product(values: &[usize]) -> Result<usize, GeometryError> {
    values.iter().try_fold(1usize, |acc, &value| {
        acc.checked_mul(value).ok_or(GeometryError::SizeOverflow)
    })
}

// MPI count conversion becomes live when the topology facade is introduced.
#[allow(dead_code)]
pub(crate) fn usize_to_i32(value: usize) -> Result<i32, GeometryError> {
    i32::try_from(value).map_err(|_| GeometryError::CountOverflow)
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
    fn mpi_count_conversion_rejects_large_values() {
        assert_eq!(
            usize_to_i32(i32::MAX as usize + 1),
            Err(GeometryError::CountOverflow),
        );
    }
}
