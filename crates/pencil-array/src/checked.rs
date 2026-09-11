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
