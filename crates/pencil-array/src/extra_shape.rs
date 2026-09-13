use crate::{ShapeError, checked::checked_product};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExtraShape {
    dimensions: Box<[usize]>,
    element_count: usize,
}

impl ExtraShape {
    pub fn new(dimensions: impl Into<Box<[usize]>>) -> Result<Self, ShapeError> {
        let dimensions = dimensions.into();
        let element_count = checked_product(&dimensions).map_err(|_| ShapeError::SizeOverflow)?;
        Ok(Self {
            dimensions,
            element_count,
        })
    }

    pub fn scalar() -> Self {
        Self {
            dimensions: Vec::new().into_boxed_slice(),
            element_count: 1,
        }
    }

    pub fn dimensions(&self) -> &[usize] {
        &self.dimensions
    }

    pub fn element_count(&self) -> usize {
        self.element_count
    }
}

#[cfg(test)]
mod tests {
    use super::ExtraShape;
    use crate::ShapeError;

    #[test]
    fn scalar_extra_shape_has_one_element_per_spatial_point() {
        let shape = ExtraShape::scalar();

        assert_eq!(shape.dimensions(), &[]);
        assert_eq!(shape.element_count(), 1);
    }

    #[test]
    fn extra_shape_keeps_axis_order() {
        let shape = ExtraShape::new([3, 2]).unwrap();

        assert_eq!(shape.dimensions(), &[3, 2]);
        assert_eq!(shape.element_count(), 6);
    }

    #[test]
    fn zero_extent_makes_an_empty_extra_shape() {
        let shape = ExtraShape::new([usize::MAX, 2, 0]).unwrap();

        assert_eq!(shape.dimensions(), &[usize::MAX, 2, 0]);
        assert_eq!(shape.element_count(), 0);
    }

    #[test]
    fn overflowing_extra_shape_is_rejected() {
        assert_eq!(
            ExtraShape::new([usize::MAX, 2]),
            Err(ShapeError::SizeOverflow),
        );
    }
}
