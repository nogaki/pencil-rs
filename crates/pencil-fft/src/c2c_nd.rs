use std::mem::size_of;

use thiserror::Error;

use crate::{Complex, FftReal, LocalC2cError, LocalC2cPlan};

/// Errors returned by local multidimensional C2C FFT plans.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum LocalC2cNdError {
    /// The shape is empty or contains a zero dimension.
    #[error("shape must be non-empty and every dimension must be non-zero")]
    InvalidShape,
    /// The element count or its complex byte size cannot be represented safely.
    #[error("shape element or byte size cannot be represented")]
    SizeOverflow,
    /// An axis is outside the shape rank.
    #[error("axis {axis} is outside rank {rank}")]
    AxisOutOfRange {
        /// The invalid axis index.
        axis: usize,
        /// The number of dimensions in the shape.
        rank: usize,
    },
    /// An axis appears more than once.
    #[error("axis {0} was selected more than once")]
    DuplicateAxis(usize),
    /// The input buffer does not contain exactly one shaped array.
    #[error("data length is {actual}, expected {expected}")]
    DataLength {
        /// The required number of complex values.
        expected: usize,
        /// The supplied number of complex values.
        actual: usize,
    },
    /// The underlying local one-dimensional FFT operation failed.
    #[error("local C2C FFT failed: {0}")]
    Local(#[source] LocalC2cError),
}

/// A reusable local C2C FFT plan over selected axes of a column-major array.
///
/// Axis zero is the fastest-varying dimension. Selected axes are transformed
/// in the order supplied to [`Self::new_col_major`]; all other dimensions form
/// independent batches. Forward transforms are unnormalized and use the
/// negative-exponent convention. Inverse transforms use the positive exponent
/// and normalize each selected axis by its length. The plan owns its one-
/// dimensional FFT plans and line buffers, so calls do not allocate.
pub struct LocalC2cNdPlan<R: FftReal> {
    shape: Vec<usize>,
    total_len: usize,
    axes: Vec<AxisPlan<R>>,
}

struct AxisPlan<R: FftReal> {
    #[cfg(test)]
    axis: usize,
    len: usize,
    stride: usize,
    outer: usize,
    plan: LocalC2cPlan<R>,
    line_in: Vec<Complex<R>>,
    line_out: Vec<Complex<R>>,
    scratch: Vec<Complex<R>>,
}

impl<R: FftReal> AxisPlan<R> {
    fn apply(&mut self, data: &mut [Complex<R>], inverse: bool) -> Result<(), LocalC2cNdError> {
        for outer in 0..self.outer {
            for inner in 0..self.stride {
                for k in 0..self.len {
                    let index = outer * self.len * self.stride + inner + k * self.stride;
                    self.line_in[k] = data[index];
                }

                if inverse {
                    self.plan
                        .inverse(&self.line_in, &mut self.line_out, &mut self.scratch)
                } else {
                    self.plan
                        .forward(&self.line_in, &mut self.line_out, &mut self.scratch)
                }
                .map_err(LocalC2cNdError::Local)?;

                for k in 0..self.len {
                    let index = outer * self.len * self.stride + inner + k * self.stride;
                    data[index] = self.line_out[k];
                }
            }
        }
        Ok(())
    }
}

impl<R: FftReal> LocalC2cNdPlan<R> {
    /// Builds a plan for a column-major shape and an ordered list of axes.
    ///
    /// Axis zero is the fastest-varying dimension. An empty `axes` list is
    /// valid and makes transform calls validated no-ops. Shape and axis
    /// validation completes before any FFT plans are constructed.
    pub fn new_col_major(shape: &[usize], axes: &[usize]) -> Result<Self, LocalC2cNdError> {
        let total_len = validate_shape::<R>(shape)?;
        let rank = shape.len();

        for &axis in axes {
            if axis >= rank {
                return Err(LocalC2cNdError::AxisOutOfRange { axis, rank });
            }
        }
        for index in 0..axes.len() {
            if axes[..index].contains(&axes[index]) {
                return Err(LocalC2cNdError::DuplicateAxis(axes[index]));
            }
        }

        let mut axis_plans = Vec::with_capacity(axes.len());
        for &axis in axes {
            let len = shape[axis];
            let stride = shape[..axis].iter().product();
            let outer = total_len / len / stride;
            let plan = LocalC2cPlan::new(len).map_err(LocalC2cNdError::Local)?;
            axis_plans.push(AxisPlan {
                #[cfg(test)]
                axis,
                len,
                stride,
                outer,
                line_in: vec![zero_complex(); len],
                line_out: vec![zero_complex(); len],
                scratch: vec![zero_complex(); plan.scratch_len()],
                plan,
            });
        }

        Ok(Self {
            shape: shape.to_vec(),
            total_len,
            axes: axis_plans,
        })
    }

    /// Returns the column-major shape owned by this plan.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the capacity-based size, in bytes, of this plan's directly owned `Vec` storage.
    ///
    /// This counts the allocations for `shape`, the `axes` entries (including
    /// each private axis plan's inline handles, metadata, and padding), and the
    /// `line_in`, `line_out`, and `scratch` buffers of each axis. It excludes
    /// this plan's inline storage, allocations owned internally by FFT backends,
    /// caller-owned data, and allocator overhead. This is a diagnostic of those
    /// `Vec` allocations, not total RSS or total backend memory; its value can
    /// vary with precision and enabled backend features.
    pub fn owned_bytes(&self) -> usize {
        let mut bytes = self.shape.capacity() * size_of::<usize>()
            + self.axes.capacity() * size_of::<AxisPlan<R>>();
        for axis in &self.axes {
            bytes += axis.line_in.capacity() * size_of::<Complex<R>>();
            bytes += axis.line_out.capacity() * size_of::<Complex<R>>();
            bytes += axis.scratch.capacity() * size_of::<Complex<R>>();
        }
        bytes
    }

    /// Computes the unnormalized negative-exponent transform in place.
    ///
    /// The data length is checked before any values are changed, including
    /// when no axes were selected. Calls do not allocate.
    pub fn forward(&mut self, data: &mut [Complex<R>]) -> Result<(), LocalC2cNdError> {
        self.apply(data, false)
    }

    /// Computes the normalized positive-exponent inverse in place.
    ///
    /// Each selected axis is normalized by its own length through the local
    /// one-dimensional inverse plan. The data length is checked before any
    /// values are changed, including when no axes were selected. Calls do not
    /// allocate.
    pub fn inverse(&mut self, data: &mut [Complex<R>]) -> Result<(), LocalC2cNdError> {
        self.apply(data, true)
    }

    fn apply(&mut self, data: &mut [Complex<R>], inverse: bool) -> Result<(), LocalC2cNdError> {
        if data.len() != self.total_len {
            return Err(LocalC2cNdError::DataLength {
                expected: self.total_len,
                actual: data.len(),
            });
        }
        for axis in &mut self.axes {
            axis.apply(data, inverse)?;
        }
        Ok(())
    }
}

fn validate_shape<R: FftReal>(shape: &[usize]) -> Result<usize, LocalC2cNdError> {
    if shape.is_empty() || shape.contains(&0) {
        return Err(LocalC2cNdError::InvalidShape);
    }
    let total_len = shape
        .iter()
        .try_fold(1usize, |total, &dimension| total.checked_mul(dimension));
    let total_len = total_len.ok_or(LocalC2cNdError::SizeOverflow)?;
    let bytes = total_len
        .checked_mul(size_of::<Complex<R>>())
        .ok_or(LocalC2cNdError::SizeOverflow)?;
    if bytes > isize::MAX as usize {
        return Err(LocalC2cNdError::SizeOverflow);
    }
    Ok(total_len)
}

fn zero_complex<R: FftReal>() -> Complex<R> {
    Complex::new(
        rustfft::num_traits::Zero::zero(),
        rustfft::num_traits::Zero::zero(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustfft::FftPlanner;
    use std::f64::consts::TAU;

    trait TestReal: FftReal {
        fn convert(value: f64) -> Self;
        fn as_f64(self) -> f64;
        fn tolerance() -> f64;
    }

    impl TestReal for f32 {
        fn convert(value: f64) -> Self {
            value as f32
        }

        fn as_f64(self) -> f64 {
            self as f64
        }

        fn tolerance() -> f64 {
            5e-4
        }
    }

    impl TestReal for f64 {
        fn convert(value: f64) -> Self {
            value
        }

        fn as_f64(self) -> f64 {
            self
        }

        fn tolerance() -> f64 {
            1e-10
        }
    }

    fn sample<R: TestReal>(len: usize) -> Vec<Complex<R>> {
        (0..len)
            .map(|index| {
                let real = ((17 * index + 3) % 31) as isize - 15;
                let imaginary = ((11 * index + 5) % 29) as isize - 14;
                Complex::new(
                    R::convert(real as f64 / 16.0),
                    R::convert(imaginary as f64 / 15.0),
                )
            })
            .collect()
    }

    fn coordinates(mut linear: usize, shape: &[usize]) -> [usize; 4] {
        assert!(shape.len() <= 4);
        let mut result = [0; 4];
        for (axis, &dimension) in shape.iter().enumerate() {
            result[axis] = linear % dimension;
            linear /= dimension;
        }
        result
    }

    fn direct_dft<R: TestReal>(
        input: &[Complex<R>],
        shape: &[usize],
        axes: &[usize],
        inverse: bool,
    ) -> Vec<Complex<f64>> {
        let total_len: usize = shape.iter().product();
        let sign = if inverse { 1.0 } else { -1.0 };
        let scale = if inverse {
            1.0 / axes.iter().map(|&axis| shape[axis]).product::<usize>() as f64
        } else {
            1.0
        };
        let mut expected = Vec::with_capacity(total_len);
        for output_index in 0..total_len {
            let output = coordinates(output_index, shape);
            let mut real = 0.0;
            let mut imaginary = 0.0;
            for (input_index, value) in input.iter().enumerate() {
                let source = coordinates(input_index, shape);
                if (0..shape.len())
                    .any(|axis| !axes.contains(&axis) && source[axis] != output[axis])
                {
                    continue;
                }
                let phase = axes.iter().fold(0.0, |phase, &axis| {
                    phase
                        + sign * TAU * source[axis] as f64 * output[axis] as f64
                            / shape[axis] as f64
                });
                let (sine, cosine) = phase.sin_cos();
                let value_real = value.re.as_f64();
                let value_imaginary = value.im.as_f64();
                real += value_real * cosine - value_imaginary * sine;
                imaginary += value_real * sine + value_imaginary * cosine;
            }
            expected.push(Complex::new(real * scale, imaginary * scale));
        }
        expected
    }

    fn assert_error<R: TestReal>(actual: &[Complex<R>], expected: &[Complex<f64>]) {
        assert_eq!(actual.len(), expected.len());
        let mut error = 0.0f64;
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let actual_real = actual.re.as_f64();
            let actual_imaginary = actual.im.as_f64();
            assert!(
                actual_real.is_finite()
                    && actual_imaginary.is_finite()
                    && expected.re.is_finite()
                    && expected.im.is_finite(),
                "non-finite value at {index}: actual={actual:?}, expected={expected:?}"
            );
            let delta = (actual_real - expected.re)
                .abs()
                .max((actual_imaginary - expected.im).abs());
            let denominator = 1.0 + expected.re.abs().max(expected.im.abs());
            error = error.max(delta / denominator);
        }
        assert!(
            error <= R::tolerance(),
            "normalized max error {error:e} exceeds {:e}",
            R::tolerance()
        );
    }

    fn exercise_numeric_cases<R: TestReal>() {
        const RANK4_AXES: [&[usize]; 11] = [
            &[0],
            &[1],
            &[2],
            &[3],
            &[0, 2],
            &[2, 0],
            &[1, 3],
            &[3, 1],
            &[0, 1, 2, 3],
            &[3, 2, 1, 0],
            &[],
        ];
        for shape in [&[2, 3, 5, 7][..], &[1, 3, 5, 7][..]] {
            let input = sample::<R>(shape.iter().product());
            for axes in RANK4_AXES {
                let mut plan = LocalC2cNdPlan::new_col_major(shape, axes).unwrap();
                assert_eq!(plan.shape(), shape);
                let mut forward = input.clone();
                plan.forward(&mut forward).unwrap();
                assert_error(&forward, &direct_dft(&input, shape, axes, false));

                let mut inverse = input.clone();
                plan.inverse(&mut inverse).unwrap();
                assert_error(&inverse, &direct_dft(&input, shape, axes, true));
            }
        }
        for axes in [&[0][..], &[][..]] {
            let shape = [1];
            let input = sample::<R>(1);
            let mut plan = LocalC2cNdPlan::new_col_major(&shape, axes).unwrap();
            let mut forward = input.clone();
            plan.forward(&mut forward).unwrap();
            assert_error(&forward, &direct_dft(&input, &shape, axes, false));
            let mut inverse = input.clone();
            plan.inverse(&mut inverse).unwrap();
            assert_error(&inverse, &direct_dft(&input, &shape, axes, true));
        }
    }

    #[test]
    fn f32_ranked_axes_match_independent_direct_dft() {
        exercise_numeric_cases::<f32>();
    }

    #[test]
    fn f64_ranked_axes_match_independent_direct_dft() {
        exercise_numeric_cases::<f64>();
    }

    fn expected_owned_bytes<R: FftReal>(plan: &LocalC2cNdPlan<R>) -> usize {
        plan.shape.capacity() * size_of::<usize>()
            + plan.axes.capacity() * size_of::<AxisPlan<R>>()
            + plan
                .axes
                .iter()
                .map(|axis| {
                    (axis.line_in.capacity() + axis.line_out.capacity() + axis.scratch.capacity())
                        * size_of::<Complex<R>>()
                })
                .sum::<usize>()
    }

    fn grow_capacity<T>(values: &mut Vec<T>) -> usize {
        let previous = values.capacity();
        values.reserve(previous + 1);
        let growth = values.capacity() - previous;
        assert!(growth > 0);
        growth
    }

    fn exercise_owned_bytes<R: TestReal>() {
        let empty_axes = LocalC2cNdPlan::<R>::new_col_major(&[2, 3], &[]).unwrap();
        assert!(empty_axes.axes.is_empty());
        assert_eq!(empty_axes.owned_bytes(), expected_owned_bytes(&empty_axes));

        let unit_axis = LocalC2cNdPlan::<R>::new_col_major(&[1], &[0]).unwrap();
        assert_eq!(unit_axis.owned_bytes(), expected_owned_bytes(&unit_axis));

        let shape = [2, 3, 5];
        let axes = [2, 0];
        let mut plan = LocalC2cNdPlan::<R>::new_col_major(&shape, &axes).unwrap();
        assert_eq!(plan.owned_bytes(), expected_owned_bytes(&plan));

        let before = plan.owned_bytes();
        let growth = grow_capacity(&mut plan.shape);
        assert_eq!(plan.owned_bytes() - before, growth * size_of::<usize>());
        assert_eq!(plan.owned_bytes(), expected_owned_bytes(&plan));

        let before = plan.owned_bytes();
        let growth = grow_capacity(&mut plan.axes);
        assert_eq!(
            plan.owned_bytes() - before,
            growth * size_of::<AxisPlan<R>>()
        );
        assert_eq!(plan.owned_bytes(), expected_owned_bytes(&plan));

        let before = plan.owned_bytes();
        let mut growth = 0;
        for axis in &mut plan.axes {
            for buffer in [&mut axis.line_in, &mut axis.line_out, &mut axis.scratch] {
                growth += grow_capacity(buffer);
            }
        }
        assert_eq!(
            plan.owned_bytes() - before,
            growth * size_of::<Complex<R>>()
        );
        assert_eq!(plan.owned_bytes(), expected_owned_bytes(&plan));
        assert_eq!(plan.shape(), &shape);

        let input = sample::<R>(shape.iter().product());
        let expected_forward = direct_dft(&input, &shape, &axes, false);
        let expected_inverse = direct_dft(&input, &shape, &axes, true);
        let owned_bytes = plan.owned_bytes();
        let mut caller_data = input.clone();
        let caller_capacity = caller_data.capacity();
        caller_data.reserve(caller_capacity + 1);
        assert!(caller_data.capacity() > caller_capacity);
        assert_eq!(plan.owned_bytes(), owned_bytes);

        let shape_before = plan.shape().to_vec();
        let mut forward = input.clone();
        assert_eq!(plan.owned_bytes(), owned_bytes);
        plan.forward(&mut forward).unwrap();
        assert_error(&forward, &expected_forward);
        assert_eq!(plan.owned_bytes(), owned_bytes);

        let mut inverse = input;
        assert_eq!(plan.owned_bytes(), owned_bytes);
        plan.inverse(&mut inverse).unwrap();
        assert_error(&inverse, &expected_inverse);
        assert_eq!(plan.owned_bytes(), owned_bytes);
        assert_eq!(plan.shape(), shape_before);
    }

    #[test]
    fn owned_bytes_is_capacity_based_and_read_only_for_both_precisions() {
        exercise_owned_bytes::<f32>();
        exercise_owned_bytes::<f64>();
    }

    #[test]
    fn forward_sign_amplitude_and_inverse_normalization_are_analytic() {
        let length = 5;
        let mode = 2;
        let mut plan = LocalC2cNdPlan::<f64>::new_col_major(&[length], &[0]).unwrap();
        let positive_mode: Vec<_> = (0..length)
            .map(|index| {
                let angle = TAU * mode as f64 * index as f64 / length as f64;
                Complex::new(angle.cos(), angle.sin())
            })
            .collect();
        let mut transformed = positive_mode;
        plan.forward(&mut transformed).unwrap();
        for (index, value) in transformed.iter().enumerate() {
            let expected = if index == mode {
                Complex::new(length as f64, 0.0)
            } else {
                Complex::new(0.0, 0.0)
            };
            assert!((value.re - expected.re).abs() < 1e-10);
            assert!((value.im - expected.im).abs() < 1e-10);
        }

        let impulse = Complex::new(2.0, -3.0);
        let mut inverse_input = vec![Complex::new(0.0, 0.0); length];
        inverse_input[0] = impulse;
        plan.inverse(&mut inverse_input).unwrap();
        for value in inverse_input {
            assert!((value.re - impulse.re / length as f64).abs() < 1e-10);
            assert!((value.im - impulse.im / length as f64).abs() < 1e-10);
        }

        let negative_mode: Vec<_> = (0..length)
            .map(|index| {
                let angle = -TAU * mode as f64 * index as f64 / length as f64;
                Complex::new(angle.cos(), angle.sin())
            })
            .collect();
        let mut inverse_output = negative_mode;
        plan.inverse(&mut inverse_output).unwrap();
        for (index, value) in inverse_output.iter().enumerate() {
            let expected = if index == mode { 1.0 } else { 0.0 };
            assert!((value.re - expected).abs() < 1e-10);
            assert!(value.im.abs() < 1e-10);
        }
    }

    fn col_major_index(coordinates: &[usize], shape: &[usize]) -> usize {
        let mut index = 0;
        let mut stride = 1;
        for (coordinate, dimension) in coordinates.iter().zip(shape) {
            index += coordinate * stride;
            stride *= dimension;
        }
        index
    }

    #[test]
    fn one_unselected_batch_change_does_not_affect_other_batches() {
        let shape = [2, 3, 5, 7];
        let mut plan = LocalC2cNdPlan::<f64>::new_col_major(&shape, &[1]).unwrap();
        let input = sample::<f64>(shape.iter().product());
        let mut changed_input = input.clone();
        for selected_axis_coordinate in 0..shape[1] {
            let index = col_major_index(&[1, selected_axis_coordinate, 2, 4], &shape);
            changed_input[index].re += 0.75;
            changed_input[index].im -= 0.5;
        }
        let mut original_output = input;
        let mut changed_output = changed_input;
        plan.forward(&mut original_output).unwrap();
        plan.forward(&mut changed_output).unwrap();

        let mut changed_batch_differs = false;
        for index in 0..original_output.len() {
            let coordinate = coordinates(index, &shape);
            if coordinate[0] == 1 && coordinate[2] == 2 && coordinate[3] == 4 {
                changed_batch_differs |= original_output[index] != changed_output[index];
            } else {
                assert_eq!(original_output[index], changed_output[index]);
            }
        }
        assert!(changed_batch_differs);
    }

    fn construction_error(shape: &[usize], axes: &[usize]) -> LocalC2cNdError {
        match LocalC2cNdPlan::<f64>::new_col_major(shape, axes) {
            Err(error) => error,
            Ok(_) => panic!("invalid construction unexpectedly succeeded"),
        }
    }

    #[test]
    fn construction_rejects_invalid_shapes_axes_and_sizes_before_planning() {
        assert_eq!(construction_error(&[], &[]), LocalC2cNdError::InvalidShape);
        assert_eq!(
            construction_error(&[2, 0, 3], &[]),
            LocalC2cNdError::InvalidShape
        );
        assert_eq!(
            validate_shape::<f64>(&[usize::MAX, 2]),
            Err(LocalC2cNdError::SizeOverflow)
        );

        let byte_overflow = usize::MAX / size_of::<Complex<f64>>() + 1;
        assert_eq!(
            validate_shape::<f64>(&[byte_overflow]),
            Err(LocalC2cNdError::SizeOverflow)
        );
        fn assert_addressable<R: FftReal>() {
            let max_elements = isize::MAX as usize / size_of::<Complex<R>>();
            assert!(validate_shape::<R>(&[max_elements]).is_ok());
            assert_eq!(
                validate_shape::<R>(&[max_elements + 1]),
                Err(LocalC2cNdError::SizeOverflow)
            );
        }
        assert_addressable::<f32>();
        assert_addressable::<f64>();

        assert_eq!(
            construction_error(&[2, 3], &[0, 2]),
            LocalC2cNdError::AxisOutOfRange { axis: 2, rank: 2 }
        );
        assert_eq!(
            construction_error(&[2, 3], &[0, 1, 2]),
            LocalC2cNdError::AxisOutOfRange { axis: 2, rank: 2 }
        );
        assert_eq!(
            construction_error(&[2, 3], &[0, 0]),
            LocalC2cNdError::DuplicateAxis(0)
        );
        assert_eq!(
            construction_error(&[2, 3], &[0, 1, 1]),
            LocalC2cNdError::DuplicateAxis(1)
        );
    }

    fn f64_bits(data: &[Complex<f64>]) -> Vec<(u64, u64)> {
        data.iter()
            .map(|value| (value.re.to_bits(), value.im.to_bits()))
            .collect()
    }

    #[test]
    fn data_length_errors_preserve_bits_and_plan_remains_reusable() {
        for axes in [false, true] {
            let mut plan = if axes {
                LocalC2cNdPlan::<f64>::new_col_major(&[2, 3], &[]).unwrap()
            } else {
                LocalC2cNdPlan::<f64>::new_col_major(&[2, 3], &[0, 1]).unwrap()
            };
            for inverse in [false, true] {
                for actual in [0, 5, 7] {
                    let mut data: Vec<_> = (0..actual)
                        .map(|index| Complex::new(index as f64 + 0.25, -0.0))
                        .collect();
                    let before = f64_bits(&data);
                    let result = if inverse {
                        plan.inverse(&mut data)
                    } else {
                        plan.forward(&mut data)
                    };
                    assert_eq!(
                        result,
                        Err(LocalC2cNdError::DataLength {
                            expected: 6,
                            actual,
                        })
                    );
                    assert_eq!(f64_bits(&data), before);
                }
                let mut valid = sample::<f64>(6);
                let selected_axes: &[usize] = if axes { &[] } else { &[0, 1] };
                let expected = direct_dft(&valid, &[2, 3], selected_axes, inverse);
                let result = if inverse {
                    plan.inverse(&mut valid)
                } else {
                    plan.forward(&mut valid)
                };
                assert_eq!(result, Ok(()));
                assert_error(&valid, &expected);
            }
        }
    }

    #[test]
    fn empty_axes_are_a_bitwise_noop_after_length_validation() {
        let mut plan = LocalC2cNdPlan::<f64>::new_col_major(&[2, 3], &[]).unwrap();
        let mut data = [Complex::new(-0.0, f64::from_bits(0x3ff0_0000_0000_0001)); 6];
        let before = f64_bits(&data);
        plan.forward(&mut data).unwrap();
        assert_eq!(f64_bits(&data), before);
        plan.inverse(&mut data).unwrap();
        assert_eq!(f64_bits(&data), before);
    }

    #[test]
    fn axes_order_is_retained_and_applied_in_that_order() {
        let plan = LocalC2cNdPlan::<f64>::new_col_major(&[2, 3, 5, 7], &[2, 0]).unwrap();
        assert_eq!(
            plan.axes.iter().map(|axis| axis.axis).collect::<Vec<_>>(),
            [2, 0]
        );
    }

    #[test]
    fn lower_level_error_is_retained_as_the_error_source() {
        let mut planner = FftPlanner::<f64>::new();
        let line_len = (1..=32)
            .chain([37, 97, 127, 1024])
            .find(|&len| planner.plan_fft_forward(len).get_immutable_scratch_len() > 0)
            .expect("a RustFFT candidate must require immutable scratch");
        let mut plan = LocalC2cNdPlan::<f64>::new_col_major(&[line_len], &[0]).unwrap();
        let required = plan.axes[0].plan.scratch_len();
        assert!(required > 0);
        plan.axes[0].scratch.clear();
        let mut data = sample::<f64>(line_len);
        let error = plan.forward(&mut data).unwrap_err();
        assert_eq!(
            error,
            LocalC2cNdError::Local(LocalC2cError::ScratchTooSmall {
                required,
                actual: 0,
            })
        );
        assert!(
            std::error::Error::source(&error)
                .unwrap()
                .downcast_ref::<LocalC2cError>()
                .is_some()
        );
    }
}
