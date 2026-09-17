use std::fmt;
use std::mem::size_of;
use std::sync::Arc;

use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

use super::{Complex, FftReal};

/// Errors returned by local real FFT plan construction and execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum LocalR2cError {
    /// The real FFT line length was zero.
    #[error("real FFT line length must be non-zero")]
    InvalidLength,
    /// The line length or a required derived buffer size could not be represented.
    #[error("real FFT length or derived length cannot be represented")]
    LengthOverflow,
    /// A source or destination buffer did not contain an integral number of lines.
    #[error("buffer length is not an integral number of FFT lines")]
    NonIntegralBatch,
    /// Source and destination contained different numbers of lines.
    #[error(
        "source and destination batch counts differ: source={source_count}, destination={destination_count}"
    )]
    BatchCountMismatch {
        /// The number of source lines.
        source_count: usize,
        /// The number of destination lines.
        destination_count: usize,
    },
    /// The caller-owned real line buffer was too short.
    #[error("real line buffer is too short: required {required}, actual {actual}")]
    RealLineTooSmall {
        /// The minimum initialized real line length required by this plan.
        required: usize,
        /// The supplied initialized real line length.
        actual: usize,
    },
    /// The caller-owned complex line buffer was too short.
    #[error("complex line buffer is too short: required {required}, actual {actual}")]
    ComplexLineTooSmall {
        /// The minimum initialized complex line length required by this plan.
        required: usize,
        /// The supplied initialized complex line length.
        actual: usize,
    },
    /// The caller-owned backend scratch slice was too short.
    #[error("FFT scratch is too small: required {required}, actual {actual}")]
    ScratchTooSmall {
        /// The minimum initialized scratch length required by this plan.
        required: usize,
        /// The supplied initialized scratch length.
        actual: usize,
    },
    /// A constrained real-spectrum endpoint had a non-zero imaginary component.
    #[error(
        "inverse spectrum batch {batch} has a non-zero imaginary component at complex index {index}"
    )]
    InvalidSpectrumEndpoint {
        /// The zero-based batch containing the invalid endpoint.
        batch: usize,
        /// The complex index of the invalid endpoint. This is zero for DC and
        /// is the final complex index for an even-length Nyquist endpoint.
        index: usize,
    },
}

/// An immutable local batched real-to-half-complex and half-complex-to-real FFT plan.
///
/// Each operation treats its data slices as contiguous row-major batches. The
/// plan owns immutable RealFFT plans, but does not own data, line buffers, or
/// scratch. Forward transforms preserve their real source and inverse
/// transforms preserve their complex source. There is intentionally no
/// in-place real FFT API.
pub struct LocalR2cPlan<R: FftReal> {
    real_len: usize,
    complex_len: usize,
    scratch_len: usize,
    forward: Arc<dyn RealToComplex<R>>,
    inverse: Arc<dyn ComplexToReal<R>>,
}

impl<R: FftReal> fmt::Debug for LocalR2cPlan<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LocalR2cPlan")
            .field("real_len", &self.real_len)
            .field("complex_len", &self.complex_len)
            .field("scratch_len", &self.scratch_len)
            .finish_non_exhaustive()
    }
}

impl<R: FftReal> LocalR2cPlan<R> {
    /// Builds a plan for real lines of `real_len` values.
    ///
    /// The reduced complex line length is `real_len / 2 + 1`. Zero and
    /// unaddressable real, reduced complex, or odd-length complex staging
    /// lengths are rejected before planning. Backend resource failures or
    /// backend panics are not converted into [`LocalR2cError`].
    pub fn new(real_len: usize) -> Result<Self, LocalR2cError> {
        let complex_len = validate_lengths::<R>(real_len)?;

        let mut planner = RealFftPlanner::<R>::new();
        let forward = planner.plan_fft_forward(real_len);
        let inverse = planner.plan_fft_inverse(real_len);
        let scratch_len = forward.get_scratch_len().max(inverse.get_scratch_len());

        Ok(Self {
            real_len,
            complex_len,
            scratch_len,
            forward,
            inverse,
        })
    }

    /// Returns the number of real values in each input or output line.
    pub fn real_len(&self) -> usize {
        self.real_len
    }

    /// Returns the number of independent complex values in each spectrum line.
    pub fn complex_len(&self) -> usize {
        self.complex_len
    }

    /// Returns the minimum initialized complex scratch length accepted by
    /// either direction.
    pub fn scratch_len(&self) -> usize {
        self.scratch_len
    }

    /// Computes an unnormalized forward real-to-half-complex FFT for every
    /// line in `src`.
    ///
    /// `src` must contain an integral number of real lines and `dst` an equal
    /// number of complex lines. `real_line` must be an initialized slice of at
    /// least [`Self::real_len`] real values; its exact required prefix is used
    /// as the native input because RealFFT mutates that input. `scratch` must
    /// be initialized and at least [`Self::scratch_len`] complex values long.
    /// Oversized line and scratch tails are not touched. A valid empty batch is
    /// a no-op after all length checks and does not touch the workspaces.
    ///
    /// Ordinary validation errors preserve `src`, `dst`, both workspaces, and
    /// all input values. After native execution begins, backend errors,
    /// resource failures, and panics are not converted and do not have an
    /// output-atomicity guarantee.
    pub fn forward(
        &self,
        src: &[R],
        dst: &mut [Complex<R>],
        real_line: &mut [R],
        scratch: &mut [Complex<R>],
    ) -> Result<(), LocalR2cError> {
        self.validate_forward(src.len(), dst.len(), real_line.len(), scratch.len())?;
        if src.is_empty() {
            return Ok(());
        }

        let native_scratch_len = self.forward.get_scratch_len();
        for (source_line, destination_line) in src
            .chunks_exact(self.real_len)
            .zip(dst.chunks_exact_mut(self.complex_len))
        {
            real_line[..self.real_len].copy_from_slice(source_line);
            self.forward
                .process_with_scratch(
                    &mut real_line[..self.real_len],
                    destination_line,
                    &mut scratch[..native_scratch_len],
                )
                .expect("validated RealFFT forward buffers must be accepted");
        }
        Ok(())
    }

    /// Computes a normalized inverse half-complex-to-real FFT for every line
    /// in `src`.
    ///
    /// `src` must contain an integral number of complex lines and `dst` an
    /// equal number of real lines. `complex_line` must be an initialized slice
    /// of at least [`Self::complex_len`] complex values; its exact required
    /// prefix is used as the native input because RealFFT mutates that input.
    /// `scratch` must satisfy [`Self::scratch_len`]. Oversized complex-line
    /// and scratch tails are not touched. The inverse divides each resulting
    /// line by exactly `real_len`. A valid empty batch is a no-op after all
    /// length and endpoint checks and does not touch the workspaces.
    ///
    /// The imaginary component of DC must be strict zero in every input line.
    /// For even `real_len`, the final (Nyquist) complex value must also have a
    /// strict-zero imaginary component. For odd `real_len` greater than one,
    /// the final complex value is not constrained; for `real_len == 1`, it is
    /// DC and remains constrained. Signed zero is accepted and NaN is not;
    /// interior bins are not constrained. These endpoint checks cover every
    /// batch before any destination, line buffer, or scratch write. An invalid
    /// endpoint therefore leaves all caller buffers unchanged.
    ///
    /// Ordinary validation errors and endpoint errors preserve `src`, `dst`,
    /// both workspaces, and all input values. After native execution begins,
    /// backend errors, resource failures, and panics are not converted and do
    /// not have an output-atomicity guarantee.
    pub fn inverse(
        &self,
        src: &[Complex<R>],
        dst: &mut [R],
        complex_line: &mut [Complex<R>],
        scratch: &mut [Complex<R>],
    ) -> Result<(), LocalR2cError> {
        self.validate_inverse(src.len(), dst.len(), complex_line.len(), scratch.len())?;
        // RealFFT reports endpoint errors after writing; preflight the entire batch.
        self.validate_endpoints(src)?;
        if src.is_empty() {
            return Ok(());
        }

        let native_scratch_len = self.inverse.get_scratch_len();
        let scale = R::one()
            / R::from_usize(self.real_len).expect("f32/f64 represent the validated length");
        for (source_line, destination_line) in src
            .chunks_exact(self.complex_len)
            .zip(dst.chunks_exact_mut(self.real_len))
        {
            complex_line[..self.complex_len].copy_from_slice(source_line);
            self.inverse
                .process_with_scratch(
                    &mut complex_line[..self.complex_len],
                    destination_line,
                    &mut scratch[..native_scratch_len],
                )
                .expect("validated RealFFT inverse buffers must be accepted");
            for value in destination_line {
                *value = *value * scale;
            }
        }
        Ok(())
    }

    fn validate_forward(
        &self,
        source_len: usize,
        destination_len: usize,
        real_line_len: usize,
        scratch_len: usize,
    ) -> Result<(), LocalR2cError> {
        let source_batches = batch_count(source_len, self.real_len)?;
        let destination_batches = batch_count(destination_len, self.complex_len)?;
        if source_batches != destination_batches {
            return Err(LocalR2cError::BatchCountMismatch {
                source_count: source_batches,
                destination_count: destination_batches,
            });
        }
        if real_line_len < self.real_len {
            return Err(LocalR2cError::RealLineTooSmall {
                required: self.real_len,
                actual: real_line_len,
            });
        }
        self.validate_scratch(scratch_len)?;
        Ok(())
    }

    fn validate_inverse(
        &self,
        source_len: usize,
        destination_len: usize,
        complex_line_len: usize,
        scratch_len: usize,
    ) -> Result<(), LocalR2cError> {
        let source_batches = batch_count(source_len, self.complex_len)?;
        let destination_batches = batch_count(destination_len, self.real_len)?;
        if source_batches != destination_batches {
            return Err(LocalR2cError::BatchCountMismatch {
                source_count: source_batches,
                destination_count: destination_batches,
            });
        }
        if complex_line_len < self.complex_len {
            return Err(LocalR2cError::ComplexLineTooSmall {
                required: self.complex_len,
                actual: complex_line_len,
            });
        }
        self.validate_scratch(scratch_len)?;
        Ok(())
    }

    fn validate_scratch(&self, actual: usize) -> Result<(), LocalR2cError> {
        if actual < self.scratch_len {
            return Err(LocalR2cError::ScratchTooSmall {
                required: self.scratch_len,
                actual,
            });
        }
        Ok(())
    }

    fn validate_endpoints(&self, source: &[Complex<R>]) -> Result<(), LocalR2cError> {
        for (batch, line) in source.chunks_exact(self.complex_len).enumerate() {
            if line[0].im != R::zero() {
                return Err(LocalR2cError::InvalidSpectrumEndpoint { batch, index: 0 });
            }
            if self.real_len % 2 == 0 && line[self.complex_len - 1].im != R::zero() {
                return Err(LocalR2cError::InvalidSpectrumEndpoint {
                    batch,
                    index: self.complex_len - 1,
                });
            }
        }
        Ok(())
    }
}

fn batch_count(actual: usize, line_len: usize) -> Result<usize, LocalR2cError> {
    if actual % line_len != 0 {
        return Err(LocalR2cError::NonIntegralBatch);
    }
    Ok(actual / line_len)
}

fn validate_lengths<R: FftReal>(real_len: usize) -> Result<usize, LocalR2cError> {
    if real_len == 0 {
        return Err(LocalR2cError::InvalidLength);
    }

    let complex_len = real_len / 2 + 1;
    validate_addressable(real_len, size_of::<R>())?;
    validate_addressable(complex_len, size_of::<Complex<R>>())?;
    if real_len % 2 != 0 {
        validate_addressable(real_len, size_of::<Complex<R>>())?;
    }
    Ok(complex_len)
}

fn validate_addressable(length: usize, element_size: usize) -> Result<(), LocalR2cError> {
    let bytes = length
        .checked_mul(element_size)
        .ok_or(LocalR2cError::LengthOverflow)?;
    if bytes > isize::MAX as usize {
        return Err(LocalR2cError::LengthOverflow);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LocalC2cPlan;
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
            6e-4
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
            2e-10
        }
    }

    fn initialized_complex<R: TestReal>(len: usize, offset: f64) -> Vec<Complex<R>> {
        (0..len)
            .map(|index| {
                Complex::new(
                    R::convert(offset + 0.17 * index as f64),
                    R::convert(-offset - 0.23 * index as f64),
                )
            })
            .collect()
    }

    fn dirty_complex<R: TestReal>(values: &mut [Complex<R>], offset: f64) {
        for (index, value) in values.iter_mut().enumerate() {
            *value = Complex::new(
                R::convert(offset + index as f64),
                R::convert(-offset - 2.0 * index as f64),
            );
        }
    }

    fn dft_forward<R: TestReal>(input: &[R], real_len: usize) -> Vec<Complex<R>> {
        input
            .chunks_exact(real_len)
            .flat_map(|line| {
                (0..=real_len / 2).map(move |k| {
                    let (real, imaginary) = line.iter().enumerate().fold(
                        (0.0, 0.0),
                        |(real, imaginary), (j, value)| {
                            let angle = -TAU * j as f64 * k as f64 / real_len as f64;
                            let (sine, cosine) = angle.sin_cos();
                            (
                                real + value.as_f64() * cosine,
                                imaginary + value.as_f64() * sine,
                            )
                        },
                    );
                    Complex::new(R::convert(real), R::convert(imaginary))
                })
            })
            .collect()
    }

    fn dft_inverse<R: TestReal>(input: &[Complex<R>], real_len: usize) -> Vec<R> {
        input
            .chunks_exact(real_len / 2 + 1)
            .flat_map(|line| {
                (0..real_len).map(move |j| {
                    let mut value = 0.0;
                    for (k, spectrum) in line.iter().enumerate() {
                        let angle = TAU * j as f64 * k as f64 / real_len as f64;
                        let (sine, cosine) = angle.sin_cos();
                        value += spectrum.re.as_f64() * cosine - spectrum.im.as_f64() * sine;
                        if k != 0 && (real_len % 2 != 0 || k != real_len / 2) {
                            value += spectrum.re.as_f64() * cosine - spectrum.im.as_f64() * sine;
                        }
                    }
                    R::convert(value / real_len as f64)
                })
            })
            .collect()
    }

    fn assert_complex_close<R: TestReal>(actual: &[Complex<R>], expected: &[Complex<R>]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let bound =
                R::tolerance() * (1.0 + expected.re.as_f64().abs().max(expected.im.as_f64().abs()));
            assert!(
                (actual.re.as_f64() - expected.re.as_f64()).abs() <= bound
                    && (actual.im.as_f64() - expected.im.as_f64()).abs() <= bound,
                "index {index}: actual={actual:?}, expected={expected:?}, bound={bound}"
            );
        }
    }

    fn assert_real_close<R: TestReal>(actual: &[R], expected: &[R]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let bound = R::tolerance() * (1.0 + expected.as_f64().abs());
            assert!(
                (actual.as_f64() - expected.as_f64()).abs() <= bound,
                "index {index}: actual={actual:?}, expected={expected:?}, bound={bound}"
            );
        }
    }

    fn real_input<R: TestReal>(real_len: usize, batch_count: usize) -> Vec<R> {
        (0..real_len * batch_count)
            .map(|index| {
                let x = index as f64 + 1.0;
                R::convert((0.31 * x).sin() + 0.09 * x - (0.11 * x).cos())
            })
            .collect()
    }

    fn native_scratch_len<R: FftReal>(real_len: usize) -> usize {
        let mut planner = realfft::RealFftPlanner::<R>::new();
        let forward = planner.plan_fft_forward(real_len);
        let inverse = planner.plan_fft_inverse(real_len);
        forward.get_scratch_len().max(inverse.get_scratch_len())
    }

    fn exercise_against_oracles<R: TestReal>(real_len: usize, batch_count: usize) {
        let plan = LocalR2cPlan::<R>::new(real_len).unwrap();
        assert_eq!(plan.real_len(), real_len);
        assert_eq!(plan.complex_len(), real_len / 2 + 1);

        let source = real_input::<R>(real_len, batch_count);
        let source_before = source.clone();
        let mut spectrum = vec![
            Complex::new(R::convert(17.0), R::convert(-23.0));
            plan.complex_len() * batch_count
        ];
        let mut real_line = vec![R::convert(0.0); real_len + 2];
        let real_line_before = real_line.clone();
        let mut scratch = initialized_complex::<R>(plan.scratch_len() + 3, 1.0);
        let mut scratch_before = scratch.clone();

        plan.forward(&source, &mut spectrum, &mut real_line, &mut scratch)
            .unwrap();
        assert_eq!(source, source_before);
        assert_eq!(&real_line[real_len..], &real_line_before[real_len..]);
        assert_eq!(
            &scratch[plan.scratch_len()..],
            &scratch_before[plan.scratch_len()..]
        );
        assert_complex_close(&spectrum, &dft_forward(&source, real_len));

        let c2c_plan = LocalC2cPlan::<R>::new(real_len).unwrap();
        let c2c_source = source
            .iter()
            .copied()
            .map(|value| Complex::new(value, R::convert(0.0)))
            .collect::<Vec<_>>();
        let mut c2c_spectrum =
            vec![Complex::new(R::convert(0.0), R::convert(0.0)); c2c_source.len()];
        let mut c2c_scratch = initialized_complex::<R>(c2c_plan.scratch_len(), 13.0);
        c2c_plan
            .forward(&c2c_source, &mut c2c_spectrum, &mut c2c_scratch)
            .unwrap();
        for (r2c_line, c2c_line) in spectrum
            .chunks_exact(plan.complex_len())
            .zip(c2c_spectrum.chunks_exact(real_len))
        {
            assert_complex_close(r2c_line, &c2c_line[..plan.complex_len()]);
        }

        let spectrum_before_inverse = spectrum.clone();
        let mut recovered = vec![R::convert(29.0); source.len()];
        let mut complex_line =
            vec![Complex::new(R::convert(0.0), R::convert(0.0)); plan.complex_len() + 2];
        let complex_line_before = complex_line.clone();
        dirty_complex(&mut scratch, -7.0);
        scratch_before = scratch.clone();
        plan.inverse(&spectrum, &mut recovered, &mut complex_line, &mut scratch)
            .unwrap();
        assert_eq!(spectrum, spectrum_before_inverse);
        assert_real_close(&recovered, &source_before);
        assert_eq!(
            &complex_line[plan.complex_len()..],
            &complex_line_before[plan.complex_len()..]
        );
        assert_eq!(
            &scratch[plan.scratch_len()..],
            &scratch_before[plan.scratch_len()..]
        );

        let mut arbitrary = (0..plan.complex_len() * batch_count)
            .map(|index| {
                Complex::new(
                    R::convert(0.2 + 0.13 * index as f64),
                    R::convert(-0.4 + 0.07 * index as f64),
                )
            })
            .collect::<Vec<_>>();
        for line in arbitrary.chunks_exact_mut(plan.complex_len()) {
            line[0].im = R::convert(0.0);
            if real_len % 2 == 0 {
                line[plan.complex_len() - 1].im = R::convert(0.0);
            }
        }
        let arbitrary_before = arbitrary.clone();
        let mut arbitrary_output = vec![R::convert(0.0); real_len * batch_count];
        dirty_complex(&mut complex_line, 33.0);
        dirty_complex(&mut scratch, 51.0);
        plan.inverse(
            &arbitrary,
            &mut arbitrary_output,
            &mut complex_line,
            &mut scratch,
        )
        .unwrap();
        assert_eq!(arbitrary, arbitrary_before);
        assert_real_close(&arbitrary_output, &dft_inverse(&arbitrary, real_len));
    }

    #[test]
    fn f32_odd_even_prime_composite_batches_match_independent_oracles() {
        for real_len in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13] {
            exercise_against_oracles::<f32>(real_len, 1);
            exercise_against_oracles::<f32>(real_len, 3);
        }
    }

    #[test]
    fn f64_odd_even_prime_composite_batches_match_independent_oracles() {
        for real_len in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13] {
            exercise_against_oracles::<f64>(real_len, 1);
            exercise_against_oracles::<f64>(real_len, 3);
        }
    }

    fn check_frequency_and_normalization<R: TestReal>() {
        let real_len = 8;
        let peak = 2;
        let input = (0..real_len)
            .map(|index| R::convert((TAU * peak as f64 * index as f64 / real_len as f64).cos()))
            .collect::<Vec<_>>();
        let plan = LocalR2cPlan::<R>::new(real_len).unwrap();
        let mut spectrum = vec![Complex::new(R::convert(0.0), R::convert(0.0)); plan.complex_len()];
        let mut real_line = vec![R::convert(0.0); real_len];
        let mut scratch = initialized_complex::<R>(plan.scratch_len(), 9.0);
        plan.forward(&input, &mut spectrum, &mut real_line, &mut scratch)
            .unwrap();
        for (index, value) in spectrum.iter().enumerate() {
            let expected = if index == peak {
                real_len as f64 / 2.0
            } else {
                0.0
            };
            assert!((value.re.as_f64() - expected).abs() <= R::tolerance() * real_len as f64);
            assert!(value.im.as_f64().abs() <= R::tolerance() * real_len as f64);
        }

        let dc = vec![R::convert(3.5); real_len];
        plan.forward(&dc, &mut spectrum, &mut real_line, &mut scratch)
            .unwrap();
        assert!((spectrum[0].re.as_f64() - 3.5 * real_len as f64).abs() < R::tolerance() * 10.0);
        assert!(spectrum[0].im.as_f64().abs() <= R::tolerance());
        let mut output = vec![R::convert(0.0); real_len];
        let mut complex_line =
            vec![Complex::new(R::convert(0.0), R::convert(0.0)); plan.complex_len()];
        plan.inverse(&spectrum, &mut output, &mut complex_line, &mut scratch)
            .unwrap();
        assert_real_close(&output, &dc);
    }

    #[test]
    fn f32_frequency_sign_dc_and_inverse_normalization() {
        check_frequency_and_normalization::<f32>();
    }

    #[test]
    fn f64_frequency_sign_dc_and_inverse_normalization() {
        check_frequency_and_normalization::<f64>();
    }

    fn check_endpoint_validation<R: TestReal>() {
        let plan = LocalR2cPlan::<R>::new(6).unwrap();
        let invalid_endpoints = [
            (0, 0, R::convert(1.0)),
            (2, 0, R::convert(f64::NAN)),
            (0, plan.complex_len() - 1, R::convert(-2.0)),
            (2, plan.complex_len() - 1, R::convert(f64::NAN)),
        ];
        for &(batch, index, imaginary) in &invalid_endpoints {
            let mut source =
                vec![Complex::new(R::convert(2.0), R::convert(0.0)); plan.complex_len() * 3];
            source[batch * plan.complex_len() + index].im = imaginary;
            let source_before = source.clone();
            let mut destination = vec![R::convert(41.0); plan.real_len() * 3];
            let destination_before = destination.clone();
            let mut line = initialized_complex::<R>(plan.complex_len() + 2, 7.0);
            let line_before = line.clone();
            let mut scratch = initialized_complex::<R>(plan.scratch_len() + 2, 11.0);
            let scratch_before = scratch.clone();

            assert_eq!(
                plan.inverse(&source, &mut destination, &mut line, &mut scratch),
                Err(LocalR2cError::InvalidSpectrumEndpoint { batch, index })
            );
            assert!(source.iter().zip(&source_before).all(|(actual, expected)| {
                actual.re.as_f64().to_bits() == expected.re.as_f64().to_bits()
                    && actual.im.as_f64().to_bits() == expected.im.as_f64().to_bits()
            }));
            assert_eq!(destination, destination_before);
            assert_eq!(line, line_before);
            assert_eq!(scratch, scratch_before);
        }

        let mut source =
            vec![Complex::new(R::convert(2.0), R::convert(0.0)); plan.complex_len() * 2];
        for (batch, line) in source.chunks_exact_mut(plan.complex_len()).enumerate() {
            line[0].im = if batch == 0 {
                R::convert(-0.0)
            } else {
                R::convert(0.0)
            };
            line[plan.complex_len() - 1].im = if batch == 0 {
                R::convert(0.0)
            } else {
                R::convert(-0.0)
            };
        }
        let source_before = source.clone();
        let mut destination = vec![R::convert(0.0); plan.real_len() * 2];
        let mut line = initialized_complex::<R>(plan.complex_len() + 1, 17.0);
        let mut scratch = initialized_complex::<R>(plan.scratch_len() + 1, 19.0);
        assert_eq!(
            plan.inverse(&source, &mut destination, &mut line, &mut scratch),
            Ok(())
        );
        assert!(source.iter().zip(&source_before).all(|(actual, expected)| {
            actual.re.as_f64().to_bits() == expected.re.as_f64().to_bits()
                && actual.im.as_f64().to_bits() == expected.im.as_f64().to_bits()
        }));

        let plan = LocalR2cPlan::<R>::new(1).unwrap();
        let source = [Complex::new(R::convert(0.0), R::convert(1.0))];
        let source_before = source;
        let mut destination = [R::convert(3.0)];
        let destination_before = destination;
        let mut line = [Complex::new(R::convert(4.0), R::convert(5.0))];
        let line_before = line;
        let mut scratch = initialized_complex::<R>(plan.scratch_len(), 23.0);
        let scratch_before = scratch.clone();
        assert_eq!(
            plan.inverse(&source, &mut destination, &mut line, &mut scratch),
            Err(LocalR2cError::InvalidSpectrumEndpoint { batch: 0, index: 0 })
        );
        assert_eq!(
            source[0].re.as_f64().to_bits(),
            source_before[0].re.as_f64().to_bits()
        );
        assert_eq!(
            source[0].im.as_f64().to_bits(),
            source_before[0].im.as_f64().to_bits()
        );
        assert_eq!(destination, destination_before);
        assert_eq!(line, line_before);
        assert_eq!(scratch, scratch_before);
    }

    #[test]
    fn f32_endpoint_validation_is_atomic_and_direction_aware() {
        check_endpoint_validation::<f32>();
    }

    #[test]
    fn f64_endpoint_validation_is_atomic_and_direction_aware() {
        check_endpoint_validation::<f64>();
    }

    #[test]
    fn zero_batch_preserves_workspaces_on_success() {
        let plan = LocalR2cPlan::<f64>::new(5).unwrap();
        assert!(plan.scratch_len() > 0);
        let source = Vec::new();
        let mut spectrum = Vec::new();
        let mut real_line = vec![9.0; plan.real_len() + 1];
        let real_line_before = real_line.clone();
        let mut complex_line = initialized_complex(plan.complex_len() + 1, 2.0);
        let complex_line_before = complex_line.clone();
        let mut scratch = initialized_complex(plan.scratch_len() + 1, 3.0);
        let scratch_before = scratch.clone();

        assert_eq!(
            plan.forward(&source, &mut spectrum, &mut real_line, &mut scratch),
            Ok(())
        );
        assert_eq!(real_line, real_line_before);
        assert_eq!(scratch, scratch_before);
        assert_eq!(
            plan.inverse(&spectrum, &mut Vec::new(), &mut complex_line, &mut scratch),
            Ok(())
        );
        assert_eq!(complex_line, complex_line_before);
        assert_eq!(scratch, scratch_before);
    }

    #[test]
    fn validation_rejects_incomplete_batches_and_batch_count_mismatch_without_changes() {
        let plan = LocalR2cPlan::<f64>::new(4).unwrap();

        for (source_len, destination_len) in [
            (plan.real_len() - 1, plan.complex_len()),
            (plan.real_len(), plan.complex_len() - 1),
        ] {
            let source = vec![1.0; source_len];
            let source_before = source.clone();
            let mut destination = initialized_complex::<f64>(destination_len, 1.0);
            let destination_before = destination.clone();
            let mut line = vec![2.0; plan.real_len() + 1];
            let line_before = line.clone();
            let mut scratch = initialized_complex::<f64>(plan.scratch_len() + 1, 3.0);
            let scratch_before = scratch.clone();
            assert_eq!(
                plan.forward(&source, &mut destination, &mut line, &mut scratch),
                Err(LocalR2cError::NonIntegralBatch)
            );
            assert_eq!(source, source_before);
            assert_eq!(destination, destination_before);
            assert_eq!(line, line_before);
            assert_eq!(scratch, scratch_before);
        }

        let source = real_input::<f64>(plan.real_len(), 1);
        let source_before = source.clone();
        let mut destination = Vec::new();
        let destination_before = destination.clone();
        let mut line = vec![4.0; plan.real_len() + 1];
        let line_before = line.clone();
        let mut scratch = initialized_complex::<f64>(plan.scratch_len() + 1, 5.0);
        let scratch_before = scratch.clone();
        assert_eq!(
            plan.forward(&source, &mut destination, &mut line, &mut scratch),
            Err(LocalR2cError::BatchCountMismatch {
                source_count: 1,
                destination_count: 0,
            })
        );
        assert_eq!(source, source_before);
        assert_eq!(destination, destination_before);
        assert_eq!(line, line_before);
        assert_eq!(scratch, scratch_before);

        for (source_len, destination_len) in [
            (plan.complex_len() + 1, plan.real_len()),
            (plan.complex_len(), plan.real_len() - 1),
        ] {
            let mut source = initialized_complex::<f64>(source_len, 6.0);
            if source.len() >= plan.complex_len() {
                source[0].im = 0.0;
                source[plan.complex_len() - 1].im = 0.0;
            }
            let source_before = source.clone();
            let mut destination = vec![7.0; destination_len];
            let destination_before = destination.clone();
            let mut line = initialized_complex::<f64>(plan.complex_len() + 1, 8.0);
            let line_before = line.clone();
            let mut scratch = initialized_complex::<f64>(plan.scratch_len() + 1, 9.0);
            let scratch_before = scratch.clone();
            assert_eq!(
                plan.inverse(&source, &mut destination, &mut line, &mut scratch),
                Err(LocalR2cError::NonIntegralBatch)
            );
            assert_eq!(source, source_before);
            assert_eq!(destination, destination_before);
            assert_eq!(line, line_before);
            assert_eq!(scratch, scratch_before);
        }

        let mut source = initialized_complex::<f64>(plan.complex_len(), 10.0);
        source[0].im = 0.0;
        source[plan.complex_len() - 1].im = 0.0;
        let source_before = source.clone();
        let mut destination = Vec::new();
        let destination_before = destination.clone();
        let mut line = initialized_complex::<f64>(plan.complex_len() + 1, 11.0);
        let line_before = line.clone();
        let mut scratch = initialized_complex::<f64>(plan.scratch_len() + 1, 12.0);
        let scratch_before = scratch.clone();
        assert_eq!(
            plan.inverse(&source, &mut destination, &mut line, &mut scratch),
            Err(LocalR2cError::BatchCountMismatch {
                source_count: 1,
                destination_count: 0,
            })
        );
        assert_eq!(source, source_before);
        assert_eq!(destination, destination_before);
        assert_eq!(line, line_before);
        assert_eq!(scratch, scratch_before);
    }

    #[test]
    fn short_line_and_scratch_errors_preserve_all_buffers() {
        let plan = LocalR2cPlan::<f64>::new(5).unwrap();
        assert!(plan.scratch_len() > 0);

        for batch_count in [0, 1] {
            let source = real_input::<f64>(plan.real_len(), batch_count);
            let source_before = source.clone();
            let mut spectrum = initialized_complex::<f64>(plan.complex_len() * batch_count, 5.0);
            let spectrum_before = spectrum.clone();
            let mut line = vec![0.0; plan.real_len() - 1];
            let line_before = line.clone();
            let mut scratch = initialized_complex::<f64>(plan.scratch_len(), 6.0);
            let scratch_before = scratch.clone();
            assert_eq!(
                plan.forward(&source, &mut spectrum, &mut line, &mut scratch),
                Err(LocalR2cError::RealLineTooSmall {
                    required: plan.real_len(),
                    actual: plan.real_len() - 1,
                })
            );
            assert_eq!(source, source_before);
            assert_eq!(spectrum, spectrum_before);
            assert_eq!(line, line_before);
            assert_eq!(scratch, scratch_before);

            let mut spectrum = initialized_complex::<f64>(plan.complex_len() * batch_count, 9.0);
            let spectrum_before = spectrum.clone();
            let mut line = vec![0.0; plan.real_len() + 1];
            let line_before = line.clone();
            let mut short_scratch = initialized_complex::<f64>(plan.scratch_len() - 1, 10.0);
            let short_scratch_before = short_scratch.clone();
            assert_eq!(
                plan.forward(&source, &mut spectrum, &mut line, &mut short_scratch),
                Err(LocalR2cError::ScratchTooSmall {
                    required: plan.scratch_len(),
                    actual: plan.scratch_len() - 1,
                })
            );
            assert_eq!(source, source_before);
            assert_eq!(spectrum, spectrum_before);
            assert_eq!(line, line_before);
            assert_eq!(short_scratch, short_scratch_before);

            let mut inverse_source =
                initialized_complex::<f64>(plan.complex_len() * batch_count, 7.0);
            for line in inverse_source.chunks_exact_mut(plan.complex_len()) {
                line[0].im = 0.0;
            }
            let inverse_source_before = inverse_source.clone();
            let mut output = vec![0.0; plan.real_len() * batch_count];
            let output_before = output.clone();
            let mut complex_line = vec![Complex::new(0.0, 0.0); plan.complex_len() - 1];
            let complex_line_before = complex_line.clone();
            let mut scratch = initialized_complex::<f64>(plan.scratch_len(), 11.0);
            let scratch_before = scratch.clone();
            assert_eq!(
                plan.inverse(
                    &inverse_source,
                    &mut output,
                    &mut complex_line,
                    &mut scratch,
                ),
                Err(LocalR2cError::ComplexLineTooSmall {
                    required: plan.complex_len(),
                    actual: plan.complex_len() - 1,
                })
            );
            assert_eq!(inverse_source, inverse_source_before);
            assert_eq!(output, output_before);
            assert_eq!(complex_line, complex_line_before);
            assert_eq!(scratch, scratch_before);

            let mut output = vec![0.0; plan.real_len() * batch_count];
            let output_before = output.clone();
            let mut full_complex_line = initialized_complex::<f64>(plan.complex_len() + 1, 12.0);
            let full_complex_line_before = full_complex_line.clone();
            let mut short_scratch = initialized_complex::<f64>(plan.scratch_len() - 1, 13.0);
            let short_scratch_before = short_scratch.clone();
            assert_eq!(
                plan.inverse(
                    &inverse_source,
                    &mut output,
                    &mut full_complex_line,
                    &mut short_scratch,
                ),
                Err(LocalR2cError::ScratchTooSmall {
                    required: plan.scratch_len(),
                    actual: plan.scratch_len() - 1,
                })
            );
            assert_eq!(inverse_source, inverse_source_before);
            assert_eq!(output, output_before);
            assert_eq!(full_complex_line, full_complex_line_before);
            assert_eq!(short_scratch, short_scratch_before);
        }
    }

    #[test]
    fn plan_scratch_len_is_the_shared_native_maximum() {
        for real_len in [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 17] {
            let f32_plan = LocalR2cPlan::<f32>::new(real_len).unwrap();
            let f64_plan = LocalR2cPlan::<f64>::new(real_len).unwrap();
            assert_eq!(f32_plan.scratch_len(), native_scratch_len::<f32>(real_len));
            assert_eq!(f64_plan.scratch_len(), native_scratch_len::<f64>(real_len));
        }
    }

    fn assert_length_overflow_before_planning<R: FftReal>(real_len: usize) {
        assert_eq!(
            validate_lengths::<R>(real_len),
            Err(LocalR2cError::LengthOverflow)
        );
        assert!(matches!(
            LocalR2cPlan::<R>::new(real_len),
            Err(LocalR2cError::LengthOverflow)
        ));
    }

    fn check_length_boundaries<R: FftReal>() {
        let max_complex_len = isize::MAX as usize / size_of::<Complex<R>>();
        let max_real_len = max_complex_len * 2 - 2;
        assert!(validate_lengths::<R>(max_real_len).is_ok());

        let real_address_overflow = isize::MAX as usize / size_of::<R>() + 1;
        assert_length_overflow_before_planning::<R>(real_address_overflow);

        let complex_address_overflow = max_complex_len * 2;
        assert_length_overflow_before_planning::<R>(complex_address_overflow);

        let odd_staging_overflow = max_complex_len + 2;
        assert_eq!(odd_staging_overflow % 2, 1);
        assert_length_overflow_before_planning::<R>(odd_staging_overflow);
    }

    #[test]
    fn zero_lengths_are_rejected_for_both_scalars() {
        assert!(matches!(
            LocalR2cPlan::<f32>::new(0),
            Err(LocalR2cError::InvalidLength)
        ));
        assert!(matches!(
            LocalR2cPlan::<f64>::new(0),
            Err(LocalR2cError::InvalidLength)
        ));
    }

    #[test]
    fn impossible_lengths_are_rejected_before_realfft_planning_for_both_scalars() {
        check_length_boundaries::<f32>();
        check_length_boundaries::<f64>();
    }
}
