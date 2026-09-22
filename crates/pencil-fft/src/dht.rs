use std::fmt;
use std::marker::PhantomData;
use std::mem::size_of;
use std::sync::Arc;

use rustfft::{Fft, FftPlanner};

use super::r2r::{LocalR2rError, R2rScalar, r2r_from_complex, r2r_to_complex};
use super::{Complex, FftReal};

/// A local batched discrete Hartley transform plan.
///
/// `T` may be `f32`, `f64`, `Complex<f32>`, or `Complex<f64>`. Each data
/// slice contains consecutive lines of [`Self::line_len`] values. The plan
/// owns an immutable forward RustFFT plan; callers own an initialized complex
/// embedding line of at least [`Self::embedding_len`] values and initialized
/// native scratch of at least [`Self::scratch_len`] values.
///
/// `forward` and `backward` compute the same unnormalized Hartley transform.
/// `inverse` computes that transform and divides each line by `line_len`.
/// For real `T`, only the real component is written; for complex `T`, the real
/// and imaginary components are transformed independently.
///
/// The implementation uses one native complex FFT per line and no full-array
/// temporary storage. Non-finite values follow ordinary IEEE arithmetic.
pub struct LocalDhtPlan<T: R2rScalar> {
    line_len: usize,
    embedding_len: usize,
    scratch_len: usize,
    fft: Arc<dyn Fft<T::Real>>,
    marker: PhantomData<T>,
    backend: super::BackendKind,
    #[cfg(feature = "fftw")]
    backend_options: Option<super::PlanOptions>,
}

impl<T: R2rScalar> fmt::Debug for LocalDhtPlan<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LocalDhtPlan")
            .field("line_len", &self.line_len)
            .field("embedding_len", &self.embedding_len)
            .field("scratch_len", &self.scratch_len)
            .finish_non_exhaustive()
    }
}

impl<T: R2rScalar> LocalDhtPlan<T> {
    /// Builds a plan for lines of `line_len` values.
    ///
    /// Zero and unaddressable line lengths are rejected before RustFFT
    /// planning. Backend resource failures and backend panics are not
    /// converted into [`LocalR2rError`].
    pub fn new(line_len: usize) -> Result<Self, LocalR2rError> {
        validate_lengths::<T>(line_len)?;

        let mut planner = FftPlanner::<T::Real>::new();
        let fft = planner.plan_fft_forward(line_len);
        Self::from_embedding(
            line_len,
            fft,
            super::BackendKind::RustFft,
            #[cfg(feature = "fftw")]
            None,
        )
    }

    fn from_embedding(
        line_len: usize,
        fft: Arc<dyn Fft<T::Real>>,
        backend: super::BackendKind,
        #[cfg(feature = "fftw")] backend_options: Option<super::PlanOptions>,
    ) -> Result<Self, LocalR2rError> {
        let scratch_len = fft.get_inplace_scratch_len();
        validate_addressable(scratch_len, size_of::<Complex<T::Real>>())?;
        Ok(Self {
            line_len,
            embedding_len: line_len,
            scratch_len,
            fft,
            marker: PhantomData,
            backend,
            #[cfg(feature = "fftw")]
            backend_options,
        })
    }

    /// Builds a plan using the runtime-loaded FFTW backend.
    #[cfg(feature = "fftw")]
    #[allow(private_bounds)]
    pub fn new_fftw(
        line_len: usize,
        options: super::PlanOptions,
    ) -> Result<Self, super::BackendInitError<LocalR2rError>>
    where
        T::Real: super::backend::FftwReal,
    {
        validate_lengths::<T>(line_len).map_err(super::BackendInitError::Local)?;
        let fft = super::backend::c2c(line_len, rustfft::FftDirection::Forward, options)
            .map_err(super::BackendInitError::Native)?;
        Self::from_embedding(line_len, fft, super::BackendKind::Fftw, Some(options))
            .map_err(super::BackendInitError::Local)
    }

    /// Returns the selected backend.
    pub fn backend_kind(&self) -> super::BackendKind {
        self.backend
    }

    /// Returns the FFTW options, when this plan uses FFTW.
    #[cfg(feature = "fftw")]
    pub fn backend_options(&self) -> Option<super::PlanOptions> {
        self.backend_options
    }

    /// Returns the number of values in each transformed line.
    pub fn line_len(&self) -> usize {
        self.line_len
    }

    /// Returns the number of complex values required in the embedding line.
    pub fn embedding_len(&self) -> usize {
        self.embedding_len
    }

    /// Returns the native RustFFT in-place scratch length.
    pub fn scratch_len(&self) -> usize {
        self.scratch_len
    }

    /// Returns the raw forward/backward composition factor.
    pub fn normalization_factor(&self) -> usize {
        self.line_len
    }

    /// Computes the unnormalized Hartley transform for every source line.
    ///
    /// `src` and `dst` must have equal lengths, each a multiple of
    /// [`Self::line_len`]. The embedding line and scratch must be initialized
    /// and satisfy [`Self::embedding_len`] and [`Self::scratch_len`]. Source
    /// data and oversized workspace tails are preserved. A valid empty batch
    /// is a checked no-op.
    pub fn forward(
        &self,
        src: &[T],
        dst: &mut [T],
        embedding_line: &mut [Complex<T::Real>],
        scratch: &mut [Complex<T::Real>],
    ) -> Result<(), LocalR2rError> {
        self.execute_out_of_place(src, dst, embedding_line, scratch, false)
    }

    /// Computes the raw positive-sign backward Hartley transform.
    ///
    /// The discrete Hartley transform is self-paired, so this is the same
    /// unnormalized operation as [`Self::forward`].
    pub fn backward(
        &self,
        src: &[T],
        dst: &mut [T],
        embedding_line: &mut [Complex<T::Real>],
        scratch: &mut [Complex<T::Real>],
    ) -> Result<(), LocalR2rError> {
        self.execute_out_of_place(src, dst, embedding_line, scratch, false)
    }

    /// Computes the normalized self-paired inverse Hartley transform.
    ///
    /// Each output line is divided by [`Self::normalization_factor`].
    pub fn inverse(
        &self,
        src: &[T],
        dst: &mut [T],
        embedding_line: &mut [Complex<T::Real>],
        scratch: &mut [Complex<T::Real>],
    ) -> Result<(), LocalR2rError> {
        self.execute_out_of_place(src, dst, embedding_line, scratch, true)
    }

    /// Computes the unnormalized Hartley transform in place.
    pub fn forward_in_place(
        &self,
        data: &mut [T],
        embedding_line: &mut [Complex<T::Real>],
        scratch: &mut [Complex<T::Real>],
    ) -> Result<(), LocalR2rError> {
        self.execute_in_place(data, embedding_line, scratch, false)
    }

    /// Computes the raw positive-sign backward Hartley transform in place.
    pub fn backward_in_place(
        &self,
        data: &mut [T],
        embedding_line: &mut [Complex<T::Real>],
        scratch: &mut [Complex<T::Real>],
    ) -> Result<(), LocalR2rError> {
        self.execute_in_place(data, embedding_line, scratch, false)
    }

    /// Computes the normalized self-paired inverse Hartley transform in place.
    pub fn inverse_in_place(
        &self,
        data: &mut [T],
        embedding_line: &mut [Complex<T::Real>],
        scratch: &mut [Complex<T::Real>],
    ) -> Result<(), LocalR2rError> {
        self.execute_in_place(data, embedding_line, scratch, true)
    }

    fn execute_out_of_place(
        &self,
        src: &[T],
        dst: &mut [T],
        embedding_line: &mut [Complex<T::Real>],
        scratch: &mut [Complex<T::Real>],
        normalize: bool,
    ) -> Result<(), LocalR2rError> {
        self.validate_out_of_place(src.len(), dst.len(), embedding_line.len(), scratch.len())?;
        if src.is_empty() {
            return Ok(());
        }
        if self.line_len == 1 {
            dst.copy_from_slice(src);
            return Ok(());
        }

        for (source_line, destination_line) in src
            .chunks_exact(self.line_len)
            .zip(dst.chunks_exact_mut(self.line_len))
        {
            self.execute_line(
                source_line,
                destination_line,
                embedding_line,
                scratch,
                normalize,
            );
        }
        Ok(())
    }

    fn execute_in_place(
        &self,
        data: &mut [T],
        embedding_line: &mut [Complex<T::Real>],
        scratch: &mut [Complex<T::Real>],
        normalize: bool,
    ) -> Result<(), LocalR2rError> {
        self.validate_in_place(data.len(), embedding_line.len(), scratch.len())?;
        if data.is_empty() || self.line_len == 1 {
            return Ok(());
        }

        for data_line in data.chunks_exact_mut(self.line_len) {
            self.execute_in_place_line(data_line, embedding_line, scratch, normalize);
        }
        Ok(())
    }

    fn execute_line(
        &self,
        source: &[T],
        destination: &mut [T],
        embedding_line: &mut [Complex<T::Real>],
        scratch: &mut [Complex<T::Real>],
        normalize: bool,
    ) {
        self.fill_and_fft(source, embedding_line, scratch);
        self.write_line(destination, embedding_line, normalize);
    }

    fn execute_in_place_line(
        &self,
        data: &mut [T],
        embedding_line: &mut [Complex<T::Real>],
        scratch: &mut [Complex<T::Real>],
        normalize: bool,
    ) {
        self.fill_and_fft(data, embedding_line, scratch);
        self.write_line(data, embedding_line, normalize);
    }

    fn fill_and_fft(
        &self,
        source: &[T],
        embedding_line: &mut [Complex<T::Real>],
        scratch: &mut [Complex<T::Real>],
    ) {
        for (embedding, &value) in embedding_line[..self.embedding_len].iter_mut().zip(source) {
            *embedding = r2r_to_complex(value);
        }
        self.fft.process_with_scratch(
            &mut embedding_line[..self.embedding_len],
            &mut scratch[..self.scratch_len],
        );
    }

    fn write_line(
        &self,
        destination: &mut [T],
        embedding_line: &[Complex<T::Real>],
        normalize: bool,
    ) {
        for (k, destination) in destination.iter_mut().enumerate() {
            let value = embedding_line[k];
            let value = if k == 0 || (self.line_len % 2 == 0 && k == self.line_len / 2) {
                scaled(value, normalize, self.line_len)
            } else {
                let mirror = embedding_line[self.line_len - k];
                let real = half_sum(real_as_f64(value.re), real_as_f64(mirror.re))
                    + half_sum(real_as_f64(mirror.im), -real_as_f64(value.im));
                let imaginary = half_sum(real_as_f64(value.im), real_as_f64(mirror.im))
                    + half_sum(real_as_f64(value.re), -real_as_f64(mirror.re));
                scaled_f64(real, imaginary, normalize, self.line_len)
            };
            *destination = r2r_from_complex(value);
        }
    }

    fn validate_out_of_place(
        &self,
        source_len: usize,
        destination_len: usize,
        embedding_len: usize,
        scratch_len: usize,
    ) -> Result<(), LocalR2rError> {
        if source_len != destination_len {
            return Err(LocalR2rError::BufferLengthMismatch);
        }
        self.validate_in_place(source_len, embedding_len, scratch_len)
    }

    fn validate_in_place(
        &self,
        data_len: usize,
        embedding_len: usize,
        scratch_len: usize,
    ) -> Result<(), LocalR2rError> {
        if data_len % self.line_len != 0 {
            return Err(LocalR2rError::NonIntegralBatch);
        }
        if embedding_len < self.embedding_len {
            return Err(LocalR2rError::ComplexLineTooSmall {
                required: self.embedding_len,
                actual: embedding_len,
            });
        }
        if scratch_len < self.scratch_len {
            return Err(LocalR2rError::ScratchTooSmall {
                required: self.scratch_len,
                actual: scratch_len,
            });
        }
        Ok(())
    }
}

fn validate_lengths<T: R2rScalar>(line_len: usize) -> Result<(), LocalR2rError> {
    if line_len == 0 {
        return Err(LocalR2rError::InvalidLength);
    }
    validate_addressable(line_len, size_of::<T>())?;
    validate_addressable(line_len, size_of::<Complex<T::Real>>())?;
    Ok(())
}

fn validate_addressable(length: usize, element_size: usize) -> Result<(), LocalR2rError> {
    let bytes = length
        .checked_mul(element_size)
        .ok_or(LocalR2rError::LengthOverflow)?;
    if bytes > isize::MAX as usize {
        return Err(LocalR2rError::LengthOverflow);
    }
    Ok(())
}

fn real_as_f64<R: FftReal>(value: R) -> f64 {
    <R as crate::private::Sealed>::pencil_fft_as_f64(value)
}

fn real_from_f64<R: FftReal>(value: f64) -> R {
    R::from_f64(value).expect("f32 and f64 can represent every finite conversion result")
}

fn scaled<R: FftReal>(value: Complex<R>, normalize: bool, line_len: usize) -> Complex<R> {
    if !normalize {
        value
    } else {
        scaled_f64(real_as_f64(value.re), real_as_f64(value.im), true, line_len)
    }
}

fn scaled_f64<R: FftReal>(
    real: f64,
    imaginary: f64,
    normalize: bool,
    line_len: usize,
) -> Complex<R> {
    if normalize {
        let scale = line_len as f64;
        Complex::new(
            real_from_f64::<R>(real / scale),
            real_from_f64::<R>(imaginary / scale),
        )
    } else {
        Complex::new(real_from_f64::<R>(real), real_from_f64::<R>(imaginary))
    }
}

fn half_sum(x: f64, y: f64) -> f64 {
    if x.abs() <= f64::MAX / 2.0 && y.abs() <= f64::MAX / 2.0 {
        (x + y) * 0.5
    } else {
        x * 0.5 + y * 0.5
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    trait TestScalar: R2rScalar + PartialEq {
        fn from_f64(real: f64, imaginary: f64) -> Self;
        fn as_f64(self) -> Complex<f64>;
        fn tolerance() -> f64;
    }

    impl TestScalar for f32 {
        fn from_f64(real: f64, _imaginary: f64) -> Self {
            real as f32
        }

        fn as_f64(self) -> Complex<f64> {
            Complex::new(self as f64, 0.0)
        }

        fn tolerance() -> f64 {
            5e-4
        }
    }

    impl TestScalar for f64 {
        fn from_f64(real: f64, _imaginary: f64) -> Self {
            real
        }

        fn as_f64(self) -> Complex<f64> {
            Complex::new(self, 0.0)
        }

        fn tolerance() -> f64 {
            2e-10
        }
    }

    impl TestScalar for Complex<f32> {
        fn from_f64(real: f64, imaginary: f64) -> Self {
            Complex::new(real as f32, imaginary as f32)
        }

        fn as_f64(self) -> Complex<f64> {
            Complex::new(self.re as f64, self.im as f64)
        }

        fn tolerance() -> f64 {
            6e-4
        }
    }

    impl TestScalar for Complex<f64> {
        fn from_f64(real: f64, imaginary: f64) -> Self {
            Complex::new(real, imaginary)
        }

        fn as_f64(self) -> Complex<f64> {
            self
        }

        fn tolerance() -> f64 {
            3e-10
        }
    }

    fn input<T: TestScalar>(line_len: usize, batches: usize) -> Vec<T> {
        (0..line_len * batches)
            .map(|index| {
                let x = index as f64 + 0.37;
                T::from_f64((0.17 * x).cos() - 0.04 * x, (0.23 * x).sin() + 0.06 * x)
            })
            .collect()
    }

    fn complex_input<T: TestScalar>(values: &[T]) -> Vec<Complex<f64>> {
        values.iter().copied().map(TestScalar::as_f64).collect()
    }

    fn direct(input: &[Complex<f64>]) -> Vec<Complex<f64>> {
        let line_len = input.len();
        (0..line_len)
            .map(|k| {
                let mut real = 0.0;
                let mut imaginary = 0.0;
                for (j, value) in input.iter().enumerate() {
                    let angle = TAU * j as f64 * k as f64 / line_len as f64;
                    let coefficient = angle.cos() + angle.sin();
                    real += value.re * coefficient;
                    imaginary += value.im * coefficient;
                }
                Complex::new(real, imaginary)
            })
            .collect()
    }

    fn expected<T: TestScalar>(
        source: &[T],
        line_len: usize,
        normalize: bool,
    ) -> Vec<Complex<f64>> {
        let mut result = source
            .chunks_exact(line_len)
            .flat_map(|line| direct(&complex_input(line)))
            .collect::<Vec<_>>();
        if normalize {
            let scale = line_len as f64;
            for value in &mut result {
                value.re /= scale;
                value.im /= scale;
            }
        }
        result
    }

    fn assert_close<T: TestScalar>(actual: &[T], expected: &[Complex<f64>]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().copied().zip(expected).enumerate() {
            let actual = actual.as_f64();
            let real_bound = T::tolerance() * (1.0 + expected.re.abs());
            let imaginary_bound = T::tolerance() * (1.0 + expected.im.abs());
            let real_error = (actual.re - expected.re).abs();
            let imaginary_error = (actual.im - expected.im).abs();
            assert!(
                real_error <= real_bound && imaginary_error <= imaginary_bound,
                "index {index}: actual={actual:?}, expected={expected:?}, real_bound={real_bound}, imaginary_bound={imaginary_bound}"
            );
        }
    }

    fn workspace<T: TestScalar>(length: usize, offset: f64) -> Vec<Complex<T::Real>> {
        (0..length)
            .map(|index| {
                Complex::new(
                    real_from_f64(offset + index as f64),
                    real_from_f64(-offset - index as f64),
                )
            })
            .collect()
    }

    #[derive(Clone, Copy)]
    enum Direction {
        Forward,
        Backward,
        Inverse,
    }

    const DIRECTIONS: [Direction; 3] =
        [Direction::Forward, Direction::Backward, Direction::Inverse];

    impl Direction {
        fn normalize(self) -> bool {
            matches!(self, Self::Inverse)
        }

        fn out_of_place<T: R2rScalar>(
            self,
            plan: &LocalDhtPlan<T>,
            source: &[T],
            destination: &mut [T],
            embedding: &mut [Complex<T::Real>],
            scratch: &mut [Complex<T::Real>],
        ) -> Result<(), LocalR2rError> {
            match self {
                Self::Forward => plan.forward(source, destination, embedding, scratch),
                Self::Backward => plan.backward(source, destination, embedding, scratch),
                Self::Inverse => plan.inverse(source, destination, embedding, scratch),
            }
        }

        fn in_place<T: R2rScalar>(
            self,
            plan: &LocalDhtPlan<T>,
            data: &mut [T],
            embedding: &mut [Complex<T::Real>],
            scratch: &mut [Complex<T::Real>],
        ) -> Result<(), LocalR2rError> {
            match self {
                Self::Forward => plan.forward_in_place(data, embedding, scratch),
                Self::Backward => plan.backward_in_place(data, embedding, scratch),
                Self::Inverse => plan.inverse_in_place(data, embedding, scratch),
            }
        }
    }

    fn assert_in_place_error(
        direction: Direction,
        plan: &LocalDhtPlan<f64>,
        data: &mut [f64],
        embedding: &mut [Complex<f64>],
        scratch: &mut [Complex<f64>],
        expected: LocalR2rError,
    ) {
        let data_before = data.to_vec();
        let embedding_before = embedding.to_vec();
        let scratch_before = scratch.to_vec();
        assert_eq!(
            direction.in_place(plan, data, embedding, scratch),
            Err(expected)
        );
        assert_eq!(data, data_before.as_slice());
        assert_eq!(embedding, embedding_before.as_slice());
        assert_eq!(scratch, scratch_before.as_slice());
    }

    fn exercise<T: TestScalar>(line_len: usize, batches: usize) {
        let plan = LocalDhtPlan::<T>::new(line_len).unwrap();
        assert_eq!(plan.line_len(), line_len);
        assert_eq!(plan.embedding_len(), line_len);
        assert_eq!(plan.normalization_factor(), line_len);

        let source = input::<T>(line_len, batches);
        for direction in DIRECTIONS {
            let expected = expected(&source, line_len, direction.normalize());
            let mut destination = vec![T::from_f64(17.0, -19.0); source.len()];
            let mut embedding = workspace::<T>(plan.embedding_len() + 2, 100.0);
            let embedding_tail = embedding[plan.embedding_len()..].to_vec();
            let mut scratch = workspace::<T>(plan.scratch_len() + 2, 200.0);
            let scratch_tail = scratch[plan.scratch_len()..].to_vec();
            direction
                .out_of_place(
                    &plan,
                    &source,
                    &mut destination,
                    &mut embedding,
                    &mut scratch,
                )
                .unwrap();
            assert_close(&destination, &expected);
            assert_eq!(&embedding[plan.embedding_len()..], &embedding_tail);
            assert_eq!(&scratch[plan.scratch_len()..], &scratch_tail);

            let mut in_place = source.clone();
            let mut embedding = workspace::<T>(plan.embedding_len() + 2, 300.0);
            let embedding_tail = embedding[plan.embedding_len()..].to_vec();
            let mut scratch = workspace::<T>(plan.scratch_len() + 2, 400.0);
            let scratch_tail = scratch[plan.scratch_len()..].to_vec();
            direction
                .in_place(&plan, &mut in_place, &mut embedding, &mut scratch)
                .unwrap();
            assert_close(&in_place, &expected);
            assert_eq!(&embedding[plan.embedding_len()..], &embedding_tail);
            assert_eq!(&scratch[plan.scratch_len()..], &scratch_tail);
        }

        let mut transformed = vec![T::from_f64(0.0, 0.0); source.len()];
        let mut embedding = workspace::<T>(plan.embedding_len(), 500.0);
        let mut scratch = workspace::<T>(plan.scratch_len(), 600.0);
        plan.forward(&source, &mut transformed, &mut embedding, &mut scratch)
            .unwrap();
        let mut raw = vec![T::from_f64(0.0, 0.0); source.len()];
        plan.backward(&transformed, &mut raw, &mut embedding, &mut scratch)
            .unwrap();
        let raw_expected = source
            .iter()
            .copied()
            .map(|value| {
                let value = value.as_f64();
                Complex::new(value.re * line_len as f64, value.im * line_len as f64)
            })
            .collect::<Vec<_>>();
        assert_close(&raw, &raw_expected);
    }

    #[test]
    fn all_scalar_types_and_lengths_match_independent_dht_oracles() {
        for line_len in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13] {
            exercise::<f32>(line_len, 2);
            exercise::<f64>(line_len, 2);
            exercise::<Complex<f32>>(line_len, 2);
            exercise::<Complex<f64>>(line_len, 2);
        }
    }

    #[test]
    fn empty_batches_and_validation_errors_are_atomic_for_all_operations() {
        let plan = LocalDhtPlan::<f64>::new(4).unwrap();
        for direction in DIRECTIONS {
            let mut embedding = workspace::<f64>(plan.embedding_len() + 1, 10.0);
            let embedding_before = embedding.clone();
            let mut scratch = workspace::<f64>(plan.scratch_len() + 1, 20.0);
            let scratch_before = scratch.clone();
            let mut destination = Vec::new();
            assert_eq!(
                direction.out_of_place(&plan, &[], &mut destination, &mut embedding, &mut scratch,),
                Ok(())
            );
            assert!(destination.is_empty());
            assert_eq!(embedding, embedding_before);
            assert_eq!(scratch, scratch_before);

            let mut data = Vec::new();
            assert_eq!(
                direction.in_place(&plan, &mut data, &mut embedding, &mut scratch),
                Ok(())
            );
        }

        for direction in DIRECTIONS {
            let source = [1.0, 2.0, 3.0, 4.0];
            let mut destination = [9.0, 9.0, 9.0];
            let destination_before = destination;
            let mut embedding = workspace::<f64>(plan.embedding_len(), 30.0);
            let embedding_before = embedding.clone();
            let mut scratch = workspace::<f64>(plan.scratch_len(), 40.0);
            let scratch_before = scratch.clone();
            assert_eq!(
                direction.out_of_place(
                    &plan,
                    &source,
                    &mut destination,
                    &mut embedding,
                    &mut scratch,
                ),
                Err(LocalR2rError::BufferLengthMismatch)
            );
            assert_eq!(destination, destination_before);
            assert_eq!(embedding, embedding_before);
            assert_eq!(scratch, scratch_before);

            let mut data = [1.0, 2.0, 3.0];
            let data_before = data;
            assert_eq!(
                direction.in_place(&plan, &mut data, &mut embedding, &mut scratch),
                Err(LocalR2rError::NonIntegralBatch)
            );
            assert_eq!(data, data_before);

            let source = [1.0, 2.0, 3.0];
            let mut destination = [9.0, 9.0, 9.0];
            assert_eq!(
                direction.out_of_place(
                    &plan,
                    &source,
                    &mut destination,
                    &mut embedding,
                    &mut scratch,
                ),
                Err(LocalR2rError::NonIntegralBatch)
            );
            assert_eq!(destination, [9.0, 9.0, 9.0]);

            let source = [1.0, 2.0, 3.0, 4.0];
            let mut destination = [9.0, 9.0, 9.0, 9.0];
            let mut short_embedding = workspace::<f64>(plan.embedding_len() - 1, 50.0);
            let embedding_before = short_embedding.clone();
            let mut scratch = workspace::<f64>(plan.scratch_len(), 60.0);
            let scratch_before = scratch.clone();
            assert_eq!(
                direction.out_of_place(
                    &plan,
                    &source,
                    &mut destination,
                    &mut short_embedding,
                    &mut scratch,
                ),
                Err(LocalR2rError::ComplexLineTooSmall {
                    required: plan.embedding_len(),
                    actual: plan.embedding_len() - 1,
                })
            );
            assert_eq!(destination, [9.0, 9.0, 9.0, 9.0]);
            assert_eq!(short_embedding, embedding_before);
            assert_eq!(scratch, scratch_before);

            if plan.scratch_len() > 0 {
                let mut destination = [9.0, 9.0, 9.0, 9.0];
                let mut embedding = workspace::<f64>(plan.embedding_len(), 70.0);
                let embedding_before = embedding.clone();
                let mut short_scratch = workspace::<f64>(plan.scratch_len() - 1, 80.0);
                let scratch_before = short_scratch.clone();
                assert_eq!(
                    direction.out_of_place(
                        &plan,
                        &source,
                        &mut destination,
                        &mut embedding,
                        &mut short_scratch,
                    ),
                    Err(LocalR2rError::ScratchTooSmall {
                        required: plan.scratch_len(),
                        actual: plan.scratch_len() - 1,
                    })
                );
                assert_eq!(destination, [9.0, 9.0, 9.0, 9.0]);
                assert_eq!(embedding, embedding_before);
                assert_eq!(short_scratch, scratch_before);
            }
        }
    }

    #[test]
    fn in_place_validation_errors_are_atomic_for_all_operations() {
        let plan = LocalDhtPlan::<f64>::new(37).unwrap();
        assert!(plan.scratch_len() > 0);

        for direction in DIRECTIONS {
            let mut data = vec![1.0; plan.line_len() - 1];
            let mut embedding = workspace::<f64>(plan.embedding_len(), 10.0);
            let mut scratch = workspace::<f64>(plan.scratch_len(), 20.0);
            assert_in_place_error(
                direction,
                &plan,
                &mut data,
                &mut embedding,
                &mut scratch,
                LocalR2rError::NonIntegralBatch,
            );

            let mut data = vec![1.0; plan.line_len()];
            let mut embedding = workspace::<f64>(plan.embedding_len() - 1, 30.0);
            let mut scratch = workspace::<f64>(plan.scratch_len(), 40.0);
            assert_in_place_error(
                direction,
                &plan,
                &mut data,
                &mut embedding,
                &mut scratch,
                LocalR2rError::ComplexLineTooSmall {
                    required: plan.embedding_len(),
                    actual: plan.embedding_len() - 1,
                },
            );

            let mut data = vec![1.0; plan.line_len()];
            let mut embedding = workspace::<f64>(plan.embedding_len(), 50.0);
            let mut scratch = workspace::<f64>(plan.scratch_len() - 1, 60.0);
            assert_in_place_error(
                direction,
                &plan,
                &mut data,
                &mut embedding,
                &mut scratch,
                LocalR2rError::ScratchTooSmall {
                    required: plan.scratch_len(),
                    actual: plan.scratch_len() - 1,
                },
            );
        }
    }

    #[test]
    fn self_bins_and_n_one_preserve_the_special_cases() {
        let plan = LocalDhtPlan::<Complex<f64>>::new(1).unwrap();
        let source = [Complex::new(-0.0, f64::NAN)];
        let mut destination = [Complex::new(1.0, 2.0)];
        let mut embedding = [Complex::new(3.0, 4.0)];
        let mut scratch = vec![Complex::new(0.0, 0.0); plan.scratch_len()];
        plan.forward(&source, &mut destination, &mut embedding, &mut scratch)
            .unwrap();
        assert_eq!(destination[0].re.to_bits(), source[0].re.to_bits());
        assert!(destination[0].im.is_nan());

        let plan = LocalDhtPlan::<Complex<f64>>::new(2).unwrap();
        let source = [Complex::new(1.25, -2.5), Complex::new(-3.0, 4.0)];
        let mut destination = [Complex::new(0.0, 0.0); 2];
        let mut embedding = [Complex::new(0.0, 0.0); 2];
        let mut scratch = vec![Complex::new(0.0, 0.0); plan.scratch_len()];
        plan.forward(&source, &mut destination, &mut embedding, &mut scratch)
            .unwrap();
        assert_eq!(destination[0], Complex::new(-1.75, 1.5));
        assert_eq!(destination[1], Complex::new(4.25, -6.5));
    }

    #[test]
    fn half_sum_handles_large_and_tiny_frequency_pairs() {
        let plan = LocalDhtPlan::<f64>::new(4).unwrap();
        let half_max = f64::MAX / 2.0;
        let source = [half_max, 0.0, -half_max, 0.0];
        let mut destination = [0.0; 4];
        let mut embedding = vec![Complex::new(0.0, 0.0); plan.embedding_len()];
        let mut scratch = vec![Complex::new(0.0, 0.0); plan.scratch_len()];
        plan.forward(&source, &mut destination, &mut embedding, &mut scratch)
            .unwrap();
        assert_eq!(destination[1].to_bits(), f64::MAX.to_bits());
        assert_eq!(destination[3].to_bits(), f64::MAX.to_bits());

        let min = f64::from_bits(1);
        let source = [min, 0.0, 0.0, 0.0];
        plan.forward(&source, &mut destination, &mut embedding, &mut scratch)
            .unwrap();
        assert_eq!(destination[1].to_bits(), min.to_bits());
        assert_eq!(destination[3].to_bits(), min.to_bits());
        assert_eq!(half_sum(f64::MAX, f64::MAX), f64::MAX);
        assert_eq!(half_sum(min, min), min);
    }

    #[test]
    fn nonfinite_values_are_not_rejected() {
        let plan = LocalDhtPlan::<f64>::new(3).unwrap();
        let source = [f64::INFINITY, f64::NAN, -f64::INFINITY];
        let mut destination = [0.0; 3];
        let mut embedding = vec![Complex::new(0.0, 0.0); plan.embedding_len()];
        let mut scratch = vec![Complex::new(0.0, 0.0); plan.scratch_len()];
        assert!(
            plan.forward(&source, &mut destination, &mut embedding, &mut scratch)
                .is_ok()
        );
        assert!(
            plan.backward(&source, &mut destination, &mut embedding, &mut scratch)
                .is_ok()
        );
        assert!(
            plan.inverse(&source, &mut destination, &mut embedding, &mut scratch)
                .is_ok()
        );
    }

    #[test]
    fn invalid_lengths_are_rejected_before_planning() {
        assert!(matches!(
            LocalDhtPlan::<f64>::new(0),
            Err(LocalR2rError::InvalidLength)
        ));
        assert!(matches!(
            LocalDhtPlan::<f64>::new(usize::MAX),
            Err(LocalR2rError::LengthOverflow)
        ));
    }
}
