#![forbid(unsafe_code)]
#![warn(missing_docs)]

//! Local one-dimensional FFTs.
//!
//! [`LocalC2cPlan`] treats each input slice as a row-major batch of contiguous
//! complex lines. [`LocalR2cPlan`] does the same for real-to-half-complex and
//! half-complex-to-real transforms, and also provides a packed, state-checked
//! single-allocation in-place API. Plans own immutable backend plans, while
//! callers own initialized line buffers, scratch storage, and data buffers.
//! Local C2C and local R2C/C2R forward and backward transforms are
//! unnormalized; backward uses the positive-sign convention. Local C2C
//! inverse and R2C/C2R inverse divide by the line length: the complex length
//! for C2C, or the original real length for R2C/C2R.
//!
//! [`LocalR2rPlan`] adds the eight FFTW-compatible DCT/DST-I-IV kinds for the
//! four supported scalar types, with component-wise complex transforms and raw
//! paired backward plus normalized inverse operations. [`LocalDhtPlan`] adds
//! the local self-paired discrete Hartley transform over the same scalar types.
//!
//! With the opt-in `distributed` feature, `C2cPlan` and its workspaces provide
//! input-preserving distributed C2C transforms, while `R2cPlan` provides
//! input-preserving out-of-place R2C/C2R forward, normalized inverse, and raw
//! backward with a selected-axis reduction. `AxisSelection` keeps the full
//! canonical route while making unselected stages identities; C2C also retains
//! its state-checked single-buffer API; distributed R2C also provides a
//! state-checked single-allocation real/complex buffer. `R2rPlan` adds per-axis
//! `Option<R2rKind>` identity selection for all eight DCT/DST kinds, real and
//! complex `f32`/`f64`, with the same route, checked transports, raw paired
//! backward, normalized inverse, and poisoned in-place state contract. All
//! distributed plans support `N >= 2`, `1 <= M < N`, and checked Alltoallv or
//! receive-before-send point-to-point transitions; legacy constructors default
//! to Alltoallv. Plan construction and transform calls are collective on the
//! topology's Cartesian communicator; workspace allocation and views are
//! noncollective. Allocation failures must be coordinated by callers before
//! the next collective call. The default feature set remains MPI-free.
//!
//! # Example
//!
//! ```
//! use pencil_fft::{Complex, LocalC2cError, LocalC2cPlan};
//!
//! fn main() -> Result<(), LocalC2cError> {
//!     let plan = LocalC2cPlan::<f64>::new(4)?;
//!     let source = vec![
//!         Complex::new(1.0, 0.0),
//!         Complex::new(2.0, -1.0),
//!         Complex::new(-1.0, 0.5),
//!         Complex::new(0.25, 2.0),
//!     ];
//!     let mut transformed = vec![Complex::new(0.0, 0.0); source.len()];
//!     let mut scratch = vec![Complex::new(0.0, 0.0); plan.scratch_len()];
//!
//!     plan.forward(&source, &mut transformed, &mut scratch)?;
//!     let mut raw_backward = vec![Complex::new(0.0, 0.0); source.len()];
//!     plan.backward(&transformed, &mut raw_backward, &mut scratch)?;
//!     assert!(raw_backward.iter().zip(&source).all(|(actual, expected)| {
//!         (actual.re - 4.0 * expected.re).abs() < 1e-10
//!             && (actual.im - 4.0 * expected.im).abs() < 1e-10
//!     }));
//!
//!     let mut recovered = vec![Complex::new(0.0, 0.0); source.len()];
//!     plan.inverse(&transformed, &mut recovered, &mut scratch)?;
//!     assert!(recovered.iter().zip(&source).all(|(actual, expected)| {
//!         (actual.re - expected.re).abs() < 1e-10
//!             && (actual.im - expected.im).abs() < 1e-10
//!     }));
//!     Ok(())
//! }
//! ```
//!
//! `FftReal` is sealed so that backend-specific numeric requirements remain
//! private:
//!
//! ```compile_fail
//! use pencil_fft::FftReal;
//!
//! #[derive(Clone, Copy, Debug)]
//! struct ExternalReal(f64);
//!
//! impl FftReal for ExternalReal {}
//! ```
//!
//! An out-of-place real transform uses caller-owned buffers for the native
//! line operation. The packed in-place API is available through
//! [`LocalR2cPlan::allocate_in_place`]:
//!
//! ```
//! use pencil_fft::{Complex, LocalR2cError, LocalR2cPlan};
//!
//! fn main() -> Result<(), LocalR2cError> {
//!     let plan = LocalR2cPlan::<f64>::new(4)?;
//!     let source = [1.0, 2.0, -1.0, 0.25];
//!     let mut spectrum = vec![Complex::new(0.0, 0.0); plan.complex_len()];
//!     let mut real_line = vec![0.0; plan.real_len()];
//!     let mut scratch = vec![Complex::new(0.0, 0.0); plan.scratch_len()];
//!
//!     plan.forward(&source, &mut spectrum, &mut real_line, &mut scratch)?;
//!     let mut recovered = vec![0.0; plan.real_len()];
//!     let mut complex_line = vec![Complex::new(0.0, 0.0); plan.complex_len()];
//!     plan.inverse(&spectrum, &mut recovered, &mut complex_line, &mut scratch)?;
//!     assert!(recovered
//!         .iter()
//!         .zip(source)
//!         .all(|(actual, expected)| (actual - expected).abs() < 1e-10));
//!     Ok(())
//! }
//! ```

use std::fmt;
use std::mem::size_of;
use std::sync::Arc;

use rustfft::{Fft, FftPlanner};

/// The complex number type used by local FFT plans.
///
/// This is the `num_complex::Complex` type used by RustFFT. It is
/// backend-neutral at this crate's public API boundary.
pub use num_complex::Complex;

mod private {
    use super::Complex;

    #[cfg(feature = "distributed")]
    pub trait DistributedReal: mpi::datatype::Equivalence {}

    #[cfg(not(feature = "distributed"))]
    pub trait DistributedReal {}

    impl DistributedReal for f32 {}
    impl DistributedReal for f64 {}

    pub trait Sealed: rustfft::FftNum + bytemuck::Pod {
        fn normalize_inverse(values: &mut [Complex<Self>], line_len: usize);
        fn pencil_fft_as_f64(self) -> f64;
        fn pencil_fft_epsilon_f64() -> f64;
        fn pencil_fft_min_subnormal_f64() -> f64;
    }

    impl Sealed for f32 {
        fn normalize_inverse(values: &mut [Complex<Self>], line_len: usize) {
            let scale = 1.0 / line_len as f32;
            for value in values {
                value.re *= scale;
                value.im *= scale;
            }
        }

        fn pencil_fft_as_f64(self) -> f64 {
            self as f64
        }

        fn pencil_fft_epsilon_f64() -> f64 {
            f32::EPSILON as f64
        }

        fn pencil_fft_min_subnormal_f64() -> f64 {
            f32::from_bits(1) as f64
        }
    }

    impl Sealed for f64 {
        fn normalize_inverse(values: &mut [Complex<Self>], line_len: usize) {
            let scale = 1.0 / line_len as f64;
            for value in values {
                value.re *= scale;
                value.im *= scale;
            }
        }

        fn pencil_fft_as_f64(self) -> f64 {
            self
        }

        fn pencil_fft_epsilon_f64() -> f64 {
            f64::EPSILON
        }

        fn pencil_fft_min_subnormal_f64() -> f64 {
            f64::from_bits(1)
        }
    }
}

/// A real scalar supported by the local FFT plans.
///
/// This trait is sealed and is implemented only for `f32` and `f64`.
pub trait FftReal:
    private::Sealed + private::DistributedReal + Copy + Send + Sync + 'static
{
}

impl FftReal for f32 {}
impl FftReal for f64 {}

mod dht;
mod r2c;
mod r2r;

pub use dht::LocalDhtPlan;
pub use r2c::{
    LocalR2cError, LocalR2cInPlaceArray, LocalR2cInPlaceWorkspace, LocalR2cPlan, R2cState,
};
pub use r2r::{LocalR2rError, LocalR2rPlan, R2rKind, R2rScalar};

#[cfg(feature = "distributed")]
mod distributed;

#[cfg(feature = "distributed")]
pub use distributed::{
    AxisSelection, AxisSelectionError, C2cInPlaceArray, C2cInPlaceWorkspace,
    C2cOutOfPlaceWorkspace, C2cPlan, C2cState, FftError, R2cError, R2cInPlaceArray,
    R2cInPlaceWorkspace, R2cPlan, R2cWorkspace, R2rError, R2rInPlaceArray, R2rInPlaceWorkspace,
    R2rPlan, R2rState, R2rWorkspace, TransposeMethod,
};

/// Errors returned by local C2C plan construction and execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum LocalC2cError {
    /// The FFT line length was zero.
    #[error("FFT line length must be non-zero")]
    InvalidLength,
    /// The line length or a required derived value could not be represented.
    #[error("FFT length or derived length cannot be represented")]
    LengthOverflow,
    /// A data buffer did not contain an integral number of FFT lines.
    #[error("buffer length is not an integral number of FFT lines")]
    NonIntegralBatch,
    /// The source and destination buffers had different lengths.
    #[error("source and destination lengths differ")]
    BufferLengthMismatch,
    /// The caller-provided scratch slice was too short.
    #[error("FFT scratch is too small")]
    ScratchTooSmall {
        /// The minimum scratch length required by this plan.
        required: usize,
        /// The scratch length supplied by the caller.
        actual: usize,
    },
}

/// An immutable local batched complex-to-complex FFT plan.
///
/// `R` is restricted to `f32` and `f64` by the sealed [`FftReal`] trait. Data
/// buffers contain consecutive lines of length [`Self::line_len`]. The plan
/// does not own or retain any caller-provided data or scratch storage.
pub struct LocalC2cPlan<R: FftReal> {
    line_len: usize,
    scratch_len: usize,
    forward: Arc<dyn Fft<R>>,
    inverse: Arc<dyn Fft<R>>,
}

impl<R: FftReal> fmt::Debug for LocalC2cPlan<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LocalC2cPlan")
            .field("line_len", &self.line_len)
            .field("scratch_len", &self.scratch_len)
            .finish_non_exhaustive()
    }
}

impl<R: FftReal> LocalC2cPlan<R> {
    /// Builds a plan for lines of `line_len` complex values.
    ///
    /// A zero length is rejected. Before invoking the backend, this also
    /// checks that one line occupies an addressable number of bytes in a
    /// `Complex<R>` slice. Backend resource failures or backend panics are not
    /// converted into [`LocalC2cError`].
    pub fn new(line_len: usize) -> Result<Self, LocalC2cError> {
        validate_line_len::<R>(line_len)?;

        let mut planner = FftPlanner::<R>::new();
        let forward = planner.plan_fft_forward(line_len);
        let inverse = planner.plan_fft_inverse(line_len);
        let scratch_len = forward
            .get_immutable_scratch_len()
            .max(forward.get_inplace_scratch_len())
            .max(inverse.get_immutable_scratch_len())
            .max(inverse.get_inplace_scratch_len());

        Ok(Self {
            line_len,
            scratch_len,
            forward,
            inverse,
        })
    }

    /// Returns the number of complex values in each transformed line.
    pub fn line_len(&self) -> usize {
        self.line_len
    }

    /// Returns the minimum initialized scratch-slice length accepted by all
    /// forward/backward/inverse and out-of-place/in-place operations.
    pub fn scratch_len(&self) -> usize {
        self.scratch_len
    }

    /// Computes an unnormalized forward FFT for every line in `src`.
    ///
    /// `src` and `dst` must have equal lengths, each a multiple of
    /// [`Self::line_len`]. The source is preserved. `scratch` must be an
    /// initialized slice of at least [`Self::scratch_len`] complex values; its
    /// contents may be changed on success. An empty batch is a successful
    /// no-op after the same length and scratch checks.
    pub fn forward(
        &self,
        src: &[Complex<R>],
        dst: &mut [Complex<R>],
        scratch: &mut [Complex<R>],
    ) -> Result<(), LocalC2cError> {
        self.validate_out_of_place(src.len(), dst.len(), scratch.len())?;
        if src.is_empty() {
            return Ok(());
        }

        self.forward
            .process_immutable_with_scratch(src, dst, scratch);
        Ok(())
    }

    /// Computes an unnormalized positive-sign backward FFT for every line in
    /// `src`.
    ///
    /// This is the raw local C2C backward transform: it does not divide by
    /// [`Self::line_len`]. The source is preserved. Validation and scratch
    /// behavior are the same as for [`Self::forward`].
    pub fn backward(
        &self,
        src: &[Complex<R>],
        dst: &mut [Complex<R>],
        scratch: &mut [Complex<R>],
    ) -> Result<(), LocalC2cError> {
        self.validate_out_of_place(src.len(), dst.len(), scratch.len())?;
        if src.is_empty() {
            return Ok(());
        }

        self.inverse
            .process_immutable_with_scratch(src, dst, scratch);
        Ok(())
    }

    /// Computes a normalized positive-sign inverse FFT for every line in
    /// `src`, dividing each output line by [`Self::line_len`].
    ///
    /// This is the normalized local C2C inverse; [`Self::backward`] exposes the
    /// same positive-sign transform without this division. The source is
    /// preserved. Validation and scratch behavior are the same as for
    /// [`Self::forward`].
    pub fn inverse(
        &self,
        src: &[Complex<R>],
        dst: &mut [Complex<R>],
        scratch: &mut [Complex<R>],
    ) -> Result<(), LocalC2cError> {
        self.backward(src, dst, scratch)?;
        R::normalize_inverse(dst, self.line_len);
        Ok(())
    }

    /// Computes an unnormalized forward FFT in place for every line in `data`.
    ///
    /// `data.len()` must be a multiple of [`Self::line_len`]. `scratch` must
    /// satisfy [`Self::scratch_len`], and may be changed on success. An empty
    /// batch is a successful no-op after validation.
    pub fn forward_in_place(
        &self,
        data: &mut [Complex<R>],
        scratch: &mut [Complex<R>],
    ) -> Result<(), LocalC2cError> {
        self.validate_in_place(data.len(), scratch.len())?;
        if data.is_empty() {
            return Ok(());
        }

        self.forward.process_with_scratch(data, scratch);
        Ok(())
    }

    /// Computes an unnormalized positive-sign backward FFT in place for every
    /// line in `data`.
    ///
    /// The result is not divided by [`Self::line_len`]. Validation and scratch
    /// behavior are the same as for [`Self::forward_in_place`].
    pub fn backward_in_place(
        &self,
        data: &mut [Complex<R>],
        scratch: &mut [Complex<R>],
    ) -> Result<(), LocalC2cError> {
        self.validate_in_place(data.len(), scratch.len())?;
        if data.is_empty() {
            return Ok(());
        }

        self.inverse.process_with_scratch(data, scratch);
        Ok(())
    }

    /// Computes a normalized positive-sign inverse FFT in place for every line
    /// in `data`, dividing each output line by [`Self::line_len`].
    ///
    /// This is the normalized local C2C inverse; [`Self::backward_in_place`]
    /// exposes the same transform without this division. Validation and
    /// scratch behavior are the same as for [`Self::forward_in_place`].
    pub fn inverse_in_place(
        &self,
        data: &mut [Complex<R>],
        scratch: &mut [Complex<R>],
    ) -> Result<(), LocalC2cError> {
        self.backward_in_place(data, scratch)?;
        R::normalize_inverse(data, self.line_len);
        Ok(())
    }

    fn validate_out_of_place(
        &self,
        src_len: usize,
        dst_len: usize,
        scratch_len: usize,
    ) -> Result<(), LocalC2cError> {
        if src_len != dst_len {
            return Err(LocalC2cError::BufferLengthMismatch);
        }
        self.validate_batch(src_len)?;
        self.validate_scratch(scratch_len)
    }

    fn validate_in_place(&self, data_len: usize, scratch_len: usize) -> Result<(), LocalC2cError> {
        self.validate_batch(data_len)?;
        self.validate_scratch(scratch_len)
    }

    fn validate_batch(&self, buffer_len: usize) -> Result<(), LocalC2cError> {
        if buffer_len % self.line_len != 0 {
            return Err(LocalC2cError::NonIntegralBatch);
        }
        Ok(())
    }

    fn validate_scratch(&self, actual: usize) -> Result<(), LocalC2cError> {
        if actual < self.scratch_len {
            return Err(LocalC2cError::ScratchTooSmall {
                required: self.scratch_len,
                actual,
            });
        }
        Ok(())
    }
}

fn validate_line_len<R: FftReal>(line_len: usize) -> Result<(), LocalC2cError> {
    if line_len == 0 {
        return Err(LocalC2cError::InvalidLength);
    }

    let bytes = line_len
        .checked_mul(size_of::<Complex<R>>())
        .ok_or(LocalC2cError::LengthOverflow)?;
    if bytes > isize::MAX as usize {
        return Err(LocalC2cError::LengthOverflow);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
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

    type OutOfPlace = fn(
        &LocalC2cPlan<f64>,
        &[Complex<f64>],
        &mut [Complex<f64>],
        &mut [Complex<f64>],
    ) -> Result<(), LocalC2cError>;
    type InPlace = fn(
        &LocalC2cPlan<f64>,
        &mut [Complex<f64>],
        &mut [Complex<f64>],
    ) -> Result<(), LocalC2cError>;
    const OPERATIONS: [(OutOfPlace, InPlace); 3] = [
        (LocalC2cPlan::forward, LocalC2cPlan::forward_in_place),
        (LocalC2cPlan::backward, LocalC2cPlan::backward_in_place),
        (LocalC2cPlan::inverse, LocalC2cPlan::inverse_in_place),
    ];

    fn native_scratch_len<R: FftReal>(line_len: usize) -> usize {
        let mut planner = FftPlanner::<R>::new();
        let forward = planner.plan_fft_forward(line_len);
        let inverse = planner.plan_fft_inverse(line_len);
        forward
            .get_immutable_scratch_len()
            .max(forward.get_inplace_scratch_len())
            .max(inverse.get_immutable_scratch_len())
            .max(inverse.get_inplace_scratch_len())
    }

    fn initialized_scratch<R: TestReal>(len: usize) -> Vec<Complex<R>> {
        (0..len)
            .map(|index| {
                Complex::new(
                    R::convert(1.25 + index as f64),
                    R::convert(-2.5 - index as f64),
                )
            })
            .collect()
    }

    fn dirty_scratch<R: TestReal>(scratch: &mut [Complex<R>]) {
        for (index, value) in scratch.iter_mut().enumerate() {
            *value = Complex::new(
                R::convert(91.0 + index as f64),
                R::convert(-37.0 - index as f64),
            );
        }
    }

    fn sample_input<R: TestReal>(line_len: usize, batch_count: usize) -> Vec<Complex<R>> {
        (0..line_len * batch_count)
            .map(|index| {
                let x = index as f64 + 1.0;
                Complex::new(
                    R::convert((0.37 * x).sin() + 0.07 * x),
                    R::convert((0.19 * x).cos() - 0.11 * x),
                )
            })
            .collect()
    }

    fn dft_oracle<R: TestReal>(
        input: &[Complex<R>],
        line_len: usize,
        sign: f64,
        scale: f64,
    ) -> Vec<Complex<R>> {
        input
            .chunks_exact(line_len)
            .flat_map(|line| {
                (0..line_len).map(move |k| {
                    let (real, imaginary) = line.iter().enumerate().fold(
                        (0.0, 0.0),
                        |(real, imaginary), (j, value)| {
                            let angle = sign * TAU * j as f64 * k as f64 / line_len as f64;
                            let (sine, cosine) = angle.sin_cos();
                            (
                                real + value.re.as_f64() * cosine - value.im.as_f64() * sine,
                                imaginary + value.re.as_f64() * sine + value.im.as_f64() * cosine,
                            )
                        },
                    );
                    Complex::new(R::convert(real * scale), R::convert(imaginary * scale))
                })
            })
            .collect()
    }

    fn assert_close<R: TestReal>(actual: &[Complex<R>], expected: &[Complex<R>]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let bound =
                R::tolerance() * (1.0 + expected.re.as_f64().abs().max(expected.im.as_f64().abs()));
            let real_error = (actual.re.as_f64() - expected.re.as_f64()).abs();
            let imaginary_error = (actual.im.as_f64() - expected.im.as_f64()).abs();
            assert!(
                real_error <= bound && imaginary_error <= bound,
                "index {index}: actual={actual:?}, expected={expected:?}, bound={bound}"
            );
        }
    }

    fn exercise_against_oracle<R: TestReal>(line_len: usize, batch_count: usize) {
        let plan = LocalC2cPlan::<R>::new(line_len).unwrap();
        assert_eq!(plan.line_len(), line_len);
        assert_eq!(plan.scratch_len(), native_scratch_len::<R>(line_len));

        let source = sample_input::<R>(line_len, batch_count);
        let source_before_transforms = source.clone();
        let scratch_len = plan.scratch_len();
        let mut forward = vec![Complex::new(R::convert(13.0), R::convert(-9.0)); source.len()];
        let mut scratch = initialized_scratch::<R>(scratch_len);
        plan.forward(&source, &mut forward, &mut scratch).unwrap();
        assert_eq!(source, source_before_transforms);
        assert_close(&forward, &dft_oracle(&source, line_len, -1.0, 1.0));

        let mut backward = vec![Complex::new(R::convert(-4.0), R::convert(8.0)); source.len()];
        dirty_scratch(&mut scratch);
        plan.backward(&source, &mut backward, &mut scratch).unwrap();
        assert_eq!(source, source_before_transforms);
        assert_close(&backward, &dft_oracle(&source, line_len, 1.0, 1.0));

        let mut inverse = vec![Complex::new(R::convert(-4.0), R::convert(8.0)); source.len()];
        dirty_scratch(&mut scratch);
        plan.inverse(&source, &mut inverse, &mut scratch).unwrap();
        assert_eq!(source, source_before_transforms);
        assert_close(
            &inverse,
            &dft_oracle(&source, line_len, 1.0, 1.0 / line_len as f64),
        );

        let forward_before_roundtrip = forward.clone();
        let mut backward_roundtrip =
            vec![Complex::new(R::convert(0.0), R::convert(0.0)); source.len()];
        dirty_scratch(&mut scratch);
        plan.backward(&forward, &mut backward_roundtrip, &mut scratch)
            .unwrap();
        assert_eq!(forward, forward_before_roundtrip);
        let expected_backward_roundtrip: Vec<_> = source_before_transforms
            .iter()
            .map(|value| {
                Complex::new(
                    R::convert(value.re.as_f64() * line_len as f64),
                    R::convert(value.im.as_f64() * line_len as f64),
                )
            })
            .collect();
        assert_close(&backward_roundtrip, &expected_backward_roundtrip);

        let mut roundtrip = vec![Complex::new(R::convert(0.0), R::convert(0.0)); source.len()];
        dirty_scratch(&mut scratch);
        plan.inverse(&forward, &mut roundtrip, &mut scratch)
            .unwrap();
        assert_eq!(forward, forward_before_roundtrip);
        assert_close(&roundtrip, &source_before_transforms);

        let mut independent_lines = Vec::with_capacity(source.len());
        for line in source.chunks_exact(line_len) {
            let line_before = line.to_vec();
            let mut transformed = vec![Complex::new(R::convert(0.0), R::convert(0.0)); line_len];
            dirty_scratch(&mut scratch);
            plan.forward(line, &mut transformed, &mut scratch).unwrap();
            assert_eq!(line, line_before.as_slice());
            independent_lines.extend(transformed);
        }
        assert_close(&independent_lines, &forward);

        let mut forward_in_place = source.clone();
        dirty_scratch(&mut scratch);
        plan.forward_in_place(&mut forward_in_place, &mut scratch)
            .unwrap();
        assert_close(&forward_in_place, &forward);

        let mut backward_in_place = source.clone();
        dirty_scratch(&mut scratch);
        plan.backward_in_place(&mut backward_in_place, &mut scratch)
            .unwrap();
        assert_close(&backward_in_place, &dft_oracle(&source, line_len, 1.0, 1.0));

        let mut inverse_in_place = forward.clone();
        dirty_scratch(&mut scratch);
        plan.inverse_in_place(&mut inverse_in_place, &mut scratch)
            .unwrap();
        assert_close(&inverse_in_place, &source);
        assert_eq!(plan.scratch_len(), scratch_len);
    }

    #[test]
    fn f32_prime_composite_single_and_multiple_batches_match_dft() {
        for line_len in [1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13] {
            exercise_against_oracle::<f32>(line_len, 1);
            exercise_against_oracle::<f32>(line_len, 3);
        }
    }

    #[test]
    fn f64_prime_composite_single_and_multiple_batches_match_dft() {
        for line_len in [1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13] {
            exercise_against_oracle::<f64>(line_len, 1);
            exercise_against_oracle::<f64>(line_len, 3);
        }
    }

    fn check_frequency_peak_and_dc<R: TestReal>() {
        let line_len = 5;
        let peak = 2;
        let single_frequency: Vec<_> = (0..line_len)
            .map(|j| {
                let angle = TAU * peak as f64 * j as f64 / line_len as f64;
                Complex::new(R::convert(angle.cos()), R::convert(angle.sin()))
            })
            .collect();
        let mut transformed = vec![Complex::new(R::convert(0.0), R::convert(0.0)); line_len];
        let plan = LocalC2cPlan::<R>::new(line_len).unwrap();
        let mut scratch = initialized_scratch::<R>(plan.scratch_len());
        let single_frequency_before = single_frequency.clone();
        plan.forward(&single_frequency, &mut transformed, &mut scratch)
            .unwrap();
        assert_eq!(single_frequency, single_frequency_before);
        let expected_peak: Vec<_> = (0..line_len)
            .map(|index| {
                if index == peak {
                    Complex::new(R::convert(line_len as f64), R::convert(0.0))
                } else {
                    Complex::new(R::convert(0.0), R::convert(0.0))
                }
            })
            .collect();
        assert_close(&transformed, &expected_peak);

        let dc_value = Complex::new(R::convert(2.0), R::convert(-3.0));
        let dc_input = vec![dc_value; line_len];
        let dc_input_before = dc_input.clone();
        dirty_scratch(&mut scratch);
        plan.forward(&dc_input, &mut transformed, &mut scratch)
            .unwrap();
        assert_eq!(dc_input, dc_input_before);
        let mut expected_dc = vec![Complex::new(R::convert(0.0), R::convert(0.0)); line_len];
        expected_dc[0] = Complex::new(
            R::convert(2.0 * line_len as f64),
            R::convert(-3.0 * line_len as f64),
        );
        assert_close(&transformed, &expected_dc);
    }

    #[test]
    fn f32_peak_sign_and_forward_dc_scale() {
        check_frequency_peak_and_dc::<f32>();
    }

    #[test]
    fn f64_peak_sign_and_forward_dc_scale() {
        check_frequency_peak_and_dc::<f64>();
    }

    #[test]
    fn zero_batch_is_checked_for_all_directions_and_modes() {
        let plan = LocalC2cPlan::<f64>::new(1).unwrap();
        let scratch_len = plan.scratch_len();

        for &(out_of_place, in_place) in &OPERATIONS {
            let source: Vec<Complex<f64>> = Vec::new();
            let source_before = source.clone();
            let mut destination = Vec::new();
            let mut scratch = initialized_scratch::<f64>(scratch_len.max(1));
            dirty_scratch(&mut scratch);
            let scratch_before = scratch.clone();
            let result = out_of_place(&plan, &source, &mut destination, &mut scratch);
            assert_eq!(result, Ok(()));
            assert_eq!(source, source_before);
            assert!(destination.is_empty());
            assert_eq!(scratch, scratch_before);

            let mut destination = vec![Complex::new(31.0, -17.0)];
            let destination_before = destination.clone();
            let mut scratch = initialized_scratch::<f64>(scratch_len.max(1));
            dirty_scratch(&mut scratch);
            let scratch_before = scratch.clone();
            let result = out_of_place(&plan, &source, &mut destination, &mut scratch);
            assert_eq!(result, Err(LocalC2cError::BufferLengthMismatch));
            assert_eq!(source, source_before);
            assert_eq!(destination, destination_before);
            assert_eq!(scratch, scratch_before);

            let mut data = Vec::new();
            let mut scratch = initialized_scratch::<f64>(scratch_len.max(1));
            dirty_scratch(&mut scratch);
            let scratch_before = scratch.clone();
            let result = in_place(&plan, &mut data, &mut scratch);
            assert_eq!(result, Ok(()));
            assert!(data.is_empty());
            assert_eq!(scratch, scratch_before);
        }

        assert_eq!(plan.scratch_len(), scratch_len);
        assert!(matches!(
            LocalC2cPlan::<f64>::new(0),
            Err(LocalC2cError::InvalidLength)
        ));
    }

    #[test]
    fn invalid_buffers_are_rejected_before_any_buffer_changes() {
        let plan = LocalC2cPlan::<f64>::new(4).unwrap();

        for (operation, destination_value) in [
            (
                LocalC2cPlan::<f64>::forward as OutOfPlace,
                Complex::new(31.0, -17.0),
            ),
            (LocalC2cPlan::<f64>::backward, Complex::new(23.0, -19.0)),
        ] {
            let source = sample_input::<f64>(4, 1);
            let source_before = source.clone();
            let mut destination = vec![destination_value; 3];
            let destination_before = destination.clone();
            let mut scratch = initialized_scratch::<f64>(plan.scratch_len());
            let scratch_before = scratch.clone();
            assert_eq!(
                operation(&plan, &source, &mut destination, &mut scratch),
                Err(LocalC2cError::BufferLengthMismatch)
            );
            assert_eq!(source, source_before);
            assert_eq!(destination, destination_before);
            assert_eq!(scratch, scratch_before);
        }

        for (operation, destination_value) in [
            (
                LocalC2cPlan::<f64>::forward as OutOfPlace,
                Complex::new(29.0, 13.0),
            ),
            (LocalC2cPlan::<f64>::backward, Complex::new(17.0, -31.0)),
        ] {
            let source = sample_input::<f64>(5, 1);
            let source_before = source.clone();
            let mut destination = vec![destination_value; 5];
            let destination_before = destination.clone();
            let mut scratch = initialized_scratch::<f64>(plan.scratch_len());
            let scratch_before = scratch.clone();
            assert_eq!(
                operation(&plan, &source, &mut destination, &mut scratch),
                Err(LocalC2cError::NonIntegralBatch)
            );
            assert_eq!(source, source_before);
            assert_eq!(destination, destination_before);
            assert_eq!(scratch, scratch_before);
        }

        for operation in [
            LocalC2cPlan::<f64>::inverse_in_place as InPlace,
            LocalC2cPlan::<f64>::backward_in_place,
        ] {
            let mut data = sample_input::<f64>(5, 1);
            let data_before = data.clone();
            let mut scratch = initialized_scratch::<f64>(plan.scratch_len());
            let scratch_before = scratch.clone();
            assert_eq!(
                operation(&plan, &mut data, &mut scratch),
                Err(LocalC2cError::NonIntegralBatch)
            );
            assert_eq!(data, data_before);
            assert_eq!(scratch, scratch_before);
        }
    }

    #[test]
    fn scratch_shortage_uses_native_requirement_and_preserves_buffers() {
        let candidate = (1usize..=32)
            .chain([37, 97, 127, 1024])
            .find_map(|line_len| {
                let required = native_scratch_len::<f64>(line_len);
                (required > 0).then_some((line_len, required))
            });
        assert!(
            candidate.is_some(),
            "RustFFT 6.4.1 must expose a positive scratch requirement for a fixture length"
        );
        let (line_len, native_required) = candidate.expect("positive-scratch fixture was checked");
        assert!(native_required > 0);
        let plan = LocalC2cPlan::<f64>::new(line_len).unwrap();
        assert_eq!(plan.scratch_len(), native_required);

        for &(out_of_place, in_place) in &OPERATIONS {
            let source: Vec<Complex<f64>> = Vec::new();
            let source_before = source.clone();
            let mut destination = Vec::new();
            let destination_before = destination.clone();
            let mut scratch = initialized_scratch::<f64>(native_required - 1);
            dirty_scratch(&mut scratch);
            let scratch_before = scratch.clone();
            let result = out_of_place(&plan, &source, &mut destination, &mut scratch);
            assert_eq!(
                result,
                Err(LocalC2cError::ScratchTooSmall {
                    required: native_required,
                    actual: native_required - 1,
                })
            );
            assert_eq!(source, source_before);
            assert_eq!(destination, destination_before);
            assert_eq!(scratch, scratch_before);

            let mut data = Vec::new();
            let data_before = data.clone();
            let mut scratch = initialized_scratch::<f64>(native_required - 1);
            dirty_scratch(&mut scratch);
            let scratch_before = scratch.clone();
            let result = in_place(&plan, &mut data, &mut scratch);
            assert_eq!(
                result,
                Err(LocalC2cError::ScratchTooSmall {
                    required: native_required,
                    actual: native_required - 1,
                })
            );
            assert_eq!(data, data_before);
            assert_eq!(scratch, scratch_before);
        }

        for (operation, destination_value) in [
            (
                LocalC2cPlan::<f64>::forward as OutOfPlace,
                Complex::new(7.0, -11.0),
            ),
            (LocalC2cPlan::<f64>::backward, Complex::new(19.0, -23.0)),
        ] {
            let source = sample_input::<f64>(line_len, 1);
            let source_before = source.clone();
            let mut destination = vec![destination_value; line_len];
            let destination_before = destination.clone();
            let mut scratch = initialized_scratch::<f64>(native_required - 1);
            let scratch_before = scratch.clone();
            assert_eq!(
                operation(&plan, &source, &mut destination, &mut scratch),
                Err(LocalC2cError::ScratchTooSmall {
                    required: native_required,
                    actual: native_required - 1,
                })
            );
            assert_eq!(source, source_before);
            assert_eq!(destination, destination_before);
            assert_eq!(scratch, scratch_before);
        }

        for operation in [
            LocalC2cPlan::<f64>::forward_in_place as InPlace,
            LocalC2cPlan::<f64>::backward_in_place,
        ] {
            let mut data = sample_input::<f64>(line_len, 1);
            let data_before = data.clone();
            let mut scratch = initialized_scratch::<f64>(native_required - 1);
            let scratch_before = scratch.clone();
            assert_eq!(
                operation(&plan, &mut data, &mut scratch),
                Err(LocalC2cError::ScratchTooSmall {
                    required: native_required,
                    actual: native_required - 1,
                })
            );
            assert_eq!(data, data_before);
            assert_eq!(scratch, scratch_before);
        }
    }

    fn assert_line_len_boundary<R: FftReal>() {
        let max_line_len = isize::MAX as usize / size_of::<Complex<R>>();
        assert!(validate_line_len::<R>(max_line_len).is_ok());
        assert_eq!(
            validate_line_len::<R>(max_line_len + 1),
            Err(LocalC2cError::LengthOverflow)
        );
    }

    #[test]
    fn impossible_lengths_are_rejected_before_planning() {
        assert_ne!(
            size_of::<Complex<f32>>(),
            size_of::<Complex<f64>>(),
            "the boundary test must cover the distinct scalar sizes"
        );
        assert_line_len_boundary::<f32>();
        assert_line_len_boundary::<f64>();
        assert!(matches!(
            LocalC2cPlan::<f64>::new(usize::MAX),
            Err(LocalC2cError::LengthOverflow)
        ));
    }
}
