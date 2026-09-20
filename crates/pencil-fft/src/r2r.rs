use std::fmt;
use std::marker::PhantomData;
use std::mem::size_of;
use std::sync::Arc;

use rustfft::num_traits::{FromPrimitive, One, Zero};
use rustfft::{Fft, FftPlanner};

use super::{Complex, FftReal};

/// The eight FFTW-compatible one-dimensional real-to-real transform kinds.
///
/// The transforms use the unnormalized FFTW definitions. [`DctII`](Self::DctII)
/// and [`DctIII`](Self::DctIII), and likewise [`DstII`](Self::DstII) and
/// [`DstIII`](Self::DstIII), are paired inverse kinds. Type I and type IV are
/// self-paired.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum R2rKind {
    /// DCT-I, valid for lengths at least two.
    DctI,
    /// DCT-II.
    DctII,
    /// DCT-III.
    DctIII,
    /// DCT-IV.
    DctIV,
    /// DST-I.
    DstI,
    /// DST-II.
    DstII,
    /// DST-III.
    DstIII,
    /// DST-IV.
    DstIV,
}

impl R2rKind {
    #[cfg(feature = "distributed")]
    pub(crate) const fn descriptor_code(self) -> u64 {
        match self {
            Self::DctI => 1,
            Self::DctII => 2,
            Self::DctIII => 3,
            Self::DctIV => 4,
            Self::DstI => 5,
            Self::DstII => 6,
            Self::DstIII => 7,
            Self::DstIV => 8,
        }
    }

    /// Returns the kind used by [`LocalR2rPlan::backward`].
    pub const fn backward_kind(self) -> Self {
        match self {
            Self::DctI => Self::DctI,
            Self::DctII => Self::DctIII,
            Self::DctIII => Self::DctII,
            Self::DctIV => Self::DctIV,
            Self::DstI => Self::DstI,
            Self::DstII => Self::DstIII,
            Self::DstIII => Self::DstII,
            Self::DstIV => Self::DstIV,
        }
    }

    /// Returns whether this kind is a sine transform.
    pub const fn is_dst(self) -> bool {
        matches!(self, Self::DstI | Self::DstII | Self::DstIII | Self::DstIV)
    }
}

/// The transform family assigned to one distributed real-to-real axis.
///
/// `Fftw` preserves the eight legacy DCT/DST kinds. `Dht` selects the
/// self-paired discrete Hartley transform without changing [`R2rKind`].
#[cfg(feature = "distributed")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AxisR2rKind {
    /// One of the legacy FFTW-compatible DCT/DST kinds.
    Fftw(R2rKind),
    /// The discrete Hartley transform.
    Dht,
}

#[cfg(feature = "distributed")]
impl AxisR2rKind {
    pub(crate) const fn descriptor_code(self) -> u64 {
        match self {
            Self::Fftw(kind) => kind.descriptor_code(),
            Self::Dht => 9,
        }
    }
}

/// A scalar accepted by [`LocalR2rPlan`].
///
/// This trait is sealed and is implemented for exactly `f32`, `f64`,
/// `Complex<f32>`, and `Complex<f64>`. For complex element types the real and
/// imaginary components are transformed independently. For real element types
/// the imaginary component of the FFT embedding is discarded on output.
///
/// ```compile_fail
/// use pencil_fft::R2rScalar;
///
/// #[derive(Clone, Copy)]
/// struct External;
///
/// impl R2rScalar for External {
///     type Real = f64;
/// }
/// ```
pub trait R2rScalar: private::SealedR2rScalar + Copy + Send + Sync + 'static {
    /// The real type used by the native RustFFT embedding.
    type Real: FftReal;
}

mod private {
    use super::{Complex, FftReal, R2rScalar};

    pub trait SealedR2rScalar {
        const VALUE_KIND: u64;

        fn to_complex(self) -> Complex<<Self as R2rScalar>::Real>
        where
            Self: R2rScalar;

        fn from_complex(value: Complex<<Self as R2rScalar>::Real>) -> Self
        where
            Self: R2rScalar;
    }

    impl<R: FftReal> SealedR2rScalar for R {
        const VALUE_KIND: u64 = 1;

        fn to_complex(self) -> Complex<<Self as R2rScalar>::Real> {
            Complex::new(self, rustfft::num_traits::Zero::zero())
        }

        fn from_complex(value: Complex<<Self as R2rScalar>::Real>) -> Self {
            value.re
        }
    }

    impl<R: FftReal> SealedR2rScalar for Complex<R> {
        const VALUE_KIND: u64 = 2;

        fn to_complex(self) -> Complex<<Self as R2rScalar>::Real> {
            self
        }

        fn from_complex(value: Complex<<Self as R2rScalar>::Real>) -> Self {
            value
        }
    }
}

// `FftReal` is sealed to `f32` and `f64`, so these blanket impls expose
// exactly the legacy four scalar representations while letting the mixed
// generic kernels name their real and complex endpoint types.
impl<R: FftReal> R2rScalar for R {
    type Real = R;
}

impl<R: FftReal> R2rScalar for Complex<R> {
    type Real = R;
}

pub(crate) fn r2r_to_complex<T: R2rScalar>(value: T) -> Complex<T::Real> {
    <T as private::SealedR2rScalar>::to_complex(value)
}

pub(crate) fn r2r_from_complex<T: R2rScalar>(value: Complex<T::Real>) -> T {
    <T as private::SealedR2rScalar>::from_complex(value)
}

#[cfg(feature = "distributed")]
pub(crate) fn r2r_value_kind<T: R2rScalar>() -> u64 {
    <T as private::SealedR2rScalar>::VALUE_KIND
}

#[cfg(feature = "distributed")]
pub(crate) fn r2r_zero<T: R2rScalar>() -> T {
    T::from_complex(Complex::new(
        rustfft::num_traits::Zero::zero(),
        rustfft::num_traits::Zero::zero(),
    ))
}

/// Errors returned by local real-to-real plan construction and execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum LocalR2rError {
    /// The line length is zero or is one for DCT-I.
    #[error("invalid real-to-real transform line length")]
    InvalidLength,
    /// The line length or a required derived value cannot be represented.
    #[error("real-to-real FFT length or derived length cannot be represented")]
    LengthOverflow,
    /// A data buffer did not contain an integral number of transform lines.
    #[error("buffer length is not an integral number of transform lines")]
    NonIntegralBatch,
    /// The source and destination buffers have different lengths.
    #[error("source and destination lengths differ")]
    BufferLengthMismatch,
    /// The caller-owned complex embedding line is too short.
    #[error("complex embedding line is too short: required {required}, actual {actual}")]
    ComplexLineTooSmall {
        /// The minimum embedding-line length required by this plan.
        required: usize,
        /// The supplied embedding-line length.
        actual: usize,
    },
    /// The caller-owned native FFT scratch slice is too short.
    #[error("FFT scratch is too small: required {required}, actual {actual}")]
    ScratchTooSmall {
        /// The minimum scratch length required by this plan.
        required: usize,
        /// The supplied scratch length.
        actual: usize,
    },
}

/// An immutable local batched FFTW-style DCT/DST plan.
///
/// `T` may be `f32`, `f64`, `Complex<f32>`, or `Complex<f64>`. Every data
/// slice contains consecutive lines of [`Self::line_len`] values. Each call
/// uses a caller-owned initialized complex embedding line of at least
/// [`Self::embedding_len`] values and initialized native scratch of at least
/// [`Self::scratch_len`] values; neither workspace is retained by the plan.
/// The out-of-place methods preserve their source slices.
///
/// `forward` computes the selected unnormalized kind. `backward` computes its
/// paired raw inverse kind without dividing. `inverse` computes the same
/// paired transform and divides by [`Self::normalization_factor`], which is
/// `2 * (n - 1)` for DCT-I, `2 * (n + 1)` for DST-I, and `2 * n` for every
/// other kind. This divisor is the logical FFTW transform factor, never the
/// complex embedding length.
///
/// The deliberately simple embedding uses at most `8 * n` complex values per
/// line, plus the queried native scratch. `ponytail: replace the extension
/// line with a compact embedding only if measured memory use requires it.`
///
/// # Example
///
/// ```
/// use pencil_fft::{LocalR2rPlan, R2rKind};
///
/// # fn main() -> Result<(), pencil_fft::LocalR2rError> {
/// let plan = LocalR2rPlan::<f64>::new(4, R2rKind::DctII)?;
/// let source = [1.0, -2.0, 0.5, 3.0];
/// let mut transformed = vec![0.0; source.len()];
/// let mut embedding = vec![pencil_fft::Complex::new(0.0, 0.0); plan.embedding_len()];
/// let mut scratch = vec![pencil_fft::Complex::new(0.0, 0.0); plan.scratch_len()];
///
/// plan.forward(&source, &mut transformed, &mut embedding, &mut scratch)?;
/// let mut raw = vec![0.0; source.len()];
/// plan.backward(&transformed, &mut raw, &mut embedding, &mut scratch)?;
/// for (actual, expected) in raw.iter().zip(source) {
///     assert!((actual - plan.normalization_factor() as f64 * expected).abs() < 1e-10);
/// }
///
/// let mut recovered = vec![0.0; source.len()];
/// plan.inverse(&transformed, &mut recovered, &mut embedding, &mut scratch)?;
/// for (actual, expected) in recovered.iter().zip(source) {
///     assert!((actual - expected).abs() < 1e-10);
/// }
/// # Ok(())
/// # }
/// ```
pub struct LocalR2rPlan<T: R2rScalar> {
    line_len: usize,
    kind: R2rKind,
    embedding_len: usize,
    scratch_len: usize,
    normalization_factor: usize,
    fft: Arc<dyn Fft<T::Real>>,
    marker: PhantomData<T>,
}

impl<T: R2rScalar> fmt::Debug for LocalR2rPlan<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LocalR2rPlan")
            .field("line_len", &self.line_len)
            .field("kind", &self.kind)
            .field("embedding_len", &self.embedding_len)
            .field("scratch_len", &self.scratch_len)
            .field("normalization_factor", &self.normalization_factor)
            .finish_non_exhaustive()
    }
}

impl<T: R2rScalar> LocalR2rPlan<T> {
    /// Builds a plan for `line_len` values and the selected DCT/DST `kind`.
    ///
    /// Zero, DCT-I length-one, overflowing, and unaddressable logical or
    /// derived lengths are rejected before RustFFT planning. Backend resource
    /// failures and backend panics are not converted into
    /// [`LocalR2rError`].
    pub fn new(line_len: usize, kind: R2rKind) -> Result<Self, LocalR2rError> {
        let (embedding_len, normalization_factor) = validate_lengths::<T>(line_len, kind)?;
        let mut planner = FftPlanner::<T::Real>::new();
        let fft = planner.plan_fft_forward(embedding_len);
        let scratch_len = fft.get_inplace_scratch_len();
        validate_addressable(scratch_len, size_of::<Complex<T::Real>>())?;

        Ok(Self {
            line_len,
            kind,
            embedding_len,
            scratch_len,
            normalization_factor,
            fft,
            marker: PhantomData,
        })
    }

    /// Returns the number of values in each logical input or output line.
    pub fn line_len(&self) -> usize {
        self.line_len
    }

    /// Returns the selected forward transform kind.
    pub fn kind(&self) -> R2rKind {
        self.kind
    }

    /// Returns the kind used by [`Self::backward`] and [`Self::inverse`].
    pub fn backward_kind(&self) -> R2rKind {
        self.kind.backward_kind()
    }

    /// Returns the number of complex values required in the embedding line.
    pub fn embedding_len(&self) -> usize {
        self.embedding_len
    }

    /// Returns the native RustFFT in-place scratch length.
    pub fn scratch_len(&self) -> usize {
        self.scratch_len
    }

    /// Returns the logical FFTW raw forward/backward composition factor.
    pub fn normalization_factor(&self) -> usize {
        self.normalization_factor
    }

    /// Computes the selected unnormalized transform for every source line.
    ///
    /// `src` and `dst` must have equal lengths, each a multiple of
    /// [`Self::line_len`]. `embedding_line` and `scratch` must be initialized
    /// and satisfy [`Self::embedding_len`] and [`Self::scratch_len`]. The
    /// source and oversized workspace tails are preserved. A valid empty
    /// batch validates all lengths and then does nothing.
    pub fn forward(
        &self,
        src: &[T],
        dst: &mut [T],
        embedding_line: &mut [Complex<T::Real>],
        scratch: &mut [Complex<T::Real>],
    ) -> Result<(), LocalR2rError> {
        self.validate_out_of_place(src.len(), dst.len(), embedding_line.len(), scratch.len())?;
        if src.is_empty() {
            return Ok(());
        }

        for (source_line, destination_line) in src
            .chunks_exact(self.line_len)
            .zip(dst.chunks_exact_mut(self.line_len))
        {
            self.execute_out_of_place_line(
                self.kind,
                source_line,
                destination_line,
                embedding_line,
                scratch,
                None,
            );
        }
        Ok(())
    }

    /// Computes the raw paired inverse transform for every source line.
    ///
    /// This method does not divide by [`Self::normalization_factor`]. Use
    /// [`Self::inverse`] for the normalized result. Validation and workspace
    /// behavior match [`Self::forward`].
    pub fn backward(
        &self,
        src: &[T],
        dst: &mut [T],
        embedding_line: &mut [Complex<T::Real>],
        scratch: &mut [Complex<T::Real>],
    ) -> Result<(), LocalR2rError> {
        self.validate_out_of_place(src.len(), dst.len(), embedding_line.len(), scratch.len())?;
        if src.is_empty() {
            return Ok(());
        }

        for (source_line, destination_line) in src
            .chunks_exact(self.line_len)
            .zip(dst.chunks_exact_mut(self.line_len))
        {
            self.execute_out_of_place_line(
                self.backward_kind(),
                source_line,
                destination_line,
                embedding_line,
                scratch,
                None,
            );
        }
        Ok(())
    }

    /// Computes the normalized paired inverse transform for every source line.
    ///
    /// The paired raw transform is divided in the real type by exactly
    /// [`Self::normalization_factor`], not by the complex embedding length.
    /// The source and oversized workspace tails are preserved.
    pub fn inverse(
        &self,
        src: &[T],
        dst: &mut [T],
        embedding_line: &mut [Complex<T::Real>],
        scratch: &mut [Complex<T::Real>],
    ) -> Result<(), LocalR2rError> {
        self.validate_out_of_place(src.len(), dst.len(), embedding_line.len(), scratch.len())?;
        if src.is_empty() {
            return Ok(());
        }
        let scale = self.normalization_scale();

        for (source_line, destination_line) in src
            .chunks_exact(self.line_len)
            .zip(dst.chunks_exact_mut(self.line_len))
        {
            self.execute_out_of_place_line(
                self.backward_kind(),
                source_line,
                destination_line,
                embedding_line,
                scratch,
                Some(scale),
            );
        }
        Ok(())
    }

    /// Computes the selected unnormalized transform in place.
    ///
    /// `data` must contain an integral batch of logical lines. Validation is
    /// complete before `data`, `embedding_line`, or `scratch` is changed.
    /// Oversized workspace tails are preserved; a valid empty batch is a
    /// no-op.
    pub fn forward_in_place(
        &self,
        data: &mut [T],
        embedding_line: &mut [Complex<T::Real>],
        scratch: &mut [Complex<T::Real>],
    ) -> Result<(), LocalR2rError> {
        self.validate_in_place(data.len(), embedding_line.len(), scratch.len())?;
        if data.is_empty() {
            return Ok(());
        }

        for data_line in data.chunks_exact_mut(self.line_len) {
            self.execute_in_place_line(self.kind, data_line, embedding_line, scratch, None);
        }
        Ok(())
    }

    /// Computes the raw paired inverse transform in place.
    ///
    /// The result is not divided by [`Self::normalization_factor`].
    pub fn backward_in_place(
        &self,
        data: &mut [T],
        embedding_line: &mut [Complex<T::Real>],
        scratch: &mut [Complex<T::Real>],
    ) -> Result<(), LocalR2rError> {
        self.validate_in_place(data.len(), embedding_line.len(), scratch.len())?;
        if data.is_empty() {
            return Ok(());
        }

        for data_line in data.chunks_exact_mut(self.line_len) {
            self.execute_in_place_line(
                self.backward_kind(),
                data_line,
                embedding_line,
                scratch,
                None,
            );
        }
        Ok(())
    }

    /// Computes the normalized paired inverse transform in place.
    ///
    /// The result is divided by [`Self::normalization_factor`], using a scale
    /// represented in `T::Real`, never by the complex embedding length.
    pub fn inverse_in_place(
        &self,
        data: &mut [T],
        embedding_line: &mut [Complex<T::Real>],
        scratch: &mut [Complex<T::Real>],
    ) -> Result<(), LocalR2rError> {
        self.validate_in_place(data.len(), embedding_line.len(), scratch.len())?;
        if data.is_empty() {
            return Ok(());
        }
        let scale = self.normalization_scale();

        for data_line in data.chunks_exact_mut(self.line_len) {
            self.execute_in_place_line(
                self.backward_kind(),
                data_line,
                embedding_line,
                scratch,
                Some(scale),
            );
        }
        Ok(())
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

    fn normalization_scale(&self) -> T::Real {
        T::Real::one()
            / T::Real::from_usize(self.normalization_factor)
                .expect("f32 and f64 can represent every validated normalization factor")
    }

    fn execute_out_of_place_line(
        &self,
        kind: R2rKind,
        source: &[T],
        destination: &mut [T],
        embedding_line: &mut [Complex<T::Real>],
        scratch: &mut [Complex<T::Real>],
        scale: Option<T::Real>,
    ) {
        self.build_embedding(kind, source, embedding_line);
        self.fft.process_with_scratch(
            &mut embedding_line[..self.embedding_len],
            &mut scratch[..self.scratch_len],
        );
        self.write_output(kind, destination, embedding_line, scale);
    }

    fn execute_in_place_line(
        &self,
        kind: R2rKind,
        data: &mut [T],
        embedding_line: &mut [Complex<T::Real>],
        scratch: &mut [Complex<T::Real>],
        scale: Option<T::Real>,
    ) {
        self.build_embedding(kind, data, embedding_line);
        self.fft.process_with_scratch(
            &mut embedding_line[..self.embedding_len],
            &mut scratch[..self.scratch_len],
        );
        self.write_output(kind, data, embedding_line, scale);
    }

    fn build_embedding(
        &self,
        kind: R2rKind,
        source: &[T],
        embedding_line: &mut [Complex<T::Real>],
    ) {
        let zero = Complex::new(T::Real::zero(), T::Real::zero());
        let embedding = &mut embedding_line[..self.embedding_len];
        embedding.fill(zero);

        match kind {
            R2rKind::DctI => {
                embedding[0] = source[0].to_complex();
                embedding[self.line_len - 1] = source[self.line_len - 1].to_complex();
                for (j, value) in source.iter().enumerate().skip(1).take(self.line_len - 2) {
                    put_pair(embedding, j, value.to_complex(), false);
                }
            }
            R2rKind::DctII => {
                for (j, value) in source.iter().enumerate() {
                    put_pair(embedding, 2 * j + 1, value.to_complex(), false);
                }
            }
            R2rKind::DctIII => {
                embedding[0] = source[0].to_complex();
                for (j, value) in source.iter().enumerate().skip(1) {
                    put_pair(embedding, j, value.to_complex(), false);
                }
            }
            R2rKind::DctIV => {
                for (j, value) in source.iter().enumerate() {
                    put_pair(embedding, 2 * j + 1, value.to_complex(), false);
                }
            }
            R2rKind::DstI => {
                for (j, value) in source.iter().enumerate() {
                    put_pair(embedding, j + 1, value.to_complex(), true);
                }
            }
            R2rKind::DstII => {
                for (j, value) in source.iter().enumerate() {
                    put_pair(embedding, 2 * j + 1, value.to_complex(), true);
                }
            }
            R2rKind::DstIII => {
                for (j, value) in source.iter().enumerate().take(self.line_len - 1) {
                    put_pair(embedding, j + 1, value.to_complex(), true);
                }
                embedding[self.line_len] = source[self.line_len - 1].to_complex();
            }
            R2rKind::DstIV => {
                for (j, value) in source.iter().enumerate() {
                    put_pair(embedding, 2 * j + 1, value.to_complex(), true);
                }
            }
        }
    }

    fn write_output(
        &self,
        kind: R2rKind,
        destination: &mut [T],
        embedding_line: &[Complex<T::Real>],
        scale: Option<T::Real>,
    ) {
        for (k, destination_value) in destination.iter_mut().enumerate() {
            let mut value = embedding_line[output_bin(kind, k)];
            if kind.is_dst() {
                value = Complex::new(-value.im, value.re);
            }
            if let Some(scale) = scale {
                value.re = value.re * scale;
                value.im = value.im * scale;
            }
            *destination_value = T::from_complex(value);
        }
    }
}

fn put_pair<R: FftReal>(embedding: &mut [Complex<R>], p: usize, value: Complex<R>, dst: bool) {
    embedding[p] = value;
    let mirror = embedding.len() - p;
    embedding[mirror] = if dst {
        Complex::new(-value.re, -value.im)
    } else {
        value
    };
}

fn output_bin(kind: R2rKind, k: usize) -> usize {
    match kind {
        R2rKind::DctI | R2rKind::DctII => k,
        R2rKind::DctIII | R2rKind::DctIV | R2rKind::DstIII | R2rKind::DstIV => 2 * k + 1,
        R2rKind::DstI | R2rKind::DstII => k + 1,
    }
}

fn validate_lengths<T: R2rScalar>(
    line_len: usize,
    kind: R2rKind,
) -> Result<(usize, usize), LocalR2rError> {
    if line_len == 0 || (kind == R2rKind::DctI && line_len == 1) {
        return Err(LocalR2rError::InvalidLength);
    }

    validate_addressable(line_len, size_of::<T>())?;
    let (embedding_len, normalization_factor) = match kind {
        R2rKind::DctI => {
            let n_minus_one = line_len
                .checked_sub(1)
                .ok_or(LocalR2rError::LengthOverflow)?;
            let length = n_minus_one
                .checked_mul(2)
                .ok_or(LocalR2rError::LengthOverflow)?;
            (length, length)
        }
        R2rKind::DstI => {
            let n_plus_one = line_len
                .checked_add(1)
                .ok_or(LocalR2rError::LengthOverflow)?;
            let length = n_plus_one
                .checked_mul(2)
                .ok_or(LocalR2rError::LengthOverflow)?;
            (length, length)
        }
        R2rKind::DctII | R2rKind::DctIII | R2rKind::DstII | R2rKind::DstIII => {
            let embedding_len = line_len
                .checked_mul(4)
                .ok_or(LocalR2rError::LengthOverflow)?;
            let normalization_factor = line_len
                .checked_mul(2)
                .ok_or(LocalR2rError::LengthOverflow)?;
            (embedding_len, normalization_factor)
        }
        R2rKind::DctIV | R2rKind::DstIV => {
            let embedding_len = line_len
                .checked_mul(8)
                .ok_or(LocalR2rError::LengthOverflow)?;
            let normalization_factor = line_len
                .checked_mul(2)
                .ok_or(LocalR2rError::LengthOverflow)?;
            (embedding_len, normalization_factor)
        }
    };
    validate_addressable(embedding_len, size_of::<Complex<T::Real>>())?;
    validate_addressable(normalization_factor, size_of::<T::Real>())?;
    Ok((embedding_len, normalization_factor))
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

#[cfg(test)]
mod tests {
    use super::*;
    use rustfft::num_traits::FromPrimitive;
    use std::f64::consts::PI;

    trait TestScalar: R2rScalar {
        fn from_f64(re: f64, im: f64) -> Self;
        fn as_f64(self) -> (f64, f64);
        fn tolerance() -> f64;
    }

    impl TestScalar for f32 {
        fn from_f64(re: f64, _im: f64) -> Self {
            re as f32
        }

        fn as_f64(self) -> (f64, f64) {
            (self as f64, 0.0)
        }

        fn tolerance() -> f64 {
            4e-4
        }
    }

    impl TestScalar for f64 {
        fn from_f64(re: f64, _im: f64) -> Self {
            re
        }

        fn as_f64(self) -> (f64, f64) {
            (self, 0.0)
        }

        fn tolerance() -> f64 {
            2e-10
        }
    }

    impl TestScalar for Complex<f32> {
        fn from_f64(re: f64, im: f64) -> Self {
            Complex::new(re as f32, im as f32)
        }

        fn as_f64(self) -> (f64, f64) {
            (self.re as f64, self.im as f64)
        }

        fn tolerance() -> f64 {
            5e-4
        }
    }

    impl TestScalar for Complex<f64> {
        fn from_f64(re: f64, im: f64) -> Self {
            Complex::new(re, im)
        }

        fn as_f64(self) -> (f64, f64) {
            (self.re, self.im)
        }

        fn tolerance() -> f64 {
            3e-10
        }
    }

    fn kinds() -> [R2rKind; 8] {
        [
            R2rKind::DctI,
            R2rKind::DctII,
            R2rKind::DctIII,
            R2rKind::DctIV,
            R2rKind::DstI,
            R2rKind::DstII,
            R2rKind::DstIII,
            R2rKind::DstIV,
        ]
    }

    fn direct(kind: R2rKind, input: &[Complex<f64>]) -> Vec<Complex<f64>> {
        let n = input.len();
        (0..n)
            .map(|k| {
                let mut output = Complex::new(0.0, 0.0);
                for (j, value) in input.iter().enumerate() {
                    let coefficient = match kind {
                        R2rKind::DctI => {
                            if j == 0 || j == n - 1 {
                                if j == n - 1 && k % 2 == 1 { -1.0 } else { 1.0 }
                            } else {
                                2.0 * (PI * j as f64 * k as f64 / (n - 1) as f64).cos()
                            }
                        }
                        R2rKind::DctII => 2.0 * (PI * (j as f64 + 0.5) * k as f64 / n as f64).cos(),
                        R2rKind::DctIII => {
                            if j == 0 {
                                1.0
                            } else {
                                2.0 * (PI * j as f64 * (k as f64 + 0.5) / n as f64).cos()
                            }
                        }
                        R2rKind::DctIV => {
                            2.0 * (PI * (j as f64 + 0.5) * (k as f64 + 0.5) / n as f64).cos()
                        }
                        R2rKind::DstI => {
                            2.0 * (PI * (j as f64 + 1.0) * (k as f64 + 1.0) / (n + 1) as f64).sin()
                        }
                        R2rKind::DstII => {
                            2.0 * (PI * (j as f64 + 0.5) * (k as f64 + 1.0) / n as f64).sin()
                        }
                        R2rKind::DstIII => {
                            if j == n - 1 {
                                if k % 2 == 0 { 1.0 } else { -1.0 }
                            } else {
                                2.0 * (PI * (j as f64 + 1.0) * (k as f64 + 0.5) / n as f64).sin()
                            }
                        }
                        R2rKind::DstIV => {
                            2.0 * (PI * (j as f64 + 0.5) * (k as f64 + 0.5) / n as f64).sin()
                        }
                    };
                    output.re += value.re * coefficient;
                    output.im += value.im * coefficient;
                }
                output
            })
            .collect()
    }

    fn input<T: TestScalar>(n: usize, batches: usize) -> Vec<T> {
        (0..n * batches)
            .map(|index| {
                let x = index as f64 + 1.0;
                T::from_f64((0.19 * x).sin() + 0.07 * x, (0.23 * x).cos() - 0.11 * x)
            })
            .collect()
    }

    fn independent_input<T: TestScalar>(n: usize, batches: usize) -> Vec<T> {
        (0..n * batches)
            .map(|index| {
                let x = index as f64 + 0.37;
                T::from_f64((0.13 * x).cos() - 0.05 * x, (0.29 * x).sin() + 0.09 * x)
            })
            .collect()
    }

    fn assert_close<T: TestScalar>(actual: &[T], expected: &[Complex<f64>]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().copied().zip(expected).enumerate() {
            let (actual_re, actual_im) = actual.as_f64();
            let bound = T::tolerance() * (1.0 + expected.re.abs().max(expected.im.abs()));
            assert!(
                (actual_re - expected.re).abs() <= bound
                    && (actual_im - expected.im).abs() <= bound,
                "index {index}: actual=({actual_re}, {actual_im}), expected={expected:?}, bound={bound}"
            );
        }
    }

    fn as_f64_complex<T: TestScalar>(value: T) -> Complex<f64> {
        let (re, im) = value.as_f64();
        Complex::new(re, im)
    }

    fn assert_scalar_preserved<T: TestScalar>(actual: &[T], expected: &[T]) {
        for (actual, expected) in actual.iter().copied().zip(expected) {
            let (actual_re, actual_im) = actual.as_f64();
            let (expected_re, expected_im) = expected.as_f64();
            assert_eq!(actual_re.to_bits(), expected_re.to_bits());
            assert_eq!(actual_im.to_bits(), expected_im.to_bits());
        }
    }

    fn initialized_workspace<T: TestScalar>(len: usize, offset: usize) -> Vec<Complex<T::Real>> {
        (0..len)
            .map(|index| {
                Complex::new(
                    T::Real::from_usize(offset + index + 1).unwrap(),
                    T::Real::from_usize(offset + index + 2).unwrap(),
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
        fn out_of_place<T: R2rScalar>(
            self,
            plan: &LocalR2rPlan<T>,
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
            plan: &LocalR2rPlan<T>,
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

    fn oracle_pair_kind(kind: R2rKind) -> R2rKind {
        match kind {
            R2rKind::DctI => R2rKind::DctI,
            R2rKind::DctII => R2rKind::DctIII,
            R2rKind::DctIII => R2rKind::DctII,
            R2rKind::DctIV => R2rKind::DctIV,
            R2rKind::DstI => R2rKind::DstI,
            R2rKind::DstII => R2rKind::DstIII,
            R2rKind::DstIII => R2rKind::DstII,
            R2rKind::DstIV => R2rKind::DstIV,
        }
    }

    fn oracle_normalization_factor(kind: R2rKind, n: usize) -> f64 {
        match kind {
            R2rKind::DctI => (2 * (n - 1)) as f64,
            R2rKind::DstI => (2 * (n + 1)) as f64,
            _ => (2 * n) as f64,
        }
    }

    fn expected<T: TestScalar>(
        kind: R2rKind,
        direction: Direction,
        n: usize,
        input: &[T],
    ) -> Vec<Complex<f64>> {
        let oracle_kind = match direction {
            Direction::Forward => kind,
            Direction::Backward | Direction::Inverse => oracle_pair_kind(kind),
        };
        let mut expected = input
            .chunks_exact(n)
            .flat_map(|line| {
                direct(
                    oracle_kind,
                    &line.iter().copied().map(as_f64_complex).collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();
        if matches!(direction, Direction::Inverse) {
            let scale = 1.0 / oracle_normalization_factor(kind, n);
            for value in &mut expected {
                value.re *= scale;
                value.im *= scale;
            }
        }
        expected
    }

    fn exercise_kind<T: TestScalar>(kind: R2rKind, n: usize, batches: usize) {
        let plan = LocalR2rPlan::<T>::new(n, kind).unwrap();
        let arbitrary = independent_input::<T>(n, batches);
        let arbitrary_before = arbitrary.clone();

        for direction in DIRECTIONS {
            let mut destination = vec![T::from_f64(17.0, -19.0); arbitrary.len()];
            let mut embedding = initialized_workspace::<T>(plan.embedding_len() + 2, 0);
            let embedding_tail = embedding[plan.embedding_len()..].to_vec();
            let mut scratch = initialized_workspace::<T>(plan.scratch_len() + 3, 100);
            let scratch_tail = scratch[plan.scratch_len()..].to_vec();

            direction
                .out_of_place(
                    &plan,
                    &arbitrary,
                    &mut destination,
                    &mut embedding,
                    &mut scratch,
                )
                .unwrap();
            assert_scalar_preserved(&arbitrary, &arbitrary_before);
            assert_close(&destination, &expected(kind, direction, n, &arbitrary));
            assert_eq!(&embedding[plan.embedding_len()..], &embedding_tail);
            assert_eq!(&scratch[plan.scratch_len()..], &scratch_tail);

            let mut in_place = arbitrary.clone();
            let mut embedding = initialized_workspace::<T>(plan.embedding_len() + 2, 200);
            let embedding_tail = embedding[plan.embedding_len()..].to_vec();
            let mut scratch = initialized_workspace::<T>(plan.scratch_len() + 3, 300);
            let scratch_tail = scratch[plan.scratch_len()..].to_vec();
            direction
                .in_place(&plan, &mut in_place, &mut embedding, &mut scratch)
                .unwrap();
            assert_close(&in_place, &expected(kind, direction, n, &arbitrary));
            assert_eq!(&embedding[plan.embedding_len()..], &embedding_tail);
            assert_eq!(&scratch[plan.scratch_len()..], &scratch_tail);
        }

        let source = input::<T>(n, batches);
        let source_before = source.clone();
        let expected_source = source
            .iter()
            .copied()
            .map(as_f64_complex)
            .collect::<Vec<_>>();
        let mut transformed = vec![T::from_f64(17.0, -19.0); source.len()];
        let mut embedding = initialized_workspace::<T>(plan.embedding_len(), 400);
        let mut scratch = initialized_workspace::<T>(plan.scratch_len(), 500);
        plan.forward(&source, &mut transformed, &mut embedding, &mut scratch)
            .unwrap();
        assert_scalar_preserved(&source, &source_before);

        let mut recovered_raw = vec![T::from_f64(-31.0, 13.0); source.len()];
        plan.backward(
            &transformed,
            &mut recovered_raw,
            &mut embedding,
            &mut scratch,
        )
        .unwrap();
        let scale = oracle_normalization_factor(kind, n);
        let expected_raw_roundtrip = expected_source
            .iter()
            .map(|value| Complex::new(value.re * scale, value.im * scale))
            .collect::<Vec<_>>();
        assert_close(&recovered_raw, &expected_raw_roundtrip);

        let mut recovered = vec![T::from_f64(-37.0, 21.0); source.len()];
        plan.inverse(&transformed, &mut recovered, &mut embedding, &mut scratch)
            .unwrap();
        assert_close(&recovered, &expected_source);

        let mut in_place = source.clone();
        plan.forward_in_place(&mut in_place, &mut embedding, &mut scratch)
            .unwrap();
        plan.inverse_in_place(&mut in_place, &mut embedding, &mut scratch)
            .unwrap();
        assert_close(&in_place, &expected_source);
    }

    #[test]
    fn all_kinds_and_scalar_types_match_independent_definitions() {
        for kind in kinds() {
            for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11] {
                if kind == R2rKind::DctI && n == 1 {
                    continue;
                }
                exercise_kind::<f32>(kind, n, 2);
                exercise_kind::<f64>(kind, n, 2);
                exercise_kind::<Complex<f32>>(kind, n, 2);
                exercise_kind::<Complex<f64>>(kind, n, 2);
            }
        }
    }

    #[test]
    fn metadata_pairing_and_normalization_are_logical() {
        assert_eq!(
            LocalR2rPlan::<f64>::new(5, R2rKind::DctI)
                .unwrap()
                .embedding_len(),
            8
        );
        assert_eq!(
            LocalR2rPlan::<f64>::new(5, R2rKind::DstI)
                .unwrap()
                .embedding_len(),
            12
        );
        for kind in kinds() {
            let plan = LocalR2rPlan::<f64>::new(5, kind).unwrap();
            assert_eq!(
                plan.normalization_factor(),
                if kind == R2rKind::DctI {
                    8
                } else if kind == R2rKind::DstI {
                    12
                } else {
                    10
                }
            );
        }
        assert_eq!(R2rKind::DctII.backward_kind(), R2rKind::DctIII);
        assert_eq!(R2rKind::DctIII.backward_kind(), R2rKind::DctII);
        assert_eq!(R2rKind::DstII.backward_kind(), R2rKind::DstIII);
        assert_eq!(R2rKind::DstIII.backward_kind(), R2rKind::DstII);
    }

    #[test]
    fn n_one_rules_and_dct_i_rejection() {
        assert!(matches!(
            LocalR2rPlan::<f64>::new(1, R2rKind::DctI),
            Err(LocalR2rError::InvalidLength)
        ));
        for kind in kinds() {
            if kind == R2rKind::DctI {
                continue;
            }
            assert!(LocalR2rPlan::<Complex<f32>>::new(1, kind).is_ok());
        }
    }

    fn assert_out_of_place_error<T: TestScalar>(
        direction: Direction,
        plan: &LocalR2rPlan<T>,
        source: &[T],
        destination: &mut [T],
        embedding: &mut [Complex<T::Real>],
        scratch: &mut [Complex<T::Real>],
        expected: LocalR2rError,
    ) {
        let source_before = source.to_vec();
        let destination_before = destination.to_vec();
        let embedding_before = embedding.to_vec();
        let scratch_before = scratch.to_vec();
        assert_eq!(
            direction.out_of_place(plan, source, destination, embedding, scratch),
            Err(expected)
        );
        assert_scalar_preserved(source, &source_before);
        assert_scalar_preserved(destination, &destination_before);
        assert_eq!(embedding, embedding_before);
        assert_eq!(scratch, scratch_before);
    }

    fn assert_in_place_error<T: TestScalar>(
        direction: Direction,
        plan: &LocalR2rPlan<T>,
        data: &mut [T],
        embedding: &mut [Complex<T::Real>],
        scratch: &mut [Complex<T::Real>],
        expected: LocalR2rError,
    ) {
        let data_before = data.to_vec();
        let embedding_before = embedding.to_vec();
        let scratch_before = scratch.to_vec();
        assert_eq!(
            direction.in_place(plan, data, embedding, scratch),
            Err(expected)
        );
        assert_scalar_preserved(data, &data_before);
        assert_eq!(embedding, embedding_before);
        assert_eq!(scratch, scratch_before);
    }

    fn validation_cases<T: TestScalar>() {
        let plan = LocalR2rPlan::<T>::new(4, R2rKind::DctIV).unwrap();
        let empty = Vec::new();

        for direction in DIRECTIONS {
            let mut embedding = initialized_workspace::<T>(plan.embedding_len() + 2, 0);
            let embedding_before = embedding.clone();
            let mut scratch = initialized_workspace::<T>(plan.scratch_len() + 2, 100);
            let scratch_before = scratch.clone();
            let mut destination = Vec::new();
            assert_eq!(
                direction.out_of_place(
                    &plan,
                    &empty,
                    &mut destination,
                    &mut embedding,
                    &mut scratch,
                ),
                Ok(())
            );
            assert!(destination.is_empty());
            assert_eq!(embedding, embedding_before);
            assert_eq!(scratch, scratch_before);

            let mut data = Vec::new();
            let mut embedding = initialized_workspace::<T>(plan.embedding_len() + 2, 200);
            let embedding_before = embedding.clone();
            let mut scratch = initialized_workspace::<T>(plan.scratch_len() + 2, 300);
            let scratch_before = scratch.clone();
            assert_eq!(
                direction.in_place(&plan, &mut data, &mut embedding, &mut scratch),
                Ok(())
            );
            assert!(data.is_empty());
            assert_eq!(embedding, embedding_before);
            assert_eq!(scratch, scratch_before);
        }

        let source = input::<T>(plan.line_len(), 1);
        for direction in DIRECTIONS {
            let mut destination = vec![T::from_f64(2.0, -3.0); source.len()];
            let mut embedding = initialized_workspace::<T>(plan.embedding_len() + 2, 400);
            let embedding_tail = embedding[plan.embedding_len()..].to_vec();
            let mut scratch = initialized_workspace::<T>(plan.scratch_len() + 2, 500);
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
            assert_eq!(&embedding[plan.embedding_len()..], &embedding_tail);
            assert_eq!(&scratch[plan.scratch_len()..], &scratch_tail);

            let mut data = source.clone();
            let mut embedding = initialized_workspace::<T>(plan.embedding_len() + 2, 600);
            let embedding_tail = embedding[plan.embedding_len()..].to_vec();
            let mut scratch = initialized_workspace::<T>(plan.scratch_len() + 2, 700);
            let scratch_tail = scratch[plan.scratch_len()..].to_vec();
            direction
                .in_place(&plan, &mut data, &mut embedding, &mut scratch)
                .unwrap();
            assert_eq!(&embedding[plan.embedding_len()..], &embedding_tail);
            assert_eq!(&scratch[plan.scratch_len()..], &scratch_tail);
        }

        for direction in DIRECTIONS {
            let source = input::<T>(plan.line_len(), 1);
            let mut destination = vec![T::from_f64(2.0, -3.0); plan.line_len() - 1];
            let mut embedding = initialized_workspace::<T>(plan.embedding_len(), 800);
            let mut scratch = initialized_workspace::<T>(plan.scratch_len(), 900);
            assert_out_of_place_error(
                direction,
                &plan,
                &source,
                &mut destination,
                &mut embedding,
                &mut scratch,
                LocalR2rError::BufferLengthMismatch,
            );

            let source = input::<T>(plan.line_len() - 1, 1);
            let mut destination = vec![T::from_f64(2.0, -3.0); source.len()];
            let mut embedding = initialized_workspace::<T>(plan.embedding_len(), 1000);
            let mut scratch = initialized_workspace::<T>(plan.scratch_len(), 1100);
            assert_out_of_place_error(
                direction,
                &plan,
                &source,
                &mut destination,
                &mut embedding,
                &mut scratch,
                LocalR2rError::NonIntegralBatch,
            );

            let mut data = input::<T>(plan.line_len() - 1, 1);
            let mut embedding = initialized_workspace::<T>(plan.embedding_len(), 1200);
            let mut scratch = initialized_workspace::<T>(plan.scratch_len(), 1300);
            assert_in_place_error(
                direction,
                &plan,
                &mut data,
                &mut embedding,
                &mut scratch,
                LocalR2rError::NonIntegralBatch,
            );

            let source = input::<T>(plan.line_len(), 1);
            let mut destination = vec![T::from_f64(2.0, -3.0); source.len()];
            let mut embedding = initialized_workspace::<T>(plan.embedding_len() - 1, 1400);
            let mut scratch = initialized_workspace::<T>(plan.scratch_len(), 1500);
            assert_out_of_place_error(
                direction,
                &plan,
                &source,
                &mut destination,
                &mut embedding,
                &mut scratch,
                LocalR2rError::ComplexLineTooSmall {
                    required: plan.embedding_len(),
                    actual: plan.embedding_len() - 1,
                },
            );

            let mut data = input::<T>(plan.line_len(), 1);
            let mut embedding = initialized_workspace::<T>(plan.embedding_len() - 1, 1600);
            let mut scratch = initialized_workspace::<T>(plan.scratch_len(), 1700);
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

            if plan.scratch_len() > 0 {
                let source = input::<T>(plan.line_len(), 1);
                let mut destination = vec![T::from_f64(2.0, -3.0); source.len()];
                let mut embedding = initialized_workspace::<T>(plan.embedding_len(), 1800);
                let mut scratch = initialized_workspace::<T>(plan.scratch_len() - 1, 1900);
                assert_out_of_place_error(
                    direction,
                    &plan,
                    &source,
                    &mut destination,
                    &mut embedding,
                    &mut scratch,
                    LocalR2rError::ScratchTooSmall {
                        required: plan.scratch_len(),
                        actual: plan.scratch_len() - 1,
                    },
                );

                let mut data = input::<T>(plan.line_len(), 1);
                let mut embedding = initialized_workspace::<T>(plan.embedding_len(), 2000);
                let mut scratch = initialized_workspace::<T>(plan.scratch_len() - 1, 2100);
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
    }

    #[test]
    fn validation_empty_batches_tails_and_errors_are_atomic_all_directions() {
        validation_cases::<f32>();
        validation_cases::<f64>();
        validation_cases::<Complex<f32>>();
        validation_cases::<Complex<f64>>();
    }

    #[test]
    fn impossible_lengths_are_rejected_before_planning() {
        for kind in kinds() {
            assert!(matches!(
                LocalR2rPlan::<f64>::new(usize::MAX, kind),
                Err(LocalR2rError::LengthOverflow)
            ));
        }
        assert_eq!(
            validate_lengths::<f32>(usize::MAX / 8 + 1, R2rKind::DctIV),
            Err(LocalR2rError::LengthOverflow)
        );
    }
}
