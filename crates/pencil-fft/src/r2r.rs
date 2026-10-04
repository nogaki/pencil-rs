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
    #[cfg(feature = "fftw")]
    fn native_kind(self) -> pencil_fftw::R2rKind {
        match self {
            Self::DctI => pencil_fftw::R2rKind::DctI,
            Self::DctII => pencil_fftw::R2rKind::DctII,
            Self::DctIII => pencil_fftw::R2rKind::DctIII,
            Self::DctIV => pencil_fftw::R2rKind::DctIV,
            Self::DstI => pencil_fftw::R2rKind::DstI,
            Self::DstII => pencil_fftw::R2rKind::DstII,
            Self::DstIII => pencil_fftw::R2rKind::DstIII,
            Self::DstIV => pencil_fftw::R2rKind::DstIV,
        }
    }

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
/// RustFFT discards the imaginary component of its embedding on output; FFTW
/// transforms only the real component.
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
    /// The real type used by the selected transform backend.
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
/// uses a caller-owned initialized complex workspace line of at least
/// [`Self::embedding_len`] values and initialized FFT scratch of at least
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
/// RustFFT uses an embedding of at most `8 * n` complex values plus its queried
/// scratch. FFTW uses native DCT/DST kernels, requiring only `n` complex workspace
/// values as two contiguous real component buffers, with zero FFT scratch.
/// Larger workspaces remain valid; their unused tails and all native scratch
/// are preserved. Mixed-plan error estimates use a separate, backend-independent
/// model length, unchanged by this compact native workspace.
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
    #[cfg(feature = "distributed")]
    error_model_len: usize,
    scratch_len: usize,
    normalization_factor: usize,
    fft: R2rKernel<T::Real>,
    marker: PhantomData<T>,
    backend: super::BackendKind,
    #[cfg(feature = "fftw")]
    backend_options: Option<super::PlanOptions>,
}

pub(crate) enum R2rKernel<R: FftReal> {
    Rust(Arc<dyn Fft<R>>),
    #[cfg(feature = "fftw")]
    Native([Arc<pencil_fftw::R2rPlan<R>>; 2]),
}

// ponytail: native staging needs n complex values; RustFFT needs its full embedding.
// Numerical error-model length stays backend-independent.
#[cfg(feature = "fftw")]
pub(crate) fn native_r2r_line<T: R2rScalar>(
    plan: &pencil_fftw::R2rPlan<T::Real>,
    source: &[T],
    workspace: &mut [Complex<T::Real>],
) {
    let n = plan.len();
    let (real, imaginary) =
        bytemuck::cast_slice_mut::<Complex<T::Real>, T::Real>(&mut workspace[..n]).split_at_mut(n);
    for ((re, im), &value) in real.iter_mut().zip(imaginary.iter_mut()).zip(source) {
        let value = r2r_to_complex(value);
        *re = value.re;
        *im = value.im;
    }
    plan.process_in_place(real)
        .expect("validated native line length");
    if <T as private::SealedR2rScalar>::VALUE_KIND == 2 {
        plan.process_in_place(imaginary)
            .expect("validated native line length");
    }
}

#[cfg(feature = "fftw")]
pub(crate) fn native_r2r_output<R: FftReal>(
    workspace: &[Complex<R>],
    n: usize,
) -> impl Iterator<Item = Complex<R>> + '_ {
    let (real, imaginary) = bytemuck::cast_slice::<Complex<R>, R>(&workspace[..n]).split_at(n);
    real.iter()
        .zip(imaginary)
        .map(|(&re, &im)| Complex::new(re, im))
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
        let (model_len, normalization_factor) = validate_lengths::<T>(line_len, kind)?;
        let mut planner = FftPlanner::<T::Real>::new();
        let fft = planner.plan_fft_forward(model_len);
        let scratch_len = fft.get_inplace_scratch_len();
        validate_addressable(scratch_len, size_of::<Complex<T::Real>>())?;
        Ok(Self {
            line_len,
            kind,
            embedding_len: model_len,
            #[cfg(feature = "distributed")]
            error_model_len: model_len,
            scratch_len,
            normalization_factor,
            fft: R2rKernel::Rust(fft),
            marker: PhantomData,
            backend: super::BackendKind::RustFft,
            #[cfg(feature = "fftw")]
            backend_options: None,
        })
    }

    /// Builds native DCT/DST plans using runtime-loaded FFTW.
    /// Logical and derived length validation matches [`Self::new`] and precedes
    /// native loading, even though workspace capacity is only `line_len` complex
    /// values. Native lengths must also fit `c_int`; no complex FFT embedding
    /// is executed.
    #[cfg(feature = "fftw")]
    pub fn new_fftw(
        line_len: usize,
        kind: R2rKind,
        options: super::PlanOptions,
    ) -> Result<Self, super::BackendInitError<LocalR2rError>> {
        let (_model_len, normalization_factor) =
            validate_lengths::<T>(line_len, kind).map_err(super::BackendInitError::Local)?;
        let forward = super::backend::r2r(line_len, kind.native_kind(), options)
            .map_err(super::BackendInitError::Native)?;
        let backward = if kind.backward_kind() == kind {
            Arc::clone(&forward)
        } else {
            super::backend::r2r(line_len, kind.backward_kind().native_kind(), options)
                .map_err(super::BackendInitError::Native)?
        };
        Ok(Self {
            line_len,
            kind,
            embedding_len: line_len,
            #[cfg(feature = "distributed")]
            error_model_len: _model_len,
            normalization_factor,
            scratch_len: 0,
            fft: R2rKernel::Native([forward, backward]),
            marker: PhantomData,
            backend: super::BackendKind::Fftw,
            backend_options: Some(options),
        })
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

    /// Returns the minimum initialized complex workspace capacity.
    ///
    /// RustFFT requires its full FFT embedding: `2 * (n - 1)` for DCT-I,
    /// `2 * (n + 1)` for DST-I, `4 * n` for type II/III, and `8 * n` for type IV.
    /// FFTW requires only `n = line_len` complex values for native staging.
    /// This backend-specific capacity is independent of the numerical model
    /// and normalization factor. Oversized workspace tails are preserved.
    pub fn embedding_len(&self) -> usize {
        self.embedding_len
    }

    /// Returns the backend-independent model length for mixed-plan error estimates.
    #[cfg(feature = "distributed")]
    pub(crate) fn error_model_len(&self) -> usize {
        self.error_model_len
    }

    /// Returns the FFT scratch length (zero for native FFTW R2R).
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
        self.transform_line(kind, source, embedding_line, scratch);
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
        self.transform_line(kind, data, embedding_line, scratch);
        self.write_output(kind, data, embedding_line, scale);
    }

    fn transform_line(
        &self,
        kind: R2rKind,
        source: &[T],
        embedding_line: &mut [Complex<T::Real>],
        scratch: &mut [Complex<T::Real>],
    ) {
        match &self.fft {
            R2rKernel::Rust(fft) => {
                self.build_embedding(kind, source, embedding_line);
                fft.process_with_scratch(
                    &mut embedding_line[..self.embedding_len],
                    &mut scratch[..self.scratch_len],
                );
            }
            #[cfg(feature = "fftw")]
            R2rKernel::Native(plans) => {
                native_r2r_line(
                    &plans[usize::from(kind != self.kind)],
                    source,
                    embedding_line,
                );
            }
        }
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
        #[cfg(feature = "fftw")]
        if matches!(self.fft, R2rKernel::Native(_)) {
            for (out, mut value) in destination
                .iter_mut()
                .zip(native_r2r_output(embedding_line, self.line_len))
            {
                if let Some(scale) = scale {
                    value.re = value.re * scale;
                    value.im = value.im * scale;
                }
                *out = T::from_complex(value);
            }
            return;
        }
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

    trait TestScalar: R2rScalar + bytemuck::Pod {
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

    fn assert_scalar_preserved<T: bytemuck::Pod>(actual: &[T], expected: &[T]) {
        assert_eq!(
            bytemuck::cast_slice::<T, u8>(actual),
            bytemuck::cast_slice::<T, u8>(expected)
        );
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

    fn exercise_plan<T: TestScalar>(plan: LocalR2rPlan<T>, batches: usize) {
        let kind = plan.kind();
        let n = plan.line_len();
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
                exercise_plan(LocalR2rPlan::<f32>::new(n, kind).unwrap(), 2);
                exercise_plan(LocalR2rPlan::<f64>::new(n, kind).unwrap(), 2);
                exercise_plan(LocalR2rPlan::<Complex<f32>>::new(n, kind).unwrap(), 2);
                exercise_plan(LocalR2rPlan::<Complex<f64>>::new(n, kind).unwrap(), 2);
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

    #[cfg(all(feature = "distributed", feature = "fftw"))]
    #[test]
    #[ignore = "requires both native FFTW runtimes"]
    fn numerical_model_contract() {
        fn check<T: TestScalar>(n: usize, kind: R2rKind, expected: (usize, usize)) {
            for mut plan in [
                LocalR2rPlan::<T>::new(n, kind).unwrap(),
                LocalR2rPlan::<T>::new_fftw(n, kind, super::super::PlanOptions::default()).unwrap(),
            ] {
                assert_eq!(
                    plan.embedding_len(),
                    match plan.backend_kind() {
                        super::super::BackendKind::RustFft => expected.0,
                        super::super::BackendKind::Fftw => n,
                    }
                );
                assert_eq!(
                    (plan.error_model_len(), plan.normalization_factor()),
                    expected
                );
                if plan.backend_kind() == super::super::BackendKind::RustFft && n < expected.0 {
                    // A compact native buffer is not sufficient for RustFFT.
                    let source = input::<T>(n, 1);
                    for direction in DIRECTIONS {
                        let mut destination = source.clone();
                        let mut embedding = initialized_workspace::<T>(n, 0);
                        let mut scratch = initialized_workspace::<T>(plan.scratch_len(), 100);
                        let error = LocalR2rError::ComplexLineTooSmall {
                            required: expected.0,
                            actual: n,
                        };
                        assert_out_of_place_error(
                            direction,
                            &plan,
                            &source,
                            &mut destination,
                            &mut embedding,
                            &mut scratch,
                            error,
                        );
                        assert_in_place_error(
                            direction,
                            &plan,
                            &mut destination,
                            &mut embedding,
                            &mut scratch,
                            error,
                        );
                    }
                }
                // Metadata-only mutation: never execute this altered plan.
                plan.embedding_len = 0;
                assert_eq!(
                    (plan.error_model_len(), plan.normalization_factor()),
                    expected
                );
            }
        }

        // Independent (model length, normalization) table for n = 1, 2, 5, 8.
        // DCT-I at n = 1 is invalid; its placeholder is skipped below.
        for (kind, cases) in [
            (R2rKind::DctI, [(0, 0), (2, 2), (8, 8), (14, 14)]),
            (R2rKind::DctII, [(4, 2), (8, 4), (20, 10), (32, 16)]),
            (R2rKind::DctIII, [(4, 2), (8, 4), (20, 10), (32, 16)]),
            (R2rKind::DctIV, [(8, 2), (16, 4), (40, 10), (64, 16)]),
            (R2rKind::DstI, [(4, 4), (6, 6), (12, 12), (18, 18)]),
            (R2rKind::DstII, [(4, 2), (8, 4), (20, 10), (32, 16)]),
            (R2rKind::DstIII, [(4, 2), (8, 4), (20, 10), (32, 16)]),
            (R2rKind::DstIV, [(8, 2), (16, 4), (40, 10), (64, 16)]),
        ] {
            for (n, expected) in [1, 2, 5, 8].into_iter().zip(cases) {
                if kind == R2rKind::DctI && n == 1 {
                    continue;
                }
                check::<f32>(n, kind, expected);
                check::<f64>(n, kind, expected);
                check::<Complex<f32>>(n, kind, expected);
                check::<Complex<f64>>(n, kind, expected);
            }
        }
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

    fn validation_cases<T: TestScalar>(plan: LocalR2rPlan<T>) {
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

            if plan.line_len() > 1 {
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
            }

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
        validation_cases(LocalR2rPlan::<f32>::new(4, R2rKind::DctIV).unwrap());
        validation_cases(LocalR2rPlan::<f64>::new(4, R2rKind::DctIV).unwrap());
        validation_cases(LocalR2rPlan::<Complex<f32>>::new(4, R2rKind::DctIV).unwrap());
        validation_cases(LocalR2rPlan::<Complex<f64>>::new(4, R2rKind::DctIV).unwrap());
    }

    #[cfg(feature = "fftw")]
    fn native_workspace_cases<T: TestScalar>(plan: &LocalR2rPlan<T>) {
        let n = plan.line_len();
        let kind = plan.kind();
        // Independent historical capacities, never the new capacity getter.
        let legacy = match kind {
            R2rKind::DctI => 2 * (n - 1),
            R2rKind::DstI => 2 * (n + 1),
            R2rKind::DctII | R2rKind::DctIII | R2rKind::DstII | R2rKind::DstIII => 4 * n,
            R2rKind::DctIV | R2rKind::DstIV => 8 * n,
        };
        let nan = f64::from_bits(0x7ff8_1234_5678_9abc);
        let guard = T::from_f64(nan, -nan);
        let workspace_guard = Complex::new(
            T::Real::from_f64(nan).unwrap(),
            T::Real::from_f64(-nan).unwrap(),
        );
        for batches in [0, 2] {
            let arbitrary = independent_input::<T>(n, batches);
            let end = 1 + arbitrary.len();
            let mut source = vec![guard; end + 1];
            source[1..end].copy_from_slice(&arbitrary);
            let source_before = source.clone();
            for direction in DIRECTIONS {
                let oracle = expected(kind, direction, n, &arbitrary);
                for (capacity, scratch_len) in
                    [(n - 1, 3), (n, 0), (n, 3), (legacy, 3), (legacy + 3, 3)]
                {
                    for in_place in [false, true] {
                        let mut data = vec![guard; source.len()];
                        if in_place {
                            data[1..end].copy_from_slice(&arbitrary);
                        }
                        let data_before = data.clone();
                        let mut embedding = vec![workspace_guard; capacity + 2];
                        let embedding_before = embedding.clone();
                        let mut scratch = vec![workspace_guard; scratch_len + 2];
                        let scratch_before = scratch.clone();
                        // Offset every supplied slice, including zero-length scratch.
                        let result = if in_place {
                            direction.in_place(
                                plan,
                                &mut data[1..end],
                                &mut embedding[1..1 + capacity],
                                &mut scratch[1..1 + scratch_len],
                            )
                        } else {
                            direction.out_of_place(
                                plan,
                                &source[1..end],
                                &mut data[1..end],
                                &mut embedding[1..1 + capacity],
                                &mut scratch[1..1 + scratch_len],
                            )
                        };
                        assert_scalar_preserved(&source, &source_before);
                        assert_scalar_preserved(&scratch, &scratch_before);
                        if capacity < n {
                            assert_eq!(
                                result,
                                Err(LocalR2rError::ComplexLineTooSmall {
                                    required: n,
                                    actual: n - 1,
                                })
                            );
                            assert_scalar_preserved(&data, &data_before);
                            assert_scalar_preserved(&embedding, &embedding_before);
                        } else {
                            assert_eq!(result, Ok(()));
                            assert_close(&data[1..end], &oracle);
                            assert_scalar_preserved(&data[..1], &data_before[..1]);
                            assert_scalar_preserved(&data[end..], &data_before[end..]);
                            let used = if batches == 0 { 0 } else { n };
                            assert_scalar_preserved(&embedding[..1], &embedding_before[..1]);
                            assert_scalar_preserved(
                                &embedding[1 + used..],
                                &embedding_before[1 + used..],
                            );
                        }
                    }
                }
            }
        }
    }

    #[cfg(feature = "fftw")]
    #[test]
    #[ignore = "requires both native FFTW runtimes"]
    fn native_r2r_oracles_and_workspace_contracts() {
        fn check<T: TestScalar>(n: usize, kind: R2rKind) {
            let plan = || {
                LocalR2rPlan::<T>::new_fftw(n, kind, super::super::PlanOptions::default()).unwrap()
            };
            assert_eq!(plan().embedding_len(), n);
            assert_eq!(plan().scratch_len(), 0);
            native_workspace_cases(&plan());
            exercise_plan(plan(), 2);
            validation_cases(plan());
        }
        for kind in kinds() {
            for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11] {
                if kind == R2rKind::DctI && n == 1 {
                    continue;
                }
                check::<f32>(n, kind);
                check::<f64>(n, kind);
                check::<Complex<f32>>(n, kind);
                check::<Complex<f64>>(n, kind);
            }
        }
    }

    #[cfg(feature = "fftw")]
    #[test]
    fn native_r2r_invalid_lengths_precede_loading() {
        use super::super::{BackendInitError, PlanOptions};
        fn check<T: R2rScalar>() {
            for n in [0, 1] {
                assert!(matches!(
                    LocalR2rPlan::<T>::new_fftw(n, R2rKind::DctI, PlanOptions::default()),
                    Err(BackendInitError::Local(LocalR2rError::InvalidLength))
                ));
            }
            // Both the logical line and compact workspace fit, but even the
            // smallest derived model (DCT-I) exceeds complex addressability.
            let derived_overflow = isize::MAX as usize / size_of::<Complex<T::Real>>() / 2 + 2;
            assert!(derived_overflow <= isize::MAX as usize / size_of::<T>());
            assert!(derived_overflow <= isize::MAX as usize / size_of::<Complex<T::Real>>());
            for kind in kinds() {
                for n in [usize::MAX, derived_overflow] {
                    assert!(matches!(
                        LocalR2rPlan::<T>::new_fftw(n, kind, PlanOptions::default()),
                        Err(BackendInitError::Local(LocalR2rError::LengthOverflow))
                    ));
                }
            }
        }
        check::<f32>();
        check::<f64>();
        check::<Complex<f32>>();
        check::<Complex<f64>>();
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
