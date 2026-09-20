use std::fmt;
use std::mem::size_of;
use std::sync::Arc;

use bytemuck::{try_cast_slice, try_cast_slice_mut};
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

use super::{Complex, FftReal};

/// Completion state of a real-to-half-complex in-place array.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum R2cState {
    /// The initialized prefix contains packed real input values.
    RealInput,
    /// The initialized prefix contains half-complex output values.
    ComplexOutput,
    /// An in-place operation started but did not complete.
    Poisoned,
}

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
    /// A requested initialized allocation could not be made.
    #[error("failed to allocate {required} elements")]
    AllocationFailed {
        /// The requested number of elements.
        required: usize,
    },
    /// The supplied in-place array was allocated for another real line length.
    #[error("in-place array does not match this real FFT plan")]
    ArrayMismatch,
    /// The supplied in-place workspace was allocated for another real line length.
    #[error("in-place workspace does not match this real FFT plan")]
    WorkspaceMismatch,
    /// The in-place backing vector length was not the plan's required length.
    #[error("in-place storage length {actual} does not equal required length {required}")]
    StorageLengthMismatch {
        /// The required number of complex storage elements.
        required: usize,
        /// The actual number of complex storage elements.
        actual: usize,
    },
    /// The backing storage could not be viewed as its sealed real scalar type.
    #[error("in-place storage cannot be safely viewed as its real scalar type")]
    StorageLayoutMismatch,
    /// An in-place operation was requested from the wrong valid state.
    #[error("in-place array is in the wrong state")]
    WrongState,
    /// An in-place operation or view was requested after an incomplete operation.
    #[error("in-place array is poisoned")]
    Poisoned,
    /// A constrained real-spectrum endpoint had a non-zero imaginary component.
    #[error(
        "inverse spectrum batch {batch} has a non-zero imaginary component at complex index {index}"
    )]
    InvalidSpectrumEndpoint {
        /// The zero-based batch containing the invalid endpoint.
        batch: usize,
        /// The complex index of the endpoint. This is zero for DC and is the
        /// final complex index for an even-length Nyquist endpoint.
        index: usize,
    },
}

/// An opaque single-allocation array for local real-to-half-complex execution.
///
/// The backing storage is a `Vec<Complex<R>>` with `complex_len * batch_count`
/// elements. In [`R2cState::RealInput`], only the packed real prefix of
/// `real_len * batch_count` scalars is exposed. In [`R2cState::ComplexOutput`],
/// the complex prefix is exposed. The two views borrow the same allocation and
/// cannot be held mutably at the same time.
///
/// ```compile_fail
/// use pencil_fft::LocalR2cPlan;
///
/// # fn main() -> Result<(), pencil_fft::LocalR2cError> {
/// let plan = LocalR2cPlan::<f64>::new(4)?;
/// let mut array = plan.allocate_in_place(1)?;
/// let real = array.real_view_mut()?;
/// let _complex = array.complex_view_mut()?;
/// let _ = real.len();
/// # Ok(())
/// # }
/// ```
pub struct LocalR2cInPlaceArray<R: FftReal> {
    storage: Vec<Complex<R>>,
    real_len: usize,
    batch_count: usize,
    state: R2cState,
}

impl<R: FftReal> fmt::Debug for LocalR2cInPlaceArray<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LocalR2cInPlaceArray")
            .field("real_len", &self.real_len)
            .field("batch_count", &self.batch_count)
            .field("storage_len", &self.storage.len())
            .field("state", &self.state)
            .finish_non_exhaustive()
    }
}

impl<R: FftReal> LocalR2cInPlaceArray<R> {
    /// Returns the current mathematical contents state of the array.
    pub fn state(&self) -> R2cState {
        self.state
    }

    /// Borrows the initialized packed real prefix in [`R2cState::RealInput`].
    pub fn real_view(&self) -> Result<&[R], LocalR2cError> {
        self.ensure_state(R2cState::RealInput)?;
        let length = self.validate_shape()?.0;
        let storage = cast_real_slice(&self.storage)?;
        Ok(&storage[..length])
    }

    /// Borrows the initialized packed real prefix mutably in
    /// [`R2cState::RealInput`].
    pub fn real_view_mut(&mut self) -> Result<&mut [R], LocalR2cError> {
        self.ensure_state(R2cState::RealInput)?;
        let length = self.validate_shape()?.0;
        let storage = cast_real_slice_mut(&mut self.storage)?;
        Ok(&mut storage[..length])
    }

    /// Borrows the initialized complex prefix in [`R2cState::ComplexOutput`].
    pub fn complex_view(&self) -> Result<&[Complex<R>], LocalR2cError> {
        self.ensure_state(R2cState::ComplexOutput)?;
        let length = self.validate_shape()?.1;
        Ok(&self.storage[..length])
    }

    /// Borrows the initialized complex prefix mutably in
    /// [`R2cState::ComplexOutput`].
    pub fn complex_view_mut(&mut self) -> Result<&mut [Complex<R>], LocalR2cError> {
        self.ensure_state(R2cState::ComplexOutput)?;
        let length = self.validate_shape()?.1;
        Ok(&mut self.storage[..length])
    }

    fn ensure_state(&self, expected: R2cState) -> Result<(), LocalR2cError> {
        match self.state {
            R2cState::Poisoned => Err(LocalR2cError::Poisoned),
            state if state == expected => Ok(()),
            _ => Err(LocalR2cError::WrongState),
        }
    }

    fn validate_shape(&self) -> Result<(usize, usize), LocalR2cError> {
        let complex_len = self.real_len / 2 + 1;
        let real_values = self
            .real_len
            .checked_mul(self.batch_count)
            .ok_or(LocalR2cError::LengthOverflow)?;
        let complex_values = complex_len
            .checked_mul(self.batch_count)
            .ok_or(LocalR2cError::LengthOverflow)?;
        if self.storage.len() != complex_values {
            return Err(LocalR2cError::StorageLengthMismatch {
                required: complex_values,
                actual: self.storage.len(),
            });
        }
        let _ = cast_real_slice(&self.storage)?;
        Ok((real_values, complex_values))
    }
}

/// Reusable initialized line storage for local real-to-half-complex in-place
/// execution.
///
/// The workspace contains one real line, one complex line, and the shared
/// native complex scratch prefix. It is private to the line length that
/// allocated it; oversized tails, if present through internal tests, are not
/// touched.
pub struct LocalR2cInPlaceWorkspace<R: FftReal> {
    real_line: Vec<R>,
    complex_line: Vec<Complex<R>>,
    scratch: Vec<Complex<R>>,
    real_len: usize,
    complex_len: usize,
}

impl<R: FftReal> fmt::Debug for LocalR2cInPlaceWorkspace<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LocalR2cInPlaceWorkspace")
            .field("real_len", &self.real_len)
            .field("complex_len", &self.complex_len)
            .field("real_line_len", &self.real_line.len())
            .field("complex_line_len", &self.complex_line.len())
            .field("scratch_len", &self.scratch.len())
            .finish_non_exhaustive()
    }
}

/// An immutable local batched real-to-half-complex and half-complex-to-real FFT plan.
///
/// Each operation treats its data slices as contiguous row-major batches. The
/// plan owns immutable RealFFT plans, but does not own out-of-place data or
/// caller-provided workspaces. Forward transforms preserve their real source
/// and inverse transforms preserve their complex source. The
/// [`Self::allocate_in_place`] API additionally provides one packed
/// `Vec<Complex<R>>` allocation with state-checked real and complex views.
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

    #[cfg(all(test, feature = "distributed"))]
    pub(crate) fn inject_scratch_shortage_for_test(&mut self) {
        self.scratch_len += 1;
    }

    /// Allocates one zero-initialized packed array for `batch_count` real
    /// lines.
    ///
    /// The allocation contains exactly `complex_len * batch_count`
    /// `Complex<R>` elements and is noncollective. Products, byte sizes, and
    /// the fallible allocation are checked before initialization. Execution
    /// never resizes this storage or allocates a full-array temporary.
    pub fn allocate_in_place(
        &self,
        batch_count: usize,
    ) -> Result<LocalR2cInPlaceArray<R>, LocalR2cError> {
        let (_, complex_values) = self.checked_in_place_lengths(batch_count)?;
        let mut storage = Vec::new();
        storage
            .try_reserve_exact(complex_values)
            .map_err(|_| LocalR2cError::AllocationFailed {
                required: complex_values,
            })?;
        storage.resize(complex_values, Complex::new(R::zero(), R::zero()));
        Ok(LocalR2cInPlaceArray {
            storage,
            real_len: self.real_len,
            batch_count,
            state: R2cState::RealInput,
        })
    }

    /// Allocates initialized one-line storage and shared native scratch for
    /// packed in-place execution.
    ///
    /// The workspace is noncollective, plan-bound by its line metadata, and
    /// contains no copy of the full transform array. Allocation failures are
    /// returned as [`LocalR2cError::AllocationFailed`].
    pub fn allocate_in_place_workspace(
        &self,
    ) -> Result<LocalR2cInPlaceWorkspace<R>, LocalR2cError> {
        validate_addressable(self.real_len, size_of::<R>())?;
        validate_addressable(self.complex_len, size_of::<Complex<R>>())?;
        validate_addressable(self.scratch_len, size_of::<Complex<R>>())?;
        Ok(LocalR2cInPlaceWorkspace {
            real_line: initialized_vec(self.real_len, R::zero())?,
            complex_line: initialized_vec(self.complex_len, Complex::new(R::zero(), R::zero()))?,
            scratch: initialized_vec(self.scratch_len, Complex::new(R::zero(), R::zero()))?,
            real_len: self.real_len,
            complex_len: self.complex_len,
        })
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
        self.execute_inverse(src, dst, complex_line, scratch, true)
    }

    /// Computes an unnormalized positive-sign backward half-complex-to-real
    /// FFT for every line in `src`.
    ///
    /// This is the raw local C2R transform: it does not divide each result by
    /// [`Self::real_len`]. Validation, endpoint checks, source preservation,
    /// line and scratch buffer requirements, and oversized-tail behavior are
    /// the same as for [`Self::inverse`].
    pub fn backward(
        &self,
        src: &[Complex<R>],
        dst: &mut [R],
        complex_line: &mut [Complex<R>],
        scratch: &mut [Complex<R>],
    ) -> Result<(), LocalR2cError> {
        self.execute_inverse(src, dst, complex_line, scratch, false)
    }

    /// Computes an unnormalized forward transform in the packed in-place
    /// array.
    ///
    /// The array must be in [`R2cState::RealInput`]. Real rows are copied into
    /// the workspace and processed from back to front because the complex
    /// output occupies more bytes than the real input. All validation is done
    /// before the array enters [`R2cState::Poisoned`]. After execution starts,
    /// a backend panic leaves it poisoned; only complete success commits
    /// [`R2cState::ComplexOutput`].
    pub fn forward_in_place(
        &self,
        array: &mut LocalR2cInPlaceArray<R>,
        workspace: &mut LocalR2cInPlaceWorkspace<R>,
    ) -> Result<(), LocalR2cError> {
        let batch_count = self.validate_in_place(array, workspace, R2cState::RealInput)?;
        let native_scratch_len = self.forward.get_scratch_len();
        if native_scratch_len > workspace.scratch.len() {
            return Err(LocalR2cError::ScratchTooSmall {
                required: native_scratch_len,
                actual: workspace.scratch.len(),
            });
        }
        array.state = R2cState::Poisoned;
        for batch in (0..batch_count).rev() {
            let source_start = batch * self.real_len;
            let source_end = source_start + self.real_len;
            {
                let storage = cast_real_slice(&array.storage)
                    .expect("validated in-place storage must have a real view");
                workspace.real_line[..self.real_len]
                    .copy_from_slice(&storage[source_start..source_end]);
            }
            let destination_start = batch * self.complex_len;
            let destination_end = destination_start + self.complex_len;
            self.forward
                .process_with_scratch(
                    &mut workspace.real_line[..self.real_len],
                    &mut array.storage[destination_start..destination_end],
                    &mut workspace.scratch[..native_scratch_len],
                )
                .expect("validated RealFFT forward buffers must be accepted");
        }
        array.state = R2cState::ComplexOutput;
        Ok(())
    }

    /// Computes a normalized inverse transform in the packed in-place array.
    ///
    /// The array must be in [`R2cState::ComplexOutput`]. Every DC endpoint,
    /// and every Nyquist endpoint for an even line length, is checked before
    /// any array, state, or workspace mutation. The result is divided by
    /// exactly [`Self::real_len`].
    pub fn inverse_in_place(
        &self,
        array: &mut LocalR2cInPlaceArray<R>,
        workspace: &mut LocalR2cInPlaceWorkspace<R>,
    ) -> Result<(), LocalR2cError> {
        self.execute_in_place_inverse(array, workspace, true)
    }

    /// Computes an unnormalized positive-sign backward transform in the
    /// packed in-place array.
    ///
    /// This is the raw local C2R operation and does not divide by
    /// [`Self::real_len`]. Validation and endpoint rules are the same as for
    /// [`Self::inverse_in_place`].
    pub fn backward_in_place(
        &self,
        array: &mut LocalR2cInPlaceArray<R>,
        workspace: &mut LocalR2cInPlaceWorkspace<R>,
    ) -> Result<(), LocalR2cError> {
        self.execute_in_place_inverse(array, workspace, false)
    }

    fn execute_in_place_inverse(
        &self,
        array: &mut LocalR2cInPlaceArray<R>,
        workspace: &mut LocalR2cInPlaceWorkspace<R>,
        normalize: bool,
    ) -> Result<(), LocalR2cError> {
        let batch_count = self.validate_in_place(array, workspace, R2cState::ComplexOutput)?;
        let scale = if normalize {
            Some(
                R::one()
                    / R::from_usize(self.real_len).expect("f32/f64 represent the validated length"),
            )
        } else {
            None
        };
        let native_scratch_len = self.inverse.get_scratch_len();
        if native_scratch_len > workspace.scratch.len() {
            return Err(LocalR2cError::ScratchTooSmall {
                required: native_scratch_len,
                actual: workspace.scratch.len(),
            });
        }
        array.state = R2cState::Poisoned;
        for batch in 0..batch_count {
            let source_start = batch * self.complex_len;
            let source_end = source_start + self.complex_len;
            workspace.complex_line[..self.complex_len]
                .copy_from_slice(&array.storage[source_start..source_end]);

            let destination_start = batch * self.real_len;
            let destination_end = destination_start + self.real_len;
            {
                let storage = cast_real_slice_mut(&mut array.storage)
                    .expect("validated in-place storage must have a real view");
                self.inverse
                    .process_with_scratch(
                        &mut workspace.complex_line[..self.complex_len],
                        &mut storage[destination_start..destination_end],
                        &mut workspace.scratch[..native_scratch_len],
                    )
                    .expect("validated RealFFT inverse buffers must be accepted");
            }
            if let Some(scale) = scale {
                let storage = cast_real_slice_mut(&mut array.storage)
                    .expect("validated in-place storage must have a real view");
                for value in &mut storage[destination_start..destination_end] {
                    *value = *value * scale;
                }
            }
        }
        array.state = R2cState::RealInput;
        Ok(())
    }

    fn execute_inverse(
        &self,
        src: &[Complex<R>],
        dst: &mut [R],
        complex_line: &mut [Complex<R>],
        scratch: &mut [Complex<R>],
        normalize: bool,
    ) -> Result<(), LocalR2cError> {
        self.validate_inverse(src.len(), dst.len(), complex_line.len(), scratch.len())?;
        // RealFFT reports endpoint errors after writing; preflight the entire batch.
        self.validate_endpoints(src)?;
        if src.is_empty() {
            return Ok(());
        }

        let native_scratch_len = self.inverse.get_scratch_len();
        let scale = if normalize {
            Some(
                R::one()
                    / R::from_usize(self.real_len).expect("f32/f64 represent the validated length"),
            )
        } else {
            None
        };
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
            if let Some(scale) = scale {
                for value in destination_line {
                    *value = *value * scale;
                }
            }
        }
        Ok(())
    }

    fn checked_in_place_lengths(
        &self,
        batch_count: usize,
    ) -> Result<(usize, usize), LocalR2cError> {
        let real_values = self
            .real_len
            .checked_mul(batch_count)
            .ok_or(LocalR2cError::LengthOverflow)?;
        let complex_values = self
            .complex_len
            .checked_mul(batch_count)
            .ok_or(LocalR2cError::LengthOverflow)?;
        validate_addressable(real_values, size_of::<R>())?;
        validate_addressable(complex_values, size_of::<Complex<R>>())?;
        Ok((real_values, complex_values))
    }

    fn validate_in_place(
        &self,
        array: &LocalR2cInPlaceArray<R>,
        workspace: &LocalR2cInPlaceWorkspace<R>,
        expected_state: R2cState,
    ) -> Result<usize, LocalR2cError> {
        match array.state {
            R2cState::Poisoned => return Err(LocalR2cError::Poisoned),
            state if state != expected_state => return Err(LocalR2cError::WrongState),
            _ => {}
        }
        if array.real_len != self.real_len {
            return Err(LocalR2cError::ArrayMismatch);
        }
        if workspace.real_len != self.real_len || workspace.complex_len != self.complex_len {
            return Err(LocalR2cError::WorkspaceMismatch);
        }
        let (_, complex_values) = self.checked_in_place_lengths(array.batch_count)?;
        if array.storage.len() != complex_values {
            return Err(LocalR2cError::StorageLengthMismatch {
                required: complex_values,
                actual: array.storage.len(),
            });
        }
        let storage = cast_real_slice(&array.storage)?;
        let real_values = self
            .real_len
            .checked_mul(array.batch_count)
            .ok_or(LocalR2cError::LengthOverflow)?;
        if storage.len() < real_values {
            return Err(LocalR2cError::StorageLayoutMismatch);
        }
        if workspace.real_line.len() < self.real_len {
            return Err(LocalR2cError::RealLineTooSmall {
                required: self.real_len,
                actual: workspace.real_line.len(),
            });
        }
        if workspace.complex_line.len() < self.complex_len {
            return Err(LocalR2cError::ComplexLineTooSmall {
                required: self.complex_len,
                actual: workspace.complex_line.len(),
            });
        }
        self.validate_scratch(workspace.scratch.len())?;
        if expected_state == R2cState::ComplexOutput {
            self.validate_endpoints(&array.storage[..complex_values])?;
        }
        Ok(array.batch_count)
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

fn cast_real_slice<R: FftReal>(storage: &[Complex<R>]) -> Result<&[R], LocalR2cError> {
    try_cast_slice(storage).map_err(|_| LocalR2cError::StorageLayoutMismatch)
}

fn cast_real_slice_mut<R: FftReal>(storage: &mut [Complex<R>]) -> Result<&mut [R], LocalR2cError> {
    try_cast_slice_mut(storage).map_err(|_| LocalR2cError::StorageLayoutMismatch)
}

fn initialized_vec<T: Clone>(length: usize, value: T) -> Result<Vec<T>, LocalR2cError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| LocalR2cError::AllocationFailed { required: length })?;
    values.resize(length, value);
    Ok(values)
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

    fn dft_inverse<R: TestReal>(input: &[Complex<R>], real_len: usize, raw: bool) -> Vec<R> {
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
                    if !raw {
                        value /= real_len as f64;
                    }
                    R::convert(value)
                })
            })
            .collect()
    }

    fn assert_complex_close<R: TestReal>(actual: &[Complex<R>], expected: &[Complex<R>]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let atol = R::tolerance();
            let rtol = R::tolerance();
            let real_bound = atol + rtol * expected.re.as_f64().abs();
            let imaginary_bound = atol + rtol * expected.im.as_f64().abs();
            let real_error = (actual.re.as_f64() - expected.re.as_f64()).abs();
            let imaginary_error = (actual.im.as_f64() - expected.im.as_f64()).abs();
            assert!(
                real_error <= real_bound && imaginary_error <= imaginary_bound,
                "index {index}: actual={actual:?}, expected={expected:?}, real_bound={real_bound}, imaginary_bound={imaginary_bound}"
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

        let mut raw_recovered = vec![R::convert(29.0); source.len()];
        dirty_complex(&mut complex_line, 41.0);
        dirty_complex(&mut scratch, 43.0);
        let complex_line_tail_before = complex_line[plan.complex_len()..].to_vec();
        let scratch_tail_before = scratch[plan.scratch_len()..].to_vec();
        plan.backward(
            &spectrum,
            &mut raw_recovered,
            &mut complex_line,
            &mut scratch,
        )
        .unwrap();
        let raw_expected = source_before
            .iter()
            .map(|value| R::convert(value.as_f64() * real_len as f64))
            .collect::<Vec<_>>();
        assert_eq!(spectrum, spectrum_before_inverse);
        assert_real_close(&raw_recovered, &raw_expected);
        assert_eq!(
            &complex_line[plan.complex_len()..],
            &complex_line_tail_before
        );
        assert_eq!(&scratch[plan.scratch_len()..], &scratch_tail_before);

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
        assert_real_close(&arbitrary_output, &dft_inverse(&arbitrary, real_len, false));

        let mut arbitrary_raw_output = vec![R::convert(0.0); real_len * batch_count];
        plan.backward(
            &arbitrary,
            &mut arbitrary_raw_output,
            &mut complex_line,
            &mut scratch,
        )
        .unwrap();
        let arbitrary_raw_expected = dft_inverse(&arbitrary, real_len, true);
        assert_eq!(arbitrary, arbitrary_before);
        assert_real_close(&arbitrary_raw_output, &arbitrary_raw_expected);
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

        let mut raw_output = vec![R::convert(0.0); real_len];
        plan.backward(&spectrum, &mut raw_output, &mut complex_line, &mut scratch)
            .unwrap();
        let raw_expected = dc
            .iter()
            .map(|value| R::convert(value.as_f64() * real_len as f64))
            .collect::<Vec<_>>();
        assert_real_close(&raw_output, &raw_expected);
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
            (1, 0, R::convert(3.0)),
            (0, plan.complex_len() - 1, R::convert(-2.0)),
            (2, plan.complex_len() - 1, R::convert(f64::NAN)),
            (1, plan.complex_len() - 1, R::convert(4.0)),
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

            assert_eq!(
                plan.backward(&source, &mut destination, &mut line, &mut scratch),
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
        assert_eq!(
            plan.backward(&source, &mut destination, &mut line, &mut scratch),
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
        assert_eq!(
            plan.backward(&source, &mut destination, &mut line, &mut scratch),
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
        assert_eq!(
            plan.backward(&spectrum, &mut Vec::new(), &mut complex_line, &mut scratch),
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
            assert_eq!(
                plan.backward(&source, &mut destination, &mut line, &mut scratch),
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
        assert_eq!(
            plan.backward(&source, &mut destination, &mut line, &mut scratch),
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
            assert_eq!(
                plan.backward(
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
            assert_eq!(
                plan.backward(
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

    fn exercise_in_place<R: TestReal>(real_len: usize, batch_count: usize) {
        let plan = LocalR2cPlan::<R>::new(real_len).unwrap();
        let source = real_input::<R>(real_len, batch_count);
        let expected_spectrum = dft_forward(&source, real_len);
        let mut oop_spectrum = vec![
            Complex::new(R::convert(17.0), R::convert(-23.0));
            plan.complex_len() * batch_count
        ];
        let mut oop_real_line = vec![R::convert(0.0); real_len];
        let mut oop_scratch = initialized_complex::<R>(plan.scratch_len(), 3.0);
        plan.forward(
            &source,
            &mut oop_spectrum,
            &mut oop_real_line,
            &mut oop_scratch,
        )
        .unwrap();
        assert_complex_close(&oop_spectrum, &expected_spectrum);

        let mut array = plan.allocate_in_place(batch_count).unwrap();
        assert_eq!(array.state(), R2cState::RealInput);
        assert_eq!(array.real_view().unwrap().len(), real_len * batch_count);
        assert_eq!(
            array.real_view_mut().unwrap().as_mut_ptr() as *const R,
            array.real_view().unwrap().as_ptr()
        );
        array.real_view_mut().unwrap().copy_from_slice(&source);
        assert!(matches!(
            array.complex_view(),
            Err(LocalR2cError::WrongState)
        ));
        let real_pointer = array.real_view().unwrap().as_ptr();
        let mut workspace = plan.allocate_in_place_workspace().unwrap();

        plan.forward_in_place(&mut array, &mut workspace).unwrap();
        assert_eq!(array.state(), R2cState::ComplexOutput);
        let spectrum = array.complex_view().unwrap();
        assert_eq!(spectrum.len(), plan.complex_len() * batch_count);
        assert_complex_close(spectrum, &expected_spectrum);
        if batch_count != 0 {
            assert_eq!(spectrum.as_ptr() as *const R, real_pointer);
        }
        assert!(matches!(array.real_view(), Err(LocalR2cError::WrongState)));

        let spectrum_before = spectrum.to_vec();
        plan.inverse_in_place(&mut array, &mut workspace).unwrap();
        assert_eq!(array.state(), R2cState::RealInput);
        assert_real_close(array.real_view().unwrap(), &source);
        assert_eq!(array.real_view().unwrap().as_ptr(), real_pointer);
        assert_complex_close(&spectrum_before, &expected_spectrum);

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
        let expected_arbitrary = dft_inverse(&arbitrary, real_len, false);
        let mut oop_arbitrary = vec![R::convert(0.0); real_len * batch_count];
        let mut oop_complex_line = initialized_complex::<R>(plan.complex_len(), 11.0);
        plan.inverse(
            &arbitrary,
            &mut oop_arbitrary,
            &mut oop_complex_line,
            &mut oop_scratch,
        )
        .unwrap();
        assert_real_close(&oop_arbitrary, &expected_arbitrary);

        plan.forward_in_place(&mut array, &mut workspace).unwrap();
        array
            .complex_view_mut()
            .unwrap()
            .copy_from_slice(&arbitrary);
        let output_pointer = array.complex_view().unwrap().as_ptr();
        plan.inverse_in_place(&mut array, &mut workspace).unwrap();
        assert_eq!(
            array.real_view().unwrap().as_ptr() as *const _,
            output_pointer as *const _
        );
        assert_real_close(array.real_view().unwrap(), &expected_arbitrary);

        plan.forward_in_place(&mut array, &mut workspace).unwrap();
        array
            .complex_view_mut()
            .unwrap()
            .copy_from_slice(&arbitrary);
        let expected_raw = dft_inverse(&arbitrary, real_len, true);
        let mut oop_raw = vec![R::convert(0.0); real_len * batch_count];
        plan.backward(
            &arbitrary,
            &mut oop_raw,
            &mut oop_complex_line,
            &mut oop_scratch,
        )
        .unwrap();
        assert_real_close(&oop_raw, &expected_raw);
        plan.backward_in_place(&mut array, &mut workspace).unwrap();
        assert_real_close(array.real_view().unwrap(), &expected_raw);
    }

    #[test]
    fn in_place_f32_matches_oop_for_empty_single_and_many_batches() {
        for real_len in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13] {
            for batch_count in [0, 1, 3] {
                exercise_in_place::<f32>(real_len, batch_count);
            }
        }
    }

    #[test]
    fn in_place_f64_matches_oop_for_empty_single_and_many_batches() {
        for real_len in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13] {
            for batch_count in [0, 1, 3] {
                exercise_in_place::<f64>(real_len, batch_count);
            }
        }
    }

    #[test]
    fn in_place_preflight_is_atomic_and_poisoned_arrays_are_closed() {
        let plan = LocalR2cPlan::<f64>::new(6).unwrap();
        let mut array = plan.allocate_in_place(3).unwrap();
        array
            .real_view_mut()
            .unwrap()
            .copy_from_slice(&real_input::<f64>(6, 3));
        let mut workspace = plan.allocate_in_place_workspace().unwrap();
        plan.forward_in_place(&mut array, &mut workspace).unwrap();

        let mut complex_before = array.complex_view().unwrap().to_vec();
        complex_before[2 * plan.complex_len()].im = f64::NAN;
        array
            .complex_view_mut()
            .unwrap()
            .copy_from_slice(&complex_before);
        let array_pointer = array.storage.as_ptr();
        let array_before = bytemuck::cast_slice::<_, u8>(&array.storage).to_vec();
        let workspace_real_before = workspace.real_line.clone();
        let workspace_complex_before = workspace.complex_line.clone();
        let workspace_scratch_before = workspace.scratch.clone();
        assert_eq!(array.state(), R2cState::ComplexOutput);
        for backward in [false, true] {
            let result = if backward {
                plan.backward_in_place(&mut array, &mut workspace)
            } else {
                plan.inverse_in_place(&mut array, &mut workspace)
            };
            assert_eq!(
                result,
                Err(LocalR2cError::InvalidSpectrumEndpoint { batch: 2, index: 0 })
            );
            assert_eq!(array.state(), R2cState::ComplexOutput);
            assert_eq!(array.storage.as_ptr(), array_pointer);
            assert_eq!(
                bytemuck::cast_slice::<_, u8>(&array.storage),
                array_before.as_slice()
            );
            assert_eq!(workspace.real_line, workspace_real_before);
            assert_eq!(workspace.complex_line, workspace_complex_before);
            assert_eq!(workspace.scratch, workspace_scratch_before);
        }

        assert_eq!(
            plan.forward_in_place(&mut array, &mut workspace),
            Err(LocalR2cError::WrongState)
        );
        assert!(matches!(array.real_view(), Err(LocalR2cError::WrongState)));
        array.state = R2cState::Poisoned;
        assert_eq!(array.state(), R2cState::Poisoned);
        assert_eq!(array.complex_view(), Err(LocalR2cError::Poisoned));
        assert_eq!(
            plan.inverse_in_place(&mut array, &mut workspace),
            Err(LocalR2cError::Poisoned)
        );
    }

    struct PanickingForward {
        inner: Arc<dyn RealToComplex<f64>>,
    }

    impl RealToComplex<f64> for PanickingForward {
        fn process(
            &self,
            _input: &mut [f64],
            _output: &mut [Complex<f64>],
        ) -> Result<(), realfft::FftError> {
            panic!("injected local real FFT backend failure");
        }

        fn process_with_scratch(
            &self,
            _input: &mut [f64],
            _output: &mut [Complex<f64>],
            _scratch: &mut [Complex<f64>],
        ) -> Result<(), realfft::FftError> {
            panic!("injected local real FFT backend failure");
        }

        fn get_scratch_len(&self) -> usize {
            self.inner.get_scratch_len()
        }

        fn len(&self) -> usize {
            self.inner.len()
        }

        fn make_input_vec(&self) -> Vec<f64> {
            self.inner.make_input_vec()
        }

        fn make_output_vec(&self) -> Vec<Complex<f64>> {
            self.inner.make_output_vec()
        }

        fn make_scratch_vec(&self) -> Vec<Complex<f64>> {
            self.inner.make_scratch_vec()
        }
    }

    struct PanickingInverse {
        inner: Arc<dyn ComplexToReal<f64>>,
    }

    impl ComplexToReal<f64> for PanickingInverse {
        fn process(
            &self,
            _input: &mut [Complex<f64>],
            _output: &mut [f64],
        ) -> Result<(), realfft::FftError> {
            panic!("injected local real FFT backend failure");
        }

        fn process_with_scratch(
            &self,
            _input: &mut [Complex<f64>],
            _output: &mut [f64],
            _scratch: &mut [Complex<f64>],
        ) -> Result<(), realfft::FftError> {
            panic!("injected local real FFT backend failure");
        }

        fn get_scratch_len(&self) -> usize {
            self.inner.get_scratch_len()
        }

        fn len(&self) -> usize {
            self.inner.len()
        }

        fn make_input_vec(&self) -> Vec<Complex<f64>> {
            self.inner.make_input_vec()
        }

        fn make_output_vec(&self) -> Vec<f64> {
            self.inner.make_output_vec()
        }

        fn make_scratch_vec(&self) -> Vec<Complex<f64>> {
            self.inner.make_scratch_vec()
        }
    }

    fn assert_poisoned_views_and_retries(
        plan: &LocalR2cPlan<f64>,
        array: &mut LocalR2cInPlaceArray<f64>,
        workspace: &mut LocalR2cInPlaceWorkspace<f64>,
    ) {
        assert_eq!(array.state(), R2cState::Poisoned);
        assert!(matches!(array.real_view(), Err(LocalR2cError::Poisoned)));
        assert!(matches!(
            array.real_view_mut(),
            Err(LocalR2cError::Poisoned)
        ));
        assert!(matches!(array.complex_view(), Err(LocalR2cError::Poisoned)));
        assert!(matches!(
            array.complex_view_mut(),
            Err(LocalR2cError::Poisoned)
        ));

        let array_before = array.storage.clone();
        let workspace_before = (
            workspace.real_line.clone(),
            workspace.complex_line.clone(),
            workspace.scratch.clone(),
        );
        for direction in 0..3 {
            let result = match direction {
                0 => plan.forward_in_place(array, workspace),
                1 => plan.inverse_in_place(array, workspace),
                _ => plan.backward_in_place(array, workspace),
            };
            assert_eq!(result, Err(LocalR2cError::Poisoned));
            assert_eq!(array.state(), R2cState::Poisoned);
            assert_eq!(array.storage, array_before);
            assert_eq!(
                (
                    workspace.real_line.clone(),
                    workspace.complex_line.clone(),
                    workspace.scratch.clone()
                ),
                workspace_before
            );
        }
    }

    #[test]
    fn injected_forward_backend_panic_poisoning_closes_views_and_retries() {
        let mut plan = LocalR2cPlan::<f64>::new(5).unwrap();
        let mut array = plan.allocate_in_place(2).unwrap();
        array
            .real_view_mut()
            .unwrap()
            .copy_from_slice(&real_input::<f64>(5, 2));
        let mut workspace = plan.allocate_in_place_workspace().unwrap();
        let inner = Arc::clone(&plan.forward);
        plan.forward = Arc::new(PanickingForward { inner });

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = plan.forward_in_place(&mut array, &mut workspace);
        }));
        assert!(panic.is_err());
        assert_poisoned_views_and_retries(&plan, &mut array, &mut workspace);
    }

    #[test]
    fn injected_inverse_and_backward_backend_panics_poison_and_close_views() {
        let mut plan = LocalR2cPlan::<f64>::new(5).unwrap();
        let inner = Arc::clone(&plan.inverse);
        plan.inverse = Arc::new(PanickingInverse { inner });

        for backward in [false, true] {
            let mut array = plan.allocate_in_place(2).unwrap();
            array
                .real_view_mut()
                .unwrap()
                .copy_from_slice(&real_input::<f64>(5, 2));
            let mut workspace = plan.allocate_in_place_workspace().unwrap();
            plan.forward_in_place(&mut array, &mut workspace).unwrap();

            let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if backward {
                    let _ = plan.backward_in_place(&mut array, &mut workspace);
                } else {
                    let _ = plan.inverse_in_place(&mut array, &mut workspace);
                }
            }));
            assert!(panic.is_err());
            assert_poisoned_views_and_retries(&plan, &mut array, &mut workspace);
        }
    }

    fn assert_forward_preflight_unchanged(
        plan: &LocalR2cPlan<f64>,
        array: &mut LocalR2cInPlaceArray<f64>,
        workspace: &mut LocalR2cInPlaceWorkspace<f64>,
        expected: LocalR2cError,
    ) {
        let array_pointer = array.storage.as_ptr();
        let array_before = array.storage.clone();
        let workspace_before = (
            workspace.real_line.clone(),
            workspace.complex_line.clone(),
            workspace.scratch.clone(),
        );
        assert_eq!(plan.forward_in_place(array, workspace), Err(expected));
        assert_eq!(array.state(), R2cState::RealInput);
        assert_eq!(array.storage.as_ptr(), array_pointer);
        assert_eq!(array.storage, array_before);
        assert_eq!(
            (
                workspace.real_line.clone(),
                workspace.complex_line.clone(),
                workspace.scratch.clone()
            ),
            workspace_before
        );
    }

    #[test]
    fn in_place_rejects_foreign_workspace_and_short_resources_without_changes() {
        let plan = LocalR2cPlan::<f64>::new(5).unwrap();
        let foreign = LocalR2cPlan::<f64>::new(6).unwrap();
        assert!(plan.scratch_len() > 0);

        for case in 0..4 {
            let mut array = plan.allocate_in_place(2).unwrap();
            array
                .real_view_mut()
                .unwrap()
                .copy_from_slice(&real_input::<f64>(5, 2));
            let mut workspace = if case == 0 {
                foreign.allocate_in_place_workspace().unwrap()
            } else {
                plan.allocate_in_place_workspace().unwrap()
            };
            let expected = match case {
                0 => LocalR2cError::WorkspaceMismatch,
                1 => {
                    workspace.complex_line.truncate(plan.complex_len() - 1);
                    LocalR2cError::ComplexLineTooSmall {
                        required: plan.complex_len(),
                        actual: plan.complex_len() - 1,
                    }
                }
                2 => {
                    workspace.scratch.truncate(plan.scratch_len() - 1);
                    LocalR2cError::ScratchTooSmall {
                        required: plan.scratch_len(),
                        actual: plan.scratch_len() - 1,
                    }
                }
                _ => {
                    workspace.real_line.truncate(plan.real_len() - 1);
                    LocalR2cError::RealLineTooSmall {
                        required: plan.real_len(),
                        actual: plan.real_len() - 1,
                    }
                }
            };
            assert_forward_preflight_unchanged(&plan, &mut array, &mut workspace, expected);
        }
    }

    #[test]
    fn in_place_oversized_workspace_tails_are_untouched() {
        let plan = LocalR2cPlan::<f64>::new(5).unwrap();
        let mut array = plan.allocate_in_place(2).unwrap();
        array
            .real_view_mut()
            .unwrap()
            .copy_from_slice(&real_input::<f64>(5, 2));
        let mut workspace = plan.allocate_in_place_workspace().unwrap();
        workspace.real_line.extend([101.0, 102.0]);
        workspace
            .complex_line
            .extend([Complex::new(103.0, -104.0), Complex::new(105.0, -106.0)]);
        workspace
            .scratch
            .extend([Complex::new(107.0, -108.0), Complex::new(109.0, -110.0)]);
        let tails = (
            workspace.real_line[plan.real_len()..].to_vec(),
            workspace.complex_line[plan.complex_len()..].to_vec(),
            workspace.scratch[plan.scratch_len()..].to_vec(),
        );
        let assert_tails = |workspace: &LocalR2cInPlaceWorkspace<f64>| {
            assert_eq!(&workspace.real_line[plan.real_len()..], tails.0.as_slice());
            assert_eq!(
                &workspace.complex_line[plan.complex_len()..],
                tails.1.as_slice()
            );
            assert_eq!(&workspace.scratch[plan.scratch_len()..], tails.2.as_slice());
        };

        plan.forward_in_place(&mut array, &mut workspace).unwrap();
        assert_tails(&workspace);
        plan.inverse_in_place(&mut array, &mut workspace).unwrap();
        assert_tails(&workspace);
        plan.forward_in_place(&mut array, &mut workspace).unwrap();
        assert_tails(&workspace);
        plan.backward_in_place(&mut array, &mut workspace).unwrap();
        assert_tails(&workspace);
    }

    #[test]
    fn in_place_allocation_checks_overflow_before_allocating() {
        let plan = LocalR2cPlan::<f64>::new(5).unwrap();
        assert!(matches!(
            plan.allocate_in_place(usize::MAX),
            Err(LocalR2cError::LengthOverflow)
        ));
    }
}
