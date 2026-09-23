#![doc = include_str!("../README.md")]
#![deny(unsafe_code)]
#[allow(unsafe_code)]
mod ffi;

use num_complex::Complex;
use realfft::{ComplexToReal, FftError, RealToComplex};
use rustfft::{Direction, Fft, FftDirection, FftNum, Length};
use std::{sync::Arc, time::Duration};

mod sealed {
    pub trait Sealed {
        const SINGLE: bool;
    }
    impl Sealed for f32 {
        const SINGLE: bool = true;
    }
    impl Sealed for f64 {
        const SINGLE: bool = false;
    }
}
/// Native FFTW scalar, sealed to f32 and f64.
pub trait Real: FftNum + Default + sealed::Sealed {}
impl Real for f32 {}
impl Real for f64 {}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PlanningRigor {
    #[default]
    Estimate,
    Measure,
    Patient,
    Exhaustive,
}
#[derive(Clone, Copy, Debug)]
pub struct PlanOptions {
    rigor: PlanningRigor,
    time_limit: Option<Duration>,
    threads: i32,
}
impl Default for PlanOptions {
    fn default() -> Self {
        Self {
            rigor: PlanningRigor::Estimate,
            time_limit: None,
            threads: 1,
        }
    }
}
impl PlanOptions {
    pub fn new(rigor: PlanningRigor, time_limit: Option<Duration>) -> Result<Self, FftwError> {
        if time_limit.is_some_and(|t| t.is_zero() || !t.as_secs_f64().is_finite()) {
            return Err(FftwError::InvalidOptions(
                "time limit must be positive and finite",
            ));
        }
        Ok(Self {
            rigor,
            time_limit,
            threads: 1,
        })
    }
    /// Request CPU planning threads; this is not a utilization guarantee.
    pub fn with_threads(mut self, count: usize) -> Result<Self, FftwError> {
        self.threads =
            i32::try_from(count)
                .ok()
                .filter(|&n| n > 0)
                .ok_or(FftwError::InvalidOptions(
                    "thread count must be positive and fit c_int",
                ))?;
        Ok(self)
    }
    pub fn requested_threads(self) -> usize {
        self.threads as usize
    }
    pub fn rigor(self) -> PlanningRigor {
        self.rigor
    }
    pub fn time_limit(self) -> Option<Duration> {
        self.time_limit
    }
    fn flags(self) -> u32 {
        2 | match self.rigor {
            PlanningRigor::Estimate => 64,
            PlanningRigor::Measure => 0,
            PlanningRigor::Patient => 32,
            PlanningRigor::Exhaustive => 8,
        }
    }
}
#[derive(Debug, thiserror::Error)]
pub enum FftwError {
    #[error("FFTW load failed: {0}")]
    Load(String),
    #[error("FFTW symbol failed: {0}")]
    Symbol(String),
    #[error("invalid FFTW options: {0}")]
    InvalidOptions(&'static str),
    #[error("FFTW length or allocation size overflow")]
    Overflow,
    #[error("FFTW returned a null plan")]
    NullPlan,
    #[error("FFTW planning buffer allocation failed: {0}")]
    Allocation(#[from] std::collections::TryReserveError),
}
/// Wisdom errors are separate to preserve the closed backend error API.
#[derive(Debug, thiserror::Error)]
pub enum WisdomError {
    #[error("wisdom contains an interior NUL")]
    InteriorNul,
    #[error("FFTW rejected wisdom")]
    InvalidWisdom,
    #[error("FFTW returned null wisdom")]
    NullExport,
    #[error("FFTW exported non-UTF-8 wisdom")]
    InvalidUtf8,
    #[error(transparent)]
    Backend(#[from] FftwError),
}
/// Import native wisdom. Rejected input is not guaranteed to leave wisdom unchanged.
pub fn import_wisdom<R: Real>(wisdom: &str) -> Result<(), WisdomError> {
    ffi::import_wisdom::<R>(wisdom)
}
/// Export an owned copy of this precision's native wisdom.
pub fn export_wisdom<R: Real>() -> Result<String, WisdomError> {
    ffi::export_wisdom::<R>()
}
/// Forget cached wisdom, without invalidating existing plans.
pub fn forget_wisdom<R: Real>() -> Result<(), WisdomError> {
    ffi::forget_wisdom::<R>()
}
fn checked_len<R: Real>(n: usize) -> Result<i32, FftwError> {
    if n == 0 {
        return Err(FftwError::InvalidOptions("length must be nonzero"));
    }
    let native = i32::try_from(n).map_err(|_| FftwError::Overflow)?;
    if n.checked_mul(std::mem::size_of::<Complex<R>>())
        .filter(|&b| b <= isize::MAX as usize)
        .is_none()
    {
        return Err(FftwError::Overflow);
    }
    Ok(native)
}
fn zeros<T: Default + Clone>(n: usize) -> Result<Vec<T>, FftwError> {
    let mut v = Vec::new();
    v.try_reserve_exact(n)?;
    v.resize(n, T::default());
    Ok(v)
}
/// Query the loaded runtime (not the installed header) for one precision.
pub fn runtime_version<R: Real>() -> Result<String, FftwError> {
    ffi::version::<R>()
}

/// Plans distinct in-place and out-of-place native transforms. Unnormalized.
pub fn plan_c2c<R: Real>(
    n: usize,
    direction: FftDirection,
    options: PlanOptions,
) -> Result<Arc<dyn Fft<R>>, FftwError> {
    checked_len::<R>(n)?;
    Ok(Arc::new(ComplexPlan {
        n,
        direction,
        oop: ffi::Plan::new(n, ffi::Kind::Complex(direction, false), options)?,
        ip: ffi::Plan::new(n, ffi::Kind::Complex(direction, true), options)?,
    }))
}
pub fn plan_r2c<R: Real>(
    n: usize,
    options: PlanOptions,
) -> Result<Arc<dyn RealToComplex<R>>, FftwError> {
    Ok(Arc::new(RealPlan {
        n,
        plan: ffi::Plan::new(n, ffi::Kind::Forward, options)?,
    }))
}
pub fn plan_c2r<R: Real>(
    n: usize,
    options: PlanOptions,
) -> Result<Arc<dyn ComplexToReal<R>>, FftwError> {
    Ok(Arc::new(RealPlan {
        n,
        plan: ffi::Plan::new(n, ffi::Kind::Inverse, options)?,
    }))
}
// Native execution requires no caller scratch. The trait accepts any extra
// scratch capacity; it and its tail remain untouched.
struct ComplexPlan<R: Real> {
    n: usize,
    direction: FftDirection,
    oop: ffi::Plan<R>,
    ip: ffi::Plan<R>,
}
impl<R: Real> Length for ComplexPlan<R> {
    fn len(&self) -> usize {
        self.n
    }
}
impl<R: Real> Direction for ComplexPlan<R> {
    fn fft_direction(&self) -> FftDirection {
        self.direction
    }
}
impl<R: Real> Fft<R> for ComplexPlan<R> {
    fn process_with_scratch(&self, buffer: &mut [Complex<R>], _: &mut [Complex<R>]) {
        assert!(buffer.len() >= self.n, "buffer shorter than plan length");
        assert_eq!(buffer.len() % self.n, 0, "invalid batch length");
        for b in buffer.chunks_exact_mut(self.n) {
            self.ip.inplace(b);
        }
    }
    fn process_outofplace_with_scratch(
        &self,
        input: &mut [Complex<R>],
        output: &mut [Complex<R>],
        scratch: &mut [Complex<R>],
    ) {
        self.process_immutable_with_scratch(input, output, scratch);
    }
    fn process_immutable_with_scratch(
        &self,
        input: &[Complex<R>],
        output: &mut [Complex<R>],
        _: &mut [Complex<R>],
    ) {
        assert!(input.len() >= self.n, "input shorter than plan length");
        assert_eq!(input.len(), output.len(), "output length");
        assert_eq!(input.len() % self.n, 0, "invalid batch length");
        for (i, o) in input
            .chunks_exact(self.n)
            .zip(output.chunks_exact_mut(self.n))
        {
            self.oop.complex(i, o);
        }
    }
    fn get_inplace_scratch_len(&self) -> usize {
        0
    }
    fn get_outofplace_scratch_len(&self) -> usize {
        0
    }
    fn get_immutable_scratch_len(&self) -> usize {
        0
    }
}
struct RealPlan<R: Real> {
    n: usize,
    plan: ffi::Plan<R>,
}
fn preflight(i: usize, o: usize, ni: usize, no: usize) -> Result<(), FftError> {
    if i != ni {
        return Err(FftError::InputBuffer(ni, i));
    }
    if o != no {
        return Err(FftError::OutputBuffer(no, o));
    }
    Ok(())
}
impl<R: Real> RealToComplex<R> for RealPlan<R> {
    fn process(&self, i: &mut [R], o: &mut [Complex<R>]) -> Result<(), FftError> {
        RealToComplex::process_with_scratch(self, i, o, &mut [])
    }
    fn process_with_scratch(
        &self,
        i: &mut [R],
        o: &mut [Complex<R>],
        _: &mut [Complex<R>],
    ) -> Result<(), FftError> {
        preflight(i.len(), o.len(), self.n, self.n / 2 + 1)?;
        self.plan.forward(i, o);
        Ok(())
    }
    fn len(&self) -> usize {
        self.n
    }
    fn get_scratch_len(&self) -> usize {
        0
    }
    fn make_input_vec(&self) -> Vec<R> {
        vec![R::zero(); self.n]
    }
    fn make_output_vec(&self) -> Vec<Complex<R>> {
        vec![Complex::default(); self.n / 2 + 1]
    }
    fn make_scratch_vec(&self) -> Vec<Complex<R>> {
        Vec::new()
    }
}
impl<R: Real> ComplexToReal<R> for RealPlan<R> {
    fn process(&self, i: &mut [Complex<R>], o: &mut [R]) -> Result<(), FftError> {
        ComplexToReal::process_with_scratch(self, i, o, &mut [])
    }
    fn process_with_scratch(
        &self,
        i: &mut [Complex<R>],
        o: &mut [R],
        _: &mut [Complex<R>],
    ) -> Result<(), FftError> {
        preflight(i.len(), o.len(), self.n / 2 + 1, self.n)?;
        let first = i[0].im != R::zero();
        let last = self.n % 2 == 0 && i[self.n / 2].im != R::zero();
        self.plan.inverse(i, o);
        if first || last {
            Err(FftError::InputValues(first, last))
        } else {
            Ok(())
        }
    }
    fn len(&self) -> usize {
        self.n
    }
    fn get_scratch_len(&self) -> usize {
        0
    }
    fn make_input_vec(&self) -> Vec<Complex<R>> {
        vec![Complex::default(); self.n / 2 + 1]
    }
    fn make_output_vec(&self) -> Vec<R> {
        vec![R::zero(); self.n]
    }
    fn make_scratch_vec(&self) -> Vec<Complex<R>> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests;
