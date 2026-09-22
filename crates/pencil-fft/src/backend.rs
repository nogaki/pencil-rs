#[cfg(feature = "fftw")]
use std::sync::Arc;

#[cfg(feature = "fftw")]
use rustfft::{Fft, FftDirection};

#[cfg(feature = "fftw")]
use crate::FftReal;

/// The implementation used by a local or distributed plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    /// The portable RustFFT/RealFFT implementation.
    RustFft,
    /// The optional runtime-loaded FFTW implementation.
    Fftw,
}

/// Failure while initializing a selected backend.
#[derive(Debug, thiserror::Error)]
pub enum BackendInitError<E> {
    /// The local plan arguments were invalid.
    #[error("local validation failed: {0}")]
    Local(E),
    /// The native backend could not be loaded or planned.
    #[cfg(feature = "fftw")]
    #[error("native FFTW initialization failed: {0}")]
    Native(pencil_fftw::FftwError),
    /// A distributed peer rejected initialization during preflight.
    #[error("peer rejected backend initialization before execution")]
    PeerPreflight,
}

impl<E> From<E> for BackendInitError<E> {
    fn from(error: E) -> Self {
        Self::Local(error)
    }
}

#[cfg(feature = "fftw")]
pub(crate) trait FftwReal: FftReal + pencil_fftw::Real + crate::private::FftwBound {}

#[cfg(feature = "fftw")]
impl<T: FftReal + pencil_fftw::Real> FftwReal for T {}

#[cfg(feature = "fftw")]
pub(crate) fn c2c<R: FftwReal>(
    n: usize,
    direction: FftDirection,
    options: pencil_fftw::PlanOptions,
) -> Result<Arc<dyn Fft<R>>, pencil_fftw::FftwError> {
    #[cfg(all(test, feature = "distributed"))]
    crate::distributed::fftw_tests::before_factory()?;
    pencil_fftw::plan_c2c(n, direction, options)
}
