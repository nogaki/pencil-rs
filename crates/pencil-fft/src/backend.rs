#[cfg(feature = "fftw")]
use std::sync::Arc;

#[cfg(feature = "fftw")]
use realfft::{ComplexToReal, RealToComplex};
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

/// The backend selection held by a local plan.
///
/// The FFTW options are only meaningful for [`BackendKind::Fftw`], so the kind
/// and its options stay together instead of being stored as separate fields.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LocalBackend {
    kind: BackendKind,
    #[cfg(feature = "fftw")]
    options: Option<pencil_fftw::PlanOptions>,
}

impl LocalBackend {
    /// The portable RustFFT/RealFFT backend.
    pub(crate) const RUST: Self = Self {
        kind: BackendKind::RustFft,
        #[cfg(feature = "fftw")]
        options: None,
    };

    /// The native FFTW backend planned with `options`.
    #[cfg(feature = "fftw")]
    pub(crate) const fn fftw(options: pencil_fftw::PlanOptions) -> Self {
        Self {
            kind: BackendKind::Fftw,
            options: Some(options),
        }
    }

    /// Returns the selected backend kind.
    pub(crate) const fn kind(self) -> BackendKind {
        self.kind
    }

    /// Returns the FFTW options, when this backend uses FFTW.
    #[cfg(feature = "fftw")]
    pub(crate) const fn options(self) -> Option<pencil_fftw::PlanOptions> {
        self.options
    }
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

#[cfg(feature = "distributed")]
impl<E> BackendInitError<E> {
    pub(crate) fn map_local<F>(self, map: impl FnOnce(E) -> F) -> BackendInitError<F> {
        match self {
            Self::Local(error) => BackendInitError::Local(map(error)),
            #[cfg(feature = "fftw")]
            Self::Native(error) => BackendInitError::Native(error),
            Self::PeerPreflight => BackendInitError::PeerPreflight,
        }
    }
}

impl<E> From<E> for BackendInitError<E> {
    fn from(error: E) -> Self {
        Self::Local(error)
    }
}

#[cfg(feature = "fftw")]
pub(crate) fn c2c<R: FftReal>(
    n: usize,
    direction: FftDirection,
    options: pencil_fftw::PlanOptions,
) -> Result<Arc<dyn Fft<R>>, pencil_fftw::FftwError> {
    #[cfg(all(test, feature = "distributed"))]
    crate::distributed::fftw_tests::before_factory()?;
    pencil_fftw::plan_c2c(n, direction, options)
}

#[cfg(feature = "fftw")]
pub(crate) fn r2r<R: FftReal>(
    n: usize,
    kind: pencil_fftw::R2rKind,
    options: pencil_fftw::PlanOptions,
) -> Result<Arc<pencil_fftw::R2rPlan<R>>, pencil_fftw::FftwError> {
    #[cfg(all(test, feature = "distributed"))]
    crate::distributed::fftw_tests::before_factory()?;
    pencil_fftw::plan_r2r(n, kind, options)
}

/// The forward and inverse native plans of a real-to-complex transform.
#[cfg(feature = "fftw")]
type R2cPlans<R> = (Arc<dyn RealToComplex<R>>, Arc<dyn ComplexToReal<R>>);

#[cfg(feature = "fftw")]
pub(crate) fn r2c<R: FftReal>(
    real_len: usize,
    options: pencil_fftw::PlanOptions,
) -> Result<R2cPlans<R>, pencil_fftw::FftwError> {
    #[cfg(all(test, feature = "distributed"))]
    crate::distributed::fftw_tests::before_factory()?;
    Ok((
        pencil_fftw::plan_r2c(real_len, options)?,
        pencil_fftw::plan_c2r(real_len, options)?,
    ))
}

#[cfg(all(test, feature = "distributed"))]
mod tests {
    use super::BackendInitError;

    #[test]
    fn map_local_preserves_failure_kind() {
        assert!(matches!(
            BackendInitError::Local("bad").map_local(str::len),
            BackendInitError::Local(3)
        ));
        let unexpected = |_: &str| -> usize { panic!("not a local failure") };
        assert!(matches!(
            BackendInitError::PeerPreflight.map_local(unexpected),
            BackendInitError::PeerPreflight
        ));
        #[cfg(feature = "fftw")]
        assert!(matches!(
            BackendInitError::Native(pencil_fftw::FftwError::InvalidOptions("native"))
                .map_local(unexpected),
            BackendInitError::Native(pencil_fftw::FftwError::InvalidOptions("native"))
        ));
    }
}
