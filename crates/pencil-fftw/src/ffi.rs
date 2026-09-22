//! The entire unsafe boundary. FFTW new-array execution is thread-safe on shared
//! immutable plans with disjoint buffers. Planning/destruction/time limits are
//! serialized per precision, including across independently loaded tables.
//! Foreign FFTW callers must coordinate separately; FFTW has no time-limit getter.
use super::*;
use libloading::Library;
use std::{
    ffi::{CStr, c_int, c_uint, c_void},
    ptr::NonNull,
    sync::{Mutex, MutexGuard},
};

#[cfg(test)]
thread_local! {
    pub(crate) static COMPLEX_EXECUTIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

type Handle = *mut c_void;
type PlanComplex<R> =
    unsafe extern "C" fn(c_int, *mut Complex<R>, *mut Complex<R>, c_int, c_uint) -> Handle;
type PlanForward<R> = unsafe extern "C" fn(c_int, *mut R, *mut Complex<R>, c_uint) -> Handle;
type PlanInverse<R> = unsafe extern "C" fn(c_int, *mut Complex<R>, *mut R, c_uint) -> Handle;
type ExecComplex<R> = unsafe extern "C" fn(Handle, *mut Complex<R>, *mut Complex<R>);
type ExecForward<R> = unsafe extern "C" fn(Handle, *mut R, *mut Complex<R>);
type ExecInverse<R> = unsafe extern "C" fn(Handle, *mut Complex<R>, *mut R);
static SINGLE: Mutex<()> = Mutex::new(());
static DOUBLE: Mutex<()> = Mutex::new(());
fn lock<R: Real>() -> MutexGuard<'static, ()> {
    (if R::SINGLE { &SINGLE } else { &DOUBLE })
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}
struct Table<R> {
    pc: PlanComplex<R>,
    pf: PlanForward<R>,
    pi: PlanInverse<R>,
    ec: ExecComplex<R>,
    ef: ExecForward<R>,
    ei: ExecInverse<R>,
    destroy: unsafe extern "C" fn(Handle),
    limit: unsafe extern "C" fn(f64),
    version: String,
    _library: Library,
}
impl<R: Real> Table<R> {
    fn load() -> Result<Self, FftwError> {
        Self::load_from(if R::SINGLE {
            "libfftw3f.so.3"
        } else {
            "libfftw3.so.3"
        })
    }
    fn load_from(path: &str) -> Result<Self, FftwError> {
        // SAFETY: loading the explicitly selected native FFTW ABI. No user code
        // executes through symbols until all required symbols have been checked.
        let library = unsafe { Library::new(path) }.map_err(|e| FftwError::Load(e.to_string()))?;
        let prefix = if R::SINGLE { "fftwf_" } else { "fftw_" };
        macro_rules! symbol {
            ($name:literal, $ty:ty) => {{
                // SAFETY: signatures are exactly fftw3.h for the sealed precision.
                unsafe {
                    *library
                        .get::<$ty>(format!("{prefix}{}\0", $name).as_bytes())
                        .map_err(|e| FftwError::Symbol(e.to_string()))?
                }
            }};
        }
        let version = symbol!("version", *const std::ffi::c_char);
        // SAFETY: FFTW exports version as a static NUL-terminated char array.
        let version = unsafe { CStr::from_ptr(version) }
            .to_string_lossy()
            .into_owned();
        Ok(Self {
            pc: symbol!("plan_dft_1d", PlanComplex<R>),
            pf: symbol!("plan_dft_r2c_1d", PlanForward<R>),
            pi: symbol!("plan_dft_c2r_1d", PlanInverse<R>),
            ec: symbol!("execute_dft", ExecComplex<R>),
            ef: symbol!("execute_dft_r2c", ExecForward<R>),
            ei: symbol!("execute_dft_c2r", ExecInverse<R>),
            destroy: symbol!("destroy_plan", unsafe extern "C" fn(Handle)),
            limit: symbol!("set_timelimit", unsafe extern "C" fn(f64)),
            version,
            _library: library,
        })
    }
}
pub(crate) fn version<R: Real>() -> Result<String, FftwError> {
    Ok(Table::<R>::load()?.version)
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Complex(FftDirection, bool),
    Forward,
    Inverse,
}
pub(crate) struct Plan<R: Real> {
    handle: NonNull<c_void>,
    table: Table<R>,
    n: usize,
    kind: Kind,
}
// SAFETY: only new-array execution is exposed, all dimensions and alias modes
// checked below. No mutable plan state is accessed during execution. The library
// outlives the handle; destruction and all planner state are locked.
unsafe impl<R: Real> Send for Plan<R> {}
unsafe impl<R: Real> Sync for Plan<R> {}
struct Reset<'a>(&'a unsafe extern "C" fn(f64));
impl Drop for Reset<'_> {
    fn drop(&mut self) {
        // SAFETY: the borrowed function pointer's library remains live.
        unsafe { (self.0)(-1.0) }
    }
}
impl<R: Real> Plan<R> {
    pub(crate) fn new(n: usize, kind: Kind, options: PlanOptions) -> Result<Self, FftwError> {
        let native = checked_len::<R>(n)?;
        let table = Table::load()?;
        // Initialized, private buffers: destructive planning never sees caller data.
        let mut a = zeros::<Complex<R>>(n)?;
        let mut b = zeros::<Complex<R>>(n)?;
        let mut r = zeros::<R>(n)?;
        let _guard = lock::<R>();
        let _reset = Reset(&table.limit);
        // SAFETY: dimensions fit c_int and all arrays hold at least n initialized
        // elements. Complex<T> has repr(C), two consecutive T fields (num-complex).
        // UNALIGNED removes SIMD alignment constraints, NOT alias constraints.
        let handle = unsafe {
            (table.limit)(options.time_limit.map_or(-1.0, |t| t.as_secs_f64()));
            match kind {
                Kind::Complex(d, ip) => (table.pc)(
                    native,
                    a.as_mut_ptr(),
                    if ip { a.as_mut_ptr() } else { b.as_mut_ptr() },
                    if d == FftDirection::Forward { -1 } else { 1 },
                    options.flags() | 16,
                ),
                Kind::Forward => {
                    (table.pf)(native, r.as_mut_ptr(), b.as_mut_ptr(), options.flags())
                }
                Kind::Inverse => {
                    (table.pi)(native, a.as_mut_ptr(), r.as_mut_ptr(), options.flags())
                }
            }
        };
        drop(_reset);
        let handle = NonNull::new(handle).ok_or(FftwError::NullPlan)?;
        Ok(Self {
            handle,
            table,
            n,
            kind,
        })
    }
    pub(crate) fn inplace(&self, b: &mut [Complex<R>]) {
        assert!(matches!(self.kind, Kind::Complex(_, true)));
        assert_eq!(b.len(), self.n);
        #[cfg(test)]
        COMPLEX_EXECUTIONS.with(|count| count.set(count.get() + 1));
        unsafe { (self.table.ec)(self.handle.as_ptr(), b.as_mut_ptr(), b.as_mut_ptr()) }
    }
    pub(crate) fn complex(&self, i: &[Complex<R>], o: &mut [Complex<R>]) {
        assert!(matches!(self.kind, Kind::Complex(_, false)));
        assert_eq!(i.len(), self.n);
        assert_eq!(o.len(), self.n);
        #[cfg(test)]
        COMPLEX_EXECUTIONS.with(|count| count.set(count.get() + 1));
        // PRESERVE_INPUT explicitly requested for this OOP plan. Rust borrows
        // guarantee disjoint arrays; FFTW accepts a mutable pointer but won't write i.
        unsafe { (self.table.ec)(self.handle.as_ptr(), i.as_ptr().cast_mut(), o.as_mut_ptr()) }
    }
    pub(crate) fn forward(&self, i: &mut [R], o: &mut [Complex<R>]) {
        assert_eq!(self.kind, Kind::Forward);
        assert_eq!(i.len(), self.n);
        assert_eq!(o.len(), self.n / 2 + 1);
        unsafe { (self.table.ef)(self.handle.as_ptr(), i.as_mut_ptr(), o.as_mut_ptr()) }
    }
    pub(crate) fn inverse(&self, i: &mut [Complex<R>], o: &mut [R]) {
        assert_eq!(self.kind, Kind::Inverse);
        assert_eq!(i.len(), self.n / 2 + 1);
        assert_eq!(o.len(), self.n);
        unsafe { (self.table.ei)(self.handle.as_ptr(), i.as_mut_ptr(), o.as_mut_ptr()) }
    }
}
impl<R: Real> Drop for Plan<R> {
    fn drop(&mut self) {
        let _guard = lock::<R>();
        unsafe { (self.table.destroy)(self.handle.as_ptr()) }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reset_on_success_error_and_unwind() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static RESETS: AtomicUsize = AtomicUsize::new(0);
        unsafe extern "C" fn record(value: f64) {
            if value == -1.0 {
                RESETS.fetch_add(1, Ordering::SeqCst);
            }
        }
        let callback: unsafe extern "C" fn(f64) = record;
        {
            let _reset = Reset(&callback);
        }
        let result = (|| -> Result<(), FftwError> {
            let _reset = Reset(&callback);
            NonNull::<c_void>::new(std::ptr::null_mut()).ok_or(FftwError::NullPlan)?;
            Ok(())
        })();
        assert!(matches!(result, Err(FftwError::NullPlan)));
        assert!(
            std::panic::catch_unwind(|| {
                let _reset = Reset(&callback);
                panic!("planning unwind");
            })
            .is_err()
        );
        assert_eq!(RESETS.load(Ordering::SeqCst), 3);
        assert_eq!(std::mem::size_of::<c_int>(), 4);
        assert_eq!(
            std::mem::size_of::<Complex<f32>>(),
            2 * std::mem::size_of::<f32>()
        );
        assert_eq!(
            std::mem::size_of::<Complex<f64>>(),
            2 * std::mem::size_of::<f64>()
        );
    }
    #[test]
    #[ignore = "requires both native FFTW runtimes"]
    fn native_partial_construction_cleanup() {
        fn check<R: Real>() {
            // A later construction error must drop the already created handle,
            // then allow subsequent planning/execution (no stale planner lock).
            let result = (|| -> Result<(), FftwError> {
                let _first = Plan::<R>::new(5, Kind::Forward, PlanOptions::default())?;
                Plan::<R>::new(0, Kind::Inverse, PlanOptions::default())?;
                Ok(())
            })();
            assert!(matches!(result, Err(FftwError::InvalidOptions(_))));
            let p = Plan::<R>::new(5, Kind::Forward, PlanOptions::default()).unwrap();
            // Wrong execution kind must fail before touching either array.
            let mut input = vec![Complex::<R>::default(); 5];
            let before = input.clone();
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| p.inplace(&mut input)))
                    .is_err()
            );
            assert_eq!(input, before);
            p.forward(&mut [R::zero(); 5], &mut [Complex::default(); 3]);
        }
        check::<f32>();
        check::<f64>();
    }
    #[test]
    fn missing_library_and_symbol() {
        assert!(matches!(
            Table::<f64>::load_from("/nonexistent/pencil-fftw.so"),
            Err(FftwError::Load(_))
        ));
        #[cfg(target_os = "linux")]
        assert!(matches!(
            Table::<f64>::load_from("libc.so.6"),
            Err(FftwError::Symbol(_))
        ));
    }
}
