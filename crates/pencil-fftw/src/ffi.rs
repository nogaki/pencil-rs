//! The entire unsafe boundary. FFTW new-array execution is thread-safe on shared
//! immutable plans with disjoint buffers. Planning/destruction/time limits are
//! serialized per precision, including across independently loaded tables.
//! Foreign FFTW callers must coordinate separately; FFTW has no time-limit getter.
use super::*;
use libloading::Library;
use std::{
    ffi::{CStr, CString, c_char, c_int, c_uint, c_void},
    ptr::NonNull,
    sync::{Mutex, MutexGuard},
};

#[cfg(test)]
thread_local! {
    pub(crate) static COMPLEX_EXECUTIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static TRACE: std::cell::RefCell<Vec<&'static str>> = const { std::cell::RefCell::new(Vec::new()) };
    static FLAG_TRACE: std::cell::RefCell<Vec<u32>> = const { std::cell::RefCell::new(Vec::new()) };
    // Observes completed API calls, not FFTW's (unqueryable) internal state.
    static LIMIT_TRACE: std::cell::RefCell<Vec<f64>> = const { std::cell::RefCell::new(Vec::new()) };
    static PANIC_AFTER_LIMIT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
#[cfg(test)]
fn trace(event: &'static str) {
    TRACE.with(|events| events.borrow_mut().push(event));
}
#[cfg(test)]
fn trace_flags(flags: u32) {
    FLAG_TRACE.with(|events| events.borrow_mut().push(flags));
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
    import: unsafe extern "C" fn(*const c_char) -> c_int,
    export: unsafe extern "C" fn() -> *mut c_char,
    forget: unsafe extern "C" fn(),
    free: unsafe extern "C" fn(*mut c_void),
    _library: Arc<Library>,
}
impl<R: Real> Table<R> {
    fn load() -> Result<Self, FftwError> {
        static SINGLE_LIBRARY: Mutex<Option<Arc<Library>>> = Mutex::new(None);
        static DOUBLE_LIBRARY: Mutex<Option<Arc<Library>>> = Mutex::new(None);
        let mut cache = (if R::SINGLE {
            &SINGLE_LIBRARY
        } else {
            &DOUBLE_LIBRARY
        })
        .lock()
        .unwrap_or_else(|e| e.into_inner());
        if let Some(library) = cache.as_ref() {
            return Self::from_library(library.clone());
        }
        let table = Self::load_from(if R::SINGLE {
            "libfftw3f.so.3"
        } else {
            "libfftw3.so.3"
        })?;
        // ponytail: bounded to one base library per precision, process lifetime.
        // Keeping the library resident preserves wisdom when every plan drops.
        *cache = Some(table._library.clone());
        Ok(table)
    }
    fn load_from(path: &str) -> Result<Self, FftwError> {
        // SAFETY: loading the explicitly selected native FFTW ABI. No user code
        // executes through symbols until all required symbols have been checked.
        let library = unsafe { Library::new(path) }.map_err(|e| FftwError::Load(e.to_string()))?;
        Self::from_library(Arc::new(library))
    }
    fn from_library(library: Arc<Library>) -> Result<Self, FftwError> {
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
            import: symbol!(
                "import_wisdom_from_string",
                unsafe extern "C" fn(*const c_char) -> c_int
            ),
            export: symbol!(
                "export_wisdom_to_string",
                unsafe extern "C" fn() -> *mut c_char
            ),
            forget: symbol!("forget_wisdom", unsafe extern "C" fn()),
            free: symbol!("free", unsafe extern "C" fn(*mut c_void)),
            version,
            _library: library,
        })
    }
}
// Accessed only while the corresponding planner lock is held. The separate
// mutex stores the callback without unsafe mutable statics.
type ThreadSetter = unsafe extern "C" fn(c_int);
struct ThreadState {
    result: Option<Result<ThreadSetter, FftwError>>,
    init_attempted: bool,
}
static SINGLE_THREADS: Mutex<ThreadState> = Mutex::new(ThreadState {
    result: None,
    init_attempted: false,
});
static DOUBLE_THREADS: Mutex<ThreadState> = Mutex::new(ThreadState {
    result: None,
    init_attempted: false,
});
struct ThreadReset(Option<ThreadSetter>);
impl Drop for ThreadReset {
    fn drop(&mut self) {
        if let Some(set) = self.0 {
            // SAFETY: thread library is pinned, caller still holds planner lock.
            #[cfg(test)]
            trace("reset1");
            unsafe { set(1) }
        }
    }
}
fn load_threads<R: Real>(
    path: &str,
    base: Option<&Table<R>>,
    init_attempted: &mut bool,
) -> Result<ThreadSetter, FftwError> {
    // SAFETY: explicitly selected FFTW pthread ABI.
    let library = unsafe { Library::new(path) }.map_err(|e| FftwError::Load(e.to_string()))?;
    let prefix = if R::SINGLE { "fftwf_" } else { "fftw_" };
    // SAFETY: exact FFTW signatures, both checked before initialization.
    let (init, set) = unsafe {
        (
            *library
                .get::<unsafe extern "C" fn() -> c_int>(
                    format!("{prefix}init_threads\0").as_bytes(),
                )
                .map_err(|e| FftwError::Symbol(e.to_string()))?,
            *library
                .get::<ThreadSetter>(format!("{prefix}plan_with_nthreads\0").as_bytes())
                .map_err(|e| FftwError::Symbol(e.to_string()))?,
        )
    };
    // Resolve through the thread handle's dependency scope, not a version string.
    // Reject a different loaded base instance before registering any callbacks.
    let base = base.ok_or_else(|| FftwError::Load("missing pinned FFTW base".into()))?;
    let dependency = Table::<R>::from_library(Arc::new(library))?;
    if dependency.pc as usize != base.pc as usize
        || dependency.limit as usize != base.limit as usize
        || dependency.import as usize != base.import as usize
        || dependency.forget as usize != base.forget as usize
    {
        return Err(FftwError::Load(
            "FFTW thread library uses a different base instance".into(),
        ));
    }
    // ponytail: one thread library per precision stays resident for registered
    // native callbacks. Never cleanup_threads (nor cleanup) behind live plans.
    let _ = Box::leak(Box::new(dependency));
    *init_attempted = true;
    if unsafe { init() } == 0 {
        return Err(FftwError::Load("FFTW init_threads returned failure".into()));
    }
    #[cfg(test)]
    {
        trace("init");
        if std::env::var_os("PENCIL_FFTW_TEST_INIT_FAILURE").is_some() {
            return Err(FftwError::Load(
                "injected failure after successful native init".into(),
            ));
        }
    }
    Ok(set)
}
fn thread_setter<R: Real>(
    table: &Table<R>,
    count: c_int,
) -> Result<Option<ThreadSetter>, FftwError> {
    let mut slot = (if R::SINGLE {
        &SINGLE_THREADS
    } else {
        &DOUBLE_THREADS
    })
    .lock()
    .unwrap_or_else(|e| e.into_inner());
    if slot.result.is_none() {
        let path = if R::SINGLE {
            "libfftw3f_threads.so.3"
        } else {
            "libfftw3_threads.so.3"
        };
        // Test-only override, read only in an isolated child process.
        #[cfg(test)]
        let override_path = std::env::var("PENCIL_FFTW_TEST_THREADS_LIBRARY").ok();
        #[cfg(test)]
        let path = override_path.as_deref().unwrap_or(path);
        slot.result = Some(load_threads::<R>(
            path,
            Some(table),
            &mut slot.init_attempted,
        ));
    }
    let setter = match slot.result.as_ref() {
        // A loader/symbol failure never called native init: serial stays base-only.
        Some(Err(_)) if count == 1 && !slot.init_attempted => None,
        Some(Ok(set)) => Some(*set),
        Some(Err(FftwError::Symbol(message))) => return Err(FftwError::Symbol(message.clone())),
        Some(Err(FftwError::Load(message))) => return Err(FftwError::Load(message.clone())),
        Some(Err(_)) => unreachable!("thread loader returns only Load or Symbol"),
        None => None,
    };
    Ok(setter)
}
fn set_threads<R: Real>(table: &Table<R>, count: c_int) -> Result<ThreadReset, FftwError> {
    let setter = thread_setter(table, count)?;
    let reset = ThreadReset(setter);
    if let Some(set) = setter {
        // SAFETY: initialized library, positive validated count, planner lock held.
        #[cfg(test)]
        trace(if count == 1 { "set1" } else { "set>1" });
        unsafe { set(count) }
    }
    Ok(reset)
}
pub(crate) fn import_wisdom<R: Real>(text: &str) -> Result<(), WisdomError> {
    if text.as_bytes().contains(&0) {
        return Err(WisdomError::InteriorNul);
    }
    let size = text
        .len()
        .checked_add(1)
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or(FftwError::Overflow)?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(size).map_err(FftwError::from)?;
    bytes.extend_from_slice(text.as_bytes());
    bytes.push(0);
    let text = CString::from_vec_with_nul(bytes).map_err(|_| WisdomError::InteriorNul)?;
    let table = Table::<R>::load()?;
    let _guard = lock::<R>();
    thread_setter(&table, 1)?;
    #[cfg(test)]
    trace("import");
    // SAFETY: owned NUL-terminated input; native function does not retain it.
    if unsafe { (table.import)(text.as_ptr()) } == 0 {
        return Err(WisdomError::InvalidWisdom);
    }
    Ok(())
}
pub(crate) fn export_wisdom<R: Real>() -> Result<String, WisdomError> {
    let table = Table::<R>::load()?;
    let _guard = lock::<R>();
    thread_setter(&table, 1)?;
    #[cfg(test)]
    trace("export");
    let pointer = NonNull::new(unsafe { (table.export)() }).ok_or(WisdomError::NullExport)?;
    struct Export(NonNull<c_char>, unsafe extern "C" fn(*mut c_void));
    impl Drop for Export {
        fn drop(&mut self) {
            unsafe { (self.1)(self.0.as_ptr().cast()) }
        }
    }
    // Install the matching allocator guard before any copying/allocation.
    let owned = Export(pointer, table.free);
    // SAFETY: FFTW promises a NUL-terminated allocated string.
    let text = unsafe { CStr::from_ptr(owned.0.as_ptr()) }
        .to_str()
        .map_err(|_| WisdomError::InvalidUtf8)?;
    let mut result = String::new();
    result
        .try_reserve_exact(text.len())
        .map_err(FftwError::from)?;
    result.push_str(text);
    Ok(result)
}
pub(crate) fn forget_wisdom<R: Real>() -> Result<(), WisdomError> {
    let table = Table::<R>::load()?;
    let _guard = lock::<R>();
    thread_setter(&table, 1)?;
    #[cfg(test)]
    trace("forget");
    unsafe { (table.forget)() };
    Ok(())
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
        #[cfg(test)]
        LIMIT_TRACE.with(|events| events.borrow_mut().push(-1.0));
    }
}
impl<R: Real> Plan<R> {
    pub(crate) fn new(n: usize, kind: Kind, options: PlanOptions) -> Result<Self, FftwError> {
        Self::new_flags(n, kind, options, options.flags())
    }
    fn new_flags(
        n: usize,
        kind: Kind,
        options: PlanOptions,
        flags: u32,
    ) -> Result<Self, FftwError> {
        let native = checked_len::<R>(n)?;
        let table = Table::load()?;
        // Initialized, private buffers: destructive planning never sees caller data.
        let mut a = zeros::<Complex<R>>(n)?;
        let mut b = zeros::<Complex<R>>(n)?;
        let mut r = zeros::<R>(n)?;
        let _guard = lock::<R>();
        let _threads = set_threads(&table, options.threads)?;
        #[cfg(test)]
        trace("limit/plan");
        let _reset = Reset(&table.limit);
        // SAFETY: dimensions fit c_int and all arrays hold at least n initialized
        // elements. Complex<T> has repr(C), two consecutive T fields (num-complex).
        // UNALIGNED removes SIMD alignment constraints, NOT alias constraints.
        let handle = unsafe {
            (table.limit)(options.time_limit.map_or(-1.0, |t| t.as_secs_f64()));
            #[cfg(test)]
            {
                LIMIT_TRACE.with(|events| {
                    events
                        .borrow_mut()
                        .push(options.time_limit.map_or(-1.0, |t| t.as_secs_f64()));
                });
                assert!(
                    !PANIC_AFTER_LIMIT.with(|needle| needle.replace(false)),
                    "planning unwind"
                );
            }
            match kind {
                Kind::Complex(d, ip) => {
                    #[cfg(test)]
                    trace_flags(flags | FFTW_PRESERVE_INPUT);
                    (table.pc)(
                        native,
                        a.as_mut_ptr(),
                        if ip { a.as_mut_ptr() } else { b.as_mut_ptr() },
                        if d == FftDirection::Forward { -1 } else { 1 },
                        flags | FFTW_PRESERVE_INPUT,
                    )
                }
                Kind::Forward => {
                    #[cfg(test)]
                    trace_flags(flags);
                    (table.pf)(native, r.as_mut_ptr(), b.as_mut_ptr(), flags)
                }
                Kind::Inverse => {
                    #[cfg(test)]
                    trace_flags(flags);
                    (table.pi)(native, a.as_mut_ptr(), r.as_mut_ptr(), flags)
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
        #[cfg(test)]
        trace("destroy");
        unsafe { (self.table.destroy)(self.handle.as_ptr()) }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires both native FFTW runtimes"]
    fn native_timelimit_calls_on_success_error_and_unwind() {
        let _serial = crate::tests::NATIVE_TEST
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        fn check<R: Real>() {
            let options = PlanOptions::new(
                PlanningRigor::Measure,
                Some(std::time::Duration::from_millis(1)),
            )
            .unwrap();
            for mode in 0..3 {
                forget_wisdom::<R>().unwrap();
                LIMIT_TRACE.with(|events| events.borrow_mut().clear());
                PANIC_AFTER_LIMIT.with(|needle| needle.set(mode == 2));
                let result = std::panic::catch_unwind(|| {
                    Plan::<R>::new_flags(
                        17,
                        Kind::Complex(FftDirection::Forward, false),
                        options,
                        options.flags() | if mode == 1 { FFTW_WISDOM_ONLY } else { 0 },
                    )
                });
                // No subsequent constructor/setter may overwrite the evidence.
                // Real API calls/arguments only; native state relies on FFTW's contract.
                LIMIT_TRACE.with(|events| {
                    assert_eq!(
                        *events.borrow(),
                        [0.001, -1.0],
                        "single={}, mode={mode}",
                        R::SINGLE
                    );
                });
                match mode {
                    0 => assert!(matches!(result, Ok(Ok(_)))),
                    1 => assert!(matches!(result, Ok(Err(FftwError::NullPlan)))),
                    _ => assert!(result.is_err()),
                }
            }
        }
        check::<f32>();
        check::<f64>();
    }
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
    #[ignore = "requires both native FFTW runtimes and pthread libraries"]
    fn native_c2c_second_plan_failure_drops_first_and_resets_threads() {
        let _serial = crate::tests::NATIVE_TEST
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        fn check<R: Real>() {
            let trained = PlanOptions::new(PlanningRigor::Measure, None)
                .unwrap()
                .with_threads(2)
                .unwrap()
                .with_conserve_memory(true);
            let kind = Kind::Complex(FftDirection::Forward, false);
            forget_wisdom::<R>().unwrap();
            drop(Plan::<R>::new(31, kind, trained.with_threads(1).unwrap()).unwrap());
            drop(Plan::<R>::new(17, kind, trained).unwrap());
            let wisdom = export_wisdom::<R>().unwrap();
            forget_wisdom::<R>().unwrap();
            import_wisdom::<R>(&wisdom).unwrap();
            let only = trained.with_wisdom_only(true);
            TRACE.with(|events| events.borrow_mut().clear());
            FLAG_TRACE.with(|events| events.borrow_mut().clear());
            let result = crate::plan_c2c::<R>(17, FftDirection::Forward, only);
            assert!(matches!(result, Err(FftwError::NullPlan)));
            TRACE.with(|events| {
                let events = events.borrow();
                assert_eq!(
                    events.iter().filter(|&&event| event == "destroy").count(),
                    1
                );
                assert_eq!(events.iter().filter(|&&event| event == "reset1").count(), 2);
            });
            FLAG_TRACE.with(|flags| {
                let flags = flags.borrow();
                assert_eq!(flags.len(), 2);
                assert!(flags.iter().all(|&value| value & FFTW_CONSERVE_MEMORY != 0));
                assert!(flags.iter().all(|&value| value & FFTW_PRESERVE_INPUT != 0));
                assert!(flags.iter().all(|&value| value & FFTW_WISDOM_ONLY != 0));
            });
            // Bypass our setter: one-thread wisdom can succeed only if the
            // failed second plan restored the native thread count to one.
            {
                let table = Table::<R>::load().unwrap();
                let _guard = lock::<R>();
                let mut input = vec![Complex::<R>::default(); 31];
                let mut output = input.clone();
                // SAFETY: initialized disjoint buffers of the trained length;
                // matching flags and live table, with the planner lock held.
                let handle = unsafe {
                    (table.pc)(
                        31,
                        input.as_mut_ptr(),
                        output.as_mut_ptr(),
                        -1,
                        only.flags() | FFTW_PRESERVE_INPUT,
                    )
                };
                assert!(
                    !handle.is_null(),
                    "failed plan did not reset native threads"
                );
                // SAFETY: non-null owned handle; table and planner lock still live.
                unsafe { (table.destroy)(handle) };
            }
            FLAG_TRACE.with(|events| events.borrow_mut().clear());
            drop(Plan::<R>::new(17, Kind::Forward, trained).unwrap());
            drop(Plan::<R>::new(17, Kind::Inverse, trained).unwrap());
            FLAG_TRACE.with(|flags| {
                assert_eq!(*flags.borrow(), [trained.flags(), trained.flags()]);
            });
            crate::forget_wisdom::<R>().unwrap();
        }
        check::<f32>();
        check::<f64>();
    }

    #[test]
    #[ignore = "requires both native FFTW runtimes"]
    fn native_partial_construction_cleanup() {
        let _serial = crate::tests::NATIVE_TEST
            .lock()
            .unwrap_or_else(|e| e.into_inner());
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
    #[ignore = "requires both native FFTW runtimes and pthread libraries"]
    fn native_wisdom_lifetime_and_threads() {
        let _serial = crate::tests::NATIVE_TEST
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        fn check<R: Real>() -> String {
            forget_wisdom::<R>().unwrap();
            let options = PlanOptions::default();
            let kind = Kind::Complex(FftDirection::Forward, false);
            let only = options.flags() | (1 << 21);
            assert!(matches!(
                Plan::<R>::new_flags(19, kind, options, only),
                Err(FftwError::NullPlan)
            ));
            drop(Plan::<R>::new(19, kind, options).unwrap());
            let wisdom = export_wisdom::<R>().unwrap();
            forget_wisdom::<R>().unwrap();
            assert!(matches!(
                Plan::<R>::new_flags(19, kind, options, only),
                Err(FftwError::NullPlan)
            ));
            import_wisdom::<R>(&wisdom).unwrap(); // no plan or temporary table survives
            let p = Plan::<R>::new_flags(19, kind, options, only).unwrap();
            forget_wisdom::<R>().unwrap();
            let input = vec![Complex::new(R::one(), R::zero()); 19];
            let mut output = vec![Complex::default(); 19];
            p.complex(&input, &mut output);
            assert_eq!(output[0].re, R::from_usize(19).unwrap());
            assert!(matches!(
                import_wisdom::<R>("bad wisdom"),
                Err(WisdomError::InvalidWisdom)
            ));
            assert!(matches!(
                import_wisdom::<R>("a\0b"),
                Err(WisdomError::InteriorNul)
            ));
            for count in [2, 3, 1] {
                let options = options.with_threads(count).unwrap();
                drop(Plan::<R>::new(23, kind, options).unwrap());
                let _guard = lock::<R>();
                let slot = if R::SINGLE {
                    &SINGLE_THREADS
                } else {
                    &DOUBLE_THREADS
                };
                assert!(matches!(slot.lock().unwrap().result, Some(Ok(_))));
            }
            // Prove native reset without a nonexistent getter: seed one-thread
            // wisdom, then call WISDOM_ONLY directly, bypassing our setter.
            forget_wisdom::<R>().unwrap();
            drop(Plan::<R>::new(31, kind, options).unwrap());
            for mode in 0..3 {
                let threaded = options.with_threads(3).unwrap();
                match mode {
                    0 => drop(Plan::<R>::new(29, kind, threaded).unwrap()),
                    1 => assert!(matches!(
                        Plan::<R>::new_flags(37, kind, threaded, only),
                        Err(FftwError::NullPlan)
                    )),
                    _ => assert!(
                        std::panic::catch_unwind(|| {
                            let _guard = lock::<R>();
                            let table = Table::<R>::load().unwrap();
                            let _reset = set_threads(&table, 3).unwrap();
                            panic!("native thread reset unwind");
                        })
                        .is_err()
                    ),
                }
                let table = Table::<R>::load().unwrap();
                let _guard = lock::<R>();
                let mut input = vec![Complex::<R>::default(); 31];
                let mut output = input.clone();
                // SAFETY: exact initialized dimensions, matching OOP wisdom flags.
                let handle = unsafe {
                    (table.pc)(31, input.as_mut_ptr(), output.as_mut_ptr(), -1, only | 16)
                };
                assert!(
                    !handle.is_null(),
                    "native thread count was not reset to one: mode={mode}, single={}",
                    R::SINGLE
                );
                unsafe { (table.destroy)(handle) };
            }
            let handles: Vec<_> = (0..3)
                .map(|count| {
                    std::thread::spawn(move || {
                        for _ in 0..4 {
                            let p =
                                Plan::<R>::new(7, kind, options.with_threads(count + 1).unwrap())
                                    .unwrap();
                            let wisdom = export_wisdom::<R>().unwrap();
                            forget_wisdom::<R>().unwrap();
                            import_wisdom::<R>(&wisdom).unwrap();
                            let mut output = vec![Complex::default(); 7];
                            p.complex(&[Complex::new(R::one(), R::zero()); 7], &mut output);
                            assert_eq!(output[0].re, R::from_usize(7).unwrap());
                        }
                    })
                })
                .collect();
            for h in handles {
                h.join().unwrap();
            }
            wisdom
        }
        let single = check::<f32>();
        let double = check::<f64>();
        assert!(matches!(
            import_wisdom::<f64>(&single),
            Err(WisdomError::InvalidWisdom)
        ));
        assert!(matches!(
            import_wisdom::<f32>(&double),
            Err(WisdomError::InvalidWisdom)
        ));
        let before = export_wisdom::<f64>().unwrap();
        forget_wisdom::<f32>().unwrap();
        assert_eq!(export_wisdom::<f64>().unwrap(), before);
    }
    #[test]
    fn thread_reset_on_success_error_unwind() {
        use std::sync::atomic::{AtomicI32, Ordering};
        static COUNT: AtomicI32 = AtomicI32::new(0);
        unsafe extern "C" fn set(n: c_int) {
            COUNT.store(n, Ordering::SeqCst);
        }
        for mode in 0..3 {
            COUNT.store(3, Ordering::SeqCst);
            let result = std::panic::catch_unwind(|| -> Result<(), FftwError> {
                let _reset = ThreadReset(Some(set));
                match mode {
                    1 => Err(FftwError::NullPlan),
                    2 => panic!("unwind"),
                    _ => Ok(()),
                }
            });
            assert_eq!(COUNT.load(Ordering::SeqCst), 1);
            assert_eq!(result.is_err(), mode == 2);
        }
    }
    #[test]
    #[ignore = "requires both native FFTW base runtimes; isolated loader failures"]
    fn native_cached_thread_load_failure() {
        const CHILD: &str = "PENCIL_FFTW_TEST_CACHE_CHILD";
        if let Ok(precision) = std::env::var(CHILD) {
            fn check<R: Real>() {
                let options = PlanOptions::default();
                let kind = Kind::Complex(FftDirection::Forward, false);
                let base = Plan::<R>::new(7, kind, options).unwrap();
                let fail = || {
                    let result = Plan::<R>::new(7, kind, options.with_threads(2).unwrap());
                    match std::env::var("PENCIL_FFTW_TEST_THREADS_LIBRARY")
                        .unwrap()
                        .as_str()
                    {
                        "libc.so.6" => assert!(matches!(result, Err(FftwError::Symbol(_)))),
                        _ => assert!(matches!(result, Err(FftwError::Load(_)))),
                    }
                };
                fail();
                let serial = Plan::<R>::new(7, kind, options).unwrap();
                for plan in [&base, &serial] {
                    let mut output = vec![Complex::default(); 7];
                    // A unit impulse has a unit spectrum: actual native FFT math.
                    let mut input = vec![Complex::default(); 7];
                    input[0].re = R::one();
                    plan.complex(&input, &mut output);
                    assert_eq!(output, vec![Complex::new(R::one(), R::zero()); 7]);
                }
                fail();
                drop((base, serial));
                let wisdom = export_wisdom::<R>().unwrap();
                forget_wisdom::<R>().unwrap();
                import_wisdom::<R>(&wisdom).unwrap();
                drop(Plan::<R>::new_flags(7, kind, options, options.flags() | (1 << 21)).unwrap());
                TRACE.with(|events| {
                    let events = events.borrow();
                    assert!(!events.contains(&"init"));
                    assert!(!events.contains(&"set1"));
                    assert!(!events.contains(&"set>1"));
                    assert!(!events.contains(&"reset1"));
                });
                let slot = if R::SINGLE {
                    &SINGLE_THREADS
                } else {
                    &DOUBLE_THREADS
                };
                let state = slot.lock().unwrap();
                assert!(!state.init_attempted);
                assert!(state.result.as_ref().unwrap().is_err());
            }
            match precision.as_str() {
                "f32" => check::<f32>(),
                "f64" => check::<f64>(),
                _ => panic!("unknown child precision"),
            }
            return;
        }
        for precision in ["f32", "f64"] {
            for library in ["/nonexistent/pencil-fftw-threads.so", "libc.so.6"] {
                let status = std::process::Command::new(std::env::current_exe().unwrap())
                    .args([
                        "--exact",
                        "ffi::tests::native_cached_thread_load_failure",
                        "--ignored",
                        "--nocapture",
                    ])
                    .env(CHILD, precision)
                    .env("PENCIL_FFTW_TEST_THREADS_LIBRARY", library)
                    .status()
                    .unwrap();
                assert!(status.success(), "{precision}: {library}");
            }
        }
    }
    #[test]
    #[ignore = "requires native FFTW; fresh-process initialization order"]
    fn native_initialization_order() {
        const CHILD: &str = "PENCIL_FFTW_TEST_ORDER_CHILD";
        if let Ok(mode) = std::env::var(CHILD) {
            fn check<R: Real>(mode: &str) {
                let options = PlanOptions::default();
                let kind = Kind::Complex(FftDirection::Forward, false);
                // Static version access must not initialize or call base routines.
                eprintln!("runtime single={}: {}", R::SINGLE, version::<R>().unwrap());
                TRACE.with(|events| assert!(events.borrow().is_empty()));
                if mode == "failure" {
                    assert!(export_wisdom::<R>().is_err());
                    assert!(forget_wisdom::<R>().is_err());
                    assert!(import_wisdom::<R>("bad wisdom").is_err());
                    for count in [1, 2, 1] {
                        assert!(matches!(
                            Plan::<R>::new(7, kind, options.with_threads(count).unwrap()),
                            Err(FftwError::Load(_))
                        ));
                    }
                    TRACE.with(|events| assert_eq!(&*events.borrow(), &["init"]));
                    return;
                }
                match mode {
                    "serial" => drop(Plan::<R>::new(7, kind, options).unwrap()),
                    "export" => {
                        export_wisdom::<R>().unwrap();
                    }
                    "forget" => forget_wisdom::<R>().unwrap(),
                    "import" => assert!(matches!(
                        import_wisdom::<R>("bad wisdom"),
                        Err(WisdomError::InvalidWisdom)
                    )),
                    "threaded" => (),
                    _ => panic!("unknown order"),
                }
                for count in [2, 1] {
                    let plan =
                        Plan::<R>::new(7, kind, options.with_threads(count).unwrap()).unwrap();
                    let mut input = vec![Complex::default(); 7];
                    input[0].re = R::one();
                    let mut output = input.clone();
                    plan.complex(&input, &mut output);
                    assert_eq!(output, vec![Complex::new(R::one(), R::zero()); 7]);
                }
                let wisdom = export_wisdom::<R>().unwrap();
                forget_wisdom::<R>().unwrap();
                import_wisdom::<R>(&wisdom).unwrap();
                drop(Plan::<R>::new_flags(7, kind, options, options.flags() | (1 << 21)).unwrap());
                TRACE.with(|events| {
                    let events = events.borrow();
                    assert_eq!(events[0], "init", "{events:?}");
                    assert_eq!(events.iter().filter(|&&e| e == "init").count(), 1);
                    assert!(events.contains(&"set>1"));
                    assert!(events.contains(&"set1"));
                    assert!(events.contains(&"reset1"));
                    eprintln!("single={}: {events:?}", R::SINGLE);
                });
            }
            match std::env::var("PENCIL_FFTW_TEST_PRECISION")
                .unwrap()
                .as_str()
            {
                "f32" => check::<f32>(&mode),
                "f64" => check::<f64>(&mode),
                _ => unreachable!(),
            }
            return;
        }
        for precision in ["f32", "f64"] {
            for mode in [
                "serial", "export", "import", "forget", "threaded", "failure",
            ] {
                let mut command = std::process::Command::new(std::env::current_exe().unwrap());
                command
                    .args([
                        "--exact",
                        "ffi::tests::native_initialization_order",
                        "--ignored",
                        "--nocapture",
                    ])
                    .env(CHILD, mode)
                    .env("PENCIL_FFTW_TEST_PRECISION", precision);
                if mode == "failure" {
                    command.env("PENCIL_FFTW_TEST_INIT_FAILURE", "1");
                }
                assert!(command.status().unwrap().success(), "{precision}: {mode}");
            }
        }
    }
    #[test]
    fn missing_library_and_symbol() {
        assert!(matches!(
            load_threads::<f64>("/nonexistent/threads.so", None, &mut false),
            Err(FftwError::Load(_))
        ));
        #[cfg(target_os = "linux")]
        assert!(matches!(
            load_threads::<f32>("libc.so.6", None, &mut false),
            Err(FftwError::Symbol(_))
        ));
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
