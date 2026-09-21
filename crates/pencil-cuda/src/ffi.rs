#![allow(unsafe_code)]
use libloading::Library;
use std::{
    cell::Cell,
    ffi::{CStr, c_char, c_void},
};

#[cfg(not(target_pointer_width = "64"))]
compile_error!("pencil-cuda requires a 64-bit host for CUDA device-address pointer conversion");

pub type CufftReal = f32;
pub type CufftDoubleReal = f64;
#[repr(C, align(8))]
pub struct CufftComplex {
    pub x: f32,
    pub y: f32,
}
#[repr(C, align(16))]
pub struct CufftDoubleComplex {
    pub x: f64,
    pub y: f64,
}

// Tests inject only AFTER a real successful native call; this is not an ABI mock.
#[cfg(test)]
thread_local! { pub static FAIL_AFTER: Cell<&'static str> = const { Cell::new("") }; }
fn after_call(operation: &str, result: i32) -> i32 {
    #[cfg(test)]
    if result == 0
        && FAIL_AFTER.with(|f| {
            if f.get() == operation {
                f.set("");
                true
            } else {
                false
            }
        })
    {
        return 999;
    }
    let _ = operation;
    result
}
pub type CUdevice = i32;
pub type CUcontext = *mut c_void;
pub type CUdeviceptr = u64;
pub type CUresult = i32;
pub type CufftResult = i32;
pub type CufftHandle = i32;
pub const CUDA_SUCCESS: i32 = 0;
pub const CUDA_ERROR_OUT_OF_MEMORY: i32 = 2;
pub const CUFFT_SUCCESS: i32 = 0;
pub const CUFFT_C2C: i32 = 0x29;
pub const CUFFT_R2C: i32 = 0x2a;
pub const CUFFT_C2R: i32 = 0x2c;
pub const CUFFT_Z2Z: i32 = 0x69;
pub const CUFFT_D2Z: i32 = 0x6a;
pub const CUFFT_Z2D: i32 = 0x6c;
pub const CUFFT_FORWARD: i32 = -1;
pub const CUFFT_INVERSE: i32 = 1;
#[derive(Debug)]
pub enum LoadError {
    Library(String),
    Symbol(&'static [u8]),
}
macro_rules! get {
    ($l:expr,$n:literal,$t:ty) => {
        unsafe { *$l.get::<$t>($n).map_err(|_| LoadError::Symbol($n))? }
    };
}
type Init = unsafe extern "C" fn(u32) -> i32;
type Count = unsafe extern "C" fn(*mut i32) -> i32;
type Dev = unsafe extern "C" fn(*mut CUdevice, i32) -> i32;
type Name = unsafe extern "C" fn(*mut c_char, i32, CUdevice) -> i32;
type Create = unsafe extern "C" fn(*mut CUcontext, u32, CUdevice) -> i32;
type Destroy = unsafe extern "C" fn(CUcontext) -> i32;
type Push = unsafe extern "C" fn(CUcontext) -> i32;
type Pop = unsafe extern "C" fn(*mut CUcontext) -> i32;
type GetCurrent = unsafe extern "C" fn(*mut CUcontext) -> i32;
type Synchronize = unsafe extern "C" fn() -> i32;
type Alloc = unsafe extern "C" fn(*mut CUdeviceptr, usize) -> i32;
type Free = unsafe extern "C" fn(CUdeviceptr) -> i32;
type H2D = unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> i32;
type D2H = unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> i32;
type D2D = unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, usize) -> i32;
type Memset = unsafe extern "C" fn(CUdeviceptr, u8, usize) -> i32;
type ModuleLoad = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32;
type ModuleUnload = unsafe extern "C" fn(*mut c_void) -> i32;
type Function = unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> i32;
type Launch = unsafe extern "C" fn(
    *mut c_void,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    *mut c_void,
    *mut *mut c_void,
    *mut *mut c_void,
) -> i32;
type Err = unsafe extern "C" fn(i32, *mut *const c_char) -> i32;
type Plan = unsafe extern "C" fn(
    *mut i32,
    i32,
    *mut i32,
    *mut i32,
    i32,
    i32,
    *mut i32,
    i32,
    i32,
    i32,
    i32,
) -> i32;
type PDestroy = unsafe extern "C" fn(i32) -> i32;
type ExecC2C =
    unsafe extern "C" fn(CufftHandle, *mut CufftComplex, *mut CufftComplex, i32) -> CufftResult;
type ExecZ2Z = unsafe extern "C" fn(
    CufftHandle,
    *mut CufftDoubleComplex,
    *mut CufftDoubleComplex,
    i32,
) -> CufftResult;
type ExecR2C = unsafe extern "C" fn(CufftHandle, *mut CufftReal, *mut CufftComplex) -> CufftResult;
type ExecC2R = unsafe extern "C" fn(CufftHandle, *mut CufftComplex, *mut CufftReal) -> CufftResult;
type ExecD2Z =
    unsafe extern "C" fn(CufftHandle, *mut CufftDoubleReal, *mut CufftDoubleComplex) -> CufftResult;
type ExecZ2D =
    unsafe extern "C" fn(CufftHandle, *mut CufftDoubleComplex, *mut CufftDoubleReal) -> CufftResult;
pub struct Api {
    _cuda: Option<Library>,
    _cufft: Option<Library>,
    pub uncertain: Cell<bool>,
    init: Init,
    count: Count,
    dev: Dev,
    name: Name,
    create: Create,
    destroy: Destroy,
    push: Push,
    pop: Pop,
    get_current: GetCurrent,
    synchronize: Synchronize,
    alloc: Alloc,
    free: Free,
    h2d: H2D,
    d2h: D2H,
    d2d: D2D,
    memset: Memset,
    module_load: ModuleLoad,
    module_unload: ModuleUnload,
    function: Function,
    launch: Launch,
    err: Err,
    plan: Plan,
    pdestroy: PDdestroy,
    c2c: ExecC2C,
    z2z: ExecZ2Z,
    r2c: ExecR2C,
    c2r: ExecC2R,
    d2z: ExecD2Z,
    z2d: ExecZ2D,
}
impl Drop for Api {
    fn drop(&mut self) {
        if self.uncertain.get() {
            // A leaked native handle must never outlive its executable code.
            std::mem::forget(self._cuda.take());
            std::mem::forget(self._cufft.take());
            #[cfg(test)]
            RELEASES.with(|r| r.borrow_mut().push("retained libraries"));
        }
    }
}
#[cfg(test)]
thread_local! { pub static RELEASES: std::cell::RefCell<Vec<&'static str>> = const { std::cell::RefCell::new(Vec::new()) }; }
fn releasing(_kind: &'static str) {
    #[cfg(test)]
    RELEASES.with(|r| r.borrow_mut().push(_kind));
}
type PDdestroy = PDestroy;
fn open(a: &[&str]) -> Result<Library, LoadError> {
    a.iter()
        .find_map(|x| unsafe { Library::new(x).ok() })
        .ok_or_else(|| LoadError::Library(a.join(",")))
}
impl Api {
    pub fn load() -> Result<Self, LoadError> {
        let c = open(&["libcuda.so.1", "libcuda.so"])?;
        let f = open(&["libcufft.so.12", "libcufft.so.11", "libcufft.so"])?;
        Ok(Self {
            init: get!(c, b"cuInit\0", Init),
            count: get!(c, b"cuDeviceGetCount\0", Count),
            dev: get!(c, b"cuDeviceGet\0", Dev),
            name: get!(c, b"cuDeviceGetName\0", Name),
            create: get!(c, b"cuCtxCreate_v2\0", Create),
            destroy: get!(c, b"cuCtxDestroy_v2\0", Destroy),
            push: get!(c, b"cuCtxPushCurrent_v2\0", Push),
            pop: get!(c, b"cuCtxPopCurrent_v2\0", Pop),
            get_current: get!(c, b"cuCtxGetCurrent\0", GetCurrent),
            synchronize: get!(c, b"cuCtxSynchronize\0", Synchronize),
            alloc: get!(c, b"cuMemAlloc_v2\0", Alloc),
            free: get!(c, b"cuMemFree_v2\0", Free),
            h2d: get!(c, b"cuMemcpyHtoD_v2\0", H2D),
            d2h: get!(c, b"cuMemcpyDtoH_v2\0", D2H),
            d2d: get!(c, b"cuMemcpyDtoD_v2\0", D2D),
            memset: get!(c, b"cuMemsetD8_v2\0", Memset),
            module_load: get!(c, b"cuModuleLoadData\0", ModuleLoad),
            module_unload: get!(c, b"cuModuleUnload\0", ModuleUnload),
            function: get!(c, b"cuModuleGetFunction\0", Function),
            launch: get!(c, b"cuLaunchKernel\0", Launch),
            err: get!(c, b"cuGetErrorString\0", Err),
            _cuda: Some(c),
            uncertain: Cell::new(false),
            plan: get!(f, b"cufftPlanMany\0", Plan),
            pdestroy: get!(f, b"cufftDestroy\0", PDestroy),
            c2c: get!(f, b"cufftExecC2C\0", ExecC2C),
            z2z: get!(f, b"cufftExecZ2Z\0", ExecZ2Z),
            r2c: get!(f, b"cufftExecR2C\0", ExecR2C),
            c2r: get!(f, b"cufftExecC2R\0", ExecC2R),
            d2z: get!(f, b"cufftExecD2Z\0", ExecD2Z),
            z2d: get!(f, b"cufftExecZ2D\0", ExecZ2D),
            _cufft: Some(f),
        })
    }
}
pub fn text(p: *const c_char) -> String {
    if p.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(p).to_string_lossy().into_owned() }
    }
}
impl Api {
    pub fn init(&self) -> i32 {
        unsafe { (self.init)(0) }
    }
    pub fn count(&self) -> (i32, i32) {
        let mut x = 0;
        (unsafe { (self.count)(&mut x) }, x)
    }
    pub fn device(&self, n: i32) -> (i32, CUdevice) {
        let mut x = 0;
        (unsafe { (self.dev)(&mut x, n) }, x)
    }
    pub fn name(&self, d: CUdevice) -> (i32, String) {
        let mut x = [0i8; 256];
        (
            unsafe { (self.name)(x.as_mut_ptr(), 256, d) },
            text(x.as_ptr()),
        )
    }
    pub fn create(&self, d: CUdevice) -> (i32, CUcontext) {
        let mut x = std::ptr::null_mut();
        (
            after_call("create", unsafe { (self.create)(&mut x, 0, d) }),
            x,
        )
    }
    pub fn destroy_ctx(&self, x: CUcontext) -> i32 {
        releasing("context");
        unsafe { (self.destroy)(x) }
    }
    pub fn push(&self, x: CUcontext) -> i32 {
        after_call("push", unsafe { (self.push)(x) })
    }
    pub fn pop(&self) -> (i32, CUcontext) {
        let mut x = std::ptr::null_mut();
        (after_call("pop", unsafe { (self.pop)(&mut x) }), x)
    }
    pub fn current(&self) -> (i32, CUcontext) {
        let mut x = std::ptr::null_mut();
        (unsafe { (self.get_current)(&mut x) }, x)
    }
    pub fn synchronize(&self) -> i32 {
        if self.uncertain.get() {
            return 999;
        }
        let r = after_call("sync", unsafe { (self.synchronize)() });
        if r != 0 {
            self.uncertain.set(true);
        }
        r
    }
    pub fn alloc_mem(&self, n: usize) -> (i32, CUdeviceptr) {
        let mut x = 0;
        (unsafe { (self.alloc)(&mut x, n) }, x)
    }
    pub fn free_mem(&self, x: CUdeviceptr) -> i32 {
        releasing("buffer");
        unsafe { (self.free)(x) }
    }
    pub fn h2d(&self, d: CUdeviceptr, b: &[u8]) -> i32 {
        unsafe { (self.h2d)(d, b.as_ptr().cast(), b.len()) }
    }
    pub fn d2h(&self, d: CUdeviceptr, n: usize) -> (i32, Vec<u8>) {
        let mut b = Vec::new();
        if b.try_reserve_exact(n).is_err() {
            return (CUDA_ERROR_OUT_OF_MEMORY, b);
        }
        // SAFETY: the allocation above reserves n writable bytes, and the native
        // call writes at most that requested transfer size.
        unsafe { b.set_len(n) };
        (unsafe { (self.d2h)(b.as_mut_ptr().cast(), d, n) }, b)
    }
    pub fn d2d(&self, a: CUdeviceptr, b: CUdeviceptr, n: usize) -> i32 {
        unsafe { (self.d2d)(a, b, n) }
    }
    pub fn err(&self, r: i32) -> String {
        let mut p = std::ptr::null();
        unsafe { (self.err)(r, &mut p) };
        text(p)
    }
    pub fn plan(&self, n: usize, b: usize, k: i32) -> (i32, i32) {
        let mut h = 0;
        let (Ok(mut nn), Ok(batch)) = (i32::try_from(n), i32::try_from(b)) else {
            return (8, 0); // CUFFT_INVALID_SIZE, before any native call.
        };
        if nn == 0 || batch == 0 {
            return (8, 0);
        }
        let distance = nn;
        (
            unsafe {
                (self.plan)(
                    &mut h,
                    1,
                    &mut nn,
                    std::ptr::null_mut(),
                    1,
                    distance,
                    std::ptr::null_mut(),
                    1,
                    distance,
                    k,
                    batch,
                )
            },
            h,
        )
    }
    pub fn destroy_plan(&self, h: i32) -> i32 {
        releasing("plan");
        unsafe { (self.pdestroy)(h) }
    }
    pub fn c2c(&self, h: i32, a: CUdeviceptr, b: CUdeviceptr, d: i32) -> i32 {
        unsafe { (self.c2c)(h, a as *mut CufftComplex, b as *mut CufftComplex, d) }
    }
    pub fn z2z(&self, h: i32, a: CUdeviceptr, b: CUdeviceptr, d: i32) -> i32 {
        unsafe {
            (self.z2z)(
                h,
                a as *mut CufftDoubleComplex,
                b as *mut CufftDoubleComplex,
                d,
            )
        }
    }
    pub fn r2c(&self, h: i32, a: CUdeviceptr, b: CUdeviceptr) -> i32 {
        unsafe { (self.r2c)(h, a as *mut CufftReal, b as *mut CufftComplex) }
    }
    pub fn c2r(&self, h: i32, a: CUdeviceptr, b: CUdeviceptr) -> i32 {
        unsafe { (self.c2r)(h, a as *mut CufftComplex, b as *mut CufftReal) }
    }
    pub fn d2z(&self, h: i32, a: CUdeviceptr, b: CUdeviceptr) -> i32 {
        unsafe { (self.d2z)(h, a as *mut CufftDoubleReal, b as *mut CufftDoubleComplex) }
    }
    pub fn z2d(&self, h: i32, a: CUdeviceptr, b: CUdeviceptr) -> i32 {
        unsafe { (self.z2d)(h, a as *mut CufftDoubleComplex, b as *mut CufftDoubleReal) }
    }
}

impl Api {
    pub fn zero(&self, p: CUdeviceptr, bytes: usize) -> i32 {
        unsafe { (self.memset)(p, 0, bytes) }
    }
    // Caller holds the owning context current until synchronization and unload.
    pub fn scale(&self, mut ptr: CUdeviceptr, mut count: u64, factor: f64, double: bool) -> i32 {
        struct Module<'a>(&'a Api, *mut c_void);
        impl Drop for Module<'_> {
            fn drop(&mut self) {
                if !self.1.is_null() && !self.0.uncertain.get() {
                    releasing("module");
                    if unsafe { (self.0.module_unload)(self.1) } != 0 {
                        self.0.uncertain.set(true);
                    }
                }
            }
        }
        let mut module = std::ptr::null_mut();
        let ptx = concat!(include_str!("scale.ptx"), "\0");
        let r = unsafe { (self.module_load)(&mut module, ptx.as_ptr().cast()) };
        let _module = Module(self, module);
        if r != 0 {
            return r;
        }
        let mut function = std::ptr::null_mut();
        let name = if double { b"scale64\0" } else { b"scale32\0" };
        let r = unsafe { (self.function)(&mut function, module, name.as_ptr().cast()) };
        if r != 0 {
            return r;
        }
        let mut f64_factor = factor;
        let mut f32_factor = factor as f32;
        let factor_ptr: *mut c_void = if double {
            (&mut f64_factor as *mut f64).cast()
        } else {
            (&mut f32_factor as *mut f32).cast()
        };
        let mut args = [
            (&mut ptr as *mut u64).cast(),
            (&mut count as *mut u64).cast(),
            factor_ptr,
        ];
        let grid = count.div_ceil(256).min(65535) as u32;
        let r = unsafe {
            (self.launch)(
                function,
                grid,
                1,
                1,
                256,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        let sync = self.synchronize();
        if r != 0 { r } else { sync }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn loaded_exec_signatures_are_typed_pointers() {
        // Compile-time regression check only, not native ABI validation.
        let _check = |api: &Api| {
            let _: unsafe extern "C" fn(i32, *mut CufftComplex, *mut CufftComplex, i32) -> i32 =
                api.c2c;
            let _: unsafe extern "C" fn(
                i32,
                *mut CufftDoubleComplex,
                *mut CufftDoubleComplex,
                i32,
            ) -> i32 = api.z2z;
            let _: unsafe extern "C" fn(i32, *mut CufftReal, *mut CufftComplex) -> i32 = api.r2c;
            let _: unsafe extern "C" fn(i32, *mut CufftComplex, *mut CufftReal) -> i32 = api.c2r;
            let _: unsafe extern "C" fn(i32, *mut CufftDoubleReal, *mut CufftDoubleComplex) -> i32 =
                api.d2z;
            let _: unsafe extern "C" fn(i32, *mut CufftDoubleComplex, *mut CufftDoubleReal) -> i32 =
                api.z2d;
        };
    }
    #[test]
    fn host_layout_and_post_success_fault_hooks() {
        use std::mem::{align_of, size_of};
        assert_eq!(size_of::<usize>(), 8);
        assert_eq!(
            (size_of::<CufftComplex>(), align_of::<CufftComplex>()),
            (8, 8)
        );
        assert_eq!(
            (
                size_of::<CufftDoubleComplex>(),
                align_of::<CufftDoubleComplex>()
            ),
            (16, 16)
        );
        for operation in ["create", "push", "pop", "sync"] {
            FAIL_AFTER.with(|f| f.set(operation));
            assert_eq!(after_call(operation, 17), 17);
            assert_eq!(after_call("unrelated", 0), 0);
            assert_eq!(after_call(operation, 0), 999);
            assert_eq!(after_call(operation, 0), 0);
        }
    }
    #[test]
    fn missing_library_is_a_loader_error() {
        assert!(matches!(
            open(&["/nonexistent/pencil-cuda/no-fallback.so"]),
            Err(LoadError::Library(_))
        ));
    }
    #[test]
    #[cfg(unix)]
    fn missing_symbol_is_a_loader_error() {
        fn probe() -> Result<Init, LoadError> {
            let library: Library = libloading::os::unix::Library::this().into();
            Ok(get!(
                library,
                b"pencil_cuda_intentionally_missing_symbol_4873\0",
                Init
            ))
        }
        assert!(matches!(probe(), Err(LoadError::Symbol(_))));
    }
}
