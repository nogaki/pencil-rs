#![deny(unsafe_code)]
#![deny(missing_debug_implementations)]

#[cfg(feature = "distributed")]
pub mod distributed;
#[cfg(feature = "distributed")]
mod distributed_real;

mod ffi;
use ffi::{Api, CUcontext, CUdeviceptr, CUresult, CufftHandle, CufftResult};
use num_complex::Complex;
use std::{marker::PhantomData, rc::Rc};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaError {
    #[error("CUDA/cuFFT libraries unavailable: {0}")]
    MissingLibrary(String),
    #[error("CUDA symbol unavailable: {0}")]
    MissingSymbol(String),
    #[error("CUDA error {code}: {message}")]
    Driver { code: i32, message: String },
    #[error("cuFFT error {0}")]
    Fft(i32),
    #[error("invalid size: {0}")]
    Size(String),
    #[error("device has no CUDA device")]
    NoDevice,
    #[error("buffer belongs to another CUDA device/context")]
    WrongDevice,
}
fn drv(api: &Api, r: CUresult) -> Result<(), CudaError> {
    if r == ffi::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(CudaError::Driver {
            code: r,
            message: api.err(r),
        })
    }
}
fn fft(r: CufftResult) -> Result<(), CudaError> {
    if r == ffi::CUFFT_SUCCESS {
        Ok(())
    } else {
        Err(CudaError::Fft(r))
    }
}
fn fallible_vec<T>(len: usize) -> Result<Vec<T>, CudaError> {
    let mut v = Vec::new();
    v.try_reserve_exact(len)
        .map_err(|_| CudaError::Size("host allocation failed".into()))?;
    Ok(v)
}

// CUDA may change the stack even when returning an asynchronous error.
// Never disarm based on a return code: verify the actual ambient context.
struct Pop<'a> {
    api: &'a Api,
    prior: CUcontext,
    private: CUcontext,
    active: bool,
}
impl Pop<'_> {
    fn restore(&mut self) -> Result<(), CudaError> {
        let (r, current) = self.api.current();
        if r != 0 {
            restoration_failed();
        }
        let mut result = Ok(());
        if current != self.prior {
            if current != self.private {
                restoration_failed();
            }
            result = drv(self.api, self.api.pop().0);
        }
        if self.api.current() != (0, self.prior) {
            restoration_failed();
        }
        self.active = false;
        result
    }
}
fn restoration_failed() -> ! {
    eprintln!("pencil-cuda: cannot confirm ambient context restoration; aborting");
    std::process::abort()
}
impl Drop for Pop<'_> {
    fn drop(&mut self) {
        if self.active {
            let _ = self.restore();
        }
    }
}
struct Inner {
    api: Api,
    ctx: CUcontext,
    device: i32,
}
impl Drop for Inner {
    fn drop(&mut self) {
        if !self.api.uncertain.get() && self.api.destroy_ctx(self.ctx) != 0 {
            self.api.uncertain.set(true);
        }
    }
}
#[derive(Clone)]
pub struct CudaDevice(Rc<Inner>);
impl std::fmt::Debug for CudaDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaDevice")
            .field("device", &self.0.device)
            .finish()
    }
}
impl CudaDevice {
    pub fn new() -> Result<Self, CudaError> {
        Self::with_ordinal(0)
    }

    pub fn with_ordinal(ordinal: usize) -> Result<Self, CudaError> {
        let api = Api::load().map_err(|e| match e {
            ffi::LoadError::Library(s) => CudaError::MissingLibrary(s),
            ffi::LoadError::Symbol(s) => {
                CudaError::MissingSymbol(String::from_utf8_lossy(s).into_owned())
            }
        })?;
        drv(&api, api.init())?;
        let (r, n) = api.count();
        drv(&api, r)?;
        if n == 0 || ordinal >= n as usize {
            return Err(CudaError::NoDevice);
        }
        let (r, d) = api.device(ordinal as i32);
        drv(&api, r)?;
        let (r, prior) = api.current();
        drv(&api, r)?;
        let mut pop = Pop {
            api: &api,
            prior,
            private: std::ptr::null_mut(),
            active: true,
        };
        let (r, c) = api.create(d);
        pop.private = c;
        let restored = pop.restore();
        drop(pop);
        let result = drv(&api, r).and(restored);
        if let Err(e) = result {
            // On a failed create, an output handle need not have been supplied.
            // If a handle was supplied, restore first, then dispose of it.
            if !c.is_null() && api.destroy_ctx(c) != 0 {
                api.uncertain.set(true);
            }
            if api.current() != (0, prior) {
                restoration_failed();
            }
            return Err(e);
        }
        Ok(Self(Rc::new(Inner {
            api,
            ctx: c,
            device: d,
        })))
    }

    pub fn device_name(&self) -> Result<String, CudaError> {
        self.with_context(|i| {
            let (r, n) = i.api.name(i.device);
            drv(&i.api, r)?;
            Ok(n)
        })
    }
    fn with_context<R>(
        &self,
        f: impl FnOnce(&Inner) -> Result<R, CudaError>,
    ) -> Result<R, CudaError> {
        if self.0.api.uncertain.get() {
            return Err(CudaError::Driver {
                code: 999,
                message: "context poisoned: completion unconfirmed; native resources retained"
                    .into(),
            });
        }
        let (r, prior) = self.0.api.current();
        drv(&self.0.api, r)?;
        // Avoid ambiguous duplicate stack entries, including during nested drops.
        if prior == self.0.ctx {
            return f(&self.0);
        }
        let mut pop = Pop {
            api: &self.0.api,
            prior,
            private: self.0.ctx,
            active: true,
        };
        drv(&self.0.api, self.0.api.push(self.0.ctx))?;
        let (r, current) = self.0.api.current();
        drv(&self.0.api, r)?;
        if current != self.0.ctx {
            return Err(CudaError::WrongDevice);
        }
        let result = f(&self.0);
        match (result, pop.restore()) {
            (Err(e), _) => Err(e),
            (Ok(_), Err(e)) => Err(e),
            (Ok(v), Ok(())) => Ok(v),
        }
    }
    fn drop_plan(&self, h: CufftHandle) {
        if self
            .with_context(|i| {
                drv(&i.api, i.api.synchronize())?;
                fft(i.api.destroy_plan(h))
            })
            .is_err()
        {
            self.0.api.uncertain.set(true);
        }
    }
    fn same(&self, other: &CudaDevice) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

pub trait DeviceScalar: sealed::Sealed + Copy + 'static {
    const BYTES: usize;
}
mod sealed {
    pub trait Sealed {}
}
impl sealed::Sealed for f32 {}
impl DeviceScalar for f32 {
    const BYTES: usize = 4;
}
impl sealed::Sealed for f64 {}
impl DeviceScalar for f64 {
    const BYTES: usize = 8;
}
impl sealed::Sealed for Complex<f32> {}
impl DeviceScalar for Complex<f32> {
    const BYTES: usize = 8;
}
impl sealed::Sealed for Complex<f64> {}
impl DeviceScalar for Complex<f64> {
    const BYTES: usize = 16;
}

#[derive(Debug)]
pub struct CudaBuffer<T: DeviceScalar> {
    device: CudaDevice,
    ptr: CUdeviceptr,
    len: usize,
    _t: PhantomData<T>,
}
impl<T: DeviceScalar> CudaBuffer<T> {
    pub fn new(device: &CudaDevice, len: usize) -> Result<Self, CudaError> {
        let bytes = len
            .checked_mul(T::BYTES)
            .filter(|&b| b <= isize::MAX as usize)
            .ok_or_else(|| CudaError::Size("byte count overflow".into()))?;
        device.with_context(|i| {
            let mut buffer = Self {
                device: device.clone(),
                ptr: 0,
                len,
                _t: PhantomData,
            };
            if bytes != 0 {
                let (r, p) = i.api.alloc_mem(bytes);
                buffer.ptr = p;
                drv(&i.api, r)?;
                drv(&i.api, i.api.zero(p, bytes))?;
            }
            Ok(buffer)
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn device(&self) -> CudaDevice {
        self.device.clone()
    }
    pub fn copy_from_bytes(&mut self, bytes: &[u8]) -> Result<(), CudaError> {
        let expected = self
            .len
            .checked_mul(T::BYTES)
            .filter(|&b| b <= isize::MAX as usize)
            .ok_or_else(|| CudaError::Size("byte count overflow".into()))?;
        if bytes.len() != expected {
            return Err(CudaError::Size("host length does not match buffer".into()));
        }
        if bytes.is_empty() {
            return Ok(());
        }
        self.device
            .with_context(|i| drv(&i.api, i.api.h2d(self.ptr, bytes)))
    }
    pub fn copy_to_bytes(&self) -> Result<Vec<u8>, CudaError> {
        if self.len == 0 {
            return Ok(Vec::new());
        }
        self.device.with_context(|i| {
            let bytes = self
                .len
                .checked_mul(T::BYTES)
                .filter(|&b| b <= isize::MAX as usize)
                .ok_or_else(|| CudaError::Size("byte count overflow".into()))?;
            let (r, b) = i.api.d2h(self.ptr, bytes);
            if r == ffi::CUDA_ERROR_OUT_OF_MEMORY && b.len() != bytes {
                return Err(CudaError::Size("host allocation failed".into()));
            }
            drv(&i.api, r)?;
            Ok(b)
        })
    }
}
macro_rules! typed_buffer {
    ($t:ty, $n:expr, $encode:expr, $decode:expr) => {
        impl CudaBuffer<$t> {
            pub fn upload(&mut self, values: &[$t]) -> Result<(), CudaError> {
                if values.len() != self.len {
                    return Err(CudaError::Size("host length does not match buffer".into()));
                }
                let mut bytes = fallible_vec(
                    values
                        .len()
                        .checked_mul($n)
                        .ok_or_else(|| CudaError::Size("host allocation size overflow".into()))?,
                )?;
                for v in values {
                    bytes.extend_from_slice(&$encode(*v));
                }
                self.copy_from_bytes(&bytes)
            }
            pub fn download(&self) -> Result<Vec<$t>, CudaError> {
                let bytes = self.copy_to_bytes()?;
                let mut values = fallible_vec(self.len)?;
                values.extend(bytes.chunks_exact($n).map($decode));
                Ok(values)
            }
        }
    };
}
typed_buffer!(f32, 4, f32::to_ne_bytes, |b: &[u8]| f32::from_ne_bytes(
    b.try_into().unwrap()
));
typed_buffer!(f64, 8, f64::to_ne_bytes, |b: &[u8]| f64::from_ne_bytes(
    b.try_into().unwrap()
));
impl CudaBuffer<Complex<f32>> {
    pub fn upload(&mut self, values: &[Complex<f32>]) -> Result<(), CudaError> {
        if values.len() != self.len {
            return Err(CudaError::Size("host length does not match buffer".into()));
        }
        let mut b = fallible_vec(
            values
                .len()
                .checked_mul(8)
                .ok_or_else(|| CudaError::Size("host allocation size overflow".into()))?,
        )?;
        for v in values {
            b.extend(v.re.to_ne_bytes());
            b.extend(v.im.to_ne_bytes());
        }
        self.copy_from_bytes(&b)
    }
    pub fn download(&self) -> Result<Vec<Complex<f32>>, CudaError> {
        let b = self.copy_to_bytes()?;
        let mut values = fallible_vec(self.len)?;
        values.extend(b.chunks_exact(8).map(|x| {
            Complex::new(
                f32::from_ne_bytes(x[..4].try_into().unwrap()),
                f32::from_ne_bytes(x[4..].try_into().unwrap()),
            )
        }));
        Ok(values)
    }
}
impl CudaBuffer<Complex<f64>> {
    pub fn upload(&mut self, values: &[Complex<f64>]) -> Result<(), CudaError> {
        if values.len() != self.len {
            return Err(CudaError::Size("host length does not match buffer".into()));
        }
        let mut b = fallible_vec(
            values
                .len()
                .checked_mul(16)
                .ok_or_else(|| CudaError::Size("host allocation size overflow".into()))?,
        )?;
        for v in values {
            b.extend(v.re.to_ne_bytes());
            b.extend(v.im.to_ne_bytes());
        }
        self.copy_from_bytes(&b)
    }
    pub fn download(&self) -> Result<Vec<Complex<f64>>, CudaError> {
        let b = self.copy_to_bytes()?;
        let mut values = fallible_vec(self.len)?;
        values.extend(b.chunks_exact(16).map(|x| {
            Complex::new(
                f64::from_ne_bytes(x[..8].try_into().unwrap()),
                f64::from_ne_bytes(x[8..].try_into().unwrap()),
            )
        }));
        Ok(values)
    }
}
impl<T: DeviceScalar> Drop for CudaBuffer<T> {
    fn drop(&mut self) {
        if self.ptr != 0
            && self
                .device
                .with_context(|i| {
                    drv(&i.api, i.api.synchronize())?;
                    drv(&i.api, i.api.free_mem(self.ptr))
                })
                .is_err()
        {
            self.device.0.api.uncertain.set(true);
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Precision {
    F32,
    F64,
}
fn checked(n: usize, batch: usize) -> Result<(), CudaError> {
    if n == 0 {
        return Err(CudaError::Size("zero transform length".into()));
    }
    if n > i32::MAX as usize || batch > i32::MAX as usize {
        return Err(CudaError::Size("cuFFT dimensions exceed i32".into()));
    }
    n.checked_mul(batch)
        .and_then(|v| v.checked_mul(16))
        .filter(|&v| v <= isize::MAX as usize)
        .ok_or_else(|| CudaError::Size("batch length overflow".into()))?;
    (n / 2 + 1)
        .checked_mul(batch)
        .ok_or_else(|| CudaError::Size("reduced length overflow".into()))?;
    Ok(())
}
#[derive(Debug)]
pub struct C2CPlan<T: DeviceScalar> {
    device: CudaDevice,
    n: usize,
    batch: usize,
    handle: Option<CufftHandle>,
    p: Precision,
    _t: PhantomData<T>,
}
pub type C2CPlanF32 = C2CPlan<Complex<f32>>;
pub type C2CPlanF64 = C2CPlan<Complex<f64>>;
impl<T: DeviceScalar> C2CPlan<T> {
    fn make(device: &CudaDevice, n: usize, batch: usize, p: Precision) -> Result<Self, CudaError> {
        checked(n, batch)?;
        device.with_context(|i| {
            let mut plan = Self {
                device: device.clone(),
                n,
                batch,
                handle: None,
                p,
                _t: PhantomData,
            };
            if batch != 0 {
                let (r, h) = i.api.plan(
                    n,
                    batch,
                    match p {
                        Precision::F32 => ffi::CUFFT_C2C,
                        Precision::F64 => ffi::CUFFT_Z2Z,
                    },
                );
                if h != 0 {
                    plan.handle = Some(h);
                }
                fft(r)?;
                plan.handle = Some(h);
            }
            Ok(plan)
        })
    }
    pub fn len(&self) -> usize {
        self.n * self.batch
    }
    pub fn is_empty(&self) -> bool {
        self.batch == 0
    }
    pub fn execute(
        &self,
        input: &CudaBuffer<T>,
        output: &mut CudaBuffer<T>,
        inverse: bool,
    ) -> Result<(), CudaError> {
        if !self.device.same(&input.device) || !self.device.same(&output.device) {
            return Err(CudaError::WrongDevice);
        }
        if input.len() != self.len() || output.len() != self.len() {
            return Err(CudaError::Size("C2C length mismatch".into()));
        }
        self.run(input.ptr, output.ptr, inverse, inverse)
    }
    fn run(
        &self,
        input: CUdeviceptr,
        output: CUdeviceptr,
        inverse: bool,
        normalize: bool,
    ) -> Result<(), CudaError> {
        if self.batch == 0 {
            return Ok(());
        }
        let h = self.handle.unwrap();
        self.device.with_context(|i| {
            let r = match self.p {
                Precision::F32 => i.api.c2c(
                    h,
                    input,
                    output,
                    if inverse {
                        ffi::CUFFT_INVERSE
                    } else {
                        ffi::CUFFT_FORWARD
                    },
                ),
                Precision::F64 => i.api.z2z(
                    h,
                    input,
                    output,
                    if inverse {
                        ffi::CUFFT_INVERSE
                    } else {
                        ffi::CUFFT_FORWARD
                    },
                ),
            };
            let result = fft(r);
            let sync = drv(&i.api, i.api.synchronize());
            result?;
            sync?;
            if normalize {
                drv(
                    &i.api,
                    i.api.scale(
                        output,
                        (self.len() * 2) as u64,
                        1.0 / self.n as f64,
                        matches!(self.p, Precision::F64),
                    ),
                )?;
            }
            Ok(())
        })
    }
    pub fn execute_forward(
        &self,
        input: &CudaBuffer<T>,
        output: &mut CudaBuffer<T>,
    ) -> Result<(), CudaError> {
        self.execute(input, output, false)
    }
    pub fn execute_backward(
        &self,
        input: &CudaBuffer<T>,
        output: &mut CudaBuffer<T>,
    ) -> Result<(), CudaError> {
        if !self.device.same(&input.device) || !self.device.same(&output.device) {
            return Err(CudaError::WrongDevice);
        }
        if input.len != self.len() || output.len != self.len() {
            return Err(CudaError::Size("C2C length mismatch".into()));
        }
        self.run(input.ptr, output.ptr, true, false)
    }
    pub fn execute_inverse(
        &self,
        input: &CudaBuffer<T>,
        output: &mut CudaBuffer<T>,
    ) -> Result<(), CudaError> {
        self.execute(input, output, true)
    }
    pub fn execute_in_place(
        &self,
        buf: &mut CudaBuffer<T>,
        inverse: bool,
    ) -> Result<(), CudaError> {
        if !self.device.same(&buf.device) {
            return Err(CudaError::WrongDevice);
        }
        if buf.len != self.len() {
            return Err(CudaError::Size("C2C length mismatch".into()));
        }
        self.run(buf.ptr, buf.ptr, inverse, inverse)
    }
    pub fn execute_backward_in_place(&self, buf: &mut CudaBuffer<T>) -> Result<(), CudaError> {
        if !self.device.same(&buf.device) {
            return Err(CudaError::WrongDevice);
        }
        if buf.len != self.len() {
            return Err(CudaError::Size("C2C length mismatch".into()));
        }
        self.run(buf.ptr, buf.ptr, true, false)
    }
}
impl C2CPlanF32 {
    pub fn new(d: &CudaDevice, n: usize, b: usize) -> Result<Self, CudaError> {
        Self::make(d, n, b, Precision::F32)
    }
}
impl C2CPlanF64 {
    pub fn new(d: &CudaDevice, n: usize, b: usize) -> Result<Self, CudaError> {
        Self::make(d, n, b, Precision::F64)
    }
}
impl<T: DeviceScalar> Drop for C2CPlan<T> {
    fn drop(&mut self) {
        if let Some(h) = self.handle {
            self.device.drop_plan(h);
        }
    }
}

#[derive(Debug)]
pub struct R2CPlan<T: DeviceScalar, U: DeviceScalar> {
    device: CudaDevice,
    n: usize,
    batch: usize,
    handle: Option<CufftHandle>,
    p: Precision,
    _t: PhantomData<(T, U)>,
}
pub type R2CPlanF32 = R2CPlan<f32, Complex<f32>>;
pub type R2CPlanF64 = R2CPlan<f64, Complex<f64>>;
impl<T: DeviceScalar, U: DeviceScalar> R2CPlan<T, U> {
    fn make(d: &CudaDevice, n: usize, b: usize, p: Precision) -> Result<Self, CudaError> {
        checked(n, b)?;
        d.with_context(|i| {
            let mut plan = Self {
                device: d.clone(),
                n,
                batch: b,
                handle: None,
                p,
                _t: PhantomData,
            };
            if b != 0 {
                let (r, h) = i.api.plan(
                    n,
                    b,
                    match p {
                        Precision::F32 => ffi::CUFFT_R2C,
                        Precision::F64 => ffi::CUFFT_D2Z,
                    },
                );
                if h != 0 {
                    plan.handle = Some(h);
                }
                fft(r)?;
                plan.handle = Some(h);
            }
            Ok(plan)
        })
    }
    pub fn input_len(&self) -> usize {
        self.n * self.batch
    }
    pub fn output_len(&self) -> usize {
        (self.n / 2 + 1) * self.batch
    }
    pub fn execute(
        &self,
        input: &CudaBuffer<T>,
        output: &mut CudaBuffer<U>,
    ) -> Result<(), CudaError> {
        if !self.device.same(&input.device) || !self.device.same(&output.device) {
            return Err(CudaError::WrongDevice);
        }
        if input.len() != self.input_len() || output.len() != self.output_len() {
            return Err(CudaError::Size("R2C buffer length mismatch".into()));
        }
        if input.ptr != 0 && input.ptr == output.ptr {
            return Err(CudaError::Size("R2C input/output alias".into()));
        }
        if self.batch == 0 {
            return Ok(());
        }
        let scratch = CudaBuffer::<T>::new(&self.device, input.len())?;
        self.device.with_context(|i| {
            drv(
                &i.api,
                i.api.d2d(scratch.ptr, input.ptr, input.len() * T::BYTES),
            )?;
            let result = fft(match self.p {
                Precision::F32 => i.api.r2c(self.handle.unwrap(), scratch.ptr, output.ptr),
                Precision::F64 => i.api.d2z(self.handle.unwrap(), scratch.ptr, output.ptr),
            });
            let sync = drv(&i.api, i.api.synchronize());
            result.and(sync)
        })
    }
}
impl R2CPlanF32 {
    pub fn new(d: &CudaDevice, n: usize, b: usize) -> Result<Self, CudaError> {
        Self::make(d, n, b, Precision::F32)
    }
}
impl R2CPlanF64 {
    pub fn new(d: &CudaDevice, n: usize, b: usize) -> Result<Self, CudaError> {
        Self::make(d, n, b, Precision::F64)
    }
}

#[derive(Debug)]
pub struct C2RPlan<T: DeviceScalar, U: DeviceScalar> {
    device: CudaDevice,
    n: usize,
    batch: usize,
    handle: Option<CufftHandle>,
    p: Precision,
    _t: PhantomData<(T, U)>,
}
pub type C2RPlanF32 = C2RPlan<Complex<f32>, f32>;
pub type C2RPlanF64 = C2RPlan<Complex<f64>, f64>;
impl<T: DeviceScalar, U: DeviceScalar> C2RPlan<T, U> {
    fn make(d: &CudaDevice, n: usize, b: usize, p: Precision) -> Result<Self, CudaError> {
        checked(n, b)?;
        d.with_context(|i| {
            let mut plan = Self {
                device: d.clone(),
                n,
                batch: b,
                handle: None,
                p,
                _t: PhantomData,
            };
            if b != 0 {
                let (r, h) = i.api.plan(
                    n,
                    b,
                    match p {
                        Precision::F32 => ffi::CUFFT_C2R,
                        Precision::F64 => ffi::CUFFT_Z2D,
                    },
                );
                if h != 0 {
                    plan.handle = Some(h);
                }
                fft(r)?;
                plan.handle = Some(h);
            }
            Ok(plan)
        })
    }
    pub fn input_len(&self) -> usize {
        (self.n / 2 + 1) * self.batch
    }
    pub fn output_len(&self) -> usize {
        self.n * self.batch
    }
    pub fn execute(
        &self,
        input: &CudaBuffer<T>,
        output: &mut CudaBuffer<U>,
    ) -> Result<(), CudaError> {
        self.execute_inverse(input, output)
    }
    pub fn execute_inverse(
        &self,
        input: &CudaBuffer<T>,
        output: &mut CudaBuffer<U>,
    ) -> Result<(), CudaError> {
        self.execute_backward(input, output)?;
        if self.batch == 0 {
            return Ok(());
        }
        self.device.with_context(|i| {
            drv(
                &i.api,
                i.api.scale(
                    output.ptr,
                    self.output_len() as u64,
                    1.0 / self.n as f64,
                    matches!(self.p, Precision::F64),
                ),
            )
        })
    }
    pub fn execute_backward(
        &self,
        input: &CudaBuffer<T>,
        output: &mut CudaBuffer<U>,
    ) -> Result<(), CudaError> {
        if !self.device.same(&input.device) || !self.device.same(&output.device) {
            return Err(CudaError::WrongDevice);
        }
        if input.len() != self.input_len() || output.len() != self.output_len() {
            return Err(CudaError::Size("C2R buffer length mismatch".into()));
        }
        if input.ptr != 0 && input.ptr == output.ptr {
            return Err(CudaError::Size("C2R input/output alias".into()));
        }
        if self.batch == 0 {
            return Ok(());
        }
        // cuFFT's real inverse requires real DC and (for even n) Nyquist bins.
        // Check only those endpoints; the remaining spectrum is intentionally left on-device.
        let scalar = match self.p {
            Precision::F32 => 4,
            Precision::F64 => 8,
        };
        let stride = (self.n / 2 + 1) * scalar * 2;
        for batch in 0..self.batch {
            for bin in [0, self.n / 2]
                .into_iter()
                .filter(|&k| k < self.n / 2 + 1 && (self.n % 2 == 0 || k == 0))
            {
                let off = batch * stride + bin * scalar * 2 + scalar;
                let bytes = self.device.with_context(|i| {
                    let (r, bytes) = i.api.d2h(input.ptr + off as u64, scalar);
                    drv(&i.api, r)?;
                    Ok(bytes)
                })?;
                let imag = if scalar == 4 {
                    f32::from_ne_bytes(bytes.as_slice().try_into().unwrap()) as f64
                } else {
                    f64::from_ne_bytes(bytes.as_slice().try_into().unwrap())
                };
                if imag != 0.0 {
                    return Err(CudaError::Size("C2R endpoint is not real".into()));
                }
            }
        }
        let scratch = CudaBuffer::<T>::new(&self.device, input.len())?;
        self.device.with_context(|i| {
            drv(
                &i.api,
                i.api.d2d(scratch.ptr, input.ptr, input.len() * T::BYTES),
            )?;
            let result = fft(match self.p {
                Precision::F32 => i.api.c2r(self.handle.unwrap(), scratch.ptr, output.ptr),
                Precision::F64 => i.api.z2d(self.handle.unwrap(), scratch.ptr, output.ptr),
            });
            let sync = drv(&i.api, i.api.synchronize());
            result.and(sync)
        })
    }
}
impl C2RPlanF32 {
    pub fn new(d: &CudaDevice, n: usize, b: usize) -> Result<Self, CudaError> {
        Self::make(d, n, b, Precision::F32)
    }
}
impl C2RPlanF64 {
    pub fn new(d: &CudaDevice, n: usize, b: usize) -> Result<Self, CudaError> {
        Self::make(d, n, b, Precision::F64)
    }
}
impl<T: DeviceScalar, U: DeviceScalar> Drop for C2RPlan<T, U> {
    fn drop(&mut self) {
        if let Some(h) = self.handle {
            self.device.drop_plan(h);
        }
    }
}

impl<T: DeviceScalar, U: DeviceScalar> Drop for R2CPlan<T, U> {
    fn drop(&mut self) {
        if let Some(h) = self.handle {
            self.device.drop_plan(h);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires CUDA driver and cuFFT"]
    fn context_restore_on_operation_error_and_unwind() {
        let d = CudaDevice::new().expect("explicit hardware run requires CUDA");
        let api = &d.0.api;
        let prior = api.current();
        assert_eq!(prior.0, 0);
        drv(api, api.push(d.0.ctx)).unwrap();
        let other = CudaDevice::new().unwrap();
        assert_eq!(api.current(), (0, d.0.ctx));
        let error: Result<(), CudaError> = other.with_context(|_| Err(CudaError::WrongDevice));
        assert!(error.is_err());
        assert_eq!(api.current(), (0, d.0.ctx));
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _: Result<(), CudaError> = other.with_context(|_| panic!("test unwind"));
        }));
        assert!(panic.is_err());
        assert_eq!(api.current(), (0, d.0.ctx));
        other.device_name().unwrap();
        drop(other);
        assert_eq!(api.current(), (0, d.0.ctx));
        assert_eq!(api.pop(), (0, d.0.ctx));
        assert_eq!(api.current(), prior);
    }
    #[test]
    #[ignore = "requires CUDA driver and cuFFT"]
    fn context_restore_after_native_success_reported_as_error() {
        let ambient = CudaDevice::new().expect("explicit hardware run requires CUDA");
        let api = &ambient.0.api;
        let prior = api.current();
        drv(api, api.push(ambient.0.ctx)).unwrap();
        for operation in ["create", "pop"] {
            ffi::FAIL_AFTER.with(|f| f.set(operation));
            assert!(
                CudaDevice::new().is_err(),
                "genuine constructor error: {operation}"
            );
            assert_eq!(api.current(), (0, ambient.0.ctx));
            ffi::FAIL_AFTER.with(|f| assert_eq!(f.get(), ""));
        }
        let other = CudaDevice::new().unwrap();
        for operation in ["push", "pop"] {
            ffi::FAIL_AFTER.with(|f| f.set(operation));
            assert!(other.device_name().is_err());
            assert_eq!(api.current(), (0, ambient.0.ctx));
            other.device_name().unwrap();
        }
        drop(other);
        assert_eq!(api.pop(), (0, ambient.0.ctx));
        assert_eq!(api.current(), prior);
    }

    #[test]
    #[ignore = "requires CUDA driver and cuFFT; deliberately retains poisoned native resources"]
    fn uncertain_completion_retains_resources_context_and_libraries() {
        for failed in ["buffer", "c2c", "r2c", "c2r", "module", "none"] {
            let d = CudaDevice::new().expect("explicit hardware run requires CUDA");
            let prior = d.0.api.current();
            let b = CudaBuffer::<f32>::new(&d, 8).unwrap();
            let c2c = C2CPlanF32::new(&d, 4, 1).unwrap();
            let r2c = R2CPlanF32::new(&d, 4, 1).unwrap();
            let c2r = C2RPlanF32::new(&d, 4, 1).unwrap();
            ffi::RELEASES.with(|r| r.borrow_mut().clear());
            if failed != "none" {
                ffi::FAIL_AFTER.with(|f| f.set("sync"));
            }
            match failed {
                "buffer" => {
                    drop(b);
                    drop(c2c);
                    drop(r2c);
                    drop(c2r);
                }
                "c2c" => {
                    drop(c2c);
                    drop(b);
                    drop(r2c);
                    drop(c2r);
                }
                "r2c" => {
                    drop(r2c);
                    drop(b);
                    drop(c2c);
                    drop(c2r);
                }
                "c2r" => {
                    drop(c2r);
                    drop(b);
                    drop(c2c);
                    drop(r2c);
                }
                _ => {
                    let result =
                        d.with_context(|i| drv(&i.api, i.api.scale(b.ptr, 8, 0.25, false)));
                    assert_eq!(result.is_err(), failed == "module");
                    drop(b);
                    drop(c2c);
                    drop(r2c);
                    drop(c2r);
                }
            }
            assert_eq!(d.0.api.current(), prior);
            assert_eq!(d.device_name().is_err(), failed != "none");
            drop(d);
            ffi::RELEASES.with(|r| {
                let r = r.borrow();
                if failed == "none" {
                    assert_eq!(
                        &*r,
                        &["module", "buffer", "plan", "plan", "plan", "context"]
                    );
                } else {
                    assert_eq!(&*r, &["retained libraries"], "{failed}");
                }
            });
        }
    }
    #[test]
    fn checked_native_shapes_and_byte_bounds() {
        for n in [1, 2, 3, 17, i32::MAX as usize] {
            checked(n, 0).unwrap();
        }
        checked(1, 1).unwrap();
        assert!(checked(0, 0).is_err());
        assert!(checked(0, 1).is_err());
        assert!(checked(i32::MAX as usize + 1, 0).is_err());
        assert!(checked(1, i32::MAX as usize + 1).is_err());
        assert!(checked(i32::MAX as usize, i32::MAX as usize).is_err());
        assert!(checked(usize::MAX, usize::MAX).is_err());
    }
    #[test]
    fn shipped_kernels_are_unsigned_bounded_grid_stride() {
        let ptx = include_str!("scale.ptx");
        for precision in [32, 64] {
            assert!(ptx.contains(&format!(".entry scale{precision}")));
            assert!(ptx.contains(&format!("mul.rn.f{precision}")));
        }
        assert_eq!(ptx.matches("setp.ge.u64 %p, %i, %n").count(), 2);
        assert_eq!(ptx.matches("setp.le.u64 %p, %left, %stride").count(), 2);
        assert_eq!(ptx.matches("mul.wide.u32 %stride").count(), 2);
    }
}
