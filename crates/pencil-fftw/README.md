# pencil-fftw — optional native CPU adapter

**Licensing: this wrapper's source is MIT; FFTW is NOT MIT.** No FFTW source or
binary is vendored. The installed FFTW distribution is GPL v2 or later
(`/usr/share/doc/libfftw3-double3/copyright`); commercial licenses are available
from FFTW's rights holders. **Dynamic loading is not an exemption from GPL
obligations.** Distributors must evaluate the GPL or obtain appropriate
commercial terms. This is not legal advice.

No MPI, GPU, build-time FFTW link, pkg-config, or header dependency. Merely
building or using another workspace crate does not require FFTW. Explicit
factories load `libfftw3.so.3` (f64) or `libfftw3f.so.3` (f32); failures return
errors, never a RustFFT fallback. These sonames currently target Linux.

## Integration API

```rust
use pencil_fftw::{plan_c2c, PlanOptions};
use rustfft::FftDirection;
# fn example() -> Result<(), pencil_fftw::FftwError> {
let fft = plan_c2c::<f64>(16, FftDirection::Forward, PlanOptions::default())?;
assert_eq!(fft.len(), 16);
# Ok(())
# }
```

- `Real: rustfft::FftNum + Default` (sealed; only f32/f64).
- `plan_c2c<R: Real>(usize, FftDirection, PlanOptions) -> Result<Arc<dyn rustfft::Fft<R>>, FftwError>`.
- `plan_r2c<R: Real>(usize, PlanOptions) -> Result<Arc<dyn realfft::RealToComplex<R>>, FftwError>`.
- `plan_c2r<R: Real>(usize, PlanOptions) -> Result<Arc<dyn realfft::ComplexToReal<R>>, FftwError>`.
- `runtime_version<R: Real>() -> Result<String, FftwError>` queries the loaded runtime, not headers.
- `PlanOptions::new(PlanningRigor, Option<Duration>) -> Result<PlanOptions, FftwError>`;
  private fields, `rigor()` and `time_limit()` getters. Default: Estimate, no limit.
- `PlanningRigor::{Estimate, Measure, Patient, Exhaustive}`. A supplied limit must
  be positive; FFTW treats it as an approximate planning budget, not a deadline.
- `FftwError::{Load(String), Symbol(String), InvalidOptions(&'static str), Overflow,
  NullPlan, Allocation(TryReserveError)}`. Existing crates' errors are untouched.

All transforms are raw/unnormalized, like RustFFT/RealFFT. C2C supports one or
more whole batches; all RustFFT process forms panic for input/buffer lengths
shorter than the plan, including zero. Consumers allowing empty batches must
return early before calling the adapter. Real traits process exactly one transform
per call (consumers loop for batching). Invalid C2C dimensions panic before native
execution, preserving input, output and scratch; real dimensions return the
existing `FftError` variants before writes.
C2R reports `InputValues` **after** execution for imaginary DC/even Nyquist.
Real inputs are mutable and may be destroyed. Consumers needing preservation
must copy them. C2C immutable input is explicitly preserved. Scratch lengths
are zero; any supplied scratch (including its tail) is untouched. Convenience
vectors and private planning buffers are fully initialized. Planning buffers use
fallible allocation; convenience trait Vec methods follow ordinary Rust allocation
behavior because their signatures cannot return allocation errors.

C2C owns distinct native IP/OOP plans. All plans use UNALIGNED, so ordinary Rust
slices and offset subslices are valid. No native DCT/DST/DHT API is provided:
consumer R2R/DHT integration must use complex embedding kernels, not claim direct
native real-to-real transforms.

## Safety and limitations

Unsafe code is confined to private `ffi.rs`. Function signatures follow fftw3.h;
num-complex's repr(C) pair matches FFTW's two-real complex representation.
Libraries outlive all their function pointers and plan handles. Planning and
plan destruction are serialized with separate global f32/f64 locks. FFTW's
new-array execution may concurrently share immutable plans with disjoint buffers.
Private buffers isolate even destructive MEASURE/PATIENT planning from callers.
Time limits are set under the planner lock and reset to `FFTW_NO_TIMELIMIT` (-1)
on exit; FFTW provides no getter, so previous foreign state is not restored.
There is no `fftw_cleanup` call. **Uncoordinated planning/destruction/time-limit
changes by foreign FFTW users in the process are outside this guarantee.**

## Checks

`cargo test -p pencil-fftw` runs host-only validation/load-error checks.
`cargo test -p pencil-fftw -- --ignored` explicitly requires BOTH native runtimes
and fails if either is unavailable. Native checks use independent direct DFTs
and cover both precisions, all rigors with bounded planning, both directions,
odd/even/unit sizes, batches, IP/OOP, offset buffers, preservation/error paths,
arbitrary spectra, raw normalization and shared-plan concurrency.
