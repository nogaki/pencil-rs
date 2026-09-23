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
  private fields, `rigor()` and `time_limit()` getters. Default: Estimate, no limit,
  one CPU thread. `with_threads(usize)` validates a positive count fitting `c_int`;
  `requested_threads()` reports the request, not measured native utilization.
  `with_wisdom_only(bool)` and `with_conserve_memory(bool)` set the matching
  native planning flags (both default false); `wisdom_only()` and
  `conserve_memory()` report them. Wisdom-only misses return `NullPlan`, with no
  fallback. Conserving memory is a planner hint, not a performance guarantee.
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

## Wisdom and CPU threads

```rust
use pencil_fftw::{export_wisdom, import_wisdom, forget_wisdom, plan_c2c, PlanOptions};
# fn example() -> Result<(), Box<dyn std::error::Error>> {
let options = PlanOptions::default().with_threads(2)?;
let plan = plan_c2c::<f64>(64, rustfft::FftDirection::Forward, options)?;
drop(plan);
let wisdom = export_wisdom::<f64>()?;
std::fs::write("double.wisdom", &wisdom)?;
forget_wisdom::<f64>()?;
import_wisdom::<f64>(&std::fs::read_to_string("double.wisdom")?)?;
# Ok(())
# }
```

These sealed precision-generic functions use native FFTW wisdom. Export returns
an owned `String`, freeing the native allocation even on conversion/allocation
failure. `WisdomError` distinguishes interior NUL, rejected wisdom, null export,
invalid UTF-8 and wrapped `FftwError`. Bad imports have no transactional guarantee.
Forgetting wisdom does not invalidate existing plans. Wisdom is precision-specific
and native-version/flags/thread-setting compatibility remains FFTW's decision.

Successful base-library loads are cached for process lifetime (one per precision),
so wisdom imported without any live plans survives temporary tables dropping.
Before the first stateful FFTW call per precision (including serial planning or
wisdom import/export/forget), the matching pthread library
`libfftw3[f]_threads.so.3` is optionally loaded under the precision planner lock.
Core function addresses resolved through its dependency scope must match the
pinned base instance before `init_threads` is called, **before any base routine**.
Static version-symbol reads are not routine calls. Successful initialization
pins both libraries for process lifetime because it registers native callbacks.
Default RustFFT does not load FFTW. Explicit serial FFTW does not require the
thread runtime, but initializes it when available for future threaded plans.
Missing libraries/symbols or a mismatched base instance are cached per precision
until process exit (no retry): default/count-one planning and wisdom remain
base-only, with no setter, and requests above one fail without fallback. No
later unsafe enablement is attempted after base-only use. If initialization was
attempted and failed, **all planning and wisdom operations** fail with the cached
`FftwError::Load`; potentially changed native state is not ignored.
After successful initialization, each planning operation (including count one)
sets its requested count after initialization and resets to known one on success,
error, or unwind. This is not restoration of foreign state: FFTW 3.3.8 has no
thread-count getter. No speedup or actual thread utilization is promised.

## Safety and limitations

Initialization follows FFTW's “Usage of Multi-threaded FFTW” requirement that
`fftw_init_threads` precede any other FFTW routine. This ordering is enforced for
calls coordinated by this crate; prior FFTW calls by foreign code are outside
the guarantee. No cleanup/reinitialization is performed behind existing plans.

Unsafe code is confined to private `ffi.rs`. Function signatures follow fftw3.h;
num-complex's repr(C) pair matches FFTW's two-real complex representation.
Libraries outlive all their function pointers and plan handles. Planning and
plan destruction, wisdom operations and thread initialization/settings are
serialized with separate global f32/f64 locks. FFTW's
new-array execution may concurrently share immutable plans with disjoint buffers.
Private buffers isolate even destructive MEASURE/PATIENT planning from callers.
Time limits are set under the planner lock and reset to `FFTW_NO_TIMELIMIT` (-1)
on exit; FFTW provides no getter, so previous foreign state is not restored.
Neither `fftw_cleanup` nor `fftw_cleanup_threads` is called. The bounded library
cache intentionally stays resident until process exit. **Uncoordinated planning,
destruction, wisdom, time-limit or thread-setting changes by foreign FFTW users
in the process are outside this guarantee.**

## Checks

`cargo test -p pencil-fftw` runs host-only validation/load-error checks.
`cargo test -p pencil-fftw -- --ignored` explicitly requires BOTH native runtimes
and fails if either is unavailable. Native checks use independent direct DFTs
and cover both precisions, all rigors with bounded planning, both directions,
odd/even/unit sizes, batches, IP/OOP, offset buffers, preservation/error paths,
arbitrary spectra, raw normalization and shared-plan concurrency with requested
counts 1/2/3. Native wisdom tests prove reuse using public wisdom-only options
after all original plans drop, precision isolation, existing-plan validity after
forget, invalid/NUL input, and concurrent planning/execution/wisdom.
Fresh-process traces cover serial-first, each wisdom operation first, and
threaded-first ordering in both precisions. Missing-library/symbol children
prove permanent base-only planning and wisdom without native initialization or
setters. Fatal-state injection occurs only after real successful initialization;
it does not simulate a genuine native out-of-memory failure. Reset checks use a
raw WISDOM_ONLY probe without setting the count before observation.
