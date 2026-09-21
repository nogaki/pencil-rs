# pencil-cuda

Local CUDA and optional host-staged distributed CUDA. CUDA driver and cuFFT are dynamically loaded; building needs neither CUDA headers nor NVRTC. There is no CPU FFT or host-scaling fallback. Default dependencies are MPI-free; `distributed` enables checked pencil routing and MPI.

## Local API

- `CudaDevice::with_ordinal(usize)` selects a device explicitly; `new()` selects ordinal 0. Rank/node device mapping belongs to the caller. `CudaDevice` owns a private context (not a primary context). Operations restore the ambient context, including on Rust unwind. Children retain the context and loaded libraries. Objects are intentionally thread-local (`Rc`, not `Send`/`Sync`).
- `CudaBuffer<T>` is zero-initialized, with typed upload/download. Writes require mutable references. Empty buffers/batches are valid; transform length zero and native/byte-count overflow are rejected before cuFFT calls.
- `C2CPlanF32/F64`: `execute_forward` (negative sign), `execute_inverse` (positive sign, divided by n), `execute_backward` (raw positive sign). `execute(..., true)` means normalized inverse. Explicit `execute_in_place(..., inverse)` and `execute_backward_in_place` support genuine aliased C2C execution.
- `R2CPlanF32/F64::execute` is forward. `C2RPlanF32/F64::execute` and `execute_inverse` are normalized; `execute_backward` is raw. Out-of-place real transforms use device-to-device scratch to preserve inputs. C2R checks only DC/Nyquist imaginary components, strictly equal to zero, before output writes; non-finite real components are not rejected.
- Normalization uses bundled f32/f64 PTX through `cuModuleLoadData`, `cuModuleGetFunction`, and `cuLaunchKernel`. Unsigned 64-bit bounded grid-stride loops cover scalar tails. No CUDA compilation occurs at build time. Execution synchronizes before returning, and resource drops synchronize before release.

True padded real/complex in-place R2C/C2R state is **not implemented**. The real APIs are out-of-place only; they do not pretend scratch execution is in-place. Module loading currently occurs per normalized execution rather than being cached.

## Safety ceiling

Only 64-bit hosts are supported (compile-time gate). cuFFT symbols use typed C pointers, with CUDA float2/double2-compatible complex layouts; integer device addresses are converted only inside the FFI boundary. Host layout checks do not prove the native ABI.

Context creation/push installs a restoration guard before inspecting the result. The guard captures the prior context and verifies it after popping, even if CUDA reports an asynchronous error after changing state. Nested calls already in the private context do not push duplicate entries. If querying or restoring the prior context cannot be confirmed, the process **aborts**, including during unwinding; continuing with a possibly wrong ambient context is forbidden.

A failed synchronization permanently poisons the device. Further native operations are rejected; affected buffers, plans, PTX modules, the private context, and both loaded libraries are deliberately retained until process exit. Cleanup failures also retain the context/libraries. There is no recovery/retry claim: repeated failures can exhaust GPU/host resources, so callers should terminate/restart after such an error. Successful synchronization retains normal cleanup. This is a conservative safety policy, not proof that failed synchronization means work remains active.

## Checks

```sh
cargo test -p pencil-cuda
cargo check -p pencil-cuda --all-features
cargo clippy -p pencil-cuda --all-targets --all-features -- -D warnings
```

Host tests unconditionally check shape/native/byte bounds, missing libraries/symbols, bundled kernel loop guards, and the default dependency boundary. Hardware tests compile in normal test runs, but are ignored. To explicitly run them:

```sh
cargo test -p pencil-cuda --test hardware -- --ignored --nocapture
cargo test -p pencil-cuda --lib -- --ignored --nocapture
```

Explicit runs fail if CUDA/driver/cuFFT is absent; they never silently skip. Tests compare f32/f64 C2C and real transforms against independent CPU direct-DFT oracles (no CPU FFT dependency): odd/even/n=1, batches, partial blocks, normalized/raw, arbitrary valid spectra, preservation, C2C in-place, wrong contexts, and endpoint rejection before writes. Ignored library tests check ambient context restoration and unwind, inject errors after actual successful create/push/pop calls (including genuine constructor failure), and inject synchronization failure to check that no buffer/plan/module/context release occurs and libraries remain loaded. A control case checks normal releases. These are native GPU checks, not ABI mocks.

Real-hardware status: **UNVERIFIED**. PTX execution and numerical results require an actual CUDA device. Host-validation logs are local build artifacts, not checked-in hardware evidence. This implementation remains draft-only until the hardware release gate below is satisfied.

## Distributed API (`--features distributed`)

`distributed::DistributedPlan<R, N, M>` supports f32/f64:

- `c2c(input, extra, selection, layout, signs, ordinal)` and `r2c(input, extra, selection, layout, ordinal)` collectively construct the GPU context and cached local cuFFT batch plans. CPU `C2cPlan`/`R2cPlan` constructors supply immutable checked stage geometry only; no CPU FFT execution occurs. Input is the CPU canonical identity-memory pencil, decomposed along the first M axes.
- `allocate_input()` (complex), `allocate_real_input()`, `allocate_output()` (complex), and `allocate_workspace()` are collective. Arrays own CUDA storage and `Arc<Pencil>`/`ExtraShape`. Workspaces are caller-owned and reusable, retaining per-stage host/device arrays and transport scratch. Different plans' workspaces are not interchangeable.
- `forward` / `inverse` / `backward` execute C2C; `forward_real` / `inverse_real` / `backward_real` execute R2C/C2R. Inverse divides by selected spatial extents; backward is raw. C2C signs and paired reverse signs follow constructor configuration. Backward signs on unselected axes are rejected.
- RFFT reduces the largest selected axis, including lengths 1 and 2. Full canonical routes and identity stages remain present. Both transpose transports, both `permute_dims` policies, partial selections, extra batches and empty ranks are supported. R2C requires at least one selected axis.
- Array `upload`/`download` are **local**, checked physical row-major transfers: extra dimensions first, then pencil memory axes. Callers must coordinate any errors in their own local preparation before entering the next collective. Prefer the collective plan allocation methods over local `CudaPencilArray::new` when preparing distributed execution.

See the compiled `distributed` module documentation for a minimal example.

### Protocol and failure boundary

Every public GPU collective begins with the same pair of five-u64 MIN/MAX reductions. A distinct GPU namespace and operation word precede exact global descriptors (shape, grid, decomposition, permutation, extras, selected axes, signs, precision, kind, transport and layout). Local ranges, strides and device ordinals are deliberately not compared across ranks. CUDA setup/allocation/native errors are agreed across ranks before subsequent MPI payloads. No root gather or CUDA-aware MPI is used.

Sources are preserved. Initial descriptor/layout/context/workspace failures leave destination and workspace unchanged. Once execution starts, an error poisons the workspace; allocate a replacement to recover from ordinary validation failures. Destination upload failures can leave destination contents uncertain. Device synchronization/context failures retain the stricter local safety policy above, not a recovery promise. MPI failures, process loss, aborting allocation failures in dependencies, or inability to restore a native context do not promise global recovery.

Before C2R writes the destination, inverse complex suffixes are completed and DC/Nyquist endpoint planes are checked for finiteness and normwise absolute-or-relative imaginary noise **separately for every extra batch and endpoint plane**. Raw scaling adjusts the absolute floor. Only accepted planes are projected to exactly zero imaginary part for the strict native C2R check; the caller's spectrum is never projected or modified.

### Traffic and memory ceiling

This is host-staged, not a device-only transport: API source downloads once, each selected local stage packs on the host, uploads contiguous lines, runs cuFFT, downloads/scatters, and the final destination uploads once. MPI transposes operate on host staging. C2R additionally downloads input for its strict local endpoint check and uses a device-to-device preservation copy. Thus budget two full local payload transfers per complex/forward-real stage, three per reverse-real boundary, plus API input/output transfers (real and complex byte counts differ). Identity stages do not invoke cuFFT or stage transfers.

Per-stage host/device storage and cuFFT plans are retained in the plan/workspace, rather than a minimal two-buffer pool. Byte serialization/download vectors, the C2R preservation temporary and normalization PTX loading still occur during execution; this is not an allocation-free or overlap/performance claim. Host packing can be replaced by GPU kernels and CUDA-aware MPI only as a separate implementation.

| Feature | CPU | CUDA local | CUDA distributed |
|---|---|---|---|
| f32/f64 C2C, normalized inverse, raw backward | yes | yes | yes |
| f32/f64 R2C/C2R, normalized/raw | yes | out-of-place | out-of-place |
| Selected axes, extra batches, both transports/layouts | yes | contiguous batches | yes |
| Configurable C2C forward/paired reverse signs | yes | explicit forward/backward | yes |
| R2R, DHT, mixed transforms | yes | no | no |
| True in-place C2C | yes | yes | no |
| Padded real in-place state | yes | no | no |
| CUDA-aware MPI / GPU-only packing | n/a | n/a | no |

### Phase-2 checks

```sh
cargo fmt -p pencil-cuda --check
cargo clippy -p pencil-cuda --all-targets -- -D warnings
cargo clippy -p pencil-cuda --all-features --all-targets -- -D warnings
cargo test -p pencil-cuda --all-targets
cargo test -p pencil-cuda --all-features --all-targets
cargo test -p pencil-cuda --all-features --doc

# Build once, then launch the test executable, not cargo, under MPI.
cargo test -p pencil-cuda --features distributed --test distributed_host --no-run
# Use the executable path printed above:
for n in 1 4 6; do
  timeout 90 mpirun -n "$n" PATH_TO_DISTRIBUTED_HOST --nocapture
done
# Likewise run the library executable with this endpoint-only filter:
# timeout 90 mpirun -n N PATH_TO_LIB_TEST mpi_helper_collective_geometry_and_projection --nocapture

# On REAL hardware, also repeat the distributed_host loop above: its
# one-rank bad ordinal / peers-valid constructor case is NOT hardware-verified
# by passing on a host where every rank lacks CUDA.
# REAL hardware matrix: explicit requests fail without CUDA, never skip.
cargo test -p pencil-cuda --features distributed --test distributed_hardware --no-run
for n in 1 4 6; do
  timeout 1800 mpirun -n "$n" PATH_TO_DISTRIBUTED_HARDWARE --ignored --nocapture
done
```

Always-run tests cover physical packing/indexing, checked shapes/overflow, empty ranks, endpoint finiteness/norm policy and per-batch/per-plane isolation. Host MPI tests force invalid ordinals (also on GPU nodes), exercise exact descriptor failures and retry, and emit `CUDA_DISTRIBUTED_HOST_OK` / `CUDA_ENDPOINT_HOST_OK` markers. The ignored hardware matrix covers f32/f64, both transports/layouts, partial/empty selections, extra batches, empty ranks, odd/even/1/2 lengths, independent direct-DFT/RFFT references, arbitrary inverse spectra, raw scale, source preservation, endpoint rejection, wrong workspace/context and recovery. The ignored matrix also checks forward-vs-inverse and allocate_input-vs-allocate_output operation mismatches after successful GPU plan setup (ranks 4/6): every rank must return `Error::Descriptor`, preserve both arrays and leave the workspace unpoisoned, then reuse the same resources without recovery. Every numerical zip comparison first asserts equal lengths.

**Hardware release gate: UNVERIFIED. Publish GPU support only as a draft until the actual CUDA+MPI ranks 1/4/6 matrix and the existing one-rank-bad-ordinal/peers-valid constructor test pass on hardware.** Host-only success and ignored-test compilation do not satisfy this gate.
