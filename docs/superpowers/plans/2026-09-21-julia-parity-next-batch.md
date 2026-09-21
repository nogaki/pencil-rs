# Next Julia parity batch

User approved all seven categories after comparison with PencilArrays 0.19.11 and PencilFFTs 0.15.3. Baseline: `b4325a879f6b58afb643ec9199cd6feac78f4d8d` (tree `26697ef5c41730a8833aa42c4312a2135fd2e815`). This is feature work, not a prerequisite whole-repo simplification pass.

## Boundaries

- Preserve existing public enum variants and struct-literal APIs, numeric conventions, initial-preflight atomicity, source preservation, and post-start poison/ownership contracts.
- Keep default local CPU FFT MPI-free, array independent of FFT/I/O, and `pencil-fft` unsafe-free. Exact MSRV is Rust 1.85.0, independently of normal stable.
- New collectives begin with the existing two five-word MIN/MAX header reductions. Agree exact descriptors before entry-specific collectives, allocation-dependent communication, or payloads. Callback semantics/associativity remain caller responsibilities.
- Workers do not publish git changes. Parent integrates, reviews, verifies and publishes. Every worktree and MSRV build has a separate Cargo target directory.
- Luna initial implementation, Astra completion/regression tests, and independent Sol review. The attempted `gpt-5.6-luna-max` model was rejected by the account; the user was informed of both the configured Luna fallback and Astra completion.
- CI billing/protection/concurrency settings are not changed. The existing September 2026 CI-success waiver is retained; do not claim CI success.

## Features and acceptance

- [x] Checked pointwise/broadcast operations. Closures, output-specified and in-place forms; identical spatial layout/topology, equal extra rank with singleton-extra broadcasting, scalars via captures. Validate mapping/overflow before writes. No implicit spatial redistribution or expression-template engine. Callback panic may leave partial output; ordinary validation errors do not. New error type instead of changing `ArrayError`.
- [x] Multi-input reductions. Two-input custom associative map/reduce with explicit neutral init, exact scalar/init/layout descriptor, deterministic rank-partial all-gather/fold (document O(P)); callbacks must not panic or call MPI. Sealed existing output scalars. Add checked/IEEE-consistent zip sum/norm and mapped min/max through existing policies, not unchecked generic arithmetic. No direct Many collective wrappers.
- [x] Named multiple datasets and append. Add a new named I/O API/error type; existing exclusive-create v1 APIs stay unchanged. Append adds a new named dataset, not in-place extent growth or overwrite. MPI container has distinct framing and independently committed append-only records; no supposedly atomic mutable global index. Prior committed records remain readable across incomplete tails; appending after an invalid tail may be refused, never truncate it. HDF5 opens existing files RDWR, stores named records with per-dataset metadata/commit and injective UTF-8 name encoding, and never replaces earlier datasets. Agree full names, reject duplicates before mutation, stage read results until cleanup succeeds. Preserve CommitUncertain/fail-stop distinctions; do not promise crash-atomic HDF5 journaling.
- [x] Per-axis FFT/BFFT signs. Add a direction value/configuration API that creates a fresh plan core, covering C2C and both mixed plans. Forward uses the configured complex sign; inverse/backward use its pair, only inverse normalizes. Positive-sign forward's inverse uses negative FFT followed by scaling. RFFT/R2R/identity stages do not silently acquire a Fourier sign. Keep closed `AxisTransform` and `DistributedLayout` unchanged; descriptors include directions, old workspaces reject the new core. Update independent Julia fixtures and corruption checks without shrinking the matrix.
- [x] Per-stage timing. Fixed/const-N records, `std::time::Instant`, local-rank measurements for transform/pack/collective-or-receive-wait/send-wait/unpack/total. No implicit timing reductions or fallible post-start reporting allocation. Distinct new operation words for profiled entry points. Overlapped intervals need not sum to total.
- [x] Genuine out-of-place P2P send/FFT overlap (matching Julia's overlap path; in-place profiling remains supported). Nested receive scope inside send scope, all reservations agreed before packing/posting, receives before sends, receive wait and scope end before unpack/callback, next FFT before send wait. Requests never escape their scope. Wait sends on callback error/unwind, then perform the same panic-status agreement on all ranks before a local panic is resumed or peers return a typed peer-panic error. Normal callback errors are collectively agreed before subsequent payloads; peer-panic paths must not enter another unmatched outer agreement. Keep existing synchronous entries unchanged and test scheduling, recovery and ownership.
- [x] Concrete CUDA implementation (draft; hardware release gate remains open). New `pencil-cuda` with dynamic CUDA/cuFFT loading (`libloading = "=0.8.9"`), resident buffers and actual GPU FFTs, no disguised CPU fallback. Local C2C/R2C/C2R for f32/f64 and normalized/raw directions; optional `distributed` uses checked host-staged MPI, never root-gather FFT. Reuse immutable stage geometry through additive accessors rather than exposing internal executors. One audited unsafe FFI boundary, retained library/context lifetimes, checked sizes/strides/device ownership and statuses. CPU crates/default behavior unchanged.

## Verification

Use existing test style and independent oracles. Add feature-specific normal/negative checks, mismatched old/new API entries, empty ranks/batches, both memory policies/transports where applicable, arbitrary inverse input, source/input preservation and post-start poison checks. Run format, Clippy, default/all-feature builds, unit/rustdoc tests and exact 1.85.0 checks; MPI 1/4/6 rank matrices and the expanded Julia reference runner use timeouts and explicit executed-test markers. Named I/O tests must show prior datasets survive recoverable failed appends/reads and duplicate names cause no writes.

The current environment has Open MPI 4.1.7 and parallel HDF5 1.10.7, but no visible CUDA toolkit, CUDA libraries or NVIDIA device tools. CUDA missing-runtime/validation checks and compile checks run here. Real GPU numerical and GPU+MPI checks are explicit opt-in tests that fail if requested without their runtime; hardware execution remains unverified until run on CUDA hardware. No system package installation or global configuration changes are authorized/needed.

### Array worktree verification

The array checkboxes above are backed by `crates/pencil-array/tests/array_ops.rs`
(one MPI initialization). It covers broadcast order/permutation and zero extents,
non-Clone in-place arrays/views, untouched output on shape/topology errors,
rank-fold instrumentation, exact neutral-bit disagreement, old/new API mismatch,
rank-local invalid topology, actual ZST-driven allocation/count failures and recovery,
mapped extrema, checked zip overflow, and stable real/complex norms.
`array_ops`, `collectives`, and `array_access` pass at 1/4/6 ranks under timeout
with one passed marker per rank. Workspace format, Clippy `-D warnings`, unit/docs,
and exact Rust 1.85.0 default all-target check/new integration build pass.
Worktree logs are in `logs/`; the isolated MSRV target is recorded in
`logs/msrv-target.log`. No CI or hardware claims are made by this array worktree.

- [x] Review blocker: `sum_by` prepares/agrees U integer partial storage before
  mapping (`collectives.rs:1336-1341`), then uses `sum_values_prepared`; removed
  the now-unused late-preparation helper. ZST oversized `sum_by` regression
  (`tests/array_ops.rs:246-264`) checks allocation error on the nonempty rank,
  peer agreement, zero callbacks on every rank, and immediate successful recovery.
  This exercises mapped-storage exhaustion, not injected partial-storage OOM.
  Reran workspace fmt/Clippy (`-D warnings`), `array_ops` and legacy `collectives`
  at 1/4/6 MPI ranks (120-second timeouts), exact Rust 1.85.0 workspace all-target
  check and both integration-test builds: all passed. Logs: `logs/review-*.log`,
  executable build records: `logs/review-build.jsonl`; targets: this worktree's
  `target` and `target/msrv-review`. No commits/pushes.

## Integration and release gates

The six CPU categories have implementations, focused MPI checks, and independent
Sol review. The integration additionally preserves the original closed `FftError`
by returning the new `FftOverlapError<E>` only from overlap APIs; an exhaustive
legacy-enum compatibility test guards this boundary. Parent full-tree validation
is recorded separately before publication.

CUDA local and distributed code is implemented on the separate CUDA branch,
including real cuFFT calls, resident arrays, host-staged MPI, normalization PTX,
context/resource safeguards, and host/MPI checks. The implementation checkbox
above is not release approval: no real CUDA numerical/ABI/PTX execution is
available here. Publish this branch as a draft until the explicit hardware
matrix passes, not as a validated GPU release.

- [ ] CUDA hardware release gate: local numeric/resource tests and the 1/4/6-rank
  distributed CUDA matrix, including valid peers plus one invalid device ordinal,
  must pass on actual hardware before release.

The CUDA implementation does not claim GPU R2R/mixed transforms, CUDA-aware MPI,
distributed in-place execution, padded real in-place execution, or allocation-free
GPU execution. Existing CPU implementations of those transform families remain.

## Worktree ownership

- Array: pointwise, collectives, associated tests/docs and this plan.
- I/O: named MPI/HDF5 formats/FFI/tests and crate docs.
- FFT: signs, timing, P2P overlap, array transpose glue, stage geometry accessors and Julia/reference tests.
- CUDA: new crate and workspace manifests/lockfile; local backend first, distributed integration after geometry interfaces settle.
- Parent: resolve additive export conflicts, integrate README/CI coverage, independent reviews, final checks and publication.
