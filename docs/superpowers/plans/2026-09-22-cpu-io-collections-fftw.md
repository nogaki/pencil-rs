# CPU parity: raw I/O, controls, collections, FFTW

User approved all four CPU-only categories. Baseline is main `dfde370041c7a83d07122d4c6d52eae6838e95e4`. GPU work, including draft PR29, remains deferred and must not be changed or merged.

## Boundaries

- Preserve old APIs, closed enums/struct literals, canonical formats, numerical conventions, ordinary preflight atomicity and post-start ownership/poison behavior.
- Every new collective starts with the existing two five-u64 MIN/MAX reductions. Agree exact metadata before native handles, typed payloads and allocation-dependent communication. Never validate separate collection members through independently entered collectives.
- CPU `pencil-fft` remains unsafe-free and default MPI-/FFTW-runtime-free; `pencil-array` stays FFT/I/O-independent. Native unsafe remains in audited private adapter boundaries.
- Exact Rust 1.85.0 checks use isolated targets, separately from normal stable. No Cargo target directory is shared between source worktrees.
- Astra implementation, independent Sol review, parent integration/publication. No GPU, global configuration changes, system package installation, or inspection/staging/removal of unrelated environment files.
- Existing September 2026 CI-success waiver remains applicable; do not claim CI passed or change billing/protection/concurrency.

## Scope

- [x] Explicit raw MPI binary input: caller-provided view determines type/shape; options carry byte offset and native/little/big byte order. Canonical logical `[extra..., spatial...]` row-major file interpretation, not automatic format/type/Julia-wire detection. Check offset/end/native count limits and `file_size >= end`, allowing unrelated prefix/trailing records. Reuse collective file views, stage/decode before cleanup, and update destination only after cleanup agreement. Old strict v1/named readers do not silently fall back to raw input.
- [x] I/O controls: additive MPI/HDF5 option-bearing APIs; real collective/independent payload calls, sorted validated MPI.Info hints passed to native open/views/FAPL, and actual HDF5 dataset-creation chunk properties. Metadata/open/commit/cleanup remain collective. Chunk rank/extents/native/product limits are checked; zero-size datasets remain supported. Existing APIs retain defaults. No MPI rank-chunk file format, dataset resizing/overwrite, compression framework, or arbitrary property system is required by this batch. External serialization of same-file writers remains a caller obligation and is documented.
- [x] Array collections: standard slices/Vec, no new owning array framework. FFT forward/inverse/raw backward (and existing in-place families) validate operation/count/all members/workspace before executing the first member, then reuse an existing workspace sequentially. Post-start failure reports member index and retains underlying semantics; no whole-collection rollback promise. I/O packs components into one `[component, extra..., spatial...]` payload; reads stage every member before any destination commit. Empty collections must be rejected or handled collectively, never fail locally before the common header. Rust standard loops suffice for allocation conveniences.
- [x] Optional FFTW backend: new MPI-free `pencil-fftw` adapter using existing RustFFT/RealFFT trait interfaces and dynamic `libloading = "=0.8.9"`; `pencil-fft/fftw` is opt-in. Real f32/f64 native C2C/R2C/C2R execution, correct separate C2C in-place/out-of-place plans, initialized planning buffers, `FFTW_UNALIGNED`, checked sizes, retained library lifetimes and serialized planner/destructor access. No `fftw_cleanup` and no caller-data MEASURE planning. R2R/DHT retain existing embeddings with FFTW-backed complex FFT kernels; do not claim direct native DCT/DST optimizations. Expose planning rigor (ESTIMATE/MEASURE/PATIENT/EXHAUSTIVE) and an optional Duration limit. The adapter controls its native planner time-limit setting under its lock and resets to NO_TIMELIMIT; foreign uncoordinated planning is outside its concurrency guarantee. Default constructors still choose RustFFT even with the feature enabled. Add backend constructors/reconfiguration with fresh cores and exact collective backend/rigor/time descriptors; native failures use new error wrappers, not new variants in old errors.

## Native environment and licensing

Open MPI 4.1.7 and parallel HDF5 1.10.7 are installed. `pkg-config fftw3` and local headers report 3.3.10, but the SONAMEs actually loaded by the parent report `fftw-3.3.8-sse2-avx` for both double and single precision. Tests must record actual loaded versions, not confuse header/pkg-config provenance with runtime ABI.

The installed FFTW copyright specifies GPL version 2 or later; FFTW also offers commercial licensing. New Rust wrapper source can remain MIT, with no FFTW source or binary vendored. Clearly document that enabling/distributing FFTW-backed applications requires compliance with the applicable FFTW GPL/commercial terms: dynamic loading is not a license exemption. Preserve the default RustFFT path and existing project license/attribution.

## Acceptance

New tests must fail a no-op/ignored-option implementation. Cover raw prefixes/trailing records/endian/offset overflow; mode/hint disagreement and invalid hints; real independent transfers; actual native chunk layout including zero extents; collection count/member mismatch before writes, indexed post-start errors and all-member read staging; native FFTW flags/time limits, alignment/aliasing, missing libraries/symbols/null plans, concurrent plan lifetime rules, arbitrary spectra, inverse/raw normalization, source preservation, both precisions, all CPU transform families, signs/layouts/transports and in-place behavior.

Run default/all-feature checks, format, Clippy, units/rustdoc, exact 1.85.0 builds, MPI 1/4/6 matrices and independent direct-DFT/R2R/Julia oracles. Native FFTW runtime tests are opt-in so default users require no FFTW installation; this environment has both precision libraries and must actually run those tests. Keep existing reference fixtures and corruption checks. Parent verifies the combined tree and protects old enum compatibility.

## I/O implementation and verification

Implemented under `crates/pencil-io` only:

- `raw.rs`: `read_mpi_raw(path, view_mut, RawReadOptions)`, explicit canonical raw input, full global end checks, per-component endian conversion, physical staging and cleanup-before-commit. Requested and effective byte order are both agreed. All new entry points also agree decomposition axes without modifying the legacy descriptor or collective sequence.
- `options.rs`, `hdf5_options.rs`: private-field builders `MpiIoOptions`, `RawReadOptions`, `Hdf5ReadOptions`, `Hdf5WriteOptions`; non-exhaustive `MpiIoMode` and `RawByteOrder`. Hints retain duplicate entries until collective rejection, are sorted, length/NUL checked, and agreed as exact length-delimited bytes.
- `mpi_io.rs`, `named_mpi.rs`, `hdf5_io.rs`, `ffi.rs`: additive `read_*_with_options`, `write_*_with_options`, and named read/write/append variants. Independent MPI calls are actual `MPI_File_read`/`MPI_File_write`; HDF5 DXPL uses the requested native mode. An owned MPI info handle reaches open/view/FAPL and is released after use. Old entry points retain their original collective sequence and null-info/default transfer behavior. Chunked HDF5 creation uses DCPL and bounded native dimensions/products; maximum extents accommodate chunks larger than current/zero extents without providing resize APIs.
- `collections.rs`: `write_mpi_collection`, `read_mpi_collection`, `write_hdf5_collection`, `read_hdf5_collection`, and non-exhaustive `CollectionIoError`. Callers pass the Cartesian communicator explicitly and standard view slices. Empty input is rejected after the common header. Members must have identical local layout/extra shape on that communicator. One validated temporary `PencilArray` supplies the leading component axis; reads commit only after the combined native read and cleanup succeed. Member validation reports an index; shared-payload failures are not falsely attributed to a member. No named collection aliases or whole-file transaction guarantees are added.
- `lib.rs` documents required all-rank participation and external same-file writer serialization across jobs/communicators, including append.

Regression coverage lives in `tests/new_options.rs`, `tests/collections.rs`, and the existing single-initialization private unit/fault test in `lib.rs`. It includes all twelve raw scalar representations in all three byte orders, complex component order, prefixes/trailers, truncation/overflow, extra axes/permutations/empty ranks; option and old/new-entry mismatches with recovery; invalid/duplicate hints; actual HDF5 chunk inspection including zero extents; two/three-member payload oracles and read preservation. Test-only FFI instrumentation observes the native MPI payload entry points, the native MPI info arguments, and `H5Pget_dxpl_mpio` results, so merely accepting but ignoring controls fails tests. The raw cleanup-fault hook also verifies no destination mutation before final agreement.

Validation uses Open MPI 4.1.7 and parallel HDF5 1.10.7, with the worktree's own `target`. Commands/logs (outside tracked source):

- `cargo fmt --all -- --check`: `/tmp/cpu3-fmt.log`.
- `cargo check --offline -p pencil-io` and `--all-features`: `/tmp/cpu3-check-{default,all}.log`.
- `cargo clippy --offline -p pencil-io [--all-features] --all-targets -- -D warnings`: `/tmp/cpu3-clippy{,-default}.log`.
- Default package tests and all-feature rustdoc tests: `/tmp/cpu3-test-default.log`, `/tmp/cpu3-doc.log` (no rustdoc examples in this crate). `RUSTDOCFLAGS='-D warnings' cargo doc --offline -p pencil-io --all-features --no-deps` also passed: `/tmp/cpu3-rustdoc.log`.
- All-feature unit/fault, old MPI/HDF5/named, and new options/collection binaries at 1/4/6 ranks: `/tmp/cpu3-matrix-summary.log`, per-run `/tmp/cpu3-final-*.log`. The built-in post-cleanup/commit fault driver is included; no separate external native-fault interposer is present in this worktree.
- Exact `RUSTUP_HOME=/tmp/pencil-rs-rustup-msrv.0vuEDM cargo +1.85.0 build --offline -p pencil-io --all-features --tests` uses a fresh `/tmp/pencil-cpu3-io-final-msrv.*` target, recorded in `/tmp/cpu3-final-msrv-target.txt`; log `/tmp/cpu3-final-msrv.log`.

These are local checks, not a CI claim or a substitute for the parent's independent review/publication. Root manifests, README, CI, GPU code and FFT work remain untouched.

## Integrated implementation and review

FFT collections cover all six CPU families and reuse the complete stored plan
descriptor, including the optional backend configuration. Member execution panic
is explicitly fail-stop: Rust-owned guards unwind, then the collection invokes
MPI abort; it does not insert an unknown-phase recovery Allreduce. Ordinary
errors retain indexed collective reporting. Asymmetric panic subprocess tests
require exit 86 plus specific markers and reject timeouts.

The native adapter and consumer use the existing RustFFT/RealFFT traits. The
adapter honors the RustFFT short/empty-buffer panic contract; the public local
pencil FFT wrappers retain their existing empty-batch success by returning before
calling that trait. All four local and six distributed families support explicit
FFTW selection, and native backend/direction configuration order is preserved.
Closed legacy errors are unchanged and have exhaustive compile checks.

Independent Sol reviews approved I/O, collection progress/whole-preflight, native
FFI lifetime/aliasing/initialization, and full consumer integration. The reviews
also led to isolated old/new I/O API mismatch tests (default options, so option
disagreement cannot mask dispatch errors). Parent final integrated verification
and publication remain separate from worktree logs.

The official reference runner accepts `PENCIL_FFT_BACKEND=rustfft|fftw` and
`PENCIL_FFT_DIRECTION_ORDER=native-first|directions-first`, validates them before
launch, and enables the actual requested backend. It preserves all 87 fixtures,
312 configurations and 30 corruption checks. No temporary edited runner or
feature-enabled-but-still-Rust fallback is required.

## Work ownership

- I/O worktree: raw read, native controls/options, collection I/O, tests and this plan.
- Collection FFT worktree: collection methods/tests and minimal private validation seams; backend integration follows after the native adapter is ready.
- FFTW worktree: new native adapter crate, its tests/licensing notes and workspace manifests/lockfile; no existing FFT implementation edits initially.
- Parent: cross-worktree integration, public README/CI coverage, independent review, exact tested-tree publication.
