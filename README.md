# Pencil Arrays for Rust

A row-major, MPI-distributed multidimensional array foundation inspired by
PencilArrays.jl, with a separately layered FFT implementation.

The workspace contains the `pencil-array` core crate and the local FFT
`pencil-fft` crate. `pencil-array` is intentionally independent of RustFFT,
RealFFT, FFTW, and any FFT-specific API. The local `pencil-fft` path accepts
flat slices, uses RustFFT/RealFFT, and is independent of MPI and
`pencil-array`. Local C2C provides unnormalized forward and positive-sign
`backward` transforms plus normalized `inverse`. An opt-in
`pencil-fft/distributed` feature adds out-of-place, input-preserving and
single-buffer in-place distributed C2C FFTs, including raw positive-sign
`backward`, and out-of-place distributed R2C/C2R forward, normalized inverse,
and raw backward over checked Alltoallv or point-to-point transitions. Its
R2C/C2R API also provides a state-checked single-allocation in-place buffer.
Local out-of-place and packed single-allocation in-place R2C/C2R are also
available. `LocalR2rPlan` provides all eight
FFTW-compatible DCT/DST-I-IV kinds for real and complex `f32`/`f64`, with
out-of-place and in-place execution. Its `forward`/`backward` operations use
raw unnormalized FFTW conventions, while `inverse` is normalized by the
logical transform factor. Each line uses at most `8n` complex embedding values
plus queried native scratch, using the existing RustFFT backend without MPI.
`LocalDhtPlan` provides the local self-paired discrete Hartley transform for
the same four scalar types, with the same caller-owned line/scratch and
out-of-place/in-place conventions.
Distributed R2R uses the same eight kinds per logical axis, with `None` identity
stages, for real and complex `f32`/`f64` over the existing checked transports.
`DhtPlan` provides selected-axis self-paired discrete Hartley transforms without
exposing the legacy R2R axis-kind constructors.

`Pencil` describes spatial distribution. `PencilArray` owns one layout and
one local buffer. `ManyPencilArray` owns a buffer large enough for several
registered layouts and exposes only its active layout. Array shapes use
logical order `[extra..., spatial...]`; their row-major buffers use memory
order `[extra..., permuted spatial...]`. `get_global` is a noncollective,
local-only lookup: remotely owned or out-of-range coordinates return `None`.
`Pencil::local_grid` borrows one complete global coordinate slice per spatial
axis and iterates this rank's spatial coordinates in physical memory order;
extra batches repeat that spatial grid. `LocalTransposePlan` provides
process-local memory-axis permutations. `AllToAllvTransposePlan` and
`PointToPointTransposePlan` provide checked distributed redistribution through
out-of-place views and shared-storage in-place execution. The optional
`pencil-fft/distributed` feature composes these checked transitions into
out-of-place and in-place distributed C2C forward/inverse/backward transforms and
out-of-place plus single-allocation in-place R2C/C2R; this does not change
the MPI-free local FFT default.
`C2cPlan` and `R2cPlan` provide matching selection-aware shape, pencil, and
array constructors through `AxisSelection<N>`, plus the existing
`from_*_with_method` forms. `AxisSelection::all()` is the default; its
`from_indices` input is validated for duplicates/out-of-bounds values and its
order is ignored. The legacy constructors retain the Alltoallv default; R2C
output reduces the selected reduction axis to `n/2+1`.

Alltoallv and point-to-point construction and execution are collective: every
rank must use the same source communicator context, API, order, `T`, and
correct `Equivalence`. Source and destination pencils must share the same
`MpiTopology` object and global shape and differ in exactly one ordered
decomposition position. Descriptor checks catch common mismatches but do not
replace that communicator, collective-order, type, or `Equivalence` contract.

`TransposeError`, `TransposeWorkspace<T>`, and
`TransposeWorkspaceRequirements` are the canonical shared API. The older
`AllToAllvTransposeError`, `AllToAllvTransposeWorkspace<T>`, and
`AllToAllvTransposeWorkspaceRequirements` names are compatibility aliases for
the same types. `workspace_requirements` and
`TransposeWorkspace::from_vecs` are noncollective and do not call MPI;
execution checks initialized `len` (not capacity), does not resize or
reallocate workspace vectors, and checks count/length/displacement/offset
limits before payload communication. Ordinary out-of-place preflight errors
preserve the source, destination, and workspace. Ordinary in-place preflight
errors preserve the array state, active data, and workspace for both
Alltoallv and point-to-point. Point-to-point uses a fixed internal tag on the
topology-owned changed-axis context, posts all receives before sends, waits for
every request, and must not overlap unfinished transposes on that context. MPI
failures, arbitrary panics, and process loss do not guarantee global recovery;
an unfinished request scope may abort.

## Global array collectives

`pencil-array` provides collective `global_sum`, `global_min`, `global_max`,
`l2_norm`, `any`, `all`, `sum_by`, and `norm_by` operations, plus root
`gather`. Numeric reductions use the sealed `i8`-through-`u64`, `f32`/`f64`,
and complex scalar set; integer sums are checked and floating-point NaN and
infinity results follow the documented IEEE policy. Methods are available on
`PencilArray` and `PencilArrayView`; free
functions take a validated view. `ManyPencilArray` intentionally has no direct
collective methods: coordinate `active_view()` validity across ranks first,
then call the corresponding view method or free function. Gather returns
logical `[extra..., spatial...]` row-major order on the root and `None`
elsewhere; all ranks validate descriptors, counts, and root allocation before
payload communication.

## Distributed R2C/C2R FFT

`R2cPlan<R, N, M>` accepts a canonical real input pencil and a non-empty
`AxisSelection<N>`. The largest selected Rust axis `r` is the real-to-complex
axis and its extent `n` is reduced to `n/2+1`; selected axes below `r` use
complex FFTs and every unselected axis is an identity stage. The final output
uses decomposition `[1..=M]` and reversed spatial memory order by default;
`DistributedLayout::permute_dims = false` keeps identity memory order and uses
strided line kernels. The route nevertheless always contains all `N` stages
and `N-1` transitions, so
unselected axes remain real-prefix or complex-suffix transposes rather than
being skipped. `allocate_workspace` is noncollective; coordinate any local
allocation failure before the next collective. Non-last real axes add an
optional real intermediate and real transpose workspace. `forward`, normalized `inverse`, and unnormalized positive-sign `backward`
are exposed in both out-of-place and state-checked single-allocation in-place
forms. Empty R2C selections are collectively rejected before
planning. The constructors and operations require the same communicator
context, API/order, scalar type, selection, method, and layouts on every rank.
Legacy constructors use Alltoallv; point-to-point uses the fixed `0x5054` tag
and must not overlap unfinished transposes on its context. The in-place array
exposes only the state-matching real or complex `PencilArrayView`; its one
allocation is recast safely between separate real-prefix and reduced-complex
suffix registries. Forward processes packed boundary rows back-to-front and
C2R processes them front-to-back. Initial preflight errors preserve the array;
after start, an error, invalid spectrum, or panic leaves it poisoned. In-place
operation words are 25/26/27; 18..24 remain reserved for distributed R2R.

The inverse performs the transverse complex inverse stages first, then
validates each extra batch and constrained DC/Nyquist plane. It accepts a
plane when either `max(abs(imaginary)) <= 128 * min_subnormal_R * D` or its
imaginary L2 norm is at most `128 * epsilon_R * D` times its real L2 norm,
where `D = 1 + sum(ceil(log2(n_a)))` over selected axes below the real axis.
This is an explicit normwise-relative/componentwise-absolute acceptance
policy, not a formal RustFFT error bound. All constrained endpoint values
must be finite; odd lengths constrain DC only (for `n=1` the final bin is DC),
even lengths also constrain Nyquist, and interior bins have no blanket finite
policy. Nonfinite or materially non-real constrained planes return
`R2cError::InvalidSpectrum` before any real destination write; the workspace
may already have changed on that post-start error path, while source and
destination remain unchanged. `backward` uses the same relative endpoint
criterion, but its componentwise absolute threshold is the inverse threshold
times the product of the selected transverse extents, excluding the real axis
and extra dimensions. That finite-positive factor is collectively validated at
plan construction. A forward/backward pair scales by the product of all
selected spatial extents; identity axes do not contribute.

## Distributed C2C FFT

A C2C plan also accepts `AxisSelection<N>`. It always follows the canonical
full route from axis `N-1` through `0`, with one stage and transition per
axis. A selected stage performs its local FFT; an unselected stage is an
identity with no native plan or scaling. Thus an empty selection is a valid
identity transform whose selected-layout transposes still run. The output
pencil uses decomposition `[1..=M]` and the reversed permutation by default;
`DistributedLayout::permute_dims = false` keeps identity memory order and uses
strided line kernels. Inverse/raw-backward normalization includes only
selected axes. `AxisSelection::all()` preserves the legacy transform exactly.

Enable the feature in this workspace with
`cargo check -p pencil-fft --features distributed --locked`. The public
`C2cPlan` supports `N >= 2` and `1 <= M < N`, canonical identity input
pencils, exact extra shapes, reusable plan-bound
`C2cOutOfPlaceWorkspace`, and input-preserving forward, normalized inverse,
and raw positive-sign backward execution. Forward consumes the canonical input
layout and produces the selected output layout; both inverse and raw backward
consume that output layout and produce canonical input. Raw backward does not
normalize, so a forward/backward pair scales by the product of selected
spatial extents; identity axes and extra dimensions do not contribute. Construction and execution are collective
on the topology's Cartesian communicator; every rank must call matching
operations in order and select the same `TransposeMethod`. The selected method is appended to the
minimal checked C2C descriptor (after global shape, process grid, extra shape,
axis mask, value kind, precision width, and method), so a rank-local method
mismatch returns
`FftError::CollectiveDescriptorMismatch` before native FFT, output, workspace,
or in-place state changes. Each rank checks its actual source and destination
against the plan before a full-Cartesian preflight agreement, and no data is
changed before that initial agreement. Once execution starts, a resource or
later transition metadata failure may mutate the workspace; no general
allocation-free rollback is promised. Point-to-point uses the existing
changed-axis topology context, fixed internal `0x5054` tag, receive-before-send,
wait-all, and MPI failure contract; it reserves request metadata. Do not
overlap unfinished point-to-point transposes on that context. The
`distributed` API is not enabled by default.

## Distributed R2R DCT/DST

`R2rPlan<T, N, M>` accepts `[Option<R2rKind>; N]` in logical-axis order;
`None` keeps that canonical route stage as an identity. It preserves the
original global shape, uses the same `N` stages and `N - 1` checked
transitions, and produces the reversed output layout by default; an explicit
`DistributedLayout` can retain identity memory order with strided line
kernels. `forward`, normalized paired `inverse`, and raw paired `backward` are input-preserving and support
real or complex `f32`/`f64`. The in-place API uses a plan-bound
`ManyPencilArray` and the shared `Input -> Poisoned -> Output` state contract;
public `R2rState` reuses that completion-state enum. R2R descriptor agreement
includes scalar value kind, underlying precision, all
per-axis kind codes, and transport before local native planning or data writes.

Plan construction and transform calls are collective. In-place array/workspace
allocation and views are noncollective; callers must coordinate an allocation
failure before the next collective call. `R2rPlan::allocate_in_place` returns
an opaque `R2rInPlaceArray` and `allocate_in_place_workspace` returns its
plan-bound scratch. The array exposes only `state`, `view`, and `view_mut`:
`Input -> Poisoned -> Output` for forward and `Output -> Poisoned -> Input`
for normalized inverse or raw backward. Initial collective preflight errors
preserve state, data, and workspace; once execution begins, the array is
poisoned before its first write, and any later failure or panic leaves it
`Poisoned`, so callers reallocate it. A mutable output view may contain
arbitrary spectral data for either reverse operation.

## Distributed mixed-axis plans

`MixedC2cPlan<R, N, M>` and `MixedR2cPlan<R, N, M>` apply a typed
`[AxisTransform; N]` in canonical descending route order. `None` is an
identity, `Fft` is a complex FFT, `R2r(AxisR2rKind::Fftw(...))` selects a DCT
or DST, and `R2r(AxisR2rKind::Dht)` selects a Hartley transform. `MixedC2cPlan`
rejects `Rfft`; `MixedR2cPlan` requires exactly one `Rfft`, permits only
identity/R2R/DHT stages on the real prefix and identity/FFT/R2R stages on the
complex suffix, and reduces only the boundary axis. Both plans provide
input-preserving out-of-place forward/inverse/raw-backward operations,
plan-bound workspaces, checked single-allocation in-place arrays, both
Alltoallv and point-to-point transitions, and the same normalized/raw pairing
as the homogeneous plans. `MixedR2cPlan` validates finite DC/Nyquist planes
collectively before C2R output and preserves the poisoned in-place state after
post-start errors or panics; a representation handoff panic may discard the
backing owner. `DistributedLayout::permute_dims = false` selects the strided
per-line kernels. The mixed descriptors use distinct collective
operation words, scalar widths, reduced shape, axis transform codes, and
transport policy, so mixed and legacy calls cannot accidentally agree.

The existing `tools/fftw-reference` Julia/FFTW checker remains the independent
numerical reference for the component FFT, R2R, DHT, and R2C kernels. Mixed
plans are validated by composing those same one-axis references in route order.
The reusable oracle is `tools/fftw-reference/mixed_reference.jl`; focused MPI
coverage is in `crates/pencil-fft/tests/distributed_mixed.rs`.

## Local Julia/FFTW reference validation

The opt-in Milestone 9 checker generates temporary Julia 1.12.6/FFTW.jl
1.10.0 references and validates the distributed C2C, R2C/C2R, R2R, and DHT
forward/inverse/raw backward APIs at 1, 4, and 6 MPI ranks with both
transpose methods and both memory-layout policies. It covers exactly 82
fixtures and 136 valid case/layout combinations per policy (272 with both
policies), including mixed-axis C2C/R2C and real/complex R2R and DHT `f32`/`f64`.
The external comparison is explicitly opt-in; normal Rust tests need no Julia.
See [`tools/fftw-reference/README.md`](tools/fftw-reference/README.md) and run
`tools/fftw-reference/check.sh` only when Julia, FFTW.jl, and MPI are locally
available.

## Native collective I/O

`pencil-io` adds collective, decomposition-independent persistence for
`PencilArray` views without changing `pencil-array` or `pencil-fft`. The native
MPI-IO backend writes a versioned header and canonical little-endian row-major
payload, and uses MPI byte-subarray file views; readers stage and validate the
complete payload before mutating the destination. Writes use exclusive file
creation and a flushed commit marker, so incomplete and committed states are
reported separately. The format records type, width, logical shapes, writer
process grid, and writer permutation; writer layout metadata is provenance,
not a read-layout requirement.

Enable the optional native parallel HDF5 backend with
`pencil-io/parallel-hdf5`. It stores the same logical order and strict scalar
or `{r,i}` compound little-endian type under `/pencil_io_v1/data`, with typed
metadata attributes and the same incomplete/committed protocol. This feature
requires a parallel HDF5 installation discoverable by `pkg-config` or
`HDF5_DIR`; it is intentionally not enabled by default. Both APIs are
collective over the view's Cartesian communicator, and every rank must enter
calls in the same order without overlapping another operation on that
communicator. The HDF5 path explicitly creates a dataset-transfer property
list with `H5Pset_dxpl_mpio(..., H5FD_MPIO_COLLECTIVE)`; verify that the HDF5
and MPI libraries resolved by the build and runtime are the same ABI. The
lockfile's shared `mpi-sys` dependency is not ABI evidence: the pair is
validated only by inspecting the test executable's `ldd` output and recording
runtime MPI and HDF5 versions from that same environment.

Run the MPI-IO integration test at the required 1/4/6 rank matrix:

```bash
for n in 1 4 6; do
  timeout --foreground 120s mpiexec --oversubscribe -n "$n" \
    cargo test -p pencil-io --test mpi_io --locked -- --nocapture --test-threads=1
done
```

The integration cases cover every supported scalar family (`i8`/`u8`,
`i16`/`u16`, `i32`/`u32`, `i64`/`u64`, `f32`/`f64`, and complex `f32`/`f64`),
changed decompositions and permutations, committed-marker rejection, and
empty local ranks.

With parallel HDF5 configured, run the corresponding HDF5 matrix:

```bash
for n in 1 4 6; do
  timeout --foreground 180s mpiexec --oversubscribe -n "$n" \
    cargo test -p pencil-io --features parallel-hdf5 --test hdf5_io --locked \
      -- --nocapture --test-threads=1
done
```

The feature-enabled build checks used by CI are:

```bash
pkg-config --exists hdf5-openmpi
pkg-config --exists hdf5
cargo check -p pencil-io --all-features --all-targets --locked
cargo test -p pencil-io --no-default-features --test mpi_io --no-run --locked
cargo test -p pencil-io --features parallel-hdf5 --test hdf5_io --no-run --locked
cargo clippy -p pencil-io --all-features --all-targets --locked -- -D warnings
cargo doc -p pencil-io --all-features --no-deps --locked
timeout --foreground 120s mpiexec --oversubscribe -n 1 \
  cargo test -p pencil-io --features parallel-hdf5 --lib \
  post_cleanup_errors_preserve_destination_and_valid_commits --locked \
  -- --nocapture --test-threads=1
```

For the declared MSRV, run
`cargo +1.85.0 check --workspace --all-targets --locked` and
`cargo +1.85.0 check -p pencil-io --all-features --all-targets --locked` before the native
matrix.

After building the HDF5 test binary, inspect its resolved native ABI before
running the matrix (the exact binary name is printed by Cargo):

```bash
ldd target/debug/deps/hdf5_io-* | grep -E 'lib(hdf5|mpi|open-rte|open-pal)'
mpiexec --version
pkg-config --modversion hdf5-openmpi
```

## Prerequisites

- Rust stable, with a minimum supported Rust version of 1.85
- A C MPI implementation such as Open MPI or MPICH (for `pencil-array` and
  distributed tests; local `pencil-fft` tests do not require MPI)
- `mpicc` and `mpiexec` available on `PATH`
- libclang and its C development headers, required by bindgen while building
  the MPI bindings

## Verification

```bash
set -euo pipefail
cargo fmt --all -- --check
cargo test -p pencil-fft --no-default-features --locked
cargo tree -p pencil-fft --no-default-features --edges normal --locked | tee /tmp/pencil-fft-local-tree.txt
if grep -Eq '(^|[[:space:]])(mpi|pencil-array)([[:space:]]|$)' /tmp/pencil-fft-local-tree.txt; then exit 1; fi
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo clippy -p pencil-fft --features distributed --all-targets --locked -- -D warnings
cargo test --workspace --lib --locked
cargo check -p pencil-fft --features distributed --all-targets --locked
cargo test -p pencil-fft --features distributed --lib --locked
cargo test -p pencil-fft --features distributed --lib distributed::r2r::tests::in_place_error_panic_backend_and_short_workspace_poison_contracts --locked -- --ignored --nocapture

mpiexec -n 1 cargo test -p pencil-array --test topology --locked -- --nocapture --test-threads=1
mpiexec -n 4 cargo test -p pencil-array --test topology --locked -- --nocapture --test-threads=1
mpiexec -n 1 cargo test -p pencil-array --test pencil --locked -- --nocapture --test-threads=1
mpiexec -n 4 cargo test -p pencil-array --test pencil --locked -- --nocapture --test-threads=1
mpiexec -n 1 cargo test -p pencil-array --test array --locked -- --nocapture --test-threads=1
mpiexec -n 4 cargo test -p pencil-array --test array --locked -- --nocapture --test-threads=1
mpiexec -n 1 cargo test -p pencil-array --test many --locked -- --nocapture --test-threads=1
mpiexec -n 4 cargo test -p pencil-array --test many --locked -- --nocapture --test-threads=1
# The local operation is noncollective; keep a timeout to catch deadlocks.
timeout --foreground 120s mpiexec -n 1 cargo test -p pencil-array --test local_transpose --locked -- --nocapture --test-threads=1
timeout --foreground 120s mpiexec -n 4 cargo test -p pencil-array --test local_transpose --locked -- --nocapture --test-threads=1
timeout --foreground 120s mpiexec -n 1 cargo test -p pencil-array --test alltoallv_transpose --locked -- --nocapture --test-threads=1
timeout --foreground 120s mpiexec -n 4 cargo test -p pencil-array --test alltoallv_transpose --locked -- --nocapture --test-threads=1
timeout --foreground 120s mpiexec -n 6 cargo test -p pencil-array --test alltoallv_transpose --locked -- --nocapture --test-threads=1
timeout --foreground 120s mpiexec -n 1 cargo test -p pencil-array --test collectives --locked -- --nocapture --test-threads=1
timeout --foreground 120s mpiexec -n 4 cargo test -p pencil-array --test collectives --locked -- --nocapture --test-threads=1
timeout --foreground 120s mpiexec -n 6 cargo test -p pencil-array --test collectives --locked -- --nocapture --test-threads=1

cargo doc --workspace --no-deps --locked
cargo doc -p pencil-fft --features distributed --no-deps --locked
cargo test --workspace --doc --locked -- --show-output
cargo test -p pencil-fft --features distributed --doc --locked -- --show-output
# The distributed suites cover C2C, R2C/C2R, R2R, and mixed-axis plans over
# Alltoallv/P2P; distributed C2C, R2C/C2R, R2R, and mixed plans also have
# in-place APIs; local C2C
# and local R2C also have in-place APIs.
for n in 1 4 6; do
  timeout --foreground 120s mpiexec --oversubscribe -n "$n" \
    cargo test -p pencil-fft --features distributed --test distributed_c2c \
      --locked -- --nocapture --test-threads=1 || exit 1
  timeout --foreground 120s mpiexec --oversubscribe -n "$n" \
    cargo test -p pencil-fft --features distributed --test distributed_r2r \
      --locked -- --nocapture --test-threads=1 || exit 1
  timeout --foreground 120s mpiexec --oversubscribe -n "$n" \
    cargo test -p pencil-fft --features distributed --test distributed_mixed \
      --locked -- --nocapture --test-threads=1 || exit 1
done
```

Rustdoc tests cover usage examples and compile-time borrowing and visibility
restrictions. `LocalR2cPlan` preserves its source by copying one line at a time
into caller-owned initialized storage, uses caller-owned complex scratch, and
provides a packed single-allocation in-place API. It exposes the original real
length `n`, the reduced complex length `n/2+1`, and the shared native scratch
requirement; forward and backward are unscaled and inverse divides each line by
`n`. The in-place API owns one `Vec<Complex<R>>` and exposes only its
initialized real or complex prefix for the current state. Its inverse and
backward accept only strict-zero DC and, for even lengths, Nyquist imaginary
components; for odd lengths greater than one, the final-bin imaginary
component is unconstrained, while `n = 1` DC remains constrained. Ordinary
validation errors preserve data and workspace, while backend/resource panics
are not converted.

The topology constructors are collective over an MPI intracommunicator. Run
each integration-test binary with one MPI initialization per process, and drop
all topologies and arrays before MPI finalizes.

## Design and plans

- `docs/superpowers/specs/2026-09-11-pencil-arrays-rust-port-design.md`
- `docs/superpowers/plans/2026-09-11-pencil-array-core-implementation.md`
- `docs/superpowers/plans/2026-09-14-local-transpose-implementation.md`
- `docs/superpowers/plans/2026-09-14-alltoallv-transpose-implementation.md`
- `docs/superpowers/plans/2026-09-14-alltoallv-in-place-implementation.md`
- [Point-to-point transpose implementation plan](docs/superpowers/plans/2026-09-15-point-to-point-transpose-implementation.md)
- [Point-to-point in-place implementation plan](docs/superpowers/plans/2026-09-15-point-to-point-in-place-implementation.md)
- [Local C2C implementation plan](docs/superpowers/plans/2026-09-15-local-c2c-implementation.md)
- [Local R2C/C2R implementation plan](docs/superpowers/plans/2026-09-17-local-r2c-implementation.md)
- [Distributed C2C implementation plan](docs/superpowers/plans/2026-09-17-distributed-c2c-implementation.md)
- [Distributed C2C in-place implementation plan](docs/superpowers/plans/2026-09-18-distributed-c2c-in-place-implementation.md)
- [Distributed C2C point-to-point implementation plan](docs/superpowers/plans/2026-09-18-distributed-c2c-point-to-point-implementation.md)
- [Distributed R2C/C2R implementation plan](docs/superpowers/plans/2026-09-18-distributed-r2c-implementation.md)
- [Julia/FFTW cross-validation plan](docs/superpowers/plans/2026-09-19-fftw-cross-validation.md)
