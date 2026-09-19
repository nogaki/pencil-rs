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
and raw backward over checked Alltoallv or point-to-point transitions. Local
out-of-place R2C/C2R are also available. `LocalR2rPlan` provides all eight
FFTW-compatible DCT/DST-I-IV kinds for real and complex `f32`/`f64`, with
out-of-place and in-place execution. Its `forward`/`backward` operations use
raw unnormalized FFTW conventions, while `inverse` is normalized by the
logical transform factor. Each line uses at most `8n` complex embedding values
plus queried native scratch; it adds no MPI or new dependencies. Distributed R2R
will follow separately.

`Pencil` describes spatial distribution. `PencilArray` owns one layout and
one local buffer. `ManyPencilArray` owns a buffer large enough for several
registered layouts and exposes only its active layout. Array shapes use
logical order `[extra..., spatial...]`; their row-major buffers use memory
order `[extra..., permuted spatial...]`. `LocalTransposePlan` provides
process-local memory-axis permutations. `AllToAllvTransposePlan` and
`PointToPointTransposePlan` provide checked distributed redistribution through
out-of-place views and shared-storage in-place execution. The optional
`pencil-fft/distributed` feature composes these checked transitions into
out-of-place and in-place distributed C2C forward/inverse/backward transforms
and out-of-place R2C/C2R; it does not change the MPI-free local FFT default.
`C2cPlan` and
`R2cPlan` provide matching `from_pencil_with_method`,
`from_array_with_method`, and `from_shape_with_method` constructors. The
legacy constructors retain the Alltoallv default; R2C output uses the original
shape with its final extent reduced to `n/2+1`.

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

## Distributed R2C/C2R FFT

`R2cPlan<R, N, M>` accepts a canonical real input pencil and returns a
complex output pencil whose original final extent `n` is reduced to `n/2+1`.
The initial reduced-complex stage keeps decomposition `[0..M)`, while the
final output uses `[1..=M]` and reversed spatial memory order. Other axes,
process-grid choices, and extra dimensions are retained. `allocate_workspace`
is noncollective; coordinate any local allocation failure before the next
collective. It owns one reduced-complex intermediate, checked transpose
storage, native scratch, and one real/complex line buffer. `forward`,
normalized `inverse`, and unnormalized positive-sign `backward` are exposed—
there is no real in-place API. The constructors and
operations require the same communicator context, API/order, scalar type,
method, and layouts on every rank. Legacy constructors use Alltoallv;
point-to-point uses the fixed `0x5054` tag and must not overlap unfinished
transposes on its context.

The inverse performs the transverse complex inverse stages first, then
validates each extra batch and constrained DC/Nyquist plane. It accepts a
plane when either `max(abs(imaginary)) <= 128 * min_subnormal_R * D` or its
imaginary L2 norm is at most `128 * epsilon_R * D` times its real L2 norm,
where `D = 1 + sum(ceil(log2(n_a)))` over original axes before the real axis.
This is an explicit normwise-relative/componentwise-absolute acceptance
policy, not a formal RustFFT error bound. All constrained endpoint values
must be finite; odd lengths constrain DC only (for `n=1` the final bin is DC),
even lengths also constrain Nyquist, and interior bins have no blanket finite
policy. Nonfinite or materially non-real constrained planes return
`R2cError::InvalidSpectrum` before any real destination write; the workspace
may already have changed on that post-start error path, while source and
destination remain unchanged. `backward` uses the same relative endpoint
criterion, but its componentwise absolute threshold is the inverse threshold
times the product of the original transverse extents, excluding the real axis
and extra dimensions. That finite-positive factor is collectively validated at
plan construction. A forward/backward pair scales by the product of all
original spatial extents.

## Distributed C2C FFT

Enable the feature in this workspace with
`cargo check -p pencil-fft --features distributed --locked`. The public
`C2cPlan` supports `N >= 2` and `1 <= M < N`, canonical identity input
pencils, exact extra shapes, reusable plan-bound
`C2cOutOfPlaceWorkspace`, and input-preserving forward, normalized inverse,
and raw positive-sign backward execution. Forward consumes the canonical input
layout and produces the reversed output layout; both inverse and raw backward
consume that output layout and produce canonical input. Raw backward does not
normalize, so a forward/backward pair scales by the product of global spatial
extents, excluding extra dimensions. Construction and execution are collective
on the topology's Cartesian communicator; every rank must call matching
operations in order and select the same `TransposeMethod`. The selected method is appended to the
minimal checked C2C descriptor (after global shape, process grid, extra shape,
and scalar width), so a rank-local method mismatch returns
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

Plan construction and transform calls are collective. In-place array/workspace
allocation and views are noncollective; callers must coordinate an allocation
failure before the next collective call. `C2cPlan::allocate_in_place` returns
an opaque `C2cInPlaceArray` and `allocate_in_place_workspace` returns its
plan-bound scratch. The array exposes only `state`, `view`, and `view_mut`:
`Input -> Poisoned -> Output` for forward and `Output -> Poisoned -> Input`
for normalized inverse or raw backward. Initial collective preflight errors
preserve state, data, and workspace; once execution begins, the array is
poisoned before its first write, and any later failure or panic leaves it
`Poisoned`, so callers reallocate it. A mutable output view may contain
arbitrary spectral data for either reverse operation.

## Local Julia/FFTW reference validation

The opt-in Milestone 9 checker generates temporary Julia 1.12.6/FFTW.jl
1.10.0 references and validates the distributed C2C forward/inverse/raw
backward and R2C/C2R APIs at 1, 4, and 6 MPI ranks with both transpose methods
and both precisions. It
covers 16 fixtures and 26 valid case/layout combinations per method and rank.
The external comparison is explicitly opt-in; normal Rust tests need no Julia.
See [`tools/fftw-reference/README.md`](tools/fftw-reference/README.md) and run
`tools/fftw-reference/check.sh` only when Julia, FFTW.jl, and MPI are locally
available.

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

cargo doc --workspace --no-deps --locked
cargo doc -p pencil-fft --features distributed --no-deps --locked
cargo test --workspace --doc --locked -- --show-output
cargo test -p pencil-fft --features distributed --doc --locked -- --show-output
# The existing suite binary covers distributed C2C forward/inverse/backward and R2C/C2R forward/inverse/backward over Alltoallv/P2P; only C2C has in-place APIs.
for n in 1 4 6; do
  timeout --foreground 120s mpiexec --oversubscribe -n "$n" \
    cargo test -p pencil-fft --features distributed --test distributed_c2c \
      --locked -- --nocapture --test-threads=1 || exit 1
done
```

Rustdoc tests cover usage examples and compile-time borrowing and visibility
restrictions. `LocalR2cPlan` preserves its source by copying one line at a time
into caller-owned initialized storage, uses caller-owned complex scratch, and
provides no real in-place API. It exposes the original real length `n`, the
reduced complex length `n/2+1`, and the shared native scratch requirement;
forward and backward are unscaled and inverse divides each line by `n`. Its
inverse and backward accept only strict-zero DC and, for even lengths, Nyquist
imaginary components; for
odd lengths greater than one, the final-bin imaginary component is
unconstrained, while `n = 1` DC remains constrained. Ordinary validation
errors preserve data and workspace, while backend/resource panics are not
converted.

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
