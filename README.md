# Pencil Arrays for Rust

A row-major, MPI-distributed multidimensional array foundation inspired by
PencilArrays.jl, with a separately layered FFT implementation.

The workspace contains the `pencil-array` core crate and the local FFT
`pencil-fft` crate. `pencil-array` is intentionally independent of RustFFT,
RealFFT, FFTW, and any FFT-specific API. The local `pencil-fft` path accepts
flat slices, uses RustFFT/RealFFT, and is independent of MPI and
`pencil-array`. Local C2C and out-of-place R2C/C2R are available; distributed
FFT integration remains follow-up work.

`Pencil` describes spatial distribution. `PencilArray` owns one layout and
one local buffer. `ManyPencilArray` owns a buffer large enough for several
registered layouts and exposes only its active layout. Array shapes use
logical order `[extra..., spatial...]`; their row-major buffers use memory
order `[extra..., permuted spatial...]`. `LocalTransposePlan` provides
process-local memory-axis permutations. `AllToAllvTransposePlan` and
`PointToPointTransposePlan` provide checked distributed redistribution through
out-of-place views and shared-storage in-place execution. Distributed FFT
APIs remain next-stage work.

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
Alltoallv and point-to-point. Point-to-point uses a fixed internal tag on the topology-owned
changed-axis context, posts all receives before sends, waits for every request,
and must not overlap unfinished transposes on that context. MPI failures,
arbitrary panics, and process loss do not guarantee global recovery; an
unfinished request scope may abort.

## Prerequisites

- Rust stable, with a minimum supported Rust version of 1.85
- A C MPI implementation such as Open MPI or MPICH (for `pencil-array` and
  distributed tests; local `pencil-fft` tests do not require MPI)
- `mpicc` and `mpiexec` available on `PATH`
- libclang and its C development headers, required by bindgen while building
  the MPI bindings

## Verification

```bash
cargo fmt --all -- --check
cargo test -p pencil-fft --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --lib --locked

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
# The existing suite binary covers Alltoallv/P2P out-of-place and both in-place APIs.
timeout --foreground 120s mpiexec -n 1 cargo test -p pencil-array --test alltoallv_transpose --locked -- --nocapture --test-threads=1
timeout --foreground 120s mpiexec -n 4 cargo test -p pencil-array --test alltoallv_transpose --locked -- --nocapture --test-threads=1
timeout --foreground 120s mpiexec -n 6 cargo test -p pencil-array --test alltoallv_transpose --locked -- --nocapture --test-threads=1

cargo doc --workspace --no-deps --locked
cargo test --workspace --doc --locked -- --show-output
```

Rustdoc tests cover usage examples and compile-time borrowing and visibility
restrictions. `LocalR2cPlan` preserves its source by copying one line at a time
into caller-owned initialized storage, uses caller-owned complex scratch, and
provides no real in-place API. It exposes the original real length `n`, the
reduced complex length `n/2+1`, and the shared native scratch requirement;
forward is unscaled and inverse divides each line by `n`. Its inverse accepts
only strict-zero DC and, for even lengths, Nyquist imaginary components; for
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
