# Pencil Arrays for Rust

A row-major, MPI-distributed multidimensional array foundation inspired by
PencilArrays.jl, with a separately layered distributed FFT implementation
planned on top.

The repository currently contains the `pencil-array` core crate. The Array
layer is intentionally independent of RustFFT, RealFFT, FFTW, and any
FFT-specific API.

`Pencil` describes spatial distribution. `PencilArray` owns one layout and
one local buffer. `ManyPencilArray` owns a buffer large enough for several
registered layouts and exposes only its active layout. Array shapes use
logical order `[extra..., spatial...]`; their row-major buffers use memory
order `[extra..., permuted spatial...]`. The core crate does not redistribute
or transpose data.

## Prerequisites

- Rust stable, with a minimum supported Rust version of 1.85
- A C MPI implementation such as Open MPI or MPICH
- `mpicc` and `mpiexec` available on `PATH`
- libclang and its C development headers, required by bindgen while building
  the MPI bindings

## Verification

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --lib --locked
cargo test -p pencil-array --test ui --locked

mpiexec -n 1 cargo test -p pencil-array --test topology --locked -- --nocapture --test-threads=1
mpiexec -n 4 cargo test -p pencil-array --test topology --locked -- --nocapture --test-threads=1
mpiexec -n 1 cargo test -p pencil-array --test pencil --locked -- --nocapture --test-threads=1
mpiexec -n 4 cargo test -p pencil-array --test pencil --locked -- --nocapture --test-threads=1
mpiexec -n 1 cargo test -p pencil-array --test array --locked -- --nocapture --test-threads=1
mpiexec -n 4 cargo test -p pencil-array --test array --locked -- --nocapture --test-threads=1
mpiexec -n 1 cargo test -p pencil-array --test many --locked -- --nocapture --test-threads=1
mpiexec -n 4 cargo test -p pencil-array --test many --locked -- --nocapture --test-threads=1

cargo doc --workspace --no-deps --locked
cargo test --workspace --doc --locked
```

The topology constructors are collective over an MPI intracommunicator. Run
each integration-test binary with one MPI initialization per process, and drop
all topologies and arrays before MPI finalizes.

## Design and plans

- `docs/superpowers/specs/2026-09-11-pencil-arrays-rust-port-design.md`
- `docs/superpowers/plans/2026-09-11-pencil-array-core-implementation.md`
