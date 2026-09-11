# Pencil Arrays for Rust

A row-major, MPI-distributed multidimensional array foundation inspired by
PencilArrays.jl, with a separately layered distributed FFT implementation
planned on top.

The repository currently contains the `pencil-array` core crate. The Array
layer is intentionally independent of RustFFT, RealFFT, FFTW, and any
FFT-specific API.

## Prerequisites

- Rust stable, with a minimum supported Rust version of 1.85
- A C MPI implementation such as Open MPI or MPICH
- `mpicc` and `mpiexec` available on `PATH`

## Verification

```bash
cargo test -p pencil-array --lib
mpiexec -n 4 cargo test -p pencil-array --test topology -- --nocapture
```

The multi-rank command will become available when the topology integration
suite is added.

## Design and plans

- `docs/superpowers/specs/2026-09-11-pencil-arrays-rust-port-design.md`
- `docs/superpowers/plans/2026-09-11-pencil-array-core-implementation.md`
