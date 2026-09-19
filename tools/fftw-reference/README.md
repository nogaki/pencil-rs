# Local Julia/FFTW reference validation

Milestone 9 is an opt-in checker for the existing distributed `pencil-fft`
C2C and R2C/C2R APIs. It changes no production FFT code, tolerances, CI, or
checked-in numeric data.

## Run

The pinned environment is Julia 1.12.6 with FFTW.jl 1.10.0. The native FFTW
version is recorded separately in each fixture. A C MPI launcher and GNU
`timeout` are required; each MPI run has a 120-second process-group timeout
with forced termination after a further 5 seconds. HUP/INT/TERM exit nonzero
before temporary-file cleanup. Julia and the launcher are executable names
or absolute paths:

```bash
PATH="$HOME/.cargo/bin:$PATH" \
JULIA_DEPOT_PATH="/tmp/pencil-rs-julia-depot:$HOME/.julia" \
JULIA_NUM_THREADS=1 JULIA_NUM_PRECOMPILE_TASKS=1 \
JULIA=julia MPIEXEC=mpiexec \
./tools/fftw-reference/check.sh
```

Set `JULIA` or `MPIEXEC` explicitly when they are not on `PATH`. The script
checks Julia 1.12.6 before package setup, copies `Project.toml` and
`Manifest.toml` (including the provider preference in the project) to a
private temporary project, instantiates there, and compares the resulting
manifest with the checked-in one. It never runs `Pkg` against this tracked
directory. An unset `JULIA_DEPOT_PATH` gets a private temporary depot; an
explicit depot is reused and never deleted by the runner. Package setup may
populate or precompile into its first entry. Startup files are disabled and
`JULIA_LOAD_PATH` contains only the temporary project and standard libraries;
user preferences are never edited. Open MPI/OpenRTE gets `--oversubscribe` unless
`PENCIL_FFTW_NO_OVERSUBSCRIBE=1` is set.

The runner generates 16 temporary, non-empty fixtures, builds the Rust test
once with `cargo test --no-run`, runs its ordinary parser self-check, then
runs the explicit ignored test through `cargo test -- --ignored` at 1, 4, and
6 MPI ranks. It verifies a test-run marker so a missing ignored test cannot
pass silently. It also runs a private corrupted-fixture copy and requires the
comparison error marker. Missing Julia, MPI, fixtures, or matrix members is a
failure, never a skip. The runner uses Cargo's selected toolchain and target
directory; it never selects an executable by filename or timestamp. Set
`RUSTUP_TOOLCHAIN=1.85.0` to run the reference matrix with an installed MSRV
toolchain.

## Isolated MSRV check

Run these commands from the repository root to check exactly Rust 1.85.0
without changing the global Rustup configuration. Setup may download the
toolchain and Cargo dependencies; MPI development tools are still required.

```bash
(
  set -euo pipefail
  WORK=$(mktemp -d)
  trap 'rm -rf -- "$WORK"' EXIT
  export RUSTUP_HOME="$WORK/rustup" CARGO_HOME="$WORK/cargo"
  export CARGO_TARGET_DIR="$WORK/target"
  rustup toolchain install 1.85.0 --profile minimal --no-self-update
  cargo +1.85.0 check --workspace --all-targets --locked
  cargo +1.85.0 check --workspace --all-targets --all-features --locked
)
```

Formatting, Clippy, and numerical tests use the normal project toolchain;
MSRV checks use the isolated 1.85.0 installation. Reference provenance is
printed by every generation run and recorded in every fixture.

## Fixture contract

Every file has this fixed order:

```text
PENCIL_FFTW_REFERENCE 1
runtime julia=1.12.6 fftw_jl=1.10.0 native=... provider=fftw
case CASE_ID
kind c2c|r2c
precision f32|f64
original_shape N0 N1 ...
extra_shape [optional positive extents]
section input real|complex COUNT
... COUNT values ...
end
section inverse_input complex COUNT
... COUNT values ...
end
section forward_expected complex COUNT
... COUNT values ...
end
section inverse_expected real|complex COUNT
... COUNT values ...
end
```

The four sections are counted and parsed in that order. Real values are one
finite number and are represented internally as a zero-imaginary complex
value; complex values are two finite numbers. Rust derives the reduced output
shape and topology grids and uses fixed precision-specific comparison bounds.
The generator fixes `ESTIMATE` and one FFTW thread; the Rust checker runs both
methods. These constants are not serialized repeatedly.
The parser rejects unknown, duplicate, reordered, trailing, malformed, empty,
non-finite, or incomplete data, zero dimensions, invalid ranks, overflowed
products, and counts before reserving section storage.

The serialized order is global logical Rust row-major `[extra..., spatial...]`.
Julia allocates `reverse([extra..., spatial...])`, transforms only spatial
axes, and asserts the actual input, reduced output, and result array sizes.
C2C uses independent complex forward and inverse inputs. R2C creates the
independent inverse spectrum from a real input, snapshots it before planning,
and executes C2R on a private copy. FFTW `ESTIMATE`, one thread, the `fftw`
provider, and native-version metadata remain explicit.

There are eight base cases, both precisions, and therefore 16 files:

- C2C: `[3,4]`, `[3,2,5]` with extras `[2,3]`, `[2,1,3,4]` with extra `[2]`.
- R2C/C2R: `[3,1]`, `[3,2]`, `[3,1,4]` and `[3,1,5]` with extras `[2,3]`,
  and `[2,1,3,3]` with extra `[2]`.

No fixture topology is serialized. For every file Rust runs all valid `M=1`
layouts for `N=2,3,4` and also `M=2` for `N=3,4`: 26 case/layout
combinations per transpose method per rank. Both Alltoallv and point-to-point
are checked. Small leading axes deliberately create empty local ranges.

The checker derives raw physical-local to global-logical offsets from each
pencil's permutation and local ranges, checks the input identity and output
reversed permutations plus expected decompositions, and fills/compares raw
slices without `get_local`. A Cartesian integer all-reduce proves every tiny
fixture element is owned exactly once. It also checks out-of-place source and
spectrum preservation and C2C in-place values.

## Short plan

Keep the Julia project locked and the generator deterministic; keep fixtures
temporary; use the existing ignored distributed test as the sole MPI runner;
keep this README as the canonical format, matrix, and execution contract.
