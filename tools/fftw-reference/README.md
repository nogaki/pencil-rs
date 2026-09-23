# Local Julia/FFTW reference validation

This opt-in checker validates the distributed `pencil-fft` C2C, R2C/C2R, R2R,
and DHT forward/inverse/backward APIs. C2C backward is the positive-sign,
unnormalized route from reversed output layout to canonical input layout; R2C
backward is the corresponding raw C2R route; R2R backward applies the paired
raw DCT/DST kind. It changes no production tolerances, CI, or checked-in
numeric data. Mixed-axis plans compose these independent one-axis FFTW/R2R/DHT
values in route order. Mixed R2C fixtures use exactly one `rfft` boundary;
the format is not an arbitrary multi-RFFT/BRFFT graph description. Their
focused MPI coverage is kept separate in
`crates/pencil-fft/tests/distributed_mixed.rs` for focused state/layout
coverage. The reusable Julia oracle is
`tools/fftw-reference/mixed_reference.jl`; it accepts Rust-axis transform
symbols and is included by the fixture generator for the mixed cases.

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

The default/unset `PENCIL_FFT_BACKEND` (or explicit `rustfft`) builds
`--features distributed`. Set `PENCIL_FFT_BACKEND=fftw` to build
`--features distributed,fftw` and actually configure every plan with
`with_fftw` (ESTIMATE). This requires installed native FFTW runtimes for
both f32 and f64; missing libraries fail the explicit run, never fall back
to RustFFT. Ordinary Cargo tests need neither Julia nor native FFTW.
The consumer runtime observed here is FFTW 3.3.8, while the Julia oracle
currently reports FFTW 3.3.11. These are observed runtime versions, not a
general constant or a promise that consumer and oracle versions match.

`PENCIL_FFT_THREADS` defaults to `1` and accepts canonical positive decimal
integers through `2147483647` (positive `c_int`; no signs, whitespace, or leading
zeros). Native plans use `PlanOptions.with_threads` and assert the resulting
plan's `requested_threads`, including after either direction configuration order.
This is a requested native planner thread cap, not observed utilization or a
speed guarantee. RustFFT accepts only `1`; larger values fail rather than
pretending to control its threads. The Julia oracle remains fixed at one FFTW
thread, independently of this selector.

`PENCIL_FFT_DIRECTION_ORDER` accepts `native-first` (default) or
`directions-first`, applying native planning before or after Fourier direction
configuration. Unknown values (including empty strings) for any selector
fail before any tool is launched; direct ignored Rust test invocations reject
them too. Run both native configuration orders with the same full checker:

```bash
export JULIA=/home/kosuke/local/bin/julia
export JULIA_DEPOT_PATH=/tmp/pencil-rs-julia-depot.Q6bSFs:$HOME/.julia
export JULIA_NUM_THREADS=1 JULIA_NUM_PRECOMPILE_TASKS=1 CARGO_BUILD_JOBS=2
export CARGO_TARGET_DIR=$(mktemp -d /tmp/pencil-reference-target.XXXXXX)
PENCIL_FFT_BACKEND=rustfft ./tools/fftw-reference/check.sh
PENCIL_FFT_BACKEND=fftw PENCIL_FFT_THREADS=2 PENCIL_FFT_DIRECTION_ORDER=native-first ./tools/fftw-reference/check.sh
PENCIL_FFT_BACKEND=fftw PENCIL_FFT_THREADS=2 PENCIL_FFT_DIRECTION_ORDER=directions-first ./tools/fftw-reference/check.sh
# Exact installed MSRV, isolated target; equivalent to cargo +1.85.0 inside the runner:
RUSTUP_HOME=/tmp/pencil-rs-rustup-msrv.0vuEDM RUSTUP_TOOLCHAIN=1.85.0 \
CARGO_TARGET_DIR=$(mktemp -d /tmp/pencil-reference-msrv-target.XXXXXX) \
PENCIL_FFT_THREADS=2 PENCIL_FFT_BACKEND=fftw PENCIL_FFT_DIRECTION_ORDER=native-first ./tools/fftw-reference/check.sh
# Pure parser checks (no environment mutation, MPI launch, or runtime loading):
cargo test -p pencil-fft --features distributed --test fftw_reference --locked
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

The runner generates exactly 82 legacy temporary, non-empty fixtures:
the original 28 C2C/R2C fixtures, 24 R2R fixtures (six cases × real and
complex `f32`/`f64`), 16 DHT fixtures (four cases × real and complex
`f32`/`f64`), and 14 mixed-axis C2C/R2C fixtures. The mixed fixtures include
FFT, RFFT, DCT, DHT, extra dimensions, and both odd/even reduced lengths.
It builds the Rust test once with `cargo test --no-run`, runs its ordinary
parser self-check, then runs the explicit ignored test
through `cargo test -- --ignored` at 1, 4, and 6 MPI ranks. The original full
and partial C2C/R2C cases are retained. For every R2C or mixed R2C fixture and
transport, the Rust checker compares the same independent expected values through
out-of-place and single-allocation real in-place forward/inverse/raw backward
paths. All fixtures contain raw
`backward_expected` values. It verifies a test-run marker so a missing ignored
test cannot pass silently. It also runs eight independent pristine-fixture copies for legacy C2C, R2C,
R2R, and DHT, plus twelve independent mixed-axis corruption copies covering
both mixed plan families and all forward/inverse/raw-backward expected sections;
each one-rank run must fail with status 1 or 101, the executed test's START
and failed-test markers, and comparison markers with matching kind/operation
context. Missing Julia, MPI, fixtures, or matrix members is a failure, never a
skip. The runner uses Cargo's selected toolchain and target directory; it never
selects an executable by filename or timestamp. Set `RUSTUP_TOOLCHAIN=1.85.0`
to run the reference matrix with an installed MSRV toolchain.

## Per-axis Fourier directions

A separate `directions_reference.jl` generator preserves the 82 legacy fixtures
and 272 base policy/layout runs unchanged. It adds exactly 5 direction fixtures:
2 C2C, 1 mixed C2C (FFT/DCT/DHT), and 2 mixed R2C (odd/even real axis with a
positive-sign complex suffix). Each runs both precisions, transports, and
memory policies: 40 additional configurations per MPI size (1, 4, and 6).
Forward, independent inverse input, normalized inverse, and raw backward are
compared numerically. The combined total is 87 fixtures and 312 configurations
per MPI size, unchanged for either backend or configuration order. Ten direction
corruptions plus the twenty legacy/mixed corruptions give 30 required rejections;
status, START, failed-test, and intended-reason guards reject runner/build/timeout
failures as evidence. Missing/duplicate fixtures, altered signs, and altered
expected outputs must fail.

Direction format 1 starts with `PENCIL_FFTW_DIRECTION_REFERENCE 1`, the pinned
runtime line, then `case`, `shape`, `transforms`, and `directions` (one
`forward`/`backward` per logical axis). The same five counted sections follow,
all serialized as complex pairs (real sections have zero imaginary parts).
The single RFFT axis determines reduced section lengths; non-FFT axes require
`forward`. `backward` configures an unscaled positive forward exponent, with
opposite signs for the paired inverse/raw-backward operations.

The standalone numerical regression also exercises in-place execution, partial
selection, core identity, and rank-dependent configuration rejection:

```bash
cargo build -p pencil-fft --features distributed --example mpi_directions
for ranks in 1 4 6; do
    timeout --kill-after=5s 120s mpiexec -n "$ranks" target/debug/examples/mpi_directions
done
```

Each successful run emits exactly one `mpi_directions: PASSED` marker.

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

Every file has this fixed order (format version 7):

```text
PENCIL_FFTW_REFERENCE 7
runtime julia=1.12.6 fftw_jl=1.10.0 native=... provider=fftw
case CASE_ID
kind c2c|r2c|r2r|dht|mixed_c2c|mixed_r2c
element_kind real|complex
precision f32|f64
original_shape N0 N1 ...
extra_shape [optional positive extents]
axis_kinds none|fft|rfft|dcti|dctii|dctiii|dctiv|dsti|dstii|dstiii|dstiv|dht [one per Rust axis]
selected_axes [canonical ascending Rust zero-based axes]
original_n N (mixed_r2c only, the original RFFT extent)
section input real|complex COUNT
... COUNT values ...
end
section inverse_input real|complex COUNT
... COUNT values ...
end
section forward_expected real|complex COUNT
... COUNT values ...
end
section inverse_expected real|complex COUNT
... COUNT values ...
end
section backward_expected real|complex COUNT
... COUNT values ...
end
```

Every fixture has five counted sections in this order. Section typing is fixed
by transform kind: C2C and mixed C2C are complex throughout; R2C and mixed R2C
are half-complex, with real `input`, `inverse_expected`, and
`backward_expected` plus complex `inverse_input` and `forward_expected`; R2R
and DHT use real or complex for all five sections according to
`element_kind`. DHT is the self-paired
separable Hartley transform; its normalized inverse divides by selected-axis
lengths and its backward result is raw. C2C `backward_expected` is generated by
FFTW's unnormalized
`bfft(copy(inverse_input), spatial_dims)`. R2C `backward_expected` is generated
by direct unnormalized `brfft(copy(inverse_input), original_real_n, spatial_dims)`.
R2R uses raw `plan_r2r` plans for the forward kind and its independently planned
paired kind; normalized inverse values are the raw paired result divided by the
logical FFTW factor (`2*(n-1)` for DCT-I, `2*(n+1)` for DST-I, otherwise `2*n`)
for each transformed axis. R2C comparison uses its raw result directly. Real
values are one finite number and are represented internally as a zero-imaginary
complex value; complex values are two finite numbers. Rust derives the reduced
output shape and topology grids and uses fixed precision-specific comparison
bounds. The generator fixes `ESTIMATE` and one FFTW thread; the Rust checker
runs both methods. These constants are not serialized repeatedly.
The parser rejects unknown, duplicate, reordered, trailing, malformed, empty,
non-finite, or incomplete data, zero dimensions, invalid ranks, overflowed
products, and counts before reserving section storage. Format 6 is rejected; format 7 is required.

The serialized order is global logical Rust row-major `[extra..., spatial...]`.
`selected_axes` is canonical Rust zero-based metadata; Julia maps Rust axis `a`
to Julia dimension `N-a`. `axis_kinds` is always present in canonical logical
axis order; legacy non-R2R/DHT fixtures use `none` on every axis. Mixed fixtures
use `none`, `fft`, `rfft`, DCT/DST, and `dht` directly and require the selected
axes to match the non-identity transforms. Julia allocates
`reverse([extra..., spatial...])`. C2C and mixed C2C apply transforms in
canonical descending Rust route order. Legacy R2C uses the maximum selected
Rust axis as its real FFT boundary; mixed R2C uses its single `rfft` axis and requires only identity/R2R/DHT
stages on the real prefix (higher Rust axes) and identity/FFT/R2R stages on
the complex suffix (lower Rust axes); `original_n` must match that axis. Only that
extent is reduced. C2C/R2R use independent forward and inverse inputs; R2C and
mixed R2C create the independent inverse spectrum from a real input, snapshot it
before planning, and execute both C2R and raw `brfft` on private copies. FFTW
`ESTIMATE`, one thread, the `fftw` provider, and native-version metadata remain
explicit.

There are eight full-axis base cases and six partial-axis cases. Both
precisions therefore produce the original 28 files. R2R adds six bounded cases
for each of four element/precision combinations, for 24 more files; DHT adds
four bounded cases (including the independent empty selection case) for the
same four combinations; mixed C2C/R2C adds seven cases at two precisions, for 82
total:

- Full-axis C2C: `[3,4]`, `[3,2,5]` with extras `[2,3]`, `[2,1,3,4]`
  with extra `[2]`.
- Full-axis R2C/C2R: `[3,1]`, `[3,2]`, `[3,1,4]` and `[3,1,5]` with
  extras `[2,3]`, and `[2,1,3,3]` with extra `[2]`.
- Partial-axis C2C: `[2,3,2,3]` with selections `[]` and `[0,3]`.
- Partial-axis R2C/C2R: `[2,3,2,3]` with `[0]`, `[0,2]`, and `[0,3]`, plus
  `[2,3,4]` with extra `[2]` and `[0,2]`.
- DHT: `[3,4]` full, `[3,2,4]` partial with extra `[2,3]`, and
  `[2,3,2,3]` partial with `[0,3]` plus an independent empty selection.
- Mixed: C2C `[3,2,4]` with `fft-dctii-dht`, `[2,3,2,3]` with
  `none-fft-dctiv-dht`, and two `[2,3,2,3]` cases covering all DCT/DST kinds;
  R2C `[4,3,5]` with `rfft-dctii-dht`, `[3,4,5]` with extra `[2]` and
  `fft-rfft-dht`, and `[3,4]` with `rfft-dht`.

The original and mixed cases are generated for f32 and f64. R2R and DHT cases
are generated for real and complex f32 and f64. No fixture topology is serialized.
For every file Rust runs all valid `M=1` layouts for `N=2,3,4` and also `M=2`
for `N=3,4`: 136 base case/layout configurations per memory-layout policy per
rank. Each configuration checks both Alltoallv and point-to-point, and both
`DistributedLayout::permute_dims` values give 272 total policy/layout runs.
Small leading axes (including the R2R `[2,1,3,3]` case) deliberately create
empty local ranges.

The checker derives raw physical-local to global-logical offsets from each
pencil's permutation and local ranges, checks the input identity and both
output permutations (reversed and identity) plus expected decompositions, and
fills/compares raw slices without `get_local`. A Cartesian integer all-reduce proves every tiny
fixture element is owned exactly once. It also checks out-of-place source and
spectrum preservation, normalized inverse, raw backward, and C2C/R2R in-place
values, including independent real and imaginary component bounds. Mixed
fixtures use an independent per-axis FFTW/R2R/DHT oracle rather than
round-trip-only assertions. DHT uses an independent direct Hartley oracle.

## Short plan

Keep the Julia project locked and the generator deterministic; keep fixtures
temporary; use the existing ignored distributed test as the sole MPI runner;
keep this README as the canonical format, matrix, and execution contract.
