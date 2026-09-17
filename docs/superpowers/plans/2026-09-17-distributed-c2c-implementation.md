# Distributed C2C FFT implementation

- Date: 2026-09-17
- Base: `d03b91805bdba3caf6900ef57a652910e28297c0`
- Scope: the opt-in `pencil-fft` distributed C2C feature only.
- Boundary: no distributed R2C/C2R, in-place FFT API, new transport, unsafe
  code, new third-party packages, or changes to the local FFT implementations.

## API and feature boundary

The `distributed` feature enables `C2cPlan`, `C2cOutOfPlaceWorkspace`, and
`FftError`. The plan accepts `N >= 2` and `1 <= M < N`, and requires
`Complex<R>: mpi::datatype::Equivalence`; MPI's built-in complex datatype is
selected by the existing `mpi/complex` feature. `R` remains sealed to `f32`
and `f64`.

`from_pencil`, `from_array`, and `from_shape` construct the plan collectively
on the topology-owned Cartesian communicator. All ranks must use the same
communicator context, API, order, dimensions, scalar, and compatible inputs.
The canonical input has identity memory order and decomposition `[0..M)`.
Forward is input-layout to output-layout; inverse uses those layouts in
reverse. Both operations are out of place and preserve their source.

## Route and local/native plans

The route starts at the canonical input and processes axes `N-2` down to `0`.
Each axis is stably moved to the memory tail. If it is currently decomposed,
that decomposition entry advances from `a` to `a + 1`. The route has `N`
stages and `N-1` edges, ending with decomposition `[1..=M]` and reversed
memory order. Exactly `M` edges are Alltoallv and the rest are local
permutations.

Route construction is local preparation and is kept as a `Result` until a
whole-Cartesian scalar agreement. It performs no native FFT planning. Only
after that agreement succeeds are the per-stage `LocalC2cPlan`s constructed;
stage geometry (FFT axis in the memory tail, not decomposed, and full local
axis length) is validated once during this preparation.

Each edge stores only a `LocalTransposePlan` or an
`AllToAllvTransposePlan`, in forward and backward directions. One transition
vector is reserved and its reservation is scalar-agreed before edge creation.
Local pair preparation is agreed before the next edge. Alltoallv constructors
are already collective and are not wrapped in another agreement; their local
workspace requirements are scalar-agreed before the next edge. Construction
folds the maximum initialized send and receive lengths and returns those maxima
with the transition vector rather than retaining per-edge requirement copies.

## Memory and adapter boundary

A workspace owns one `ManyPencilArray` registered for every route stage, one
shared initialized `TransposeWorkspace`, and native FFT scratch. The workspace
stores the plan's `Arc` identity. Allocation uses checked lengths and
`try_reserve_exact`; execution checks initialized `len`, not capacity.

Local transitions use the safe array-level adapter that stages into the
initialized send prefix without resizing or exposing the private buffer.
Alltoallv transitions use the existing checked in-place operation. The initial
execution preflight checks source and destination layouts, exact extra shape,
workspace identity, active intermediate validity, and actual FFT/send/receive
lengths against the stored maxima. Immutable route geometry and per-edge
requirements are not rechecked there.

## Exact descriptor and collective ordering

Every constructor or execution first reduces the fixed five-word header:
`schema=1`, operation (`7` plan, `8` forward, `9` inverse), `N`, `M`, and the
checked payload length. The payload is exactly, in order:

1. `global_shape[N]`;
2. `process_grid[M]`;
3. extra-shape rank and its dimensions;
4. `size_of::<R>()` (4 or 8 bytes for the sealed scalar).

Its checked length is `N + M + 2 + extra_rank`. Exact payload words are compared
with the existing two-buffer min/max reduction. Descriptor readiness, MPI
count, allocation, and conversion failures are agreed before the reduction.
The constructor descriptor has no type names, alignments, stage/layout values,
transition kinds, or invalid route placeholders.

Execution borrows the plan descriptor; it does not rebuild or append source or
destination descriptors. Actual source/destination layouts and exact extras are
validated locally, then success is reduced over the full Cartesian communicator
before the first write. This retains collective detection of wrong layouts,
shapes, directions, scalar widths, workspace identity, and initialized lengths
without putting rank-local layout state in the descriptor.

## Normalization and failure contract

Each local inverse stage is normalized once by its spatial line length. The
complete inverse therefore scales by `1 / product(global spatial lengths)`;
extra dimensions do not affect the scale.

A descriptor or initial preflight error returns collectively before any source,
destination, or workspace write, and leaves all buffers unchanged. After the
first FFT stage starts, a checked Alltoallv may prepare metadata and return a
collectively agreed preparation/allocation error. The source remains preserved,
but workspace changes are then allowed; no speculative rollback is attempted.
Native FFT/MPI failures, panics, and process loss remain outside recovery
guarantees.

## Runnable example and checks

The feature module contains one public `C2cPlan` doctest: one MPI
initialization, a 2D `M=1` topology sized from `world.size()`, input/output and
workspace allocation, forward, inverse, scoped drops before the universe, and
a round-trip assertion.

Local verification passed: 56 default workspace unit tests and 17 doctests;
58 unit tests and 18 doctests with `pencil-fft/distributed`; distributed C2C
at 1/4/6 ranks and all 13 existing array MPI runs; formatting, Clippy, docs,
dependency boundaries, and lockfile checks. Independent final review approved.
The local FFT implementation is unchanged. Rust 1.85 is checked by GitHub CI.

Parent/CI checks use the requested environment:

```text
PATH=/home/kosuke/.cargo/bin:$PATH
CARGO_HOME=/tmp/pencil-rs-cargo.wgYrAO
cargo check -p pencil-fft --features distributed --all-targets --locked
cargo test -p pencil-fft --features distributed --doc --locked -- --show-output
cargo clippy -p pencil-fft --features distributed --all-targets --locked -- -D warnings
cargo doc -p pencil-fft --features distributed --no-deps --locked
for n in 1 4 6; do
  timeout --foreground 120s mpiexec --oversubscribe -n "$n" \
    cargo test -p pencil-fft --features distributed --test distributed_c2c \
      --locked -- --nocapture --test-threads=1 || exit 1
done
```
