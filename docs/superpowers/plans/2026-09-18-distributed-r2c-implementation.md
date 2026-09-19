# Distributed R2C/C2R implementation plan

- Date: 2026-09-18
- Base: `d617ac7774e90aa224fbcc4d023b24a9e9761b15`
- Status: implemented and independently reviewed.
- Scope: feature-gated, out-of-place distributed R2C/C2R, including raw backward.
- Constraints: `N >= 2`, `1 <= M < N`, canonical identity input with
  decomposition `[0..M)`, `f32`/`f64`, Alltoallv or PointToPoint, no unsafe
  code or new dependencies. Preserve `FftError`, local R2C behavior, the
  default MPI-free build, and the existing transport contracts. No real
  in-place API.

## Public API and file layout

- `crates/pencil-fft/src/distributed.rs` owns the feature-gated public
  exports/errors and the shared private route, stage, transition, descriptor,
  preflight, and workspace machinery.
- `crates/pencil-fft/src/distributed/r2c.rs` owns the typed `R2cPlan`,
  `R2cWorkspace`, constructor wrappers, real-line workspace, endpoint policy,
  and R2C/C2R execution. It does not define a second plan core.
- `crates/pencil-fft/src/lib.rs` keeps `FftReal` sealed and exports the R2C
  types. Numeric helpers remain private `Sealed` methods; there are no root
  `pub(crate)` forwarding wrappers.
- `README.md`, the public R2C rustdoc, the main design spec, and CI labels
  describe the same R2C/C2R boundary. The existing single MPI integration
  binary remains the test target; its test-file changes are owned separately.

## Shared production core

- `TransformStage` stores explicit `input` and `output` pencil endpoints and
  `LocalTransform::{Complex(LocalC2cPlan), RealComplex(LocalR2cPlan)}`. No
  public framework, marker hierarchy, `ValueKind`, or R2C-specific stage type
  is introduced.
- `TransformPlanCore` contains only `stages`, `transitions`, `extra_shape`,
  `descriptor`, `fft_scratch_len`, `transpose_send_len`, and
  `transpose_receive_len`. Both public plan wrappers and their workspaces
  hold the same plan `Arc`; in-place C2C poisoning keeps its existing identity
  checks.
- The real input is `core.stages[0].input`; the original real and reduced
  complex line lengths come from the first `LocalR2cPlan` when an operation
  needs them. Inverse boundary depth is computed cheaply from that input's
  original shape, not retained as another core field. The R2C plan stores only
  the prevalidated raw absolute endpoint threshold needed by `backward`.

## Construction and execution

- `validate_input` performs the canonical dimensions, nonzero shape,
  topology/shape, identity permutation, and `[0..M)` decomposition checks and
  returns the input `Arc`. C2C calls it through `build_route`; R2C collectively
  agrees it before creating its reduced route, so no full original-shape route
  is built merely for validation.
- R2C creates one reduced shape `[global_shape[..N-1], n/2+1]` and runs the
  existing route algorithm on it for axes `N-2` through `0`. Its first stage
  is `RealComplex(original input -> reduced canonical pencil)`; the reduced
  route's first endpoint is not separately transformed.
- One `prepare_stages(route, shape, first)` serves C2C and R2C. C2C passes
  `None`; R2C passes its already-native-planned and collectively-agreed real
  first stage. It reserves the final stage count once, fills the remaining
  complex stages, and computes the maximum native scratch including the
  supplied first stage.
- Transitions use the current stage output and next stage input endpoints.
  Native distributed transition constructors retain their own collective
  agreement; locally fallible vector reservation and workspace-requirement
  preparation remain collectively agreed. The existing local transition path
  is unchanged.
- A shared out-of-place preflight checks plan/workspace identity, endpoint
  layouts, extra shapes, registered intermediate layouts, and common
  initialized FFT/transpose lengths. R2C adds only its real-line and
  complex-line length checks. All initial errors are agreed before writes.

## R2C/C2R math and error boundary

- Forward performs local R2C on each original last-axis line, then the shared
  complex tail; the real source is preserved. Inverse and raw backward reverse
  the complex tail first, validate constrained planes, zero accepted endpoint
  imaginary parts only in the private intermediate, and call the strict local
  C2R operation. Only inverse applies normalization, exactly once per original
  spatial axis; extra dimensions and `m = n/2+1` are excluded.
- For every extra batch and constrained plane `z(x)`, all endpoint real and
  imaginary values must be finite. Normalized inverse acceptance is exactly:

  ```text
  D = 1 + sum_{a=0..N-2} ceil(log2(n_a))
  relative_R = 128 * epsilon_R * D
  absolute_R = 128 * min_subnormal_R * D
  max_x abs(Im z(x)) <= absolute_R
  OR ||Im z||_2 <= relative_R * ||Re z||_2
  ```

  `n_a` are original axes before the real axis; length one contributes zero.
  Raw backward keeps the relative threshold and uses
  `absolute_raw = absolute_inverse * product(n_0..n_{N-2})`, excluding the
  real axis and extras. That factor and threshold are finite-positive
  collectively validated during construction before native planning. DC is
  always constrained. Nyquist is constrained only for even `n`; for odd
  `n > 1` the final bin is unconstrained, while `n == 1` has only DC. This is
  the approved explicit componentwise-absolute/normwise-relative policy, not
  a formal RustFFT error bound. Interior bins have no blanket finite policy.
- The fixed four-word max and sum reductions run for every batch and plane on
  the full Cartesian communicator, including empty local ranks, followed by
  one validity reduction before any real destination write. Invalid boundaries
  return `R2cError::InvalidSpectrum`; source and destination remain unchanged,
  while post-start workspace mutation is allowed. No public tolerance knob,
  full-spectrum gather, offender-ID reduction, or global rollback promise is
  added.

## Protocol and documentation status

- R2C construction/forward/inverse/raw backward use operation words
  12/13/14/17. The existing five-word header and words 1--11 stay unchanged.
  Raw and normalized reverse calls, forward, C2C operations, and out-of-place/
  in-place operations therefore remain distinct in the full Cartesian header.
  The descriptor records the
  original real shape, process grid, extra rank/extents, scalar width, and
  method, so even and odd shapes sharing `n/2+1` remain distinct.
- Public docs state that the reduced canonical stage initially keeps
  decomposition `[0..M)`, while final output uses `[1..=M]` and reversed
  memory order. They include a runnable MPI example with an odd length,
  forward/inverse/raw-backward coverage, shape queries, and source preservation, plus the
  exact endpoint policy and compile-fail associated-method lookup.
- The main design spec records this implementation in a dated Milestone 8
  section; README links this plan but not a separate addendum. CI job count and
  targets remain unchanged; distributed suite labels say C2C/R2C/C2R.

## Checks

Parent verification passed: 56 default workspace unit tests and 17 doctests;
60 feature-enabled unit tests and 22 doctests; the combined FFT MPI suite on
1/4/6 ranks plus all 13 existing array MPI runs (16 MPI runs total). Formatting,
default/distributed Clippy with `-D warnings`, documentation, public R2C example
visibility, and default dependency boundaries passed. The no-real-in-place
doctest fails specifically because the associated method is absent. Existing
C2C test functions remain intact; local FFT bodies and the array/manifests/lock
are unchanged. Rust 1.85 is checked by the existing GitHub CI job.
