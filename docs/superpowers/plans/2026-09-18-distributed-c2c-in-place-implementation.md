# Distributed C2C in-place FFT revision record

- Date: 2026-09-18
- Base: `cbd1c172cb649a731f251f78135d291e2de1a242`
- Status: implemented in the existing distributed C2C module; no array crate,
  manifest, lockfile, or local FFT implementation changes.

## API and compatibility

- `C2cInPlaceArray`, `C2cInPlaceWorkspace`, and `C2cState` are feature-gated
  public types. Allocation and views are noncollective; plan construction and
  transform calls are collective. Callers coordinate local allocation failure
  before the next collective.
- `FftError` remains exactly the base enum. In-place foreign-plan arrays map to
  `Array(ArrayError::IncompatiblePencils)`, poisoned arrays to
  `Array(ArrayError::Poisoned)`, and wrong `Input`/`Output` state or active
  endpoint layout to the existing `InputLayoutMismatch`. Foreign workspaces
  remain `WorkspaceMismatch`.
- Forward uses `Input -> Poisoned -> Output`; inverse uses
  `Output -> Poisoned -> Input`. Preflight preserves state/data/workspace.
  Poison is set before the first write; post-start `Err` or panic leaves it set.
  There is no production `catch_unwind` or recovery hook.

## Implementation

- Shared the existing five-word execution header/descriptor agreement between
  out-of-place and in-place calls, and shared the FFT/send/receive initialized
  length checks. No generic execution framework was added.
- In-place allocation clones the exact stage `Arc` registry. Preflight retains
  state, active endpoint, exact extra shape, plan/workspace identity, and all
  length checks without an O(N^2) registry rescan.
- In-place execution reuses the existing local FFT and local/Alltoallv route in
  one `ManyPencilArray`. The test DFT helpers accept `PencilArrayView`, so the
  same independent oracle checks cover both APIs.

## Verification coverage

- The integration binary covers f32/f64, N=2/3/4, M=1/2, uneven and empty
  local ranks, zero extra batches, reversed communicators, independent DFT
  checks, arbitrary spectra, round trips, reuse, pointer stability across
  layout-length changes, and collective negative cases.
- A live multi-rank operation-10/operation-11 mismatch snapshots state/data and
  workspaces before rejection, then realigns and reuses the workspace.
- The single-rank private unit test loops over both directions and Err/panic,
  observes Poisoned before callback writes, checks both views and both public
  retries, and exercises a corrupted final local plan returning
  `LocalC2c(NonIntegralBatch)` after the first stage/transition.
- Privacy doctest, README/spec/plan docs, and CI command names describe the
  collective/noncollective and pre-start poison contracts. Verification passed:
  default pencil-fft 20 unit + 3 doc tests, distributed pencil-fft 24 unit +
  6 doc tests, workspace 56 library + 17 doc tests, Clippy with `-D warnings`,
  both documentation builds, and one integration test at each MPI size 1, 4,
  and 6 (all under 120-second timeouts).
- Parent verification also passed the feature-enabled workspace's 60 unit tests
  and 20 doctests, all 13 existing array MPI runs (16 MPI runs total), and the
  default dependency boundary. The privacy doctest fails specifically on the
  private field. Final independent review approved; Rust 1.85 is checked in CI.
