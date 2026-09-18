# Distributed C2C PointToPoint implementation

- Date: 2026-09-18
- Base: `f8efdbe3f495f100df82b4b722deff13f5af74d3`
- Branch: `feat/distributed-c2c-p2p`
- Scope: the feature-gated distributed C2C transport choice, for both OOP and IP.
- Parent owns commit, push, PR, and merge; this worktree does none of them.

## API

- Add distributed-only `TransposeMethod::{AllToAllv, PointToPoint}`.
- Keep `from_pencil`, `from_array`, and `from_shape` signatures and make them
  delegate to `TransposeMethod::AllToAllv`.
- Add `from_pencil_with_method(input, extra_shape, method)`,
  `from_array_with_method(input, method)`, and
  `from_shape_with_method(topology, global_shape, extra_shape, method)`.
- Reexport the enum only with the `distributed` feature. Keep `FftError`, local
  FFT APIs, operation words 7--11, and the default local path unchanged.

## Core implementation

- Add one `PointToPoint` arm to the existing private `C2cTransition`.
- Build each distributed forward/backward edge with the selected existing
  native plan. Agree each forward and backward `workspace_requirements` result
  before the next collective, then keep the maximum shared lengths.
- Dispatch the new arm to `PointToPointTransposePlan::execute_in_place`.
  Reuse local transitions, `ManyPencilArray`, `TransposeWorkspace`, FFT scratch,
  and existing OOP/IP loops. Do not add a backend trait, factory, framework, or
  workspace type, and do not duplicate request handling.

## Protocol and contracts

- Preserve the fixed five-word header/schema and operations 7--11.
- Append one stable method word to the existing minimal descriptor, after scalar
  width: `0` for Alltoallv and `1` for PointToPoint. The checked payload length
  becomes `N + M + 3 + extra_rank`.
- Include that descriptor in construction and execution before native planning,
  FFT/output/workspace writes, or in-place poisoning. A rank-local valid method
  mismatch must return `CollectiveDescriptorMismatch` on every rank.
- Preserve source/input behavior, normalized inverse, shape limits, exact extras,
  plan-bound array/workspace identity, valid views, and Input/Output/Poisoned
  states. Keep allocations local; transforms and plan construction collective.
- Inherit the array P2P context/tag (`0x5054`), receive-before-send,
  wait-all, request reservation, and MPI failure contract. Do not claim
  allocation-free execution or global recovery after MPI/panic failure.

## Verification results

- `cargo fmt --all -- --check`, distributed all-targets check, workspace and
  feature Clippy with `-D warnings`, and both workspace/feature rustdoc builds:
  passed.
- Default `pencil-fft`: 20 unit tests, 2 positive doctests, and 1 compile-fail
  doctest passed. Distributed library tests: 24 passed.
- Workspace library tests: `pencil-array` 36 passed and default `pencil-fft`
  20 passed. Workspace doctests: 14 `pencil-array` and 3 default `pencil-fft`
  passed; distributed `pencil-fft` doctests: 6 passed.
- The combined `distributed_c2c` MPI binary passed at 1, 4, and 6 ranks under
  the 120-second timeout (one integration test per rank launch). Cross-transport
  comparison now checks four returned snapshots: OOP forward, arbitrary-spectrum
  OOP inverse, in-place forward, and arbitrary-spectrum in-place inverse.
- The private one-rank poison test constructs all three `_with_method` paths for
  both methods and checks both transition directions; legacy constructors still
  inspect as Alltoallv. Transport-negative coverage uses one layout helper for
  the 2D/M1 case and the 6-rank 4D/M2 rank-5 outside-subgroup case, covering
  constructor plus all four execution mismatches with snapshots and reuse.
- Parent verification passed 60 feature-enabled workspace unit tests and
  20 doctests, all 13 existing array MPI runs (16 MPI runs total), dependency
  boundary checks, and public rustdoc example/contract visibility. Final
  independent review approved. Array/manifests/lock and existing execution/poison
  loops are unchanged; `FftError` variants are unchanged.
- Rust 1.85 is verified by the existing GitHub CI job; no local toolchain or
  global configuration changes were made.
