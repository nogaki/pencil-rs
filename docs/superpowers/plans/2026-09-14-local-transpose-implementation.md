# Local transpose implementation plan

- Date: 2026-09-14
- Base: `216ad9e`
- Scope: Milestone 4 local transpose implementation, verification, and documentation.

## Boundary

Local transpose changes only the row-major memory-axis permutation between two
layouts with the same `MpiTopology` object, global shape, and ordered
decomposition (`Pencil::same_distribution`). The permutation may be identical
or any other valid permutation. It preserves every logical value at the same
local extra/spatial index. It does not redistribute ownership, call MPI, or run
an FFT.

The local API is deliberately separate from the future collective distributed
`TransposePlan` API. The `TransposePlanKind::Local` entry in section 15 is a
future composition point; this stage does not expose or implicitly wrap a
`TransposePlan`. `LocalTransposePlan` construction and execution are
process-local and noncollective: ranks that do not call it are valid. MPI
initialization/topology construction keeps its existing collective contract;
the local operation itself has no rank agreement or hidden communicator call.

Implement it in the existing `pencil-array` crate, rather than adding a new
crate or dependency. This keeps the existing crate-private
`LayoutWriteGuard` usable. Do not add `MpiElement`, a communication-method
enum/variant, a macro, or a general workspace abstraction.

## API contract

Add a small `local_transpose` module and re-export:

```rust
#[derive(Debug)]
pub struct LocalTransposePlan<const N: usize, const M: usize> { /* source, destination */ }

#[derive(Debug, thiserror::Error)]
pub enum LocalTransposeError {
    #[error("source and destination distributions are incompatible")]
    IncompatibleDistribution,
    #[error("source view or active layout does not match the plan")]
    SourceLayoutMismatch,
    #[error("destination view or registered layout does not match the plan")]
    DestinationLayoutMismatch,
    #[error("source and destination extra shapes differ")]
    ExtraShapeMismatch,
    #[error("scratch capacity {actual} is less than required {required}")]
    ScratchTooSmall { required: usize, actual: usize },
    #[error(transparent)]
    Array(#[from] ArrayError),
}

impl<const N: usize, const M: usize> LocalTransposePlan<N, M> {
    pub fn new(
        source: Arc<Pencil<N, M>>,
        destination: Arc<Pencil<N, M>>,
    ) -> Result<Self, LocalTransposeError>;

    pub fn execute_views<T: Clone>(
        &self,
        source: PencilArrayView<'_, T, N, M>,
        destination: PencilArrayViewMut<'_, T, N, M>,
    ) -> Result<(), LocalTransposeError>;

    pub fn execute_in_place<T: Clone>(
        &self,
        array: &mut ManyPencilArray<T, N, M>,
        scratch: &mut Vec<T>,
    ) -> Result<(), LocalTransposeError>;
}
```

`LocalTransposeError` has only the variants shown above: distribution/layout
mismatch, exact extra-shape mismatch, scratch-capacity failure, and wrapped
`ArrayError` (including checked-arithmetic and poisoned-state errors). `T:
Clone` is the only element constraint: local code does not need `MpiElement`,
`Copy`, or `Send`.

`execute_views` is the out-of-place path. Callers obtain views from
`PencilArray::view`/`view_mut`; view constructors and arbitrary inactive
`ManyPencilArray` views remain private/unavailable. It checks, before any
write, that both views match the plan and that their `ExtraShape` values are
exactly equal (same rank and extents, not merely the same element count). It
then clones each logical `[extra..., spatial...]` value into the destination
memory order. The source is unchanged on success and on a clone panic; a clone
panic may leave the separate destination partially written.

`execute_in_place` requires the active array layout to be the plan source and
the plan destination to be registered. `ManyPencilArray` supplies one exact
shared `ExtraShape` for both layouts; validate that shape and its checked
storage length rather than comparing only a product. Check
`checked_product([local_len, extra_shape.element_count()])` and
`scratch.capacity() >= required` before changing the array or clearing the
scratch vector. The required prefix is the only scratch data used; this
capacity contract means execution does not grow or reallocate the scratch
backing `Vec`. This does not prohibit allocation performed internally by
`T::clone` or allocation for indexing metadata. Public `PencilArray` and
`ManyPencilArray` constructors reject an overflowing checked storage product,
so malformed huge-shape arrays cannot be supplied to either
`LocalTransposePlan::execute_views` or `execute_in_place`; a huge shape paired
with a zero extra extent may instead validate as an empty no-op.

The in-place sequence is fixed:

1. Clone source values in source physical storage order into scratch while
   the array is still valid. The existing logical-index mapping selects each
   staged source value and its destination offset during the body write.
2. If clearing scratch or staging panics, propagate the panic, leave the
   array valid with the source layout, and make no claim about scratch contents.
3. After staging completes, create `LayoutWriteGuard`, which marks the array
   `Poisoned`, and copy the staged values into the backing storage.
4. Commit the registered destination layout only after the complete copy
   succeeds. A clone panic during this post-poison copy leaves the array
   `Poisoned`; recovery is a later complete `overwrite_with`.

All ordinary validation errors occur before destination/body writes. A
successful in-place operation is the only path that commits the destination
state. The implementation uses the existing views, `ManyPencilArray`, and
crate-private `LayoutWriteGuard`; it exposes no raw storage or unsafe fast
path.

## Files and implementation order

1. Create `crates/pencil-array/src/local_transpose.rs` with the error, plan,
   checked preflight, logical-index mapping, out-of-place copy, and staged
   in-place path.
2. Modify `crates/pencil-array/src/lib.rs` only to register the module and
   re-export `LocalTransposePlan` and `LocalTransposeError`.
3. Create `crates/pencil-array/tests/local_transpose.rs`. Do not modify
   `ManyPencilArray` unless an implementation bug reveals that the already
   present private hooks are insufficient; do not change `Cargo.toml`.

## Verification

Use one integration-test binary with one MPI initialization per process, as in
the existing tests (`--test-threads=1`). The test covers:

- 2D and 3D nontrivial permutations, plus identical source/destination layout;
- scalar extra shape, nonempty extra dimensions, and a zero extent such as
  `[2, 0]`;
- a 4-rank fixture with an empty local region;
- direct logical-index expected-value checks, forward/backward round trips, and
  out-of-place input preservation;
- plan/view distribution mismatch, exact extra-shape mismatch (including
  same-count shapes such as `[2, 3]` versus `[6]`), checked overflow using a
  maximal pencil, and insufficient scratch capacity, all before data mutation;
- in-place success, a clone panic while staging (array remains valid), and a
  clone panic after poisoning (array remains poisoned);
- normal execution on all ranks and a separate branch where only selected
  ranks execute the local operation, proving that no other rank is required to
  call it.

Run:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --lib --locked
timeout --foreground 120s mpiexec -n 1 cargo test -p pencil-array --test local_transpose --locked -- --nocapture --test-threads=1
timeout --foreground 120s mpiexec -n 4 cargo test -p pencil-array --test local_transpose --locked -- --nocapture --test-threads=1
mpiexec -n 1 cargo test -p pencil-array --test many --locked -- --nocapture --test-threads=1
mpiexec -n 4 cargo test -p pencil-array --test many --locked -- --nocapture --test-threads=1
cargo test --workspace --doc --locked -- --show-output
```

No commit or push is part of this stage.
