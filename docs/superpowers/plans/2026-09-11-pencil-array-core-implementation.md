# Pencil Array Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the greenfield Cargo workspace and complete the independently testable `pencil-array` core through `PencilArray` and `ManyPencilArray`, without implementing distributed transposition or FFT.

**Architecture:** The crate separates pure geometry from MPI-owned topology. `MpiTopology<M>` owns Cartesian and one-axis subcommunicators; immutable `Pencil<N, M>` values share it through `Arc`. `PencilArray` owns one local `Vec<T>`, while `ManyPencilArray` owns one maximum-sized local buffer plus a checked active-layout state and creates temporary borrowed views.

**Tech Stack:** Rust 2024 edition; MSRV 1.85; `mpi = 0.8.2`; `thiserror = 2`; `proptest = 1`; Rustdoc compile-fail tests; Open MPI or MPICH for integration tests.

**Spec:** `docs/superpowers/specs/2026-09-11-pencil-arrays-rust-port-design.md`

## Global Constraints

- `pencil-array` must not depend on RustFFT, RealFFT, FFTW, or any FFT-specific type.
- Spatial dimension count `N` and MPI topology dimension count `M` are const generics.
- Accept `1 <= M <= N`, including `M == N`.
- Storage is contiguous `Vec<T>` and uses row-major order.
- Public logical axis order is `[extra..., spatial...]`.
- Physical memory order is `[extra..., permuted spatial...]`.
- Extra dimensions are never MPI-distributed and are never changed by spatial permutations.
- `Pencil`, topology metadata, and configuration objects are immutable after construction.
- Share immutable topology and pencil values with `Arc`; do not put communication buffers in `Pencil`.
- Public APIs return structured errors for caller-controlled invalid input; do not panic on such input.
- Do not add a runtime-dimensional `DynPencil`, GPU storage abstraction, FFT API, distributed transpose implementation, or executor wrapper in this plan.
- Preserve the upstream PencilArrays.jl MIT copyright notice in `NOTICE.md` because the range-partitioning and API semantics are derived from it.

---

## File Map

```text
Cargo.toml                              Workspace manifest and shared dependency versions
rust-toolchain.toml                     Stable toolchain declaration
.gitignore                              Rust build artefacts
LICENSE                                 MIT licence for this implementation
NOTICE.md                               Upstream PencilArrays/PencilFFTs attribution
README.md                               Workspace purpose and build prerequisites
crates/pencil-array/Cargo.toml          Core crate manifest
crates/pencil-array/src/lib.rs          Public module/re-export boundary and compile-fail doctests
crates/pencil-array/src/error.rs        Core error types
crates/pencil-array/src/checked.rs      Checked products and integer conversions
crates/pencil-array/src/axis.rs         SpatialAxis and AxisPermutation
crates/pencil-array/src/geometry.rs     Partitioning, shapes, ranges, and row-major offsets
crates/pencil-array/src/topology.rs     Owned MPI Cartesian topology and subcommunicators
crates/pencil-array/src/pencil.rs       Immutable distributed layout
crates/pencil-array/src/extra_shape.rs  Checked extra-dimension shape
crates/pencil-array/src/view.rs         Borrowed immutable/mutable array views
crates/pencil-array/src/array.rs        Owning PencilArray
crates/pencil-array/src/many.rs         ManyPencilArray, layout state, overwrite transaction
crates/pencil-array/tests/topology.rs   Multi-rank topology integration tests
crates/pencil-array/tests/pencil.rs     Multi-rank Pencil integration tests
```

---

### Task 1: Bootstrap the Cargo workspace

**Files:**
- Create: `Cargo.toml`
- Create: `rust-toolchain.toml`
- Create: `.gitignore`
- Create: `LICENSE`
- Create: `NOTICE.md`
- Create: `README.md`
- Create: `crates/pencil-array/Cargo.toml`
- Create: `crates/pencil-array/src/lib.rs`

**Interfaces:**
- Consumes: none
- Produces: a buildable workspace containing the `pencil-array` library crate

- [ ] **Step 1: Create the workspace manifest**

```toml
[workspace]
members = ["crates/pencil-array"]
resolver = "3"

[workspace.package]
edition = "2024"
rust-version = "1.85"
license = "MIT"

[workspace.dependencies]
mpi = "0.8.2"
thiserror = "2"
proptest = "1"
```

Do not add a `repository` field until the actual hosting URL is known.

- [ ] **Step 2: Pin the toolchain channel and components**

```toml
[toolchain]
channel = "stable"
components = ["clippy", "rustfmt"]
profile = "minimal"
```

- [ ] **Step 3: Create the crate manifest**

```toml
[package]
name = "pencil-array"
version = "0.1.0"
edition.workspace = true
rust-version.workspace = true
license.workspace = true

[dependencies]
mpi.workspace = true
thiserror.workspace = true

[dev-dependencies]
proptest.workspace = true
```

- [ ] **Step 4: Create the initial library root**

```rust
#![forbid(unsafe_op_in_unsafe_fn)]
#![deny(missing_debug_implementations)]

//! MPI-distributed, row-major multidimensional array foundations.
```

Add each module declaration and re-export in the task that creates that module, so every intermediate commit remains buildable.

- [ ] **Step 5: Add licence and attribution files**

Use the standard MIT licence text in `LICENSE`. In `NOTICE.md`, include:

```text
This project is an independent Rust implementation inspired by:

PencilArrays.jl
Copyright (c) 2020 Juan Ignacio Polanco <jipolanc@gmail.com> and contributors
https://github.com/jipolanco/PencilArrays.jl
MIT License

PencilFFTs.jl
Copyright (c) 2019 Juan Ignacio Polanco
https://github.com/jipolanco/PencilFFTs.jl
MIT License

Reference commits used by the design:
PencilArrays.jl 12229b99b827e07880517982c3365a18d1f9b8dc
PencilFFTs.jl   1d98a3ff790c40445987ad64b99eb3b946a11034
```

- [ ] **Step 6: Add build prerequisites to `README.md`**

Document Rust stable, a C MPI implementation, `mpicc`, and `mpiexec`. Include these commands:

```bash
cargo test -p pencil-array --lib
mpiexec -n 4 cargo test -p pencil-array --test topology -- --nocapture
```

- [ ] **Step 7: Verify workspace metadata**

Run:

```bash
cargo metadata --no-deps --format-version 1
cargo fmt --all -- --check
cargo test -p pencil-array --lib
```

Expected: all commands exit successfully; the new crate has zero tests at this point.

- [ ] **Step 8: Commit**

```bash
git add Cargo.toml rust-toolchain.toml .gitignore LICENSE NOTICE.md README.md crates/pencil-array
git commit -m "chore: bootstrap pencil array workspace"
```

---

### Task 2: Add structured errors and checked arithmetic

**Files:**
- Create: `crates/pencil-array/src/error.rs`
- Create: `crates/pencil-array/src/checked.rs`
- Modify: `crates/pencil-array/src/lib.rs`

**Interfaces:**
- Consumes: none
- Produces:
  - `AxisError`
  - `GeometryError`
  - crate-private `checked_product`
  - crate-private `usize_to_i32`

- [ ] **Step 1: Write failing unit tests for checked products**

Add to `checked.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn product_of_empty_shape_is_one() {
        assert_eq!(checked_product(&[]).unwrap(), 1);
    }

    #[test]
    fn product_detects_overflow() {
        assert_eq!(checked_product(&[usize::MAX, 2]), Err(GeometryError::SizeOverflow));
    }

    #[test]
    fn mpi_count_conversion_rejects_large_values() {
        assert_eq!(usize_to_i32(i32::MAX as usize + 1), Err(GeometryError::CountOverflow));
    }
}
```

- [ ] **Step 2: Run the focused test and confirm failure**

Run:

```bash
cargo test -p pencil-array checked::tests --lib
```

Expected: compilation fails because the error variants and helper functions are absent.

- [ ] **Step 3: Implement the error types**

```rust
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum AxisError {
    #[error("axis {axis} is outside 0..{dimensions}")]
    OutOfBounds { axis: usize, dimensions: usize },

    #[error("axis {axis} occurs more than once")]
    Duplicate { axis: usize },
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum GeometryError {
    #[error("shape or offset calculation overflowed usize")]
    SizeOverflow,

    #[error("value does not fit MPI Count (i32)")]
    CountOverflow,

    #[error("process coordinate {coordinate} is outside 0..{extent} on topology axis {axis}")]
    ProcessCoordinateOutOfBounds {
        axis: usize,
        coordinate: usize,
        extent: usize,
    },

    #[error("local index {index} is outside 0..{extent} on logical axis {axis}")]
    LocalIndexOutOfBounds {
        axis: usize,
        index: usize,
        extent: usize,
    },
}
```

- [ ] **Step 4: Implement checked helpers**

```rust
use crate::GeometryError;

pub(crate) fn checked_product(values: &[usize]) -> Result<usize, GeometryError> {
    values
        .iter()
        .try_fold(1usize, |acc, &value| acc.checked_mul(value).ok_or(GeometryError::SizeOverflow))
}

pub(crate) fn usize_to_i32(value: usize) -> Result<i32, GeometryError> {
    i32::try_from(value).map_err(|_| GeometryError::CountOverflow)
}
```

- [ ] **Step 5: Export errors and run tests**

Run:

```bash
cargo test -p pencil-array checked::tests --lib
cargo clippy -p pencil-array --all-targets -- -D warnings
```

Expected: all tests pass and Clippy reports no warnings.

- [ ] **Step 6: Commit**

```bash
git add crates/pencil-array/src/error.rs crates/pencil-array/src/checked.rs crates/pencil-array/src/lib.rs
git commit -m "feat: add checked geometry primitives"
```

---

### Task 3: Implement `SpatialAxis` and `AxisPermutation<N>`

**Files:**
- Create: `crates/pencil-array/src/axis.rs`
- Modify: `crates/pencil-array/src/lib.rs`

**Interfaces:**
- Consumes: `AxisError`
- Produces:

```rust
pub struct SpatialAxis(usize);

impl SpatialAxis {
    pub fn new<const N: usize>(index: usize) -> Result<Self, AxisError>;
    pub const fn index(self) -> usize;
}

pub struct AxisPermutation<const N: usize>;

impl<const N: usize> AxisPermutation<N> {
    pub fn new(axes: [usize; N]) -> Result<Self, AxisError>;
    pub fn identity() -> Self;
    pub fn axes(&self) -> &[SpatialAxis; N];
    pub fn position_of(&self, axis: SpatialAxis) -> usize;
    pub fn permute<T: Copy>(&self, logical: [T; N]) -> [T; N];
    pub fn unpermute<T: Copy>(&self, memory: [T; N]) -> [T; N];
}
```

- [ ] **Step 1: Write failing constructor and round-trip tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spatial_axis_rejects_out_of_bounds_index() {
        assert_eq!(
            SpatialAxis::new::<3>(3),
            Err(AxisError::OutOfBounds { axis: 3, dimensions: 3 }),
        );
    }

    #[test]
    fn permutation_rejects_duplicate_axes() {
        assert_eq!(
            AxisPermutation::<3>::new([0, 1, 1]),
            Err(AxisError::Duplicate { axis: 1 }),
        );
    }

    #[test]
    fn permutation_maps_logical_values_to_memory_order() {
        let permutation = AxisPermutation::<3>::new([0, 2, 1]).unwrap();
        assert_eq!(permutation.permute([10, 20, 30]), [10, 30, 20]);
        assert_eq!(permutation.unpermute([10, 30, 20]), [10, 20, 30]);
    }
}
```

- [ ] **Step 2: Run the focused tests and confirm failure**

```bash
cargo test -p pencil-array axis::tests --lib
```

Expected: compilation fails because the types are not implemented.

- [ ] **Step 3: Implement validated axes and permutations**

Use this representation:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SpatialAxis(usize);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AxisPermutation<const N: usize> {
    axes_in_memory_order: [SpatialAxis; N],
    logical_to_memory: [usize; N],
}
```

`AxisPermutation::new` must:

1. reject every index `>= N`;
2. reject repeated indices;
3. fill `logical_to_memory[logical_axis] = memory_position`.

Use `std::array::from_fn` for fixed-size arrays. Do not allocate a `Vec` in `permute` or `unpermute`.

- [ ] **Step 4: Add property tests for all six 3D permutations**

```rust
proptest::proptest! {
    #[test]
    fn permute_then_unpermute_is_identity(values in proptest::array::uniform3(any::<u16>())) {
        for axes in [
            [0, 1, 2], [0, 2, 1], [1, 0, 2],
            [1, 2, 0], [2, 0, 1], [2, 1, 0],
        ] {
            let permutation = AxisPermutation::<3>::new(axes).unwrap();
            prop_assert_eq!(permutation.unpermute(permutation.permute(values)), values);
        }
    }
}
```

- [ ] **Step 5: Run tests and documentation checks**

```bash
cargo test -p pencil-array axis::tests --lib
cargo test -p pencil-array --doc
cargo clippy -p pencil-array --all-targets -- -D warnings
```

- [ ] **Step 6: Commit**

```bash
git add crates/pencil-array/src/axis.rs crates/pencil-array/src/lib.rs
git commit -m "feat: add validated spatial axes"
```

---

### Task 4: Implement pure partitioning and row-major indexing

**Files:**
- Create: `crates/pencil-array/src/geometry.rs`
- Modify: `crates/pencil-array/src/lib.rs`

**Interfaces:**
- Consumes: `AxisPermutation<N>`, `GeometryError`, `checked_product`
- Produces:

```rust
pub fn partition_range(length: usize, parts: usize, coordinate: usize) -> Result<Range<usize>, GeometryError>;

pub(crate) fn local_ranges_for<const N: usize, const M: usize>(
    global_shape: [usize; N],
    process_grid: [usize; M],
    process_coords: [usize; M],
    decomposition: [SpatialAxis; M],
) -> Result<[Range<usize>; N], GeometryError>;

pub(crate) fn shape_from_ranges<const N: usize>(ranges: &[Range<usize>; N]) -> [usize; N];

pub(crate) fn row_major_offset(shape: &[usize], indices: &[usize]) -> Result<usize, GeometryError>;
```

- [ ] **Step 1: Write failing exact-range tests**

```rust
#[test]
fn partition_matches_julia_floor_rule() {
    assert_eq!(partition_range(10, 3, 0).unwrap(), 0..3);
    assert_eq!(partition_range(10, 3, 1).unwrap(), 3..6);
    assert_eq!(partition_range(10, 3, 2).unwrap(), 6..10);
}

#[test]
fn partition_allows_empty_ranges() {
    assert_eq!(partition_range(2, 4, 0).unwrap(), 0..0);
    assert_eq!(partition_range(2, 4, 1).unwrap(), 0..1);
    assert_eq!(partition_range(2, 4, 2).unwrap(), 1..1);
    assert_eq!(partition_range(2, 4, 3).unwrap(), 1..2);
}
```

- [ ] **Step 2: Run tests and confirm failure**

```bash
cargo test -p pencil-array geometry::tests --lib
```

- [ ] **Step 3: Implement overflow-safe partitioning**

Use quotient/remainder arithmetic instead of evaluating `length * coordinate`, which can overflow. Implement the boundary with the equivalent decomposition below:

```rust
fn boundary(length: usize, parts: usize, p: usize) -> Result<usize, GeometryError> {
    let q = length / parts;
    let r = length % parts;
    q.checked_mul(p)
        .and_then(|base| r.checked_mul(p).map(|rp| (base, rp)))
        .and_then(|(base, rp)| base.checked_add(rp / parts))
        .ok_or(GeometryError::SizeOverflow)
}
```

Reject `parts == 0` with a dedicated `GeometryError::ZeroPartitions` variant added in `error.rs`.

- [ ] **Step 4: Implement multidimensional range assignment**

Start all spatial axes as `0..global_shape[d]`. For topology axis `t`, replace spatial axis `decomposition[t]` with `partition_range(global_shape[d], process_grid[t], process_coords[t])`.

- [ ] **Step 5: Implement row-major offsets**

For `shape = [d0, d1, ..., dk]`, compute:

```text
offset = (((i0 * d1 + i1) * d2 + i2) ... ) * dk + ik
```

Check equal rank of `shape` and `indices`, each index bound, multiplication overflow, and addition overflow.

- [ ] **Step 6: Add property tests**

Verify for random `length in 1..128` and `parts in 1..32`:

```rust
let ranges = (0..parts)
    .map(|p| partition_range(length, parts, p).unwrap())
    .collect::<Vec<_>>();

prop_assert_eq!(ranges.first().unwrap().start, 0);
prop_assert_eq!(ranges.last().unwrap().end, length);
for pair in ranges.windows(2) {
    prop_assert_eq!(pair[0].end, pair[1].start);
}
prop_assert_eq!(ranges.iter().map(Range::len).sum::<usize>(), length);
```

Also test an unsorted decomposition such as `[2, 0]` with process grid `[2, 3]`.

- [ ] **Step 7: Run all pure tests**

```bash
cargo test -p pencil-array --lib
cargo clippy -p pencil-array --all-targets -- -D warnings
```

- [ ] **Step 8: Commit**

```bash
git add crates/pencil-array/src/geometry.rs crates/pencil-array/src/error.rs crates/pencil-array/src/lib.rs
git commit -m "feat: add distributed geometry calculations"
```

---

### Task 5: Implement owned MPI Cartesian topology

**Files:**
- Create: `crates/pencil-array/src/topology.rs`
- Create: `crates/pencil-array/tests/topology.rs`
- Modify: `crates/pencil-array/src/error.rs`
- Modify: `crates/pencil-array/src/lib.rs`
- Modify: `README.md`

**Interfaces:**
- Consumes: `mpi::topology::{CartesianCommunicator, Communicator}`, checked integer helpers
- Produces:

```rust
pub struct MpiTopology<const M: usize>;

impl<const M: usize> MpiTopology<M> {
    pub fn new<C: Communicator>(comm: &C, process_grid: [usize; M]) -> Result<Arc<Self>, TopologyError>;
    pub fn auto<C: Communicator>(comm: &C) -> Result<Arc<Self>, TopologyError>;
    pub fn from_cartesian(comm: &CartesianCommunicator) -> Result<Arc<Self>, TopologyError>;
    pub fn process_grid(&self) -> &[usize; M];
    pub fn local_coords(&self) -> &[usize; M];
    pub fn rank(&self) -> i32;
    pub fn size(&self) -> usize;
    pub fn rank_at(&self, coords: [usize; M]) -> Result<i32, TopologyError>;
    pub(crate) fn cartesian(&self) -> &CartesianCommunicator;
    pub(crate) fn subcommunicator(&self, topology_axis: usize) -> &CartesianCommunicator;
}
```

- [ ] **Step 1: Write a failing four-rank topology test**

```rust
use mpi::traits::*;
use pencil_array::MpiTopology;

#[test]
fn cartesian_topology_maps_all_coordinates() {
    let universe = mpi::initialize().expect("MPI initialization failed");
    let world = universe.world();
    assert_eq!(world.size(), 4, "run this test with mpiexec -n 4");

    let topology = MpiTopology::<2>::new(&world, [2, 2]).unwrap();
    assert_eq!(topology.process_grid(), &[2, 2]);
    assert_eq!(topology.size(), 4);

    for i in 0..2 {
        for j in 0..2 {
            let rank = topology.rank_at([i, j]).unwrap();
            assert!((0..4).contains(&rank));
        }
    }
}
```

- [ ] **Step 2: Run the integration test and confirm failure**

```bash
mpiexec -n 4 cargo test -p pencil-array --test topology -- --nocapture
```

Expected: compilation fails because `MpiTopology` does not exist.

- [ ] **Step 3: Add `TopologyError`**

```rust
#[derive(Debug, thiserror::Error)]
pub enum TopologyError {
    #[error("topology dimension count must be positive")]
    ZeroDimensions,

    #[error("process-grid extent on axis {axis} must be positive")]
    ZeroExtent { axis: usize },

    #[error("process-grid size {grid_size} does not equal communicator size {communicator_size}")]
    CommunicatorSizeMismatch {
        grid_size: usize,
        communicator_size: usize,
    },

    #[error("MPI did not create a Cartesian communicator")]
    CartesianCreationFailed,

    #[error("Cartesian communicator has {actual} dimensions, expected {expected}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("coordinate {coordinate} is outside 0..{extent} on topology axis {axis}")]
    CoordinateOutOfBounds { axis: usize, coordinate: usize, extent: usize },

    #[error("MPI topology operation failed: {0}")]
    Mpi(String),

    #[error(transparent)]
    Geometry(#[from] GeometryError),
}
```

- [ ] **Step 4: Implement `MpiTopology::new` using rsmpi's high-level Cartesian API**

Use:

```rust
let dims = process_grid
    .iter()
    .copied()
    .map(i32::try_from)
    .collect::<Result<Vec<_>, _>>()?;
let periods = vec![false; M];
let cartesian = comm
    .create_cartesian_communicator(&dims, &periods, false)
    .ok_or(TopologyError::CartesianCreationFailed)?;
```

Store:

```rust
pub struct MpiTopology<const M: usize> {
    cartesian: CartesianCommunicator,
    subcommunicators: Box<[CartesianCommunicator]>,
    process_grid: [usize; M],
    local_coords: [usize; M],
    ranks: Box<[i32]>,
}
```

Build one one-dimensional subcommunicator per topology axis with `cartesian.subgroup(&retain)`, where only `retain[axis]` is `true`.

- [ ] **Step 5: Implement `from_cartesian` with an owned duplicate**

Use the raw-handle conversion supplied by rsmpi only inside a small private unsafe function:

```rust
fn duplicate_cartesian(
    source: &CartesianCommunicator,
) -> Result<CartesianCommunicator, TopologyError> {
    // MPI_Comm_dup preserves topology attributes.
    // Call mpi::ffi::MPI_Comm_dup, check its return code, then transfer the
    // resulting non-system handle into CartesianCommunicator::try_from_raw.
}
```

The unsafe block must contain only the FFI call and `try_from_raw`. Document why the new handle is live, owned, non-system, intra-communicator, and no longer used through another owner.

- [ ] **Step 6: Implement `auto` around `MPI_Dims_create`**

Initialize `[0i32; M]`, call `mpi::ffi::MPI_Dims_create(comm.size(), M as i32, dims.as_mut_ptr())`, check the return code, convert to `[usize; M]`, then delegate to `new`.

Keep this unsafe call in a private helper named `dims_create<const M: usize>` with a safety comment and focused tests.

- [ ] **Step 7: Add topology assertions**

Expand `tests/topology.rs` to verify:

```text
local_coords maps back to local rank
rank_at rejects an out-of-range coordinate
subcommunicator(axis).size() == process_grid[axis]
MpiTopology::<1>::auto creates a one-axis grid with extent world.size()
```

- [ ] **Step 8: Run topology tests under one and four ranks**

```bash
mpiexec -n 1 cargo test -p pencil-array --test topology -- --nocapture
mpiexec -n 4 cargo test -p pencil-array --test topology -- --nocapture
cargo clippy -p pencil-array --all-targets -- -D warnings
```

Expected: all commands pass. If the MPI launcher refuses root execution in a container, set the launcher-specific allow-root variables in the test environment rather than embedding them in library code.

- [ ] **Step 9: Commit**

```bash
git add crates/pencil-array/src/topology.rs crates/pencil-array/src/error.rs crates/pencil-array/src/lib.rs crates/pencil-array/tests/topology.rs README.md
git commit -m "feat: add owned MPI cartesian topology"
```

---

### Task 6: Implement immutable `Pencil<N, M>` layouts

**Files:**
- Create: `crates/pencil-array/src/pencil.rs`
- Create: `crates/pencil-array/tests/pencil.rs`
- Modify: `crates/pencil-array/src/error.rs`
- Modify: `crates/pencil-array/src/lib.rs`

**Interfaces:**
- Consumes: `Arc<MpiTopology<M>>`, `AxisPermutation<N>`, pure geometry helpers
- Produces:

```rust
pub struct Pencil<const N: usize, const M: usize>;
pub struct PencilConfig<const N: usize, const M: usize>;

impl<const N: usize, const M: usize> Pencil<N, M> {
    pub fn new(topology: Arc<MpiTopology<M>>, global_shape: [usize; N], decomposition: [usize; M]) -> Result<Arc<Self>, PencilError>;
    pub fn new_default(topology: Arc<MpiTopology<M>>, global_shape: [usize; N]) -> Result<Arc<Self>, PencilError>;
    pub fn new_permuted(topology: Arc<MpiTopology<M>>, global_shape: [usize; N], decomposition: [usize; M], permutation: AxisPermutation<N>) -> Result<Arc<Self>, PencilError>;
    pub fn with_decomposition(self: &Arc<Self>, decomposition: [usize; M]) -> Result<Arc<Self>, PencilError>;
    pub fn with_permutation(self: &Arc<Self>, permutation: AxisPermutation<N>) -> Result<Arc<Self>, PencilError>;
    pub fn with_global_shape(self: &Arc<Self>, global_shape: [usize; N]) -> Result<Arc<Self>, PencilError>;
    pub fn reconfigured(self: &Arc<Self>, config: PencilConfig<N, M>) -> Result<Arc<Self>, PencilError>;
}
```

- [ ] **Step 1: Write failing validation tests**

Add unit tests for:

```rust
#[test]
fn default_decomposition_uses_leading_axes_for_row_major() {
    assert_eq!(default_decomposition::<5, 2>().unwrap(), [SpatialAxis::new::<5>(0).unwrap(), SpatialAxis::new::<5>(1).unwrap()]);
}

#[test]
fn decomposition_order_is_semantic() {
    assert_ne!([0usize, 2], [2usize, 0]);
}
```

Add MPI integration tests that create `[2, 2]` topology and compare `[0, 1]` with `[1, 0]` local ranges.

- [ ] **Step 2: Run tests and confirm failure**

```bash
mpiexec -n 4 cargo test -p pencil-array --test pencil -- --nocapture
```

- [ ] **Step 3: Add `PencilError`**

Include exact variants for:

```rust
InvalidDimensionRelation { spatial: usize, topology: usize }
ZeroGlobalExtent { axis: usize }
InvalidDecomposition(AxisError)
InvalidPermutation(AxisError)
SizeOverflow
Topology(TopologyError)
```

- [ ] **Step 4: Implement the private value constructor**

Use this field layout:

```rust
#[derive(Debug)]
pub struct Pencil<const N: usize, const M: usize> {
    topology: Arc<MpiTopology<M>>,
    global_shape: [usize; N],
    decomposition: [SpatialAxis; M],
    permutation: AxisPermutation<N>,
    local_ranges: [Range<usize>; N],
    local_shape_logical: [usize; N],
    local_shape_memory: [usize; N],
    local_len: usize,
    global_len: usize,
}
```

The private constructor must validate all user values before computing caches.

- [ ] **Step 5: Implement public constructors and derivation methods**

`with_decomposition` and other derivation methods must preserve the same `Arc<MpiTopology<M>>` and create a new immutable `Pencil`. They must not perform communication or touch array storage.

`PencilConfig` is a complete value, not a partial builder:

```rust
#[derive(Clone, Debug)]
pub struct PencilConfig<const N: usize, const M: usize> {
    pub global_shape: [usize; N],
    pub decomposition: [usize; M],
    pub permutation: AxisPermutation<N>,
}
```

Implement `From<&Pencil<N, M>> for PencilConfig<N, M>`.

- [ ] **Step 6: Implement reference and comparison APIs**

Implement exactly:

```rust
pub fn topology(&self) -> &Arc<MpiTopology<M>>;
pub fn global_shape(&self) -> &[usize; N];
pub fn decomposition(&self) -> &[SpatialAxis; M];
pub fn permutation(&self) -> &AxisPermutation<N>;
pub fn local_ranges(&self) -> &[Range<usize>; N];
pub fn local_shape_logical(&self) -> [usize; N];
pub fn local_shape_memory(&self) -> [usize; N];
pub fn local_len(&self) -> usize;
pub fn global_len(&self) -> usize;
pub fn ranges_at(&self, process_coords: [usize; M]) -> Result<[Range<usize>; N], PencilError>;
pub fn same_topology(&self, other: &Self) -> bool;
pub fn same_distribution(&self, other: &Self) -> bool;
pub fn same_layout(&self, other: &Self) -> bool;
```

`same_topology` uses `Arc::ptr_eq`. `same_distribution` adds global shape and ordered decomposition. `same_layout` also adds the permutation.

- [ ] **Step 7: Test derivation and `M == N`**

Add tests that verify:

```text
with_decomposition preserves topology and global shape
with_permutation changes only memory shape
M == N construction succeeds
P > L produces empty local ranges without error
unsorted decomposition produces the expected topology-axis mapping
```

- [ ] **Step 8: Run tests**

```bash
cargo test -p pencil-array --lib
mpiexec -n 4 cargo test -p pencil-array --test pencil -- --nocapture
cargo clippy -p pencil-array --all-targets -- -D warnings
```

- [ ] **Step 9: Commit**

```bash
git add crates/pencil-array/src/pencil.rs crates/pencil-array/src/error.rs crates/pencil-array/src/lib.rs crates/pencil-array/tests/pencil.rs
git commit -m "feat: add immutable pencil layouts"
```

---

### Task 7: Implement `ExtraShape` and borrowed views

**Files:**
- Create: `crates/pencil-array/src/extra_shape.rs`
- Create: `crates/pencil-array/src/view.rs`
- Modify: `crates/pencil-array/src/error.rs`
- Modify: `crates/pencil-array/src/lib.rs`

**Interfaces:**
- Consumes: checked products, `Pencil<N, M>`
- Produces:

```rust
pub struct ExtraShape;
pub struct PencilArrayView<'a, T, const N: usize, const M: usize>;
pub struct PencilArrayViewMut<'a, T, const N: usize, const M: usize>;
```

- [ ] **Step 1: Write failing `ExtraShape` tests**

```rust
#[test]
fn scalar_extra_shape_has_one_element_per_spatial_point() {
    let shape = ExtraShape::scalar();
    assert_eq!(shape.dimensions(), &[]);
    assert_eq!(shape.element_count(), 1);
}

#[test]
fn extra_shape_keeps_axis_order() {
    let shape = ExtraShape::new([3, 2]).unwrap();
    assert_eq!(shape.dimensions(), &[3, 2]);
    assert_eq!(shape.element_count(), 6);
}
```

Decide and test that zero-sized extra extents are allowed, producing `element_count == 0`; this matches ordinary empty array semantics and does not invalidate the `Pencil` spatial geometry.

- [ ] **Step 2: Run tests and confirm failure**

```bash
cargo test -p pencil-array extra_shape --lib
```

- [ ] **Step 3: Implement `ExtraShape`**

```rust
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExtraShape {
    dimensions: Box<[usize]>,
    element_count: usize,
}
```

Use `checked_product` and expose only immutable accessors.

- [ ] **Step 4: Implement read-only and mutable view structs**

Both views hold `&Pencil`, `&ExtraShape`, and a slice. Their constructors are crate-private and validate slice length. Expose the same shape, indexing, and slice APIs planned for `PencilArray` so local algorithms can be generic over owning arrays and views.

Create a crate-private trait:

```rust
pub(crate) trait LocalArrayLayout<T, const N: usize, const M: usize> {
    fn pencil(&self) -> &Pencil<N, M>;
    fn extra_shape(&self) -> &ExtraShape;
    fn as_slice(&self) -> &[T];
}
```

Do not expose this trait publicly in the first release.

- [ ] **Step 5: Test view length validation and shapes**

Use a one-rank MPI topology in an integration test to construct a `Pencil`, then verify:

```text
logical shape = [extra..., local logical spatial...]
memory shape = [extra..., local memory spatial...]
mutable changes through a view are visible in the backing Vec
```

- [ ] **Step 6: Run tests and commit**

```bash
cargo test -p pencil-array --lib
mpiexec -n 1 cargo test -p pencil-array --test pencil -- --nocapture
cargo clippy -p pencil-array --all-targets -- -D warnings

git add crates/pencil-array/src/extra_shape.rs crates/pencil-array/src/view.rs crates/pencil-array/src/error.rs crates/pencil-array/src/lib.rs crates/pencil-array/tests/pencil.rs
git commit -m "feat: add extra shapes and array views"
```

---

### Task 8: Implement owning `PencilArray<T, N, M>`

**Files:**
- Create: `crates/pencil-array/src/array.rs`
- Modify: `crates/pencil-array/src/error.rs`
- Modify: `crates/pencil-array/src/lib.rs`
- Modify: `crates/pencil-array/tests/pencil.rs`

**Interfaces:**
- Consumes: `Arc<Pencil<N, M>>`, `ExtraShape`, borrowed views
- Produces:

```rust
pub struct PencilArray<T, const N: usize, const M: usize>;
```

with the complete construction, shape, slice, view, and local-index API from the design spec.

- [ ] **Step 1: Write failing construction tests**

Test:

```text
from_vec accepts exact local length
from_vec rejects shorter and longer vectors
from_elem fills every local element
from_fn is called exactly required_len times
```

Use a one-rank `Pencil::<3, 2>` with global shape `[2, 3, 4]` and extra shape `[2]`; required length is `48`.

- [ ] **Step 2: Run tests and confirm failure**

```bash
mpiexec -n 1 cargo test -p pencil-array --test pencil pencil_array -- --nocapture
```

- [ ] **Step 3: Add `ArrayError`**

```rust
#[derive(Debug, thiserror::Error)]
pub enum ArrayError {
    #[error("storage length {actual} does not equal required length {required}")]
    StorageLengthMismatch { required: usize, actual: usize },

    #[error("array layouts are incompatible")]
    IncompatiblePencils,

    #[error("active layout index {index} is outside 0..{layout_count}")]
    InvalidActiveLayout { index: usize, layout_count: usize },

    #[error("array data is poisoned by an incomplete in-place operation")]
    Poisoned,

    #[error(transparent)]
    Geometry(#[from] GeometryError),
}
```

- [ ] **Step 4: Implement ownership and view APIs**

Use:

```rust
#[derive(Debug)]
pub struct PencilArray<T, const N: usize, const M: usize> {
    pencil: Arc<Pencil<N, M>>,
    extra_shape: ExtraShape,
    storage: Vec<T>,
}
```

Implement `view()` and `view_mut()` by borrowing the entire array and constructing the corresponding validated view.

- [ ] **Step 5: Implement logical indexing**

For `get_local(extra_indices, spatial_indices)`:

1. verify extra rank and bounds;
2. verify spatial bounds in logical order;
3. construct memory indices as `[extra indices..., spatial indices in permutation.axes() order]`;
4. construct memory shape as `[extra shape..., pencil.local_shape_memory()]`;
5. compute `row_major_offset`;
6. index the flat slice.

Add an explicit `ArrayError::ExtraIndexRankMismatch` variant for fallible internal APIs. Preserve the public `Option<&T>` contract by converting any index failure to `None`.

- [ ] **Step 6: Add exhaustive small indexing tests**

For extra shape `[2]`, local logical spatial shape `[2, 3, 4]`, and permutation `[0, 2, 1]`, fill storage with its flat offset. Iterate every logical index and verify `get_local` returns the offset computed from physical shape `[2, 2, 4, 3]` and physical indices `[extra, x, z, y]`.

- [ ] **Step 7: Run tests and commit**

```bash
cargo test -p pencil-array --lib
mpiexec -n 1 cargo test -p pencil-array --test pencil -- --nocapture
cargo clippy -p pencil-array --all-targets -- -D warnings

git add crates/pencil-array/src/array.rs crates/pencil-array/src/error.rs crates/pencil-array/src/lib.rs crates/pencil-array/tests/pencil.rs
git commit -m "feat: add owning pencil arrays"
```

---

### Task 9: Implement `ManyPencilArray` and layout-state transactions

**Files:**
- Create: `crates/pencil-array/src/many.rs`
- Modify: `crates/pencil-array/src/error.rs`
- Modify: `crates/pencil-array/src/lib.rs`
- Modify: `crates/pencil-array/tests/pencil.rs`

**Interfaces:**
- Consumes: `PencilArrayView`, `PencilArrayViewMut`, `Pencil::same_layout`
- Produces:

```rust
pub struct ManyPencilArray<T, const N: usize, const M: usize>;
pub enum OverwriteError<E>;
```

with:

```rust
pub fn from_vec(...);
pub fn from_elem(...);
pub fn pencils(&self) -> &[Arc<Pencil<N, M>>];
pub fn active_pencil(&self) -> Result<&Pencil<N, M>, ArrayError>;
pub fn active_view(&self) -> Result<PencilArrayView<'_, T, N, M>, ArrayError>;
pub fn active_view_mut(&mut self) -> Result<PencilArrayViewMut<'_, T, N, M>, ArrayError>;
pub fn overwrite_with<F, E>(&mut self, target: &Pencil<N, M>, write: F) -> Result<(), OverwriteError<E>>;
```

- [ ] **Step 1: Write failing invariant tests**

Cover:

```text
empty pencil list rejected
invalid active index rejected
different topology rejected
different global shape rejected
duplicate same_layout rejected
storage must equal max(local_len) * extra_count
```

- [ ] **Step 2: Run tests and confirm failure**

```bash
mpiexec -n 4 cargo test -p pencil-array --test pencil many_pencil_array -- --nocapture
```

- [ ] **Step 3: Implement the private state model**

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LayoutState {
    Valid(usize),
    Poisoned,
}

#[derive(Debug)]
pub struct ManyPencilArray<T, const N: usize, const M: usize> {
    pencils: Box<[Arc<Pencil<N, M>>]>,
    extra_shape: ExtraShape,
    storage: Vec<T>,
    state: LayoutState,
}
```

Do not expose layout indices or an arbitrary `view_at` API.

- [ ] **Step 4: Implement active views with exact used length**

When layout `i` is active, expose only:

```text
pencils[i].local_len() * extra_shape.element_count()
```

from the start of `storage`. Never expose unused tail capacity as active data.

- [ ] **Step 5: Implement `overwrite_with` as a poison-first transaction**

Use a private guard whose constructor sets `state = Poisoned`. The guard's `commit(index)` writes `Valid(index)`. If the closure returns `Err` or unwinds, the guard is dropped without committing and state remains poisoned.

Define:

```rust
#[derive(Debug, thiserror::Error)]
pub enum OverwriteError<E> {
    #[error(transparent)]
    Array(#[from] ArrayError),

    #[error("overwrite closure failed")]
    Writer(E),
}
```

`overwrite_with` may operate on an already poisoned array because a successful full overwrite is the recovery path.

- [ ] **Step 6: Test success, error, and panic paths**

Use:

```rust
let result = array.overwrite_with(target, |mut view| {
    view.as_mut_slice().fill(7u32);
    Ok::<_, &'static str>(())
});
assert!(result.is_ok());
assert!(array.active_pencil().unwrap().same_layout(target));
```

For error, return `Err("stop")` after modifying one element and verify `active_view()` returns `ArrayError::Poisoned`.

For panic, wrap the call in `std::panic::catch_unwind(std::panic::AssertUnwindSafe(...))`, then verify the poisoned state.

- [ ] **Step 7: Add crate-private transition hooks for the transpose plan**

Add only the minimal crate-private methods needed by the next implementation plan:

```rust
pub(crate) fn active_index(&self) -> Result<usize, ArrayError>;
pub(crate) fn find_layout(&self, pencil: &Pencil<N, M>) -> Option<usize>;
pub(crate) fn begin_in_place_write(&mut self) -> Result<LayoutWriteGuard<'_, T, N, M>, ArrayError>;
```

`LayoutWriteGuard` gives mutable access to the backing storage and can commit exactly one previously registered destination index. It must not be public.

- [ ] **Step 8: Run all tests**

```bash
cargo test -p pencil-array --lib
mpiexec -n 1 cargo test -p pencil-array --test pencil -- --nocapture
mpiexec -n 4 cargo test -p pencil-array --test pencil -- --nocapture
cargo clippy -p pencil-array --all-targets -- -D warnings
```

- [ ] **Step 9: Commit**

```bash
git add crates/pencil-array/src/many.rs crates/pencil-array/src/error.rs crates/pencil-array/src/lib.rs crates/pencil-array/tests/pencil.rs
git commit -m "feat: add shared-storage pencil arrays"
```

---

### Task 10: Lock down the public API and write crate-level documentation

**Files:**
- Modify: `crates/pencil-array/src/lib.rs`
- Modify: `crates/pencil-array/src/*.rs`
- Modify: `README.md`

**Interfaces:**
- Consumes: all interfaces from Tasks 1–9
- Produces: documented, intentionally minimal public API for the core milestone

- [ ] **Step 1: Add crate-level documentation**

At the top of `lib.rs`, document:

```text
Pencil describes spatial distribution only.
PencilArray owns one layout and one local buffer.
ManyPencilArray owns one buffer usable under several layouts, but exposes only the active layout.
Logical order is [extra..., spatial...].
Memory order is [extra..., permuted spatial...] in row-major storage.
```

Include a one-rank example that constructs a topology, pencil, extra shape, and array.

- [ ] **Step 2: Add compile-fail doctests for forbidden APIs**

Add these examples to the crate-level documentation in `lib.rs`:

```rust,compile_fail
# use pencil_array::ManyPencilArray;
# fn no_arbitrary_many_view(many: &ManyPencilArray<u32, 3, 2>) {
let _ = many.view_at(1);
# }
```

```rust,compile_fail
# use pencil_array::ManyPencilArray;
# fn no_active_layout_setter(many: &mut ManyPencilArray<u32, 3, 2>) {
many.set_active_layout(1);
# }
```

Run `cargo test -p pencil-array --doc -- --show-output` and inspect that each example fails because the method does not exist or is private, not because its setup is invalid. No separate harness or diagnostic snapshots are needed.

- [ ] **Step 3: Audit visibility**

Run:

```bash
cargo doc -p pencil-array --no-deps
```

Inspect the public item list. Keep the following private:

```text
LayoutState
LayoutWriteGuard
checked arithmetic helpers
raw Cartesian communicator accessor
subcommunicator accessor
local indexing implementation helpers
```

- [ ] **Step 4: Run the complete core verification set**

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --lib
cargo test -p pencil-array --doc
mpiexec -n 1 cargo test -p pencil-array --test topology -- --nocapture
mpiexec -n 4 cargo test -p pencil-array --test topology -- --nocapture
mpiexec -n 1 cargo test -p pencil-array --test pencil -- --nocapture
mpiexec -n 4 cargo test -p pencil-array --test pencil -- --nocapture
cargo doc --workspace --no-deps
```

Expected: every command exits successfully.

- [ ] **Step 5: Commit**

```bash
git add README.md crates/pencil-array/src crates/pencil-array/tests
git commit -m "docs: define pencil array core API"
```

---

## Plan Boundary and Follow-on Plans

This plan stops before local or distributed transposition. It leaves a working, independently testable data-model crate with the exact private hooks required by the transpose subsystem.

Create and approve separate plans in this order before implementing their code:

1. `pencil-array-transpose`: local permutation, `ExchangePattern`, `AllToAllV`, point-to-point, collective precondition checks, and `TransposeWorkspace`.
2. `pencil-fft`: RustFFT/RealFFT local stages, common transform path, one-intermediate out-of-place C2C/R2C, and C2C in-place state wrapper.
3. `pencil-validation`: Julia cross-language drivers, MPI failure matrix, benchmarks, memory accounting, and release documentation.

## Self-Review Record

- Spec coverage in this plan: design sections 6–13 and the core portions of sections 30–31.
- Intentionally delegated: design sections 14–29 and distributed portions of sections 31–34.
- Type names checked across all tasks: `MpiTopology`, `SpatialAxis`, `AxisPermutation`, `Pencil`, `PencilConfig`, `ExtraShape`, `PencilArray`, `PencilArrayView`, `PencilArrayViewMut`, `ManyPencilArray`.
- No task requires FFT dependencies or public storage abstraction.
- Every task ends with a runnable verification command and an isolated commit.
