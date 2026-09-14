#![forbid(unsafe_op_in_unsafe_fn)]
#![deny(missing_debug_implementations)]
#![warn(missing_docs)]

//! MPI-distributed, row-major multidimensional array foundations.
//!
//! A [`Pencil`] describes spatial distribution only. A [`PencilArray`] owns
//! one pencil layout and one local buffer. A [`ManyPencilArray`] owns one
//! buffer sized for several registered layouts, while exposing a view of only
//! its active layout.
//!
//! Array dimensions are interpreted in logical order as
//! `[extra..., spatial...]`. The row-major buffer uses memory order
//! `[extra..., permuted spatial...]`; [`AxisPermutation`] changes only the
//! spatial suffix. [`ExtraShape::scalar`] represents zero extra dimensions
//! and contributes one element per local spatial point.
//!
//! Topology construction is collective and accepts only MPI
//! intracommunicators. Every topology, pencil, array, and borrowed view must be
//! dropped before MPI is finalized. The owning arrays and their views enforce
//! storage lengths and borrowing in safe Rust. [`LocalTransposePlan`] provides
//! process-local memory-axis permutations; data redistribution, distributed
//! transposition, and FFTs are not yet implemented.
//!
//! # One-rank example
//!
//! Run this example as one ordinary process. The array is declared after the
//! MPI universe, so it and its topology are dropped before MPI finalization.
//!
//! ```
//! use mpi::traits::*;
//! use pencil_array::{
//!     AxisPermutation, ExtraShape, LocalTransposePlan, MpiTopology, Pencil, PencilArray,
//! };
//!
//! let universe = mpi::initialize().expect("MPI must not already be initialized");
//! let world = universe.world();
//! assert_eq!(world.size(), 1, "this example requires exactly one MPI rank");
//!
//! let topology = MpiTopology::<1>::new(&world, [1])?;
//! let source_pencil = Pencil::<2, 1>::new(topology, [4, 6], [0])?;
//! let destination_pencil =
//!     source_pencil.with_permutation(AxisPermutation::new([1, 0])?)?;
//! let plan = LocalTransposePlan::new(source_pencil.clone(), destination_pencil.clone())?;
//! let extra_shape = ExtraShape::scalar();
//! let source = PencilArray::from_elem(source_pencil, extra_shape.clone(), 0.0_f64)?;
//! let mut destination =
//!     PencilArray::from_elem(destination_pencil, extra_shape, 0.0_f64)?;
//! plan.execute_views(source.view(), destination.view_mut())?;
//!
//! assert_eq!(source.logical_shape(), [4, 6]);
//! assert_eq!(destination.memory_shape(), [6, 4]);
//! assert_eq!(destination.len(), 24);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Borrowing and API boundaries
//!
//! Shared and exclusive views can be borrowed in sequence once the earlier
//! borrow is no longer used:
//!
//! ```
//! use pencil_array::{ManyPencilArray, PencilArray};
//!
//! fn borrow_active<T>(many: &mut ManyPencilArray<T, 3, 2>) {
//!     let _view = many.active_view().unwrap();
//!     let _view_mut = many.active_view_mut().unwrap();
//! }
//!
//! fn borrow_owner<T>(array: &mut PencilArray<T, 3, 2>) {
//!     let _view = array.view();
//!     let _view_mut = array.view_mut();
//! }
//! ```
//!
//! Two live mutable views cannot alias the same storage:
//!
//! ```compile_fail
//! # use pencil_array::ManyPencilArray;
//! # fn alias_mutable_view<T>(many: &mut ManyPencilArray<T, 3, 2>) {
//! let first = many.active_view_mut().unwrap();
//! let second = many.active_view_mut().unwrap();
//! let _ = (first, second);
//! # }
//! ```
//!
//! A view cannot outlive its owning array:
//!
//! ```compile_fail
//! # use pencil_array::{PencilArray, PencilArrayView};
//! fn outlive_owner<'a, T>(array: PencilArray<T, 3, 2>) -> PencilArrayView<'a, T, 3, 2> {
//!     array.view()
//! }
//! ```
//!
//! Shared-storage arrays expose only the active layout, not an arbitrary view:
//!
//! ```compile_fail
//! # use pencil_array::ManyPencilArray;
//! # fn arbitrary_view<T>(many: &ManyPencilArray<T, 3, 2>) {
//! let _ = many.view_at(1);
//! # }
//! ```
//!
//! The active layout cannot be changed without a complete overwrite:
//!
//! ```compile_fail
//! # use pencil_array::ManyPencilArray;
//! # fn set_active<T>(many: &mut ManyPencilArray<T, 3, 2>) {
//! many.set_active_layout(1);
//! # }
//! ```
//!
//! Internal write transactions are not part of the public API:
//!
//! ```compile_fail
//! # use pencil_array::ManyPencilArray;
//! # fn start_internal_write<T>(many: &mut ManyPencilArray<T, 3, 2>) {
//! let _ = many.begin_in_place_write();
//! # }
//! ```
//!
//! Views must be borrowed through an owning array, not constructed directly:
//!
//! ```compile_fail
//! # use pencil_array::{ExtraShape, Pencil, PencilArrayView};
//! # fn construct_view<T>(pencil: &Pencil<3, 2>, extra_shape: &ExtraShape, storage: &[T]) {
//! let _ = PencilArrayView::new(pencil, extra_shape, storage);
//! # }
//! ```
//!
//! ```compile_fail
//! # use pencil_array::{ExtraShape, Pencil, PencilArrayViewMut};
//! # fn construct_view_mut<T>(pencil: &Pencil<3, 2>, extra_shape: &ExtraShape, storage: &mut [T]) {
//! let _ = PencilArrayViewMut::new(pencil, extra_shape, storage);
//! # }
//! ```
//!
//! The topology's owned communicators are private:
//!
//! ```compile_fail
//! # use pencil_array::MpiTopology;
//! # fn expose_cartesian(topology: &MpiTopology<2>) {
//! let _ = topology.cartesian();
//! # }
//! ```
//!
//! ```compile_fail
//! # use pencil_array::MpiTopology;
//! # fn expose_subcommunicator(topology: &MpiTopology<2>) {
//! let _ = topology.subcommunicator(0);
//! # }
//! ```
//!
//! Internal layout state, write guards, and implementation traits are not exported:
//!
//! ```compile_fail
//! use pencil_array::LayoutState;
//! ```
//!
//! ```compile_fail
//! use pencil_array::LayoutWriteGuard;
//! ```
//!
//! ```compile_fail
//! use pencil_array::LocalArrayLayout;
//! ```

// ponytail: rustdoc checks API restrictions; snapshots only if exact diagnostics become a contract.
mod array;
mod axis;
mod checked;
mod decomposition;
mod error;
mod extra_shape;
mod geometry;
mod local_transpose;
mod many;
mod pencil;
mod topology;
mod view;

pub use array::PencilArray;
pub use axis::{AxisPermutation, SpatialAxis};
pub use decomposition::Decomposition;
pub use error::{ArrayError, AxisError, GeometryError, PencilError, ShapeError, TopologyError};
pub use extra_shape::ExtraShape;
pub use geometry::partition_range;
pub use local_transpose::{LocalTransposeError, LocalTransposePlan};
pub use many::{ManyPencilArray, OverwriteError};
pub use pencil::{Pencil, PencilConfig};
pub use topology::MpiTopology;
pub use view::{PencilArrayView, PencilArrayViewMut};
