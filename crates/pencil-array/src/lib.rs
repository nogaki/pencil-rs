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
//! storage lengths and borrowing in safe Rust; this crate does not yet perform
//! redistribution, transposition, or FFTs.
//!
//! # One-rank example
//!
//! Run this example as one ordinary process. The array is declared after the
//! MPI universe, so it and its topology are dropped before MPI finalization.
//!
//! ```
//! use mpi::traits::*;
//! use pencil_array::{ExtraShape, MpiTopology, Pencil, PencilArray};
//!
//! let universe = mpi::initialize().expect("MPI must not already be initialized");
//! let world = universe.world();
//! assert_eq!(world.size(), 1, "this example requires exactly one MPI rank");
//!
//! let topology = MpiTopology::<1>::new(&world, [1])?;
//! let pencil = Pencil::<2, 1>::new(topology, [4, 6], [0])?;
//! let extra_shape = ExtraShape::scalar();
//! let array = PencilArray::from_elem(pencil, extra_shape, 0.0_f64)?;
//!
//! assert_eq!(array.logical_shape(), [4, 6]);
//! assert_eq!(array.memory_shape(), [4, 6]);
//! assert_eq!(array.len(), 24);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod array;
mod axis;
mod checked;
mod decomposition;
mod error;
mod extra_shape;
mod geometry;
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
pub use many::{ManyPencilArray, OverwriteError};
pub use pencil::{Pencil, PencilConfig};
pub use topology::MpiTopology;
pub use view::{PencilArrayView, PencilArrayViewMut};
