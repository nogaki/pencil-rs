#![forbid(unsafe_op_in_unsafe_fn)]
#![deny(missing_debug_implementations)]

//! MPI-distributed, row-major multidimensional array foundations.

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
