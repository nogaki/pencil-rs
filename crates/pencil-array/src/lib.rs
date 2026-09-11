#![forbid(unsafe_op_in_unsafe_fn)]
#![deny(missing_debug_implementations)]

//! MPI-distributed, row-major multidimensional array foundations.

mod axis;
mod checked;
mod error;

pub use axis::{AxisPermutation, SpatialAxis};
pub use error::{AxisError, GeometryError};
