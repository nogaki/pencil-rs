#![forbid(unsafe_op_in_unsafe_fn)]
#![deny(missing_debug_implementations)]

//! MPI-distributed, row-major multidimensional array foundations.

mod checked;
mod error;

pub use error::{AxisError, GeometryError};
