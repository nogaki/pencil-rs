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

    #[error(
        "process coordinate {coordinate} is outside 0..{extent} on topology axis {axis}"
    )]
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
