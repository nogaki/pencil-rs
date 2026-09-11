use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum AxisError {
    #[error("axis {axis} is outside 0..{dimensions}")]
    OutOfBounds { axis: usize, dimensions: usize },

    #[error("axis {axis} occurs more than once")]
    Duplicate { axis: usize },

    #[error(
        "topology dimension M={topology_dimensions} must satisfy 1 <= M <= N={spatial_dimensions}"
    )]
    InvalidDecompositionRank {
        spatial_dimensions: usize,
        topology_dimensions: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum GeometryError {
    #[error("shape or offset calculation overflowed usize")]
    SizeOverflow,

    #[error("value does not fit MPI Count (i32)")]
    CountOverflow,

    #[error("process grid extent is zero on topology axis {axis}")]
    ZeroProcessExtent { axis: usize },

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
