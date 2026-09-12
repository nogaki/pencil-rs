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

#[derive(Debug, Clone, PartialEq, Eq, Error)]
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
    CoordinateOutOfBounds {
        axis: usize,
        coordinate: usize,
        extent: usize,
    },

    #[error("topology axis {axis} is outside 0..{dimensions}")]
    AxisOutOfBounds { axis: usize, dimensions: usize },

    #[error("subcommunicator on topology axis {axis} has size {actual}, expected {expected}")]
    SubcommunicatorSizeMismatch {
        axis: usize,
        expected: usize,
        actual: usize,
    },

    #[error("MPI topology operation {operation} failed with error code {code}")]
    Mpi { operation: &'static str, code: i32 },

    #[error(transparent)]
    Geometry(#[from] GeometryError),
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PencilError {
    #[error("topology dimension M={topology} must satisfy 1 <= M <= spatial dimension N={spatial}")]
    InvalidDimensionRelation { spatial: usize, topology: usize },

    #[error("global extent on spatial axis {axis} must be positive")]
    ZeroGlobalExtent { axis: usize },

    #[error("invalid decomposition: {0}")]
    InvalidDecomposition(AxisError),

    #[error("invalid memory-axis permutation: {0}")]
    InvalidPermutation(AxisError),

    #[error("pencil shape or range calculation overflowed usize")]
    SizeOverflow,

    #[error(transparent)]
    Topology(#[from] TopologyError),
}
