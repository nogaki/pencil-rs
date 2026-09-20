use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
/// Errors validating spatial axes, permutations, and decompositions.
pub enum AxisError {
    /// An axis was outside the valid dimension range.
    #[error("axis {axis} is outside 0..{dimensions}")]
    OutOfBounds {
        /// The rejected zero-based axis.
        axis: usize,
        /// The number of available dimensions.
        dimensions: usize,
    },

    /// An axis appeared more than once where distinct axes are required.
    #[error("axis {axis} occurs more than once")]
    Duplicate {
        /// The repeated zero-based axis.
        axis: usize,
    },

    /// The topology rank was zero or exceeded the spatial rank.
    #[error(
        "topology dimension M={topology_dimensions} must satisfy 1 <= M <= N={spatial_dimensions}"
    )]
    InvalidDecompositionRank {
        /// The number of spatial dimensions.
        spatial_dimensions: usize,
        /// The number of topology dimensions.
        topology_dimensions: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
/// Errors computing local geometry and row-major offsets.
pub enum GeometryError {
    /// A shape, length, or offset did not fit `usize`.
    #[error("shape or offset calculation overflowed usize")]
    SizeOverflow,

    /// A value did not fit the MPI `Count` representation.
    #[error("value does not fit MPI Count (i32)")]
    CountOverflow,

    /// A partition operation was requested with zero parts.
    #[error("partition count must be positive")]
    ZeroPartitions,

    /// An index tuple had a different rank from its shape.
    #[error("shape rank {shape_rank} does not equal index rank {index_rank}")]
    RankMismatch {
        /// The number of shape dimensions.
        shape_rank: usize,
        /// The number of supplied indices.
        index_rank: usize,
    },

    /// A process-grid extent was zero.
    #[error("process grid extent is zero on topology axis {axis}")]
    ZeroProcessExtent {
        /// The topology axis with zero extent.
        axis: usize,
    },

    /// A process coordinate was outside its topology-axis extent.
    #[error("process coordinate {coordinate} is outside 0..{extent} on topology axis {axis}")]
    ProcessCoordinateOutOfBounds {
        /// The topology axis being indexed.
        axis: usize,
        /// The rejected coordinate.
        coordinate: usize,
        /// The process-grid extent.
        extent: usize,
    },

    /// A local index was outside its extent in the supplied shape.
    #[error("local index {index} is outside 0..{extent} on axis {axis} of the supplied shape")]
    LocalIndexOutOfBounds {
        /// The axis position in the supplied shape.
        axis: usize,
        /// The rejected local index.
        index: usize,
        /// The corresponding extent in the supplied shape.
        extent: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
/// Errors validating undistributed extra dimensions.
pub enum ShapeError {
    /// The product of the extra dimensions did not fit `usize`.
    #[error("extra shape element count overflowed usize")]
    SizeOverflow,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
/// Errors constructing a borrowed [`crate::LocalGrid`].
pub enum LocalGridError {
    /// A supplied global coordinate axis had the wrong length.
    #[error("coordinate axis {axis} has length {actual}, expected global extent {expected}")]
    AxisLengthMismatch {
        /// The zero-based spatial axis with the mismatch.
        axis: usize,
        /// The expected global extent from the pencil.
        expected: usize,
        /// The supplied coordinate-axis length.
        actual: usize,
    },

    /// The local coordinate-grid shape did not fit `usize`.
    #[error("local coordinate-grid shape overflowed usize")]
    SizeOverflow,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
/// Errors constructing or accessing local array storage.
pub enum ArrayError {
    /// The supplied storage did not have the required exact length.
    #[error("storage length {actual} does not equal required length {required}")]
    StorageLengthMismatch {
        /// The length required by the layout.
        required: usize,
        /// The supplied storage length.
        actual: usize,
    },

    /// Allocation for a validated storage length failed.
    #[error("failed to allocate storage for {required} elements")]
    AllocationFailed {
        /// The requested number of elements.
        required: usize,
    },

    /// The number of extra indices did not match the extra shape rank.
    #[error("extra index rank {actual} does not equal required rank {required}")]
    ExtraIndexRankMismatch {
        /// The required number of extra indices.
        required: usize,
        /// The supplied number of extra indices.
        actual: usize,
    },

    /// Registered or requested pencil layouts were incompatible.
    #[error("array layouts are incompatible")]
    IncompatiblePencils,

    /// The active layout index was outside the registry.
    #[error("active layout index {index} is outside 0..{layout_count}")]
    InvalidActiveLayout {
        /// The rejected active index.
        index: usize,
        /// The number of registered layouts.
        layout_count: usize,
    },

    /// An incomplete write left the contents unusable through active access.
    #[error("array data is poisoned by an incomplete in-place operation")]
    Poisoned,

    /// A lower-level geometry calculation failed.
    #[error(transparent)]
    Geometry(#[from] GeometryError),
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
/// Errors returned by global array reductions and root gathers.
///
/// Reduction and gather calls perform their ordinary validation collectively
/// before any variable-count or point-to-point communication. MPI failures,
/// process loss, and a callback that panics are outside that recovery
/// guarantee.
pub enum CollectiveError {
    /// Ranks supplied different operation or array descriptors.
    #[error("collective array descriptors differ between ranks")]
    CollectiveDescriptorMismatch,

    /// At least one rank rejected a collective precondition.
    #[error("a collective array precondition failed on another rank")]
    CollectivePreconditionFailed,

    /// Descriptor, metadata, or allocation preparation failed collectively.
    #[error("collective array metadata preparation failed")]
    PreparationFailed,

    /// A root rank was not a rank in the topology Cartesian communicator.
    #[error("root rank {root} is outside 0..{size}")]
    RootOutOfBounds {
        /// The supplied root rank after conversion to a signed MPI rank.
        root: i64,
        /// The communicator size.
        size: usize,
    },

    /// The payload type has no addressable elements and cannot be seeded safely.
    #[error("zero-sized gather payload types are unsupported")]
    ZeroSizedTypeUnsupported,

    /// A local count, global length, or MPI count conversion overflowed.
    #[error("array collective count or size overflowed")]
    CountOverflow,

    /// A required descriptor, reduction scratch, or root gather buffer could
    /// not be reserved.
    #[error("array collective allocation failed for {elements} elements")]
    AllocationFailed {
        /// The number of elements that could not be reserved.
        elements: usize,
    },

    /// A checked integer sum overflowed on at least one rank.
    #[error("checked integer array sum overflowed")]
    IntegerOverflow,

    /// A lower-level local array validation failed.
    #[error(transparent)]
    Array(#[from] ArrayError),
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
/// Errors constructing or querying an MPI Cartesian topology.
pub enum TopologyError {
    /// The supplied communicator was an intercommunicator.
    #[error("Cartesian topology requires an intracommunicator")]
    InterCommunicatorUnsupported,

    /// Ranks supplied different topology dimension counts.
    #[error("topology dimension counts differ between communicator ranks")]
    InconsistentDimensions,

    /// Ranks supplied different process-grid extents.
    #[error("process grids differ between communicator ranks")]
    InconsistentProcessGrid,

    /// Another rank rejected its local constructor input.
    #[error("another communicator rank failed topology validation")]
    CollectivePreconditionFailed,

    /// A zero-dimensional Cartesian topology was requested.
    #[error("topology dimension count must be positive")]
    ZeroDimensions,

    /// A process-grid axis had zero extent.
    #[error("process-grid extent on axis {axis} must be positive")]
    ZeroExtent {
        /// The topology axis with zero extent.
        axis: usize,
    },

    /// The process-grid product did not match the communicator size.
    #[error("process-grid size {grid_size} does not equal communicator size {communicator_size}")]
    CommunicatorSizeMismatch {
        /// The product of the requested process-grid extents.
        grid_size: usize,
        /// The number of processes in the communicator.
        communicator_size: usize,
    },

    /// MPI could not create or duplicate a Cartesian communicator.
    #[error("MPI did not create a Cartesian communicator")]
    CartesianCreationFailed,

    /// An existing Cartesian communicator had an unexpected dimension count.
    #[error("Cartesian communicator has {actual} dimensions, expected {expected}")]
    DimensionMismatch {
        /// The compile-time topology dimension count.
        expected: usize,
        /// The communicator's actual dimension count.
        actual: usize,
    },

    /// A Cartesian coordinate was outside its process-grid extent.
    #[error("coordinate {coordinate} is outside 0..{extent} on topology axis {axis}")]
    CoordinateOutOfBounds {
        /// The topology axis being indexed.
        axis: usize,
        /// The rejected coordinate.
        coordinate: usize,
        /// The process-grid extent.
        extent: usize,
    },

    /// A topology axis was outside `0..M`.
    #[error("topology axis {axis} is outside 0..{dimensions}")]
    AxisOutOfBounds {
        /// The rejected topology axis.
        axis: usize,
        /// The number of topology dimensions.
        dimensions: usize,
    },

    /// A derived axis communicator had an unexpected size.
    #[error("subcommunicator on topology axis {axis} has size {actual}, expected {expected}")]
    SubcommunicatorSizeMismatch {
        /// The topology axis used to form the communicator.
        axis: usize,
        /// The expected communicator size.
        expected: usize,
        /// The actual communicator size.
        actual: usize,
    },

    /// A direct MPI operation returned a nonzero status code.
    #[error("MPI topology operation {operation} failed with error code {code}")]
    Mpi {
        /// The MPI operation that failed.
        operation: &'static str,
        /// The status code returned by MPI.
        code: i32,
    },

    /// A topology value could not be represented safely.
    #[error(transparent)]
    Geometry(#[from] GeometryError),
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
/// Errors validating or deriving a [`crate::Pencil`] layout.
pub enum PencilError {
    /// The topology rank was zero or exceeded the spatial rank.
    #[error("topology dimension M={topology} must satisfy 1 <= M <= spatial dimension N={spatial}")]
    InvalidDimensionRelation {
        /// The number of spatial dimensions.
        spatial: usize,
        /// The number of topology dimensions.
        topology: usize,
    },

    /// A global spatial extent was zero.
    #[error("global extent on spatial axis {axis} must be positive")]
    ZeroGlobalExtent {
        /// The spatial axis with zero extent.
        axis: usize,
    },

    /// The ordered decomposition was invalid.
    #[error("invalid decomposition: {0}")]
    InvalidDecomposition(AxisError),

    /// The row-major memory-axis permutation was invalid.
    #[error("invalid memory-axis permutation: {0}")]
    InvalidPermutation(AxisError),

    /// A pencil shape or range calculation did not fit `usize`.
    #[error("pencil shape or range calculation overflowed usize")]
    SizeOverflow,

    /// Topology validation or lookup failed.
    #[error(transparent)]
    Topology(#[from] TopologyError),
}
