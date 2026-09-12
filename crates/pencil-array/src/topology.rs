use std::{fmt, sync::Arc};

use mpi::topology::{CartesianCommunicator, Communicator, IntoTopology};

use crate::{GeometryError, TopologyError, checked::checked_product};

/// An owned Cartesian MPI topology and its one-dimensional axis communicators.
pub struct MpiTopology<const M: usize> {
    cartesian: CartesianCommunicator,
    subcommunicators: Box<[CartesianCommunicator]>,
    process_grid: [usize; M],
    local_coords: [usize; M],
}

impl<const M: usize> MpiTopology<M> {
    /// Creates a non-periodic Cartesian topology with the requested process grid.
    ///
    /// This is collective over `comm`.
    pub fn new<C: Communicator>(
        comm: &C,
        process_grid: [usize; M],
    ) -> Result<Arc<Self>, TopologyError> {
        Self::validate_dimension_count()?;

        for (axis, extent) in process_grid.iter().copied().enumerate() {
            if extent == 0 {
                return Err(TopologyError::ZeroExtent { axis });
            }
        }

        let grid_size = checked_product(&process_grid)?;
        let communicator_size = usize::try_from(comm.size())
            .map_err(|_| TopologyError::Geometry(GeometryError::CountOverflow))?;
        if grid_size != communicator_size {
            return Err(TopologyError::CommunicatorSizeMismatch {
                grid_size,
                communicator_size,
            });
        }

        let mut dims = [0i32; M];
        for (slot, extent) in dims.iter_mut().zip(process_grid) {
            *slot = i32::try_from(extent).map_err(|_| GeometryError::CountOverflow)?;
        }
        let periods = [false; M];
        let cartesian = comm
            .create_cartesian_communicator(&dims, &periods, false)
            .ok_or(TopologyError::CartesianCreationFailed)?;

        Self::from_owned_cartesian(cartesian)
    }

    /// Creates a balanced non-periodic Cartesian topology using `MPI_Dims_create`.
    ///
    /// This is collective over `comm`.
    pub fn auto<C: Communicator>(comm: &C) -> Result<Arc<Self>, TopologyError> {
        Self::validate_dimension_count()?;

        let dimensions = i32::try_from(M).map_err(|_| GeometryError::CountOverflow)?;
        let mut dims = [0i32; M];
        // SAFETY: `dims` contains exactly `dimensions` writable MPI Count values,
        // and `comm.size()` is a valid positive communicator size.
        let status =
            unsafe { mpi::ffi::MPI_Dims_create(comm.size(), dimensions, dims.as_mut_ptr()) };
        if status != 0 {
            return Err(TopologyError::Mpi {
                operation: "MPI_Dims_create",
                code: status,
            });
        }

        let mut process_grid = [0usize; M];
        for (slot, extent) in process_grid.iter_mut().zip(dims) {
            *slot = usize::try_from(extent).map_err(|_| GeometryError::CountOverflow)?;
        }

        Self::new(comm, process_grid)
    }

    /// Duplicates an existing Cartesian communicator and owns the duplicate.
    ///
    /// This is collective over `comm`.
    pub fn from_cartesian(comm: &CartesianCommunicator) -> Result<Arc<Self>, TopologyError> {
        let duplicated = match comm.duplicate().into_topology() {
            IntoTopology::Cartesian(cartesian) => cartesian,
            _ => return Err(TopologyError::CartesianCreationFailed),
        };
        Self::from_owned_cartesian(duplicated)
    }

    /// Returns the Cartesian process-grid extents.
    pub fn process_grid(&self) -> &[usize; M] {
        &self.process_grid
    }

    /// Returns the calling rank's Cartesian coordinates.
    pub fn local_coords(&self) -> &[usize; M] {
        &self.local_coords
    }

    /// Returns the calling process's rank in the Cartesian communicator.
    pub fn rank(&self) -> i32 {
        self.cartesian.rank()
    }

    /// Returns the number of processes in the Cartesian communicator.
    pub fn size(&self) -> usize {
        usize::try_from(self.cartesian.size()).expect("MPI communicator size is non-negative")
    }

    /// Maps valid zero-based Cartesian coordinates to a communicator rank.
    pub fn rank_at(&self, coords: [usize; M]) -> Result<i32, TopologyError> {
        let mut mpi_coords = [0i32; M];
        for axis in 0..M {
            let coordinate = coords[axis];
            let extent = self.process_grid[axis];
            if coordinate >= extent {
                return Err(TopologyError::CoordinateOutOfBounds {
                    axis,
                    coordinate,
                    extent,
                });
            }
            mpi_coords[axis] =
                i32::try_from(coordinate).map_err(|_| GeometryError::CountOverflow)?;
        }

        // SAFETY: `mpi_coords` has exactly M entries, matching the communicator
        // dimensionality, and every coordinate was checked against its extent.
        Ok(unsafe { self.cartesian.coordinates_to_rank_unchecked(&mpi_coords) })
    }

    /// Returns the number of processes in the one-dimensional communicator for an axis.
    pub fn subcommunicator_size(&self, topology_axis: usize) -> Result<usize, TopologyError> {
        let communicator =
            self.subcommunicators
                .get(topology_axis)
                .ok_or(TopologyError::AxisOutOfBounds {
                    axis: topology_axis,
                    dimensions: M,
                })?;
        usize::try_from(communicator.size()).map_err(|_| GeometryError::CountOverflow.into())
    }

    fn validate_dimension_count() -> Result<(), TopologyError> {
        if M == 0 {
            Err(TopologyError::ZeroDimensions)
        } else {
            Ok(())
        }
    }

    fn from_owned_cartesian(cartesian: CartesianCommunicator) -> Result<Arc<Self>, TopologyError> {
        Self::validate_dimension_count()?;

        let layout = cartesian.get_layout();
        if layout.dims.len() != M || layout.coords.len() != M {
            return Err(TopologyError::DimensionMismatch {
                expected: M,
                actual: layout.dims.len(),
            });
        }

        let mut process_grid = [0usize; M];
        for (slot, extent) in process_grid.iter_mut().zip(layout.dims.iter().copied()) {
            *slot = usize::try_from(extent).map_err(|_| GeometryError::CountOverflow)?;
        }

        let mut local_coords = [0usize; M];
        for (slot, coordinate) in local_coords.iter_mut().zip(layout.coords.iter().copied()) {
            *slot = usize::try_from(coordinate)
                .map_err(|_| TopologyError::Geometry(GeometryError::CountOverflow))?;
        }

        let mut subcommunicators = Vec::with_capacity(M);
        for axis in 0..M {
            let mut retained_axes = vec![false; M];
            retained_axes[axis] = true;
            let subcommunicator = cartesian.subgroup(&retained_axes);
            let actual = usize::try_from(subcommunicator.size())
                .map_err(|_| GeometryError::CountOverflow)?;
            let expected = process_grid[axis];
            if actual != expected {
                return Err(TopologyError::SubcommunicatorSizeMismatch {
                    axis,
                    expected,
                    actual,
                });
            }
            subcommunicators.push(subcommunicator);
        }

        Ok(Self {
            cartesian,
            subcommunicators: subcommunicators.into_boxed_slice(),
            process_grid,
            local_coords,
        }
        .into_shared())
    }

    #[allow(clippy::arc_with_non_send_sync)]
    fn into_shared(self) -> Arc<Self> {
        // rsmpi communicators are intentionally !Send and !Sync unless an MPI
        // threading contract is established. Arc is used here only for shared
        // ownership across arrays and plans; it does not make this value
        // transferable between threads.
        Arc::new(self)
    }
}

impl<const M: usize> fmt::Debug for MpiTopology<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MpiTopology")
            .field("process_grid", &self.process_grid)
            .field("local_coords", &self.local_coords)
            .field("rank", &self.rank())
            .field("size", &self.size())
            .finish_non_exhaustive()
    }
}
