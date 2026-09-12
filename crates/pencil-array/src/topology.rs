use std::{fmt, sync::Arc};

use mpi::topology::{CartesianCommunicator, Communicator};

use crate::{GeometryError, TopologyError, checked::checked_product};

/// An owned, non-periodic Cartesian MPI topology.
pub struct MpiTopology<const M: usize> {
    cartesian: CartesianCommunicator,
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
        if M == 0 {
            return Err(TopologyError::ZeroDimensions);
        }

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

        let layout = cartesian.get_layout();
        if layout.dims.len() != M || layout.coords.len() != M {
            return Err(TopologyError::DimensionMismatch {
                expected: M,
                actual: layout.dims.len(),
            });
        }

        let mut local_coords = [0usize; M];
        for (slot, coordinate) in local_coords.iter_mut().zip(layout.coords) {
            *slot = usize::try_from(coordinate)
                .map_err(|_| TopologyError::Geometry(GeometryError::CountOverflow))?;
        }

        Ok(Arc::new(Self {
            cartesian,
            process_grid,
            local_coords,
        }))
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
