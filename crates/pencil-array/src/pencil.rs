use std::{ops::Range, sync::Arc};

use crate::{
    AxisPermutation, Decomposition, GeometryError, MpiTopology, PencilError, SpatialAxis,
    decomposition::local_data_range,
};

/// An immutable MPI distribution and row-major memory layout for an N-dimensional array.
#[derive(Debug)]
pub struct Pencil<const N: usize, const M: usize> {
    topology: Arc<MpiTopology<M>>,
    global_shape: [usize; N],
    decomposition: [SpatialAxis; M],
    permutation: AxisPermutation<N>,
    local_ranges: [Range<usize>; N],
}

impl<const N: usize, const M: usize> Pencil<N, M> {
    /// Creates an unpermuted pencil with an explicit ordered decomposition.
    pub fn new(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        decomposition: [usize; M],
    ) -> Result<Arc<Self>, PencilError> {
        Self::new_permuted(
            topology,
            global_shape,
            decomposition,
            AxisPermutation::identity(),
        )
    }

    /// Creates an unpermuted row-major pencil decomposed over the leading M axes.
    pub fn new_default(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
    ) -> Result<Arc<Self>, PencilError> {
        Self::new(topology, global_shape, std::array::from_fn(|axis| axis))
    }

    /// Creates a pencil with explicit distribution and memory-axis permutation.
    pub fn new_permuted(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        decomposition: [usize; M],
        permutation: AxisPermutation<N>,
    ) -> Result<Arc<Self>, PencilError> {
        if M == 0 || M > N {
            return Err(PencilError::InvalidDimensionRelation {
                spatial: N,
                topology: M,
            });
        }
        for (axis, extent) in global_shape.iter().copied().enumerate() {
            if extent == 0 {
                return Err(PencilError::ZeroGlobalExtent { axis });
            }
        }

        let decomposition =
            Decomposition::<N, M>::new(decomposition).map_err(PencilError::InvalidDecomposition)?;
        let local_ranges = ranges_for(
            &topology,
            global_shape,
            &decomposition,
            *topology.local_coords(),
        )?;

        Ok(Self {
            topology,
            global_shape,
            decomposition: *decomposition.axes(),
            permutation,
            local_ranges,
        }
        .into_shared())
    }

    /// Returns the ordered spatial axes associated with topology axes.
    pub fn decomposition(&self) -> &[SpatialAxis; M] {
        &self.decomposition
    }

    /// Returns the logical-to-memory spatial-axis permutation.
    pub fn permutation(&self) -> &AxisPermutation<N> {
        &self.permutation
    }

    /// Returns the calling rank's zero-based half-open logical ranges.
    pub fn local_ranges(&self) -> &[Range<usize>; N] {
        &self.local_ranges
    }

    /// Returns true when both pencils share the identical topology object.
    pub fn same_topology(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.topology, &other.topology)
    }

    /// Returns true when topology, global shape, and ordered decomposition match.
    pub fn same_distribution(&self, other: &Self) -> bool {
        self.same_topology(other)
            && self.global_shape == other.global_shape
            && self.decomposition == other.decomposition
    }

    #[allow(clippy::arc_with_non_send_sync)]
    fn into_shared(self) -> Arc<Self> {
        // Pencil shares an rsmpi-backed topology. Arc provides ownership
        // sharing only and does not make the value Send or Sync.
        Arc::new(self)
    }
}

fn ranges_for<const N: usize, const M: usize>(
    topology: &MpiTopology<M>,
    global_shape: [usize; N],
    decomposition: &Decomposition<N, M>,
    process_coords: [usize; M],
) -> Result<[Range<usize>; N], PencilError> {
    topology.rank_at(process_coords)?;
    let process_grid = decomposition.complete_process_grid(*topology.process_grid());
    let process_coords = decomposition.complete_process_coords(process_coords);
    let mut ranges = std::array::from_fn(|_| 0..0);

    for axis in 0..N {
        ranges[axis] =
            local_data_range(process_coords[axis], process_grid[axis], global_shape[axis])
                .map_err(map_geometry_error)?;
    }

    Ok(ranges)
}

fn map_geometry_error(error: GeometryError) -> PencilError {
    match error {
        GeometryError::SizeOverflow | GeometryError::CountOverflow => PencilError::SizeOverflow,
        other => unreachable!("validated pencil geometry produced {other}"),
    }
}
