use std::{ops::Range, sync::Arc};

use crate::{
    AxisPermutation, Decomposition, GeometryError, LocalGrid, LocalGridError, MpiTopology,
    PencilError, SpatialAxis,
    checked::checked_product,
    geometry::{local_ranges_for, shape_from_ranges},
};

/// Configuration values used to derive a new [`Pencil`] from an existing one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PencilConfig<const N: usize, const M: usize> {
    /// The global spatial shape in logical-axis order.
    pub global_shape: [usize; N],
    /// The distributed spatial axes in topology-axis order.
    pub decomposition: [usize; M],
    /// The spatial axes in row-major memory order.
    pub permutation: AxisPermutation<N>,
}

impl<const N: usize, const M: usize> From<&Pencil<N, M>> for PencilConfig<N, M> {
    fn from(pencil: &Pencil<N, M>) -> Self {
        Self {
            global_shape: pencil.global_shape,
            decomposition: std::array::from_fn(|axis| pencil.decomposition[axis].index()),
            permutation: pencil.permutation.clone(),
        }
    }
}

/// An immutable MPI distribution and row-major memory layout for an N-dimensional array.
#[derive(Debug)]
pub struct Pencil<const N: usize, const M: usize> {
    topology: Arc<MpiTopology<M>>,
    global_shape: [usize; N],
    decomposition: [SpatialAxis; M],
    permutation: AxisPermutation<N>,
    local_ranges: [Range<usize>; N],
    local_shape_logical: [usize; N],
    local_shape_memory: [usize; N],
    local_len: usize,
    global_len: usize,
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

        let global_len = checked_product(&global_shape).map_err(map_geometry_error)?;
        let decomposition =
            Decomposition::<N, M>::new(decomposition).map_err(PencilError::InvalidDecomposition)?;
        let local_ranges = ranges_for(
            &topology,
            global_shape,
            &decomposition,
            *topology.local_coords(),
        )?;
        let local_shape_logical = shape_from_ranges(&local_ranges);
        let local_shape_memory = permutation.permute(local_shape_logical);
        let local_len = checked_product(&local_shape_logical).map_err(map_geometry_error)?;

        Ok(Self {
            topology,
            global_shape,
            decomposition: *decomposition.axes(),
            permutation,
            local_ranges,
            local_shape_logical,
            local_shape_memory,
            local_len,
            global_len,
        }
        .into_shared())
    }

    /// Returns the shared Cartesian topology.
    pub fn topology(&self) -> &Arc<MpiTopology<M>> {
        &self.topology
    }

    /// Returns the global spatial shape in logical-axis order.
    pub fn global_shape(&self) -> &[usize; N] {
        &self.global_shape
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

    /// Returns the calling rank's local spatial shape in logical-axis order.
    pub fn local_shape_logical(&self) -> [usize; N] {
        self.local_shape_logical
    }

    /// Returns the calling rank's local spatial shape in memory-axis order.
    pub fn local_shape_memory(&self) -> [usize; N] {
        self.local_shape_memory
    }

    /// Returns the number of local spatial elements.
    pub fn local_len(&self) -> usize {
        self.local_len
    }

    /// Returns the number of global spatial elements.
    pub fn global_len(&self) -> usize {
        self.global_len
    }

    /// Borrows caller-provided global coordinate axes for this pencil's local grid.
    ///
    /// The input contains one complete coordinate slice per logical spatial
    /// axis. Every slice must have the corresponding [`Self::global_shape`]
    /// length. The returned grid keeps only local coordinate slices and no
    /// reference to this pencil or its MPI topology.
    pub fn local_grid<'a, C>(
        &self,
        coordinates: [&'a [C]; N],
    ) -> Result<LocalGrid<'a, C, N>, LocalGridError> {
        LocalGrid::from_pencil(self, coordinates)
    }

    /// Computes logical ranges for arbitrary valid Cartesian process coordinates.
    pub fn ranges_at(&self, process_coords: [usize; M]) -> Result<[Range<usize>; N], PencilError> {
        let decomposition = Decomposition::<N, M>::new(std::array::from_fn(|axis| {
            self.decomposition[axis].index()
        }))
        .expect("stored decomposition is valid");
        ranges_for(
            &self.topology,
            self.global_shape,
            &decomposition,
            process_coords,
        )
    }

    /// Derives a pencil with a different ordered decomposition.
    pub fn with_decomposition(&self, decomposition: [usize; M]) -> Result<Arc<Self>, PencilError> {
        self.reconfigured(PencilConfig {
            global_shape: self.global_shape,
            decomposition,
            permutation: self.permutation.clone(),
        })
    }

    /// Derives a pencil with a different memory-axis permutation.
    pub fn with_permutation(
        &self,
        permutation: AxisPermutation<N>,
    ) -> Result<Arc<Self>, PencilError> {
        self.reconfigured(PencilConfig {
            global_shape: self.global_shape,
            decomposition: std::array::from_fn(|axis| self.decomposition[axis].index()),
            permutation,
        })
    }

    /// Derives a pencil with a different global spatial shape.
    pub fn with_global_shape(&self, global_shape: [usize; N]) -> Result<Arc<Self>, PencilError> {
        self.reconfigured(PencilConfig {
            global_shape,
            decomposition: std::array::from_fn(|axis| self.decomposition[axis].index()),
            permutation: self.permutation.clone(),
        })
    }

    /// Derives a pencil by replacing all configurable layout values at once.
    pub fn reconfigured(&self, config: PencilConfig<N, M>) -> Result<Arc<Self>, PencilError> {
        Self::new_permuted(
            Arc::clone(&self.topology),
            config.global_shape,
            config.decomposition,
            config.permutation,
        )
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

    /// Returns true when distribution and memory-axis permutation both match.
    pub fn same_layout(&self, other: &Self) -> bool {
        self.same_distribution(other) && self.permutation == other.permutation
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
    local_ranges_for(
        global_shape,
        *topology.process_grid(),
        process_coords,
        *decomposition.axes(),
    )
    .map_err(map_geometry_error)
}

fn map_geometry_error(error: GeometryError) -> PencilError {
    match error {
        GeometryError::SizeOverflow | GeometryError::CountOverflow => PencilError::SizeOverflow,
        other => unreachable!("validated pencil geometry produced {other}"),
    }
}
