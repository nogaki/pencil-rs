use std::{
    any::type_name,
    mem::{align_of, size_of},
    ops::Range,
    sync::Arc,
};

use mpi::{
    Count,
    collective::{CommunicatorCollectives, SystemOperation},
    datatype::Equivalence,
    topology::Communicator,
};
use thiserror::Error;

use crate::{
    ArrayError, ExtraShape, Pencil, PencilArrayView, PencilArrayViewMut, checked::checked_product,
    geometry::row_major_offset,
};

const DESCRIPTOR_SCHEMA: u64 = 1;
// The operation word combines transport and operation; 1..=3 are the
// existing Alltoallv codes and 4..=5 are the point-to-point additions.
const OPERATION_ALLTOALLV_NEW: u64 = 1;
const OPERATION_ALLTOALLV_VIEWS: u64 = 2;
pub(crate) const OPERATION_ALLTOALLV_IN_PLACE: u64 = 3;
const OPERATION_POINT_TO_POINT_NEW: u64 = 4;
const OPERATION_POINT_TO_POINT_VIEWS: u64 = 5;
const INVALID_AXIS: u64 = u64::MAX;
const HEADER_WORDS: usize = 5;
pub(crate) const POINT_TO_POINT_RESERVED_TAG: mpi::Tag = 0x5054;

/// The private transport selection used by the checked transpose plans.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CommunicationMode {
    AllToAllv,
    PointToPoint,
}

impl CommunicationMode {
    pub(crate) fn new_operation(self) -> u64 {
        match self {
            Self::AllToAllv => OPERATION_ALLTOALLV_NEW,
            Self::PointToPoint => OPERATION_POINT_TO_POINT_NEW,
        }
    }

    pub(crate) fn views_operation(self) -> u64 {
        match self {
            Self::AllToAllv => OPERATION_ALLTOALLV_VIEWS,
            Self::PointToPoint => OPERATION_POINT_TO_POINT_VIEWS,
        }
    }
}

/// Errors returned by checked Alltoallv and point-to-point transpose operations.
///
/// Both distributed transports use this canonical error type, so their
/// collective validation, ordinary preflight, and workspace contracts cannot
/// diverge. Normal preflight errors are agreed by the source topology before
/// either transport starts payload communication; MPI failures, arbitrary
/// panics, and process loss are outside the recovery guarantee.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum TransposeError {
    /// The source and destination pencils do not share the same topology object.
    #[error("source and destination topologies are incompatible")]
    IncompatibleTopology,

    /// The source and destination pencils have different global shapes.
    #[error("source and destination global shapes differ")]
    IncompatibleGlobalShape,

    /// The decomposition is unchanged or differs at more than one topology position.
    #[error("source and destination decompositions do not differ at exactly one position")]
    UnsupportedDecompositionChange,

    /// The supplied source view or active array layout does not match the
    /// plan's source layout.
    #[error("source view does not match the transpose plan")]
    SourceLayoutMismatch,

    /// The supplied destination view or registered array layout does not match
    /// the plan's destination layout.
    #[error("destination view does not match the transpose plan")]
    DestinationLayoutMismatch,

    /// The source and destination extra shapes differ.
    #[error("source and destination extra shapes differ")]
    ExtraShapeMismatch,

    /// A workspace vector has fewer initialized elements than required.
    #[error(
        "workspace is too small: send requires {send_required} elements but has {send_len}, receive requires {receive_required} elements but has {receive_len}"
    )]
    WorkspaceTooSmall {
        /// The required initialized send length.
        send_required: usize,
        /// The supplied initialized send length.
        send_len: usize,
        /// The required initialized receive length.
        receive_required: usize,
        /// The supplied initialized receive length.
        receive_len: usize,
    },

    /// A count, displacement, total, or checked multiplication does not fit.
    #[error("MPI count or displacement overflowed")]
    CountOverflow,

    /// Checked metadata or offset preparation failed.
    #[error("transpose metadata preparation failed")]
    PreparationFailed,

    /// Ranks supplied different fixed or variable collective descriptors.
    #[error("collective transpose descriptors differ between ranks")]
    CollectiveDescriptorMismatch,

    /// At least one rank rejected a collective precondition.
    #[error("a collective transpose precondition failed on another rank")]
    CollectivePreconditionFailed,

    /// A local array validation failed.
    #[error(transparent)]
    Array(#[from] ArrayError),
}

/// Initialized send and receive storage shared by Alltoallv and point-to-point
/// distributed transposes.
///
/// A workspace is local state: constructing it never calls MPI and is not
/// collective. Execution checks the vectors' initialized `len`, ignores excess
/// capacity, and uses only the required prefixes without resizing or
/// reallocating them. Ordinary preflight errors leave its vectors unchanged;
/// MPI failures or arbitrary panics after communication starts are not covered
/// by that guarantee.
#[derive(Debug)]
pub struct TransposeWorkspace<T> {
    pub(crate) send_buffer: Vec<T>,
    pub(crate) receive_buffer: Vec<T>,
}

impl<T> TransposeWorkspace<T> {
    /// Creates a workspace from initialized vectors without calling MPI.
    ///
    /// This constructor is noncollective and performs no allocation of its own.
    /// Execution checks the vectors' initialized lengths, rather than their
    /// capacities, before communication. It uses only the required prefixes;
    /// neither vector is resized or reallocated.
    pub fn from_vecs(send: Vec<T>, receive: Vec<T>) -> Self {
        Self {
            send_buffer: send,
            receive_buffer: receive,
        }
    }
}

/// The initialized send and receive lengths needed by either transport for one
/// transpose and one exact extra shape.
///
/// Requirements are computed locally and noncollectively. A caller must obtain
/// and satisfy the requirements independently on each rank.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransposeWorkspaceRequirements {
    /// Number of initialized send elements required.
    pub send_len: usize,
    /// Number of initialized receive elements required.
    pub receive_len: usize,
}

/// Shared checked metadata and preflight state for distributed transpose
/// transports. It owns the source-to-destination layout metadata, while the
/// public transport wrappers own their communication protocol.
#[derive(Debug)]
pub(crate) struct TransposePlanCore<const N: usize, const M: usize> {
    source: Arc<Pencil<N, M>>,
    destination: Arc<Pencil<N, M>>,
    changed_topology_axis: usize,
    peers: Vec<PeerMetadata<N>>,
    descriptor: Vec<u64>,
}

#[derive(Debug)]
pub(crate) struct PeerMetadata<const N: usize> {
    pub(crate) peer_rank: Count,
    pub(crate) peer_coordinate: usize,
    pub(crate) send_region: [Range<usize>; N],
    pub(crate) receive_region: [Range<usize>; N],
    pub(crate) send_spatial_len: usize,
    pub(crate) receive_spatial_len: usize,
    pub(crate) send_spatial_displacement: usize,
    pub(crate) receive_spatial_displacement: usize,
}

#[derive(Debug)]
pub(crate) struct PreparedExchange {
    pub(crate) requirements: TransposeWorkspaceRequirements,
    pub(crate) send_counts: Vec<Count>,
    pub(crate) send_displacements: Vec<Count>,
    pub(crate) receive_counts: Vec<Count>,
    pub(crate) receive_displacements: Vec<Count>,
}

impl<const N: usize, const M: usize> TransposePlanCore<N, M> {
    /// Collectively validates and constructs a distributed transpose plan.
    ///
    /// All ranks must call this method in the same order on the same source
    /// communicator context. It performs scalar header agreement and exact
    /// descriptor comparison before entering any changed-axis communicator.
    /// It rejects an unchanged decomposition rather than silently selecting
    /// the local transpose API. Header and descriptor preparation failures
    /// return before payload communication; no rank-local early return may
    /// skip the required collective protocol.
    pub(crate) fn new(
        source: Arc<Pencil<N, M>>,
        destination: Arc<Pencil<N, M>>,
        mode: CommunicationMode,
    ) -> Result<Self, TransposeError> {
        let communicator = source.topology().cartesian();
        let expected_len = plan_descriptor_len::<N, M>();
        let operation = mode.new_operation();
        let descriptor_result = expected_len
            .ok()
            .and_then(|_| build_plan_descriptor(&source, &destination, operation).ok());
        let descriptor_len_word = expected_len.map_or(INVALID_AXIS, |len| len as u64);
        let header = [
            DESCRIPTOR_SCHEMA,
            operation,
            N as u64,
            M as u64,
            descriptor_len_word,
        ];
        if !agree_header(communicator, header) {
            return Err(TransposeError::CollectiveDescriptorMismatch);
        }
        let descriptor = collective_descriptor(communicator, descriptor_result, expected_len.ok())?;

        let local_validation = validate_plan_layout(&source, &destination);
        if !collective_valid(communicator, local_validation.is_ok()) {
            return Err(local_validation
                .err()
                .unwrap_or(TransposeError::CollectivePreconditionFailed));
        }
        let changed_topology_axis = local_validation.expect("collective validation succeeded");

        let pattern = build_peer_metadata(&source, &destination, changed_topology_axis);
        if !collective_valid(communicator, pattern.is_ok()) {
            return Err(pattern
                .err()
                .unwrap_or(TransposeError::CollectivePreconditionFailed));
        }
        let peers = pattern.expect("collective metadata validation succeeded");

        Ok(Self {
            source,
            destination,
            changed_topology_axis,
            peers,
            descriptor,
        })
    }

    pub(crate) fn source(&self) -> &Arc<Pencil<N, M>> {
        &self.source
    }

    pub(crate) fn destination(&self) -> &Arc<Pencil<N, M>> {
        &self.destination
    }

    pub(crate) fn peers(&self) -> &[PeerMetadata<N>] {
        &self.peers
    }

    pub(crate) fn changed_topology_axis(&self) -> usize {
        self.changed_topology_axis
    }

    pub(crate) fn request_counts(&self, prepared: &PreparedExchange) -> (usize, usize) {
        debug_assert_eq!(prepared.receive_counts.len(), self.peers.len());
        debug_assert_eq!(prepared.send_counts.len(), self.peers.len());
        let receive = prepared
            .receive_counts
            .iter()
            .filter(|&&count| count != 0)
            .count();
        let send = prepared
            .send_counts
            .iter()
            .filter(|&&count| count != 0)
            .count();
        (receive, send)
    }

    /// Returns the initialized workspace lengths required for `extra_shape`.
    ///
    /// This method is noncollective and does not allocate workspace storage.
    /// The returned lengths include all peer segments and are checked against
    /// `mpi::Count` for the counts and displacements used by execution.
    pub(crate) fn workspace_requirements(
        &self,
        extra_shape: &ExtraShape,
    ) -> Result<TransposeWorkspaceRequirements, TransposeError> {
        let batch = extra_shape.element_count();
        let mut send_len = 0usize;
        let mut receive_len = 0usize;
        for peer in &self.peers {
            let send_count = peer
                .send_spatial_len
                .checked_mul(batch)
                .ok_or(TransposeError::CountOverflow)?;
            let receive_count = peer
                .receive_spatial_len
                .checked_mul(batch)
                .ok_or(TransposeError::CountOverflow)?;
            let send_displacement = peer
                .send_spatial_displacement
                .checked_mul(batch)
                .ok_or(TransposeError::CountOverflow)?;
            let receive_displacement = peer
                .receive_spatial_displacement
                .checked_mul(batch)
                .ok_or(TransposeError::CountOverflow)?;
            checked_count(send_count, send_displacement)?;
            checked_count(receive_count, receive_displacement)?;
            send_len = send_len
                .checked_add(send_count)
                .ok_or(TransposeError::CountOverflow)?;
            receive_len = receive_len
                .checked_add(receive_count)
                .ok_or(TransposeError::CountOverflow)?;
        }
        Count::try_from(send_len).map_err(|_| TransposeError::CountOverflow)?;
        Count::try_from(receive_len).map_err(|_| TransposeError::CountOverflow)?;
        Ok(TransposeWorkspaceRequirements {
            send_len,
            receive_len,
        })
    }

    pub(crate) fn prepare_execution<T>(
        &self,
        source: &PencilArrayView<'_, T, N, M>,
        destination: &PencilArrayViewMut<'_, T, N, M>,
        workspace: &TransposeWorkspace<T>,
    ) -> Result<PreparedExchange, TransposeError>
    where
        T: Equivalence + Copy,
    {
        if !source.pencil().same_layout(self.source.as_ref()) {
            return Err(TransposeError::SourceLayoutMismatch);
        }
        if !destination.pencil().same_layout(self.destination.as_ref()) {
            return Err(TransposeError::DestinationLayoutMismatch);
        }
        if source.extra_shape() != destination.extra_shape() {
            return Err(TransposeError::ExtraShapeMismatch);
        }
        self.prepare_common(
            source.extra_shape(),
            source.len(),
            destination.len(),
            workspace,
        )
    }

    pub(crate) fn prepare_common<T>(
        &self,
        extra_shape: &ExtraShape,
        source_len: usize,
        destination_len: usize,
        workspace: &TransposeWorkspace<T>,
    ) -> Result<PreparedExchange, TransposeError> {
        let requirements = self.workspace_requirements(extra_shape)?;
        if workspace.send_buffer.len() < requirements.send_len
            || workspace.receive_buffer.len() < requirements.receive_len
        {
            return Err(TransposeError::WorkspaceTooSmall {
                send_required: requirements.send_len,
                send_len: workspace.send_buffer.len(),
                receive_required: requirements.receive_len,
                receive_len: workspace.receive_buffer.len(),
            });
        }

        let prepared = prepare_exchange_counts(self, extra_shape)?;
        if requirements.send_len != source_len || requirements.receive_len != destination_len {
            return Err(TransposeError::PreparationFailed);
        }
        for (counts, expected_len) in [
            (prepared.send_counts.as_slice(), source_len),
            (prepared.receive_counts.as_slice(), destination_len),
        ] {
            let total = counts.iter().try_fold(0usize, |total, &count| {
                let count =
                    usize::try_from(count).map_err(|_| TransposeError::PreparationFailed)?;
                total
                    .checked_add(count)
                    .ok_or(TransposeError::PreparationFailed)
            })?;
            if total != expected_len {
                return Err(TransposeError::PreparationFailed);
            }
        }

        for peer in &self.peers {
            let send_len = region_spatial_len(&peer.send_region)?;
            let receive_len = region_spatial_len(&peer.receive_region)?;
            if send_len != peer.send_spatial_len || receive_len != peer.receive_spatial_len {
                return Err(TransposeError::PreparationFailed);
            }
            validate_region_offsets(
                self.source.as_ref(),
                source_len,
                &peer.send_region,
                extra_shape.element_count(),
            )?;
            validate_region_offsets(
                self.destination.as_ref(),
                destination_len,
                &peer.receive_region,
                extra_shape.element_count(),
            )?;
        }
        Ok(prepared)
    }
}

fn validate_plan_layout<const N: usize, const M: usize>(
    source: &Pencil<N, M>,
    destination: &Pencil<N, M>,
) -> Result<usize, TransposeError> {
    if !source.same_topology(destination) {
        return Err(TransposeError::IncompatibleTopology);
    }
    if source.global_shape() != destination.global_shape() {
        return Err(TransposeError::IncompatibleGlobalShape);
    }
    let mut changed = None;
    for axis in 0..M {
        if source.decomposition()[axis] != destination.decomposition()[axis] {
            if changed.is_some() {
                return Err(TransposeError::UnsupportedDecompositionChange);
            }
            changed = Some(axis);
        }
    }
    changed.ok_or(TransposeError::UnsupportedDecompositionChange)
}

fn build_peer_metadata<const N: usize, const M: usize>(
    source: &Pencil<N, M>,
    destination: &Pencil<N, M>,
    changed_topology_axis: usize,
) -> Result<Vec<PeerMetadata<N>>, TransposeError> {
    let subcommunicator = source.topology().subcommunicator(changed_topology_axis);
    let peer_count =
        usize::try_from(subcommunicator.size()).map_err(|_| TransposeError::PreparationFailed)?;
    let mut peers = Vec::new();
    peers
        .try_reserve_exact(peer_count)
        .map_err(|_| TransposeError::PreparationFailed)?;
    let mut send_displacement = 0usize;
    let mut receive_displacement = 0usize;
    let local_coords = *source.topology().local_coords();
    let mut one_axis_coordinate = [0i32; 1];

    for peer_index in 0..peer_count {
        let peer_rank =
            Count::try_from(peer_index).map_err(|_| TransposeError::PreparationFailed)?;
        subcommunicator.rank_to_coordinates_into(peer_rank, &mut one_axis_coordinate);
        let peer_coordinate = usize::try_from(one_axis_coordinate[0])
            .map_err(|_| TransposeError::PreparationFailed)?;
        let mut peer_coords = local_coords;
        peer_coords[changed_topology_axis] = peer_coordinate;

        let source_peer_ranges = source
            .ranges_at(peer_coords)
            .map_err(|_| TransposeError::PreparationFailed)?;
        let destination_peer_ranges = destination
            .ranges_at(peer_coords)
            .map_err(|_| TransposeError::PreparationFailed)?;
        let send_region = intersect_ranges(source.local_ranges(), &destination_peer_ranges);
        let receive_region = intersect_ranges(&source_peer_ranges, destination.local_ranges());
        let send_spatial_len = region_spatial_len(&send_region)?;
        let receive_spatial_len = region_spatial_len(&receive_region)?;
        let peer = PeerMetadata {
            peer_rank,
            peer_coordinate,
            send_region,
            receive_region,
            send_spatial_len,
            receive_spatial_len,
            send_spatial_displacement: send_displacement,
            receive_spatial_displacement: receive_displacement,
        };
        peers.push(peer);
        send_displacement = send_displacement
            .checked_add(send_spatial_len)
            .ok_or(TransposeError::PreparationFailed)?;
        receive_displacement = receive_displacement
            .checked_add(receive_spatial_len)
            .ok_or(TransposeError::PreparationFailed)?;
    }
    Ok(peers)
}

fn intersect_ranges<const N: usize>(
    left: &[Range<usize>; N],
    right: &[Range<usize>; N],
) -> [Range<usize>; N] {
    std::array::from_fn(|axis| {
        let start = left[axis].start.max(right[axis].start);
        let end = left[axis].end.min(right[axis].end);
        if start < end {
            start..end
        } else {
            start..start
        }
    })
}

fn region_spatial_len<const N: usize>(region: &[Range<usize>; N]) -> Result<usize, TransposeError> {
    let mut shape = [0usize; N];
    for axis in 0..N {
        shape[axis] = region[axis]
            .end
            .checked_sub(region[axis].start)
            .ok_or(TransposeError::PreparationFailed)?;
    }
    checked_product(&shape).map_err(|_| TransposeError::PreparationFailed)
}

fn prepare_exchange_counts<const N: usize, const M: usize>(
    plan: &TransposePlanCore<N, M>,
    extra_shape: &ExtraShape,
) -> Result<PreparedExchange, TransposeError> {
    let requirements = plan.workspace_requirements(extra_shape)?;
    let peer_count = plan.peers.len();
    let mut send_counts = zeroed_vec(peer_count)?;
    let mut send_displacements = zeroed_vec(peer_count)?;
    let mut receive_counts = zeroed_vec(peer_count)?;
    let mut receive_displacements = zeroed_vec(peer_count)?;
    let batch = extra_shape.element_count();
    let coordinate_extent = plan.source.topology().process_grid()[plan.changed_topology_axis];

    for peer in &plan.peers {
        let index =
            usize::try_from(peer.peer_rank).map_err(|_| TransposeError::PreparationFailed)?;
        if index >= peer_count || peer.peer_coordinate >= coordinate_extent {
            return Err(TransposeError::PreparationFailed);
        }
        let send_count = peer
            .send_spatial_len
            .checked_mul(batch)
            .ok_or(TransposeError::CountOverflow)?;
        let receive_count = peer
            .receive_spatial_len
            .checked_mul(batch)
            .ok_or(TransposeError::CountOverflow)?;
        let send_displacement = peer
            .send_spatial_displacement
            .checked_mul(batch)
            .ok_or(TransposeError::CountOverflow)?;
        let receive_displacement = peer
            .receive_spatial_displacement
            .checked_mul(batch)
            .ok_or(TransposeError::CountOverflow)?;
        send_counts[index] =
            checked_partition_count(send_count, send_displacement, requirements.send_len)?;
        send_displacements[index] =
            Count::try_from(send_displacement).map_err(|_| TransposeError::CountOverflow)?;
        receive_counts[index] = checked_partition_count(
            receive_count,
            receive_displacement,
            requirements.receive_len,
        )?;
        receive_displacements[index] =
            Count::try_from(receive_displacement).map_err(|_| TransposeError::CountOverflow)?;
    }
    Ok(PreparedExchange {
        requirements,
        send_counts,
        send_displacements,
        receive_counts,
        receive_displacements,
    })
}

fn checked_partition_count(
    count: usize,
    displacement: usize,
    total: usize,
) -> Result<Count, TransposeError> {
    let count_as_mpi = Count::try_from(count).map_err(|_| TransposeError::CountOverflow)?;
    let displacement_as_mpi =
        Count::try_from(displacement).map_err(|_| TransposeError::CountOverflow)?;
    count_as_mpi
        .checked_add(displacement_as_mpi)
        .ok_or(TransposeError::CountOverflow)?;
    if count
        .checked_add(displacement)
        .ok_or(TransposeError::CountOverflow)?
        > total
    {
        return Err(TransposeError::PreparationFailed);
    }
    Ok(count_as_mpi)
}

fn checked_count(count: usize, displacement: usize) -> Result<(), TransposeError> {
    let count_as_mpi = Count::try_from(count).map_err(|_| TransposeError::CountOverflow)?;
    let displacement_as_mpi =
        Count::try_from(displacement).map_err(|_| TransposeError::CountOverflow)?;
    count_as_mpi
        .checked_add(displacement_as_mpi)
        .ok_or(TransposeError::CountOverflow)?;
    Ok(())
}

fn checked_count_len(len: usize) -> Result<(), TransposeError> {
    Count::try_from(len)
        .map(|_| ())
        .map_err(|_| TransposeError::CountOverflow)
}

fn zeroed_vec<T: Copy + Default>(len: usize) -> Result<Vec<T>, TransposeError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| TransposeError::PreparationFailed)?;
    values.resize(len, T::default());
    Ok(values)
}

fn validate_region_offsets<const N: usize, const M: usize>(
    pencil: &Pencil<N, M>,
    storage_len: usize,
    region: &[Range<usize>; N],
    extra_count: usize,
) -> Result<(), TransposeError> {
    if extra_count == 0 {
        return Ok(());
    }
    let spatial_len = region_spatial_len(region)?;
    if spatial_len == 0 {
        return Ok(());
    }
    for extra_linear in 0..extra_count {
        let extra_offset = extra_linear
            .checked_mul(pencil.local_len())
            .ok_or(TransposeError::PreparationFailed)?;
        for linear in 0..spatial_len {
            let global = region_index(region, linear)?;
            let spatial_offset = local_spatial_offset(pencil, global)?;
            let offset = extra_offset
                .checked_add(spatial_offset)
                .ok_or(TransposeError::PreparationFailed)?;
            if offset >= storage_len {
                return Err(TransposeError::PreparationFailed);
            }
        }
    }
    Ok(())
}

fn region_index<const N: usize>(
    region: &[Range<usize>; N],
    mut linear: usize,
) -> Result<[usize; N], TransposeError> {
    let mut shape = [0usize; N];
    for axis in 0..N {
        shape[axis] = region[axis]
            .end
            .checked_sub(region[axis].start)
            .ok_or(TransposeError::PreparationFailed)?;
    }
    let mut index = [0usize; N];
    for axis in (0..N).rev() {
        if shape[axis] == 0 {
            return Err(TransposeError::PreparationFailed);
        }
        index[axis] = region[axis].start + linear % shape[axis];
        linear /= shape[axis];
    }
    Ok(index)
}

fn local_spatial_offset<const N: usize, const M: usize>(
    pencil: &Pencil<N, M>,
    global: [usize; N],
) -> Result<usize, TransposeError> {
    let mut local = [0usize; N];
    for axis in 0..N {
        let range = &pencil.local_ranges()[axis];
        if global[axis] < range.start || global[axis] >= range.end {
            return Err(TransposeError::PreparationFailed);
        }
        local[axis] = global[axis] - range.start;
    }
    let memory = pencil.permutation().permute(local);
    row_major_offset(&pencil.local_shape_memory(), &memory)
        .map_err(|_| TransposeError::PreparationFailed)
}

// ponytail: pack/unpack stays the element-by-element reference mapping;
// add no index table or unmeasured optimization.
pub(crate) fn pack_source<T: Copy, const N: usize, const M: usize>(
    peers: &[PeerMetadata<N>],
    source_pencil: &Pencil<N, M>,
    source_storage: &[T],
    send_buffer: &mut [T],
    send_len: usize,
    extra_count: usize,
) {
    debug_assert_eq!(send_buffer.len(), send_len);
    if extra_count == 0 {
        return;
    }
    for peer in peers {
        if peer.send_spatial_len == 0 {
            continue;
        }
        let segment_start = peer
            .send_spatial_displacement
            .checked_mul(extra_count)
            .expect("checked exchange preflight validated send displacement");
        for extra_linear in 0..extra_count {
            let source_extra_offset = extra_linear
                .checked_mul(source_pencil.local_len())
                .expect("checked exchange preflight validated source offset");
            let payload_offset = segment_start
                .checked_add(
                    extra_linear
                        .checked_mul(peer.send_spatial_len)
                        .expect("checked exchange preflight validated send segment"),
                )
                .expect("checked exchange preflight validated send payload");
            for linear in 0..peer.send_spatial_len {
                let global = region_index(&peer.send_region, linear)
                    .expect("checked exchange preflight validated send region");
                let source_offset = source_extra_offset
                    .checked_add(
                        local_spatial_offset(source_pencil, global)
                            .expect("checked exchange preflight validated source mapping"),
                    )
                    .expect("checked exchange preflight validated source offset");
                let payload_index = payload_offset
                    .checked_add(linear)
                    .expect("checked exchange preflight validated send index");
                send_buffer[payload_index] = source_storage[source_offset];
            }
        }
    }
}

pub(crate) fn unpack_destination<T: Copy, const N: usize, const M: usize>(
    peers: &[PeerMetadata<N>],
    destination_pencil: &Pencil<N, M>,
    destination_storage: &mut [T],
    receive_buffer: &[T],
    extra_count: usize,
) {
    if extra_count == 0 {
        return;
    }
    for peer in peers {
        if peer.receive_spatial_len == 0 {
            continue;
        }
        let segment_start = peer
            .receive_spatial_displacement
            .checked_mul(extra_count)
            .expect("checked exchange preflight validated receive displacement");
        for extra_linear in 0..extra_count {
            let destination_extra_offset = extra_linear
                .checked_mul(destination_pencil.local_len())
                .expect("checked exchange preflight validated destination offset");
            let payload_offset = segment_start
                .checked_add(
                    extra_linear
                        .checked_mul(peer.receive_spatial_len)
                        .expect("checked exchange preflight validated receive segment"),
                )
                .expect("checked exchange preflight validated receive payload");
            for linear in 0..peer.receive_spatial_len {
                let global = region_index(&peer.receive_region, linear)
                    .expect("checked exchange preflight validated receive region");
                let destination_offset = destination_extra_offset
                    .checked_add(
                        local_spatial_offset(destination_pencil, global)
                            .expect("checked exchange preflight validated destination mapping"),
                    )
                    .expect("checked exchange preflight validated destination offset");
                let payload_index = payload_offset
                    .checked_add(linear)
                    .expect("checked exchange preflight validated receive index");
                destination_storage[destination_offset] = receive_buffer[payload_index];
            }
        }
    }
}

fn plan_descriptor_len<const N: usize, const M: usize>() -> Result<usize, ()> {
    let topology_words = M.checked_mul(4).ok_or(())?;
    let spatial_words = N.checked_mul(4).ok_or(())?;
    5usize
        .checked_add(topology_words)
        .and_then(|value| value.checked_add(spatial_words))
        .ok_or(())
}

fn build_plan_descriptor<const N: usize, const M: usize>(
    source: &Pencil<N, M>,
    destination: &Pencil<N, M>,
    operation: u64,
) -> Result<Vec<u64>, ()> {
    let len = plan_descriptor_len::<N, M>()?;
    let mut descriptor = Vec::new();
    descriptor.try_reserve_exact(len).map_err(|_| ())?;
    descriptor.push(DESCRIPTOR_SCHEMA);
    descriptor.push(operation);
    descriptor.push(N as u64);
    descriptor.push(M as u64);
    descriptor.push(changed_axis_sentinel(source, destination));
    append_usizes(&mut descriptor, source.topology().process_grid());
    append_usizes(&mut descriptor, destination.topology().process_grid());
    append_pencil_values(&mut descriptor, source);
    append_pencil_values(&mut descriptor, destination);
    if descriptor.len() != len {
        return Err(());
    }
    Ok(descriptor)
}

fn append_pencil_values<const N: usize, const M: usize>(
    descriptor: &mut Vec<u64>,
    pencil: &Pencil<N, M>,
) {
    append_usizes(descriptor, pencil.global_shape());
    descriptor.extend(
        pencil
            .decomposition()
            .iter()
            .map(|axis| axis.index() as u64),
    );
    descriptor.extend(
        pencil
            .permutation()
            .axes()
            .iter()
            .map(|axis| axis.index() as u64),
    );
}

fn changed_axis_sentinel<const N: usize, const M: usize>(
    source: &Pencil<N, M>,
    destination: &Pencil<N, M>,
) -> u64 {
    let mut changed = None;
    for axis in 0..M {
        if source.decomposition()[axis] != destination.decomposition()[axis] {
            if changed.is_some() {
                return INVALID_AXIS;
            }
            changed = Some(axis as u64);
        }
    }
    changed.unwrap_or(INVALID_AXIS)
}

fn append_usizes(descriptor: &mut Vec<u64>, values: &[usize]) {
    descriptor.extend(values.iter().copied().map(|value| value as u64));
}

pub(crate) fn agree_execute_descriptor<C, T, const N: usize, const M: usize>(
    plan: &TransposePlanCore<N, M>,
    communicator: &C,
    source_extra_shape: &ExtraShape,
    destination_extra_shape: &ExtraShape,
    operation: u64,
) -> Result<(), TransposeError>
where
    C: CommunicatorCollectives,
    T: Equivalence,
{
    let expected_len =
        execute_descriptor_len::<T, N, M>(plan, source_extra_shape, destination_extra_shape);
    let descriptor_result = expected_len.ok().and_then(|_| {
        build_execute_descriptor::<T, N, M>(
            plan,
            source_extra_shape,
            destination_extra_shape,
            operation,
        )
        .ok()
    });
    let descriptor_len_word = expected_len.map_or(INVALID_AXIS, |len| len as u64);
    let header = [
        DESCRIPTOR_SCHEMA,
        operation,
        N as u64,
        M as u64,
        descriptor_len_word,
    ];
    if !agree_header(communicator, header) {
        return Err(TransposeError::CollectiveDescriptorMismatch);
    }
    collective_descriptor(communicator, descriptor_result, expected_len.ok())?;
    Ok(())
}

fn execute_descriptor_len<T: Equivalence, const N: usize, const M: usize>(
    plan: &TransposePlanCore<N, M>,
    source_extra_shape: &ExtraShape,
    destination_extra_shape: &ExtraShape,
) -> Result<usize, ()> {
    let source_name_len = type_name::<T>().len();
    let mpi_name_len = type_name::<<T as Equivalence>::Out>().len();
    let mut length = 4usize.checked_add(plan.descriptor.len()).ok_or(())?;
    length = length
        .checked_add(
            1usize
                .checked_add(source_extra_shape.dimensions().len())
                .ok_or(())?,
        )
        .ok_or(())?;
    length = length
        .checked_add(
            1usize
                .checked_add(destination_extra_shape.dimensions().len())
                .ok_or(())?,
        )
        .ok_or(())?;
    length = length.checked_add(2).ok_or(())?;
    length = length
        .checked_add(1usize.checked_add(source_name_len).ok_or(())?)
        .ok_or(())?;
    length = length
        .checked_add(1usize.checked_add(mpi_name_len).ok_or(())?)
        .ok_or(())?;
    Ok(length)
}

fn build_execute_descriptor<T, const N: usize, const M: usize>(
    plan: &TransposePlanCore<N, M>,
    source_extra_shape: &ExtraShape,
    destination_extra_shape: &ExtraShape,
    operation: u64,
) -> Result<Vec<u64>, ()>
where
    T: Equivalence,
{
    let len = execute_descriptor_len::<T, N, M>(plan, source_extra_shape, destination_extra_shape)?;
    let mut descriptor = Vec::new();
    descriptor.try_reserve_exact(len).map_err(|_| ())?;
    descriptor.push(DESCRIPTOR_SCHEMA);
    descriptor.push(operation);
    descriptor.push(N as u64);
    descriptor.push(M as u64);
    descriptor.extend(plan.descriptor.iter().copied());
    append_shape(&mut descriptor, source_extra_shape);
    append_shape(&mut descriptor, destination_extra_shape);
    descriptor.push(size_of::<T>() as u64);
    descriptor.push(align_of::<T>() as u64);
    append_type_name(&mut descriptor, type_name::<T>());
    append_type_name(&mut descriptor, type_name::<<T as Equivalence>::Out>());
    if descriptor.len() != len {
        return Err(());
    }
    Ok(descriptor)
}

fn append_shape(descriptor: &mut Vec<u64>, shape: &ExtraShape) {
    descriptor.push(shape.dimensions().len() as u64);
    append_usizes(descriptor, shape.dimensions());
}

fn append_type_name(descriptor: &mut Vec<u64>, name: &str) {
    descriptor.push(name.len() as u64);
    descriptor.extend(name.as_bytes().iter().copied().map(u64::from));
}

fn agree_header<C: CommunicatorCollectives>(comm: &C, header: [u64; HEADER_WORDS]) -> bool {
    let mut minimum = [0u64; HEADER_WORDS];
    let mut maximum = [0u64; HEADER_WORDS];
    comm.all_reduce_into(&header[..], &mut minimum[..], SystemOperation::min());
    comm.all_reduce_into(&header[..], &mut maximum[..], SystemOperation::max());
    minimum == maximum
}

pub(crate) fn collective_valid<C: CommunicatorCollectives>(comm: &C, local_valid: bool) -> bool {
    let value = i32::from(local_valid);
    let mut result = 0i32;
    comm.all_reduce_into(&value, &mut result, SystemOperation::min());
    result != 0
}

fn collective_descriptor<C: CommunicatorCollectives>(
    comm: &C,
    local_descriptor: Option<Vec<u64>>,
    expected_len: Option<usize>,
) -> Result<Vec<u64>, TransposeError> {
    let mut minimum: Option<Vec<u64>> = None;
    let mut maximum: Option<Vec<u64>> = None;
    let mut ready = false;
    if let (Some(descriptor), Some(expected_len)) = (&local_descriptor, expected_len) {
        ready = descriptor.len() == expected_len && checked_count_len(expected_len).is_ok();
        if ready {
            minimum = zeroed_vec(expected_len).ok();
            maximum = zeroed_vec(expected_len).ok();
            if minimum.is_none() || maximum.is_none() {
                ready = false;
            }
        }
    }

    let all_ready = collective_valid(comm, ready);
    if !all_ready {
        return Err(TransposeError::PreparationFailed);
    }

    let descriptor = local_descriptor.expect("ready descriptors have a value");
    let mut minimum = minimum.expect("ready descriptors have a minimum buffer");
    let mut maximum = maximum.expect("ready descriptors have a maximum buffer");
    comm.all_reduce_into(
        descriptor.as_slice(),
        minimum.as_mut_slice(),
        SystemOperation::min(),
    );
    comm.all_reduce_into(
        descriptor.as_slice(),
        maximum.as_mut_slice(),
        SystemOperation::max(),
    );
    // ponytail: exact native word comparison replaces a full all-rank gather;
    // keep only the two descriptor-sized reduction buffers.
    if minimum != maximum {
        return Err(TransposeError::CollectiveDescriptorMismatch);
    }
    Ok(descriptor)
}

#[cfg(test)]
mod tests {
    use super::checked_count_len;
    use mpi::Count;

    #[test]
    fn descriptor_count_boundary_is_checked_without_allocation() {
        let maximum = usize::try_from(Count::MAX).expect("Count::MAX fits usize");
        assert!(checked_count_len(maximum).is_ok());
        assert!(checked_count_len(maximum + 1).is_err());
    }
}
