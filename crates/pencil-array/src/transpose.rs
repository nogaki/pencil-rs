use std::{
    any::type_name,
    mem::{align_of, size_of},
    ops::Range,
    sync::Arc,
    time::Duration,
};

use mpi::{
    Count,
    collective::{CommunicatorCollectives, SystemOperation},
    datatype::Equivalence,
    topology::Communicator,
};
use thiserror::Error;

use crate::{
    ArrayError, AxisPermutation, ExtraShape, ManyPencilArray, Pencil, PencilArrayView,
    PencilArrayViewMut,
    checked::{checked_product, local_storage_len},
    geometry::row_major_offset,
    preflight::{agree_header, expand_for_max},
};

pub(crate) use crate::preflight::agree_preflight;

const DESCRIPTOR_SCHEMA: u64 = 1;
// The operation word combines transport and operation; keep these values
// stable because they are part of the collective preflight protocol.
const OPERATION_ALLTOALLV_NEW: u64 = 1;
const OPERATION_ALLTOALLV_VIEWS: u64 = 2;
pub(crate) const OPERATION_ALLTOALLV_IN_PLACE: u64 = 3;
const OPERATION_POINT_TO_POINT_NEW: u64 = 4;
const OPERATION_POINT_TO_POINT_VIEWS: u64 = 5;
const OPERATION_POINT_TO_POINT_IN_PLACE: u64 = 6;
// Keep additive array operations outside the FFT protocol's operation range.
const OPERATION_POINT_TO_POINT_CALLBACK: u64 = 0x10001;
const OPERATION_POINT_TO_POINT_TIMED_VIEWS: u64 = 0x10002;
const OPERATION_POINT_TO_POINT_TIMED_IN_PLACE: u64 = 0x10003;
const OPERATION_ALLTOALLV_TIMED_VIEWS: u64 = 0x10004;
const OPERATION_ALLTOALLV_TIMED_IN_PLACE: u64 = 0x10005;
const OPERATION_POINT_TO_POINT_CALLBACK_IN_PLACE: u64 = 0x10006;
const INVALID_AXIS: u64 = u64::MAX;
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

    pub(crate) fn in_place_operation(self) -> u64 {
        match self {
            Self::AllToAllv => OPERATION_ALLTOALLV_IN_PLACE,
            Self::PointToPoint => OPERATION_POINT_TO_POINT_IN_PLACE,
        }
    }

    pub(crate) fn timed_views_operation(self) -> u64 {
        match self {
            Self::PointToPoint => OPERATION_POINT_TO_POINT_TIMED_VIEWS,
            Self::AllToAllv => OPERATION_ALLTOALLV_TIMED_VIEWS,
        }
    }

    pub(crate) fn timed_in_place_operation(self) -> u64 {
        match self {
            Self::PointToPoint => OPERATION_POINT_TO_POINT_TIMED_IN_PLACE,
            Self::AllToAllv => OPERATION_ALLTOALLV_TIMED_IN_PLACE,
        }
    }

    pub(crate) fn callback_operation(self) -> u64 {
        match self {
            Self::PointToPoint => OPERATION_POINT_TO_POINT_CALLBACK,
            Self::AllToAllv => OPERATION_ALLTOALLV_VIEWS,
        }
    }

    pub(crate) fn callback_in_place_operation(self) -> u64 {
        match self {
            Self::PointToPoint => OPERATION_POINT_TO_POINT_CALLBACK_IN_PLACE,
            Self::AllToAllv => OPERATION_ALLTOALLV_IN_PLACE,
        }
    }
}

/// Errors returned by checked Alltoallv and point-to-point transpose operations.
///
/// Both distributed transports use this canonical error type and share
/// collective validation, ordinary preflight guarantees, and
/// initialized-workspace contracts. MPI count limits depend on the transport.
/// Normal preflight errors are agreed by the source topology before
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

    /// A required MPI value does not fit `mpi::Count`, or checked length
    /// arithmetic overflows.
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

/// Error returned by the callback-based point-to-point overlap APIs.
#[derive(Debug, Error)]
pub enum OverlapError<E> {
    /// The local transpose failed before or during communication.
    #[error(transparent)]
    Transpose(#[from] TransposeError),
    /// The callback returned an error on this rank.
    #[error("the overlap callback failed")]
    Callback(E),
    /// Another rank's callback panicked. The origin resumes unwinding after agreement.
    #[error("the overlap callback panicked on another rank")]
    PeerPanicked,
    /// Another rank's callback returned an error.
    #[error("the overlap callback failed on another rank")]
    PeerCallbackFailed,
    /// A callback result could not be agreed by the full Cartesian communicator.
    #[error("the overlap callback result could not be agreed collectively")]
    CollectivePreconditionFailed,
}

/// Timing collected by a profiled distributed transpose.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TransposeTiming {
    /// Time spent packing local source data.
    pub pack: Duration,
    /// Time spent posting point-to-point receives.
    pub post_receive: Duration,
    /// Time spent waiting for point-to-point receives.
    pub receive_wait: Duration,
    /// Time spent in the Alltoallv collective.
    pub collective_wait: Duration,
    /// Time spent unpacking received data and copying the self region.
    pub unpack: Duration,
    /// Time spent waiting for point-to-point sends.
    pub send_wait: Duration,
    /// Total local wall-clock time, including preflight.
    pub total: Duration,
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

    /// Returns the initialized send-buffer length without exposing its storage.
    pub fn send_len(&self) -> usize {
        self.send_buffer.len()
    }

    /// Returns the initialized receive-buffer length without exposing its storage.
    pub fn receive_len(&self) -> usize {
        self.receive_buffer.len()
    }
}

/// The initialized send and receive lengths needed by either transport for one
/// transpose and one exact extra shape.
///
/// Requirements are computed locally and noncollectively. A caller must obtain
/// and satisfy the requirements independently on each rank. Lengths use checked
/// `usize` arithmetic; the selected plan checks the MPI values required by its
/// transport.
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
    mode: CommunicationMode,
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

// Displacement vectors are populated only for Alltoallv; point-to-point uses
// PeerMetadata's usize displacements.
#[derive(Debug)]
pub(crate) struct PreparedExchange {
    pub(crate) requirements: TransposeWorkspaceRequirements,
    pub(crate) send_counts: Vec<Count>,
    pub(crate) send_displacements: Vec<Count>,
    pub(crate) receive_counts: Vec<Count>,
    pub(crate) receive_displacements: Vec<Count>,
}

#[derive(Debug)]
pub(crate) struct PreparedInPlace {
    pub(crate) destination_index: usize,
    pub(crate) exchange: PreparedExchange,
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

        let changed_topology_axis = agree_preflight(
            communicator,
            validate_plan_layout(&source, &destination),
            TransposeError::CollectivePreconditionFailed,
        )?;
        let peers = agree_preflight(
            communicator,
            build_peer_metadata(&source, &destination, changed_topology_axis),
            TransposeError::CollectivePreconditionFailed,
        )?;

        Ok(Self {
            source,
            destination,
            changed_topology_axis,
            peers,
            descriptor,
            mode,
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

    pub(crate) fn self_peer(&self) -> &PeerMetadata<N> {
        let coordinate = self.source.topology().local_coords()[self.changed_topology_axis];
        self.peers
            .iter()
            .find(|peer| peer.peer_coordinate == coordinate)
            .expect("peer metadata contains the calling rank")
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
    /// The returned lengths include all peer segments. Each count must fit
    /// `mpi::Count`; Alltoallv additionally checks each displacement against
    /// that type. Segment ends and total lengths use checked `usize` arithmetic.
    pub(crate) fn workspace_requirements(
        &self,
        extra_shape: &ExtraShape,
    ) -> Result<TransposeWorkspaceRequirements, TransposeError> {
        workspace_requirements_for_peers(&self.peers, extra_shape.element_count(), self.mode)
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

        let prepared = prepare_exchange_counts(
            &self.peers,
            extra_shape.element_count(),
            self.source.topology().process_grid()[self.changed_topology_axis],
            self.mode,
        )?;
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
                self.source.local_ranges(),
                &self.source.local_shape_memory(),
                self.source.permutation(),
                self.source.local_len(),
                source_len,
                &peer.send_region,
                extra_shape.element_count(),
            )?;
            validate_region_offsets(
                self.destination.local_ranges(),
                &self.destination.local_shape_memory(),
                self.destination.permutation(),
                self.destination.local_len(),
                destination_len,
                &peer.receive_region,
                extra_shape.element_count(),
            )?;
        }
        Ok(prepared)
    }
}

pub(crate) fn prepare_in_place<T, const N: usize, const M: usize>(
    plan: &TransposePlanCore<N, M>,
    array: &ManyPencilArray<T, N, M>,
    workspace: &TransposeWorkspace<T>,
) -> Result<PreparedInPlace, TransposeError>
where
    T: Equivalence + Copy,
{
    let source_index = array.active_index().map_err(TransposeError::from)?;
    let active_source = array
        .pencils()
        .get(source_index)
        .ok_or(TransposeError::SourceLayoutMismatch)?;
    if !active_source.same_layout(plan.source.as_ref()) {
        return Err(TransposeError::SourceLayoutMismatch);
    }
    let destination_index = array
        .find_layout(plan.destination.as_ref())
        .ok_or(TransposeError::DestinationLayoutMismatch)?;
    let extra_count = array.extra_shape().element_count();
    let source_len = local_storage_len(active_source.local_len(), extra_count)
        .map_err(|error| TransposeError::Array(ArrayError::from(error)))?;
    let destination_pencil = array
        .pencils()
        .get(destination_index)
        .ok_or(TransposeError::DestinationLayoutMismatch)?;
    let destination_len = local_storage_len(destination_pencil.local_len(), extra_count)
        .map_err(|error| TransposeError::Array(ArrayError::from(error)))?;
    let exchange =
        plan.prepare_common(array.extra_shape(), source_len, destination_len, workspace)?;
    Ok(PreparedInPlace {
        destination_index,
        exchange,
    })
}

pub(crate) fn finish_in_place<T: Copy, const N: usize, const M: usize>(
    plan: &TransposePlanCore<N, M>,
    array: &mut ManyPencilArray<T, N, M>,
    destination_index: usize,
    exchange: &PreparedExchange,
    receive_buffer: &[T],
    extra_count: usize,
) {
    let mut guard = array
        .begin_in_place_write()
        .expect("collective in-place preflight validated the active state");
    unpack_destination(
        plan.peers(),
        plan.destination.as_ref(),
        &mut guard.storage_mut()[..exchange.requirements.receive_len],
        receive_buffer,
        extra_count,
    );
    guard
        .commit(destination_index)
        .expect("collective in-place preflight validated the destination layout");
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

fn workspace_requirements_for_peers<const N: usize>(
    peers: &[PeerMetadata<N>],
    batch: usize,
    mode: CommunicationMode,
) -> Result<TransposeWorkspaceRequirements, TransposeError> {
    let mut send_len = 0usize;
    let mut receive_len = 0usize;
    for peer in peers {
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
        checked_count(send_count, send_displacement, mode)?;
        checked_count(receive_count, receive_displacement, mode)?;
        send_len = send_len
            .checked_add(send_count)
            .ok_or(TransposeError::CountOverflow)?;
        receive_len = receive_len
            .checked_add(receive_count)
            .ok_or(TransposeError::CountOverflow)?;
    }
    Ok(TransposeWorkspaceRequirements {
        send_len,
        receive_len,
    })
}

fn prepare_exchange_counts<const N: usize>(
    peers: &[PeerMetadata<N>],
    batch: usize,
    coordinate_extent: usize,
    mode: CommunicationMode,
) -> Result<PreparedExchange, TransposeError> {
    let requirements = workspace_requirements_for_peers(peers, batch, mode)?;
    let peer_count = peers.len();
    let mut send_counts = zeroed_vec(peer_count)?;
    let mut send_displacements = if mode == CommunicationMode::AllToAllv {
        zeroed_vec(peer_count)?
    } else {
        Vec::new()
    };
    let mut receive_counts = zeroed_vec(peer_count)?;
    let mut receive_displacements = if mode == CommunicationMode::AllToAllv {
        zeroed_vec(peer_count)?
    } else {
        Vec::new()
    };

    for peer in peers {
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
            checked_partition_count(send_count, send_displacement, requirements.send_len, mode)?;
        if mode == CommunicationMode::AllToAllv {
            send_displacements[index] =
                Count::try_from(send_displacement).map_err(|_| TransposeError::CountOverflow)?;
        }
        receive_counts[index] = checked_partition_count(
            receive_count,
            receive_displacement,
            requirements.receive_len,
            mode,
        )?;
        if mode == CommunicationMode::AllToAllv {
            receive_displacements[index] =
                Count::try_from(receive_displacement).map_err(|_| TransposeError::CountOverflow)?;
        }
    }
    Ok(PreparedExchange {
        requirements,
        send_counts,
        send_displacements,
        receive_counts,
        receive_displacements,
    })
}

fn checked_count(
    count: usize,
    displacement: usize,
    mode: CommunicationMode,
) -> Result<Count, TransposeError> {
    let count_as_mpi = Count::try_from(count).map_err(|_| TransposeError::CountOverflow)?;
    if mode == CommunicationMode::AllToAllv {
        Count::try_from(displacement).map_err(|_| TransposeError::CountOverflow)?;
    }
    count
        .checked_add(displacement)
        .ok_or(TransposeError::CountOverflow)?;
    Ok(count_as_mpi)
}

fn checked_partition_count(
    count: usize,
    displacement: usize,
    total: usize,
    mode: CommunicationMode,
) -> Result<Count, TransposeError> {
    let count_as_mpi = checked_count(count, displacement, mode)?;
    let end = count
        .checked_add(displacement)
        .ok_or(TransposeError::CountOverflow)?;
    if end > total {
        return Err(TransposeError::PreparationFailed);
    }
    Ok(count_as_mpi)
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

// Bounds the region with metadata only; the elementwise scan survives as the
// #[cfg(test)] reference implementation below, which this must match exactly.
fn validate_region_offsets<const N: usize>(
    local_ranges: &[Range<usize>; N],
    local_shape_memory: &[usize; N],
    permutation: &AxisPermutation<N>,
    local_len: usize,
    storage_len: usize,
    region: &[Range<usize>; N],
    extra_count: usize,
) -> Result<(), TransposeError> {
    if extra_count == 0 {
        return Ok(());
    }
    if region_spatial_len(region)? == 0 {
        return Ok(());
    }

    let mut last_local = [0usize; N];
    for axis in 0..N {
        let local = &local_ranges[axis];
        let requested = &region[axis];
        if requested.start < local.start || requested.end > local.end {
            return Err(TransposeError::PreparationFailed);
        }
        last_local[axis] = requested
            .end
            .checked_sub(1)
            .and_then(|last| last.checked_sub(local.start))
            .ok_or(TransposeError::PreparationFailed)?;
    }

    let last_memory = permutation.permute(last_local);
    let max_spatial = row_major_offset(local_shape_memory, &last_memory)
        .map_err(|_| TransposeError::PreparationFailed)?;
    let max_offset = (extra_count - 1)
        .checked_mul(local_len)
        .and_then(|extra_offset| extra_offset.checked_add(max_spatial))
        .ok_or(TransposeError::PreparationFailed)?;
    if max_offset >= storage_len {
        return Err(TransposeError::PreparationFailed);
    }
    Ok(())
}

// The pack and unpack walks no longer map elements individually; only the
// test-only elementwise references below use this linear index mapping.
#[cfg(test)]
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

/// The local pencil metadata that the region walk needs. It is a plain slice
/// of the pencil's published layout so that the walk can be compared against an
/// elementwise reference without an MPI topology.
struct LocalLayout<'a, const N: usize> {
    local_ranges: &'a [Range<usize>; N],
    shape_memory: [usize; N],
    permutation: &'a AxisPermutation<N>,
    local_len: usize,
}

impl<'a, const N: usize> LocalLayout<'a, N> {
    fn new<const M: usize>(pencil: &'a Pencil<N, M>) -> Self {
        Self {
            local_ranges: pencil.local_ranges(),
            shape_memory: pencil.local_shape_memory(),
            permutation: pencil.permutation(),
            local_len: pencil.local_len(),
        }
    }
}

/// Precomputed memory walk for one peer region.
///
/// The payload order is the global row-major order of the region, so the walk
/// carries an odometer over the outer global axes and copies the last global
/// axis as a single run.
struct RegionWalk<const N: usize> {
    /// Memory offset of the region's start corner inside one extra block.
    start: usize,
    /// Memory stride of each global axis.
    stride: [usize; N],
    /// Region extent along each global axis.
    extent: [usize; N],
    /// Number of odometer positions over all axes but the last one.
    outer_len: usize,
    /// Number of elements along the last global axis.
    run_len: usize,
    /// Memory stride of the last global axis.
    run_stride: usize,
    /// Largest memory offset the walk reaches inside one extra block.
    max_offset: usize,
}

fn region_walk<const N: usize>(
    layout: &LocalLayout<'_, N>,
    region: &[Range<usize>; N],
) -> Result<RegionWalk<N>, TransposeError> {
    // The pencil publishes the local length as the checked product of the
    // memory shape, so these partial products cannot overflow.
    let mut stride = [0usize; N];
    let mut running = 1usize;
    for position in (0..N).rev() {
        let axis = layout.permutation.axes()[position].index();
        stride[axis] = running;
        running *= layout.shape_memory[position];
    }

    let mut extent = [0usize; N];
    let mut start = 0usize;
    let mut offset_span = 0usize;
    let mut outer_len = 1usize;
    let mut run_len = 0usize;
    let mut run_stride = 0usize;
    for axis in 0..N {
        let local = &layout.local_ranges[axis];
        let requested = &region[axis];
        if requested.start < local.start || requested.end > local.end {
            return Err(TransposeError::PreparationFailed);
        }
        let length = requested
            .end
            .checked_sub(requested.start)
            .ok_or(TransposeError::PreparationFailed)?;
        // Callers skip empty regions, so a zero extent here breaks preflight.
        if length == 0 {
            return Err(TransposeError::PreparationFailed);
        }
        extent[axis] = length;
        start = start
            .checked_add(
                (requested.start - local.start)
                    .checked_mul(stride[axis])
                    .ok_or(TransposeError::PreparationFailed)?,
            )
            .ok_or(TransposeError::PreparationFailed)?;
        offset_span = offset_span
            .checked_add(
                (length - 1)
                    .checked_mul(stride[axis])
                    .ok_or(TransposeError::PreparationFailed)?,
            )
            .ok_or(TransposeError::PreparationFailed)?;
        if axis + 1 < N {
            outer_len = outer_len
                .checked_mul(length)
                .ok_or(TransposeError::PreparationFailed)?;
        } else {
            run_len = length;
            run_stride = stride[axis];
        }
    }

    Ok(RegionWalk {
        start,
        stride,
        extent,
        outer_len,
        run_len,
        run_stride,
        max_offset: start
            .checked_add(offset_span)
            .ok_or(TransposeError::PreparationFailed)?,
    })
}

/// Advances the odometer over the outer global axes to the next payload
/// position. The payload order is global row-major, so only the storage offset
/// needs carrying.
///
/// A wrapping axis first gives back the advance it already accumulated
/// (`counter * stride`) rather than adding its stride and subtracting the whole
/// extent afterwards. The stride of a length-1 axis is not bounded by
/// `max_offset` (that axis contributes `0 * stride` to the span), so the
/// add-first form can overflow `usize` even though every visited offset is
/// inside the region.
fn advance_region<const N: usize>(
    walk: &RegionWalk<N>,
    counter: &mut [usize; N],
    offset: &mut usize,
) {
    for axis in (0..N.saturating_sub(1)).rev() {
        if counter[axis] + 1 < walk.extent[axis] {
            counter[axis] += 1;
            *offset += walk.stride[axis];
            return;
        }
        *offset -= counter[axis] * walk.stride[axis];
        counter[axis] = 0;
    }
}

/// Visits every region run in global row-major payload order, passing the
/// storage offset and payload offset of each run to `visit`.
fn walk_region<const N: usize>(walk: &RegionWalk<N>, mut visit: impl FnMut(usize, usize)) {
    if walk.run_len == 0 {
        return;
    }
    let mut counter = [0usize; N];
    let mut offset = walk.start;
    let mut payload_offset = 0usize;
    for position in 0..walk.outer_len {
        visit(offset, payload_offset);
        payload_offset += walk.run_len;
        if position + 1 == walk.outer_len {
            break;
        }
        advance_region(walk, &mut counter, &mut offset);
    }
}

/// Copies one region from pencil storage into payload order.
fn copy_region_to_payload<T: Copy, const N: usize>(
    walk: &RegionWalk<N>,
    storage: &[T],
    payload: &mut [T],
) {
    walk_region(walk, |offset, payload_offset| {
        let payload_end = payload_offset + walk.run_len;
        if walk.run_stride == 1 {
            payload[payload_offset..payload_end]
                .copy_from_slice(&storage[offset..offset + walk.run_len]);
        } else {
            for index in 0..walk.run_len {
                payload[payload_offset + index] = storage[offset + index * walk.run_stride];
            }
        }
    });
}

/// Copies one region from payload order into pencil storage.
fn copy_region_from_payload<T: Copy, const N: usize>(
    walk: &RegionWalk<N>,
    payload: &[T],
    storage: &mut [T],
) {
    walk_region(walk, |offset, payload_offset| {
        let payload_end = payload_offset + walk.run_len;
        if walk.run_stride == 1 {
            storage[offset..offset + walk.run_len]
                .copy_from_slice(&payload[payload_offset..payload_end]);
        } else {
            for index in 0..walk.run_len {
                storage[offset + index * walk.run_stride] = payload[payload_offset + index];
            }
        }
    });
}

fn pack_source_regions<T: Copy, const N: usize>(
    peers: &[PeerMetadata<N>],
    layout: &LocalLayout<'_, N>,
    source_storage: &[T],
    send_buffer: &mut [T],
    send_len: usize,
    extra_count: usize,
) {
    pack_source_regions_skipping(
        peers,
        None,
        layout,
        source_storage,
        send_buffer,
        send_len,
        extra_count,
    );
}

fn pack_source_regions_skipping<T: Copy, const N: usize>(
    peers: &[PeerMetadata<N>],
    skip_peer_coordinate: Option<usize>,
    layout: &LocalLayout<'_, N>,
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
        if skip_peer_coordinate == Some(peer.peer_coordinate) || peer.send_spatial_len == 0 {
            continue;
        }
        let walk = region_walk(layout, &peer.send_region)
            .expect("checked exchange preflight validated send region");
        debug_assert_eq!(walk.outer_len * walk.run_len, peer.send_spatial_len);
        let max_offset = (extra_count - 1)
            .checked_mul(layout.local_len)
            .and_then(|extra_offset| extra_offset.checked_add(walk.max_offset))
            .expect("checked exchange preflight validated source offset");
        debug_assert!(max_offset < layout.local_len.saturating_mul(extra_count));
        let segment_start = peer
            .send_spatial_displacement
            .checked_mul(extra_count)
            .expect("checked exchange preflight validated send displacement");
        for extra_linear in 0..extra_count {
            let source_extra_offset = extra_linear
                .checked_mul(layout.local_len)
                .expect("checked exchange preflight validated source offset");
            let payload_offset = segment_start
                .checked_add(
                    extra_linear
                        .checked_mul(peer.send_spatial_len)
                        .expect("checked exchange preflight validated send segment"),
                )
                .expect("checked exchange preflight validated send payload");
            let payload_end = payload_offset
                .checked_add(peer.send_spatial_len)
                .expect("checked exchange preflight validated send index");
            copy_region_to_payload(
                &walk,
                &source_storage[source_extra_offset..],
                &mut send_buffer[payload_offset..payload_end],
            );
        }
    }
}

pub(crate) fn pack_source<T: Copy, const N: usize, const M: usize>(
    peers: &[PeerMetadata<N>],
    source_pencil: &Pencil<N, M>,
    source_storage: &[T],
    send_buffer: &mut [T],
    send_len: usize,
    extra_count: usize,
) {
    pack_source_regions(
        peers,
        &LocalLayout::new(source_pencil),
        source_storage,
        send_buffer,
        send_len,
        extra_count,
    );
}

pub(crate) fn pack_source_except_self<T: Copy, const N: usize, const M: usize>(
    peers: &[PeerMetadata<N>],
    self_coordinate: usize,
    source_pencil: &Pencil<N, M>,
    source_storage: &[T],
    send_buffer: &mut [T],
    send_len: usize,
    extra_count: usize,
) {
    pack_source_regions_skipping(
        peers,
        Some(self_coordinate),
        &LocalLayout::new(source_pencil),
        source_storage,
        send_buffer,
        send_len,
        extra_count,
    );
}

fn unpack_destination_regions<T: Copy, const N: usize>(
    peers: &[PeerMetadata<N>],
    layout: &LocalLayout<'_, N>,
    destination_storage: &mut [T],
    receive_buffer: &[T],
    extra_count: usize,
) {
    unpack_destination_regions_skipping(
        peers,
        None,
        layout,
        destination_storage,
        receive_buffer,
        extra_count,
    );
}

fn unpack_destination_regions_skipping<T: Copy, const N: usize>(
    peers: &[PeerMetadata<N>],
    skip_peer_coordinate: Option<usize>,
    layout: &LocalLayout<'_, N>,
    destination_storage: &mut [T],
    receive_buffer: &[T],
    extra_count: usize,
) {
    if extra_count == 0 {
        return;
    }
    for peer in peers {
        if skip_peer_coordinate == Some(peer.peer_coordinate) || peer.receive_spatial_len == 0 {
            continue;
        }
        let walk = region_walk(layout, &peer.receive_region)
            .expect("checked exchange preflight validated receive region");
        debug_assert_eq!(walk.outer_len * walk.run_len, peer.receive_spatial_len);
        let max_offset = (extra_count - 1)
            .checked_mul(layout.local_len)
            .and_then(|extra_offset| extra_offset.checked_add(walk.max_offset))
            .expect("checked exchange preflight validated destination offset");
        debug_assert!(max_offset < layout.local_len.saturating_mul(extra_count));
        let segment_start = peer
            .receive_spatial_displacement
            .checked_mul(extra_count)
            .expect("checked exchange preflight validated receive displacement");
        for extra_linear in 0..extra_count {
            let destination_extra_offset = extra_linear
                .checked_mul(layout.local_len)
                .expect("checked exchange preflight validated destination offset");
            let payload_offset = segment_start
                .checked_add(
                    extra_linear
                        .checked_mul(peer.receive_spatial_len)
                        .expect("checked exchange preflight validated receive segment"),
                )
                .expect("checked exchange preflight validated receive payload");
            let payload_end = payload_offset
                .checked_add(peer.receive_spatial_len)
                .expect("checked exchange preflight validated receive index");
            copy_region_from_payload(
                &walk,
                &receive_buffer[payload_offset..payload_end],
                &mut destination_storage[destination_extra_offset..],
            );
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
    unpack_destination_regions(
        peers,
        &LocalLayout::new(destination_pencil),
        destination_storage,
        receive_buffer,
        extra_count,
    );
}

pub(crate) fn unpack_destination_except_self<T: Copy, const N: usize, const M: usize>(
    peers: &[PeerMetadata<N>],
    self_coordinate: usize,
    destination_pencil: &Pencil<N, M>,
    destination_storage: &mut [T],
    receive_buffer: &[T],
    extra_count: usize,
) {
    unpack_destination_regions_skipping(
        peers,
        Some(self_coordinate),
        &LocalLayout::new(destination_pencil),
        destination_storage,
        receive_buffer,
        extra_count,
    );
}

/// Copies one region between two pencil layouts in global row-major order.
fn copy_region_between<T: Copy, const N: usize>(
    source_walk: &RegionWalk<N>,
    destination_walk: &RegionWalk<N>,
    source_storage: &[T],
    destination_storage: &mut [T],
) {
    if source_walk.run_len == 0 {
        return;
    }
    debug_assert_eq!(source_walk.outer_len, destination_walk.outer_len);
    debug_assert_eq!(source_walk.run_len, destination_walk.run_len);
    let mut source_counter = [0usize; N];
    let mut destination_counter = [0usize; N];
    let mut source_offset = source_walk.start;
    let mut destination_offset = destination_walk.start;
    for position in 0..source_walk.outer_len {
        if source_walk.run_stride == 1 && destination_walk.run_stride == 1 {
            destination_storage[destination_offset..destination_offset + destination_walk.run_len]
                .copy_from_slice(
                    &source_storage[source_offset..source_offset + source_walk.run_len],
                );
        } else {
            for index in 0..source_walk.run_len {
                destination_storage[destination_offset + index * destination_walk.run_stride] =
                    source_storage[source_offset + index * source_walk.run_stride];
            }
        }
        if position + 1 == source_walk.outer_len {
            break;
        }
        advance_region(source_walk, &mut source_counter, &mut source_offset);
        advance_region(
            destination_walk,
            &mut destination_counter,
            &mut destination_offset,
        );
    }
}

fn copy_self_region_regions<T: Copy, const N: usize>(
    source_layout: &LocalLayout<'_, N>,
    source_region: &[Range<usize>; N],
    source_storage: &[T],
    destination_layout: &LocalLayout<'_, N>,
    destination_region: &[Range<usize>; N],
    destination_storage: &mut [T],
    extra_count: usize,
) {
    if extra_count == 0 {
        return;
    }
    let spatial_len = region_spatial_len(source_region)
        .expect("checked exchange preflight validated self send region");
    if spatial_len == 0 {
        return;
    }
    let source_walk = region_walk(source_layout, source_region)
        .expect("checked exchange preflight validated self send region");
    let destination_walk = region_walk(destination_layout, destination_region)
        .expect("checked exchange preflight validated self receive region");
    debug_assert_eq!(source_walk.outer_len * source_walk.run_len, spatial_len);
    debug_assert_eq!(
        destination_walk.outer_len * destination_walk.run_len,
        spatial_len
    );
    for extra_linear in 0..extra_count {
        let source_extra_offset = extra_linear
            .checked_mul(source_layout.local_len)
            .expect("checked exchange preflight validated source offset");
        let destination_extra_offset = extra_linear
            .checked_mul(destination_layout.local_len)
            .expect("checked exchange preflight validated destination offset");
        copy_region_between(
            &source_walk,
            &destination_walk,
            &source_storage[source_extra_offset..],
            &mut destination_storage[destination_extra_offset..],
        );
    }
}

pub(crate) fn copy_self_region<T: Copy, const N: usize, const M: usize>(
    peer: &PeerMetadata<N>,
    source_pencil: &Pencil<N, M>,
    source_storage: &[T],
    destination_pencil: &Pencil<N, M>,
    destination_storage: &mut [T],
    extra_count: usize,
) {
    copy_self_region_regions(
        &LocalLayout::new(source_pencil),
        &peer.send_region,
        source_storage,
        &LocalLayout::new(destination_pencil),
        &peer.receive_region,
        destination_storage,
        extra_count,
    );
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

/// Returns whether every rank reported `true` through one MIN all-reduce.
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
    let mut paired: Option<Vec<u64>> = None;
    let mut reduced: Option<Vec<u64>> = None;
    let mut length = 0usize;
    let mut ready = false;
    if let (Some(descriptor), Some(expected_len)) = (&local_descriptor, expected_len) {
        let doubled = expected_len.checked_mul(2);
        ready = descriptor.len() == expected_len
            && doubled.is_some_and(|len| checked_count_len(len).is_ok());
        if ready {
            let doubled = doubled.expect("ready descriptor length fits twice");
            paired = zeroed_vec(doubled).ok();
            reduced = zeroed_vec(doubled).ok();
            if paired.is_none() || reduced.is_none() {
                ready = false;
            } else {
                length = expected_len;
            }
        }
    }

    let all_ready = collective_valid(comm, ready);
    if !all_ready {
        return Err(TransposeError::PreparationFailed);
    }

    let descriptor = local_descriptor.expect("ready descriptors have a value");
    let mut paired = paired.expect("ready descriptors have a paired buffer");
    let mut reduced = reduced.expect("ready descriptors have a reduction buffer");
    expand_for_max(&descriptor, &mut paired);
    comm.all_reduce_into(
        paired.as_slice(),
        reduced.as_mut_slice(),
        SystemOperation::max(),
    );
    // ponytail: exact native word comparison replaces a full all-rank gather;
    // the second half is the complement of the global minimum.
    let (maximum, complemented_minimum) = reduced.split_at(length);
    if maximum
        .iter()
        .zip(complemented_minimum)
        .any(|(&maximum, &complemented_minimum)| maximum != !complemented_minimum)
    {
        return Err(TransposeError::CollectiveDescriptorMismatch);
    }
    Ok(descriptor)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Local pencil metadata that the offset validator needs.
    struct RegionMetadata<const N: usize> {
        local_ranges: [Range<usize>; N],
        local_shape_memory: [usize; N],
        permutation: AxisPermutation<N>,
        local_len: usize,
    }

    /// Element-by-element copy of the pre-metadata validator. It stays test-only
    /// so production preflight never scans spatial_len * extra_count elements.
    fn validate_region_offsets_reference<const N: usize>(
        local_ranges: &[Range<usize>; N],
        local_shape_memory: &[usize; N],
        permutation: &AxisPermutation<N>,
        local_len: usize,
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
                .checked_mul(local_len)
                .ok_or(TransposeError::PreparationFailed)?;
            for linear in 0..spatial_len {
                let global = region_index(region, linear)?;
                let mut local = [0usize; N];
                for axis in 0..N {
                    let range = &local_ranges[axis];
                    if global[axis] < range.start || global[axis] >= range.end {
                        return Err(TransposeError::PreparationFailed);
                    }
                    local[axis] = global[axis] - range.start;
                }
                let memory = permutation.permute(local);
                let spatial_offset = row_major_offset(local_shape_memory, &memory)
                    .map_err(|_| TransposeError::PreparationFailed)?;
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

    /// Runs both validators, asserts that they agree, and returns the
    /// production result.
    fn checked_offsets<const N: usize>(
        metadata: &RegionMetadata<N>,
        region: &[Range<usize>; N],
        extra_count: usize,
        storage_len: usize,
    ) -> Result<(), TransposeError> {
        let actual = validate_region_offsets(
            &metadata.local_ranges,
            &metadata.local_shape_memory,
            &metadata.permutation,
            metadata.local_len,
            storage_len,
            region,
            extra_count,
        );
        let reference = validate_region_offsets_reference(
            &metadata.local_ranges,
            &metadata.local_shape_memory,
            &metadata.permutation,
            metadata.local_len,
            storage_len,
            region,
            extra_count,
        );
        assert_eq!(
            actual,
            reference,
            "region {region:?}, extra_count {extra_count}, storage_len {storage_len}, permutation {:?}",
            metadata.permutation.axes(),
        );
        actual
    }

    fn all_permutations<const N: usize>() -> Vec<AxisPermutation<N>> {
        fn fill<const N: usize>(
            position: usize,
            current: &mut [usize; N],
            used: &mut [bool; N],
            out: &mut Vec<[usize; N]>,
        ) {
            if position == N {
                out.push(*current);
                return;
            }
            for axis in 0..N {
                if !used[axis] {
                    used[axis] = true;
                    current[position] = axis;
                    fill(position + 1, current, used, out);
                    used[axis] = false;
                }
            }
        }
        let mut axes = Vec::new();
        fill(0, &mut [0usize; N], &mut [false; N], &mut axes);
        axes.into_iter()
            .map(|axes| AxisPermutation::new(axes).expect("valid permutation"))
            .collect()
    }

    fn region_candidates(local: &Range<usize>) -> [Range<usize>; 7] {
        let (start, end) = (local.start, local.end);
        [
            start..end,
            start + 1..end,
            start..end - 1,
            end + 1..end + 1,
            start - 1..end,
            start..end + 1,
            end..start,
        ]
    }

    fn check_exhaustive_against_reference<const N: usize>() {
        let local_ranges: [Range<usize>; N] = std::array::from_fn(|axis| {
            let start = 3 + 2 * axis;
            start..start + 2 + axis
        });
        let logical_shape: [usize; N] = std::array::from_fn(|axis| 2 + axis);
        let local_len = logical_shape.iter().product::<usize>();
        let candidates: Vec<[Range<usize>; 7]> =
            local_ranges.iter().map(region_candidates).collect();

        for permutation in all_permutations::<N>() {
            let metadata = RegionMetadata {
                local_ranges: local_ranges.clone(),
                local_shape_memory: permutation.permute(logical_shape),
                permutation,
                local_len,
            };
            for code in 0..7usize.pow(N as u32) {
                let mut remainder = code;
                let region: [Range<usize>; N] = std::array::from_fn(|axis| {
                    let choice = remainder % 7;
                    remainder /= 7;
                    candidates[axis][choice].clone()
                });
                for extra_count in 0..=3 {
                    let max_storage = local_len * extra_count + 1;
                    for storage_len in 0..=max_storage {
                        let _ = checked_offsets(&metadata, &region, extra_count, storage_len);
                    }
                }
            }
        }
    }

    #[test]
    fn region_offsets_match_elementwise_reference() {
        check_exhaustive_against_reference::<1>();
        check_exhaustive_against_reference::<2>();
        check_exhaustive_against_reference::<3>();
    }

    #[test]
    fn region_offsets_preserve_zero_and_empty_cases() {
        let metadata = RegionMetadata {
            local_ranges: [5..8, 0..4],
            local_shape_memory: [4, 3],
            permutation: AxisPermutation::new([1, 0]).expect("valid permutation"),
            local_len: 12,
        };
        let cases = [
            ([6..8, 1..4], 2, 24, Ok(())),
            ([6..8, 1..4], 2, 23, Err(TransposeError::PreparationFailed)),
            ([4..7, 1..4], 2, 24, Err(TransposeError::PreparationFailed)),
            ([9..9, 0..4], 2, 0, Ok(())),
            ([Range { start: 8, end: 7 }, 0..4], 0, 0, Ok(())),
            (
                [9..9, Range { start: 4, end: 3 }],
                1,
                0,
                Err(TransposeError::PreparationFailed),
            ),
        ];
        for (region, extra_count, storage_len, expected) in cases {
            assert_eq!(
                checked_offsets(&metadata, &region, extra_count, storage_len),
                expected,
                "region {region:?}, extra_count {extra_count}, storage_len {storage_len}",
            );
        }
    }

    // clippy's single_range_in_vec_init misreads one-axis `[Range; 1]` arrays.
    #[allow(clippy::single_range_in_vec_init)]
    #[test]
    fn region_offsets_check_storage_and_usize_boundaries() {
        let maximum = usize::MAX;
        let half = maximum / 2 + 1;

        let tall = RegionMetadata {
            local_ranges: [0..half],
            local_shape_memory: [half],
            permutation: AxisPermutation::identity(),
            local_len: half,
        };
        assert_eq!(checked_offsets(&tall, &[0..1], 2, half + 1), Ok(()));
        assert_eq!(
            checked_offsets(&tall, &[0..1], 3, maximum),
            Err(TransposeError::PreparationFailed),
        );

        let huge_local = RegionMetadata {
            local_ranges: [0..maximum],
            local_shape_memory: [maximum],
            permutation: AxisPermutation::identity(),
            local_len: maximum,
        };
        assert_eq!(
            checked_offsets(&huge_local, &[1..2], 2, maximum),
            Err(TransposeError::PreparationFailed),
        );

        let single = RegionMetadata {
            local_ranges: [0..1],
            local_shape_memory: [1],
            permutation: AxisPermutation::identity(),
            local_len: 1,
        };
        assert_eq!(checked_offsets(&single, &[0..1], 1, 1), Ok(()));
        assert_eq!(
            checked_offsets(&single, &[0..1], 1, 0),
            Err(TransposeError::PreparationFailed),
        );

        let row_multiplication = RegionMetadata {
            local_ranges: [0..3, 0..maximum],
            local_shape_memory: [3, maximum],
            permutation: AxisPermutation::identity(),
            local_len: 1,
        };
        assert_eq!(
            checked_offsets(&row_multiplication, &[2..3, 0..1], 1, maximum),
            Err(TransposeError::PreparationFailed),
        );

        let row_addition = RegionMetadata {
            local_ranges: [0..2, 0..maximum],
            local_shape_memory: [2, maximum],
            permutation: AxisPermutation::identity(),
            local_len: 1,
        };
        assert_eq!(
            checked_offsets(&row_addition, &[1..2, 1..2], 1, maximum),
            Err(TransposeError::PreparationFailed),
        );

        let region_product = RegionMetadata {
            local_ranges: [0..maximum, 0..2],
            local_shape_memory: [maximum, 2],
            permutation: AxisPermutation::identity(),
            local_len: 1,
        };
        assert_eq!(
            checked_offsets(&region_product, &[0..maximum, 0..2], 1, maximum),
            Err(TransposeError::PreparationFailed),
        );

        let zero_extent = RegionMetadata {
            local_ranges: [0..2, 0..2],
            local_shape_memory: [0, 2],
            permutation: AxisPermutation::identity(),
            local_len: 1,
        };
        assert_eq!(
            checked_offsets(&zero_extent, &[0..1, 0..1], 1, maximum),
            Err(TransposeError::PreparationFailed),
        );

        let short_memory = RegionMetadata {
            local_ranges: [0..2, 0..2],
            local_shape_memory: [1, 2],
            permutation: AxisPermutation::identity(),
            local_len: 1,
        };
        assert_eq!(
            checked_offsets(&short_memory, &[1..2, 0..1], 1, maximum),
            Err(TransposeError::PreparationFailed),
        );

        let zero_product = RegionMetadata {
            local_ranges: [0..maximum, 0..2, 0..0],
            local_shape_memory: [maximum, 2, 0],
            permutation: AxisPermutation::identity(),
            local_len: 0,
        };
        assert_eq!(
            checked_offsets(&zero_product, &[0..maximum, 0..2, 0..0], 1, 0),
            Ok(()),
        );
    }

    #[test]
    fn region_offsets_large_region_uses_metadata() {
        let shape = [512usize, 512, 512];
        let local_len = 512usize.pow(3);
        let local_ranges: [Range<usize>; 3] = std::array::from_fn(|axis| 0..shape[axis]);
        let permutation = AxisPermutation::<3>::identity();
        let storage_len = 3 * local_len;
        let region = local_ranges.clone();
        assert_eq!(
            validate_region_offsets(
                &local_ranges,
                &shape,
                &permutation,
                local_len,
                storage_len,
                &region,
                3,
            ),
            Ok(()),
        );
    }

    /// Element-by-element copy of the pre-optimisation pack mapping. It stays
    /// test-only so the odometer walk can be checked against the mapping it
    /// replaced.
    fn reference_region_offsets<const N: usize>(
        local_ranges: &[Range<usize>; N],
        shape_memory: &[usize; N],
        permutation: &AxisPermutation<N>,
        region: &[Range<usize>; N],
    ) -> Vec<usize> {
        let spatial_len = region_spatial_len(region).expect("valid region");
        (0..spatial_len)
            .map(|linear| {
                let global = region_index(region, linear).expect("valid region index");
                let mut local = [0usize; N];
                for axis in 0..N {
                    let range = &local_ranges[axis];
                    assert!(range.contains(&global[axis]), "region inside the pencil");
                    local[axis] = global[axis] - range.start;
                }
                let memory = permutation.permute(local);
                row_major_offset(shape_memory, &memory).expect("valid memory index")
            })
            .collect()
    }

    #[test]
    fn pack_and_unpack_match_elementwise_reference() {
        let local_ranges: [Range<usize>; 3] = [1..4, 1..3, 2..5];
        let logical_shape: [usize; 3] =
            std::array::from_fn(|axis| local_ranges[axis].end - local_ranges[axis].start);
        let local_len = logical_shape.iter().product::<usize>();

        // Regions start at non-zero global offsets and include a length-1 last
        // axis and empty regions, which pack and unpack skip.
        let send_regions = [
            [1..4, 1..3, 2..5],
            [2..4, 1..3, 2..4],
            [1..2, 2..3, 3..4],
            [2..3, 1..2, 2..2],
        ];
        let receive_regions = [
            [1..3, 1..3, 2..5],
            [2..4, 2..3, 3..5],
            [1..4, 1..2, 3..4],
            [1..3, 1..3, 4..4],
        ];

        let mut peers = Vec::new();
        let mut send_displacement = 0usize;
        let mut receive_displacement = 0usize;
        for (index, (send_region, receive_region)) in
            send_regions.iter().zip(&receive_regions).enumerate()
        {
            let send_spatial_len = region_spatial_len(send_region).expect("valid region");
            let receive_spatial_len = region_spatial_len(receive_region).expect("valid region");
            peers.push(PeerMetadata {
                peer_rank: Count::try_from(index).expect("peer rank fits an MPI count"),
                peer_coordinate: index,
                send_region: send_region.clone(),
                receive_region: receive_region.clone(),
                send_spatial_len,
                receive_spatial_len,
                send_spatial_displacement: send_displacement,
                receive_spatial_displacement: receive_displacement,
            });
            send_displacement += send_spatial_len;
            receive_displacement += receive_spatial_len;
        }

        for permutation in all_permutations::<3>() {
            let shape_memory = permutation.permute(logical_shape);
            let layout = LocalLayout {
                local_ranges: &local_ranges,
                shape_memory,
                permutation: &permutation,
                local_len,
            };
            for extra_count in [1usize, 3] {
                let source: Vec<u32> =
                    (0..u32::try_from(local_len * extra_count).unwrap()).collect();
                let send_len = send_displacement * extra_count;
                let mut send_buffer = vec![0u32; send_len];
                let mut expected_send = vec![0u32; send_len];
                for peer in &peers {
                    if peer.send_spatial_len == 0 {
                        continue;
                    }
                    let offsets = reference_region_offsets(
                        &local_ranges,
                        &shape_memory,
                        &permutation,
                        &peer.send_region,
                    );
                    for extra_linear in 0..extra_count {
                        for (linear, &offset) in offsets.iter().enumerate() {
                            let payload = peer.send_spatial_displacement * extra_count
                                + extra_linear * peer.send_spatial_len
                                + linear;
                            expected_send[payload] = source[extra_linear * local_len + offset];
                        }
                    }
                }
                pack_source_regions(
                    &peers,
                    &layout,
                    &source,
                    &mut send_buffer,
                    send_len,
                    extra_count,
                );
                assert_eq!(
                    send_buffer,
                    expected_send,
                    "pack permutation {:?}, extra_count {extra_count}",
                    permutation.axes(),
                );

                let receive_len = receive_displacement * extra_count;
                let receive_buffer: Vec<u32> = (0..u32::try_from(receive_len).unwrap())
                    .map(|value| 1000 + value)
                    .collect();
                let mut storage = vec![0u32; local_len * extra_count];
                let mut expected_storage = vec![0u32; local_len * extra_count];
                for peer in &peers {
                    if peer.receive_spatial_len == 0 {
                        continue;
                    }
                    let offsets = reference_region_offsets(
                        &local_ranges,
                        &shape_memory,
                        &permutation,
                        &peer.receive_region,
                    );
                    for extra_linear in 0..extra_count {
                        for (linear, &offset) in offsets.iter().enumerate() {
                            let payload = peer.receive_spatial_displacement * extra_count
                                + extra_linear * peer.receive_spatial_len
                                + linear;
                            expected_storage[extra_linear * local_len + offset] =
                                receive_buffer[payload];
                        }
                    }
                }
                unpack_destination_regions(
                    &peers,
                    &layout,
                    &mut storage,
                    &receive_buffer,
                    extra_count,
                );
                assert_eq!(
                    storage,
                    expected_storage,
                    "unpack permutation {:?}, extra_count {extra_count}",
                    permutation.axes(),
                );
            }
        }
    }

    #[test]
    fn self_region_copy_matches_elementwise_reference() {
        let local_ranges: [Range<usize>; 3] = [1..4, 2..6, 0..3];
        let logical_shape: [usize; 3] =
            std::array::from_fn(|axis| local_ranges[axis].end - local_ranges[axis].start);
        let local_len = logical_shape.iter().product::<usize>();
        // The self peer's send and receive regions are the same global region.
        let region: [Range<usize>; 3] = [2..4, 2..5, 1..3];

        for source_permutation in all_permutations::<3>() {
            let source_shape_memory = source_permutation.permute(logical_shape);
            let source_layout = LocalLayout {
                local_ranges: &local_ranges,
                shape_memory: source_shape_memory,
                permutation: &source_permutation,
                local_len,
            };
            for destination_permutation in all_permutations::<3>() {
                let destination_shape_memory = destination_permutation.permute(logical_shape);
                let destination_layout = LocalLayout {
                    local_ranges: &local_ranges,
                    shape_memory: destination_shape_memory,
                    permutation: &destination_permutation,
                    local_len,
                };
                for extra_count in [1usize, 3] {
                    let source: Vec<u32> =
                        (0..u32::try_from(local_len * extra_count).unwrap()).collect();
                    let mut destination = vec![u32::MAX; local_len * extra_count];
                    let mut expected = vec![u32::MAX; local_len * extra_count];
                    let source_offsets = reference_region_offsets(
                        &local_ranges,
                        &source_shape_memory,
                        &source_permutation,
                        &region,
                    );
                    let destination_offsets = reference_region_offsets(
                        &local_ranges,
                        &destination_shape_memory,
                        &destination_permutation,
                        &region,
                    );
                    for extra_linear in 0..extra_count {
                        for (&source_offset, &destination_offset) in
                            source_offsets.iter().zip(&destination_offsets)
                        {
                            expected[extra_linear * local_len + destination_offset] =
                                source[extra_linear * local_len + source_offset];
                        }
                    }
                    copy_self_region_regions(
                        &source_layout,
                        &region,
                        &source,
                        &destination_layout,
                        &region,
                        &mut destination,
                        extra_count,
                    );
                    assert_eq!(
                        destination,
                        expected,
                        "self copy source {:?}, destination {:?}, extra_count {extra_count}",
                        source_permutation.axes(),
                        destination_permutation.axes(),
                    );
                }
            }
        }
    }

    /// A length-1 axis contributes `0 * stride` to the region's span, so its
    /// stride is not bounded by `max_offset`. Wrapping that axis must not add
    /// the stride to an offset that is already near `usize::MAX`; the old
    /// add-then-subtract order overflowed here. A zero-sized element type keeps
    /// the huge extent free.
    #[test]
    fn region_walk_wraps_length_one_axis_without_overflow() {
        let huge = 3 * (1usize << (usize::BITS - 2));
        let local_ranges: [Range<usize>; 3] = [0..huge, 0..1, 0..1];
        let shape_memory = [1, huge, 1];
        let permutation = AxisPermutation::new([1, 0, 2]).expect("valid permutation");
        let layout = LocalLayout {
            local_ranges: &local_ranges,
            shape_memory,
            permutation: &permutation,
            local_len: huge,
        };
        // The region stops one element short of the pencil's end, so
        // max_offset is `huge - 1` while the length-1 axis 1 has stride
        // `huge`. Advancing across that axis adds to `huge - 2`.
        let start = huge - 2;
        let region = [start..huge, 0..1, 0..1];
        let peer = PeerMetadata {
            peer_rank: Count::try_from(0).expect("peer rank fits an MPI count"),
            peer_coordinate: 0,
            send_region: region.clone(),
            receive_region: region,
            send_spatial_len: 2,
            receive_spatial_len: 2,
            send_spatial_displacement: 0,
            receive_spatial_displacement: 0,
        };
        let storage = vec![(); huge];
        let mut send_buffer = vec![(); 2];
        pack_source_regions(&[peer], &layout, &storage, &mut send_buffer, 2, 1);
    }

    #[test]
    fn descriptor_count_boundary_is_checked_without_allocation() {
        let maximum = usize::try_from(Count::MAX).expect("Count::MAX fits usize");
        assert!(checked_count_len(maximum).is_ok());
        assert!(checked_count_len(maximum + 1).is_err());
    }

    /// One-dimensional peers whose send and receive segments are identical.
    /// Each tuple is `(spatial_count, spatial_displacement)`.
    #[allow(clippy::single_range_in_vec_init)]
    fn count_test_peers(segments: &[(usize, usize)]) -> Vec<PeerMetadata<1>> {
        segments
            .iter()
            .enumerate()
            .map(|(index, &(count, displacement))| PeerMetadata {
                peer_rank: Count::try_from(index).unwrap(),
                peer_coordinate: index,
                send_region: [0..count],
                receive_region: [0..count],
                send_spatial_len: count,
                receive_spatial_len: count,
                send_spatial_displacement: displacement,
                receive_spatial_displacement: displacement,
            })
            .collect()
    }

    /// Keeps the boundary value on one direction and zeroes the other.
    #[allow(clippy::single_range_in_vec_init)]
    fn zero_direction(peers: &mut [PeerMetadata<1>], send_side: bool) {
        for peer in peers {
            if send_side {
                peer.receive_region = [0..0];
                peer.receive_spatial_len = 0;
                peer.receive_spatial_displacement = 0;
            } else {
                peer.send_region = [0..0];
                peer.send_spatial_len = 0;
                peer.send_spatial_displacement = 0;
            }
        }
    }

    fn count_values(values: &[usize]) -> Vec<Count> {
        values
            .iter()
            .map(|&value| Count::try_from(value).expect("Count conversion"))
            .collect()
    }

    fn both_modes() -> [CommunicationMode; 2] {
        [
            CommunicationMode::AllToAllv,
            CommunicationMode::PointToPoint,
        ]
    }

    #[test]
    fn count_limits_accept_issue_peer_sizes() {
        let count = usize::try_from(1_500_000_000_u64).unwrap();
        let total = usize::try_from(3_000_000_000_u64).unwrap();
        let peers = count_test_peers(&[(count, 0), (count, count)]);
        for mode in both_modes() {
            let requirements = workspace_requirements_for_peers(&peers, 1, mode).unwrap();
            assert_eq!(
                requirements,
                TransposeWorkspaceRequirements {
                    send_len: total,
                    receive_len: total,
                }
            );

            let prepared = prepare_exchange_counts(&peers, 1, peers.len(), mode).unwrap();
            assert_eq!(prepared.requirements, requirements);
            assert_eq!(prepared.send_counts, count_values(&[count, count]));
            assert_eq!(prepared.receive_counts, count_values(&[count, count]));
            if mode == CommunicationMode::AllToAllv {
                assert_eq!(prepared.send_displacements, count_values(&[0, count]));
                assert_eq!(prepared.receive_displacements, count_values(&[0, count]));
            } else {
                assert!(prepared.send_displacements.is_empty());
                assert!(prepared.receive_displacements.is_empty());
            }
        }
    }

    #[test]
    fn count_limits_keep_point_to_point_displacements_in_usize() {
        let count = usize::try_from(1_100_000_000_u64).unwrap();
        let total = 3 * count;
        let peers = count_test_peers(&[(count, 0), (count, count), (count, 2 * count)]);

        let requirements =
            workspace_requirements_for_peers(&peers, 1, CommunicationMode::PointToPoint).unwrap();
        assert_eq!(
            requirements,
            TransposeWorkspaceRequirements {
                send_len: total,
                receive_len: total,
            }
        );
        let prepared =
            prepare_exchange_counts(&peers, 1, peers.len(), CommunicationMode::PointToPoint)
                .unwrap();
        assert_eq!(prepared.requirements, requirements);
        assert_eq!(prepared.send_counts, count_values(&[count, count, count]));
        assert_eq!(
            prepared.receive_counts,
            count_values(&[count, count, count])
        );
        assert!(prepared.send_displacements.is_empty());
        assert!(prepared.receive_displacements.is_empty());

        assert_eq!(
            workspace_requirements_for_peers(&peers, 1, CommunicationMode::AllToAllv).unwrap_err(),
            TransposeError::CountOverflow,
        );
        assert_eq!(
            prepare_exchange_counts(&peers, 1, peers.len(), CommunicationMode::AllToAllv)
                .unwrap_err(),
            TransposeError::CountOverflow,
        );
    }

    #[test]
    fn count_limits_bound_peer_counts_and_arithmetic() {
        let maximum = usize::try_from(Count::MAX).unwrap();
        let above_maximum = maximum + 1;
        let half = usize::MAX / 2 + 1;

        for mode in both_modes() {
            let peers = count_test_peers(&[(maximum, 0)]);
            let requirements = workspace_requirements_for_peers(&peers, 1, mode).unwrap();
            assert_eq!(
                requirements,
                TransposeWorkspaceRequirements {
                    send_len: maximum,
                    receive_len: maximum,
                }
            );
            let prepared = prepare_exchange_counts(&peers, 1, peers.len(), mode).unwrap();
            assert_eq!(prepared.requirements, requirements);
            assert_eq!(prepared.send_counts, count_values(&[maximum]));
            assert_eq!(prepared.receive_counts, count_values(&[maximum]));
        }

        let overflow_cases: &[(&[(usize, usize)], usize)] = &[
            (&[(above_maximum, 0)], 1),
            (&[(maximum, 0)], 2),
            (&[(half, 0)], 2),
            (&[(0, half)], 2),
        ];
        for mode in both_modes() {
            for &(segments, batch) in overflow_cases {
                let peers = count_test_peers(segments);
                assert_eq!(
                    workspace_requirements_for_peers(&peers, batch, mode).unwrap_err(),
                    TransposeError::CountOverflow,
                );
                assert_eq!(
                    prepare_exchange_counts(&peers, batch, peers.len(), mode).unwrap_err(),
                    TransposeError::CountOverflow,
                );
            }

            let peers = count_test_peers(&[(1, usize::MAX)]);
            assert_eq!(
                workspace_requirements_for_peers(&peers, 1, mode).unwrap_err(),
                TransposeError::CountOverflow,
            );
            assert_eq!(
                prepare_exchange_counts(&peers, 1, peers.len(), mode).unwrap_err(),
                TransposeError::CountOverflow,
            );

            let peers = count_test_peers(&[(1, 1)]);
            assert_eq!(
                workspace_requirements_for_peers(&peers, 1, mode).unwrap(),
                TransposeWorkspaceRequirements {
                    send_len: 1,
                    receive_len: 1,
                }
            );
            assert_eq!(
                prepare_exchange_counts(&peers, 1, peers.len(), mode).unwrap_err(),
                TransposeError::PreparationFailed,
            );

            let peers = count_test_peers(&[(usize::MAX, usize::MAX)]);
            let requirements = workspace_requirements_for_peers(&peers, 0, mode).unwrap();
            assert_eq!(
                requirements,
                TransposeWorkspaceRequirements {
                    send_len: 0,
                    receive_len: 0,
                }
            );
            let prepared = prepare_exchange_counts(&peers, 0, peers.len(), mode).unwrap();
            assert_eq!(prepared.requirements, requirements);
            assert_eq!(prepared.send_counts, count_values(&[0]));
            assert_eq!(prepared.receive_counts, count_values(&[0]));
            if mode == CommunicationMode::AllToAllv {
                assert_eq!(prepared.send_displacements, count_values(&[0]));
                assert_eq!(prepared.receive_displacements, count_values(&[0]));
            } else {
                assert!(prepared.send_displacements.is_empty());
                assert!(prepared.receive_displacements.is_empty());
            }
        }
    }

    #[test]
    fn count_limits_check_send_and_receive_independently() {
        let maximum = usize::try_from(Count::MAX).unwrap();
        let above_maximum = maximum + 1;
        let half = usize::MAX / 2 + 1;

        for mode in both_modes() {
            for send_side in [true, false] {
                for (segments, batch) in [(vec![(half, 0)], 2usize), (vec![(above_maximum, 0)], 1)]
                {
                    let mut peers = count_test_peers(&segments);
                    zero_direction(&mut peers, send_side);
                    assert_eq!(
                        workspace_requirements_for_peers(&peers, batch, mode).unwrap_err(),
                        TransposeError::CountOverflow,
                    );
                    assert_eq!(
                        prepare_exchange_counts(&peers, batch, peers.len(), mode).unwrap_err(),
                        TransposeError::CountOverflow,
                    );
                }

                let mut peers = count_test_peers(&[(1, 1)]);
                zero_direction(&mut peers, send_side);
                let (send_len, receive_len) = if send_side { (1, 0) } else { (0, 1) };
                assert_eq!(
                    workspace_requirements_for_peers(&peers, 1, mode).unwrap(),
                    TransposeWorkspaceRequirements {
                        send_len,
                        receive_len,
                    }
                );
                assert_eq!(
                    prepare_exchange_counts(&peers, 1, peers.len(), mode).unwrap_err(),
                    TransposeError::PreparationFailed,
                );
            }
        }
    }

    #[test]
    fn count_limits_reject_invalid_peer_indices() {
        let expected = TransposeWorkspaceRequirements {
            send_len: 1,
            receive_len: 1,
        };
        for mode in both_modes() {
            let mut peers = count_test_peers(&[(1, 0)]);
            peers[0].peer_rank = -1;
            assert_eq!(
                workspace_requirements_for_peers(&peers, 1, mode).unwrap(),
                expected,
            );
            assert_eq!(
                prepare_exchange_counts(&peers, 1, peers.len(), mode).unwrap_err(),
                TransposeError::PreparationFailed,
            );

            let mut peers = count_test_peers(&[(1, 0)]);
            peers[0].peer_rank = 1;
            assert_eq!(
                workspace_requirements_for_peers(&peers, 1, mode).unwrap(),
                expected,
            );
            assert_eq!(
                prepare_exchange_counts(&peers, 1, peers.len(), mode).unwrap_err(),
                TransposeError::PreparationFailed,
            );

            let mut peers = count_test_peers(&[(1, 0)]);
            peers[0].peer_coordinate = 1;
            assert_eq!(
                workspace_requirements_for_peers(&peers, 1, mode).unwrap(),
                expected,
            );
            assert_eq!(
                prepare_exchange_counts(&peers, 1, 1, mode).unwrap_err(),
                TransposeError::PreparationFailed,
            );
        }
    }

    #[test]
    fn checked_count_and_partition_count_boundaries() {
        let maximum = usize::try_from(Count::MAX).unwrap();
        let above_maximum = maximum + 1;

        for mode in both_modes() {
            assert_eq!(checked_count(maximum, 0, mode), Ok(Count::MAX));
            assert_eq!(
                checked_count(above_maximum, 0, mode).unwrap_err(),
                TransposeError::CountOverflow,
            );
            let wide_displacement = checked_count(0, above_maximum, mode);
            if mode == CommunicationMode::AllToAllv {
                assert_eq!(
                    wide_displacement.unwrap_err(),
                    TransposeError::CountOverflow
                );
            } else {
                assert_eq!(wide_displacement, Ok(0));
            }
            assert_eq!(
                checked_count(1, usize::MAX, mode).unwrap_err(),
                TransposeError::CountOverflow,
            );

            assert_eq!(
                checked_partition_count(maximum, 0, maximum, mode),
                Ok(Count::MAX),
            );
            assert_eq!(
                checked_partition_count(1, maximum, above_maximum, mode),
                Ok(1),
            );
            assert_eq!(
                checked_partition_count(1, maximum, maximum, mode).unwrap_err(),
                TransposeError::PreparationFailed,
            );
            assert_eq!(checked_partition_count(0, maximum, maximum, mode), Ok(0));
            assert_eq!(
                checked_partition_count(0, maximum, maximum - 1, mode).unwrap_err(),
                TransposeError::PreparationFailed,
            );
            let wide_count = checked_partition_count(1, above_maximum, maximum + 2, mode);
            if mode == CommunicationMode::AllToAllv {
                assert_eq!(wide_count.unwrap_err(), TransposeError::CountOverflow);
            } else {
                assert_eq!(wide_count, Ok(1));
            }
            assert_eq!(
                checked_partition_count(1, usize::MAX, usize::MAX, mode).unwrap_err(),
                TransposeError::CountOverflow,
            );
            let wide_zero = checked_partition_count(0, usize::MAX, usize::MAX, mode);
            if mode == CommunicationMode::AllToAllv {
                assert_eq!(wide_zero.unwrap_err(), TransposeError::CountOverflow);
            } else {
                assert_eq!(wide_zero, Ok(0));
            }
            assert_eq!(checked_partition_count(0, 0, 0, mode), Ok(0));
        }
    }
}
