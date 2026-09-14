use std::{
    any::type_name,
    mem::{align_of, size_of},
    ops::Range,
    sync::Arc,
};

use mpi::{
    Count,
    collective::{CommunicatorCollectives, SystemOperation},
    datatype::{Equivalence, Partition, PartitionMut},
    topology::Communicator,
};
use thiserror::Error;

use crate::{
    ArrayError, ExtraShape, Pencil, PencilArrayView, PencilArrayViewMut, checked::checked_product,
    geometry::row_major_offset,
};

const DESCRIPTOR_SCHEMA: u64 = 1;
const OPERATION_NEW: u64 = 1;
const OPERATION_EXECUTE: u64 = 2;
const INVALID_AXIS: u64 = u64::MAX;
const HEADER_WORDS: usize = 5;

/// Errors returned by an out-of-place `MPI_Alltoallv` distributed transpose.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum AllToAllvTransposeError {
    /// The source and destination pencils do not share the same topology object.
    #[error("source and destination topologies are incompatible")]
    IncompatibleTopology,

    /// The source and destination pencils have different global shapes.
    #[error("source and destination global shapes differ")]
    IncompatibleGlobalShape,

    /// The decomposition is unchanged or differs at more than one topology position.
    #[error("source and destination decompositions do not differ at exactly one position")]
    UnsupportedDecompositionChange,

    /// The supplied source view does not match the plan's source layout.
    #[error("source view does not match the transpose plan")]
    SourceLayoutMismatch,

    /// The supplied destination view does not match the plan's destination layout.
    #[error("destination view does not match the transpose plan")]
    DestinationLayoutMismatch,

    /// The source and destination views have different extra shapes.
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

/// Initialized send and receive storage for an Alltoallv transpose.
///
/// A workspace is local state: constructing it never calls MPI. Execution
/// checks the vectors' initialized `len`, ignores excess capacity, and uses
/// only the required prefixes without resizing or reallocating them.
#[derive(Debug)]
pub struct AllToAllvTransposeWorkspace<T> {
    send_buffer: Vec<T>,
    receive_buffer: Vec<T>,
}

impl<T> AllToAllvTransposeWorkspace<T> {
    /// Creates a workspace from initialized vectors without calling MPI.
    ///
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

/// The initialized send and receive lengths needed by one transpose and one
/// exact extra shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AllToAllvTransposeWorkspaceRequirements {
    /// Number of initialized send elements required.
    pub send_len: usize,
    /// Number of initialized receive elements required.
    pub receive_len: usize,
}

/// A checked, out-of-place distributed transpose using one `MPI_Alltoallv`.
///
/// [`Self::new`] is collective on the source topology's Cartesian
/// communicator. Every rank must call it in the same order with the same
/// source communicator context and matching source-to-destination layouts.
/// [`Self::execute_views`] has the same collective contract and additionally
/// requires every rank to use the same `T` with a correct
/// [`Equivalence`](mpi::datatype::Equivalence) implementation. These are API
/// contracts; descriptor checks can catch common mismatches but cannot prove
/// type identity, the correctness of an unsafe `Equivalence` implementation,
/// or communicator/collective-order correctness.
///
/// Counts and displacements are checked against `mpi::Count` before any MPI
/// payload call. Ordinary descriptor or preflight errors leave the source
/// unchanged and do not write the destination. MPI failures, arbitrary panics,
/// and process loss do not guarantee that a `Result` is returned or that the
/// destination remains unchanged.
#[derive(Debug)]
pub struct AllToAllvTransposePlan<const N: usize, const M: usize> {
    source: Arc<Pencil<N, M>>,
    destination: Arc<Pencil<N, M>>,
    changed_topology_axis: usize,
    peers: Vec<PeerMetadata<N>>,
    descriptor: Vec<u64>,
}

#[derive(Debug)]
struct PeerMetadata<const N: usize> {
    peer_rank: Count,
    peer_coordinate: usize,
    send_region: [Range<usize>; N],
    receive_region: [Range<usize>; N],
    send_spatial_len: usize,
    receive_spatial_len: usize,
    send_spatial_displacement: usize,
    receive_spatial_displacement: usize,
}

#[derive(Debug)]
struct PreparedExchange {
    requirements: AllToAllvTransposeWorkspaceRequirements,
    send_counts: Vec<Count>,
    send_displacements: Vec<Count>,
    receive_counts: Vec<Count>,
    receive_displacements: Vec<Count>,
}

impl<const N: usize, const M: usize> AllToAllvTransposePlan<N, M> {
    /// Collectively validates and constructs a distributed transpose plan.
    ///
    /// All ranks must call this method in the same order on the same source
    /// communicator context. It performs scalar header agreement and exact
    /// descriptor comparison before entering any changed-axis communicator.
    /// It rejects an unchanged decomposition rather than silently selecting
    /// the local transpose API. Header and descriptor preparation failures
    /// return before payload communication; no rank-local early return may
    /// skip the required collective protocol.
    pub fn new(
        source: Arc<Pencil<N, M>>,
        destination: Arc<Pencil<N, M>>,
    ) -> Result<Self, AllToAllvTransposeError> {
        let communicator = source.topology().cartesian();
        let expected_len = plan_descriptor_len::<N, M>();
        let descriptor_result = expected_len
            .ok()
            .and_then(|_| build_plan_descriptor(&source, &destination).ok());
        let descriptor_len_word = expected_len.map_or(INVALID_AXIS, |len| len as u64);
        let header = [
            DESCRIPTOR_SCHEMA,
            OPERATION_NEW,
            N as u64,
            M as u64,
            descriptor_len_word,
        ];
        if !agree_header(communicator, header) {
            return Err(AllToAllvTransposeError::CollectiveDescriptorMismatch);
        }
        let descriptor = collective_descriptor(communicator, descriptor_result, expected_len.ok())?;

        let local_validation = validate_plan_layout(&source, &destination);
        if !collective_valid(communicator, local_validation.is_ok()) {
            return Err(local_validation
                .err()
                .unwrap_or(AllToAllvTransposeError::CollectivePreconditionFailed));
        }
        let changed_topology_axis = local_validation.expect("collective validation succeeded");

        let pattern = build_peer_metadata(&source, &destination, changed_topology_axis);
        if !collective_valid(communicator, pattern.is_ok()) {
            return Err(pattern
                .err()
                .unwrap_or(AllToAllvTransposeError::CollectivePreconditionFailed));
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

    /// Returns the initialized workspace lengths required for `extra_shape`.
    ///
    /// This method is noncollective and does not allocate workspace storage.
    /// The returned lengths include all peer segments and are checked against
    /// `mpi::Count` for the counts and displacements used by execution.
    pub fn workspace_requirements(
        &self,
        extra_shape: &ExtraShape,
    ) -> Result<AllToAllvTransposeWorkspaceRequirements, AllToAllvTransposeError> {
        let batch = extra_shape.element_count();
        let mut send_len = 0usize;
        let mut receive_len = 0usize;
        for peer in &self.peers {
            let send_count = peer
                .send_spatial_len
                .checked_mul(batch)
                .ok_or(AllToAllvTransposeError::CountOverflow)?;
            let receive_count = peer
                .receive_spatial_len
                .checked_mul(batch)
                .ok_or(AllToAllvTransposeError::CountOverflow)?;
            let send_displacement = peer
                .send_spatial_displacement
                .checked_mul(batch)
                .ok_or(AllToAllvTransposeError::CountOverflow)?;
            let receive_displacement = peer
                .receive_spatial_displacement
                .checked_mul(batch)
                .ok_or(AllToAllvTransposeError::CountOverflow)?;
            checked_count(send_count, send_displacement)?;
            checked_count(receive_count, receive_displacement)?;
            send_len = send_len
                .checked_add(send_count)
                .ok_or(AllToAllvTransposeError::CountOverflow)?;
            receive_len = receive_len
                .checked_add(receive_count)
                .ok_or(AllToAllvTransposeError::CountOverflow)?;
        }
        Count::try_from(send_len).map_err(|_| AllToAllvTransposeError::CountOverflow)?;
        Count::try_from(receive_len).map_err(|_| AllToAllvTransposeError::CountOverflow)?;
        Ok(AllToAllvTransposeWorkspaceRequirements {
            send_len,
            receive_len,
        })
    }

    /// Executes the transpose after a whole-source-topology collective
    /// preflight.
    ///
    /// Every rank must call this method in the same order on the same source
    /// communicator context, with the same `T` and a correct
    /// `Equivalence` implementation. The `type_name::<T>()`, size, alignment,
    /// and `type_name::<T::Out>()` fields in the descriptor are only misuse
    /// detection; they are not a proof of type identity or of an unsafe
    /// `Equivalence` impl.
    ///
    /// The source is never written. Ordinary validation errors are returned on
    /// all ranks before packing, communication, or destination writes, so the
    /// destination is unchanged on those paths. Workspace lengths are checked
    /// by initialized `len`, and only the required prefixes are used. Every
    /// MPI count and displacement must fit `mpi::Count`. After the
    /// `MPI_Alltoallv` call starts, an MPI failure, arbitrary panic, or process
    /// loss does not guarantee that a `Result` is recovered or that the
    /// destination is unchanged. The source and destination views must use the
    /// layouts held by this plan and must have exactly equal extra shapes.
    pub fn execute_views<T>(
        &self,
        source: PencilArrayView<'_, T, N, M>,
        mut destination: PencilArrayViewMut<'_, T, N, M>,
        workspace: &mut AllToAllvTransposeWorkspace<T>,
    ) -> Result<(), AllToAllvTransposeError>
    where
        T: Equivalence + Copy,
    {
        let communicator = self.source.topology().cartesian();
        let expected_len = execute_descriptor_len::<T, N, M>(self, &source, &destination);
        let descriptor_result = expected_len
            .ok()
            .and_then(|_| build_execute_descriptor(self, &source, &destination).ok());
        let descriptor_len_word = expected_len.map_or(INVALID_AXIS, |len| len as u64);
        let header = [
            DESCRIPTOR_SCHEMA,
            OPERATION_EXECUTE,
            N as u64,
            M as u64,
            descriptor_len_word,
        ];
        if !agree_header(communicator, header) {
            return Err(AllToAllvTransposeError::CollectiveDescriptorMismatch);
        }
        let _descriptor =
            collective_descriptor(communicator, descriptor_result, expected_len.ok())?;

        let local_preflight = self.prepare_execution(&source, &destination, workspace);
        if !collective_valid(communicator, local_preflight.is_ok()) {
            return Err(local_preflight
                .err()
                .unwrap_or(AllToAllvTransposeError::CollectivePreconditionFailed));
        }
        let prepared = local_preflight.expect("collective execution preflight succeeded");

        pack_source(
            &self.peers,
            &source,
            &mut workspace.send_buffer[..prepared.requirements.send_len],
            prepared.requirements.send_len,
            source.extra_shape().element_count(),
        );

        {
            let subcommunicator = self
                .source
                .topology()
                .subcommunicator(self.changed_topology_axis);
            if prepared.requirements.send_len == 0 && prepared.requirements.receive_len == 0 {
                // Some MPI implementations require a distinct real buffer
                // address even when every count is zero. Rust empty slices have
                // a non-null dangling pointer, but this initialized i32 buffer
                // avoids relying on that address and never touches the user's
                // workspace. Every MPI type signature here has count zero, so
                // the dummy i32 does not fabricate a T value.
                let send_dummy = [0i32; 1];
                let mut receive_dummy = [0i32; 1];
                let send_partition = Partition::new(
                    &send_dummy[..],
                    prepared.send_counts.as_slice(),
                    prepared.send_displacements.as_slice(),
                );
                let mut receive_partition = PartitionMut::new(
                    &mut receive_dummy[..],
                    prepared.receive_counts.as_slice(),
                    prepared.receive_displacements.as_slice(),
                );
                subcommunicator.all_to_all_varcount_into(&send_partition, &mut receive_partition);
            } else {
                let send_buffer = &workspace.send_buffer[..prepared.requirements.send_len];
                let receive_buffer =
                    &mut workspace.receive_buffer[..prepared.requirements.receive_len];
                let send_partition = Partition::new(
                    send_buffer,
                    prepared.send_counts.as_slice(),
                    prepared.send_displacements.as_slice(),
                );
                let mut receive_partition = PartitionMut::new(
                    receive_buffer,
                    prepared.receive_counts.as_slice(),
                    prepared.receive_displacements.as_slice(),
                );
                subcommunicator.all_to_all_varcount_into(&send_partition, &mut receive_partition);
            }
        }

        let destination_extra_count = destination.extra_shape().element_count();
        unpack_destination(
            &self.peers,
            &mut destination,
            &workspace.receive_buffer[..prepared.requirements.receive_len],
            destination_extra_count,
        );
        Ok(())
    }

    fn prepare_execution<T>(
        &self,
        source: &PencilArrayView<'_, T, N, M>,
        destination: &PencilArrayViewMut<'_, T, N, M>,
        workspace: &AllToAllvTransposeWorkspace<T>,
    ) -> Result<PreparedExchange, AllToAllvTransposeError>
    where
        T: Equivalence + Copy,
    {
        if !source.pencil().same_layout(self.source.as_ref()) {
            return Err(AllToAllvTransposeError::SourceLayoutMismatch);
        }
        if !destination.pencil().same_layout(self.destination.as_ref()) {
            return Err(AllToAllvTransposeError::DestinationLayoutMismatch);
        }
        if source.extra_shape() != destination.extra_shape() {
            return Err(AllToAllvTransposeError::ExtraShapeMismatch);
        }

        let requirements = self.workspace_requirements(source.extra_shape())?;
        if workspace.send_buffer.len() < requirements.send_len
            || workspace.receive_buffer.len() < requirements.receive_len
        {
            return Err(AllToAllvTransposeError::WorkspaceTooSmall {
                send_required: requirements.send_len,
                send_len: workspace.send_buffer.len(),
                receive_required: requirements.receive_len,
                receive_len: workspace.receive_buffer.len(),
            });
        }

        let prepared = prepare_exchange_counts(self, source.extra_shape())?;
        for peer in &self.peers {
            let send_len = region_spatial_len(&peer.send_region)?;
            let receive_len = region_spatial_len(&peer.receive_region)?;
            if send_len != peer.send_spatial_len || receive_len != peer.receive_spatial_len {
                return Err(AllToAllvTransposeError::PreparationFailed);
            }
            validate_region_offsets(
                self.source.as_ref(),
                source.len(),
                &peer.send_region,
                source.extra_shape().element_count(),
            )?;
            validate_region_offsets(
                self.destination.as_ref(),
                destination.len(),
                &peer.receive_region,
                destination.extra_shape().element_count(),
            )?;
        }
        Ok(prepared)
    }
}

fn validate_plan_layout<const N: usize, const M: usize>(
    source: &Pencil<N, M>,
    destination: &Pencil<N, M>,
) -> Result<usize, AllToAllvTransposeError> {
    if !source.same_topology(destination) {
        return Err(AllToAllvTransposeError::IncompatibleTopology);
    }
    if source.global_shape() != destination.global_shape() {
        return Err(AllToAllvTransposeError::IncompatibleGlobalShape);
    }
    let mut changed = None;
    for axis in 0..M {
        if source.decomposition()[axis] != destination.decomposition()[axis] {
            if changed.is_some() {
                return Err(AllToAllvTransposeError::UnsupportedDecompositionChange);
            }
            changed = Some(axis);
        }
    }
    changed.ok_or(AllToAllvTransposeError::UnsupportedDecompositionChange)
}

fn build_peer_metadata<const N: usize, const M: usize>(
    source: &Pencil<N, M>,
    destination: &Pencil<N, M>,
    changed_topology_axis: usize,
) -> Result<Vec<PeerMetadata<N>>, AllToAllvTransposeError> {
    let subcommunicator = source.topology().subcommunicator(changed_topology_axis);
    let peer_count = usize::try_from(subcommunicator.size())
        .map_err(|_| AllToAllvTransposeError::PreparationFailed)?;
    let mut peers = Vec::new();
    peers
        .try_reserve_exact(peer_count)
        .map_err(|_| AllToAllvTransposeError::PreparationFailed)?;
    let mut send_displacement = 0usize;
    let mut receive_displacement = 0usize;
    let local_coords = *source.topology().local_coords();
    let mut one_axis_coordinate = [0i32; 1];

    for peer_index in 0..peer_count {
        let peer_rank =
            Count::try_from(peer_index).map_err(|_| AllToAllvTransposeError::PreparationFailed)?;
        subcommunicator.rank_to_coordinates_into(peer_rank, &mut one_axis_coordinate);
        let peer_coordinate = usize::try_from(one_axis_coordinate[0])
            .map_err(|_| AllToAllvTransposeError::PreparationFailed)?;
        let mut peer_coords = local_coords;
        peer_coords[changed_topology_axis] = peer_coordinate;

        let source_peer_ranges = source
            .ranges_at(peer_coords)
            .map_err(|_| AllToAllvTransposeError::PreparationFailed)?;
        let destination_peer_ranges = destination
            .ranges_at(peer_coords)
            .map_err(|_| AllToAllvTransposeError::PreparationFailed)?;
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
            .ok_or(AllToAllvTransposeError::PreparationFailed)?;
        receive_displacement = receive_displacement
            .checked_add(receive_spatial_len)
            .ok_or(AllToAllvTransposeError::PreparationFailed)?;
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

fn region_spatial_len<const N: usize>(
    region: &[Range<usize>; N],
) -> Result<usize, AllToAllvTransposeError> {
    let mut shape = [0usize; N];
    for axis in 0..N {
        shape[axis] = region[axis]
            .end
            .checked_sub(region[axis].start)
            .ok_or(AllToAllvTransposeError::PreparationFailed)?;
    }
    checked_product(&shape).map_err(|_| AllToAllvTransposeError::PreparationFailed)
}

fn prepare_exchange_counts<const N: usize, const M: usize>(
    plan: &AllToAllvTransposePlan<N, M>,
    extra_shape: &ExtraShape,
) -> Result<PreparedExchange, AllToAllvTransposeError> {
    let requirements = plan.workspace_requirements(extra_shape)?;
    let peer_count = plan.peers.len();
    let mut send_counts = zeroed_vec(peer_count)?;
    let mut send_displacements = zeroed_vec(peer_count)?;
    let mut receive_counts = zeroed_vec(peer_count)?;
    let mut receive_displacements = zeroed_vec(peer_count)?;
    let batch = extra_shape.element_count();
    let coordinate_extent = plan.source.topology().process_grid()[plan.changed_topology_axis];

    for peer in &plan.peers {
        let index = usize::try_from(peer.peer_rank)
            .map_err(|_| AllToAllvTransposeError::PreparationFailed)?;
        if index >= peer_count || peer.peer_coordinate >= coordinate_extent {
            return Err(AllToAllvTransposeError::PreparationFailed);
        }
        let send_count = peer
            .send_spatial_len
            .checked_mul(batch)
            .ok_or(AllToAllvTransposeError::CountOverflow)?;
        let receive_count = peer
            .receive_spatial_len
            .checked_mul(batch)
            .ok_or(AllToAllvTransposeError::CountOverflow)?;
        let send_displacement = peer
            .send_spatial_displacement
            .checked_mul(batch)
            .ok_or(AllToAllvTransposeError::CountOverflow)?;
        let receive_displacement = peer
            .receive_spatial_displacement
            .checked_mul(batch)
            .ok_or(AllToAllvTransposeError::CountOverflow)?;
        send_counts[index] =
            checked_partition_count(send_count, send_displacement, requirements.send_len)?;
        send_displacements[index] = Count::try_from(send_displacement)
            .map_err(|_| AllToAllvTransposeError::CountOverflow)?;
        receive_counts[index] = checked_partition_count(
            receive_count,
            receive_displacement,
            requirements.receive_len,
        )?;
        receive_displacements[index] = Count::try_from(receive_displacement)
            .map_err(|_| AllToAllvTransposeError::CountOverflow)?;
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
) -> Result<Count, AllToAllvTransposeError> {
    let count_as_mpi =
        Count::try_from(count).map_err(|_| AllToAllvTransposeError::CountOverflow)?;
    let displacement_as_mpi =
        Count::try_from(displacement).map_err(|_| AllToAllvTransposeError::CountOverflow)?;
    count_as_mpi
        .checked_add(displacement_as_mpi)
        .ok_or(AllToAllvTransposeError::CountOverflow)?;
    if count
        .checked_add(displacement)
        .ok_or(AllToAllvTransposeError::CountOverflow)?
        > total
    {
        return Err(AllToAllvTransposeError::PreparationFailed);
    }
    Ok(count_as_mpi)
}

fn checked_count(count: usize, displacement: usize) -> Result<(), AllToAllvTransposeError> {
    let count_as_mpi =
        Count::try_from(count).map_err(|_| AllToAllvTransposeError::CountOverflow)?;
    let displacement_as_mpi =
        Count::try_from(displacement).map_err(|_| AllToAllvTransposeError::CountOverflow)?;
    count_as_mpi
        .checked_add(displacement_as_mpi)
        .ok_or(AllToAllvTransposeError::CountOverflow)?;
    Ok(())
}

fn checked_count_len(len: usize) -> Result<(), AllToAllvTransposeError> {
    Count::try_from(len)
        .map(|_| ())
        .map_err(|_| AllToAllvTransposeError::CountOverflow)
}

fn zeroed_vec<T: Copy + Default>(len: usize) -> Result<Vec<T>, AllToAllvTransposeError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| AllToAllvTransposeError::PreparationFailed)?;
    values.resize(len, T::default());
    Ok(values)
}

fn validate_region_offsets<const N: usize, const M: usize>(
    pencil: &Pencil<N, M>,
    storage_len: usize,
    region: &[Range<usize>; N],
    extra_count: usize,
) -> Result<(), AllToAllvTransposeError> {
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
            .ok_or(AllToAllvTransposeError::PreparationFailed)?;
        for linear in 0..spatial_len {
            let global = region_index(region, linear)?;
            let spatial_offset = local_spatial_offset(pencil, global)?;
            let offset = extra_offset
                .checked_add(spatial_offset)
                .ok_or(AllToAllvTransposeError::PreparationFailed)?;
            if offset >= storage_len {
                return Err(AllToAllvTransposeError::PreparationFailed);
            }
        }
    }
    Ok(())
}

fn region_index<const N: usize>(
    region: &[Range<usize>; N],
    mut linear: usize,
) -> Result<[usize; N], AllToAllvTransposeError> {
    let mut shape = [0usize; N];
    for axis in 0..N {
        shape[axis] = region[axis]
            .end
            .checked_sub(region[axis].start)
            .ok_or(AllToAllvTransposeError::PreparationFailed)?;
    }
    let mut index = [0usize; N];
    for axis in (0..N).rev() {
        if shape[axis] == 0 {
            return Err(AllToAllvTransposeError::PreparationFailed);
        }
        index[axis] = region[axis].start + linear % shape[axis];
        linear /= shape[axis];
    }
    Ok(index)
}

fn local_spatial_offset<const N: usize, const M: usize>(
    pencil: &Pencil<N, M>,
    global: [usize; N],
) -> Result<usize, AllToAllvTransposeError> {
    let mut local = [0usize; N];
    for axis in 0..N {
        let range = &pencil.local_ranges()[axis];
        if global[axis] < range.start || global[axis] >= range.end {
            return Err(AllToAllvTransposeError::PreparationFailed);
        }
        local[axis] = global[axis] - range.start;
    }
    let memory = pencil.permutation().permute(local);
    row_major_offset(&pencil.local_shape_memory(), &memory)
        .map_err(|_| AllToAllvTransposeError::PreparationFailed)
}

// ponytail: pack/unpack stays the element-by-element reference mapping;
// add no index table or unmeasured optimization.
fn pack_source<T: Copy, const N: usize, const M: usize>(
    peers: &[PeerMetadata<N>],
    source: &PencilArrayView<'_, T, N, M>,
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
                .checked_mul(source.pencil().local_len())
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
                        local_spatial_offset(source.pencil(), global)
                            .expect("checked exchange preflight validated source mapping"),
                    )
                    .expect("checked exchange preflight validated source offset");
                let payload_index = payload_offset
                    .checked_add(linear)
                    .expect("checked exchange preflight validated send index");
                send_buffer[payload_index] = source.as_slice()[source_offset];
            }
        }
    }
}

fn unpack_destination<T: Copy, const N: usize, const M: usize>(
    peers: &[PeerMetadata<N>],
    destination: &mut PencilArrayViewMut<'_, T, N, M>,
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
                .checked_mul(destination.pencil().local_len())
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
                        local_spatial_offset(destination.pencil(), global)
                            .expect("checked exchange preflight validated destination mapping"),
                    )
                    .expect("checked exchange preflight validated destination offset");
                let payload_index = payload_offset
                    .checked_add(linear)
                    .expect("checked exchange preflight validated receive index");
                destination.as_mut_slice()[destination_offset] = receive_buffer[payload_index];
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
) -> Result<Vec<u64>, ()> {
    let len = plan_descriptor_len::<N, M>()?;
    let mut descriptor = Vec::new();
    descriptor.try_reserve_exact(len).map_err(|_| ())?;
    descriptor.push(DESCRIPTOR_SCHEMA);
    descriptor.push(OPERATION_NEW);
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

fn execute_descriptor_len<T: Equivalence, const N: usize, const M: usize>(
    plan: &AllToAllvTransposePlan<N, M>,
    source: &PencilArrayView<'_, T, N, M>,
    destination: &PencilArrayViewMut<'_, T, N, M>,
) -> Result<usize, ()> {
    let source_name_len = type_name::<T>().len();
    let mpi_name_len = type_name::<<T as Equivalence>::Out>().len();
    let mut length = 4usize.checked_add(plan.descriptor.len()).ok_or(())?;
    length = length
        .checked_add(
            1usize
                .checked_add(source.extra_shape().dimensions().len())
                .ok_or(())?,
        )
        .ok_or(())?;
    length = length
        .checked_add(
            1usize
                .checked_add(destination.extra_shape().dimensions().len())
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
    plan: &AllToAllvTransposePlan<N, M>,
    source: &PencilArrayView<'_, T, N, M>,
    destination: &PencilArrayViewMut<'_, T, N, M>,
) -> Result<Vec<u64>, ()>
where
    T: Equivalence,
{
    let len = execute_descriptor_len(plan, source, destination)?;
    let mut descriptor = Vec::new();
    descriptor.try_reserve_exact(len).map_err(|_| ())?;
    descriptor.push(DESCRIPTOR_SCHEMA);
    descriptor.push(OPERATION_EXECUTE);
    descriptor.push(N as u64);
    descriptor.push(M as u64);
    descriptor.extend(plan.descriptor.iter().copied());
    append_shape(&mut descriptor, source.extra_shape());
    append_shape(&mut descriptor, destination.extra_shape());
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

fn collective_valid<C: CommunicatorCollectives>(comm: &C, local_valid: bool) -> bool {
    let value = i32::from(local_valid);
    let mut result = 0i32;
    comm.all_reduce_into(&value, &mut result, SystemOperation::min());
    result != 0
}

fn collective_descriptor<C: CommunicatorCollectives>(
    comm: &C,
    local_descriptor: Option<Vec<u64>>,
    expected_len: Option<usize>,
) -> Result<Vec<u64>, AllToAllvTransposeError> {
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
        return Err(AllToAllvTransposeError::PreparationFailed);
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
        return Err(AllToAllvTransposeError::CollectiveDescriptorMismatch);
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
