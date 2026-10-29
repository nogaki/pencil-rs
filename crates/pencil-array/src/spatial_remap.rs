//! Checked affine redistribution between arbitrary pencil layouts.

use std::{
    any::type_name,
    fmt,
    mem::{MaybeUninit, align_of, size_of},
    sync::Arc,
};

use mpi::{
    Count,
    collective::{CommunicatorCollectives, SystemOperation},
    datatype::{DatatypeRef, Equivalence, MutView, Partition, PartitionMut, UserDatatype, View},
    raw::traits::{AsRaw, FromRaw},
    topology::Communicator,
};
use thiserror::Error;

use crate::{
    ExtraShape, MpiTopology, Pencil, PencilError, TopologyError,
    checked::checked_product,
    geometry::{partition_range, row_major_offset},
};

const DESCRIPTOR_SCHEMA: u64 = 1;
const OP_PREPARE: u64 = 0x3601;
const OP_APPLY: u64 = 0x3602;
const HEADER_WORDS: usize = 5;
const DESCRIPTOR_CHUNK: usize = 64;

/// Errors returned while preparing or applying a [`SpatialRemapPlan`].
#[derive(Debug, Error)]
pub enum SpatialRemapError {
    /// The two pencils do not share the same topology object.
    #[error("source and destination pencils use different topology objects")]
    DifferentTopology,

    /// The source and destination global spatial shapes differ.
    #[error("source and destination global spatial shapes differ")]
    DifferentShape,

    /// An affine-map sign is not `-1` or `1`.
    #[error("affine sign {sign} on spatial axis {axis} is not -1 or 1")]
    InvalidAffineSign {
        /// The spatial axis with the invalid sign.
        axis: usize,
        /// The rejected sign.
        sign: i8,
    },

    /// A supplied view's extra dimensions differ from the prepared shape.
    #[error("view extra shape differs from the spatial remap plan")]
    DifferentExtraShape,

    /// A supplied view's pencil layout differs from the prepared layout.
    #[error("view layout differs from the spatial remap plan")]
    LayoutMismatch,

    /// At least one rank rejected a local collective precondition.
    #[error("a collective spatial-remap precondition failed on another rank")]
    CollectivePreconditionFailed,

    /// Ranks supplied different operation or plan descriptors.
    #[error("collective spatial-remap descriptors differ between ranks")]
    CollectiveDescriptorMismatch,

    /// A shape, linear index, byte count, or displacement overflowed `usize`.
    #[error("spatial-remap size or index overflowed usize")]
    SizeOverflow,

    /// An element count or displacement does not fit MPI's `Count`.
    #[error("spatial-remap count or displacement does not fit MPI Count")]
    MpiCountOverflow,

    /// Allocation failed during collective preparation.
    #[error("allocation failed while preparing {stage} ({elements} elements)")]
    Allocation {
        /// The data being prepared.
        stage: &'static str,
        /// The requested number of elements.
        elements: usize,
    },

    /// An index sent to this rank is not owned by this rank.
    #[error("rank {rank} received a request for non-owned global index {global_index}")]
    WrongOwner {
        /// The rank that received the invalid request.
        rank: i32,
        /// The requested row-major global spatial index.
        global_index: u64,
    },

    /// Prepared metadata contradicted the validated pencil geometry.
    #[error("spatial-remap metadata is internally inconsistent")]
    StructuralInconsistency,

    /// The payload type or its MPI datatype cannot safely describe one Rust value.
    #[error("payload type has an unsupported MPI datatype layout")]
    UnsupportedDatatype,

    /// A pencil operation failed while preparing the remap.
    #[error(transparent)]
    Pencil(#[from] PencilError),

    /// A topology operation failed while preparing the remap.
    #[error(transparent)]
    Topology(#[from] TopologyError),
}

#[derive(Clone, Copy)]
struct RequestRoute {
    owner: usize,
    global_index: u64,
    destination_offset: usize,
}

struct OwnerRoutes {
    entries: Vec<RequestRoute>,
    counts: Vec<Count>,
    displacements: Vec<Count>,
    total_count: Count,
}

struct RequestBuffers {
    send_indices: Vec<u64>,
    destination_offsets: Vec<usize>,
    receive_indices: Vec<u64>,
}

struct ReceiveLayout {
    displacements: Vec<Count>,
    total: usize,
}

struct PayloadLayout {
    send_counts: Vec<Count>,
    send_displacements: Vec<Count>,
    receive_counts: Vec<Count>,
    receive_displacements: Vec<Count>,
    send_len: usize,
    receive_len: usize,
    send_count: Count,
    receive_count: Count,
}

/// A reusable, collectively prepared affine redistribution plan.
///
/// `prepare` and every `apply_views` call are collective over the source
/// topology's full Cartesian communicator. The source and destination pencils
/// must share that rank-local topology object and global shape; arbitrary
/// decomposition and memory permutations are supported. Each logical point
/// maps as `source[a] = (signs[a] * destination[a] + offsets[a]) mod extent[a]`.
/// The raw `i128` offsets are collectively compared without normalization;
/// reduction modulo the positive extent is used only while preparing routes.
/// Every extra-shape group uses the same spatial map, and its axes are not
/// transformed.
///
/// The plan owns the pencils, exact extra shape, descriptor, routes, a duplicated
/// standard MPI datatype, and initialized send/receive storage. Reuse the plan
/// for repeated calls with checked views of those layouts. `apply_views` does
/// not allocate or call user `Clone`, `Default`, `Equivalence`, or datatype
/// methods. Its collective preflight runs before packing and ordinary failures
/// preserve both views and all plan contents. The payload type must have a
/// correct unsafe `Equivalence` implementation whose typemap accesses only
/// initialized fields within each `T`; the plan validates extent and true
/// extent, but cannot prove a custom typemap's semantic correctness.
///
/// Drop the plan and its topology before MPI is finalized. MPI failures,
/// process loss, and arbitrary panics are outside the recovery guarantee.
pub struct SpatialRemapPlan<T, const N: usize, const M: usize> {
    source: Arc<Pencil<N, M>>,
    destination: Arc<Pencil<N, M>>,
    extra_shape: ExtraShape,
    signs: [i8; N],
    offsets: [i128; N],
    descriptor: Vec<u64>,
    request_send_counts: Vec<Count>,
    request_send_displacements: Vec<Count>,
    request_receive_counts: Vec<Count>,
    request_receive_displacements: Vec<Count>,
    destination_offsets: Vec<usize>,
    source_offsets: Vec<usize>,
    payload_send_counts: Vec<Count>,
    payload_send_displacements: Vec<Count>,
    payload_receive_counts: Vec<Count>,
    payload_receive_displacements: Vec<Count>,
    send_buffer: Vec<T>,
    receive_buffer: Vec<T>,
    send_len: usize,
    receive_len: usize,
    send_count: Count,
    receive_count: Count,
    datatype: UserDatatype,
}

impl<T, const N: usize, const M: usize> fmt::Debug for SpatialRemapPlan<T, N, M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpatialRemapPlan")
            .field("source", &self.source)
            .field("destination", &self.destination)
            .field("extra_shape", &self.extra_shape)
            .field("signs", &self.signs)
            .field("offsets", &self.offsets)
            .field("send_len", &self.send_len)
            .field("receive_len", &self.receive_len)
            .finish_non_exhaustive()
    }
}

impl<T, const N: usize, const M: usize> SpatialRemapPlan<T, N, M>
where
    T: Equivalence + Copy,
{
    /// Collectively prepares a reusable affine redistribution plan.
    ///
    /// All ranks must use the same source communicator context, `T`, `N`, `M`,
    /// call order, and correct `Equivalence` implementation. Preparation
    /// compares the complete descriptor and allocates all routes and initialized
    /// payload buffers before returning. `buffer_init` is copied into those
    /// buffers; it need not implement `Default`. Zero-sized payload types and
    /// payload counts beyond MPI `Count` are rejected rather than chunked.
    pub fn prepare(
        source: Arc<Pencil<N, M>>,
        destination: Arc<Pencil<N, M>>,
        extra_shape: ExtraShape,
        signs: [i8; N],
        offsets: [i128; N],
        buffer_init: T,
    ) -> Result<Self, SpatialRemapError> {
        let communicator = source.topology().communicator();
        let header = [
            DESCRIPTOR_SCHEMA,
            OP_PREPARE,
            dimension_word(N),
            dimension_word(M),
            0,
        ];
        if !agree_header(communicator, header) {
            return Err(SpatialRemapError::CollectiveDescriptorMismatch);
        }

        agree_signs(communicator, &signs)?;
        agree_pencil_metadata(communicator, &source, &destination)?;
        let descriptor = agree_plan_descriptor::<T, N, M, _>(
            communicator,
            &source,
            &destination,
            &extra_shape,
            &signs,
            &offsets,
        )?;

        if size_of::<T>() == 0 {
            return Err(SpatialRemapError::UnsupportedDatatype);
        }

        let routes = agree_stage(
            communicator,
            build_owner_routes(&source, &destination, &signs, &offsets),
        )?;
        let communicator_size = source.topology().size();
        let receive_counts_result =
            filled_vec(communicator_size, 0 as Count, "request receive counts");
        let mut receive_counts = agree_stage(communicator, receive_counts_result)?;
        communicator.all_to_all_into(&routes.counts[..], &mut receive_counts[..]);

        let receive_layout = agree_stage(
            communicator,
            receive_layout(&receive_counts, source.local_len()),
        )?;
        let mut request_buffers = agree_stage(
            communicator,
            allocate_request_buffers(&routes, receive_layout.total),
        )?;

        let send_partition = Partition::new(
            &request_buffers.send_indices[..],
            routes.counts.as_slice(),
            routes.displacements.as_slice(),
        );
        let mut receive_partition = PartitionMut::new(
            &mut request_buffers.receive_indices[..],
            receive_counts.as_slice(),
            receive_layout.displacements.as_slice(),
        );
        communicator.all_to_all_varcount_into(&send_partition, &mut receive_partition);

        let source_offsets = agree_stage(
            communicator,
            incoming_source_offsets(
                &source,
                &request_buffers.receive_indices[..receive_layout.total],
            ),
        )?;
        let payload = agree_stage(
            communicator,
            payload_layout::<T>(
                &routes,
                &receive_counts,
                &receive_layout.displacements,
                source.local_len(),
                destination.local_len(),
                extra_shape.element_count(),
                communicator_size,
            ),
        )?;
        let (send_buffer, receive_buffer) = agree_stage(
            communicator,
            allocate_payload_buffers(payload.send_len, payload.receive_len, buffer_init),
        )?;
        let datatype = agree_stage(
            communicator,
            duplicate_checked_datatype::<T, M>(source.topology().as_ref()),
        )?;

        Ok(Self {
            source,
            destination,
            extra_shape,
            signs,
            offsets,
            descriptor,
            request_send_counts: routes.counts,
            request_send_displacements: routes.displacements,
            request_receive_counts: receive_counts,
            request_receive_displacements: receive_layout.displacements,
            destination_offsets: request_buffers.destination_offsets,
            source_offsets,
            payload_send_counts: payload.send_counts,
            payload_send_displacements: payload.send_displacements,
            payload_receive_counts: payload.receive_counts,
            payload_receive_displacements: payload.receive_displacements,
            send_buffer,
            receive_buffer,
            send_len: payload.send_len,
            receive_len: payload.receive_len,
            send_count: payload.send_count,
            receive_count: payload.receive_count,
            datatype,
        })
    }

    /// Collectively applies the prepared map to checked source and destination views.
    ///
    /// View construction itself is local; callers must agree any failure there
    /// before entering this method. This method first agrees the operation and
    /// immutable plan descriptor, then validates extra shapes and pencil layouts
    /// on all ranks. Only after successful validation does it pack, execute one
    /// full-Cartesian `MPI_Alltoallv`, and unpack. It performs no Rust
    /// allocation. Ordinary preflight errors leave source, destination, and all
    /// plan contents unchanged.
    pub fn apply_views(
        &mut self,
        source: crate::PencilArrayView<'_, T, N, M>,
        mut destination: crate::PencilArrayViewMut<'_, T, N, M>,
    ) -> Result<(), SpatialRemapError> {
        let topology = Arc::clone(self.source.topology());
        let communicator = topology.communicator();
        let header = [
            DESCRIPTOR_SCHEMA,
            OP_APPLY,
            dimension_word(N),
            dimension_word(M),
            dimension_word(self.descriptor.len()),
        ];
        if !agree_header(communicator, header) {
            return Err(SpatialRemapError::CollectiveDescriptorMismatch);
        }
        if !descriptors_equal(communicator, &self.descriptor) {
            return Err(SpatialRemapError::CollectiveDescriptorMismatch);
        }

        let validation = validate_views(self, &source, &destination);
        agree_stage(communicator, validation)?;

        pack_payload(self, source.as_slice());
        {
            let send_view =
                // SAFETY: `prepare` duplicated and checked this datatype: its extent
                // equals `size_of::<T>()`, its true typemap lies within one T, and
                // `Equivalence` guarantees it does not read uninitialized padding.
                // Counts/displacements were checked against the initialized logical
                // lengths; the physical one-element dummy is used only for count zero.
                unsafe {
                    View::with_count_and_datatype(
                        &self.send_buffer[..],
                        self.send_count,
                        &self.datatype,
                    )
                };
            let mut receive_view =
                // SAFETY: the corresponding receive datatype, bounds, and logical
                // count were validated during preparation; the exclusive view cannot
                // alias the source or its separately owned send buffer.
                unsafe {
                    MutView::with_count_and_datatype(
                        &mut self.receive_buffer[..],
                        self.receive_count,
                        &self.datatype,
                    )
                };
            let send_partition = Partition::new(
                &send_view,
                self.payload_send_counts.as_slice(),
                self.payload_send_displacements.as_slice(),
            );
            let mut receive_partition = PartitionMut::new(
                &mut receive_view,
                self.payload_receive_counts.as_slice(),
                self.payload_receive_displacements.as_slice(),
            );
            communicator.all_to_all_varcount_into(&send_partition, &mut receive_partition);
        }
        unpack_payload(self, destination.as_mut_slice());
        Ok(())
    }
}

fn dimension_word(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn agree_header<C: CommunicatorCollectives>(comm: &C, header: [u64; HEADER_WORDS]) -> bool {
    let mut minimum = [0u64; HEADER_WORDS];
    let mut maximum = [0u64; HEADER_WORDS];
    comm.all_reduce_into(&header[..], &mut minimum[..], SystemOperation::min());
    comm.all_reduce_into(&header[..], &mut maximum[..], SystemOperation::max());
    minimum == maximum
}

fn agree_stage<C, V>(comm: &C, local: Result<V, SpatialRemapError>) -> Result<V, SpatialRemapError>
where
    C: CommunicatorCollectives,
{
    if all_ranks_succeeded_from_comm(comm, local.is_ok()) {
        local
    } else {
        Err(local
            .err()
            .unwrap_or(SpatialRemapError::CollectivePreconditionFailed))
    }
}

fn all_ranks_succeeded_from_comm<C: CommunicatorCollectives>(comm: &C, ok: bool) -> bool {
    let local = i32::from(ok);
    let mut all = 0i32;
    comm.all_reduce_into(&local, &mut all, SystemOperation::min());
    all != 0
}

fn agree_signs<C: CommunicatorCollectives, const N: usize>(
    comm: &C,
    signs: &[i8; N],
) -> Result<(), SpatialRemapError> {
    let mut equal = true;
    for chunk in signs.chunks(DESCRIPTOR_CHUNK) {
        let mut minimum = [i8::MAX; DESCRIPTOR_CHUNK];
        let mut maximum = [i8::MIN; DESCRIPTOR_CHUNK];
        comm.all_reduce_into(chunk, &mut minimum[..chunk.len()], SystemOperation::min());
        comm.all_reduce_into(chunk, &mut maximum[..chunk.len()], SystemOperation::max());
        equal &= minimum[..chunk.len()] == maximum[..chunk.len()];
    }
    if !equal {
        return Err(SpatialRemapError::CollectiveDescriptorMismatch);
    }
    let valid = signs
        .iter()
        .enumerate()
        .find(|(_, sign)| **sign != -1 && **sign != 1)
        .map(|(axis, sign)| SpatialRemapError::InvalidAffineSign { axis, sign: *sign })
        .map_or(Ok(()), Err);
    agree_stage(comm, valid)
}

fn agree_pencil_metadata<C, const N: usize, const M: usize>(
    comm: &C,
    source: &Pencil<N, M>,
    destination: &Pencil<N, M>,
) -> Result<(), SpatialRemapError>
where
    C: CommunicatorCollectives,
{
    let same_topology = if source.same_topology(destination) {
        Ok(())
    } else {
        Err(SpatialRemapError::DifferentTopology)
    };
    agree_stage(comm, same_topology)?;

    let same_shape = if source.global_shape() == destination.global_shape() {
        Ok(())
    } else {
        Err(SpatialRemapError::DifferentShape)
    };
    agree_stage(comm, same_shape)?;

    agree_usize_array(
        comm,
        source.topology().process_grid(),
        SpatialRemapError::DifferentTopology,
    )?;
    let size = source.topology().size();
    let mut minimum_size = 0usize;
    let mut maximum_size = 0usize;
    comm.all_reduce_into(&size, &mut minimum_size, SystemOperation::min());
    comm.all_reduce_into(&size, &mut maximum_size, SystemOperation::max());
    if minimum_size != maximum_size {
        return Err(SpatialRemapError::DifferentTopology);
    }

    agree_usize_array(
        comm,
        source.global_shape(),
        SpatialRemapError::DifferentShape,
    )
}

fn agree_usize_array<C, const K: usize>(
    comm: &C,
    values: &[usize; K],
    mismatch: SpatialRemapError,
) -> Result<(), SpatialRemapError>
where
    C: CommunicatorCollectives,
{
    agree_stage(
        comm,
        Count::try_from(K)
            .map(|_| ())
            .map_err(|_| SpatialRemapError::MpiCountOverflow),
    )?;
    if K == 0 {
        return Ok(());
    }
    let mut minimum = [0usize; K];
    let mut maximum = [0usize; K];
    comm.all_reduce_into(&values[..], &mut minimum[..], SystemOperation::min());
    comm.all_reduce_into(&values[..], &mut maximum[..], SystemOperation::max());
    if minimum == maximum {
        Ok(())
    } else {
        Err(mismatch)
    }
}

fn descriptor_length<T: Equivalence, const N: usize, const M: usize>(
    extra_shape: &ExtraShape,
) -> Result<usize, SpatialRemapError> {
    let mut length = 4usize;
    for amount in [
        1usize
            .checked_add(M)
            .ok_or(SpatialRemapError::SizeOverflow)?,
        N.checked_mul(4).ok_or(SpatialRemapError::SizeOverflow)?,
        M.checked_mul(2).ok_or(SpatialRemapError::SizeOverflow)?,
        1usize
            .checked_add(extra_shape.dimensions().len())
            .ok_or(SpatialRemapError::SizeOverflow)?,
        N,
        N.checked_mul(2).ok_or(SpatialRemapError::SizeOverflow)?,
        2,
        1usize
            .checked_add(type_name::<T>().len())
            .ok_or(SpatialRemapError::SizeOverflow)?,
        1usize
            .checked_add(type_name::<<T as Equivalence>::Out>().len())
            .ok_or(SpatialRemapError::SizeOverflow)?,
    ] {
        length = length
            .checked_add(amount)
            .ok_or(SpatialRemapError::SizeOverflow)?;
    }
    Ok(length)
}

fn agree_plan_descriptor<T, const N: usize, const M: usize, C>(
    comm: &C,
    source: &Pencil<N, M>,
    destination: &Pencil<N, M>,
    extra_shape: &ExtraShape,
    signs: &[i8; N],
    offsets: &[i128; N],
) -> Result<Vec<u64>, SpatialRemapError>
where
    T: Equivalence,
    C: CommunicatorCollectives,
{
    let descriptor_len = agree_stage(comm, descriptor_length::<T, N, M>(extra_shape))?;
    let length_word = agree_stage(
        comm,
        u64::try_from(descriptor_len).map_err(|_| SpatialRemapError::SizeOverflow),
    )?;
    let mut minimum_len = 0u64;
    let mut maximum_len = 0u64;
    comm.all_reduce_into(&length_word, &mut minimum_len, SystemOperation::min());
    comm.all_reduce_into(&length_word, &mut maximum_len, SystemOperation::max());
    if minimum_len != maximum_len {
        return Err(SpatialRemapError::CollectiveDescriptorMismatch);
    }

    let descriptor = agree_stage(
        comm,
        build_descriptor::<T, N, M>(
            descriptor_len,
            source,
            destination,
            extra_shape,
            signs,
            offsets,
        ),
    )?;
    if descriptors_equal(comm, &descriptor) {
        Ok(descriptor)
    } else {
        Err(SpatialRemapError::CollectiveDescriptorMismatch)
    }
}

fn build_descriptor<T, const N: usize, const M: usize>(
    length: usize,
    source: &Pencil<N, M>,
    destination: &Pencil<N, M>,
    extra_shape: &ExtraShape,
    signs: &[i8; N],
    offsets: &[i128; N],
) -> Result<Vec<u64>, SpatialRemapError>
where
    T: Equivalence,
{
    let mut descriptor = Vec::new();
    descriptor
        .try_reserve_exact(length)
        .map_err(|_| SpatialRemapError::Allocation {
            stage: "collective descriptor",
            elements: length,
        })?;
    descriptor.extend([
        DESCRIPTOR_SCHEMA,
        OP_PREPARE,
        dimension_word(N),
        dimension_word(M),
        dimension_word(source.topology().size()),
    ]);
    append_usizes(&mut descriptor, source.topology().process_grid())?;
    append_pencil(&mut descriptor, source)?;
    append_pencil(&mut descriptor, destination)?;
    push_usize(&mut descriptor, extra_shape.dimensions().len())?;
    append_usizes(&mut descriptor, extra_shape.dimensions())?;
    for sign in signs {
        descriptor.push(u64::from(sign.to_le_bytes()[0]));
    }
    for offset in offsets {
        let bytes = offset.to_le_bytes();
        let mut low = [0u8; 8];
        let mut high = [0u8; 8];
        low.copy_from_slice(&bytes[..8]);
        high.copy_from_slice(&bytes[8..]);
        descriptor.push(u64::from_le_bytes(low));
        descriptor.push(u64::from_le_bytes(high));
    }
    push_usize(&mut descriptor, size_of::<T>())?;
    push_usize(&mut descriptor, align_of::<T>())?;
    append_type_name(&mut descriptor, type_name::<T>())?;
    append_type_name(&mut descriptor, type_name::<<T as Equivalence>::Out>())?;
    if descriptor.len() != length {
        return Err(SpatialRemapError::StructuralInconsistency);
    }
    Ok(descriptor)
}

fn append_pencil<const N: usize, const M: usize>(
    descriptor: &mut Vec<u64>,
    pencil: &Pencil<N, M>,
) -> Result<(), SpatialRemapError> {
    append_usizes(descriptor, pencil.global_shape())?;
    for axis in pencil.decomposition() {
        push_usize(descriptor, axis.index())?;
    }
    for axis in pencil.permutation().axes() {
        push_usize(descriptor, axis.index())?;
    }
    Ok(())
}

fn append_usizes(descriptor: &mut Vec<u64>, values: &[usize]) -> Result<(), SpatialRemapError> {
    for value in values {
        push_usize(descriptor, *value)?;
    }
    Ok(())
}

fn push_usize(descriptor: &mut Vec<u64>, value: usize) -> Result<(), SpatialRemapError> {
    descriptor.push(u64::try_from(value).map_err(|_| SpatialRemapError::SizeOverflow)?);
    Ok(())
}

fn append_type_name(descriptor: &mut Vec<u64>, name: &str) -> Result<(), SpatialRemapError> {
    push_usize(descriptor, name.len())?;
    descriptor.extend(name.as_bytes().iter().copied().map(u64::from));
    Ok(())
}

fn descriptors_equal<C: CommunicatorCollectives>(comm: &C, descriptor: &[u64]) -> bool {
    let mut equal = true;
    for chunk in descriptor.chunks(DESCRIPTOR_CHUNK) {
        let mut minimum = [0u64; DESCRIPTOR_CHUNK];
        let mut maximum = [0u64; DESCRIPTOR_CHUNK];
        comm.all_reduce_into(chunk, &mut minimum[..chunk.len()], SystemOperation::min());
        comm.all_reduce_into(chunk, &mut maximum[..chunk.len()], SystemOperation::max());
        equal &= minimum[..chunk.len()] == maximum[..chunk.len()];
    }
    equal
}

fn filled_vec<T: Copy>(
    length: usize,
    value: T,
    stage: &'static str,
) -> Result<Vec<T>, SpatialRemapError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| SpatialRemapError::Allocation {
            stage,
            elements: length,
        })?;
    values.extend(std::iter::repeat_with(|| value).take(length));
    Ok(values)
}

fn build_owner_routes<const N: usize, const M: usize>(
    source: &Pencil<N, M>,
    destination: &Pencil<N, M>,
    signs: &[i8; N],
    offsets: &[i128; N],
) -> Result<OwnerRoutes, SpatialRemapError> {
    let rank_count = source.topology().size();
    let mut entries = Vec::new();
    entries
        .try_reserve_exact(destination.local_len())
        .map_err(|_| SpatialRemapError::Allocation {
            stage: "owner request routes",
            elements: destination.local_len(),
        })?;

    for destination_offset in 0..destination.local_len() {
        let local = local_coordinate(destination, destination_offset)?;
        let mut destination_global = [0usize; N];
        for axis in 0..N {
            destination_global[axis] = destination.local_ranges()[axis]
                .start
                .checked_add(local[axis])
                .ok_or(SpatialRemapError::SizeOverflow)?;
        }
        let source_global =
            affine_coordinate(destination_global, *source.global_shape(), signs, offsets)?;
        let owner = owner_rank(source, source_global)?;
        let global_linear = row_major_offset(&source.global_shape()[..], &source_global[..])
            .map_err(|_| SpatialRemapError::StructuralInconsistency)?;
        entries.push(RequestRoute {
            owner,
            global_index: u64::try_from(global_linear)
                .map_err(|_| SpatialRemapError::SizeOverflow)?,
            destination_offset,
        });
    }
    entries.sort_unstable_by_key(|entry| entry.owner);

    let mut owner_counts = filled_vec(rank_count, 0usize, "owner request counts")?;
    for entry in &entries {
        let count = owner_counts
            .get_mut(entry.owner)
            .ok_or(SpatialRemapError::StructuralInconsistency)?;
        *count = count
            .checked_add(1)
            .ok_or(SpatialRemapError::SizeOverflow)?;
    }
    let mut counts = filled_vec(rank_count, 0 as Count, "owner MPI counts")?;
    let mut displacements = filled_vec(rank_count, 0 as Count, "owner MPI displacements")?;
    let mut displacement = 0usize;
    for peer in 0..rank_count {
        let count = owner_counts[peer];
        counts[peer] = Count::try_from(count).map_err(|_| SpatialRemapError::MpiCountOverflow)?;
        displacements[peer] =
            Count::try_from(displacement).map_err(|_| SpatialRemapError::MpiCountOverflow)?;
        checked_count_end(counts[peer], displacements[peer])?;
        displacement = displacement
            .checked_add(count)
            .ok_or(SpatialRemapError::SizeOverflow)?;
    }
    let total_count =
        Count::try_from(displacement).map_err(|_| SpatialRemapError::MpiCountOverflow)?;
    if displacement != entries.len() {
        return Err(SpatialRemapError::StructuralInconsistency);
    }
    validate_partition_layout(&counts, &displacements, entries.len())?;
    Ok(OwnerRoutes {
        entries,
        counts,
        displacements,
        total_count,
    })
}

fn local_coordinate<const N: usize, const M: usize>(
    pencil: &Pencil<N, M>,
    memory_linear: usize,
) -> Result<[usize; N], SpatialRemapError> {
    if memory_linear >= pencil.local_len() {
        return Err(SpatialRemapError::StructuralInconsistency);
    }
    let memory_shape = pencil.local_shape_memory();
    let mut remainder = memory_linear;
    let mut memory_coordinate = [0usize; N];
    for axis in (0..N).rev() {
        if memory_shape[axis] == 0 {
            return Err(SpatialRemapError::StructuralInconsistency);
        }
        memory_coordinate[axis] = remainder % memory_shape[axis];
        remainder /= memory_shape[axis];
    }
    if remainder != 0 {
        return Err(SpatialRemapError::StructuralInconsistency);
    }
    let mut logical = [0usize; N];
    for (memory_axis, axis) in pencil.permutation().axes().iter().enumerate() {
        logical[axis.index()] = memory_coordinate[memory_axis];
    }
    Ok(logical)
}

fn affine_coordinate<const N: usize>(
    destination: [usize; N],
    shape: [usize; N],
    signs: &[i8; N],
    offsets: &[i128; N],
) -> Result<[usize; N], SpatialRemapError> {
    let mut source = [0usize; N];
    for axis in 0..N {
        let extent = i128::try_from(shape[axis]).map_err(|_| SpatialRemapError::SizeOverflow)?;
        if extent <= 0 || (signs[axis] != -1 && signs[axis] != 1) {
            return Err(SpatialRemapError::StructuralInconsistency);
        }
        let destination =
            i128::try_from(destination[axis]).map_err(|_| SpatialRemapError::SizeOverflow)?;
        let signed = if signs[axis] == 1 {
            destination
        } else {
            -destination
        };
        let translated = signed
            .checked_add(offsets[axis].rem_euclid(extent))
            .ok_or(SpatialRemapError::SizeOverflow)?;
        source[axis] = usize::try_from(translated.rem_euclid(extent))
            .map_err(|_| SpatialRemapError::SizeOverflow)?;
    }
    Ok(source)
}

fn owner_rank<const N: usize, const M: usize>(
    source: &Pencil<N, M>,
    global: [usize; N],
) -> Result<usize, SpatialRemapError> {
    let mut owner_coords = [0usize; M];
    for (topology_axis, owner_coordinate) in owner_coords.iter_mut().enumerate() {
        let spatial_axis = source.decomposition()[topology_axis].index();
        *owner_coordinate = partition_owner(
            global[spatial_axis],
            source.global_shape()[spatial_axis],
            source.topology().process_grid()[topology_axis],
        )?;
    }
    let rank = source.topology().rank_at(owner_coords)?;
    let rank = usize::try_from(rank).map_err(|_| SpatialRemapError::StructuralInconsistency)?;
    if rank >= source.topology().size() {
        return Err(SpatialRemapError::StructuralInconsistency);
    }
    Ok(rank)
}

fn partition_owner(index: usize, extent: usize, parts: usize) -> Result<usize, SpatialRemapError> {
    if extent == 0 || index >= extent || parts == 0 {
        return Err(SpatialRemapError::StructuralInconsistency);
    }
    let mut low = 0usize;
    let mut high = parts;
    while low < high {
        let middle = low + (high - low) / 2;
        let range = partition_range(extent, parts, middle)
            .map_err(|_| SpatialRemapError::StructuralInconsistency)?;
        if index < range.start {
            high = middle;
        } else if index >= range.end {
            low = middle + 1;
        } else {
            return Ok(middle);
        }
    }
    Err(SpatialRemapError::StructuralInconsistency)
}

fn receive_layout(
    counts: &[Count],
    expected_total: usize,
) -> Result<ReceiveLayout, SpatialRemapError> {
    let mut displacements = filled_vec(counts.len(), 0 as Count, "request receive displacements")?;
    let mut total = 0usize;
    for (peer, count) in counts.iter().copied().enumerate() {
        if count < 0 {
            return Err(SpatialRemapError::MpiCountOverflow);
        }
        displacements[peer] =
            Count::try_from(total).map_err(|_| SpatialRemapError::MpiCountOverflow)?;
        checked_count_end(count, displacements[peer])?;
        total = total
            .checked_add(usize::try_from(count).map_err(|_| SpatialRemapError::MpiCountOverflow)?)
            .ok_or(SpatialRemapError::SizeOverflow)?;
    }
    Count::try_from(total).map_err(|_| SpatialRemapError::MpiCountOverflow)?;
    validate_partition_layout(counts, &displacements, total)?;
    if total != expected_total {
        return Err(SpatialRemapError::StructuralInconsistency);
    }
    Ok(ReceiveLayout {
        displacements,
        total,
    })
}

fn allocate_request_buffers(
    routes: &OwnerRoutes,
    receive_len: usize,
) -> Result<RequestBuffers, SpatialRemapError> {
    let send_physical_len = routes.entries.len().max(1);
    let receive_physical_len = receive_len.max(1);
    let mut send_indices = filled_vec(send_physical_len, 0u64, "request send buffer")?;
    let mut destination_offsets =
        filled_vec(routes.entries.len(), 0usize, "destination request indices")?;
    for (index, entry) in routes.entries.iter().enumerate() {
        send_indices[index] = entry.global_index;
        destination_offsets[index] = entry.destination_offset;
    }
    let receive_indices = filled_vec(receive_physical_len, 0u64, "request receive buffer")?;
    Ok(RequestBuffers {
        send_indices,
        destination_offsets,
        receive_indices,
    })
}

fn incoming_source_offsets<const N: usize, const M: usize>(
    source: &Pencil<N, M>,
    requests: &[u64],
) -> Result<Vec<usize>, SpatialRemapError> {
    let mut offsets = filled_vec(requests.len(), 0usize, "incoming source indices")?;
    let rank = source.topology().rank();
    let local_rank =
        usize::try_from(rank).map_err(|_| SpatialRemapError::StructuralInconsistency)?;
    for (slot, global_index) in requests.iter().copied().enumerate() {
        let global = global_coordinate(source, global_index)
            .ok_or(SpatialRemapError::WrongOwner { rank, global_index })?;
        if owner_rank(source, global)? != local_rank {
            return Err(SpatialRemapError::WrongOwner { rank, global_index });
        }
        offsets[slot] = local_source_offset(source, global)
            .ok_or(SpatialRemapError::WrongOwner { rank, global_index })?;
    }
    if offsets.len() != source.local_len() {
        return Err(SpatialRemapError::StructuralInconsistency);
    }
    Ok(offsets)
}

fn global_coordinate<const N: usize, const M: usize>(
    pencil: &Pencil<N, M>,
    global_index: u64,
) -> Option<[usize; N]> {
    let mut remainder = usize::try_from(global_index).ok()?;
    if remainder >= pencil.global_len() {
        return None;
    }
    let mut global = [0usize; N];
    for axis in (0..N).rev() {
        let extent = pencil.global_shape()[axis];
        if extent == 0 {
            return None;
        }
        global[axis] = remainder % extent;
        remainder /= extent;
    }
    (remainder == 0).then_some(global)
}

fn local_source_offset<const N: usize, const M: usize>(
    pencil: &Pencil<N, M>,
    global: [usize; N],
) -> Option<usize> {
    let mut local = [0usize; N];
    for axis in 0..N {
        let range = &pencil.local_ranges()[axis];
        if !range.contains(&global[axis]) {
            return None;
        }
        local[axis] = global[axis] - range.start;
    }
    let mut memory = [0usize; N];
    for (position, axis) in pencil.permutation().axes().iter().enumerate() {
        memory[position] = local[axis.index()];
    }
    let shape = pencil.local_shape_memory();
    row_major_offset(&shape[..], &memory[..]).ok()
}

fn payload_layout<T>(
    routes: &OwnerRoutes,
    request_receive_counts: &[Count],
    request_receive_displacements: &[Count],
    source_spatial_len: usize,
    destination_spatial_len: usize,
    extra_count: usize,
    communicator_size: usize,
) -> Result<PayloadLayout, SpatialRemapError> {
    if routes.counts.len() != communicator_size
        || routes.displacements.len() != communicator_size
        || request_receive_counts.len() != communicator_size
        || request_receive_displacements.len() != communicator_size
        || usize::try_from(routes.total_count).ok() != Some(routes.entries.len())
        || size_of::<T>() == 0
    {
        return Err(if size_of::<T>() == 0 {
            SpatialRemapError::UnsupportedDatatype
        } else {
            SpatialRemapError::StructuralInconsistency
        });
    }
    validate_partition_layout(&routes.counts, &routes.displacements, routes.entries.len())?;
    let source_request_total = validate_partition_layout(
        request_receive_counts,
        request_receive_displacements,
        source_spatial_len,
    )?;
    if source_request_total != source_spatial_len {
        return Err(SpatialRemapError::StructuralInconsistency);
    }

    let send_len = source_spatial_len
        .checked_mul(extra_count)
        .ok_or(SpatialRemapError::SizeOverflow)?;
    let receive_len = destination_spatial_len
        .checked_mul(extra_count)
        .ok_or(SpatialRemapError::SizeOverflow)?;
    send_len
        .checked_mul(size_of::<T>())
        .ok_or(SpatialRemapError::SizeOverflow)?;
    receive_len
        .checked_mul(size_of::<T>())
        .ok_or(SpatialRemapError::SizeOverflow)?;
    checked_product(&[source_spatial_len, extra_count, size_of::<T>()])
        .map_err(|_| SpatialRemapError::SizeOverflow)?;
    checked_product(&[destination_spatial_len, extra_count, size_of::<T>()])
        .map_err(|_| SpatialRemapError::SizeOverflow)?;

    let mut send_counts = filled_vec(communicator_size, 0 as Count, "payload send counts")?;
    let mut send_displacements =
        filled_vec(communicator_size, 0 as Count, "payload send displacements")?;
    let mut receive_counts = filled_vec(communicator_size, 0 as Count, "payload receive counts")?;
    let mut receive_displacements = filled_vec(
        communicator_size,
        0 as Count,
        "payload receive displacements",
    )?;
    for peer in 0..communicator_size {
        let incoming = usize::try_from(request_receive_counts[peer])
            .map_err(|_| SpatialRemapError::MpiCountOverflow)?;
        let incoming_displacement = usize::try_from(request_receive_displacements[peer])
            .map_err(|_| SpatialRemapError::MpiCountOverflow)?;
        let outgoing = usize::try_from(routes.counts[peer])
            .map_err(|_| SpatialRemapError::MpiCountOverflow)?;
        let outgoing_displacement = usize::try_from(routes.displacements[peer])
            .map_err(|_| SpatialRemapError::MpiCountOverflow)?;

        let send_elements = incoming
            .checked_mul(extra_count)
            .ok_or(SpatialRemapError::SizeOverflow)?;
        let send_offset = incoming_displacement
            .checked_mul(extra_count)
            .ok_or(SpatialRemapError::SizeOverflow)?;
        send_counts[peer] =
            Count::try_from(send_elements).map_err(|_| SpatialRemapError::MpiCountOverflow)?;
        send_displacements[peer] =
            Count::try_from(send_offset).map_err(|_| SpatialRemapError::MpiCountOverflow)?;
        checked_count_end(send_counts[peer], send_displacements[peer])?;

        let receive_elements = outgoing
            .checked_mul(extra_count)
            .ok_or(SpatialRemapError::SizeOverflow)?;
        let receive_offset = outgoing_displacement
            .checked_mul(extra_count)
            .ok_or(SpatialRemapError::SizeOverflow)?;
        receive_counts[peer] =
            Count::try_from(receive_elements).map_err(|_| SpatialRemapError::MpiCountOverflow)?;
        receive_displacements[peer] =
            Count::try_from(receive_offset).map_err(|_| SpatialRemapError::MpiCountOverflow)?;
        checked_count_end(receive_counts[peer], receive_displacements[peer])?;
    }
    validate_partition_layout(&send_counts, &send_displacements, send_len)?;
    validate_partition_layout(&receive_counts, &receive_displacements, receive_len)?;
    let send_count = Count::try_from(send_len).map_err(|_| SpatialRemapError::MpiCountOverflow)?;
    let receive_count =
        Count::try_from(receive_len).map_err(|_| SpatialRemapError::MpiCountOverflow)?;
    Ok(PayloadLayout {
        send_counts,
        send_displacements,
        receive_counts,
        receive_displacements,
        send_len,
        receive_len,
        send_count,
        receive_count,
    })
}

fn checked_count_end(count: Count, displacement: Count) -> Result<(), SpatialRemapError> {
    if count < 0 || displacement < 0 || count.checked_add(displacement).is_none() {
        return Err(SpatialRemapError::MpiCountOverflow);
    }
    Ok(())
}

fn validate_partition_layout(
    counts: &[Count],
    displacements: &[Count],
    logical_len: usize,
) -> Result<usize, SpatialRemapError> {
    if counts.len() != displacements.len() {
        return Err(SpatialRemapError::StructuralInconsistency);
    }
    let mut next = 0usize;
    for (&count, &displacement) in counts.iter().zip(displacements) {
        checked_count_end(count, displacement)?;
        let count = usize::try_from(count).map_err(|_| SpatialRemapError::MpiCountOverflow)?;
        let displacement =
            usize::try_from(displacement).map_err(|_| SpatialRemapError::MpiCountOverflow)?;
        if displacement != next {
            return Err(SpatialRemapError::StructuralInconsistency);
        }
        let end = displacement
            .checked_add(count)
            .ok_or(SpatialRemapError::SizeOverflow)?;
        if end > logical_len {
            return Err(SpatialRemapError::StructuralInconsistency);
        }
        next = end;
    }
    if next != logical_len {
        return Err(SpatialRemapError::StructuralInconsistency);
    }
    Ok(next)
}

fn allocate_payload_buffers<T: Copy>(
    send_len: usize,
    receive_len: usize,
    init: T,
) -> Result<(Vec<T>, Vec<T>), SpatialRemapError> {
    let send = filled_vec(send_len.max(1), init, "payload send buffer")?;
    let receive = filled_vec(receive_len.max(1), init, "payload receive buffer")?;
    Ok((send, receive))
}

fn duplicate_checked_datatype<T, const M: usize>(
    topology: &MpiTopology<M>,
) -> Result<UserDatatype, SpatialRemapError>
where
    T: Equivalence,
{
    let output = T::equivalent_datatype();
    // Convert even user datatypes to the standard borrowed handle first. Do
    // not call `output.dup()`: its associated return type is user-controlled.
    let borrowed = unsafe { DatatypeRef::from_raw(output.as_raw()) };
    let mut duplicated = MaybeUninit::uninit();
    // SAFETY: `borrowed` is valid for the duration of this call and the output
    // remains uninitialized unless MPI reports success. On failure, abort before
    // reading it, as MPI provides no recoverable initialized handle.
    let status = unsafe { mpi::ffi::MPI_Type_dup(borrowed.as_raw(), duplicated.as_mut_ptr()) };
    if status != 0 {
        topology.communicator().abort(status);
    }
    // SAFETY: MPI_Type_dup returned success, which initializes a valid owned
    // datatype handle. UserDatatype's standard Drop owns MPI_Type_free.
    let datatype = unsafe { UserDatatype::from_raw(duplicated.assume_init()) };

    let mut lower_bound = MaybeUninit::uninit();
    let mut extent = MaybeUninit::uninit();
    // SAFETY: the duplicated committed datatype is valid; output parameters are
    // read only after MPI reports success.
    let status = unsafe {
        mpi::ffi::MPI_Type_get_extent(
            datatype.as_raw(),
            lower_bound.as_mut_ptr(),
            extent.as_mut_ptr(),
        )
    };
    if status != 0 {
        topology.communicator().abort(status);
    }
    // SAFETY: successful MPI_Type_get_extent initialized both output values.
    let _lower_bound = unsafe { lower_bound.assume_init() };
    let extent = unsafe { extent.assume_init() };

    let mut true_lower_bound = MaybeUninit::uninit();
    let mut true_extent = MaybeUninit::uninit();
    // SAFETY: same valid duplicated handle; outputs are untouched on failure.
    let status = unsafe {
        mpi::ffi::MPI_Type_get_true_extent(
            datatype.as_raw(),
            true_lower_bound.as_mut_ptr(),
            true_extent.as_mut_ptr(),
        )
    };
    if status != 0 {
        topology.communicator().abort(status);
    }
    // SAFETY: successful MPI_Type_get_true_extent initialized both outputs.
    let true_lower_bound = unsafe { true_lower_bound.assume_init() };
    let true_extent = unsafe { true_extent.assume_init() };

    let rust_size =
        i128::try_from(size_of::<T>()).map_err(|_| SpatialRemapError::UnsupportedDatatype)?;
    let extent = i128::try_from(extent).map_err(|_| SpatialRemapError::UnsupportedDatatype)?;
    let true_lower_bound =
        i128::try_from(true_lower_bound).map_err(|_| SpatialRemapError::UnsupportedDatatype)?;
    let true_extent =
        i128::try_from(true_extent).map_err(|_| SpatialRemapError::UnsupportedDatatype)?;
    let true_end = true_lower_bound
        .checked_add(true_extent)
        .ok_or(SpatialRemapError::UnsupportedDatatype)?;
    if rust_size <= 0
        || extent != rust_size
        || true_lower_bound < 0
        || true_extent < 0
        || true_end > rust_size
    {
        return Err(SpatialRemapError::UnsupportedDatatype);
    }
    Ok(datatype)
}

fn validate_views<T, const N: usize, const M: usize>(
    plan: &SpatialRemapPlan<T, N, M>,
    source: &crate::PencilArrayView<'_, T, N, M>,
    destination: &crate::PencilArrayViewMut<'_, T, N, M>,
) -> Result<(), SpatialRemapError>
where
    T: Equivalence + Copy,
{
    if source.extra_shape() != &plan.extra_shape
        || destination.extra_shape() != &plan.extra_shape
        || source.extra_shape() != destination.extra_shape()
    {
        return Err(SpatialRemapError::DifferentExtraShape);
    }
    if !source.pencil().same_layout(plan.source.as_ref())
        || !destination.pencil().same_layout(plan.destination.as_ref())
        || source.len() != plan.send_len
        || destination.len() != plan.receive_len
    {
        return Err(SpatialRemapError::LayoutMismatch);
    }
    Ok(())
}

fn pack_payload<T: Copy, const N: usize, const M: usize>(
    plan: &mut SpatialRemapPlan<T, N, M>,
    source: &[T],
) {
    let extra_count = plan.extra_shape.element_count();
    for peer in 0..plan.request_receive_counts.len() {
        let request_start = usize::try_from(plan.request_receive_displacements[peer])
            .expect("prepared request displacement is nonnegative");
        let request_count = usize::try_from(plan.request_receive_counts[peer])
            .expect("prepared request count is nonnegative");
        let payload_start = usize::try_from(plan.payload_send_displacements[peer])
            .expect("prepared payload displacement is nonnegative");
        for extra in 0..extra_count {
            let source_extra = extra * plan.source.local_len();
            let payload_extra = payload_start + extra * request_count;
            for index in 0..request_count {
                plan.send_buffer[payload_extra + index] =
                    source[source_extra + plan.source_offsets[request_start + index]];
            }
        }
    }
}

fn unpack_payload<T: Copy, const N: usize, const M: usize>(
    plan: &SpatialRemapPlan<T, N, M>,
    destination: &mut [T],
) {
    let extra_count = plan.extra_shape.element_count();
    for peer in 0..plan.request_send_counts.len() {
        let request_start = usize::try_from(plan.request_send_displacements[peer])
            .expect("prepared request displacement is nonnegative");
        let request_count = usize::try_from(plan.request_send_counts[peer])
            .expect("prepared request count is nonnegative");
        let payload_start = usize::try_from(plan.payload_receive_displacements[peer])
            .expect("prepared payload displacement is nonnegative");
        for extra in 0..extra_count {
            let destination_extra = extra * plan.destination.local_len();
            let payload_extra = payload_start + extra * request_count;
            for index in 0..request_count {
                destination[destination_extra + plan.destination_offsets[request_start + index]] =
                    plan.receive_buffer[payload_extra + index];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        SpatialRemapError, SpatialRemapPlan, affine_coordinate, checked_count_end,
        local_coordinate, partition_owner, validate_partition_layout,
    };
    use mpi::Count;

    #[test]
    fn affine_mapping_handles_full_width_offsets_and_signs() {
        let shape = [5usize, 7];
        assert_eq!(
            affine_coordinate([0, 6], shape, &[1, -1], &[i128::MAX, i128::MIN]).unwrap(),
            [2, 6],
        );
        assert_eq!(
            affine_coordinate([4, 3], shape, &[-1, 1], &[i128::MIN, i128::MAX]).unwrap(),
            [3, 4],
        );
    }

    #[test]
    fn floor_partition_owner_handles_empty_partitions() {
        assert_eq!(partition_owner(0, 2, 4).unwrap(), 1);
        assert_eq!(partition_owner(1, 2, 4).unwrap(), 3);
        assert_eq!(partition_owner(8, 10, 3).unwrap(), 2);
        assert!(matches!(
            partition_owner(2, 2, 3),
            Err(SpatialRemapError::StructuralInconsistency),
        ));
    }

    #[test]
    fn count_layout_checks_packed_nonoverlapping_segments() {
        assert_eq!(
            validate_partition_layout(&[2, 0, 3], &[0, 2, 2], 5).unwrap(),
            5
        );
        assert!(matches!(
            validate_partition_layout(&[2, 2], &[0, 1], 4),
            Err(SpatialRemapError::StructuralInconsistency),
        ));
        assert_eq!(
            validate_partition_layout(&[Count::MAX], &[0], usize::try_from(Count::MAX).unwrap())
                .unwrap(),
            usize::try_from(Count::MAX).unwrap(),
        );
        assert!(matches!(
            checked_count_end(Count::MAX, 1),
            Err(SpatialRemapError::MpiCountOverflow),
        ));
    }

    #[test]
    #[ignore = "MPI contract test; run under mpiexec with --ignored"]
    fn mpi_preflight_failure_preserves_all_plan_state() {
        use std::sync::Arc;

        use mpi::{raw::traits::AsRaw, traits::*};

        use crate::{
            AxisPermutation, ExtraShape, MpiTopology, Pencil, PencilArrayView, PencilArrayViewMut,
        };

        let universe = mpi::initialize().expect("MPI must not already be initialized");
        let world = universe.world();
        let size = usize::try_from(world.size()).unwrap();
        let topology = MpiTopology::<1>::new(&world, [size]).unwrap();
        let source = Pencil::<2, 1>::new(Arc::clone(&topology), [7, 5], [0]).unwrap();
        let destination = Pencil::<2, 1>::new_permuted(
            Arc::clone(&topology),
            [7, 5],
            [0],
            AxisPermutation::new([1, 0]).unwrap(),
        )
        .unwrap();
        let extra = ExtraShape::new([2, 3]).unwrap();
        let signs = [1, -1];
        let offsets = [2, -3];
        let mut plan = SpatialRemapPlan::<i64, 2, 1>::prepare(
            Arc::clone(&source),
            Arc::clone(&destination),
            extra.clone(),
            signs,
            offsets,
            0,
        )
        .unwrap();
        let mut source_data = Vec::with_capacity(source.local_len() * extra.element_count());
        for extra_index in 0..extra.element_count() {
            for memory_index in 0..source.local_len() {
                let local = local_coordinate(&source, memory_index).unwrap();
                let global = [
                    source.local_ranges()[0].start + local[0],
                    source.local_ranges()[1].start + local[1],
                ];
                let global_linear = global[0] * source.global_shape()[1] + global[1];
                source_data.push(
                    i64::try_from(extra_index * source.global_len() + global_linear + 1).unwrap(),
                );
            }
        }
        let mut destination_data = vec![91i64; destination.local_len() * extra.element_count()];
        let wrong_destination = Pencil::<2, 1>::new(Arc::clone(&topology), [7, 5], [0]).unwrap();
        let verify_oracle = |actual: &[i64]| {
            for extra_index in 0..extra.element_count() {
                for memory_index in 0..destination.local_len() {
                    let local = local_coordinate(&destination, memory_index).unwrap();
                    let destination_global = [
                        destination.local_ranges()[0].start + local[0],
                        destination.local_ranges()[1].start + local[1],
                    ];
                    let source_global = affine_coordinate(
                        destination_global,
                        *source.global_shape(),
                        &signs,
                        &offsets,
                    )
                    .unwrap();
                    let source_linear =
                        source_global[0] * source.global_shape()[1] + source_global[1];
                    let expected =
                        i64::try_from(extra_index * source.global_len() + source_linear + 1)
                            .unwrap();
                    assert_eq!(
                        actual[extra_index * destination.local_len() + memory_index],
                        expected,
                    );
                }
            }
        };

        assert!(source_data.iter().all(|value| *value != 0));
        let source_view = PencilArrayView::from_slice(&source, &extra, &source_data).unwrap();
        let destination_view =
            PencilArrayViewMut::from_slice_mut(&destination, &extra, &mut destination_data)
                .unwrap();
        plan.apply_views(source_view, destination_view).unwrap();
        verify_oracle(&destination_data);
        assert!(plan.send_buffer.iter().any(|value| *value != 0));
        assert!(plan.receive_buffer.iter().any(|value| *value != 0));

        let send_before = plan.send_buffer.clone();
        let receive_before = plan.receive_buffer.clone();
        let descriptor_before = plan.descriptor.clone();
        let route_state_before = (
            plan.request_send_counts.clone(),
            plan.request_send_displacements.clone(),
            plan.request_receive_counts.clone(),
            plan.request_receive_displacements.clone(),
            plan.destination_offsets.clone(),
            plan.source_offsets.clone(),
            plan.payload_send_counts.clone(),
            plan.payload_send_displacements.clone(),
            plan.payload_receive_counts.clone(),
            plan.payload_receive_displacements.clone(),
        );
        let metadata_before = (
            plan.extra_shape.clone(),
            plan.signs,
            plan.offsets,
            plan.send_len,
            plan.receive_len,
            plan.send_count,
            plan.receive_count,
            plan.datatype.as_raw(),
        );
        let source_before = source_data.clone();
        let destination_before = destination_data.clone();

        let source_view = PencilArrayView::from_slice(&source, &extra, &source_data).unwrap();
        let destination_view = if topology.rank() == 0 {
            PencilArrayViewMut::from_slice_mut(&wrong_destination, &extra, &mut destination_data)
                .unwrap()
        } else {
            PencilArrayViewMut::from_slice_mut(&destination, &extra, &mut destination_data).unwrap()
        };
        let error = plan.apply_views(source_view, destination_view).unwrap_err();
        if topology.rank() == 0 {
            assert!(matches!(error, SpatialRemapError::LayoutMismatch));
        } else {
            assert!(matches!(
                error,
                SpatialRemapError::CollectivePreconditionFailed
            ));
        }
        assert_eq!(plan.send_buffer, send_before);
        assert_eq!(plan.receive_buffer, receive_before);
        assert_eq!(plan.descriptor, descriptor_before);
        assert_eq!(
            (
                plan.request_send_counts.clone(),
                plan.request_send_displacements.clone(),
                plan.request_receive_counts.clone(),
                plan.request_receive_displacements.clone(),
                plan.destination_offsets.clone(),
                plan.source_offsets.clone(),
                plan.payload_send_counts.clone(),
                plan.payload_send_displacements.clone(),
                plan.payload_receive_counts.clone(),
                plan.payload_receive_displacements.clone(),
            ),
            route_state_before,
        );
        assert_eq!(
            (
                plan.extra_shape.clone(),
                plan.signs,
                plan.offsets,
                plan.send_len,
                plan.receive_len,
                plan.send_count,
                plan.receive_count,
                plan.datatype.as_raw(),
            ),
            metadata_before,
        );
        assert!(Arc::ptr_eq(&plan.source, &source));
        assert!(Arc::ptr_eq(&plan.destination, &destination));
        assert_eq!(source_data, source_before);
        assert_eq!(destination_data, destination_before);

        let source_view = PencilArrayView::from_slice(&source, &extra, &source_data).unwrap();
        let destination_view =
            PencilArrayViewMut::from_slice_mut(&destination, &extra, &mut destination_data)
                .unwrap();
        plan.apply_views(source_view, destination_view).unwrap();
        verify_oracle(&destination_data);
    }
}
