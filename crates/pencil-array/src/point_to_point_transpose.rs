use std::sync::Arc;

use mpi::{datatype::Equivalence, request::scope, traits::*};

use crate::{
    ExtraShape, Pencil, PencilArrayView, PencilArrayViewMut,
    transpose::{
        CommunicationMode, POINT_TO_POINT_RESERVED_TAG, TransposeError, TransposePlanCore,
        TransposeWorkspace, TransposeWorkspaceRequirements, collective_valid, pack_source,
        unpack_destination,
    },
};

/// A checked distributed transpose using nonblocking point-to-point messages.
///
/// [`Self::new`] and [`Self::execute_views`] are collective on the source
/// topology's Cartesian communicator. Every rank must use the same source
/// communicator context, API, order, `T`, and correct
/// [`Equivalence`](mpi::datatype::Equivalence) implementation. The source and
/// destination pencils must share that topology object and global shape and
/// differ in exactly one ordered decomposition position. Descriptor checks
/// catch common mismatches, but cannot prove type identity, the correctness of
/// an unsafe `Equivalence` implementation, or communicator/collective-order
/// correctness.
///
/// [`Self::workspace_requirements`] and [`TransposeWorkspace::from_vecs`] are
/// noncollective local operations. Requirements and workspace checks use
/// initialized `len`, not capacity, and all counts, displacements, totals,
/// offsets, and source/destination/workspace lengths are checked before any
/// payload request is posted. Ordinary descriptor or preflight errors leave
/// the source, destination, and workspace unchanged.
///
/// Payloads use the changed-axis subcommunicator owned by the topology and the
/// fixed internal `POINT_TO_POINT_RESERVED_TAG` (`0x5054`).
/// That context and tag are not a per-operation namespace: callers must not
/// overlap unfinished transposes on the same context. The implementation posts
/// every nonzero receive before any send, waits for every request, and returns
/// only after all requests and their borrowed segments are complete.
/// `mpi::request::scope` may abort the process if it exits with unfinished
/// requests, including while unwinding an arbitrary panic. MPI failures,
/// arbitrary panics, and process loss do not
/// guarantee that a `Result` is recovered or that storage remains unchanged.
///
/// Point-to-point in-place transpose and FFT APIs are not implemented.
#[derive(Debug)]
pub struct PointToPointTransposePlan<const N: usize, const M: usize> {
    core: TransposePlanCore<N, M>,
}

impl<const N: usize, const M: usize> PointToPointTransposePlan<N, M> {
    /// Collectively validates and constructs a point-to-point transpose plan.
    ///
    /// All ranks must call this method in the same order on the same source
    /// communicator context. It validates the same topology, global-shape, and
    /// exactly-one-decomposition-change contract as Alltoallv before returning.
    pub fn new(
        source: Arc<Pencil<N, M>>,
        destination: Arc<Pencil<N, M>>,
    ) -> Result<Self, TransposeError> {
        Ok(Self {
            core: TransposePlanCore::new(source, destination, CommunicationMode::PointToPoint)?,
        })
    }

    /// Returns the initialized workspace lengths required for `extra_shape`.
    ///
    /// This method is noncollective and does not call MPI or allocate storage.
    /// The result is local to the calling rank; use initialized workspace
    /// `len` values at least as large as the returned send and receive lengths.
    pub fn workspace_requirements(
        &self,
        extra_shape: &ExtraShape,
    ) -> Result<TransposeWorkspaceRequirements, TransposeError> {
        self.core.workspace_requirements(extra_shape)
    }

    /// Executes the transpose with all receives posted before all sends.
    ///
    /// Every rank must call this method in the same order on the same source
    /// communicator context, with the same `T` and a correct `Equivalence`
    /// implementation. The source is preserved. Ordinary descriptor or
    /// preflight errors are returned on all ranks before packing or posting any
    /// request; the source, destination, and workspace are unchanged on those
    /// paths. Workspace vectors are used only through initialized prefixes
    /// and are never resized or reallocated. Every count and displacement,
    /// including checked sums, totals, and view/workspace lengths, must fit
    /// `mpi::Count` and the buffer-prefix limits. After a request is posted,
    /// MPI failure, arbitrary panic, or process loss does not guarantee a
    /// recovered `Result` or unchanged destination. All requests are waited
    /// before this method returns; the fixed `0x5054` tag must not be used for
    /// another unfinished
    /// transpose on the same topology context.
    pub fn execute_views<T>(
        &self,
        source: PencilArrayView<'_, T, N, M>,
        mut destination: PencilArrayViewMut<'_, T, N, M>,
        workspace: &mut TransposeWorkspace<T>,
    ) -> Result<(), TransposeError>
    where
        T: Equivalence + Copy,
    {
        let communicator = self.core.source().topology().cartesian();
        crate::transpose::agree_execute_descriptor::<_, T, N, M>(
            &self.core,
            communicator,
            source.extra_shape(),
            destination.extra_shape(),
            CommunicationMode::PointToPoint.views_operation(),
        )?;

        let local_preflight = (|| {
            let prepared = self
                .core
                .prepare_execution(&source, &destination, workspace)?;
            let request_counts = self.core.request_counts(&prepared);
            Ok::<_, TransposeError>((prepared, request_counts))
        })();
        if !collective_valid(communicator, local_preflight.is_ok()) {
            return Err(local_preflight
                .err()
                .unwrap_or(TransposeError::CollectivePreconditionFailed));
        }
        let (prepared, (receive_slots, send_slots)) =
            local_preflight.expect("collective point-to-point preflight succeeded");
        let extra_count = source.extra_shape().element_count();

        scope(|scope| {
            let mut receive_requests = Vec::new();
            let mut send_requests = Vec::new();
            let receive_reserved = receive_requests.try_reserve(receive_slots).is_ok();
            let send_reserved = send_requests.try_reserve(send_slots).is_ok();
            let reserved = receive_reserved && send_reserved;
            if !collective_valid(communicator, reserved) {
                return if reserved {
                    Err(TransposeError::CollectivePreconditionFailed)
                } else {
                    Err(TransposeError::PreparationFailed)
                };
            }

            pack_source(
                self.core.peers(),
                self.core.source().as_ref(),
                source.as_slice(),
                &mut workspace.send_buffer[..prepared.requirements.send_len],
                prepared.requirements.send_len,
                extra_count,
            );

            let subcommunicator = self
                .core
                .source()
                .topology()
                .subcommunicator(self.core.changed_topology_axis());
            let mut receive_tail =
                &mut workspace.receive_buffer[..prepared.requirements.receive_len];
            for peer in self.core.peers() {
                let count = peer
                    .receive_spatial_len
                    .checked_mul(extra_count)
                    .expect("point-to-point preflight validated receive count");
                if count == 0 {
                    continue;
                }
                let (segment, rest) = receive_tail.split_at_mut(count);
                receive_tail = rest;
                let request = subcommunicator
                    .process_at_rank(peer.peer_rank)
                    .immediate_receive_into_with_tag(scope, segment, POINT_TO_POINT_RESERVED_TAG);
                receive_requests.push(request);
            }

            let send_buffer = &workspace.send_buffer[..prepared.requirements.send_len];
            let mut send_tail = send_buffer;
            for peer in self.core.peers() {
                let count = peer
                    .send_spatial_len
                    .checked_mul(extra_count)
                    .expect("point-to-point preflight validated send count");
                if count == 0 {
                    continue;
                }
                let (segment, rest) = send_tail.split_at(count);
                send_tail = rest;
                let request = subcommunicator
                    .process_at_rank(peer.peer_rank)
                    .immediate_send_with_tag(scope, segment, POINT_TO_POINT_RESERVED_TAG);
                send_requests.push(request);
            }

            for request in receive_requests {
                request.wait_without_status();
            }
            for request in send_requests {
                request.wait_without_status();
            }
            Ok(())
        })?;

        unpack_destination(
            self.core.peers(),
            self.core.destination().as_ref(),
            destination.as_mut_slice(),
            &workspace.receive_buffer[..prepared.requirements.receive_len],
            extra_count,
        );
        Ok(())
    }
}
