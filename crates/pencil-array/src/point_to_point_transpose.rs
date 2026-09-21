use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Arc,
    time::Instant,
};

#[cfg(test)]
thread_local! {
    static EVENT_TRACE: std::cell::RefCell<Vec<&'static str>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(test)]
pub(crate) fn test_trace_start() {
    EVENT_TRACE.with(|trace| trace.borrow_mut().clear());
}

#[cfg(test)]
pub(crate) fn test_trace_event(event: &'static str) {
    EVENT_TRACE.with(|trace| trace.borrow_mut().push(event));
}

#[cfg(test)]
pub(crate) fn test_trace_finish() -> Vec<&'static str> {
    EVENT_TRACE.with(|trace| std::mem::take(&mut *trace.borrow_mut()))
}

use mpi::{datatype::Equivalence, request::scope, traits::*};

use crate::{
    ExtraShape, ManyPencilArray, Pencil, PencilArrayView, PencilArrayViewMut,
    transpose::{
        CommunicationMode, OverlapError, POINT_TO_POINT_RESERVED_TAG, PreparedExchange,
        TransposeError, TransposePlanCore, TransposeTiming, TransposeWorkspace,
        TransposeWorkspaceRequirements, collective_valid, finish_in_place, pack_source,
        prepare_in_place, unpack_destination,
    },
};

/// A checked distributed transpose using nonblocking point-to-point messages.
///
/// [`Self::new`], [`Self::execute_views`], and [`Self::execute_in_place`] are
/// collective on the source topology's Cartesian communicator. Every rank must
/// use the same source communicator context, API, order, `T`, and correct
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
/// the source, destination, and workspace unchanged. `execute_views` preserves
/// its source on success and writes the destination view from the received data.
/// `execute_in_place` replaces the active layout and its data on success; it
/// writes only the destination prefix, preserving any excess registered
/// storage tail. Its ordinary errors leave the array state, active data, and
/// workspace unchanged. Exact source/destination extra-shape agreement,
/// initialized workspace lengths, checked counts and offsets, and the
/// registered in-place layouts are required before payload communication.
///
/// Payloads use the changed-axis subcommunicator owned by the topology and the
/// fixed internal `POINT_TO_POINT_RESERVED_TAG` (`0x5054`). That context and
/// tag are not a per-operation namespace: callers must not overlap unfinished
/// transposes on the same context. Before packing, the implementation reserves
/// all request slots and agrees that reservation succeeded on every Cartesian
/// rank. It then posts every nonzero receive before any send, waits for every
/// request, and returns only after all requests and their borrowed segments are
/// complete. For in-place execution, the scope ends before the array is marked
/// `Poisoned`, the destination prefix is unpacked, and the destination layout is
/// committed.
/// `mpi::request::scope` may abort the process if it exits with unfinished
/// requests, including while unwinding an arbitrary panic. MPI failures,
/// arbitrary panics, and process loss do not
/// guarantee that a `Result` is recovered or that storage remains unchanged.
///
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
    /// implementation. The source is preserved and the destination receives the
    /// transposed physical buffer on success. Ordinary descriptor or preflight
    /// errors are returned on all ranks before packing or posting any request;
    /// the source, destination, and workspace are unchanged on those paths.
    /// Workspace vectors are used only through initialized prefixes and are
    /// never resized or reallocated. Every count and displacement, including
    /// checked sums, totals, exact extra-shape agreement, and view/workspace
    /// lengths, must fit `mpi::Count` and the buffer-prefix limits. All requests
    /// are waited before this method returns; the fixed `0x5054` tag must not be
    /// used for another unfinished transpose on the same topology context.
    /// After a request is posted, MPI failure, arbitrary panic, or process loss
    /// does not guarantee a recovered `Result` or unchanged destination.
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

        let local_preflight = self
            .core
            .prepare_execution(&source, &destination, workspace);
        if !collective_valid(communicator, local_preflight.is_ok()) {
            return Err(local_preflight
                .err()
                .unwrap_or(TransposeError::CollectivePreconditionFailed));
        }
        let prepared = local_preflight.expect("collective point-to-point preflight succeeded");
        let extra_count = source.extra_shape().element_count();

        execute_point_to_point_exchange(
            &self.core,
            communicator,
            source.as_slice(),
            &prepared,
            workspace,
            extra_count,
        )?;

        unpack_destination(
            self.core.peers(),
            self.core.destination().as_ref(),
            destination.as_mut_slice(),
            &workspace.receive_buffer[..prepared.requirements.receive_len],
            extra_count,
        );
        Ok(())
    }

    /// Executes `execute_views` and returns wall-clock timings for its local phases.
    pub fn execute_views_with_timing<T>(
        &self,
        source: PencilArrayView<'_, T, N, M>,
        mut destination: PencilArrayViewMut<'_, T, N, M>,
        workspace: &mut TransposeWorkspace<T>,
    ) -> Result<TransposeTiming, TransposeError>
    where
        T: Equivalence + Copy,
    {
        let total = Instant::now();
        let communicator = self.core.source().topology().cartesian();
        crate::transpose::agree_execute_descriptor::<_, T, N, M>(
            &self.core,
            communicator,
            source.extra_shape(),
            destination.extra_shape(),
            CommunicationMode::PointToPoint.timed_views_operation(),
        )?;
        let local_preflight = self
            .core
            .prepare_execution(&source, &destination, workspace);
        if !collective_valid(communicator, local_preflight.is_ok()) {
            return Err(local_preflight
                .err()
                .unwrap_or(TransposeError::CollectivePreconditionFailed));
        }
        let prepared = local_preflight.expect("collective point-to-point preflight succeeded");
        let extra_count = source.extra_shape().element_count();
        let mut timing = TransposeTiming::default();
        execute_point_to_point_exchange_timed(
            &self.core,
            communicator,
            source.as_slice(),
            &prepared,
            workspace,
            extra_count,
            &mut timing,
        )?;
        let unpack = Instant::now();
        unpack_destination(
            self.core.peers(),
            self.core.destination().as_ref(),
            destination.as_mut_slice(),
            &workspace.receive_buffer[..prepared.requirements.receive_len],
            extra_count,
        );
        timing.unpack = unpack.elapsed();
        timing.total = total.elapsed();
        Ok(timing)
    }

    /// Runs a local callback after receive completion and unpacking, before send waits.
    /// Callback errors are agreed across the Cartesian communicator. Panics drain
    /// sends and are agreed before the origin resumes unwinding and peers return
    /// `OverlapError::PeerPanicked`. The callback must not call MPI.
    pub fn execute_views_with_callback<T, F, E>(
        &self,
        source: PencilArrayView<'_, T, N, M>,
        mut destination: PencilArrayViewMut<'_, T, N, M>,
        workspace: &mut TransposeWorkspace<T>,
        callback: F,
    ) -> Result<(), OverlapError<E>>
    where
        T: Equivalence + Copy,
        F: FnOnce(&mut [T]) -> Result<(), E>,
        E: std::fmt::Debug,
    {
        let communicator = self.core.source().topology().cartesian();
        crate::transpose::agree_execute_descriptor::<_, T, N, M>(
            &self.core,
            communicator,
            source.extra_shape(),
            destination.extra_shape(),
            CommunicationMode::PointToPoint.callback_operation(),
        )
        .map_err(OverlapError::Transpose)?;
        let local = self
            .core
            .prepare_execution(&source, &destination, workspace);
        if !collective_valid(communicator, local.is_ok()) {
            return Err(local.err().map_or(
                OverlapError::CollectivePreconditionFailed,
                OverlapError::Transpose,
            ));
        }
        let prepared = local.expect("collective point-to-point preflight succeeded");
        let extra_count = source.extra_shape().element_count();
        execute_point_to_point_exchange_callback(
            &self.core,
            communicator,
            &prepared,
            workspace,
            extra_count,
            |send_buffer| {
                pack_source(
                    self.core.peers(),
                    self.core.source().as_ref(),
                    source.as_slice(),
                    send_buffer,
                    prepared.requirements.send_len,
                    extra_count,
                );
            },
            |receive_buffer| {
                unpack_destination(
                    self.core.peers(),
                    self.core.destination().as_ref(),
                    destination.as_mut_slice(),
                    receive_buffer,
                    extra_count,
                );
                #[cfg(test)]
                crate::point_to_point_transpose::test_trace_event("unpack");
                callback(destination.as_mut_slice())
            },
        )
    }

    /// Executes the in-place transpose, then runs a callback on the destination.
    ///
    /// The array is poisoned before the destination is unpacked. It is committed
    /// only when the callback succeeds on every rank; callback errors leave it
    /// poisoned and peers report `PeerCallbackFailed`. A callback panic drains
    /// sends before the origin resumes unwinding.
    pub fn execute_in_place_with_callback<T, F, E>(
        &self,
        array: &mut ManyPencilArray<T, N, M>,
        workspace: &mut TransposeWorkspace<T>,
        callback: F,
    ) -> Result<(), OverlapError<E>>
    where
        T: Equivalence + Copy,
        F: FnOnce(&mut [T]) -> Result<(), E>,
        E: std::fmt::Debug,
    {
        let communicator = self.core.source().topology().cartesian();
        crate::transpose::agree_execute_descriptor::<_, T, N, M>(
            &self.core,
            communicator,
            array.extra_shape(),
            array.extra_shape(),
            CommunicationMode::PointToPoint.callback_in_place_operation(),
        )
        .map_err(OverlapError::Transpose)?;
        let local = prepare_in_place(&self.core, array, workspace);
        if !collective_valid(communicator, local.is_ok()) {
            return Err(local.err().map_or(
                OverlapError::CollectivePreconditionFailed,
                OverlapError::Transpose,
            ));
        }
        let prepared = local.expect("collective in-place preflight succeeded");
        let extra_count = array.extra_shape().element_count();
        // The transport reserves and agrees before invoking pack. Delay poisoning
        // until then so reservation failure is still an atomic preflight failure.
        let guard_cell = std::cell::RefCell::new(None);
        let result = execute_point_to_point_exchange_callback(
            &self.core,
            communicator,
            &prepared.exchange,
            workspace,
            extra_count,
            |send_buffer| {
                let mut guard = array
                    .begin_in_place_write()
                    .expect("in-place callback preflight validated active state");
                pack_source(
                    self.core.peers(),
                    self.core.source().as_ref(),
                    guard.storage_mut(),
                    send_buffer,
                    prepared.exchange.requirements.send_len,
                    extra_count,
                );
                *guard_cell.borrow_mut() = Some(guard);
            },
            |receive_buffer| {
                let mut guard = guard_cell.borrow_mut();
                let destination = &mut guard.as_mut().expect("packing created guard").storage_mut()
                    [..prepared.exchange.requirements.receive_len];
                unpack_destination(
                    self.core.peers(),
                    self.core.destination().as_ref(),
                    destination,
                    receive_buffer,
                    extra_count,
                );
                #[cfg(test)]
                crate::point_to_point_transpose::test_trace_event("unpack");
                callback(destination)
            },
        );
        if result.is_ok() {
            guard_cell
                .into_inner()
                .expect("successful callback created guard")
                .commit(prepared.destination_index)
                .map_err(|error| OverlapError::Transpose(error.into()))?;
        }
        result
    }

    /// Executes the transpose by replacing the active layout in shared storage.
    ///
    /// Every rank must call this method in the same order on the same source
    /// communicator context with the same `T` and a correct `Equivalence`
    /// implementation. The active layout must match the plan's source, the
    /// destination must be registered in the array, and the extra shape must be
    /// identical on every rank. Descriptor and ordinary preflight errors are
    /// agreed before packing or posting any request, leaving the array state,
    /// active data, and workspace unchanged. On success, the source remains
    /// valid through packing and communication; after the request scope has
    /// ended, the array is marked `Poisoned`, the destination prefix is
    /// unpacked, and the destination layout is committed. Any unused storage
    /// tail is preserved. MPI failures, arbitrary panics, and process loss do
    /// not guarantee a recovered `Result` or unchanged storage.
    pub fn execute_in_place<T>(
        &self,
        array: &mut ManyPencilArray<T, N, M>,
        workspace: &mut TransposeWorkspace<T>,
    ) -> Result<(), TransposeError>
    where
        T: Equivalence + Copy,
    {
        let communicator = self.core.source().topology().cartesian();
        crate::transpose::agree_execute_descriptor::<_, T, N, M>(
            &self.core,
            communicator,
            array.extra_shape(),
            array.extra_shape(),
            CommunicationMode::PointToPoint.in_place_operation(),
        )?;

        let local_preflight = prepare_in_place(&self.core, array, workspace);
        if !collective_valid(communicator, local_preflight.is_ok()) {
            return Err(local_preflight
                .err()
                .unwrap_or(TransposeError::CollectivePreconditionFailed));
        }
        let prepared = local_preflight.expect("collective in-place preflight succeeded");
        let extra_count = array.extra_shape().element_count();
        {
            let source = array
                .active_view()
                .expect("collective in-place preflight validated the active source");
            execute_point_to_point_exchange(
                &self.core,
                communicator,
                source.as_slice(),
                &prepared.exchange,
                workspace,
                extra_count,
            )?;
        }
        finish_in_place(
            &self.core,
            array,
            prepared.destination_index,
            &prepared.exchange,
            &workspace.receive_buffer[..prepared.exchange.requirements.receive_len],
            extra_count,
        );
        Ok(())
    }

    /// Executes the in-place transpose and reports local phase timings.
    pub fn execute_in_place_with_timing<T>(
        &self,
        array: &mut ManyPencilArray<T, N, M>,
        workspace: &mut TransposeWorkspace<T>,
    ) -> Result<TransposeTiming, TransposeError>
    where
        T: Equivalence + Copy,
    {
        let total = Instant::now();
        let communicator = self.core.source().topology().cartesian();
        crate::transpose::agree_execute_descriptor::<_, T, N, M>(
            &self.core,
            communicator,
            array.extra_shape(),
            array.extra_shape(),
            CommunicationMode::PointToPoint.timed_in_place_operation(),
        )?;
        let local = prepare_in_place(&self.core, array, workspace);
        if !collective_valid(communicator, local.is_ok()) {
            return Err(local
                .err()
                .unwrap_or(TransposeError::CollectivePreconditionFailed));
        }
        let prepared = local.expect("collective in-place preflight succeeded");
        let extra_count = array.extra_shape().element_count();
        let mut timing = TransposeTiming::default();
        {
            let source = array.active_view().expect("validated active source");
            execute_point_to_point_exchange_timed(
                &self.core,
                communicator,
                source.as_slice(),
                &prepared.exchange,
                workspace,
                extra_count,
                &mut timing,
            )?;
        }
        let unpack = Instant::now();
        finish_in_place(
            &self.core,
            array,
            prepared.destination_index,
            &prepared.exchange,
            &workspace.receive_buffer[..prepared.exchange.requirements.receive_len],
            extra_count,
        );
        timing.unpack = unpack.elapsed();
        timing.total = total.elapsed();
        Ok(timing)
    }
}

fn execute_point_to_point_exchange<C, T, const N: usize, const M: usize>(
    plan: &TransposePlanCore<N, M>,
    communicator: &C,
    source_storage: &[T],
    prepared: &PreparedExchange,
    workspace: &mut TransposeWorkspace<T>,
    extra_count: usize,
) -> Result<(), TransposeError>
where
    C: CommunicatorCollectives,
    T: Equivalence + Copy,
{
    execute_point_to_point_exchange_timed(
        plan,
        communicator,
        source_storage,
        prepared,
        workspace,
        extra_count,
        &mut TransposeTiming::default(),
    )
}

fn execute_point_to_point_exchange_callback<C, T, P, F, E, const N: usize, const M: usize>(
    plan: &TransposePlanCore<N, M>,
    communicator: &C,
    prepared: &PreparedExchange,
    workspace: &mut TransposeWorkspace<T>,
    extra_count: usize,
    pack: P,
    callback: F,
) -> Result<(), OverlapError<E>>
where
    C: CommunicatorCollectives,
    T: Equivalence + Copy,
    P: FnOnce(&mut [T]),
    F: FnOnce(&mut [T]) -> Result<(), E>,
    E: std::fmt::Debug,
{
    let (receive_slots, send_slots) = plan.request_counts(prepared);
    let send_buffer = &mut workspace.send_buffer[..prepared.requirements.send_len];
    let receive_buffer = &mut workspace.receive_buffer[..prepared.requirements.receive_len];
    let sub = plan
        .source()
        .topology()
        .subcommunicator(plan.changed_topology_axis());

    let callback_result = scope(|send_scope| {
        let mut sends = Vec::new();
        // Reserve the actual request vectors before packing or posting.
        scope(|receive_scope| {
            let mut receives = Vec::new();
            let receive_reserved = receives.try_reserve(receive_slots).is_ok();
            let send_reserved = sends.try_reserve(send_slots).is_ok();
            if !collective_valid(communicator, receive_reserved && send_reserved) {
                return Err(OverlapError::CollectivePreconditionFailed);
            }
            pack(send_buffer);
            let mut receive_tail = &mut receive_buffer[..];
            for peer in plan.peers() {
                let count = peer
                    .receive_spatial_len
                    .checked_mul(extra_count)
                    .expect("validated count");
                if count == 0 {
                    continue;
                }
                let (segment, rest) = receive_tail.split_at_mut(count);
                receive_tail = rest;
                receives.push(
                    sub.process_at_rank(peer.peer_rank)
                        .immediate_receive_into_with_tag(
                            receive_scope,
                            segment,
                            POINT_TO_POINT_RESERVED_TAG,
                        ),
                );
            }
            let mut send_tail = &send_buffer[..];
            for peer in plan.peers() {
                let count = peer
                    .send_spatial_len
                    .checked_mul(extra_count)
                    .expect("validated count");
                if count == 0 {
                    continue;
                }
                let (segment, rest) = send_tail.split_at(count);
                send_tail = rest;
                sends.push(sub.process_at_rank(peer.peer_rank).immediate_send_with_tag(
                    send_scope,
                    segment,
                    POINT_TO_POINT_RESERVED_TAG,
                ));
            }
            for request in receives {
                request.wait_without_status();
            }
            Ok(())
        })?;
        let result = catch_unwind(AssertUnwindSafe(|| callback(receive_buffer)));
        #[cfg(test)]
        test_trace_event("send_wait");
        for request in sends {
            request.wait_without_status();
        }
        Ok::<_, OverlapError<E>>(result)
    })?;
    let local_status: i32 = match callback_result {
        Ok(Ok(())) => 0,
        Ok(Err(_)) => 1,
        Err(_) => 2,
    };
    let mut status = 0i32;
    communicator.all_reduce_into(
        &local_status,
        &mut status,
        mpi::collective::SystemOperation::max(),
    );
    match callback_result {
        Ok(Ok(())) if status == 0 => Ok(()),
        Ok(Err(error)) if status == 1 => Err(OverlapError::Callback(error)),
        Err(payload) => std::panic::resume_unwind(payload),
        _ if status == 2 => Err(OverlapError::PeerPanicked),
        _ => Err(OverlapError::PeerCallbackFailed),
    }
}

fn execute_point_to_point_exchange_timed<C, T, const N: usize, const M: usize>(
    plan: &TransposePlanCore<N, M>,
    communicator: &C,
    source_storage: &[T],
    prepared: &PreparedExchange,
    workspace: &mut TransposeWorkspace<T>,
    extra_count: usize,
    timing: &mut TransposeTiming,
) -> Result<(), TransposeError>
where
    C: CommunicatorCollectives,
    T: Equivalence + Copy,
{
    let (receive_slots, send_slots) = plan.request_counts(prepared);
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

        let started = Instant::now();
        pack_source(
            plan.peers(),
            plan.source().as_ref(),
            source_storage,
            &mut workspace.send_buffer[..prepared.requirements.send_len],
            prepared.requirements.send_len,
            extra_count,
        );
        timing.pack = started.elapsed();
        let started = Instant::now();

        let subcommunicator = plan
            .source()
            .topology()
            .subcommunicator(plan.changed_topology_axis());
        let mut receive_tail = &mut workspace.receive_buffer[..prepared.requirements.receive_len];
        for peer in plan.peers() {
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

        timing.post_receive = started.elapsed();
        let send_buffer = &workspace.send_buffer[..prepared.requirements.send_len];
        let mut send_tail = send_buffer;
        for peer in plan.peers() {
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

        let started = Instant::now();
        for request in receive_requests {
            request.wait_without_status();
        }
        timing.receive_wait = started.elapsed();
        let started = Instant::now();
        for request in send_requests {
            request.wait_without_status();
        }
        timing.send_wait = started.elapsed();
        Ok(())
    })
}
