use std::{sync::Arc, time::Instant};

use mpi::{
    collective::CommunicatorCollectives,
    datatype::{Equivalence, Partition, PartitionMut},
};

use crate::{
    ExtraShape, ManyPencilArray, Pencil, PencilArrayView, PencilArrayViewMut,
    transpose::{
        CommunicationMode, PreparedExchange, TransposeError, TransposePlanCore, TransposeWorkspace,
        TransposeWorkspaceRequirements, agree_execute_descriptor, collective_valid,
        finish_in_place, pack_source, prepare_in_place, unpack_destination,
    },
};

/// A checked distributed transpose using one `MPI_Alltoallv`.
///
/// [`Self::new`] and both execution methods are collective on the source
/// topology's Cartesian communicator. Every rank must call the same API in the
/// same order with the same source communicator context and matching
/// source-to-destination layouts. Both execution methods additionally require
/// every rank to use the same `T` with a correct
/// [`Equivalence`](mpi::datatype::Equivalence) implementation. These are API
/// contracts; descriptor checks can catch common mismatches but cannot prove
/// type identity, the correctness of an unsafe `Equivalence` implementation,
/// or communicator/collective-order correctness.
///
/// The source and destination pencils must share the same topology object,
/// global shape, and differ in exactly one ordered decomposition position.
/// `workspace_requirements` and [`TransposeWorkspace::from_vecs`] are local
/// operations and do not call MPI; requirements and workspace lengths are
/// checked using initialized `len`, not capacity. Every count, displacement,
/// total, and required view/workspace prefix must fit the checked limits before
/// payload communication starts.
///
/// [`Self::execute_views`] is out-of-place: it preserves the source and leaves
/// the destination and workspace unchanged for ordinary descriptor or
/// preflight errors. [`Self::execute_in_place`] instead replaces the active
/// source layout on success; its ordinary descriptor or preflight errors leave
/// the array state, contents, and workspace unchanged. MPI failures, arbitrary
/// panics, and process loss do not guarantee that a `Result` is recovered or
/// that storage remains unchanged.
#[derive(Debug)]
pub struct AllToAllvTransposePlan<const N: usize, const M: usize> {
    core: TransposePlanCore<N, M>,
}

impl<const N: usize, const M: usize> AllToAllvTransposePlan<N, M> {
    /// Collectively validates and constructs an Alltoallv transpose plan.
    ///
    /// All ranks must call this method in the same order on the same source
    /// communicator context. It performs fixed-header and exact descriptor
    /// agreement before entering any changed-axis subcommunicator. It rejects
    /// an unchanged decomposition rather than silently selecting the local
    /// transpose API. Header, descriptor, and layout failures return before
    /// payload communication; no rank-local early return may skip the required
    /// collective protocol.
    pub fn new(
        source: Arc<Pencil<N, M>>,
        destination: Arc<Pencil<N, M>>,
    ) -> Result<Self, TransposeError> {
        Ok(Self {
            core: TransposePlanCore::new(source, destination, CommunicationMode::AllToAllv)?,
        })
    }

    /// Returns the initialized workspace lengths required for `extra_shape`.
    ///
    /// This method is noncollective and does not call MPI or allocate the
    /// returned storage. The result is local to the calling rank because peer
    /// partitions can have different lengths. It includes all peer segments
    /// and checks the `mpi::Count` bounds used by execution.
    pub fn workspace_requirements(
        &self,
        extra_shape: &ExtraShape,
    ) -> Result<TransposeWorkspaceRequirements, TransposeError> {
        self.core.workspace_requirements(extra_shape)
    }

    /// Executes the transpose after a whole-source-topology collective
    /// preflight through one `MPI_Alltoallv`.
    ///
    /// Every rank must call this method in the same order on the same source
    /// communicator context, with the same `T` and a correct `Equivalence`
    /// implementation. The source is never written. Ordinary descriptor or
    /// preflight errors are returned on all ranks before packing,
    /// communication, or destination writes, so the destination and workspace
    /// remain unchanged on those paths. Workspace initialized lengths are
    /// checked by `len`; only required prefixes are used and no vector is
    /// resized or reallocated. Every count and displacement, including their
    /// checked sum and total, must fit `mpi::Count`/buffer limits. After the
    /// `MPI_Alltoallv` call starts, an MPI failure, arbitrary panic, or process
    /// loss does not guarantee that a `Result` is recovered or that the
    /// destination is unchanged. The source and destination views must use the
    /// layouts held by this plan and have exactly equal extra shapes.
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
        agree_execute_descriptor::<_, T, N, M>(
            &self.core,
            communicator,
            source.extra_shape(),
            destination.extra_shape(),
            CommunicationMode::AllToAllv.views_operation(),
        )?;

        let local_preflight = self
            .core
            .prepare_execution(&source, &destination, workspace);
        if !collective_valid(communicator, local_preflight.is_ok()) {
            return Err(local_preflight
                .err()
                .unwrap_or(TransposeError::CollectivePreconditionFailed));
        }
        let prepared = local_preflight.expect("collective execution preflight succeeded");
        let extra_count = source.extra_shape().element_count();
        pack_source(
            self.core.peers(),
            self.core.source().as_ref(),
            source.as_slice(),
            &mut workspace.send_buffer[..prepared.requirements.send_len],
            prepared.requirements.send_len,
            extra_count,
        );
        execute_exchange(&self.core, &prepared, workspace);
        unpack_destination(
            self.core.peers(),
            self.core.destination().as_ref(),
            destination.as_mut_slice(),
            &workspace.receive_buffer[..prepared.requirements.receive_len],
            extra_count,
        );
        Ok(())
    }

    /// Executes the transpose and returns local pack, collective wait, unpack,
    /// and total wall-clock timings.
    pub fn execute_views_with_timing<T>(
        &self,
        source: PencilArrayView<'_, T, N, M>,
        mut destination: PencilArrayViewMut<'_, T, N, M>,
        workspace: &mut TransposeWorkspace<T>,
    ) -> Result<crate::TransposeTiming, TransposeError>
    where
        T: Equivalence + Copy,
    {
        let total = Instant::now();
        let communicator = self.core.source().topology().cartesian();
        agree_execute_descriptor::<_, T, N, M>(
            &self.core,
            communicator,
            source.extra_shape(),
            destination.extra_shape(),
            CommunicationMode::AllToAllv.timed_views_operation(),
        )?;
        let local = self
            .core
            .prepare_execution(&source, &destination, workspace);
        if !collective_valid(communicator, local.is_ok()) {
            return Err(local
                .err()
                .unwrap_or(TransposeError::CollectivePreconditionFailed));
        }
        let prepared = local.expect("collective execution preflight succeeded");
        let extra_count = source.extra_shape().element_count();
        let mut timing = crate::TransposeTiming::default();
        let started = Instant::now();
        pack_source(
            self.core.peers(),
            self.core.source().as_ref(),
            source.as_slice(),
            &mut workspace.send_buffer[..prepared.requirements.send_len],
            prepared.requirements.send_len,
            extra_count,
        );
        timing.pack = started.elapsed();
        let started = Instant::now();
        execute_exchange(&self.core, &prepared, workspace);
        timing.collective_wait = started.elapsed();
        let started = Instant::now();
        unpack_destination(
            self.core.peers(),
            self.core.destination().as_ref(),
            destination.as_mut_slice(),
            &workspace.receive_buffer[..prepared.requirements.receive_len],
            extra_count,
        );
        timing.unpack = started.elapsed();
        timing.total = total.elapsed();
        Ok(timing)
    }

    /// Executes the transpose by replacing the active layout in shared storage.
    ///
    /// Every rank must call this method in the same order on the same source
    /// communicator context, with the same `T` and a correct `Equivalence`
    /// implementation. The active layout must match the plan's source and the
    /// destination must be registered. Ordinary descriptor or preflight errors
    /// leave the array state, contents, and workspace unchanged. The array
    /// remains valid with its source layout through pack and communication;
    /// only after communication completes is it poisoned, then its destination
    /// prefix is unpacked and committed. A successful operation therefore
    /// replaces the active source; it does not preserve source contents. MPI
    /// failures, arbitrary panics, and process loss do not guarantee that a
    /// `Result` is recovered or that storage is unchanged.
    pub fn execute_in_place<T>(
        &self,
        array: &mut ManyPencilArray<T, N, M>,
        workspace: &mut TransposeWorkspace<T>,
    ) -> Result<(), TransposeError>
    where
        T: Equivalence + Copy,
    {
        let communicator = self.core.source().topology().cartesian();
        agree_execute_descriptor::<_, T, N, M>(
            &self.core,
            communicator,
            array.extra_shape(),
            array.extra_shape(),
            CommunicationMode::AllToAllv.in_place_operation(),
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
            pack_source(
                self.core.peers(),
                self.core.source().as_ref(),
                source.as_slice(),
                &mut workspace.send_buffer[..prepared.exchange.requirements.send_len],
                prepared.exchange.requirements.send_len,
                extra_count,
            );
        }
        execute_exchange(&self.core, &prepared.exchange, workspace);
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

    /// Executes the in-place transpose and returns local phase timings.
    pub fn execute_in_place_with_timing<T>(
        &self,
        array: &mut ManyPencilArray<T, N, M>,
        workspace: &mut TransposeWorkspace<T>,
    ) -> Result<crate::TransposeTiming, TransposeError>
    where
        T: Equivalence + Copy,
    {
        let total = Instant::now();
        let communicator = self.core.source().topology().cartesian();
        agree_execute_descriptor::<_, T, N, M>(
            &self.core,
            communicator,
            array.extra_shape(),
            array.extra_shape(),
            CommunicationMode::AllToAllv.timed_in_place_operation(),
        )?;
        let local = prepare_in_place(&self.core, array, workspace);
        if !collective_valid(communicator, local.is_ok()) {
            return Err(local
                .err()
                .unwrap_or(TransposeError::CollectivePreconditionFailed));
        }
        let prepared = local.expect("collective in-place preflight succeeded");
        let extra_count = array.extra_shape().element_count();
        let mut timing = crate::TransposeTiming::default();
        let started = Instant::now();
        {
            let source = array.active_view().expect("validated active source");
            pack_source(
                self.core.peers(),
                self.core.source().as_ref(),
                source.as_slice(),
                &mut workspace.send_buffer[..prepared.exchange.requirements.send_len],
                prepared.exchange.requirements.send_len,
                extra_count,
            );
        }
        timing.pack = started.elapsed();
        let started = Instant::now();
        execute_exchange(&self.core, &prepared.exchange, workspace);
        timing.collective_wait = started.elapsed();
        let started = Instant::now();
        finish_in_place(
            &self.core,
            array,
            prepared.destination_index,
            &prepared.exchange,
            &workspace.receive_buffer[..prepared.exchange.requirements.receive_len],
            extra_count,
        );
        timing.unpack = started.elapsed();
        timing.total = total.elapsed();
        Ok(timing)
    }
}

fn execute_exchange<T, const N: usize, const M: usize>(
    plan: &TransposePlanCore<N, M>,
    prepared: &PreparedExchange,
    workspace: &mut TransposeWorkspace<T>,
) where
    T: Equivalence + Copy,
{
    let subcommunicator = plan
        .source()
        .topology()
        .subcommunicator(plan.changed_topology_axis());
    if prepared.requirements.send_len == 0 && prepared.requirements.receive_len == 0 {
        // Some MPI implementations require a distinct real buffer address
        // even when every count is zero. This initialized dummy storage is
        // never exposed as a T value and is only used for zero-count MPI calls.
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
        let receive_buffer = &mut workspace.receive_buffer[..prepared.requirements.receive_len];
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
