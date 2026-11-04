//! Small collective-preflight helpers for callers coordinating local work.

use mpi::collective::{CommunicatorCollectives, SystemOperation};

use crate::{MpiTopology, transpose::collective_valid};

/// Words in the fixed header agreed before any variable-length metadata.
pub(crate) const HEADER_WORDS: usize = 5;

/// Agrees one fixed five-word header across `comm`.
///
/// This fixed exchange is shared by the crate's transpose, spatial-remap, and
/// reduction protocols. Do not turn it into per-word scalar reductions: a rank
/// may be entering a different checked API, and the MPI collective count and
/// order must still match while that mismatch is reported. `pencil-fft` keeps
/// its own identical copy of this protocol.
pub(crate) fn agree_header<C: CommunicatorCollectives>(
    comm: &C,
    header: [u64; HEADER_WORDS],
) -> bool {
    let mut minimum = [0u64; HEADER_WORDS];
    let mut maximum = [0u64; HEADER_WORDS];
    comm.all_reduce_into(&header[..], &mut minimum[..], SystemOperation::min());
    comm.all_reduce_into(&header[..], &mut maximum[..], SystemOperation::max());
    minimum == maximum
}

/// Returns whether every rank accepted its local precondition.
///
/// This performs one reduction on the topology's Cartesian communicator. Every
/// rank must call it in the same order. A `false` result does not identify the
/// rank that rejected its precondition.
pub fn all_ranks_succeeded<const M: usize>(topology: &MpiTopology<M>, local_success: bool) -> bool {
    collective_valid(topology.communicator(), local_success)
}
