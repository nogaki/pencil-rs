//! Small collective-preflight helper for callers coordinating local work.

use crate::{MpiTopology, transpose::collective_valid};

/// Returns whether every rank accepted its local precondition.
///
/// This performs one reduction on the topology's Cartesian communicator. Every
/// rank must call it in the same order. A `false` result does not identify the
/// rank that rejected its precondition.
pub fn all_ranks_succeeded<const M: usize>(topology: &MpiTopology<M>, local_success: bool) -> bool {
    collective_valid(topology.communicator(), local_success)
}
