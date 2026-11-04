//! Small collective-preflight helpers for callers coordinating local work.

use mpi::collective::{CommunicatorCollectives, SystemOperation};

use crate::{MpiTopology, transpose::collective_valid};

/// Words in the fixed header agreed before any variable-length metadata.
pub(crate) const HEADER_WORDS: usize = 5;

/// Writes `[values, !values]` into `paired` for one max all-reduce.
pub(crate) fn expand_for_max(values: &[u64], paired: &mut [u64]) {
    let (top, bottom) = paired.split_at_mut(values.len());
    top.copy_from_slice(values);
    for (destination, &value) in bottom.iter_mut().zip(values) {
        *destination = !value;
    }
}

/// Splits a max-reduced `[values, !values]` buffer into global min and max.
///
/// Each complemented word's maximum is the complement of the original word's
/// minimum, so the second half recovers the minimum exactly.
pub(crate) fn split_min_max(reduced: &[u64], minimum: &mut [u64], maximum: &mut [u64]) {
    let (top, bottom) = reduced.split_at(reduced.len() / 2);
    maximum.copy_from_slice(top);
    for (destination, &value) in minimum.iter_mut().zip(bottom) {
        *destination = !value;
    }
}

/// Agrees one fixed five-word header across `comm`.
///
/// This fixed exchange is shared by the crate's transpose, spatial-remap, and
/// reduction protocols. Do not turn it into per-word scalar reductions: a rank
/// may be entering a different checked API, and the MPI collective count and
/// order must still match while that mismatch is reported. It is one max
/// reduction over `[header, !header]`; `pencil-fft` keeps its own identical copy
/// of this protocol.
pub(crate) fn agree_header<C: CommunicatorCollectives>(
    comm: &C,
    header: [u64; HEADER_WORDS],
) -> bool {
    let mut paired = [0u64; 2 * HEADER_WORDS];
    let mut reduced = [0u64; 2 * HEADER_WORDS];
    expand_for_max(&header, &mut paired);
    comm.all_reduce_into(&paired[..], &mut reduced[..], SystemOperation::max());
    let mut minimum = [0u64; HEADER_WORDS];
    let mut maximum = [0u64; HEADER_WORDS];
    split_min_max(&reduced, &mut minimum, &mut maximum);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complemented_max_reduction_recovers_min_and_max() {
        let left = [7u64, 1, u64::MAX, 0, 42];
        let right = [3u64, 9, u64::MAX - 1, 0, 7];
        let mut maximum = [0u64; HEADER_WORDS];
        let mut complemented_minimum = [0u64; HEADER_WORDS];
        for index in 0..HEADER_WORDS {
            maximum[index] = left[index].max(right[index]);
            complemented_minimum[index] = (!left[index]).max(!right[index]);
        }
        let mut reduced = [0u64; 2 * HEADER_WORDS];
        reduced[..HEADER_WORDS].copy_from_slice(&maximum);
        reduced[HEADER_WORDS..].copy_from_slice(&complemented_minimum);

        let mut minimum = [0u64; HEADER_WORDS];
        let mut maximum = [0u64; HEADER_WORDS];
        split_min_max(&reduced, &mut minimum, &mut maximum);
        for index in 0..HEADER_WORDS {
            assert_eq!(minimum[index], left[index].min(right[index]));
            assert_eq!(maximum[index], left[index].max(right[index]));
        }
    }

    #[test]
    fn expanded_buffer_holds_each_word_and_its_complement() {
        let header = [1u64, 2, 3, 4, 5];
        let mut paired = [0u64; 2 * HEADER_WORDS];
        expand_for_max(&header, &mut paired);
        assert_eq!(&paired[..HEADER_WORDS], &header);
        for index in 0..HEADER_WORDS {
            assert_eq!(paired[HEADER_WORDS + index], !header[index]);
        }
    }
}
