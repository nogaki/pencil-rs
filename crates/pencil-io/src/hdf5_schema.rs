//! HDF5 dataset schema shared by the serial and parallel backends.
//!
//! Only MPI-independent type mapping and post-read value rules live here.
//! Native handle ownership, MPI agreement, and expected-shape comparison stay
//! in each backend.

use std::ffi::CStr;

use hdf5_metno_sys::h5i::hid_t;

#[cfg(feature = "parallel-hdf5")]
use crate::MAX_PROTOCOL_RANK;
use crate::{COMMIT_MARKER, INCOMPLETE_MARKER};

pub(crate) const ATTR_VERSION: &CStr = c"pencil_io_version";
pub(crate) const ATTR_COMMIT: &CStr = c"pencil_io_commit";
pub(crate) const ATTR_N: &CStr = c"pencil_io_n";
pub(crate) const ATTR_TYPE: &CStr = c"pencil_io_type";
pub(crate) const ATTR_WIDTH: &CStr = c"pencil_io_width";
pub(crate) const ATTR_EXTRA_RANK: &CStr = c"pencil_io_extra_rank";
pub(crate) const ATTR_EXTRA: &CStr = c"pencil_io_extra_shape";
pub(crate) const ATTR_GLOBAL: &CStr = c"pencil_io_global_shape";
pub(crate) const ATTR_GRID: &CStr = c"pencil_io_writer_grid";
pub(crate) const ATTR_PERM: &CStr = c"pencil_io_writer_permutation";

pub(crate) struct ScalarSpec {
    pub(crate) base: hid_t,
    pub(crate) complex: bool,
    pub(crate) width: usize,
}

/// Maps a sealed `IoElement` type code to its canonical HDF5 scalar.
pub(crate) fn scalar_spec(code: u64) -> Option<ScalarSpec> {
    use hdf5_metno::globals::*;
    let (base, complex, width) = match code {
        1 => (*H5T_STD_I8LE, false, 1),
        2 => (*H5T_STD_U8LE, false, 1),
        3 => (*H5T_STD_I16LE, false, 2),
        4 => (*H5T_STD_U16LE, false, 2),
        5 => (*H5T_STD_I32LE, false, 4),
        6 => (*H5T_STD_U32LE, false, 4),
        7 => (*H5T_STD_I64LE, false, 8),
        8 => (*H5T_STD_U64LE, false, 8),
        9 => (*H5T_IEEE_F32LE, false, 4),
        10 => (*H5T_IEEE_F64LE, false, 8),
        11 => (*H5T_IEEE_F32LE, true, 8),
        12 => (*H5T_IEEE_F64LE, true, 16),
        _ => return None,
    };
    Some(ScalarSpec {
        base,
        complex,
        width,
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum CommitState {
    Complete,
    Incomplete,
    Invalid,
}

pub(crate) fn commit_state(value: u64) -> CommitState {
    match value {
        COMMIT_MARKER => CommitState::Complete,
        INCOMPLETE_MARKER => CommitState::Incomplete,
        _ => CommitState::Invalid,
    }
}

#[cfg(feature = "parallel-hdf5")]
pub(crate) fn rank_in_range(rank: u64) -> bool {
    rank <= MAX_PROTOCOL_RANK as u64
}

#[cfg(feature = "parallel-hdf5")]
pub(crate) fn combined_rank_in_range(rank: u64, extra_rank: u64) -> bool {
    rank.checked_add(extra_rank).is_some_and(rank_in_range)
}

#[cfg(feature = "parallel-hdf5")]
pub(crate) fn valid_global_shape(values: &[u64]) -> bool {
    !values.is_empty()
        && values
            .iter()
            .copied()
            .try_fold(1u64, |product, value| product.checked_mul(value))
            .is_some()
}

/// Checks the byte count with the combined shape product first and the width
/// last, matching the catalog's historical overflow points.
#[cfg(feature = "parallel-hdf5")]
pub(crate) fn valid_shape_bytes(global: &[u64], extra: &[u64], width: u64) -> bool {
    global
        .iter()
        .chain(extra)
        .try_fold(1u64, |product, &value| product.checked_mul(value))
        .and_then(|product| product.checked_mul(width))
        .is_some()
}

/// Checks the byte count starting from the element width, matching the
/// session catalog's historical overflow points.
#[cfg(feature = "parallel-hdf5")]
pub(crate) fn valid_shape_bytes_from_width(global: &[u64], extra: &[u64], width: u64) -> bool {
    global
        .iter()
        .chain(extra)
        .try_fold(width, |product, &value| product.checked_mul(value))
        .is_some()
}

/// Checks a writer grid without validating the axis permutation; permutation
/// scratch ownership differs between the serial and parallel backends.
pub(crate) fn valid_writer_grid(grid: &[u64], spatial_rank: usize) -> bool {
    !grid.is_empty()
        && grid.len() <= spatial_rank
        && grid.iter().all(|&value| value > 0)
        && grid
            .iter()
            .try_fold(1u64, |product, &value| product.checked_mul(value))
            .is_some()
}

/// Checks that `values` is a permutation of `0..n`, using backend-owned
/// scratch. `seen` must hold at least `n` flags.
pub(crate) fn valid_permutation(values: &[u64], n: usize, seen: &mut [bool]) -> bool {
    if values.len() != n || seen.len() < n {
        return false;
    }
    seen[..n].fill(false);
    for &value in values {
        let Ok(index) = usize::try_from(value) else {
            return false;
        };
        if index >= n || seen[index] {
            return false;
        }
        seen[index] = true;
    }
    true
}
