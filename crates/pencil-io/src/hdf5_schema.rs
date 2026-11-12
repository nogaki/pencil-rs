//! HDF5 dataset schema shared by the serial and parallel backends.
//!
//! Only MPI-independent type mapping and post-read value rules live here.
//! Native handle ownership, MPI agreement, and expected-shape comparison stay
//! in each backend.

use std::ffi::CStr;

use hdf5_metno_sys::h5i::hid_t;

use crate::{COMMIT_MARKER, FORMAT_VERSION, INCOMPLETE_MARKER, MAX_PROTOCOL_RANK};

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

macro_rules! attributes {
    ($($variant:ident => $name:ident),+ $(,)?) => {
        /// The ten attributes of the v1 metadata schema, in write and read order.
        #[derive(Clone, Copy, PartialEq, Eq)]
        pub(crate) enum Attribute {
            $($variant,)+
        }

        impl Attribute {
            pub(crate) const ALL: [Self; 10] = [$(Self::$variant,)+];

            pub(crate) fn name(self) -> &'static CStr {
                match self {
                    $(Self::$variant => $name,)+
                }
            }
        }
    };
}

attributes! {
    Version => ATTR_VERSION,
    Commit => ATTR_COMMIT,
    Rank => ATTR_N,
    TypeCode => ATTR_TYPE,
    Width => ATTR_WIDTH,
    ExtraRank => ATTR_EXTRA_RANK,
    ExtraShape => ATTR_EXTRA,
    GlobalShape => ATTR_GLOBAL,
    WriterGrid => ATTR_GRID,
    WriterPermutation => ATTR_PERM,
}

impl Attribute {
    /// The extra shape exists only when the extra rank is nonzero; every other
    /// attribute is always present.
    pub(crate) fn present(self, extra_rank: usize) -> bool {
        self != Self::ExtraShape || extra_rank > 0
    }

    /// Expected and maximum read lengths for spatial rank `n` and extra rank
    /// `er`. The writer grid has no fixed length and may be empty.
    pub(crate) fn read_len(self, n: usize, er: usize) -> (Option<usize>, usize) {
        match self {
            Self::ExtraShape => (Some(er), er),
            Self::GlobalShape | Self::WriterPermutation => (Some(n), n),
            Self::WriterGrid => (None, MAX_PROTOCOL_RANK),
            _ => (Some(1), 1),
        }
    }
}

/// One attribute's write value or read-side expectation. `None` means the
/// reader makes no check for that attribute; the catalogs only expect the
/// version.
#[derive(Clone, Copy)]
enum Slot<'a> {
    None,
    Scalar(u64),
    Values(&'a [u64]),
    Shape(&'a [usize]),
}

/// Values of the ten schema attributes. Writers use [`Self::for_write`];
/// readers use [`Self::expected`] or [`Self::version_only`] and [`Self::matches`].
pub(crate) struct Metadata<'a> {
    slots: [Slot<'a>; 10],
}

impl<'a> Metadata<'a> {
    pub(crate) fn for_write(
        rank: u64,
        type_code: u64,
        width: u64,
        extra: &'a [u64],
        global: &'a [u64],
        grid: &'a [u64],
        permutation: &'a [u64],
    ) -> Self {
        Self {
            slots: [
                Slot::Scalar(FORMAT_VERSION),
                Slot::Scalar(INCOMPLETE_MARKER),
                Slot::Scalar(rank),
                Slot::Scalar(type_code),
                Slot::Scalar(width),
                Slot::Scalar(extra.len() as u64),
                Slot::Values(extra),
                Slot::Values(global),
                Slot::Values(grid),
                Slot::Values(permutation),
            ],
        }
    }

    pub(crate) fn expected(
        rank: u64,
        type_code: u64,
        width: u64,
        extra_rank: u64,
        extra: &'a [usize],
        global: &'a [usize],
    ) -> Self {
        Self {
            slots: [
                Slot::Scalar(FORMAT_VERSION),
                Slot::None,
                Slot::Scalar(rank),
                Slot::Scalar(type_code),
                Slot::Scalar(width),
                Slot::Scalar(extra_rank),
                Slot::Shape(extra),
                Slot::Shape(global),
                Slot::None,
                Slot::None,
            ],
        }
    }

    #[cfg(feature = "parallel-hdf5")]
    pub(crate) fn version_only() -> Self {
        let mut slots = [Slot::None; 10];
        slots[Attribute::Version as usize] = Slot::Scalar(FORMAT_VERSION);
        Self { slots }
    }

    pub(crate) fn value(&self, attribute: Attribute) -> &[u64] {
        match &self.slots[attribute as usize] {
            Slot::Scalar(value) => std::slice::from_ref(value),
            Slot::Values(values) => values,
            Slot::None | Slot::Shape(_) => &[],
        }
    }

    /// True when `read` satisfies the expectation for `attribute`, or when the
    /// reader has no expectation for it.
    pub(crate) fn matches(&self, attribute: Attribute, read: &[u64]) -> bool {
        match &self.slots[attribute as usize] {
            Slot::None => true,
            Slot::Scalar(expected) => read.iter().copied().eq(std::iter::once(*expected)),
            Slot::Values(expected) => read.iter().copied().eq(expected.iter().copied()),
            Slot::Shape(expected) => read.iter().copied().eq(expected.iter().map(|&v| v as u64)),
        }
    }
}

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

pub(crate) fn rank_in_range(rank: u64) -> bool {
    rank <= MAX_PROTOCOL_RANK as u64
}

pub(crate) fn combined_rank_in_range(rank: u64, extra_rank: u64) -> bool {
    rank.checked_add(extra_rank).is_some_and(rank_in_range)
}

pub(crate) fn valid_global_shape(values: &[u64]) -> bool {
    !values.is_empty()
        && values
            .iter()
            .copied()
            .try_fold(1u64, |product, value| product.checked_mul(value))
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

/// Writer provenance: a bounded positive grid and an axis permutation.
pub(crate) fn valid_writer_provenance(
    grid: &[u64],
    permutation: &[u64],
    spatial_rank: usize,
) -> bool {
    let mut seen = [false; MAX_PROTOCOL_RANK];
    valid_writer_grid(grid, spatial_rank) && valid_permutation(permutation, spatial_rank, &mut seen)
}

/// Which extra shape/overflow rule a reader applies to stored metadata.
#[derive(Clone, Copy)]
#[cfg_attr(not(feature = "parallel-hdf5"), allow(dead_code))]
pub(crate) enum ShapeRule {
    /// Compare the stored shape with the caller's expected shape (serial read).
    Expected,
    /// Require a nonempty global shape whose product fits (parallel read).
    Global,
    /// Require the combined shape product and width to fit (catalog).
    Bytes,
    /// Require the width-first combined product to fit (session catalog).
    BytesFromWidth,
}

/// Stored shape and provenance values already read from a dataset.
pub(crate) struct ShapeMetadata<'a> {
    pub(crate) extra: &'a [u64],
    pub(crate) global: &'a [u64],
    pub(crate) grid: &'a [u64],
    pub(crate) permutation: &'a [u64],
    pub(crate) spatial_rank: usize,
    pub(crate) width: u64,
    pub(crate) rule: ShapeRule,
}

/// Why shared metadata acceptance rejected a dataset.
pub(crate) enum MetadataIssue {
    /// The commit marker records an unfinished write.
    Incomplete,
    /// The commit marker is not a recognized value.
    InvalidCommit,
    /// The stored shape or writer provenance is invalid.
    Shape,
}

/// The one acceptance rule for commit, shape and writer provenance metadata.
/// Readers pass only the group they have read so far, preserving their own
/// attribute read and error order.
pub(crate) fn metadata_acceptance(
    expected: &Metadata<'_>,
    commit: Option<u64>,
    shape: Option<ShapeMetadata<'_>>,
) -> Result<(), MetadataIssue> {
    if let Some(value) = commit {
        match commit_state(value) {
            CommitState::Complete => {}
            CommitState::Incomplete => return Err(MetadataIssue::Incomplete),
            CommitState::Invalid => return Err(MetadataIssue::InvalidCommit),
        }
    }
    let Some(shape) = shape else {
        return Ok(());
    };
    // Multiplication is commutative, so the catalog (`Bytes`) and session
    // (`BytesFromWidth`) orderings have the same result once zero extents are
    // handled together: one zero factor makes the element count zero, and no
    // later factor can overflow. Only a nonzero product may be rejected.
    let mut dims = shape.global.iter().chain(shape.extra).copied();
    let width = match shape.rule {
        ShapeRule::Bytes | ShapeRule::BytesFromWidth => Some(shape.width),
        ShapeRule::Expected | ShapeRule::Global => None,
    };
    let bytes_ok = width == Some(0)
        || dims.clone().any(|value| value == 0)
        || dims
            .try_fold(1u64, |product, value| product.checked_mul(value))
            .and_then(|product| width.map_or(Some(product), |width| product.checked_mul(width)))
            .is_some();
    let rule_ok = match shape.rule {
        ShapeRule::Expected | ShapeRule::BytesFromWidth => true,
        ShapeRule::Global => valid_global_shape(shape.global),
        ShapeRule::Bytes => {
            !shape.global.is_empty()
                && combined_rank_in_range(shape.global.len() as u64, shape.extra.len() as u64)
        }
    };
    if !bytes_ok
        || !rule_ok
        || !expected.matches(Attribute::GlobalShape, shape.global)
        || !expected.matches(Attribute::ExtraShape, shape.extra)
        || !valid_writer_provenance(shape.grid, shape.permutation, shape.spatial_rank)
    {
        return Err(MetadataIssue::Shape);
    }
    Ok(())
}
