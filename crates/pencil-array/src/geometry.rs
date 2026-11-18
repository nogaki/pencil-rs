use std::ops::Range;

use crate::{GeometryError, SpatialAxis};

/// Partitions `0..length` using the floor rule, allowing empty local ranges.
///
/// Rank `coordinate` owns `floor(length * coordinate / parts)` through
/// `floor(length * (coordinate + 1) / parts)`, with an exclusive end.
pub fn partition_range(
    length: usize,
    parts: usize,
    coordinate: usize,
) -> Result<Range<usize>, GeometryError> {
    if parts == 0 {
        return Err(GeometryError::ZeroPartitions);
    }
    if coordinate >= parts {
        return Err(GeometryError::ProcessCoordinateOutOfBounds {
            axis: 0,
            coordinate,
            extent: parts,
        });
    }

    // Rust's supported pointer widths are at most 64 bits, so both products
    // fit u128. Each quotient is <= length and therefore fits usize again.
    // coordinate < parts also guarantees that coordinate + 1 fits usize.
    let boundary = |position: usize| ((length as u128 * position as u128) / parts as u128) as usize;
    Ok(boundary(coordinate)..boundary(coordinate + 1))
}

/// Computes ranges for an already validated ordered spatial decomposition.
pub(crate) fn local_ranges_for<const N: usize, const M: usize>(
    global_shape: [usize; N],
    process_grid: [usize; M],
    process_coords: [usize; M],
    decomposition: [SpatialAxis; M],
) -> Result<[Range<usize>; N], GeometryError> {
    let mut ranges = std::array::from_fn(|axis| 0..global_shape[axis]);
    for topology_axis in 0..M {
        let extent = process_grid[topology_axis];
        let coordinate = process_coords[topology_axis];
        if extent == 0 {
            return Err(GeometryError::ZeroProcessExtent {
                axis: topology_axis,
            });
        }
        if coordinate >= extent {
            return Err(GeometryError::ProcessCoordinateOutOfBounds {
                axis: topology_axis,
                coordinate,
                extent,
            });
        }
        let spatial_axis = decomposition[topology_axis].index();
        ranges[spatial_axis] = partition_range(global_shape[spatial_axis], extent, coordinate)?;
    }
    Ok(ranges)
}

pub(crate) fn shape_from_ranges<const N: usize>(ranges: &[Range<usize>; N]) -> [usize; N] {
    std::array::from_fn(|axis| ranges[axis].len())
}

// Keep the private unit-test symbol; its old name is not a runtime layout mode.
#[cfg(test)]
fn row_major_offset(shape: &[usize], indices: &[usize]) -> Result<usize, GeometryError> {
    column_major_offset(shape, indices)
}

pub(crate) fn column_major_offset(
    shape: &[usize],
    indices: &[usize],
) -> Result<usize, GeometryError> {
    if shape.len() != indices.len() {
        return Err(GeometryError::RankMismatch {
            shape_rank: shape.len(),
            index_rank: indices.len(),
        });
    }
    let mut offset = 0usize;
    let mut stride = 1usize;
    for (axis, (&extent, &index)) in shape.iter().zip(indices).enumerate() {
        if index >= extent {
            return Err(GeometryError::LocalIndexOutOfBounds {
                axis,
                index,
                extent,
            });
        }
        offset = offset
            .checked_add(
                index
                    .checked_mul(stride)
                    .ok_or(GeometryError::SizeOverflow)?,
            )
            .ok_or(GeometryError::SizeOverflow)?;
        if axis + 1 < shape.len() {
            stride = stride
                .checked_mul(extent)
                .ok_or(GeometryError::SizeOverflow)?;
        }
    }
    Ok(offset)
}

#[cfg(test)]
#[path = "tests/geometry.rs"]
mod tests;
