use std::convert::TryInto;

use num_complex::Complex;
use pencil_array::{PencilArrayView, PencilArrayViewMut};

use crate::IoError;

mod sealed {
    pub trait Sealed {}
}

/// A sealed scalar representation supported by both native backends.
///
/// Values are identified by a stable code and encoded in canonical
/// little-endian bytes.  The set intentionally excludes platform-sized and
/// user-defined types so equal-width values cannot be confused on disk.
pub trait IoElement: sealed::Sealed + Copy + 'static {
    /// Stable on-file type code.
    const CODE: u64;
    /// Number of canonical bytes per value.
    const WIDTH: usize;

    /// Appends one canonical little-endian value to `out`.
    fn encode_le(self, out: &mut Vec<u8>);

    /// Decodes one canonical little-endian value.
    fn decode_le(bytes: &[u8]) -> Self;
}

macro_rules! impl_integer {
    ($ty:ty, $code:expr, $width:expr) => {
        impl sealed::Sealed for $ty {}
        impl IoElement for $ty {
            const CODE: u64 = $code;
            const WIDTH: usize = $width;

            fn encode_le(self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_le_bytes());
            }

            fn decode_le(bytes: &[u8]) -> Self {
                <$ty>::from_le_bytes(bytes.try_into().expect("validated scalar width"))
            }
        }
    };
}

impl_integer!(i8, 1, 1);
impl_integer!(u8, 2, 1);
impl_integer!(i16, 3, 2);
impl_integer!(u16, 4, 2);
impl_integer!(i32, 5, 4);
impl_integer!(u32, 6, 4);
impl_integer!(i64, 7, 8);
impl_integer!(u64, 8, 8);

macro_rules! impl_float {
    ($ty:ty, $bits:ty, $code:expr, $width:expr) => {
        impl sealed::Sealed for $ty {}
        impl IoElement for $ty {
            const CODE: u64 = $code;
            const WIDTH: usize = $width;

            fn encode_le(self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_bits().to_le_bytes());
            }

            fn decode_le(bytes: &[u8]) -> Self {
                <$ty>::from_bits(<$bits>::from_le_bytes(
                    bytes.try_into().expect("validated scalar width"),
                ))
            }
        }
    };
}

impl_float!(f32, u32, 9, 4);
impl_float!(f64, u64, 10, 8);

impl sealed::Sealed for Complex<f32> {}
impl IoElement for Complex<f32> {
    const CODE: u64 = 11;
    const WIDTH: usize = 8;

    fn encode_le(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.re.to_bits().to_le_bytes());
        out.extend_from_slice(&self.im.to_bits().to_le_bytes());
    }

    fn decode_le(bytes: &[u8]) -> Self {
        let re = u32::from_le_bytes(bytes[..4].try_into().expect("validated complex width"));
        let im = u32::from_le_bytes(bytes[4..8].try_into().expect("validated complex width"));
        Self::new(f32::from_bits(re), f32::from_bits(im))
    }
}

impl sealed::Sealed for Complex<f64> {}
impl IoElement for Complex<f64> {
    const CODE: u64 = 12;
    const WIDTH: usize = 16;

    fn encode_le(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.re.to_bits().to_le_bytes());
        out.extend_from_slice(&self.im.to_bits().to_le_bytes());
    }

    fn decode_le(bytes: &[u8]) -> Self {
        let re = u64::from_le_bytes(bytes[..8].try_into().expect("validated complex width"));
        let im = u64::from_le_bytes(bytes[8..16].try_into().expect("validated complex width"));
        Self::new(f64::from_bits(re), f64::from_bits(im))
    }
}

pub(crate) fn canonical_dims<T, const N: usize, const M: usize>(
    view: &PencilArrayView<'_, T, N, M>,
) -> Result<Vec<usize>, IoError> {
    let extra = view.extra_shape().dimensions();
    let rank = extra.len().checked_add(N).ok_or(IoError::SizeLimit {
        what: "logical rank",
    })?;
    if rank == 0 || rank > crate::MAX_PROTOCOL_RANK {
        return Err(IoError::SizeLimit {
            what: "logical rank",
        });
    }
    let mut dims = Vec::new();
    dims.try_reserve_exact(rank)
        .map_err(|_| IoError::AllocationFailed {
            requested: rank * 8,
        })?;
    dims.extend_from_slice(extra);
    dims.extend_from_slice(&view.local_spatial_shape());
    Ok(dims)
}

pub(crate) fn element_count(dims: &[usize]) -> Result<usize, IoError> {
    dims.iter().try_fold(1usize, |product, &extent| {
        product.checked_mul(extent).ok_or(IoError::SizeLimit {
            what: "element count",
        })
    })
}

pub(crate) fn pack_view<T: IoElement, const N: usize, const M: usize>(
    view: &PencilArrayView<'_, T, N, M>,
) -> Result<Vec<u8>, IoError> {
    let dims = canonical_dims(view)?;
    let elements = element_count(&dims)?;
    if elements != view.len() {
        return Err(IoError::InvalidInput(
            "view storage does not match logical shape",
        ));
    }
    let bytes = elements.checked_mul(T::WIDTH).ok_or(IoError::SizeLimit {
        what: "local payload bytes",
    })?;
    let mut packed = Vec::new();
    packed
        .try_reserve_exact(bytes)
        .map_err(|_| IoError::AllocationFailed { requested: bytes })?;
    let mut indices = Vec::new();
    indices
        .try_reserve_exact(dims.len())
        .map_err(|_| IoError::AllocationFailed {
            requested: dims.len() * std::mem::size_of::<usize>(),
        })?;
    indices.resize(dims.len(), 0);
    let extra_rank = view.extra_shape().dimensions().len();

    for flat in 0..elements {
        set_indices(flat, &dims, &mut indices);
        let spatial = std::array::from_fn(|axis| indices[extra_rank + axis]);
        let value =
            *view
                .get_local(&indices[..extra_rank], spatial)
                .ok_or(IoError::InvalidInput(
                    "view index was outside its validated storage",
                ))?;
        value.encode_le(&mut packed);
    }
    debug_assert_eq!(packed.len(), bytes);
    Ok(packed)
}

/// Decodes a local logical-order payload into the view's physical row-major
/// order.  All allocation and index validation happens before callers close
/// their native resources; the returned vector can therefore be committed with
/// one infallible slice copy.
pub(crate) fn prepare_physical_values<T: IoElement, const N: usize, const M: usize>(
    view: &PencilArrayViewMut<'_, T, N, M>,
    packed: &[u8],
) -> Result<Vec<T>, IoError> {
    let logical_dims = canonical_dims_mut(view)?;
    let elements = element_count(&logical_dims)?;
    let bytes = elements.checked_mul(T::WIDTH).ok_or(IoError::SizeLimit {
        what: "local payload bytes",
    })?;
    if packed.len() != bytes || view.len() != elements {
        return Err(IoError::InvalidFile {
            reason: "local staging length",
        });
    }

    let extra = view.extra_shape().dimensions();
    let extra_rank = extra.len();
    let mut physical_dims = Vec::new();
    physical_dims
        .try_reserve_exact(logical_dims.len())
        .map_err(|_| IoError::AllocationFailed {
            requested: logical_dims.len() * std::mem::size_of::<usize>(),
        })?;
    physical_dims.extend_from_slice(extra);
    physical_dims.extend_from_slice(&view.local_spatial_memory_shape());

    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| IoError::AllocationFailed {
            requested: elements.saturating_mul(std::mem::size_of::<T>()),
        })?;
    let mut physical_indices = Vec::new();
    physical_indices
        .try_reserve_exact(physical_dims.len())
        .map_err(|_| IoError::AllocationFailed {
            requested: physical_dims.len() * std::mem::size_of::<usize>(),
        })?;
    physical_indices.resize(physical_dims.len(), 0);

    for physical_flat in 0..elements {
        set_indices(physical_flat, &physical_dims, &mut physical_indices);
        let mut logical_spatial = [0usize; N];
        for (memory_axis, &axis) in view.pencil().permutation().axes().iter().enumerate() {
            logical_spatial[axis.index()] = physical_indices[extra_rank + memory_axis];
        }

        let mut logical_flat = 0usize;
        for axis in 0..logical_dims.len() {
            let index = if axis < extra_rank {
                physical_indices[axis]
            } else {
                logical_spatial[axis - extra_rank]
            };
            logical_flat = logical_flat
                .checked_mul(logical_dims[axis])
                .and_then(|flat| flat.checked_add(index))
                .ok_or(IoError::InvalidFile {
                    reason: "decoded index",
                })?;
        }
        let start = logical_flat
            .checked_mul(T::WIDTH)
            .ok_or(IoError::InvalidFile {
                reason: "decoded byte offset",
            })?;
        let end = start.checked_add(T::WIDTH).ok_or(IoError::InvalidFile {
            reason: "decoded byte range",
        })?;
        values.push(T::decode_le(&packed[start..end]));
    }
    debug_assert_eq!(values.len(), view.len());
    Ok(values)
}

fn canonical_dims_mut<T, const N: usize, const M: usize>(
    view: &PencilArrayViewMut<'_, T, N, M>,
) -> Result<Vec<usize>, IoError> {
    let extra = view.extra_shape().dimensions();
    let rank = extra.len().checked_add(N).ok_or(IoError::SizeLimit {
        what: "logical rank",
    })?;
    if rank == 0 || rank > crate::MAX_PROTOCOL_RANK {
        return Err(IoError::SizeLimit {
            what: "logical rank",
        });
    }
    let mut dims = Vec::new();
    dims.try_reserve_exact(rank)
        .map_err(|_| IoError::AllocationFailed {
            requested: rank * 8,
        })?;
    dims.extend_from_slice(extra);
    dims.extend_from_slice(&view.local_spatial_shape());
    Ok(dims)
}

fn set_indices(flat: usize, dims: &[usize], indices: &mut [usize]) {
    let mut remaining = flat;
    for axis in (0..dims.len()).rev() {
        indices[axis] = remaining % dims[axis];
        remaining /= dims[axis];
    }
}
