use std::convert::TryInto;

use num_complex::Complex;
#[cfg(feature = "mpi")]
use pencil_array::{Pencil, PencilArrayView, PencilArrayViewMut};

#[cfg(any(feature = "mpi", feature = "hdf5"))]
use crate::IoError;

mod sealed {
    pub trait Sealed {}
}

/// A sealed scalar representation supported by the native I/O backends.
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

macro_rules! impl_primitive {
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

impl_primitive!(i8, 1, 1);
impl_primitive!(u8, 2, 1);
impl_primitive!(i16, 3, 2);
impl_primitive!(u16, 4, 2);
impl_primitive!(i32, 5, 4);
impl_primitive!(u32, 6, 4);
impl_primitive!(i64, 7, 8);
impl_primitive!(u64, 8, 8);
impl_primitive!(f32, 9, 4);
impl_primitive!(f64, 10, 8);

#[cfg(test)]
#[test]
fn float_codec_preserves_ieee_bits() {
    assert_eq!(
        (f32::CODE, f32::WIDTH, f64::CODE, f64::WIDTH),
        (9, 4, 10, 8)
    );
    // Signed zero, subnormal, infinities, and signalling/quiet NaN payloads.
    for bits in [
        0, 1, 0x80000000, 0x7f800000, 0xff800000, 0x7f800001, 0x7fc01234,
    ] {
        let mut bytes = Vec::new();
        f32::from_bits(bits).encode_le(&mut bytes);
        assert_eq!(bytes, bits.to_le_bytes());
        assert_eq!(f32::decode_le(&bytes).to_bits(), bits);
    }
    for bits in [
        0,
        1,
        0x8000000000000000,
        0x7ff0000000000000,
        0xfff0000000000000,
        0x7ff0000000000001,
        0x7ff8000000001234,
    ] {
        let mut bytes = Vec::new();
        f64::from_bits(bits).encode_le(&mut bytes);
        assert_eq!(bytes, bits.to_le_bytes());
        assert_eq!(f64::decode_le(&bytes).to_bits(), bits);
    }
}

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

#[cfg(feature = "mpi")]
fn canonical_dims<const N: usize, const M: usize>(
    extra: &[usize],
    pencil: &Pencil<N, M>,
) -> Result<Vec<usize>, IoError> {
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
    dims.extend_from_slice(&pencil.local_shape_logical());
    Ok(dims)
}

#[cfg(any(feature = "mpi", feature = "hdf5"))]
pub(crate) fn element_count(dims: &[usize]) -> Result<usize, IoError> {
    dims.iter().try_fold(1usize, |product, &extent| {
        product.checked_mul(extent).ok_or(IoError::SizeLimit {
            what: "element count",
        })
    })
}

#[cfg(feature = "mpi")]
pub(crate) fn pack_view<T: IoElement, const N: usize, const M: usize>(
    view: &PencilArrayView<'_, T, N, M>,
) -> Result<Vec<u8>, IoError> {
    let dims = canonical_dims(view.extra_shape().dimensions(), view.pencil())?;
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

/// Validated layout metadata and scratch for one physical-order decode.
///
/// Retains neither the destination view nor the byte payload.
#[cfg(feature = "mpi")]
pub(crate) struct PhysicalDecodePlan<T: IoElement, const N: usize> {
    logical_dims: Vec<usize>,
    physical_dims: Vec<usize>,
    physical_indices: Vec<usize>,
    logical_spatial: [usize; N],
    memory_to_logical: [usize; N],
    extra_rank: usize,
    elements: usize,
    bytes: usize,
    _element: std::marker::PhantomData<T>,
}

/// Validates a logical-order payload and allocates O(rank) decode scratch.
///
/// Leaves the destination unchanged and retains neither argument. Decode with
/// this payload and the original view's storage only after native cleanup and
/// the final collective gate have succeeded.
#[cfg(feature = "mpi")]
pub(crate) fn prepare_physical_decode<T: IoElement, const N: usize, const M: usize>(
    view: &PencilArrayViewMut<'_, T, N, M>,
    packed: &[u8],
) -> Result<PhysicalDecodePlan<T, N>, IoError> {
    let logical_dims = canonical_dims(view.extra_shape().dimensions(), view.pencil())?;
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

    let mut physical_indices = Vec::new();
    physical_indices
        .try_reserve_exact(physical_dims.len())
        .map_err(|_| IoError::AllocationFailed {
            requested: physical_dims.len() * std::mem::size_of::<usize>(),
        })?;
    physical_indices.resize(physical_dims.len(), 0);

    let memory_to_logical =
        std::array::from_fn(|axis| view.pencil().permutation().axes()[axis].index());

    Ok(PhysicalDecodePlan {
        logical_dims,
        physical_dims,
        physical_indices,
        logical_spatial: [0usize; N],
        memory_to_logical,
        extra_rank,
        elements,
        bytes,
        _element: std::marker::PhantomData,
    })
}

#[cfg(feature = "mpi")]
impl<T: IoElement, const N: usize> PhysicalDecodePlan<T, N> {
    /// Consumes this plan and decodes directly into physical-order storage.
    ///
    /// The caller must supply the original payload and the original view's
    /// storage. Correct use performs no allocation or collective and returns
    /// no error.
    ///
    /// # Panics
    ///
    /// Panics before any destination write if either length differs from the
    /// length validated during preparation.
    pub(crate) fn decode_into(mut self, packed: &[u8], destination: &mut [T]) {
        // shortcut: equal-length substitutions are undetected; revisit if prepare/decode move to separate operations.
        assert_eq!(packed.len(), self.bytes, "validated staging length changed");
        assert_eq!(
            destination.len(),
            self.elements,
            "validated destination length changed"
        );

        // Validated products and lengths bound all indices and byte ranges.
        for (physical_flat, slot) in destination.iter_mut().enumerate() {
            set_indices(
                physical_flat,
                &self.physical_dims,
                &mut self.physical_indices,
            );

            for (memory_axis, &logical_axis) in self.memory_to_logical.iter().enumerate() {
                self.logical_spatial[logical_axis] =
                    self.physical_indices[self.extra_rank + memory_axis];
            }

            let mut logical_flat = 0usize;
            for (axis, &extent) in self.logical_dims.iter().enumerate() {
                let index = if axis < self.extra_rank {
                    self.physical_indices[axis]
                } else {
                    self.logical_spatial[axis - self.extra_rank]
                };
                logical_flat = logical_flat * extent + index;
            }

            let start = logical_flat * T::WIDTH;
            let end = start + T::WIDTH;
            *slot = T::decode_le(&packed[start..end]);
        }
    }
}

#[cfg(all(test, feature = "mpi"))]
#[test]
fn physical_decode_length_mismatch_preserves_destination() {
    // Logical and physical shape [2] with the identity axis map.
    let plan = || PhysicalDecodePlan::<i32, 1> {
        logical_dims: vec![2],
        physical_dims: vec![2],
        physical_indices: vec![0],
        logical_spatial: [0],
        memory_to_logical: [0],
        extra_rank: 0,
        elements: 2,
        bytes: 8,
        _element: std::marker::PhantomData,
    };

    for (bytes, destination_len) in [(7usize, 2usize), (9, 2), (8, 1), (8, 3)] {
        let packed = vec![0u8; bytes];
        let mut destination = vec![-999i32; destination_len];
        let before = destination.clone();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            plan().decode_into(&packed, &mut destination);
        }));
        assert!(
            outcome.is_err(),
            "mismatched {bytes}-byte staging and {destination_len}-element destination must panic"
        );
        assert_eq!(
            destination, before,
            "mismatched {bytes}-byte staging and {destination_len}-element destination wrote"
        );
    }
}

#[cfg(feature = "mpi")]
fn set_indices(flat: usize, dims: &[usize], indices: &mut [usize]) {
    let mut remaining = flat;
    for axis in (0..dims.len()).rev() {
        indices[axis] = remaining % dims[axis];
        remaining /= dims[axis];
    }
}
