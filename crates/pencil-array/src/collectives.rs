//! Checked global reductions and root gathers for [`crate::PencilArrayView`] values.
//!
//! Reduction calls are collective on the view's topology Cartesian communicator.
//! Every rank must call the same operation in the same order with the same
//! communicator, layout, callback type, and correct MPI [`Equivalence`]
//! implementations. A callback is local code: it must not call MPI or panic.
//! `sum_by` and `norm_by` stage callback results, so the callback is invoked
//! exactly once per local element before the reduction starts.

use std::{
    any::type_name,
    mem::{align_of, size_of},
    ops::Range,
};

use mpi::{
    Count,
    collective::{CommunicatorCollectives, SystemOperation},
    datatype::Equivalence,
    topology::{CartesianCommunicator, Communicator, CommunicatorRelation},
    traits::{Destination, Source},
};
use num_complex::{Complex32, Complex64};

use crate::{
    CollectiveError, ExtraShape, Pencil, PencilArray, PencilArrayView, checked::checked_product,
    error::ArrayError, transpose::collective_valid, view::LocalArrayLayout,
};

const DESCRIPTOR_SCHEMA: u64 = 3;
const OPERATION_NAMESPACE: u64 = 0x4152_0000;
const OP_GLOBAL_SUM: u64 = OPERATION_NAMESPACE + 1;
const OP_GLOBAL_MIN: u64 = OPERATION_NAMESPACE + 2;
const OP_GLOBAL_MAX: u64 = OPERATION_NAMESPACE + 3;
const OP_L2_NORM: u64 = OPERATION_NAMESPACE + 4;
const OP_ANY: u64 = OPERATION_NAMESPACE + 5;
const OP_ALL: u64 = OPERATION_NAMESPACE + 6;
const OP_ANY_BY: u64 = OPERATION_NAMESPACE + 7;
const OP_ALL_BY: u64 = OPERATION_NAMESPACE + 8;
const OP_MAP_REDUCE2: u64 = 0x4d52_3201;
const OP_MAP_REDUCE3: u64 = 0x4d52_3301;
const OP_MAP_REDUCE_MANY: u64 = 0x4d52_4d01;
const OP_SUM_MANY: u64 = OPERATION_NAMESPACE + 16;
const OP_NORM_MANY: u64 = OPERATION_NAMESPACE + 17;
const OP_MIN_MANY: u64 = OPERATION_NAMESPACE + 18;
const OP_MAX_MANY: u64 = OPERATION_NAMESPACE + 19;
const OP_SUM_BY: u64 = OPERATION_NAMESPACE + 9;
const OP_NORM_BY: u64 = OPERATION_NAMESPACE + 10;
const OP_GATHER: u64 = OPERATION_NAMESPACE + 11;
const OP_MAPPED_MIN: u64 = OPERATION_NAMESPACE + 12;
const OP_MAPPED_MAX: u64 = OPERATION_NAMESPACE + 13;
const OP_ZIP_SUM_BY: u64 = OPERATION_NAMESPACE + 14;
const OP_ZIP_NORM_BY: u64 = OPERATION_NAMESPACE + 15;
const INVALID_WORD: u64 = u64::MAX;
const HEADER_WORDS: usize = 5;
const GATHER_SEED_TAG: mpi::Tag = 0x4741;
const GATHER_PAYLOAD_TAG: mpi::Tag = 0x4742;

mod sealed {
    pub trait Scalar {}
    pub trait Ordered {}
    pub trait NormOutput {}
    pub trait Truth {}
}

/// The real scalar returned by an L2 norm.
///
/// The trait is sealed. `f32` and `f64` are the only norm result types exposed
/// by this crate.
pub trait NormOutput: sealed::NormOutput + Equivalence + Copy + 'static {
    /// Converts the checked `f64` norm calculation to this result precision.
    fn from_f64(value: f64) -> Self;
}

impl NormOutput for f32 {
    fn from_f64(value: f64) -> Self {
        value as f32
    }
}

impl NormOutput for f64 {
    fn from_f64(value: f64) -> Self {
        value
    }
}

impl sealed::NormOutput for f32 {}
impl sealed::NormOutput for f64 {}

/// A scalar supported by global sum, norm, and mapped reduction operations.
///
/// Implementations are deliberately sealed to the crate's fixed scalar set:
/// `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `f32`, `f64`,
/// `Complex32`, and `Complex64`.
pub trait SupportedScalar: sealed::Scalar + Equivalence + Copy + 'static {
    /// The precision used for this scalar's L2 norm.
    type Norm: NormOutput;

    /// Returns the additive identity.
    fn zero() -> Self;

    /// Computes a checked local sum.
    fn local_sum(values: &[Self]) -> Result<Self, CollectiveError>;

    /// Completes a global sum from a locally checked partial.
    fn collective_sum<C: CommunicatorCollectives>(
        communicator: &C,
        local: Self,
        local_flags: [u32; 6],
    ) -> Result<Self, CollectiveError>;

    /// Prepares rank partial storage before a mapped callback is invoked.
    fn prepare_collective_sum<C: CommunicatorCollectives>(
        communicator: &C,
    ) -> Result<Vec<Self>, CollectiveError> {
        let _ = communicator;
        Ok(Vec::new())
    }

    /// Completes a sum using storage prepared before callbacks.
    fn collective_sum_prepared<C: CommunicatorCollectives>(
        communicator: &C,
        local: Self,
        local_flags: [u32; 6],
        partials: &mut Vec<Self>,
    ) -> Result<Self, CollectiveError> {
        let _ = partials;
        Self::collective_sum(communicator, local, local_flags)
    }
    /// Returns non-finite flags as `[NaN_re, +Inf_re, -Inf_re, NaN_im,
    /// +Inf_im, -Inf_im]`; real scalars use only the first three entries.
    fn nonfinite_flags(self) -> [u32; 6];
    /// Returns the absolute value used by the scaled norm algorithm.
    fn norm_abs(self) -> f64;

    /// Returns the scalar's Julia-like truth value for [`any`] and [`all`].
    fn truth(self) -> bool;
    /// Appends the exact scalar bits in a portable little-endian form.
    fn append_le_bits(self, bytes: &mut Vec<u8>);
}

/// A supported scalar for order-based global minimum and maximum.
///
/// Complex values intentionally do not implement this trait.
pub trait OrderedScalar: SupportedScalar + sealed::Ordered {
    /// Returns a NaN value for the explicit floating-point NaN policy.
    ///
    /// Integer implementations return an unused zero value because integers
    /// never set the global NaN flag.
    fn nan_value() -> Self;

    /// Returns a local min or max identity when `minimum` is true or false.
    fn local_extreme(values: &[Self], minimum: bool) -> Self;

    /// Performs the native MPI min or max reduction.
    fn collective_extreme<C: CommunicatorCollectives>(
        communicator: &C,
        local: Self,
        minimum: bool,
    ) -> Self;
}

/// A scalar with a global truth value for [`any`] and [`all`].
///
/// This includes the fixed numeric set, complex values, and `bool`. Arbitrary
/// input types can use [`any_by`] and [`all_by`] instead.
pub trait TruthValue: sealed::Truth + Copy + 'static {
    /// Returns whether one value is true for a global any/all operation.
    fn truth(self) -> bool;
}

macro_rules! impl_integer_scalar {
    ($($ty:ty),+ $(,)?) => {$(
        impl sealed::Scalar for $ty {}
        impl sealed::Ordered for $ty {}
        impl sealed::Truth for $ty {}

        impl SupportedScalar for $ty {
            type Norm = f64;

            fn zero() -> Self { 0 }

            fn local_sum(values: &[Self]) -> Result<Self, CollectiveError> {
                values.iter().try_fold(0 as $ty, |sum, &value| {
                    sum.checked_add(value).ok_or(CollectiveError::IntegerOverflow)
                })
            }

            fn collective_sum<C: CommunicatorCollectives>(
                communicator: &C,
                local: Self,
                _local_flags: [u32; 6],
            ) -> Result<Self, CollectiveError> {
                integer_collective_sum(communicator, local)
            }

            fn prepare_collective_sum<C: CommunicatorCollectives>(
                communicator: &C,
            ) -> Result<Vec<Self>, CollectiveError> {
                prepare_integer_collective_sum(communicator)
            }

            fn collective_sum_prepared<C: CommunicatorCollectives>(
                communicator: &C,
                local: Self,
                _local_flags: [u32; 6],
                partials: &mut Vec<Self>,
            ) -> Result<Self, CollectiveError> {
                integer_collective_sum_prepared(communicator, local, partials)
            }

            fn nonfinite_flags(self) -> [u32; 6] {
                let _ = self;
                [0, 0, 0, 0, 0, 0]
            }

            fn norm_abs(self) -> f64 {
                (self as f64).abs()
            }

            fn truth(self) -> bool {
                self != 0
            }
            fn append_le_bits(self, bytes: &mut Vec<u8>) { bytes.extend_from_slice(&self.to_le_bytes()); }
        }

        impl OrderedScalar for $ty {
            fn nan_value() -> Self { 0 }

            fn local_extreme(values: &[Self], minimum: bool) -> Self {
                let mut result = if minimum { <$ty>::MAX } else { <$ty>::MIN };
                for &value in values {
                    result = if minimum { result.min(value) } else { result.max(value) };
                }
                result
            }

            fn collective_extreme<C: CommunicatorCollectives>(
                communicator: &C,
                local: Self,
                minimum: bool,
            ) -> Self {
                let mut result = local;
                let operation = if minimum { SystemOperation::min() } else { SystemOperation::max() };
                communicator.all_reduce_into(&local, &mut result, operation);
                result
            }
        }

        impl TruthValue for $ty {
            fn truth(self) -> bool { <Self as SupportedScalar>::truth(self) }
        }
    )+};
}

macro_rules! impl_float_scalar {
    ($ty:ty, $norm:ty) => {
        impl sealed::Scalar for $ty {}
        impl sealed::Ordered for $ty {}
        impl sealed::Truth for $ty {}

        impl SupportedScalar for $ty {
            type Norm = $norm;

            fn zero() -> Self {
                0.0
            }

            fn local_sum(values: &[Self]) -> Result<Self, CollectiveError> {
                Ok(values.iter().copied().fold(0.0, |sum, value| sum + value))
            }

            fn collective_sum<C: CommunicatorCollectives>(
                communicator: &C,
                local: Self,
                local_flags: [u32; 6],
            ) -> Result<Self, CollectiveError> {
                let mut result = local;
                communicator.all_reduce_into(&local, &mut result, SystemOperation::sum());
                // Flags describe input values, not overflowed local partials.
                // A finite partial must not manufacture the opposite infinity
                // sign and override a globally declared infinity.
                let global_flags = collective_flags(communicator, local_flags);
                if global_flags[0] != 0 {
                    Ok(<$ty>::NAN)
                } else if global_flags[1] != 0 && global_flags[2] != 0 {
                    Ok(<$ty>::NAN)
                } else if global_flags[1] != 0 {
                    Ok(<$ty>::INFINITY)
                } else if global_flags[2] != 0 {
                    Ok(<$ty>::NEG_INFINITY)
                } else {
                    Ok(result)
                }
            }

            fn nonfinite_flags(self) -> [u32; 6] {
                [
                    u32::from(self.is_nan()),
                    u32::from(self.is_infinite() && self.is_sign_positive()),
                    u32::from(self.is_infinite() && self.is_sign_negative()),
                    0,
                    0,
                    0,
                ]
            }

            fn norm_abs(self) -> f64 {
                (self as f64).abs()
            }

            fn truth(self) -> bool {
                self != 0.0
            }
            fn append_le_bits(self, bytes: &mut Vec<u8>) {
                bytes.extend_from_slice(&self.to_bits().to_le_bytes());
            }
        }

        impl OrderedScalar for $ty {
            fn nan_value() -> Self {
                <$ty>::NAN
            }

            fn local_extreme(values: &[Self], minimum: bool) -> Self {
                let mut result = if minimum {
                    <$ty>::INFINITY
                } else {
                    <$ty>::NEG_INFINITY
                };
                for &value in values {
                    if !value.is_nan() {
                        result = if minimum {
                            result.min(value)
                        } else {
                            result.max(value)
                        };
                    }
                }
                result
            }

            fn collective_extreme<C: CommunicatorCollectives>(
                communicator: &C,
                local: Self,
                minimum: bool,
            ) -> Self {
                let mut result = local;
                let operation = if minimum {
                    SystemOperation::min()
                } else {
                    SystemOperation::max()
                };
                communicator.all_reduce_into(&local, &mut result, operation);
                result
            }
        }

        impl TruthValue for $ty {
            fn truth(self) -> bool {
                <Self as SupportedScalar>::truth(self)
            }
        }
    };
}

impl_integer_scalar!(i8, i16, i32, i64, u8, u16, u32, u64);
impl_float_scalar!(f32, f32);
impl_float_scalar!(f64, f64);

impl sealed::Scalar for Complex32 {}
impl sealed::Truth for Complex32 {}
impl SupportedScalar for Complex32 {
    type Norm = f32;

    fn zero() -> Self {
        Complex32::new(0.0, 0.0)
    }

    fn local_sum(values: &[Self]) -> Result<Self, CollectiveError> {
        Ok(values
            .iter()
            .copied()
            .fold(Self::zero(), |sum, value| sum + value))
    }

    fn collective_sum<C: CommunicatorCollectives>(
        communicator: &C,
        local: Self,
        local_flags: [u32; 6],
    ) -> Result<Self, CollectiveError> {
        let mut result = local;
        communicator.all_reduce_into(&local, &mut result, SystemOperation::sum());
        let global_flags = collective_flags(communicator, local_flags);
        let real = sum_component_f32(result.re, global_flags[0], global_flags[1], global_flags[2]);
        let imaginary =
            sum_component_f32(result.im, global_flags[3], global_flags[4], global_flags[5]);
        Ok(Self::new(real, imaginary))
    }

    fn nonfinite_flags(self) -> [u32; 6] {
        complex_flags_f32(self)
    }

    fn norm_abs(self) -> f64 {
        f64::from(self.re).hypot(f64::from(self.im))
    }

    fn truth(self) -> bool {
        self.re != 0.0 || self.im != 0.0
    }
    fn append_le_bits(self, bytes: &mut Vec<u8>) {
        bytes.extend_from_slice(&self.re.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.im.to_bits().to_le_bytes());
    }
}

impl TruthValue for Complex32 {
    fn truth(self) -> bool {
        <Self as SupportedScalar>::truth(self)
    }
}

impl sealed::Scalar for Complex64 {}
impl sealed::Truth for Complex64 {}
impl SupportedScalar for Complex64 {
    type Norm = f64;

    fn zero() -> Self {
        Complex64::new(0.0, 0.0)
    }

    fn local_sum(values: &[Self]) -> Result<Self, CollectiveError> {
        Ok(values
            .iter()
            .copied()
            .fold(Self::zero(), |sum, value| sum + value))
    }

    fn collective_sum<C: CommunicatorCollectives>(
        communicator: &C,
        local: Self,
        local_flags: [u32; 6],
    ) -> Result<Self, CollectiveError> {
        let mut result = local;
        communicator.all_reduce_into(&local, &mut result, SystemOperation::sum());
        let global_flags = collective_flags(communicator, local_flags);
        let real = sum_component_f64(result.re, global_flags[0], global_flags[1], global_flags[2]);
        let imaginary =
            sum_component_f64(result.im, global_flags[3], global_flags[4], global_flags[5]);
        Ok(Self::new(real, imaginary))
    }

    fn nonfinite_flags(self) -> [u32; 6] {
        complex_flags_f64(self)
    }

    fn norm_abs(self) -> f64 {
        self.re.hypot(self.im)
    }

    fn truth(self) -> bool {
        self.re != 0.0 || self.im != 0.0
    }
    fn append_le_bits(self, bytes: &mut Vec<u8>) {
        bytes.extend_from_slice(&self.re.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.im.to_bits().to_le_bytes());
    }
}

impl TruthValue for Complex64 {
    fn truth(self) -> bool {
        <Self as SupportedScalar>::truth(self)
    }
}

impl sealed::Truth for bool {}
impl TruthValue for bool {
    fn truth(self) -> bool {
        self
    }
}

/// Computes the replicated global sum of a view.
///
/// Floating-point NaN wins globally. Opposite infinity signs produce NaN;
/// otherwise a single infinity sign is preserved. Complex components apply
/// those rules independently.
pub fn global_sum<T, const N: usize, const M: usize>(
    view: &PencilArrayView<'_, T, N, M>,
) -> Result<T, CollectiveError>
where
    T: SupportedScalar,
{
    sum_layout(view, OP_GLOBAL_SUM)
}

/// Computes the replicated global minimum. `None` is returned for a globally
/// empty extra shape.
pub fn global_min<T, const N: usize, const M: usize>(
    view: &PencilArrayView<'_, T, N, M>,
) -> Result<Option<T>, CollectiveError>
where
    T: OrderedScalar,
{
    extreme_layout(view, OP_GLOBAL_MIN, true)
}

/// Computes the replicated global maximum. `None` is returned for a globally
/// empty extra shape.
pub fn global_max<T, const N: usize, const M: usize>(
    view: &PencilArrayView<'_, T, N, M>,
) -> Result<Option<T>, CollectiveError>
where
    T: OrderedScalar,
{
    extreme_layout(view, OP_GLOBAL_MAX, false)
}

/// Computes a replicated scaled global L2 norm.
///
/// NaN produces NaN and any infinity produces positive infinity.
///
/// The callback-free form invokes no user code. The implementation combines a
/// global maximum absolute value with a globally reduced sum of scaled squares;
/// it does not form avoidable intermediate squares of the original values.
pub fn l2_norm<T, const N: usize, const M: usize>(
    view: &PencilArrayView<'_, T, N, M>,
) -> Result<T::Norm, CollectiveError>
where
    T: SupportedScalar,
{
    norm_layout(view, OP_L2_NORM)
}

/// Computes global any using the fixed scalar truth rule.
pub fn any<T, const N: usize, const M: usize>(
    view: &PencilArrayView<'_, T, N, M>,
) -> Result<bool, CollectiveError>
where
    T: TruthValue,
{
    truth_layout(view, OP_ANY, true)
}

/// Computes global all using the fixed scalar truth rule.
pub fn all<T, const N: usize, const M: usize>(
    view: &PencilArrayView<'_, T, N, M>,
) -> Result<bool, CollectiveError>
where
    T: TruthValue,
{
    truth_layout(view, OP_ALL, false)
}

/// Computes global any after applying `predicate` once to every local value.
///
/// `predicate` must not panic, call MPI, or depend on callback invocation
/// order beyond the local row-major storage order.
pub fn any_by<T, F, const N: usize, const M: usize>(
    view: &PencilArrayView<'_, T, N, M>,
    mut predicate: F,
) -> Result<bool, CollectiveError>
where
    F: FnMut(&T) -> bool,
{
    by_truth_layout(view, OP_ANY_BY, true, |value| predicate(value))
}

/// Computes global all after applying `predicate` once to every local value.
///
/// `predicate` must not panic, call MPI, or depend on callback invocation
/// order beyond the local row-major storage order.
pub fn all_by<T, F, const N: usize, const M: usize>(
    view: &PencilArrayView<'_, T, N, M>,
    mut predicate: F,
) -> Result<bool, CollectiveError>
where
    F: FnMut(&T) -> bool,
{
    by_truth_layout(view, OP_ALL_BY, false, |value| predicate(value))
}

/// Computes a global sum after invoking `f` once per local input value.
pub fn sum_by<T, U, F, const N: usize, const M: usize>(
    view: &PencilArrayView<'_, T, N, M>,
    f: F,
) -> Result<U, CollectiveError>
where
    U: SupportedScalar,
    F: FnMut(&T) -> U,
{
    mapped_sum_layout(view, OP_SUM_BY, f)
}

/// Computes a global L2 norm after invoking `f` once per local input value.
pub fn norm_by<T, U, F, const N: usize, const M: usize>(
    view: &PencilArrayView<'_, T, N, M>,
    f: F,
) -> Result<U::Norm, CollectiveError>
where
    U: SupportedScalar,
    F: FnMut(&T) -> U,
{
    mapped_norm_layout(view, OP_NORM_BY, f)
}

/// Computes a checked global minimum after mapping each value in memory order.
/// Empty global inputs return `None`; mapped NaNs propagate. Uses O(local length)
/// staging. The callback must not panic or call MPI.
pub fn min_by<T, U, F, const N: usize, const M: usize>(
    view: &PencilArrayView<'_, T, N, M>,
    f: F,
) -> Result<Option<U>, CollectiveError>
where
    U: OrderedScalar,
    F: FnMut(&T) -> U,
{
    mapped_extreme_layout(view, OP_MAPPED_MIN, true, f)
}

/// Computes a checked global maximum after mapping each value in memory order.
/// Empty global inputs return `None`; mapped NaNs propagate. Uses O(local length)
/// staging. The callback must not panic or call MPI.
pub fn max_by<T, U, F, const N: usize, const M: usize>(
    view: &PencilArrayView<'_, T, N, M>,
    f: F,
) -> Result<Option<U>, CollectiveError>
where
    U: OrderedScalar,
    F: FnMut(&T) -> U,
{
    mapped_extreme_layout(view, OP_MAPPED_MAX, false, f)
}

/// Computes a checked global sum of mapped pairs.
///
/// Traversal follows local row-major memory order (with broadcast dimensions
/// revisited). Metadata costs O(extra-rank), while staged mapped values use
/// O(localcount) storage. Captured callback state cannot be verified
/// collectively; the callback must not panic or call MPI.
pub fn zip_sum_by<A, B, U, F, const N: usize, const M: usize>(
    left: &PencilArrayView<'_, A, N, M>,
    right: &PencilArrayView<'_, B, N, M>,
    f: F,
) -> Result<U, CollectiveError>
where
    U: SupportedScalar,
    F: FnMut(&A, &B) -> U,
{
    let communicator = left.pencil().topology().communicator();
    let plan = prepare_map_reduce2(left, right, U::zero(), type_name::<F>(), OP_ZIP_SUM_BY)?;
    let mut partials = U::prepare_collective_sum(communicator)?;
    let mapped = map_pairs_collective(left, right, &plan, f)?;
    let mapped = agree_staged(communicator, Ok(mapped))?;
    sum_values_prepared(communicator, &mapped, &mut partials)
}

/// Computes a scaled global L2 norm of mapped pairs.
///
/// Traversal follows local row-major memory order (with broadcast dimensions
/// revisited). Metadata costs O(extra-rank), while staged mapped values use
/// O(localcount) storage. Captured callback state cannot be verified
/// collectively; the callback must not panic or call MPI.
pub fn zip_norm_by<A, B, U, F, const N: usize, const M: usize>(
    left: &PencilArrayView<'_, A, N, M>,
    right: &PencilArrayView<'_, B, N, M>,
    f: F,
) -> Result<U::Norm, CollectiveError>
where
    U: SupportedScalar,
    F: FnMut(&A, &B) -> U,
{
    let plan = prepare_map_reduce2(left, right, U::zero(), type_name::<F>(), OP_ZIP_NORM_BY)?;
    let mapped = map_pairs_collective(left, right, &plan, f)?;
    let mapped = agree_staged(left.pencil().topology().communicator(), Ok(mapped))?;
    let prepared = PreparedLayout {
        global_count: plan.global_count,
    };
    norm_values(left.pencil().topology().communicator(), &mapped, prepared)
}

/// Maps three locally aligned inputs and folds their mapped triples in rank order.
/// The reducer must be deterministic, associative, and neutral-compatible; its
/// result must not depend on mutable invocation count or order (observational
/// counters are fine). All ranks must use matching callbacks. Map and reduce
/// callbacks must not call MPI or panic; no recovery is guaranteed if they do.
pub fn map_reduce3<A, B, D, U, F, R, const N: usize, const M: usize>(
    communicator: &CartesianCommunicator,
    first: &PencilArrayView<'_, A, N, M>,
    second: &PencilArrayView<'_, B, N, M>,
    third: &PencilArrayView<'_, D, N, M>,
    neutral: U,
    mut map: F,
    mut reduce: R,
) -> Result<U, CollectiveError>
where
    U: SupportedScalar,
    F: FnMut(&A, &B, &D) -> U,
    R: FnMut(U, U) -> U,
{
    let plan = prepare_map_reduce3(
        communicator,
        first,
        second,
        third,
        neutral,
        type_name::<(F, R)>(),
    )?;
    let size = usize::try_from(communicator.size()).map_err(|_| CollectiveError::CountOverflow)?;
    let mut partials = Vec::new();
    let allocation = partials.try_reserve_exact(size);
    if !collective_valid(communicator, allocation.is_ok()) {
        return Err(allocation
            .err()
            .map_or(CollectiveError::CollectivePreconditionFailed, |_| {
                CollectiveError::AllocationFailed { elements: size }
            }));
    }
    partials.resize(size, neutral);
    let mut local = neutral;
    for_each_triple(first, second, third, &plan, |a, b, d| {
        local = reduce(local, map(a, b, d))
    });
    communicator.all_gather_into(&local, &mut partials[..]);
    Ok(partials.into_iter().fold(neutral, reduce))
}

/// Maps any non-empty collection of homogeneous views and folds it in rank order.
/// `map` receives reusable references; it must not retain them, call MPI, or panic.
/// The reducer must be deterministic, associative, and neutral-compatible; its
/// result must not depend on mutable invocation count or order (observational
/// counters are fine). All ranks must use matching callbacks. No recovery is
/// guaranteed if a callback calls MPI or panics.
pub fn map_reduce_many<T, U, F, R, const N: usize, const M: usize>(
    communicator: &CartesianCommunicator,
    inputs: &[PencilArrayView<'_, T, N, M>],
    neutral: U,
    mut map: F,
    mut reduce: R,
) -> Result<U, CollectiveError>
where
    T: 'static,
    U: SupportedScalar,
    F: FnMut(&[&T]) -> U,
    R: FnMut(U, U) -> U,
{
    let plan = prepare_map_reduce_many(
        communicator,
        inputs,
        neutral,
        type_name::<(F, R)>(),
        OP_MAP_REDUCE_MANY,
    )?;
    let size = usize::try_from(communicator.size()).map_err(|_| CollectiveError::CountOverflow)?;
    let mut partials = Vec::new();
    let partials_ok = partials.try_reserve_exact(size).is_ok();
    let mut refs: Vec<&T> = Vec::new();
    let refs_ok = refs.try_reserve_exact(inputs.len()).is_ok();
    if !collective_valid(communicator, partials_ok && refs_ok) {
        return Err(if !partials_ok {
            CollectiveError::AllocationFailed { elements: size }
        } else if !refs_ok {
            CollectiveError::AllocationFailed {
                elements: inputs.len(),
            }
        } else {
            CollectiveError::CollectivePreconditionFailed
        });
    }
    partials.resize(size, neutral);
    let mut local = neutral;
    for_each_many(inputs, &plan, &mut refs, |values| {
        local = reduce(local, map(values))
    });
    communicator.all_gather_into(&local, &mut partials[..]);
    Ok(partials.into_iter().fold(neutral, reduce))
}

/// Sums values produced by mapping each tuple of homogeneous inputs.
/// Integer accumulation is checked; floating-point nonfinite policies match [`sum_by`].
/// Callbacks must have identical semantics on all ranks, must not call MPI or
/// panic. All preparation is agreed before mapping; callback panic has no
/// collective recovery guarantee.
pub fn sum_many_by<T, U, F, const N: usize, const M: usize>(
    communicator: &CartesianCommunicator,
    inputs: &[PencilArrayView<'_, T, N, M>],
    f: F,
) -> Result<U, CollectiveError>
where
    T: 'static,
    U: SupportedScalar,
    F: FnMut(&[&T]) -> U,
{
    let plan = prepare_map_reduce_many(
        communicator,
        inputs,
        U::zero(),
        type_name::<F>(),
        OP_SUM_MANY,
    )?;
    let mut partials = U::prepare_collective_sum(communicator)?;
    let values = map_many_values_prepared(communicator, inputs, &plan, f)?;
    sum_values_prepared(communicator, &values, &mut partials)
}

/// Computes a scaled norm after mapping each tuple of homogeneous inputs.
/// Uses the scaled and nonfinite policies of [`norm_by`]. The callback contract
/// and preflight guarantees are those of [`sum_many_by`].
/// The callback must not call MPI or panic; no recovery is guaranteed if it does.
pub fn norm_many_by<T, U, F, const N: usize, const M: usize>(
    communicator: &CartesianCommunicator,
    inputs: &[PencilArrayView<'_, T, N, M>],
    f: F,
) -> Result<U::Norm, CollectiveError>
where
    T: 'static,
    U: SupportedScalar,
    F: FnMut(&[&T]) -> U,
{
    let plan = prepare_map_reduce_many(
        communicator,
        inputs,
        U::zero(),
        type_name::<F>(),
        OP_NORM_MANY,
    )?;
    let values = map_many_values_prepared(communicator, inputs, &plan, f)?;
    norm_values(
        communicator,
        &values,
        PreparedLayout {
            global_count: plan.global_count,
        },
    )
}

/// Computes the minimum after mapping each tuple of homogeneous inputs.
/// Preserves [`min_by`]'s NaN policy and globally-empty `None` result.
/// The callback contract and preflight guarantees are those of [`sum_many_by`].
/// The callback must not call MPI or panic; no recovery is guaranteed if it does.
pub fn min_many_by<T, U, F, const N: usize, const M: usize>(
    communicator: &CartesianCommunicator,
    inputs: &[PencilArrayView<'_, T, N, M>],
    f: F,
) -> Result<Option<U>, CollectiveError>
where
    T: 'static,
    U: OrderedScalar,
    F: FnMut(&[&T]) -> U,
{
    let plan = prepare_map_reduce_many(
        communicator,
        inputs,
        U::zero(),
        type_name::<F>(),
        OP_MIN_MANY,
    )?;
    let values = map_many_values_prepared(communicator, inputs, &plan, f)?;
    let local = U::local_extreme(&values, true);
    let result = U::collective_extreme(communicator, local, true);
    let flags = collective_flags(communicator, values_nonfinite_flags(values.iter().copied()));
    Ok((plan.global_count != 0).then_some(if flags[0] != 0 {
        U::nan_value()
    } else {
        result
    }))
}

/// Computes the maximum after mapping each tuple of homogeneous inputs.
/// Preserves [`max_by`]'s NaN policy and globally-empty `None` result.
/// The callback contract and preflight guarantees are those of [`sum_many_by`].
/// The callback must not call MPI or panic; no recovery is guaranteed if it does.
pub fn max_many_by<T, U, F, const N: usize, const M: usize>(
    communicator: &CartesianCommunicator,
    inputs: &[PencilArrayView<'_, T, N, M>],
    f: F,
) -> Result<Option<U>, CollectiveError>
where
    T: 'static,
    U: OrderedScalar,
    F: FnMut(&[&T]) -> U,
{
    let plan = prepare_map_reduce_many(
        communicator,
        inputs,
        U::zero(),
        type_name::<F>(),
        OP_MAX_MANY,
    )?;
    let values = map_many_values_prepared(communicator, inputs, &plan, f)?;
    let local = U::local_extreme(&values, false);
    let result = U::collective_extreme(communicator, local, false);
    let flags = collective_flags(communicator, values_nonfinite_flags(values.iter().copied()));
    Ok((plan.global_count != 0).then_some(if flags[0] != 0 {
        U::nan_value()
    } else {
        result
    }))
}

/// Maps two locally aligned inputs and folds their mapped pairs in rank order.
/// Traversal follows local row-major memory order (with broadcast dimensions
/// revisited). Shape/stride metadata costs O(extra-rank); unlike staged zip
/// reductions, this implementation uses one partial per rank, so its extra
/// storage and communication cost is O(P). Captured callback state cannot be
/// verified collectively; callbacks must be associative, neutral-compatible,
/// non-panicking, and must not call MPI.
pub fn map_reduce2<A, B, U, F, R, const N: usize, const M: usize>(
    left: &PencilArrayView<'_, A, N, M>,
    right: &PencilArrayView<'_, B, N, M>,
    neutral: U,
    mut map: F,
    mut reduce: R,
) -> Result<U, CollectiveError>
where
    U: SupportedScalar,
    F: FnMut(&A, &B) -> U,
    R: FnMut(U, U) -> U,
{
    let plan = prepare_map_reduce2(left, right, neutral, type_name::<(F, R)>(), OP_MAP_REDUCE2)?;
    let communicator = left.pencil().topology().communicator();
    let size = usize::try_from(communicator.size()).map_err(|_| CollectiveError::CountOverflow)?;
    let mut partials = Vec::new();
    let allocation = partials.try_reserve_exact(size);
    if !collective_valid(communicator, allocation.is_ok()) {
        return Err(allocation
            .err()
            .map_or(CollectiveError::CollectivePreconditionFailed, |_| {
                CollectiveError::AllocationFailed { elements: size }
            }));
    }
    partials.resize(size, neutral);
    // The allocation agreement above is deliberately before the first callback.
    let mut local = neutral;
    for_each_pair(left, right, &plan, |a, b| {
        local = reduce(local, map(a, b));
    });
    communicator.all_gather_into(&local, &mut partials[..]);
    Ok(partials.into_iter().fold(neutral, reduce))
}

/// Gathers a view to `root` in global logical row-major order.
///
/// The root is a rank in the view's topology Cartesian communicator and may be
/// supplied as any integer type convertible to `usize`. Non-root ranks return
/// `Ok(None)`. The source is preserved. Root allocation and all local MPI
/// counts are checked and agreed before the first payload message. Payload
/// messages use a duplicated communicator context, isolating the internal
/// tags from user traffic on the topology communicator.
pub fn gather<T, R, const N: usize, const M: usize>(
    view: &PencilArrayView<'_, T, N, M>,
    root: R,
) -> Result<Option<Vec<T>>, CollectiveError>
where
    T: Copy + Equivalence,
    R: TryInto<usize> + Copy,
{
    gather_layout(view, root)
}

impl<T, const N: usize, const M: usize> PencilArrayView<'_, T, N, M> {
    /// See [`global_sum`].
    pub fn global_sum(&self) -> Result<T, CollectiveError>
    where
        T: SupportedScalar,
    {
        global_sum(self)
    }

    /// See [`global_min`].
    pub fn global_min(&self) -> Result<Option<T>, CollectiveError>
    where
        T: OrderedScalar,
    {
        global_min(self)
    }

    /// See [`global_max`].
    pub fn global_max(&self) -> Result<Option<T>, CollectiveError>
    where
        T: OrderedScalar,
    {
        global_max(self)
    }

    /// See [`l2_norm`].
    pub fn l2_norm(&self) -> Result<T::Norm, CollectiveError>
    where
        T: SupportedScalar,
    {
        l2_norm(self)
    }

    /// See [`any`].
    pub fn any(&self) -> Result<bool, CollectiveError>
    where
        T: TruthValue,
    {
        any(self)
    }

    /// See [`all`].
    pub fn all(&self) -> Result<bool, CollectiveError>
    where
        T: TruthValue,
    {
        all(self)
    }

    /// See [`any_by`].
    pub fn any_by<F>(&self, predicate: F) -> Result<bool, CollectiveError>
    where
        F: FnMut(&T) -> bool,
    {
        any_by(self, predicate)
    }

    /// See [`all_by`].
    pub fn all_by<F>(&self, predicate: F) -> Result<bool, CollectiveError>
    where
        F: FnMut(&T) -> bool,
    {
        all_by(self, predicate)
    }

    /// See [`sum_by`].
    pub fn sum_by<U, F>(&self, f: F) -> Result<U, CollectiveError>
    where
        U: SupportedScalar,
        F: FnMut(&T) -> U,
    {
        sum_by(self, f)
    }

    /// See [`norm_by`].
    pub fn norm_by<U, F>(&self, f: F) -> Result<U::Norm, CollectiveError>
    where
        U: SupportedScalar,
        F: FnMut(&T) -> U,
    {
        norm_by(self, f)
    }

    /// See [`map_reduce3`].
    pub fn map_reduce3<B, D, U, F, R>(
        &self,
        second: &PencilArrayView<'_, B, N, M>,
        third: &PencilArrayView<'_, D, N, M>,
        neutral: U,
        map: F,
        reduce: R,
    ) -> Result<U, CollectiveError>
    where
        U: SupportedScalar,
        F: FnMut(&T, &B, &D) -> U,
        R: FnMut(U, U) -> U,
    {
        map_reduce3(
            self.pencil().topology().communicator(),
            self,
            second,
            third,
            neutral,
            map,
            reduce,
        )
    }

    /// See [`map_reduce2`].
    pub fn map_reduce2<B, U, F, R>(
        &self,
        right: &PencilArrayView<'_, B, N, M>,
        neutral: U,
        map: F,
        reduce: R,
    ) -> Result<U, CollectiveError>
    where
        U: SupportedScalar,
        F: FnMut(&T, &B) -> U,
        R: FnMut(U, U) -> U,
    {
        map_reduce2(self, right, neutral, map, reduce)
    }

    /// See [`min_by`].
    pub fn min_by<U, F>(&self, f: F) -> Result<Option<U>, CollectiveError>
    where
        U: OrderedScalar,
        F: FnMut(&T) -> U,
    {
        min_by(self, f)
    }

    /// See [`max_by`].
    pub fn max_by<U, F>(&self, f: F) -> Result<Option<U>, CollectiveError>
    where
        U: OrderedScalar,
        F: FnMut(&T) -> U,
    {
        max_by(self, f)
    }

    /// See [`gather`].
    pub fn gather<R>(&self, root: R) -> Result<Option<Vec<T>>, CollectiveError>
    where
        T: Copy + Equivalence,
        R: TryInto<usize> + Copy,
    {
        gather(self, root)
    }
}

impl<T, const N: usize, const M: usize> PencilArray<T, N, M> {
    /// See [`global_sum`].
    pub fn global_sum(&self) -> Result<T, CollectiveError>
    where
        T: SupportedScalar,
    {
        global_sum(&self.view())
    }

    /// See [`global_min`].
    pub fn global_min(&self) -> Result<Option<T>, CollectiveError>
    where
        T: OrderedScalar,
    {
        global_min(&self.view())
    }

    /// See [`global_max`].
    pub fn global_max(&self) -> Result<Option<T>, CollectiveError>
    where
        T: OrderedScalar,
    {
        global_max(&self.view())
    }

    /// See [`l2_norm`].
    pub fn l2_norm(&self) -> Result<T::Norm, CollectiveError>
    where
        T: SupportedScalar,
    {
        l2_norm(&self.view())
    }

    /// See [`any`].
    pub fn any(&self) -> Result<bool, CollectiveError>
    where
        T: TruthValue,
    {
        any(&self.view())
    }

    /// See [`all`].
    pub fn all(&self) -> Result<bool, CollectiveError>
    where
        T: TruthValue,
    {
        all(&self.view())
    }

    /// See [`any_by`].
    pub fn any_by<F>(&self, predicate: F) -> Result<bool, CollectiveError>
    where
        F: FnMut(&T) -> bool,
    {
        any_by(&self.view(), predicate)
    }

    /// See [`all_by`].
    pub fn all_by<F>(&self, predicate: F) -> Result<bool, CollectiveError>
    where
        F: FnMut(&T) -> bool,
    {
        all_by(&self.view(), predicate)
    }

    /// See [`sum_by`].
    pub fn sum_by<U, F>(&self, f: F) -> Result<U, CollectiveError>
    where
        U: SupportedScalar,
        F: FnMut(&T) -> U,
    {
        sum_by(&self.view(), f)
    }

    /// See [`norm_by`].
    pub fn norm_by<U, F>(&self, f: F) -> Result<U::Norm, CollectiveError>
    where
        U: SupportedScalar,
        F: FnMut(&T) -> U,
    {
        norm_by(&self.view(), f)
    }

    /// See [`map_reduce3`].
    pub fn map_reduce3<B, D, U, F, R>(
        &self,
        second: &PencilArray<B, N, M>,
        third: &PencilArray<D, N, M>,
        neutral: U,
        map: F,
        reduce: R,
    ) -> Result<U, CollectiveError>
    where
        U: SupportedScalar,
        F: FnMut(&T, &B, &D) -> U,
        R: FnMut(U, U) -> U,
    {
        map_reduce3(
            self.pencil().topology().communicator(),
            &self.view(),
            &second.view(),
            &third.view(),
            neutral,
            map,
            reduce,
        )
    }

    /// See [`map_reduce2`].
    pub fn map_reduce2<B, U, F, R>(
        &self,
        right: &PencilArray<B, N, M>,
        neutral: U,
        map: F,
        reduce: R,
    ) -> Result<U, CollectiveError>
    where
        U: SupportedScalar,
        F: FnMut(&T, &B) -> U,
        R: FnMut(U, U) -> U,
    {
        map_reduce2(&self.view(), &right.view(), neutral, map, reduce)
    }

    /// See [`gather`].
    pub fn gather<R>(&self, root: R) -> Result<Option<Vec<T>>, CollectiveError>
    where
        T: Copy + Equivalence,
        R: TryInto<usize> + Copy,
    {
        gather(&self.view(), root)
    }
}

#[derive(Debug)]
struct MapPlan {
    shape: Vec<usize>,
    left_strides: Vec<usize>,
    right_strides: Vec<usize>,
    output_strides: Vec<usize>,
    spatial: usize,
    count: usize,
    global_count: usize,
}

#[derive(Debug)]
struct MapPlan3 {
    shape: Vec<usize>,
    strides: [Vec<usize>; 3],
    output_strides: Vec<usize>,
    spatial: usize,
    count: usize,
}

#[derive(Debug)]
struct ManyPlan {
    shape: Vec<usize>,
    strides: Vec<Vec<usize>>,
    output_strides: Vec<usize>,
    spatial: usize,
    extra: usize,
    count: usize,
    global_count: usize,
}

fn prepare_map_reduce3<A, B, D, U, const N: usize, const M: usize>(
    c: &CartesianCommunicator,
    a: &PencilArrayView<'_, A, N, M>,
    b: &PencilArrayView<'_, B, N, M>,
    d: &PencilArrayView<'_, D, N, M>,
    neutral: U,
    callback: &str,
) -> Result<MapPlan3, CollectiveError>
where
    U: SupportedScalar,
{
    let header = [
        DESCRIPTOR_SCHEMA,
        OP_MAP_REDUCE3,
        u64::try_from(N).unwrap_or(INVALID_WORD),
        u64::try_from(M).unwrap_or(INVALID_WORD),
        0,
    ];
    if !agree_header(c, header) {
        return Err(CollectiveError::CollectiveDescriptorMismatch);
    }
    let lengths = [
        descriptor_len::<A, U, N, M>(a, 0, callback),
        descriptor_len::<B, U, N, M>(b, 0, callback),
        descriptor_len::<D, U, N, M>(d, 0, callback),
    ];
    let expected = size_of::<U>().checked_mul(2).ok_or(()).and_then(|base| {
        lengths.iter().try_fold(base, |n, x| {
            n.checked_add(*x.as_ref().map_err(|_| ())?).ok_or(())
        })
    });
    let header = [
        DESCRIPTOR_SCHEMA,
        OP_MAP_REDUCE3,
        u64::try_from(N).unwrap_or(INVALID_WORD),
        u64::try_from(M).unwrap_or(INVALID_WORD),
        expected
            .as_ref()
            .ok()
            .and_then(|n| u64::try_from(*n).ok())
            .unwrap_or(INVALID_WORD),
    ];
    if !agree_header(c, header) {
        return Err(CollectiveError::CollectiveDescriptorMismatch);
    }
    let descriptor = (|| {
        let mut out = Vec::new();
        out.try_reserve_exact(expected.map_err(|_| ())?)
            .map_err(|_| ())?;
        let words_a = build_descriptor::<_, A, U, N, M>(a, OP_MAP_REDUCE3, 0, callback)?;
        let words_b = build_descriptor::<_, B, U, N, M>(b, OP_MAP_REDUCE3, 0, callback)?;
        let words_d = build_descriptor::<_, D, U, N, M>(d, OP_MAP_REDUCE3, 0, callback)?;
        for (words, len) in [
            (words_a, lengths[0]),
            (words_b, lengths[1]),
            (words_d, lengths[2]),
        ] {
            if Some(words.len()) != len.ok() {
                return Err(());
            }
            out.extend(words);
        }
        for byte in neutral_bytes(neutral)? {
            append_word(&mut out, u64::from(byte));
        }
        Ok(out)
    })();
    let _ = collective_descriptor(c, descriptor.ok(), expected.ok())?;
    let shape = broadcast_shape3(a.extra_shape(), b.extra_shape(), d.extra_shape());
    if !collective_valid(c, shape.is_ok()) {
        return Err(CollectiveError::CollectivePreconditionFailed);
    }
    let shape = shape?;
    let valid = a.pencil().same_layout(b.pencil())
        && a.pencil().same_layout(d.pencil())
        && matches!(
            c.compare(a.pencil().topology().communicator()),
            CommunicatorRelation::Identical | CommunicatorRelation::Congruent
        )
        && matches!(
            c.compare(b.pencil().topology().communicator()),
            CommunicatorRelation::Identical | CommunicatorRelation::Congruent
        )
        && matches!(
            c.compare(d.pencil().topology().communicator()),
            CommunicatorRelation::Identical | CommunicatorRelation::Congruent
        )
        && shape.is_some();
    if !collective_valid(c, valid) || !valid {
        return Err(CollectiveError::CollectivePreconditionFailed);
    }
    let shape = shape.unwrap();
    let dims = [
        a.extra_shape().dimensions(),
        b.extra_shape().dimensions(),
        d.extra_shape().dimensions(),
    ];
    let strides = dims.map(strides_for_dims);
    let output_strides = strides_for_dims(&shape);
    let ok = strides.iter().all(Result::is_ok) && output_strides.is_ok();
    if !collective_valid(c, ok) || !ok {
        return Err(CollectiveError::PreparationFailed);
    }
    let extra = extra_count(&shape)?;
    let count = a.pencil().local_len().checked_mul(extra);
    let global_count = a.pencil().global_len().checked_mul(extra);
    if !collective_valid(c, count.is_some() && global_count.is_some()) {
        return Err(CollectiveError::CountOverflow);
    }
    let count = count.ok_or(CollectiveError::CountOverflow)?;
    Ok(MapPlan3 {
        shape,
        strides: strides.map(Result::unwrap),
        output_strides: output_strides.unwrap(),
        spatial: a.pencil().local_len(),
        count,
    })
}

fn broadcast_shape3(
    a: &ExtraShape,
    b: &ExtraShape,
    d: &ExtraShape,
) -> Result<Option<Vec<usize>>, CollectiveError> {
    let ab = broadcast_shape(a, b)?.ok_or(CollectiveError::CollectivePreconditionFailed)?;
    broadcast_shape(
        &ExtraShape::new(ab).map_err(|_| CollectiveError::PreparationFailed)?,
        d,
    )
}

fn strides_for_dims(dims: &[usize]) -> Result<Vec<usize>, CollectiveError> {
    let mut out = Vec::new();
    out.try_reserve_exact(dims.len())
        .map_err(|_| CollectiveError::AllocationFailed {
            elements: dims.len(),
        })?;
    let mut stride: usize = 1;
    for &dim in dims.iter().rev() {
        out.push(stride);
        stride = stride
            .checked_mul(dim)
            .ok_or(CollectiveError::CountOverflow)?;
    }
    out.reverse();
    Ok(out)
}

fn for_each_triple<A, B, D, F, const N: usize, const M: usize>(
    a: &PencilArrayView<'_, A, N, M>,
    b: &PencilArrayView<'_, B, N, M>,
    d: &PencilArrayView<'_, D, N, M>,
    p: &MapPlan3,
    mut f: F,
) where
    F: FnMut(&A, &B, &D),
{
    let sa = a.as_slice();
    let sb = b.as_slice();
    let sd = d.as_slice();
    if p.spatial == 0 {
        return;
    }
    let dims = [
        a.extra_shape().dimensions(),
        b.extra_shape().dimensions(),
        d.extra_shape().dimensions(),
    ];
    let extra = p.count / p.spatial;
    for linear in 0..extra {
        let mut ix = [0; 3];
        for q in 0..3 {
            for (axis, &dim) in dims[q].iter().enumerate() {
                let i = (linear / p.output_strides[axis]) % p.shape[axis];
                if dim != 1 {
                    ix[q] += i * p.strides[q][axis];
                }
            }
        }
        for k in 0..p.spatial {
            f(
                &sa[ix[0] * p.spatial + k],
                &sb[ix[1] * p.spatial + k],
                &sd[ix[2] * p.spatial + k],
            );
        }
    }
}

fn prepare_map_reduce_many<T, U, C, const N: usize, const M: usize>(
    c: &C,
    inputs: &[PencilArrayView<'_, T, N, M>],
    neutral: U,
    callback: &str,
    operation: u64,
) -> Result<ManyPlan, CollectiveError>
where
    T: 'static,
    U: SupportedScalar,
    C: CommunicatorCollectives,
{
    // The header is deliberately first: do not inspect input-dependent lengths
    // or allocate a composite descriptor before this agreement.
    let header = [
        DESCRIPTOR_SCHEMA,
        operation,
        u64::try_from(N).unwrap_or(INVALID_WORD),
        u64::try_from(M).unwrap_or(INVALID_WORD),
        u64::try_from(inputs.len()).unwrap_or(INVALID_WORD),
    ];
    if !agree_header(c, header) {
        return Err(CollectiveError::CollectiveDescriptorMismatch);
    }
    let prepared = (|| {
        if inputs.is_empty() {
            return Err(CollectiveError::CollectivePreconditionFailed);
        }
        let mut lengths = Vec::new();
        lengths
            .try_reserve_exact(inputs.len())
            .map_err(|_| CollectiveError::AllocationFailed {
                elements: inputs.len(),
            })?;
        let mut expected = size_of::<U>()
            .checked_mul(2)
            .ok_or(CollectiveError::CountOverflow)?;
        for input in inputs {
            let length = descriptor_len::<T, U, N, M>(input, 0, callback)
                .map_err(|_| CollectiveError::CountOverflow)?;
            expected = expected
                .checked_add(length)
                .ok_or(CollectiveError::CountOverflow)?;
            lengths.push(length);
        }
        let mut descriptor = Vec::new();
        descriptor
            .try_reserve_exact(expected)
            .map_err(|_| CollectiveError::AllocationFailed { elements: expected })?;
        for (input, length) in inputs.iter().zip(lengths) {
            let words = build_descriptor::<_, T, U, N, M>(input, operation, 0, callback)
                .map_err(|_| CollectiveError::PreparationFailed)?;
            if words.len() != length {
                return Err(CollectiveError::PreparationFailed);
            }
            descriptor.extend(words);
        }
        for byte in neutral_bytes(neutral).map_err(|_| CollectiveError::AllocationFailed {
            elements: size_of::<U>(),
        })? {
            append_word(&mut descriptor, u64::from(byte));
        }
        if descriptor.len() != expected {
            return Err(CollectiveError::PreparationFailed);
        }
        let mut shape = Vec::new();
        shape
            .try_reserve_exact(inputs[0].extra_shape().dimensions().len())
            .map_err(|_| CollectiveError::AllocationFailed {
                elements: inputs[0].extra_shape().dimensions().len(),
            })?;
        shape.extend_from_slice(inputs[0].extra_shape().dimensions());
        for input in &inputs[1..] {
            shape = broadcast_shape(
                &ExtraShape::new(shape).map_err(|_| CollectiveError::PreparationFailed)?,
                input.extra_shape(),
            )?
            .ok_or(CollectiveError::CollectivePreconditionFailed)?;
        }
        let mut strides = Vec::new();
        strides
            .try_reserve_exact(inputs.len())
            .map_err(|_| CollectiveError::AllocationFailed {
                elements: inputs.len(),
            })?;
        for input in inputs {
            strides.push(strides_for_dims(input.extra_shape().dimensions())?);
        }
        let output_strides = strides_for_dims(&shape)?;
        let spatial = inputs[0].pencil().local_len();
        let extra = extra_count(&shape)?;
        let count = spatial
            .checked_mul(extra)
            .ok_or(CollectiveError::CountOverflow)?;
        let global_count = inputs[0]
            .pencil()
            .global_len()
            .checked_mul(extra)
            .ok_or(CollectiveError::CountOverflow)?;
        Ok((
            descriptor,
            expected,
            shape,
            strides,
            output_strides,
            spatial,
            extra,
            count,
            global_count,
        ))
    })();
    let ready = prepared.is_ok();
    if !collective_valid(c, ready) {
        return Err(prepared
            .err()
            .unwrap_or(CollectiveError::CollectivePreconditionFailed));
    }
    let (descriptor, expected, shape, strides, output_strides, spatial, extra, count, global_count) =
        prepared.expect("collective many preparation succeeded");
    let length = u64::try_from(expected).unwrap_or(INVALID_WORD);
    let mut minimum = length;
    let mut maximum = length;
    c.all_reduce_into(&length, &mut minimum, SystemOperation::min());
    c.all_reduce_into(&length, &mut maximum, SystemOperation::max());
    if minimum != maximum || minimum == INVALID_WORD {
        return Err(CollectiveError::CollectiveDescriptorMismatch);
    }
    let _ = collective_descriptor(c, Some(descriptor), Some(expected))?;
    let layout_ok = inputs.iter().all(|input| {
        input.pencil().same_layout(inputs[0].pencil())
            && matches!(
                c.compare(input.pencil().topology().communicator()),
                CommunicatorRelation::Identical | CommunicatorRelation::Congruent
            )
    });
    if !collective_valid(c, layout_ok) || !layout_ok {
        return Err(CollectiveError::CollectivePreconditionFailed);
    }
    Ok(ManyPlan {
        shape,
        strides,
        output_strides,
        spatial,
        extra,
        count,
        global_count,
    })
}

fn for_each_many<'v, T, F, const N: usize, const M: usize>(
    inputs: &'v [PencilArrayView<'v, T, N, M>],
    p: &ManyPlan,
    refs: &mut Vec<&'v T>,
    mut f: F,
) where
    F: FnMut(&[&T]),
{
    if p.spatial == 0 {
        return;
    }
    for linear in 0..p.extra {
        refs.clear();
        for (q, input) in inputs.iter().enumerate() {
            let mut i = 0;
            for axis in 0..p.shape.len() {
                let x = (linear / p.output_strides[axis]) % p.shape[axis];
                if input.extra_shape().dimensions()[axis] != 1 {
                    i += x * p.strides[q][axis];
                }
            }
            refs.push(&input.as_slice()[i * p.spatial]);
        }
        for k in 0..p.spatial {
            for (q, input) in inputs.iter().enumerate() {
                let mut i = 0;
                for axis in 0..p.shape.len() {
                    let x = (linear / p.output_strides[axis]) % p.shape[axis];
                    if input.extra_shape().dimensions()[axis] != 1 {
                        i += x * p.strides[q][axis];
                    }
                }
                refs[q] = &input.as_slice()[i * p.spatial + k];
            }
            f(refs);
        }
    }
}

fn map_many_values_prepared<T, U, F, C, const N: usize, const M: usize>(
    c: &C,
    inputs: &[PencilArrayView<'_, T, N, M>],
    p: &ManyPlan,
    mut f: F,
) -> Result<Vec<U>, CollectiveError>
where
    T: 'static,
    U: SupportedScalar,
    F: FnMut(&[&T]) -> U,
    C: CommunicatorCollectives,
{
    let mut out = Vec::new();
    let out_ok = out.try_reserve_exact(p.count).is_ok();
    let mut refs = Vec::new();
    let refs_ok = refs.try_reserve_exact(inputs.len()).is_ok();
    if !collective_valid(c, out_ok && refs_ok) {
        return Err(if !out_ok {
            CollectiveError::AllocationFailed { elements: p.count }
        } else if !refs_ok {
            CollectiveError::AllocationFailed {
                elements: inputs.len(),
            }
        } else {
            CollectiveError::CollectivePreconditionFailed
        });
    }
    for_each_many(inputs, p, &mut refs, |v| out.push(f(v)));
    Ok(out)
}

fn prepare_map_reduce2<A, B, U, const N: usize, const M: usize>(
    left: &PencilArrayView<'_, A, N, M>,
    right: &PencilArrayView<'_, B, N, M>,
    neutral: U,
    callback_name: &str,
    operation: u64,
) -> Result<MapPlan, CollectiveError>
where
    U: SupportedScalar,
{
    let communicator = left.pencil().topology().communicator();
    // The fixed five-word exchange is the first collective.  In particular,
    // do not allocate/build either composite descriptor until all ranks have
    // agreed on the operation and its exact length.
    let left_len = descriptor_len::<A, U, N, M>(left, 0, callback_name);
    let right_len = descriptor_len::<B, U, N, M>(right, 0, callback_name);
    let expected = left_len
        .and_then(|left_len| {
            right_len.and_then(|right_len| {
                left_len
                    .checked_add(right_len)
                    .and_then(|length| length.checked_add(size_of::<U>().checked_mul(2)?))
                    .ok_or(())
            })
        })
        .ok();
    let header = [
        DESCRIPTOR_SCHEMA,
        operation,
        u64::try_from(N).unwrap_or(INVALID_WORD),
        u64::try_from(M).unwrap_or(INVALID_WORD),
        expected
            .and_then(|length| u64::try_from(length).ok())
            .unwrap_or(INVALID_WORD),
    ];
    if !agree_header(communicator, header) {
        return Err(CollectiveError::CollectiveDescriptorMismatch);
    }

    let descriptor = match (left_len, right_len) {
        (Ok(_), Ok(_)) => (|| {
            let mut left_words =
                build_descriptor::<_, A, U, N, M>(left, operation, 0, callback_name)?;
            let right_words =
                build_descriptor::<_, B, U, N, M>(right, operation, 0, callback_name)?;
            left_words
                .try_reserve_exact(expected.ok_or(())?.saturating_sub(left_words.len()))
                .map_err(|_| ())?;
            left_words.extend(right_words);
            for byte in neutral_bytes(neutral)? {
                append_word(&mut left_words, u64::from(byte));
            }
            if Some(left_words.len()) != expected {
                return Err(());
            }
            Ok(left_words)
        })(),
        _ => Err(()),
    };
    let _ = collective_descriptor(communicator, descriptor.ok(), expected)?;
    let shape = broadcast_shape(left.extra_shape(), right.extra_shape());
    if !collective_valid(communicator, shape.is_ok()) {
        return Err(shape
            .err()
            .unwrap_or(CollectiveError::CollectivePreconditionFailed));
    }
    let shape = shape?;
    let valid = left.pencil().same_layout(right.pencil()) && shape.is_some();
    if !collective_valid(communicator, valid) || !valid {
        return Err(CollectiveError::CollectivePreconditionFailed);
    }
    let shape = shape.expect("broadcast shape validated");
    let left_dims = left.extra_shape().dimensions();
    let right_dims = right.extra_shape().dimensions();
    let strides = |dims: &[usize]| -> Result<Vec<usize>, CollectiveError> {
        let mut out = Vec::new();
        out.try_reserve_exact(dims.len())
            .map_err(|_| CollectiveError::AllocationFailed {
                elements: dims.len(),
            })?;
        let mut stride = 1usize;
        for i in (0..dims.len()).rev() {
            out.push(stride);
            stride = stride
                .checked_mul(dims[i])
                .ok_or(CollectiveError::CountOverflow)?;
        }
        out.reverse();
        Ok(out)
    };
    let prepared = (strides(left_dims), strides(right_dims), strides(&shape));
    let preparation_error = prepared
        .0
        .as_ref()
        .err()
        .or_else(|| prepared.1.as_ref().err())
        .or_else(|| prepared.2.as_ref().err())
        .cloned();
    if !collective_valid(
        communicator,
        prepared.0.is_ok() && prepared.1.is_ok() && prepared.2.is_ok(),
    ) {
        return Err(preparation_error.unwrap_or(CollectiveError::CollectivePreconditionFailed));
    }
    let extra = extra_count(&shape).ok();
    let count = extra.and_then(|extra| left.pencil().local_len().checked_mul(extra));
    let global_count = extra.and_then(|extra| left.pencil().global_len().checked_mul(extra));
    if !collective_valid(communicator, count.is_some() && global_count.is_some()) {
        return Err(CollectiveError::CountOverflow);
    }
    Ok(MapPlan {
        left_strides: prepared.0?,
        right_strides: prepared.1?,
        output_strides: prepared.2?,
        spatial: left.pencil().local_len(),
        count: count.ok_or(CollectiveError::CountOverflow)?,
        global_count: global_count.ok_or(CollectiveError::CountOverflow)?,
        shape,
    })
}

fn neutral_bytes<U: SupportedScalar>(value: U) -> Result<Vec<u8>, ()> {
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(size_of::<U>()).map_err(|_| ())?;
    value.append_le_bits(&mut bytes);
    if bytes.len() != size_of::<U>() {
        return Err(());
    }
    Ok(bytes)
}

fn broadcast_shape(
    left: &ExtraShape,
    right: &ExtraShape,
) -> Result<Option<Vec<usize>>, CollectiveError> {
    if left.dimensions().len() != right.dimensions().len() {
        return Ok(None);
    }
    let mut shape = Vec::new();
    shape
        .try_reserve_exact(left.dimensions().len())
        .map_err(|_| CollectiveError::AllocationFailed {
            elements: left.dimensions().len(),
        })?;
    for (&a, &b) in left.dimensions().iter().zip(right.dimensions()) {
        let dimension = if a == b {
            a
        } else if a == 1 {
            b
        } else if b == 1 {
            a
        } else {
            return Ok(None);
        };
        shape.push(dimension);
    }
    Ok(Some(shape))
}

fn extra_count(shape: &[usize]) -> Result<usize, CollectiveError> {
    shape.iter().try_fold(1usize, |a, &b| {
        a.checked_mul(b).ok_or(CollectiveError::CountOverflow)
    })
}
fn map_pairs_collective<A, B, U, F, const N: usize, const M: usize>(
    left: &PencilArrayView<'_, A, N, M>,
    right: &PencilArrayView<'_, B, N, M>,
    plan: &MapPlan,
    mut f: F,
) -> Result<Vec<U>, CollectiveError>
where
    F: FnMut(&A, &B) -> U,
{
    let count = plan.count;
    let mut mapped = Vec::new();
    let allocation = mapped.try_reserve_exact(count);
    if !collective_valid(left.pencil().topology().communicator(), allocation.is_ok()) {
        return Err(allocation
            .err()
            .map_or(CollectiveError::CollectivePreconditionFailed, |_| {
                CollectiveError::AllocationFailed { elements: count }
            }));
    }
    for_each_pair(left, right, plan, |a, b| mapped.push(f(a, b)));
    Ok(mapped)
}

fn for_each_pair<A, B, F, const N: usize, const M: usize>(
    left: &PencilArrayView<'_, A, N, M>,
    right: &PencilArrayView<'_, B, N, M>,
    plan: &MapPlan,
    mut f: F,
) where
    F: FnMut(&A, &B),
{
    let ls = left.as_slice();
    let rs = right.as_slice();
    if plan.count == 0 {
        return;
    }
    let extra = plan.count / plan.spatial;
    for linear in 0..extra {
        let mut li = 0;
        let mut ri = 0;
        for axis in 0..plan.shape.len() {
            let index = (linear / plan.output_strides[axis]) % plan.shape[axis];
            if left.extra_shape().dimensions()[axis] != 1 {
                li += index * plan.left_strides[axis];
            }
            if right.extra_shape().dimensions()[axis] != 1 {
                ri += index * plan.right_strides[axis];
            }
        }
        for k in 0..plan.spatial {
            f(&ls[li * plan.spatial + k], &rs[ri * plan.spatial + k]);
        }
    }
}

fn sum_layout<L, T, const N: usize, const M: usize>(
    layout: &L,
    operation: u64,
) -> Result<T, CollectiveError>
where
    L: LocalArrayLayout<T, N, M>,
    T: SupportedScalar,
{
    let prepared = prepare_layout::<L, T, T, N, M>(layout, operation, 0, type_name::<()>())?;
    let communicator = layout.pencil().topology().communicator();
    let value = agree_sum_result(communicator, T::local_sum(layout.as_slice()))?;
    let local_flags = values_nonfinite_flags(layout.as_slice().iter().copied());
    let _ = prepared;
    T::collective_sum(
        layout.pencil().topology().communicator(),
        value,
        local_flags,
    )
}

fn extreme_layout<L, T, const N: usize, const M: usize>(
    layout: &L,
    operation: u64,
    minimum: bool,
) -> Result<Option<T>, CollectiveError>
where
    L: LocalArrayLayout<T, N, M>,
    T: OrderedScalar,
{
    let prepared = prepare_layout::<L, T, T, N, M>(layout, operation, 0, type_name::<()>())?;
    let communicator = layout.pencil().topology().communicator();
    let local = T::local_extreme(layout.as_slice(), minimum);
    let result = T::collective_extreme(communicator, local, minimum);
    let flags = collective_flags(
        communicator,
        values_nonfinite_flags(layout.as_slice().iter().copied()),
    );
    if prepared.global_count == 0 {
        Ok(None)
    } else if flags[0] != 0 {
        Ok(Some(T::nan_value()))
    } else {
        Ok(Some(result))
    }
}

fn norm_layout<L, T, const N: usize, const M: usize>(
    layout: &L,
    operation: u64,
) -> Result<T::Norm, CollectiveError>
where
    L: LocalArrayLayout<T, N, M>,
    T: SupportedScalar,
{
    let prepared = prepare_layout::<L, T, T::Norm, N, M>(layout, operation, 0, type_name::<()>())?;
    norm_values(
        layout.pencil().topology().communicator(),
        layout.as_slice(),
        prepared,
    )
}

fn truth_layout<L, T, const N: usize, const M: usize>(
    layout: &L,
    operation: u64,
    any_operation: bool,
) -> Result<bool, CollectiveError>
where
    L: LocalArrayLayout<T, N, M>,
    T: TruthValue,
{
    let _ = prepare_layout::<L, T, bool, N, M>(layout, operation, 0, type_name::<()>())?;
    let local = if any_operation {
        layout.as_slice().iter().copied().any(TruthValue::truth)
    } else {
        layout.as_slice().iter().copied().all(TruthValue::truth)
    };
    let communicator = layout.pencil().topology().communicator();
    let local_word = i32::from(local);
    let mut result = 0i32;
    let operation = if any_operation {
        SystemOperation::max()
    } else {
        SystemOperation::min()
    };
    communicator.all_reduce_into(&local_word, &mut result, operation);
    Ok(result != 0)
}

fn by_truth_layout<L, T, F, const N: usize, const M: usize>(
    layout: &L,
    operation: u64,
    any_operation: bool,
    mut predicate: F,
) -> Result<bool, CollectiveError>
where
    L: LocalArrayLayout<T, N, M>,
    F: FnMut(&T) -> bool,
{
    let _ = prepare_layout::<L, T, bool, N, M>(layout, operation, 0, type_name::<F>())?;
    let local = if any_operation {
        layout
            .as_slice()
            .iter()
            .fold(false, |result, value| result | predicate(value))
    } else {
        layout
            .as_slice()
            .iter()
            .fold(true, |result, value| result & predicate(value))
    };
    let communicator = layout.pencil().topology().communicator();
    let local_word = i32::from(local);
    let mut result = 0i32;
    let operation = if any_operation {
        SystemOperation::max()
    } else {
        SystemOperation::min()
    };
    communicator.all_reduce_into(&local_word, &mut result, operation);
    Ok(result != 0)
}

fn mapped_sum_layout<L, T, U, F, const N: usize, const M: usize>(
    layout: &L,
    operation: u64,
    f: F,
) -> Result<U, CollectiveError>
where
    L: LocalArrayLayout<T, N, M>,
    U: SupportedScalar,
    F: FnMut(&T) -> U,
{
    let _ = prepare_layout::<L, T, U, N, M>(layout, operation, 0, type_name::<F>())?;
    let communicator = layout.pencil().topology().communicator();
    let mut partials = U::prepare_collective_sum(communicator)?;
    let mapped = map_values_collective(communicator, layout.as_slice(), f)?;
    let mapped = agree_staged(communicator, Ok(mapped))?;
    sum_values_prepared(communicator, &mapped, &mut partials)
}

fn mapped_extreme_layout<L, T, U, F, const N: usize, const M: usize>(
    layout: &L,
    operation: u64,
    minimum: bool,
    f: F,
) -> Result<Option<U>, CollectiveError>
where
    L: LocalArrayLayout<T, N, M>,
    U: OrderedScalar,
    F: FnMut(&T) -> U,
{
    let prepared = prepare_layout::<L, T, U, N, M>(layout, operation, 0, type_name::<F>())?;
    let mapped = map_values_collective(
        layout.pencil().topology().communicator(),
        layout.as_slice(),
        f,
    )?;
    let local = U::local_extreme(&mapped, minimum);
    let result = U::collective_extreme(layout.pencil().topology().communicator(), local, minimum);
    let flags = collective_flags(
        layout.pencil().topology().communicator(),
        values_nonfinite_flags(mapped.iter().copied()),
    );
    Ok(if prepared.global_count == 0 {
        None
    } else if flags[0] != 0 {
        Some(U::nan_value())
    } else {
        Some(result)
    })
}

fn mapped_norm_layout<L, T, U, F, const N: usize, const M: usize>(
    layout: &L,
    operation: u64,
    f: F,
) -> Result<U::Norm, CollectiveError>
where
    L: LocalArrayLayout<T, N, M>,
    U: SupportedScalar,
    F: FnMut(&T) -> U,
{
    let prepared = prepare_layout::<L, T, U::Norm, N, M>(layout, operation, 0, type_name::<F>())?;
    let mapped = map_values_collective(
        layout.pencil().topology().communicator(),
        layout.as_slice(),
        f,
    )?;
    let mapped = agree_staged(layout.pencil().topology().communicator(), Ok(mapped))?;
    norm_values(layout.pencil().topology().communicator(), &mapped, prepared)
}

fn map_values_collective<T, U, F, C: CommunicatorCollectives>(
    communicator: &C,
    values: &[T],
    mut f: F,
) -> Result<Vec<U>, CollectiveError>
where
    F: FnMut(&T) -> U,
{
    let mut mapped = Vec::new();
    let allocation = mapped.try_reserve_exact(values.len());
    if !collective_valid(communicator, allocation.is_ok()) {
        return Err(allocation
            .err()
            .map_or(CollectiveError::CollectivePreconditionFailed, |_| {
                CollectiveError::AllocationFailed {
                    elements: values.len(),
                }
            }));
    }
    for value in values {
        mapped.push(f(value));
    }
    Ok(mapped)
}

fn agree_staged<T, C: CommunicatorCollectives>(
    communicator: &C,
    staged: Result<Vec<T>, CollectiveError>,
) -> Result<Vec<T>, CollectiveError> {
    let local_ok = staged.is_ok();
    if !collective_valid(communicator, local_ok) {
        return Err(staged
            .err()
            .unwrap_or(CollectiveError::CollectivePreconditionFailed));
    }
    Ok(staged.expect("collective staging validation succeeded"))
}

fn sum_values_prepared<T, C: CommunicatorCollectives>(
    communicator: &C,
    values: &[T],
    partials: &mut Vec<T>,
) -> Result<T, CollectiveError>
where
    T: SupportedScalar,
{
    let value = agree_sum_result(communicator, T::local_sum(values))?;
    T::collective_sum_prepared(
        communicator,
        value,
        values_nonfinite_flags(values.iter().copied()),
        partials,
    )
}

fn agree_sum_result<T, C: CommunicatorCollectives>(
    communicator: &C,
    local: Result<T, CollectiveError>,
) -> Result<T, CollectiveError> {
    let local_overflow = matches!(local.as_ref().err(), Some(CollectiveError::IntegerOverflow));
    if !collective_valid(communicator, local.is_ok()) {
        let local_word = i32::from(local_overflow);
        let mut global_overflow = 0i32;
        communicator.all_reduce_into(&local_word, &mut global_overflow, SystemOperation::max());
        return Err(if global_overflow != 0 {
            CollectiveError::IntegerOverflow
        } else {
            local
                .err()
                .unwrap_or(CollectiveError::CollectivePreconditionFailed)
        });
    }
    local
}

fn norm_values<T, C: CommunicatorCollectives>(
    communicator: &C,
    values: &[T],
    prepared: PreparedLayout,
) -> Result<T::Norm, CollectiveError>
where
    T: SupportedScalar,
{
    let local_flags = values_nonfinite_flags(values.iter().copied());
    let flags = collective_flags(communicator, local_flags);
    if flags[0] != 0 || flags[3] != 0 {
        return Ok(<T::Norm as NormOutput>::from_f64(f64::NAN));
    }

    let local_max = values
        .iter()
        .copied()
        .map(SupportedScalar::norm_abs)
        .fold(0.0_f64, f64::max);
    let mut global_max = 0.0_f64;
    communicator.all_reduce_into(&local_max, &mut global_max, SystemOperation::max());
    if flags[1] != 0 || flags[2] != 0 || flags[4] != 0 || flags[5] != 0 {
        return Ok(<T::Norm as NormOutput>::from_f64(f64::INFINITY));
    }
    if global_max.is_infinite() {
        return Ok(<T::Norm as NormOutput>::from_f64(f64::INFINITY));
    }
    if global_max == 0.0 || prepared.global_count == 0 {
        return Ok(<T::Norm as NormOutput>::from_f64(0.0));
    }

    let local_scaled = values.iter().copied().fold(0.0_f64, |sum, value| {
        let ratio = value.norm_abs() / global_max;
        sum + ratio * ratio
    });
    let mut global_scaled = 0.0_f64;
    communicator.all_reduce_into(&local_scaled, &mut global_scaled, SystemOperation::sum());
    Ok(<T::Norm as NormOutput>::from_f64(
        global_max * global_scaled.sqrt(),
    ))
}

#[derive(Debug)]
struct PreparedLayout {
    global_count: usize,
}

fn prepare_layout<L, T, U, const N: usize, const M: usize>(
    layout: &L,
    operation: u64,
    root_word: u64,
    callback_name: &str,
) -> Result<PreparedLayout, CollectiveError>
where
    L: LocalArrayLayout<T, N, M>,
{
    let communicator = layout.pencil().topology().communicator();
    // Keep the five-word MIN/MAX header ahead of all descriptor allocation.
    let expected_len = descriptor_len::<T, U, N, M>(layout, root_word, callback_name).ok();
    let header = [
        DESCRIPTOR_SCHEMA,
        operation,
        u64::try_from(N).unwrap_or(INVALID_WORD),
        u64::try_from(M).unwrap_or(INVALID_WORD),
        expected_len
            .and_then(|length| u64::try_from(length).ok())
            .unwrap_or(INVALID_WORD),
    ];
    if !agree_header(communicator, header) {
        return Err(CollectiveError::CollectiveDescriptorMismatch);
    }
    let descriptor = expected_len.and_then(|_| {
        build_descriptor::<L, T, U, N, M>(layout, operation, root_word, callback_name).ok()
    });
    let _ = collective_descriptor(communicator, descriptor, expected_len)?;

    let global_count = layout
        .extra_shape()
        .element_count()
        .checked_mul(layout.pencil().global_len())
        .ok_or(CollectiveError::CountOverflow);
    if !collective_valid(communicator, global_count.is_ok()) {
        return Err(global_count
            .err()
            .unwrap_or(CollectiveError::CollectivePreconditionFailed));
    }
    Ok(PreparedLayout {
        global_count: global_count.expect("collective global count validation succeeded"),
    })
}

fn descriptor_len<T, U, const N: usize, const M: usize>(
    layout: &impl LocalArrayLayout<T, N, M>,
    root_word: u64,
    callback_name: &str,
) -> Result<usize, ()> {
    let mut length = 0usize;
    let add = |length: &mut usize, value: usize| -> Result<(), ()> {
        *length = length.checked_add(value).ok_or(())?;
        Ok(())
    };
    add(&mut length, 4)?; // schema, operation, spatial rank, topology rank
    add(&mut length, N)?; // global spatial shape
    add(&mut length, M)?; // ordered decomposition
    add(&mut length, N)?; // memory permutation
    add(&mut length, 1 + layout.extra_shape().dimensions().len())?;
    add(&mut length, M)?; // process grid
    add(&mut length, 1)?; // root word
    add(&mut length, 4)?; // size/alignment for both Rust types
    add(&mut length, 1 + type_name::<T>().len())?;
    add(&mut length, 1 + type_name::<U>().len())?;
    add(&mut length, 1 + callback_name.len())?;
    let _ = root_word;
    length.checked_mul(2).ok_or(())
}

fn build_descriptor<L, T, U, const N: usize, const M: usize>(
    layout: &L,
    operation: u64,
    root_word: u64,
    callback_name: &str,
) -> Result<Vec<u32>, ()>
where
    L: LocalArrayLayout<T, N, M>,
{
    let length = descriptor_len::<T, U, N, M>(layout, root_word, callback_name)?;
    let mut descriptor = Vec::new();
    descriptor.try_reserve_exact(length).map_err(|_| ())?;
    append_word(&mut descriptor, DESCRIPTOR_SCHEMA);
    append_word(&mut descriptor, operation);
    append_word(&mut descriptor, u64::try_from(N).map_err(|_| ())?);
    append_word(&mut descriptor, u64::try_from(M).map_err(|_| ())?);
    append_usizes(&mut descriptor, layout.pencil().global_shape())?;
    for axis in layout.pencil().decomposition() {
        append_word(
            &mut descriptor,
            u64::try_from(axis.index()).map_err(|_| ())?,
        );
    }
    for axis in layout.pencil().permutation().axes() {
        append_word(
            &mut descriptor,
            u64::try_from(axis.index()).map_err(|_| ())?,
        );
    }
    append_word(
        &mut descriptor,
        u64::try_from(layout.extra_shape().dimensions().len()).map_err(|_| ())?,
    );
    append_usizes(&mut descriptor, layout.extra_shape().dimensions())?;
    append_usizes(&mut descriptor, layout.pencil().topology().process_grid())?;
    append_word(&mut descriptor, root_word);
    append_word(&mut descriptor, size_of::<T>() as u64);
    append_word(&mut descriptor, align_of::<T>() as u64);
    append_word(&mut descriptor, size_of::<U>() as u64);
    append_word(&mut descriptor, align_of::<U>() as u64);
    append_type_name(&mut descriptor, type_name::<T>())?;
    append_type_name(&mut descriptor, type_name::<U>())?;
    append_type_name(&mut descriptor, callback_name)?;
    if descriptor.len() != length {
        return Err(());
    }
    Ok(descriptor)
}

fn append_word(words: &mut Vec<u32>, value: u64) {
    words.push(value as u32);
    words.push((value >> 32) as u32);
}

fn append_usizes(words: &mut Vec<u32>, values: &[usize]) -> Result<(), ()> {
    for &value in values {
        append_word(words, u64::try_from(value).map_err(|_| ())?);
    }
    Ok(())
}

fn append_type_name(words: &mut Vec<u32>, name: &str) -> Result<(), ()> {
    append_word(words, u64::try_from(name.len()).map_err(|_| ())?);
    for byte in name.bytes() {
        append_word(words, u64::from(byte));
    }
    Ok(())
}

fn agree_header<C: CommunicatorCollectives>(communicator: &C, header: [u64; HEADER_WORDS]) -> bool {
    // This fixed five-word exchange is shared by the transpose and FFT
    // protocols. Do not turn it into per-word scalar reductions: a rank may be
    // entering a different checked API, and MPI collective count/order must
    // still match while that mismatch is reported.
    let mut minimum = [0u64; HEADER_WORDS];
    let mut maximum = [0u64; HEADER_WORDS];
    communicator.all_reduce_into(&header[..], &mut minimum[..], SystemOperation::min());
    communicator.all_reduce_into(&header[..], &mut maximum[..], SystemOperation::max());
    minimum == maximum
}

fn collective_descriptor<C: CommunicatorCollectives>(
    communicator: &C,
    descriptor: Option<Vec<u32>>,
    expected_len: Option<usize>,
) -> Result<Vec<u32>, CollectiveError> {
    let mut ready = false;
    let mut minimum = None;
    let mut maximum = None;
    if let (Some(words), Some(length)) = (&descriptor, expected_len) {
        if words.len() == length && Count::try_from(length).is_ok() {
            let mut min_words = Vec::new();
            let mut max_words = Vec::new();
            let min_ok = min_words.try_reserve_exact(length).is_ok();
            let max_ok = max_words.try_reserve_exact(length).is_ok();
            if min_ok && max_ok {
                min_words.resize(length, 0);
                max_words.resize(length, 0);
                minimum = Some(min_words);
                maximum = Some(max_words);
                ready = true;
            }
        }
    }
    if !collective_valid(communicator, ready) {
        return Err(CollectiveError::PreparationFailed);
    }
    let words = descriptor.expect("collective descriptor preparation succeeded");
    let mut minimum = minimum.expect("collective descriptor minimum is prepared");
    let mut maximum = maximum.expect("collective descriptor maximum is prepared");
    communicator.all_reduce_into(
        words.as_slice(),
        minimum.as_mut_slice(),
        SystemOperation::min(),
    );
    communicator.all_reduce_into(
        words.as_slice(),
        maximum.as_mut_slice(),
        SystemOperation::max(),
    );
    if minimum != maximum {
        return Err(CollectiveError::CollectiveDescriptorMismatch);
    }
    Ok(words)
}

fn collective_flags<C: CommunicatorCollectives>(communicator: &C, local: [u32; 6]) -> [u32; 6] {
    let mut global = [0u32; 6];
    communicator.all_reduce_into(&local[..], &mut global[..], SystemOperation::max());
    global
}

fn values_nonfinite_flags<T>(values: impl IntoIterator<Item = T>) -> [u32; 6]
where
    T: SupportedScalar,
{
    values.into_iter().fold([0u32; 6], |flags, value| {
        merge_flags(flags, value.nonfinite_flags())
    })
}

fn merge_flags(mut left: [u32; 6], right: [u32; 6]) -> [u32; 6] {
    for index in 0..left.len() {
        left[index] = left[index].max(right[index]);
    }
    left
}

fn sum_component_f32(value: f32, nan: u32, positive: u32, negative: u32) -> f32 {
    if nan != 0 || (positive != 0 && negative != 0) {
        f32::NAN
    } else if positive != 0 {
        f32::INFINITY
    } else if negative != 0 {
        f32::NEG_INFINITY
    } else {
        value
    }
}

fn sum_component_f64(value: f64, nan: u32, positive: u32, negative: u32) -> f64 {
    if nan != 0 || (positive != 0 && negative != 0) {
        f64::NAN
    } else if positive != 0 {
        f64::INFINITY
    } else if negative != 0 {
        f64::NEG_INFINITY
    } else {
        value
    }
}

fn complex_flags_f32(value: Complex32) -> [u32; 6] {
    [
        u32::from(value.re.is_nan()),
        u32::from(value.re.is_infinite() && value.re.is_sign_positive()),
        u32::from(value.re.is_infinite() && value.re.is_sign_negative()),
        u32::from(value.im.is_nan()),
        u32::from(value.im.is_infinite() && value.im.is_sign_positive()),
        u32::from(value.im.is_infinite() && value.im.is_sign_negative()),
    ]
}

fn complex_flags_f64(value: Complex64) -> [u32; 6] {
    [
        u32::from(value.re.is_nan()),
        u32::from(value.re.is_infinite() && value.re.is_sign_positive()),
        u32::from(value.re.is_infinite() && value.re.is_sign_negative()),
        u32::from(value.im.is_nan()),
        u32::from(value.im.is_infinite() && value.im.is_sign_positive()),
        u32::from(value.im.is_infinite() && value.im.is_sign_negative()),
    ]
}

fn prepare_integer_collective_sum<T, C>(communicator: &C) -> Result<Vec<T>, CollectiveError>
where
    T: CheckedSum,
    C: CommunicatorCollectives,
{
    let size = usize::try_from(communicator.size()).map_err(|_| CollectiveError::CountOverflow)?;
    let mut partials = Vec::new();
    let allocation = partials.try_reserve_exact(size);
    if !collective_valid(communicator, allocation.is_ok()) {
        return Err(if allocation.is_err() {
            CollectiveError::AllocationFailed { elements: size }
        } else {
            CollectiveError::CollectivePreconditionFailed
        });
    }
    partials.resize(size, T::zero_for_sum());
    Ok(partials)
}

fn integer_collective_sum<T, C>(communicator: &C, local: T) -> Result<T, CollectiveError>
where
    T: CheckedSum,
    C: CommunicatorCollectives,
{
    let mut partials = prepare_integer_collective_sum(communicator)?;
    integer_collective_sum_prepared(communicator, local, &mut partials)
}

fn integer_collective_sum_prepared<T, C>(
    communicator: &C,
    local: T,
    partials: &mut [T],
) -> Result<T, CollectiveError>
where
    T: CheckedSum,
    C: CommunicatorCollectives,
{
    // ponytail: one checked scalar partial per rank is the deliberate O(P)
    // ceiling; a custom MPI integer operation would not provide the required
    // identical rank-order overflow result.
    communicator.all_gather_into(&local, partials);
    partials
        .iter()
        .copied()
        .try_fold(T::zero_for_sum(), |sum, value| {
            sum.checked_add_for_sum(value)
        })
}

trait CheckedSum: Equivalence + Copy {
    fn zero_for_sum() -> Self;
    fn checked_add_for_sum(self, rhs: Self) -> Result<Self, CollectiveError>;
}

macro_rules! impl_checked_sum {
    ($($ty:ty),+ $(,)?) => {$(
        impl CheckedSum for $ty {
            fn zero_for_sum() -> Self { 0 }
            fn checked_add_for_sum(self, rhs: Self) -> Result<Self, CollectiveError> {
                self.checked_add(rhs).ok_or(CollectiveError::IntegerOverflow)
            }
        }
    )+};
}
impl_checked_sum!(i8, i16, i32, i64, u8, u16, u32, u64);

fn gather_layout<T, R, const N: usize, const M: usize>(
    view: &PencilArrayView<'_, T, N, M>,
    root: R,
) -> Result<Option<Vec<T>>, CollectiveError>
where
    T: Copy + Equivalence,
    R: TryInto<usize> + Copy,
{
    let communicator = view.pencil().topology().communicator();
    let parsed_root = root.try_into().ok();
    let root_word = parsed_root.map_or(INVALID_WORD, |value| value as u64);
    let prepared = prepare_layout::<_, T, T, N, M>(view, OP_GATHER, root_word, type_name::<()>())?;
    let size = view.pencil().topology().size();
    let root_rank = parsed_root.and_then(|value| i32::try_from(value).ok());
    let root_valid = root_rank.is_some_and(|rank| rank >= 0 && rank < communicator.size());
    if !collective_valid(communicator, root_valid) {
        return Err(if root_valid {
            CollectiveError::CollectivePreconditionFailed
        } else {
            CollectiveError::RootOutOfBounds {
                root: parsed_root
                    .and_then(|value| i64::try_from(value).ok())
                    .unwrap_or(-1),
                size,
            }
        });
    }
    if size_of::<T>() == 0 {
        if !collective_valid(communicator, false) {
            return Err(CollectiveError::ZeroSizedTypeUnsupported);
        }
        unreachable!("zero-sized gather type agreement returned success")
    }
    let root_rank = root_rank.expect("collective root validation succeeded");
    let metadata = build_gather_metadata(view, prepared.global_count);
    let metadata_ok = metadata.is_ok();
    if !collective_valid(communicator, metadata_ok) {
        return Err(metadata
            .err()
            .unwrap_or(CollectiveError::CollectivePreconditionFailed));
    }
    let metadata = metadata.expect("collective gather metadata validation succeeded");

    let local_count = u64::try_from(view.len()).map_err(|_| CollectiveError::CountOverflow);
    let count_ok = local_count.is_ok() && Count::try_from(view.len()).is_ok();
    if !collective_valid(communicator, count_ok) {
        return Err(
            if local_count.is_err() || Count::try_from(view.len()).is_err() {
                CollectiveError::CountOverflow
            } else {
                CollectiveError::CollectivePreconditionFailed
            },
        );
    }
    let local_count = local_count.expect("collective local count validation succeeded");
    let mut maximum_count = 0u64;
    communicator.all_reduce_into(&local_count, &mut maximum_count, SystemOperation::max());
    let local_first = if local_count == 0 {
        u64::try_from(size).map_err(|_| CollectiveError::CountOverflow)?
    } else {
        u64::try_from(communicator.rank()).map_err(|_| CollectiveError::CountOverflow)?
    };
    let mut first_nonempty = 0u64;
    communicator.all_reduce_into(&local_first, &mut first_nonempty, SystemOperation::min());
    if usize::try_from(maximum_count).ok() != Some(metadata.maximum_count)
        || usize::try_from(first_nonempty).ok() != Some(metadata.first_nonempty)
    {
        if !collective_valid(communicator, false) {
            return Err(CollectiveError::CollectivePreconditionFailed);
        }
        unreachable!("gather metadata and scalar count agreements diverged")
    }

    let mut root_buffers = None;
    let mut allocation_error = None;
    if communicator.rank() == root_rank {
        let mut global = Vec::new();
        if global.try_reserve_exact(prepared.global_count).is_err() {
            allocation_error = Some(CollectiveError::AllocationFailed {
                elements: prepared.global_count,
            });
        } else {
            let mut scratch = Vec::new();
            if scratch.try_reserve_exact(metadata.maximum_count).is_err() {
                allocation_error = Some(CollectiveError::AllocationFailed {
                    elements: metadata.maximum_count,
                });
            } else {
                root_buffers = Some((global, scratch));
            }
        }
    }
    if !collective_valid(communicator, allocation_error.is_none()) {
        return Err(allocation_error.unwrap_or(CollectiveError::CollectivePreconditionFailed));
    }

    let gather_communicator = communicator.duplicate();
    let rank = usize::try_from(communicator.rank()).map_err(|_| CollectiveError::CountOverflow)?;
    let root = usize::try_from(root_rank).map_err(|_| CollectiveError::CountOverflow)?;
    if prepared.global_count != 0
        && metadata.first_nonempty != root
        && rank == metadata.first_nonempty
    {
        let seed = &view.as_slice()[0];
        gather_communicator
            .process_at_rank(root_rank)
            .send_with_tag(seed, GATHER_SEED_TAG);
    }

    if rank == root {
        let (global, scratch) = root_buffers
            .as_mut()
            .expect("root allocation agreement created root buffers");
        let seed = if prepared.global_count == 0 {
            None
        } else if metadata.first_nonempty == root {
            Some(view.as_slice()[0])
        } else {
            Some(
                gather_communicator
                    .process_at_rank(i32::try_from(metadata.first_nonempty).expect("rank fits MPI"))
                    .receive_with_tag::<T>(GATHER_SEED_TAG)
                    .0,
            )
        };
        if let Some(seed) = seed {
            global.resize_with(prepared.global_count, || seed);
            scratch.resize_with(metadata.maximum_count, || seed);
        }

        for source_rank in 0..size {
            let block = &metadata.blocks[source_rank];
            if source_rank == root {
                unpack_block(global, view.as_slice(), view.pencil(), block);
            } else {
                let scratch_block = &mut scratch[..block.count];
                gather_communicator
                    .process_at_rank(i32::try_from(source_rank).expect("rank fits MPI"))
                    .receive_into_with_tag(scratch_block, GATHER_PAYLOAD_TAG);
                unpack_block(global, scratch_block, view.pencil(), block);
            }
        }
        Ok(Some(std::mem::take(global)))
    } else {
        for source_rank in 0..size {
            if source_rank == rank {
                gather_communicator
                    .process_at_rank(root_rank)
                    .send_with_tag(view.as_slice(), GATHER_PAYLOAD_TAG);
                break;
            }
        }
        Ok(None)
    }
}

#[derive(Debug)]
struct GatherBlock<const N: usize> {
    ranges: [Range<usize>; N],
    local_shape_memory: [usize; N],
    spatial_count: usize,
    count: usize,
}

#[derive(Debug)]
struct GatherMetadata<const N: usize> {
    blocks: Vec<GatherBlock<N>>,
    maximum_count: usize,
    first_nonempty: usize,
}

fn build_gather_metadata<T, const N: usize, const M: usize>(
    view: &PencilArrayView<'_, T, N, M>,
    global_count: usize,
) -> Result<GatherMetadata<N>, CollectiveError> {
    let topology = view.pencil().topology();
    let size = topology.size();
    let mut slots: Vec<Option<GatherBlock<N>>> = Vec::new();
    slots
        .try_reserve_exact(size)
        .map_err(|_| CollectiveError::AllocationFailed { elements: size })?;
    slots.resize_with(size, || None);

    let grid = *topology.process_grid();
    let mut coordinates = [0usize; M];
    loop {
        let rank = topology
            .rank_at(coordinates)
            .map_err(|_| CollectiveError::PreparationFailed)?;
        let rank = usize::try_from(rank).map_err(|_| CollectiveError::CountOverflow)?;
        if slots.get(rank).and_then(Option::as_ref).is_some() {
            return Err(CollectiveError::PreparationFailed);
        }
        let ranges = view
            .pencil()
            .ranges_at(coordinates)
            .map_err(|_| CollectiveError::PreparationFailed)?;
        let local_shape = std::array::from_fn(|axis| ranges[axis].len());
        let spatial_count =
            checked_product(&local_shape).map_err(|_| CollectiveError::CountOverflow)?;
        let count = spatial_count
            .checked_mul(view.extra_shape().element_count())
            .ok_or(CollectiveError::CountOverflow)?;
        Count::try_from(count).map_err(|_| CollectiveError::CountOverflow)?;
        slots[rank] = Some(GatherBlock {
            ranges,
            local_shape_memory: view.pencil().permutation().permute(local_shape),
            spatial_count,
            count,
        });
        if !advance_coordinates(&mut coordinates, grid) {
            break;
        }
    }

    let mut blocks = Vec::new();
    blocks
        .try_reserve_exact(size)
        .map_err(|_| CollectiveError::AllocationFailed { elements: size })?;
    for block in slots {
        blocks.push(block.ok_or(CollectiveError::PreparationFailed)?);
    }
    let mut total = 0usize;
    let mut maximum_count = 0usize;
    let mut first_nonempty = size;
    for (rank, block) in blocks.iter().enumerate() {
        total = total
            .checked_add(block.count)
            .ok_or(CollectiveError::CountOverflow)?;
        maximum_count = maximum_count.max(block.count);
        if block.count != 0 && first_nonempty == size {
            first_nonempty = rank;
        }
    }
    if total != global_count {
        return Err(CollectiveError::PreparationFailed);
    }
    let local_rank =
        usize::try_from(topology.rank()).map_err(|_| CollectiveError::CountOverflow)?;
    if blocks.get(local_rank).map(|block| block.count) != Some(view.len()) {
        return Err(CollectiveError::Array(ArrayError::StorageLengthMismatch {
            required: blocks.get(local_rank).map_or(0, |block| block.count),
            actual: view.len(),
        }));
    }
    Ok(GatherMetadata {
        blocks,
        maximum_count,
        first_nonempty,
    })
}

fn advance_coordinates<const M: usize>(coordinates: &mut [usize; M], grid: [usize; M]) -> bool {
    for axis in (0..M).rev() {
        coordinates[axis] += 1;
        if coordinates[axis] < grid[axis] {
            return true;
        }
        coordinates[axis] = 0;
    }
    false
}

fn unpack_block<T, const N: usize, const M: usize>(
    destination: &mut [T],
    physical: &[T],
    pencil: &Pencil<N, M>,
    block: &GatherBlock<N>,
) where
    T: Copy,
{
    if block.count == 0 {
        return;
    }
    let global_spatial_shape = pencil.global_shape();
    let global_spatial_count = pencil.global_len();
    let extra_count = block.count / block.spatial_count;
    for (physical_offset, &physical_value) in physical.iter().enumerate() {
        let extra_offset = physical_offset / block.spatial_count;
        let mut remainder = physical_offset % block.spatial_count;
        let mut memory_coordinates = [0usize; N];
        for (axis, coordinate) in memory_coordinates.iter_mut().enumerate().rev() {
            let extent = block.local_shape_memory[axis];
            *coordinate = remainder % extent;
            remainder /= extent;
        }
        let mut global_coordinates = [0usize; N];
        for (memory_axis, &local_coordinate) in memory_coordinates.iter().enumerate() {
            let logical_axis = pencil.permutation().axes()[memory_axis].index();
            global_coordinates[logical_axis] = block.ranges[logical_axis].start + local_coordinate;
        }
        let mut global_spatial_offset = 0usize;
        for (logical_axis, &coordinate) in global_coordinates.iter().enumerate() {
            global_spatial_offset =
                global_spatial_offset * global_spatial_shape[logical_axis] + coordinate;
        }
        let global_offset = extra_offset * global_spatial_count + global_spatial_offset;
        debug_assert!(extra_offset < extra_count);
        destination[global_offset] = physical_value;
    }
}

#[cfg(test)]
mod tests {
    use super::advance_coordinates;

    #[test]
    fn coordinate_increment_visits_the_last_tuple_once() {
        let mut coordinates = [0usize, 0usize];
        let grid = [2usize, 3usize];
        let mut count = 1;
        while advance_coordinates(&mut coordinates, grid) {
            count += 1;
        }
        assert_eq!(count, 6);
        assert_eq!(coordinates, [0, 0]);
    }
}
