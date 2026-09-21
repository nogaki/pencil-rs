//! Exhaustive-match compatibility for five distributed FFT/transpose error enums
//! at b4325a8 (not an audit of every public enum in the workspace).
//! No wildcard arms: adding a legacy variant must break this downstream check.
#![cfg(feature = "distributed")]

use pencil_array::TransposeError;
use pencil_fft::{FftError, MixedError, R2cError, R2rError};

#[allow(dead_code)]
fn exhaustive_ffterror(error: FftError) {
    match error {
        FftError::InvalidDimensions => (),
        FftError::InvalidInputLayout => (),
        FftError::InputLayoutMismatch => (),
        FftError::OutputLayoutMismatch => (),
        FftError::ExtraShapeMismatch => (),
        FftError::WorkspaceMismatch => (),
        FftError::WorkspaceTooSmall { .. } => (),
        FftError::CollectiveDescriptorMismatch => (),
        FftError::CollectivePreconditionFailed => (),
        FftError::PreparationFailed => (),
        FftError::AllocationFailed { .. } => (),
        FftError::StorageLayoutMismatch => (),
        FftError::LocalC2c(_) => (),
        FftError::Pencil(_) => (),
        FftError::Array(_) => (),
        FftError::Transpose(_) => (),
        FftError::LocalTranspose(_) => (),
    }
}

#[allow(dead_code)]
fn exhaustive_r2cerror(error: R2cError) {
    match error {
        R2cError::Fft(_) => (),
        R2cError::LocalR2c(_) => (),
        R2cError::InvalidSpectrum => (),
    }
}

#[allow(dead_code)]
fn exhaustive_r2rerror(error: R2rError) {
    match error {
        R2rError::Fft(_) => (),
        R2rError::LocalR2r(_) => (),
    }
}

#[allow(dead_code)]
fn exhaustive_mixederror(error: MixedError) {
    match error {
        MixedError::Fft(_) => (),
        MixedError::LocalC2c(_) => (),
        MixedError::LocalR2r(_) => (),
        MixedError::LocalR2c(_) => (),
        MixedError::InvalidGraph => (),
        MixedError::InvalidSpectrum => (),
    }
}

#[allow(dead_code)]
fn exhaustive_transposeerror(error: TransposeError) {
    match error {
        TransposeError::IncompatibleTopology => (),
        TransposeError::IncompatibleGlobalShape => (),
        TransposeError::UnsupportedDecompositionChange => (),
        TransposeError::SourceLayoutMismatch => (),
        TransposeError::DestinationLayoutMismatch => (),
        TransposeError::ExtraShapeMismatch => (),
        TransposeError::WorkspaceTooSmall { .. } => (),
        TransposeError::CountOverflow => (),
        TransposeError::PreparationFailed => (),
        TransposeError::CollectiveDescriptorMismatch => (),
        TransposeError::CollectivePreconditionFailed => (),
        TransposeError::Array(_) => (),
    }
}

// These pointers are checked without constructing MPI plans or starting MPI.
#[test]
fn only_overlap_methods_use_the_additive_error() {
    use pencil_array::PencilArray;
    use pencil_fft::*;
    type Call<P, S, D, W, T, E> =
        fn(&P, &PencilArray<S, 3, 2>, &mut PencilArray<D, 3, 2>, &mut W) -> Result<T, E>;
    macro_rules! check {
        ($plan:ident, $work:ident, $src:ty, $dst:ty, $error:ty) => {{
            type P = $plan<f64, 3, 2>;
            type W = $work<f64, 3, 2>;
            let _: Call<P, $src, $dst, W, (), $error> = P::forward;
            let _: Call<P, $dst, $src, W, (), $error> = P::inverse;
            let _: Call<P, $dst, $src, W, (), $error> = P::backward;
            let _: Call<P, $src, $dst, W, TransformTiming<3>, $error> = P::forward_with_timing;
            let _: Call<P, $dst, $src, W, TransformTiming<3>, $error> = P::inverse_with_timing;
            let _: Call<P, $dst, $src, W, TransformTiming<3>, $error> = P::backward_with_timing;
            let _: Call<P, $src, $dst, W, (), FftOverlapError<$error>> = P::forward_with_overlap;
            let _: Call<P, $dst, $src, W, (), FftOverlapError<$error>> = P::inverse_with_overlap;
            let _: Call<P, $dst, $src, W, (), FftOverlapError<$error>> = P::backward_with_overlap;
        }};
    }
    check!(
        C2cPlan,
        C2cOutOfPlaceWorkspace,
        Complex<f64>,
        Complex<f64>,
        FftError
    );
    check!(R2cPlan, R2cWorkspace, f64, Complex<f64>, R2cError);
    check!(R2rPlan, R2rWorkspace, f64, f64, R2rError);
    check!(DhtPlan, R2rWorkspace, f64, f64, R2rError);
    check!(
        MixedC2cPlan,
        MixedC2cWorkspace,
        Complex<f64>,
        Complex<f64>,
        MixedError
    );
    check!(
        MixedR2cPlan,
        MixedR2cWorkspace,
        f64,
        Complex<f64>,
        MixedError
    );
}
