use pencil_fft::{
    BackendKind, LocalC2cError, LocalC2cPlan, LocalDhtPlan, LocalR2cError, LocalR2cPlan,
    LocalR2rError, LocalR2rPlan, R2rKind,
};

fn c2c(error: LocalC2cError) -> &'static str {
    match error {
        LocalC2cError::InvalidLength => "invalid",
        LocalC2cError::LengthOverflow => "overflow",
        LocalC2cError::NonIntegralBatch => "batch",
        LocalC2cError::BufferLengthMismatch => "buffers",
        LocalC2cError::ScratchTooSmall { .. } => "scratch",
    }
}
fn r2c(error: LocalR2cError) -> &'static str {
    match error {
        LocalR2cError::InvalidLength => "invalid",
        LocalR2cError::LengthOverflow => "overflow",
        LocalR2cError::NonIntegralBatch => "batch",
        LocalR2cError::BatchCountMismatch { .. } => "count",
        LocalR2cError::RealLineTooSmall { .. } => "real",
        LocalR2cError::ComplexLineTooSmall { .. } => "complex",
        LocalR2cError::ScratchTooSmall { .. } => "scratch",
        LocalR2cError::AllocationFailed { .. } => "allocation",
        LocalR2cError::ArrayMismatch => "array",
        LocalR2cError::WorkspaceMismatch => "workspace",
        LocalR2cError::StorageLengthMismatch { .. } => "storage",
        LocalR2cError::StorageLayoutMismatch => "layout",
        LocalR2cError::WrongState => "state",
        LocalR2cError::Poisoned => "poisoned",
        LocalR2cError::InvalidSpectrumEndpoint { .. } => "endpoint",
    }
}
fn r2r(error: LocalR2rError) -> &'static str {
    match error {
        LocalR2rError::InvalidLength => "invalid",
        LocalR2rError::LengthOverflow => "overflow",
        LocalR2rError::NonIntegralBatch => "batch",
        LocalR2rError::BufferLengthMismatch => "buffers",
        LocalR2rError::ComplexLineTooSmall { .. } => "complex",
        LocalR2rError::ScratchTooSmall { .. } => "scratch",
    }
}

#[test]
fn legacy_local_error_surfaces_are_closed_and_defaults_are_rust() {
    assert_eq!(c2c(LocalC2cError::InvalidLength), "invalid");
    assert_eq!(r2c(LocalR2cError::WrongState), "state");
    assert_eq!(r2r(LocalR2rError::InvalidLength), "invalid");
    assert_eq!(
        LocalC2cPlan::<f64>::new(1).unwrap().backend_kind(),
        BackendKind::RustFft
    );
    assert_eq!(
        LocalR2cPlan::<f64>::new(1).unwrap().backend_kind(),
        BackendKind::RustFft
    );
    assert_eq!(
        LocalR2rPlan::<f64>::new(2, R2rKind::DctI)
            .unwrap()
            .backend_kind(),
        BackendKind::RustFft
    );
    assert_eq!(
        LocalDhtPlan::<f64>::new(1).unwrap().backend_kind(),
        BackendKind::RustFft
    );
}
