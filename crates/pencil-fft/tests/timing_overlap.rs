#![cfg(feature = "distributed")]

use std::sync::Arc;

use mpi::traits::*;
use pencil_array::{ExtraShape, MpiTopology, PencilArray};
use pencil_fft::{
    AxisR2rKind, AxisTransform, C2cPlan, Complex, DistributedLayout, FftError, FftReal,
    FourierDirection, FourierDirections, MixedC2cPlan, MixedR2cPlan, R2cPlan, R2rKind, R2rPlan,
    TransposeMethod,
};

trait TestReal: FftReal + mpi::datatype::Equivalence + std::fmt::Debug {
    fn from_f64(value: f64) -> Self;
    fn to_f64(value: Self) -> f64;
}
impl TestReal for f32 {
    fn from_f64(value: f64) -> Self {
        value as f32
    }
    fn to_f64(value: Self) -> f64 {
        value as f64
    }
}
impl TestReal for f64 {
    fn from_f64(value: f64) -> Self {
        value
    }
    fn to_f64(value: Self) -> f64 {
        value
    }
}

#[test]
fn distributed_timing_overlap_one_mpi_binary() {
    let universe = mpi::initialize().expect("MPI initialization failed");
    let world = universe.world();
    let size = usize::try_from(world.size()).unwrap();
    assert!(matches!(size, 1 | 4 | 6), "run with 1, 4, or 6 ranks");
    let topology = MpiTopology::<1>::new(&world, [size]).unwrap();

    for &method in &[TransposeMethod::AllToAllv, TransposeMethod::PointToPoint] {
        for &permute_dims in &[false, true] {
            for directions in [
                FourierDirections::forward(),
                FourierDirections::new([FourierDirection::Forward, FourierDirection::Backward]),
            ] {
                run_case::<f64>(&topology, method, permute_dims, directions, false);
                run_case::<f32>(&topology, method, permute_dims, directions, false);
            }
        }
    }
    run_baselines(&topology);

    // A zero-sized extra dimension is a real empty local array, including on
    // otherwise nonempty ranks. It must still pass construction and timing.
    run_case::<f64>(
        &topology,
        TransposeMethod::PointToPoint,
        false,
        FourierDirections::forward(),
        true,
    );
}

fn run_case<R: TestReal>(
    topology: &Arc<MpiTopology<1>>,
    method: TransposeMethod,
    permute_dims: bool,
    directions: FourierDirections<2>,
    empty: bool,
) where
    Complex<R>: mpi::datatype::Equivalence,
{
    let extra = if empty {
        ExtraShape::new([0]).unwrap()
    } else {
        ExtraShape::scalar()
    };
    let plan = C2cPlan::<R, 2, 1>::from_shape_with_layout(
        Arc::clone(topology),
        [5, 7],
        extra.clone(),
        DistributedLayout {
            transpose_method: method,
            permute_dims,
        },
    )
    .unwrap()
    .with_fft_directions(directions)
    .unwrap();
    let mut source = plan.allocate_input().unwrap();
    fill_mixed_signs(&mut source);
    let source_before = source.as_slice().to_vec();
    let tolerance = if std::mem::size_of::<R>() == 4 {
        2e-2
    } else {
        1e-9
    };
    let mut output = plan.allocate_output().unwrap();
    let mut workspace = plan.allocate_out_of_place_workspace().unwrap();

    let timing = plan
        .forward_with_timing(&source, &mut output, &mut workspace)
        .unwrap();
    assert_c2c_timing(&timing);
    if !empty {
        assert_dft(&output, [5, 7], directions);
    }

    // The profiled path is numerically the old path, and all three directions
    // are checked without relying on a positive wall-clock duration.
    let mut inverse = plan.allocate_input().unwrap();
    let inverse_timing = plan
        .inverse_with_timing(&output, &mut inverse, &mut workspace)
        .unwrap();
    assert_c2c_timing(&inverse_timing);
    assert_close(
        inverse.as_slice(),
        &source_before,
        <R as TestReal>::from_f64(tolerance),
    );
    let mut backward = plan.allocate_input().unwrap();
    let backward_timing = plan
        .backward_with_timing(&output, &mut backward, &mut workspace)
        .unwrap();
    assert_c2c_timing(&backward_timing);
    for (actual, expected) in backward.as_slice().iter().zip(source_before.iter()) {
        assert!((R::to_f64(actual.re) - R::to_f64(expected.re) * 35.0).abs() < tolerance);
    }

    for direction in 0..3 {
        let mut in_place = plan.allocate_in_place().unwrap();
        in_place
            .view_mut()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(source.as_slice());
        let mut ws = plan.allocate_in_place_workspace().unwrap();
        if direction != 0 {
            plan.forward_in_place(&mut in_place, &mut ws).unwrap();
        }
        let timing = match direction {
            0 => plan.forward_in_place_with_timing(&mut in_place, &mut ws),
            1 => plan.inverse_in_place_with_timing(&mut in_place, &mut ws),
            _ => plan.backward_in_place_with_timing(&mut in_place, &mut ws),
        }
        .unwrap();
        assert_c2c_timing(&timing);
        let view = in_place.view().unwrap();
        let expected = match direction {
            0 => output.as_slice(),
            1 => source.as_slice(),
            _ => backward.as_slice(),
        };
        assert_close(
            view.as_slice(),
            expected,
            <R as TestReal>::from_f64(tolerance),
        );
    }

    if method == TransposeMethod::PointToPoint {
        let mut overlap = plan.allocate_output().unwrap();
        plan.forward_with_overlap(&source, &mut overlap, &mut workspace)
            .unwrap();
        assert_close(
            overlap.as_slice(),
            output.as_slice(),
            <R as TestReal>::from_f64(tolerance),
        );
        let mut overlap_input = plan.allocate_input().unwrap();
        plan.inverse_with_overlap(&overlap, &mut overlap_input, &mut workspace)
            .unwrap();
        assert_close(
            overlap_input.as_slice(),
            source_before.as_slice(),
            <R as TestReal>::from_f64(tolerance),
        );
        let mut overlap_backward = plan.allocate_input().unwrap();
        plan.backward_with_overlap(&overlap, &mut overlap_backward, &mut workspace)
            .unwrap();
        assert_close_scaled(overlap_backward.as_slice(), source_before.as_slice(), 35.0);
    } else {
        let mut overlap = plan.allocate_output().unwrap();
        let before = overlap.as_slice().to_vec();
        let result = plan.forward_with_overlap(&source, &mut overlap, &mut workspace);
        assert!(matches!(result, Err(FftError::OverlapUnsupported)));
        assert_eq!(source.as_slice(), source_before.as_slice());
        assert_eq!(overlap.as_slice(), before.as_slice());
    }

    // Initial preflight errors preserve every caller-owned buffer.
    let wrong = C2cPlan::<R, 2, 1>::from_shape_with_layout(
        Arc::clone(topology),
        [6, 7],
        extra.clone(),
        DistributedLayout {
            transpose_method: method,
            permute_dims,
        },
    )
    .unwrap();
    let mut wrong_output = wrong.allocate_output().unwrap();
    let wrong_before = wrong_output.as_slice().to_vec();
    let result = plan.forward(&source, &mut wrong_output, &mut workspace);
    assert!(result.is_err());
    assert_eq!(source.as_slice(), source_before.as_slice());
    assert_eq!(wrong_output.as_slice(), wrong_before.as_slice());

    // The legacy, profiled, and overlap entry points all reject a foreign
    // workspace collectively before touching caller buffers.
    let foreign = C2cPlan::<R, 2, 1>::from_shape_with_layout(
        Arc::clone(topology),
        [5, 7],
        extra.clone(),
        DistributedLayout {
            transpose_method: method,
            permute_dims,
        },
    )
    .unwrap();
    let mut foreign_workspace = foreign.allocate_out_of_place_workspace().unwrap();
    let mut profile_output = plan.allocate_output().unwrap();
    let profile_before = profile_output.as_slice().to_vec();
    let profiled = plan.forward_with_timing(&source, &mut profile_output, &mut foreign_workspace);
    assert!(matches!(profiled, Err(FftError::WorkspaceMismatch)));
    assert_eq!(profile_output.as_slice(), profile_before.as_slice());
    if method == TransposeMethod::PointToPoint {
        let mut overlap_output = plan.allocate_output().unwrap();
        let overlap_before = overlap_output.as_slice().to_vec();
        let overlap =
            plan.forward_with_overlap(&source, &mut overlap_output, &mut foreign_workspace);
        assert!(matches!(overlap, Err(FftError::WorkspaceMismatch)));
        assert_eq!(overlap_output.as_slice(), overlap_before.as_slice());
    }
    // Deliberately disagree on the operation, not merely on a workspace.
    // Every rank enters exactly one header exchange in each block.
    if method == TransposeMethod::PointToPoint && !empty {
        assert_api_mismatch_recovery(&plan, topology, &source, &source_before, &mut workspace);
    }

    // The collective mismatch preflight completes before the next payload.
}

fn run_baselines(topology: &Arc<MpiTopology<1>>) {
    let layout = DistributedLayout {
        transpose_method: TransposeMethod::PointToPoint,
        permute_dims: true,
    };

    let r2c = R2cPlan::<f64, 2, 1>::from_shape_with_layout(
        Arc::clone(topology),
        [5, 7],
        ExtraShape::scalar(),
        layout,
    )
    .unwrap();
    let source = r2c.allocate_input().unwrap();
    let mut output = r2c.allocate_output().unwrap();
    let mut workspace = r2c.allocate_workspace().unwrap();
    assert_timing(
        &r2c.forward_with_timing(&source, &mut output, &mut workspace)
            .unwrap(),
    );

    let r2r = R2rPlan::<f64, 2, 1>::from_shape_with_layout(
        Arc::clone(topology),
        [5, 7],
        ExtraShape::scalar(),
        [Some(R2rKind::DctII), Some(R2rKind::DctII)],
        layout,
    )
    .unwrap();
    let source = r2r.allocate_input().unwrap();
    let mut output = r2r.allocate_output().unwrap();
    let mut workspace = r2r.allocate_workspace().unwrap();
    assert_timing(
        &r2r.forward_with_timing(&source, &mut output, &mut workspace)
            .unwrap(),
    );

    let dht = pencil_fft::DhtPlan::<f64, 2, 1>::from_shape_with_layout(
        Arc::clone(topology),
        [5, 7],
        ExtraShape::scalar(),
        layout,
    )
    .unwrap();
    let source = dht.allocate_input().unwrap();
    let mut output = dht.allocate_output().unwrap();
    let mut workspace = dht.allocate_workspace().unwrap();
    dht.forward(&source, &mut output, &mut workspace).unwrap();

    let mixed = MixedC2cPlan::<f64, 2, 1>::from_shape_with_fft_directions(
        Arc::clone(topology),
        [5, 7],
        ExtraShape::scalar(),
        [AxisTransform::Fft, AxisTransform::R2r(AxisR2rKind::Dht)],
        FourierDirections::forward(),
    )
    .unwrap();
    let source = mixed.allocate_input().unwrap();
    let mut output = mixed.allocate_output().unwrap();
    let mut workspace = mixed.allocate_out_of_place_workspace().unwrap();
    assert_timing(
        &mixed
            .forward_with_timing(&source, &mut output, &mut workspace)
            .unwrap(),
    );

    let mixed_r2c = MixedR2cPlan::<f64, 2, 1>::from_shape_with_fft_directions(
        Arc::clone(topology),
        [5, 7],
        ExtraShape::scalar(),
        [AxisTransform::Rfft, AxisTransform::R2r(AxisR2rKind::Dht)],
        FourierDirections::forward(),
    )
    .unwrap();
    let source = mixed_r2c.allocate_input().unwrap();
    let mut output = mixed_r2c.allocate_output().unwrap();
    let mut workspace = mixed_r2c.allocate_workspace().unwrap();
    assert_timing(
        &mixed_r2c
            .forward_with_timing(&source, &mut output, &mut workspace)
            .unwrap(),
    );
}

fn assert_c2c_timing<const N: usize>(timing: &pencil_fft::TransformTiming<N>) {
    assert_timing(timing);
    for (index, stage) in timing.stages.iter().enumerate() {
        assert_eq!(stage.fft_calls, 1, "C2C stage {index}");
        assert_eq!(
            stage.transition_calls,
            if index + 1 < N { 1 } else { 0 },
            "C2C transition {index}"
        );
    }
}

fn assert_timing<const N: usize>(timing: &pencil_fft::TransformTiming<N>) {
    assert!(timing.total >= timing.stages.iter().map(|stage| stage.total).sum());
    for stage in timing.stages {
        assert!(stage.total >= stage.fft);
        assert!(stage.total >= stage.transpose);
        assert!(stage.communication.pack <= stage.transpose);
        assert!(stage.communication.unpack <= stage.transpose);
        assert!(stage.communication.collective_wait <= stage.transpose);
    }
}

fn assert_api_mismatch_recovery<R: TestReal>(
    plan: &C2cPlan<R, 2, 1>,
    topology: &Arc<MpiTopology<1>>,
    source: &PencilArray<Complex<R>, 2, 1>,
    source_before: &[Complex<R>],
    workspace: &mut pencil_fft::C2cOutOfPlaceWorkspace<R, 2, 1>,
) where
    Complex<R>: mpi::datatype::Equivalence,
{
    if topology.communicator().size() == 1 {
        return;
    }
    let rank_zero = topology.rank() == 0;
    let mut destination = plan.allocate_output().unwrap();
    let destination_before = destination.as_slice().to_vec();

    // rank 0 uses the legacy header; peers use the profiled header.
    let result = if rank_zero {
        plan.forward(source, &mut destination, workspace)
            .map(|_| ())
    } else {
        plan.forward_with_timing(source, &mut destination, workspace)
            .map(|_| ())
    };
    assert!(matches!(
        result,
        Err(FftError::CollectiveDescriptorMismatch)
    ));
    assert_eq!(source.as_slice(), source_before);
    assert_eq!(destination.as_slice(), destination_before.as_slice());

    // rank 0 uses the profiled header; peers use the overlap header.
    let destination_before = destination.as_slice().to_vec();
    let result = if rank_zero {
        plan.forward_with_timing(source, &mut destination, workspace)
            .map(|_| ())
    } else {
        plan.forward_with_overlap(source, &mut destination, workspace)
    };
    assert!(matches!(
        result,
        Err(FftError::CollectiveDescriptorMismatch)
    ));
    assert_eq!(source.as_slice(), source_before);
    assert_eq!(destination.as_slice(), destination_before.as_slice());

    // The same buffers and workspace remain usable after both failed calls.
    plan.forward(source, &mut destination, workspace).unwrap();
    assert_eq!(source.as_slice(), source_before);
}

fn fill_mixed_signs<R: TestReal>(array: &mut PencilArray<Complex<R>, 2, 1>) {
    if array.is_empty() {
        return;
    }
    let ranges = array.pencil().local_ranges().clone();
    let shape = array.local_spatial_shape();
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            let x = ranges[0].start + i;
            let y = ranges[1].start + j;
            let value = if (x + y) % 2 == 0 {
                (x + 1) as f64
            } else {
                -((y + 1) as f64)
            };
            array.get_local_mut(&[], [i, j]).unwrap().re = <R as TestReal>::from_f64(value);
        }
    }
}

fn assert_dft<R: TestReal>(
    array: &PencilArray<Complex<R>, 2, 1>,
    shape: [usize; 2],
    directions: FourierDirections<2>,
) {
    let ranges = array.pencil().local_ranges();
    let tolerance = if std::mem::size_of::<R>() == 4 {
        2e-2
    } else {
        1e-9
    };
    for i in 0..array.local_spatial_shape()[0] {
        for j in 0..array.local_spatial_shape()[1] {
            let kx = ranges[0].start + i;
            let ky = ranges[1].start + j;
            let mut expected = Complex::new(0.0, 0.0);
            for x in 0..shape[0] {
                for y in 0..shape[1] {
                    let value = if (x + y) % 2 == 0 {
                        (x + 1) as f64
                    } else {
                        -((y + 1) as f64)
                    };
                    let signs = [
                        if directions.get(0) == Some(FourierDirection::Forward) {
                            -1.0
                        } else {
                            1.0
                        },
                        if directions.get(1) == Some(FourierDirection::Forward) {
                            -1.0
                        } else {
                            1.0
                        },
                    ];
                    let angle = std::f64::consts::TAU
                        * (signs[0] * kx as f64 * x as f64 / shape[0] as f64
                            + signs[1] * ky as f64 * y as f64 / shape[1] as f64);
                    expected += Complex::new(value * angle.cos(), value * angle.sin());
                }
            }
            let actual = array.get_local(&[], [i, j]).unwrap();
            assert!((R::to_f64(actual.re) - expected.re).abs() < tolerance);
            assert!((R::to_f64(actual.im) - expected.im).abs() < tolerance);
        }
    }
}

fn assert_close<R: TestReal>(actual: &[Complex<R>], expected: &[Complex<R>], tol: R) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert!((R::to_f64(a.re) - R::to_f64(b.re)).abs() < R::to_f64(tol));
        assert!((R::to_f64(a.im) - R::to_f64(b.im)).abs() < R::to_f64(tol));
    }
}
fn assert_close_scaled<R: TestReal>(actual: &[Complex<R>], expected: &[Complex<R>], scale: f64) {
    for (a, b) in actual.iter().zip(expected) {
        assert!((R::to_f64(a.re) - R::to_f64(b.re) * scale).abs() < 3e-3);
        assert!(R::to_f64(a.im).abs() < 3e-3);
    }
}
