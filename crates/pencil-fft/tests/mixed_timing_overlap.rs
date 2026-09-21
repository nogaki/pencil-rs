#![cfg(feature = "distributed")]

use std::sync::Arc;

use mpi::traits::*;
use pencil_array::{ExtraShape, MpiTopology};
use pencil_fft::{
    AxisTransform, Complex, DistributedLayout, FftError, FftOverlapError, FftReal,
    FourierDirection, FourierDirections, MixedC2cPlan, MixedError, MixedR2cPlan, TransformTiming,
    TransposeMethod,
};

// Debug includes the private workspace buffers and active ownership state.
macro_rules! reject_unsupported {
    ($plan:ident, $src:ident, $dst:ident, $ws:ident, $overlap:ident, $ordinary:ident) => {{
        let before = (
            $src.as_slice().to_vec(),
            $dst.as_slice().to_vec(),
            format!("{:?}", $ws),
        );
        let result = $plan.$overlap(&$src, &mut $dst, &mut $ws);
        assert!(
            matches!(result, Err(FftOverlapError::UnsupportedTransport)),
            "{result:?}"
        );
        assert_eq!($src.as_slice(), before.0);
        assert_eq!($dst.as_slice(), before.1);
        assert_eq!(format!("{:?}", $ws), before.2);
        $plan.$ordinary(&$src, &mut $dst, &mut $ws).unwrap();
    }};
}

trait Real: FftReal + mpi::datatype::Equivalence {
    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;
}
impl Real for f32 {
    fn from_f64(x: f64) -> Self {
        x as f32
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
}
impl Real for f64 {
    fn from_f64(x: f64) -> Self {
        x
    }
    fn to_f64(self) -> f64 {
        self
    }
}

fn phase(sign: f64, k: usize, n: usize) -> Complex<f64> {
    let a = sign * std::f64::consts::TAU * k as f64 / n as f64;
    Complex::new(a.cos(), a.sin())
}
fn assert_complex<R: Real>(actual: Complex<R>, expected: Complex<f64>) {
    let tol = if std::mem::size_of::<R>() == 4 {
        3e-4
    } else {
        1e-10
    };
    assert!(
        (actual.re.to_f64() - expected.re).abs() < tol,
        "re actual={:?} expected={:?}",
        actual.re.to_f64(),
        expected.re
    );
    assert!(
        (actual.im.to_f64() - expected.im).abs() < tol,
        "im actual={:?} expected={:?}",
        actual.im.to_f64(),
        expected.im
    );
}

fn check_timing<const N: usize>(t: &TransformTiming<N>) {
    for stage in t.stages {
        assert_eq!(stage.fft_calls, 1, "every logical stage is profiled once");
        assert_eq!(stage.total, stage.fft + stage.transpose);
        assert!(stage.total <= t.total);
    }
}

fn fill_c2c<R: Real, const N: usize, const M: usize>(
    a: &mut pencil_array::PencilArray<Complex<R>, N, M>,
) {
    if let Some(x) = a.get_global_mut(&[], [1; N]) {
        *x = Complex::new(<R as Real>::from_f64(1.25), <R as Real>::from_f64(3.25));
    }
}
fn fill_r2c<R: Real, const N: usize, const M: usize>(a: &mut pencil_array::PencilArray<R, N, M>) {
    if let Some(x) = a.get_global_mut(&[], [1; N]) {
        *x = <R as Real>::from_f64(1.25);
    }
}

fn c2c_expected(k: [usize; 3]) -> Complex<f64> {
    let dht = (std::f64::consts::TAU * k[1] as f64 / 7.0).cos()
        + (std::f64::consts::TAU * k[1] as f64 / 7.0).sin();
    Complex::new(1.25, 3.25) * phase(1.0, k[0], 5) * dht * phase(-1.0, k[2], 6)
}

fn check_c2c_spectrum<R: Real, const M: usize>(a: &pencil_array::PencilArray<Complex<R>, 3, M>) {
    for i in 0..5 {
        for j in 0..7 {
            for k in 0..6 {
                if let Some(x) = a.get_global(&[], [i, j, k]) {
                    assert_complex(*x, c2c_expected([i, j, k]));
                }
            }
        }
    }
}

fn r2c_expected(k: [usize; 3], boundary: usize) -> Complex<f64> {
    let mut out = Complex::new(1.25, 0.0);
    for (axis, &n) in [5, 7, 6].iter().enumerate() {
        let d = std::f64::consts::TAU * k[axis] as f64 / n as f64;
        out *= if axis > boundary {
            Complex::new(d.cos() + d.sin(), 0.0)
        } else {
            phase(if axis < boundary { 1.0 } else { -1.0 }, k[axis], n)
        };
    }
    out
}

fn check_r2c_spectrum<R: Real, const M: usize>(
    a: &pencil_array::PencilArray<Complex<R>, 3, M>,
    boundary: usize,
) {
    let lengths = [5, 7, 6];
    for i in 0..lengths[0] {
        for j in 0..lengths[1] {
            for k in 0..lengths[2] {
                let q = [i, j, k];
                if q[boundary] <= lengths[boundary] / 2 {
                    if let Some(x) = a.get_global(&[], q) {
                        assert_complex(*x, r2c_expected(q, boundary));
                    }
                }
            }
        }
    }
}

fn c2c_case<R: Real>(topology: &Arc<MpiTopology<1>>, layout: DistributedLayout, empty: bool)
where
    Complex<R>: Equivalence,
{
    let extra = if empty {
        ExtraShape::new([0]).unwrap()
    } else {
        ExtraShape::scalar()
    };
    let dirs = FourierDirections::new([
        FourierDirection::Backward,
        FourierDirection::Forward,
        FourierDirection::Forward,
    ]);
    let plan = MixedC2cPlan::<R, 3, 1>::from_shape_with_layout(
        Arc::clone(topology),
        [5, 7, 6],
        extra,
        [
            AxisTransform::Fft,
            AxisTransform::R2r(pencil_fft::AxisR2rKind::Dht),
            AxisTransform::Fft,
        ],
        layout,
    )
    .unwrap()
    .with_fft_directions(dirs)
    .unwrap();
    let mut input = plan.allocate_input().unwrap();
    fill_c2c(&mut input);
    let mut output = plan.allocate_output().unwrap();
    let mut ws = plan.allocate_workspace().unwrap();
    let foreign = MixedC2cPlan::<R, 3, 1>::from_shape_with_layout(
        Arc::clone(topology),
        [5, 7, 6],
        ExtraShape::scalar(),
        [
            AxisTransform::Fft,
            AxisTransform::R2r(pencil_fft::AxisR2rKind::Dht),
            AxisTransform::Fft,
        ],
        layout,
    )
    .unwrap()
    .with_fft_directions(dirs)
    .unwrap();
    let mut foreign_ws = foreign.allocate_workspace().unwrap();
    assert!(matches!(
        plan.forward_with_timing(&input, &mut output, &mut foreign_ws),
        Err(MixedError::Fft(FftError::WorkspaceMismatch))
    ));
    let timing = plan
        .forward_with_timing(&input, &mut output, &mut ws)
        .unwrap();
    check_timing(&timing);
    for f in [
        plan.inverse_with_timing(&output, &mut { plan.allocate_input().unwrap() }, &mut ws),
        plan.backward_with_timing(&output, &mut { plan.allocate_input().unwrap() }, &mut ws),
    ] {
        check_timing(&f.unwrap());
    }

    check_c2c_spectrum(&output);
    if layout.transpose_method == TransposeMethod::AllToAllv {
        reject_unsupported!(plan, input, output, ws, forward_with_overlap, forward);
        let mut recovered = plan.allocate_input().unwrap();
        reject_unsupported!(plan, output, recovered, ws, inverse_with_overlap, inverse);
        reject_unsupported!(plan, output, recovered, ws, backward_with_overlap, backward);
    }
    // The overlap path is only meaningful for point-to-point routes. Compare it
    // with the independently executed established (non-overlapped) kernel.
    if layout.transpose_method == TransposeMethod::PointToPoint {
        let mut overlap = plan.allocate_output().unwrap();
        plan.forward_with_overlap(&input, &mut overlap, &mut ws)
            .unwrap();
        assert_eq!(overlap.as_slice(), output.as_slice());
        let mut oi = plan.allocate_input().unwrap();
        plan.inverse_with_overlap(&output, &mut oi, &mut ws)
            .unwrap();
        let mut expected = plan.allocate_input().unwrap();
        plan.inverse(&output, &mut expected, &mut ws).unwrap();
        assert_eq!(oi.as_slice(), expected.as_slice());
        let mut ob = plan.allocate_input().unwrap();
        plan.backward_with_overlap(&output, &mut ob, &mut ws)
            .unwrap();
        let mut eb = plan.allocate_input().unwrap();
        plan.backward(&output, &mut eb, &mut ws).unwrap();
        assert_eq!(ob.as_slice(), eb.as_slice());
    }
    for direction in 0..3 {
        let mut a = plan.allocate_in_place().unwrap();
        {
            let mut view = a.view_mut().unwrap();
            for x in view.as_mut_slice() {
                *x = Complex::new(<R as Real>::from_f64(0.0), <R as Real>::from_f64(0.0));
            }
            if let Some(x) = view.get_global_mut(&[], [1; 3]) {
                *x = Complex::new(<R as Real>::from_f64(1.25), <R as Real>::from_f64(3.25));
            }
        }
        let mut iw = plan.allocate_in_place_workspace().unwrap();
        if direction != 0 {
            plan.forward_in_place(&mut a, &mut iw).unwrap();
        }
        let t = match direction {
            0 => plan.forward_in_place_with_timing(&mut a, &mut iw),
            1 => plan.inverse_in_place_with_timing(&mut a, &mut iw),
            _ => plan.backward_in_place_with_timing(&mut a, &mut iw),
        };
        check_timing(&t.unwrap());
    }
}

fn r2c_case<R: Real>(
    topology: &Arc<MpiTopology<1>>,
    layout: DistributedLayout,
    boundary: usize,
    empty: bool,
) where
    Complex<R>: Equivalence,
{
    let extra = if empty {
        ExtraShape::new([0]).unwrap()
    } else {
        ExtraShape::scalar()
    };
    let mut transforms = [AxisTransform::R2r(pencil_fft::AxisR2rKind::Dht); 3];
    transforms[..boundary].fill(AxisTransform::Fft);
    transforms[boundary] = AxisTransform::Rfft;
    transforms[boundary + 1..].fill(AxisTransform::R2r(pencil_fft::AxisR2rKind::Dht));
    let mut directions = [FourierDirection::Forward; 3];
    directions[..boundary].fill(FourierDirection::Backward);
    let plan = MixedR2cPlan::<R, 3, 1>::from_shape_with_layout(
        Arc::clone(topology),
        [5, 7, 6],
        extra,
        transforms,
        layout,
    )
    .unwrap()
    .with_fft_directions(FourierDirections::new(directions))
    .unwrap();
    let mut input = plan.allocate_input().unwrap();
    fill_r2c(&mut input);
    let mut output = plan.allocate_output().unwrap();
    let mut ws = plan.allocate_workspace().unwrap();
    let foreign = MixedR2cPlan::<R, 3, 1>::from_shape_with_layout(
        Arc::clone(topology),
        [5, 7, 6],
        ExtraShape::scalar(),
        transforms,
        layout,
    )
    .unwrap();
    let mut foreign_ws = foreign.allocate_workspace().unwrap();
    assert!(matches!(
        plan.forward_with_timing(&input, &mut output, &mut foreign_ws),
        Err(MixedError::Fft(FftError::WorkspaceMismatch))
    ));
    let timing = plan
        .forward_with_timing(&input, &mut output, &mut ws)
        .unwrap();
    check_timing(&timing);
    for f in [
        plan.inverse_with_timing(&output, &mut { plan.allocate_input().unwrap() }, &mut ws),
        plan.backward_with_timing(&output, &mut { plan.allocate_input().unwrap() }, &mut ws),
    ] {
        check_timing(&f.unwrap());
    }
    check_r2c_spectrum(&output, boundary);
    if layout.transpose_method == TransposeMethod::AllToAllv {
        reject_unsupported!(plan, input, output, ws, forward_with_overlap, forward);
        let mut recovered = plan.allocate_input().unwrap();
        reject_unsupported!(plan, output, recovered, ws, inverse_with_overlap, inverse);
        reject_unsupported!(plan, output, recovered, ws, backward_with_overlap, backward);
    }
    if layout.transpose_method == TransposeMethod::PointToPoint {
        let mut overlap = plan.allocate_output().unwrap();
        plan.forward_with_overlap(&input, &mut overlap, &mut ws)
            .unwrap();
        assert_eq!(overlap.as_slice(), output.as_slice());
        let mut oi = plan.allocate_input().unwrap();
        plan.inverse_with_overlap(&output, &mut oi, &mut ws)
            .unwrap();
        let mut expected = plan.allocate_input().unwrap();
        plan.inverse(&output, &mut expected, &mut ws).unwrap();
        assert_eq!(oi.as_slice(), expected.as_slice());
        let mut ob = plan.allocate_input().unwrap();
        plan.backward_with_overlap(&output, &mut ob, &mut ws)
            .unwrap();
        let mut eb = plan.allocate_input().unwrap();
        plan.backward(&output, &mut eb, &mut ws).unwrap();
        assert_eq!(ob.as_slice(), eb.as_slice());
    }
    for direction in 0..3 {
        let mut a = plan.allocate_in_place().unwrap();
        {
            let mut view = a.real_view_mut().unwrap();
            view.as_mut_slice().copy_from_slice(input.as_slice());
        }
        let mut iw = plan.allocate_in_place_workspace().unwrap();
        if direction != 0 {
            plan.forward_in_place(&mut a, &mut iw).unwrap();
        }
        let t = match direction {
            0 => plan.forward_in_place_with_timing(&mut a, &mut iw),
            1 => plan.inverse_in_place_with_timing(&mut a, &mut iw),
            _ => plan.backward_in_place_with_timing(&mut a, &mut iw),
        };
        check_timing(&t.unwrap());
    }
}

#[test]
fn mixed_timing_overlap_mpi_1_4_6() {
    let universe = mpi::initialize().expect("MPI initialization failed");
    let world = universe.world();
    let size = usize::try_from(world.size()).unwrap();
    assert!(matches!(size, 1 | 4 | 6), "run with 1, 4, or 6 ranks");
    let topology = MpiTopology::<1>::new(&world, [size]).unwrap();
    for &method in &[TransposeMethod::AllToAllv, TransposeMethod::PointToPoint] {
        for &permute_dims in &[false, true] {
            let layout = DistributedLayout {
                transpose_method: method,
                permute_dims,
            };
            c2c_case::<f32>(&topology, layout, false);
            c2c_case::<f64>(&topology, layout, false);
            for boundary in 0..3 {
                r2c_case::<f32>(&topology, layout, boundary, false);
                r2c_case::<f64>(&topology, layout, boundary, false);
            }
        }
    }
    // Empty extra dimensions exercise ranks with no owned data as well.
    c2c_case::<f64>(&topology, DistributedLayout::default(), true);
    r2c_case::<f64>(&topology, DistributedLayout::default(), 1, true);
}
