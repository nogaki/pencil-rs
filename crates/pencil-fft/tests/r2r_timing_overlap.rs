#![cfg(feature = "distributed")]

use std::sync::Arc;
use std::time::Duration;

use mpi::traits::*;
use pencil_array::{ExtraShape, MpiTopology, PencilArray};
use pencil_fft::{
    Complex, DhtPlan, DistributedLayout, R2rKind, R2rPlan, R2rScalar, TransformTiming,
    TransposeMethod,
};

trait Value: R2rScalar + mpi::datatype::Equivalence {
    fn make(re: f64, im: f64) -> Self;
    fn parts(self) -> (f64, f64);
}
impl Value for f32 {
    fn make(re: f64, _: f64) -> Self {
        re as f32
    }
    fn parts(self) -> (f64, f64) {
        (self as f64, 0.0)
    }
}
impl Value for f64 {
    fn make(re: f64, _: f64) -> Self {
        re
    }
    fn parts(self) -> (f64, f64) {
        (self, 0.0)
    }
}
impl Value for Complex<f32> {
    fn make(re: f64, im: f64) -> Self {
        Complex::new(re as f32, im as f32)
    }
    fn parts(self) -> (f64, f64) {
        (self.re as f64, self.im as f64)
    }
}
impl Value for Complex<f64> {
    fn make(re: f64, im: f64) -> Self {
        Complex::new(re, im)
    }
    fn parts(self) -> (f64, f64) {
        (self.re, self.im)
    }
}

#[test]
fn r2r_and_dht_timing_overlap_cover_values_layouts_and_empty_arrays() {
    let universe = mpi::initialize().expect("MPI initialization failed");
    let world = universe.world();
    let size = usize::try_from(world.size()).unwrap();
    assert!(matches!(size, 1 | 4 | 6), "run with 1, 4, or 6 ranks");
    let topology = MpiTopology::<1>::new(&world, [size]).unwrap();

    for &permute_dims in &[false, true] {
        let layout = DistributedLayout {
            transpose_method: TransposeMethod::PointToPoint,
            permute_dims,
        };
        run_r2r::<f32>(&topology, layout);
        run_r2r::<f64>(&topology, layout);
        run_r2r::<Complex<f32>>(&topology, layout);
        run_r2r::<Complex<f64>>(&topology, layout);
        run_dht::<f32>(&topology, layout);
        run_dht::<f64>(&topology, layout);
        run_dht::<Complex<f32>>(&topology, layout);
        run_dht::<Complex<f64>>(&topology, layout);
    }

    // Zero extra shape exercises the empty-array/empty-rank path without
    // weakening the timing assertions (the route still has all three stages).
    let layout = DistributedLayout {
        transpose_method: TransposeMethod::PointToPoint,
        ..Default::default()
    };
    run_empty::<f64>(&topology, layout);
}

fn assert_timing<const N: usize>(timing: &TransformTiming<N>) {
    assert!(timing.total >= timing.stages.iter().map(|s| s.total).sum::<Duration>());
    for (i, stage) in timing.stages.iter().enumerate() {
        assert_eq!(stage.fft_calls, 1, "stage {i}");
        assert_eq!(stage.transition_calls, u64::from(i + 1 < N), "stage {i}");
        assert!(stage.total >= stage.fft && stage.total >= stage.transpose);
        assert!(stage.communication.pack <= stage.transpose);
        assert!(stage.communication.unpack <= stage.transpose);
        assert!(stage.communication.collective_wait <= stage.transpose);
    }
}

fn fill<T: Value>(a: &mut PencilArray<T, 3, 1>) {
    if a.is_empty() {
        return;
    }
    let ranges = a.pencil().local_ranges().clone();
    let shape = a.local_spatial_shape();
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                if [
                    ranges[0].start + i,
                    ranges[1].start + j,
                    ranges[2].start + k,
                ] == [1, 1, 1]
                {
                    *a.get_local_mut(&[], [i, j, k]).unwrap() = T::make(1.0, 2.0);
                }
            }
        }
    }
}

fn assert_close<T: Value>(a: &PencilArray<T, 3, 1>, b: &PencilArray<T, 3, 1>, eps: f64) {
    for (x, y) in a.as_slice().iter().zip(b.as_slice()) {
        let (xr, xi) = x.parts();
        let (yr, yi) = y.parts();
        assert!((xr - yr).abs() < eps && (xi - yi).abs() < eps);
    }
}

fn run_r2r<T: Value>(topology: &Arc<MpiTopology<1>>, layout: DistributedLayout) {
    let plan = R2rPlan::<T, 3, 1>::from_shape_with_layout(
        Arc::clone(topology),
        [4, 5, 6],
        ExtraShape::scalar(),
        [Some(R2rKind::DctII); 3],
        layout,
    )
    .unwrap();
    let mut input = plan.allocate_input().unwrap();
    fill(&mut input);
    let mut output = plan.allocate_output().unwrap();
    let mut workspace = plan.allocate_workspace().unwrap();
    let timing = plan
        .forward_with_timing(&input, &mut output, &mut workspace)
        .unwrap();
    assert_timing(&timing);
    assert_dct::<T>(&output, [4, 5, 6]);

    let mut overlap = plan.allocate_output().unwrap();
    plan.forward_with_overlap(&input, &mut overlap, &mut workspace)
        .unwrap();
    assert_close(&overlap, &output, tolerance::<T>());

    // The three timing entry points are deliberately exercised, not just forward.
    let mut inverse = plan.allocate_input().unwrap();
    assert_timing(
        &plan
            .inverse_with_timing(&output, &mut inverse, &mut workspace)
            .unwrap(),
    );
    let mut backward = plan.allocate_input().unwrap();
    assert_timing(
        &plan
            .backward_with_timing(&output, &mut backward, &mut workspace)
            .unwrap(),
    );
    let mut inverse_overlap = plan.allocate_input().unwrap();
    plan.inverse_with_overlap(&output, &mut inverse_overlap, &mut workspace)
        .unwrap();
    let mut backward_overlap = plan.allocate_input().unwrap();
    plan.backward_with_overlap(&output, &mut backward_overlap, &mut workspace)
        .unwrap();
    assert_close(&inverse, &input, tolerance::<T>());
    assert_close(&inverse_overlap, &inverse, tolerance::<T>());
    assert_close(&backward_overlap, &backward, tolerance::<T>() * 120.0);
    for direction in 0..3 {
        let mut array = plan.allocate_in_place().unwrap();
        array
            .view_mut()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(input.as_slice());
        let mut ws = plan.allocate_in_place_workspace().unwrap();
        if direction != 0 {
            plan.forward_in_place(&mut array, &mut ws).unwrap();
        }
        let timing = match direction {
            0 => plan.forward_in_place_with_timing(&mut array, &mut ws),
            1 => plan.inverse_in_place_with_timing(&mut array, &mut ws),
            _ => plan.backward_in_place_with_timing(&mut array, &mut ws),
        }
        .unwrap();
        assert_timing(&timing);
        let expected = match direction {
            0 => &output,
            1 => &inverse,
            _ => &backward,
        };
        for (a, b) in array
            .view()
            .unwrap()
            .as_slice()
            .iter()
            .zip(expected.as_slice())
        {
            let (ar, ai) = a.parts();
            let (br, bi) = b.parts();
            assert!(
                (ar - br).abs() < tolerance::<T>() * 120.0
                    && (ai - bi).abs() < tolerance::<T>() * 120.0
            );
        }
    }
}

fn assert_dct<T: Value>(a: &PencilArray<T, 3, 1>, shape: [usize; 3]) {
    let ranges = a.pencil().local_ranges();
    let eps = tolerance::<T>();
    for i in 0..a.local_spatial_shape()[0] {
        for j in 0..a.local_spatial_shape()[1] {
            for k in 0..a.local_spatial_shape()[2] {
                let q = [
                    ranges[0].start + i,
                    ranges[1].start + j,
                    ranges[2].start + k,
                ];
                let factor = (0..3)
                    .map(|d| {
                        2.0 * (std::f64::consts::PI * 1.5 * q[d] as f64 / shape[d] as f64).cos()
                    })
                    .product::<f64>();
                let (re, im) = a.get_local(&[], [i, j, k]).unwrap().parts();
                let (expected_re, expected_im) = T::make(factor, 2.0 * factor).parts();
                assert!(
                    (re - expected_re).abs() < eps && (im - expected_im).abs() < eps,
                    "{q:?}: {re} {im} != {factor}"
                );
            }
        }
    }
}

fn run_dht<T: Value>(topology: &Arc<MpiTopology<1>>, layout: DistributedLayout) {
    let plan = DhtPlan::<T, 3, 1>::from_shape_with_layout(
        Arc::clone(topology),
        [4, 5, 6],
        ExtraShape::scalar(),
        layout,
    )
    .unwrap();
    let mut input = plan.allocate_input().unwrap();
    fill(&mut input);
    let mut output = plan.allocate_output().unwrap();
    let mut workspace = plan.allocate_workspace().unwrap();
    let timing = plan
        .forward_with_timing(&input, &mut output, &mut workspace)
        .unwrap();
    assert_timing(&timing);
    assert_dht::<T>(&output, [4, 5, 6]);
    let mut overlap = plan.allocate_output().unwrap();
    plan.forward_with_overlap(&input, &mut overlap, &mut workspace)
        .unwrap();
    assert_close(&overlap, &output, tolerance::<T>());
    let mut inverse = plan.allocate_input().unwrap();
    assert_timing(
        &plan
            .inverse_with_timing(&output, &mut inverse, &mut workspace)
            .unwrap(),
    );
    let mut backward = plan.allocate_input().unwrap();
    assert_timing(
        &plan
            .backward_with_timing(&output, &mut backward, &mut workspace)
            .unwrap(),
    );
    let mut io = plan.allocate_input().unwrap();
    plan.inverse_with_overlap(&output, &mut io, &mut workspace)
        .unwrap();
    let mut bo = plan.allocate_input().unwrap();
    plan.backward_with_overlap(&output, &mut bo, &mut workspace)
        .unwrap();
    assert_close(&inverse, &input, tolerance::<T>());
    assert_close(&io, &inverse, tolerance::<T>());
    assert_close(&bo, &backward, tolerance::<T>() * 120.0);
    for direction in 0..3 {
        let mut array = plan.allocate_in_place().unwrap();
        array
            .view_mut()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(input.as_slice());
        let mut ws = plan.allocate_in_place_workspace().unwrap();
        if direction != 0 {
            plan.forward_in_place(&mut array, &mut ws).unwrap();
        }
        let timing = match direction {
            0 => plan.forward_in_place_with_timing(&mut array, &mut ws),
            1 => plan.inverse_in_place_with_timing(&mut array, &mut ws),
            _ => plan.backward_in_place_with_timing(&mut array, &mut ws),
        }
        .unwrap();
        assert_timing(&timing);
        let expected = match direction {
            0 => &output,
            1 => &inverse,
            _ => &backward,
        };
        for (a, b) in array
            .view()
            .unwrap()
            .as_slice()
            .iter()
            .zip(expected.as_slice())
        {
            let (ar, ai) = a.parts();
            let (br, bi) = b.parts();
            assert!(
                (ar - br).abs() < tolerance::<T>() * 120.0
                    && (ai - bi).abs() < tolerance::<T>() * 120.0
            );
        }
    }
}

fn assert_dht<T: Value>(a: &PencilArray<T, 3, 1>, shape: [usize; 3]) {
    let ranges = a.pencil().local_ranges();
    let eps = tolerance::<T>();
    for i in 0..a.local_spatial_shape()[0] {
        for j in 0..a.local_spatial_shape()[1] {
            for k in 0..a.local_spatial_shape()[2] {
                let q = [
                    ranges[0].start + i,
                    ranges[1].start + j,
                    ranges[2].start + k,
                ];
                let factor = (0..3)
                    .map(|d| {
                        let theta = std::f64::consts::TAU * q[d] as f64 / shape[d] as f64;
                        theta.cos() + theta.sin()
                    })
                    .product::<f64>();
                let (re, im) = a.get_local(&[], [i, j, k]).unwrap().parts();
                let (expected_re, expected_im) = T::make(factor, 2.0 * factor).parts();
                assert!((re - expected_re).abs() < eps && (im - expected_im).abs() < eps);
            }
        }
    }
}

fn run_empty<T: Value>(topology: &Arc<MpiTopology<1>>, layout: DistributedLayout) {
    let plan = R2rPlan::<T, 3, 1>::from_shape_with_layout(
        Arc::clone(topology),
        [2, 2, 2],
        ExtraShape::new([0]).unwrap(),
        [Some(R2rKind::DctII); 3],
        layout,
    )
    .unwrap();
    let input = plan.allocate_input().unwrap();
    let mut output = plan.allocate_output().unwrap();
    let mut ws = plan.allocate_workspace().unwrap();
    assert!(input.is_empty() && output.is_empty());
    assert_timing(
        &plan
            .forward_with_timing(&input, &mut output, &mut ws)
            .unwrap(),
    );
}

fn tolerance<T: Value>() -> f64 {
    if std::mem::size_of::<T::Real>() == 4 {
        2e-4
    } else {
        1e-10
    }
}
