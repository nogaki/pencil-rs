#![cfg(feature = "distributed")]

use std::f64::consts::TAU;
use std::sync::Arc;

use mpi::traits::*;
use pencil_array::{ExtraShape, MpiTopology};
use pencil_fft::{AxisSelection, Complex, DistributedLayout, FftReal, R2cPlan, TransposeMethod};

trait Real: FftReal + mpi::datatype::Equivalence {
    fn make(x: f64) -> Self;
    fn error(x: Self) -> f64;
}
impl Real for f32 {
    fn make(x: f64) -> Self {
        x as f32
    }
    fn error(x: Self) -> f64 {
        x.abs() as f64
    }
}
impl Real for f64 {
    fn make(x: f64) -> Self {
        x
    }
    fn error(x: Self) -> f64 {
        x.abs()
    }
}

#[test]
fn r2c_timing_and_overlap_all_layouts() {
    let universe = mpi::initialize().expect("MPI initialization failed");
    let world = universe.world();
    let size = usize::try_from(world.size()).unwrap();
    assert!(matches!(size, 1 | 4 | 6), "run with 1, 4, or 6 ranks");
    let topology = MpiTopology::<1>::new(&world, [size]).unwrap();
    for &permute_dims in &[false, true] {
        run::<f32>(&topology, permute_dims);
        run::<f64>(&topology, permute_dims);
        interior_boundary::<f32>(&topology, permute_dims);
        interior_boundary::<f64>(&topology, permute_dims);
    }
}

fn interior_boundary<R: Real>(topology: &Arc<MpiTopology<1>>, permute_dims: bool)
where
    Complex<R>: Equivalence,
{
    let plan = R2cPlan::<R, 4, 1>::from_shape_with_selection_and_layout(
        Arc::clone(topology),
        [3, 4, 2, 2],
        ExtraShape::scalar(),
        AxisSelection::from_indices([0, 1]).unwrap(),
        DistributedLayout {
            transpose_method: TransposeMethod::PointToPoint,
            permute_dims,
        },
    )
    .unwrap();
    let check = |timing: &pencil_fft::TransformTiming<4>| {
        assert_eq!(timing.stages.map(|stage| stage.fft_calls), [0, 0, 1, 1]);
        assert_eq!(
            timing.stages.map(|stage| stage.transition_calls),
            [1, 1, 1, 0]
        );
        for stage in timing.stages {
            assert_eq!(stage.total, stage.fft + stage.transpose);
        }
    };
    let mut input = plan.allocate_input().unwrap();
    input.as_mut_slice().fill(R::make(1.0));
    let mut output = plan.allocate_output().unwrap();
    let mut ws = plan.allocate_workspace().unwrap();
    check(
        &plan
            .forward_with_timing(&input, &mut output, &mut ws)
            .unwrap(),
    );
    let mut recovered = plan.allocate_input().unwrap();
    check(
        &plan
            .inverse_with_timing(&output, &mut recovered, &mut ws)
            .unwrap(),
    );
    for value in recovered.as_slice() {
        assert!(R::error(*value - R::make(1.0)) < 1e-5);
    }
    check(
        &plan
            .backward_with_timing(&output, &mut recovered, &mut ws)
            .unwrap(),
    );
    for value in recovered.as_slice() {
        assert!(R::error(*value - R::make(12.0)) < 1e-4);
    }
    for direction in 0..3 {
        let mut array = plan.allocate_in_place().unwrap();
        array
            .real_view_mut()
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
        check(&timing);
    }
}

fn expected_fft_calls<R: Real>(plan: &R2cPlan<R, 3, 1>, selection: AxisSelection<3>) -> [u64; 3]
where
    Complex<R>: mpi::datatype::Equivalence,
{
    // One record is a phase invocation, not one record per line.  This remains
    // one on an empty rank and with a zero-sized extra shape.
    let boundary = plan
        .stage_geometry()
        .iter()
        .position(|stage| selection.contains(stage.axis))
        .unwrap();
    // Real-prefix identities are not invoked. Complex-tail identity stages
    // still invoke the local transform helper (copy/no-op), which is measured.
    std::array::from_fn(|stage_index| u64::from(stage_index >= boundary))
}

fn assert_timing<R: Real>(
    plan: &R2cPlan<R, 3, 1>,
    selection: AxisSelection<3>,
    timing: &pencil_fft::TransformTiming<3>,
) where
    Complex<R>: mpi::datatype::Equivalence,
{
    assert!(timing.total >= timing.stages.iter().map(|stage| stage.total).sum());
    let expected = expected_fft_calls(plan, selection);
    for (index, stage) in timing.stages.iter().enumerate() {
        assert_eq!(
            stage.fft_calls, expected[index],
            "stage {index} FFT call count"
        );
        assert_eq!(stage.total, stage.fft + stage.transpose);
        assert_eq!(stage.transition_calls, u64::from(index + 1 < 3));
    }
}

fn run<R: Real>(topology: &Arc<MpiTopology<1>>, permute_dims: bool)
where
    Complex<R>: mpi::datatype::Equivalence,
{
    let layout = DistributedLayout {
        transpose_method: TransposeMethod::PointToPoint,
        permute_dims,
    };
    let selection = AxisSelection::all();
    let plan = R2cPlan::<R, 3, 1>::from_shape_with_selection_and_layout(
        Arc::clone(topology),
        [4, 5, 6],
        ExtraShape::scalar(),
        selection,
        layout,
    )
    .unwrap();
    let mut input = plan.allocate_input().unwrap();
    fill_analytic(&mut input);
    let original = input.as_slice().to_vec();
    let mut reference = plan.allocate_output().unwrap();
    let mut workspace = plan.allocate_workspace().unwrap();
    let timing = plan
        .forward_with_timing(&input, &mut reference, &mut workspace)
        .unwrap();
    assert_timing(&plan, selection, &timing);

    let mut overlap = plan.allocate_output().unwrap();
    plan.forward_with_overlap(&input, &mut overlap, &mut workspace)
        .unwrap();
    assert_analytic_forward(&input, &overlap);

    let mut recovered = plan.allocate_input().unwrap();
    let inverse = plan
        .inverse_with_timing(&overlap, &mut recovered, &mut workspace)
        .unwrap();
    assert_timing(&plan, selection, &inverse);
    let mut overlap_recovered = plan.allocate_input().unwrap();
    plan.inverse_with_overlap(&overlap, &mut overlap_recovered, &mut workspace)
        .unwrap();
    let eps = if std::mem::size_of::<R>() == 4 {
        2e-3
    } else {
        1e-9
    };
    for (a, b) in recovered.as_slice().iter().zip(&original) {
        assert!(R::error(*a - *b) < eps * 20.0);
    }

    let mut backward = plan.allocate_input().unwrap();
    let backward_timing = plan
        .backward_with_timing(&overlap, &mut backward, &mut workspace)
        .unwrap();
    assert_timing(&plan, selection, &backward_timing);
    let mut overlap_backward = plan.allocate_input().unwrap();
    plan.backward_with_overlap(&overlap, &mut overlap_backward, &mut workspace)
        .unwrap();
    for (a, b) in overlap_recovered.as_slice().iter().zip(&original) {
        assert!(R::error(*a - *b) < eps * 20.0);
    }
    for (a, b) in backward.as_slice().iter().zip(&original) {
        assert!(R::error(*a - *b * R::make(120.0)) < eps * 120.0);
    }
    for (a, b) in overlap_backward.as_slice().iter().zip(&original) {
        assert!(R::error(*a - *b * R::make(120.0)) < eps * 120.0);
    }

    for direction in 0..3 {
        let mut array = plan.allocate_in_place().unwrap();
        array
            .real_view_mut()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(&original);
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
        assert_timing(&plan, selection, &timing);
        if direction == 0 {
            for (a, b) in array
                .complex_view()
                .unwrap()
                .as_slice()
                .iter()
                .zip(reference.as_slice())
            {
                assert!(R::error(a.re - b.re) < eps * 120.0 && R::error(a.im - b.im) < eps * 120.0);
            }
        } else {
            let scale = if direction == 1 { 1.0 } else { 120.0 };
            for (a, b) in array.real_view().unwrap().as_slice().iter().zip(&original) {
                assert!(R::error(*a - *b * R::make(scale)) < eps * 120.0);
            }
        }
    }

    // Boundary regressions: the real stage may be the first or last route stage.
    for indices in [[2usize], [0usize]] {
        let selected = AxisSelection::from_indices(indices).unwrap();
        let boundary_plan = R2cPlan::<R, 3, 1>::from_shape_with_selection_and_layout(
            Arc::clone(topology),
            [4, 5, 6],
            ExtraShape::scalar(),
            selected,
            layout,
        )
        .unwrap();
        let mut source = boundary_plan.allocate_input().unwrap();
        fill_analytic(&mut source);
        let original = source.as_slice().to_vec();
        let mut destination = boundary_plan.allocate_output().unwrap();
        let mut boundary_workspace = boundary_plan.allocate_workspace().unwrap();
        let timing = boundary_plan
            .forward_with_timing(&source, &mut destination, &mut boundary_workspace)
            .unwrap();
        assert_timing(&boundary_plan, selected, &timing);
        let mut overlap_forward = boundary_plan.allocate_output().unwrap();
        boundary_plan
            .forward_with_overlap(&source, &mut overlap_forward, &mut boundary_workspace)
            .unwrap();
        let forward_eps = if std::mem::size_of::<R>() == 4 {
            3e-3
        } else {
            2e-10
        };
        for (actual, expected) in overlap_forward
            .as_slice()
            .iter()
            .zip(destination.as_slice())
        {
            assert!(R::error(actual.re - expected.re) < forward_eps);
            assert!(R::error(actual.im - expected.im) < forward_eps);
        }
        let mut inverse = boundary_plan.allocate_input().unwrap();
        let timing = boundary_plan
            .inverse_with_timing(&destination, &mut inverse, &mut boundary_workspace)
            .unwrap();
        assert_timing(&boundary_plan, selected, &timing);
        let mut overlap_inverse = boundary_plan.allocate_input().unwrap();
        boundary_plan
            .inverse_with_overlap(&destination, &mut overlap_inverse, &mut boundary_workspace)
            .unwrap();
        let eps = if std::mem::size_of::<R>() == 4 {
            3e-3
        } else {
            2e-10
        };
        for (actual, expected) in overlap_inverse.as_slice().iter().zip(&original) {
            assert!(R::error(*actual - *expected) < eps * 20.0);
        }
        let mut overlap_backward = boundary_plan.allocate_input().unwrap();
        boundary_plan
            .backward_with_overlap(&destination, &mut overlap_backward, &mut boundary_workspace)
            .unwrap();
        let backward_scale = if indices[0] == 2 { 6.0 } else { 4.0 };
        for (actual, expected) in overlap_backward.as_slice().iter().zip(&original) {
            assert!(R::error(*actual - *expected * R::make(backward_scale)) < eps * 40.0);
        }
    }

    // Empty extra dimensions and ranks with no local points are valid routes.
    let empty = R2cPlan::<R, 3, 1>::from_shape_with_layout(
        Arc::clone(topology),
        [1, 1, 1],
        ExtraShape::new([0]).unwrap(),
        layout,
    )
    .unwrap();
    let source = empty.allocate_input().unwrap();
    let mut destination = empty.allocate_output().unwrap();
    let mut empty_workspace = empty.allocate_workspace().unwrap();
    let timing = empty
        .forward_with_timing(&source, &mut destination, &mut empty_workspace)
        .unwrap();
    assert_timing(&empty, AxisSelection::all(), &timing);
}

fn fill_analytic<R: Real, const M: usize>(input: &mut pencil_array::PencilArray<R, 3, M>) {
    let ranges = input.pencil().local_ranges().clone();
    for x in ranges[0].clone() {
        for y in ranges[1].clone() {
            for z in ranges[2].clone() {
                *input.get_global_mut(&[], [x, y, z]).unwrap() = R::make(analytic_input(x, y, z));
            }
        }
    }
}

fn assert_analytic_forward<R: Real, const M: usize>(
    input: &pencil_array::PencilArray<R, 3, M>,
    output: &pencil_array::PencilArray<Complex<R>, 3, M>,
) {
    let shape = [4usize, 5, 6];
    let eps = if std::mem::size_of::<R>() == 4 {
        3e-3
    } else {
        2e-10
    };
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..(shape[2] / 2 + 1) {
                let mut expected = (0.0, 0.0);
                for x in 0..shape[0] {
                    for y in 0..shape[1] {
                        for z in 0..shape[2] {
                            let value = analytic_input(x, y, z);
                            let phase = TAU * (i * x) as f64 / shape[0] as f64
                                + TAU * (j * y) as f64 / shape[1] as f64
                                + TAU * (k * z) as f64 / shape[2] as f64;
                            expected.0 += value * phase.cos();
                            expected.1 -= value * phase.sin();
                        }
                    }
                }
                if let Some(actual) = output.get_global(&[], [i, j, k]) {
                    assert!(R::error(actual.re - R::make(expected.0)) < eps);
                    assert!(R::error(actual.im - R::make(expected.1)) < eps);
                }
            }
        }
    }
    assert_eq!(input.pencil().global_shape(), &shape);
}

fn analytic_input(x: usize, y: usize, z: usize) -> f64 {
    0.25 + (0.7 * x as f64 + 0.31 * y as f64 + 0.17 * z as f64).sin()
}
