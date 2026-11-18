#![cfg(feature = "distributed")]

use std::{f64::consts::TAU, sync::Arc};

use mpi::traits::*;
use pencil_array::{ExtraShape, MpiTopology, Pencil, PencilArray, SpatialAxis};
use pencil_fft::{
    AxisSelection, C2cPlan, C2cState, Complex, DistributedLayout, FftReal, FourierDirection,
    FourierDirections, TransposeMethod,
};

trait TestReal: FftReal + std::fmt::Debug {
    fn from_f64(value: f64) -> Self;
    fn to_f64(value: Self) -> f64;
    fn tolerance() -> f64;
}

impl TestReal for f32 {
    fn from_f64(value: f64) -> Self {
        value as f32
    }
    fn to_f64(value: Self) -> f64 {
        value as f64
    }
    fn tolerance() -> f64 {
        4e-4
    }
}

impl TestReal for f64 {
    fn from_f64(value: f64) -> Self {
        value
    }
    fn to_f64(value: Self) -> f64 {
        value
    }
    fn tolerance() -> f64 {
        2e-9
    }
}

fn process_grid(size: usize) -> [usize; 2] {
    match size {
        1 => [1, 1],
        4 => [2, 2],
        6 => [2, 3],
        other => panic!("run with 1, 4, or 6 MPI ranks, got {other}"),
    }
}

fn axis_order<const N: usize, const M: usize>(pencil: &Pencil<N, M>) -> [usize; N] {
    pencil.permutation().axes().map(SpatialAxis::index)
}

fn decomposition<const N: usize, const M: usize>(pencil: &Pencil<N, M>) -> [usize; M] {
    pencil.decomposition().map(SpatialAxis::index)
}

fn sample(extra: [usize; 2], x: [usize; 4]) -> Complex<f64> {
    Complex::new(
        (extra[0] * 300 + extra[1] * 70 + x[0] * 19 + x[1] * 7 + x[2] * 3 + x[3]) as f64 / 11.0,
        (extra[0] * 13 + extra[1] * 5 + x[0] * 2) as f64 - (x[1] + x[2] * 3 + x[3]) as f64 / 9.0,
    )
}

fn independent_offset<const N: usize, const M: usize>(
    pencil: &Pencil<N, M>,
    extra: [usize; 2],
    logical: [usize; N],
) -> usize {
    let axes = pencil.permutation().axes().map(SpatialAxis::index);
    let memory_shape = pencil.local_shape_memory();
    let mut spatial_offset = 0;
    let mut prefix_stride = 1;
    for position in 0..N {
        spatial_offset += logical[axes[position]] * prefix_stride;
        prefix_stride *= memory_shape[position];
    }
    (extra[0] + 2 * extra[1]) * pencil.local_len() + spatial_offset
}

fn fill<R: TestReal>(array: &mut PencilArray<Complex<R>, 4, 2>, extra_bias: [usize; 2]) {
    let pencil = Arc::clone(array.pencil());
    let ranges = pencil.local_ranges().clone();
    let shape = pencil.local_shape_logical();
    for e1 in 0..3 {
        for e0 in 0..2 {
            for i0 in 0..shape[0] {
                for i1 in 0..shape[1] {
                    for i2 in 0..shape[2] {
                        for i3 in 0..shape[3] {
                            let local = [i0, i1, i2, i3];
                            let global = [
                                ranges[0].start + i0,
                                ranges[1].start + i1,
                                ranges[2].start + i2,
                                ranges[3].start + i3,
                            ];
                            let value = sample([e0 + extra_bias[0], e1 + extra_bias[1]], global);
                            array.as_mut_slice()[independent_offset(&pencil, [e0, e1], local)] =
                                Complex::new(
                                    <R as TestReal>::from_f64(value.re),
                                    <R as TestReal>::from_f64(value.im),
                                );
                        }
                    }
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum DftDirection {
    Forward,
    Inverse,
    Backward,
}

fn direct_value(
    extra: [usize; 2],
    output: [usize; 4],
    direction: DftDirection,
    forward_signs: [f64; 4],
    extra_bias: [usize; 2],
) -> Complex<f64> {
    let inverse = !matches!(direction, DftDirection::Forward);
    let mut result = Complex::new(0.0, 0.0);
    for x0 in 0..3 {
        for x2 in 0..7 {
            let input = sample(
                [extra[0] + extra_bias[0], extra[1] + extra_bias[1]],
                [x0, output[1], x2, output[3]],
            );
            let inverse_sign = if inverse { -1.0 } else { 1.0 };
            let angle = inverse_sign
                * TAU
                * (forward_signs[0] * x0 as f64 * output[0] as f64 / 3.0
                    + forward_signs[2] * x2 as f64 * output[2] as f64 / 7.0);
            let (sin, cos) = angle.sin_cos();
            result.re += input.re * cos - input.im * sin;
            result.im += input.re * sin + input.im * cos;
        }
    }
    if matches!(direction, DftDirection::Inverse) {
        result / 21.0
    } else {
        result
    }
}

fn assert_direct<R: TestReal>(
    pencil: &Pencil<4, 2>,
    values: &[Complex<R>],
    direction: DftDirection,
    forward_signs: [f64; 4],
    extra_bias: [usize; 2],
) {
    let ranges = pencil.local_ranges();
    let shape = pencil.local_shape_logical();
    for e1 in 0..3 {
        for e0 in 0..2 {
            for i0 in 0..shape[0] {
                for i1 in 0..shape[1] {
                    for i2 in 0..shape[2] {
                        for i3 in 0..shape[3] {
                            let local = [i0, i1, i2, i3];
                            let global = [
                                ranges[0].start + i0,
                                ranges[1].start + i1,
                                ranges[2].start + i2,
                                ranges[3].start + i3,
                            ];
                            let actual = values[independent_offset(pencil, [e0, e1], local)];
                            let expected = direct_value(
                                [e0, e1],
                                global,
                                direction,
                                forward_signs,
                                extra_bias,
                            );
                            let tolerance = R::tolerance() * (1.0 + expected.norm());
                            assert!(
                                (R::to_f64(actual.re) - expected.re).abs() <= tolerance
                                    && (R::to_f64(actual.im) - expected.im).abs() <= tolerance,
                                "CM_FFT_C2C_CONTRACT: {direction:?} at {global:?}, extra={:?}: {actual:?} != {expected:?}",
                                [e0, e1]
                            );
                        }
                    }
                }
            }
        }
    }
}

fn expected_memory_shape<const N: usize, const M: usize>(
    pencil: &Pencil<N, M>,
    extra: &[usize],
) -> Vec<usize> {
    let logical = pencil.local_shape_logical();
    let mut expected: Vec<_> = pencil
        .permutation()
        .axes()
        .iter()
        .map(|axis| logical[axis.index()])
        .collect();
    expected.extend_from_slice(extra);
    expected
}

fn check_prefix_stride<const N: usize, const M: usize>(
    pencil: &Pencil<N, M>,
    axis: usize,
) -> usize {
    let axes = pencil.permutation().axes().map(SpatialAxis::index);
    let position = axes.iter().position(|&current| current == axis).unwrap();
    pencil.local_shape_memory()[..position].iter().product()
}

fn run_math<R: TestReal>(topology: &Arc<MpiTopology<2>>, layout: DistributedLayout)
where
    Complex<R>: mpi::datatype::Equivalence,
{
    let directions = FourierDirections::new([
        FourierDirection::Backward,
        FourierDirection::Forward,
        FourierDirection::Forward,
        FourierDirection::Forward,
    ]);
    let signs = [1.0, -1.0, -1.0, 1.0];
    let plan = C2cPlan::<R, 4, 2>::from_shape_with_selection_and_layout(
        Arc::clone(topology),
        [3, 5, 7, 11],
        ExtraShape::new([2, 3]).unwrap(),
        AxisSelection::from_indices([2, 0]).unwrap(),
        layout,
    )
    .unwrap()
    .with_fft_directions(directions)
    .unwrap();
    assert_eq!(
        plan.stage_geometry()
            .iter()
            .map(|stage| stage.axis)
            .collect::<Vec<_>>(),
        [0, 2],
        "CM_FFT_C2C_CONTRACT: selected logical axes run ascending; unselected axes have no stages"
    );
    if !layout.permute_dims {
        assert_eq!(
            check_prefix_stride(plan.stage_geometry()[1].source.as_ref(), 2),
            plan.stage_geometry()[1].source.local_shape_memory()[0]
                * plan.stage_geometry()[1].source.local_shape_memory()[1],
            "CM_FFT_C2C_CONTRACT: no-permute axis-2 line stride is the independent memory prefix product"
        );
    }

    let mut input = plan.allocate_input().unwrap();
    fill(&mut input, [0, 0]);
    let original = input.as_slice().to_vec();
    let mut spectrum = plan.allocate_output().unwrap();
    let mut workspace = plan.allocate_out_of_place_workspace().unwrap();
    let timing = plan
        .forward_with_timing(&input, &mut spectrum, &mut workspace)
        .unwrap();
    assert_eq!(
        timing.stages.len(),
        plan.stage_geometry().len(),
        "CM_FFT_C2C_CONTRACT: profiling reports the routed stage count"
    );
    for (index, stage) in timing.stages.iter().enumerate() {
        assert_eq!(stage.fft_calls, 1, "C2C route stage {index}");
        assert_eq!(
            stage.transition_calls,
            u64::from(index + 1 < timing.stages.len()),
            "C2C route transition {index}"
        );
    }
    assert_direct(
        spectrum.pencil(),
        spectrum.as_slice(),
        DftDirection::Forward,
        signs,
        [0, 0],
    );
    assert_eq!(input.as_slice(), original.as_slice());

    let mut arbitrary = plan.allocate_output().unwrap();
    fill(&mut arbitrary, [7, 9]);
    let arbitrary_before = arbitrary.as_slice().to_vec();
    let mut recovered = plan.allocate_input().unwrap();
    plan.inverse(&arbitrary, &mut recovered, &mut workspace)
        .unwrap();
    assert_eq!(arbitrary.as_slice(), arbitrary_before.as_slice());
    assert_direct(
        recovered.pencil(),
        recovered.as_slice(),
        DftDirection::Inverse,
        signs,
        [7, 9],
    );

    let mut raw = plan.allocate_input().unwrap();
    plan.backward(&arbitrary, &mut raw, &mut workspace).unwrap();
    assert_direct(
        raw.pencil(),
        raw.as_slice(),
        DftDirection::Backward,
        signs,
        [7, 9],
    );

    let mut inplace = plan.allocate_in_place().unwrap();
    inplace
        .view_mut()
        .unwrap()
        .as_mut_slice()
        .copy_from_slice(&original);
    let pointer = inplace.view().unwrap().as_slice().as_ptr();
    let mut inplace_workspace = plan.allocate_in_place_workspace().unwrap();
    plan.forward_in_place(&mut inplace, &mut inplace_workspace)
        .unwrap();
    assert_eq!(inplace.state(), C2cState::Output);
    assert_eq!(inplace.view().unwrap().as_slice().as_ptr(), pointer);
    let output_view = inplace.view().unwrap();
    assert_direct(
        output_view.pencil(),
        output_view.as_slice(),
        DftDirection::Forward,
        signs,
        [0, 0],
    );
    plan.inverse_in_place(&mut inplace, &mut inplace_workspace)
        .unwrap();
    assert_eq!(inplace.state(), C2cState::Input);
    let recovered_view = inplace.view().unwrap();
    assert_eq!(recovered_view.as_slice().as_ptr(), pointer);
    for (actual, expected) in recovered_view.as_slice().iter().zip(&original) {
        let tolerance = R::tolerance() * (1.0 + R::to_f64(expected.re).abs());
        assert!((R::to_f64(actual.re) - R::to_f64(expected.re)).abs() <= tolerance);
        assert!((R::to_f64(actual.im) - R::to_f64(expected.im)).abs() <= tolerance);
    }

    if layout.transpose_method == TransposeMethod::PointToPoint {
        let mut overlap = plan.allocate_output().unwrap();
        plan.forward_with_overlap(&input, &mut overlap, &mut workspace)
            .unwrap();
        assert_eq!(overlap.as_slice(), spectrum.as_slice());
    }

    let foreign = C2cPlan::<R, 4, 2>::from_shape_with_selection_and_layout(
        Arc::clone(topology),
        [3, 5, 7, 11],
        ExtraShape::new([2, 3]).unwrap(),
        AxisSelection::from_indices([0, 2]).unwrap(),
        layout,
    )
    .unwrap();
    let mut foreign_workspace = foreign.allocate_out_of_place_workspace().unwrap();
    let mut untouched = plan.allocate_output().unwrap();
    untouched.as_mut_slice().fill(Complex::new(
        <R as TestReal>::from_f64(17.0),
        <R as TestReal>::from_f64(-5.0),
    ));
    let before = untouched.as_slice().to_vec();
    assert!(
        plan.forward(&input, &mut untouched, &mut foreign_workspace)
            .is_err()
    );
    assert_eq!(input.as_slice(), original.as_slice());
    assert_eq!(untouched.as_slice(), before.as_slice());
}

#[test]
fn column_major_c2c_contract() {
    let universe = mpi::initialize().expect("MPI initialization failed");
    let world = universe.world();
    let size = usize::try_from(world.size()).unwrap();
    let topology = MpiTopology::<2>::new(&world, process_grid(size)).unwrap();

    let suffix_input = Pencil::<3, 2>::new_default(Arc::clone(&topology), [3, 5, 7]).unwrap();
    let suffix_constructor =
        C2cPlan::<f64, 3, 2>::from_pencil(Arc::clone(&suffix_input), ExtraShape::scalar());
    assert!(
        suffix_constructor.is_ok(),
        "CM_FFT_C2C_CONTRACT: C2C constructor must accept identity memory order with default trailing decomposition; got {:?}",
        suffix_constructor.err()
    );

    let full = suffix_constructor.unwrap();
    assert_eq!(axis_order(full.input_pencil()), [0, 1, 2]);
    assert_eq!(decomposition(full.input_pencil()), [1, 2]);
    let stages = full.stage_geometry();
    assert_eq!(
        stages.iter().map(|stage| stage.axis).collect::<Vec<_>>(),
        [0, 1, 2]
    );
    assert_eq!(axis_order(&stages[0].output), [0, 1, 2]);
    assert_eq!(decomposition(&stages[0].output), [1, 2]);
    assert_eq!(axis_order(&stages[1].output), [1, 0, 2]);
    assert_eq!(decomposition(&stages[1].output), [0, 2]);
    assert!(stages[0].source.same_layout(full.input_pencil()));
    assert!(stages[1].source.same_layout(&stages[0].output));
    assert!(stages[2].source.same_layout(&stages[1].output));
    assert_eq!(axis_order(&stages[2].output), [2, 1, 0]);
    assert_eq!(decomposition(&stages[2].output), [0, 1]);

    let selected = AxisSelection::<4>::from_indices([0, 2]).unwrap();
    let subset = C2cPlan::<f64, 4, 2>::from_shape_with_selection_and_layout(
        Arc::clone(&topology),
        [3, 5, 7, 11],
        ExtraShape::new([2, 3]).unwrap(),
        selected,
        DistributedLayout::default(),
    )
    .unwrap();
    assert_eq!(decomposition(subset.input_pencil()), [2, 3]);
    assert_eq!(axis_order(subset.input_pencil()), [0, 1, 2, 3]);
    assert_eq!(
        subset
            .stage_geometry()
            .iter()
            .map(|stage| stage.axis)
            .collect::<Vec<_>>(),
        [0, 2]
    );
    assert_eq!(axis_order(subset.output_pencil()), [2, 0, 1, 3]);
    assert_eq!(decomposition(subset.output_pencil()), [0, 3]);
    let subset_input = subset.allocate_input().unwrap();
    let subset_output = subset.allocate_output().unwrap();
    assert_eq!(
        subset_input.memory_shape(),
        expected_memory_shape(subset.input_pencil(), &[2, 3])
    );
    assert_eq!(
        subset_output.memory_shape(),
        expected_memory_shape(subset.output_pencil(), &[2, 3])
    );
    let mut expected_logical = subset.input_pencil().local_shape_logical().to_vec();
    expected_logical.extend([2, 3]);
    assert_eq!(subset_input.logical_shape(), expected_logical);

    let unordered = C2cPlan::<f64, 4, 2>::from_shape_with_selection_and_layout(
        Arc::clone(&topology),
        [3, 5, 7, 11],
        ExtraShape::new([2, 3]).unwrap(),
        AxisSelection::from_indices([2, 0]).unwrap(),
        DistributedLayout::default(),
    )
    .unwrap();
    assert!(
        subset
            .output_pencil()
            .same_layout(unordered.output_pencil())
    );

    let strided = C2cPlan::<f64, 4, 2>::from_shape_with_selection_and_layout(
        Arc::clone(&topology),
        [3, 5, 7, 11],
        ExtraShape::scalar(),
        selected,
        DistributedLayout {
            permute_dims: false,
            ..DistributedLayout::default()
        },
    )
    .unwrap();
    assert_eq!(axis_order(strided.output_pencil()), [0, 1, 2, 3]);
    assert_eq!(decomposition(strided.output_pencil()), [0, 3]);
    assert_eq!(
        check_prefix_stride(strided.stage_geometry()[1].source.as_ref(), 2),
        15
    );

    let empty = C2cPlan::<f64, 3, 2>::from_shape_with_selection_and_layout(
        Arc::clone(&topology),
        [3, 5, 7],
        ExtraShape::scalar(),
        AxisSelection::empty(),
        DistributedLayout::default(),
    )
    .unwrap();
    assert!(empty.stage_geometry().is_empty());
    assert!(empty.input_pencil().same_layout(empty.output_pencil()));
    let mut empty_input = empty.allocate_input().unwrap();
    for (index, value) in empty_input.as_mut_slice().iter_mut().enumerate() {
        *value = Complex::new(index as f64 + 0.25, -(index as f64) - 0.5);
    }
    let before = empty_input.as_slice().to_vec();
    let mut empty_output = empty.allocate_output().unwrap();
    let mut empty_workspace = empty.allocate_out_of_place_workspace().unwrap();
    let empty_timing = empty
        .forward_with_timing(&empty_input, &mut empty_output, &mut empty_workspace)
        .unwrap();
    assert!(empty_timing.stages.is_empty());
    assert_eq!(empty_output.as_slice(), before.as_slice());

    let zero_extra = C2cPlan::<f64, 4, 2>::from_shape_with_selection_and_layout(
        Arc::clone(&topology),
        [3, 5, 7, 11],
        ExtraShape::new([0]).unwrap(),
        selected,
        DistributedLayout::default(),
    )
    .unwrap();
    assert_eq!(zero_extra.stage_geometry().len(), 2);
    let zero_input = zero_extra.allocate_input().unwrap();
    let mut zero_output = zero_extra.allocate_output().unwrap();
    let mut zero_workspace = zero_extra.allocate_out_of_place_workspace().unwrap();
    zero_extra
        .forward(&zero_input, &mut zero_output, &mut zero_workspace)
        .unwrap();
    assert!(zero_input.is_empty());
    assert!(zero_output.is_empty());

    run_math::<f32>(&topology, DistributedLayout::default());
    run_math::<f64>(&topology, DistributedLayout::default());
    run_math::<f64>(
        &topology,
        DistributedLayout {
            transpose_method: TransposeMethod::PointToPoint,
            permute_dims: false,
        },
    );

    #[cfg(feature = "fftw")]
    {
        let native = subset
            .with_fft_directions(FourierDirections::new([
                FourierDirection::Backward,
                FourierDirection::Forward,
                FourierDirection::Forward,
                FourierDirection::Forward,
            ]))
            .unwrap()
            .with_fftw(pencil_fft::PlanOptions::default())
            .unwrap();
        assert_eq!(native.backend_kind(), pencil_fft::BackendKind::Fftw);
        run_math_with_plan::<f64>(&topology, native);

        let native_f32 = C2cPlan::<f32, 4, 2>::from_shape_with_selection_and_layout(
            Arc::clone(&topology),
            [3, 5, 7, 11],
            ExtraShape::new([2, 3]).unwrap(),
            selected,
            DistributedLayout::default(),
        )
        .unwrap()
        .with_fft_directions(FourierDirections::new([
            FourierDirection::Backward,
            FourierDirection::Forward,
            FourierDirection::Forward,
            FourierDirection::Forward,
        ]))
        .unwrap()
        .with_fftw(pencil_fft::PlanOptions::default())
        .unwrap();
        assert_eq!(native_f32.backend_kind(), pencil_fft::BackendKind::Fftw);
        run_math_with_plan::<f32>(&topology, native_f32);
    }

    let empty_rank = C2cPlan::<f64, 4, 2>::from_shape_with_selection_and_layout(
        Arc::clone(&topology),
        [2, 3, 2, 2],
        ExtraShape::scalar(),
        selected,
        DistributedLayout::default(),
    )
    .unwrap();
    let mut rank_input = empty_rank.allocate_input().unwrap();
    let local_empty = i32::from(rank_input.is_empty());
    let mut empty_ranks = 0;
    topology.communicator().all_reduce_into(
        &local_empty,
        &mut empty_ranks,
        mpi::collective::SystemOperation::sum(),
    );
    if size == 6 {
        assert!(
            empty_ranks > 0,
            "CM_FFT_C2C_CONTRACT: matrix includes empty ranks"
        );
    }
    rank_input.as_mut_slice().fill(Complex::new(0.0, 0.0));
    let mut rank_output = empty_rank.allocate_output().unwrap();
    let mut rank_workspace = empty_rank.allocate_out_of_place_workspace().unwrap();
    empty_rank
        .forward(&rank_input, &mut rank_output, &mut rank_workspace)
        .unwrap();
    assert!(
        rank_output
            .as_slice()
            .iter()
            .all(|value| *value == Complex::new(0.0, 0.0))
    );
    let mut rank_recovered = empty_rank.allocate_input().unwrap();
    empty_rank
        .inverse(&rank_output, &mut rank_recovered, &mut rank_workspace)
        .unwrap();
    assert!(
        rank_recovered
            .as_slice()
            .iter()
            .all(|value| *value == Complex::new(0.0, 0.0))
    );
}

fn run_math_with_plan<R: TestReal>(_topology: &Arc<MpiTopology<2>>, plan: C2cPlan<R, 4, 2>)
where
    Complex<R>: mpi::datatype::Equivalence,
{
    let mut input = plan.allocate_input().unwrap();
    fill(&mut input, [0, 0]);
    let mut output = plan.allocate_output().unwrap();
    let mut workspace = plan.allocate_out_of_place_workspace().unwrap();
    assert_eq!(plan.stage_geometry().len(), 2);
    plan.forward(&input, &mut output, &mut workspace).unwrap();
    assert_direct(
        output.pencil(),
        output.as_slice(),
        DftDirection::Forward,
        [1.0, -1.0, -1.0, 1.0],
        [0, 0],
    );

    let mut arbitrary = plan.allocate_output().unwrap();
    fill(&mut arbitrary, [7, 9]);
    let mut inverse = plan.allocate_input().unwrap();
    plan.inverse(&arbitrary, &mut inverse, &mut workspace)
        .unwrap();
    assert_direct(
        inverse.pencil(),
        inverse.as_slice(),
        DftDirection::Inverse,
        [1.0, -1.0, -1.0, 1.0],
        [7, 9],
    );
    let mut backward = plan.allocate_input().unwrap();
    plan.backward(&arbitrary, &mut backward, &mut workspace)
        .unwrap();
    assert_direct(
        backward.pencil(),
        backward.as_slice(),
        DftDirection::Backward,
        [1.0, -1.0, -1.0, 1.0],
        [7, 9],
    );
}
