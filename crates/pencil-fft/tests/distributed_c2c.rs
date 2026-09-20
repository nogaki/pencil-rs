use std::{f64::consts::TAU, sync::Arc};

use mpi::{
    collective::SystemOperation,
    topology::{Color, Communicator, Key},
    traits::CommunicatorCollectives,
};
use pencil_array::{
    AllToAllvTransposePlan, ArrayError, AxisPermutation, ExtraShape, ManyPencilArray, MpiTopology,
    Pencil, PencilArray, PencilArrayView, PointToPointTransposePlan, TransposeWorkspace,
};
use pencil_fft::{
    AxisSelection, C2cInPlaceArray, C2cInPlaceWorkspace, C2cOutOfPlaceWorkspace, C2cPlan, C2cState,
    Complex, FftError, FftReal, R2cError, R2cPlan, R2cState, R2cWorkspace, TransposeMethod,
};

trait TestReal: FftReal + mpi::datatype::Equivalence + std::fmt::Debug {
    fn from_f64(value: f64) -> Self;
    fn to_f64(value: Self) -> f64;
    fn tolerance() -> f64;
    fn epsilon() -> f64;
    fn min_subnormal() -> f64;
    fn huge() -> f64;
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

    fn epsilon() -> f64 {
        f32::EPSILON as f64
    }

    fn min_subnormal() -> f64 {
        f32::from_bits(1) as f64
    }

    fn huge() -> f64 {
        1e20
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
        2e-10
    }

    fn epsilon() -> f64 {
        f64::EPSILON
    }

    fn min_subnormal() -> f64 {
        f64::from_bits(1)
    }

    fn huge() -> f64 {
        1e200
    }
}

type ResultSnapshots<R> = (
    Vec<Complex<R>>,
    Vec<Complex<R>>,
    Vec<Complex<R>>,
    Vec<Complex<R>>,
    Vec<Complex<R>>,
    Vec<Complex<R>>,
);

struct OopTransport<'a, R: TestReal, const N: usize, const M: usize> {
    plan: &'a C2cPlan<R, N, M>,
    source: &'a PencilArray<Complex<R>, N, M>,
    destination: &'a mut PencilArray<Complex<R>, N, M>,
    workspace: &'a mut C2cOutOfPlaceWorkspace<R, N, M>,
}

struct InPlaceTransport<'a, R: TestReal, const N: usize, const M: usize> {
    plan: &'a C2cPlan<R, N, M>,
    array: &'a mut C2cInPlaceArray<R, N, M>,
    workspace: &'a mut C2cInPlaceWorkspace<R, N, M>,
}

#[derive(Clone, Copy)]
enum C2cOperation {
    Forward,
    Inverse,
    Backward,
}

#[test]
fn distributed_c2c_one_mpi_binary() {
    let universe = mpi::initialize().expect("MPI initialization failed");
    let world = universe.world();
    let size = usize::try_from(world.size()).unwrap();
    assert!(matches!(size, 1 | 4 | 6), "run with 1, 4, or 6 ranks");

    let grid_1d = [size];
    let topology_1d = MpiTopology::<1>::new(&world, grid_1d).unwrap();
    assert_eq!(topology_1d.communicator().rank(), topology_1d.rank());
    assert_eq!(topology_1d.communicator().size(), world.size());
    let rank_value = topology_1d.rank();
    let mut minimum = 0;
    let mut maximum = 0;
    topology_1d
        .communicator()
        .all_reduce_into(&rank_value, &mut minimum, SystemOperation::min());
    topology_1d
        .communicator()
        .all_reduce_into(&rank_value, &mut maximum, SystemOperation::max());
    assert_eq!(minimum, 0);
    assert_eq!(maximum, world.size() - 1);
    topology_1d.communicator().barrier();

    run_case::<f64, 2, 1>(&topology_1d, [3, 4], ExtraShape::scalar(), 0.0, true);
    run_case::<f64, 3, 1>(
        &topology_1d,
        [3, 2, 4],
        ExtraShape::new([2, 3]).unwrap(),
        1.0,
        false,
    );
    run_case::<f64, 4, 1>(&topology_1d, [3, 2, 3, 4], ExtraShape::scalar(), 2.0, false);

    let grid_2d = match size {
        1 => [1, 1],
        4 => [2, 2],
        6 => [2, 3],
        _ => unreachable!(),
    };
    let topology_2d = MpiTopology::<2>::new(&world, grid_2d).unwrap();
    run_case::<f64, 3, 2>(&topology_2d, [3, 1, 4], ExtraShape::scalar(), 3.0, false);
    run_case::<f64, 4, 2>(
        &topology_2d,
        [3, 1, 3, 4],
        ExtraShape::new([0]).unwrap(),
        4.0,
        false,
    );

    run_case::<f32, 2, 1>(&topology_1d, [3, 4], ExtraShape::scalar(), 5.0, false);
    run_case::<f32, 3, 1>(&topology_1d, [3, 2, 4], ExtraShape::scalar(), 6.0, false);
    run_case::<f32, 4, 1>(&topology_1d, [3, 2, 3, 4], ExtraShape::scalar(), 7.0, false);
    run_case::<f32, 3, 2>(
        &topology_2d,
        [3, 1, 4],
        ExtraShape::new([2, 3]).unwrap(),
        8.0,
        false,
    );
    run_case::<f32, 4, 2>(&topology_2d, [3, 1, 3, 4], ExtraShape::scalar(), 9.0, false);

    // A reversed rank-order communicator exercises the same route on a
    // different Cartesian context without changing the public API.
    let reversed = world.split_by_color_with_key(
        Color::with_value(0),
        Key::try_from(world.size() - world.rank()).unwrap(),
    );
    let reversed = reversed.expect("all ranks join the reversed communicator");
    let reversed_topology = MpiTopology::<1>::new(&reversed, [size]).unwrap();
    run_case::<f64, 3, 1>(
        &reversed_topology,
        [3, 2, 4],
        ExtraShape::scalar(),
        10.0,
        false,
    );

    run_r2c_cases(&topology_1d, &topology_2d);
    run_r2c_in_place_cases(&topology_1d, &topology_2d);
    run_partial_subset_cases(&topology_1d, &topology_2d);
    assert_r2c_method_parity::<f64, 3, 1>(
        &reversed_topology,
        [3, 2, 5],
        ExtraShape::scalar(),
        26.0,
    );
    negative_collective_cases(&world, &topology_1d, &topology_2d);
    negative_r2c_collective_cases(&world, &topology_1d, &topology_2d);
}

fn run_case<R: TestReal, const N: usize, const M: usize>(
    topology: &Arc<MpiTopology<M>>,
    global_shape: [usize; N],
    extra_shape: ExtraShape,
    seed: f64,
    constructor_checks: bool,
) where
    Complex<R>: mpi::datatype::Equivalence,
{
    let outputs = [TransposeMethod::AllToAllv, TransposeMethod::PointToPoint].map(|method| {
        run_case_with_method::<R, N, M>(
            topology,
            global_shape,
            extra_shape.clone(),
            seed,
            constructor_checks,
            method,
        )
    });
    assert_eq!(outputs[0], outputs[1]);
}

fn run_case_with_method<R: TestReal, const N: usize, const M: usize>(
    topology: &Arc<MpiTopology<M>>,
    global_shape: [usize; N],
    extra_shape: ExtraShape,
    seed: f64,
    constructor_checks: bool,
    method: TransposeMethod,
) -> ResultSnapshots<R>
where
    Complex<R>: mpi::datatype::Equivalence,
{
    let plan = C2cPlan::<R, N, M>::from_shape_with_method(
        Arc::clone(topology),
        global_shape,
        extra_shape.clone(),
        method,
    )
    .unwrap();
    let mut source = plan.allocate_input().unwrap();
    fill_input(&mut source, seed);

    if constructor_checks {
        let legacy_pencil =
            C2cPlan::<R, N, M>::from_pencil(Arc::clone(plan.input_pencil()), extra_shape.clone())
                .unwrap();
        let legacy_array = C2cPlan::<R, N, M>::from_array(&source).unwrap();
        let explicit_pencil = C2cPlan::<R, N, M>::from_pencil_with_method(
            Arc::clone(plan.input_pencil()),
            extra_shape.clone(),
            method,
        )
        .unwrap();
        let explicit_array = C2cPlan::<R, N, M>::from_array_with_method(&source, method).unwrap();
        assert!(
            legacy_pencil
                .output_pencil()
                .same_layout(legacy_array.output_pencil().as_ref())
        );
        assert!(
            explicit_pencil
                .output_pencil()
                .same_layout(explicit_array.output_pencil().as_ref())
        );
        assert!(
            explicit_pencil
                .output_pencil()
                .same_layout(plan.output_pencil().as_ref())
        );
        assert_eq!(legacy_array.extra_shape(), &extra_shape);
        assert_eq!(explicit_array.extra_shape(), &extra_shape);
        if matches!(method, TransposeMethod::AllToAllv) {
            let legacy_shape = C2cPlan::<R, N, M>::from_shape(
                Arc::clone(topology),
                global_shape,
                extra_shape.clone(),
            )
            .unwrap();
            assert!(
                legacy_shape
                    .output_pencil()
                    .same_layout(plan.output_pencil().as_ref())
            );
            let mut legacy_output = legacy_shape.allocate_output().unwrap();
            let mut legacy_workspace = legacy_shape.allocate_out_of_place_workspace().unwrap();
            let mut explicit_output = plan.allocate_output().unwrap();
            let mut explicit_workspace = plan.allocate_out_of_place_workspace().unwrap();
            legacy_shape
                .forward(&source, &mut legacy_output, &mut legacy_workspace)
                .unwrap();
            plan.forward(&source, &mut explicit_output, &mut explicit_workspace)
                .unwrap();
            assert_eq!(legacy_output.as_slice(), explicit_output.as_slice());
        }
    }

    let source_before = source.as_slice().to_vec();
    let mut spectrum = plan.allocate_output().unwrap();
    let spectrum_before = spectrum.as_slice().to_vec();
    let mut workspace = plan.allocate_out_of_place_workspace().unwrap();
    let mut second_spectrum = plan.allocate_output().unwrap();
    let mut second_workspace = plan.allocate_out_of_place_workspace().unwrap();

    plan.forward(&source, &mut spectrum, &mut workspace)
        .unwrap();
    plan.forward(&source, &mut second_spectrum, &mut second_workspace)
        .unwrap();
    assert_eq!(spectrum.as_slice(), second_spectrum.as_slice());
    assert_eq!(source.as_slice(), source_before.as_slice());
    check_forward(&spectrum.view(), &extra_shape, global_shape, seed);
    let oop_forward_snapshot = spectrum.as_slice().to_vec();

    // Inverse is checked against an independently generated spectrum rather
    // than only against the forward output.
    let mut arbitrary_spectrum = plan.allocate_output().unwrap();
    fill_spectrum(&mut arbitrary_spectrum, seed + 31.0);
    let arbitrary_before = arbitrary_spectrum.as_slice().to_vec();
    let mut inverse = plan.allocate_input().unwrap();
    plan.inverse(&arbitrary_spectrum, &mut inverse, &mut workspace)
        .unwrap();
    assert_eq!(arbitrary_spectrum.as_slice(), arbitrary_before.as_slice());
    check_inverse(
        &inverse.view(),
        &extra_shape,
        global_shape,
        seed + 31.0,
        true,
    );
    let oop_inverse_snapshot = inverse.as_slice().to_vec();

    let mut backward = plan.allocate_input().unwrap();
    plan.backward(&arbitrary_spectrum, &mut backward, &mut workspace)
        .unwrap();
    assert_eq!(arbitrary_spectrum.as_slice(), arbitrary_before.as_slice());
    check_inverse(
        &backward.view(),
        &extra_shape,
        global_shape,
        seed + 31.0,
        false,
    );
    let oop_backward_snapshot = backward.as_slice().to_vec();

    let mut roundtrip = plan.allocate_input().unwrap();
    plan.inverse(&spectrum, &mut roundtrip, &mut second_workspace)
        .unwrap();
    assert_eq!(spectrum.as_slice(), second_spectrum.as_slice());
    for (actual, expected) in roundtrip.as_slice().iter().zip(source_before.iter()) {
        assert_close(*actual, *expected);
    }
    let mut raw_roundtrip = plan.allocate_input().unwrap();
    plan.backward(&spectrum, &mut raw_roundtrip, &mut second_workspace)
        .unwrap();
    let spatial_scale = global_shape.iter().product::<usize>() as f64;
    for (actual, expected) in raw_roundtrip.as_slice().iter().zip(source_before.iter()) {
        let expected = Complex::new(
            <R as TestReal>::from_f64(R::to_f64(expected.re) * spatial_scale),
            <R as TestReal>::from_f64(R::to_f64(expected.im) * spatial_scale),
        );
        assert_close(*actual, expected);
    }

    let mut in_place = plan.allocate_in_place().unwrap();
    fill_in_place(&mut in_place, seed, false);
    let in_place_storage = in_place.view().unwrap().as_slice().as_ptr();
    let in_place_original = in_place.view().unwrap().as_slice().to_vec();
    let mut in_place_workspace = plan.allocate_in_place_workspace().unwrap();
    assert_eq!(in_place.state(), C2cState::Input);
    assert!(
        in_place
            .view()
            .unwrap()
            .pencil()
            .same_layout(plan.input_pencil().as_ref())
    );
    plan.forward_in_place(&mut in_place, &mut in_place_workspace)
        .unwrap();
    assert_eq!(in_place.state(), C2cState::Output);
    assert_eq!(
        in_place.view().unwrap().as_slice().as_ptr(),
        in_place_storage
    );
    assert!(
        in_place
            .view()
            .unwrap()
            .pencil()
            .same_layout(plan.output_pencil().as_ref())
    );
    for (actual, expected) in in_place
        .view()
        .unwrap()
        .as_slice()
        .iter()
        .zip(spectrum.as_slice())
    {
        assert_close(*actual, *expected);
    }
    check_forward(&in_place.view().unwrap(), &extra_shape, global_shape, seed);
    let in_place_forward_snapshot = in_place.view().unwrap().as_slice().to_vec();

    plan.inverse_in_place(&mut in_place, &mut in_place_workspace)
        .unwrap();
    assert_eq!(in_place.state(), C2cState::Input);
    assert_eq!(
        in_place.view().unwrap().as_slice().as_ptr(),
        in_place_storage
    );
    assert!(
        in_place
            .view()
            .unwrap()
            .pencil()
            .same_layout(plan.input_pencil().as_ref())
    );
    for (actual, expected) in in_place
        .view()
        .unwrap()
        .as_slice()
        .iter()
        .zip(&in_place_original)
    {
        assert_close(*actual, *expected);
    }

    // Output remains mutable so callers can supply arbitrary spectra to the
    // normalized inverse, including every extra batch independently.
    plan.forward_in_place(&mut in_place, &mut in_place_workspace)
        .unwrap();
    fill_in_place(&mut in_place, seed + 31.0, true);
    plan.inverse_in_place(&mut in_place, &mut in_place_workspace)
        .unwrap();
    assert_eq!(in_place.state(), C2cState::Input);
    assert_eq!(
        in_place.view().unwrap().as_slice().as_ptr(),
        in_place_storage
    );
    for (actual, expected) in in_place
        .view()
        .unwrap()
        .as_slice()
        .iter()
        .zip(inverse.as_slice())
    {
        assert_close(*actual, *expected);
    }
    check_inverse(
        &in_place.view().unwrap(),
        &extra_shape,
        global_shape,
        seed + 31.0,
        true,
    );
    let in_place_inverse_snapshot = in_place.view().unwrap().as_slice().to_vec();

    // Raw backward uses the same output-to-input state route without local
    // normalization, and preserves the single backing allocation.
    plan.forward_in_place(&mut in_place, &mut in_place_workspace)
        .unwrap();
    fill_in_place(&mut in_place, seed + 31.0, true);
    plan.backward_in_place(&mut in_place, &mut in_place_workspace)
        .unwrap();
    assert_eq!(in_place.state(), C2cState::Input);
    assert_eq!(
        in_place.view().unwrap().as_slice().as_ptr(),
        in_place_storage
    );
    assert!(
        in_place
            .view()
            .unwrap()
            .pencil()
            .same_layout(plan.input_pencil().as_ref())
    );
    for (actual, expected) in in_place
        .view()
        .unwrap()
        .as_slice()
        .iter()
        .zip(&oop_backward_snapshot)
    {
        assert_close(*actual, *expected);
    }
    check_inverse(
        &in_place.view().unwrap(),
        &extra_shape,
        global_shape,
        seed + 31.0,
        false,
    );
    let in_place_backward_snapshot = in_place.view().unwrap().as_slice().to_vec();

    fill_in_place(&mut in_place, seed, false);
    plan.forward_in_place(&mut in_place, &mut in_place_workspace)
        .unwrap();
    assert_eq!(in_place.state(), C2cState::Output);
    assert_eq!(
        in_place.view().unwrap().as_slice().as_ptr(),
        in_place_storage
    );
    plan.backward_in_place(&mut in_place, &mut in_place_workspace)
        .unwrap();
    assert_eq!(in_place.state(), C2cState::Input);
    assert_eq!(
        in_place.view().unwrap().as_slice().as_ptr(),
        in_place_storage
    );
    for (actual, expected) in in_place
        .view()
        .unwrap()
        .as_slice()
        .iter()
        .zip(&in_place_original)
    {
        let expected = Complex::new(
            <R as TestReal>::from_f64(R::to_f64(expected.re) * spatial_scale),
            <R as TestReal>::from_f64(R::to_f64(expected.im) * spatial_scale),
        );
        assert_close(*actual, expected);
    }

    // The same array and workspace can be used again after a successful pair.
    plan.forward_in_place(&mut in_place, &mut in_place_workspace)
        .unwrap();
    plan.inverse_in_place(&mut in_place, &mut in_place_workspace)
        .unwrap();
    assert_eq!(in_place.state(), C2cState::Input);
    assert_eq!(
        in_place.view().unwrap().as_slice().as_ptr(),
        in_place_storage
    );

    if extra_shape.element_count() == 0 {
        assert!(spectrum.is_empty());
        assert!(inverse.is_empty());
    } else if !spectrum.is_empty() {
        assert_ne!(spectrum.as_slice(), spectrum_before.as_slice());
    }
    let snapshots = (
        oop_forward_snapshot,
        oop_inverse_snapshot,
        oop_backward_snapshot,
        in_place_forward_snapshot,
        in_place_inverse_snapshot,
        in_place_backward_snapshot,
    );
    topology_barrier(topology);
    snapshots
}

fn run_partial_subset_cases(topology_1d: &Arc<MpiTopology<1>>, topology_2d: &Arc<MpiTopology<2>>) {
    let shape = [2, 3, 2, 4];
    let selections = [
        AxisSelection::from_indices([2, 0]).unwrap(),
        AxisSelection::empty(),
        AxisSelection::all(),
    ];
    for selection in selections {
        for method in [TransposeMethod::AllToAllv, TransposeMethod::PointToPoint] {
            run_partial_c2c_delta::<4, 2>(topology_2d, shape, selection, method);
        }
    }
    for method in [TransposeMethod::AllToAllv, TransposeMethod::PointToPoint] {
        run_empty_c2c_in_place::<4, 2>(topology_2d, shape, method);
    }
    for selection in [
        AxisSelection::from_indices([0]).unwrap(),
        AxisSelection::from_indices([2, 0]).unwrap(),
        AxisSelection::from_indices([3, 0]).unwrap(),
    ] {
        for method in [TransposeMethod::AllToAllv, TransposeMethod::PointToPoint] {
            run_partial_r2c_delta::<4, 2>(topology_2d, shape, selection, method);
        }
    }
    // The same non-last real boundary also exercises the one-dimensional
    // process-grid route.
    for method in [TransposeMethod::AllToAllv, TransposeMethod::PointToPoint] {
        run_partial_r2c_delta::<4, 1>(
            topology_1d,
            shape,
            AxisSelection::from_indices([0]).unwrap(),
            method,
        );
    }
}

fn run_empty_c2c_in_place<const N: usize, const M: usize>(
    topology: &Arc<MpiTopology<M>>,
    shape: [usize; N],
    method: TransposeMethod,
) where
    Complex<f64>: mpi::datatype::Equivalence,
{
    let selection = AxisSelection::empty();
    let plan = C2cPlan::<f64, N, M>::from_shape_with_selection_and_method(
        Arc::clone(topology),
        shape,
        ExtraShape::scalar(),
        selection,
        method,
    )
    .unwrap();
    let mut array = plan.allocate_in_place().unwrap();
    set_delta_complex_in_place(&mut array);
    let pointer = array.view().unwrap().as_slice().as_ptr();
    let mut workspace = plan.allocate_in_place_workspace().unwrap();
    assert_eq!(array.state(), C2cState::Input);
    assert!(
        array
            .view()
            .unwrap()
            .pencil()
            .same_layout(plan.input_pencil().as_ref())
    );

    plan.forward_in_place(&mut array, &mut workspace).unwrap();
    assert_eq!(array.state(), C2cState::Output);
    assert_eq!(array.view().unwrap().as_slice().as_ptr(), pointer);
    assert!(
        array
            .view()
            .unwrap()
            .pencil()
            .same_layout(plan.output_pencil().as_ref())
    );
    assert_delta_complex_view(&array.view().unwrap(), selection, 1.0);

    let output_before = array.view().unwrap().as_slice().to_vec();
    let workspace_before = format!("{workspace:?}");
    assert!(matches!(
        plan.forward_in_place(&mut array, &mut workspace),
        Err(FftError::InputLayoutMismatch)
    ));
    assert_eq!(array.state(), C2cState::Output);
    assert_eq!(array.view().unwrap().as_slice(), output_before.as_slice());
    assert_eq!(format!("{workspace:?}"), workspace_before);

    plan.inverse_in_place(&mut array, &mut workspace).unwrap();
    assert_eq!(array.state(), C2cState::Input);
    assert_eq!(array.view().unwrap().as_slice().as_ptr(), pointer);
    assert!(
        array
            .view()
            .unwrap()
            .pencil()
            .same_layout(plan.input_pencil().as_ref())
    );
    assert_delta_complex_view(&array.view().unwrap(), selection, 1.0);

    plan.forward_in_place(&mut array, &mut workspace).unwrap();
    plan.backward_in_place(&mut array, &mut workspace).unwrap();
    assert_eq!(array.state(), C2cState::Input);
    assert_eq!(array.view().unwrap().as_slice().as_ptr(), pointer);
    assert_delta_complex_view(&array.view().unwrap(), selection, 1.0);
}

fn run_partial_c2c_delta<const N: usize, const M: usize>(
    topology: &Arc<MpiTopology<M>>,
    shape: [usize; N],
    selection: AxisSelection<N>,
    method: TransposeMethod,
) where
    Complex<f64>: mpi::datatype::Equivalence,
{
    let plan = C2cPlan::<f64, N, M>::from_shape_with_selection_and_method(
        Arc::clone(topology),
        shape,
        ExtraShape::scalar(),
        selection,
        method,
    )
    .unwrap();
    let mut source = plan.allocate_input().unwrap();
    set_delta_complex(&mut source);
    let source_before = source.as_slice().to_vec();
    let mut spectrum = plan.allocate_output().unwrap();
    let mut workspace = plan.allocate_out_of_place_workspace().unwrap();
    plan.forward(&source, &mut spectrum, &mut workspace)
        .unwrap();
    assert_eq!(source.as_slice(), source_before.as_slice());
    assert_delta_complex_spectrum(&spectrum, selection, 1.0);
    let spectrum_before = spectrum.as_slice().to_vec();
    let mut recovered = plan.allocate_input().unwrap();
    plan.inverse(&spectrum, &mut recovered, &mut workspace)
        .unwrap();
    assert_eq!(spectrum.as_slice(), spectrum_before.as_slice());
    assert_delta_complex_input(&recovered, 1.0);
    let mut raw = plan.allocate_input().unwrap();
    plan.backward(&spectrum, &mut raw, &mut workspace).unwrap();
    let scale = (0..N)
        .filter(|&axis| selection.contains(axis))
        .map(|axis| shape[axis])
        .product::<usize>() as f64;
    assert_delta_complex_input(&raw, scale);
}

fn run_partial_r2c_delta<const N: usize, const M: usize>(
    topology: &Arc<MpiTopology<M>>,
    shape: [usize; N],
    selection: AxisSelection<N>,
    method: TransposeMethod,
) where
    f64: mpi::datatype::Equivalence,
    Complex<f64>: mpi::datatype::Equivalence,
{
    let plan = R2cPlan::<f64, N, M>::from_shape_with_selection_and_method(
        Arc::clone(topology),
        shape,
        ExtraShape::scalar(),
        selection,
        method,
    )
    .unwrap();
    let reduction_axis = (0..N).rev().find(|&axis| selection.contains(axis)).unwrap();
    assert_eq!(
        plan.output_pencil().global_shape()[reduction_axis],
        shape[reduction_axis] / 2 + 1
    );
    let mut source = plan.allocate_input().unwrap();
    set_delta_real(&mut source);
    let source_before = source.as_slice().to_vec();
    let mut spectrum = plan.allocate_output().unwrap();
    let mut workspace = plan.allocate_workspace().unwrap();
    plan.forward(&source, &mut spectrum, &mut workspace)
        .unwrap();
    assert_eq!(source.as_slice(), source_before.as_slice());
    assert_delta_r2c_spectrum(&spectrum, selection);
    let spectrum_before = spectrum.as_slice().to_vec();
    let mut recovered = plan.allocate_input().unwrap();
    plan.inverse(&spectrum, &mut recovered, &mut workspace)
        .unwrap();
    assert_eq!(spectrum.as_slice(), spectrum_before.as_slice());
    assert_delta_real(&recovered, 1.0);
    let mut raw = plan.allocate_input().unwrap();
    plan.backward(&spectrum, &mut raw, &mut workspace).unwrap();
    let scale = (0..N)
        .filter(|&axis| selection.contains(axis))
        .map(|axis| shape[axis])
        .product::<usize>() as f64;
    assert_delta_real(&raw, scale);
}

fn set_delta_complex<const N: usize, const M: usize>(array: &mut PencilArray<Complex<f64>, N, M>) {
    let local_shape = array.pencil().local_shape_logical();
    let starts: [usize; N] = std::array::from_fn(|axis| array.pencil().local_ranges()[axis].start);
    for linear in 0..array.pencil().local_len() {
        let local = unravel_spatial(linear, local_shape);
        let global: [usize; N] = std::array::from_fn(|axis| starts[axis] + local[axis]);
        *array.get_local_mut(&[], local).unwrap() = if global.iter().all(|&value| value == 0) {
            Complex::new(1.0, 0.0)
        } else {
            Complex::new(0.0, 0.0)
        };
    }
}

fn set_delta_real<const N: usize, const M: usize>(array: &mut PencilArray<f64, N, M>) {
    let local_shape = array.pencil().local_shape_logical();
    let starts: [usize; N] = std::array::from_fn(|axis| array.pencil().local_ranges()[axis].start);
    for linear in 0..array.pencil().local_len() {
        let local = unravel_spatial(linear, local_shape);
        let global: [usize; N] = std::array::from_fn(|axis| starts[axis] + local[axis]);
        *array.get_local_mut(&[], local).unwrap() = if global.iter().all(|&value| value == 0) {
            1.0
        } else {
            0.0
        };
    }
}

fn set_delta_complex_in_place<const N: usize, const M: usize>(
    array: &mut C2cInPlaceArray<f64, N, M>,
) {
    let mut view = array.view_mut().unwrap();
    let local_shape = view.pencil().local_shape_logical();
    let starts: [usize; N] = std::array::from_fn(|axis| view.pencil().local_ranges()[axis].start);
    for linear in 0..view.pencil().local_len() {
        let local = unravel_spatial(linear, local_shape);
        let global: [usize; N] = std::array::from_fn(|axis| starts[axis] + local[axis]);
        *view.get_local_mut(&[], local).unwrap() = if global.iter().all(|&value| value == 0) {
            Complex::new(1.0, 0.0)
        } else {
            Complex::new(0.0, 0.0)
        };
    }
}

fn assert_delta_complex_spectrum<const N: usize, const M: usize>(
    array: &PencilArray<Complex<f64>, N, M>,
    selection: AxisSelection<N>,
    scale: f64,
) {
    let view = array.view();
    assert_delta_complex_view(&view, selection, scale);
}

fn assert_delta_complex_view<const N: usize, const M: usize>(
    view: &PencilArrayView<'_, Complex<f64>, N, M>,
    selection: AxisSelection<N>,
    scale: f64,
) {
    let local_shape = view.pencil().local_shape_logical();
    let starts: [usize; N] = std::array::from_fn(|axis| view.pencil().local_ranges()[axis].start);
    for linear in 0..view.pencil().local_len() {
        let local = unravel_spatial(linear, local_shape);
        let global: [usize; N] = std::array::from_fn(|axis| starts[axis] + local[axis]);
        let expected = if global
            .iter()
            .enumerate()
            .all(|(axis, &value)| selection.contains(axis) || value == 0)
        {
            scale
        } else {
            0.0
        };
        let actual = view.get_local(&[], local).unwrap();
        assert!((actual.re - expected).abs() < 1e-9);
        assert!(actual.im.abs() < 1e-9);
    }
}

fn assert_delta_complex_input<const N: usize, const M: usize>(
    array: &PencilArray<Complex<f64>, N, M>,
    scale: f64,
) {
    let view = array.view();
    let local_shape = view.pencil().local_shape_logical();
    let starts: [usize; N] = std::array::from_fn(|axis| view.pencil().local_ranges()[axis].start);
    for linear in 0..view.pencil().local_len() {
        let local = unravel_spatial(linear, local_shape);
        let global: [usize; N] = std::array::from_fn(|axis| starts[axis] + local[axis]);
        let expected = if global.iter().all(|&value| value == 0) {
            scale
        } else {
            0.0
        };
        let actual = view.get_local(&[], local).unwrap();
        assert!((actual.re - expected).abs() < 1e-9);
        assert!(actual.im.abs() < 1e-9);
    }
}

fn assert_delta_r2c_spectrum<const N: usize, const M: usize>(
    array: &PencilArray<Complex<f64>, N, M>,
    selection: AxisSelection<N>,
) {
    assert_delta_complex_spectrum(array, selection, 1.0);
}

fn assert_delta_real<const N: usize, const M: usize>(array: &PencilArray<f64, N, M>, scale: f64) {
    let view = array.view();
    let local_shape = view.pencil().local_shape_logical();
    let starts: [usize; N] = std::array::from_fn(|axis| view.pencil().local_ranges()[axis].start);
    for linear in 0..view.pencil().local_len() {
        let local = unravel_spatial(linear, local_shape);
        let global: [usize; N] = std::array::from_fn(|axis| starts[axis] + local[axis]);
        let expected = if global.iter().all(|&value| value == 0) {
            scale
        } else {
            0.0
        };
        let actual = *view.get_local(&[], local).unwrap();
        assert!((actual - expected).abs() < 1e-9);
    }
}

fn run_r2c_cases(topology_1d: &Arc<MpiTopology<1>>, topology_2d: &Arc<MpiTopology<2>>) {
    // Both transports run every case; comparing the two returned local
    // matrices catches parity errors in either the forward or inverse path.
    assert_r2c_method_parity::<f64, 2, 1>(topology_1d, [3, 1], ExtraShape::scalar(), 20.0);
    assert_r2c_method_parity::<f64, 2, 1>(
        topology_1d,
        [3, 2],
        ExtraShape::new([2, 3]).unwrap(),
        20.5,
    );
    assert_r2c_method_parity::<f64, 3, 1>(topology_1d, [2, 3, 2], ExtraShape::scalar(), 21.0);
    assert_r2c_method_parity::<f64, 3, 1>(
        topology_1d,
        [3, 2, 5],
        ExtraShape::new([2]).unwrap(),
        21.5,
    );
    assert_r2c_method_parity::<f64, 4, 2>(
        topology_2d,
        [3, 1, 2, 4],
        ExtraShape::new([2, 3]).unwrap(),
        22.0,
    );
    assert_r2c_method_parity::<f64, 4, 2>(topology_2d, [3, 1, 2, 3], ExtraShape::scalar(), 22.5);
    assert_r2c_method_parity::<f64, 4, 2>(
        topology_2d,
        [3, 1, 2, 4],
        ExtraShape::new([0]).unwrap(),
        23.0,
    );

    assert_r2c_method_parity::<f32, 2, 1>(topology_1d, [3, 1], ExtraShape::scalar(), 24.0);
    assert_r2c_method_parity::<f32, 2, 1>(topology_1d, [3, 2], ExtraShape::new([2]).unwrap(), 24.5);
    assert_r2c_method_parity::<f32, 3, 1>(
        topology_1d,
        [2, 3, 2],
        ExtraShape::new([0]).unwrap(),
        25.0,
    );
    assert_r2c_method_parity::<f32, 4, 2>(
        topology_2d,
        [3, 1, 2, 4],
        ExtraShape::new([2, 3]).unwrap(),
        25.5,
    );
    assert_r2c_method_parity::<f32, 4, 2>(topology_2d, [3, 1, 2, 3], ExtraShape::scalar(), 26.0);

    run_r2c_boundary_cases(topology_1d);
    run_r2c_boundary_hard_cases(topology_1d, topology_2d);
}

fn run_r2c_in_place_cases(topology_1d: &Arc<MpiTopology<1>>, topology_2d: &Arc<MpiTopology<2>>) {
    run_r2c_in_place_case::<f64, 2, 1>(
        topology_1d,
        [3, 4],
        AxisSelection::all(),
        ExtraShape::scalar(),
        30.0,
        TransposeMethod::AllToAllv,
    );
    run_r2c_in_place_case::<f64, 3, 1>(
        topology_1d,
        [3, 2, 5],
        AxisSelection::from_indices([1]).unwrap(),
        ExtraShape::new([2]).unwrap(),
        30.5,
        TransposeMethod::PointToPoint,
    );
    run_r2c_in_place_case::<f64, 3, 1>(
        topology_1d,
        [3, 2, 5],
        AxisSelection::from_indices([0]).unwrap(),
        ExtraShape::scalar(),
        31.0,
        TransposeMethod::AllToAllv,
    );
    run_r2c_in_place_case::<f32, 4, 2>(
        topology_2d,
        [3, 1, 2, 4],
        AxisSelection::from_indices([2]).unwrap(),
        ExtraShape::new([2, 3]).unwrap(),
        31.5,
        TransposeMethod::PointToPoint,
    );
    run_r2c_in_place_case::<f32, 4, 2>(
        topology_2d,
        [3, 1, 2, 4],
        AxisSelection::from_indices([0]).unwrap(),
        ExtraShape::scalar(),
        32.0,
        TransposeMethod::AllToAllv,
    );
    run_r2c_in_place_case::<f64, 4, 2>(
        topology_2d,
        [3, 1, 2, 4],
        AxisSelection::all(),
        ExtraShape::new([0]).unwrap(),
        32.5,
        TransposeMethod::PointToPoint,
    );
    run_r2c_in_place_poison_case(topology_1d);
}

fn run_r2c_in_place_poison_case(topology: &Arc<MpiTopology<1>>) {
    let plan = R2cPlan::<f64, 2, 1>::from_shape_with_method(
        Arc::clone(topology),
        [3, 4],
        ExtraShape::scalar(),
        TransposeMethod::AllToAllv,
    )
    .unwrap();
    let foreign_plan = R2cPlan::<f64, 2, 1>::from_shape_with_method(
        Arc::clone(topology),
        [3, 4],
        ExtraShape::scalar(),
        TransposeMethod::AllToAllv,
    )
    .unwrap();
    let mut foreign_array = foreign_plan.allocate_in_place().unwrap();
    foreign_array
        .real_view_mut()
        .unwrap()
        .as_mut_slice()
        .fill(2.0);
    let mut foreign_array_workspace = plan.allocate_in_place_workspace().unwrap();
    let foreign_array_before = foreign_array.real_view().unwrap().as_slice().to_vec();
    let foreign_array_workspace_before = format!("{foreign_array_workspace:?}");
    assert!(matches!(
        plan.forward_in_place(&mut foreign_array, &mut foreign_array_workspace),
        Err(R2cError::Fft(FftError::Array(
            ArrayError::IncompatiblePencils
        )))
    ));
    assert_eq!(foreign_array.state(), R2cState::RealInput);
    assert_eq!(
        foreign_array.real_view().unwrap().as_slice(),
        foreign_array_before.as_slice()
    );
    assert_eq!(
        format!("{foreign_array_workspace:?}"),
        foreign_array_workspace_before
    );

    let mut workspace_foreign = foreign_plan.allocate_in_place_workspace().unwrap();
    let mut own_array = plan.allocate_in_place().unwrap();
    own_array.real_view_mut().unwrap().as_mut_slice().fill(3.0);
    let own_array_before = own_array.real_view().unwrap().as_slice().to_vec();
    let workspace_foreign_before = format!("{workspace_foreign:?}");
    assert!(matches!(
        plan.forward_in_place(&mut own_array, &mut workspace_foreign),
        Err(R2cError::Fft(FftError::WorkspaceMismatch))
    ));
    assert_eq!(own_array.state(), R2cState::RealInput);
    assert_eq!(
        own_array.real_view().unwrap().as_slice(),
        own_array_before.as_slice()
    );
    assert_eq!(format!("{workspace_foreign:?}"), workspace_foreign_before);

    let mut array = plan.allocate_in_place().unwrap();
    array.real_view_mut().unwrap().as_mut_slice().fill(1.0);
    let before = array.real_view().unwrap().as_slice().to_vec();
    let pointer = array.real_view().unwrap().as_slice().as_ptr() as usize;
    let mut workspace = plan.allocate_in_place_workspace().unwrap();
    assert!(matches!(
        plan.inverse_in_place(&mut array, &mut workspace),
        Err(R2cError::Fft(FftError::InputLayoutMismatch))
    ));
    assert_eq!(array.state(), R2cState::RealInput);
    assert_eq!(array.real_view().unwrap().as_slice(), before.as_slice());
    assert_eq!(
        array.real_view().unwrap().as_slice().as_ptr() as usize,
        pointer
    );

    plan.forward_in_place(&mut array, &mut workspace).unwrap();
    if let Some(value) = array.complex_view_mut().unwrap().as_mut_slice().first_mut() {
        value.im = 1.0;
    }
    assert!(matches!(
        plan.inverse_in_place(&mut array, &mut workspace),
        Err(R2cError::InvalidSpectrum)
    ));
    assert_eq!(array.state(), R2cState::Poisoned);
    assert!(array.real_view().is_err());
    assert!(array.complex_view().is_err());
    assert!(matches!(
        plan.forward_in_place(&mut array, &mut workspace),
        Err(R2cError::Fft(FftError::Array(_)))
    ));
}

fn run_r2c_in_place_case<R: TestReal, const N: usize, const M: usize>(
    topology: &Arc<MpiTopology<M>>,
    global_shape: [usize; N],
    selection: AxisSelection<N>,
    extra_shape: ExtraShape,
    seed: f64,
    method: TransposeMethod,
) where
    Complex<R>: mpi::datatype::Equivalence,
{
    let plan = R2cPlan::<R, N, M>::from_shape_with_selection_and_method(
        Arc::clone(topology),
        global_shape,
        extra_shape.clone(),
        selection,
        method,
    )
    .unwrap();
    let mut source = plan.allocate_input().unwrap();
    fill_r2c_input(&mut source, seed);
    let source_before = source.as_slice().to_vec();
    let mut spectrum = plan.allocate_output().unwrap();
    let mut oop_workspace = plan.allocate_workspace().unwrap();
    plan.forward(&source, &mut spectrum, &mut oop_workspace)
        .unwrap();
    let spectrum_before = spectrum.as_slice().to_vec();
    let mut expected_inverse = plan.allocate_input().unwrap();
    plan.inverse(&spectrum, &mut expected_inverse, &mut oop_workspace)
        .unwrap();
    let mut expected_backward = plan.allocate_input().unwrap();
    plan.backward(&spectrum, &mut expected_backward, &mut oop_workspace)
        .unwrap();

    let mut array = plan.allocate_in_place().unwrap();
    {
        let mut view = array.real_view_mut().unwrap();
        assert_eq!(view.len(), source_before.len());
        view.as_mut_slice().copy_from_slice(&source_before);
    }
    assert!(matches!(
        array.complex_view(),
        Err(R2cError::Fft(FftError::OutputLayoutMismatch))
    ));
    let pointer = array.real_view().unwrap().as_slice().as_ptr() as usize;
    let mut workspace = plan.allocate_in_place_workspace().unwrap();
    plan.forward_in_place(&mut array, &mut workspace).unwrap();
    assert_eq!(array.state(), R2cState::ComplexOutput);
    assert_eq!(
        array.complex_view().unwrap().as_slice().as_ptr() as usize,
        pointer
    );
    if selection.is_all() {
        check_r2c_forward(
            &array.complex_view().unwrap(),
            &extra_shape,
            global_shape,
            seed,
        );
    }
    for (actual, expected) in array
        .complex_view()
        .unwrap()
        .as_slice()
        .iter()
        .zip(&spectrum_before)
    {
        assert_close(*actual, *expected);
    }
    assert!(matches!(
        array.real_view(),
        Err(R2cError::Fft(FftError::InputLayoutMismatch))
    ));

    if let Err(error) = plan.inverse_in_place(&mut array, &mut workspace) {
        panic!(
            "in-place inverse failed for shape={global_shape:?}, selection={selection:?}, method={method:?}: {error:?}"
        );
    }
    assert_eq!(array.state(), R2cState::RealInput);
    assert_eq!(
        array.real_view().unwrap().as_slice().as_ptr() as usize,
        pointer
    );
    for (actual, expected) in array
        .real_view()
        .unwrap()
        .as_slice()
        .iter()
        .zip(expected_inverse.as_slice())
    {
        let bound = R::tolerance() * (1.0 + R::to_f64(*expected).abs());
        assert!((R::to_f64(*actual) - R::to_f64(*expected)).abs() <= bound);
    }

    plan.forward_in_place(&mut array, &mut workspace).unwrap();
    plan.backward_in_place(&mut array, &mut workspace).unwrap();
    assert_eq!(array.state(), R2cState::RealInput);
    assert_eq!(
        array.real_view().unwrap().as_slice().as_ptr() as usize,
        pointer
    );
    for (actual, expected) in array
        .real_view()
        .unwrap()
        .as_slice()
        .iter()
        .zip(expected_backward.as_slice())
    {
        let bound = R::tolerance() * (1.0 + R::to_f64(*expected).abs());
        assert!((R::to_f64(*actual) - R::to_f64(*expected)).abs() <= bound);
    }
}

fn assert_r2c_method_parity<R: TestReal, const N: usize, const M: usize>(
    topology: &Arc<MpiTopology<M>>,
    global_shape: [usize; N],
    extra_shape: ExtraShape,
    seed: f64,
) -> (Vec<Complex<R>>, Vec<R>)
where
    Complex<R>: mpi::datatype::Equivalence,
{
    let alltoallv = run_r2c_case::<R, N, M>(
        topology,
        global_shape,
        extra_shape.clone(),
        seed,
        TransposeMethod::AllToAllv,
    );
    let point_to_point = run_r2c_case::<R, N, M>(
        topology,
        global_shape,
        extra_shape,
        seed,
        TransposeMethod::PointToPoint,
    );
    assert_eq!(alltoallv, point_to_point);
    alltoallv
}

fn run_r2c_case<R: TestReal, const N: usize, const M: usize>(
    topology: &Arc<MpiTopology<M>>,
    global_shape: [usize; N],
    extra_shape: ExtraShape,
    seed: f64,
    method: TransposeMethod,
) -> (Vec<Complex<R>>, Vec<R>)
where
    Complex<R>: mpi::datatype::Equivalence,
{
    let plan = R2cPlan::<R, N, M>::from_shape_with_method(
        Arc::clone(topology),
        global_shape,
        extra_shape.clone(),
        method,
    )
    .unwrap();
    let mut source = plan.allocate_input().unwrap();
    fill_r2c_input(&mut source, seed);
    assert_r2c_layout(&plan, global_shape, &extra_shape);

    // Exercise the legacy/default constructors and their method-selecting
    // forms. They must describe the same reduced endpoint.
    let legacy_shape =
        R2cPlan::<R, N, M>::from_shape(Arc::clone(topology), global_shape, extra_shape.clone())
            .unwrap();
    let legacy_pencil =
        R2cPlan::<R, N, M>::from_pencil(Arc::clone(plan.input_pencil()), extra_shape.clone())
            .unwrap();
    let legacy_array = R2cPlan::<R, N, M>::from_array(&source).unwrap();
    let explicit_pencil = R2cPlan::<R, N, M>::from_pencil_with_method(
        Arc::clone(plan.input_pencil()),
        extra_shape.clone(),
        method,
    )
    .unwrap();
    let explicit_array = R2cPlan::<R, N, M>::from_array_with_method(&source, method).unwrap();
    for candidate in [
        &legacy_shape,
        &legacy_pencil,
        &legacy_array,
        &explicit_pencil,
        &explicit_array,
    ] {
        assert!(
            candidate
                .output_pencil()
                .same_layout(plan.output_pencil().as_ref())
        );
        assert_eq!(candidate.extra_shape(), &extra_shape);
    }
    let source_before = source.as_slice().to_vec();
    let mut spectrum = plan.allocate_output().unwrap();
    let mut workspace = plan.allocate_workspace().unwrap();
    plan.forward(&source, &mut spectrum, &mut workspace)
        .unwrap();
    assert_eq!(source.as_slice(), source_before.as_slice());
    check_r2c_forward(&spectrum.view(), &extra_shape, global_shape, seed);

    let forward_snapshot = spectrum.as_slice().to_vec();
    let mut recovered = plan.allocate_input().unwrap();
    plan.inverse(&spectrum, &mut recovered, &mut workspace)
        .unwrap();
    assert_eq!(spectrum.as_slice(), forward_snapshot.as_slice());
    for (actual, expected) in recovered.as_slice().iter().zip(&source_before) {
        assert!((R::to_f64(*actual) - R::to_f64(*expected)).abs() <= R::tolerance());
    }
    let mut raw_roundtrip = plan.allocate_input().unwrap();
    plan.backward(&spectrum, &mut raw_roundtrip, &mut workspace)
        .unwrap();
    let spatial_scale = global_shape.iter().product::<usize>() as f64;
    for (actual, expected) in raw_roundtrip.as_slice().iter().zip(&source_before) {
        assert!(
            (R::to_f64(*actual) - R::to_f64(*expected) * spatial_scale).abs()
                <= R::tolerance() * (1.0 + R::to_f64(*expected).abs() * spatial_scale)
        );
    }

    fill_r2c_spectrum(&mut spectrum, global_shape, seed + 17.0);
    assert_raw_boundary_imaginary(&spectrum.view(), &extra_shape, global_shape);
    let arbitrary_before = spectrum.as_slice().to_vec();
    let mut arbitrary_inverse = plan.allocate_input().unwrap();
    plan.inverse(&spectrum, &mut arbitrary_inverse, &mut workspace)
        .unwrap();
    assert_eq!(spectrum.as_slice(), arbitrary_before.as_slice());
    check_r2c_inverse(
        &arbitrary_inverse.view(),
        &extra_shape,
        global_shape,
        seed + 17.0,
    );
    let mut arbitrary_backward = plan.allocate_input().unwrap();
    plan.backward(&spectrum, &mut arbitrary_backward, &mut workspace)
        .unwrap();
    check_r2c_backward(
        &arbitrary_backward.view(),
        &extra_shape,
        global_shape,
        seed + 17.0,
    );

    plan.forward(&source, &mut spectrum, &mut workspace)
        .unwrap();
    assert_eq!(source.as_slice(), source_before.as_slice());
    (forward_snapshot, arbitrary_inverse.as_slice().to_vec())
}

fn assert_r2c_layout<R: TestReal, const N: usize, const M: usize>(
    plan: &R2cPlan<R, N, M>,
    original_shape: [usize; N],
    extra_shape: &ExtraShape,
) where
    Complex<R>: mpi::datatype::Equivalence,
{
    assert_eq!(plan.input_pencil().global_shape(), &original_shape);
    assert_eq!(plan.extra_shape(), extra_shape);
    let output = plan.output_pencil();
    let expected_shape = std::array::from_fn(|axis| {
        if axis + 1 == N {
            original_shape[axis] / 2 + 1
        } else {
            original_shape[axis]
        }
    });
    assert_eq!(output.global_shape(), &expected_shape);
    assert_eq!(
        output
            .decomposition()
            .iter()
            .map(|axis| axis.index())
            .collect::<Vec<_>>(),
        (1..=M).collect::<Vec<_>>()
    );
    assert_eq!(
        output
            .permutation()
            .axes()
            .iter()
            .map(|axis| axis.index())
            .collect::<Vec<_>>(),
        (0..N).rev().collect::<Vec<_>>()
    );
}

fn assert_raw_boundary_imaginary<R: TestReal, const N: usize, const M: usize>(
    array: &PencilArrayView<'_, Complex<R>, N, M>,
    extra_shape: &ExtraShape,
    original_shape: [usize; N],
) {
    if extra_shape.element_count() == 0 || !original_shape[..N - 1].iter().any(|&n| n > 2) {
        return;
    }
    let plane_count = if original_shape[N - 1] % 2 == 0 { 2 } else { 1 };
    let mut local = [0.0_f64; 2];
    let ranges = array.pencil().local_ranges();
    let local_shape = array.pencil().local_shape_logical();
    for extra_linear in 0..extra_shape.element_count() {
        let extra = unravel_extra(extra_linear, extra_shape.dimensions());
        for spatial_linear in 0..array.pencil().local_len() {
            let spatial = unravel_spatial(spatial_linear, local_shape);
            let frequency: [usize; N] =
                std::array::from_fn(|axis| ranges[axis].start + spatial[axis]);
            let plane = if frequency[N - 1] == 0 {
                Some(0)
            } else if plane_count == 2 && frequency[N - 1] == original_shape[N - 1] / 2 {
                Some(1)
            } else {
                None
            };
            if let Some(plane) = plane {
                let value = array
                    .get_local(&extra, spatial)
                    .expect("raw half-spectrum index is local");
                local[plane] = local[plane].max(R::to_f64(value.im).abs());
            }
        }
    }
    let mut global = [0.0_f64; 2];
    array.pencil().topology().communicator().all_reduce_into(
        &local,
        &mut global,
        SystemOperation::max(),
    );
    for (plane, value) in global.iter().enumerate().take(plane_count) {
        assert!(
            *value > 0.0,
            "valid raw boundary plane {plane} unexpectedly has zero imaginary part"
        );
    }
}

fn fill_r2c_input<R: TestReal, const N: usize, const M: usize>(
    array: &mut PencilArray<R, N, M>,
    seed: f64,
) {
    let dimensions = array.extra_shape().dimensions().to_vec();
    let local_shape = array.pencil().local_shape_logical();
    for extra_linear in 0..array.extra_shape().element_count() {
        let extra = unravel_extra(extra_linear, &dimensions);
        for spatial_linear in 0..array.pencil().local_len() {
            let spatial = unravel_spatial(spatial_linear, local_shape);
            let global: [usize; N] = std::array::from_fn(|axis| {
                array.pencil().local_ranges()[axis].start + spatial[axis]
            });
            *array
                .get_local_mut(&extra, spatial)
                .expect("real input index is local") =
                <R as TestReal>::from_f64(r2c_real_value(&extra, global, seed));
        }
    }
}

fn fill_r2c_spectrum<R: TestReal, const N: usize, const M: usize>(
    array: &mut PencilArray<Complex<R>, N, M>,
    original_shape: [usize; N],
    seed: f64,
) {
    let dimensions = array.extra_shape().dimensions().to_vec();
    let local_shape = array.pencil().local_shape_logical();
    for extra_linear in 0..array.extra_shape().element_count() {
        let extra = unravel_extra(extra_linear, &dimensions);
        for spatial_linear in 0..array.pencil().local_len() {
            let spatial = unravel_spatial(spatial_linear, local_shape);
            let frequency = std::array::from_fn(|axis| {
                array.pencil().local_ranges()[axis].start + spatial[axis]
            });
            let value = arbitrary_half_value(&extra, frequency, original_shape, seed);
            *array
                .get_local_mut(&extra, spatial)
                .expect("half-spectrum index is local") = Complex::new(
                <R as TestReal>::from_f64(value.re),
                <R as TestReal>::from_f64(value.im),
            );
        }
    }
}

fn check_r2c_forward<R: TestReal, const N: usize, const M: usize>(
    array: &PencilArrayView<'_, Complex<R>, N, M>,
    extra_shape: &ExtraShape,
    original_shape: [usize; N],
    seed: f64,
) {
    let local_shape = array.pencil().local_shape_logical();
    let ranges = array.pencil().local_ranges();
    let dimensions = extra_shape.dimensions();
    for extra_linear in 0..extra_shape.element_count() {
        let extra = unravel_extra(extra_linear, dimensions);
        for spatial_linear in 0..array.pencil().local_len() {
            let spatial = unravel_spatial(spatial_linear, local_shape);
            let frequency: [usize; N] =
                std::array::from_fn(|axis| ranges[axis].start + spatial[axis]);
            let expected = r2c_half_value::<R, N>(&extra, frequency, original_shape, seed);
            let actual = *array
                .get_local(&extra, spatial)
                .expect("forward half-spectrum index is local");
            let bound = R::tolerance() * (1.0 + expected.re.abs().max(expected.im.abs()));
            assert!((R::to_f64(actual.re) - expected.re).abs() <= bound);
            assert!((R::to_f64(actual.im) - expected.im).abs() <= bound);
        }
    }
}

fn check_r2c_inverse<R: TestReal, const N: usize, const M: usize>(
    array: &PencilArrayView<'_, R, N, M>,
    extra_shape: &ExtraShape,
    original_shape: [usize; N],
    seed: f64,
) {
    let local_shape = array.pencil().local_shape_logical();
    let ranges = array.pencil().local_ranges();
    let dimensions = extra_shape.dimensions();
    for extra_linear in 0..extra_shape.element_count() {
        let extra = unravel_extra(extra_linear, dimensions);
        for spatial_linear in 0..array.pencil().local_len() {
            let spatial = unravel_spatial(spatial_linear, local_shape);
            let target = std::array::from_fn(|axis| ranges[axis].start + spatial[axis]);
            let expected = r2c_inverse_value::<R, N>(&extra, target, original_shape, seed);
            let actual = R::to_f64(
                *array
                    .get_local(&extra, spatial)
                    .expect("inverse real index is local"),
            );
            assert!((actual - expected).abs() <= R::tolerance() * (1.0 + expected.abs()));
        }
    }
}

fn check_r2c_backward<R: TestReal, const N: usize, const M: usize>(
    array: &PencilArrayView<'_, R, N, M>,
    extra_shape: &ExtraShape,
    original_shape: [usize; N],
    seed: f64,
) {
    let local_shape = array.pencil().local_shape_logical();
    let ranges = array.pencil().local_ranges();
    let dimensions = extra_shape.dimensions();
    let scale = original_shape.iter().product::<usize>() as f64;
    for extra_linear in 0..extra_shape.element_count() {
        let extra = unravel_extra(extra_linear, dimensions);
        for spatial_linear in 0..array.pencil().local_len() {
            let spatial = unravel_spatial(spatial_linear, local_shape);
            let target = std::array::from_fn(|axis| ranges[axis].start + spatial[axis]);
            let expected = r2c_inverse_value::<R, N>(&extra, target, original_shape, seed) * scale;
            let actual = R::to_f64(
                *array
                    .get_local(&extra, spatial)
                    .expect("backward real index is local"),
            );
            assert!((actual - expected).abs() <= R::tolerance() * (1.0 + expected.abs()));
        }
    }
}

fn r2c_real_value<const N: usize>(extra: &[usize], spatial: [usize; N], seed: f64) -> f64 {
    let mut value = 0.17 + seed * 0.013;
    for (index, &coordinate) in extra.iter().enumerate() {
        value += (index + 2) as f64 * (coordinate + 1) as f64 * 0.041;
    }
    for (axis, coordinate) in spatial.into_iter().enumerate() {
        value += (axis + 1) as f64 * (coordinate + 1) as f64 * 0.23;
        value += ((coordinate + axis + 1) as f64).sin() * 0.019;
    }
    // Pairwise products make the real fixture genuinely nonseparable. A
    // separable sum cannot expose a mistaken mirror on another transverse axis.
    for left in 0..N {
        for right in left + 1..N {
            value += 0.013
                * (left + 1) as f64
                * (right + 2) as f64
                * (spatial[left] + 1) as f64
                * (spatial[right] + 2) as f64;
        }
    }
    value
}

fn r2c_half_value<R: TestReal, const N: usize>(
    extra: &[usize],
    frequency: [usize; N],
    shape: [usize; N],
    seed: f64,
) -> Complex<f64> {
    let mut real = 0.0;
    let mut imaginary = 0.0;
    let total = shape.iter().product::<usize>();
    for linear in 0..total {
        let spatial = unravel_spatial(linear, shape);
        let value = R::to_f64(<R as TestReal>::from_f64(r2c_real_value(
            extra, spatial, seed,
        )));
        let phase = -TAU
            * (0..N)
                .map(|axis| spatial[axis] as f64 * frequency[axis] as f64 / shape[axis] as f64)
                .sum::<f64>();
        let (sin, cos) = phase.sin_cos();
        real += value * cos;
        imaginary += value * sin;
    }
    Complex::new(real, imaginary)
}

fn arbitrary_base_value<const N: usize>(
    extra: &[usize],
    frequency: [usize; N],
    shape: [usize; N],
    seed: f64,
) -> Complex<f64> {
    let mut real = 0.31 + seed * 0.017;
    let mut imaginary = -0.29 - seed * 0.011;
    for (index, &coordinate) in extra.iter().enumerate() {
        real += (index + 2) as f64 * (coordinate + 1) as f64 * 0.067;
        imaginary -= (index + 3) as f64 * (coordinate + 1) as f64 * 0.053;
    }
    for (axis, &coordinate) in frequency.iter().enumerate() {
        real += (axis + 1) as f64 * (coordinate + 1) as f64 * 0.173;
        imaginary += (axis + 2) as f64 * (coordinate + 1) as f64 * 0.119;
        real += ((coordinate + axis + 2) as f64).sin() * 0.037;
    }
    for left in 0..N.saturating_sub(1) {
        for right in left + 1..N.saturating_sub(1) {
            real += 0.021 * (frequency[left] + 1) as f64 * (frequency[right] + 2) as f64;
            imaginary -=
                0.015 * (shape[left] + frequency[left] + 1) as f64 * (frequency[right] + 1) as f64;
        }
    }
    Complex::new(real, imaginary)
}

fn arbitrary_half_value<const N: usize>(
    extra: &[usize],
    frequency: [usize; N],
    shape: [usize; N],
    seed: f64,
) -> Complex<f64> {
    let last = frequency[N - 1];
    let half = shape[N - 1] / 2;
    let base = arbitrary_base_value(extra, frequency, shape, seed);
    if last != 0 && !(shape[N - 1] % 2 == 0 && last == half) {
        return base;
    }
    let mirror = std::array::from_fn(|axis| {
        if axis + 1 == N {
            last
        } else if frequency[axis] == 0 {
            0
        } else {
            shape[axis] - frequency[axis]
        }
    });
    let partner = arbitrary_base_value(extra, mirror, shape, seed);
    Complex::new(base.re + partner.re, base.im - partner.im)
}

fn converted_complex<R: TestReal>(value: Complex<f64>) -> Complex<f64> {
    Complex::new(
        R::to_f64(<R as TestReal>::from_f64(value.re)),
        R::to_f64(<R as TestReal>::from_f64(value.im)),
    )
}

fn r2c_inverse_value<R: TestReal, const N: usize>(
    extra: &[usize],
    target: [usize; N],
    shape: [usize; N],
    seed: f64,
) -> f64 {
    let total = shape.iter().product::<usize>();
    let mut result = 0.0;
    for linear in 0..total {
        let frequency = unravel_spatial(linear, shape);
        let spectrum = if frequency[N - 1] <= shape[N - 1] / 2 {
            converted_complex::<R>(arbitrary_half_value(extra, frequency, shape, seed))
        } else {
            let mirror = std::array::from_fn(|axis| {
                if axis + 1 == N {
                    shape[axis] - frequency[axis]
                } else if frequency[axis] == 0 {
                    0
                } else {
                    shape[axis] - frequency[axis]
                }
            });
            converted_complex::<R>(arbitrary_half_value(extra, mirror, shape, seed)).conj()
        };
        let phase = TAU
            * (0..N)
                .map(|axis| target[axis] as f64 * frequency[axis] as f64 / shape[axis] as f64)
                .sum::<f64>();
        let (sin, cos) = phase.sin_cos();
        result += spectrum.re * cos - spectrum.im * sin;
    }
    result / total as f64
}

fn set_r2c_global<R: TestReal, const N: usize, const M: usize>(
    array: &mut PencilArray<Complex<R>, N, M>,
    extra: &[usize],
    global: [usize; N],
    value: Complex<R>,
) {
    let local: [Option<usize>; N] = std::array::from_fn(|axis| {
        let range = &array.pencil().local_ranges()[axis];
        if range.start <= global[axis] && global[axis] < range.end {
            Some(global[axis] - range.start)
        } else {
            None
        }
    });
    if local.iter().all(Option::is_some) {
        let local = std::array::from_fn(|axis| local[axis].expect("local owner was checked"));
        *array
            .get_local_mut(extra, local)
            .expect("global endpoint is local on its owner") = value;
    }
}

fn run_r2c_boundary_cases(topology: &Arc<MpiTopology<1>>) {
    for method in [TransposeMethod::AllToAllv, TransposeMethod::PointToPoint] {
        let plan = R2cPlan::<f32, 3, 1>::from_shape_with_method(
            Arc::clone(topology),
            [128, 128, 3],
            ExtraShape::scalar(),
            method,
        )
        .unwrap();
        let mut spectrum = plan.allocate_output().unwrap();
        set_r2c_global(&mut spectrum, &[], [0, 0, 0], Complex::new(0.0, 1.0));
        let source_before = spectrum.as_slice().to_vec();
        let mut destination = plan.allocate_input().unwrap();
        for value in destination.as_mut_slice() {
            *value = 19.0;
        }
        let destination_before = destination.as_slice().to_vec();
        let mut workspace = plan.allocate_workspace().unwrap();
        assert!(matches!(
            plan.inverse(&spectrum, &mut destination, &mut workspace),
            Err(R2cError::InvalidSpectrum)
        ));
        assert_eq!(spectrum.as_slice(), source_before.as_slice());
        assert_eq!(destination.as_slice(), destination_before.as_slice());

        let mut real_source = plan.allocate_input().unwrap();
        fill_r2c_input(&mut real_source, 25.0);
        plan.forward(&real_source, &mut spectrum, &mut workspace)
            .unwrap();
        plan.inverse(&spectrum, &mut destination, &mut workspace)
            .unwrap();

        let batched = R2cPlan::<f32, 3, 1>::from_shape_with_method(
            Arc::clone(topology),
            [8, 8, 4],
            ExtraShape::new([2]).unwrap(),
            method,
        )
        .unwrap();
        let mut batched_spectrum = batched.allocate_output().unwrap();
        set_r2c_global(
            &mut batched_spectrum,
            &[1],
            [0, 0, 0],
            Complex::new(0.0, 1.0),
        );
        let batched_source_before = batched_spectrum.as_slice().to_vec();
        let mut batched_destination = batched.allocate_input().unwrap();
        let batched_destination_before = batched_destination.as_slice().to_vec();
        let mut batched_workspace = batched.allocate_workspace().unwrap();
        assert!(matches!(
            batched.inverse(
                &batched_spectrum,
                &mut batched_destination,
                &mut batched_workspace,
            ),
            Err(R2cError::InvalidSpectrum)
        ));
        assert_eq!(
            batched_spectrum.as_slice(),
            batched_source_before.as_slice()
        );
        assert_eq!(
            batched_destination.as_slice(),
            batched_destination_before.as_slice()
        );

        let small = R2cPlan::<f64, 2, 1>::from_shape_with_method(
            Arc::clone(topology),
            [1, 2],
            ExtraShape::scalar(),
            method,
        )
        .unwrap();
        let mut small_spectrum = small.allocate_output().unwrap();
        let mut small_destination = small.allocate_input().unwrap();
        let mut small_workspace = small.allocate_workspace().unwrap();
        set_r2c_global(
            &mut small_spectrum,
            &[],
            [0, 0],
            Complex::new(1.0, f64::from_bits(1)),
        );
        small
            .inverse(
                &small_spectrum,
                &mut small_destination,
                &mut small_workspace,
            )
            .unwrap();
        set_r2c_global(&mut small_spectrum, &[], [0, 0], Complex::new(1.0, 1e-15));
        small
            .inverse(
                &small_spectrum,
                &mut small_destination,
                &mut small_workspace,
            )
            .unwrap();
        set_r2c_global(&mut small_spectrum, &[], [0, 0], Complex::new(1.0, 1e-12));
        assert!(matches!(
            small.inverse(
                &small_spectrum,
                &mut small_destination,
                &mut small_workspace,
            ),
            Err(R2cError::InvalidSpectrum)
        ));
        set_r2c_global(
            &mut small_spectrum,
            &[],
            [0, 0],
            Complex::new(f64::NAN, 0.0),
        );
        assert!(matches!(
            small.inverse(
                &small_spectrum,
                &mut small_destination,
                &mut small_workspace,
            ),
            Err(R2cError::InvalidSpectrum)
        ));

        let odd = R2cPlan::<f64, 2, 1>::from_shape_with_method(
            Arc::clone(topology),
            [2, 3],
            ExtraShape::scalar(),
            method,
        )
        .unwrap();
        let mut odd_spectrum = odd.allocate_output().unwrap();
        set_r2c_global(&mut odd_spectrum, &[], [0, 1], Complex::new(0.0, 1.0));
        let mut odd_destination = odd.allocate_input().unwrap();
        let mut odd_workspace = odd.allocate_workspace().unwrap();
        odd.inverse(&odd_spectrum, &mut odd_destination, &mut odd_workspace)
            .unwrap();
    }
}

fn real_bits<R: TestReal>(values: &[R]) -> Vec<u64> {
    values
        .iter()
        .map(|value| R::to_f64(*value).to_bits())
        .collect()
}

fn complex_bits<R: TestReal>(values: &[Complex<R>]) -> Vec<(u64, u64)> {
    values
        .iter()
        .map(|value| (R::to_f64(value.re).to_bits(), R::to_f64(value.im).to_bits()))
        .collect()
}

fn dirty_real<R: TestReal, const N: usize, const M: usize>(array: &mut PencilArray<R, N, M>) {
    for (index, value) in array.as_mut_slice().iter_mut().enumerate() {
        *value = <R as TestReal>::from_f64(31.0 + index as f64 * 0.25);
    }
}

fn r2c_depth<const N: usize>(shape: [usize; N]) -> f64 {
    1.0 + shape[..N - 1]
        .iter()
        .map(|&length| {
            if length <= 1 {
                0.0
            } else {
                (usize::BITS - (length - 1).leading_zeros()) as f64
            }
        })
        .sum::<f64>()
}

fn r2c_partial_policy<const N: usize>(
    shape: [usize; N],
    selection: AxisSelection<N>,
) -> (f64, f64) {
    let reduction_axis = (0..N)
        .rev()
        .find(|&axis| selection.contains(axis))
        .expect("partial R2C policy needs a selected axis");
    let depth = 1.0
        + (0..reduction_axis)
            .filter(|&axis| selection.contains(axis) && shape[axis] > 1)
            .map(|axis| (usize::BITS - (shape[axis] - 1).leading_zeros()) as f64)
            .sum::<f64>();
    let transverse = (0..reduction_axis)
        .filter(|&axis| selection.contains(axis))
        .map(|axis| shape[axis] as f64)
        .product::<f64>();
    (depth, transverse)
}

fn r2c_partial_boundary_check<R: TestReal, const N: usize, const M: usize>(
    plan: &R2cPlan<R, N, M>,
    extra: &[usize],
    endpoint: [usize; N],
    real: f64,
    imaginary: f64,
    normalize: bool,
    accepted: bool,
) where
    Complex<R>: mpi::datatype::Equivalence,
{
    let mut source = plan.allocate_output().unwrap();
    set_r2c_global(
        &mut source,
        extra,
        endpoint,
        Complex::new(
            <R as TestReal>::from_f64(real),
            <R as TestReal>::from_f64(imaginary),
        ),
    );
    let source_before = complex_bits(source.as_slice());
    let mut destination = plan.allocate_input().unwrap();
    dirty_real(&mut destination);
    let destination_before = real_bits(destination.as_slice());
    let mut workspace = plan.allocate_workspace().unwrap();
    let result = if normalize {
        plan.inverse(&source, &mut destination, &mut workspace)
    } else {
        plan.backward(&source, &mut destination, &mut workspace)
    };
    if accepted {
        assert!(result.is_ok(), "partial endpoint result: {result:?}");
    } else {
        let max_imaginary = source
            .as_slice()
            .iter()
            .map(|value| R::to_f64(value.im).abs())
            .fold(0.0, f64::max);
        assert!(
            matches!(result, Err(R2cError::InvalidSpectrum)),
            "partial endpoint result normalize={normalize} real={real} imag={imaginary} max_imaginary={max_imaginary}: {result:?}"
        );
        assert_eq!(real_bits(destination.as_slice()), destination_before);
    }
    assert_eq!(complex_bits(source.as_slice()), source_before);
    reuse_r2c_after_boundary(plan, &mut workspace);
}

fn run_r2c_partial_boundary_policy<R: TestReal>(
    topology: &Arc<MpiTopology<1>>,
    method: TransposeMethod,
) where
    Complex<R>: mpi::datatype::Equivalence,
{
    let shape = [8, 9, 4, 3];
    let selected = AxisSelection::from_indices([0, 2]).unwrap();
    let (depth, transverse) = r2c_partial_policy(shape, selected);
    let inverse_absolute = 128.0 * R::min_subnormal() * depth;
    let raw_absolute = inverse_absolute * transverse;
    let relative = 128.0 * R::epsilon() * depth;
    let partial = R2cPlan::<R, 4, 1>::from_shape_with_selection_and_method(
        Arc::clone(topology),
        shape,
        ExtraShape::new([2, 3]).unwrap(),
        selected,
        method,
    )
    .unwrap();
    assert_eq!(partial.output_pencil().global_shape(), &[8, 9, 3, 3]);
    for normalize in [true, false] {
        // These are frequency-DC impulse amplitudes, not boundary-plane values.
        // The normalized tail divides them by T; the raw tail does not. Thus
        // both source thresholds are A*T despite different boundary thresholds.
        let source_absolute_threshold = if normalize {
            inverse_absolute * transverse
        } else {
            raw_absolute
        };
        r2c_partial_boundary_check(
            &partial,
            &[1, 2],
            [0, 0, 0, 0],
            0.0,
            0.5 * source_absolute_threshold,
            normalize,
            true,
        );
        r2c_partial_boundary_check(
            &partial,
            &[1, 2],
            [0, 0, 0, 0],
            0.0,
            2.0 * source_absolute_threshold,
            normalize,
            false,
        );
        r2c_partial_boundary_check(
            &partial,
            &[1, 2],
            [0, 0, 0, 0],
            1.0,
            0.5 * relative,
            normalize,
            true,
        );
        r2c_partial_boundary_check(
            &partial,
            &[1, 2],
            [0, 0, 0, 0],
            1.0,
            2.0 * relative,
            normalize,
            false,
        );
        r2c_partial_boundary_check(
            &partial,
            &[1, 2],
            [0, 0, 0, 0],
            f64::NAN,
            0.0,
            normalize,
            false,
        );
        r2c_partial_boundary_check(
            &partial,
            &[1, 2],
            [0, 0, 0, 0],
            0.0,
            f64::INFINITY,
            normalize,
            false,
        );
    }

    let head_only_selection = AxisSelection::from_indices([2]).unwrap();
    let (head_depth, head_transverse) = r2c_partial_policy(shape, head_only_selection);
    assert_eq!(head_depth, 1.0);
    assert_eq!(head_transverse, 1.0);
    let head_only = R2cPlan::<R, 4, 1>::from_shape_with_selection_and_method(
        Arc::clone(topology),
        shape,
        ExtraShape::new([2, 3]).unwrap(),
        head_only_selection,
        method,
    )
    .unwrap();
    for normalize in [true, false] {
        let absolute = 128.0 * R::min_subnormal();
        r2c_partial_boundary_check(
            &head_only,
            &[1, 2],
            [0, 0, 0, 0],
            0.0,
            0.5 * absolute,
            normalize,
            true,
        );
        r2c_partial_boundary_check(
            &head_only,
            &[1, 2],
            [0, 0, 0, 0],
            0.0,
            2.0 * absolute,
            normalize,
            false,
        );
    }
}

fn r2c_boundary_reject<R: TestReal, const N: usize, const M: usize, F>(
    plan: &R2cPlan<R, N, M>,
    mutate: F,
) where
    Complex<R>: mpi::datatype::Equivalence,
    F: FnOnce(&mut PencilArray<Complex<R>, N, M>),
{
    let mut source = plan.allocate_output().unwrap();
    mutate(&mut source);
    let source_before = complex_bits(source.as_slice());
    let mut destination = plan.allocate_input().unwrap();
    dirty_real(&mut destination);
    let destination_before = real_bits(destination.as_slice());
    let mut workspace = plan.allocate_workspace().unwrap();
    let result = plan.inverse(&source, &mut destination, &mut workspace);
    assert!(matches!(result, Err(R2cError::InvalidSpectrum)));
    assert_eq!(complex_bits(source.as_slice()), source_before);
    assert_eq!(real_bits(destination.as_slice()), destination_before);
    // The post-tail boundary check is allowed to use the workspace before it
    // rejects, so only the caller-owned source and destination are compared.
    reuse_r2c_after_boundary(plan, &mut workspace);
}

fn r2c_boundary_accept<R: TestReal, const N: usize, const M: usize, F>(
    plan: &R2cPlan<R, N, M>,
    mutate: F,
) where
    Complex<R>: mpi::datatype::Equivalence,
    F: FnOnce(&mut PencilArray<Complex<R>, N, M>),
{
    let mut source = plan.allocate_output().unwrap();
    mutate(&mut source);
    let source_before = complex_bits(source.as_slice());
    let mut destination = plan.allocate_input().unwrap();
    dirty_real(&mut destination);
    let mut workspace = plan.allocate_workspace().unwrap();
    plan.inverse(&source, &mut destination, &mut workspace)
        .unwrap();
    assert_eq!(complex_bits(source.as_slice()), source_before);
    reuse_r2c_after_boundary(plan, &mut workspace);
}

fn r2c_raw_boundary_check(
    plan: &R2cPlan<f32, 2, 1>,
    extra: &[usize],
    endpoint: usize,
    real: f32,
    imaginary: f32,
    accepted: bool,
) {
    let mut source = plan.allocate_output().unwrap();
    set_r2c_global(
        &mut source,
        extra,
        [0, endpoint],
        Complex::new(real, imaginary),
    );
    let source_before = complex_bits(source.as_slice());
    let mut destination = plan.allocate_input().unwrap();
    dirty_real(&mut destination);
    let destination_before = real_bits(destination.as_slice());
    let mut workspace = plan.allocate_workspace().unwrap();
    let result = plan.backward(&source, &mut destination, &mut workspace);
    if accepted {
        assert!(result.is_ok(), "raw endpoint result: {result:?}");
    } else {
        assert!(
            matches!(result, Err(R2cError::InvalidSpectrum)),
            "raw endpoint result: {result:?}"
        );
    }
    assert_eq!(complex_bits(source.as_slice()), source_before);
    if !accepted {
        assert_eq!(real_bits(destination.as_slice()), destination_before);
    }
}

fn run_r2c_raw_absolute_boundary(topology: &Arc<MpiTopology<1>>, method: TransposeMethod) {
    let depth = r2c_depth([3, 2]);
    let raw_absolute = 128.0 * f32::from_bits(1) as f64 * depth * 3.0;
    let just_below = (0.5 * raw_absolute) as f32;
    let just_above = (2.0 * raw_absolute) as f32;
    let relative = (0.5 * 128.0 * f32::EPSILON as f64 * depth) as f32;
    let relative_above = (2.0 * 128.0 * f32::EPSILON as f64 * depth) as f32;

    let scalar_n2 = R2cPlan::<f32, 2, 1>::from_shape_with_method(
        Arc::clone(topology),
        [3, 2],
        ExtraShape::scalar(),
        method,
    )
    .unwrap();
    r2c_raw_boundary_check(&scalar_n2, &[], 0, 0.0, just_below, true);
    r2c_raw_boundary_check(&scalar_n2, &[], 0, 0.0, just_above, false);
    r2c_raw_boundary_check(&scalar_n2, &[], 0, 1.0, relative, true);
    r2c_raw_boundary_check(&scalar_n2, &[], 0, 1.0, relative_above, false);

    let scalar_n4 = R2cPlan::<f32, 2, 1>::from_shape_with_method(
        Arc::clone(topology),
        [3, 4],
        ExtraShape::scalar(),
        method,
    )
    .unwrap();
    r2c_raw_boundary_check(&scalar_n4, &[], 0, 0.0, just_below, true);
    r2c_raw_boundary_check(&scalar_n4, &[], 0, 0.0, just_above, false);

    let batched = R2cPlan::<f32, 2, 1>::from_shape_with_method(
        Arc::clone(topology),
        [3, 2],
        ExtraShape::new([2]).unwrap(),
        method,
    )
    .unwrap();
    r2c_raw_boundary_check(&batched, &[0], 0, 0.0, just_below, true);
    r2c_raw_boundary_check(&batched, &[0], 0, 0.0, just_above, false);

    for extra in [0, 1] {
        let extra = [extra];
        for endpoint in [0, 1] {
            for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                r2c_raw_boundary_check(&batched, &extra, endpoint, bad, 0.0, false);
                r2c_raw_boundary_check(&batched, &extra, endpoint, 0.0, bad, false);
            }
        }
    }
}

fn r2c_interior_nonfinite<R: TestReal, const M: usize>(plan: &R2cPlan<R, 2, M>, value: R)
where
    Complex<R>: mpi::datatype::Equivalence,
{
    let mut source = plan.allocate_output().unwrap();
    set_r2c_global(
        &mut source,
        &[],
        [0, 1],
        Complex::new(value, <R as TestReal>::from_f64(0.0)),
    );
    let source_before = complex_bits(source.as_slice());
    let mut destination = plan.allocate_input().unwrap();
    dirty_real(&mut destination);
    let mut workspace = plan.allocate_workspace().unwrap();
    assert!(
        plan.inverse(&source, &mut destination, &mut workspace)
            .is_ok()
    );
    assert_eq!(complex_bits(source.as_slice()), source_before);
    reuse_r2c_after_boundary(plan, &mut workspace);
}

fn reuse_r2c_after_boundary<R: TestReal, const N: usize, const M: usize>(
    plan: &R2cPlan<R, N, M>,
    workspace: &mut R2cWorkspace<R, N, M>,
) where
    Complex<R>: mpi::datatype::Equivalence,
{
    let mut source = plan.allocate_input().unwrap();
    fill_r2c_input(&mut source, 71.0);
    let mut spectrum = plan.allocate_output().unwrap();
    plan.forward(&source, &mut spectrum, workspace).unwrap();
    let mut recovered = plan.allocate_input().unwrap();
    plan.inverse(&spectrum, &mut recovered, workspace).unwrap();
}

fn run_r2c_boundary_precision<R: TestReal>(topology: &Arc<MpiTopology<1>>, method: TransposeMethod)
where
    Complex<R>: mpi::datatype::Equivalence,
{
    let small = R2cPlan::<R, 2, 1>::from_shape_with_method(
        Arc::clone(topology),
        [1, 2],
        ExtraShape::scalar(),
        method,
    )
    .unwrap();
    let relative = 128.0 * R::epsilon() * r2c_depth([1, 2]);
    let absolute = 128.0 * R::min_subnormal() * r2c_depth([1, 2]);
    r2c_boundary_accept(&small, |spectrum| {
        set_r2c_global(
            spectrum,
            &[],
            [0, 0],
            Complex::new(
                <R as TestReal>::from_f64(1.0),
                <R as TestReal>::from_f64(0.5 * relative),
            ),
        );
    });
    r2c_boundary_reject(&small, |spectrum| {
        set_r2c_global(
            spectrum,
            &[],
            [0, 0],
            Complex::new(
                <R as TestReal>::from_f64(1.0),
                <R as TestReal>::from_f64(2.0 * relative),
            ),
        );
    });
    r2c_boundary_accept(&small, |spectrum| {
        set_r2c_global(
            spectrum,
            &[],
            [0, 0],
            Complex::new(
                <R as TestReal>::from_f64(0.0),
                <R as TestReal>::from_f64(0.5 * absolute),
            ),
        );
    });
    r2c_boundary_reject(&small, |spectrum| {
        set_r2c_global(
            spectrum,
            &[],
            [0, 0],
            Complex::new(
                <R as TestReal>::from_f64(0.0),
                <R as TestReal>::from_f64(2.0 * absolute),
            ),
        );
    });

    for re in [0.0, -0.0] {
        for im in [0.0, -0.0] {
            r2c_boundary_accept(&small, |spectrum| {
                set_r2c_global(
                    spectrum,
                    &[],
                    [0, 0],
                    Complex::new(<R as TestReal>::from_f64(re), <R as TestReal>::from_f64(im)),
                );
            });
        }
    }
    for (re, im) in [(R::huge(), 0.0), (1e-20, 0.0), (1e4, 0.0)] {
        r2c_boundary_accept(&small, |spectrum| {
            set_r2c_global(
                spectrum,
                &[],
                [0, 0],
                Complex::new(<R as TestReal>::from_f64(re), <R as TestReal>::from_f64(im)),
            );
        });
    }

    let cancellation = R2cPlan::<R, 2, 1>::from_shape_with_method(
        Arc::clone(topology),
        [3, 2],
        ExtraShape::scalar(),
        method,
    )
    .unwrap();
    r2c_boundary_accept(&cancellation, |spectrum| {
        set_r2c_global(
            spectrum,
            &[],
            [0, 0],
            Complex::new(
                <R as TestReal>::from_f64(3.0),
                <R as TestReal>::from_f64(0.0),
            ),
        );
        set_r2c_global(
            spectrum,
            &[],
            [1, 0],
            Complex::new(
                <R as TestReal>::from_f64(2.0),
                <R as TestReal>::from_f64(3.0),
            ),
        );
        set_r2c_global(
            spectrum,
            &[],
            [2, 0],
            Complex::new(
                <R as TestReal>::from_f64(2.0),
                <R as TestReal>::from_f64(-3.0),
            ),
        );
    });

    let endpoints = R2cPlan::<R, 2, 1>::from_shape_with_method(
        Arc::clone(topology),
        [1, 4],
        ExtraShape::scalar(),
        method,
    )
    .unwrap();
    for endpoint in [0, 2] {
        for component in 0..2 {
            for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                r2c_boundary_reject(&endpoints, |spectrum| {
                    let value = if component == 0 {
                        Complex::new(
                            <R as TestReal>::from_f64(bad),
                            <R as TestReal>::from_f64(0.0),
                        )
                    } else {
                        Complex::new(
                            <R as TestReal>::from_f64(0.0),
                            <R as TestReal>::from_f64(bad),
                        )
                    };
                    set_r2c_global(spectrum, &[], [0, endpoint], value);
                });
            }
        }
    }
    r2c_interior_nonfinite(&endpoints, <R as TestReal>::from_f64(f64::NAN));
    r2c_interior_nonfinite(&endpoints, <R as TestReal>::from_f64(f64::INFINITY));
    let odd_end = R2cPlan::<R, 2, 1>::from_shape_with_method(
        Arc::clone(topology),
        [1, 3],
        ExtraShape::scalar(),
        method,
    )
    .unwrap();
    r2c_interior_nonfinite(&odd_end, <R as TestReal>::from_f64(f64::NAN));

    run_r2c_boundary_isolation(topology, method);
}

fn run_r2c_boundary_isolation<R: TestReal>(topology: &Arc<MpiTopology<1>>, method: TransposeMethod)
where
    Complex<R>: mpi::datatype::Equivalence,
{
    let batched = R2cPlan::<R, 2, 1>::from_shape_with_method(
        Arc::clone(topology),
        [1, 4],
        ExtraShape::new([2]).unwrap(),
        method,
    )
    .unwrap();
    r2c_boundary_reject(&batched, |spectrum| {
        set_r2c_global(
            spectrum,
            &[0],
            [0, 0],
            Complex::new(
                <R as TestReal>::from_f64(R::huge()),
                <R as TestReal>::from_f64(0.0),
            ),
        );
        set_r2c_global(
            spectrum,
            &[1],
            [0, 0],
            Complex::new(
                <R as TestReal>::from_f64(0.0),
                <R as TestReal>::from_f64(1.0),
            ),
        );
    });

    let interior = R2cPlan::<R, 2, 1>::from_shape_with_method(
        Arc::clone(topology),
        [1, 4],
        ExtraShape::scalar(),
        method,
    )
    .unwrap();
    r2c_boundary_reject(&interior, |spectrum| {
        set_r2c_global(
            spectrum,
            &[],
            [0, 0],
            Complex::new(
                <R as TestReal>::from_f64(0.0),
                <R as TestReal>::from_f64(1.0),
            ),
        );
        set_r2c_global(
            spectrum,
            &[],
            [0, 1],
            Complex::new(
                <R as TestReal>::from_f64(R::huge()),
                <R as TestReal>::from_f64(0.0),
            ),
        );
    });

    r2c_boundary_reject(&interior, |spectrum| {
        set_r2c_global(
            spectrum,
            &[],
            [0, 0],
            Complex::new(
                <R as TestReal>::from_f64(0.0),
                <R as TestReal>::from_f64(1.0),
            ),
        );
        set_r2c_global(
            spectrum,
            &[],
            [0, 2],
            Complex::new(
                <R as TestReal>::from_f64(R::huge()),
                <R as TestReal>::from_f64(0.0),
            ),
        );
    });
    r2c_boundary_reject(&interior, |spectrum| {
        set_r2c_global(
            spectrum,
            &[],
            [0, 0],
            Complex::new(
                <R as TestReal>::from_f64(R::huge()),
                <R as TestReal>::from_f64(0.0),
            ),
        );
        set_r2c_global(
            spectrum,
            &[],
            [0, 2],
            Complex::new(
                <R as TestReal>::from_f64(0.0),
                <R as TestReal>::from_f64(1.0),
            ),
        );
    });
}

fn run_r2c_analytic_f32_boundary(topology: &Arc<MpiTopology<1>>, method: TransposeMethod) {
    let plan = R2cPlan::<f32, 3, 1>::from_shape_with_method(
        Arc::clone(topology),
        [128, 128, 3],
        ExtraShape::scalar(),
        method,
    )
    .unwrap();
    r2c_boundary_reject(&plan, |spectrum| {
        set_r2c_global(spectrum, &[], [0, 0, 0], Complex::new(0.0, 1.0));
    });
}

fn run_r2c_boundary_hard_cases(
    topology_1d: &Arc<MpiTopology<1>>,
    topology_2d: &Arc<MpiTopology<2>>,
) {
    for method in [TransposeMethod::AllToAllv, TransposeMethod::PointToPoint] {
        run_r2c_boundary_precision::<f32>(topology_1d, method);
        run_r2c_boundary_precision::<f64>(topology_1d, method);
        run_r2c_raw_absolute_boundary(topology_1d, method);
        run_r2c_analytic_f32_boundary(topology_1d, method);
        run_r2c_partial_boundary_policy::<f32>(topology_1d, method);
        run_r2c_partial_boundary_policy::<f64>(topology_1d, method);
    }
    if topology_2d.communicator().size() == 6 {
        run_r2c_six_rank_endpoint_failure(topology_2d);
    }
}

fn run_r2c_six_rank_endpoint_failure(topology: &Arc<MpiTopology<2>>) {
    for method in [TransposeMethod::AllToAllv, TransposeMethod::PointToPoint] {
        let plan = R2cPlan::<f64, 4, 2>::from_shape_with_method(
            Arc::clone(topology),
            [2, 1, 2, 4],
            ExtraShape::scalar(),
            method,
        )
        .unwrap();
        let mut source = plan.allocate_output().unwrap();
        for k0 in 0..2 {
            for k1 in 0..1 {
                for k2 in 0..2 {
                    set_r2c_global(
                        &mut source,
                        &[],
                        [k0, k1, k2, 0],
                        Complex::new(0.0, if k0 == 0 { 1.0 } else { -1.0 }),
                    );
                }
            }
        }
        let source_before = complex_bits(source.as_slice());
        let mut destination = plan.allocate_input().unwrap();
        dirty_real(&mut destination);
        let destination_before = real_bits(destination.as_slice());
        let mut workspace = plan.allocate_workspace().unwrap();
        let result = plan.inverse(&source, &mut destination, &mut workspace);
        assert!(matches!(result, Err(R2cError::InvalidSpectrum)));
        assert_eq!(complex_bits(source.as_slice()), source_before);
        assert_eq!(real_bits(destination.as_slice()), destination_before);
        reuse_r2c_after_boundary(&plan, &mut workspace);
    }
}

fn fill_input<R: TestReal, const N: usize, const M: usize>(
    array: &mut PencilArray<Complex<R>, N, M>,
    seed: f64,
) {
    let extra_dimensions = array.extra_shape().dimensions().to_vec();
    let extra_count = array.extra_shape().element_count();
    let local_shape = array.pencil().local_shape_logical();
    for extra_linear in 0..extra_count {
        let extra_indices = unravel_extra(extra_linear, &extra_dimensions);
        for spatial_linear in 0..array.pencil().local_len() {
            let spatial = unravel_spatial(spatial_linear, local_shape);
            let global: [usize; N] = std::array::from_fn(|axis| {
                array.pencil().local_ranges()[axis].start + spatial[axis]
            });
            *array
                .get_local_mut(&extra_indices, spatial)
                .expect("filled input index is local") = input_value(&extra_indices, global, seed);
        }
    }
}

fn fill_in_place<R: TestReal, const N: usize, const M: usize>(
    array: &mut C2cInPlaceArray<R, N, M>,
    seed: f64,
    spectrum: bool,
) where
    Complex<R>: mpi::datatype::Equivalence,
{
    let mut view = array.view_mut().unwrap();
    let extra_dimensions = view.extra_shape().dimensions().to_vec();
    let extra_count = view.extra_shape().element_count();
    let local_shape = view.pencil().local_shape_logical();
    let starts = std::array::from_fn::<_, N, _>(|axis| view.pencil().local_ranges()[axis].start);
    for extra_linear in 0..extra_count {
        let extra_indices = unravel_extra(extra_linear, &extra_dimensions);
        for spatial_linear in 0..view.pencil().local_len() {
            let spatial = unravel_spatial(spatial_linear, local_shape);
            let global: [usize; N] = std::array::from_fn(|axis| starts[axis] + spatial[axis]);
            let value = if spectrum {
                spectrum_value(&extra_indices, global, seed)
            } else {
                input_value(&extra_indices, global, seed)
            };
            *view
                .get_local_mut(&extra_indices, spatial)
                .expect("filled in-place index is local") = value;
        }
    }
}

fn fill_spectrum<R: TestReal, const N: usize, const M: usize>(
    array: &mut PencilArray<Complex<R>, N, M>,
    seed: f64,
) {
    let extra_dimensions = array.extra_shape().dimensions().to_vec();
    let extra_count = array.extra_shape().element_count();
    let local_shape = array.pencil().local_shape_logical();
    for extra_linear in 0..extra_count {
        let extra_indices = unravel_extra(extra_linear, &extra_dimensions);
        for spatial_linear in 0..array.pencil().local_len() {
            let spatial = unravel_spatial(spatial_linear, local_shape);
            let global: [usize; N] = std::array::from_fn(|axis| {
                array.pencil().local_ranges()[axis].start + spatial[axis]
            });
            *array
                .get_local_mut(&extra_indices, spatial)
                .expect("filled spectrum index is local") =
                spectrum_value(&extra_indices, global, seed);
        }
    }
}

fn check_forward<R: TestReal, const N: usize, const M: usize>(
    array: &PencilArrayView<'_, Complex<R>, N, M>,
    extra_shape: &ExtraShape,
    global_shape: [usize; N],
    seed: f64,
) {
    let local_shape = array.pencil().local_shape_logical();
    let ranges = array.pencil().local_ranges();
    for extra_linear in 0..extra_shape.element_count() {
        let extra_indices = unravel_extra(extra_linear, extra_shape.dimensions());
        for spatial_linear in 0..array.pencil().local_len() {
            let local = unravel_spatial(spatial_linear, local_shape);
            let wave_number = std::array::from_fn(|axis| ranges[axis].start + local[axis]);
            let expected = dft_value::<R, N>(
                &extra_indices,
                wave_number,
                global_shape,
                seed,
                false,
                false,
            );
            let actual = *array
                .get_local(&extra_indices, local)
                .expect("checked forward index is local");
            assert_close(actual, expected);
        }
    }
}

fn check_inverse<R: TestReal, const N: usize, const M: usize>(
    array: &PencilArrayView<'_, Complex<R>, N, M>,
    extra_shape: &ExtraShape,
    global_shape: [usize; N],
    seed: f64,
    normalize_inverse: bool,
) {
    let local_shape = array.pencil().local_shape_logical();
    let ranges = array.pencil().local_ranges();
    for extra_linear in 0..extra_shape.element_count() {
        let extra_indices = unravel_extra(extra_linear, extra_shape.dimensions());
        for spatial_linear in 0..array.pencil().local_len() {
            let local = unravel_spatial(spatial_linear, local_shape);
            let spatial = std::array::from_fn(|axis| ranges[axis].start + local[axis]);
            let expected = dft_value::<R, N>(
                &extra_indices,
                spatial,
                global_shape,
                seed,
                true,
                normalize_inverse,
            );
            let actual = *array
                .get_local(&extra_indices, local)
                .expect("checked inverse index is local");
            assert_close(actual, expected);
        }
    }
}

fn dft_value<R: TestReal, const N: usize>(
    extra_indices: &[usize],
    target: [usize; N],
    global_shape: [usize; N],
    seed: f64,
    inverse: bool,
    normalize_inverse: bool,
) -> Complex<R> {
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut sum = Complex::new(0.0, 0.0);
    let total = global_shape.iter().product::<usize>();
    for linear in 0..total {
        let spatial = unravel_spatial(linear, global_shape);
        let value = if inverse {
            spectrum_value(extra_indices, spatial, seed)
        } else {
            input_value(extra_indices, spatial, seed)
        };
        let phase = sign
            * TAU
            * (0..N)
                .map(|axis| spatial[axis] as f64 * target[axis] as f64 / global_shape[axis] as f64)
                .sum::<f64>();
        let (sin, cos) = phase.sin_cos();
        sum += Complex::new(
            R::to_f64(value.re) * cos - R::to_f64(value.im) * sin,
            R::to_f64(value.re) * sin + R::to_f64(value.im) * cos,
        );
    }
    if inverse && normalize_inverse {
        let scale = 1.0 / total as f64;
        sum.re *= scale;
        sum.im *= scale;
    }
    Complex::new(
        <R as TestReal>::from_f64(sum.re),
        <R as TestReal>::from_f64(sum.im),
    )
}

fn input_value<R: TestReal, const N: usize>(
    extra_indices: &[usize],
    spatial: [usize; N],
    seed: f64,
) -> Complex<R> {
    let mut real = 0.13 + seed * 0.017;
    let mut imaginary = -0.27 - seed * 0.011;
    for (index, &value) in extra_indices.iter().enumerate() {
        real += (index + 2) as f64 * (value + 1) as f64 * 0.071;
        imaginary -= (index + 3) as f64 * (value + 1) as f64 * 0.043;
    }
    for (axis, value) in spatial.into_iter().enumerate() {
        real += (axis + 1) as f64 * (value + 1) as f64 * 0.19
            + ((value + 1 + axis) as f64).sin() * 0.013;
        imaginary += (axis + 2) as f64 * (value + 1) as f64 * 0.11
            + ((value + 2 + axis) as f64).cos() * 0.017;
    }
    Complex::new(
        <R as TestReal>::from_f64(real),
        <R as TestReal>::from_f64(imaginary),
    )
}

fn spectrum_value<R: TestReal, const N: usize>(
    extra_indices: &[usize],
    wave_number: [usize; N],
    seed: f64,
) -> Complex<R> {
    let mut real = 0.41 + seed * 0.009;
    let mut imaginary = -0.18 - seed * 0.007;
    for (index, &value) in extra_indices.iter().enumerate() {
        real += (index + 1) as f64 * (value + 2) as f64 * 0.037;
        imaginary += (index + 4) as f64 * (value + 1) as f64 * 0.029;
    }
    for (axis, value) in wave_number.into_iter().enumerate() {
        real += (axis + 2) as f64 * (value + 1) as f64 * 0.083;
        imaginary -= (axis + 1) as f64 * (value + 2) as f64 * 0.047;
    }
    Complex::new(
        <R as TestReal>::from_f64(real),
        <R as TestReal>::from_f64(imaginary),
    )
}

fn assert_close<R: TestReal>(actual: Complex<R>, expected: Complex<R>) {
    let bound = R::tolerance()
        * (1.0
            + R::to_f64(expected.re)
                .abs()
                .max(R::to_f64(expected.im).abs()));
    assert!((R::to_f64(actual.re) - R::to_f64(expected.re)).abs() <= bound);
    assert!((R::to_f64(actual.im) - R::to_f64(expected.im)).abs() <= bound);
}

fn unravel_extra(linear: usize, dimensions: &[usize]) -> Vec<usize> {
    let mut result = vec![0; dimensions.len()];
    let mut remainder = linear;
    for axis in (0..dimensions.len()).rev() {
        result[axis] = remainder % dimensions[axis];
        remainder /= dimensions[axis];
    }
    result
}

fn unravel_spatial<const N: usize>(linear: usize, shape: [usize; N]) -> [usize; N] {
    let mut result = [0; N];
    let mut remainder = linear;
    for axis in (0..N).rev() {
        result[axis] = remainder % shape[axis];
        remainder /= shape[axis];
    }
    result
}

fn topology_barrier<const M: usize>(topology: &Arc<MpiTopology<M>>) {
    topology.communicator().barrier();
}

fn assert_c2c_rejected<R, const N: usize, const M: usize, F>(
    source: &PencilArray<Complex<R>, N, M>,
    destination: &mut PencilArray<Complex<R>, N, M>,
    workspace: &mut C2cOutOfPlaceWorkspace<R, N, M>,
    execute: F,
) where
    R: TestReal + std::fmt::Debug,
    Complex<R>: mpi::datatype::Equivalence,
    F: FnOnce(
        &PencilArray<Complex<R>, N, M>,
        &mut PencilArray<Complex<R>, N, M>,
        &mut C2cOutOfPlaceWorkspace<R, N, M>,
    ) -> Result<(), FftError>,
{
    let source_before = source.as_slice().to_vec();
    let destination_before = destination.as_slice().to_vec();
    let workspace_before = format!("{workspace:?}");
    assert!(execute(source, destination, workspace).is_err());
    assert_eq!(source.as_slice(), source_before.as_slice());
    assert_eq!(destination.as_slice(), destination_before.as_slice());
    assert_eq!(format!("{workspace:?}"), workspace_before);
}

fn assert_c2c_rejected_exact<R, const N: usize, const M: usize, F, P>(
    source: &PencilArray<Complex<R>, N, M>,
    destination: &mut PencilArray<Complex<R>, N, M>,
    workspace: &mut C2cOutOfPlaceWorkspace<R, N, M>,
    execute: F,
    predicate: P,
) where
    R: TestReal + std::fmt::Debug,
    Complex<R>: mpi::datatype::Equivalence,
    F: FnOnce(
        &PencilArray<Complex<R>, N, M>,
        &mut PencilArray<Complex<R>, N, M>,
        &mut C2cOutOfPlaceWorkspace<R, N, M>,
    ) -> Result<(), FftError>,
    P: FnOnce(&FftError) -> bool,
{
    let source_before = source.as_slice().to_vec();
    let destination_before = destination.as_slice().to_vec();
    let workspace_before = format!("{workspace:?}");
    let error =
        execute(source, destination, workspace).expect_err("C2C call unexpectedly succeeded");
    assert!(predicate(&error), "unexpected C2C error: {error:?}");
    assert_eq!(source.as_slice(), source_before.as_slice());
    assert_eq!(destination.as_slice(), destination_before.as_slice());
    assert_eq!(format!("{workspace:?}"), workspace_before);
}

fn assert_in_place_rejected<R, const N: usize, const M: usize, F>(
    array: &mut C2cInPlaceArray<R, N, M>,
    workspace: &mut C2cInPlaceWorkspace<R, N, M>,
    execute: F,
) where
    R: TestReal + std::fmt::Debug,
    Complex<R>: mpi::datatype::Equivalence,
    F: FnOnce(
        &mut C2cInPlaceArray<R, N, M>,
        &mut C2cInPlaceWorkspace<R, N, M>,
    ) -> Result<(), FftError>,
{
    let state_before = array.state();
    let array_before = format!("{array:?}");
    let workspace_before = format!("{workspace:?}");
    assert!(execute(array, workspace).is_err());
    assert_eq!(array.state(), state_before);
    assert_eq!(format!("{array:?}"), array_before);
    assert_eq!(format!("{workspace:?}"), workspace_before);
}

fn fresh_in_place<R: TestReal, const N: usize, const M: usize>(
    plan: &C2cPlan<R, N, M>,
    seed: f64,
) -> (C2cInPlaceArray<R, N, M>, C2cInPlaceWorkspace<R, N, M>)
where
    Complex<R>: mpi::datatype::Equivalence,
{
    let mut array = plan.allocate_in_place().unwrap();
    fill_in_place(&mut array, seed, false);
    let workspace = plan.allocate_in_place_workspace().unwrap();
    (array, workspace)
}

fn reuse_forward<R: TestReal, const N: usize, const M: usize>(
    plan: &C2cPlan<R, N, M>,
    source: &PencilArray<Complex<R>, N, M>,
    workspace: &mut C2cOutOfPlaceWorkspace<R, N, M>,
) where
    Complex<R>: mpi::datatype::Equivalence,
{
    let mut destination = plan.allocate_output().unwrap();
    plan.forward(source, &mut destination, workspace).unwrap();
}

fn reuse_backward<R: TestReal, const N: usize, const M: usize>(
    plan: &C2cPlan<R, N, M>,
    source: &PencilArray<Complex<R>, N, M>,
    workspace: &mut C2cOutOfPlaceWorkspace<R, N, M>,
) where
    Complex<R>: mpi::datatype::Equivalence,
{
    let mut destination = plan.allocate_input().unwrap();
    plan.backward(source, &mut destination, workspace).unwrap();
}

fn assert_mixed_direction_oop(
    plan: &C2cPlan<f64, 2, 1>,
    world: &mpi::topology::SimpleCommunicator,
    root_operation: C2cOperation,
    peer_operation: C2cOperation,
    seed: f64,
) {
    let rank = world.rank();
    let mut input = plan.allocate_input().unwrap();
    fill_input(&mut input, seed);
    let mut output = plan.allocate_output().unwrap();
    fill_spectrum(&mut output, seed + 1.0);
    let mut input_destination = plan.allocate_input().unwrap();
    let mut output_destination = plan.allocate_output().unwrap();
    let mut workspace = plan.allocate_out_of_place_workspace().unwrap();
    let input_before = input.as_slice().to_vec();
    let output_before = output.as_slice().to_vec();
    let input_destination_before = input_destination.as_slice().to_vec();
    let output_destination_before = output_destination.as_slice().to_vec();
    let workspace_before = format!("{workspace:?}");
    let operation = if rank == 0 {
        root_operation
    } else {
        peer_operation
    };
    let result = match operation {
        C2cOperation::Forward => plan.forward(&input, &mut output_destination, &mut workspace),
        C2cOperation::Inverse => plan.inverse(&output, &mut input_destination, &mut workspace),
        C2cOperation::Backward => plan.backward(&output, &mut input_destination, &mut workspace),
    };
    assert!(matches!(
        result,
        Err(FftError::CollectiveDescriptorMismatch)
    ));
    assert_eq!(input.as_slice(), input_before.as_slice());
    assert_eq!(output.as_slice(), output_before.as_slice());
    assert_eq!(
        input_destination.as_slice(),
        input_destination_before.as_slice()
    );
    assert_eq!(
        output_destination.as_slice(),
        output_destination_before.as_slice()
    );
    assert_eq!(format!("{workspace:?}"), workspace_before);

    plan.forward(&input, &mut output_destination, &mut workspace)
        .unwrap();
    world.barrier();
}

fn assert_mixed_direction_in_place(
    plan: &C2cPlan<f64, 2, 1>,
    world: &mpi::topology::SimpleCommunicator,
    root_operation: C2cOperation,
    peer_operation: C2cOperation,
    seed: f64,
) {
    let rank = world.rank();
    let mut array = plan.allocate_in_place().unwrap();
    fill_in_place(&mut array, seed, false);
    let mut workspace = plan.allocate_in_place_workspace().unwrap();
    plan.forward_in_place(&mut array, &mut workspace).unwrap();
    let operation = if rank == 0 {
        root_operation
    } else {
        peer_operation
    };
    if matches!(operation, C2cOperation::Forward) {
        array = plan.allocate_in_place().unwrap();
        fill_in_place(&mut array, seed, false);
    }
    let state_before = array.state();
    let array_before = format!("{array:?}");
    let workspace_before = format!("{workspace:?}");
    let result = match operation {
        C2cOperation::Forward => plan.forward_in_place(&mut array, &mut workspace),
        C2cOperation::Inverse => plan.inverse_in_place(&mut array, &mut workspace),
        C2cOperation::Backward => plan.backward_in_place(&mut array, &mut workspace),
    };
    assert!(matches!(
        result,
        Err(FftError::CollectiveDescriptorMismatch)
    ));
    assert_eq!(array.state(), state_before);
    assert_eq!(format!("{array:?}"), array_before);
    assert_eq!(format!("{workspace:?}"), workspace_before);

    array = plan.allocate_in_place().unwrap();
    fill_in_place(&mut array, seed, false);
    plan.forward_in_place(&mut array, &mut workspace).unwrap();
    plan.backward_in_place(&mut array, &mut workspace).unwrap();
    world.barrier();
}

fn assert_mixed_transport_oop<R: TestReal + std::fmt::Debug, const N: usize, const M: usize>(
    use_point_to_point: bool,
    operation: C2cOperation,
    alltoallv: OopTransport<'_, R, N, M>,
    point_to_point: OopTransport<'_, R, N, M>,
) where
    Complex<R>: mpi::datatype::Equivalence,
{
    let OopTransport {
        plan: alltoallv_plan,
        source: alltoallv_source,
        destination: alltoallv_destination,
        workspace: alltoallv_workspace,
    } = alltoallv;
    let OopTransport {
        plan: point_to_point_plan,
        source: point_to_point_source,
        destination: point_to_point_destination,
        workspace: point_to_point_workspace,
    } = point_to_point;
    let alltoallv_source_before = alltoallv_source.as_slice().to_vec();
    let alltoallv_destination_before = alltoallv_destination.as_slice().to_vec();
    let alltoallv_workspace_before = format!("{alltoallv_workspace:?}");
    let point_to_point_source_before = point_to_point_source.as_slice().to_vec();
    let point_to_point_destination_before = point_to_point_destination.as_slice().to_vec();
    let point_to_point_workspace_before = format!("{point_to_point_workspace:?}");

    let result = match (use_point_to_point, operation) {
        (true, C2cOperation::Forward) => point_to_point_plan.forward(
            point_to_point_source,
            point_to_point_destination,
            point_to_point_workspace,
        ),
        (true, C2cOperation::Inverse) => point_to_point_plan.inverse(
            point_to_point_source,
            point_to_point_destination,
            point_to_point_workspace,
        ),
        (true, C2cOperation::Backward) => point_to_point_plan.backward(
            point_to_point_source,
            point_to_point_destination,
            point_to_point_workspace,
        ),
        (false, C2cOperation::Forward) => {
            alltoallv_plan.forward(alltoallv_source, alltoallv_destination, alltoallv_workspace)
        }
        (false, C2cOperation::Inverse) => {
            alltoallv_plan.inverse(alltoallv_source, alltoallv_destination, alltoallv_workspace)
        }
        (false, C2cOperation::Backward) => {
            alltoallv_plan.backward(alltoallv_source, alltoallv_destination, alltoallv_workspace)
        }
    };
    assert!(matches!(
        result,
        Err(FftError::CollectiveDescriptorMismatch)
    ));
    assert_eq!(
        alltoallv_source.as_slice(),
        alltoallv_source_before.as_slice()
    );
    assert_eq!(
        alltoallv_destination.as_slice(),
        alltoallv_destination_before.as_slice()
    );
    assert_eq!(
        format!("{alltoallv_workspace:?}"),
        alltoallv_workspace_before
    );
    assert_eq!(
        point_to_point_source.as_slice(),
        point_to_point_source_before.as_slice()
    );
    assert_eq!(
        point_to_point_destination.as_slice(),
        point_to_point_destination_before.as_slice()
    );
    assert_eq!(
        format!("{point_to_point_workspace:?}"),
        point_to_point_workspace_before
    );
}

fn assert_mixed_transport_in_place<R: TestReal + std::fmt::Debug, const N: usize, const M: usize>(
    use_point_to_point: bool,
    operation: C2cOperation,
    alltoallv: InPlaceTransport<'_, R, N, M>,
    point_to_point: InPlaceTransport<'_, R, N, M>,
) where
    Complex<R>: mpi::datatype::Equivalence,
{
    let InPlaceTransport {
        plan: alltoallv_plan,
        array: alltoallv_array,
        workspace: alltoallv_workspace,
    } = alltoallv;
    let InPlaceTransport {
        plan: point_to_point_plan,
        array: point_to_point_array,
        workspace: point_to_point_workspace,
    } = point_to_point;
    let alltoallv_state = alltoallv_array.state();
    let alltoallv_data = alltoallv_array.view().unwrap().as_slice().to_vec();
    let alltoallv_workspace_before = format!("{alltoallv_workspace:?}");
    let point_to_point_state = point_to_point_array.state();
    let point_to_point_data = point_to_point_array.view().unwrap().as_slice().to_vec();
    let point_to_point_workspace_before = format!("{point_to_point_workspace:?}");

    let result = match (use_point_to_point, operation) {
        (true, C2cOperation::Forward) => {
            point_to_point_plan.forward_in_place(point_to_point_array, point_to_point_workspace)
        }
        (true, C2cOperation::Inverse) => {
            point_to_point_plan.inverse_in_place(point_to_point_array, point_to_point_workspace)
        }
        (true, C2cOperation::Backward) => {
            point_to_point_plan.backward_in_place(point_to_point_array, point_to_point_workspace)
        }
        (false, C2cOperation::Forward) => {
            alltoallv_plan.forward_in_place(alltoallv_array, alltoallv_workspace)
        }
        (false, C2cOperation::Inverse) => {
            alltoallv_plan.inverse_in_place(alltoallv_array, alltoallv_workspace)
        }
        (false, C2cOperation::Backward) => {
            alltoallv_plan.backward_in_place(alltoallv_array, alltoallv_workspace)
        }
    };
    assert!(matches!(
        result,
        Err(FftError::CollectiveDescriptorMismatch)
    ));
    assert_eq!(alltoallv_array.state(), alltoallv_state);
    assert_eq!(
        alltoallv_array.view().unwrap().as_slice(),
        alltoallv_data.as_slice()
    );
    assert_eq!(
        format!("{alltoallv_workspace:?}"),
        alltoallv_workspace_before
    );
    assert_eq!(point_to_point_array.state(), point_to_point_state);
    assert_eq!(
        point_to_point_array.view().unwrap().as_slice(),
        point_to_point_data.as_slice()
    );
    assert_eq!(
        format!("{point_to_point_workspace:?}"),
        point_to_point_workspace_before
    );
}

fn negative_transport_cases(
    world: &mpi::topology::SimpleCommunicator,
    topology_1d: &Arc<MpiTopology<1>>,
    topology_2d: &Arc<MpiTopology<2>>,
) {
    if world.size() == 1 {
        return;
    }
    negative_transport_layout(world, topology_1d, [3, 4], 0, 61.0);
    if world.size() == 6 {
        // Rank 5 is outside the changed-axis subgroup; the layout helper also
        // reuses both valid plans after every rejected operation.
        negative_transport_layout(world, topology_2d, [3, 1, 3, 4], 5, 63.0);
    }
}

fn negative_transport_layout<const N: usize, const M: usize>(
    world: &mpi::topology::SimpleCommunicator,
    topology: &Arc<MpiTopology<M>>,
    global_shape: [usize; N],
    mismatch_rank: i32,
    seed: f64,
) where
    Complex<f64>: mpi::datatype::Equivalence,
{
    let alltoallv_plan = C2cPlan::<f64, N, M>::from_shape_with_method(
        Arc::clone(topology),
        global_shape,
        ExtraShape::scalar(),
        TransposeMethod::AllToAllv,
    )
    .unwrap();
    let point_to_point_plan = C2cPlan::<f64, N, M>::from_shape_with_method(
        Arc::clone(topology),
        global_shape,
        ExtraShape::scalar(),
        TransposeMethod::PointToPoint,
    )
    .unwrap();
    let use_point_to_point = world.rank() == mismatch_rank;

    let mut alltoallv_source = alltoallv_plan.allocate_input().unwrap();
    let mut point_to_point_source = point_to_point_plan.allocate_input().unwrap();
    fill_input(&mut alltoallv_source, seed);
    fill_input(&mut point_to_point_source, seed);
    let mut alltoallv_output = alltoallv_plan.allocate_output().unwrap();
    let mut point_to_point_output = point_to_point_plan.allocate_output().unwrap();
    let mut alltoallv_inverse = alltoallv_plan.allocate_input().unwrap();
    let mut point_to_point_inverse = point_to_point_plan.allocate_input().unwrap();
    let mut alltoallv_workspace = alltoallv_plan.allocate_out_of_place_workspace().unwrap();
    let mut point_to_point_workspace = point_to_point_plan
        .allocate_out_of_place_workspace()
        .unwrap();
    let (mut alltoallv_array, mut alltoallv_in_place_workspace) =
        fresh_in_place(&alltoallv_plan, seed + 1.0);
    let (mut point_to_point_array, mut point_to_point_in_place_workspace) =
        fresh_in_place(&point_to_point_plan, seed + 1.0);

    let alltoallv_resources_before = format!(
        "{:?}|{:?}|{:?}|{:?}|{:?}|{:?}",
        alltoallv_source.as_slice(),
        alltoallv_output.as_slice(),
        alltoallv_inverse.as_slice(),
        alltoallv_workspace,
        alltoallv_array,
        alltoallv_in_place_workspace,
    );
    let point_to_point_resources_before = format!(
        "{:?}|{:?}|{:?}|{:?}|{:?}|{:?}",
        point_to_point_source.as_slice(),
        point_to_point_output.as_slice(),
        point_to_point_inverse.as_slice(),
        point_to_point_workspace,
        point_to_point_array,
        point_to_point_in_place_workspace,
    );
    let constructor_mismatch = if use_point_to_point {
        C2cPlan::<f64, N, M>::from_shape_with_method(
            Arc::clone(topology),
            global_shape,
            ExtraShape::scalar(),
            TransposeMethod::PointToPoint,
        )
    } else {
        C2cPlan::<f64, N, M>::from_shape_with_method(
            Arc::clone(topology),
            global_shape,
            ExtraShape::scalar(),
            TransposeMethod::AllToAllv,
        )
    };
    assert!(matches!(
        constructor_mismatch,
        Err(FftError::CollectiveDescriptorMismatch)
    ));
    assert_eq!(
        format!(
            "{:?}|{:?}|{:?}|{:?}|{:?}|{:?}",
            alltoallv_source.as_slice(),
            alltoallv_output.as_slice(),
            alltoallv_inverse.as_slice(),
            alltoallv_workspace,
            alltoallv_array,
            alltoallv_in_place_workspace,
        ),
        alltoallv_resources_before
    );
    assert_eq!(
        format!(
            "{:?}|{:?}|{:?}|{:?}|{:?}|{:?}",
            point_to_point_source.as_slice(),
            point_to_point_output.as_slice(),
            point_to_point_inverse.as_slice(),
            point_to_point_workspace,
            point_to_point_array,
            point_to_point_in_place_workspace,
        ),
        point_to_point_resources_before
    );
    world.barrier();

    // Reuse both valid plans after the constructor rejection.
    alltoallv_plan
        .forward(
            &alltoallv_source,
            &mut alltoallv_output,
            &mut alltoallv_workspace,
        )
        .unwrap();
    point_to_point_plan
        .forward(
            &point_to_point_source,
            &mut point_to_point_output,
            &mut point_to_point_workspace,
        )
        .unwrap();
    world.barrier();

    assert_mixed_transport_oop(
        use_point_to_point,
        C2cOperation::Forward,
        OopTransport {
            plan: &alltoallv_plan,
            source: &alltoallv_source,
            destination: &mut alltoallv_output,
            workspace: &mut alltoallv_workspace,
        },
        OopTransport {
            plan: &point_to_point_plan,
            source: &point_to_point_source,
            destination: &mut point_to_point_output,
            workspace: &mut point_to_point_workspace,
        },
    );
    alltoallv_plan
        .forward(
            &alltoallv_source,
            &mut alltoallv_output,
            &mut alltoallv_workspace,
        )
        .unwrap();
    point_to_point_plan
        .forward(
            &point_to_point_source,
            &mut point_to_point_output,
            &mut point_to_point_workspace,
        )
        .unwrap();
    world.barrier();

    assert_mixed_transport_oop(
        use_point_to_point,
        C2cOperation::Inverse,
        OopTransport {
            plan: &alltoallv_plan,
            source: &alltoallv_output,
            destination: &mut alltoallv_inverse,
            workspace: &mut alltoallv_workspace,
        },
        OopTransport {
            plan: &point_to_point_plan,
            source: &point_to_point_output,
            destination: &mut point_to_point_inverse,
            workspace: &mut point_to_point_workspace,
        },
    );
    alltoallv_plan
        .inverse(
            &alltoallv_output,
            &mut alltoallv_inverse,
            &mut alltoallv_workspace,
        )
        .unwrap();
    point_to_point_plan
        .inverse(
            &point_to_point_output,
            &mut point_to_point_inverse,
            &mut point_to_point_workspace,
        )
        .unwrap();
    world.barrier();

    assert_mixed_transport_oop(
        use_point_to_point,
        C2cOperation::Backward,
        OopTransport {
            plan: &alltoallv_plan,
            source: &alltoallv_output,
            destination: &mut alltoallv_inverse,
            workspace: &mut alltoallv_workspace,
        },
        OopTransport {
            plan: &point_to_point_plan,
            source: &point_to_point_output,
            destination: &mut point_to_point_inverse,
            workspace: &mut point_to_point_workspace,
        },
    );
    alltoallv_plan
        .backward(
            &alltoallv_output,
            &mut alltoallv_inverse,
            &mut alltoallv_workspace,
        )
        .unwrap();
    point_to_point_plan
        .backward(
            &point_to_point_output,
            &mut point_to_point_inverse,
            &mut point_to_point_workspace,
        )
        .unwrap();
    world.barrier();

    assert_mixed_transport_in_place(
        use_point_to_point,
        C2cOperation::Forward,
        InPlaceTransport {
            plan: &alltoallv_plan,
            array: &mut alltoallv_array,
            workspace: &mut alltoallv_in_place_workspace,
        },
        InPlaceTransport {
            plan: &point_to_point_plan,
            array: &mut point_to_point_array,
            workspace: &mut point_to_point_in_place_workspace,
        },
    );
    alltoallv_plan
        .forward_in_place(&mut alltoallv_array, &mut alltoallv_in_place_workspace)
        .unwrap();
    point_to_point_plan
        .forward_in_place(
            &mut point_to_point_array,
            &mut point_to_point_in_place_workspace,
        )
        .unwrap();
    world.barrier();

    assert_mixed_transport_in_place(
        use_point_to_point,
        C2cOperation::Backward,
        InPlaceTransport {
            plan: &alltoallv_plan,
            array: &mut alltoallv_array,
            workspace: &mut alltoallv_in_place_workspace,
        },
        InPlaceTransport {
            plan: &point_to_point_plan,
            array: &mut point_to_point_array,
            workspace: &mut point_to_point_in_place_workspace,
        },
    );
    alltoallv_plan
        .backward_in_place(&mut alltoallv_array, &mut alltoallv_in_place_workspace)
        .unwrap();
    point_to_point_plan
        .backward_in_place(
            &mut point_to_point_array,
            &mut point_to_point_in_place_workspace,
        )
        .unwrap();
    alltoallv_plan
        .forward_in_place(&mut alltoallv_array, &mut alltoallv_in_place_workspace)
        .unwrap();
    point_to_point_plan
        .forward_in_place(
            &mut point_to_point_array,
            &mut point_to_point_in_place_workspace,
        )
        .unwrap();
    world.barrier();

    assert_mixed_transport_in_place(
        use_point_to_point,
        C2cOperation::Inverse,
        InPlaceTransport {
            plan: &alltoallv_plan,
            array: &mut alltoallv_array,
            workspace: &mut alltoallv_in_place_workspace,
        },
        InPlaceTransport {
            plan: &point_to_point_plan,
            array: &mut point_to_point_array,
            workspace: &mut point_to_point_in_place_workspace,
        },
    );
    alltoallv_plan
        .inverse_in_place(&mut alltoallv_array, &mut alltoallv_in_place_workspace)
        .unwrap();
    point_to_point_plan
        .inverse_in_place(
            &mut point_to_point_array,
            &mut point_to_point_in_place_workspace,
        )
        .unwrap();
    world.barrier();
}

fn negative_collective_cases(
    world: &mpi::topology::SimpleCommunicator,
    topology_1d: &Arc<MpiTopology<1>>,
    topology_2d: &Arc<MpiTopology<2>>,
) {
    let rank = world.rank();
    let world_size = usize::try_from(world.size()).unwrap();

    let zero_shape =
        C2cPlan::<f64, 2, 1>::from_shape(Arc::clone(topology_1d), [0, 4], ExtraShape::scalar());
    assert!(zero_shape.is_err());
    world.barrier();

    // These dimensions are valid for Pencil, but not for distributed C2C.
    let one_dimensional =
        C2cPlan::<f64, 1, 1>::from_shape(Arc::clone(topology_1d), [4], ExtraShape::scalar());
    assert!(one_dimensional.is_err());
    world.barrier();
    let fully_distributed =
        C2cPlan::<f64, 2, 2>::from_shape(Arc::clone(topology_2d), [3, 4], ExtraShape::scalar());
    assert!(fully_distributed.is_err());
    world.barrier();

    let canonical = Pencil::<2, 1>::new(Arc::clone(topology_1d), [3, 4], [0]).unwrap();
    let noncanonical_permutation = canonical
        .with_permutation(AxisPermutation::new([1, 0]).unwrap())
        .unwrap();
    let noncanonical_decomposition = canonical.with_decomposition([1]).unwrap();
    for noncanonical in [noncanonical_permutation, noncanonical_decomposition] {
        let input = if rank == 0 {
            Arc::clone(&noncanonical)
        } else {
            Arc::clone(&canonical)
        };
        assert!(C2cPlan::<f64, 2, 1>::from_pencil(input, ExtraShape::scalar()).is_err());
        world.barrier();
    }

    if world_size > 1 {
        let shape_mismatch = C2cPlan::<f64, 2, 1>::from_shape(
            Arc::clone(topology_1d),
            if rank == 0 { [3, 4] } else { [2, 6] },
            ExtraShape::scalar(),
        );
        assert!(shape_mismatch.is_err());
        world.barrier();

        let extra_mismatch = C2cPlan::<f64, 2, 1>::from_shape(
            Arc::clone(topology_1d),
            [3, 4],
            ExtraShape::new(if rank == 0 { [2, 3] } else { [3, 2] }).unwrap(),
        );
        assert!(extra_mismatch.is_err());
        world.barrier();

        let scalar_mismatch = if rank == 0 {
            C2cPlan::<f32, 2, 1>::from_shape(Arc::clone(topology_1d), [3, 4], ExtraShape::scalar())
                .is_err()
        } else {
            C2cPlan::<f64, 2, 1>::from_shape(Arc::clone(topology_1d), [3, 4], ExtraShape::scalar())
                .is_err()
        };
        assert!(scalar_mismatch);
        world.barrier();

        let dimension_mismatch = if rank == 0 {
            C2cPlan::<f64, 3, 1>::from_shape(
                Arc::clone(topology_1d),
                [3, 2, 4],
                ExtraShape::scalar(),
            )
            .is_err()
        } else {
            C2cPlan::<f64, 2, 1>::from_shape(Arc::clone(topology_1d), [3, 4], ExtraShape::scalar())
                .is_err()
        };
        assert!(dimension_mismatch);
        world.barrier();
    }

    // Build a same-layout foreign topology collectively before using it on a
    // selected rank. The pointer identity is part of Pencil layout identity.
    let foreign_topology = MpiTopology::<1>::new(world, [world_size]).unwrap();
    let foreign_input_pencil =
        Pencil::<2, 1>::new(Arc::clone(&foreign_topology), [3, 4], [0]).unwrap();
    let foreign_output_pencil = foreign_input_pencil
        .with_decomposition([1])
        .unwrap()
        .with_permutation(AxisPermutation::new([1, 0]).unwrap())
        .unwrap();
    let mut foreign_source = PencilArray::from_elem(
        Arc::clone(&foreign_input_pencil),
        ExtraShape::scalar(),
        Complex::new(17.0_f64, -17.0),
    )
    .unwrap();
    let mut foreign_destination = PencilArray::from_elem(
        Arc::clone(&foreign_output_pencil),
        ExtraShape::scalar(),
        Complex::new(19.0_f64, -19.0),
    )
    .unwrap();

    let valid_plan =
        C2cPlan::<f64, 2, 1>::from_shape(Arc::clone(topology_1d), [3, 4], ExtraShape::scalar())
            .unwrap();
    let mut source = valid_plan.allocate_input().unwrap();
    fill_input(&mut source, 21.0);
    let mut workspace = valid_plan.allocate_out_of_place_workspace().unwrap();
    let mut valid_output = valid_plan.allocate_output().unwrap();
    valid_plan
        .forward(&source, &mut valid_output, &mut workspace)
        .unwrap();

    let wrong_source = valid_plan.allocate_output().unwrap();
    let mut source_layout_destination = valid_plan.allocate_output().unwrap();
    if rank == 0 {
        assert_c2c_rejected(
            &wrong_source,
            &mut source_layout_destination,
            &mut workspace,
            |source, destination, workspace| valid_plan.forward(source, destination, workspace),
        );
    } else {
        assert_c2c_rejected(
            &source,
            &mut source_layout_destination,
            &mut workspace,
            |source, destination, workspace| valid_plan.forward(source, destination, workspace),
        );
    }
    reuse_forward(&valid_plan, &source, &mut workspace);
    world.barrier();

    let mut backward_layout_destination = valid_plan.allocate_input().unwrap();
    if rank == 0 {
        assert_c2c_rejected(
            &source,
            &mut backward_layout_destination,
            &mut workspace,
            |source, destination, workspace| valid_plan.backward(source, destination, workspace),
        );
    } else {
        assert_c2c_rejected(
            &valid_output,
            &mut backward_layout_destination,
            &mut workspace,
            |source, destination, workspace| valid_plan.backward(source, destination, workspace),
        );
    }
    reuse_backward(&valid_plan, &valid_output, &mut workspace);
    world.barrier();

    let mut wrong_destination = valid_plan.allocate_input().unwrap();
    let mut destination_layout_destination = valid_plan.allocate_output().unwrap();
    if rank == 0 {
        assert_c2c_rejected(
            &source,
            &mut wrong_destination,
            &mut workspace,
            |source, destination, workspace| valid_plan.forward(source, destination, workspace),
        );
    } else {
        assert_c2c_rejected(
            &source,
            &mut destination_layout_destination,
            &mut workspace,
            |source, destination, workspace| valid_plan.forward(source, destination, workspace),
        );
    }
    reuse_forward(&valid_plan, &source, &mut workspace);
    world.barrier();

    if rank == 0 {
        assert_c2c_rejected(
            &valid_output,
            &mut destination_layout_destination,
            &mut workspace,
            |source, destination, workspace| valid_plan.backward(source, destination, workspace),
        );
    } else {
        assert_c2c_rejected(
            &valid_output,
            &mut backward_layout_destination,
            &mut workspace,
            |source, destination, workspace| valid_plan.backward(source, destination, workspace),
        );
    }
    reuse_backward(&valid_plan, &valid_output, &mut workspace);
    world.barrier();

    let wrong_global_pencil = Pencil::<2, 1>::new(Arc::clone(topology_1d), [2, 6], [0]).unwrap();
    let wrong_global_source = PencilArray::from_elem(
        wrong_global_pencil,
        ExtraShape::scalar(),
        Complex::new(0.0_f64, 0.0),
    )
    .unwrap();
    let mut global_destination = valid_plan.allocate_output().unwrap();
    if rank == 0 {
        assert_c2c_rejected(
            &wrong_global_source,
            &mut global_destination,
            &mut workspace,
            |source, destination, workspace| valid_plan.forward(source, destination, workspace),
        );
    } else {
        assert_c2c_rejected(
            &source,
            &mut global_destination,
            &mut workspace,
            |source, destination, workspace| valid_plan.forward(source, destination, workspace),
        );
    }
    reuse_forward(&valid_plan, &source, &mut workspace);
    world.barrier();

    // Both foreign arrays have the same visible layout values as the plan's
    // endpoints, but their topology Arc is different. Check each endpoint
    // independently so neither mismatch can be hidden by the other.
    let mut foreign_source_destination = valid_plan.allocate_output().unwrap();
    if rank == 0 {
        assert_c2c_rejected(
            &foreign_source,
            &mut foreign_source_destination,
            &mut workspace,
            |source, destination, workspace| valid_plan.forward(source, destination, workspace),
        );
    } else {
        assert_c2c_rejected(
            &source,
            &mut foreign_source_destination,
            &mut workspace,
            |source, destination, workspace| valid_plan.forward(source, destination, workspace),
        );
    }
    reuse_forward(&valid_plan, &source, &mut workspace);
    world.barrier();

    let mut foreign_destination_case = valid_plan.allocate_output().unwrap();
    if rank == 0 {
        assert_c2c_rejected(
            &source,
            &mut foreign_destination,
            &mut workspace,
            |source, destination, workspace| valid_plan.forward(source, destination, workspace),
        );
    } else {
        assert_c2c_rejected(
            &source,
            &mut foreign_destination_case,
            &mut workspace,
            |source, destination, workspace| valid_plan.forward(source, destination, workspace),
        );
    }
    reuse_forward(&valid_plan, &source, &mut workspace);
    world.barrier();

    let mut foreign_backward_destination = valid_plan.allocate_input().unwrap();
    if rank == 0 {
        assert_c2c_rejected(
            &foreign_destination,
            &mut foreign_backward_destination,
            &mut workspace,
            |source, destination, workspace| valid_plan.backward(source, destination, workspace),
        );
    } else {
        assert_c2c_rejected(
            &valid_output,
            &mut foreign_backward_destination,
            &mut workspace,
            |source, destination, workspace| valid_plan.backward(source, destination, workspace),
        );
    }
    reuse_backward(&valid_plan, &valid_output, &mut workspace);
    world.barrier();

    let foreign_backward_source = valid_plan.allocate_output().unwrap();
    if rank == 0 {
        assert_c2c_rejected(
            &foreign_backward_source,
            &mut foreign_source,
            &mut workspace,
            |source, destination, workspace| valid_plan.backward(source, destination, workspace),
        );
    } else {
        assert_c2c_rejected(
            &valid_output,
            &mut foreign_backward_destination,
            &mut workspace,
            |source, destination, workspace| valid_plan.backward(source, destination, workspace),
        );
    }
    reuse_backward(&valid_plan, &valid_output, &mut workspace);
    world.barrier();

    let batch_extra = ExtraShape::new([2, 3]).unwrap();
    let batch_plan =
        C2cPlan::<f64, 2, 1>::from_shape(Arc::clone(topology_1d), [3, 4], batch_extra).unwrap();
    let batch_source = batch_plan.allocate_input().unwrap();
    let wrong_extra = ExtraShape::new([3, 2]).unwrap();
    let wrong_extra_source = PencilArray::from_elem(
        Arc::clone(batch_plan.input_pencil()),
        wrong_extra.clone(),
        Complex::new(0.0_f64, 0.0),
    )
    .unwrap();
    let wrong_backward_source = PencilArray::from_elem(
        Arc::clone(batch_plan.output_pencil()),
        wrong_extra.clone(),
        Complex::new(0.0_f64, 0.0),
    )
    .unwrap();
    let mut wrong_backward_destination = PencilArray::from_elem(
        Arc::clone(batch_plan.input_pencil()),
        wrong_extra.clone(),
        Complex::new(0.0_f64, 0.0),
    )
    .unwrap();
    let mut source_extra_destination = batch_plan.allocate_output().unwrap();
    let mut batch_output = batch_plan.allocate_output().unwrap();
    let mut batch_workspace = batch_plan.allocate_out_of_place_workspace().unwrap();
    batch_plan
        .forward(&batch_source, &mut batch_output, &mut batch_workspace)
        .unwrap();
    if rank == 0 {
        assert_c2c_rejected(
            &wrong_extra_source,
            &mut source_extra_destination,
            &mut batch_workspace,
            |source, destination, workspace| batch_plan.forward(source, destination, workspace),
        );
    } else {
        assert_c2c_rejected(
            &batch_source,
            &mut source_extra_destination,
            &mut batch_workspace,
            |source, destination, workspace| batch_plan.forward(source, destination, workspace),
        );
    }
    reuse_forward(&batch_plan, &batch_source, &mut batch_workspace);
    world.barrier();

    let mut batch_backward_destination = batch_plan.allocate_input().unwrap();
    if rank == 0 {
        assert_c2c_rejected_exact(
            &wrong_backward_source,
            &mut batch_backward_destination,
            &mut batch_workspace,
            |source, destination, workspace| batch_plan.backward(source, destination, workspace),
            |error| matches!(error, FftError::ExtraShapeMismatch),
        );
    } else {
        assert_c2c_rejected_exact(
            &batch_output,
            &mut batch_backward_destination,
            &mut batch_workspace,
            |source, destination, workspace| batch_plan.backward(source, destination, workspace),
            |error| matches!(error, FftError::CollectivePreconditionFailed),
        );
    }
    reuse_backward(&batch_plan, &batch_output, &mut batch_workspace);
    world.barrier();

    let mut wrong_extra_destination = PencilArray::from_elem(
        Arc::clone(batch_plan.output_pencil()),
        wrong_extra,
        Complex::new(9.0_f64, -9.0),
    )
    .unwrap();
    let mut destination_extra_destination = batch_plan.allocate_output().unwrap();
    if rank == 0 {
        assert_c2c_rejected(
            &batch_source,
            &mut wrong_extra_destination,
            &mut batch_workspace,
            |source, destination, workspace| batch_plan.forward(source, destination, workspace),
        );
    } else {
        assert_c2c_rejected(
            &batch_source,
            &mut destination_extra_destination,
            &mut batch_workspace,
            |source, destination, workspace| batch_plan.forward(source, destination, workspace),
        );
    }
    reuse_forward(&batch_plan, &batch_source, &mut batch_workspace);
    world.barrier();

    if rank == 0 {
        assert_c2c_rejected_exact(
            &batch_output,
            &mut wrong_backward_destination,
            &mut batch_workspace,
            |source, destination, workspace| batch_plan.backward(source, destination, workspace),
            |error| matches!(error, FftError::ExtraShapeMismatch),
        );
    } else {
        assert_c2c_rejected_exact(
            &batch_output,
            &mut batch_backward_destination,
            &mut batch_workspace,
            |source, destination, workspace| batch_plan.backward(source, destination, workspace),
            |error| matches!(error, FftError::CollectivePreconditionFailed),
        );
    }
    reuse_backward(&batch_plan, &batch_output, &mut batch_workspace);
    world.barrier();

    let other_plan =
        C2cPlan::<f64, 2, 1>::from_shape(Arc::clone(topology_1d), [3, 4], ExtraShape::scalar())
            .unwrap();
    let mut other_workspace = other_plan.allocate_out_of_place_workspace().unwrap();
    let mut workspace_destination = valid_plan.allocate_output().unwrap();
    if rank == 0 {
        assert_c2c_rejected(
            &source,
            &mut workspace_destination,
            &mut other_workspace,
            |source, destination, workspace| valid_plan.forward(source, destination, workspace),
        );
    } else {
        assert_c2c_rejected(
            &source,
            &mut workspace_destination,
            &mut workspace,
            |source, destination, workspace| valid_plan.forward(source, destination, workspace),
        );
    }
    reuse_forward(&valid_plan, &source, &mut workspace);
    world.barrier();

    let mut backward_workspace_destination = valid_plan.allocate_input().unwrap();
    if rank == 0 {
        assert_c2c_rejected(
            &valid_output,
            &mut backward_workspace_destination,
            &mut other_workspace,
            |source, destination, workspace| valid_plan.backward(source, destination, workspace),
        );
    } else {
        assert_c2c_rejected(
            &valid_output,
            &mut backward_workspace_destination,
            &mut workspace,
            |source, destination, workspace| valid_plan.backward(source, destination, workspace),
        );
    }
    reuse_backward(&valid_plan, &valid_output, &mut workspace);
    world.barrier();

    if world_size > 1 {
        // A constructor on one rank must not be allowed to pair with an FFT
        // execution on the other ranks.
        let mut constructor_execution_destination = valid_plan.allocate_output().unwrap();
        let mut constructor_execution_workspace =
            valid_plan.allocate_out_of_place_workspace().unwrap();
        let source_before = source.as_slice().to_vec();
        let destination_before = constructor_execution_destination.as_slice().to_vec();
        let workspace_before = format!("{constructor_execution_workspace:?}");
        let mixed_result = if rank == 0 {
            C2cPlan::<f64, 2, 1>::from_pencil(
                Arc::clone(valid_plan.input_pencil()),
                ExtraShape::scalar(),
            )
            .map(|_| ())
        } else {
            valid_plan.forward(
                &source,
                &mut constructor_execution_destination,
                &mut constructor_execution_workspace,
            )
        };
        assert!(mixed_result.is_err());
        assert_eq!(source.as_slice(), source_before.as_slice());
        assert_eq!(
            constructor_execution_destination.as_slice(),
            destination_before.as_slice()
        );
        assert_eq!(
            format!("{constructor_execution_workspace:?}"),
            workspace_before
        );
        world.barrier();
        reuse_forward(&valid_plan, &source, &mut constructor_execution_workspace);
        world.barrier();

        let mut inverse_destination = valid_plan.allocate_input().unwrap();
        let mut arbitrary_spectrum = valid_plan.allocate_output().unwrap();
        fill_spectrum(&mut arbitrary_spectrum, 25.0);
        let mut direction_workspace = valid_plan.allocate_out_of_place_workspace().unwrap();
        if rank == 0 {
            let mut forward_destination = valid_plan.allocate_output().unwrap();
            assert_c2c_rejected(
                &source,
                &mut forward_destination,
                &mut direction_workspace,
                |source, destination, workspace| valid_plan.forward(source, destination, workspace),
            );
        } else {
            assert_c2c_rejected(
                &arbitrary_spectrum,
                &mut inverse_destination,
                &mut direction_workspace,
                |source, destination, workspace| valid_plan.inverse(source, destination, workspace),
            );
        }
        reuse_forward(&valid_plan, &source, &mut direction_workspace);
        world.barrier();

        // Distinct raw backward and normalized inverse/forward operation words
        // reject at the full-Cartesian header before any buffer changes.
        assert_mixed_direction_oop(
            &valid_plan,
            world,
            C2cOperation::Backward,
            C2cOperation::Inverse,
            25.5,
        );
        assert_mixed_direction_oop(
            &valid_plan,
            world,
            C2cOperation::Backward,
            C2cOperation::Forward,
            25.75,
        );

        let transpose_plan = AllToAllvTransposePlan::new(
            Arc::clone(valid_plan.input_pencil()),
            Arc::clone(valid_plan.output_pencil()),
        )
        .unwrap();
        let requirements = transpose_plan
            .workspace_requirements(&ExtraShape::scalar())
            .unwrap();
        let mut transpose_workspace = TransposeWorkspace::from_vecs(
            vec![Complex::new(0.0_f64, 0.0); requirements.send_len],
            vec![Complex::new(0.0_f64, 0.0); requirements.receive_len],
        );
        let mut transpose_destination = valid_plan.allocate_output().unwrap();
        if rank == 0 {
            let source_before = source.as_slice().to_vec();
            let destination_before = transpose_destination.as_slice().to_vec();
            let workspace_before = format!("{transpose_workspace:?}");
            let result = transpose_plan.execute_views(
                source.view(),
                transpose_destination.view_mut(),
                &mut transpose_workspace,
            );
            assert!(result.is_err());
            assert_eq!(source.as_slice(), source_before.as_slice());
            assert_eq!(
                transpose_destination.as_slice(),
                destination_before.as_slice()
            );
            assert_eq!(format!("{transpose_workspace:?}"), workspace_before);
        } else {
            assert_c2c_rejected(
                &source,
                &mut transpose_destination,
                &mut workspace,
                |source, destination, workspace| valid_plan.forward(source, destination, workspace),
            );
        }
        world.barrier();

        // Reuse both workspaces after the mixed-API rejection. The Alltoallv
        // call also proves its workspace was not touched by the failed call.
        let mut transpose_reuse_destination = valid_plan.allocate_output().unwrap();
        transpose_plan
            .execute_views(
                source.view(),
                transpose_reuse_destination.view_mut(),
                &mut transpose_workspace,
            )
            .unwrap();
        reuse_forward(&valid_plan, &source, &mut workspace);
        world.barrier();

        let constructor_mix = if rank == 0 {
            AllToAllvTransposePlan::new(
                Arc::clone(valid_plan.input_pencil()),
                Arc::clone(valid_plan.output_pencil()),
            )
            .is_err()
        } else {
            C2cPlan::<f64, 2, 1>::from_pencil(
                Arc::clone(valid_plan.input_pencil()),
                ExtraShape::scalar(),
            )
            .is_err()
        };
        assert!(constructor_mix);
        world.barrier();

        let p2p_constructor_mix = if rank == 0 {
            PointToPointTransposePlan::new(
                Arc::clone(valid_plan.input_pencil()),
                Arc::clone(valid_plan.output_pencil()),
            )
            .is_err()
        } else {
            C2cPlan::<f64, 2, 1>::from_pencil(
                Arc::clone(valid_plan.input_pencil()),
                ExtraShape::scalar(),
            )
            .is_err()
        };
        assert!(p2p_constructor_mix);
        world.barrier();
    }

    if world_size == 6 {
        let plan_2d = C2cPlan::<f64, 4, 2>::from_shape(
            Arc::clone(topology_2d),
            [3, 1, 3, 4],
            ExtraShape::scalar(),
        )
        .unwrap();
        let mut source_2d = plan_2d.allocate_input().unwrap();
        fill_input(&mut source_2d, 41.0);
        let wrong_source_2d = plan_2d.allocate_output().unwrap();
        let mut destination_2d = plan_2d.allocate_output().unwrap();
        let mut workspace_2d = plan_2d.allocate_out_of_place_workspace().unwrap();
        if rank == 5 {
            assert_c2c_rejected(
                &wrong_source_2d,
                &mut destination_2d,
                &mut workspace_2d,
                |source, destination, workspace| plan_2d.forward(source, destination, workspace),
            );
        } else {
            assert_c2c_rejected(
                &source_2d,
                &mut destination_2d,
                &mut workspace_2d,
                |source, destination, workspace| plan_2d.forward(source, destination, workspace),
            );
        }
        world.barrier();
        reuse_forward(&plan_2d, &source_2d, &mut workspace_2d);
        world.barrier();
    }

    negative_selection_mask_cases(world, topology_1d);
    negative_transport_cases(world, topology_1d, topology_2d);
    for method in [TransposeMethod::AllToAllv, TransposeMethod::PointToPoint] {
        negative_in_place_cases(world, topology_1d, topology_2d, method);
    }
}

fn negative_selection_mask_cases(
    world: &mpi::topology::SimpleCommunicator,
    topology: &Arc<MpiTopology<1>>,
) {
    let selection_a = AxisSelection::from_indices([0, 2]).unwrap();
    let selection_b = AxisSelection::from_indices([1, 2]).unwrap();
    for method in [TransposeMethod::AllToAllv, TransposeMethod::PointToPoint] {
        let empty_r2c = R2cPlan::<f64, 3, 1>::from_shape_with_selection_and_method(
            Arc::clone(topology),
            [3, 2, 4],
            ExtraShape::scalar(),
            AxisSelection::empty(),
            method,
        );
        assert!(matches!(
            empty_r2c,
            Err(R2cError::Fft(FftError::InvalidDimensions))
        ));
        world.barrier();

        if world.size() == 1 {
            continue;
        }

        let constructor = if world.rank() == 0 {
            C2cPlan::<f64, 3, 1>::from_shape_with_selection_and_method(
                Arc::clone(topology),
                [3, 2, 4],
                ExtraShape::scalar(),
                selection_a,
                method,
            )
            .map(|_| ())
        } else {
            C2cPlan::<f64, 3, 1>::from_shape_with_selection_and_method(
                Arc::clone(topology),
                [3, 2, 4],
                ExtraShape::scalar(),
                selection_b,
                method,
            )
            .map(|_| ())
        };
        assert!(matches!(
            constructor,
            Err(FftError::CollectiveDescriptorMismatch)
        ));
        world.barrier();

        let c2c_a = C2cPlan::<f64, 3, 1>::from_shape_with_selection_and_method(
            Arc::clone(topology),
            [3, 2, 4],
            ExtraShape::scalar(),
            selection_a,
            method,
        )
        .unwrap();
        let c2c_b = C2cPlan::<f64, 3, 1>::from_shape_with_selection_and_method(
            Arc::clone(topology),
            [3, 2, 4],
            ExtraShape::scalar(),
            selection_b,
            method,
        )
        .unwrap();
        let mut c2c_source_a = c2c_a.allocate_input().unwrap();
        let mut c2c_source_b = c2c_b.allocate_input().unwrap();
        fill_input(&mut c2c_source_a, 101.0);
        fill_input(&mut c2c_source_b, 101.0);
        let mut c2c_output_a = c2c_a.allocate_output().unwrap();
        let mut c2c_output_b = c2c_b.allocate_output().unwrap();
        let mut c2c_workspace_a = c2c_a.allocate_out_of_place_workspace().unwrap();
        let mut c2c_workspace_b = c2c_b.allocate_out_of_place_workspace().unwrap();
        let c2c_source_a_before = complex_bits(c2c_source_a.as_slice());
        let c2c_source_b_before = complex_bits(c2c_source_b.as_slice());
        let c2c_output_a_before = complex_bits(c2c_output_a.as_slice());
        let c2c_output_b_before = complex_bits(c2c_output_b.as_slice());
        let c2c_workspace_a_before = format!("{c2c_workspace_a:?}");
        let c2c_workspace_b_before = format!("{c2c_workspace_b:?}");
        let c2c_result = if world.rank() == 0 {
            c2c_a.forward(&c2c_source_a, &mut c2c_output_a, &mut c2c_workspace_a)
        } else {
            c2c_b.forward(&c2c_source_b, &mut c2c_output_b, &mut c2c_workspace_b)
        };
        assert!(matches!(
            c2c_result,
            Err(FftError::CollectiveDescriptorMismatch)
        ));
        assert_eq!(complex_bits(c2c_source_a.as_slice()), c2c_source_a_before);
        assert_eq!(complex_bits(c2c_source_b.as_slice()), c2c_source_b_before);
        assert_eq!(complex_bits(c2c_output_a.as_slice()), c2c_output_a_before);
        assert_eq!(complex_bits(c2c_output_b.as_slice()), c2c_output_b_before);
        assert_eq!(format!("{c2c_workspace_a:?}"), c2c_workspace_a_before);
        assert_eq!(format!("{c2c_workspace_b:?}"), c2c_workspace_b_before);
        c2c_a
            .forward(&c2c_source_a, &mut c2c_output_a, &mut c2c_workspace_a)
            .unwrap();
        c2c_b
            .forward(&c2c_source_b, &mut c2c_output_b, &mut c2c_workspace_b)
            .unwrap();
        world.barrier();

        let r2c_a = R2cPlan::<f64, 3, 1>::from_shape_with_selection_and_method(
            Arc::clone(topology),
            [3, 2, 4],
            ExtraShape::scalar(),
            selection_a,
            method,
        )
        .unwrap();
        let r2c_b = R2cPlan::<f64, 3, 1>::from_shape_with_selection_and_method(
            Arc::clone(topology),
            [3, 2, 4],
            ExtraShape::scalar(),
            selection_b,
            method,
        )
        .unwrap();
        let mut r2c_source_a = r2c_a.allocate_input().unwrap();
        let mut r2c_source_b = r2c_b.allocate_input().unwrap();
        fill_r2c_input(&mut r2c_source_a, 102.0);
        fill_r2c_input(&mut r2c_source_b, 102.0);
        let mut r2c_output_a = r2c_a.allocate_output().unwrap();
        let mut r2c_output_b = r2c_b.allocate_output().unwrap();
        let mut r2c_workspace_a = r2c_a.allocate_workspace().unwrap();
        let mut r2c_workspace_b = r2c_b.allocate_workspace().unwrap();
        let r2c_source_a_before = real_bits(r2c_source_a.as_slice());
        let r2c_source_b_before = real_bits(r2c_source_b.as_slice());
        let r2c_output_a_before = complex_bits(r2c_output_a.as_slice());
        let r2c_output_b_before = complex_bits(r2c_output_b.as_slice());
        let r2c_workspace_a_before = format!("{r2c_workspace_a:?}");
        let r2c_workspace_b_before = format!("{r2c_workspace_b:?}");
        let r2c_result = if world.rank() == 0 {
            r2c_a.forward(&r2c_source_a, &mut r2c_output_a, &mut r2c_workspace_a)
        } else {
            r2c_b.forward(&r2c_source_b, &mut r2c_output_b, &mut r2c_workspace_b)
        };
        assert!(matches!(
            r2c_result,
            Err(R2cError::Fft(FftError::CollectiveDescriptorMismatch))
        ));
        assert_eq!(real_bits(r2c_source_a.as_slice()), r2c_source_a_before);
        assert_eq!(real_bits(r2c_source_b.as_slice()), r2c_source_b_before);
        assert_eq!(complex_bits(r2c_output_a.as_slice()), r2c_output_a_before);
        assert_eq!(complex_bits(r2c_output_b.as_slice()), r2c_output_b_before);
        assert_eq!(format!("{r2c_workspace_a:?}"), r2c_workspace_a_before);
        assert_eq!(format!("{r2c_workspace_b:?}"), r2c_workspace_b_before);
        r2c_a
            .forward(&r2c_source_a, &mut r2c_output_a, &mut r2c_workspace_a)
            .unwrap();
        r2c_b
            .forward(&r2c_source_b, &mut r2c_output_b, &mut r2c_workspace_b)
            .unwrap();
        world.barrier();
    }
}

fn negative_in_place_cases(
    world: &mpi::topology::SimpleCommunicator,
    topology_1d: &Arc<MpiTopology<1>>,
    topology_2d: &Arc<MpiTopology<2>>,
    method: TransposeMethod,
) {
    let rank = world.rank();
    let size = world.size();
    let plan = C2cPlan::<f64, 2, 1>::from_shape_with_method(
        Arc::clone(topology_1d),
        [3, 4],
        ExtraShape::scalar(),
        method,
    )
    .unwrap();

    // The wrong direction is rejected collectively without changing the
    // initial input state.
    let (mut wrong_state, mut wrong_state_workspace) = fresh_in_place(&plan, 50.0);
    assert_in_place_rejected(
        &mut wrong_state,
        &mut wrong_state_workspace,
        |array, workspace| plan.inverse_in_place(array, workspace),
    );
    assert_in_place_rejected(
        &mut wrong_state,
        &mut wrong_state_workspace,
        |array, workspace| plan.backward_in_place(array, workspace),
    );
    plan.forward_in_place(&mut wrong_state, &mut wrong_state_workspace)
        .unwrap();
    plan.inverse_in_place(&mut wrong_state, &mut wrong_state_workspace)
        .unwrap();
    plan.forward_in_place(&mut wrong_state, &mut wrong_state_workspace)
        .unwrap();
    plan.backward_in_place(&mut wrong_state, &mut wrong_state_workspace)
        .unwrap();
    world.barrier();

    if size > 1 {
        // A live in-place direction mismatch must stop at the shared header:
        // rank zero has valid input for operation 10 while every peer has
        // valid output for operation 11.
        let mut mixed_direction = plan.allocate_in_place().unwrap();
        fill_in_place(&mut mixed_direction, 50.25, false);
        let mut mixed_direction_workspace = plan.allocate_in_place_workspace().unwrap();
        plan.forward_in_place(&mut mixed_direction, &mut mixed_direction_workspace)
            .unwrap();
        if rank == 0 {
            mixed_direction = plan.allocate_in_place().unwrap();
            fill_in_place(&mut mixed_direction, 50.25, false);
        }
        let mixed_state_before = mixed_direction.state();
        let mixed_data_before = mixed_direction.view().unwrap().as_slice().to_vec();
        let mixed_workspace_before = format!("{mixed_direction_workspace:?}");
        let result = if rank == 0 {
            plan.forward_in_place(&mut mixed_direction, &mut mixed_direction_workspace)
        } else {
            plan.inverse_in_place(&mut mixed_direction, &mut mixed_direction_workspace)
        };
        assert!(matches!(
            result,
            Err(FftError::CollectiveDescriptorMismatch)
        ));
        assert_eq!(mixed_direction.state(), mixed_state_before);
        assert_eq!(
            mixed_direction.view().unwrap().as_slice(),
            mixed_data_before.as_slice()
        );
        assert_eq!(
            format!("{mixed_direction_workspace:?}"),
            mixed_workspace_before
        );

        // Re-align all ranks with fresh input while reusing the workspace.
        mixed_direction = plan.allocate_in_place().unwrap();
        fill_in_place(&mut mixed_direction, 50.25, false);
        plan.forward_in_place(&mut mixed_direction, &mut mixed_direction_workspace)
            .unwrap();
        plan.inverse_in_place(&mut mixed_direction, &mut mixed_direction_workspace)
            .unwrap();
        world.barrier();

        assert_mixed_direction_in_place(
            &plan,
            world,
            C2cOperation::Backward,
            C2cOperation::Inverse,
            50.75,
        );
        assert_mixed_direction_in_place(
            &plan,
            world,
            C2cOperation::Backward,
            C2cOperation::Forward,
            50.875,
        );
    }

    // Replacing only rank zero after a successful collective creates a
    // rank-local state mismatch. The full Cartesian preflight must reject it
    // before any rank writes.
    let (mut rank_local_state, mut rank_local_workspace) = fresh_in_place(&plan, 50.5);
    plan.forward_in_place(&mut rank_local_state, &mut rank_local_workspace)
        .unwrap();
    if rank == 0 {
        rank_local_state = plan.allocate_in_place().unwrap();
        fill_in_place(&mut rank_local_state, 50.5, false);
    }
    let rank_local_state_before = format!("{rank_local_state:?}");
    let rank_local_workspace_before = format!("{rank_local_workspace:?}");
    let result = plan.forward_in_place(&mut rank_local_state, &mut rank_local_workspace);
    if size == 1 {
        assert!(result.is_ok());
    } else {
        assert!(result.is_err());
        assert_eq!(format!("{rank_local_state:?}"), rank_local_state_before);
        assert_eq!(
            format!("{rank_local_workspace:?}"),
            rank_local_workspace_before
        );
    }
    rank_local_state = plan.allocate_in_place().unwrap();
    fill_in_place(&mut rank_local_state, 50.5, false);
    plan.forward_in_place(&mut rank_local_state, &mut rank_local_workspace)
        .unwrap();
    plan.inverse_in_place(&mut rank_local_state, &mut rank_local_workspace)
        .unwrap();
    world.barrier();

    // A second forward is rejected while the completed output remains usable.
    let (mut twice, mut twice_workspace) = fresh_in_place(&plan, 51.0);
    plan.forward_in_place(&mut twice, &mut twice_workspace)
        .unwrap();
    assert_eq!(twice.state(), C2cState::Output);
    assert_in_place_rejected(&mut twice, &mut twice_workspace, |array, workspace| {
        plan.forward_in_place(array, workspace)
    });
    plan.inverse_in_place(&mut twice, &mut twice_workspace)
        .unwrap();
    world.barrier();

    // Same-layout arrays from a separately constructed plan are still foreign
    // because plan identity is an Arc contract, not just a layout comparison.
    let foreign_plan = C2cPlan::<f64, 2, 1>::from_shape_with_method(
        Arc::clone(topology_1d),
        [3, 4],
        ExtraShape::scalar(),
        method,
    )
    .unwrap();
    let (mut valid_array, mut valid_workspace) = fresh_in_place(&plan, 52.0);
    let mut foreign_array = foreign_plan.allocate_in_place().unwrap();
    fill_in_place(&mut foreign_array, 52.0, false);
    let valid_array_before = format!("{valid_array:?}");
    let foreign_array_before = format!("{foreign_array:?}");
    let valid_workspace_before = format!("{valid_workspace:?}");
    let result = if rank == 0 {
        plan.forward_in_place(&mut foreign_array, &mut valid_workspace)
    } else {
        plan.forward_in_place(&mut valid_array, &mut valid_workspace)
    };
    assert!(result.is_err());
    assert_eq!(format!("{valid_array:?}"), valid_array_before);
    assert_eq!(format!("{foreign_array:?}"), foreign_array_before);
    assert_eq!(format!("{valid_workspace:?}"), valid_workspace_before);
    plan.forward_in_place(&mut valid_array, &mut valid_workspace)
        .unwrap();
    plan.inverse_in_place(&mut valid_array, &mut valid_workspace)
        .unwrap();
    world.barrier();

    // Check the workspace identity independently from the array identity.
    let (mut workspace_array, mut workspace_for_plan) = fresh_in_place(&plan, 53.0);
    let mut foreign_workspace = foreign_plan.allocate_in_place_workspace().unwrap();
    let workspace_array_before = format!("{workspace_array:?}");
    let workspace_for_plan_before = format!("{workspace_for_plan:?}");
    let foreign_workspace_before = format!("{foreign_workspace:?}");
    let result = if rank == 0 {
        plan.forward_in_place(&mut workspace_array, &mut foreign_workspace)
    } else {
        plan.forward_in_place(&mut workspace_array, &mut workspace_for_plan)
    };
    assert!(result.is_err());
    assert_eq!(format!("{workspace_array:?}"), workspace_array_before);
    assert_eq!(format!("{workspace_for_plan:?}"), workspace_for_plan_before);
    assert_eq!(format!("{foreign_workspace:?}"), foreign_workspace_before);
    plan.forward_in_place(&mut workspace_array, &mut workspace_for_plan)
        .unwrap();
    plan.inverse_in_place(&mut workspace_array, &mut workspace_for_plan)
        .unwrap();
    world.barrier();

    // In-place and out-of-place forward calls have distinct operation words.
    let (mut mixed_array, mut mixed_workspace) = fresh_in_place(&plan, 54.0);
    let mut mixed_source = plan.allocate_input().unwrap();
    fill_input(&mut mixed_source, 54.0);
    let mut mixed_destination = plan.allocate_output().unwrap();
    let mut mixed_oop_workspace = plan.allocate_out_of_place_workspace().unwrap();
    let mixed_array_before = format!("{mixed_array:?}");
    let mixed_workspace_before = format!("{mixed_workspace:?}");
    let mixed_source_before = mixed_source.as_slice().to_vec();
    let mixed_destination_before = mixed_destination.as_slice().to_vec();
    let mixed_oop_workspace_before = format!("{mixed_oop_workspace:?}");
    let result = if rank == 0 {
        plan.forward(
            &mixed_source,
            &mut mixed_destination,
            &mut mixed_oop_workspace,
        )
    } else {
        plan.forward_in_place(&mut mixed_array, &mut mixed_workspace)
    };
    if size == 1 {
        assert!(result.is_ok());
    } else {
        assert!(result.is_err());
        assert_eq!(mixed_source.as_slice(), mixed_source_before.as_slice());
        assert_eq!(
            mixed_destination.as_slice(),
            mixed_destination_before.as_slice()
        );
        assert_eq!(
            format!("{mixed_oop_workspace:?}"),
            mixed_oop_workspace_before
        );
    }
    assert_eq!(format!("{mixed_array:?}"), mixed_array_before);
    assert_eq!(format!("{mixed_workspace:?}"), mixed_workspace_before);
    plan.forward_in_place(&mut mixed_array, &mut mixed_workspace)
        .unwrap();
    plan.inverse_in_place(&mut mixed_array, &mut mixed_workspace)
        .unwrap();
    world.barrier();

    // Raw out-of-place and in-place calls also have distinct operation words.
    let (mut raw_array, mut raw_in_place_workspace) = fresh_in_place(&plan, 54.5);
    plan.forward_in_place(&mut raw_array, &mut raw_in_place_workspace)
        .unwrap();
    let mut raw_source = plan.allocate_output().unwrap();
    fill_spectrum(&mut raw_source, 54.5);
    let mut raw_destination = plan.allocate_input().unwrap();
    let mut raw_oop_workspace = plan.allocate_out_of_place_workspace().unwrap();
    let raw_array_before = format!("{raw_array:?}");
    let raw_in_place_workspace_before = format!("{raw_in_place_workspace:?}");
    let raw_source_before = raw_source.as_slice().to_vec();
    let raw_destination_before = raw_destination.as_slice().to_vec();
    let raw_oop_workspace_before = format!("{raw_oop_workspace:?}");
    let result = if rank == 0 {
        plan.backward(&raw_source, &mut raw_destination, &mut raw_oop_workspace)
    } else {
        plan.backward_in_place(&mut raw_array, &mut raw_in_place_workspace)
    };
    if size == 1 {
        assert!(result.is_ok());
    } else {
        assert!(matches!(
            result,
            Err(FftError::CollectiveDescriptorMismatch)
        ));
        assert_eq!(raw_source.as_slice(), raw_source_before.as_slice());
        assert_eq!(
            raw_destination.as_slice(),
            raw_destination_before.as_slice()
        );
        assert_eq!(format!("{raw_oop_workspace:?}"), raw_oop_workspace_before);
    }
    assert_eq!(format!("{raw_array:?}"), raw_array_before);
    assert_eq!(
        format!("{raw_in_place_workspace:?}"),
        raw_in_place_workspace_before
    );
    plan.backward_in_place(&mut raw_array, &mut raw_in_place_workspace)
        .unwrap();
    assert_eq!(raw_array.state(), C2cState::Input);
    world.barrier();

    // A constructor cannot pair with an in-place execution on another rank.
    let (mut constructor_array, mut constructor_workspace) = fresh_in_place(&plan, 55.0);
    let constructor_array_before = format!("{constructor_array:?}");
    let constructor_workspace_before = format!("{constructor_workspace:?}");
    let result = if rank == 0 {
        C2cPlan::<f64, 2, 1>::from_pencil_with_method(
            Arc::clone(plan.input_pencil()),
            ExtraShape::scalar(),
            method,
        )
        .map(|_| ())
    } else {
        plan.forward_in_place(&mut constructor_array, &mut constructor_workspace)
    };
    if size == 1 {
        assert!(result.is_ok());
    } else {
        assert!(result.is_err());
    }
    assert_eq!(format!("{constructor_array:?}"), constructor_array_before);
    assert_eq!(
        format!("{constructor_workspace:?}"),
        constructor_workspace_before
    );
    plan.forward_in_place(&mut constructor_array, &mut constructor_workspace)
        .unwrap();
    plan.inverse_in_place(&mut constructor_array, &mut constructor_workspace)
        .unwrap();
    world.barrier();

    // The array-level Alltoallv in-place protocol also has a distinct fixed
    // header and must not enter the FFT path.
    let transpose_plan = AllToAllvTransposePlan::new(
        Arc::clone(plan.input_pencil()),
        Arc::clone(plan.output_pencil()),
    )
    .unwrap();
    let transpose_requirements = transpose_plan
        .workspace_requirements(&ExtraShape::scalar())
        .unwrap();
    let mut transpose_array = ManyPencilArray::from_elem(
        vec![
            Arc::clone(plan.input_pencil()),
            Arc::clone(plan.output_pencil()),
        ],
        0,
        ExtraShape::scalar(),
        Complex::new(0.0_f64, 0.0),
    )
    .unwrap();
    let mut transpose_workspace = TransposeWorkspace::from_vecs(
        vec![Complex::new(0.0_f64, 0.0); transpose_requirements.send_len],
        vec![Complex::new(0.0_f64, 0.0); transpose_requirements.receive_len],
    );
    let (mut fft_array, mut fft_workspace) = fresh_in_place(&plan, 56.0);
    let transpose_array_before = format!("{transpose_array:?}");
    let transpose_workspace_before = format!("{transpose_workspace:?}");
    let fft_array_before = format!("{fft_array:?}");
    let fft_workspace_before = format!("{fft_workspace:?}");
    let result = if rank == 0 {
        transpose_plan
            .execute_in_place(&mut transpose_array, &mut transpose_workspace)
            .map_err(FftError::Transpose)
    } else {
        plan.forward_in_place(&mut fft_array, &mut fft_workspace)
    };
    if size == 1 {
        assert!(result.is_ok());
    } else {
        assert!(result.is_err());
        assert_eq!(format!("{transpose_array:?}"), transpose_array_before);
        assert_eq!(
            format!("{transpose_workspace:?}"),
            transpose_workspace_before
        );
    }
    assert_eq!(format!("{fft_array:?}"), fft_array_before);
    assert_eq!(format!("{fft_workspace:?}"), fft_workspace_before);
    plan.forward_in_place(&mut fft_array, &mut fft_workspace)
        .unwrap();
    plan.inverse_in_place(&mut fft_array, &mut fft_workspace)
        .unwrap();
    world.barrier();

    // Rank 5 is deliberately in a different next-stage subgroup from rank
    // zero. Full-Cartesian preflight must catch its foreign array before any
    // rank starts a transition.
    if size == 6 {
        let plan_2d = C2cPlan::<f64, 4, 2>::from_shape_with_method(
            Arc::clone(topology_2d),
            [3, 1, 3, 4],
            ExtraShape::scalar(),
            method,
        )
        .unwrap();
        let foreign_plan_2d = C2cPlan::<f64, 4, 2>::from_shape_with_method(
            Arc::clone(topology_2d),
            [3, 1, 3, 4],
            ExtraShape::scalar(),
            method,
        )
        .unwrap();
        let (mut array_2d, mut workspace_2d) = fresh_in_place(&plan_2d, 57.0);
        let mut bad_array_2d = foreign_plan_2d.allocate_in_place().unwrap();
        fill_in_place(&mut bad_array_2d, 57.0, false);
        let array_2d_before = format!("{array_2d:?}");
        let bad_array_2d_before = format!("{bad_array_2d:?}");
        let workspace_2d_before = format!("{workspace_2d:?}");
        let result = if rank == 5 {
            plan_2d.forward_in_place(&mut bad_array_2d, &mut workspace_2d)
        } else {
            plan_2d.forward_in_place(&mut array_2d, &mut workspace_2d)
        };
        assert!(result.is_err());
        assert_eq!(format!("{array_2d:?}"), array_2d_before);
        assert_eq!(format!("{bad_array_2d:?}"), bad_array_2d_before);
        assert_eq!(format!("{workspace_2d:?}"), workspace_2d_before);
        plan_2d
            .forward_in_place(&mut array_2d, &mut workspace_2d)
            .unwrap();
        plan_2d
            .inverse_in_place(&mut array_2d, &mut workspace_2d)
            .unwrap();
        world.barrier();
    }
}

fn assert_r2c_forward_rejected<R: TestReal, const N: usize, const M: usize, F, P>(
    source: &PencilArray<R, N, M>,
    destination: &mut PencilArray<Complex<R>, N, M>,
    workspace: &mut R2cWorkspace<R, N, M>,
    execute: F,
    predicate: P,
) where
    Complex<R>: mpi::datatype::Equivalence,
    F: FnOnce(
        &PencilArray<R, N, M>,
        &mut PencilArray<Complex<R>, N, M>,
        &mut R2cWorkspace<R, N, M>,
    ) -> Result<(), R2cError>,
    P: Fn(&R2cError) -> bool,
{
    let source_before = real_bits(source.as_slice());
    let destination_before = complex_bits(destination.as_slice());
    let workspace_before = format!("{workspace:?}");
    let error =
        execute(source, destination, workspace).expect_err("R2C call unexpectedly succeeded");
    assert!(predicate(&error), "unexpected R2C error: {error:?}");
    assert_eq!(real_bits(source.as_slice()), source_before);
    assert_eq!(complex_bits(destination.as_slice()), destination_before);
    assert_eq!(format!("{workspace:?}"), workspace_before);
}

fn assert_r2c_inverse_rejected<R: TestReal, const N: usize, const M: usize, F, P>(
    source: &PencilArray<Complex<R>, N, M>,
    destination: &mut PencilArray<R, N, M>,
    workspace: &mut R2cWorkspace<R, N, M>,
    execute: F,
    predicate: P,
) where
    Complex<R>: mpi::datatype::Equivalence,
    F: FnOnce(
        &PencilArray<Complex<R>, N, M>,
        &mut PencilArray<R, N, M>,
        &mut R2cWorkspace<R, N, M>,
    ) -> Result<(), R2cError>,
    P: Fn(&R2cError) -> bool,
{
    let source_before = complex_bits(source.as_slice());
    let destination_before = real_bits(destination.as_slice());
    let workspace_before = format!("{workspace:?}");
    let error =
        execute(source, destination, workspace).expect_err("R2C inverse unexpectedly succeeded");
    assert!(predicate(&error), "unexpected R2C inverse error: {error:?}");
    assert_eq!(complex_bits(source.as_slice()), source_before);
    assert_eq!(real_bits(destination.as_slice()), destination_before);
    assert_eq!(format!("{workspace:?}"), workspace_before);
}

fn r2c_collective_descriptor_error(error: &R2cError) -> bool {
    matches!(error, R2cError::Fft(FftError::CollectiveDescriptorMismatch))
}

fn r2c_precondition_or(error: &R2cError, local: fn(&FftError) -> bool, root: bool) -> bool {
    match error {
        R2cError::Fft(error) if root => local(error),
        R2cError::Fft(FftError::CollectivePreconditionFailed) if !root => true,
        _ => false,
    }
}

fn r2c_collective_reuse<R: TestReal, const N: usize, const M: usize>(plan: &R2cPlan<R, N, M>)
where
    Complex<R>: mpi::datatype::Equivalence,
{
    let mut workspace = plan.allocate_workspace().unwrap();
    reuse_r2c_after_boundary(plan, &mut workspace);
}

fn negative_r2c_collective_cases(
    world: &mpi::topology::SimpleCommunicator,
    topology_1d: &Arc<MpiTopology<1>>,
    topology_2d: &Arc<MpiTopology<2>>,
) {
    negative_r2c_header_cases(world, topology_1d);
    negative_r2c_shape_and_type_cases(world, topology_1d);
    negative_r2c_layout_cases(world, topology_1d);
    if world.size() == 6 {
        negative_r2c_2d_preflight_case(world, topology_2d);
    }
}

fn negative_r2c_header_cases(
    world: &mpi::topology::SimpleCommunicator,
    topology: &Arc<MpiTopology<1>>,
) {
    let rank = world.rank();
    // Every valid plan is constructed on every rank before any deliberately
    // mixed call below.
    let r2c_all = R2cPlan::<f64, 2, 1>::from_shape_with_method(
        Arc::clone(topology),
        [3, 4],
        ExtraShape::scalar(),
        TransposeMethod::AllToAllv,
    )
    .unwrap();
    let r2c_p2p = R2cPlan::<f64, 2, 1>::from_shape_with_method(
        Arc::clone(topology),
        [3, 4],
        ExtraShape::scalar(),
        TransposeMethod::PointToPoint,
    )
    .unwrap();
    let even = R2cPlan::<f64, 2, 1>::from_shape(Arc::clone(topology), [3, 2], ExtraShape::scalar())
        .unwrap();
    let odd = R2cPlan::<f64, 2, 1>::from_shape(Arc::clone(topology), [3, 3], ExtraShape::scalar())
        .unwrap();
    let n3 =
        R2cPlan::<f64, 3, 1>::from_shape(Arc::clone(topology), [3, 1, 4], ExtraShape::scalar())
            .unwrap();
    let f32_plan =
        R2cPlan::<f32, 2, 1>::from_shape(Arc::clone(topology), [3, 4], ExtraShape::scalar())
            .unwrap();
    let c2c = C2cPlan::<f64, 2, 1>::from_shape(Arc::clone(topology), [3, 4], ExtraShape::scalar())
        .unwrap();

    if world.size() > 1 {
        let constructor_mismatch = if rank == 0 {
            C2cPlan::<f64, 2, 1>::from_shape(Arc::clone(topology), [3, 4], ExtraShape::scalar())
                .map(|_| ())
                .map_err(|error| matches!(error, FftError::CollectiveDescriptorMismatch))
        } else {
            R2cPlan::<f64, 2, 1>::from_shape(Arc::clone(topology), [3, 4], ExtraShape::scalar())
                .map(|_| ())
                .map_err(|error| r2c_collective_descriptor_error(&error))
        };
        assert!(matches!(constructor_mismatch, Err(true)));
        r2c_collective_reuse(&r2c_all);
        world.barrier();

        let mut constructor_source = r2c_all.allocate_input().unwrap();
        fill_r2c_input(&mut constructor_source, 81.0);
        let mut constructor_destination = r2c_all.allocate_output().unwrap();
        let mut constructor_workspace = r2c_all.allocate_workspace().unwrap();
        let constructor_source_before = real_bits(constructor_source.as_slice());
        let constructor_destination_before = complex_bits(constructor_destination.as_slice());
        let constructor_workspace_before = format!("{constructor_workspace:?}");
        let constructor_execution = if rank == 0 {
            R2cPlan::<f64, 2, 1>::from_shape(Arc::clone(topology), [3, 4], ExtraShape::scalar())
                .map(|_| ())
                .map_err(|error| r2c_collective_descriptor_error(&error))
        } else {
            r2c_all
                .forward(
                    &constructor_source,
                    &mut constructor_destination,
                    &mut constructor_workspace,
                )
                .map(|_| ())
                .map_err(|error| r2c_collective_descriptor_error(&error))
        };
        assert!(matches!(constructor_execution, Err(true)));
        assert_eq!(
            real_bits(constructor_source.as_slice()),
            constructor_source_before
        );
        assert_eq!(
            complex_bits(constructor_destination.as_slice()),
            constructor_destination_before
        );
        assert_eq!(
            format!("{constructor_workspace:?}"),
            constructor_workspace_before
        );
        r2c_collective_reuse(&r2c_all);
        world.barrier();

        let mut c2c_source = c2c.allocate_input().unwrap();
        fill_input(&mut c2c_source, 82.0);
        let mut c2c_destination = c2c.allocate_output().unwrap();
        let mut c2c_workspace = c2c.allocate_out_of_place_workspace().unwrap();
        let mut r2c_source = r2c_all.allocate_input().unwrap();
        fill_r2c_input(&mut r2c_source, 82.0);
        let mut r2c_destination = r2c_all.allocate_output().unwrap();
        let mut r2c_workspace = r2c_all.allocate_workspace().unwrap();
        let c2c_source_before = complex_bits(c2c_source.as_slice());
        let c2c_destination_before = complex_bits(c2c_destination.as_slice());
        let c2c_workspace_before = format!("{c2c_workspace:?}");
        let r2c_source_before = real_bits(r2c_source.as_slice());
        let r2c_destination_before = complex_bits(r2c_destination.as_slice());
        let r2c_workspace_before = format!("{r2c_workspace:?}");
        let execution_mismatch = if rank == 0 {
            c2c.forward(&c2c_source, &mut c2c_destination, &mut c2c_workspace)
                .map(|_| ())
                .map_err(|error| matches!(error, FftError::CollectiveDescriptorMismatch))
        } else {
            r2c_all
                .forward(&r2c_source, &mut r2c_destination, &mut r2c_workspace)
                .map(|_| ())
                .map_err(|error| r2c_collective_descriptor_error(&error))
        };
        assert!(matches!(execution_mismatch, Err(true)));
        assert_eq!(complex_bits(c2c_source.as_slice()), c2c_source_before);
        assert_eq!(
            complex_bits(c2c_destination.as_slice()),
            c2c_destination_before
        );
        assert_eq!(format!("{c2c_workspace:?}"), c2c_workspace_before);
        assert_eq!(real_bits(r2c_source.as_slice()), r2c_source_before);
        assert_eq!(
            complex_bits(r2c_destination.as_slice()),
            r2c_destination_before
        );
        assert_eq!(format!("{r2c_workspace:?}"), r2c_workspace_before);
        c2c.forward(&c2c_source, &mut c2c_destination, &mut c2c_workspace)
            .unwrap();
        r2c_all
            .forward(&r2c_source, &mut r2c_destination, &mut r2c_workspace)
            .unwrap();
        world.barrier();

        let mut forward_source = r2c_all.allocate_input().unwrap();
        fill_r2c_input(&mut forward_source, 83.0);
        let mut forward_destination = r2c_all.allocate_output().unwrap();
        let mut forward_workspace = r2c_all.allocate_workspace().unwrap();
        let mut inverse_source = r2c_all.allocate_output().unwrap();
        fill_r2c_spectrum(&mut inverse_source, [3, 4], 84.0);
        let mut inverse_destination = r2c_all.allocate_input().unwrap();
        let mut inverse_workspace = r2c_all.allocate_workspace().unwrap();
        let forward_source_before = real_bits(forward_source.as_slice());
        let forward_destination_before = complex_bits(forward_destination.as_slice());
        let forward_workspace_before = format!("{forward_workspace:?}");
        let inverse_source_before = complex_bits(inverse_source.as_slice());
        let inverse_destination_before = real_bits(inverse_destination.as_slice());
        let inverse_workspace_before = format!("{inverse_workspace:?}");
        let direction_mismatch = if rank == 0 {
            r2c_all
                .forward(
                    &forward_source,
                    &mut forward_destination,
                    &mut forward_workspace,
                )
                .map(|_| ())
                .map_err(|error| r2c_collective_descriptor_error(&error))
        } else {
            r2c_all
                .inverse(
                    &inverse_source,
                    &mut inverse_destination,
                    &mut inverse_workspace,
                )
                .map(|_| ())
                .map_err(|error| r2c_collective_descriptor_error(&error))
        };
        assert!(matches!(direction_mismatch, Err(true)));
        assert_eq!(real_bits(forward_source.as_slice()), forward_source_before);
        assert_eq!(
            complex_bits(forward_destination.as_slice()),
            forward_destination_before
        );
        assert_eq!(format!("{forward_workspace:?}"), forward_workspace_before);
        assert_eq!(
            complex_bits(inverse_source.as_slice()),
            inverse_source_before
        );
        assert_eq!(
            real_bits(inverse_destination.as_slice()),
            inverse_destination_before
        );
        assert_eq!(format!("{inverse_workspace:?}"), inverse_workspace_before);
        r2c_all
            .forward(
                &forward_source,
                &mut forward_destination,
                &mut forward_workspace,
            )
            .unwrap();
        r2c_all
            .inverse(
                &forward_destination,
                &mut inverse_destination,
                &mut inverse_workspace,
            )
            .unwrap();
        world.barrier();

        let mut raw_source = r2c_all.allocate_output().unwrap();
        fill_r2c_spectrum(&mut raw_source, [3, 4], 84.5);
        let raw_source_before = complex_bits(raw_source.as_slice());
        let mut raw_inverse_destination = r2c_all.allocate_input().unwrap();
        dirty_real(&mut raw_inverse_destination);
        let raw_inverse_destination_before = real_bits(raw_inverse_destination.as_slice());
        let mut raw_inverse_workspace = r2c_all.allocate_workspace().unwrap();
        let raw_inverse_workspace_before = format!("{raw_inverse_workspace:?}");
        let mut raw_backward_destination = r2c_all.allocate_input().unwrap();
        dirty_real(&mut raw_backward_destination);
        let raw_backward_destination_before = real_bits(raw_backward_destination.as_slice());
        let mut raw_backward_workspace = r2c_all.allocate_workspace().unwrap();
        let raw_backward_workspace_before = format!("{raw_backward_workspace:?}");
        let reverse_direction_mismatch = if rank == 0 {
            r2c_all
                .inverse(
                    &raw_source,
                    &mut raw_inverse_destination,
                    &mut raw_inverse_workspace,
                )
                .map(|_| ())
                .map_err(|error| r2c_collective_descriptor_error(&error))
        } else {
            r2c_all
                .backward(
                    &raw_source,
                    &mut raw_backward_destination,
                    &mut raw_backward_workspace,
                )
                .map(|_| ())
                .map_err(|error| r2c_collective_descriptor_error(&error))
        };
        assert!(matches!(reverse_direction_mismatch, Err(true)));
        assert_eq!(complex_bits(raw_source.as_slice()), raw_source_before);
        assert_eq!(
            real_bits(raw_inverse_destination.as_slice()),
            raw_inverse_destination_before
        );
        assert_eq!(
            format!("{raw_inverse_workspace:?}"),
            raw_inverse_workspace_before
        );
        assert_eq!(
            real_bits(raw_backward_destination.as_slice()),
            raw_backward_destination_before
        );
        assert_eq!(
            format!("{raw_backward_workspace:?}"),
            raw_backward_workspace_before
        );
        r2c_all
            .inverse(
                &raw_source,
                &mut raw_inverse_destination,
                &mut raw_inverse_workspace,
            )
            .unwrap();
        r2c_all
            .backward(
                &raw_source,
                &mut raw_backward_destination,
                &mut raw_backward_workspace,
            )
            .unwrap();
        world.barrier();

        let mut all_source = r2c_all.allocate_input().unwrap();
        fill_r2c_input(&mut all_source, 85.0);
        let mut all_destination = r2c_all.allocate_output().unwrap();
        let mut all_workspace = r2c_all.allocate_workspace().unwrap();
        let mut p2p_source = r2c_p2p.allocate_input().unwrap();
        fill_r2c_input(&mut p2p_source, 85.0);
        let mut p2p_destination = r2c_p2p.allocate_output().unwrap();
        let mut p2p_workspace = r2c_p2p.allocate_workspace().unwrap();
        mixed_r2c_method_forward(
            world,
            &r2c_all,
            &r2c_p2p,
            &mut all_source,
            &mut all_destination,
            &mut all_workspace,
            &mut p2p_source,
            &mut p2p_destination,
            &mut p2p_workspace,
            true,
        );
        mixed_r2c_method_forward(
            world,
            &r2c_all,
            &r2c_p2p,
            &mut all_source,
            &mut all_destination,
            &mut all_workspace,
            &mut p2p_source,
            &mut p2p_destination,
            &mut p2p_workspace,
            false,
        );
    }
    let _ = (even, odd, n3, f32_plan);
}

#[allow(clippy::too_many_arguments)]
fn mixed_r2c_method_forward(
    world: &mpi::topology::SimpleCommunicator,
    alltoallv: &R2cPlan<f64, 2, 1>,
    point_to_point: &R2cPlan<f64, 2, 1>,
    all_source: &mut PencilArray<f64, 2, 1>,
    all_destination: &mut PencilArray<Complex<f64>, 2, 1>,
    all_workspace: &mut R2cWorkspace<f64, 2, 1>,
    p2p_source: &mut PencilArray<f64, 2, 1>,
    p2p_destination: &mut PencilArray<Complex<f64>, 2, 1>,
    p2p_workspace: &mut R2cWorkspace<f64, 2, 1>,
    rank_zero_uses_alltoallv: bool,
) {
    let rank = world.rank();
    let all_source_before = real_bits(all_source.as_slice());
    let all_destination_before = complex_bits(all_destination.as_slice());
    let all_workspace_before = format!("{all_workspace:?}");
    let p2p_source_before = real_bits(p2p_source.as_slice());
    let p2p_destination_before = complex_bits(p2p_destination.as_slice());
    let p2p_workspace_before = format!("{p2p_workspace:?}");
    let result = if (rank == 0) == rank_zero_uses_alltoallv {
        alltoallv.forward(all_source, all_destination, all_workspace)
    } else {
        point_to_point.forward(p2p_source, p2p_destination, p2p_workspace)
    };
    let error = result.expect_err("mixed R2C transports unexpectedly succeeded");
    assert!(r2c_collective_descriptor_error(&error));
    assert_eq!(real_bits(all_source.as_slice()), all_source_before);
    assert_eq!(
        complex_bits(all_destination.as_slice()),
        all_destination_before
    );
    assert_eq!(format!("{all_workspace:?}"), all_workspace_before);
    assert_eq!(real_bits(p2p_source.as_slice()), p2p_source_before);
    assert_eq!(
        complex_bits(p2p_destination.as_slice()),
        p2p_destination_before
    );
    assert_eq!(format!("{p2p_workspace:?}"), p2p_workspace_before);

    alltoallv
        .forward(all_source, all_destination, all_workspace)
        .unwrap();
    point_to_point
        .forward(p2p_source, p2p_destination, p2p_workspace)
        .unwrap();
    world.barrier();
}

fn negative_r2c_shape_and_type_cases(
    world: &mpi::topology::SimpleCommunicator,
    topology: &Arc<MpiTopology<1>>,
) {
    let rank = world.rank();
    let even = R2cPlan::<f64, 2, 1>::from_shape(Arc::clone(topology), [3, 2], ExtraShape::scalar())
        .unwrap();
    let odd = R2cPlan::<f64, 2, 1>::from_shape(Arc::clone(topology), [3, 3], ExtraShape::scalar())
        .unwrap();
    let n2 = R2cPlan::<f64, 2, 1>::from_shape(Arc::clone(topology), [3, 4], ExtraShape::scalar())
        .unwrap();
    let n3 =
        R2cPlan::<f64, 3, 1>::from_shape(Arc::clone(topology), [3, 1, 4], ExtraShape::scalar())
            .unwrap();
    let f32_plan =
        R2cPlan::<f32, 2, 1>::from_shape(Arc::clone(topology), [3, 4], ExtraShape::scalar())
            .unwrap();
    let f64_plan =
        R2cPlan::<f64, 2, 1>::from_shape(Arc::clone(topology), [3, 4], ExtraShape::scalar())
            .unwrap();
    assert_eq!(
        even.output_pencil().global_shape(),
        odd.output_pencil().global_shape()
    );

    if world.size() == 1 {
        r2c_collective_reuse(&even);
        r2c_collective_reuse(&odd);
        r2c_collective_reuse(&n2);
        r2c_collective_reuse(&n3);
        r2c_collective_reuse(&f32_plan);
        r2c_collective_reuse(&f64_plan);
        return;
    }

    let shape_ctor = if rank == 0 {
        R2cPlan::<f64, 2, 1>::from_shape(Arc::clone(topology), [3, 2], ExtraShape::scalar())
    } else {
        R2cPlan::<f64, 2, 1>::from_shape(Arc::clone(topology), [3, 3], ExtraShape::scalar())
    };
    let shape_error = shape_ctor.expect_err("even and odd constructor unexpectedly agreed");
    assert!(r2c_collective_descriptor_error(&shape_error));
    r2c_collective_reuse(&even);
    r2c_collective_reuse(&odd);
    world.barrier();

    let mut even_source = even.allocate_input().unwrap();
    fill_r2c_input(&mut even_source, 86.0);
    let mut even_destination = even.allocate_output().unwrap();
    let mut even_workspace = even.allocate_workspace().unwrap();
    let mut odd_source = odd.allocate_input().unwrap();
    fill_r2c_input(&mut odd_source, 86.0);
    let mut odd_destination = odd.allocate_output().unwrap();
    let mut odd_workspace = odd.allocate_workspace().unwrap();
    mixed_r2c_shape_forward(
        world,
        &even,
        &odd,
        &mut even_source,
        &mut even_destination,
        &mut even_workspace,
        &mut odd_source,
        &mut odd_destination,
        &mut odd_workspace,
        true,
    );
    mixed_r2c_shape_forward(
        world,
        &even,
        &odd,
        &mut even_source,
        &mut even_destination,
        &mut even_workspace,
        &mut odd_source,
        &mut odd_destination,
        &mut odd_workspace,
        false,
    );

    let n2_ctor = if rank == 0 {
        R2cPlan::<f64, 2, 1>::from_shape(Arc::clone(topology), [3, 4], ExtraShape::scalar())
            .map(|_| ())
            .map_err(|error| r2c_collective_descriptor_error(&error))
    } else {
        R2cPlan::<f64, 3, 1>::from_shape(Arc::clone(topology), [3, 1, 4], ExtraShape::scalar())
            .map(|_| ())
            .map_err(|error| r2c_collective_descriptor_error(&error))
    };
    assert!(matches!(n2_ctor, Err(true)));
    r2c_collective_reuse(&n2);
    r2c_collective_reuse(&n3);
    world.barrier();

    let mut n2_source = n2.allocate_input().unwrap();
    fill_r2c_input(&mut n2_source, 87.0);
    let mut n2_destination = n2.allocate_output().unwrap();
    let mut n2_workspace = n2.allocate_workspace().unwrap();
    let mut n3_source = n3.allocate_input().unwrap();
    fill_r2c_input(&mut n3_source, 87.0);
    let mut n3_destination = n3.allocate_output().unwrap();
    let mut n3_workspace = n3.allocate_workspace().unwrap();
    let n2_source_before = real_bits(n2_source.as_slice());
    let n2_destination_before = complex_bits(n2_destination.as_slice());
    let n2_workspace_before = format!("{n2_workspace:?}");
    let n3_source_before = real_bits(n3_source.as_slice());
    let n3_destination_before = complex_bits(n3_destination.as_slice());
    let n3_workspace_before = format!("{n3_workspace:?}");
    let n_execution = if rank == 0 {
        n2.forward(&n2_source, &mut n2_destination, &mut n2_workspace)
            .map(|_| ())
            .map_err(|error| r2c_collective_descriptor_error(&error))
    } else {
        n3.forward(&n3_source, &mut n3_destination, &mut n3_workspace)
            .map(|_| ())
            .map_err(|error| r2c_collective_descriptor_error(&error))
    };
    assert!(matches!(n_execution, Err(true)));
    assert_eq!(real_bits(n2_source.as_slice()), n2_source_before);
    assert_eq!(
        complex_bits(n2_destination.as_slice()),
        n2_destination_before
    );
    assert_eq!(format!("{n2_workspace:?}"), n2_workspace_before);
    assert_eq!(real_bits(n3_source.as_slice()), n3_source_before);
    assert_eq!(
        complex_bits(n3_destination.as_slice()),
        n3_destination_before
    );
    assert_eq!(format!("{n3_workspace:?}"), n3_workspace_before);
    n2.forward(&n2_source, &mut n2_destination, &mut n2_workspace)
        .unwrap();
    n3.forward(&n3_source, &mut n3_destination, &mut n3_workspace)
        .unwrap();
    world.barrier();

    let scalar_ctor = if rank == 0 {
        R2cPlan::<f32, 2, 1>::from_shape(Arc::clone(topology), [3, 4], ExtraShape::scalar())
            .map(|_| ())
            .map_err(|error| r2c_collective_descriptor_error(&error))
    } else {
        R2cPlan::<f64, 2, 1>::from_shape(Arc::clone(topology), [3, 4], ExtraShape::scalar())
            .map(|_| ())
            .map_err(|error| r2c_collective_descriptor_error(&error))
    };
    assert!(matches!(scalar_ctor, Err(true)));
    r2c_collective_reuse(&f32_plan);
    r2c_collective_reuse(&f64_plan);
    world.barrier();

    let mut f32_source = f32_plan.allocate_input().unwrap();
    fill_r2c_input(&mut f32_source, 88.0);
    let mut f32_destination = f32_plan.allocate_output().unwrap();
    let mut f32_workspace = f32_plan.allocate_workspace().unwrap();
    let mut f64_source = f64_plan.allocate_input().unwrap();
    fill_r2c_input(&mut f64_source, 88.0);
    let mut f64_destination = f64_plan.allocate_output().unwrap();
    let mut f64_workspace = f64_plan.allocate_workspace().unwrap();
    let f32_source_before = real_bits(f32_source.as_slice());
    let f32_destination_before = complex_bits(f32_destination.as_slice());
    let f32_workspace_before = format!("{f32_workspace:?}");
    let f64_source_before = real_bits(f64_source.as_slice());
    let f64_destination_before = complex_bits(f64_destination.as_slice());
    let f64_workspace_before = format!("{f64_workspace:?}");
    let scalar_execution = if rank == 0 {
        f32_plan
            .forward(&f32_source, &mut f32_destination, &mut f32_workspace)
            .map(|_| ())
            .map_err(|error| r2c_collective_descriptor_error(&error))
    } else {
        f64_plan
            .forward(&f64_source, &mut f64_destination, &mut f64_workspace)
            .map(|_| ())
            .map_err(|error| r2c_collective_descriptor_error(&error))
    };
    assert!(matches!(scalar_execution, Err(true)));
    assert_eq!(real_bits(f32_source.as_slice()), f32_source_before);
    assert_eq!(
        complex_bits(f32_destination.as_slice()),
        f32_destination_before
    );
    assert_eq!(format!("{f32_workspace:?}"), f32_workspace_before);
    assert_eq!(real_bits(f64_source.as_slice()), f64_source_before);
    assert_eq!(
        complex_bits(f64_destination.as_slice()),
        f64_destination_before
    );
    assert_eq!(format!("{f64_workspace:?}"), f64_workspace_before);
    f32_plan
        .forward(&f32_source, &mut f32_destination, &mut f32_workspace)
        .unwrap();
    f64_plan
        .forward(&f64_source, &mut f64_destination, &mut f64_workspace)
        .unwrap();
    world.barrier();
}

#[allow(clippy::too_many_arguments)]
fn mixed_r2c_shape_forward(
    world: &mpi::topology::SimpleCommunicator,
    even: &R2cPlan<f64, 2, 1>,
    odd: &R2cPlan<f64, 2, 1>,
    even_source: &mut PencilArray<f64, 2, 1>,
    even_destination: &mut PencilArray<Complex<f64>, 2, 1>,
    even_workspace: &mut R2cWorkspace<f64, 2, 1>,
    odd_source: &mut PencilArray<f64, 2, 1>,
    odd_destination: &mut PencilArray<Complex<f64>, 2, 1>,
    odd_workspace: &mut R2cWorkspace<f64, 2, 1>,
    rank_zero_uses_even: bool,
) {
    let rank = world.rank();
    let even_source_before = real_bits(even_source.as_slice());
    let even_destination_before = complex_bits(even_destination.as_slice());
    let even_workspace_before = format!("{even_workspace:?}");
    let odd_source_before = real_bits(odd_source.as_slice());
    let odd_destination_before = complex_bits(odd_destination.as_slice());
    let odd_workspace_before = format!("{odd_workspace:?}");
    let result = if (rank == 0) == rank_zero_uses_even {
        even.forward(even_source, even_destination, even_workspace)
    } else {
        odd.forward(odd_source, odd_destination, odd_workspace)
    };
    let error = result.expect_err("even/odd R2C execution unexpectedly succeeded");
    assert!(r2c_collective_descriptor_error(&error));
    assert_eq!(real_bits(even_source.as_slice()), even_source_before);
    assert_eq!(
        complex_bits(even_destination.as_slice()),
        even_destination_before
    );
    assert_eq!(format!("{even_workspace:?}"), even_workspace_before);
    assert_eq!(real_bits(odd_source.as_slice()), odd_source_before);
    assert_eq!(
        complex_bits(odd_destination.as_slice()),
        odd_destination_before
    );
    assert_eq!(format!("{odd_workspace:?}"), odd_workspace_before);
    even.forward(even_source, even_destination, even_workspace)
        .unwrap();
    odd.forward(odd_source, odd_destination, odd_workspace)
        .unwrap();
    world.barrier();
}

fn negative_r2c_layout_cases(
    world: &mpi::topology::SimpleCommunicator,
    topology: &Arc<MpiTopology<1>>,
) {
    let plan = R2cPlan::<f64, 2, 1>::from_shape(Arc::clone(topology), [3, 4], ExtraShape::scalar())
        .unwrap();
    let same_layout_plan =
        R2cPlan::<f64, 2, 1>::from_shape(Arc::clone(topology), [3, 4], ExtraShape::scalar())
            .unwrap();
    let world_size = usize::try_from(world.size()).unwrap();
    let foreign_topology = MpiTopology::<1>::new(world, [world_size]).unwrap();
    let foreign_plan = R2cPlan::<f64, 2, 1>::from_shape(
        Arc::clone(&foreign_topology),
        [3, 4],
        ExtraShape::scalar(),
    )
    .unwrap();
    let rank = world.rank();
    let root = rank == 0;

    let canonical = Arc::clone(plan.input_pencil());
    let noncanonical = canonical
        .with_permutation(AxisPermutation::new([1, 0]).unwrap())
        .unwrap();
    let constructor_result = if root {
        R2cPlan::<f64, 2, 1>::from_pencil(noncanonical, ExtraShape::scalar())
    } else {
        R2cPlan::<f64, 2, 1>::from_pencil(canonical, ExtraShape::scalar())
    };
    assert!(matches!(constructor_result, Err(R2cError::Fft(_))));
    r2c_collective_reuse(&plan);
    world.barrier();

    let mut valid_source = plan.allocate_input().unwrap();
    fill_r2c_input(&mut valid_source, 91.0);
    let mut valid_destination = plan.allocate_output().unwrap();
    let mut valid_workspace = plan.allocate_workspace().unwrap();
    let wrong_source = PencilArray::from_elem(
        Arc::clone(plan.output_pencil()),
        ExtraShape::scalar(),
        9.0_f64,
    )
    .unwrap();
    assert_r2c_forward_rejected(
        if root { &wrong_source } else { &valid_source },
        &mut valid_destination,
        &mut valid_workspace,
        |source, destination, workspace| plan.forward(source, destination, workspace),
        |error| {
            r2c_precondition_or(
                error,
                |error| matches!(error, FftError::InputLayoutMismatch),
                root,
            )
        },
    );
    plan.forward(&valid_source, &mut valid_destination, &mut valid_workspace)
        .unwrap();
    world.barrier();

    let mut wrong_destination = PencilArray::from_elem(
        Arc::clone(plan.input_pencil()),
        ExtraShape::scalar(),
        Complex::new(8.0_f64, -8.0),
    )
    .unwrap();
    let mut destination_workspace = plan.allocate_workspace().unwrap();
    assert_r2c_forward_rejected(
        &valid_source,
        if root {
            &mut wrong_destination
        } else {
            &mut valid_destination
        },
        &mut destination_workspace,
        |source, destination, workspace| plan.forward(source, destination, workspace),
        |error| {
            r2c_precondition_or(
                error,
                |error| matches!(error, FftError::OutputLayoutMismatch),
                root,
            )
        },
    );
    plan.forward(
        &valid_source,
        &mut valid_destination,
        &mut destination_workspace,
    )
    .unwrap();
    world.barrier();

    let wrong_extra = ExtraShape::new([2]).unwrap();
    let wrong_extra_source = PencilArray::from_elem(
        Arc::clone(plan.input_pencil()),
        wrong_extra.clone(),
        7.0_f64,
    )
    .unwrap();
    let mut extra_source_workspace = plan.allocate_workspace().unwrap();
    assert_r2c_forward_rejected(
        if root {
            &wrong_extra_source
        } else {
            &valid_source
        },
        &mut valid_destination,
        &mut extra_source_workspace,
        |source, destination, workspace| plan.forward(source, destination, workspace),
        |error| {
            r2c_precondition_or(
                error,
                |error| matches!(error, FftError::ExtraShapeMismatch),
                root,
            )
        },
    );
    plan.forward(
        &valid_source,
        &mut valid_destination,
        &mut extra_source_workspace,
    )
    .unwrap();
    world.barrier();

    let mut wrong_extra_destination = PencilArray::from_elem(
        Arc::clone(plan.output_pencil()),
        wrong_extra,
        Complex::new(6.0_f64, -6.0),
    )
    .unwrap();
    let mut extra_destination_workspace = plan.allocate_workspace().unwrap();
    assert_r2c_forward_rejected(
        &valid_source,
        if root {
            &mut wrong_extra_destination
        } else {
            &mut valid_destination
        },
        &mut extra_destination_workspace,
        |source, destination, workspace| plan.forward(source, destination, workspace),
        |error| {
            r2c_precondition_or(
                error,
                |error| matches!(error, FftError::ExtraShapeMismatch),
                root,
            )
        },
    );
    plan.forward(
        &valid_source,
        &mut valid_destination,
        &mut extra_destination_workspace,
    )
    .unwrap();
    world.barrier();

    let mut foreign_source = foreign_plan.allocate_input().unwrap();
    fill_r2c_input(&mut foreign_source, 92.0);
    let mut foreign_source_workspace = plan.allocate_workspace().unwrap();
    assert_r2c_forward_rejected(
        if root { &foreign_source } else { &valid_source },
        &mut valid_destination,
        &mut foreign_source_workspace,
        |source, destination, workspace| plan.forward(source, destination, workspace),
        |error| {
            r2c_precondition_or(
                error,
                |error| matches!(error, FftError::InputLayoutMismatch),
                root,
            )
        },
    );
    plan.forward(
        &valid_source,
        &mut valid_destination,
        &mut foreign_source_workspace,
    )
    .unwrap();
    world.barrier();

    let mut foreign_destination = foreign_plan.allocate_output().unwrap();
    let mut foreign_destination_workspace = plan.allocate_workspace().unwrap();
    assert_r2c_forward_rejected(
        &valid_source,
        if root {
            &mut foreign_destination
        } else {
            &mut valid_destination
        },
        &mut foreign_destination_workspace,
        |source, destination, workspace| plan.forward(source, destination, workspace),
        |error| {
            r2c_precondition_or(
                error,
                |error| matches!(error, FftError::OutputLayoutMismatch),
                root,
            )
        },
    );
    plan.forward(
        &valid_source,
        &mut valid_destination,
        &mut foreign_destination_workspace,
    )
    .unwrap();
    world.barrier();

    let mut foreign_workspace = same_layout_plan.allocate_workspace().unwrap();
    let mut workspace_source = same_layout_plan.allocate_input().unwrap();
    fill_r2c_input(&mut workspace_source, 93.0);
    let mut workspace_destination = same_layout_plan.allocate_output().unwrap();
    let foreign_workspace_before = format!("{foreign_workspace:?}");
    assert_r2c_forward_rejected(
        &valid_source,
        &mut valid_destination,
        if root {
            &mut foreign_workspace
        } else {
            &mut valid_workspace
        },
        |source, destination, workspace| plan.forward(source, destination, workspace),
        |error| {
            r2c_precondition_or(
                error,
                |error| matches!(error, FftError::WorkspaceMismatch),
                root,
            )
        },
    );
    assert_eq!(format!("{foreign_workspace:?}"), foreign_workspace_before);
    plan.forward(&valid_source, &mut valid_destination, &mut valid_workspace)
        .unwrap();
    same_layout_plan
        .forward(
            &workspace_source,
            &mut workspace_destination,
            &mut foreign_workspace,
        )
        .unwrap();
    world.barrier();

    let mut inverse_source = plan.allocate_output().unwrap();
    fill_r2c_spectrum(&mut inverse_source, [3, 4], 93.5);
    let mut inverse_destination = plan.allocate_input().unwrap();
    dirty_real(&mut inverse_destination);
    let wrong_inverse_source = PencilArray::from_elem(
        Arc::clone(plan.input_pencil()),
        ExtraShape::scalar(),
        Complex::new(3.0_f64, -3.0),
    )
    .unwrap();
    let mut inverse_workspace = plan.allocate_workspace().unwrap();
    assert_r2c_inverse_rejected(
        if root {
            &wrong_inverse_source
        } else {
            &inverse_source
        },
        &mut inverse_destination,
        &mut inverse_workspace,
        |source, destination, workspace| plan.inverse(source, destination, workspace),
        |error| {
            r2c_precondition_or(
                error,
                |error| matches!(error, FftError::InputLayoutMismatch),
                root,
            )
        },
    );
    plan.inverse(
        &inverse_source,
        &mut inverse_destination,
        &mut inverse_workspace,
    )
    .unwrap();
    world.barrier();

    let mut wrong_inverse_destination = PencilArray::from_elem(
        Arc::clone(plan.output_pencil()),
        ExtraShape::scalar(),
        2.0_f64,
    )
    .unwrap();
    let mut inverse_destination_workspace = plan.allocate_workspace().unwrap();
    assert_r2c_inverse_rejected(
        &inverse_source,
        if root {
            &mut wrong_inverse_destination
        } else {
            &mut inverse_destination
        },
        &mut inverse_destination_workspace,
        |source, destination, workspace| plan.inverse(source, destination, workspace),
        |error| {
            r2c_precondition_or(
                error,
                |error| matches!(error, FftError::OutputLayoutMismatch),
                root,
            )
        },
    );
    plan.inverse(
        &inverse_source,
        &mut inverse_destination,
        &mut inverse_destination_workspace,
    )
    .unwrap();
    world.barrier();

    let wrong_global_pencil = Pencil::<2, 1>::new(Arc::clone(topology), [2, 6], [0]).unwrap();
    let wrong_global_source =
        PencilArray::from_elem(wrong_global_pencil, ExtraShape::scalar(), 4.0_f64).unwrap();
    let mut global_workspace = plan.allocate_workspace().unwrap();
    assert_r2c_forward_rejected(
        if root {
            &wrong_global_source
        } else {
            &valid_source
        },
        &mut valid_destination,
        &mut global_workspace,
        |source, destination, workspace| plan.forward(source, destination, workspace),
        |error| {
            r2c_precondition_or(
                error,
                |error| matches!(error, FftError::InputLayoutMismatch),
                root,
            )
        },
    );
    plan.forward(&valid_source, &mut valid_destination, &mut global_workspace)
        .unwrap();
    world.barrier();
}

fn negative_r2c_2d_preflight_case(
    world: &mpi::topology::SimpleCommunicator,
    topology: &Arc<MpiTopology<2>>,
) {
    let plan =
        R2cPlan::<f64, 4, 2>::from_shape(Arc::clone(topology), [2, 1, 2, 4], ExtraShape::scalar())
            .unwrap();
    let rank = world.rank();
    let bad_rank = rank == 5;
    let constructor_result = if bad_rank {
        R2cPlan::<f64, 4, 2>::from_pencil(Arc::clone(plan.output_pencil()), ExtraShape::scalar())
    } else {
        R2cPlan::<f64, 4, 2>::from_pencil(Arc::clone(plan.input_pencil()), ExtraShape::scalar())
    };
    assert!(matches!(constructor_result, Err(R2cError::Fft(_))));
    r2c_collective_reuse(&plan);
    world.barrier();

    let mut source = plan.allocate_input().unwrap();
    fill_r2c_input(&mut source, 94.0);
    let mut destination = plan.allocate_output().unwrap();
    let mut workspace = plan.allocate_workspace().unwrap();
    let wrong_source = PencilArray::from_elem(
        Arc::clone(plan.output_pencil()),
        ExtraShape::scalar(),
        5.0_f64,
    )
    .unwrap();
    assert_r2c_forward_rejected(
        if bad_rank { &wrong_source } else { &source },
        &mut destination,
        &mut workspace,
        |source, destination, workspace| plan.forward(source, destination, workspace),
        |error| {
            r2c_precondition_or(
                error,
                |error| matches!(error, FftError::InputLayoutMismatch),
                bad_rank,
            )
        },
    );
    plan.forward(&source, &mut destination, &mut workspace)
        .unwrap();
    world.barrier();
}
