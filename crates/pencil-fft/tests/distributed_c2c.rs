use std::{f64::consts::TAU, sync::Arc};

use mpi::{
    collective::SystemOperation,
    topology::{Color, Communicator, Key},
    traits::CommunicatorCollectives,
};
use pencil_array::{
    AllToAllvTransposePlan, AxisPermutation, ExtraShape, ManyPencilArray, MpiTopology, Pencil,
    PencilArray, PencilArrayView, PointToPointTransposePlan, TransposeWorkspace,
};
use pencil_fft::{
    C2cInPlaceArray, C2cInPlaceWorkspace, C2cOutOfPlaceWorkspace, C2cPlan, C2cState, Complex,
    FftError, FftReal,
};

trait TestReal: FftReal + mpi::datatype::Equivalence {
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
        2e-10
    }
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

    negative_collective_cases(&world, &topology_1d, &topology_2d);
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
    let plan =
        C2cPlan::<R, N, M>::from_shape(Arc::clone(topology), global_shape, extra_shape.clone())
            .unwrap();
    let mut source = plan.allocate_input().unwrap();
    fill_input(&mut source, seed);

    if constructor_checks {
        let from_pencil =
            C2cPlan::<R, N, M>::from_pencil(Arc::clone(plan.input_pencil()), extra_shape.clone())
                .unwrap();
        let from_array = C2cPlan::<R, N, M>::from_array(&source).unwrap();
        assert!(
            from_pencil
                .output_pencil()
                .same_layout(from_array.output_pencil().as_ref())
        );
        assert_eq!(from_array.extra_shape(), &extra_shape);
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

    // Inverse is checked against an independently generated spectrum rather
    // than only against the forward output.
    let mut arbitrary_spectrum = plan.allocate_output().unwrap();
    fill_spectrum(&mut arbitrary_spectrum, seed + 31.0);
    let arbitrary_before = arbitrary_spectrum.as_slice().to_vec();
    let mut inverse = plan.allocate_input().unwrap();
    plan.inverse(&arbitrary_spectrum, &mut inverse, &mut workspace)
        .unwrap();
    assert_eq!(arbitrary_spectrum.as_slice(), arbitrary_before.as_slice());
    check_inverse(&inverse.view(), &extra_shape, global_shape, seed + 31.0);

    let mut roundtrip = plan.allocate_input().unwrap();
    plan.inverse(&spectrum, &mut roundtrip, &mut second_workspace)
        .unwrap();
    assert_eq!(spectrum.as_slice(), second_spectrum.as_slice());
    for (actual, expected) in roundtrip.as_slice().iter().zip(source_before.iter()) {
        assert_close(*actual, *expected);
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
    );

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
    topology_barrier(topology);
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
            let expected =
                dft_value::<R, N>(&extra_indices, wave_number, global_shape, seed, false);
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
) {
    let local_shape = array.pencil().local_shape_logical();
    let ranges = array.pencil().local_ranges();
    for extra_linear in 0..extra_shape.element_count() {
        let extra_indices = unravel_extra(extra_linear, extra_shape.dimensions());
        for spatial_linear in 0..array.pencil().local_len() {
            let local = unravel_spatial(spatial_linear, local_shape);
            let spatial = std::array::from_fn(|axis| ranges[axis].start + local[axis]);
            let expected = dft_value::<R, N>(&extra_indices, spatial, global_shape, seed, true);
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
    if inverse {
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
    let foreign_source = PencilArray::from_elem(
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
    let mut source_extra_destination = batch_plan.allocate_output().unwrap();
    let mut batch_workspace = batch_plan.allocate_out_of_place_workspace().unwrap();
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

    negative_in_place_cases(world, topology_1d, topology_2d);
}

fn negative_in_place_cases(
    world: &mpi::topology::SimpleCommunicator,
    topology_1d: &Arc<MpiTopology<1>>,
    topology_2d: &Arc<MpiTopology<2>>,
) {
    let rank = world.rank();
    let size = world.size();
    let plan =
        C2cPlan::<f64, 2, 1>::from_shape(Arc::clone(topology_1d), [3, 4], ExtraShape::scalar())
            .unwrap();

    // The wrong direction is rejected collectively without changing the
    // initial input state.
    let (mut wrong_state, mut wrong_state_workspace) = fresh_in_place(&plan, 50.0);
    assert_in_place_rejected(
        &mut wrong_state,
        &mut wrong_state_workspace,
        |array, workspace| plan.inverse_in_place(array, workspace),
    );
    plan.forward_in_place(&mut wrong_state, &mut wrong_state_workspace)
        .unwrap();
    plan.inverse_in_place(&mut wrong_state, &mut wrong_state_workspace)
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
    let foreign_plan =
        C2cPlan::<f64, 2, 1>::from_shape(Arc::clone(topology_1d), [3, 4], ExtraShape::scalar())
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

    // A constructor cannot pair with an in-place execution on another rank.
    let (mut constructor_array, mut constructor_workspace) = fresh_in_place(&plan, 55.0);
    let constructor_array_before = format!("{constructor_array:?}");
    let constructor_workspace_before = format!("{constructor_workspace:?}");
    let result = if rank == 0 {
        C2cPlan::<f64, 2, 1>::from_pencil(Arc::clone(plan.input_pencil()), ExtraShape::scalar())
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
        let plan_2d = C2cPlan::<f64, 4, 2>::from_shape(
            Arc::clone(topology_2d),
            [3, 1, 3, 4],
            ExtraShape::scalar(),
        )
        .unwrap();
        let foreign_plan_2d = C2cPlan::<f64, 4, 2>::from_shape(
            Arc::clone(topology_2d),
            [3, 1, 3, 4],
            ExtraShape::scalar(),
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
