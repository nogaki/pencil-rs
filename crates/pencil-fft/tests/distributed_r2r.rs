use std::{fmt::Debug, sync::Arc};

use mpi::{
    collective::{CommunicatorCollectives, SystemOperation},
    datatype::Equivalence,
    topology::{Color, Key},
    traits::*,
};
use pencil_array::{ExtraShape, MpiTopology, Pencil, PencilArray, PencilArrayViewMut, SpatialAxis};
use pencil_fft::{
    Complex, FftError, R2rError, R2rKind, R2rPlan, R2rScalar, R2rState, TransposeMethod,
};

trait TestValue: R2rScalar + Equivalence + Copy + Debug + PartialEq {
    fn from_parts(real: f64, imaginary: f64) -> Self;
    fn parts(value: Self) -> Complex<f64>;
    fn tolerance() -> (f64, f64);
}

impl TestValue for f32 {
    fn from_parts(real: f64, _imaginary: f64) -> Self {
        real as f32
    }

    fn parts(value: Self) -> Complex<f64> {
        Complex::new(value as f64, 0.0)
    }

    fn tolerance() -> (f64, f64) {
        (3e-4, 3e-5)
    }
}

impl TestValue for f64 {
    fn from_parts(real: f64, _imaginary: f64) -> Self {
        real
    }

    fn parts(value: Self) -> Complex<f64> {
        Complex::new(value, 0.0)
    }

    fn tolerance() -> (f64, f64) {
        (2e-10, 2e-12)
    }
}

impl TestValue for Complex<f32> {
    fn from_parts(real: f64, imaginary: f64) -> Self {
        Complex::new(real as f32, imaginary as f32)
    }

    fn parts(value: Self) -> Complex<f64> {
        Complex::new(value.re as f64, value.im as f64)
    }

    fn tolerance() -> (f64, f64) {
        (4e-4, 4e-5)
    }
}

impl TestValue for Complex<f64> {
    fn from_parts(real: f64, imaginary: f64) -> Self {
        Complex::new(real, imaginary)
    }

    fn parts(value: Self) -> Complex<f64> {
        value
    }

    fn tolerance() -> (f64, f64) {
        (2e-10, 2e-12)
    }
}

#[derive(Debug, PartialEq)]
struct Snapshots<T>(Vec<T>, Vec<T>, Vec<T>, Vec<T>, Vec<T>, Vec<T>);

#[derive(Clone, Copy)]
enum Direction {
    Forward,
    Inverse,
    Backward,
}

fn paired_kind(kind: R2rKind) -> R2rKind {
    match kind {
        R2rKind::DctI => R2rKind::DctI,
        R2rKind::DctII => R2rKind::DctIII,
        R2rKind::DctIII => R2rKind::DctII,
        R2rKind::DctIV => R2rKind::DctIV,
        R2rKind::DstI => R2rKind::DstI,
        R2rKind::DstII => R2rKind::DstIII,
        R2rKind::DstIII => R2rKind::DstII,
        R2rKind::DstIV => R2rKind::DstIV,
    }
}

fn logical_scale(kind: R2rKind, length: usize) -> f64 {
    match kind {
        R2rKind::DctI => (2 * (length - 1)) as f64,
        R2rKind::DstI => (2 * (length + 1)) as f64,
        _ => (2 * length) as f64,
    }
}

fn coefficient(kind: R2rKind, source: usize, target: usize, length: usize) -> f64 {
    let pi = std::f64::consts::PI;
    match kind {
        R2rKind::DctI => {
            if source == 0 || source == length - 1 {
                if source == length - 1 && target % 2 == 1 {
                    -1.0
                } else {
                    1.0
                }
            } else {
                2.0 * (pi * source as f64 * target as f64 / (length - 1) as f64).cos()
            }
        }
        R2rKind::DctII => 2.0 * (pi * (source as f64 + 0.5) * target as f64 / length as f64).cos(),
        R2rKind::DctIII => {
            if source == 0 {
                1.0
            } else {
                2.0 * (pi * source as f64 * (target as f64 + 0.5) / length as f64).cos()
            }
        }
        R2rKind::DctIV => {
            2.0 * (pi * (source as f64 + 0.5) * (target as f64 + 0.5) / length as f64).cos()
        }
        R2rKind::DstI => {
            2.0 * (pi * (source as f64 + 1.0) * (target as f64 + 1.0) / (length + 1) as f64).sin()
        }
        R2rKind::DstII => {
            2.0 * (pi * (source as f64 + 0.5) * (target as f64 + 1.0) / length as f64).sin()
        }
        R2rKind::DstIII => {
            if source == length - 1 {
                if target % 2 == 0 { 1.0 } else { -1.0 }
            } else {
                2.0 * (pi * (source as f64 + 1.0) * (target as f64 + 0.5) / length as f64).sin()
            }
        }
        R2rKind::DstIV => {
            2.0 * (pi * (source as f64 + 0.5) * (target as f64 + 0.5) / length as f64).sin()
        }
    }
}

fn product(values: &[usize]) -> usize {
    values.iter().copied().product()
}

fn unravel(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    let mut result = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        result[axis] = linear % shape[axis];
        linear /= shape[axis];
    }
    result
}

fn row_offset(shape: &[usize], indices: &[usize]) -> usize {
    shape
        .iter()
        .zip(indices)
        .fold(0, |offset, (&extent, &index)| offset * extent + index)
}

fn value_tensor<T: TestValue, const N: usize>(
    shape: [usize; N],
    extra: &[usize],
    seed: f64,
) -> Vec<Complex<f64>> {
    let mut full_shape = extra.to_vec();
    full_shape.extend(shape);
    let mut values = Vec::with_capacity(product(&full_shape));
    for linear in 0..product(&full_shape) {
        let coordinates = unravel(linear, &full_shape);
        let split = extra.len();
        let extra_coordinates = &coordinates[..split];
        let spatial = &coordinates[split..];
        let mut real = 0.217 + seed * 0.013;
        let mut imaginary = -0.319 - seed * 0.017;
        for (axis, &coordinate) in extra_coordinates.iter().enumerate() {
            real += (axis + 2) as f64 * (coordinate + 1) as f64 * 0.071;
            imaginary -= (axis + 3) as f64 * (coordinate + 2) as f64 * 0.043;
        }
        for (axis, &coordinate) in spatial.iter().enumerate() {
            real += (axis + 1) as f64 * (coordinate + 1) as f64 * 0.173;
            real += ((coordinate + axis + 2) as f64 * 0.31 + seed).sin() * 0.029;
            imaginary += (axis + 2) as f64 * (coordinate + 2) as f64 * 0.119;
            imaginary += ((coordinate + axis + 3) as f64 * 0.27 - seed).cos() * 0.023;
        }
        for left in 0..spatial.len() {
            for right in left + 1..spatial.len() {
                real += 0.0091
                    * (left + 1) as f64
                    * (right + 2) as f64
                    * (spatial[left] + 1) as f64
                    * (spatial[right] + 2) as f64;
                imaginary -= 0.0063
                    * (left + 2) as f64
                    * (right + 1) as f64
                    * (spatial[left] + 2) as f64
                    * (spatial[right] + 1) as f64;
            }
        }
        values.push(T::parts(T::from_parts(real, imaginary)));
    }
    values
}

fn tensor_oracle<const N: usize>(
    input: &[Complex<f64>],
    shape: [usize; N],
    extra: &[usize],
    kinds: [Option<R2rKind>; N],
    reverse: bool,
    normalize: bool,
) -> Vec<Complex<f64>> {
    let spatial_count = product(&shape);
    let mut full_shape = extra.to_vec();
    full_shape.extend(shape);
    let total = product(&full_shape);
    let mut result = vec![Complex::new(0.0, 0.0); total];
    let mut factor = 1.0;
    if normalize {
        for (axis, kind) in kinds.iter().enumerate() {
            if let Some(kind) = kind {
                factor *= logical_scale(*kind, shape[axis]);
            }
        }
    }

    for (target_linear, target_value) in result.iter_mut().enumerate() {
        let target_coordinates = unravel(target_linear, &full_shape);
        let extra_coordinates = &target_coordinates[..extra.len()];
        let target = &target_coordinates[extra.len()..];
        let mut sum = Complex::new(0.0, 0.0);
        for source_linear in 0..spatial_count {
            let source = unravel(source_linear, &shape);
            let mut coefficient_product = 1.0;
            for (axis, kind) in kinds.iter().enumerate() {
                match kind {
                    Some(kind) => {
                        let kind = if reverse { paired_kind(*kind) } else { *kind };
                        coefficient_product *=
                            coefficient(kind, source[axis], target[axis], shape[axis]);
                    }
                    None if source[axis] != target[axis] => {
                        coefficient_product = 0.0;
                        break;
                    }
                    None => {}
                }
            }
            if coefficient_product != 0.0 {
                let mut index = extra_coordinates.to_vec();
                index.extend(&source);
                let source_offset = row_offset(&full_shape, &index);
                sum.re += input[source_offset].re * coefficient_product;
                sum.im += input[source_offset].im * coefficient_product;
            }
        }
        *target_value = Complex::new(sum.re / factor, sum.im / factor);
    }
    result
}

fn raw_offsets<const N: usize, const M: usize>(
    pencil: &Pencil<N, M>,
    extra: &ExtraShape,
    length: usize,
) -> Vec<usize> {
    let permutation = pencil.permutation().axes().map(SpatialAxis::index);
    let local_shape: [usize; N] =
        std::array::from_fn(|memory_axis| pencil.local_ranges()[permutation[memory_axis]].len());
    let mut physical_shape = extra.dimensions().to_vec();
    physical_shape.extend(local_shape);
    let mut global_shape = extra.dimensions().to_vec();
    global_shape.extend(*pencil.global_shape());
    let mut offsets = Vec::with_capacity(length);
    for linear in 0..length {
        let physical = unravel(linear, &physical_shape);
        let mut global = physical[..extra.dimensions().len()].to_vec();
        let mut spatial = [0usize; N];
        for (memory_axis, &logical_axis) in permutation.iter().enumerate() {
            spatial[logical_axis] = pencil.local_ranges()[logical_axis].start
                + physical[extra.dimensions().len() + memory_axis];
        }
        global.extend(spatial);
        offsets.push(row_offset(&global_shape, &global));
    }
    offsets
}

fn fill_slice<T: TestValue, const N: usize, const M: usize>(
    storage: &mut [T],
    pencil: &Pencil<N, M>,
    extra: &ExtraShape,
    values: &[Complex<f64>],
) {
    let offsets = raw_offsets(pencil, extra, storage.len());
    for (slot, offset) in storage.iter_mut().zip(offsets) {
        *slot = T::from_parts(values[offset].re, values[offset].im);
    }
}

fn fill_view<T: TestValue, const N: usize, const M: usize>(
    view: &mut PencilArrayViewMut<'_, T, N, M>,
    values: &[Complex<f64>],
) {
    let offsets = raw_offsets(view.pencil(), view.extra_shape(), view.len());
    for (slot, offset) in view.as_mut_slice().iter_mut().zip(offsets) {
        *slot = T::from_parts(values[offset].re, values[offset].im);
    }
}

fn assert_close_component(actual: f64, expected: f64, tolerance: (f64, f64), label: &str) {
    let bound = tolerance.0 + tolerance.1 * actual.abs().max(expected.abs());
    assert!(
        (actual - expected).abs() <= bound,
        "{label}: actual={actual:.17e} expected={expected:.17e} bound={bound:.3e}"
    );
}

fn check_slice<T: TestValue, const N: usize, const M: usize>(
    storage: &[T],
    pencil: &Pencil<N, M>,
    extra: &ExtraShape,
    values: &[Complex<f64>],
    label: &str,
) {
    let offsets = raw_offsets(pencil, extra, storage.len());
    for (index, (value, offset)) in storage.iter().zip(offsets).enumerate() {
        let actual = T::parts(*value);
        let expected = values[offset];
        assert_close_component(
            actual.re,
            expected.re,
            T::tolerance(),
            &format!("{label} slot={index} real"),
        );
        assert_close_component(
            actual.im,
            expected.im,
            T::tolerance(),
            &format!("{label} slot={index} imaginary"),
        );
    }
}

fn assert_ownership<const N: usize, const M: usize>(
    pencil: &Pencil<N, M>,
    extra: &ExtraShape,
    local_len: usize,
    label: &str,
) {
    let mut global_shape = extra.dimensions().to_vec();
    global_shape.extend(*pencil.global_shape());
    let count = product(&global_shape);
    if count == 0 {
        return;
    }
    let offsets = raw_offsets(pencil, extra, local_len);
    let mut local = vec![0i32; count];
    for offset in offsets {
        local[offset] += 1;
    }
    let mut global = vec![0i32; count];
    pencil
        .topology()
        .communicator()
        .all_reduce_into(&local, &mut global, SystemOperation::sum());
    assert!(
        global.iter().all(|&count| count == 1),
        "{label}: {global:?}"
    );
}

fn assert_endpoint<const N: usize, const M: usize>(
    pencil: &Pencil<N, M>,
    shape: [usize; N],
    extra: &ExtraShape,
    output: bool,
    label: &str,
) {
    assert_eq!(*pencil.global_shape(), shape, "{label}: global shape");
    let expected_permutation = if output {
        std::array::from_fn(|axis| N - axis - 1)
    } else {
        std::array::from_fn(|axis| axis)
    };
    assert_eq!(
        pencil.permutation().axes().map(SpatialAxis::index),
        expected_permutation,
        "{label}: permutation"
    );
    let expected_decomposition = if output {
        std::array::from_fn(|axis| axis + 1)
    } else {
        std::array::from_fn(|axis| axis)
    };
    assert_eq!(
        pencil.decomposition().map(SpatialAxis::index),
        expected_decomposition,
        "{label}: decomposition"
    );
    assert!(
        !extra.dimensions().contains(&usize::MAX),
        "{label}: extra shape"
    );
}

fn run_case<T: TestValue, const N: usize, const M: usize>(
    topology: &Arc<MpiTopology<M>>,
    shape: [usize; N],
    extra: ExtraShape,
    kinds: [Option<R2rKind>; N],
    method: TransposeMethod,
    seed: f64,
    check_constructors: bool,
) -> Snapshots<T> {
    let plan = R2rPlan::<T, N, M>::from_shape_with_method(
        Arc::clone(topology),
        shape,
        extra.clone(),
        kinds,
        method,
    )
    .unwrap();
    assert_eq!(plan.kinds(), kinds);
    assert_eq!(plan.extra_shape(), &extra);
    assert_endpoint(plan.input_pencil(), shape, &extra, false, "R2R input");
    assert_endpoint(plan.output_pencil(), shape, &extra, true, "R2R output");

    let input_values = value_tensor::<T, N>(shape, extra.dimensions(), seed);
    let reverse_values = value_tensor::<T, N>(shape, extra.dimensions(), seed + 31.75);
    let forward_values = tensor_oracle(
        &input_values,
        shape,
        extra.dimensions(),
        kinds,
        false,
        false,
    );
    let inverse_values = tensor_oracle(
        &reverse_values,
        shape,
        extra.dimensions(),
        kinds,
        true,
        true,
    );
    let backward_values = tensor_oracle(
        &reverse_values,
        shape,
        extra.dimensions(),
        kinds,
        true,
        false,
    );

    let mut source = plan.allocate_input().unwrap();
    let source_pencil = source.pencil().clone();
    let source_extra = source.extra_shape().clone();
    fill_slice(
        source.as_mut_slice(),
        source_pencil.as_ref(),
        &source_extra,
        &input_values,
    );
    assert_ownership(
        source.pencil(),
        source.extra_shape(),
        source.len(),
        "R2R input ownership",
    );
    let source_before = source.as_slice().to_vec();
    let mut output = plan.allocate_output().unwrap();
    assert_ownership(
        output.pencil(),
        output.extra_shape(),
        output.len(),
        "R2R output ownership",
    );
    let mut workspace = plan.allocate_workspace().unwrap();
    if check_constructors {
        let from_pencil = R2rPlan::<T, N, M>::from_pencil_with_method(
            Arc::clone(plan.input_pencil()),
            extra.clone(),
            kinds,
            method,
        )
        .unwrap();
        let from_array =
            R2rPlan::<T, N, M>::from_array_with_method(&source, kinds, method).unwrap();
        assert!(
            from_pencil
                .output_pencil()
                .same_layout(plan.output_pencil().as_ref())
        );
        assert!(
            from_array
                .output_pencil()
                .same_layout(plan.output_pencil().as_ref())
        );
        assert_eq!(from_pencil.kinds(), kinds);
        assert_eq!(from_array.extra_shape(), &extra);
    }

    plan.forward(&source, &mut output, &mut workspace).unwrap();
    assert_eq!(source.as_slice(), source_before.as_slice());
    check_slice(
        output.as_slice(),
        output.pencil(),
        output.extra_shape(),
        &forward_values,
        "R2R OOP forward",
    );
    let forward_snapshot = output.as_slice().to_vec();

    let mut reverse_source = plan.allocate_output().unwrap();
    let reverse_pencil = reverse_source.pencil().clone();
    let reverse_extra = reverse_source.extra_shape().clone();
    fill_slice(
        reverse_source.as_mut_slice(),
        reverse_pencil.as_ref(),
        &reverse_extra,
        &reverse_values,
    );
    let reverse_before = reverse_source.as_slice().to_vec();
    let mut inverse = plan.allocate_input().unwrap();
    plan.inverse(&reverse_source, &mut inverse, &mut workspace)
        .unwrap();
    assert_eq!(reverse_source.as_slice(), reverse_before.as_slice());
    check_slice(
        inverse.as_slice(),
        inverse.pencil(),
        inverse.extra_shape(),
        &inverse_values,
        "R2R OOP inverse",
    );
    let inverse_snapshot = inverse.as_slice().to_vec();

    let mut backward = plan.allocate_input().unwrap();
    plan.backward(&reverse_source, &mut backward, &mut workspace)
        .unwrap();
    assert_eq!(reverse_source.as_slice(), reverse_before.as_slice());
    check_slice(
        backward.as_slice(),
        backward.pencil(),
        backward.extra_shape(),
        &backward_values,
        "R2R OOP backward",
    );
    let backward_snapshot = backward.as_slice().to_vec();

    let mut inplace = plan.allocate_in_place().unwrap();
    {
        let mut view = inplace.view_mut().unwrap();
        fill_view(&mut view, &input_values);
    }
    let pointer = inplace.view().unwrap().as_slice().as_ptr();
    let mut inplace_workspace = plan.allocate_in_place_workspace().unwrap();
    plan.forward_in_place(&mut inplace, &mut inplace_workspace)
        .unwrap();
    assert_eq!(inplace.state(), R2rState::Output);
    assert_eq!(inplace.view().unwrap().as_slice().as_ptr(), pointer);
    assert_endpoint(
        inplace.view().unwrap().pencil(),
        shape,
        &extra,
        true,
        "R2R in-place forward",
    );
    check_slice(
        inplace.view().unwrap().as_slice(),
        inplace.view().unwrap().pencil(),
        inplace.view().unwrap().extra_shape(),
        &forward_values,
        "R2R in-place forward",
    );
    let inplace_forward = inplace.view().unwrap().as_slice().to_vec();

    {
        let mut view = inplace.view_mut().unwrap();
        fill_view(&mut view, &reverse_values);
    }
    plan.inverse_in_place(&mut inplace, &mut inplace_workspace)
        .unwrap();
    assert_eq!(inplace.state(), R2rState::Input);
    assert_eq!(inplace.view().unwrap().as_slice().as_ptr(), pointer);
    check_slice(
        inplace.view().unwrap().as_slice(),
        inplace.view().unwrap().pencil(),
        inplace.view().unwrap().extra_shape(),
        &inverse_values,
        "R2R in-place inverse",
    );
    let inplace_inverse = inplace.view().unwrap().as_slice().to_vec();

    plan.forward_in_place(&mut inplace, &mut inplace_workspace)
        .unwrap();
    {
        let mut view = inplace.view_mut().unwrap();
        fill_view(&mut view, &reverse_values);
    }
    plan.backward_in_place(&mut inplace, &mut inplace_workspace)
        .unwrap();
    assert_eq!(inplace.state(), R2rState::Input);
    assert_eq!(inplace.view().unwrap().as_slice().as_ptr(), pointer);
    check_slice(
        inplace.view().unwrap().as_slice(),
        inplace.view().unwrap().pencil(),
        inplace.view().unwrap().extra_shape(),
        &backward_values,
        "R2R in-place backward",
    );
    let inplace_backward = inplace.view().unwrap().as_slice().to_vec();

    // Successful reuse is intentional: no resource is consumed by a call.
    plan.forward(&source, &mut output, &mut workspace).unwrap();
    plan.inverse(&reverse_source, &mut inverse, &mut workspace)
        .unwrap();
    plan.forward_in_place(&mut inplace, &mut inplace_workspace)
        .unwrap();
    plan.inverse_in_place(&mut inplace, &mut inplace_workspace)
        .unwrap();

    Snapshots(
        forward_snapshot,
        inverse_snapshot,
        backward_snapshot,
        inplace_forward,
        inplace_inverse,
        inplace_backward,
    )
}

fn parity_case<T: TestValue, const N: usize, const M: usize>(
    topology: &Arc<MpiTopology<M>>,
    shape: [usize; N],
    extra: ExtraShape,
    kinds: [Option<R2rKind>; N],
    seed: f64,
    check_constructors: bool,
) {
    let alltoallv = run_case::<T, N, M>(
        topology,
        shape,
        extra.clone(),
        kinds,
        TransposeMethod::AllToAllv,
        seed,
        check_constructors,
    );
    let point_to_point = run_case::<T, N, M>(
        topology,
        shape,
        extra,
        kinds,
        TransposeMethod::PointToPoint,
        seed,
        false,
    );
    assert_eq!(alltoallv, point_to_point, "R2R transport parity");
}

macro_rules! all_values {
    ($topology:expr, $n:expr, $m:expr, $shape:expr, $extra:expr, $kinds:expr, $seed:expr, $constructors:expr $(,)?) => {{
        parity_case::<f32, $n, $m>(
            $topology,
            $shape,
            $extra.clone(),
            $kinds,
            $seed,
            $constructors,
        );
        parity_case::<f64, $n, $m>(
            $topology,
            $shape,
            $extra.clone(),
            $kinds,
            $seed + 1.0,
            false,
        );
        parity_case::<Complex<f32>, $n, $m>(
            $topology,
            $shape,
            $extra.clone(),
            $kinds,
            $seed + 2.0,
            false,
        );
        parity_case::<Complex<f64>, $n, $m>($topology, $shape, $extra, $kinds, $seed + 3.0, false);
    }};
}

fn positive_cases(
    world: &mpi::topology::SimpleCommunicator,
    topology_1d: &Arc<MpiTopology<1>>,
    topology_2d: &Arc<MpiTopology<2>>,
) {
    let pairs = [
        [Some(R2rKind::DctI), Some(R2rKind::DctII)],
        [Some(R2rKind::DctIII), Some(R2rKind::DctIV)],
        [Some(R2rKind::DstI), Some(R2rKind::DstII)],
        [Some(R2rKind::DstIII), Some(R2rKind::DstIV)],
    ];
    for (index, kinds) in pairs.into_iter().enumerate() {
        all_values!(
            topology_1d,
            2,
            1,
            [3, 4],
            ExtraShape::scalar(),
            kinds,
            10.0 + index as f64 * 10.0,
            index == 0,
        );
    }

    all_values!(
        topology_1d,
        3,
        1,
        [2, 3, 4],
        ExtraShape::new([2, 3]).unwrap(),
        [Some(R2rKind::DctII), None, Some(R2rKind::DstII)],
        60.0,
        false,
    );
    all_values!(
        topology_1d,
        3,
        1,
        [2, 3, 4],
        ExtraShape::new([0]).unwrap(),
        [Some(R2rKind::DstI), None, Some(R2rKind::DctIV)],
        70.0,
        false,
    );
    all_values!(
        topology_1d,
        4,
        1,
        [2, 1, 3, 3],
        ExtraShape::new([2]).unwrap(),
        [None, None, None, None],
        80.0,
        false,
    );
    all_values!(
        topology_1d,
        4,
        1,
        [2, 1, 3, 3],
        ExtraShape::scalar(),
        [
            Some(R2rKind::DctI),
            None,
            Some(R2rKind::DstIII),
            Some(R2rKind::DctIV)
        ],
        90.0,
        false,
    );

    // This is a genuinely two-dimensional process grid. The extent one on
    // logical axis 1 deliberately creates empty local ranks at size 4/6.
    all_values!(
        topology_2d,
        4,
        2,
        [2, 1, 3, 3],
        ExtraShape::new([2]).unwrap(),
        [None, None, None, None],
        100.0,
        false,
    );
    let local_empty = i32::from(
        R2rPlan::<f64, 4, 2>::from_shape(
            Arc::clone(topology_2d),
            [2, 1, 3, 3],
            ExtraShape::new([2]).unwrap(),
            [None, None, None, None],
        )
        .unwrap()
        .input_pencil()
        .local_len()
            == 0,
    );
    let mut any_empty = 0;
    topology_2d.communicator().all_reduce_into(
        &local_empty,
        &mut any_empty,
        SystemOperation::max(),
    );
    if world.size() > 1 {
        assert_eq!(
            any_empty, 1,
            "the M=2 R2R case must have an empty local rank"
        );
    }

    // Reordered communicator: values and layouts must not depend on world rank
    // numbering or on the communicator's construction history.
    let reversed = world.split_by_color_with_key(
        Color::with_value(0),
        Key::try_from(world.size() - world.rank()).unwrap(),
    );
    let reversed = reversed.expect("all ranks join reordered communicator");
    let reordered = MpiTopology::<1>::new(&reversed, [world.size() as usize]).unwrap();
    parity_case::<Complex<f64>, 3, 1>(
        &reordered,
        [2, 3, 4],
        ExtraShape::new([2]).unwrap(),
        [Some(R2rKind::DctII), None, Some(R2rKind::DstII)],
        111.0,
        false,
    );
}

fn is_descriptor_mismatch(error: &R2rError) -> bool {
    matches!(error, R2rError::Fft(FftError::CollectiveDescriptorMismatch))
}

fn precondition_or(error: &R2rError, local: fn(&FftError) -> bool, root: bool) -> bool {
    match error {
        R2rError::Fft(error) if root => local(error),
        R2rError::Fft(FftError::CollectivePreconditionFailed) if !root => true,
        _ => false,
    }
}

fn assert_oop_rejected<T: TestValue, const N: usize, const M: usize, F, P>(
    source: &PencilArray<T, N, M>,
    destination: &mut PencilArray<T, N, M>,
    workspace: &mut pencil_fft::R2rWorkspace<T, N, M>,
    execute: F,
    predicate: P,
) where
    F: FnOnce(
        &PencilArray<T, N, M>,
        &mut PencilArray<T, N, M>,
        &mut pencil_fft::R2rWorkspace<T, N, M>,
    ) -> Result<(), R2rError>,
    P: FnOnce(&R2rError) -> bool,
{
    let source_before = source.as_slice().to_vec();
    let destination_before = destination.as_slice().to_vec();
    let workspace_before = format!("{workspace:?}");
    let error = execute(source, destination, workspace).expect_err("R2R rejection succeeded");
    assert!(predicate(&error), "unexpected R2R error: {error:?}");
    assert_eq!(source.as_slice(), source_before.as_slice());
    assert_eq!(destination.as_slice(), destination_before.as_slice());
    assert_eq!(format!("{workspace:?}"), workspace_before);
}

fn mixed_operation_oop_ip(world: &mpi::topology::SimpleCommunicator, plan: &R2rPlan<f64, 2, 1>) {
    let rank = world.rank();
    let source = plan.allocate_input().unwrap();
    let mut output = plan.allocate_output().unwrap();
    let mut workspace = plan.allocate_workspace().unwrap();
    let mut array = plan.allocate_in_place().unwrap();
    let mut inplace_workspace = plan.allocate_in_place_workspace().unwrap();
    let source_before = format!("{source:?}");
    let output_before = format!("{output:?}");
    let workspace_before = format!("{workspace:?}");
    let array_before = format!("{array:?}");
    let inplace_workspace_before = format!("{inplace_workspace:?}");
    let result = if rank == 0 {
        plan.forward(&source, &mut output, &mut workspace)
    } else {
        plan.forward_in_place(&mut array, &mut inplace_workspace)
    };
    assert!(matches!(
        result,
        Err(R2rError::Fft(FftError::CollectiveDescriptorMismatch))
    ));
    assert_eq!(format!("{source:?}"), source_before);
    assert_eq!(format!("{output:?}"), output_before);
    assert_eq!(format!("{workspace:?}"), workspace_before);
    assert_eq!(format!("{array:?}"), array_before);
    assert_eq!(format!("{inplace_workspace:?}"), inplace_workspace_before);
    plan.forward(&source, &mut output, &mut workspace).unwrap();
    plan.forward_in_place(&mut array, &mut inplace_workspace)
        .unwrap();
    world.barrier();
}

fn mixed_operation_oop(
    world: &mpi::topology::SimpleCommunicator,
    plan: &R2rPlan<f64, 2, 1>,
    root_direction: Direction,
    peer_direction: Direction,
) {
    let rank = world.rank();
    let mut input = plan.allocate_input().unwrap();
    let mut output = plan.allocate_output().unwrap();
    let mut input_destination = plan.allocate_input().unwrap();
    let mut workspace = plan.allocate_workspace().unwrap();
    let result = if (rank == 0) == matches!(root_direction, Direction::Forward) {
        match root_direction {
            Direction::Forward => plan.forward(&input, &mut output, &mut workspace),
            Direction::Inverse => plan.inverse(&output, &mut input_destination, &mut workspace),
            Direction::Backward => plan.backward(&output, &mut input_destination, &mut workspace),
        }
    } else {
        match peer_direction {
            Direction::Forward => plan.forward(&input, &mut output, &mut workspace),
            Direction::Inverse => plan.inverse(&output, &mut input_destination, &mut workspace),
            Direction::Backward => plan.backward(&output, &mut input_destination, &mut workspace),
        }
    };
    assert!(matches!(
        result,
        Err(R2rError::Fft(FftError::CollectiveDescriptorMismatch))
    ));
    let _ = (&mut input, &mut output, &mut input_destination);
    world.barrier();
}

fn negative_cases(
    world: &mpi::topology::SimpleCommunicator,
    topology_1d: &Arc<MpiTopology<1>>,
    topology_2d: &Arc<MpiTopology<2>>,
) {
    let rank = world.rank();
    let size = usize::try_from(world.size()).unwrap();

    assert!(matches!(
        R2rPlan::<f64, 1, 1>::from_shape(
            Arc::clone(topology_1d),
            [4],
            ExtraShape::scalar(),
            [Some(R2rKind::DctII)],
        ),
        Err(R2rError::Fft(FftError::InvalidDimensions))
    ));
    assert!(matches!(
        R2rPlan::<f64, 2, 1>::from_shape(
            Arc::clone(topology_1d),
            [1, 3],
            ExtraShape::scalar(),
            [Some(R2rKind::DctI), None],
        ),
        Err(R2rError::LocalR2r(pencil_fft::LocalR2rError::InvalidLength))
    ));
    assert!(matches!(
        R2rPlan::<f64, 2, 1>::from_shape(
            Arc::clone(topology_1d),
            [usize::MAX, 2],
            ExtraShape::scalar(),
            [Some(R2rKind::DctII), None],
        ),
        Err(R2rError::Fft(FftError::Pencil(
            pencil_array::PencilError::SizeOverflow
        )))
    ));
    world.barrier();

    let plan = R2rPlan::<f64, 2, 1>::from_shape(
        Arc::clone(topology_1d),
        [3, 4],
        ExtraShape::scalar(),
        [Some(R2rKind::DctII), Some(R2rKind::DstIII)],
    )
    .unwrap();
    let mismatch_plan = R2rPlan::<f64, 2, 1>::from_shape(
        Arc::clone(topology_1d),
        [3, 4],
        ExtraShape::scalar(),
        [Some(R2rKind::DctIII), Some(R2rKind::DstIII)],
    )
    .unwrap();
    let _ = mismatch_plan;

    if size > 1 {
        let result = if rank == 0 {
            R2rPlan::<f64, 2, 1>::from_shape(
                Arc::clone(topology_1d),
                [3, 4],
                ExtraShape::scalar(),
                [Some(R2rKind::DctII), Some(R2rKind::DstII)],
            )
            .map(|_| ())
        } else {
            R2rPlan::<f64, 2, 1>::from_shape(
                Arc::clone(topology_1d),
                [3, 4],
                ExtraShape::scalar(),
                [Some(R2rKind::DctIII), Some(R2rKind::DstII)],
            )
            .map(|_| ())
        };
        assert!(result.as_ref().err().is_some_and(is_descriptor_mismatch));
        let value_kind = if rank == 0 {
            R2rPlan::<f64, 2, 1>::from_shape(
                Arc::clone(topology_1d),
                [3, 4],
                ExtraShape::scalar(),
                [Some(R2rKind::DctII), None],
            )
            .map(|_| ())
        } else {
            R2rPlan::<Complex<f64>, 2, 1>::from_shape(
                Arc::clone(topology_1d),
                [3, 4],
                ExtraShape::scalar(),
                [Some(R2rKind::DctII), None],
            )
            .map(|_| ())
        };
        assert!(
            value_kind
                .as_ref()
                .err()
                .is_some_and(is_descriptor_mismatch)
        );
        let precision = if rank == 0 {
            R2rPlan::<f64, 2, 1>::from_shape(
                Arc::clone(topology_1d),
                [3, 4],
                ExtraShape::scalar(),
                [Some(R2rKind::DctII), None],
            )
            .map(|_| ())
        } else {
            R2rPlan::<f32, 2, 1>::from_shape(
                Arc::clone(topology_1d),
                [3, 4],
                ExtraShape::scalar(),
                [Some(R2rKind::DctII), None],
            )
            .map(|_| ())
        };
        assert!(precision.as_ref().err().is_some_and(is_descriptor_mismatch));
        // Equal byte width is intentionally a separate check: real f64 and
        // complex f32 must differ by element kind, not by size_of::<T>().
        let equal_bytes = if rank == 0 {
            R2rPlan::<f64, 2, 1>::from_shape(
                Arc::clone(topology_1d),
                [3, 4],
                ExtraShape::scalar(),
                [Some(R2rKind::DctII), None],
            )
            .map(|_| ())
        } else {
            R2rPlan::<Complex<f32>, 2, 1>::from_shape(
                Arc::clone(topology_1d),
                [3, 4],
                ExtraShape::scalar(),
                [Some(R2rKind::DctII), None],
            )
            .map(|_| ())
        };
        assert!(
            equal_bytes
                .as_ref()
                .err()
                .is_some_and(is_descriptor_mismatch)
        );
        let dimension = if rank == 0 {
            R2rPlan::<f64, 2, 1>::from_shape(
                Arc::clone(topology_1d),
                [3, 4],
                ExtraShape::scalar(),
                [Some(R2rKind::DctII), None],
            )
            .map(|_| ())
        } else {
            R2rPlan::<f64, 3, 1>::from_shape(
                Arc::clone(topology_1d),
                [3, 2, 4],
                ExtraShape::scalar(),
                [Some(R2rKind::DctII), None, None],
            )
            .map(|_| ())
        };
        assert!(dimension.as_ref().err().is_some_and(is_descriptor_mismatch));
        let shape = if rank == 0 {
            R2rPlan::<f64, 2, 1>::from_shape(
                Arc::clone(topology_1d),
                [3, 4],
                ExtraShape::scalar(),
                [Some(R2rKind::DctII), None],
            )
            .map(|_| ())
        } else {
            R2rPlan::<f64, 2, 1>::from_shape(
                Arc::clone(topology_1d),
                [2, 6],
                ExtraShape::scalar(),
                [Some(R2rKind::DctII), None],
            )
            .map(|_| ())
        };
        assert!(shape.as_ref().err().is_some_and(is_descriptor_mismatch));
        let extra = if rank == 0 {
            R2rPlan::<f64, 2, 1>::from_shape(
                Arc::clone(topology_1d),
                [3, 4],
                ExtraShape::new([2, 3]).unwrap(),
                [Some(R2rKind::DctII), None],
            )
            .map(|_| ())
        } else {
            R2rPlan::<f64, 2, 1>::from_shape(
                Arc::clone(topology_1d),
                [3, 4],
                ExtraShape::new([3, 2]).unwrap(),
                [Some(R2rKind::DctII), None],
            )
            .map(|_| ())
        };
        assert!(extra.as_ref().err().is_some_and(is_descriptor_mismatch));
        let transport = if rank == 0 {
            R2rPlan::<f64, 2, 1>::from_shape_with_method(
                Arc::clone(topology_1d),
                [3, 4],
                ExtraShape::scalar(),
                [Some(R2rKind::DctII), None],
                TransposeMethod::AllToAllv,
            )
            .map(|_| ())
        } else {
            R2rPlan::<f64, 2, 1>::from_shape_with_method(
                Arc::clone(topology_1d),
                [3, 4],
                ExtraShape::scalar(),
                [Some(R2rKind::DctII), None],
                TransposeMethod::PointToPoint,
            )
            .map(|_| ())
        };
        assert!(transport.as_ref().err().is_some_and(is_descriptor_mismatch));
        world.barrier();
    }

    if size > 1 {
        mixed_operation_oop(world, &plan, Direction::Backward, Direction::Inverse);
        mixed_operation_oop(world, &plan, Direction::Backward, Direction::Forward);
        mixed_operation_oop_ip(world, &plan);
    }

    let source = plan.allocate_input().unwrap();
    let mut output = plan.allocate_output().unwrap();
    let mut workspace = plan.allocate_workspace().unwrap();
    let wrong_source = plan.allocate_output().unwrap();
    let root = rank == 0;
    assert_oop_rejected(
        if root { &wrong_source } else { &source },
        &mut output,
        &mut workspace,
        |source, destination, workspace| plan.forward(source, destination, workspace),
        |error| {
            precondition_or(
                error,
                |error| matches!(error, FftError::InputLayoutMismatch),
                root,
            )
        },
    );
    plan.forward(&source, &mut output, &mut workspace).unwrap();
    world.barrier();

    let mut wrong_destination = plan.allocate_input().unwrap();
    assert_oop_rejected(
        &source,
        if root {
            &mut wrong_destination
        } else {
            &mut output
        },
        &mut workspace,
        |source, destination, workspace| plan.forward(source, destination, workspace),
        |error| {
            precondition_or(
                error,
                |error| matches!(error, FftError::OutputLayoutMismatch),
                root,
            )
        },
    );
    plan.forward(&source, &mut output, &mut workspace).unwrap();
    world.barrier();

    let batch_plan = R2rPlan::<f64, 2, 1>::from_shape(
        Arc::clone(topology_1d),
        [3, 4],
        ExtraShape::new([2, 3]).unwrap(),
        [Some(R2rKind::DctII), None],
    )
    .unwrap();
    let batch_source = batch_plan.allocate_input().unwrap();
    let mut batch_output = batch_plan.allocate_output().unwrap();
    let mut batch_workspace = batch_plan.allocate_workspace().unwrap();
    let wrong_extra = PencilArray::from_elem(
        Arc::clone(batch_plan.input_pencil()),
        ExtraShape::new([3, 2]).unwrap(),
        1.0_f64,
    )
    .unwrap();
    assert_oop_rejected(
        if root { &wrong_extra } else { &batch_source },
        &mut batch_output,
        &mut batch_workspace,
        |source, destination, workspace| batch_plan.forward(source, destination, workspace),
        |error| {
            precondition_or(
                error,
                |error| matches!(error, FftError::ExtraShapeMismatch),
                root,
            )
        },
    );
    batch_plan
        .forward(&batch_source, &mut batch_output, &mut batch_workspace)
        .unwrap();
    world.barrier();

    let foreign_topology = MpiTopology::<1>::new(world, [size]).unwrap();
    let foreign_plan = R2rPlan::<f64, 2, 1>::from_shape(
        Arc::clone(&foreign_topology),
        [3, 4],
        ExtraShape::scalar(),
        [Some(R2rKind::DctII), None],
    )
    .unwrap();
    let foreign_source = foreign_plan.allocate_input().unwrap();
    let other_plan = R2rPlan::<f64, 2, 1>::from_shape(
        Arc::clone(topology_1d),
        [3, 4],
        ExtraShape::scalar(),
        [Some(R2rKind::DctII), None],
    )
    .unwrap();
    let mut foreign_workspace = other_plan.allocate_workspace().unwrap();
    assert_oop_rejected(
        if root { &foreign_source } else { &source },
        &mut output,
        if root {
            &mut foreign_workspace
        } else {
            &mut workspace
        },
        |source, destination, workspace| plan.forward(source, destination, workspace),
        |error| {
            if root {
                matches!(
                    error,
                    R2rError::Fft(FftError::InputLayoutMismatch)
                        | R2rError::Fft(FftError::WorkspaceMismatch)
                )
            } else {
                matches!(error, R2rError::Fft(FftError::WorkspaceMismatch))
                    || matches!(error, R2rError::Fft(FftError::CollectivePreconditionFailed))
            }
        },
    );
    plan.forward(&source, &mut output, &mut workspace).unwrap();
    world.barrier();

    // In-place wrong state, foreign array, foreign workspace, and OOP/IP
    // operation mixing all preserve the preflight state and bytes.
    let mut array = plan.allocate_in_place().unwrap();
    let mut inplace_workspace = plan.allocate_in_place_workspace().unwrap();
    let mut foreign_array = foreign_plan.allocate_in_place().unwrap();
    let foreign_array_before = format!("{foreign_array:?}");
    let result = plan.forward_in_place(&mut foreign_array, &mut inplace_workspace);
    assert!(matches!(
        result,
        Err(R2rError::Fft(FftError::Array(
            pencil_array::ArrayError::IncompatiblePencils
        )))
    ));
    assert_eq!(foreign_array.state(), R2rState::Input);
    assert_eq!(format!("{foreign_array:?}"), foreign_array_before);
    let mut foreign_inplace_workspace = foreign_plan.allocate_in_place_workspace().unwrap();
    let array_before = format!("{array:?}");
    let result = plan.forward_in_place(&mut array, &mut foreign_inplace_workspace);
    assert!(matches!(
        result,
        Err(R2rError::Fft(FftError::WorkspaceMismatch))
    ));
    assert_eq!(array.state(), R2rState::Input);
    assert_eq!(format!("{array:?}"), array_before);
    assert!(matches!(
        plan.inverse_in_place(&mut array, &mut inplace_workspace),
        Err(R2rError::Fft(FftError::InputLayoutMismatch))
    ));
    assert_eq!(array.state(), R2rState::Input);
    plan.forward_in_place(&mut array, &mut inplace_workspace)
        .unwrap();
    assert!(matches!(
        plan.forward_in_place(&mut array, &mut inplace_workspace),
        Err(R2rError::Fft(FftError::InputLayoutMismatch))
    ));
    plan.inverse_in_place(&mut array, &mut inplace_workspace)
        .unwrap();
    world.barrier();

    if size > 1 {
        let all_plan = R2rPlan::<f64, 2, 1>::from_shape_with_method(
            Arc::clone(topology_1d),
            [3, 4],
            ExtraShape::scalar(),
            [Some(R2rKind::DctII), None],
            TransposeMethod::AllToAllv,
        )
        .unwrap();
        let p2p_plan = R2rPlan::<f64, 2, 1>::from_shape_with_method(
            Arc::clone(topology_1d),
            [3, 4],
            ExtraShape::scalar(),
            [Some(R2rKind::DctII), None],
            TransposeMethod::PointToPoint,
        )
        .unwrap();
        let all_source = all_plan.allocate_input().unwrap();
        let mut all_output = all_plan.allocate_output().unwrap();
        let mut all_workspace = all_plan.allocate_workspace().unwrap();
        let p2p_source = p2p_plan.allocate_input().unwrap();
        let mut p2p_output = p2p_plan.allocate_output().unwrap();
        let mut p2p_workspace = p2p_plan.allocate_workspace().unwrap();
        let result = if rank == 0 {
            p2p_plan.forward(&p2p_source, &mut p2p_output, &mut p2p_workspace)
        } else {
            all_plan.forward(&all_source, &mut all_output, &mut all_workspace)
        };
        assert!(matches!(
            result,
            Err(R2rError::Fft(FftError::CollectiveDescriptorMismatch))
        ));
        all_plan
            .forward(&all_source, &mut all_output, &mut all_workspace)
            .unwrap();
        p2p_plan
            .forward(&p2p_source, &mut p2p_output, &mut p2p_workspace)
            .unwrap();
        world.barrier();

        let mut all_array = all_plan.allocate_in_place().unwrap();
        let mut all_ip_workspace = all_plan.allocate_in_place_workspace().unwrap();
        let mut p2p_array = p2p_plan.allocate_in_place().unwrap();
        let mut p2p_ip_workspace = p2p_plan.allocate_in_place_workspace().unwrap();
        let result = if rank == 0 {
            all_plan.forward_in_place(&mut all_array, &mut all_ip_workspace)
        } else {
            p2p_plan.forward_in_place(&mut p2p_array, &mut p2p_ip_workspace)
        };
        assert!(matches!(
            result,
            Err(R2rError::Fft(FftError::CollectiveDescriptorMismatch))
        ));
        all_plan
            .forward_in_place(&mut all_array, &mut all_ip_workspace)
            .unwrap();
        p2p_plan
            .forward_in_place(&mut p2p_array, &mut p2p_ip_workspace)
            .unwrap();
        world.barrier();
    }

    if world.size() == 6 {
        // Rank 5 is in an empty rank/changed-subgroup corner. The check is
        // deliberately full-Cartesian, so it must not deadlock or pass merely
        // because rank 5 is outside the first changed-axis subgroup.
        let plan_2d = R2rPlan::<f64, 4, 2>::from_shape_with_method(
            Arc::clone(topology_2d),
            [2, 1, 3, 3],
            ExtraShape::new([2]).unwrap(),
            [None, None, None, None],
            TransposeMethod::PointToPoint,
        )
        .unwrap();
        let source_2d = plan_2d.allocate_input().unwrap();
        let wrong_2d = plan_2d.allocate_output().unwrap();
        let mut output_2d = plan_2d.allocate_output().unwrap();
        let mut workspace_2d = plan_2d.allocate_workspace().unwrap();
        assert_oop_rejected(
            if rank == 5 { &wrong_2d } else { &source_2d },
            &mut output_2d,
            &mut workspace_2d,
            |source, destination, workspace| plan_2d.forward(source, destination, workspace),
            |error| {
                if rank == 5 {
                    matches!(error, R2rError::Fft(FftError::InputLayoutMismatch))
                } else {
                    matches!(error, R2rError::Fft(FftError::CollectivePreconditionFailed))
                }
            },
        );
        plan_2d
            .forward(&source_2d, &mut output_2d, &mut workspace_2d)
            .unwrap();
        world.barrier();
    }
}

#[test]
fn distributed_r2r_oracle_matrix_and_negative_contracts() {
    let universe = mpi::initialize().expect("MPI must not already be initialized");
    let world = universe.world();
    let size = usize::try_from(world.size()).unwrap();
    assert!(matches!(size, 1 | 4 | 6), "run with 1, 4, or 6 ranks");
    let topology_1d = MpiTopology::<1>::new(&world, [size]).unwrap();
    let grid = match size {
        1 => [1, 1],
        4 => [2, 2],
        6 => [2, 3],
        _ => unreachable!(),
    };
    let topology_2d = MpiTopology::<2>::new(&world, grid).unwrap();
    positive_cases(&world, &topology_1d, &topology_2d);
    negative_cases(&world, &topology_1d, &topology_2d);
}
