#![cfg(feature = "distributed")]
#![allow(clippy::arc_with_non_send_sync, clippy::needless_range_loop)]

use std::{f64::consts::PI, sync::Arc};

use mpi::{
    collective::{CommunicatorCollectives, SystemOperation},
    topology::Communicator,
};
use pencil_array::{ExtraShape, MpiTopology, PencilArray};
use pencil_fft::{
    AxisR2rKind, AxisTransform, Complex, DistributedLayout, FftError, FftReal, MixedC2cPlan,
    MixedError, MixedR2cPlan, R2cState, R2rKind, TransposeMethod,
};

type C64 = Complex<f64>;

#[derive(Clone, Copy)]
enum Op {
    None,
    Fft,
    Rfft,
    DctII,
    Dht,
}

trait TestReal: FftReal {
    fn from_f64(value: f64) -> Self;
    fn to_f64(self) -> f64;
}

impl TestReal for f32 {
    fn from_f64(value: f64) -> Self {
        value as f32
    }

    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl TestReal for f64 {
    fn from_f64(value: f64) -> Self {
        value
    }

    fn to_f64(self) -> f64 {
        self
    }
}

fn product(shape: &[usize]) -> usize {
    shape.iter().copied().product()
}

fn unravel(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    let mut coordinates = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        coordinates[axis] = linear % shape[axis];
        linear /= shape[axis];
    }
    coordinates
}

fn offset(shape: &[usize], coordinates: &[usize]) -> usize {
    shape
        .iter()
        .zip(coordinates)
        .fold(0, |value, (&extent, &coordinate)| {
            value * extent + coordinate
        })
}

fn batches<const N: usize>(
    extra: &[usize],
    shape: [usize; N],
    mut value: impl FnMut(usize, [usize; N]) -> C64,
) -> Vec<Vec<C64>> {
    (0..product(extra))
        .map(|batch| {
            (0..product(&shape))
                .map(|linear| {
                    let coordinates: [usize; N] = unravel(linear, &shape).try_into().unwrap();
                    value(batch, coordinates)
                })
                .collect()
        })
        .collect()
}

fn seed(batch: usize, coordinates: &[usize]) -> C64 {
    let mut real = 0.17 + 0.031 * batch as f64;
    let mut imaginary = -0.23 - 0.019 * batch as f64;
    for (axis, &coordinate) in coordinates.iter().enumerate() {
        real += (0.41 + 0.07 * axis as f64) * (coordinate + 1) as f64;
        imaginary += (0.29 + 0.05 * axis as f64) * (coordinate + 2) as f64;
    }
    C64::new(real, imaginary)
}

fn twiddle(angle: f64) -> C64 {
    C64::new(angle.cos(), angle.sin())
}

fn line_transform(line: &[C64], op: Op, reverse: bool, normalize: bool) -> Vec<C64> {
    let n = line.len();
    match op {
        Op::None => line.to_vec(),
        Op::Rfft => unreachable!("RFFT is handled at its boundary"),
        Op::Fft => (0..n)
            .map(|k| {
                let mut sum = C64::new(0.0, 0.0);
                for (j, &value) in line.iter().enumerate() {
                    let sign = if reverse { 1.0 } else { -1.0 };
                    sum += value * twiddle(sign * 2.0 * PI * (j * k) as f64 / n as f64);
                }
                if reverse && normalize {
                    sum / n as f64
                } else {
                    sum
                }
            })
            .collect(),
        Op::Dht => (0..n)
            .map(|k| {
                let mut sum = C64::new(0.0, 0.0);
                for (j, &value) in line.iter().enumerate() {
                    let angle = PI * 2.0 * (j * k) as f64 / n as f64;
                    sum += value * (angle.cos() + angle.sin());
                }
                if reverse && normalize {
                    sum / n as f64
                } else {
                    sum
                }
            })
            .collect(),
        Op::DctII if !reverse => (0..n)
            .map(|k| {
                (0..n).fold(C64::new(0.0, 0.0), |sum, j| {
                    sum + line[j] * (2.0 * (PI * (j as f64 + 0.5) * k as f64 / n as f64).cos())
                })
            })
            .collect(),
        Op::DctII => (0..n)
            .map(|j| {
                let mut sum = line[0];
                for (k, &value) in line.iter().enumerate().skip(1) {
                    sum += value * (2.0 * (PI * k as f64 * (j as f64 + 0.5) / n as f64).cos());
                }
                if normalize {
                    sum / (2.0 * n as f64)
                } else {
                    sum
                }
            })
            .collect(),
    }
}

fn apply_axis(
    data: &mut Vec<C64>,
    shape: &[usize],
    axis: usize,
    op: Op,
    reverse: bool,
    normalize: bool,
) {
    if matches!(op, Op::None) || shape[axis] == 0 {
        return;
    }
    let line_len = shape[axis];
    let mut result = data.clone();
    for linear in 0..data.len() {
        let mut coordinates = unravel(linear, shape);
        if coordinates[axis] != 0 {
            continue;
        }
        let mut line = Vec::with_capacity(line_len);
        for k in 0..line_len {
            coordinates[axis] = k;
            line.push(data[offset(shape, &coordinates)]);
        }
        let transformed = line_transform(&line, op, reverse, normalize);
        for (k, value) in transformed.into_iter().enumerate() {
            coordinates[axis] = k;
            result[offset(shape, &coordinates)] = value;
        }
    }
    *data = result;
}

fn rfft_forward(data: &[C64], shape: &[usize], axis: usize) -> (Vec<C64>, Vec<usize>) {
    let n = shape[axis];
    let reduced = n / 2 + 1;
    let mut output_shape = shape.to_vec();
    output_shape[axis] = reduced;
    let mut output = vec![C64::new(0.0, 0.0); product(&output_shape)];
    for linear in 0..data.len() {
        let mut coordinates = unravel(linear, shape);
        if coordinates[axis] != 0 {
            continue;
        }
        let mut line = Vec::with_capacity(n);
        for k in 0..n {
            coordinates[axis] = k;
            line.push(data[offset(shape, &coordinates)]);
        }
        for k in 0..reduced {
            let mut sum = C64::new(0.0, 0.0);
            for (j, &value) in line.iter().enumerate() {
                sum += value * twiddle(-2.0 * PI * (j * k) as f64 / n as f64);
            }
            coordinates[axis] = k;
            output[offset(&output_shape, &coordinates)] = sum;
        }
    }
    (output, output_shape)
}

fn rfft_reverse(
    data: &[C64],
    shape: &[usize],
    axis: usize,
    original_n: usize,
    normalize: bool,
) -> (Vec<C64>, Vec<usize>) {
    let reduced = shape[axis];
    let mut output_shape = shape.to_vec();
    output_shape[axis] = original_n;
    let mut output = vec![C64::new(0.0, 0.0); product(&output_shape)];
    for linear in 0..data.len() {
        let mut coordinates = unravel(linear, shape);
        if coordinates[axis] != 0 {
            continue;
        }
        let mut line = Vec::with_capacity(reduced);
        for k in 0..reduced {
            coordinates[axis] = k;
            line.push(data[offset(shape, &coordinates)]);
        }
        for j in 0..original_n {
            let mut sum = line[0];
            for (k, &value) in line.iter().enumerate().skip(1) {
                let endpoint = original_n % 2 == 0 && k == reduced - 1;
                let angle = 2.0 * PI * (j * k) as f64 / original_n as f64;
                let contribution = value * twiddle(angle);
                sum += if endpoint {
                    contribution
                } else {
                    C64::new(2.0 * contribution.re, 0.0)
                };
            }
            coordinates[axis] = j;
            output[offset(&output_shape, &coordinates)] = if normalize {
                C64::new(sum.re / original_n as f64, 0.0)
            } else {
                C64::new(sum.re, 0.0)
            };
        }
    }
    (output, output_shape)
}

fn c2c_transform<const N: usize>(
    mut data: Vec<C64>,
    shape: [usize; N],
    ops: [Op; N],
    reverse: bool,
    normalize: bool,
) -> Vec<C64> {
    if reverse {
        for axis in 0..N {
            apply_axis(&mut data, &shape, axis, ops[axis], true, normalize);
        }
    } else {
        for axis in (0..N).rev() {
            apply_axis(&mut data, &shape, axis, ops[axis], false, false);
        }
    }
    data
}

fn mixed_r2c_forward<const N: usize>(
    mut data: Vec<C64>,
    shape: [usize; N],
    ops: [Op; N],
    boundary: usize,
) -> (Vec<C64>, Vec<usize>) {
    let mut shape = shape.to_vec();
    for axis in (boundary + 1..N).rev() {
        apply_axis(&mut data, &shape, axis, ops[axis], false, false);
    }
    let (mut data, reduced_shape) = rfft_forward(&data, &shape, boundary);
    shape = reduced_shape;
    for axis in (0..boundary).rev() {
        apply_axis(&mut data, &shape, axis, ops[axis], false, false);
    }
    (data, shape)
}

fn mixed_r2c_reverse<const N: usize>(
    mut data: Vec<C64>,
    reduced_shape: &[usize],
    original_shape: [usize; N],
    ops: [Op; N],
    boundary: usize,
    normalize: bool,
) -> (Vec<C64>, Vec<usize>) {
    let mut shape = reduced_shape.to_vec();
    for axis in 0..boundary {
        apply_axis(&mut data, &shape, axis, ops[axis], true, normalize);
    }
    let (mut data, full_shape) =
        rfft_reverse(&data, &shape, boundary, original_shape[boundary], normalize);
    shape = full_shape;
    for axis in boundary + 1..N {
        apply_axis(&mut data, &shape, axis, ops[axis], true, normalize);
    }
    (data, shape)
}

fn extra_indices(mut linear: usize, extra: &[usize]) -> Vec<usize> {
    let mut result = vec![0; extra.len()];
    for axis in (0..extra.len()).rev() {
        result[axis] = linear % extra[axis];
        linear /= extra[axis];
    }
    result
}

fn extra_shape(extra: &[usize]) -> ExtraShape {
    ExtraShape::new(extra.to_vec()).unwrap()
}

fn fill_complex<R: TestReal, const N: usize, const M: usize>(
    array: &mut PencilArray<Complex<R>, N, M>,
    values: &[Vec<C64>],
    extra: &[usize],
    shape: [usize; N],
) {
    for (batch, values) in values.iter().enumerate() {
        let extra_indices = extra_indices(batch, extra);
        for linear in 0..values.len() {
            let coordinates: [usize; N] = unravel(linear, &shape).try_into().unwrap();
            if let Some(slot) = array.get_global_mut(&extra_indices, coordinates) {
                *slot = Complex::new(
                    <R as TestReal>::from_f64(values[linear].re),
                    <R as TestReal>::from_f64(values[linear].im),
                );
            }
        }
    }
}

fn fill_real<R: TestReal, const N: usize, const M: usize>(
    array: &mut PencilArray<R, N, M>,
    values: &[Vec<C64>],
    extra: &[usize],
    shape: [usize; N],
) {
    for (batch, values) in values.iter().enumerate() {
        let extra_indices = extra_indices(batch, extra);
        for linear in 0..values.len() {
            let coordinates: [usize; N] = unravel(linear, &shape).try_into().unwrap();
            if let Some(slot) = array.get_global_mut(&extra_indices, coordinates) {
                *slot = <R as TestReal>::from_f64(values[linear].re);
            }
        }
    }
}

fn check_complex<R: TestReal, C: CommunicatorCollectives, const N: usize>(
    communicator: &C,
    values: &[Vec<C64>],
    extra: &[usize],
    shape: [usize; N],
    get: impl Fn(&[usize], [usize; N]) -> Option<Complex<R>>,
    tolerance: f64,
    label: &str,
) {
    let mut local_max: f64 = 0.0;
    let mut local_owned = 0i32;
    for (batch, expected) in values.iter().enumerate() {
        let extra_indices = extra_indices(batch, extra);
        for linear in 0..expected.len() {
            let coordinates: [usize; N] = unravel(linear, &shape).try_into().unwrap();
            if let Some(actual) = get(&extra_indices, coordinates) {
                local_owned += 1;
                local_max = local_max.max((actual.re.to_f64() - expected[linear].re).abs());
                local_max = local_max.max((actual.im.to_f64() - expected[linear].im).abs());
            }
        }
    }
    let mut max_error = 0.0;
    communicator.all_reduce_into(&local_max, &mut max_error, SystemOperation::max());
    let mut owned = 0i32;
    communicator.all_reduce_into(&local_owned, &mut owned, SystemOperation::sum());
    let expected_owned = values.iter().map(Vec::len).sum::<usize>() as i32;
    assert_eq!(owned, expected_owned, "{label}: global ownership");
    assert!(
        max_error <= tolerance,
        "{label}: error {max_error} > {tolerance}"
    );
}

fn check_real<R: TestReal, C: CommunicatorCollectives, const N: usize>(
    communicator: &C,
    values: &[Vec<C64>],
    extra: &[usize],
    shape: [usize; N],
    get: impl Fn(&[usize], [usize; N]) -> Option<R>,
    tolerance: f64,
    label: &str,
) {
    let mut local_max: f64 = 0.0;
    let mut local_owned = 0i32;
    for (batch, expected) in values.iter().enumerate() {
        let extra_indices = extra_indices(batch, extra);
        for linear in 0..expected.len() {
            let coordinates: [usize; N] = unravel(linear, &shape).try_into().unwrap();
            if let Some(actual) = get(&extra_indices, coordinates) {
                local_owned += 1;
                local_max = local_max.max((actual.to_f64() - expected[linear].re).abs());
            }
        }
    }
    let mut max_error = 0.0;
    communicator.all_reduce_into(&local_max, &mut max_error, SystemOperation::max());
    let mut owned = 0i32;
    communicator.all_reduce_into(&local_owned, &mut owned, SystemOperation::sum());
    let expected_owned = values.iter().map(Vec::len).sum::<usize>() as i32;
    assert_eq!(owned, expected_owned, "{label}: global ownership");
    assert!(
        max_error <= tolerance,
        "{label}: error {max_error} > {tolerance}"
    );
}

fn run_c2c_case<R: TestReal, const N: usize, const M: usize>(
    topology: Arc<MpiTopology<M>>,
    shape: [usize; N],
    extra: &[usize],
    ops: [Op; N],
    method: TransposeMethod,
    permute_dims: bool,
) where
    Complex<R>: mpi::datatype::Equivalence,
{
    let plan = MixedC2cPlan::<R, N, M>::from_shape_with_layout(
        Arc::clone(&topology),
        shape,
        extra_shape(extra),
        ops.map(|op| match op {
            Op::None => AxisTransform::None,
            Op::Fft => AxisTransform::Fft,
            Op::Rfft => AxisTransform::Rfft,
            Op::DctII => AxisTransform::R2r(AxisR2rKind::Fftw(R2rKind::DctII)),
            Op::Dht => AxisTransform::R2r(AxisR2rKind::Dht),
        }),
        DistributedLayout {
            transpose_method: method,
            permute_dims,
        },
    )
    .unwrap();
    let input = batches(extra, shape, |batch, coordinates| seed(batch, &coordinates));
    let inverse_input = batches(extra, shape, |batch, coordinates| {
        seed(batch + 11, &coordinates)
    });
    let forward: Vec<_> = input
        .iter()
        .map(|values| c2c_transform(values.clone(), shape, ops, false, false))
        .collect();
    let inverse: Vec<_> = inverse_input
        .iter()
        .map(|values| c2c_transform(values.clone(), shape, ops, true, true))
        .collect();
    let backward: Vec<_> = inverse_input
        .iter()
        .map(|values| c2c_transform(values.clone(), shape, ops, true, false))
        .collect();

    let mut source = plan.allocate_input().unwrap();
    fill_complex(&mut source, &input, extra, shape);
    let source_before = source.as_slice().to_vec();
    let mut output = plan.allocate_output().unwrap();
    let mut workspace = plan.allocate_workspace().unwrap();
    plan.forward(&source, &mut output, &mut workspace).unwrap();
    check_complex(
        topology.communicator(),
        &forward,
        extra,
        shape,
        |extra, coordinates| output.get_global(extra, coordinates).copied(),
        if std::mem::size_of::<R>() == 4 {
            3e-3
        } else {
            1e-9
        },
        "mixed C2C forward oracle",
    );
    assert_eq!(source.as_slice(), source_before.as_slice());

    let mut inverse_source = plan.allocate_output().unwrap();
    fill_complex(&mut inverse_source, &inverse_input, extra, shape);
    let inverse_source_before = inverse_source.as_slice().to_vec();
    let mut recovered = plan.allocate_input().unwrap();
    plan.inverse(&inverse_source, &mut recovered, &mut workspace)
        .unwrap();
    assert_eq!(inverse_source.as_slice(), inverse_source_before.as_slice());
    check_complex(
        topology.communicator(),
        &inverse,
        extra,
        shape,
        |extra, coordinates| {
            recovered
                .get_global(extra, coordinates)
                .map(|value| Complex::new(value.re, value.im))
        },
        if std::mem::size_of::<R>() == 4 {
            3e-3
        } else {
            1e-9
        },
        "mixed C2C inverse oracle",
    );

    let mut backwards = plan.allocate_input().unwrap();
    plan.backward(&inverse_source, &mut backwards, &mut workspace)
        .unwrap();
    assert_eq!(inverse_source.as_slice(), inverse_source_before.as_slice());
    check_complex(
        topology.communicator(),
        &backward,
        extra,
        shape,
        |extra, coordinates| {
            backwards
                .get_global(extra, coordinates)
                .map(|value| Complex::new(value.re, value.im))
        },
        if std::mem::size_of::<R>() == 4 {
            3e-3
        } else {
            1e-9
        },
        "mixed C2C backward oracle",
    );

    let mut array = plan.allocate_in_place().unwrap();
    fill_complex_view(&mut array, &input, extra, shape);
    let mut ip_workspace = plan.allocate_in_place_workspace().unwrap();
    plan.forward_in_place(&mut array, &mut ip_workspace)
        .unwrap();
    {
        let view = array.view().unwrap();
        check_complex(
            topology.communicator(),
            &forward,
            extra,
            shape,
            |extra, coordinates| view.get_global(extra, coordinates).copied(),
            if std::mem::size_of::<R>() == 4 {
                3e-3
            } else {
                1e-9
            },
            "mixed C2C in-place forward oracle",
        );
    }
    plan.inverse_in_place(&mut array, &mut ip_workspace)
        .unwrap();
    {
        let view = array.view().unwrap();
        check_complex(
            topology.communicator(),
            &input,
            extra,
            shape,
            |extra, coordinates| view.get_global(extra, coordinates).copied(),
            if std::mem::size_of::<R>() == 4 {
                3e-3
            } else {
                1e-9
            },
            "mixed C2C in-place inverse oracle",
        );
    }
    let in_place_backward: Vec<_> = inverse_input
        .iter()
        .map(|values| {
            c2c_transform(
                c2c_transform(values.clone(), shape, ops, false, false),
                shape,
                ops,
                true,
                false,
            )
        })
        .collect();
    let mut raw_array = plan.allocate_in_place().unwrap();
    fill_complex_view(&mut raw_array, &inverse_input, extra, shape);
    let mut raw_workspace = plan.allocate_in_place_workspace().unwrap();
    plan.forward_in_place(&mut raw_array, &mut raw_workspace)
        .unwrap();
    plan.backward_in_place(&mut raw_array, &mut raw_workspace)
        .unwrap();
    {
        let view = raw_array.view().unwrap();
        check_complex(
            topology.communicator(),
            &in_place_backward,
            extra,
            shape,
            |extra, coordinates| view.get_global(extra, coordinates).copied(),
            if std::mem::size_of::<R>() == 4 {
                3e-3
            } else {
                1e-9
            },
            "mixed C2C in-place backward oracle",
        );
    }
}

fn fill_complex_view<R: TestReal, const N: usize, const M: usize>(
    array: &mut pencil_fft::MixedC2cInPlaceArray<R, N, M>,
    values: &[Vec<C64>],
    extra: &[usize],
    shape: [usize; N],
) {
    let mut view = array.view_mut().unwrap();
    for (batch, values) in values.iter().enumerate() {
        let extra_indices = extra_indices(batch, extra);
        for linear in 0..values.len() {
            let coordinates: [usize; N] = unravel(linear, &shape).try_into().unwrap();
            if let Some(slot) = view.get_global_mut(&extra_indices, coordinates) {
                *slot = Complex::new(
                    <R as TestReal>::from_f64(values[linear].re),
                    <R as TestReal>::from_f64(values[linear].im),
                );
            }
        }
    }
}

fn run_r2c_case<R: TestReal, const N: usize, const M: usize>(
    topology: Arc<MpiTopology<M>>,
    shape: [usize; N],
    extra: &[usize],
    ops: [Op; N],
    boundary: usize,
    method: TransposeMethod,
    permute_dims: bool,
) where
    Complex<R>: mpi::datatype::Equivalence,
{
    let transforms = ops.map(|op| match op {
        Op::None => AxisTransform::None,
        Op::Fft => AxisTransform::Fft,
        Op::Rfft => AxisTransform::Rfft,
        Op::DctII => AxisTransform::R2r(AxisR2rKind::Fftw(R2rKind::DctII)),
        Op::Dht => AxisTransform::R2r(AxisR2rKind::Dht),
    });
    let plan = MixedR2cPlan::<R, N, M>::from_shape_with_layout(
        Arc::clone(&topology),
        shape,
        extra_shape(extra),
        transforms,
        DistributedLayout {
            transpose_method: method,
            permute_dims,
        },
    )
    .unwrap();
    let input = batches(extra, shape, |batch, coordinates| {
        let value = seed(batch, &coordinates);
        C64::new(value.re, 0.0)
    });
    let inverse_real = batches(extra, shape, |batch, coordinates| {
        let value = seed(batch + 17, &coordinates);
        C64::new(value.re, 0.0)
    });
    let mut forward = Vec::new();
    let mut inverse = Vec::new();
    let mut backward = Vec::new();
    let mut reduced_shape = shape;
    reduced_shape[boundary] = shape[boundary] / 2 + 1;
    for values in &input {
        let (expected, _) = mixed_r2c_forward(values.clone(), shape, ops, boundary);
        forward.push(expected);
    }
    for values in &inverse_real {
        let (spectrum, spectrum_shape) = mixed_r2c_forward(values.clone(), shape, ops, boundary);
        let (expected_inverse, _) = mixed_r2c_reverse(
            spectrum.clone(),
            &spectrum_shape,
            shape,
            ops,
            boundary,
            true,
        );
        let (expected_backward, _) =
            mixed_r2c_reverse(spectrum, &spectrum_shape, shape, ops, boundary, false);
        inverse.push(expected_inverse);
        backward.push(expected_backward);
    }
    // The spectrum is generated independently from the inverse real tensor; this is
    // deliberately not a roundtrip expectation.
    let inverse_spectrum_values: Vec<Vec<C64>> = inverse_real
        .iter()
        .map(|values| mixed_r2c_forward(values.clone(), shape, ops, boundary).0)
        .collect();

    let mut source = plan.allocate_input().unwrap();
    fill_real(&mut source, &input, extra, shape);
    let source_before = source.as_slice().to_vec();
    let mut spectrum = plan.allocate_output().unwrap();
    let mut workspace = plan.allocate_workspace().unwrap();
    plan.forward(&source, &mut spectrum, &mut workspace)
        .unwrap();
    check_complex(
        topology.communicator(),
        &forward,
        extra,
        reduced_shape,
        |extra, coordinates| spectrum.get_global(extra, coordinates).copied(),
        if std::mem::size_of::<R>() == 4 {
            4e-3
        } else {
            1e-8
        },
        "mixed R2C forward oracle",
    );
    assert_eq!(source.as_slice(), source_before.as_slice());

    let mut inverse_source = plan.allocate_output().unwrap();
    fill_complex(
        &mut inverse_source,
        &inverse_spectrum_values,
        extra,
        reduced_shape,
    );
    let inverse_source_before = inverse_source.as_slice().to_vec();
    let mut restored = plan.allocate_input().unwrap();
    plan.inverse(&inverse_source, &mut restored, &mut workspace)
        .unwrap();
    assert_eq!(inverse_source.as_slice(), inverse_source_before.as_slice());
    check_real(
        topology.communicator(),
        &inverse,
        extra,
        shape,
        |extra, coordinates| restored.get_global(extra, coordinates).copied(),
        if std::mem::size_of::<R>() == 4 {
            4e-3
        } else {
            1e-8
        },
        "mixed R2C inverse oracle",
    );
    let mut raw = plan.allocate_input().unwrap();
    plan.backward(&inverse_source, &mut raw, &mut workspace)
        .unwrap();
    assert_eq!(inverse_source.as_slice(), inverse_source_before.as_slice());
    check_real(
        topology.communicator(),
        &backward,
        extra,
        shape,
        |extra, coordinates| raw.get_global(extra, coordinates).copied(),
        if std::mem::size_of::<R>() == 4 {
            4e-3
        } else {
            1e-8
        },
        "mixed R2C backward oracle",
    );

    let mut array = plan.allocate_in_place().unwrap();
    {
        let mut view = array.real_view_mut().unwrap();
        fill_real_view(&mut view, &input, extra, shape);
    }
    let mut ip_workspace = plan.allocate_in_place_workspace().unwrap();
    plan.forward_in_place(&mut array, &mut ip_workspace)
        .unwrap();
    {
        let view = array.complex_view().unwrap();
        check_complex(
            topology.communicator(),
            &forward,
            extra,
            reduced_shape,
            |extra, coordinates| view.get_global(extra, coordinates).copied(),
            if std::mem::size_of::<R>() == 4 {
                4e-3
            } else {
                1e-8
            },
            "mixed R2C in-place forward oracle",
        );
    }
    plan.inverse_in_place(&mut array, &mut ip_workspace)
        .unwrap();
    {
        let view = array.real_view().unwrap();
        check_real(
            topology.communicator(),
            &input,
            extra,
            shape,
            |extra, coordinates| view.get_global(extra, coordinates).copied(),
            if std::mem::size_of::<R>() == 4 {
                4e-3
            } else {
                1e-8
            },
            "mixed R2C in-place inverse oracle",
        );
    }
    let mut raw_array = plan.allocate_in_place().unwrap();
    {
        let mut view = raw_array.real_view_mut().unwrap();
        fill_real_view(&mut view, &inverse_real, extra, shape);
    }
    let mut raw_workspace = plan.allocate_in_place_workspace().unwrap();
    plan.forward_in_place(&mut raw_array, &mut raw_workspace)
        .unwrap();
    plan.backward_in_place(&mut raw_array, &mut raw_workspace)
        .unwrap();
    {
        let view = raw_array.real_view().unwrap();
        check_real(
            topology.communicator(),
            &backward,
            extra,
            shape,
            |extra, coordinates| view.get_global(extra, coordinates).copied(),
            if std::mem::size_of::<R>() == 4 {
                4e-3
            } else {
                1e-8
            },
            "mixed R2C in-place backward oracle",
        );
    }
}

fn fill_real_view<R: TestReal, const N: usize, const M: usize>(
    view: &mut pencil_array::PencilArrayViewMut<'_, R, N, M>,
    values: &[Vec<C64>],
    extra: &[usize],
    shape: [usize; N],
) {
    for (batch, values) in values.iter().enumerate() {
        let extra_indices = extra_indices(batch, extra);
        for linear in 0..values.len() {
            let coordinates: [usize; N] = unravel(linear, &shape).try_into().unwrap();
            if let Some(slot) = view.get_global_mut(&extra_indices, coordinates) {
                *slot = <R as TestReal>::from_f64(values[linear].re);
            }
        }
    }
}

fn run_zero_extra<R: TestReal>(topology: Arc<MpiTopology<1>>)
where
    Complex<R>: mpi::datatype::Equivalence,
{
    let plan = MixedR2cPlan::<R, 2, 1>::from_shape(
        topology,
        [2, 3],
        extra_shape(&[0]),
        [AxisTransform::Fft, AxisTransform::Rfft],
    )
    .unwrap();
    let source = plan.allocate_input().unwrap();
    let mut output = plan.allocate_output().unwrap();
    let mut workspace = plan.allocate_workspace().unwrap();
    plan.forward(&source, &mut output, &mut workspace).unwrap();
    let mut restored = plan.allocate_input().unwrap();
    plan.inverse(&output, &mut restored, &mut workspace)
        .unwrap();
    let mut array = plan.allocate_in_place().unwrap();
    let mut ip_workspace = plan.allocate_in_place_workspace().unwrap();
    plan.forward_in_place(&mut array, &mut ip_workspace)
        .unwrap();
    plan.inverse_in_place(&mut array, &mut ip_workspace)
        .unwrap();
}

fn run_mixed_tensor_axis_oracles(world: &mpi::topology::SimpleCommunicator) {
    assert!(matches!(world.size(), 1 | 4 | 6));
    let size = usize::try_from(world.size()).unwrap();
    let topology_one = Arc::new(MpiTopology::<1>::new(world, [size]).unwrap());
    let grid = match size {
        1 => [1, 1],
        4 => [2, 2],
        6 => [2, 3],
        _ => unreachable!(),
    };
    let topology_two = Arc::new(MpiTopology::<2>::new(world, grid).unwrap());
    let methods = [TransposeMethod::AllToAllv, TransposeMethod::PointToPoint];
    for method in methods {
        for permute_dims in [false, true] {
            run_c2c_case::<f32, 3, 1>(
                Arc::clone(&topology_one),
                [2, 3, 4],
                &[2],
                [Op::Fft, Op::DctII, Op::Dht],
                method,
                permute_dims,
            );
            run_r2c_case::<f32, 3, 1>(
                Arc::clone(&topology_one),
                [2, 3, 4],
                &[2],
                [Op::Fft, Op::Rfft, Op::Dht],
                1,
                method,
                permute_dims,
            );
            run_c2c_case::<f64, 3, 1>(
                Arc::clone(&topology_one),
                [2, 3, 4],
                &[2],
                [Op::Fft, Op::DctII, Op::Dht],
                method,
                permute_dims,
            );
            run_r2c_case::<f64, 3, 1>(
                Arc::clone(&topology_one),
                [2, 3, 4],
                &[2],
                [Op::Fft, Op::Rfft, Op::Dht],
                1,
                method,
                permute_dims,
            );
            run_r2c_case::<f64, 3, 1>(
                Arc::clone(&topology_one),
                [2, 4, 3],
                &[2],
                [Op::Fft, Op::Rfft, Op::Dht],
                1,
                method,
                permute_dims,
            );
            run_c2c_case::<f64, 4, 2>(
                Arc::clone(&topology_two),
                [1, 3, 2, 4],
                &[2],
                [Op::Fft, Op::DctII, Op::Dht, Op::None],
                method,
                permute_dims,
            );
            run_r2c_case::<f64, 4, 2>(
                Arc::clone(&topology_two),
                [1, 3, 2, 4],
                &[2],
                [Op::Fft, Op::Rfft, Op::Dht, Op::None],
                1,
                method,
                permute_dims,
            );
        }
    }
    run_zero_extra::<f64>(Arc::clone(&topology_one));
}

#[derive(Clone, Copy)]
enum NonfiniteComponent {
    Real(f64),
    Imaginary(f64),
}

fn reject_invalid_endpoint(
    plan: &MixedR2cPlan<f64, 2, 1>,
    coordinate: [usize; 2],
    component: NonfiniteComponent,
    world: &mpi::topology::SimpleCommunicator,
) {
    let mut input = plan.allocate_input().unwrap();
    input.as_mut_slice().fill(0.25);
    let mut spectrum = plan.allocate_output().unwrap();
    let mut workspace = plan.allocate_workspace().unwrap();
    plan.forward(&input, &mut spectrum, &mut workspace).unwrap();
    let mut destination = plan.allocate_input().unwrap();
    destination.as_mut_slice().fill(7.0);
    let destination_before = destination.as_slice().to_vec();
    let mut owned = 0i32;
    if let Some(slot) = spectrum.get_global_mut(&[], coordinate) {
        owned = 1;
        match component {
            NonfiniteComponent::Real(value) => slot.re = value,
            NonfiniteComponent::Imaginary(value) => slot.im = value,
        }
    }
    let mut global_owned = 0i32;
    world.all_reduce_into(&owned, &mut global_owned, SystemOperation::sum());
    assert_eq!(global_owned, 1);
    let spectrum_before: Vec<_> = spectrum
        .as_slice()
        .iter()
        .map(|value| (value.re.to_bits(), value.im.to_bits()))
        .collect();
    assert!(matches!(
        plan.inverse(&spectrum, &mut destination, &mut workspace),
        Err(MixedError::InvalidSpectrum)
    ));
    let spectrum_after: Vec<_> = spectrum
        .as_slice()
        .iter()
        .map(|value| (value.re.to_bits(), value.im.to_bits()))
        .collect();
    assert_eq!(spectrum_after, spectrum_before);
    assert_eq!(destination.as_slice(), destination_before.as_slice());
}

fn is_descriptor_mismatch(error: &MixedError) -> bool {
    matches!(
        error,
        MixedError::Fft(FftError::CollectiveDescriptorMismatch)
    )
}

#[test]
fn mixed_tensor_oracles_and_collective_failures() {
    let universe = mpi::initialize().expect("MPI initialization failed");
    let world = universe.world();
    assert!(matches!(world.size(), 1 | 4 | 6));
    let size = usize::try_from(world.size()).unwrap();
    let topology = Arc::new(MpiTopology::<1>::new(&world, [size]).unwrap());
    let rank = world.rank();
    run_mixed_tensor_axis_oracles(&world);

    if size > 1 {
        let mismatched = if rank == 0 {
            [AxisTransform::Fft, AxisTransform::R2r(AxisR2rKind::Dht)]
        } else {
            [AxisTransform::Fft, AxisTransform::Fft]
        };
        let result = MixedC2cPlan::<f64, 2, 1>::from_shape(
            Arc::clone(&topology),
            [2, 3],
            ExtraShape::scalar(),
            mismatched,
        );
        assert!(result.as_ref().is_err_and(is_descriptor_mismatch));
    }

    let good_c2c = MixedC2cPlan::<f64, 2, 1>::from_shape(
        Arc::clone(&topology),
        [2, 3],
        ExtraShape::scalar(),
        [AxisTransform::Fft, AxisTransform::R2r(AxisR2rKind::Dht)],
    )
    .unwrap();
    let foreign_c2c = MixedC2cPlan::<f64, 2, 1>::from_shape(
        Arc::clone(&topology),
        [2, 3],
        ExtraShape::scalar(),
        [
            AxisTransform::Fft,
            AxisTransform::R2r(AxisR2rKind::Fftw(R2rKind::DctII)),
        ],
    )
    .unwrap();
    let mut source = good_c2c.allocate_input().unwrap();
    source.as_mut_slice().fill(Complex::new(1.0, -0.5));
    let mut destination = good_c2c.allocate_output().unwrap();
    let mut bad_workspace = if rank == 0 {
        foreign_c2c.allocate_workspace().unwrap()
    } else {
        good_c2c.allocate_workspace().unwrap()
    };
    let source_before = source.as_slice().to_vec();
    let destination_before = destination.as_slice().to_vec();
    let workspace_before = format!("{bad_workspace:?}");
    let result = good_c2c.forward(&source, &mut destination, &mut bad_workspace);
    assert!(
        matches!(result, Err(MixedError::Fft(FftError::WorkspaceMismatch)))
            || matches!(
                result,
                Err(MixedError::Fft(FftError::CollectivePreconditionFailed))
            )
    );
    assert_eq!(source.as_slice(), source_before.as_slice());
    assert_eq!(destination.as_slice(), destination_before.as_slice());
    assert_eq!(format!("{bad_workspace:?}"), workspace_before);
    let mut workspace = good_c2c.allocate_workspace().unwrap();
    good_c2c
        .forward(&source, &mut destination, &mut workspace)
        .unwrap();

    let bad_graph = MixedR2cPlan::<f64, 2, 1>::from_shape(
        Arc::clone(&topology),
        [2, 3],
        ExtraShape::scalar(),
        [AxisTransform::Rfft, AxisTransform::Rfft],
    );
    assert!(matches!(bad_graph, Err(MixedError::InvalidGraph)));
    let good_r2c = MixedR2cPlan::<f64, 2, 1>::from_shape(
        Arc::clone(&topology),
        [2, 3],
        ExtraShape::scalar(),
        [AxisTransform::None, AxisTransform::Rfft],
    )
    .unwrap();
    let foreign_r2c = MixedR2cPlan::<f64, 2, 1>::from_shape(
        Arc::clone(&topology),
        [2, 3],
        ExtraShape::scalar(),
        [AxisTransform::Fft, AxisTransform::Rfft],
    )
    .unwrap();
    let mut real = good_r2c.allocate_input().unwrap();
    real.as_mut_slice().fill(0.75);
    let mut spectrum = good_r2c.allocate_output().unwrap();
    let mut bad_r2c_workspace = if rank == 0 {
        foreign_r2c.allocate_workspace().unwrap()
    } else {
        good_r2c.allocate_workspace().unwrap()
    };
    let real_before = real.as_slice().to_vec();
    let spectrum_before = spectrum.as_slice().to_vec();
    let workspace_before = format!("{bad_r2c_workspace:?}");
    let result = good_r2c.forward(&real, &mut spectrum, &mut bad_r2c_workspace);
    assert!(
        matches!(result, Err(MixedError::Fft(FftError::WorkspaceMismatch)))
            || matches!(
                result,
                Err(MixedError::Fft(FftError::CollectivePreconditionFailed))
            )
    );
    assert_eq!(real.as_slice(), real_before.as_slice());
    assert_eq!(spectrum.as_slice(), spectrum_before.as_slice());
    assert_eq!(format!("{bad_r2c_workspace:?}"), workspace_before);
    let mut workspace = good_r2c.allocate_workspace().unwrap();
    good_r2c
        .forward(&real, &mut spectrum, &mut workspace)
        .unwrap();

    let local_mutator = if spectrum.is_empty() { i32::MAX } else { rank };
    let mut mutator = i32::MAX;
    world.all_reduce_into(&local_mutator, &mut mutator, SystemOperation::min());
    for value in spectrum.as_mut_slice() {
        if rank == mutator {
            value.im = f64::NAN;
        }
    }
    let mut restored = good_r2c.allocate_input().unwrap();
    let result = good_r2c.inverse(&spectrum, &mut restored, &mut workspace);
    assert!(matches!(result, Err(MixedError::InvalidSpectrum)));

    reject_invalid_endpoint(
        &good_r2c,
        [0, 0],
        NonfiniteComponent::Real(f64::NAN),
        &world,
    );
    reject_invalid_endpoint(
        &good_r2c,
        [0, 0],
        NonfiniteComponent::Imaginary(f64::INFINITY),
        &world,
    );
    let even_r2c = MixedR2cPlan::<f64, 2, 1>::from_shape(
        Arc::clone(&topology),
        [2, 4],
        ExtraShape::scalar(),
        [AxisTransform::None, AxisTransform::Rfft],
    )
    .unwrap();
    reject_invalid_endpoint(
        &even_r2c,
        [0, 0],
        NonfiniteComponent::Real(f64::NEG_INFINITY),
        &world,
    );
    reject_invalid_endpoint(
        &even_r2c,
        [0, 0],
        NonfiniteComponent::Imaginary(f64::NAN),
        &world,
    );
    reject_invalid_endpoint(
        &even_r2c,
        [0, 2],
        NonfiniteComponent::Real(f64::NAN),
        &world,
    );
    reject_invalid_endpoint(
        &even_r2c,
        [0, 2],
        NonfiniteComponent::Imaginary(f64::NEG_INFINITY),
        &world,
    );

    let mut poisoned = even_r2c.allocate_in_place().unwrap();
    let mut poisoned_workspace = even_r2c.allocate_in_place_workspace().unwrap();
    even_r2c
        .forward_in_place(&mut poisoned, &mut poisoned_workspace)
        .unwrap();
    let mut owned = 0i32;
    if let Some(value) = poisoned
        .complex_view_mut()
        .unwrap()
        .get_global_mut(&[], [0, 2])
    {
        owned = 1;
        value.im = f64::INFINITY;
    }
    let mut global_owned = 0i32;
    world.all_reduce_into(&owned, &mut global_owned, SystemOperation::sum());
    assert_eq!(global_owned, 1);
    assert!(matches!(
        even_r2c.inverse_in_place(&mut poisoned, &mut poisoned_workspace),
        Err(MixedError::InvalidSpectrum)
    ));
    assert_eq!(poisoned.state(), R2cState::Poisoned);
    assert!(matches!(
        poisoned.complex_view(),
        Err(MixedError::Fft(FftError::Array(
            pencil_array::ArrayError::Poisoned
        )))
    ));
    assert!(matches!(
        poisoned.real_view(),
        Err(MixedError::Fft(FftError::Array(
            pencil_array::ArrayError::Poisoned
        )))
    ));
    for direction in [0, 1, 2] {
        let result = match direction {
            0 => even_r2c.forward_in_place(&mut poisoned, &mut poisoned_workspace),
            1 => even_r2c.inverse_in_place(&mut poisoned, &mut poisoned_workspace),
            _ => even_r2c.backward_in_place(&mut poisoned, &mut poisoned_workspace),
        };
        assert!(matches!(
            result,
            Err(MixedError::Fft(FftError::Array(
                pencil_array::ArrayError::Poisoned
            )))
        ));
    }

    good_r2c
        .forward(&real, &mut spectrum, &mut workspace)
        .unwrap();
    for value in spectrum.as_mut_slice() {
        if rank == mutator {
            value.im = 1.0;
        }
    }
    let result = good_r2c.inverse(&spectrum, &mut restored, &mut workspace);
    assert!(matches!(result, Err(MixedError::InvalidSpectrum)));

    good_r2c
        .forward(&real, &mut spectrum, &mut workspace)
        .unwrap();
    good_r2c
        .inverse(&spectrum, &mut restored, &mut workspace)
        .unwrap();
}
