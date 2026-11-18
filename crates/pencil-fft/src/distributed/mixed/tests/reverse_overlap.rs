use super::*;
use mpi::topology::Communicator;
use std::cell::Cell;
thread_local! { static CALLBACKS: Cell<usize> = const { Cell::new(0) }; }

pub(super) fn callback() -> Result<(), MixedError> {
    CALLBACKS.with(|count| count.set(count.get() + 1));
    super::super::consume_c2c_callback_injection().map_err(MixedError::Fft)
}

fn endpoint_threshold_probes<R: FftReal, const N: usize, const M: usize>(
    plan: &MixedR2cPlan<R, N, M>,
    extra_indices: &[usize],
    normalize: bool,
    depth: f64,
    factor: f64,
) where
    Complex<R>: Equivalence,
{
    let core = &plan.core;
    let stage = &core.stages[core.real_stage_index];
    let pencil = &stage.output;
    let relative = 128.0 * <R as crate::private::Sealed>::pencil_fft_epsilon_f64() * depth;
    let absolute = 128.0
        * <R as crate::private::Sealed>::pencil_fft_min_subnormal_f64()
        * depth
        * if normalize { 1.0 } else { factor };
    let mut data = PencilArray::from_elem(
        Arc::clone(pencil),
        core.extra_shape.clone(),
        zero_complex::<R>(),
    )
    .unwrap();
    let plane_count = if core.real_len.is_multiple_of(2) {
        2
    } else {
        1
    };
    for plane in 0..plane_count {
        let mut position = [0; N];
        position[stage.axis] = if plane == 0 { 0 } else { core.complex_len - 1 };
        for (re, threshold) in [(1.0, relative), (0.0, absolute)] {
            let neighbors = endpoint_ulp_neighbors::<R>(threshold);
            for (im, accepted) in neighbors.into_iter().zip([true, true, false]) {
                // One globally nonzero entry avoids norm-summation roundoff.
                data.as_mut_slice().fill(zero_complex::<R>());
                if let Some(value) = data.get_global_mut(extra_indices, position) {
                    *value = Complex::new(R::from_f64(re).unwrap(), R::from_f64(im).unwrap());
                }
                let result =
                    validate_mixed_boundary_data(core, Ok((pencil, data.as_slice())), normalize);
                assert_eq!(
                    result.is_ok(),
                    accepted,
                    "real_len={} batch={extra_indices:?} plane={plane} normalize={normalize} re={re} im={im}: {result:?}",
                    core.real_len,
                );
                if !accepted {
                    assert!(matches!(result, Err(MixedError::InvalidSpectrum)));
                }
            }
        }
    }
}

// Direct boundary data avoids FFT roundoff and an expected-value self-oracle.
fn endpoint_contract<R: FftReal>(topology: &Arc<MpiTopology<2>>, backend: BackendChoice)
where
    Complex<R>: Equivalence,
{
    for real_len in [1, 7, 8] {
        let plan = MixedR2cPlan::<R, 4, 2>::from_shape_with_layout(
            Arc::clone(topology),
            [6, 12, 8, real_len],
            ExtraShape::scalar(),
            [
                AxisTransform::Fft,
                AxisTransform::R2r(AxisR2rKind::Fftw(crate::R2rKind::DctIV)),
                AxisTransform::R2r(AxisR2rKind::Dht),
                AxisTransform::Rfft,
            ],
            DistributedLayout::default(),
        )
        .unwrap();
        let plan = match backend {
            BackendChoice::RustFft => plan,
            #[cfg(feature = "fftw")]
            BackendChoice::Fftw(options) => plan.with_fftw(options).unwrap(),
        };
        // DCT-IV: E=8*12=96, factor=2*12=24; DHT: E=factor=8;
        // FFT: E=factor=6. Ceil logs are 7, 3, 3, respectively.
        let (depth, factor) = (18.0, 1152.0);
        assert_eq!(mixed_endpoint_policy(&plan.core).unwrap(), (depth, factor));
        let epsilon = <R as crate::private::Sealed>::pencil_fft_epsilon_f64();
        let min_subnormal = <R as crate::private::Sealed>::pencil_fft_min_subnormal_f64();
        let pencil = &plan.core.stages[plan.core.real_stage_index].output;
        for normalize in [true, false] {
            let absolute = 128.0 * min_subnormal * depth * if normalize { 1.0 } else { factor };
            if !normalize {
                assert_eq!(plan.core.raw_absolute_threshold, absolute);
            }
            endpoint_threshold_probes(&plan, &[], normalize, depth, factor);
            for (re, im, accepted) in [
                (1.0, 0.99 * 128.0 * epsilon * depth, true),
                (1.0, 1.01 * 128.0 * epsilon * depth, false),
                (0.0, absolute, true),
                (0.0, 2.0 * absolute, false),
                (f64::INFINITY, 0.0, false),
                (0.0, f64::NAN, false),
            ] {
                let data = vec![
                    Complex::new(R::from_f64(re).unwrap(), R::from_f64(im).unwrap());
                    pencil.local_len()
                ];
                let result =
                    validate_mixed_boundary_data(&plan.core, Ok((pencil, &data)), normalize);
                assert_eq!(
                    result.is_ok(),
                    accepted,
                    "real_len={real_len} normalize={normalize} re={re} im={im}: {result:?}"
                );
                if !accepted {
                    assert!(matches!(result, Err(MixedError::InvalidSpectrum)));
                }
            }
        }
    }
}

fn batched_endpoint_contract<R: FftReal>(topology: &Arc<MpiTopology<2>>, backend: BackendChoice)
where
    Complex<R>: Equivalence,
{
    for real_len in [7, 8] {
        // Execution runs right-to-left: real DCT-IV precedes RFFT;
        // only the complex DHT contributes depth/factor. Identities on both sides.
        let plan = MixedR2cPlan::<R, 5, 2>::from_shape_with_layout(
            Arc::clone(topology),
            [6, 12, real_len, 8, 6],
            ExtraShape::new(vec![2]).unwrap(),
            [
                AxisTransform::None,
                AxisTransform::R2r(AxisR2rKind::Dht),
                AxisTransform::Rfft,
                AxisTransform::R2r(AxisR2rKind::Fftw(crate::R2rKind::DctIV)),
                AxisTransform::None,
            ],
            DistributedLayout::default(),
        )
        .unwrap();
        let plan = match backend {
            BackendChoice::RustFft => plan,
            #[cfg(feature = "fftw")]
            BackendChoice::Fftw(options) => plan.with_fftw(options).unwrap(),
        };
        assert_eq!(mixed_endpoint_policy(&plan.core).unwrap(), (7.0, 12.0));
        let pencil = &plan.core.stages[plan.core.real_stage_index].output;
        let stride = memory_stride(pencil, 2).unwrap();
        let local_len = pencil.local_len();
        let reduced = real_len / 2 + 1;
        for normalize in [true, false] {
            for batch in 0..2 {
                endpoint_threshold_probes(&plan, &[batch], normalize, 7.0, 12.0);
                for bin in [0, reduced - 1] {
                    // Huge clean other planes/batches must not mask this plane's error.
                    let mut data =
                        vec![Complex::new(R::from_f64(1e20).unwrap(), R::zero()); 2 * local_len];
                    for index in 0..local_len {
                        if (index / stride) % reduced == bin {
                            data[batch * local_len + index] =
                                Complex::new(R::zero(), R::from_f64(1.0).unwrap());
                        }
                    }
                    let result =
                        validate_mixed_boundary_data(&plan.core, Ok((pencil, &data)), normalize);
                    if real_len == 7 && bin == reduced - 1 {
                        result.unwrap(); // Odd final bin is not an endpoint.
                    } else {
                        assert!(matches!(result, Err(MixedError::InvalidSpectrum)));
                    }
                }
            }
        }
        for raw in [false, true] {
            let mut source = plan.allocate_output().unwrap();
            source
                .as_mut_slice()
                .fill(Complex::new(R::zero(), R::from_f64(1.0).unwrap()));
            let source_before = source.as_slice().to_vec();
            let mut destination = plan.allocate_input().unwrap();
            destination.as_mut_slice().fill(R::from_f64(7.0).unwrap());
            let before = destination.as_slice().to_vec();
            let mut workspace = plan.allocate_workspace().unwrap();
            let result = if raw {
                plan.backward(&source, &mut destination, &mut workspace)
            } else {
                plan.inverse(&source, &mut destination, &mut workspace)
            };
            assert!(matches!(result, Err(MixedError::InvalidSpectrum)));
            assert_eq!(source.as_slice(), source_before);
            assert_eq!(destination.as_slice(), before);

            let mut array = plan.allocate_in_place().unwrap();
            let mut workspace = plan.allocate_in_place_workspace().unwrap();
            plan.forward_in_place(&mut array, &mut workspace).unwrap();
            array
                .complex_view_mut()
                .unwrap()
                .as_mut_slice()
                .copy_from_slice(source.as_slice());
            let result = if raw {
                plan.backward_in_place(&mut array, &mut workspace)
            } else {
                plan.inverse_in_place(&mut array, &mut workspace)
            };
            assert!(matches!(result, Err(MixedError::InvalidSpectrum)));
            assert_eq!(array.state(), crate::R2cState::Poisoned);
        }
        // Ordinary public IP preflight must preserve the real state and data.
        let mut array = plan.allocate_in_place().unwrap();
        array
            .real_view_mut()
            .unwrap()
            .as_mut_slice()
            .fill(R::from_f64(0.25).unwrap());
        let before = array.real_view().unwrap().as_slice().to_vec();
        let state = array.state();
        let mut workspace = plan.allocate_in_place_workspace().unwrap();
        assert!(plan.inverse_in_place(&mut array, &mut workspace).is_err());
        assert_eq!(array.state(), state);
        assert_eq!(array.real_view().unwrap().as_slice(), before);
    }
}

#[cfg(feature = "fftw")]
fn native_workspace_contract<R: FftReal>(topology: &Arc<MpiTopology<1>>)
where
    Complex<R>: Equivalence,
{
    let rank = topology.communicator().rank();
    let value = R::from_f64(0.25).unwrap();
    let complex_value = Complex::new(value, -value);
    let sentinel = Complex::new(R::from_f64(7.0).unwrap(), R::from_f64(-3.0).unwrap());
    let nan = Complex::new(
        R::from_f64(f64::NAN).unwrap(),
        R::from_f64(-f64::NAN).unwrap(),
    );
    let dct = AxisTransform::R2r(AxisR2rKind::Fftw(crate::R2rKind::DctIV));
    let dht = AxisTransform::R2r(AxisR2rKind::Dht);
    // DCT-IV n=8 previously needed 64; the shared max only shrinks for h=16.
    // The final graph has only FFT/RFFT/identity stages and needs no embedding.
    for (h, required, rust_required) in [(16, 16, 64), (64, 64, 64), (128, 128, 128), (16, 0, 0)] {
        let tail_bits =
            |line: &[Complex<R>]| bytemuck::cast_slice::<_, u8>(&line[required..]).to_vec();
        let assert_short_error = |error: MixedError| match (rank, error) {
            (
                0,
                MixedError::Fft(FftError::WorkspaceTooSmall {
                    kind: "complex embedding",
                    required: actual_required,
                    actual,
                }),
            ) => assert_eq!((actual_required, actual), (required, required - 1)),
            (_, MixedError::Fft(FftError::CollectivePreconditionFailed)) if rank != 0 => {}
            (_, error) => panic!("unexpected rank {rank} embedding preflight error: {error:?}"),
        };
        let transforms = if required == 0 {
            [AxisTransform::Fft, AxisTransform::None]
        } else {
            [dct, dht]
        };
        let rust_plan = MixedC2cPlan::<R, 2, 1>::from_shape(
            Arc::clone(topology),
            [8, h],
            ExtraShape::scalar(),
            transforms,
        )
        .unwrap();
        assert_eq!(rust_plan.core.embedding_len, rust_required);
        assert_eq!(
            rust_plan.allocate_workspace().unwrap().embedding_line.len(),
            rust_required
        );
        assert_eq!(
            rust_plan
                .allocate_in_place_workspace()
                .unwrap()
                .embedding_line
                .len(),
            rust_required
        );
        let plan = rust_plan.with_fftw(PlanOptions::default()).unwrap();
        assert_eq!(
            (plan.core.embedding_len, plan.core.fft_scratch_len),
            (required, 0)
        );
        // Allocate from the rebuilt plan so strict endpoint/core identity stays valid.
        let mut source = plan.allocate_input().unwrap();
        source.as_mut_slice().fill(complex_value);
        let source_before = source.as_slice().to_vec();
        let mut destination = plan.allocate_output().unwrap();
        destination.as_mut_slice().fill(sentinel);
        let mut workspace = plan.allocate_workspace().unwrap();
        let mut array = plan.allocate_in_place().unwrap();
        array.view_mut().unwrap().as_mut_slice().fill(complex_value);
        let mut ip_workspace = plan.allocate_in_place_workspace().unwrap();
        // These are initialized lengths, not allocator capacities.
        assert_eq!(
            (workspace.embedding_line.len(), workspace.fft_scratch.len()),
            (required, 0)
        );
        assert_eq!(
            (
                ip_workspace.embedding_line.len(),
                ip_workspace.fft_scratch.len()
            ),
            (required, 0)
        );
        if h == 16 && required != 0 {
            workspace.embedding_line.fill(sentinel);
            ip_workspace.embedding_line.fill(sentinel);
            if rank == 0 {
                // truncate retains capacity: validation must still reject M-1.
                workspace.embedding_line.truncate(required - 1);
                ip_workspace.embedding_line.truncate(required - 1);
            }
            let destination_before = destination.as_slice().to_vec();
            let workspace_before = format!("{workspace:?}");
            assert_short_error(
                plan.forward(&source, &mut destination, &mut workspace)
                    .unwrap_err(),
            );
            assert_eq!(source.as_slice(), source_before);
            assert_eq!(destination.as_slice(), destination_before);
            assert_eq!(format!("{workspace:?}"), workspace_before);

            let array_before = format!("{array:?}");
            let workspace_before = format!("{ip_workspace:?}");
            assert_short_error(
                plan.forward_in_place(&mut array, &mut ip_workspace)
                    .unwrap_err(),
            );
            assert_eq!(array.state(), C2cState::Input);
            assert_eq!(format!("{array:?}"), array_before);
            assert_eq!(format!("{ip_workspace:?}"), workspace_before);
            workspace
                .embedding_line
                .resize(required, zero_complex::<R>());
            ip_workspace
                .embedding_line
                .resize(required, zero_complex::<R>());
        }
        for padding in [0, 3] {
            workspace.embedding_line.resize(required + padding, nan);
            ip_workspace.embedding_line.resize(required + padding, nan);
            // Only [M..] is a tail: other stages may use the DCT's [n..M).
            let tail = tail_bits(&workspace.embedding_line);
            let ip_tail = tail_bits(&ip_workspace.embedding_line);
            plan.forward(&source, &mut destination, &mut workspace)
                .unwrap();
            assert_eq!(source.as_slice(), source_before);
            assert_eq!(tail_bits(&workspace.embedding_line), tail);
            let spectrum_before = destination.as_slice().to_vec();
            let mut recovered = plan.allocate_input().unwrap();
            for raw in [false, true] {
                if raw {
                    plan.backward(&destination, &mut recovered, &mut workspace)
                        .unwrap();
                } else {
                    plan.inverse(&destination, &mut recovered, &mut workspace)
                        .unwrap();
                }
                assert_eq!(destination.as_slice(), spectrum_before);
                assert_eq!(tail_bits(&workspace.embedding_line), tail);
                array
                    .view_mut()
                    .unwrap()
                    .as_mut_slice()
                    .copy_from_slice(&source_before);
                plan.forward_in_place(&mut array, &mut ip_workspace)
                    .unwrap();
                assert_eq!(array.state(), C2cState::Output);
                assert_eq!(tail_bits(&ip_workspace.embedding_line), ip_tail);
                if raw {
                    plan.backward_in_place(&mut array, &mut ip_workspace)
                        .unwrap();
                } else {
                    plan.inverse_in_place(&mut array, &mut ip_workspace)
                        .unwrap();
                }
                assert_eq!(array.state(), C2cState::Input);
                assert_eq!(tail_bits(&ip_workspace.embedding_line), ip_tail);
            }
        }

        let transforms = if required == 0 {
            [AxisTransform::Fft, AxisTransform::None, AxisTransform::Rfft]
        } else {
            [dct, dht, AxisTransform::Rfft]
        };
        let rust_plan = MixedR2cPlan::<R, 3, 1>::from_shape(
            Arc::clone(topology),
            [8, h, 8],
            ExtraShape::scalar(),
            transforms,
        )
        .unwrap();
        assert_eq!(rust_plan.core.embedding_len, rust_required);
        assert_eq!(
            rust_plan.allocate_workspace().unwrap().embedding_line.len(),
            rust_required
        );
        assert_eq!(
            rust_plan
                .allocate_in_place_workspace()
                .unwrap()
                .embedding_line
                .len(),
            rust_required
        );
        let plan = rust_plan.with_fftw(PlanOptions::default()).unwrap();
        assert_eq!(
            (plan.core.embedding_len, plan.core.fft_scratch_len),
            (required, 0)
        );
        let mut source = plan.allocate_input().unwrap();
        source.as_mut_slice().fill(value);
        let source_before = source.as_slice().to_vec();
        let mut destination = plan.allocate_output().unwrap();
        destination.as_mut_slice().fill(sentinel);
        let mut workspace = plan.allocate_workspace().unwrap();
        let mut array = plan.allocate_in_place().unwrap();
        array.real_view_mut().unwrap().as_mut_slice().fill(value);
        let mut ip_workspace = plan.allocate_in_place_workspace().unwrap();
        assert_eq!(
            (workspace.embedding_line.len(), workspace.fft_scratch.len()),
            (required, 0)
        );
        assert_eq!(
            (
                ip_workspace.embedding_line.len(),
                ip_workspace.fft_scratch.len()
            ),
            (required, 0)
        );
        if h == 16 && required != 0 {
            workspace.embedding_line.fill(sentinel);
            ip_workspace.embedding_line.fill(sentinel);
            if rank == 0 {
                workspace.embedding_line.truncate(required - 1);
                ip_workspace.embedding_line.truncate(required - 1);
            }
            let destination_before = destination.as_slice().to_vec();
            let workspace_before = format!("{workspace:?}");
            assert_short_error(
                plan.forward(&source, &mut destination, &mut workspace)
                    .unwrap_err(),
            );
            assert_eq!(source.as_slice(), source_before);
            assert_eq!(destination.as_slice(), destination_before);
            assert_eq!(format!("{workspace:?}"), workspace_before);

            let array_before = format!("{array:?}");
            let workspace_before = format!("{ip_workspace:?}");
            assert_short_error(
                plan.forward_in_place(&mut array, &mut ip_workspace)
                    .unwrap_err(),
            );
            assert_eq!(array.state(), R2cState::RealInput);
            assert_eq!(format!("{array:?}"), array_before);
            assert_eq!(format!("{ip_workspace:?}"), workspace_before);
            workspace
                .embedding_line
                .resize(required, zero_complex::<R>());
            ip_workspace
                .embedding_line
                .resize(required, zero_complex::<R>());
        }
        for padding in [0, 3] {
            workspace.embedding_line.resize(required + padding, nan);
            ip_workspace.embedding_line.resize(required + padding, nan);
            let tail = tail_bits(&workspace.embedding_line);
            let ip_tail = tail_bits(&ip_workspace.embedding_line);
            plan.forward(&source, &mut destination, &mut workspace)
                .unwrap();
            assert_eq!(source.as_slice(), source_before);
            assert_eq!(tail_bits(&workspace.embedding_line), tail);
            let spectrum_before = destination.as_slice().to_vec();
            let mut recovered = plan.allocate_input().unwrap();
            for raw in [false, true] {
                if raw {
                    plan.backward(&destination, &mut recovered, &mut workspace)
                        .unwrap();
                } else {
                    plan.inverse(&destination, &mut recovered, &mut workspace)
                        .unwrap();
                }
                assert_eq!(destination.as_slice(), spectrum_before);
                assert_eq!(tail_bits(&workspace.embedding_line), tail);
                array
                    .real_view_mut()
                    .unwrap()
                    .as_mut_slice()
                    .copy_from_slice(&source_before);
                plan.forward_in_place(&mut array, &mut ip_workspace)
                    .unwrap();
                assert_eq!(array.state(), R2cState::ComplexOutput);
                assert_eq!(tail_bits(&ip_workspace.embedding_line), ip_tail);
                if raw {
                    plan.backward_in_place(&mut array, &mut ip_workspace)
                        .unwrap();
                } else {
                    plan.inverse_in_place(&mut array, &mut ip_workspace)
                        .unwrap();
                }
                assert_eq!(array.state(), R2cState::RealInput);
                assert_eq!(tail_bits(&ip_workspace.embedding_line), ip_tail);
            }
        }
    }
}

#[cfg(feature = "fftw")]
#[test]
#[ignore = "MPI and native FFTW require a separate test process (1, 4, or 6 ranks)"]
fn native_workspace_capacity_contract() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size() as usize;
    assert!(matches!(size, 1 | 4 | 6));
    let topology = pencil_array::MpiTopology::<1>::new(&world, [size]).unwrap();
    native_workspace_contract::<f32>(&topology);
    native_workspace_contract::<f64>(&topology);
}

#[cfg(feature = "fftw")]
#[test]
#[ignore = "MPI and native FFTW require a separate test process (1, 4, or 6 ranks)"]
fn native_endpoint_policy_contract() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size() as usize;
    assert!(matches!(size, 1 | 4 | 6));
    let topology = pencil_array::MpiTopology::<2>::new(&world, [size, 1]).unwrap();
    let backend = BackendChoice::Fftw(PlanOptions::default());
    endpoint_contract::<f32>(&topology, backend);
    endpoint_contract::<f64>(&topology, backend);
    batched_endpoint_contract::<f32>(&topology, backend);
    batched_endpoint_contract::<f64>(&topology, backend);
}

#[test]
#[ignore = "MPI must be initialized in a separate test process"]
fn reverse_local_routes_run_callbacks_and_independent_spectrum() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size() as usize;
    assert!(matches!(size, 1 | 4 | 6));
    let topology = pencil_array::MpiTopology::<2>::new(&world, [size, 1]).unwrap();
    endpoint_contract::<f32>(&topology, BackendChoice::RustFft);
    endpoint_contract::<f64>(&topology, BackendChoice::RustFft);
    batched_endpoint_contract::<f32>(&topology, BackendChoice::RustFft);
    batched_endpoint_contract::<f64>(&topology, BackendChoice::RustFft);
    for permute_dims in [false, true] {
        for boundary in 0..4 {
            let mut transforms = [AxisTransform::R2r(AxisR2rKind::Dht); 4];
            transforms[..boundary].fill(AxisTransform::Fft);
            transforms[boundary] = AxisTransform::Rfft;
            let plan = MixedR2cPlan::<f64, 4, 2>::from_shape_with_layout(
                Arc::clone(&topology),
                [8, 9, 10, 12],
                ExtraShape::scalar(),
                transforms,
                DistributedLayout {
                    permute_dims,
                    transpose_method: TransposeMethod::PointToPoint,
                },
            )
            .unwrap();
            let callbacks = plan
                .core
                .transitions
                .iter()
                .filter(|t| matches!(t.backward, super::super::C2cTransition::PointToPoint(_)))
                .count();
            assert!(callbacks > 0);
            assert!(
                plan.core.transitions.iter().any(|t| if permute_dims {
                    matches!(t.backward, super::super::C2cTransition::Local(_))
                } else {
                    matches!(t.backward, super::super::C2cTransition::Identity)
                }),
                "must exercise the requested local/identity segment as well"
            );
            let mut source = plan.allocate_output().unwrap();
            source.as_mut_slice().fill(Complex::new(1.0, 0.0));
            let mut destination = plan.allocate_input().unwrap();
            let mut workspace = plan.allocate_workspace().unwrap();
            for raw in [false, true] {
                CALLBACKS.with(|count| count.set(0));
                if raw {
                    plan.backward_with_overlap(&source, &mut destination, &mut workspace)
                        .unwrap();
                } else {
                    plan.inverse_with_overlap(&source, &mut destination, &mut workspace)
                        .unwrap();
                }
                assert_eq!(
                    CALLBACKS.with(Cell::get),
                    callbacks,
                    "each P2P FFT must run inside the actual transpose callback"
                );
                // An independently specified constant half-spectrum is a unit impulse.
                // Raw FFT/DHT normalization is the product of all four extents.
                for i in 0..8 {
                    for j in 0..9 {
                        for k in 0..10 {
                            for l in 0..12 {
                                if let Some(&actual) = destination.get_global(&[], [i, j, k, l]) {
                                    let expected = if [i, j, k, l] == [0; 4] {
                                        if raw { 8640.0 } else { 1.0 }
                                    } else {
                                        0.0
                                    };
                                    assert!(
                                        (actual - expected).abs() < 1e-9,
                                        "boundary={boundary} raw={raw} at {:?}: {actual} != {expected}",
                                        [i, j, k, l]
                                    );
                                }
                            }
                        }
                    }
                }
                assert!(
                    source
                        .as_slice()
                        .iter()
                        .all(|&x| x == Complex::new(1.0, 0.0))
                );
                use super::super::{C2C_CALLBACK_INJECTION, C2cCallbackInjection};
                for injection in [C2cCallbackInjection::Error, C2cCallbackInjection::Panic] {
                    let mut failed_workspace = plan.allocate_workspace().unwrap();
                    if world.rank() == 0 {
                        C2C_CALLBACK_INJECTION.with(|slot| slot.set(Some(injection)));
                    }
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        if raw {
                            plan.backward_with_overlap(
                                &source,
                                &mut destination,
                                &mut failed_workspace,
                            )
                        } else {
                            plan.inverse_with_overlap(
                                &source,
                                &mut destination,
                                &mut failed_workspace,
                            )
                        }
                    }));
                    if matches!(injection, C2cCallbackInjection::Panic) && world.rank() == 0 {
                        assert!(result.is_err());
                    } else {
                        match (injection, world.rank() == 0, result.unwrap().unwrap_err()) {
                            (
                                C2cCallbackInjection::Error,
                                true,
                                FftOverlapError::Overlap(OverlapError::Callback(MixedError::Fft(
                                    FftError::PreparationFailed,
                                ))),
                            ) => {}
                            (
                                C2cCallbackInjection::Error,
                                false,
                                FftOverlapError::Overlap(OverlapError::PeerCallbackFailed),
                            ) => {}
                            (
                                C2cCallbackInjection::Panic,
                                false,
                                FftOverlapError::Overlap(OverlapError::PeerPanicked),
                            ) => {}
                            (_, _, error) => {
                                panic!("unexpected mixed callback result: {error:?}")
                            }
                        }
                    }
                    assert!(
                        source
                            .as_slice()
                            .iter()
                            .all(|&x| x == Complex::new(1.0, 0.0))
                    );
                    assert!(
                        matches!(
                            failed_workspace.intermediate.active_view(),
                            Err(ArrayError::Poisoned)
                        ) || failed_workspace.real_intermediate.as_ref().is_some_and(
                            |real| matches!(real.active_view(), Err(ArrayError::Poisoned))
                        )
                    );
                    world.barrier(); // No extra FFT agreement after callback Err/panic.
                    plan.inverse(&source, &mut destination, &mut workspace)
                        .unwrap();
                }
            }
        }
    }
}

use super::super::{
    ENDPOINT_BATCH_CHUNK, EndpointReductionEvent, endpoint_chunk_count, endpoint_chunk_len,
    endpoint_reduction_log, endpoint_ulp_neighbors, force_endpoint_allocation_failure,
    reset_endpoint_reduction_log,
};

fn chunk_plan<R: FftReal>(
    topology: &Arc<MpiTopology<1>>,
    batch_count: usize,
    real_len: usize,
    layout: DistributedLayout,
) -> MixedR2cPlan<R, 2, 1>
where
    Complex<R>: Equivalence,
{
    MixedR2cPlan::from_shape_with_layout(
        Arc::clone(topology),
        [4, real_len],
        ExtraShape::new(vec![batch_count]).unwrap(),
        [AxisTransform::Fft, AxisTransform::Rfft],
        layout,
    )
    .unwrap()
}

fn boundary_array<R: FftReal>(plan: &MixedR2cPlan<R, 2, 1>) -> PencilArray<Complex<R>, 2, 1>
where
    Complex<R>: Equivalence,
{
    PencilArray::from_elem(
        Arc::clone(&plan.core.stages[plan.core.real_stage_index].output),
        plan.core.extra_shape.clone(),
        zero_complex::<R>(),
    )
    .unwrap()
}

// Only successful validation: three allocations of the bounded length, one
// preparation agreement, then one MAX and one SUM per chunk, then one final
// agreement.
fn chunk_protocol(batch_count: usize) {
    let chunk = batch_count.min(ENDPOINT_BATCH_CHUNK);
    let log = endpoint_reduction_log();
    let mut allocations = Vec::new();
    let mut maxima = Vec::new();
    let mut sums = Vec::new();
    let mut preparations = 0;
    let mut finals = 0;
    for event in &log {
        match *event {
            EndpointReductionEvent::Allocation { slot, length } => {
                assert_eq!(slot, allocations.len(), "batch_count={batch_count}");
                assert_eq!(length, chunk, "batch_count={batch_count}");
                allocations.push(length);
            }
            EndpointReductionEvent::PreparationAgreement => preparations += 1,
            EndpointReductionEvent::MaxReduction { elements } => maxima.push(elements),
            EndpointReductionEvent::SumReduction { elements } => sums.push(elements),
            EndpointReductionEvent::FinalAgreement => finals += 1,
        }
    }
    assert_eq!(allocations.len(), 3, "batch_count={batch_count}");
    assert_eq!(log[3], EndpointReductionEvent::PreparationAgreement);
    assert_eq!(preparations, 1, "batch_count={batch_count}");
    assert_eq!(finals, 1, "batch_count={batch_count}");
    assert_eq!(
        log.last(),
        Some(&EndpointReductionEvent::FinalAgreement),
        "batch_count={batch_count}"
    );
    let expected: Vec<usize> = (0..endpoint_chunk_count(batch_count))
        .map(|index| 4 * endpoint_chunk_len(batch_count, index * ENDPOINT_BATCH_CHUNK))
        .collect();
    assert_eq!(maxima, expected, "MAX chunks for batch_count={batch_count}");
    assert_eq!(sums, expected, "SUM chunks for batch_count={batch_count}");
}

fn assert_no_statistics_reduction(log: &[EndpointReductionEvent]) {
    assert!(
        log.iter().all(|event| !matches!(
            event,
            EndpointReductionEvent::MaxReduction { .. }
                | EndpointReductionEvent::SumReduction { .. }
                | EndpointReductionEvent::FinalAgreement
        )),
        "statistics reductions ran after a preparation failure: {log:?}"
    );
    assert_eq!(
        log.iter()
            .filter(|event| matches!(event, EndpointReductionEvent::PreparationAgreement))
            .count(),
        1,
        "{log:?}"
    );
}

fn chunked_statistics<R: FftReal>(topology: &Arc<MpiTopology<1>>)
where
    Complex<R>: Equivalence,
{
    let chunk = ENDPOINT_BATCH_CHUNK;
    for real_len in [6, 7, 1] {
        for batch_count in [0, 1, chunk - 1, chunk, chunk + 1, 2 * chunk + 1] {
            let plan = chunk_plan::<R>(
                topology,
                batch_count,
                real_len,
                DistributedLayout::default(),
            );
            let pencil = &plan.core.stages[plan.core.real_stage_index].output;
            let data = boundary_array(&plan);
            reset_endpoint_reduction_log();
            validate_mixed_boundary_data(&plan.core, Ok((pencil, data.as_slice())), true).unwrap();
            chunk_protocol(batch_count);
        }
    }
    // A permuted boundary layout keeps the same bounded protocol.
    let batch_count = chunk + 1;
    let plan = chunk_plan::<R>(
        topology,
        batch_count,
        6,
        DistributedLayout {
            permute_dims: true,
            ..DistributedLayout::default()
        },
    );
    let pencil = &plan.core.stages[plan.core.real_stage_index].output;
    let data = boundary_array(&plan);
    reset_endpoint_reduction_log();
    validate_mixed_boundary_data(&plan.core, Ok((pencil, data.as_slice())), true).unwrap();
    chunk_protocol(batch_count);
}

// Existing ULP probes, placed in the first, middle, and last chunk. Values
// are exactly representable per batch, so MPI's reduction order cannot
// change the decision.
fn threshold_fixtures<R: FftReal>(topology: &Arc<MpiTopology<1>>)
where
    Complex<R>: Equivalence,
{
    let chunk = ENDPOINT_BATCH_CHUNK;
    let batch_count = 2 * chunk + 1;
    let plan = chunk_plan::<R>(topology, batch_count, 6, DistributedLayout::default());
    let (depth, factor) = mixed_endpoint_policy(&plan.core).unwrap();
    for batch in [0, chunk - 1, chunk, chunk + 1, 2 * chunk] {
        for normalize in [true, false] {
            endpoint_threshold_probes(&plan, &[batch], normalize, depth, factor);
        }
    }
}

// Four exactly representable unit real parts make the fused SUM exactly 4,
// so the relative threshold is 2*relative. Replacing the SUM with a MAX
// would see 1 and reject the accepted neighbor, so this fixture pins the
// fused sum across batches and ranks.
fn multi_contribution_probes<R: FftReal>(topology: &Arc<MpiTopology<1>>)
where
    Complex<R>: Equivalence,
{
    let plan = MixedR2cPlan::<R, 2, 1>::from_shape(
        Arc::clone(topology),
        [4, 2],
        ExtraShape::scalar(),
        [AxisTransform::None, AxisTransform::Rfft],
    )
    .unwrap();
    let (depth, factor) = mixed_endpoint_policy(&plan.core).unwrap();
    assert_eq!((depth, factor), (1.0, 1.0));
    let pencil = &plan.core.stages[plan.core.real_stage_index].output;
    let relative = 128.0 * <R as crate::private::Sealed>::pencil_fft_epsilon_f64() * depth;
    let inverse_absolute =
        128.0 * <R as crate::private::Sealed>::pencil_fft_min_subnormal_f64() * depth;
    assert_eq!(plan.core.raw_absolute_threshold, inverse_absolute * factor);
    let one = R::from_f64(1.0).unwrap();
    let mut data = PencilArray::from_elem(
        Arc::clone(pencil),
        plan.core.extra_shape.clone(),
        zero_complex::<R>(),
    )
    .unwrap();
    let plane_count = if plan.core.real_len.is_multiple_of(2) {
        2
    } else {
        1
    };
    for normalize in [true, false] {
        let absolute = if normalize {
            inverse_absolute
        } else {
            plan.core.raw_absolute_threshold
        };
        for plane in 0..plane_count {
            let k = if plane == 0 {
                0
            } else {
                plan.core.complex_len - 1
            };
            for (im, accepted) in endpoint_ulp_neighbors::<R>(2.0 * relative)
                .into_iter()
                .zip([true, true, false])
            {
                data.as_mut_slice().fill(zero_complex::<R>());
                for row in 0..4 {
                    if let Some(value) = data.get_global_mut(&[], [row, k]) {
                        *value = Complex::new(one, R::zero());
                    }
                }
                if let Some(value) = data.get_global_mut(&[], [0, k]) {
                    *value = Complex::new(one, R::from_f64(im).unwrap());
                }
                let result = validate_mixed_boundary_data(
                    &plan.core,
                    Ok((pencil, data.as_slice())),
                    normalize,
                );
                assert_eq!(
                    result.is_ok(),
                    accepted,
                    "plane={plane} normalize={normalize} im={im}: {result:?}"
                );
                if !accepted {
                    assert!(matches!(result, Err(MixedError::InvalidSpectrum)));
                }
            }
            // The absolute test sees a single imaginary endpoint.
            for (im, accepted) in endpoint_ulp_neighbors::<R>(absolute)
                .into_iter()
                .zip([true, true, false])
            {
                data.as_mut_slice().fill(zero_complex::<R>());
                if let Some(value) = data.get_global_mut(&[], [0, k]) {
                    *value = Complex::new(R::zero(), R::from_f64(im).unwrap());
                }
                let result = validate_mixed_boundary_data(
                    &plan.core,
                    Ok((pencil, data.as_slice())),
                    normalize,
                );
                assert_eq!(
                    result.is_ok(),
                    accepted,
                    "absolute plane={plane} normalize={normalize} im={im}: {result:?}"
                );
                if !accepted {
                    assert!(matches!(result, Err(MixedError::InvalidSpectrum)));
                }
            }
        }
    }
}

fn invalid_batches<R: FftReal>(topology: &Arc<MpiTopology<1>>)
where
    Complex<R>: Equivalence,
{
    let chunk = ENDPOINT_BATCH_CHUNK;
    let batch_count = 2 * chunk + 1;
    let plan = chunk_plan::<R>(topology, batch_count, 6, DistributedLayout::default());
    let pencil = &plan.core.stages[plan.core.real_stage_index].output;
    let complex_len = plan.core.complex_len;
    for (batch, plane, re, im) in [
        (0, 0, 0.0, 1e30),
        (2 * chunk, 0, 0.0, 1e30),
        (chunk, complex_len - 1, 0.0, 1e30),
        (2 * chunk, 0, f64::NAN, 0.0),
        (0, complex_len - 1, 0.0, f64::INFINITY),
    ] {
        let mut data = boundary_array(&plan);
        for i in 0..4 {
            if let Some(value) = data.get_global_mut(&[batch], [i, plane]) {
                *value = Complex::new(R::from_f64(re).unwrap(), R::from_f64(im).unwrap());
            }
        }
        reset_endpoint_reduction_log();
        let result = validate_mixed_boundary_data(&plan.core, Ok((pencil, data.as_slice())), true);
        assert!(
            matches!(result, Err(MixedError::InvalidSpectrum)),
            "batch={batch} plane={plane} re={re} im={im}: {result:?}"
        );
        // The first or last invalid chunk does not stop later chunks.
        chunk_protocol(batch_count);
    }
}

fn plane_and_batch_separation<R: FftReal>(topology: &Arc<MpiTopology<1>>)
where
    Complex<R>: Equivalence,
{
    let chunk = ENDPOINT_BATCH_CHUNK;
    let batch_count = chunk + 1;
    let plan = chunk_plan::<R>(topology, batch_count, 6, DistributedLayout::default());
    let pencil = &plan.core.stages[plan.core.real_stage_index].output;
    let complex_len = plan.core.complex_len;
    let huge = R::from_f64(1e20).unwrap();
    let small = R::from_f64(1.0).unwrap();
    for (batch, plane) in [
        (0, 0),
        (0, complex_len - 1),
        (chunk, 0),
        (chunk, complex_len - 1),
    ] {
        // Large clean real parts elsewhere must not hide this plane.
        let mut data = boundary_array(&plan);
        data.as_mut_slice().fill(Complex::new(huge, R::zero()));
        for i in 0..4 {
            if let Some(value) = data.get_global_mut(&[batch], [i, plane]) {
                *value = Complex::new(R::zero(), small);
            }
        }
        reset_endpoint_reduction_log();
        let result = validate_mixed_boundary_data(&plan.core, Ok((pencil, data.as_slice())), true);
        assert!(
            matches!(result, Err(MixedError::InvalidSpectrum)),
            "batch={batch} plane={plane}: {result:?}"
        );
        chunk_protocol(batch_count);
    }
    // A uniformly large but real plane is accepted: the relative test is
    // scale invariant, and the absolute test uses its imaginary part.
    let mut data = boundary_array(&plan);
    data.as_mut_slice().fill(Complex::new(huge, R::zero()));
    validate_mixed_boundary_data(&plan.core, Ok((pencil, data.as_slice())), true).unwrap();
    // Signed zero is accepted.
    let mut data = boundary_array(&plan);
    data.as_mut_slice().fill(Complex::new(
        R::from_f64(-0.0).unwrap(),
        R::from_f64(-0.0).unwrap(),
    ));
    validate_mixed_boundary_data(&plan.core, Ok((pencil, data.as_slice())), true).unwrap();
    // Non-endpoint bins stay unconstrained.
    let mut data = boundary_array(&plan);
    for i in 0..4 {
        if let Some(value) = data.get_global_mut(&[chunk], [i, 1]) {
            *value = Complex::new(R::zero(), huge);
        }
    }
    validate_mixed_boundary_data(&plan.core, Ok((pencil, data.as_slice())), true).unwrap();
    // Odd and length-one real axes constrain DC only.
    for real_len in [7, 1] {
        let plan = chunk_plan::<R>(
            topology,
            batch_count,
            real_len,
            DistributedLayout::default(),
        );
        let pencil = &plan.core.stages[plan.core.real_stage_index].output;
        let complex_len = plan.core.complex_len;
        if real_len % 2 == 1 && complex_len > 1 {
            let mut data = boundary_array(&plan);
            for i in 0..4 {
                if let Some(value) = data.get_global_mut(&[batch_count - 1], [i, complex_len - 1]) {
                    *value = Complex::new(R::zero(), huge);
                }
            }
            validate_mixed_boundary_data(&plan.core, Ok((pencil, data.as_slice())), true).unwrap();
        }
        let mut data = boundary_array(&plan);
        for i in 0..4 {
            if let Some(value) = data.get_global_mut(&[batch_count - 1], [i, 0]) {
                *value = Complex::new(R::zero(), huge);
            }
        }
        let result = validate_mixed_boundary_data(&plan.core, Ok((pencil, data.as_slice())), true);
        assert!(
            matches!(result, Err(MixedError::InvalidSpectrum)),
            "real_len={real_len}: {result:?}"
        );
    }
}

fn allocation_and_preparation_failures<R: FftReal>(
    topology: &Arc<MpiTopology<1>>,
    rank: usize,
    size: usize,
) where
    Complex<R>: Equivalence,
{
    let chunk = ENDPOINT_BATCH_CHUNK;
    let batch_count = chunk + 1;
    let plan = chunk_plan::<R>(topology, batch_count, 6, DistributedLayout::default());
    let pencil = &plan.core.stages[plan.core.real_stage_index].output;
    let data = boundary_array(&plan);
    for slot in 0..3 {
        reset_endpoint_reduction_log();
        if rank == 0 {
            force_endpoint_allocation_failure(slot);
        }
        let result = validate_mixed_boundary_data(&plan.core, Ok((pencil, data.as_slice())), true);
        match (rank, result) {
            (0, Err(MixedError::Fft(FftError::AllocationFailed { required }))) => {
                assert_eq!(required, chunk, "slot={slot}")
            }
            (_, Err(MixedError::Fft(FftError::CollectivePreconditionFailed))) => {}
            (_, other) => {
                panic!("unexpected allocation failure result for slot {slot}: {other:?}")
            }
        }
        let log = endpoint_reduction_log();
        assert_no_statistics_reduction(&log);
        // The injecting rank stops at its failing slot; every other rank
        // allocated all three buffers before the preparation agreement.
        let allocated = log
            .iter()
            .filter(|event| matches!(event, EndpointReductionEvent::Allocation { .. }))
            .count();
        assert_eq!(
            allocated,
            if rank == 0 { slot + 1 } else { 3 },
            "slot={slot}: {log:?}"
        );
    }
    // A rank-local input error is agreed before allocation and reduction.
    reset_endpoint_reduction_log();
    let result = if rank == 0 {
        validate_mixed_boundary_data(
            &plan.core,
            Err(MixedError::Fft(FftError::PreparationFailed)),
            true,
        )
    } else {
        validate_mixed_boundary_data(&plan.core, Ok((pencil, data.as_slice())), true)
    };
    match (rank, result) {
        (0, Err(MixedError::Fft(FftError::PreparationFailed))) => {}
        (_, Err(MixedError::Fft(FftError::CollectivePreconditionFailed))) => {}
        (_, other) => panic!("unexpected input failure result: {other:?}"),
    }
    let log = endpoint_reduction_log();
    assert_no_statistics_reduction(&log);
    assert_eq!(
        log.iter()
            .filter(|event| matches!(event, EndpointReductionEvent::Allocation { .. }))
            .count(),
        if rank == 0 { 0 } else { 3 },
        "{log:?}"
    );
    // A short input buffer is a preparation failure on one rank. The last
    // rank owns the tail of the leading axis for every tested rank count,
    // so it always has a nonempty local buffer to shorten.
    let shortener = size - 1;
    assert!(!data.as_slice().is_empty() || rank != shortener);
    reset_endpoint_reduction_log();
    let result = if rank == shortener {
        validate_mixed_boundary_data(
            &plan.core,
            Ok((pencil, &data.as_slice()[..data.as_slice().len() - 1])),
            true,
        )
    } else {
        validate_mixed_boundary_data(&plan.core, Ok((pencil, data.as_slice())), true)
    };
    match (rank, result) {
        (_, Err(MixedError::Fft(FftError::PreparationFailed))) if rank == shortener => {}
        (_, Err(MixedError::Fft(FftError::CollectivePreconditionFailed))) => {}
        (_, other) => panic!("unexpected short input result: {other:?}"),
    }
    let log = endpoint_reduction_log();
    assert_no_statistics_reduction(&log);
    assert_eq!(
        log.iter()
            .filter(|event| matches!(event, EndpointReductionEvent::Allocation { .. }))
            .count(),
        if rank == shortener { 0 } else { 3 },
        "{log:?}"
    );
}

fn public_routes<R: FftReal>(topology: &Arc<MpiTopology<1>>)
where
    Complex<R>: Equivalence,
{
    let chunk = ENDPOINT_BATCH_CHUNK;
    let batch_count = chunk + 1;
    let plan = chunk_plan::<R>(topology, batch_count, 6, DistributedLayout::default());
    let mut source = plan.allocate_output().unwrap();
    source.as_mut_slice().fill(zero_complex::<R>());
    let mut destination = plan.allocate_input().unwrap();
    let mut workspace = plan.allocate_workspace().unwrap();
    for raw in [false, true] {
        reset_endpoint_reduction_log();
        destination.as_mut_slice().fill(R::from_f64(7.0).unwrap());
        if raw {
            plan.backward(&source, &mut destination, &mut workspace)
                .unwrap();
        } else {
            plan.inverse(&source, &mut destination, &mut workspace)
                .unwrap();
        }
        chunk_protocol(batch_count);
    }
    reset_endpoint_reduction_log();
    plan.inverse_with_timing(&source, &mut destination, &mut workspace)
        .unwrap();
    chunk_protocol(batch_count);
    reset_endpoint_reduction_log();
    plan.backward_with_timing(&source, &mut destination, &mut workspace)
        .unwrap();
    chunk_protocol(batch_count);

    let mut array = plan.allocate_in_place().unwrap();
    let mut in_place_workspace = plan.allocate_in_place_workspace().unwrap();
    for raw in [false, true] {
        plan.forward_in_place(&mut array, &mut in_place_workspace)
            .unwrap();
        array
            .complex_view_mut()
            .unwrap()
            .as_mut_slice()
            .fill(zero_complex::<R>());
        reset_endpoint_reduction_log();
        if raw {
            plan.backward_in_place(&mut array, &mut in_place_workspace)
                .unwrap();
        } else {
            plan.inverse_in_place(&mut array, &mut in_place_workspace)
                .unwrap();
        }
        chunk_protocol(batch_count);
    }

    // The overlap route validates inside its boundary callback.
    let overlap_plan = chunk_plan::<R>(
        topology,
        batch_count,
        6,
        DistributedLayout {
            transpose_method: TransposeMethod::PointToPoint,
            permute_dims: false,
        },
    );
    let mut overlap_source = overlap_plan.allocate_output().unwrap();
    overlap_source.as_mut_slice().fill(zero_complex::<R>());
    let mut overlap_destination = overlap_plan.allocate_input().unwrap();
    let mut overlap_workspace = overlap_plan.allocate_workspace().unwrap();
    reset_endpoint_reduction_log();
    overlap_plan
        .inverse_with_overlap(
            &overlap_source,
            &mut overlap_destination,
            &mut overlap_workspace,
        )
        .unwrap();
    chunk_protocol(batch_count);
    reset_endpoint_reduction_log();
    overlap_plan
        .backward_with_overlap(
            &overlap_source,
            &mut overlap_destination,
            &mut overlap_workspace,
        )
        .unwrap();
    chunk_protocol(batch_count);

    // The last batch's DC plane rejects the whole call and preserves the
    // source and real destination.
    let mut source = plan.allocate_output().unwrap();
    source.as_mut_slice().fill(zero_complex::<R>());
    for i in 0..4 {
        if let Some(value) = source.get_global_mut(&[batch_count - 1], [i, 0]) {
            *value = Complex::new(R::zero(), R::from_f64(1e30).unwrap());
        }
    }
    let source_before = source.as_slice().to_vec();
    destination.as_mut_slice().fill(R::from_f64(7.0).unwrap());
    let destination_before = destination.as_slice().to_vec();
    reset_endpoint_reduction_log();
    let result = plan.inverse(&source, &mut destination, &mut workspace);
    assert!(
        matches!(result, Err(MixedError::InvalidSpectrum)),
        "{result:?}"
    );
    chunk_protocol(batch_count);
    assert_eq!(source.as_slice(), source_before);
    assert_eq!(destination.as_slice(), destination_before);

    // In-place reverse poisons the array after the same collective check.
    let mut array = plan.allocate_in_place().unwrap();
    let mut in_place_workspace = plan.allocate_in_place_workspace().unwrap();
    plan.forward_in_place(&mut array, &mut in_place_workspace)
        .unwrap();
    {
        let mut view = array.complex_view_mut().unwrap();
        view.as_mut_slice().fill(zero_complex::<R>());
        for i in 0..4 {
            if let Some(value) = view.get_global_mut(&[batch_count - 1], [i, 0]) {
                *value = Complex::new(R::zero(), R::from_f64(1e30).unwrap());
            }
        }
    }
    reset_endpoint_reduction_log();
    let result = plan.inverse_in_place(&mut array, &mut in_place_workspace);
    assert!(
        matches!(result, Err(MixedError::InvalidSpectrum)),
        "{result:?}"
    );
    chunk_protocol(batch_count);
    assert_eq!(array.state(), crate::R2cState::Poisoned);
}

// A preparation agreement inside a P2P boundary callback fails every rank
// as a callback error: the injecting rank keeps its own error and its peers
// report the collective precondition. The boundary array stays poisoned.
fn overlap_preparation_failures<R: FftReal>(topology: &Arc<MpiTopology<1>>, rank: usize)
where
    Complex<R>: Equivalence,
{
    let chunk = ENDPOINT_BATCH_CHUNK;
    let batch_count = chunk + 1;
    let plan = chunk_plan::<R>(
        topology,
        batch_count,
        6,
        DistributedLayout {
            transpose_method: TransposeMethod::PointToPoint,
            permute_dims: false,
        },
    );
    assert!(matches!(
        plan.core.transitions[plan.core.real_stage_index].backward,
        super::super::C2cTransition::PointToPoint(_)
    ));
    let mut source = plan.allocate_output().unwrap();
    source.as_mut_slice().fill(zero_complex::<R>());
    let source_before = source.as_slice().to_vec();
    let mut destination = plan.allocate_input().unwrap();
    destination.as_mut_slice().fill(R::from_f64(7.0).unwrap());
    let destination_before = destination.as_slice().to_vec();
    for slot in 0..3 {
        let mut workspace = plan.allocate_workspace().unwrap();
        reset_endpoint_reduction_log();
        if rank == 0 {
            force_endpoint_allocation_failure(slot);
        }
        let error = plan
            .inverse_with_overlap(&source, &mut destination, &mut workspace)
            .unwrap_err();
        match (rank, error) {
            (
                0,
                FftOverlapError::Overlap(OverlapError::Callback(MixedError::Fft(
                    FftError::AllocationFailed { required },
                ))),
            ) => assert_eq!(required, chunk, "slot={slot}"),
            (
                _,
                FftOverlapError::Overlap(OverlapError::Callback(MixedError::Fft(
                    FftError::CollectivePreconditionFailed,
                ))),
            ) => {}
            (_, error) => {
                panic!("unexpected overlap allocation failure for slot {slot}: {error:?}")
            }
        }
        let log = endpoint_reduction_log();
        assert_no_statistics_reduction(&log);
        assert_eq!(
            log.iter()
                .filter(|event| matches!(event, EndpointReductionEvent::Allocation { .. }))
                .count(),
            if rank == 0 { slot + 1 } else { 3 },
            "slot={slot}: {log:?}"
        );
        assert!(matches!(
            workspace.intermediate.active_view(),
            Err(ArrayError::Poisoned)
        ));
        assert_eq!(source.as_slice(), source_before);
    }
    // The last batch's DC plane rejects the whole overlap call and keeps
    // the source and real destination.
    let mut source = plan.allocate_output().unwrap();
    source.as_mut_slice().fill(zero_complex::<R>());
    for i in 0..4 {
        if let Some(value) = source.get_global_mut(&[batch_count - 1], [i, 0]) {
            *value = Complex::new(R::zero(), R::from_f64(1e30).unwrap());
        }
    }
    let source_before = source.as_slice().to_vec();
    for raw in [false, true] {
        let mut workspace = plan.allocate_workspace().unwrap();
        destination.as_mut_slice().fill(R::from_f64(7.0).unwrap());
        reset_endpoint_reduction_log();
        let error = if raw {
            plan.backward_with_overlap(&source, &mut destination, &mut workspace)
                .unwrap_err()
        } else {
            plan.inverse_with_overlap(&source, &mut destination, &mut workspace)
                .unwrap_err()
        };
        assert!(
            matches!(
                error,
                FftOverlapError::Overlap(OverlapError::Callback(MixedError::InvalidSpectrum))
            ),
            "{error:?}"
        );
        chunk_protocol(batch_count);
        assert!(matches!(
            workspace.intermediate.active_view(),
            Err(ArrayError::Poisoned)
        ));
        assert_eq!(source.as_slice(), source_before);
        assert_eq!(destination.as_slice(), destination_before);
    }
}

#[test]
#[ignore = "MPI must be initialized in a separate test process"]
fn endpoint_reductions_are_chunked() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size() as usize;
    assert!(matches!(size, 1 | 4 | 6));
    let rank = usize::try_from(world.rank()).unwrap();
    let topology = pencil_array::MpiTopology::<1>::new(&world, [size]).unwrap();
    chunked_statistics::<f64>(&topology);
    threshold_fixtures::<f32>(&topology);
    threshold_fixtures::<f64>(&topology);
    multi_contribution_probes::<f32>(&topology);
    multi_contribution_probes::<f64>(&topology);
    invalid_batches::<f64>(&topology);
    plane_and_batch_separation::<f64>(&topology);
    allocation_and_preparation_failures::<f64>(&topology, rank, size);
    public_routes::<f64>(&topology);
    overlap_preparation_failures::<f64>(&topology, rank);
}
