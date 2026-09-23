#![cfg(all(feature = "distributed", feature = "fftw"))]

use std::{sync::Arc, time::Duration};

use mpi::traits::*;
use pencil_array::{ExtraShape, MpiTopology};
use pencil_fft::{
    AxisR2rKind, AxisSelection, AxisTransform, BackendKind, C2cPlan, Complex, DhtPlan,
    DistributedLayout, FourierDirection, FourierDirections, MixedC2cPlan, MixedR2cPlan,
    PlanOptions, PlanningRigor, R2cPlan, R2rKind, R2rPlan, TransposeMethod,
};

fn options() -> PlanOptions {
    PlanOptions::new(PlanningRigor::Estimate, Some(Duration::from_millis(1)))
        .unwrap()
        .with_threads(2)
        .unwrap()
}

fn assert_options(actual: PlanOptions, expected: PlanOptions) {
    assert_eq!(actual.requested_threads(), expected.requested_threads());
    assert_eq!(actual.rigor(), expected.rigor());
    assert_eq!(actual.time_limit(), expected.time_limit());
}

fn assert_complex_close(actual: &[Complex<f64>], expected: &[Complex<f64>], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert!(
            (a.re - e.re).abs() <= tol * (1.0 + e.re.abs()),
            "real {a:?} != {e:?}"
        );
        assert!(
            (a.im - e.im).abs() <= tol * (1.0 + e.im.abs()),
            "imag {a:?} != {e:?}"
        );
    }
}

fn assert_real_close(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() <= tol * (1.0 + e.abs()), "{a} != {e}");
    }
}

fn assert_values_close<T: Copy + Into<Complex<f64>>>(actual: &[T], expected: &[T]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        let actual = actual.into();
        let expected = expected.into();
        assert!(
            (actual - expected).norm() <= 3e-9 * (1.0 + expected.norm()),
            "{actual} != {expected}"
        );
    }
}

#[test]
#[ignore = "MPI must be launched explicitly by the parent test runner"]
fn fftw_distributed_native_matrix() {
    let universe = mpi::initialize().expect("MPI initialization failed");
    let world = universe.world();
    let size = usize::try_from(world.size()).unwrap();
    assert!(matches!(size, 1 | 4 | 6), "run with 1, 4, or 6 ranks");
    let topology = MpiTopology::<1>::new(&world, [size]).unwrap();
    let opts = options();
    if world.rank() == 0 {
        eprintln!(
            "FFTW native f32: {}",
            pencil_fft::runtime_version::<f32>().unwrap()
        );
        eprintln!(
            "FFTW native f64: {}",
            pencil_fft::runtime_version::<f64>().unwrap()
        );
    }

    // Keep the auxiliary matrix in one macro: every family exposes the same
    // collective surface, while their array element types remain distinct.
    macro_rules! aux {
        ($plan:ident, $rust:ident, $input:ident, $output:ident, $workspace:ident, $rust_input:ident, $rust_output:ident, $rust_workspace:ident, $ip_mut:ident, $ip_input:ident, $ip_output:ident) => {{
            let native_options = $plan.options().unwrap();
            assert_eq!(native_options.rigor(), opts.rigor());
            assert_eq!(native_options.time_limit(), opts.time_limit());
            assert_eq!(native_options.requested_threads(), opts.requested_threads());
            let mut inverse_expected = $plan.allocate_input().unwrap();
            let mut backward_expected = $plan.allocate_input().unwrap();
            $plan
                .inverse(&$output, &mut inverse_expected, &mut $workspace)
                .unwrap();
            $plan
                .backward(&$output, &mut backward_expected, &mut $workspace)
                .unwrap();
            let mut timed_output = $plan.allocate_output().unwrap();
            let timing = $plan
                .forward_with_timing(&$input, &mut timed_output, &mut $workspace)
                .unwrap();
            assert_eq!(timing.stages.len(), 2);
            assert_values_close(timed_output.as_slice(), $output.as_slice());
            let mut inverse = $plan.allocate_input().unwrap();
            let timing = $plan
                .inverse_with_timing(&$output, &mut inverse, &mut $workspace)
                .unwrap();
            assert_eq!(timing.stages.len(), 2);
            assert_values_close(inverse.as_slice(), inverse_expected.as_slice());
            let mut backward = $plan.allocate_input().unwrap();
            let timing = $plan
                .backward_with_timing(&$output, &mut backward, &mut $workspace)
                .unwrap();
            assert_eq!(timing.stages.len(), 2);
            assert_values_close(backward.as_slice(), backward_expected.as_slice());
            let before = timed_output.as_slice().to_vec();
            let forward = $plan.forward_with_overlap(&$input, &mut timed_output, &mut $workspace);
            let reverse = $plan.inverse_with_overlap(&$output, &mut inverse, &mut $workspace);
            let raw = $plan.backward_with_overlap(&$output, &mut backward, &mut $workspace);
            if $plan.layout().transpose_method == TransposeMethod::PointToPoint {
                forward.unwrap();
                reverse.unwrap();
                raw.unwrap();
            } else {
                assert!(forward.is_err() && reverse.is_err() && raw.is_err());
                assert_eq!(timed_output.as_slice(), before.as_slice());
            }
            assert_values_close(timed_output.as_slice(), before.as_slice());
            assert_values_close(inverse.as_slice(), inverse_expected.as_slice());
            assert_values_close(backward.as_slice(), backward_expected.as_slice());

            let before = timed_output.as_slice().to_vec();
            assert!(
                $plan
                    .forward(&$rust_input, &mut timed_output, &mut $workspace)
                    .is_err()
            );
            assert!(
                $plan
                    .forward(&$input, &mut $rust_output, &mut $workspace)
                    .is_err()
            );
            assert!(
                $plan
                    .forward(&$input, &mut timed_output, &mut $rust_workspace)
                    .is_err()
            );
            assert_eq!(timed_output.as_slice(), before.as_slice());

            let mut ip = $plan.allocate_in_place().unwrap();
            ip.$ip_mut().unwrap().as_mut_slice().copy_from_slice($input.as_slice());
            let mut ip_workspace = $plan.allocate_in_place_workspace().unwrap();
            $plan.forward_in_place_with_timing(&mut ip, &mut ip_workspace).unwrap();
            assert_values_close(ip.$ip_output().unwrap().as_slice(), $output.as_slice());
            $plan.inverse_in_place_with_timing(&mut ip, &mut ip_workspace).unwrap();
            assert_values_close(ip.$ip_input().unwrap().as_slice(), inverse_expected.as_slice());
            ip.$ip_mut().unwrap().as_mut_slice().copy_from_slice($input.as_slice());
            $plan.forward_in_place(&mut ip, &mut ip_workspace).unwrap();
            $plan.backward_in_place_with_timing(&mut ip, &mut ip_workspace).unwrap();
            assert_values_close(ip.$ip_input().unwrap().as_slice(), backward_expected.as_slice());
            let mut ip_many = vec![$plan.allocate_in_place().unwrap(), $plan.allocate_in_place().unwrap()];
            for member in &mut ip_many { member.$ip_mut().unwrap().as_mut_slice().copy_from_slice($input.as_slice()); }
            $plan.forward_many_in_place(&mut ip_many, &mut ip_workspace).unwrap();
            for member in &ip_many { assert_values_close(member.$ip_output().unwrap().as_slice(), $output.as_slice()); }
            $plan.inverse_many_in_place(&mut ip_many, &mut ip_workspace).unwrap();
            for member in &ip_many { assert_values_close(member.$ip_input().unwrap().as_slice(), inverse_expected.as_slice()); }
            for member in &mut ip_many { member.$ip_mut().unwrap().as_mut_slice().copy_from_slice($input.as_slice()); }
            $plan.forward_many_in_place(&mut ip_many, &mut ip_workspace).unwrap();
            $plan.backward_many_in_place(&mut ip_many, &mut ip_workspace).unwrap();
            for member in &ip_many { assert_values_close(member.$ip_input().unwrap().as_slice(), backward_expected.as_slice()); }

            let mut second = $plan.allocate_input().unwrap();
            second.as_mut_slice().copy_from_slice($input.as_slice());
            let many_sources = vec![$input, second];
            let mut many_destinations = vec![
                $plan.allocate_output().unwrap(),
                $plan.allocate_output().unwrap(),
            ];
            $plan
                .forward_many(&many_sources, &mut many_destinations, &mut $workspace)
                .unwrap();
            for destination in &many_destinations {
                assert_values_close(destination.as_slice(), $output.as_slice());
            }
            let mut many_inverse = vec![
                $plan.allocate_input().unwrap(),
                $plan.allocate_input().unwrap(),
            ];
            $plan
                .inverse_many(&many_destinations, &mut many_inverse, &mut $workspace)
                .unwrap();
            for destination in &many_inverse {
                assert_values_close(destination.as_slice(), inverse_expected.as_slice());
            }
            $plan
                .backward_many(&many_destinations, &mut many_inverse, &mut $workspace)
                .unwrap();
            for destination in &many_inverse {
                assert_values_close(destination.as_slice(), backward_expected.as_slice());
            }

            let foreign = if world.rank() == 0 {
                $rust.allocate_input().unwrap()
            } else {
                $plan.allocate_input().unwrap()
            };
            let mut first = $plan.allocate_input().unwrap();
            first
                .as_mut_slice()
                .copy_from_slice(many_sources[0].as_slice());
            let foreign_sources = vec![first, foreign];
            let guarded_before: Vec<Vec<_>> = many_destinations
                .iter()
                .map(|x| x.as_slice().to_vec())
                .collect();
            assert!(
                $plan
                    .forward_many(&foreign_sources, &mut many_destinations, &mut $workspace)
                    .is_err()
            );
            for (got, expected) in many_destinations.iter().zip(&guarded_before) {
                assert_eq!(got.as_slice(), expected.as_slice());
            }

            if size > 1 {
                let other = $plan
                    .with_fftw(
                        PlanOptions::new(PlanningRigor::Patient, Some(Duration::from_millis(2)))
                            .unwrap(),
                    )
                    .unwrap();
                let selected = if world.rank() == 0 { &other } else { &$plan };
                let sources = vec![selected.allocate_input().unwrap()];
                let mut destinations = vec![selected.allocate_output().unwrap()];
                let before = destinations[0].as_slice().to_vec();
                assert!(matches!(
                    selected.forward_many(&sources, &mut destinations, &mut $workspace),
                    Err(pencil_fft::CollectionError::HeaderMismatch)
                ));
                assert_eq!(destinations[0].as_slice(), before.as_slice());
            }
            $plan
                .forward_many(&many_sources, &mut many_destinations, &mut $workspace)
                .unwrap();
        }};
    }

    // One binary, one MPI init, both transports and both memory routes.
    for method in [TransposeMethod::AllToAllv, TransposeMethod::PointToPoint] {
        for permute_dims in [false, true] {
            let layout = DistributedLayout {
                transpose_method: method,
                permute_dims,
            };
            let rust = C2cPlan::<f64, 2, 1>::from_shape_with_layout(
                Arc::clone(&topology),
                [4 * size, 3],
                ExtraShape::new([2]).unwrap(),
                layout,
            )
            .unwrap();
            let plan = rust.with_fftw(opts).unwrap();
            assert_eq!(plan.layout(), layout);
            assert_eq!(plan.backend_kind(), BackendKind::Fftw);
            let mut input = plan.allocate_input().unwrap();
            input
                .as_mut_slice()
                .iter_mut()
                .enumerate()
                .for_each(|(i, x)| {
                    *x = Complex::new(i as f64 - 3.0, -(i as f64) - 0.25);
                });
            let original = input.as_slice().to_vec();
            let mut output = plan.allocate_output().unwrap();
            let mut workspace = plan.allocate_out_of_place_workspace().unwrap();
            plan.forward(&input, &mut output, &mut workspace).unwrap();

            // Compare against a separately built Rust plan, not just a
            // round-trip.  A paired inverse can hide a shared sign/layout bug.
            let mut rust_input = rust.allocate_input().unwrap();
            rust_input
                .as_mut_slice()
                .iter_mut()
                .zip(&original)
                .for_each(|(dst, src)| *dst = *src);
            let mut rust_output = rust.allocate_output().unwrap();
            let mut rust_workspace = rust.allocate_out_of_place_workspace().unwrap();
            rust.forward(&rust_input, &mut rust_output, &mut rust_workspace)
                .unwrap();
            assert_complex_close(output.as_slice(), rust_output.as_slice(), 3e-9);

            // Old-core arrays and workspaces must not cross the backend
            // boundary, even though their element types and shapes match.
            assert!(
                plan.forward(&rust_input, &mut output, &mut workspace)
                    .is_err()
            );
            assert!(
                plan.forward(&input, &mut output, &mut rust_workspace)
                    .is_err()
            );
            assert_eq!(input.as_slice(), original.as_slice());
            let timing = plan
                .forward_with_timing(&input, &mut output, &mut workspace)
                .unwrap();
            assert_eq!(timing.stages.len(), 2);
            assert!(timing.total >= Duration::ZERO);
            if method == TransposeMethod::PointToPoint {
                let mut overlap = plan.allocate_output().unwrap();
                plan.forward_with_overlap(&input, &mut overlap, &mut workspace)
                    .unwrap();
            }
            let mut recovered = plan.allocate_input().unwrap();
            plan.inverse(&output, &mut recovered, &mut workspace)
                .unwrap();
            for (a, b) in recovered.as_slice().iter().zip(&original) {
                assert!((a.re - b.re).abs() < 1e-8);
                assert!((a.im - b.im).abs() < 1e-8);
            }

            let mut backward = plan.allocate_input().unwrap();
            plan.backward(&output, &mut backward, &mut workspace)
                .unwrap();
            let scale = (4 * size * 3) as f64;
            for (a, b) in backward.as_slice().iter().zip(&original) {
                assert!((a.re - b.re * scale).abs() < 1e-7 * scale);
                assert!((a.im - b.im * scale).abs() < 1e-7 * scale);
            }
            let mut inplace = plan.allocate_in_place().unwrap();
            inplace
                .view_mut()
                .unwrap()
                .as_mut_slice()
                .copy_from_slice(&original);
            let mut ip_workspace = plan.allocate_in_place_workspace().unwrap();
            plan.forward_in_place(&mut inplace, &mut ip_workspace)
                .unwrap();
            plan.inverse_in_place(&mut inplace, &mut ip_workspace)
                .unwrap();
            assert_complex_close(inplace.view().unwrap().as_slice(), &original, 3e-9);
            plan.forward_in_place(&mut inplace, &mut ip_workspace)
                .unwrap();
            plan.backward_in_place(&mut inplace, &mut ip_workspace)
                .unwrap();
            let scaled: Vec<_> = original.iter().map(|value| value * scale).collect();
            assert_complex_close(inplace.view().unwrap().as_slice(), &scaled, 3e-9);

            // Auxiliary timing/overlap/collection coverage uses the same
            // native plan and deliberately exercises a foreign member on rank 0.
            aux!(
                plan,
                rust,
                input,
                output,
                workspace,
                rust_input,
                rust_output,
                rust_workspace,
                view_mut,
                view,
                view
            );
        }
    }

    // Explicit signs/configuration order are part of the native descriptor.
    let signed = C2cPlan::<f64, 2, 1>::from_shape_with_fft_directions(
        Arc::clone(&topology),
        [4 * size, 3],
        ExtraShape::scalar(),
        FourierDirections::new([FourierDirection::Backward, FourierDirection::Forward]),
    )
    .unwrap()
    .with_fftw(opts)
    .unwrap();
    assert_eq!(
        signed.fft_directions().get(0),
        Some(FourierDirection::Backward)
    );
    assert_eq!(signed.options().unwrap().requested_threads(), 2);
    let native_first = signed
        .with_fft_directions(FourierDirections::new([
            FourierDirection::Forward,
            FourierDirection::Backward,
        ]))
        .unwrap();
    assert_eq!(native_first.backend_kind(), BackendKind::Fftw);
    assert_options(native_first.options().unwrap(), opts);
    assert_eq!(
        native_first.fft_directions().get(1),
        Some(FourierDirection::Backward)
    );
    assert!(
        signed
            .with_fftw(PlanOptions::new(PlanningRigor::Measure, None).unwrap())
            .is_ok()
    );

    let partial_rust = C2cPlan::<f64, 2, 1>::from_shape_with_selection_and_layout(
        Arc::clone(&topology),
        [4 * size, 3],
        ExtraShape::scalar(),
        AxisSelection::from_indices([1]).unwrap(),
        DistributedLayout::default(),
    )
    .unwrap();
    let partial = partial_rust.with_fftw(opts).unwrap();
    assert_eq!(partial.backend_kind(), BackendKind::Fftw);
    let mut partial_input = partial.allocate_input().unwrap();
    partial_input
        .as_mut_slice()
        .iter_mut()
        .enumerate()
        .for_each(|(i, x)| {
            *x = Complex::new(i as f64 * 0.19 + 0.1, -(i as f64) * 0.03);
        });
    let mut partial_output = partial.allocate_output().unwrap();
    let mut partial_workspace = partial.allocate_out_of_place_workspace().unwrap();
    partial
        .forward(&partial_input, &mut partial_output, &mut partial_workspace)
        .unwrap();
    let mut partial_expected = partial_rust.allocate_output().unwrap();
    let mut partial_rust_workspace = partial_rust.allocate_out_of_place_workspace().unwrap();
    partial_rust
        .forward(
            &partial_input,
            &mut partial_expected,
            &mut partial_rust_workspace,
        )
        .unwrap();
    assert_complex_close(partial_output.as_slice(), partial_expected.as_slice(), 3e-9);

    // Descriptor mismatches are rejected collectively; a matching rebuild
    // proves the communicator and native loader remain recoverable.
    if size > 1 {
        let mismatched_rigor = PlanOptions::new(
            if world.rank() == 0 {
                PlanningRigor::Measure
            } else {
                PlanningRigor::Estimate
            },
            None,
        )
        .unwrap();
        assert!(
            C2cPlan::<f64, 2, 1>::from_shape_with_fftw(
                Arc::clone(&topology),
                [4 * size, 3],
                ExtraShape::scalar(),
                mismatched_rigor,
            )
            .is_err()
        );
        world.barrier();
        let mismatched_time = PlanOptions::new(
            PlanningRigor::Estimate,
            Some(Duration::from_millis(if world.rank() == 0 { 1 } else { 2 })),
        )
        .unwrap();
        assert!(
            C2cPlan::<f64, 2, 1>::from_shape_with_fftw(
                Arc::clone(&topology),
                [4 * size, 3],
                ExtraShape::scalar(),
                mismatched_time,
            )
            .is_err()
        );
        world.barrier();
        let mismatched_none_some = PlanOptions::new(
            PlanningRigor::Estimate,
            if world.rank() == 0 {
                None
            } else {
                Some(Duration::from_nanos(1))
            },
        )
        .unwrap();
        assert!(
            C2cPlan::<f64, 2, 1>::from_shape_with_fftw(
                Arc::clone(&topology),
                [4 * size, 3],
                ExtraShape::scalar(),
                mismatched_none_some,
            )
            .is_err()
        );
        world.barrier();
        assert!(
            C2cPlan::<f64, 2, 1>::from_shape_with_fftw(
                Arc::clone(&topology),
                [4 * size, 3],
                ExtraShape::scalar(),
                opts,
            )
            .is_ok()
        );
    }

    // Rank-local thread choices must be rejected before native planning, and a
    // matching retry must still support workspace reuse.
    if size > 1 {
        for bad_rank in [1, 2] {
            if bad_rank < size {
                let rank_opts =
                    PlanOptions::new(PlanningRigor::Estimate, Some(Duration::from_millis(1)))
                        .unwrap()
                        .with_threads(if world.rank() == bad_rank as i32 {
                            1
                        } else {
                            2
                        })
                        .unwrap();
                macro_rules! reject {
                    ($call:expr) => {{
                        assert!($call.is_err());
                        world.barrier();
                    }};
                }
                reject!(C2cPlan::<f64, 2, 1>::from_shape_with_fftw(
                    Arc::clone(&topology),
                    [4 * size, 3],
                    ExtraShape::scalar(),
                    rank_opts
                ));
                reject!(R2cPlan::<f64, 2, 1>::from_shape_with_fftw(
                    Arc::clone(&topology),
                    [4 * size, 3],
                    ExtraShape::scalar(),
                    rank_opts
                ));
                reject!(R2rPlan::<f64, 2, 1>::from_shape_with_fftw(
                    Arc::clone(&topology),
                    [4 * size, 3],
                    ExtraShape::scalar(),
                    [Some(R2rKind::DctII), Some(R2rKind::DstIII)],
                    rank_opts
                ));
                reject!(DhtPlan::<f64, 2, 1>::from_shape_with_fftw(
                    Arc::clone(&topology),
                    [4 * size, 3],
                    ExtraShape::scalar(),
                    rank_opts
                ));
                reject!(MixedC2cPlan::<f64, 2, 1>::from_shape_with_fftw(
                    Arc::clone(&topology),
                    [4 * size, 3],
                    ExtraShape::scalar(),
                    [
                        AxisTransform::Fft,
                        AxisTransform::R2r(AxisR2rKind::Fftw(R2rKind::DctII))
                    ],
                    rank_opts
                ));
                reject!(MixedR2cPlan::<f64, 2, 1>::from_shape_with_fftw(
                    Arc::clone(&topology),
                    [4 * size, 3],
                    ExtraShape::scalar(),
                    [AxisTransform::None, AxisTransform::Rfft],
                    rank_opts
                ));
                let retry = C2cPlan::<f64, 2, 1>::from_shape_with_fftw(
                    Arc::clone(&topology),
                    [4 * size, 3],
                    ExtraShape::scalar(),
                    opts,
                )
                .unwrap();
                let input = retry.allocate_input().unwrap();
                let mut output = retry.allocate_output().unwrap();
                let mut workspace = retry.allocate_out_of_place_workspace().unwrap();
                retry.forward(&input, &mut output, &mut workspace).unwrap();
                retry.forward(&input, &mut output, &mut workspace).unwrap();
                world.barrier();
            }
        }
    }

    for method in [TransposeMethod::AllToAllv, TransposeMethod::PointToPoint] {
        for permute_dims in [false, true] {
            let layout = DistributedLayout {
                transpose_method: method,
                permute_dims,
            };
            // The remaining native families are constructed and executed in the same
            // process/communicator, catching backend, rank, and descriptor regressions.
            let r2c = R2cPlan::<f64, 2, 1>::from_shape_with_layout(
                Arc::clone(&topology),
                [4 * size, 3],
                ExtraShape::scalar(),
                layout,
            )
            .unwrap()
            .with_fftw(opts)
            .unwrap();
            assert_eq!(r2c.backend_kind(), BackendKind::Fftw);
            let mut ri = r2c.allocate_input().unwrap();
            ri.as_mut_slice()
                .iter_mut()
                .enumerate()
                .for_each(|(i, x)| *x = i as f64 * 0.17 - 0.3);
            let mut ro = r2c.allocate_output().unwrap();
            let mut r2cw = r2c.allocate_workspace().unwrap();
            r2c.forward(&ri, &mut ro, &mut r2cw).unwrap();
            let rust_r2c = R2cPlan::<f64, 2, 1>::from_shape_with_layout(
                Arc::clone(&topology),
                [4 * size, 3],
                ExtraShape::scalar(),
                layout,
            )
            .unwrap();
            let mut rri = rust_r2c.allocate_input().unwrap();
            rri.as_mut_slice().copy_from_slice(ri.as_slice());
            let mut rro = rust_r2c.allocate_output().unwrap();
            let mut rrw = rust_r2c.allocate_workspace().unwrap();
            rust_r2c.forward(&rri, &mut rro, &mut rrw).unwrap();
            assert_complex_close(ro.as_slice(), rro.as_slice(), 3e-9);
            let r2r = R2rPlan::<f64, 2, 1>::from_shape_with_layout(
                Arc::clone(&topology),
                [4 * size, 3],
                ExtraShape::scalar(),
                [Some(R2rKind::DctII), Some(R2rKind::DstIII)],
                layout,
            )
            .unwrap()
            .with_fftw(opts)
            .unwrap();
            assert_eq!(r2r.backend_kind(), BackendKind::Fftw);
            assert_eq!(r2r.options().unwrap().rigor(), opts.rigor());
            let mut a = r2r.allocate_input().unwrap();
            a.as_mut_slice()
                .iter_mut()
                .enumerate()
                .for_each(|(i, x)| *x = i as f64 * 0.11 + 0.2);
            let mut b = r2r.allocate_output().unwrap();
            let mut w = r2r.allocate_workspace().unwrap();
            r2r.forward(&a, &mut b, &mut w).unwrap();
            let rust_r2r = R2rPlan::<f64, 2, 1>::from_shape_with_layout(
                Arc::clone(&topology),
                [4 * size, 3],
                ExtraShape::scalar(),
                [Some(R2rKind::DctII), Some(R2rKind::DstIII)],
                layout,
            )
            .unwrap();
            let mut ra = rust_r2r.allocate_input().unwrap();
            ra.as_mut_slice().copy_from_slice(a.as_slice());
            let mut rb = rust_r2r.allocate_output().unwrap();
            let mut rw = rust_r2r.allocate_workspace().unwrap();
            rust_r2r.forward(&ra, &mut rb, &mut rw).unwrap();
            assert_real_close(b.as_slice(), rb.as_slice(), 3e-8);
            let dht = DhtPlan::<f64, 2, 1>::from_shape_with_layout(
                Arc::clone(&topology),
                [4 * size, 3],
                ExtraShape::scalar(),
                layout,
            )
            .unwrap()
            .with_fftw(opts)
            .unwrap();
            assert_eq!(dht.backend_kind(), BackendKind::Fftw);
            assert_eq!(dht.options().unwrap().rigor(), opts.rigor());
            let mut da = dht.allocate_input().unwrap();
            da.as_mut_slice()
                .iter_mut()
                .enumerate()
                .for_each(|(i, x)| *x = i as f64 * 0.07 - 0.1);
            let mut db = dht.allocate_output().unwrap();
            let mut dw = dht.allocate_workspace().unwrap();
            dht.forward(&da, &mut db, &mut dw).unwrap();
            let rust_dht = DhtPlan::<f64, 2, 1>::from_shape_with_layout(
                Arc::clone(&topology),
                [4 * size, 3],
                ExtraShape::scalar(),
                layout,
            )
            .unwrap();
            let mut rda = rust_dht.allocate_input().unwrap();
            rda.as_mut_slice().copy_from_slice(da.as_slice());
            let mut rdb = rust_dht.allocate_output().unwrap();
            let mut rdw = rust_dht.allocate_workspace().unwrap();
            rust_dht.forward(&rda, &mut rdb, &mut rdw).unwrap();
            assert_real_close(db.as_slice(), rdb.as_slice(), 3e-8);

            let mc = MixedC2cPlan::<f64, 2, 1>::from_shape_with_layout(
                Arc::clone(&topology),
                [4 * size, 3],
                ExtraShape::scalar(),
                [
                    AxisTransform::Fft,
                    AxisTransform::R2r(AxisR2rKind::Fftw(R2rKind::DctII)),
                ],
                layout,
            )
            .unwrap()
            .with_fftw(opts)
            .unwrap();
            assert_eq!(mc.backend_kind(), BackendKind::Fftw);
            assert_eq!(mc.options().unwrap().rigor(), opts.rigor());
            let mc_signed = mc
                .with_fft_directions(FourierDirections::new([
                    FourierDirection::Backward,
                    FourierDirection::Forward,
                ]))
                .unwrap();
            assert_eq!(mc_signed.backend_kind(), BackendKind::Fftw);
            assert_eq!(mc_signed.options().unwrap().requested_threads(), 2);
            assert_options(mc_signed.options().unwrap(), opts);
            assert_eq!(
                mc_signed.fft_directions().get(0),
                Some(FourierDirection::Backward)
            );
            let mut ma = mc.allocate_input().unwrap();
            ma.as_mut_slice().iter_mut().enumerate().for_each(|(i, x)| {
                *x = Complex::new(i as f64 * 0.13 - 0.4, i as f64 * -0.09 + 0.2)
            });
            let original_mixed = ma.as_slice().to_vec();
            let mut mb = mc.allocate_output().unwrap();
            let mut mw = mc.allocate_workspace().unwrap();
            mc.forward(&ma, &mut mb, &mut mw).unwrap();
            let rust_mc = MixedC2cPlan::<f64, 2, 1>::from_shape_with_layout(
                Arc::clone(&topology),
                [4 * size, 3],
                ExtraShape::scalar(),
                [
                    AxisTransform::Fft,
                    AxisTransform::R2r(AxisR2rKind::Fftw(R2rKind::DctII)),
                ],
                layout,
            )
            .unwrap();
            let directions_first = rust_mc
                .with_fft_directions(mc_signed.fft_directions())
                .unwrap()
                .with_fftw(opts)
                .unwrap();
            assert_eq!(directions_first.backend_kind(), BackendKind::Fftw);
            assert_options(directions_first.options().unwrap(), opts);
            assert_eq!(
                directions_first.fft_directions(),
                mc_signed.fft_directions()
            );
            let mut rma = rust_mc.allocate_input().unwrap();
            rma.as_mut_slice().copy_from_slice(&original_mixed);
            let mut rmb = rust_mc.allocate_output().unwrap();
            let mut rmw = rust_mc.allocate_workspace().unwrap();
            rust_mc.forward(&rma, &mut rmb, &mut rmw).unwrap();
            assert_complex_close(mb.as_slice(), rmb.as_slice(), 3e-8);
            let mixed_directions =
                FourierDirections::new([FourierDirection::Backward, FourierDirection::Forward]);
            let mr = MixedR2cPlan::<f64, 2, 1>::from_shape_with_layout(
                Arc::clone(&topology),
                [4 * size, 3],
                ExtraShape::scalar(),
                [AxisTransform::Fft, AxisTransform::Rfft],
                layout,
            )
            .unwrap()
            .with_fftw(opts)
            .unwrap();
            assert_eq!(mr.backend_kind(), BackendKind::Fftw);
            assert_eq!(mr.options().unwrap().rigor(), opts.rigor());
            let mr_signed = mr.with_fft_directions(mixed_directions).unwrap();
            assert_eq!(mr_signed.fft_directions(), mixed_directions);
            assert_eq!(mr_signed.backend_kind(), BackendKind::Fftw);
            assert_eq!(mr_signed.options().unwrap().requested_threads(), 2);
            assert_options(mr_signed.options().unwrap(), opts);
            let mut mra = mr.allocate_input().unwrap();
            mra.as_mut_slice()
                .iter_mut()
                .enumerate()
                .for_each(|(i, x)| *x = i as f64 * 0.05 + 0.1);
            let original_mixed_real = mra.as_slice().to_vec();
            let mut mrb = mr.allocate_output().unwrap();
            let mut mrw = mr.allocate_workspace().unwrap();
            mr.forward(&mra, &mut mrb, &mut mrw).unwrap();
            let rust_mr = MixedR2cPlan::<f64, 2, 1>::from_shape_with_layout(
                Arc::clone(&topology),
                [4 * size, 3],
                ExtraShape::scalar(),
                [AxisTransform::Fft, AxisTransform::Rfft],
                layout,
            )
            .unwrap();
            let rust_mr_signed = rust_mr.with_fft_directions(mixed_directions).unwrap();
            let directions_first = rust_mr_signed.with_fftw(opts).unwrap();
            assert_eq!(directions_first.backend_kind(), BackendKind::Fftw);
            assert_options(directions_first.options().unwrap(), opts);
            assert_eq!(directions_first.fft_directions(), mixed_directions);
            assert_eq!(directions_first.options().unwrap().requested_threads(), 2);
            let mut signed_input = rust_mr_signed.allocate_input().unwrap();
            signed_input
                .as_mut_slice()
                .copy_from_slice(&original_mixed_real);
            let mut signed_expected = rust_mr_signed.allocate_output().unwrap();
            let mut signed_workspace = rust_mr_signed.allocate_workspace().unwrap();
            rust_mr_signed
                .forward(&signed_input, &mut signed_expected, &mut signed_workspace)
                .unwrap();
            for signed_plan in [&mr_signed, &directions_first] {
                let mut input = signed_plan.allocate_input().unwrap();
                input.as_mut_slice().copy_from_slice(&original_mixed_real);
                let mut output = signed_plan.allocate_output().unwrap();
                let mut workspace = signed_plan.allocate_workspace().unwrap();
                signed_plan
                    .forward(&input, &mut output, &mut workspace)
                    .unwrap();
                assert_complex_close(output.as_slice(), signed_expected.as_slice(), 3e-8);
                let mut recovered = signed_plan.allocate_input().unwrap();
                signed_plan
                    .inverse(&output, &mut recovered, &mut workspace)
                    .unwrap();
                assert_real_close(recovered.as_slice(), &original_mixed_real, 3e-8);
            }
            let mut rmra = rust_mr.allocate_input().unwrap();
            rmra.as_mut_slice().copy_from_slice(&original_mixed_real);
            let mut rmrb = rust_mr.allocate_output().unwrap();
            let mut rmrw = rust_mr.allocate_workspace().unwrap();
            rust_mr.forward(&rmra, &mut rmrb, &mut rmrw).unwrap();
            assert_complex_close(mrb.as_slice(), rmrb.as_slice(), 3e-8);

            // The same auxiliary contract is generated for the other five families.
            aux!(
                r2c,
                rust_r2c,
                ri,
                ro,
                r2cw,
                rri,
                rro,
                rrw,
                real_view_mut,
                real_view,
                complex_view
            );
            aux!(r2r, rust_r2r, a, b, w, ra, rb, rw, view_mut, view, view);
            aux!(
                dht, rust_dht, da, db, dw, rda, rdb, rdw, view_mut, view, view
            );
            aux!(mc, rust_mc, ma, mb, mw, rma, rmb, rmw, view_mut, view, view);
            aux!(
                mr,
                rust_mr,
                mra,
                mrb,
                mrw,
                rmra,
                rmrb,
                rmrw,
                real_view_mut,
                real_view,
                complex_view
            );
        }
    }

    // The generic native path must not accidentally instantiate only f64.
    let f32_rust = C2cPlan::<f32, 2, 1>::from_shape_with_method(
        Arc::clone(&topology),
        [4 * size, 3],
        ExtraShape::scalar(),
        TransposeMethod::AllToAllv,
    )
    .unwrap();
    let f32_native = f32_rust.with_fftw(opts).unwrap();
    let mut f32_input = f32_native.allocate_input().unwrap();
    f32_input
        .as_mut_slice()
        .iter_mut()
        .enumerate()
        .for_each(|(i, x)| {
            *x = Complex::new(i as f32 * 0.031 - 0.2, i as f32 * -0.017 + 0.1);
        });
    let mut f32_output = f32_native.allocate_output().unwrap();
    let mut f32_workspace = f32_native.allocate_out_of_place_workspace().unwrap();
    f32_native
        .forward(&f32_input, &mut f32_output, &mut f32_workspace)
        .unwrap();
    let mut f32_expected = f32_rust.allocate_output().unwrap();
    let mut f32_rust_workspace = f32_rust.allocate_out_of_place_workspace().unwrap();
    f32_rust
        .forward(&f32_input, &mut f32_expected, &mut f32_rust_workspace)
        .unwrap();
    for (a, e) in f32_output.as_slice().iter().zip(f32_expected.as_slice()) {
        assert!((f64::from(a.re) - f64::from(e.re)).abs() < 3e-5);
        assert!((f64::from(a.im) - f64::from(e.im)).abs() < 3e-5);
    }

    // Collection path, including reuse and recovery after a clean call.
    let cp = C2cPlan::<f64, 2, 1>::from_shape_with_fftw(
        Arc::clone(&topology),
        [4 * size, 3],
        ExtraShape::scalar(),
        opts,
    )
    .unwrap();
    let sources = vec![cp.allocate_input().unwrap(), cp.allocate_input().unwrap()];
    let mut destinations = vec![cp.allocate_output().unwrap(), cp.allocate_output().unwrap()];
    let mut cw = cp.allocate_out_of_place_workspace().unwrap();
    cp.forward_many(&sources, &mut destinations, &mut cw)
        .unwrap();
    cp.forward_many(&sources, &mut destinations, &mut cw)
        .unwrap();
    world.barrier();
    if world.rank() == 0 {
        println!("PASSED fftw_distributed_native_matrix");
    }
}
