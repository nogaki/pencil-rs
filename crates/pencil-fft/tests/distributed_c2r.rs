#![cfg(feature = "distributed")]

use std::{f64::consts::TAU, sync::Arc};

use mpi::{datatype::Equivalence, traits::*};
use pencil_array::{AxisPermutation, ExtraShape, MpiTopology, PencilArray};
use pencil_fft::{
    AxisSelection, BackendKind, C2rError, C2rPlan, Complex, DistributedLayout, FftError, FftReal,
    R2cPlan, TransposeMethod,
};

trait Real: FftReal + Into<f64> {
    const TOL: f64;
}
impl Real for f32 {
    const TOL: f64 = 4e-4;
}
impl Real for f64 {
    const TOL: f64 = 2e-10;
}

// Visit logical global indices without assuming the array's memory permutation.
fn visit<T, const N: usize, const M: usize>(
    array: &mut PencilArray<T, N, M>,
    mut f: impl FnMut(usize, [usize; N], &mut T),
) {
    let pencil = Arc::clone(array.pencil());
    for (offset, value) in array.as_mut_slice().iter_mut().enumerate() {
        let mut linear = offset % pencil.local_len();
        let mut global = [0; N];
        for axis in pencil.permutation().axes().iter().rev() {
            let a = axis.index();
            let length = pencil.local_shape_logical()[a];
            global[a] = pencil.local_ranges()[a].start + linear % length;
            linear /= length;
        }
        f(offset / pencil.local_len(), global, value);
    }
}

fn signal<const N: usize>(batch: usize, x: [usize; N]) -> f64 {
    let q = x
        .iter()
        .enumerate()
        .map(|(a, &x)| (a + 2) * (x + 1))
        .sum::<usize>() as f64;
    0.17 + 0.031 * batch as f64 + 0.071 * q + (0.3 * q).sin()
}

// Direct negative-sign DFT of the analytic real signal, not another FFT plan.
fn spectrum<const N: usize>(
    batch: usize,
    k: [usize; N],
    shape: [usize; N],
    selected: AxisSelection<N>,
) -> Complex<f64> {
    let mut sum = Complex::new(0.0, 0.0);
    for mut linear in 0..shape.iter().product() {
        let mut x = k;
        for a in (0..N).rev() {
            x[a] = linear % shape[a];
            linear /= shape[a];
        }
        if (0..N).any(|a| !selected.contains(a) && x[a] != k[a]) {
            continue;
        }
        let phase = -TAU
            * (0..N)
                .filter(|&a| selected.contains(a))
                .map(|a| (k[a] * x[a]) as f64 / shape[a] as f64)
                .sum::<f64>();
        sum += signal(batch, x) * Complex::new(phase.cos(), phase.sin());
    }
    sum
}

fn complex<R: Real>(z: Complex<f64>) -> Complex<R> {
    Complex::new(R::from_f64(z.re).unwrap(), R::from_f64(z.im).unwrap())
}

fn close<R: Real>(actual: R, expected: f64) {
    let actual: f64 = actual.into();
    assert!(
        (actual - expected).abs() <= R::TOL * (1.0 + expected.abs()),
        "{actual} != {expected}"
    );
}

fn check_spectrum<R: Real, const N: usize, const M: usize>(
    array: &mut PencilArray<Complex<R>, N, M>,
    shape: [usize; N],
    selection: AxisSelection<N>,
    scale: f64,
) {
    visit(array, |batch, k, actual| {
        let expected = spectrum(batch, k, shape, selection) * scale;
        close(actual.re, expected.re);
        close(actual.im, expected.im);
    });
}

fn run_case<R: Real, const N: usize, const M: usize>(
    topology: &Arc<MpiTopology<M>>,
    shape: [usize; N],
    selection: AxisSelection<N>,
    extra: ExtraShape,
    layout: DistributedLayout,
    native: bool,
) where
    Complex<R>: Equivalence,
{
    let reduction = (0..N).find(|&a| selection.contains(a)).unwrap();
    let real_len = shape[reduction];
    let mut reduced = shape;
    reduced[reduction] = real_len / 2 + 1;
    let plan = C2rPlan::<R, N, M>::from_shape_with_selection_and_layout(
        Arc::clone(topology),
        reduced,
        real_len,
        extra.clone(),
        selection,
        layout,
    )
    .unwrap();
    assert_eq!(plan.backend_kind(), BackendKind::RustFft);
    #[cfg(feature = "fftw")]
    let plan = if native {
        let plan = plan.with_fftw(pencil_fft::PlanOptions::default()).unwrap();
        assert_eq!(plan.backend_kind(), BackendKind::Fftw);
        assert!(plan.options().is_some());
        plan
    } else {
        plan
    };
    #[cfg(not(feature = "fftw"))]
    assert!(!native);
    assert_eq!(plan.selection(), selection);
    assert_eq!(plan.reduction_axis(), reduction);
    assert_eq!(plan.real_len(), real_len);
    assert_eq!(plan.extra_shape(), &extra);
    assert_eq!(plan.layout(), layout);
    assert_eq!(plan.input_pencil().global_shape(), &reduced);
    assert_eq!(plan.output_pencil().global_shape(), &shape);
    assert_eq!(
        plan.input_pencil().decomposition().map(|a| a.index()),
        std::array::from_fn(|a| a)
    );
    assert_eq!(
        plan.output_pencil().decomposition().map(|a| a.index()),
        std::array::from_fn(|a| a + 1)
    );
    assert_eq!(
        plan.input_pencil().permutation().axes().map(|a| a.index()),
        std::array::from_fn(|a| a)
    );
    assert_eq!(
        plan.output_pencil().permutation().axes().map(|a| a.index()),
        std::array::from_fn(|a| if layout.permute_dims { N - 1 - a } else { a })
    );
    for (i, stage) in plan.stage_geometry().iter().enumerate() {
        assert_eq!(stage.axis, N - 1 - i);
        assert_eq!(
            stage.source.global_shape(),
            if stage.axis >= reduction {
                &reduced
            } else {
                &shape
            }
        );
        assert_eq!(
            stage.output.global_shape(),
            if stage.axis > reduction {
                &reduced
            } else {
                &shape
            }
        );
    }

    let factor = (0..N)
        .filter(|&a| selection.contains(a))
        .map(|a| shape[a])
        .product::<usize>() as f64;
    let mut input = plan.allocate_input().unwrap();
    visit(&mut input, |batch, k, z| {
        *z = complex(spectrum(batch, k, shape, selection))
    });
    let before = input.as_slice().to_vec();
    let mut output = plan.allocate_output().unwrap();
    let mut recovered = plan.allocate_input().unwrap();
    let mut workspace = plan.allocate_workspace().unwrap();
    plan.forward(&input, &mut output, &mut workspace).unwrap();
    assert_eq!(input.as_slice(), before);
    visit(&mut output, |batch, x, value| {
        close(*value, factor * signal(batch, x))
    });
    let real_before = output.as_slice().to_vec();
    plan.inverse(&output, &mut recovered, &mut workspace)
        .unwrap();
    assert_eq!(output.as_slice(), real_before);
    check_spectrum(&mut recovered, shape, selection, 1.0);

    // Fresh analytic real source checks both R2C operations independently.
    visit(&mut output, |batch, x, value| {
        *value = R::from_f64(signal(batch, x)).unwrap()
    });
    let real_before = output.as_slice().to_vec();
    plan.backward(&output, &mut recovered, &mut workspace)
        .unwrap();
    assert_eq!(output.as_slice(), real_before);
    check_spectrum(&mut recovered, shape, selection, 1.0);
    plan.inverse(&output, &mut recovered, &mut workspace)
        .unwrap();
    assert_eq!(output.as_slice(), real_before);
    check_spectrum(&mut recovered, shape, selection, 1.0 / factor);

    // Existing R2C still reduces the highest selected axis and uses its old route.
    let legacy = R2cPlan::<R, N, M>::from_shape_with_selection_and_layout(
        Arc::clone(topology),
        shape,
        extra,
        selection,
        layout,
    )
    .unwrap();
    let mut real = legacy.allocate_input().unwrap();
    visit(&mut real, |batch, x, value| {
        *value = R::from_f64(signal(batch, x)).unwrap()
    });
    let before = real.as_slice().to_vec();
    let mut freq = legacy.allocate_output().unwrap();
    let mut scratch = legacy.allocate_workspace().unwrap();
    legacy.forward(&real, &mut freq, &mut scratch).unwrap();
    assert_eq!(real.as_slice(), before);
    check_spectrum(&mut freq, shape, selection, 1.0);
    legacy.inverse(&freq, &mut real, &mut scratch).unwrap();
    visit(&mut real, |batch, x, value| close(*value, signal(batch, x)));
}

fn matrix<R: Real>(one: &Arc<MpiTopology<1>>, two: &Arc<MpiTopology<2>>, native: bool)
where
    Complex<R>: Equivalence,
{
    for method in [TransposeMethod::AllToAllv, TransposeMethod::PointToPoint] {
        for permute_dims in [true, false] {
            let layout = DistributedLayout {
                transpose_method: method,
                permute_dims,
            };
            for n in [1, 2, 5, 6] {
                run_case::<R, 3, 2>(
                    two,
                    [n, 3, 2],
                    AxisSelection::all(),
                    ExtraShape::new([2, 3]).unwrap(),
                    layout,
                    native,
                );
                run_case::<R, 3, 2>(
                    two,
                    [2, n, 3],
                    AxisSelection::from_indices([1, 2]).unwrap(),
                    ExtraShape::scalar(),
                    layout,
                    native,
                );
                run_case::<R, 3, 1>(
                    one,
                    [2, 1, n],
                    AxisSelection::from_indices([2]).unwrap(),
                    ExtraShape::new([2]).unwrap(),
                    layout,
                    native,
                );
            }
            for selection in [
                AxisSelection::from_indices([0, 2]).unwrap(),
                AxisSelection::from_indices([1]).unwrap(),
                AxisSelection::from_indices([0]).unwrap(),
            ] {
                run_case::<R, 3, 1>(
                    one,
                    [3, 2, 4],
                    selection,
                    ExtraShape::new([2]).unwrap(),
                    layout,
                    native,
                );
            }
            // N > M + 1: mixed local/distributed edges must also be reversed.
            run_case::<R, 4, 1>(
                one,
                [2, 5, 2, 3],
                AxisSelection::from_indices([1, 3]).unwrap(),
                ExtraShape::scalar(),
                layout,
                native,
            );
            run_case::<R, 3, 2>(
                two,
                [1, 2, 3],
                AxisSelection::all(),
                ExtraShape::new([0]).unwrap(),
                layout,
                native,
            );
        }
    }
}

fn mismatch<T: std::fmt::Debug>(result: Result<T, C2rError>) {
    assert!(
        matches!(
            result,
            Err(C2rError::Fft(FftError::CollectiveDescriptorMismatch))
        ),
        "{result:?}"
    );
}

fn failures(topology: &Arc<MpiTopology<1>>, native: bool) {
    #[cfg(not(feature = "fftw"))]
    assert!(!native);
    let rank = topology.rank();
    let size = topology.communicator().size();
    let build = |n| {
        C2rPlan::<f64, 2, 1>::from_shape(
            Arc::clone(topology),
            [n / 2 + 1, 3],
            n,
            ExtraShape::scalar(),
        )
    };
    let plan = build(4).unwrap();
    let other = build(4).unwrap();
    let mut source = plan.allocate_input().unwrap();
    let mut target = plan.allocate_output().unwrap();
    let mut workspace = plan.allocate_workspace().unwrap();
    let mut foreign = other.allocate_workspace().unwrap();
    target.as_mut_slice().fill(71.0);
    let source_before = source.as_slice().to_vec();
    let target_before = target.as_slice().to_vec();
    let active = if rank == 0 {
        &mut foreign
    } else {
        &mut workspace
    };
    let scratch_before = format!("{active:?}");
    let result = plan.forward(&source, &mut target, active);
    assert!(matches!(
        result,
        Err(C2rError::Fft(
            FftError::WorkspaceMismatch | FftError::CollectivePreconditionFailed
        ))
    ));
    assert_eq!(format!("{active:?}"), scratch_before);
    assert_eq!(source.as_slice(), source_before);
    assert_eq!(target.as_slice(), target_before);

    // The constructors accept canonical complex pencils/arrays directly.
    let pencil_plan =
        C2rPlan::<f64, 2, 1>::from_pencil(Arc::clone(plan.input_pencil()), 4, ExtraShape::scalar())
            .unwrap();
    let array_plan = C2rPlan::from_array(&source, 4).unwrap();
    assert!(
        pencil_plan
            .output_pencil()
            .same_layout(array_plan.output_pencil())
    );
    // Noncanonical input must be rejected, including an old R2C output layout.
    let old = R2cPlan::<f64, 2, 1>::from_shape(Arc::clone(topology), [3, 4], ExtraShape::scalar())
        .unwrap();
    assert!(matches!(
        C2rPlan::<f64, 2, 1>::from_pencil(Arc::clone(old.output_pencil()), 4, ExtraShape::scalar()),
        Err(C2rError::Fft(FftError::InvalidInputLayout))
    ));
    for (shape, length, selection) in [
        ([3, 3], 0, AxisSelection::all()),
        ([2, 3], 4, AxisSelection::all()),
        ([3, 3], 4, AxisSelection::empty()),
    ] {
        assert!(
            C2rPlan::<f64, 2, 1>::from_shape_with_selection_and_layout(
                Arc::clone(topology),
                shape,
                length,
                ExtraShape::scalar(),
                selection,
                DistributedLayout::default()
            )
            .is_err()
        );
    }
    let bad_pencil = if rank == 0 {
        plan.input_pencil()
            .with_permutation(AxisPermutation::new([1, 0]).unwrap())
            .unwrap()
    } else {
        Arc::clone(plan.input_pencil())
    };
    assert!(
        C2rPlan::<f64, 2, 1>::from_pencil(Arc::clone(&bad_pencil), 4, ExtraShape::scalar())
            .is_err()
    );
    let bad_source =
        PencilArray::from_elem(bad_pencil, ExtraShape::scalar(), Complex::new(2.0, 0.0)).unwrap();
    let scratch_before = format!("{workspace:?}");
    assert!(
        plan.forward(&bad_source, &mut target, &mut workspace)
            .is_err()
    );
    assert_eq!(format!("{workspace:?}"), scratch_before);
    assert_eq!(target.as_slice(), target_before);

    if size > 1 {
        // Explicit length agreement matters even when both lengths reduce to 3.
        mismatch(C2rPlan::<f64, 2, 1>::from_shape(
            Arc::clone(topology),
            [3, 3],
            if rank == 0 { 5 } else { 4 },
            ExtraShape::scalar(),
        ));
        mismatch(C2rPlan::<f64, 2, 1>::from_shape(
            Arc::clone(topology),
            [3, if rank == 0 { 4 } else { 3 }],
            4,
            ExtraShape::scalar(),
        ));
        mismatch(C2rPlan::<f64, 2, 1>::from_shape_with_selection_and_layout(
            Arc::clone(topology),
            [3, 3],
            4,
            ExtraShape::scalar(),
            if rank == 0 {
                AxisSelection::all()
            } else {
                AxisSelection::from_indices([0]).unwrap()
            },
            DistributedLayout::default(),
        ));
        mismatch(C2rPlan::<f64, 2, 1>::from_shape(
            Arc::clone(topology),
            [3, 3],
            4,
            ExtraShape::new(if rank == 0 { [2, 3] } else { [3, 2] }).unwrap(),
        ));
        for change_method in [false, true] {
            mismatch(C2rPlan::<f64, 2, 1>::from_shape_with_selection_and_layout(
                Arc::clone(topology),
                [3, 3],
                4,
                ExtraShape::scalar(),
                AxisSelection::all(),
                DistributedLayout {
                    transpose_method: if change_method && rank == 0 {
                        TransposeMethod::PointToPoint
                    } else {
                        TransposeMethod::AllToAllv
                    },
                    permute_dims: change_method || rank == 0,
                },
            ));
        }
        let result = if rank == 0 {
            C2rPlan::<f32, 2, 1>::from_shape(Arc::clone(topology), [3, 3], 4, ExtraShape::scalar())
                .map(|_| ())
        } else {
            build(4).map(|_| ())
        };
        mismatch(result);
        let result = if rank == 0 {
            R2cPlan::<f64, 2, 1>::from_shape(Arc::clone(topology), [3, 3], ExtraShape::scalar())
                .map(|_| ())
        } else {
            build(4).map(|_| ())
        };
        mismatch(result);
        // Already-built plans must also agree on lengths, shapes, and selection.
        for (shape, n, selection) in [
            ([3, 3], 5, AxisSelection::all()),
            ([3, 4], 4, AxisSelection::all()),
            ([3, 3], 4, AxisSelection::from_indices([0]).unwrap()),
        ] {
            let alternate = C2rPlan::<f64, 2, 1>::from_shape_with_selection_and_layout(
                Arc::clone(topology),
                shape,
                n,
                ExtraShape::scalar(),
                selection,
                DistributedLayout::default(),
            )
            .unwrap();
            let alternate_source = alternate.allocate_input().unwrap();
            let mut alternate_target = alternate.allocate_output().unwrap();
            alternate_target.as_mut_slice().fill(43.0);
            let mut alternate_workspace = alternate.allocate_workspace().unwrap();
            let scratch_before = format!("{alternate_workspace:?}/{workspace:?}");
            mismatch(if rank == 0 {
                alternate.forward(
                    &alternate_source,
                    &mut alternate_target,
                    &mut alternate_workspace,
                )
            } else {
                plan.forward(&source, &mut target, &mut workspace)
            });
            assert!(alternate_target.as_slice().iter().all(|&v| v == 43.0));
            assert_eq!(source.as_slice(), source_before);
            assert_eq!(target.as_slice(), target_before);
            assert_eq!(
                format!("{alternate_workspace:?}/{workspace:?}"),
                scratch_before
            );
        }
        // Inverse and backward must disagree BEFORE transforming or scaling.
        let scratch_before = format!("{workspace:?}");
        mismatch(if rank == 0 {
            plan.inverse(&target, &mut source, &mut workspace)
        } else {
            plan.backward(&target, &mut source, &mut workspace)
        });
        assert_eq!(format!("{workspace:?}"), scratch_before);
        assert_eq!(source.as_slice(), source_before);
        assert_eq!(target.as_slice(), target_before);
        mismatch(if rank == 0 {
            plan.forward(&source, &mut target, &mut workspace)
        } else {
            plan.inverse(&target, &mut source, &mut workspace)
        });
        assert_eq!(source.as_slice(), source_before);
        assert_eq!(target.as_slice(), target_before);

        let mut old_real = old.allocate_input().unwrap();
        let mut old_freq = old.allocate_output().unwrap();
        let mut old_work = old.allocate_workspace().unwrap();
        mismatch(if rank == 0 {
            old.forward(&old_real, &mut old_freq, &mut old_work)
        } else {
            plan.forward(&source, &mut target, &mut workspace)
        });
        mismatch(if rank == 0 {
            old.backward(&old_freq, &mut old_real, &mut old_work)
        } else {
            plan.backward(&target, &mut source, &mut workspace)
        });
        assert_eq!(source.as_slice(), source_before);
        assert_eq!(target.as_slice(), target_before);
    }

    // One rank changes one constrained frequency; all ranks reject, even empty owners.
    for n in [1, 2, 5, 6] {
        let plan = build(n).unwrap();
        #[cfg(feature = "fftw")]
        let plan = if native {
            plan.with_fftw(pencil_fft::PlanOptions::default()).unwrap()
        } else {
            plan
        };
        let mut source = plan.allocate_input().unwrap();
        let mut target = plan.allocate_output().unwrap();
        let mut workspace = plan.allocate_workspace().unwrap();
        target.as_mut_slice().fill(17.0);
        for endpoint in if n % 2 == 0 { vec![0, n / 2] } else { vec![0] } {
            for nonfinite in [false, true] {
                source.as_mut_slice().fill(Complex::new(0.0, 0.0));
                visit(&mut source, |_, k, z| {
                    if k == [endpoint, 0] {
                        *z = if nonfinite {
                            Complex::new(f64::INFINITY, 0.0)
                        } else {
                            Complex::new(0.0, 1.0)
                        };
                    }
                });
                let before = source.as_slice().to_vec();
                assert!(matches!(
                    plan.forward(&source, &mut target, &mut workspace),
                    Err(C2rError::InvalidSpectrum)
                ));
                assert_eq!(source.as_slice(), before);
                assert!(target.as_slice().iter().all(|&v| v == 17.0));
            }
        }
        source.as_mut_slice().fill(Complex::new(0.0, 0.0));
        visit(&mut source, |_, k, z| {
            if k == [0, 0] {
                *z = Complex::new(1.0, 1e-16);
            }
        });
        let before = source.as_slice().to_vec();
        plan.forward(&source, &mut target, &mut workspace).unwrap();
        assert_eq!(source.as_slice(), before); // Projection stays private.
        // Pure imaginary subnormal noise exercises the absolute threshold:
        // 128 * min_subnormal * (1 + ceil(log2(3))) * 3 = 1152 subnormals.
        for (noise, accepted) in [(600.0, true), (2400.0, false)] {
            source.as_mut_slice().fill(Complex::new(0.0, 0.0));
            visit(&mut source, |_, k, z| {
                if k == [0, 0] {
                    z.im = noise * f64::from_bits(1);
                }
            });
            let before = source.as_slice().to_vec();
            let real_before = target.as_slice().to_vec();
            let result = plan.forward(&source, &mut target, &mut workspace);
            if accepted {
                result.unwrap();
                assert!(target.as_slice().iter().all(|&v| v == 0.0));
            } else {
                assert!(matches!(result, Err(C2rError::InvalidSpectrum)));
                assert_eq!(target.as_slice(), real_before);
            }
            assert_eq!(source.as_slice(), before);
        }
        if n == 5 {
            source.as_mut_slice().fill(Complex::new(0.0, 0.0));
            visit(&mut source, |_, k, z| {
                if k == [n / 2, 0] {
                    *z = Complex::new(0.0, 1.0);
                }
            });
            let before = source.as_slice().to_vec();
            plan.forward(&source, &mut target, &mut workspace).unwrap();
            assert_eq!(source.as_slice(), before);
            visit(&mut target, |_, x, value| {
                close(
                    *value,
                    -2.0 * (TAU * (n / 2 * x[0]) as f64 / n as f64).sin(),
                )
            });
        }
    }

    #[cfg(feature = "fftw")]
    if native {
        use pencil_fft::{BackendInitError, PlanOptions};
        let native = plan.with_fftw(PlanOptions::default()).unwrap();
        let mut native_source = native.allocate_input().unwrap();
        let mut native_target = native.allocate_output().unwrap();
        let mut native_work = native.allocate_workspace().unwrap();
        assert!(
            native
                .forward(&source, &mut native_target, &mut native_work)
                .is_err()
        );
        assert!(
            native
                .forward(&native_source, &mut native_target, &mut workspace)
                .is_err()
        );
        if size > 1 {
            // Rank-varying options and RustFFT/native constructor families.
            assert!(matches!(
                plan.with_fftw(
                    PlanOptions::default()
                        .with_threads(if rank == 0 { 2 } else { 1 })
                        .unwrap()
                ),
                Err(BackendInitError::Local(C2rError::Fft(
                    FftError::CollectiveDescriptorMismatch
                )))
            ));
            let result = if rank == 0 {
                plan.with_fftw(PlanOptions::default()).map(|_| ())
            } else {
                build(4).map(|_| ()).map_err(BackendInitError::Local)
            };
            assert!(matches!(
                result,
                Err(BackendInitError::Local(C2rError::Fft(
                    FftError::CollectiveDescriptorMismatch
                )))
            ));
            mismatch(if rank == 0 {
                native.backward(&native_target, &mut native_source, &mut native_work)
            } else {
                plan.backward(&target, &mut source, &mut workspace)
            });
        }
    }
}

fn run(native: bool) {
    let universe = mpi::initialize().expect("MPI initialization failed");
    let world = universe.world();
    let size = usize::try_from(world.size()).unwrap();
    assert!(matches!(size, 1 | 4 | 6), "run with 1, 4, or 6 MPI ranks");
    let one = MpiTopology::<1>::new(&world, [size]).unwrap();
    let two = MpiTopology::<2>::new(
        &world,
        match size {
            1 => [1, 1],
            4 => [2, 2],
            _ => [2, 3],
        },
    )
    .unwrap();
    matrix::<f64>(&one, &two, native);
    matrix::<f32>(&one, &two, native);
    failures(&one, native);
}

#[test]
fn distributed_c2r_one_mpi_binary() {
    run(false);
}

#[cfg(feature = "fftw")]
#[test]
#[ignore = "requires explicit native FFTW MPI run"]
fn distributed_c2r_fftw_one_mpi_binary() {
    run(true);
}
