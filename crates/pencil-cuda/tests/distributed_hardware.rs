#![cfg(feature = "distributed")]
//! Explicit execution requires real CUDA; there is deliberately no fallback.
use mpi::traits::*;
use num_complex::Complex;
use pencil_array::{ExtraShape, MpiTopology, Pencil};
use pencil_cuda::distributed::{CudaPencilArray, DistributedPlan, Error};
use pencil_fft::{
    AxisSelection, DistributedLayout, FourierDirection, FourierDirections, TransposeMethod,
};
use std::f64::consts::TAU;

fn coordinate<const N: usize, const M: usize>(p: &Pencil<N, M>, q: usize) -> [usize; N] {
    let mut t = q;
    let mut x = [0; N];
    let shape = p.local_shape_memory();
    for (m, a) in p.permutation().axes().iter().enumerate().rev() {
        x[a.index()] = t % shape[m] + p.local_ranges()[a.index()].start;
        t /= shape[m];
    }
    x
}
fn values<const N: usize, const M: usize>(
    p: &Pencil<N, M>,
    batches: usize,
    f: impl Fn(usize, [usize; N]) -> Complex<f64>,
) -> Vec<Complex<f64>> {
    (0..batches)
        .flat_map(|b| (0..p.local_len()).map(move |q| (b, coordinate(p, q))))
        .map(|(b, x)| f(b, x))
        .collect()
}
fn signal(b: usize, x: [usize; 2]) -> Complex<f64> {
    Complex::new(
        (0.37 * (1 + b + x[0] * 3 + x[1]) as f64).sin(),
        (0.19 * (2 + b + x[0] + x[1] * 2) as f64).cos(),
    )
}
// Small direct DFT, independently indexed in logical global coordinates.
fn dft(
    shape: [usize; 2],
    selected: [bool; 2],
    sign: [f64; 2],
    b: usize,
    k: [usize; 2],
    real: bool,
) -> Complex<f64> {
    let mut sum = Complex::new(0.0, 0.0);
    for x0 in 0..shape[0] {
        for x1 in 0..shape[1] {
            let x = [x0, x1];
            if (0..2).any(|a| !selected[a] && x[a] != k[a]) {
                continue;
            }
            let angle = (0..2)
                .filter(|&a| selected[a])
                .map(|a| sign[a] * TAU * (x[a] * k[a]) as f64 / shape[a] as f64)
                .sum();
            let mut z = signal(b, x);
            if real {
                z.im = 0.0;
            }
            sum += z * Complex::from_polar(1.0, angle);
        }
    }
    sum
}
fn close(a: Complex<f64>, b: Complex<f64>, eps: f64) {
    assert!(
        (a - b).norm() <= eps * (1.0 + a.norm().max(b.norm())),
        "{a:?} != {b:?}"
    );
}
macro_rules! matrix {
    ($name:ident,$r:ty,$eps:expr) => {
        fn $name(world: &impl Communicator) {
            let topology = MpiTopology::<1>::new(world, [world.size() as usize]).unwrap();
            for method in [TransposeMethod::AllToAllv, TransposeMethod::PointToPoint] {
                for permute_dims in [false, true] {
                    for shape in [[3, 5], [2, 4], [1, 1], [2, 2]] {
                        for selected in [[true, true], [false, true], [true, false], [false, false]]
                        {
                            let input =
                                Pencil::<2, 1>::new_default(topology.clone(), shape).unwrap();
                            let extra = ExtraShape::new([2, 2]).unwrap();
                            let batches = 4;
                            let selection =
                                AxisSelection::from_indices((0..2).filter(|&a| selected[a]))
                                    .unwrap();
                            let layout = DistributedLayout {
                                transpose_method: method,
                                permute_dims,
                            };
                            let signs = FourierDirections::new([
                                if selected[0] {
                                    FourierDirection::Backward
                                } else {
                                    FourierDirection::Forward
                                },
                                FourierDirection::Forward,
                            ]);
                            let plan = DistributedPlan::<$r, 2, 1>::c2c(
                                input.clone(),
                                extra.clone(),
                                selection,
                                layout,
                                signs,
                                0,
                            )
                            .unwrap();
                            let mut w = plan.allocate_workspace().unwrap();
                            let mut src = plan.allocate_input().unwrap();
                            let mut dst = plan.allocate_output().unwrap();
                            let initial = values(&input, batches, signal);
                            let upload: Vec<_> = initial
                                .iter()
                                .map(|z| Complex::new(z.re as $r, z.im as $r))
                                .collect();
                            src.upload(&upload).unwrap();
                            plan.forward(&src, &mut dst, &mut w).unwrap();
                            assert_eq!(src.download().unwrap(), upload);
                            let want = values(plan.output_pencil(), batches, |b, k| {
                                dft(shape, selected, [1.0, -1.0], b, k, false)
                            });
                            let actual = dst.download().unwrap();
                            assert_eq!(actual.len(), want.len());
                            for (a, b) in actual.iter().zip(want) {
                                close(Complex::new(a.re as f64, a.im as f64), b, $eps)
                            }
                            let spectrum = dst.download().unwrap();
                            plan.inverse(&dst, &mut src, &mut w).unwrap();
                            assert_eq!(dst.download().unwrap(), spectrum);
                            let actual = src.download().unwrap();
                            assert_eq!(actual.len(), initial.len());
                            for (a, b) in actual.iter().zip(&initial) {
                                close(Complex::new(a.re as f64, a.im as f64), *b, $eps)
                            }
                            // Arbitrary complex inverse input, not a forward output.
                            let arbitrary = values(plan.output_pencil(), batches, signal);
                            let v: Vec<_> = arbitrary
                                .iter()
                                .map(|z| Complex::new(z.re as $r, z.im as $r))
                                .collect();
                            dst.upload(&v).unwrap();
                            let scale = (0..2)
                                .filter(|&a| selected[a])
                                .map(|a| shape[a])
                                .product::<usize>() as f64;
                            for raw in [false, true] {
                                if raw {
                                    plan.backward(&dst, &mut src, &mut w).unwrap()
                                } else {
                                    plan.inverse(&dst, &mut src, &mut w).unwrap()
                                }
                                let want = values(&input, batches, |b, k| {
                                    dft(shape, selected, [-1.0, 1.0], b, k, false)
                                        / if raw { 1.0 } else { scale }
                                });
                                let actual = src.download().unwrap();
                                assert_eq!(actual.len(), want.len());
                                for (a, b) in actual.iter().zip(want) {
                                    close(Complex::new(a.re as f64, a.im as f64), b, $eps)
                                }
                                assert_eq!(dst.download().unwrap(), v);
                            }
                            // Wrong workspace and context are preflight-only failures;
                            // both leave the destination and workspace usable.
                            let other = DistributedPlan::<$r, 2, 1>::c2c(
                                input.clone(),
                                extra.clone(),
                                selection,
                                layout,
                                signs,
                                0,
                            )
                            .unwrap();
                            let mut wrong = other.allocate_workspace().unwrap();
                            let before = src.download().unwrap();
                            assert!(plan.inverse(&dst, &mut src, &mut wrong).is_err());
                            assert!(!wrong.is_poisoned());
                            assert_eq!(src.download().unwrap(), before);
                            let mut alien =
                                CudaPencilArray::new(other.device(), input.clone(), extra.clone())
                                    .unwrap();
                            assert!(plan.inverse(&dst, &mut alien, &mut w).is_err());
                            assert!(!w.is_poisoned());
                            plan.inverse(&dst, &mut src, &mut w).unwrap();
                            if !selected.iter().any(|&v| v) {
                                continue;
                            }
                            let real = DistributedPlan::<$r, 2, 1>::r2c(
                                input.clone(),
                                extra.clone(),
                                selection,
                                layout,
                                0,
                            )
                            .unwrap();
                            let mut rw = real.allocate_workspace().unwrap();
                            let mut rin = real.allocate_real_input().unwrap();
                            let mut rout = real.allocate_output().unwrap();
                            let original: Vec<$r> = initial.iter().map(|z| z.re as $r).collect();
                            rin.upload(&original).unwrap();
                            real.forward_real(&rin, &mut rout, &mut rw).unwrap();
                            assert_eq!(rin.download().unwrap(), original);
                            let want = values(real.output_pencil(), batches, |b, k| {
                                dft(shape, selected, [-1.0, -1.0], b, k, true)
                            });
                            let actual = rout.download().unwrap();
                            assert_eq!(actual.len(), want.len());
                            for (a, b) in actual.iter().zip(want) {
                                close(Complex::new(a.re as f64, a.im as f64), b, $eps)
                            }
                            // Independently constructed Hermitian spectra, rather than
                            // cuFFT roundtrips, exercise arbitrary inverse input.
                            let independent = values(real.output_pencil(), batches, |b, k| {
                                dft(shape, selected, [-1.0, -1.0], b + 7, k, true)
                            });
                            let upload: Vec<_> = independent
                                .iter()
                                .map(|z| Complex::new(z.re as $r, z.im as $r))
                                .collect();
                            rout.upload(&upload).unwrap();
                            for raw in [false, true] {
                                if raw {
                                    real.backward_real(&rout, &mut rin, &mut rw).unwrap()
                                } else {
                                    real.inverse_real(&rout, &mut rin, &mut rw).unwrap()
                                }
                                let want = values(&input, batches, |b, x| {
                                    Complex::new(
                                        signal(b + 7, x).re * if raw { scale } else { 1.0 },
                                        0.0,
                                    )
                                });
                                let actual = rin.download().unwrap();
                                assert_eq!(actual.len(), want.len());
                                for (a, b) in actual.iter().zip(want) {
                                    close(Complex::new(*a as f64, 0.0), b, $eps)
                                }
                                assert_eq!(rout.download().unwrap(), upload);
                            }
                            // A purely imaginary DC plane cannot describe real data.
                            let axis = (0..2).rfind(|&a| selected[a]).unwrap();
                            let bad = values(real.output_pencil(), batches, |_, k| {
                                Complex::new(0.0, if k[axis] == 0 { 1.0 } else { 0.0 })
                            });
                            let bad: Vec<_> = bad
                                .iter()
                                .map(|z| Complex::new(z.re as $r, z.im as $r))
                                .collect();
                            rout.upload(&bad).unwrap();
                            let saved = rin.download().unwrap();
                            assert!(real.inverse_real(&rout, &mut rin, &mut rw).is_err());
                            assert!(rw.is_poisoned());
                            assert_eq!(rin.download().unwrap(), saved);
                            assert_eq!(rout.download().unwrap(), bad);
                            rw = real.allocate_workspace().unwrap();
                            rout.upload(&upload).unwrap();
                            real.inverse_real(&rout, &mut rin, &mut rw).unwrap();
                        }
                    }
                }
            }
        }
    };
}
fn signal3(b: usize, x: [usize; 3]) -> Complex<f64> {
    let z = signal(b, [x[0] + 2 * x[2], x[1]]);
    Complex::new(z.re + 0.11 * x[2] as f64, z.im)
}
fn dft3(
    shape: [usize; 3],
    selected: [bool; 3],
    b: usize,
    k: [usize; 3],
    real: bool,
) -> Complex<f64> {
    let mut sum = Complex::new(0.0, 0.0);
    for a in 0..shape[0] {
        for c in 0..shape[1] {
            for d in 0..shape[2] {
                let x = [a, c, d];
                if (0..3).any(|i| !selected[i] && x[i] != k[i]) {
                    continue;
                }
                let angle = (0..3)
                    .filter(|&i| selected[i])
                    .map(|i| -TAU * (x[i] * k[i]) as f64 / shape[i] as f64)
                    .sum();
                let mut z = signal3(b, x);
                if real {
                    z.im = 0.0;
                }
                sum += z * Complex::from_polar(1.0, angle);
            }
        }
    }
    sum
}
macro_rules! grid_matrix {
    ($name:ident,$r:ty,$eps:expr) => {
        fn $name(world: &impl Communicator) {
            let size = world.size() as usize;
            let rows = if size % 2 == 0 { 2 } else { 1 };
            let topology = MpiTopology::<2>::new(world, [rows, size / rows]).unwrap();
            for method in [TransposeMethod::AllToAllv, TransposeMethod::PointToPoint] {
                for permute_dims in [false, true] {
                    for shape in [[2, 3, 5], [1, 2, 3]] {
                        for selected in [
                            [true, true, true],
                            [true, true, false],
                            [true, false, true],
                            [true, false, false],
                        ] {
                            let input =
                                Pencil::<3, 2>::new_default(topology.clone(), shape).unwrap();
                            let extra = ExtraShape::new([2]).unwrap();
                            let selection =
                                AxisSelection::from_indices((0..3).filter(|&i| selected[i]))
                                    .unwrap();
                            let layout = DistributedLayout {
                                transpose_method: method,
                                permute_dims,
                            };
                            let scale = (0..3)
                                .filter(|&i| selected[i])
                                .map(|i| shape[i])
                                .product::<usize>() as f64;
                            let plan = DistributedPlan::<$r, 3, 2>::c2c(
                                input.clone(),
                                extra.clone(),
                                selection,
                                layout,
                                FourierDirections::forward(),
                                0,
                            )
                            .unwrap();
                            let mut w = plan.allocate_workspace().unwrap();
                            let mut src = plan.allocate_input().unwrap();
                            let mut dst = plan.allocate_output().unwrap();
                            let reference = values(&input, 2, signal3);
                            let upload: Vec<_> = reference
                                .iter()
                                .map(|z| Complex::new(z.re as $r, z.im as $r))
                                .collect();
                            src.upload(&upload).unwrap();
                            plan.forward(&src, &mut dst, &mut w).unwrap();
                            assert_eq!(src.download().unwrap(), upload);
                            let want = values(plan.output_pencil(), 2, |b, k| {
                                dft3(shape, selected, b, k, false)
                            });
                            let actual = dst.download().unwrap();
                            assert_eq!(actual.len(), want.len());
                            for (a, b) in actual.iter().zip(want) {
                                close(Complex::new(a.re as f64, a.im as f64), b, $eps)
                            }
                            for raw in [false, true] {
                                if raw {
                                    plan.backward(&dst, &mut src, &mut w).unwrap()
                                } else {
                                    plan.inverse(&dst, &mut src, &mut w).unwrap()
                                }
                                let actual = src.download().unwrap();
                                assert_eq!(actual.len(), reference.len());
                                for (a, b) in actual.iter().zip(&reference) {
                                    close(
                                        Complex::new(a.re as f64, a.im as f64),
                                        *b * if raw { scale } else { 1.0 },
                                        $eps,
                                    )
                                }
                            }
                            let plan = DistributedPlan::<$r, 3, 2>::r2c(
                                input.clone(),
                                extra.clone(),
                                selection,
                                layout,
                                0,
                            )
                            .unwrap();
                            let mut w = plan.allocate_workspace().unwrap();
                            let mut src = plan.allocate_real_input().unwrap();
                            let mut dst = plan.allocate_output().unwrap();
                            let upload: Vec<_> = reference.iter().map(|z| z.re as $r).collect();
                            src.upload(&upload).unwrap();
                            plan.forward_real(&src, &mut dst, &mut w).unwrap();
                            assert_eq!(src.download().unwrap(), upload);
                            let want = values(plan.output_pencil(), 2, |b, k| {
                                dft3(shape, selected, b, k, true)
                            });
                            let actual = dst.download().unwrap();
                            assert_eq!(actual.len(), want.len());
                            for (a, b) in actual.iter().zip(&want) {
                                close(Complex::new(a.re as f64, a.im as f64), *b, $eps)
                            }
                            let independent: Vec<_> = want
                                .iter()
                                .map(|z| Complex::new(z.re as $r, z.im as $r))
                                .collect();
                            dst.upload(&independent).unwrap();
                            for raw in [false, true] {
                                if raw {
                                    plan.backward_real(&dst, &mut src, &mut w).unwrap()
                                } else {
                                    plan.inverse_real(&dst, &mut src, &mut w).unwrap()
                                }
                                let actual = src.download().unwrap();
                                assert_eq!(actual.len(), reference.len());
                                for (a, b) in actual.iter().zip(&reference) {
                                    close(
                                        Complex::new(*a as f64, 0.0),
                                        Complex::new(b.re * if raw { scale } else { 1.0 }, 0.0),
                                        $eps,
                                    )
                                }
                                assert_eq!(dst.download().unwrap(), independent);
                            }
                        }
                    }
                }
            }
        }
    };
}
// Called inside the single ignored MPI test so MPI is initialized only once.
fn operation_header_mismatches(world: &impl Communicator) {
    let topology = MpiTopology::<1>::new(world, [world.size() as usize]).unwrap();
    let input = Pencil::<2, 1>::new_default(topology, [3, 5]).unwrap();
    let plan = DistributedPlan::<f64, 2, 1>::c2c(
        input.clone(),
        ExtraShape::scalar(),
        AxisSelection::all(),
        DistributedLayout::default(),
        FourierDirections::forward(),
        0,
    )
    .unwrap();
    let mut w = plan.allocate_workspace().unwrap();
    let mut src = plan.allocate_input().unwrap();
    let mut dst = plan.allocate_output().unwrap();
    let original = values(&input, 1, signal);
    let sentinel = values(plan.output_pencil(), 1, |b, x| signal(b + 7, x));
    src.upload(&original).unwrap();
    dst.upload(&sentinel).unwrap();
    if world.size() > 1 {
        // All ranks must reject at the common operation header, before writes
        // or workspace poisoning. No replacement/recovery is allowed here.
        let result = if world.rank() == 0 {
            plan.forward(&src, &mut dst, &mut w)
        } else {
            plan.inverse(&dst, &mut src, &mut w)
        };
        assert!(matches!(result, Err(Error::Descriptor)));
        assert!(!w.is_poisoned());
        assert_eq!(src.download().unwrap(), original);
        assert_eq!(dst.download().unwrap(), sentinel);

        let result = if world.rank() == 0 {
            plan.allocate_input()
        } else {
            plan.allocate_output()
        };
        assert!(matches!(result, Err(Error::Descriptor)));
        assert!(!w.is_poisoned());
        assert_eq!(src.download().unwrap(), original);
        assert_eq!(dst.download().unwrap(), sentinel);
    }
    // The same plan, arrays and workspace still execute correctly, with no
    // recovery; matched allocations must also remain usable after rejection.
    let _input = plan.allocate_input().unwrap();
    let _output = plan.allocate_output().unwrap();
    plan.forward(&src, &mut dst, &mut w).unwrap();
    assert_eq!(src.download().unwrap(), original);
    let spectrum = dst.download().unwrap();
    let want = values(plan.output_pencil(), 1, |b, k| {
        dft([3, 5], [true, true], [-1.0, -1.0], b, k, false)
    });
    assert_eq!(spectrum.len(), want.len());
    for (a, b) in spectrum.iter().zip(want) {
        close(*a, b, 3e-11);
    }
    plan.inverse(&dst, &mut src, &mut w).unwrap();
    assert!(!w.is_poisoned());
    assert_eq!(dst.download().unwrap(), spectrum);
    let actual = src.download().unwrap();
    assert_eq!(actual.len(), original.len());
    for (a, b) in actual.iter().zip(original) {
        close(*a, b, 3e-11);
    }
}

grid_matrix!(grid_f32, f32, 3e-4);
grid_matrix!(grid_f64, f64, 3e-11);
matrix!(matrix_f32, f32, 3e-4);
matrix!(matrix_f64, f64, 3e-11);
#[test]
#[ignore = "UNVERIFIED: requires CUDA+cuFFT and mpirun -n 1/4/6"]
fn distributed_cuda_actual_mpi_matrix() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    // Explicit requests without CUDA fail, never skip.
    operation_header_mismatches(&world);
    matrix_f32(&world);
    matrix_f64(&world);
    grid_f32(&world);
    grid_f64(&world);
    if world.rank() == 0 {
        println!("CUDA_DISTRIBUTED_HARDWARE_OK ranks={}", world.size());
    }
}
