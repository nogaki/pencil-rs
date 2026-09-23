#![cfg(feature = "fftw")]

use pencil_fft::{
    BackendKind, Complex, FftReal, LocalC2cPlan, LocalDhtPlan, LocalR2cPlan, LocalR2rPlan,
    PlanOptions, PlanningRigor, R2rKind, R2rScalar,
};
use pencil_fft::{export_wisdom, forget_wisdom, import_wisdom};
use rustfft::num_traits::{FromPrimitive, ToPrimitive, Zero};
use std::f64::consts::TAU;
use std::time::Duration;

fn assert_options(actual: Option<PlanOptions>, expected: PlanOptions) {
    let actual = actual.expect("native options");
    assert_eq!(actual.rigor(), expected.rigor());
    assert_eq!(actual.time_limit(), expected.time_limit());
    assert_eq!(actual.requested_threads(), expected.requested_threads());
    assert_eq!(actual.wisdom_only(), expected.wisdom_only());
    assert_eq!(actual.conserve_memory(), expected.conserve_memory());
}

trait Value: R2rScalar {
    fn value(x: f64) -> Self;
    fn parts(self) -> (f64, f64);
}
macro_rules! values {
    ($($t:ty: $ctor:expr, $parts:expr);+ $(;)?) => { $(
        impl Value for $t { fn value(x: f64) -> Self { $ctor(x) } fn parts(self) -> (f64, f64) { $parts(self) } }
    )+ };
}
values! {
    f32: |x: f64| x as f32, |x: f32| (x as f64, 0.0);
    f64: |x: f64| x, |x: f64| (x, 0.0);
    Complex<f32>: |x: f64| Complex::new(x as f32, (0.31 * x).sin() as f32), |x: Complex<f32>| (x.re as f64, x.im as f64);
    Complex<f64>: |x: f64| Complex::new(x, (0.31 * x).sin()), |x: Complex<f64>| (x.re, x.im);
}

fn close<T: Value>(actual: T, expected: Complex<f64>) {
    let (re, im) = actual.parts();
    let e = if std::mem::size_of::<T::Real>() == 4 {
        3e-4
    } else {
        2e-10
    };
    assert!((re - expected.re).abs() < e * (1.0 + expected.re.abs()));
    assert!((im - expected.im).abs() < e * (1.0 + expected.im.abs()));
}
fn dft(input: &[Complex<f64>], inverse: bool) -> Vec<Complex<f64>> {
    (0..input.len())
        .map(|k| {
            input
                .iter()
                .enumerate()
                .map(|(j, &x)| {
                    let a = (if inverse { 1.0 } else { -1.0 }) * TAU * (j * k) as f64
                        / input.len() as f64;
                    x * Complex::new(a.cos(), a.sin())
                })
                .sum()
        })
        .collect()
}
fn r2r(kind: R2rKind, input: &[Complex<f64>]) -> Vec<Complex<f64>> {
    let n = input.len();
    (0..n)
        .map(|k| {
            input
                .iter()
                .enumerate()
                .map(|(j, &x)| {
                    let c = match kind {
                        R2rKind::DctI => {
                            if j == 0 || j + 1 == n {
                                if j + 1 == n && k % 2 == 1 { -1. } else { 1. }
                            } else {
                                2. * (std::f64::consts::PI * j as f64 * k as f64 / (n - 1) as f64)
                                    .cos()
                            }
                        }
                        R2rKind::DctII => {
                            2. * (std::f64::consts::PI * (j as f64 + 0.5) * k as f64 / n as f64)
                                .cos()
                        }
                        R2rKind::DctIII => {
                            if j == 0 {
                                1.
                            } else {
                                2. * (std::f64::consts::PI * j as f64 * (k as f64 + 0.5) / n as f64)
                                    .cos()
                            }
                        }
                        R2rKind::DctIV => {
                            2. * (std::f64::consts::PI * (j as f64 + 0.5) * (k as f64 + 0.5)
                                / n as f64)
                                .cos()
                        }
                        R2rKind::DstI => {
                            2. * (std::f64::consts::PI * (j as f64 + 1.) * (k as f64 + 1.)
                                / (n + 1) as f64)
                                .sin()
                        }
                        R2rKind::DstII => {
                            2. * (std::f64::consts::PI * (j as f64 + 0.5) * (k as f64 + 1.)
                                / n as f64)
                                .sin()
                        }
                        R2rKind::DstIII => {
                            if j + 1 == n {
                                if k % 2 == 0 { 1. } else { -1. }
                            } else {
                                2. * (std::f64::consts::PI * (j as f64 + 1.) * (k as f64 + 0.5)
                                    / n as f64)
                                    .sin()
                            }
                        }
                        R2rKind::DstIV => {
                            2. * (std::f64::consts::PI * (j as f64 + 0.5) * (k as f64 + 0.5)
                                / n as f64)
                                .sin()
                        }
                    };
                    x * c
                })
                .sum()
        })
        .collect()
}
fn dht(input: &[Complex<f64>]) -> Vec<Complex<f64>> {
    let n = input.len();
    (0..n)
        .map(|k| {
            input
                .iter()
                .enumerate()
                .map(|(j, &x)| {
                    let a = TAU * (j * k) as f64 / n as f64;
                    x * (a.cos() + a.sin())
                })
                .sum()
        })
        .collect()
}

fn close_complex<R: FftReal + ToPrimitive>(actual: Complex<R>, expected: Complex<f64>) {
    let re = actual.re.to_f64().unwrap();
    let im = actual.im.to_f64().unwrap();
    let e = if std::mem::size_of::<R>() == 4 {
        3e-4
    } else {
        2e-10
    };
    assert!((re - expected.re).abs() < e * (1.0 + expected.re.abs()));
    assert!((im - expected.im).abs() < e * (1.0 + expected.im.abs()));
}

fn rust_defaults() {
    for n in [1, 3, 4, 5] {
        assert_eq!(
            LocalC2cPlan::<f64>::new(n).unwrap().backend_kind(),
            BackendKind::RustFft
        );
        assert_eq!(
            LocalR2cPlan::<f64>::new(n).unwrap().backend_kind(),
            BackendKind::RustFft
        );
        assert_eq!(
            LocalR2rPlan::<f64>::new(n, R2rKind::DctII)
                .unwrap()
                .backend_kind(),
            BackendKind::RustFft
        );
        assert_eq!(
            LocalDhtPlan::<f64>::new(n).unwrap().backend_kind(),
            BackendKind::RustFft
        );
        assert!(
            LocalC2cPlan::<f64>::new(n)
                .unwrap()
                .backend_options()
                .is_none()
        );
        assert!(
            LocalR2cPlan::<f64>::new(n)
                .unwrap()
                .backend_options()
                .is_none()
        );
        assert!(
            LocalR2rPlan::<f64>::new(n, R2rKind::DctII)
                .unwrap()
                .backend_options()
                .is_none()
        );
        assert!(
            LocalDhtPlan::<f64>::new(n)
                .unwrap()
                .backend_options()
                .is_none()
        );
    }
}

fn edge_native<R: FftReal + ToPrimitive + FromPrimitive>() {
    for n in [1, 3, 4, 5] {
        let source: Vec<_> = (0..n * 2)
            .map(|i| {
                Complex::new(
                    R::from_f64(i as f64 * 0.17 - 0.2).unwrap(),
                    R::from_f64(i as f64 * 0.09).unwrap(),
                )
            })
            .collect();
        let p = LocalC2cPlan::<R>::new_fftw(
            n,
            PlanOptions::new(PlanningRigor::Estimate, Some(Duration::from_millis(2)))
                .unwrap()
                .with_threads(2)
                .unwrap(),
        )
        .unwrap();
        let mut out = vec![Complex::new(R::zero(), R::zero()); source.len()];
        let mut scratch = Vec::new();
        p.forward(&[], &mut [], &mut scratch).unwrap();
        p.backward(&[], &mut [], &mut scratch).unwrap();
        p.inverse(&[], &mut [], &mut scratch).unwrap();
        p.forward_in_place(&mut [], &mut scratch).unwrap();
        p.backward_in_place(&mut [], &mut scratch).unwrap();
        p.inverse_in_place(&mut [], &mut scratch).unwrap();
        p.forward(&source, &mut out, &mut scratch).unwrap();
        let mut back = vec![Complex::new(R::zero(), R::zero()); source.len()];
        p.backward(&out, &mut back, &mut scratch).unwrap();
        for (a, e) in back.iter().zip(source.iter()) {
            close_complex(
                *a,
                Complex::new(
                    e.re.to_f64().unwrap() * n as f64,
                    e.im.to_f64().unwrap() * n as f64,
                ),
            );
        }
        for op in 0..3 {
            let mut data = source.clone();
            match op {
                0 => p.forward_in_place(&mut data, &mut scratch),
                1 => p.backward_in_place(&mut data, &mut scratch),
                _ => p.inverse_in_place(&mut data, &mut scratch),
            }
            .unwrap();
            for (input, actual) in source.chunks(n).zip(data.chunks(n)) {
                let input: Vec<_> = input
                    .iter()
                    .map(|x| Complex::new(x.re.to_f64().unwrap(), x.im.to_f64().unwrap()))
                    .collect();
                let expected = dft(&input, op != 0);
                for (&actual, expected) in actual.iter().zip(expected) {
                    close_complex(actual, expected / if op == 2 { n as f64 } else { 1.0 });
                }
            }
        }
        let d = LocalDhtPlan::<R>::new_fftw(
            n,
            PlanOptions::new(PlanningRigor::Estimate, Some(Duration::from_millis(2)))
                .unwrap()
                .with_threads(2)
                .unwrap(),
        )
        .unwrap();
        let real: Vec<R> = (0..n * 2)
            .map(|i| R::from_f64(i as f64 - 0.3).unwrap())
            .collect();
        let mut transformed = vec![R::zero(); real.len()];
        let mut emb = vec![Complex::new(R::zero(), R::zero()); d.embedding_len()];
        d.forward(&real, &mut transformed, &mut emb, &mut [])
            .unwrap();
        let mut recovered = vec![R::zero(); real.len()];
        d.backward(&transformed, &mut recovered, &mut emb, &mut [])
            .unwrap();
        for (a, e) in recovered.iter().zip(real.iter()) {
            assert!((a.to_f64().unwrap() - e.to_f64().unwrap() * n as f64).abs() < 3e-3);
        }
        d.inverse(&transformed, &mut recovered, &mut emb, &mut [])
            .unwrap();
        for (a, e) in recovered.iter().zip(real.iter()) {
            assert!((a.to_f64().unwrap() - e.to_f64().unwrap()).abs() < 3e-3);
        }
        let zeros = vec![R::zero(); n * 2];
        let mut zero_out = vec![R::zero(); n * 2];
        d.forward(&zeros, &mut zero_out, &mut emb, &mut []).unwrap();
        assert!(zero_out.iter().all(|x| x.to_f64().unwrap() == 0.));
        d.forward(&[], &mut [], &mut emb, &mut []).unwrap();
    }
}

fn r2r_reverse_and_in_place<T: Value>(plan: &LocalR2rPlan<T>, source: &[T]) {
    let n = plan.line_len();
    let mut embedding = vec![Complex::new(T::Real::zero(), T::Real::zero()); plan.embedding_len()];
    let mut scratch = vec![Complex::new(T::Real::zero(), T::Real::zero()); plan.scratch_len()];
    let mut output = vec![T::value(91.0); source.len() + 2];
    for op in 0..3 {
        let slice = &mut output[1..source.len() + 1];
        match op {
            0 => plan.forward(source, slice, &mut embedding, &mut scratch),
            1 => plan.backward(source, slice, &mut embedding, &mut scratch),
            _ => plan.inverse(source, slice, &mut embedding, &mut scratch),
        }
        .unwrap();
        let mut inplace = source.to_vec();
        match op {
            0 => plan.forward_in_place(&mut inplace, &mut embedding, &mut scratch),
            1 => plan.backward_in_place(&mut inplace, &mut embedding, &mut scratch),
            _ => plan.inverse_in_place(&mut inplace, &mut embedding, &mut scratch),
        }
        .unwrap();
        for (batch, line) in source.chunks(n).enumerate() {
            let input: Vec<_> = line
                .iter()
                .map(|x| {
                    let (r, i) = x.parts();
                    Complex::new(r, i)
                })
                .collect();
            let expected = r2r(
                if op == 0 {
                    plan.kind()
                } else {
                    plan.backward_kind()
                },
                &input,
            );
            for (k, expected) in expected.into_iter().enumerate() {
                let expected = expected
                    / if op == 2 {
                        plan.normalization_factor() as f64
                    } else {
                        1.0
                    };
                close(output[1 + batch * n + k], expected);
                close(inplace[batch * n + k], expected);
            }
        }
        assert_eq!(output[0].parts(), T::value(91.0).parts());
        assert_eq!(output[source.len() + 1].parts(), T::value(91.0).parts());
    }
    plan.forward(&[], &mut [], &mut embedding, &mut scratch)
        .unwrap();
    plan.backward(&[], &mut [], &mut embedding, &mut scratch)
        .unwrap();
    plan.inverse(&[], &mut [], &mut embedding, &mut scratch)
        .unwrap();
}

fn native<T: Value>()
where
    T::Real: ToPrimitive + FromPrimitive,
{
    eprintln!(
        "runtime {}: {}",
        std::any::type_name::<T::Real>(),
        pencil_fft::runtime_version::<T::Real>().unwrap()
    );
    let options = [
        PlanningRigor::Estimate,
        PlanningRigor::Measure,
        PlanningRigor::Patient,
        PlanningRigor::Exhaustive,
    ]
    .into_iter()
    .flat_map(|r| {
        [1, 2, 3].map(|threads| {
            PlanOptions::new(r, Some(Duration::from_millis(2)))
                .unwrap()
                .with_threads(threads)
                .unwrap()
        })
    });
    for options in options {
        for &n in &[1, 3, 4, 5] {
            let src: Vec<T> = (0..n * 2)
                .map(|i| T::value(i as f64 * 0.37 - 0.6))
                .collect();
            let source: Vec<_> = src
                .iter()
                .copied()
                .map(|x| {
                    let (re, im) = x.parts();
                    Complex::new(re, im)
                })
                .collect();
            let mut csrc = vec![Complex::new(T::Real::zero(), T::Real::zero()); 2 * n + 2];
            for (slot, value) in csrc[1..1 + 2 * n].iter_mut().zip(&source) {
                *slot = Complex::new(
                    T::Real::from_f64(value.re).unwrap(),
                    T::Real::from_f64(value.im).unwrap(),
                );
            }
            let mut cdst = vec![
                Complex::new(
                    T::Real::from_f64(77.).unwrap(),
                    T::Real::from_f64(77.).unwrap()
                );
                csrc.len()
            ];
            let cp = LocalC2cPlan::<T::Real>::new_fftw(n, options).unwrap();
            assert_eq!(cp.backend_kind(), BackendKind::Fftw);
            assert_options(cp.backend_options(), options);
            cp.forward(&csrc[1..1 + 2 * n], &mut cdst[1..1 + 2 * n], &mut [])
                .unwrap();
            for (batch, line) in cdst[1..1 + 2 * n].chunks(n).enumerate() {
                let expected_c = dft(&source[batch * n..(batch + 1) * n], false);
                for (k, &value) in line.iter().enumerate() {
                    close_complex(value, expected_c[k]);
                }
            }
            assert_eq!(cdst[0].re.to_f64().unwrap(), 77.);
            assert_eq!(cdst.last().unwrap().re.to_f64().unwrap(), 77.);
            let real_src: Vec<T::Real> = (0..n * 2)
                .map(|i| T::Real::from_f64(i as f64 * 0.23 - 0.4).unwrap())
                .collect();
            let rp = LocalR2cPlan::<T::Real>::new_fftw(n, options).unwrap();
            assert_eq!(rp.backend_kind(), BackendKind::Fftw);
            assert_options(rp.backend_options(), options);
            let m = rp.complex_len();
            let expected = real_src
                .chunks(n)
                .flat_map(|line| {
                    dft(
                        &line
                            .iter()
                            .map(|&x| Complex::new(x.to_f64().unwrap(), 0.))
                            .collect::<Vec<_>>(),
                        false,
                    )[..m]
                        .to_vec()
                })
                .collect::<Vec<_>>();
            let mut spectrum = vec![Complex::new(T::Real::zero(), T::Real::zero()); expected.len()];
            let mut rl = vec![T::Real::zero(); n + 2];
            rl[n] = T::Real::from_f64(91.).unwrap();
            rl[n + 1] = T::Real::from_f64(92.).unwrap();
            let mut cs = Vec::new();
            rp.forward(&real_src, &mut spectrum, &mut rl, &mut cs)
                .unwrap();
            assert_eq!(rl[n].to_f64().unwrap(), 91.);
            assert_eq!(rl[n + 1].to_f64().unwrap(), 92.);
            for (a, e) in spectrum.iter().zip(expected.iter()) {
                close_complex(*a, *e);
            }
            let preserved = spectrum.clone();
            let mut recovered = vec![T::Real::zero(); real_src.len()];
            let mut cl = vec![Complex::new(T::Real::zero(), T::Real::zero()); m + 2];
            cl[m] = Complex::new(T::Real::from_f64(93.).unwrap(), T::Real::zero());
            cl[m + 1] = Complex::new(T::Real::from_f64(94.).unwrap(), T::Real::zero());
            rp.inverse(&spectrum, &mut recovered, &mut cl, &mut cs)
                .unwrap();
            assert_eq!(spectrum, preserved);
            assert_eq!(cl[m].re.to_f64().unwrap(), 93.);
            assert_eq!(cl[m + 1].re.to_f64().unwrap(), 94.);
            for (a, e) in recovered.iter().zip(real_src.iter()) {
                assert!((a.to_f64().unwrap() - e.to_f64().unwrap()).abs() < 3e-4);
            }
            let arbitrary: Vec<_> = (0..2 * m)
                .map(|i| {
                    let k = i % m;
                    Complex::new(
                        T::Real::from_f64(i as f64 * 0.13 - 0.5).unwrap(),
                        if k == 0 || (n % 2 == 0 && k + 1 == m) {
                            T::Real::zero()
                        } else {
                            T::Real::from_f64(i as f64 * 0.07).unwrap()
                        },
                    )
                })
                .collect();
            rp.backward(&arbitrary, &mut recovered, &mut cl, &mut cs)
                .unwrap();
            for (b, line) in arbitrary.chunks(m).enumerate() {
                for (j, a) in recovered[b * n..(b + 1) * n].iter().enumerate() {
                    let expected: Complex<f64> = line
                        .iter()
                        .enumerate()
                        .map(|(k, x)| {
                            let q = TAU * (j * k) as f64 / n as f64;
                            let weight = if k == 0 || (n % 2 == 0 && k == n / 2) {
                                1.0
                            } else {
                                2.0
                            };
                            Complex::new(x.re.to_f64().unwrap(), x.im.to_f64().unwrap())
                                * Complex::new(q.cos(), q.sin())
                                * weight
                        })
                        .sum();
                    assert!((a.to_f64().unwrap() - expected.re).abs() < 3e-3);
                }
            }
            rp.forward(&[], &mut [], &mut rl, &mut cs).unwrap();
            rp.backward(&[], &mut [], &mut cl, &mut cs).unwrap();
            rp.inverse(&[], &mut [], &mut cl, &mut cs).unwrap();
            let mut bad = arbitrary.clone();
            bad[m].im = T::Real::from_f64(1.0).unwrap();
            let before_output = recovered.clone();
            let before_line = cl.clone();
            for inverse in [false, true] {
                let result = if inverse {
                    rp.inverse(&bad, &mut recovered, &mut cl, &mut cs)
                } else {
                    rp.backward(&bad, &mut recovered, &mut cl, &mut cs)
                };
                assert!(result.is_err());
                assert_eq!(recovered, before_output);
                assert_eq!(cl, before_line);
            }
            let mut packed = rp.allocate_in_place(2).unwrap();
            packed.real_view_mut().unwrap().copy_from_slice(&real_src);
            let mut workspace = rp.allocate_in_place_workspace().unwrap();
            rp.forward_in_place(&mut packed, &mut workspace).unwrap();
            rp.inverse_in_place(&mut packed, &mut workspace).unwrap();
            for (a, e) in packed.real_view().unwrap().iter().zip(real_src.iter()) {
                assert!((a.to_f64().unwrap() - e.to_f64().unwrap()).abs() < 3e-4);
            }

            let mut dst = vec![T::value(99.); src.len()];
            for &kind in [
                R2rKind::DstI,
                R2rKind::DctII,
                R2rKind::DctIII,
                R2rKind::DctIV,
                R2rKind::DstII,
                R2rKind::DstIII,
                R2rKind::DstIV,
            ]
            .iter()
            {
                let p = LocalR2rPlan::<T>::new_fftw(n, kind, options).unwrap();
                r2r_reverse_and_in_place(&p, &src);
                assert_eq!(p.backend_kind(), BackendKind::Fftw);
                assert_options(p.backend_options(), options);
                let mut embedding =
                    vec![Complex::new(T::Real::zero(), T::Real::zero()); p.embedding_len()];
                let mut scratch = vec![];
                p.forward(&src, &mut dst, &mut embedding, &mut scratch)
                    .unwrap();
                for (batch, line) in dst.chunks(n).enumerate() {
                    for (k, &v) in line.iter().enumerate() {
                        close(v, r2r(kind, &source[batch * n..(batch + 1) * n])[k]);
                    }
                }
            }
            if n > 1 {
                for &kind in [R2rKind::DctI, R2rKind::DstI].iter() {
                    let p = LocalR2rPlan::<T>::new_fftw(n, kind, options).unwrap();
                    r2r_reverse_and_in_place(&p, &src);
                    let mut embedding =
                        vec![Complex::new(T::Real::zero(), T::Real::zero()); p.embedding_len()];
                    let mut scratch = vec![];
                    p.forward(&src, &mut dst, &mut embedding, &mut scratch)
                        .unwrap();
                    for (batch, line) in dst.chunks(n).enumerate() {
                        for (k, &v) in line.iter().enumerate() {
                            close(v, r2r(kind, &source[batch * n..(batch + 1) * n])[k]);
                        }
                    }
                }
            }
            let d = LocalDhtPlan::<T>::new_fftw(n, options).unwrap();
            assert_eq!(d.backend_kind(), BackendKind::Fftw);
            assert_options(d.backend_options(), options);
            let mut out = vec![T::value(99.); src.len()];
            let mut embedding =
                vec![Complex::new(T::Real::zero(), T::Real::zero()); d.embedding_len()];
            let mut scratch = vec![];
            d.forward(&src, &mut out, &mut embedding, &mut scratch)
                .unwrap();
            for (batch, line) in out.chunks(n).enumerate() {
                for (k, &v) in line.iter().enumerate() {
                    close(v, dht(&source[batch * n..(batch + 1) * n])[k]);
                }
            }
        }
    }
}

#[test]
fn defaults_are_rust_plans() {
    rust_defaults();
}

fn local_wisdom<R>()
where
    R: FftReal + FromPrimitive + ToPrimitive,
{
    let trained = PlanOptions::new(PlanningRigor::Measure, None)
        .unwrap()
        .with_threads(2)
        .unwrap()
        .with_conserve_memory(true);
    let only = trained.with_wisdom_only(true);
    let n = 7;
    forget_wisdom::<R>().unwrap();
    assert!(matches!(
        LocalR2rPlan::<R>::new_fftw(n, R2rKind::DctII, only),
        Err(pencil_fft::BackendInitError::Native(
            pencil_fft::FftwError::NullPlan
        ))
    ));
    assert!(matches!(
        LocalDhtPlan::<R>::new_fftw(n, only),
        Err(pencil_fft::BackendInitError::Native(
            pencil_fft::FftwError::NullPlan
        ))
    ));

    let trained_r2r = LocalR2rPlan::<R>::new_fftw(n, R2rKind::DctII, trained).unwrap();
    let dht = LocalDhtPlan::<R>::new_fftw(n, trained).unwrap();
    assert_options(trained_r2r.backend_options(), trained);
    assert_options(dht.backend_options(), trained);
    let wisdom = export_wisdom::<R>().unwrap();
    drop((trained_r2r, dht));
    forget_wisdom::<R>().unwrap();
    import_wisdom::<R>(&wisdom).unwrap();

    let hit = LocalR2rPlan::<R>::new_fftw(n, R2rKind::DctII, only).unwrap();
    let dht_hit = LocalDhtPlan::<R>::new_fftw(n, only).unwrap();
    assert_options(hit.backend_options(), only);
    assert_options(dht_hit.backend_options(), only);
    let reference: Vec<_> = (0..n)
        .map(|j| Complex::new(j as f64 * 0.3 - 0.4, 0.0))
        .collect();
    let source: Vec<_> = reference
        .iter()
        .map(|v| R::from_f64(v.re).unwrap())
        .collect();
    let mut output = vec![R::zero(); n];
    let mut embedding = vec![Complex::new(R::zero(), R::zero()); hit.embedding_len()];
    hit.forward(&source, &mut output, &mut embedding, &mut [])
        .unwrap();
    for (actual, expected) in output.iter().zip(r2r(R2rKind::DctII, &reference)) {
        assert!((actual.to_f64().unwrap() - expected.re).abs() < 3e-4);
    }
    let mut hartley = vec![R::zero(); n];
    let mut dht_embedding = vec![Complex::new(R::zero(), R::zero()); dht_hit.embedding_len()];
    dht_hit
        .forward(&source, &mut hartley, &mut dht_embedding, &mut [])
        .unwrap();
    for (actual, expected) in hartley.iter().zip(dft(&reference, false)) {
        assert!((actual.to_f64().unwrap() - (expected.re - expected.im)).abs() < 3e-4);
    }
}

#[test]
#[ignore = "requires native libfftw3.so.3"]
fn native_f64_wisdom_only_r2r_dht() {
    local_wisdom::<f64>();
}

#[test]
#[ignore = "requires native libfftw3f.so.3"]
fn native_f32_wisdom_only_r2r_dht() {
    local_wisdom::<f32>();
}

#[test]
#[ignore = "requires native libfftw3.so.3"]
fn native_f64_all_local() {
    edge_native::<f64>();
    native::<f64>();
    native::<Complex<f64>>();
}
#[test]
#[ignore = "requires native libfftw3f.so.3"]
fn native_f32_all_local() {
    edge_native::<f32>();
    native::<f32>();
    native::<Complex<f32>>();
}
