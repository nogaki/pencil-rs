use super::*;
fn dft(input: &[Complex<f64>], inverse: bool) -> Vec<Complex<f64>> {
    (0..input.len())
        .map(|k| {
            input
                .iter()
                .enumerate()
                .map(|(j, &v)| {
                    let angle =
                        (if inverse { 1.0 } else { -1.0 }) * std::f64::consts::TAU * (j * k) as f64
                            / input.len() as f64;
                    v * Complex::new(angle.cos(), angle.sin())
                })
                .sum()
        })
        .collect()
}
fn c<R: Real>(v: f64) -> R {
    R::from_f64(v).unwrap()
}
fn near<R: Real + rustfft::num_traits::ToPrimitive>(a: Complex<R>, b: Complex<f64>) {
    assert!(
        (a.re.to_f64().unwrap() - b.re).abs() < 2e-4
            && (a.im.to_f64().unwrap() - b.im).abs() < 2e-4,
        "{a:?} != {b:?}"
    );
}
#[test]
fn host_contracts() {
    assert!(checked_len::<f32>(0).is_err());
    assert!(matches!(
        checked_len::<f64>(i32::MAX as usize + 1),
        Err(FftwError::Overflow)
    ));
    assert_eq!(checked_len::<f64>(1).unwrap(), 1);
    assert!(zeros::<f64>(usize::MAX).is_err());
    assert_eq!(
        zeros::<Complex<f32>>(13).unwrap(),
        vec![Complex::default(); 13]
    );
    assert!(PlanOptions::new(PlanningRigor::Measure, Some(Duration::ZERO)).is_err());
    for (rigor, flags) in [
        (PlanningRigor::Estimate, 66),
        (PlanningRigor::Measure, 2),
        (PlanningRigor::Patient, 34),
        (PlanningRigor::Exhaustive, 10),
    ] {
        let o = PlanOptions::new(rigor, Some(Duration::from_millis(1))).unwrap();
        assert_eq!(o.flags(), flags);
        assert_eq!(o.rigor(), rigor);
        assert_eq!(o.time_limit(), Some(Duration::from_millis(1)));
    }
    // All factory dimension checks precede loading or allocation.
    assert!(matches!(
        plan_c2c::<f32>(0, FftDirection::Forward, PlanOptions::default()),
        Err(FftwError::InvalidOptions(_))
    ));
    assert!(matches!(
        plan_c2r::<f64>(usize::MAX, PlanOptions::default()),
        Err(FftwError::Overflow)
    ));
    assert!(matches!(
        plan_r2c::<f64>(0, PlanOptions::default()),
        Err(FftwError::InvalidOptions(_))
    ));
}
fn matrix<R: Real + rustfft::num_traits::ToPrimitive>() {
    eprintln!(
        "runtime {}: {}",
        std::any::type_name::<R>(),
        runtime_version::<R>().expect("native runtime REQUIRED")
    );
    for rigor in [
        PlanningRigor::Estimate,
        PlanningRigor::Measure,
        PlanningRigor::Patient,
        PlanningRigor::Exhaustive,
    ] {
        let options = PlanOptions::new(rigor, Some(Duration::from_millis(2))).unwrap();
        for n in [1, 3, 4, 7, 8] {
            let source: Vec<_> = (0..n)
                .map(|j| Complex::new(j as f64 * 0.3 - 0.4, 0.2 - j as f64 * 0.1))
                .collect();
            let input: Vec<_> = source
                .iter()
                .map(|v| Complex::new(c::<R>(v.re), c::<R>(v.im)))
                .collect();
            for direction in [FftDirection::Forward, FftDirection::Inverse] {
                let p = plan_c2c::<R>(n, direction, options).unwrap();
                let expected = dft(&source, direction == FftDirection::Inverse);
                let mut batch = vec![Complex::new(c::<R>(99.0), c::<R>(99.0)); 2 * n + 2];
                batch[1..n + 1].copy_from_slice(&input);
                batch[n + 1..2 * n + 1].copy_from_slice(&input);
                let original = batch.clone();
                let mut output = batch.clone();
                let mut scratch = vec![Complex::new(c::<R>(42.0), c::<R>(42.0)); 2];
                let scratch_before = scratch.clone();
                let executions = ffi::COMPLEX_EXECUTIONS.with(|count| count.get());
                p.process_immutable_with_scratch(
                    &batch[1..2 * n + 1],
                    &mut output[1..2 * n + 1],
                    &mut scratch,
                );
                assert_eq!(batch, original);
                assert_eq!(scratch, scratch_before);
                for (k, &v) in output[1..2 * n + 1].iter().enumerate() {
                    near(v, expected[k % n]);
                }
                assert_eq!(output[0], original[0]);
                assert_eq!(output[2 * n + 1], original[2 * n + 1]);
                p.process_with_scratch(&mut batch[1..2 * n + 1], &mut scratch);
                assert_eq!(batch, output);
                batch.clone_from(&original);
                p.process_outofplace_with_scratch(
                    &mut batch[1..2 * n + 1],
                    &mut output[1..2 * n + 1],
                    &mut [],
                );
                assert_eq!(batch, original);
                let after_valid = ffi::COMPLEX_EXECUTIONS.with(|count| count.get());
                assert_eq!(after_valid - executions, 6);
                // All short lengths, including zero, must panic before native
                // execution without changing input, output, scratch or tails.
                for len in 0..n {
                    for form in 0..4 {
                        let input_before = batch.clone();
                        let output_before = output.clone();
                        assert!(
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match form {
                                0 => p.process(&mut batch[1..1 + len]),
                                1 => p.process_with_scratch(&mut batch[1..1 + len], &mut scratch,),
                                2 => p.process_outofplace_with_scratch(
                                    &mut batch[1..1 + len],
                                    &mut output[1..1 + len],
                                    &mut scratch,
                                ),
                                _ => p.process_immutable_with_scratch(
                                    &batch[1..1 + len],
                                    &mut output[1..1 + len],
                                    &mut scratch,
                                ),
                            }))
                            .is_err()
                        );
                        assert_eq!(batch, input_before);
                        assert_eq!(output, output_before);
                        assert_eq!(scratch, scratch_before);
                        assert_eq!(
                            ffi::COMPLEX_EXECUTIONS.with(|count| count.get()),
                            after_valid
                        );
                    }
                }
                let before = output.clone();
                assert!(
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                        || p.process_immutable_with_scratch(&input, &mut output[..n - 1], &mut [])
                    ))
                    .is_err()
                );
                assert_eq!(output, before);
                if n > 1 {
                    assert!(
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                            || p.process_with_scratch(&mut batch[..n + 1], &mut [])
                        ))
                        .is_err()
                    );
                    assert_eq!(batch, original);
                }
                // Shared plan survives original Arc and executes concurrently.
                let handles: Vec<_> = (0..3)
                    .map(|_| {
                        let p = p.clone();
                        let mut i = input.clone();
                        std::thread::spawn(move || {
                            p.process(&mut i);
                            i
                        })
                    })
                    .collect();
                drop(p);
                for h in handles {
                    for (v, e) in h.join().unwrap().into_iter().zip(&expected) {
                        near(v, *e);
                    }
                }
            }
            let forward = plan_r2c::<R>(n, options).unwrap();
            let inverse = plan_c2r::<R>(n, options).unwrap();
            assert!(forward.make_scratch_vec().is_empty());
            assert!(inverse.make_scratch_vec().is_empty());
            assert!(forward.make_input_vec().iter().all(|&v| v == R::zero()));
            assert!(inverse.make_output_vec().iter().all(|&v| v == R::zero()));
            let mut real = vec![c::<R>(99.0); n + 2];
            for j in 0..n {
                real[j + 1] = c(source[j].re);
            }
            let mut spectrum = vec![Complex::new(c::<R>(99.0), c::<R>(99.0)); n / 2 + 3];
            forward
                .process(&mut real[1..n + 1], &mut spectrum[1..n / 2 + 2])
                .unwrap();
            let real_source: Vec<_> = source.iter().map(|v| Complex::new(v.re, 0.0)).collect();
            let expected = dft(&real_source, false);
            for k in 0..n / 2 + 1 {
                near(spectrum[k + 1], expected[k]);
            }
            assert_eq!(real[0], c(99.0));
            assert_eq!(real[n + 1], c(99.0));
            assert_eq!(spectrum[0].re, c(99.0));
            assert_eq!(spectrum[n / 2 + 2].re, c(99.0));
            inverse
                .process(&mut spectrum[1..n / 2 + 2], &mut real[1..n + 1])
                .unwrap();
            for j in 0..n {
                near(
                    Complex::new(real[j + 1], R::zero()),
                    Complex::new(n as f64 * source[j].re, 0.0),
                );
            }
            // Independent arbitrary Hermitian spectrum, not just a round trip.
            let mut half = inverse.make_input_vec();
            let mut full = vec![Complex::default(); n];
            for k in 0..half.len() {
                let z = Complex::new(
                    0.3 + k as f64,
                    if k == 0 || (n % 2 == 0 && k == n / 2) {
                        0.0
                    } else {
                        -0.2 * k as f64
                    },
                );
                half[k] = Complex::new(c(z.re), c(z.im));
                full[k] = z;
                if k > 0 && n - k != k {
                    full[n - k] = z.conj();
                }
            }
            let expected = dft(&full, true);
            let mut out = inverse.make_output_vec();
            inverse.process(&mut half, &mut out).unwrap();
            for j in 0..n {
                near(Complex::new(out[j], R::zero()), expected[j]);
            }
            let mut ri = forward.make_input_vec();
            let mut co = forward.make_output_vec();
            let ri_before = ri.clone();
            let co_before = co.clone();
            assert!(matches!(
                forward.process(&mut ri[..n - 1], &mut co),
                Err(FftError::InputBuffer(_, _))
            ));
            assert_eq!(ri, ri_before);
            assert_eq!(co, co_before);
            let short = co.len() - 1;
            assert!(matches!(
                forward.process(&mut ri, &mut co[..short]),
                Err(FftError::OutputBuffer(_, _))
            ));
            assert_eq!(ri, ri_before);
            assert_eq!(co, co_before);
            assert!(matches!(
                inverse.process(&mut co[..short], &mut ri),
                Err(FftError::InputBuffer(_, _))
            ));
            assert!(matches!(
                inverse.process(&mut co, &mut ri[..n - 1]),
                Err(FftError::OutputBuffer(_, _))
            ));
            assert_eq!(ri, ri_before);
            assert_eq!(co, co_before);
            // Repeated real batches use trait-sized calls, with shared plans
            // surviving the factory owner's Arc and concurrent same-precision planning.
            let handles: Vec<_> = (0..3)
                .map(|_| {
                    let f = forward.clone();
                    let b = inverse.clone();
                    std::thread::spawn(move || {
                        let mut scratch = vec![Complex::new(c::<R>(17.0), c::<R>(19.0)); 4];
                        let saved = scratch.clone();
                        for _ in 0..3 {
                            let mut i = vec![c::<R>(1.0); n];
                            let mut o = f.make_output_vec();
                            f.process_with_scratch(&mut i, &mut o, &mut scratch)
                                .unwrap();
                            b.process_with_scratch(&mut o, &mut i, &mut scratch)
                                .unwrap();
                            for v in i {
                                near(Complex::new(v, R::zero()), Complex::new(n as f64, 0.0));
                            }
                            assert_eq!(scratch, saved);
                            plan_r2c::<R>(n, options).unwrap();
                        }
                    })
                })
                .collect();
            co[0].im = c(1.0);
            if n % 2 == 0 {
                co[n / 2].im = c(1.0);
            }
            assert!(
                matches!(inverse.process(&mut co, &mut ri), Err(FftError::InputValues(true, last)) if last == (n%2==0))
            );
            drop(forward);
            drop(inverse);
            for h in handles {
                h.join().unwrap();
            }
        }
    }
    // Unbounded estimate after bounded planning exercises the reset/default path.
    plan_c2c::<R>(5, FftDirection::Forward, PlanOptions::default()).unwrap();
}
#[test]
#[ignore = "requires native libfftw3.so.3"]
fn native_f64() {
    matrix::<f64>();
}
#[test]
#[ignore = "requires native libfftw3f.so.3"]
fn native_f32() {
    matrix::<f32>();
}
