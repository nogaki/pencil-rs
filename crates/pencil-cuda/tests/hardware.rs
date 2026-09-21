#![allow(clippy::needless_range_loop)]

use num_complex::Complex;
use pencil_cuda::{
    C2CPlanF32, C2CPlanF64, C2RPlanF32, C2RPlanF64, CudaBuffer, CudaDevice, R2CPlanF32, R2CPlanF64,
};
use std::f64::consts::PI;

fn device() -> CudaDevice {
    CudaDevice::new().expect("explicit hardware run requires CUDA driver, device and cuFFT")
}

fn close(a: f64, b: f64, tolerance: f64) {
    let e = tolerance * (1.0 + a.abs().max(b.abs()));
    assert!((a - b).abs() <= e, "{a} != {b}");
}

macro_rules! hardware_tests {
    ($mod:ident, $r:ty, $c:ty, $c2c:ty, $r2c:ty, $c2r:ty) => {
        mod $mod {
            use super::*;

            fn c(re: f64, im: f64) -> $c {
                Complex::new(re as $r, im as $r)
            }
            fn dft(x: &[$c], inverse: bool, normalized: bool) -> Vec<$c> {
                let n = x.len();
                (0..n)
                    .map(|k| {
                        let mut z = Complex::<f64>::new(0.0, 0.0);
                        for (j, value) in x.iter().enumerate() {
                            let angle = 2.0 * PI * (j * k) as f64 / n as f64;
                            let angle = if inverse { angle } else { -angle };
                            z += Complex::new(value.re as f64, value.im as f64)
                                * Complex::new(angle.cos(), angle.sin());
                        }
                        if normalized {
                            z /= n as f64;
                        }
                        c(z.re, z.im)
                    })
                    .collect()
            }
            fn real_dft(x: &[$r]) -> Vec<$c> {
                let values: Vec<_> = x.iter().map(|v| c(*v as f64, 0.0)).collect();
                dft(&values, false, false)[..x.len() / 2 + 1].to_vec()
            }
            fn c2r_oracle(h: &[$c], n: usize, normalized: bool) -> Vec<$r> {
                let mut full = vec![c(0.0, 0.0); n];
                full[..h.len()].copy_from_slice(h);
                for k in 1..n / 2 + (n % 2) {
                    full[n - k] = h[k].conj();
                }
                dft(&full, true, normalized)
                    .into_iter()
                    .map(|z| z.re)
                    .collect()
            }
            fn check_complex(got: &[$c], want: &[$c]) {
                assert_eq!(got.len(), want.len());
                for (a, b) in got.iter().zip(want) {
                    close(
                        a.re as f64,
                        b.re as f64,
                        if <$r>::EPSILON > 1e-10 as $r {
                            4e-4
                        } else {
                            1e-9
                        },
                    );
                    close(
                        a.im as f64,
                        b.im as f64,
                        if <$r>::EPSILON > 1e-10 as $r {
                            4e-4
                        } else {
                            1e-9
                        },
                    );
                }
            }
            fn check_real(got: &[$r], want: &[$r]) {
                assert_eq!(got.len(), want.len());
                for (a, b) in got.iter().zip(want) {
                    close(
                        *a as f64,
                        *b as f64,
                        if <$r>::EPSILON > 1e-10 as $r {
                            4e-4
                        } else {
                            1e-9
                        },
                    );
                }
            }

            fn run() {
                let d = device();
                for &n in &[1usize, 2, 3, 4, 5, 8, 129, 257] {
                    let batch = 2;
                    let x: Vec<$c> = (0..n * batch)
                        .map(|i| c((i as f64 * 0.71).sin(), (i as f64 * 0.31).cos()))
                        .collect();
                    let mut src = CudaBuffer::<$c>::new(&d, x.len()).unwrap();
                    let mut out = CudaBuffer::<$c>::new(&d, x.len()).unwrap();
                    src.upload(&x).unwrap();
                    let before = src.download().unwrap();
                    let p = <$c2c>::new(&d, n, batch).unwrap();
                    p.execute(&src, &mut out, false).unwrap();
                    let got = out.download().unwrap();
                    for b in 0..batch {
                        check_complex(
                            &got[b * n..(b + 1) * n],
                            &dft(&x[b * n..(b + 1) * n], false, false),
                        );
                    }
                    assert_eq!(src.download().unwrap(), before);
                    p.execute(&out, &mut src, true).unwrap();
                    let got = src.download().unwrap();
                    for b in 0..batch {
                        check_complex(&got[b * n..(b + 1) * n], &x[b * n..(b + 1) * n]);
                    }
                    p.execute_inverse(&out, &mut src).unwrap();
                    let got = src.download().unwrap();
                    for b in 0..batch {
                        check_complex(&got[b * n..(b + 1) * n], &x[b * n..(b + 1) * n]);
                    }
                    let spectrum = out.download().unwrap();
                    p.execute_backward(&out, &mut src).unwrap();
                    let got = src.download().unwrap();
                    for b in 0..batch {
                        check_complex(
                            &got[b * n..(b + 1) * n],
                            &dft(&spectrum[b * n..(b + 1) * n], true, false),
                        );
                    }
                    let mut inplace = CudaBuffer::<$c>::new(&d, n).unwrap();
                    inplace.upload(&x[..n]).unwrap();
                    let ip = <$c2c>::new(&d, n, 1).unwrap();
                    ip.execute_in_place(&mut inplace, false).unwrap();
                    check_complex(&inplace.download().unwrap(), &dft(&x[..n], false, false));
                    ip.execute_in_place(&mut inplace, true).unwrap();
                    check_complex(&inplace.download().unwrap(), &x[..n]);
                    inplace.upload(&x[..n]).unwrap();
                    ip.execute_backward_in_place(&mut inplace).unwrap();
                    check_complex(&inplace.download().unwrap(), &dft(&x[..n], true, false));
                    inplace.upload(&x[..n]).unwrap();
                    ip.execute_in_place(&mut inplace, true).unwrap();
                    check_complex(&inplace.download().unwrap(), &dft(&x[..n], true, true));

                    let r: Vec<$r> = (0..n * batch)
                        .map(|i| (i as f64 * 0.43).sin() as $r)
                        .collect();
                    let mut rb = CudaBuffer::<$r>::new(&d, r.len()).unwrap();
                    let mut sb = CudaBuffer::<$c>::new(&d, (n / 2 + 1) * batch).unwrap();
                    rb.upload(&r).unwrap();
                    let rp = <$r2c>::new(&d, n, batch).unwrap();
                    rp.execute(&rb, &mut sb).unwrap();
                    let spectrum = sb.download().unwrap();
                    for b in 0..batch {
                        check_complex(
                            &spectrum[b * (n / 2 + 1)..(b + 1) * (n / 2 + 1)],
                            &real_dft(&r[b * n..(b + 1) * n]),
                        );
                    }
                    assert_eq!(rb.download().unwrap(), r);
                    let mut back = CudaBuffer::<$r>::new(&d, r.len()).unwrap();
                    let cp = <$c2r>::new(&d, n, batch).unwrap();
                    cp.execute(&sb, &mut back).unwrap();
                    assert_eq!(sb.download().unwrap(), spectrum);
                    let got = back.download().unwrap();
                    for b in 0..batch {
                        check_real(&got[b * n..(b + 1) * n], &r[b * n..(b + 1) * n]);
                    }
                    cp.execute_backward(&sb, &mut back).unwrap();
                    let got = back.download().unwrap();
                    for b in 0..batch {
                        check_real(
                            &got[b * n..(b + 1) * n],
                            &r[b * n..(b + 1) * n]
                                .iter()
                                .map(|v| *v * n as $r)
                                .collect::<Vec<_>>(),
                        );
                    }
                }
            }

            #[test]
            #[ignore = "requires a CUDA driver and cuFFT"]
            fn direct_dft_oracle_all_lengths_batches_and_tails() {
                run();
            }

            #[test]
            #[ignore = "requires a CUDA driver and cuFFT"]
            fn arbitrary_spectra_endpoints_and_prewrite_errors() {
                let d = device();
                let n = 7;
                let mut input = CudaBuffer::<$c>::new(&d, n / 2 + 1).unwrap();
                let mut output = CudaBuffer::<$r>::new(&d, n).unwrap();
                let spectrum = vec![c(2.0, 0.0), c(-1.0, 3.0), c(0.5, -0.25), c(4.0, 2.0)];
                input.upload(&spectrum).unwrap();
                let p = <$c2r>::new(&d, n, 1).unwrap();
                p.execute(&input, &mut output).unwrap();
                check_real(&output.download().unwrap(), &c2r_oracle(&spectrum, n, true));
                p.execute_backward(&input, &mut output).unwrap();
                check_real(
                    &output.download().unwrap(),
                    &c2r_oracle(&spectrum, n, false),
                );
                assert_eq!(input.download().unwrap(), spectrum);
                let sentinel = vec![7.0 as $r; n];
                output.upload(&sentinel).unwrap();
                let mut bad = spectrum.clone();
                bad[0].im = 1.0 as $r;
                input.upload(&bad).unwrap();
                assert!(p.execute(&input, &mut output).is_err());
                assert_eq!(output.download().unwrap(), sentinel);
                // Even-length Nyquist, NaN and infinity imaginary endpoints reject pre-write.
                let p = <$c2r>::new(&d, 6, 1).unwrap();
                let mut output = CudaBuffer::<$r>::new(&d, 6).unwrap();
                let sentinel = vec![7.0 as $r; 6];
                for bin in [0, 3] {
                    for im in [1.0, f64::NAN, f64::INFINITY] {
                        let mut bad = spectrum.clone();
                        bad[3].im = 0.0 as $r;
                        bad[bin].im = im as $r;
                        input.upload(&bad).unwrap();
                        output.upload(&sentinel).unwrap();
                        assert!(p.execute(&input, &mut output).is_err());
                        assert_eq!(output.download().unwrap(), sentinel);
                    }
                }
                // Real non-finite endpoints are permitted by the CPU contract.
                for re in [f64::NAN, f64::INFINITY] {
                    let mut valid = spectrum.clone();
                    valid[0] = c(re, 0.0);
                    valid[3].im = 0.0 as $r;
                    input.upload(&valid).unwrap();
                    p.execute(&input, &mut output).unwrap();
                }
            }

            #[test]
            #[ignore = "requires CUDA driver and cuFFT"]
            fn arbitrary_valid_spectra_both_directions_and_batches() {
                let d = device();
                for n in [1, 2, 3, 6, 7] {
                    let width = n / 2 + 1;
                    let values: Vec<_> = (0..2 * width)
                        .map(|i| {
                            let bin = i % width;
                            c(
                                i as f64 - 0.75,
                                if bin == 0 || (n % 2 == 0 && bin == n / 2) {
                                    -0.0
                                } else {
                                    i as f64 * 0.2
                                },
                            )
                        })
                        .collect();
                    let mut input = CudaBuffer::<$c>::new(&d, values.len()).unwrap();
                    input.upload(&values).unwrap();
                    let mut out = CudaBuffer::<$r>::new(&d, 2 * n).unwrap();
                    let p = <$c2r>::new(&d, n, 2).unwrap();
                    for normalized in [false, true] {
                        if normalized {
                            p.execute_inverse(&input, &mut out).unwrap();
                        } else {
                            p.execute_backward(&input, &mut out).unwrap();
                        }
                        let got = out.download().unwrap();
                        for b in 0..2 {
                            check_real(
                                &got[b * n..(b + 1) * n],
                                &c2r_oracle(&values[b * width..(b + 1) * width], n, normalized),
                            );
                        }
                        assert_eq!(input.download().unwrap(), values);
                    }
                }
            }

            #[test]
            #[ignore = "requires two CUDA contexts"]
            fn wrong_context_is_rejected_without_writes() {
                let a = device();
                let b = device();
                let input = CudaBuffer::<$c>::new(&a, 1).unwrap();
                let mut output = CudaBuffer::<$c>::new(&b, 1).unwrap();
                let p = <$c2c>::new(&a, 1, 1).unwrap();
                output.upload(&[c(9.0, -2.0)]).unwrap();
                assert!(p.execute(&input, &mut output, false).is_err());
                assert_eq!(output.download().unwrap(), vec![c(9.0, -2.0)]);
                let mut real = CudaBuffer::<$r>::new(&b, 1).unwrap();
                real.upload(&[9.0 as $r]).unwrap();
                assert!(
                    <$c2r>::new(&a, 1, 1)
                        .unwrap()
                        .execute(&input, &mut real)
                        .is_err()
                );
                assert_eq!(real.download().unwrap(), vec![9.0 as $r]);
                assert!(
                    <$r2c>::new(&a, 1, 1)
                        .unwrap()
                        .execute(&real, &mut output)
                        .is_err()
                );
                assert_eq!(output.download().unwrap(), vec![c(9.0, -2.0)]);
            }
        }
    };
}

hardware_tests!(
    f32_hardware,
    f32,
    Complex<f32>,
    C2CPlanF32,
    R2CPlanF32,
    C2RPlanF32
);
hardware_tests!(
    f64_hardware,
    f64,
    Complex<f64>,
    C2CPlanF64,
    R2CPlanF64,
    C2RPlanF64
);

#[test]
#[ignore = "requires CUDA driver and cuFFT"]
fn empty_batches_zero_initialization_and_native_bounds() {
    let d = device();
    let zero = CudaBuffer::<f64>::new(&d, 19).unwrap();
    assert_eq!(zero.download().unwrap(), vec![0.0; 19]);
    let a = CudaBuffer::<Complex<f64>>::new(&d, 0).unwrap();
    let mut b = CudaBuffer::<Complex<f64>>::new(&d, 0).unwrap();
    let mut real = CudaBuffer::<f64>::new(&d, 0).unwrap();
    C2CPlanF64::new(&d, 3, 0)
        .unwrap()
        .execute_inverse(&a, &mut b)
        .unwrap();
    R2CPlanF64::new(&d, 3, 0)
        .unwrap()
        .execute(&real, &mut b)
        .unwrap();
    C2RPlanF64::new(&d, 3, 0)
        .unwrap()
        .execute(&a, &mut real)
        .unwrap();
    assert!(C2CPlanF64::new(&d, 0, 0).is_err());
    assert!(R2CPlanF64::new(&d, i32::MAX as usize + 1, 0).is_err());
    assert!(C2RPlanF64::new(&d, 1, i32::MAX as usize + 1).is_err());
    assert!(CudaBuffer::<Complex<f64>>::new(&d, usize::MAX).is_err());
}
