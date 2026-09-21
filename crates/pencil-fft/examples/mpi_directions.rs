//! Independent per-axis directional MPI regression (one MPI initialization).
#[cfg(feature = "distributed")]
mod checks {
    use mpi::topology::Communicator;
    use pencil_array::{ExtraShape, MpiTopology};
    use pencil_fft::{
        AxisR2rKind, AxisSelection, AxisTransform, C2cPlan, Complex, DistributedLayout, FftReal,
        FourierDirection::{self, Backward as B, Forward as F},
        FourierDirections, MixedC2cPlan, MixedR2cPlan, R2rKind, TransposeMethod,
    };
    use std::sync::Arc;
    type C = Complex<f64>;
    trait Real: FftReal + mpi::datatype::Equivalence {
        fn cast(x: f64) -> Self;
        fn value(x: Self) -> f64;
    }
    impl Real for f32 {
        fn cast(x: f64) -> Self {
            x as Self
        }
        fn value(x: Self) -> f64 {
            x as f64
        }
    }
    impl Real for f64 {
        fn cast(x: f64) -> Self {
            x
        }
        fn value(x: Self) -> f64 {
            x
        }
    }
    fn near<R: Real>(a: Complex<R>, b: C) {
        let tol = if std::mem::size_of::<R>() == 4 {
            2e-4
        } else {
            2e-11
        };
        assert!(
            (R::value(a.re) - b.re).abs() < tol * (1. + b.norm())
                && (R::value(a.im) - b.im).abs() < tol * (1. + b.norm()),
            "{a:?} != {b:?}"
        );
    }
    fn convert<R: Real>(a: C) -> Complex<R> {
        Complex::new(R::cast(a.re), R::cast(a.im))
    }
    fn coord(s: [usize; 3], i: usize) -> [usize; 3] {
        [i / (s[1] * s[2]), i / s[2] % s[1], i % s[2]]
    }
    fn index(s: [usize; 3], q: [usize; 3]) -> usize {
        (q[0] * s[1] + q[1]) * s[2] + q[2]
    }
    fn values(s: [usize; 3], e: usize, inverse: bool, real: bool) -> Vec<C> {
        (0..s.iter().product())
            .map(|i| {
                let t = (i + 13 * e + usize::from(inverse) * 31) as f64;
                C::new(
                    (t * 0.71).sin() + t * 0.013,
                    if real { 0. } else { (t * 0.37).cos() - 0.2 },
                )
            })
            .collect()
    }
    fn scale(s: [usize; 3], tr: [AxisTransform; 3]) -> f64 {
        (0..3)
            .map(|a| match tr[a] {
                AxisTransform::None => 1.,
                AxisTransform::R2r(AxisR2rKind::Fftw(R2rKind::DctII)) => 2. * s[a] as f64,
                _ => s[a] as f64,
            })
            .product()
    }
    // Direct separable sums; no FFT backend or distributed results are used as an oracle.
    fn direct(
        mut v: Vec<C>,
        s: [usize; 3],
        tr: [AxisTransform; 3],
        dirs: [FourierDirection; 3],
        inverse: bool,
    ) -> Vec<C> {
        for a in 0..3 {
            if tr[a] == AxisTransform::None {
                continue;
            }
            let mut out = v.clone();
            for (i, slot) in out.iter_mut().enumerate() {
                let mut q = coord(s, i);
                let k = q[a];
                let n = s[a] as f64;
                *slot = C::new(0., 0.);
                for j in 0..s[a] {
                    q[a] = j;
                    let theta = std::f64::consts::TAU * j as f64 * k as f64 / n;
                    let weight = match tr[a] {
                        AxisTransform::Fft | AxisTransform::Rfft => {
                            let sign = if (dirs[a] == F) != inverse { -1. } else { 1. };
                            C::from_polar(1., sign * theta)
                        }
                        AxisTransform::R2r(AxisR2rKind::Dht) => {
                            C::new(theta.cos() + theta.sin(), 0.)
                        }
                        AxisTransform::R2r(AxisR2rKind::Fftw(R2rKind::DctII)) => {
                            let w = if inverse {
                                if j == 0 {
                                    1.
                                } else {
                                    2. * (std::f64::consts::PI * j as f64 * (k as f64 + 0.5) / n)
                                        .cos()
                                }
                            } else {
                                2. * (std::f64::consts::PI * (j as f64 + 0.5) * k as f64 / n).cos()
                            };
                            C::new(w, 0.)
                        }
                        _ => unreachable!(),
                    };
                    *slot += v[index(s, q)] * weight;
                }
            }
            v = out;
        }
        v
    }
    fn real_inverse(v: Vec<C>, s: [usize; 3], dirs: [FourierDirection; 3]) -> Vec<C> {
        let reduced = [s[0], s[1] / 2 + 1, s[2]];
        let v = direct(
            v,
            reduced,
            [AxisTransform::Fft, AxisTransform::None, AxisTransform::None],
            dirs,
            true,
        );
        let full = (0..s.iter().product())
            .map(|i| {
                let mut q = coord(s, i);
                if q[1] <= s[1] / 2 {
                    v[index(reduced, q)]
                } else {
                    q[1] = s[1] - q[1];
                    v[index(reduced, q)].conj()
                }
            })
            .collect();
        direct(
            full,
            s,
            [
                AxisTransform::None,
                AxisTransform::Rfft,
                AxisTransform::None,
            ],
            dirs,
            true,
        )
        .into_iter()
        .map(|x| C::new(x.re, 0.))
        .collect()
    }
    macro_rules! fill_complex {
        ($array:expr,$s:expr,$inverse:expr) => {{
            #[allow(unused_mut)]
            let mut a = $array;
            for e in 0..2 {
                for (i, x) in values($s, e, $inverse, false).into_iter().enumerate() {
                    if let Some(slot) = a.get_global_mut(&[e], coord($s, i)) {
                        *slot = convert(x);
                    }
                }
            }
        }};
    }
    macro_rules! check_complex {
        ($array:expr,$s:expr,$want:expr) => {{
            let a = $array;
            for e in 0..2 {
                let want = $want(e);
                for (i, x) in want.into_iter().enumerate() {
                    if let Some(slot) = a.get_global(&[e], coord($s, i)) {
                        near(*slot, x);
                    }
                }
            }
        }};
    }
    macro_rules! complex_checks {
        ($p:expr,$s:expr,$tr:expr,$dirs:expr,$ws:ident) => {{
            let p = $p;
            let s = $s;
            let tr = $tr;
            let dirs = $dirs;
            let forward = |e| direct(values(s, e, false, false), s, tr, dirs, false);
            let backward = |e| direct(values(s, e, true, false), s, tr, dirs, true);
            let inverse = |e| {
                backward(e)
                    .into_iter()
                    .map(|x| x / scale(s, tr))
                    .collect::<Vec<_>>()
            };
            let mut input = p.allocate_input().unwrap();
            fill_complex!(&mut input, s, false);
            let before = input.as_slice().to_vec();
            let mut output = p.allocate_output().unwrap();
            let mut ws = p.$ws().unwrap();
            p.forward(&input, &mut output, &mut ws).unwrap();
            check_complex!(&output, s, forward);
            assert_eq!(input.as_slice(), before);
            fill_complex!(&mut output, s, true);
            let spectrum = output.as_slice().to_vec();
            p.inverse(&output, &mut input, &mut ws).unwrap();
            check_complex!(&input, s, inverse);
            p.backward(&output, &mut input, &mut ws).unwrap();
            check_complex!(&input, s, backward);
            assert_eq!(output.as_slice(), spectrum);
            let mut ip = p.allocate_in_place().unwrap();
            let mut iw = p.allocate_in_place_workspace().unwrap();
            fill_complex!(ip.view_mut().unwrap(), s, false);
            p.forward_in_place(&mut ip, &mut iw).unwrap();
            check_complex!(ip.view().unwrap(), s, forward);
            fill_complex!(ip.view_mut().unwrap(), s, true);
            p.inverse_in_place(&mut ip, &mut iw).unwrap();
            check_complex!(ip.view().unwrap(), s, inverse);
            p.forward_in_place(&mut ip, &mut iw).unwrap();
            fill_complex!(ip.view_mut().unwrap(), s, true);
            p.backward_in_place(&mut ip, &mut iw).unwrap();
            check_complex!(ip.view().unwrap(), s, backward);
        }};
    }
    fn run<R: Real>(top: &Arc<MpiTopology<1>>)
    where
        Complex<R>: mpi::datatype::Equivalence,
    {
        let s = [2, 3, 4];
        for method in [TransposeMethod::AllToAllv, TransposeMethod::PointToPoint] {
            for permute_dims in [false, true] {
                let layout = DistributedLayout {
                    transpose_method: method,
                    permute_dims,
                };
                for (selected, dirs) in [
                    ([true; 3], [B; 3]),
                    ([true; 3], [F, B, F]),
                    ([true, true, false], [F, B, F]),
                ] {
                    let selection =
                        AxisSelection::from_indices((0..3).filter(|&a| selected[a])).unwrap();
                    let base = C2cPlan::<R, 3, 1>::from_shape_with_selection_and_layout(
                        Arc::clone(top),
                        s,
                        ExtraShape::new([2]).unwrap(),
                        selection,
                        layout,
                    )
                    .unwrap();
                    let p = base
                        .with_fft_directions(FourierDirections::new(dirs))
                        .unwrap();
                    let tr = selected.map(|yes| {
                        if yes {
                            AxisTransform::Fft
                        } else {
                            AxisTransform::None
                        }
                    });
                    complex_checks!(p, s, tr, dirs, allocate_out_of_place_workspace);
                }
                let tr = [
                    AxisTransform::R2r(AxisR2rKind::Fftw(R2rKind::DctII)),
                    AxisTransform::Fft,
                    AxisTransform::R2r(AxisR2rKind::Dht),
                ];
                let base = MixedC2cPlan::<R, 3, 1>::from_shape_with_layout(
                    Arc::clone(top),
                    s,
                    ExtraShape::new([2]).unwrap(),
                    tr,
                    layout,
                )
                .unwrap();
                let p = base
                    .with_fft_directions(FourierDirections::new([F, B, F]))
                    .unwrap();
                complex_checks!(p, s, tr, [F, B, F], allocate_workspace);
                for n in [3, 4] {
                    real::<R>(top, [3, n, 2], layout);
                }
            }
        }
        negatives::<R>(top);
    }
    fn real<R: Real>(top: &Arc<MpiTopology<1>>, s: [usize; 3], layout: DistributedLayout)
    where
        Complex<R>: mpi::datatype::Equivalence,
    {
        let tr = [AxisTransform::Fft, AxisTransform::Rfft, AxisTransform::None];
        let dirs = [B, F, F];
        let base = MixedR2cPlan::<R, 3, 1>::from_shape_with_layout(
            Arc::clone(top),
            s,
            ExtraShape::new([2]).unwrap(),
            tr,
            layout,
        )
        .unwrap();
        let p = base
            .with_fft_directions(FourierDirections::new(dirs))
            .unwrap();
        let reduced = [s[0], s[1] / 2 + 1, s[2]];
        let forward = |e| {
            let full = direct(values(s, e, false, true), s, tr, dirs, false);
            (0..reduced.iter().product())
                .map(|i| full[index(s, coord(reduced, i))])
                .collect::<Vec<_>>()
        };
        let raw = |e| {
            let mut v = values(reduced, e, true, false);
            for (i, x) in v.iter_mut().enumerate() {
                let k = coord(reduced, i)[1];
                if k == 0 || (s[1] % 2 == 0 && k == s[1] / 2) {
                    x.im = 0.;
                }
            }
            direct(
                v,
                reduced,
                [AxisTransform::Fft, AxisTransform::None, AxisTransform::None],
                dirs,
                false,
            )
        };
        macro_rules! fill_raw {
            ($array:expr) => {{
                #[allow(unused_mut)]
                let mut a = $array;
                for e in 0..2 {
                    for (i, x) in raw(e).into_iter().enumerate() {
                        if let Some(slot) = a.get_global_mut(&[e], coord(reduced, i)) {
                            *slot = convert(x);
                        }
                    }
                }
            }};
        }
        let backward = |e| real_inverse(raw(e), s, dirs);
        let inverse = |e| {
            backward(e)
                .into_iter()
                .map(|x| x / scale(s, tr))
                .collect::<Vec<_>>()
        };
        let mut input = p.allocate_input().unwrap();
        for e in 0..2 {
            for (i, x) in values(s, e, false, true).into_iter().enumerate() {
                if let Some(slot) = input.get_global_mut(&[e], coord(s, i)) {
                    *slot = R::cast(x.re);
                }
            }
        }
        let before = input.as_slice().to_vec();
        let mut output = p.allocate_output().unwrap();
        let mut ws = p.allocate_workspace().unwrap();
        p.forward(&input, &mut output, &mut ws).unwrap();
        check_complex!(&output, reduced, forward);
        assert_eq!(input.as_slice(), before);
        fill_raw!(&mut output);
        let spectrum = output.as_slice().to_vec();
        macro_rules! check_real {
            ($array:expr,$want:expr) => {{
                let a = $array;
                for e in 0..2 {
                    for (i, x) in $want(e).into_iter().enumerate() {
                        if let Some(slot) = a.get_global(&[e], coord(s, i)) {
                            near(Complex::new(*slot, R::cast(0.)), x);
                        }
                    }
                }
            }};
        }
        p.inverse(&output, &mut input, &mut ws).unwrap();
        check_real!(&input, inverse);
        p.backward(&output, &mut input, &mut ws).unwrap();
        check_real!(&input, backward);
        assert_eq!(output.as_slice(), spectrum);
        let mut ip = p.allocate_in_place().unwrap();
        let mut iw = p.allocate_in_place_workspace().unwrap();
        ip.real_view_mut()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(&before);
        p.forward_in_place(&mut ip, &mut iw).unwrap();
        check_complex!(ip.complex_view().unwrap(), reduced, forward);
        fill_raw!(ip.complex_view_mut().unwrap());
        p.inverse_in_place(&mut ip, &mut iw).unwrap();
        check_real!(ip.real_view().unwrap(), inverse);
        p.forward_in_place(&mut ip, &mut iw).unwrap();
        fill_raw!(ip.complex_view_mut().unwrap());
        p.backward_in_place(&mut ip, &mut iw).unwrap();
        check_real!(ip.real_view().unwrap(), backward);
    }
    fn negatives<R: Real>(top: &Arc<MpiTopology<1>>)
    where
        Complex<R>: mpi::datatype::Equivalence,
    {
        let shape = [2, 3, 4];
        let p =
            C2cPlan::<R, 3, 1>::from_shape(Arc::clone(top), shape, ExtraShape::scalar()).unwrap();
        let dirs = FourierDirections::new([F, B, F]);
        let q = p.with_fft_directions(dirs).unwrap();
        macro_rules! identities {
            ($p:expr,$q:expr,$ws:ident) => {{
                let p = &$p;
                let q = &$q;
                let a = p.allocate_input().unwrap();
                let b = q.allocate_input().unwrap();
                let mut out = q.allocate_output().unwrap();
                let mut old = p.$ws().unwrap();
                let mut new = q.$ws().unwrap();
                assert!(q.forward(&a, &mut out, &mut new).is_err());
                assert!(q.forward(&b, &mut out, &mut old).is_err());
                q.forward(&b, &mut out, &mut new).unwrap();
                let mut ip = p.allocate_in_place().unwrap();
                let mut iw = q.allocate_in_place_workspace().unwrap();
                assert!(q.forward_in_place(&mut ip, &mut iw).is_err());
                let mut ip = q.allocate_in_place().unwrap();
                let mut iw = p.allocate_in_place_workspace().unwrap();
                assert!(q.forward_in_place(&mut ip, &mut iw).is_err());
            }};
        }
        identities!(p, q, allocate_out_of_place_workspace);
        let partial = C2cPlan::<R, 3, 1>::from_shape_with_selection(
            Arc::clone(top),
            shape,
            ExtraShape::scalar(),
            AxisSelection::from_indices([0]).unwrap(),
        )
        .unwrap();
        assert!(partial.with_fft_directions(dirs).is_err());

        let mixed = MixedC2cPlan::<R, 3, 1>::from_shape(
            Arc::clone(top),
            shape,
            ExtraShape::scalar(),
            [AxisTransform::None, AxisTransform::Fft, AxisTransform::None],
        )
        .unwrap();
        let mq = mixed.with_fft_directions(dirs).unwrap();
        identities!(mixed, mq, allocate_workspace);
        assert!(
            mixed
                .with_fft_directions(FourierDirections::new([B, F, F]))
                .is_err()
        );
        let real = MixedR2cPlan::<R, 3, 1>::from_shape(
            Arc::clone(top),
            shape,
            ExtraShape::scalar(),
            [AxisTransform::Fft, AxisTransform::Rfft, AxisTransform::None],
        )
        .unwrap();
        let rq = real
            .with_fft_directions(FourierDirections::new([B, F, F]))
            .unwrap();
        identities!(real, rq, allocate_workspace);
        assert!(real.with_fft_directions(dirs).is_err());
        if top.communicator().size() > 1 {
            let rank = top.communicator().rank();
            let d = if rank == 0 {
                dirs
            } else {
                FourierDirections::default()
            };
            assert!(p.with_fft_directions(d).is_err());
            assert!(mixed.with_fft_directions(d).is_err());
            let d = if rank == 0 {
                FourierDirections::new([B, F, F])
            } else {
                FourierDirections::default()
            };
            assert!(real.with_fft_directions(d).is_err());
            let result = if rank == 0 {
                C2cPlan::<R, 3, 1>::from_shape_with_fft_directions(
                    Arc::clone(top),
                    shape,
                    ExtraShape::scalar(),
                    dirs,
                )
            } else {
                C2cPlan::<R, 3, 1>::from_shape(Arc::clone(top), shape, ExtraShape::scalar())
            };
            assert!(result.is_err());
            let transforms = mixed.transforms();
            let result = if rank == 0 {
                MixedC2cPlan::<R, 3, 1>::from_shape_with_fft_directions(
                    Arc::clone(top),
                    shape,
                    ExtraShape::scalar(),
                    transforms,
                    dirs,
                )
            } else {
                MixedC2cPlan::<R, 3, 1>::from_shape(
                    Arc::clone(top),
                    shape,
                    ExtraShape::scalar(),
                    transforms,
                )
            };
            assert!(result.is_err());
            let transforms = real.transforms();
            let result = if rank == 0 {
                MixedR2cPlan::<R, 3, 1>::from_shape_with_fft_directions(
                    Arc::clone(top),
                    shape,
                    ExtraShape::scalar(),
                    transforms,
                    FourierDirections::new([B, F, F]),
                )
            } else {
                MixedR2cPlan::<R, 3, 1>::from_shape(
                    Arc::clone(top),
                    shape,
                    ExtraShape::scalar(),
                    transforms,
                )
            };
            assert!(result.is_err());
        }
        let empty =
            C2cPlan::<R, 3, 1>::from_shape(Arc::clone(top), shape, ExtraShape::new([0]).unwrap())
                .unwrap()
                .with_fft_directions(dirs)
                .unwrap();
        empty
            .forward(
                &empty.allocate_input().unwrap(),
                &mut empty.allocate_output().unwrap(),
                &mut empty.allocate_out_of_place_workspace().unwrap(),
            )
            .unwrap();
    }
    pub fn main() {
        let u = mpi::initialize().unwrap();
        let world = u.world();
        let n = world.size() as usize;
        let top = MpiTopology::new(&world, [n]).unwrap();
        run::<f32>(&top);
        run::<f64>(&top);
        if world.rank() == 0 {
            println!("mpi_directions: PASSED ({n} ranks)");
        }
    }
}
#[cfg(feature = "distributed")]
fn main() {
    checks::main();
}
#[cfg(not(feature = "distributed"))]
fn main() {
    eprintln!("requires --features distributed");
}
