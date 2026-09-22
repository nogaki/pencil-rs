//! One MPI initialization; independent small direct-sum oracles.
#![cfg(feature = "distributed")]
use mpi::traits::*;
use pencil_array::{ExtraShape, MpiTopology, PencilArrayView, PencilArrayViewMut};
use pencil_fft::*;
use std::{f64::consts::TAU, sync::Arc};
mod collections_preflight;
const SHAPE: [usize; 2] = [4, 3];

trait Value: Copy + std::fmt::Debug + PartialEq {
    fn from_z(z: Complex<f64>) -> Self;
    fn z(self) -> Complex<f64>;
}
macro_rules! value {
    ($r:ty) => {
        impl Value for $r {
            fn from_z(z: Complex<f64>) -> Self {
                z.re as Self
            }
            fn z(self) -> Complex<f64> {
                Complex::new(self as f64, 0.0)
            }
        }
        impl Value for Complex<$r> {
            fn from_z(z: Complex<f64>) -> Self {
                Self::new(z.re as $r, z.im as $r)
            }
            fn z(self) -> Complex<f64> {
                Complex::new(self.re as f64, self.im as f64)
            }
        }
    };
}
value!(f32);
value!(f64);
#[derive(Clone, Copy)]
enum Op {
    Fft(f64),
    Dct,
    Hartley,
}
fn weight(op: Op, k: usize, x: usize, n: usize) -> Complex<f64> {
    match op {
        Op::Fft(sign) => Complex::from_polar(1.0, sign * TAU * (k * x) as f64 / n as f64),
        Op::Dct => Complex::new(
            2.0 * (TAU * (x as f64 + 0.5) * k as f64 / (2 * n) as f64).cos(),
            0.0,
        ),
        Op::Hartley => {
            let q = TAU * (x * k) as f64 / n as f64;
            Complex::new(q.cos() + q.sin(), 0.0)
        }
    }
}
fn sample(m: usize, e: usize, g: [usize; 2], complex: bool) -> Complex<f64> {
    Complex::new(
        (1 + m + 3 * e + g[0] + 2 * g[1]) as f64 / 7.0,
        if complex {
            (m as f64 + g[0] as f64 - g[1] as f64) / 5.0
        } else {
            0.0
        },
    )
}
fn fill<T: Value>(a: PencilArrayViewMut<'_, T, 2, 1>, m: usize, complex: bool, dc: bool) {
    fill_with(a, |e, g| {
        if dc {
            Complex::new(if g == [0, 0] { (1 + m + e) as f64 } else { 0.0 }, 0.0)
        } else {
            sample(m, e, g, complex)
        }
    });
}
fn fill_with<T: Value>(
    mut a: PencilArrayViewMut<'_, T, 2, 1>,
    value: impl Fn(usize, [usize; 2]) -> Complex<f64>,
) {
    let shape = a.pencil().local_shape_logical();
    let start = a.pencil().local_ranges().clone().map(|r| r.start);
    for e in 0..2 {
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                let g = [start[0] + i, start[1] + j];
                let z = value(e, g);
                *a.get_local_mut(&[e], [i, j]).unwrap() = T::from_z(z);
            }
        }
    }
}
fn check<T: Value>(
    a: PencilArrayView<'_, T, 2, 1>,
    expected: impl Fn(usize, [usize; 2]) -> Complex<f64>,
) {
    let shape = a.pencil().local_shape_logical();
    let start = a.pencil().local_ranges().clone().map(|r| r.start);
    for e in 0..2 {
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                let z = a.get_local(&[e], [i, j]).unwrap().z();
                let want = expected(e, [start[0] + i, start[1] + j]);
                assert!(
                    (z - want).norm() < 3e-4 * (1.0 + want.norm()),
                    "{z:?} != {want:?}"
                );
            }
        }
    }
}
fn direct(m: usize, e: usize, k: [usize; 2], ops: [Op; 2], complex: bool) -> Complex<f64> {
    let mut z = Complex::new(0.0, 0.0);
    for x in 0..SHAPE[0] {
        for y in 0..SHAPE[1] {
            z += sample(m, e, [x, y], complex)
                * weight(ops[0], k[0], x, SHAPE[0])
                * weight(ops[1], k[1], y, SHAPE[1]);
        }
    }
    z
}
// A non-DC spectrum defined independently of any forward execution. Reflection
// on Fourier axes makes the same full spectrum usable as Hermitian R2C input.
fn spectrum(m: usize, e: usize, mut g: [usize; 2], ops: [Op; 2]) -> Complex<f64> {
    for axis in 0..2 {
        if matches!(ops[axis], Op::Fft(_)) {
            g[axis] = g[axis].min(SHAPE[axis] - g[axis]);
        }
    }
    sample(m, e, g, false)
}
fn reverse(m: usize, e: usize, x: [usize; 2], ops: [Op; 2]) -> Complex<f64> {
    let reverse_weight = |op, k, x, n| match op {
        Op::Fft(sign) => weight(Op::Fft(-sign), k, x, n),
        Op::Hartley => weight(op, k, x, n),
        Op::Dct => Complex::new(
            if k == 0 {
                1.0
            } else {
                2.0 * (TAU * k as f64 * (x as f64 + 0.5) / (2 * n) as f64).cos()
            },
            0.0,
        ),
    };
    let mut z = Complex::new(0.0, 0.0);
    for i in 0..SHAPE[0] {
        for j in 0..SHAPE[1] {
            z += spectrum(m, e, [i, j], ops)
                * reverse_weight(ops[0], i, x[0], SHAPE[0])
                * reverse_weight(ops[1], j, x[1], SHAPE[1]);
        }
    }
    z
}
macro_rules! exercise {
    ($plan:expr, $ws:ident, $iv:ident, $ivm:ident, $ov:ident, $ovm:ident, $ops:expr, $complex:expr, $count:expr) => {{
        let p = $plan;
        let ops = $ops;
        let scale = ops
            .iter()
            .zip(SHAPE)
            .map(|(op, n)| if matches!(op, Op::Dct) { 2 * n } else { n })
            .product::<usize>() as f64;
        let mut source = (0..$count)
            .map(|m| {
                let mut a = p.allocate_input().unwrap();
                fill(a.view_mut(), m, $complex, false);
                a
            })
            .collect::<Vec<_>>();
        let saved = source
            .iter()
            .map(|a| a.as_slice().to_vec())
            .collect::<Vec<_>>();
        let mut out = (0..$count)
            .map(|_| p.allocate_output().unwrap())
            .collect::<Vec<_>>();
        let mut w = p.$ws().unwrap();
        let last = out.len() - 1;
        // Every family must reject a bad last member before touching member 0.
        let bad = pencil_array::PencilArray::from_fn(
            Arc::clone(out[last].pencil()),
            ExtraShape::new(vec![3]).unwrap(),
            || Value::from_z(Complex::new(77.0, 0.0)),
        )
        .unwrap();
        let good = std::mem::replace(&mut out[last], bad);
        let before = out
            .iter()
            .map(|a| a.as_slice().to_vec())
            .collect::<Vec<_>>();
        collections_preflight::indexed(p.forward_many(&source, &mut out, &mut w), last);
        assert_eq!(
            before,
            out.iter()
                .map(|a| a.as_slice().to_vec())
                .collect::<Vec<_>>()
        );
        out[last] = good;
        let other_plan = $plan;
        let mut foreign = other_plan.$ws().unwrap();
        let before = out
            .iter()
            .map(|a| a.as_slice().to_vec())
            .collect::<Vec<_>>();
        collections_preflight::indexed(p.forward_many(&source, &mut out, &mut foreign), 0);
        assert_eq!(
            before,
            out.iter()
                .map(|a| a.as_slice().to_vec())
                .collect::<Vec<_>>()
        );
        p.forward_many(&source, &mut out, &mut w).unwrap();
        for (m, a) in out.iter().enumerate() {
            check(a.view(), |e, g| direct(m, e, g, ops, $complex));
        }
        assert_eq!(
            saved,
            source
                .iter()
                .map(|a| a.as_slice().to_vec())
                .collect::<Vec<_>>()
        );
        p.inverse_many(&out, &mut source, &mut w).unwrap();
        for (m, a) in source.iter().enumerate() {
            check(a.view(), |e, g| sample(m, e, g, $complex));
        }
        p.backward_many(&out, &mut source, &mut w).unwrap();
        for (m, a) in source.iter().enumerate() {
            check(a.view(), |e, g| sample(m, e, g, $complex) * scale);
        }
        for (m, a) in out.iter_mut().enumerate() {
            fill_with(a.view_mut(), |e, g| spectrum(m, e, g, ops));
        }
        let spectra = out
            .iter()
            .map(|a| a.as_slice().to_vec())
            .collect::<Vec<_>>();
        p.inverse_many(&out, &mut source, &mut w).unwrap();
        for (m, a) in source.iter().enumerate() {
            check(a.view(), |e, g| reverse(m, e, g, ops) / scale);
        }
        p.backward_many(&out, &mut source, &mut w).unwrap();
        for (m, a) in source.iter().enumerate() {
            check(a.view(), |e, g| reverse(m, e, g, ops));
        }
        assert_eq!(
            spectra,
            out.iter()
                .map(|a| a.as_slice().to_vec())
                .collect::<Vec<_>>()
        );
        let mut ip = (0..$count)
            .map(|m| {
                let mut a = p.allocate_in_place().unwrap();
                fill(a.$ivm().unwrap(), m, $complex, false);
                a
            })
            .collect::<Vec<_>>();
        let mut iw = p.allocate_in_place_workspace().unwrap();
        p.forward_in_place(&mut ip[last], &mut iw).unwrap();
        let before = ip[..last]
            .iter()
            .map(|a| a.$iv().unwrap().as_slice().to_vec())
            .collect::<Vec<_>>();
        let last_before = ip[last].$ov().unwrap().as_slice().to_vec();
        collections_preflight::indexed(p.forward_many_in_place(&mut ip, &mut iw), last);
        assert_eq!(
            before,
            ip[..last]
                .iter()
                .map(|a| a.$iv().unwrap().as_slice().to_vec())
                .collect::<Vec<_>>()
        );
        assert_eq!(last_before, ip[last].$ov().unwrap().as_slice());
        p.inverse_in_place(&mut ip[last], &mut iw).unwrap();
        p.forward_many_in_place(&mut ip, &mut iw).unwrap();
        for (m, a) in ip.iter().enumerate() {
            check(a.$ov().unwrap(), |e, g| direct(m, e, g, ops, $complex));
        }
        p.inverse_many_in_place(&mut ip, &mut iw).unwrap();
        for (m, a) in ip.iter().enumerate() {
            check(a.$iv().unwrap(), |e, g| sample(m, e, g, $complex));
        }
        p.forward_many_in_place(&mut ip, &mut iw).unwrap();
        for (m, a) in ip.iter_mut().enumerate() {
            fill(a.$ovm().unwrap(), m, false, true);
        }
        p.backward_many_in_place(&mut ip, &mut iw).unwrap();
        for (m, a) in ip.iter().enumerate() {
            check(a.$iv().unwrap(), |e, _| {
                Complex::new((1 + m + e) as f64, 0.0)
            });
        }
    }};
}
macro_rules! families {
    ($r:ty, $topology:expr, $layout:expr) => {{
        let t = $topology;
        let l = $layout;
        let extra = || ExtraShape::new(vec![2]).unwrap();
        let dct = AxisTransform::R2r(AxisR2rKind::Fftw(R2rKind::DctII));
        exercise!(
            C2cPlan::<$r, 2, 1>::from_shape_with_layout(Arc::clone(t), SHAPE, extra(), l)
                .unwrap()
                .with_fft_directions(FourierDirections::new([
                    FourierDirection::Backward,
                    FourierDirection::Forward
                ]))
                .unwrap(),
            allocate_out_of_place_workspace,
            view,
            view_mut,
            view,
            view_mut,
            [Op::Fft(1.0), Op::Fft(-1.0)],
            true,
            3
        );
        exercise!(
            R2cPlan::<$r, 2, 1>::from_shape_with_layout(Arc::clone(t), SHAPE, extra(), l).unwrap(),
            allocate_workspace,
            real_view,
            real_view_mut,
            complex_view,
            complex_view_mut,
            [Op::Fft(-1.0); 2],
            false,
            2
        );
        exercise!(
            MixedC2cPlan::<$r, 2, 1>::from_shape_with_layout(
                Arc::clone(t),
                SHAPE,
                extra(),
                [AxisTransform::Fft, dct],
                l
            )
            .unwrap(),
            allocate_workspace,
            view,
            view_mut,
            view,
            view_mut,
            [Op::Fft(-1.0), Op::Dct],
            true,
            3
        );
        exercise!(
            MixedR2cPlan::<$r, 2, 1>::from_shape_with_layout(
                Arc::clone(t),
                SHAPE,
                extra(),
                [AxisTransform::Rfft, dct],
                l
            )
            .unwrap(),
            allocate_workspace,
            real_view,
            real_view_mut,
            complex_view,
            complex_view_mut,
            [Op::Fft(-1.0), Op::Dct],
            false,
            2
        );
        real_families!($r, t, l, false);
        real_families!(Complex<$r>, t, l, true);
    }};
}
macro_rules! real_families {
    ($v:ty,$t:expr,$l:expr,$complex:expr) => {{
        exercise!(
            R2rPlan::<$v, 2, 1>::from_shape_with_layout(
                Arc::clone($t),
                SHAPE,
                ExtraShape::new(vec![2]).unwrap(),
                [Some(R2rKind::DctII); 2],
                $l
            )
            .unwrap(),
            allocate_workspace,
            view,
            view_mut,
            view,
            view_mut,
            [Op::Dct; 2],
            $complex,
            2
        );
        exercise!(
            DhtPlan::<$v, 2, 1>::from_shape_with_layout(
                Arc::clone($t),
                SHAPE,
                ExtraShape::new(vec![2]).unwrap(),
                $l
            )
            .unwrap(),
            allocate_workspace,
            view,
            view_mut,
            view,
            view_mut,
            [Op::Hartley; 2],
            $complex,
            3
        );
    }};
}
#[test]
fn collections_one_mpi_binary() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let topology = MpiTopology::<1>::new(&world, [world.size() as usize]).unwrap();
    for transpose_method in [TransposeMethod::PointToPoint, TransposeMethod::AllToAllv] {
        for permute_dims in [false, true] {
            let layout = DistributedLayout {
                transpose_method,
                permute_dims,
            };
            families!(f32, &topology, layout);
            families!(f64, &topology, layout);
        }
    }
    collections_preflight::run(&topology);
    println!("COLLECTIONS_OK rank={} size={}", world.rank(), world.size());
}
