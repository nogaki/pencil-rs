use mpi::topology::Communicator;
use num_complex::Complex64;
use pencil_array::*;
use std::{cell::Cell, sync::Arc};

fn array<T: Clone>(p: &Arc<Pencil<2, 1>>, dims: &[usize], value: T) -> PencilArray<T, 2, 1> {
    PencilArray::from_elem(p.clone(), ExtraShape::new(dims).unwrap(), value).unwrap()
}

fn pointwise(p: &Arc<Pencil<2, 1>>) {
    let mut a = array(p, &[1, 3], 0i64);
    let mut b = array(p, &[2, 1], 0i64);
    for (i, v) in a.as_mut_slice().iter_mut().enumerate() {
        *v = i as i64;
    }
    for (i, v) in b.as_mut_slice().iter_mut().enumerate() {
        *v = 100 + i as i64;
    }
    let mut out = array(p, &[2, 3], -1);
    let mut calls = 0;
    let scalar = 7;
    let mut seen = Vec::with_capacity(out.len());
    pointwise2_views(a.view(), b.view(), out.view_mut(), |x, y| {
        calls += 1;
        seen.push((*x, *y));
        x + y + scalar
    })
    .unwrap();
    assert_eq!(calls, out.len());
    let n = p.local_len();
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..n {
                assert_eq!(
                    seen[(i * 3 + j) * n + k],
                    (a.as_slice()[j * n + k], b.as_slice()[i * n + k])
                );
                assert_eq!(
                    out.as_slice()[(i * 3 + j) * n + k],
                    a.as_slice()[j * n + k] + b.as_slice()[i * n + k] + scalar
                );
            }
        }
    }
    let before = out.as_slice().to_vec();
    let bad = array(p, &[2, 2], 1);
    assert!(pointwise2(&a, &bad, &mut out, |_, _| { panic!("invalid callback") }).is_err());
    assert_eq!(out.as_slice(), before);
    let other = Pencil::new(p.topology().clone(), *p.global_shape(), [0]).unwrap();
    let bad = array(&other, &[2, 1], 1);
    assert!(pointwise2(&a, &bad, &mut out, |_, _| panic!("layout callback")).is_err());
    assert_eq!(out.as_slice(), before);
    struct NonClone(i64);
    let mut nc = PencilArray::from_vec(
        p.clone(),
        ExtraShape::new([2, 3]).unwrap(),
        (0..out.len()).map(|_| NonClone(1)).collect(),
    )
    .unwrap();
    pointwise2_in_place(&mut nc, &b, |x, y| NonClone(x.0 + y)).unwrap();
    assert!(nc.as_slice().iter().all(|v| v.0 >= 101));
    pointwise2_in_place_views(nc.view_mut(), b.view(), |x, y| NonClone(x.0 - y)).unwrap();
    assert!(nc.as_slice().iter().all(|v| v.0 == 1));
    let bad_rank = array(p, &[1], 1);
    assert!(
        pointwise2(&a, &bad_rank, &mut out, |_, _| panic!(
            "extra rank callback"
        ))
        .is_err()
    );
    assert_eq!(out.as_slice(), before);
    let topology = MpiTopology::<1>::new(
        p.topology().communicator(),
        [p.topology().communicator().size() as usize],
    )
    .unwrap();
    let different = Pencil::new_permuted(
        topology,
        *p.global_shape(),
        [0],
        AxisPermutation::new([1, 0]).unwrap(),
    )
    .unwrap();
    let bad_topology = array(&different, &[2, 1], 1);
    assert!(
        pointwise2(&a, &bad_topology, &mut out, |_, _| panic!(
            "topology callback"
        ))
        .is_err()
    );
    assert_eq!(out.as_slice(), before);
    let empty = array(p, &[0], 1);
    let singleton = array(p, &[1], 1);
    let mut empty_out = array(p, &[0], 9);
    pointwise2(&empty, &singleton, &mut empty_out, |_, _| {
        panic!("empty callback")
    })
    .unwrap();
}

fn reductions(p: &Arc<Pencil<2, 1>>, size: usize, rank: usize) {
    let a = array(p, &[1, 3], 2i64);
    let b = array(p, &[2, 1], 3i64);
    assert_eq!(
        zip_sum_by(&a.view(), &b.view(), |x, y| x * y).unwrap(),
        (p.global_len() * 36) as i64
    );
    assert_eq!(
        map_reduce2(&a.view(), &b.view(), 0i64, |x, y| x + y, |x, y| x + y).unwrap(),
        (p.global_len() * 30) as i64
    );
    // Addition is associative; observe the final rank-ordered fold separately.
    let one = array(p, &[], rank as i64 + 1);
    let local_calls = one.len();
    let mut operands = Vec::with_capacity(local_calls + size);
    map_reduce2(
        &one.view(),
        &one.view(),
        0i64,
        |x, _| *x,
        |x, y| {
            operands.push(y);
            x + y
        },
    )
    .unwrap();
    let topology = p.topology();
    let comm = topology.communicator();
    let local = (rank as i64 + 1) * local_calls as i64;
    let mut expected = vec![0i64; size];
    mpi::collective::CommunicatorCollectives::all_gather_into(comm, &local, &mut expected[..]);
    assert_eq!(&operands[local_calls..], expected);
    assert!(zip_sum_by(&a.view(), &b.view(), |_, _| i32::MAX).is_err());
    for scale in [1e300, 1e-300] {
        let result = zip_norm_by(&a.view(), &b.view(), |_, _| scale).unwrap();
        let expected = scale * (p.global_len() as f64 * 6.0).sqrt();
        assert!((result / expected - 1.0).abs() < 1e-12);
    }
    let norm = zip_norm_by(&a.view(), &b.view(), |_, _| Complex64::new(3.0, 4.0)).unwrap();
    assert!((norm / (5.0 * (p.global_len() as f64 * 6.0).sqrt()) - 1.0).abs() < 1e-12);
    assert!(min_by(&a.view(), |_| f64::NAN).unwrap().unwrap().is_nan());
    assert!(max_by(&a.view(), |_| f64::NAN).unwrap().unwrap().is_nan());
    assert_eq!(
        min_by(&a.view(), |_| f64::NEG_INFINITY).unwrap(),
        Some(f64::NEG_INFINITY)
    );
    assert_eq!(
        max_by(&a.view(), |_| f64::INFINITY).unwrap(),
        Some(f64::INFINITY)
    );
    let empty = array(p, &[0], 1i64);
    let single = array(p, &[1], 1i64);
    assert_eq!(min_by(&empty.view(), |_| 1).unwrap(), None);
    assert_eq!(max_by(&empty.view(), |_| 1).unwrap(), None);
    assert_eq!(
        map_reduce2(
            &empty.view(),
            &single.view(),
            0i64,
            |_, _| panic!("empty"),
            |x, y| x + y
        )
        .unwrap(),
        0
    );
    let duplicate_topology = MpiTopology::<1>::new(p.topology().communicator(), [size]).unwrap();
    let duplicate_pencil = Pencil::new_permuted(
        duplicate_topology,
        *p.global_shape(),
        [0],
        AxisPermutation::new([1, 0]).unwrap(),
    )
    .unwrap();
    let wrong_topology = array(&duplicate_pencil, &[2, 1], 3i64);
    let calls = Cell::new(0);
    let right = if rank == 0 { &wrong_topology } else { &b };
    assert!(matches!(
        map_reduce2(
            &a.view(),
            &right.view(),
            0i64,
            |x, y| {
                calls.set(calls.get() + 1);
                x + y
            },
            |x, y| x + y
        ),
        Err(CollectiveError::CollectivePreconditionFailed)
    ));
    assert_eq!(calls.get(), 0);
    if size > 1 {
        for neutral in [
            if rank == 0 { -0.0 } else { 0.0 },
            f64::from_bits(0x7ff8000000000000 + u64::from(rank == 0)),
        ] {
            let calls = Cell::new(0);
            let result = map_reduce2(
                &a.view(),
                &b.view(),
                neutral,
                |_, _| {
                    calls.set(calls.get() + 1);
                    0.0
                },
                |x, y| x + y,
            );
            assert!(matches!(
                result,
                Err(CollectiveError::CollectiveDescriptorMismatch)
            ));
            assert_eq!(calls.get(), 0);
        }
        let bad = array(p, if rank == 0 { &[2, 2] } else { &[2, 1] }, 1i64);
        let calls = Cell::new(0);
        assert!(
            zip_sum_by(&a.view(), &bad.view(), |x, y| {
                calls.set(calls.get() + 1);
                x + y
            })
            .is_err()
        );
        assert_eq!(calls.get(), 0);
        let result = if rank == 0 {
            global_sum(&a.view())
        } else {
            zip_sum_by(&a.view(), &b.view(), |x, y| x + y)
        };
        assert!(matches!(
            result,
            Err(CollectiveError::CollectiveDescriptorMismatch)
        ));
    }
    assert_eq!(global_sum(&a.view()).unwrap(), (p.global_len() * 6) as i64);
}

// ZST storage exercises real reservation/count failures without allocating huge buffers.
fn preparation_failures(topology: &Arc<MpiTopology<1>>) {
    let p = Pencil::<2, 1>::new(topology.clone(), [1, 1], [0]).unwrap();
    let make = |dims: &[usize]| {
        let extra = ExtraShape::new(dims).unwrap();
        let len = extra.element_count() * p.local_len();
        PencilArray::from_vec(p.clone(), extra, vec![(); len]).unwrap()
    };
    let huge = make(&[isize::MAX as usize]);
    let one = make(&[1]);
    let calls = Cell::new(0);
    let result = sum_by(&huge.view(), |_| {
        calls.set(calls.get() + 1);
        1i64
    });
    // Only the nonempty rank cannot reserve mapped storage; peers must agree.
    if p.local_len() != 0 {
        assert!(matches!(
            result,
            Err(CollectiveError::AllocationFailed { .. })
        ));
    } else {
        assert!(matches!(
            result,
            Err(CollectiveError::CollectivePreconditionFailed)
        ));
    }
    assert_eq!(calls.get(), 0);
    assert_eq!(sum_by(&one.view(), |_| 1i64).unwrap(), 1);
    assert!(
        zip_sum_by(&huge.view(), &one.view(), |_, _| -> f64 {
            panic!("allocation callback")
        })
        .is_err()
    );
    assert!(
        zip_norm_by(&huge.view(), &one.view(), |_, _| -> f64 {
            panic!("allocation callback")
        })
        .is_err()
    );
    let huge = make(&[usize::MAX, 1]);
    let two = make(&[1, 2]);
    assert!(
        map_reduce2(
            &huge.view(),
            &two.view(),
            0i64,
            |_, _| panic!("overflow callback"),
            |x, y| x + y
        )
        .is_err()
    );
    let small = make(&[1]);
    assert_eq!(
        zip_sum_by(&small.view(), &small.view(), |_, _| 1i64).unwrap(),
        1
    );
}

#[test]
fn array_ops() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size() as usize;
    let topology = MpiTopology::<1>::new(&world, [size]).unwrap();
    preparation_failures(&topology);
    for shape in [[size * 2 + 1, 3], [1, 1]] {
        let p = Pencil::new_permuted(
            topology.clone(),
            shape,
            [0],
            AxisPermutation::new([1, 0]).unwrap(),
        )
        .unwrap();
        pointwise(&p);
        reductions(&p, size, p.topology().communicator().rank() as usize);
    }
}
