#[path = "support/pointwise_views.rs"]
mod pointwise_views;
#[path = "support/typed_three.rs"]
mod typed_three;

use std::{fmt, sync::Arc};

use mpi::traits::*;
use num_complex::Complex64;
use pencil_array::{
    CollectiveError, ExtraShape, MpiTopology, Pencil, PencilArray, map_reduce_many, map_reduce3,
    max_many_by, min_many_by, norm_many_by, pointwise_many, pointwise_many_in_place, pointwise3,
    pointwise3_in_place, sum_many_by,
};

#[derive(Debug, PartialEq)]
struct NonClone(u64);
impl fmt::Display for NonClone {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

fn array<T: Clone>(p: Arc<Pencil<1, 1>>, extra: ExtraShape, value: T) -> PencilArray<T, 1, 1> {
    PencilArray::from_elem(p, extra, value).unwrap()
}

fn map_i64(xs: &[&i32]) -> i64 {
    xs.iter().map(|x| i64::from(**x)).sum()
}
fn reduce_i64(x: i64, y: i64) -> i64 {
    x + y
}
fn sum_f64(xs: &[&i32]) -> f64 {
    xs.iter().map(|x| f64::from(**x)).sum()
}
fn generic_callback<T>(xs: &[&T]) -> i64 {
    xs.len() as i64
}

#[test]
fn multi_input_current_api_and_edge_cases() {
    let universe = mpi::initialize().expect("MPI initialization failed");
    let world = universe.world();
    let size = usize::try_from(world.size()).unwrap();
    assert!(matches!(size, 1 | 4 | 6), "run with -n 1, -n 4, or -n 6");
    let topology = MpiTopology::<1>::new(&world, [size]).unwrap();
    let pencil = Pencil::<1, 1>::new(Arc::clone(&topology), [size], [0]).unwrap();

    pointwise_views::run(&world);
    typed_three::run_map_reduce3_mismatch_retry_suite(&world, &topology, &pencil);

    let a = array(Arc::clone(&pencil), ExtraShape::scalar(), 2_i32);
    let b = array(Arc::clone(&pencil), ExtraShape::scalar(), 0.5_f64);
    let c = array(
        Arc::clone(&pencil),
        ExtraShape::scalar(),
        Complex64::new(1.0, 2.0),
    );
    let three = map_reduce3(
        topology.communicator(),
        &a.view(),
        &b.view(),
        &c.view(),
        0.0_f64,
        |x, y, z| f64::from(*x) + *y + z.re + z.im,
        |x, y| x + y,
    )
    .unwrap();
    assert_eq!(three, size as f64 * 5.5);

    // Affine composition is associative but noncommutative. This independently
    // verifies ascending-rank folding, not merely a commutative sum.
    fn compose(x: Complex64, y: Complex64) -> Complex64 {
        Complex64::new(x.re * y.re, x.im * y.re + y.im)
    }
    let digit = array(
        Arc::clone(&pencil),
        ExtraShape::scalar(),
        f64::from(world.rank() + 1),
    );
    let affine = map_reduce_many(
        topology.communicator(),
        &[digit.view()],
        Complex64::new(1.0, 0.0),
        |xs| Complex64::new(2.0, *xs[0]),
        compose,
    )
    .unwrap();
    let expected_digits: f64 = (0..size)
        .map(|r| (r + 1) as f64 * 2.0_f64.powi((size - 1 - r) as i32))
        .sum();
    assert_eq!(
        affine,
        Complex64::new(2.0_f64.powi(size as i32), expected_digits)
    );

    let shape13 = ExtraShape::new([1, 3]).unwrap();
    let shape23 = ExtraShape::new([2, 3]).unwrap();
    let broadcast_a = array(Arc::clone(&pencil), shape13.clone(), 1_i32);
    let broadcast_b = array(Arc::clone(&pencil), ExtraShape::new([2, 1]).unwrap(), 2_i64);
    let broadcast_c = array(Arc::clone(&pencil), ExtraShape::new([1, 1]).unwrap(), 3_u8);
    assert_eq!(
        map_reduce3(
            topology.communicator(),
            &broadcast_a.view(),
            &broadcast_b.view(),
            &broadcast_c.view(),
            0_i64,
            |x, y, z| i64::from(*x) + y + i64::from(*z),
            reduce_i64
        )
        .unwrap(),
        36 * size as i64
    );
    let inputs4: Vec<_> = (1..=4)
        .map(|n| {
            array(
                Arc::clone(&pencil),
                if n == 4 {
                    shape23.clone()
                } else {
                    shape13.clone()
                },
                n,
            )
        })
        .collect();
    let views4: Vec<_> = inputs4.iter().map(PencilArray::view).collect();
    let sum4 = sum_many_by(topology.communicator(), &views4, map_i64).unwrap();
    assert_eq!(sum4, size as i64 * 6 * 10);

    let inputs6: Vec<_> = (1..=6)
        .map(|n| array(Arc::clone(&pencil), shape23.clone(), n))
        .collect();
    let views6: Vec<_> = inputs6.iter().map(PencilArray::view).collect();
    let folded =
        map_reduce_many(topology.communicator(), &views6, 0_i64, map_i64, reduce_i64).unwrap();
    assert_eq!(folded, size as i64 * 6 * 21);
    let order = std::cell::RefCell::new(Vec::new());
    map_reduce_many(
        topology.communicator(),
        &views4,
        0_i64,
        |xs| {
            order
                .borrow_mut()
                .push(xs.iter().map(|x| **x).collect::<Vec<_>>());
            0
        },
        reduce_i64,
    )
    .unwrap();
    assert_eq!(order.borrow().first().unwrap(), &vec![1, 2, 3, 4]);

    // Callback order follows extra-major, then physical spatial storage order.
    let topo2 = MpiTopology::<1>::new(&world, [size]).unwrap();
    let pencil2 = Pencil::<2, 1>::new(Arc::clone(&topo2), [size, 2], [0]).unwrap();
    let ordered_inputs: Vec<_> = (0..5_i32)
        .map(|offset| {
            PencilArray::from_vec(
                Arc::clone(&pencil2),
                ExtraShape::new([2, 3]).unwrap(),
                (0..12_i32).map(|x| offset * 100 + x).collect(),
            )
            .unwrap()
        })
        .collect();
    let ordered_views: Vec<_> = ordered_inputs.iter().map(PencilArray::view).collect();
    let seen = std::cell::RefCell::new(Vec::new());
    map_reduce_many(
        topo2.communicator(),
        &ordered_views,
        0_i64,
        |xs| {
            seen.borrow_mut()
                .push(xs.iter().map(|x| **x).collect::<Vec<_>>());
            i64::from(*xs[0])
        },
        reduce_i64,
    )
    .unwrap();
    let expected_seen: Vec<_> = (0..12_i32)
        .map(|x| {
            (0..5_i32)
                .map(|offset| offset * 100 + x)
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(*seen.borrow(), expected_seen);

    // Same callback function, but different neutral bits: reject before invoking it.
    let calls = std::cell::Cell::new(0);
    let mismatch = if size > 1 {
        let neutral = if world.rank() == 0 { 0_i64 } else { 1_i64 };
        map_reduce_many(
            topology.communicator(),
            &views4,
            neutral,
            |_| {
                calls.set(calls.get() + 1);
                1_i64
            },
            reduce_i64,
        )
        .is_err()
    } else {
        true
    };
    assert!(mismatch);
    if size > 1 {
        assert_eq!(calls.get(), 0);

        // Input-count descriptors reject ranks using the same callback, then recover.
        let count_mismatch = if world.rank() == 0 {
            map_reduce_many(topology.communicator(), &views4, 0_i64, map_i64, reduce_i64)
        } else {
            map_reduce_many(
                topology.communicator(),
                &views6[..5],
                0_i64,
                map_i64,
                reduce_i64,
            )
        };
        assert!(matches!(
            count_mismatch,
            Err(CollectiveError::CollectiveDescriptorMismatch)
        ));
        assert_eq!(
            map_reduce_many(
                topology.communicator(),
                &views6[..5],
                0_i64,
                map_i64,
                reduce_i64,
            )
            .unwrap(),
            size as i64 * 6 * 15
        );

        // A generic callback keeps the output/callback type fixed while input dtypes differ.
        let typed_i32 = if world.rank() == 0 {
            map_reduce_many(
                topology.communicator(),
                &views4,
                0_i64,
                generic_callback,
                reduce_i64,
            )
        } else {
            let typed = array(Arc::clone(&pencil), shape13.clone(), 1_i64);
            let typed_views: Vec<_> = (0..4).map(|_| typed.view()).collect();
            map_reduce_many(
                topology.communicator(),
                &typed_views,
                0_i64,
                generic_callback,
                reduce_i64,
            )
        };
        assert!(matches!(
            typed_i32,
            Err(CollectiveError::CollectiveDescriptorMismatch)
        ));
    }
    if size > 1 {
        let signed_zero = map_reduce_many(
            topology.communicator(),
            &views4,
            if world.rank() == 0 { 0.0_f64 } else { -0.0_f64 },
            sum_f64,
            |x, y| x + y,
        );
        assert!(matches!(
            signed_zero,
            Err(CollectiveError::CollectiveDescriptorMismatch)
        ));
        assert_eq!(
            map_reduce_many(
                topology.communicator(),
                &views4,
                0.0_f64,
                sum_f64,
                |x, y| x + y,
            )
            .unwrap(),
            size as f64 * 6.0 * 10.0
        );
    }
    let same_bits = map_reduce_many(
        topology.communicator(),
        &views4,
        -0_i64,
        map_i64,
        reduce_i64,
    )
    .unwrap();
    assert_eq!(same_bits, size as i64 * 6 * 10);
    assert_eq!(
        sum_many_by(topology.communicator(), &views4, map_i64).unwrap(),
        size as i64 * 6 * 10
    );

    // Operation mismatch is also recoverable; the following collective is valid.
    if size > 1 {
        let old_new = if world.rank() == 0 {
            sum_many_by(topology.communicator(), &views4, map_i64).map(|_| ())
        } else {
            norm_many_by::<_, f64, _, 1, 1>(topology.communicator(), &views4, |_| 1.0).map(|_| ())
        };
        assert!(matches!(
            old_new,
            Err(CollectiveError::CollectiveDescriptorMismatch)
        ));
        assert_eq!(
            sum_many_by(topology.communicator(), &views4, map_i64).unwrap(),
            size as i64 * 6 * 10
        );
    }

    let overflow = sum_many_by(topology.communicator(), &views4, |_| i64::MAX);
    assert!(matches!(overflow, Err(CollectiveError::IntegerOverflow)));
    let extremes = array(
        Arc::clone(&pencil),
        ExtraShape::scalar(),
        if world.rank() == 0 {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        },
    );
    let ev = vec![extremes.view()];
    assert_eq!(
        min_many_by(topology.communicator(), &ev, |x| *x[0]).unwrap(),
        Some(f64::NEG_INFINITY)
    );
    assert_eq!(
        max_many_by(topology.communicator(), &ev, |x| *x[0]).unwrap(),
        Some(if size == 1 {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        })
    );

    let huge = array(
        Arc::clone(&pencil),
        ExtraShape::scalar(),
        Complex64::new(6e200, 8e200),
    );
    let norm = norm_many_by(topology.communicator(), &[huge.view()], |xs| *xs[0]).unwrap();
    let expected_norm = 1e201 * (size as f64).sqrt();
    assert!(norm.is_finite() && (norm / expected_norm - 1.0).abs() < 1e-12);
    let nan = array(Arc::clone(&pencil), ExtraShape::scalar(), f64::NAN);
    assert!(
        min_many_by(topology.communicator(), &[nan.view()], |xs| *xs[0])
            .unwrap()
            .unwrap()
            .is_nan()
    );
    assert!(
        max_many_by(topology.communicator(), &[nan.view()], |xs| *xs[0])
            .unwrap()
            .unwrap()
            .is_nan()
    );
    assert!(
        norm_many_by(topology.communicator(), &[nan.view()], |xs| *xs[0])
            .unwrap()
            .is_nan()
    );
    let inf = array(Arc::clone(&pencil), ExtraShape::scalar(), f64::INFINITY);
    assert_eq!(
        norm_many_by(topology.communicator(), &[inf.view()], |xs| *xs[0]).unwrap(),
        f64::INFINITY
    );

    // Empty extra shape is an empty global reduction, including norm.
    let empty = array(Arc::clone(&pencil), ExtraShape::new([0]).unwrap(), 1.0_f64);
    let empty_view = vec![empty.view()];
    assert_eq!(
        norm_many_by(topology.communicator(), &empty_view, |x| *x[0]).unwrap(),
        0.0
    );
    assert_eq!(
        min_many_by(topology.communicator(), &empty_view, |x| *x[0]).unwrap(),
        None
    );

    // A global spatial length of one creates empty local partitions on ranks 1+.
    let one_pencil = Pencil::<1, 1>::new(Arc::clone(&topology), [1], [0]).unwrap();
    let singleton = array(
        Arc::clone(&one_pencil),
        ExtraShape::new([1, 2]).unwrap(),
        3_i32,
    );
    assert_eq!(
        map_reduce3(
            topology.communicator(),
            &singleton.view(),
            &singleton.view(),
            &singleton.view(),
            0_i64,
            |x, y, z| i64::from(x + y + z),
            reduce_i64
        )
        .unwrap(),
        18
    );
    let zero_extra = array(
        Arc::clone(&one_pencil),
        ExtraShape::new([0, 2]).unwrap(),
        4_i32,
    );
    let mut zero_output = array(
        Arc::clone(&one_pencil),
        ExtraShape::new([0, 2]).unwrap(),
        9_i32,
    );
    let zero_calls = std::cell::Cell::new(0);
    pointwise_many(&[&singleton, &zero_extra], &mut zero_output, |_| {
        zero_calls.set(zero_calls.get() + 1);
        0_i32
    })
    .unwrap();
    assert_eq!(zero_calls.get(), 0);
    assert!(zero_output.as_slice().is_empty());
    let mixed_views = vec![singleton.view(), zero_extra.view()];
    assert_eq!(
        sum_many_by(topology.communicator(), &mixed_views, |_| 1_i64).unwrap(),
        0
    );

    // Pointwise operations use the real APIs and do not require Clone.
    let na =
        PencilArray::from_fn(Arc::clone(&pencil), ExtraShape::scalar(), || NonClone(2)).unwrap();
    let nb =
        PencilArray::from_fn(Arc::clone(&pencil), ExtraShape::scalar(), || NonClone(3)).unwrap();
    let nc =
        PencilArray::from_fn(Arc::clone(&pencil), ExtraShape::scalar(), || NonClone(4)).unwrap();
    let mut no =
        PencilArray::from_fn(Arc::clone(&pencil), ExtraShape::scalar(), || NonClone(0)).unwrap();
    pointwise3(&na, &nb, &nc, &mut no, |x, y, z| NonClone(x.0 + y.0 + z.0)).unwrap();
    assert_eq!(no.as_slice()[0].0, 9);
    let mut inplace =
        PencilArray::from_fn(Arc::clone(&pencil), ExtraShape::scalar(), || NonClone(1)).unwrap();
    pointwise3_in_place(&mut inplace, &nb, &nc, |x, y, z| NonClone(x.0 + y.0 + z.0)).unwrap();
    assert_eq!(inplace.as_slice()[0].0, 8);
    let refs = [&na, &nb, &nc];
    pointwise_many(&refs, &mut no, |xs| NonClone(xs.iter().map(|x| x.0).sum())).unwrap();
    assert_eq!(no.as_slice()[0].0, 9);
    pointwise_many_in_place(&mut no, &refs, |x, xs| {
        NonClone(x.0 + xs.iter().map(|v| v.0).sum::<u64>())
    })
    .unwrap();
    assert_eq!(no.as_slice()[0].0, 18);

    // Last-input failures leave the sentinel untouched.
    let bad = array(Arc::clone(&pencil), ExtraShape::new([4, 3]).unwrap(), 7_i32);
    let mut sentinel = array(Arc::clone(&pencil), shape23.clone(), 99_i32);
    let err = pointwise_many(&[&inputs4[0], &bad], &mut sentinel, |xs| *xs[0]).unwrap_err();
    assert!(matches!(
        err,
        pencil_array::MultiInputError::Input { index: 1, .. }
    ));
    assert!(sentinel.as_slice().iter().all(|x| *x == 99));
    let other_topology = MpiTopology::<1>::new(&world, [size]).unwrap();
    let wrong_pencil = Pencil::<1, 1>::new(other_topology, [size], [0]).unwrap();
    let wrong = array(wrong_pencil, shape23.clone(), 1_i32);
    let wrong_calls = std::cell::Cell::new(0);
    let err = pointwise_many(&[&inputs4[0], &wrong], &mut sentinel, |xs| {
        wrong_calls.set(wrong_calls.get() + 1);
        *xs[0]
    })
    .unwrap_err();
    assert!(matches!(
        err,
        pencil_array::MultiInputError::Input { index: 1, .. }
    ));
    assert_eq!(wrong_calls.get(), 0);
    assert!(sentinel.as_slice().iter().all(|x| *x == 99));

    // Only the last rank supplies a bad last member; every rank must reject.
    let last = if world.rank() == world.size() - 1 {
        &wrong
    } else {
        &inputs4[3]
    };
    assert!(
        sum_many_by(
            topology.communicator(),
            &[inputs4[0].view(), last.view()],
            |xs| {
                wrong_calls.set(wrong_calls.get() + 1);
                i64::from(*xs[0])
            }
        )
        .is_err()
    );
    assert_eq!(wrong_calls.get(), 0);
    let last = if world.rank() == world.size() - 1 {
        &bad
    } else {
        &inputs4[3]
    };
    assert!(
        sum_many_by(
            topology.communicator(),
            &[inputs4[3].view(), last.view()],
            |xs| {
                wrong_calls.set(wrong_calls.get() + 1);
                i64::from(*xs[0])
            }
        )
        .is_err()
    );
    assert_eq!(wrong_calls.get(), 0);

    if size > 1 {
        for old_two in [false, true] {
            let result = if world.rank() == 0 {
                if old_two {
                    pencil_array::map_reduce2(
                        &a.view(),
                        &a.view(),
                        0_i64,
                        |x, y| i64::from(*x + *y),
                        reduce_i64,
                    )
                } else {
                    pencil_array::sum_by(&a.view(), |x| i64::from(*x))
                }
            } else {
                map_reduce_many(topology.communicator(), &views4, 0_i64, map_i64, reduce_i64)
            };
            assert!(matches!(
                result,
                Err(CollectiveError::CollectiveDescriptorMismatch)
            ));
            assert_eq!(
                sum_many_by(topology.communicator(), &views4, map_i64).unwrap(),
                size as i64 * 60
            );
        }
    }
    eprintln!(
        "EXECUTED multi-input matrix rank={} size={size}",
        world.rank()
    );
}
