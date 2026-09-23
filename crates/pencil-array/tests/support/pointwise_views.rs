use std::sync::Arc;

use mpi::topology::Communicator;
use pencil_array::{
    AxisPermutation, ExtraShape, MpiTopology, MultiInputError, Pencil, PencilArray, PointwiseError,
    pointwise_many_in_place_views, pointwise_many_views, pointwise3_in_place_views,
    pointwise3_views,
};

#[derive(Debug, PartialEq)]
struct NonClone(u64);

fn array<T>(pencil: Arc<Pencil<2, 1>>, shape: ExtraShape, values: Vec<T>) -> PencilArray<T, 2, 1> {
    PencilArray::from_vec(pencil, shape, values).unwrap()
}

// Compatibility regression: the original public enum has exactly four variants.
// Do not add a wildcard or extend this match to accommodate new variants.
fn match_pointwise_error(error: PointwiseError) {
    match error {
        PointwiseError::IncompatiblePencils => {}
        PointwiseError::ExtraRankMismatch { .. } => {}
        PointwiseError::ExtraExtentMismatch { .. } => {}
        PointwiseError::Array(_) => {}
    }
}

/// Run the pointwise borrowed-view matrix from `multi_input`.
///
/// Invocation from the existing MPI test:
/// `support::pointwise_views::run(&world);`
pub fn run<C: Communicator>(world: &C) {
    let size = usize::try_from(world.size()).unwrap();
    let topology = MpiTopology::<1>::new(world, [size]).unwrap();
    let permutation = AxisPermutation::new([1, 0]).unwrap();
    let pencil =
        Pencil::<2, 1>::new_permuted(Arc::clone(&topology), [2, 3], [0], permutation.clone())
            .unwrap();
    let spatial = pencil.local_len();
    let shape_a = ExtraShape::new([1, 3]).unwrap();
    let shape_b = ExtraShape::new([2, 1]).unwrap();
    let shape_c = ExtraShape::new([1, 1]).unwrap();
    let output_shape = ExtraShape::new([2, 3]).unwrap();

    let a = array(
        Arc::clone(&pencil),
        shape_a,
        (0..3 * spatial)
            .map(|i| 10_i64 + (i / spatial) as i64 + (i % spatial) as i64)
            .collect(),
    );
    let b = array(
        Arc::clone(&pencil),
        shape_b,
        (0..2 * spatial)
            .map(|i| 100_i64 + (i / spatial) as i64 + (i % spatial) as i64)
            .collect(),
    );
    let c = array(
        Arc::clone(&pencil),
        shape_c,
        (0..spatial).map(|i| 1_000_i64 + i as i64).collect(),
    );

    // All four view APIs, including singleton broadcasting and physical order.
    let mut typed_out = array(
        Arc::clone(&pencil),
        output_shape.clone(),
        vec![0; 6 * spatial],
    );
    let mut seen = Vec::new();
    pointwise3_views(
        a.view(),
        b.view(),
        c.view(),
        typed_out.view_mut(),
        |x, y, z| {
            seen.push((*x, *y, *z));
            x + y + z
        },
    )
    .unwrap();
    for e in 0..6 {
        let expected = 10 + (e % 3) as i64 + 100 + (e / 3) as i64 + 1_000;
        for k in 0..spatial {
            assert_eq!(
                typed_out.as_slice()[e * spatial + k],
                expected + 3 * k as i64
            );
            assert_eq!(
                seen[e * spatial + k],
                (
                    10 + (e % 3) as i64 + k as i64,
                    100 + (e / 3) as i64 + k as i64,
                    1000 + k as i64
                )
            );
        }
    }

    let mut many_out = array(
        Arc::clone(&pencil),
        output_shape.clone(),
        vec![0; 6 * spatial],
    );
    pointwise_many_views(&[a.view(), b.view(), c.view()], many_out.view_mut(), |xs| {
        xs.iter().map(|x| **x).sum::<i64>()
    })
    .unwrap();
    assert_eq!(many_out.as_slice(), typed_out.as_slice());

    let mut typed_ip = array(
        Arc::clone(&pencil),
        output_shape.clone(),
        vec![7; 6 * spatial],
    );
    pointwise3_in_place_views(typed_ip.view_mut(), b.view(), c.view(), |x, y, z| x + y + z)
        .unwrap();
    for e in 0..6 {
        for k in 0..spatial {
            assert_eq!(
                typed_ip.as_slice()[e * spatial + k],
                1107 + (e / 3) as i64 + 2 * k as i64
            );
        }
    }

    // The in-place many API's output shape is the left input's shape, so use
    // the already-expanded output as the destination for this direct view call.
    let mut many_ip = array(
        Arc::clone(&pencil),
        output_shape.clone(),
        vec![7; 6 * spatial],
    );
    pointwise_many_in_place_views(many_ip.view_mut(), &[b.view(), c.view()], |x, xs| {
        *x + xs.iter().map(|v| **v).sum::<i64>()
    })
    .unwrap();
    for e in 0..6 {
        for k in 0..spatial {
            assert_eq!(
                many_ip.as_slice()[e * spatial + k],
                7 + 100 + (e / 3) as i64 + k as i64 + 1_000 + k as i64
            );
        }
    }

    // Non-Clone heterogeneous in-place typed-three input.
    let mut na =
        PencilArray::from_fn(Arc::clone(&pencil), ExtraShape::scalar(), || NonClone(2)).unwrap();
    let nb = PencilArray::from_fn(Arc::clone(&pencil), ExtraShape::scalar(), || 3_u64).unwrap();
    let nc = PencilArray::from_fn(Arc::clone(&pencil), ExtraShape::scalar(), || 4_i32).unwrap();
    pointwise3_in_place_views(na.view_mut(), nb.view(), nc.view(), |x, y, z| {
        NonClone(x.0 + *y + *z as u64)
    })
    .unwrap();
    assert!(na.as_slice().iter().all(|x| x.0 == 9));

    // Zero extras are valid and must not invoke callbacks.
    let zero = array(
        Arc::clone(&pencil),
        ExtraShape::new([0, 2]).unwrap(),
        Vec::<i32>::new(),
    );
    let mut zero_out = array(
        Arc::clone(&pencil),
        ExtraShape::new([0, 2]).unwrap(),
        Vec::new(),
    );
    let calls = std::cell::Cell::new(0);
    pointwise_many_views(&[zero.view()], zero_out.view_mut(), |_| {
        calls.set(calls.get() + 1);
        0
    })
    .unwrap();
    pointwise3_views(
        zero.view(),
        zero.view(),
        zero.view(),
        zero_out.view_mut(),
        |_, _, _| {
            calls.set(calls.get() + 1);
            0
        },
    )
    .unwrap();
    pointwise3_in_place_views(zero_out.view_mut(), zero.view(), zero.view(), |_, _, _| {
        calls.set(calls.get() + 1);
        0
    })
    .unwrap();
    pointwise_many_in_place_views(zero_out.view_mut(), &[zero.view()], |_, _| {
        calls.set(calls.get() + 1);
        0
    })
    .unwrap();
    assert_eq!(calls.get(), 0);

    // A rank-empty local geometry still validates and runs without callbacks.
    let empty_pencil =
        Pencil::<2, 1>::new_permuted(Arc::clone(&topology), [1, 3], [0], permutation).unwrap();
    let empty =
        PencilArray::from_elem(Arc::clone(&empty_pencil), ExtraShape::scalar(), 1_i32).unwrap();
    let mut empty_out =
        PencilArray::from_elem(Arc::clone(&empty_pencil), ExtraShape::scalar(), 0_i32).unwrap();
    pointwise3_views(
        empty.view(),
        empty.view(),
        empty.view(),
        empty_out.view_mut(),
        |x, y, z| {
            calls.set(calls.get() + 1);
            x + y + z
        },
    )
    .unwrap();
    pointwise_many_views(&[empty.view()], empty_out.view_mut(), |xs| {
        calls.set(calls.get() + 1);
        *xs[0]
    })
    .unwrap();
    pointwise3_in_place_views(
        empty_out.view_mut(),
        empty.view(),
        empty.view(),
        |x, y, z| {
            calls.set(calls.get() + 1);
            x + y + z
        },
    )
    .unwrap();
    pointwise_many_in_place_views(empty_out.view_mut(), &[empty.view()], |x, xs| {
        calls.set(calls.get() + 1);
        x + xs[0]
    })
    .unwrap();
    assert_eq!(calls.get(), 4 * empty_pencil.local_len());
    if empty_pencil.local_len() == 0 {
        assert!(empty_out.as_slice().is_empty());
    }

    assert!(matches!(
        pointwise_many_views::<i32, i32, _, 2, 1>(&[], empty_out.view_mut(), |_| 0),
        Err(MultiInputError::EmptyInputs)
    ));

    assert!(matches!(
        pointwise_many_in_place_views::<i32, _, 2, 1>(empty_out.view_mut(), &[], |_, _| 0),
        Err(MultiInputError::EmptyInputs)
    ));

    // Last-input validation is atomic for typed-three output and in-place APIs.
    let bad = array(
        Arc::clone(&pencil),
        ExtraShape::new([4, 3]).unwrap(),
        vec![9; 12 * spatial],
    );
    let mut sentinel = array(
        Arc::clone(&pencil),
        output_shape.clone(),
        vec![77; 6 * spatial],
    );
    let output_calls = std::cell::Cell::new(0);
    let error = pointwise3_views(
        a.view(),
        b.view(),
        bad.view(),
        sentinel.view_mut(),
        |x, y, z| {
            output_calls.set(output_calls.get() + 1);
            x + y + z
        },
    )
    .unwrap_err();
    if let MultiInputError::Input { source, .. } = error.clone() {
        match_pointwise_error(source);
    }
    assert!(matches!(error, MultiInputError::Input { index: 2, .. }));
    assert_eq!(output_calls.get(), 0);
    assert!(sentinel.as_slice().iter().all(|x| *x == 77));

    let mut inplace_sentinel = array(Arc::clone(&pencil), output_shape, vec![88; 6 * spatial]);
    let inplace_calls = std::cell::Cell::new(0);
    let error = pointwise3_in_place_views(
        inplace_sentinel.view_mut(),
        b.view(),
        bad.view(),
        |x, y, z| {
            inplace_calls.set(inplace_calls.get() + 1);
            *x + *y + *z
        },
    )
    .unwrap_err();
    if let MultiInputError::Input { source, .. } = error.clone() {
        match_pointwise_error(source);
    }
    assert!(matches!(error, MultiInputError::Input { index: 2, .. }));
    assert_eq!(inplace_calls.get(), 0);
    assert!(inplace_sentinel.as_slice().iter().all(|x| *x == 88));
    let error = pointwise_many_in_place_views(
        inplace_sentinel.view_mut(),
        &[b.view(), bad.view()],
        |x, _| {
            inplace_calls.set(inplace_calls.get() + 1);
            *x
        },
    )
    .unwrap_err();
    assert!(matches!(error, MultiInputError::Input { index: 1, .. }));
    assert_eq!(inplace_calls.get(), 0);
    assert!(inplace_sentinel.as_slice().iter().all(|x| *x == 88));
}
