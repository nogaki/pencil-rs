use std::{
    cell::Cell,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Arc,
};

use mpi::traits::*;
use pencil_array::{
    ArrayError, AxisPermutation, ExtraShape, GeometryError, ManyPencilArray, MpiTopology,
    OverwriteError, Pencil, PencilConfig,
};

#[test]
fn many_pencil_array_enforces_registry_views_and_transactions() {
    let universe = mpi::initialize().expect("MPI initialization failed");
    let world = universe.world();
    let grid = match world.size() {
        1 => [1, 1],
        4 => [2, 2],
        other => panic!("run this test with mpiexec -n 1 or -n 4, got {other}"),
    };

    let topology = MpiTopology::<2>::new(&world, grid).unwrap();
    let base = Pencil::<3, 2>::new(Arc::clone(&topology), [3, 7, 5], [0, 1]).unwrap();
    let redistributed = base.with_decomposition([1, 2]).unwrap();
    let permuted = redistributed
        .with_permutation(AxisPermutation::new([2, 0, 1]).unwrap())
        .unwrap();
    let duplicate = base
        .reconfigured(PencilConfig::from(base.as_ref()))
        .unwrap();

    let empty: Vec<Arc<Pencil<3, 2>>> = Vec::new();
    assert_eq!(
        ManyPencilArray::<u32, 3, 2>::from_vec(empty, 0, ExtraShape::scalar(), Vec::new(),)
            .unwrap_err(),
        ArrayError::IncompatiblePencils,
    );
    assert_eq!(
        ManyPencilArray::<u32, 3, 2>::from_vec(
            vec![Arc::clone(&base)],
            1,
            ExtraShape::scalar(),
            vec![0; base.local_len()],
        )
        .unwrap_err(),
        ArrayError::InvalidActiveLayout {
            index: 1,
            layout_count: 1,
        },
    );

    let distinct_topology = MpiTopology::<2>::new(&world, grid).unwrap();
    let other_topology = Pencil::<3, 2>::new(distinct_topology, [3, 7, 5], [1, 2]).unwrap();
    assert_eq!(
        ManyPencilArray::<u32, 3, 2>::from_vec(
            vec![Arc::clone(&base), other_topology],
            0,
            ExtraShape::scalar(),
            Vec::new(),
        )
        .unwrap_err(),
        ArrayError::IncompatiblePencils,
    );

    let different_shape = base.with_global_shape([3, 7, 6]).unwrap();
    assert_eq!(
        ManyPencilArray::<u32, 3, 2>::from_vec(
            vec![Arc::clone(&base), Arc::clone(&different_shape)],
            0,
            ExtraShape::scalar(),
            Vec::new(),
        )
        .unwrap_err(),
        ArrayError::IncompatiblePencils,
    );
    assert_eq!(
        ManyPencilArray::<u32, 3, 2>::from_vec(
            vec![Arc::clone(&base), duplicate],
            0,
            ExtraShape::scalar(),
            Vec::new(),
        )
        .unwrap_err(),
        ArrayError::IncompatiblePencils,
    );

    let extra_shape = ExtraShape::new([2]).unwrap();
    let required = base.local_len().max(redistributed.local_len()) * 2;
    if world.size() == 4 {
        assert_ne!(
            base.local_len(),
            redistributed.local_len(),
            "fixture must expose per-layout local-length mistakes on every rank",
        );
    }
    assert_eq!(
        ManyPencilArray::<u32, 3, 2>::from_vec(
            vec![Arc::clone(&base), Arc::clone(&redistributed)],
            0,
            extra_shape.clone(),
            vec![0; required - 1],
        )
        .unwrap_err(),
        ArrayError::StorageLengthMismatch {
            required,
            actual: required - 1,
        },
    );
    assert_eq!(
        ManyPencilArray::<u32, 3, 2>::from_vec(
            vec![Arc::clone(&base), Arc::clone(&redistributed)],
            0,
            extra_shape.clone(),
            vec![0; required + 1],
        )
        .unwrap_err(),
        ArrayError::StorageLengthMismatch {
            required,
            actual: required + 1,
        },
    );

    let linear_topology = MpiTopology::<1>::new(&world, [grid.into_iter().product()]).unwrap();
    let huge = Pencil::<2, 1>::new(linear_topology, [usize::MAX, 1], [0]).unwrap();
    assert_eq!(
        ManyPencilArray::<u8, 2, 1>::from_vec(
            vec![Arc::clone(&huge)],
            0,
            ExtraShape::new([5]).unwrap(),
            Vec::new(),
        )
        .unwrap_err(),
        ArrayError::Geometry(GeometryError::SizeOverflow),
    );
    assert_eq!(
        ManyPencilArray::<u64, 2, 1>::from_elem(
            vec![Arc::clone(&huge)],
            0,
            ExtraShape::scalar(),
            0,
        )
        .unwrap_err(),
        ArrayError::AllocationFailed {
            required: huge.local_len(),
        },
    );

    let mut array = ManyPencilArray::from_vec(
        vec![
            Arc::clone(&base),
            Arc::clone(&redistributed),
            Arc::clone(&permuted),
        ],
        0,
        extra_shape.clone(),
        vec![0u32; required],
    )
    .unwrap();
    assert_eq!(array.pencils().len(), 3);
    assert_eq!(array.extra_shape(), &extra_shape);
    assert!(array.active_pencil().unwrap().same_layout(&base));
    assert_eq!(array.active_view().unwrap().len(), base.local_len() * 2);
    array.active_view_mut().unwrap().as_mut_slice().fill(3);
    assert!(
        array
            .active_view()
            .unwrap()
            .as_slice()
            .iter()
            .all(|&x| x == 3)
    );

    let equivalent_target = redistributed
        .reconfigured(PencilConfig::from(redistributed.as_ref()))
        .unwrap();
    array
        .overwrite_with(equivalent_target.as_ref(), |mut view| {
            assert_eq!(view.len(), redistributed.local_len() * 2);
            view.as_mut_slice().fill(7);
            Ok::<_, &'static str>(())
        })
        .unwrap();
    assert!(array.active_pencil().unwrap().same_layout(&redistributed));
    assert_eq!(
        array.active_view().unwrap().len(),
        redistributed.local_len() * 2
    );
    assert!(
        array
            .active_view()
            .unwrap()
            .as_slice()
            .iter()
            .all(|&x| x == 7)
    );

    let invalid_writer_called = Cell::new(false);
    let before_invalid = array.active_view().unwrap().as_slice().to_vec();
    let invalid = array.overwrite_with(different_shape.as_ref(), |mut view| {
        invalid_writer_called.set(true);
        view.as_mut_slice().fill(99);
        Ok::<_, &'static str>(())
    });
    assert!(matches!(
        invalid,
        Err(OverwriteError::Array(ArrayError::IncompatiblePencils))
    ));
    assert!(!invalid_writer_called.get());
    assert!(array.active_pencil().unwrap().same_layout(&redistributed));
    assert_eq!(array.active_view().unwrap().as_slice(), before_invalid);

    let writer_error = array.overwrite_with(base.as_ref(), |mut view| {
        view.as_mut_slice()[0] = 13;
        Err("stop")
    });
    assert!(matches!(writer_error, Err(OverwriteError::Writer("stop"))));
    assert_eq!(array.active_pencil().unwrap_err(), ArrayError::Poisoned);
    assert_eq!(array.active_view().unwrap_err(), ArrayError::Poisoned);
    assert_eq!(array.active_view_mut().unwrap_err(), ArrayError::Poisoned);
    array
        .overwrite_with(base.as_ref(), |mut view| {
            view.as_mut_slice().fill(17);
            Ok::<_, &'static str>(())
        })
        .unwrap();
    assert!(array.active_pencil().unwrap().same_layout(&base));
    assert!(
        array
            .active_view()
            .unwrap()
            .as_slice()
            .iter()
            .all(|&x| x == 17)
    );

    let panic_result = catch_unwind(AssertUnwindSafe(|| {
        let _ = array.overwrite_with(permuted.as_ref(), |mut view| -> Result<(), ()> {
            view.as_mut_slice()[0] = 19;
            panic!("injected overwrite panic");
        });
    }));
    assert!(panic_result.is_err());
    assert_eq!(array.active_view().unwrap_err(), ArrayError::Poisoned);
    array
        .overwrite_with(permuted.as_ref(), |mut view| {
            view.as_mut_slice().fill(23);
            Ok::<_, &'static str>(())
        })
        .unwrap();
    assert!(array.active_pencil().unwrap().same_layout(&permuted));
    assert_eq!(array.active_view().unwrap().len(), permuted.local_len() * 2);
    assert!(
        array
            .active_view()
            .unwrap()
            .as_slice()
            .iter()
            .all(|&x| x == 23)
    );

    let filled = ManyPencilArray::from_elem(
        vec![Arc::clone(&base), Arc::clone(&redistributed)],
        1,
        extra_shape,
        29u32,
    )
    .unwrap();
    assert!(filled.active_pencil().unwrap().same_layout(&redistributed));
    assert!(
        filled
            .active_view()
            .unwrap()
            .as_slice()
            .iter()
            .all(|&x| x == 29)
    );
}
