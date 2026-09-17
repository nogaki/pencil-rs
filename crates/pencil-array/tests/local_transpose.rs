use std::{
    cell::Cell,
    panic::{AssertUnwindSafe, catch_unwind},
    rc::Rc,
    sync::Arc,
};

use mpi::traits::*;
use pencil_array::{
    AllToAllvTransposePlan, ArrayError, AxisPermutation, ExtraShape, GeometryError,
    LocalTransposeError, LocalTransposePlan, ManyPencilArray, MpiTopology, Pencil, PencilArray,
    TransposeWorkspace,
};

#[derive(Debug)]
struct PanicClone {
    value: usize,
    remaining: Rc<Cell<usize>>,
}

impl Clone for PanicClone {
    fn clone(&self) -> Self {
        let remaining = self.remaining.get();
        assert!(remaining > 0, "injected clone panic");
        self.remaining.set(remaining - 1);
        Self {
            value: self.value,
            remaining: Rc::clone(&self.remaining),
        }
    }
}

#[derive(Debug)]
struct PanicCloneFrom {
    value: usize,
    panic: Rc<Cell<bool>>,
}

impl Clone for PanicCloneFrom {
    fn clone(&self) -> Self {
        Self {
            value: self.value,
            panic: Rc::clone(&self.panic),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.value = source.value;
        if self.panic.get() {
            panic!("injected clone_from panic");
        }
    }
}

#[derive(Debug)]
struct PanicDrop {
    value: usize,
    panic_on_drop: Rc<Cell<bool>>,
}

impl Clone for PanicDrop {
    fn clone(&self) -> Self {
        Self {
            value: self.value,
            panic_on_drop: Rc::clone(&self.panic_on_drop),
        }
    }
}

impl Drop for PanicDrop {
    fn drop(&mut self) {
        if self.panic_on_drop.replace(false) {
            panic!("injected drop panic");
        }
    }
}

fn value_2d(pencil: &Pencil<2, 1>, extra: [usize; 2], spatial: [usize; 2]) -> u64 {
    let ranges = pencil.local_ranges();
    let x = ranges[0].start + spatial[0];
    let y = ranges[1].start + spatial[1];
    (extra[0] * 1_000_000 + extra[1] * 10_000 + x * 100 + y) as u64
}

fn value_3d(pencil: &Pencil<3, 2>, spatial: [usize; 3]) -> u64 {
    let ranges = pencil.local_ranges();
    let x = ranges[0].start + spatial[0];
    let y = ranges[1].start + spatial[1];
    let z = ranges[2].start + spatial[2];
    (x * 10_000 + y * 100 + z) as u64
}

fn assert_many_values_3d(array: &ManyPencilArray<u64, 3, 2>, pencil: &Pencil<3, 2>) {
    assert!(array.active_pencil().unwrap().same_layout(pencil));
    let shape = pencil.local_shape_logical();
    let view = array.active_view().unwrap();
    for x in 0..shape[0] {
        for y in 0..shape[1] {
            for z in 0..shape[2] {
                assert_eq!(
                    view.get_local(&[], [x, y, z]),
                    Some(&value_3d(pencil, [x, y, z])),
                );
            }
        }
    }
}

#[test]
fn local_transpose_covers_layouts_shapes_validation_and_panics() {
    let universe = mpi::initialize().expect("MPI initialization failed");
    let world = universe.world();
    let world_size = usize::try_from(world.size()).unwrap();
    assert!(matches!(world_size, 1 | 4));
    let rank = world.rank();
    let grid_2d = if world_size == 1 { [1, 1] } else { [2, 2] };

    // All MPI/topology construction is unconditional. The local operation is
    // tested below on only one rank as well.
    let topology_1d = MpiTopology::<1>::new(&world, [world_size]).unwrap();
    // Keep every topology constructor in the same order on every rank.
    let topology_1d_other = MpiTopology::<1>::new(&world, [world_size]).unwrap();
    let topology_2d = MpiTopology::<2>::new(&world, grid_2d).unwrap();

    let source_2d =
        Pencil::<2, 1>::new(Arc::clone(&topology_1d), [8 * world_size, 5], [0]).unwrap();
    let destination_2d = source_2d
        .with_permutation(AxisPermutation::new([1, 0]).unwrap())
        .unwrap();
    let incompatible_2d = source_2d.with_decomposition([1]).unwrap();

    let source_3d = Pencil::<3, 2>::new(
        Arc::clone(&topology_2d),
        [4 * grid_2d[0], 6 * grid_2d[1], 5],
        [0, 1],
    )
    .unwrap();
    let destination_3d = source_3d
        .with_permutation(AxisPermutation::new([2, 0, 1]).unwrap())
        .unwrap();
    let redistributed_3d = source_3d.with_decomposition([1, 2]).unwrap();

    let empty_source_pencil =
        Pencil::<2, 2>::new(Arc::clone(&topology_2d), [1, 4 * grid_2d[1]], [0, 1]).unwrap();
    let empty_destination_pencil = empty_source_pencil
        .with_permutation(AxisPermutation::new([1, 0]).unwrap())
        .unwrap();

    let extra = ExtraShape::new([2, 3]).unwrap();
    let mut source = PencilArray::from_elem(Arc::clone(&source_2d), extra.clone(), 0_u64).unwrap();
    let source_shape = source_2d.local_shape_logical();
    for extra_0 in 0..2 {
        for extra_1 in 0..3 {
            for x in 0..source_shape[0] {
                for y in 0..source_shape[1] {
                    *source.get_local_mut(&[extra_0, extra_1], [x, y]).unwrap() =
                        value_2d(&source_2d, [extra_0, extra_1], [x, y]);
                }
            }
        }
    }
    let source_before = source.as_slice().to_vec();

    let plan_2d =
        LocalTransposePlan::new(Arc::clone(&source_2d), Arc::clone(&destination_2d)).unwrap();
    let mut destination =
        PencilArray::from_elem(Arc::clone(&destination_2d), extra.clone(), 0_u64).unwrap();
    plan_2d
        .execute_views(source.view(), destination.view_mut())
        .unwrap();
    assert_eq!(source.as_slice(), source_before.as_slice());
    let destination_shape = destination_2d.local_shape_logical();
    for extra_0 in 0..2 {
        for extra_1 in 0..3 {
            for x in 0..destination_shape[0] {
                for y in 0..destination_shape[1] {
                    assert_eq!(
                        destination.get_local(&[extra_0, extra_1], [x, y]),
                        Some(&value_2d(&source_2d, [extra_0, extra_1], [x, y])),
                    );
                }
            }
        }
    }

    let same_layout_plan =
        LocalTransposePlan::new(Arc::clone(&source_2d), Arc::clone(&source_2d)).unwrap();
    let mut copied = PencilArray::from_elem(Arc::clone(&source_2d), extra.clone(), 0_u64).unwrap();
    same_layout_plan
        .execute_views(source.view(), copied.view_mut())
        .unwrap();
    assert_eq!(copied.as_slice(), source_before.as_slice());

    let backward_plan =
        LocalTransposePlan::new(Arc::clone(&destination_2d), Arc::clone(&source_2d)).unwrap();
    let mut roundtrip =
        PencilArray::from_elem(Arc::clone(&source_2d), extra.clone(), 0_u64).unwrap();
    backward_plan
        .execute_views(destination.view(), roundtrip.view_mut())
        .unwrap();
    assert_eq!(roundtrip.as_slice(), source_before.as_slice());

    let required_2d = source_2d.local_len() * extra.element_count();
    let mut in_place_2d = ManyPencilArray::from_vec(
        vec![Arc::clone(&source_2d), Arc::clone(&destination_2d)],
        0,
        extra.clone(),
        source_before.clone(),
    )
    .unwrap();
    let mut scratch_2d = Vec::with_capacity(required_2d);
    let scratch_2d_ptr = scratch_2d.as_ptr();
    let scratch_2d_capacity = scratch_2d.capacity();
    plan_2d
        .execute_in_place(&mut in_place_2d, &mut scratch_2d)
        .unwrap();
    assert_eq!(scratch_2d.as_ptr(), scratch_2d_ptr);
    assert_eq!(scratch_2d.capacity(), scratch_2d_capacity);
    assert_eq!(scratch_2d.as_slice(), source_before.as_slice());
    assert!(
        in_place_2d
            .active_pencil()
            .unwrap()
            .same_layout(destination_2d.as_ref())
    );
    {
        let view = in_place_2d.active_view().unwrap();
        for extra_0 in 0..2 {
            for extra_1 in 0..3 {
                for x in 0..destination_shape[0] {
                    for y in 0..destination_shape[1] {
                        assert_eq!(
                            view.get_local(&[extra_0, extra_1], [x, y]),
                            Some(&value_2d(&source_2d, [extra_0, extra_1], [x, y])),
                        );
                    }
                }
            }
        }
    }
    let mut backward_scratch_2d = Vec::with_capacity(required_2d);
    backward_plan
        .execute_in_place(&mut in_place_2d, &mut backward_scratch_2d)
        .unwrap();
    assert!(
        in_place_2d
            .active_pencil()
            .unwrap()
            .same_layout(source_2d.as_ref())
    );
    assert_eq!(
        in_place_2d.active_view().unwrap().as_slice(),
        source_before.as_slice()
    );

    let mut same_layout_array = ManyPencilArray::from_vec(
        vec![Arc::clone(&source_2d)],
        0,
        extra.clone(),
        source_before.clone(),
    )
    .unwrap();
    let mut same_layout_scratch = Vec::with_capacity(required_2d);
    same_layout_plan
        .execute_in_place(&mut same_layout_array, &mut same_layout_scratch)
        .unwrap();
    assert!(
        same_layout_array
            .active_pencil()
            .unwrap()
            .same_layout(source_2d.as_ref())
    );
    assert_eq!(
        same_layout_array.active_view().unwrap().as_slice(),
        source_before.as_slice()
    );

    assert!(matches!(
        LocalTransposePlan::new(Arc::clone(&source_2d), incompatible_2d),
        Err(LocalTransposeError::IncompatibleDistribution),
    ));
    let other_topology_pencil =
        Pencil::<2, 1>::new(Arc::clone(&topology_1d_other), [8 * world_size, 5], [0]).unwrap();
    assert!(matches!(
        LocalTransposePlan::new(Arc::clone(&source_2d), other_topology_pencil),
        Err(LocalTransposeError::IncompatibleDistribution),
    ));
    let different_shape = source_2d.with_global_shape([8 * world_size, 6]).unwrap();
    assert!(matches!(
        LocalTransposePlan::new(Arc::clone(&source_2d), different_shape),
        Err(LocalTransposeError::IncompatibleDistribution),
    ));

    let wrong_source =
        PencilArray::from_elem(Arc::clone(&destination_2d), extra.clone(), 11_u64).unwrap();
    let mut mismatch_destination =
        PencilArray::from_elem(Arc::clone(&destination_2d), extra.clone(), 17_u64).unwrap();
    let mismatch_before = mismatch_destination.as_slice().to_vec();
    assert!(matches!(
        plan_2d.execute_views(wrong_source.view(), mismatch_destination.view_mut()),
        Err(LocalTransposeError::SourceLayoutMismatch),
    ));
    assert_eq!(mismatch_destination.as_slice(), mismatch_before.as_slice());

    let wrong_destination =
        PencilArray::from_elem(Arc::clone(&source_2d), extra.clone(), 23_u64).unwrap();
    let mut wrong_destination = wrong_destination;
    let wrong_destination_before = wrong_destination.as_slice().to_vec();
    assert!(matches!(
        plan_2d.execute_views(source.view(), wrong_destination.view_mut()),
        Err(LocalTransposeError::DestinationLayoutMismatch),
    ));
    assert_eq!(
        wrong_destination.as_slice(),
        wrong_destination_before.as_slice()
    );

    let same_count_shape = ExtraShape::new([6]).unwrap();
    let mut same_count_destination =
        PencilArray::from_elem(Arc::clone(&destination_2d), same_count_shape, 29_u64).unwrap();
    let same_count_before = same_count_destination.as_slice().to_vec();
    assert!(matches!(
        plan_2d.execute_views(source.view(), same_count_destination.view_mut()),
        Err(LocalTransposeError::ExtraShapeMismatch),
    ));
    assert_eq!(
        same_count_destination.as_slice(),
        same_count_before.as_slice()
    );
    assert_eq!(source.as_slice(), source_before.as_slice());

    let zero_extra = ExtraShape::new([2, 0]).unwrap();
    let zero_source =
        PencilArray::<u8, 2, 1>::from_elem(Arc::clone(&source_2d), zero_extra.clone(), 0).unwrap();
    let mut zero_destination =
        PencilArray::<u8, 2, 1>::from_elem(Arc::clone(&destination_2d), zero_extra.clone(), 0)
            .unwrap();
    LocalTransposePlan::new(Arc::clone(&source_2d), Arc::clone(&destination_2d))
        .unwrap()
        .execute_views(zero_source.view(), zero_destination.view_mut())
        .unwrap();
    assert!(zero_source.is_empty());
    assert!(zero_destination.is_empty());

    let mut zero_many = ManyPencilArray::from_elem(
        vec![Arc::clone(&source_2d), Arc::clone(&destination_2d)],
        0,
        zero_extra,
        0_u8,
    )
    .unwrap();
    let mut zero_scratch = Vec::with_capacity(1);
    zero_scratch.push(99);
    let zero_scratch_ptr = zero_scratch.as_ptr();
    let zero_scratch_capacity = zero_scratch.capacity();
    LocalTransposePlan::new(Arc::clone(&source_2d), Arc::clone(&destination_2d))
        .unwrap()
        .execute_in_place(&mut zero_many, &mut zero_scratch)
        .unwrap();
    assert!(
        zero_many
            .active_pencil()
            .unwrap()
            .same_layout(destination_2d.as_ref())
    );
    assert!(zero_many.active_view().unwrap().is_empty());
    assert!(zero_scratch.is_empty());
    assert_eq!(zero_scratch.as_ptr(), zero_scratch_ptr);
    assert_eq!(zero_scratch.capacity(), zero_scratch_capacity);

    let huge_pencil = Pencil::<2, 1>::new(Arc::clone(&topology_1d), [usize::MAX, 1], [0]).unwrap();
    let huge_destination_pencil = huge_pencil
        .with_permutation(AxisPermutation::new([1, 0]).unwrap())
        .unwrap();
    let overflowing_extra = ExtraShape::new([5]).unwrap();
    assert!(matches!(
        PencilArray::<u8, 2, 1>::from_vec(
            Arc::clone(&huge_pencil),
            overflowing_extra.clone(),
            Vec::new(),
        ),
        Err(ArrayError::Geometry(GeometryError::SizeOverflow)),
    ));
    assert!(matches!(
        ManyPencilArray::<u8, 2, 1>::from_vec(
            vec![
                Arc::clone(&huge_pencil),
                Arc::clone(&huge_destination_pencil),
            ],
            0,
            overflowing_extra,
            Vec::new(),
        ),
        Err(ArrayError::Geometry(GeometryError::SizeOverflow)),
    ));

    let huge_zero_extra = ExtraShape::new([5, 0]).unwrap();
    let huge_zero_source = PencilArray::<u8, 2, 1>::from_vec(
        Arc::clone(&huge_pencil),
        huge_zero_extra.clone(),
        Vec::new(),
    )
    .unwrap();
    let mut huge_zero_destination = PencilArray::<u8, 2, 1>::from_vec(
        Arc::clone(&huge_destination_pencil),
        huge_zero_extra.clone(),
        Vec::new(),
    )
    .unwrap();
    let huge_zero_plan = LocalTransposePlan::new(
        Arc::clone(&huge_pencil),
        Arc::clone(&huge_destination_pencil),
    )
    .unwrap();
    huge_zero_plan
        .execute_views(huge_zero_source.view(), huge_zero_destination.view_mut())
        .unwrap();
    assert!(huge_zero_source.is_empty());
    assert!(huge_zero_destination.is_empty());

    let mut huge_zero_many = ManyPencilArray::from_vec(
        vec![
            Arc::clone(&huge_pencil),
            Arc::clone(&huge_destination_pencil),
        ],
        0,
        huge_zero_extra,
        Vec::<u8>::new(),
    )
    .unwrap();
    let mut huge_zero_scratch = Vec::new();
    huge_zero_plan
        .execute_in_place(&mut huge_zero_many, &mut huge_zero_scratch)
        .unwrap();
    assert!(
        huge_zero_many
            .active_pencil()
            .unwrap()
            .same_layout(huge_destination_pencil.as_ref())
    );
    assert!(huge_zero_many.active_view().unwrap().is_empty());
    assert!(huge_zero_scratch.is_empty());

    let mut source_3d_array =
        PencilArray::from_elem(Arc::clone(&source_3d), ExtraShape::scalar(), 0_u64).unwrap();
    let shape_3d = source_3d.local_shape_logical();
    for x in 0..shape_3d[0] {
        for y in 0..shape_3d[1] {
            for z in 0..shape_3d[2] {
                *source_3d_array.get_local_mut(&[], [x, y, z]).unwrap() =
                    value_3d(&source_3d, [x, y, z]);
            }
        }
    }
    let source_3d_before = source_3d_array.as_slice().to_vec();
    let mut destination_3d_array =
        PencilArray::from_elem(Arc::clone(&destination_3d), ExtraShape::scalar(), 0_u64).unwrap();
    let plan_3d =
        LocalTransposePlan::new(Arc::clone(&source_3d), Arc::clone(&destination_3d)).unwrap();

    // The initialized-prefix adapter keeps its Vec length and tail intact,
    // then the same workspace remains usable by Alltoallv.
    let alltoallv_redistributed_3d = source_3d.with_decomposition([0, 2]).unwrap();
    let redistributed_3d_transposed = alltoallv_redistributed_3d
        .with_permutation(AxisPermutation::new([2, 0, 1]).unwrap())
        .unwrap();
    let alltoallv_after_local = AllToAllvTransposePlan::new(
        Arc::clone(&destination_3d),
        Arc::clone(&redistributed_3d_transposed),
    )
    .unwrap();
    let alltoallv_requirements = alltoallv_after_local
        .workspace_requirements(&ExtraShape::scalar())
        .unwrap();
    let required_local = source_3d.local_len();
    assert!(required_local > 0);
    let shared_send_len = required_local.max(alltoallv_requirements.send_len);
    let tail = [0xfeed_u64, 0xcafe_u64, 0xbeef_u64];
    let mut shared_send = vec![0xabad_u64; shared_send_len];
    shared_send.extend_from_slice(&tail);
    let mut shared_workspace =
        TransposeWorkspace::from_vecs(shared_send, vec![0_u64; alltoallv_requirements.receive_len]);
    let mut shared_storage = source_3d_before.clone();
    shared_storage.resize(
        source_3d
            .local_len()
            .max(redistributed_3d_transposed.local_len()),
        0_u64,
    );
    let mut shared_many = ManyPencilArray::from_vec(
        vec![
            Arc::clone(&source_3d),
            Arc::clone(&destination_3d),
            Arc::clone(&redistributed_3d_transposed),
        ],
        0,
        ExtraShape::scalar(),
        shared_storage,
    )
    .unwrap();
    plan_3d
        .execute_in_place_with_transpose_workspace(&mut shared_many, &mut shared_workspace)
        .unwrap();
    assert_eq!(shared_workspace.send_len(), shared_send_len + tail.len());
    assert_eq!(
        shared_workspace.receive_len(),
        alltoallv_requirements.receive_len
    );
    assert_many_values_3d(&shared_many, destination_3d.as_ref());
    alltoallv_after_local
        .execute_in_place(&mut shared_many, &mut shared_workspace)
        .unwrap();
    assert_many_values_3d(&shared_many, redistributed_3d_transposed.as_ref());

    // Initialized length, not spare capacity, controls the adapter's check.
    let mut short_many = ManyPencilArray::from_vec(
        vec![Arc::clone(&source_3d), Arc::clone(&destination_3d)],
        0,
        ExtraShape::scalar(),
        source_3d_before.clone(),
    )
    .unwrap();
    let mut short_send = Vec::with_capacity(required_local);
    short_send.resize(required_local - 1, 0_u64);
    let mut short_workspace = TransposeWorkspace::from_vecs(short_send, Vec::new());
    let short_before = short_many.active_view().unwrap().as_slice().to_vec();
    assert!(matches!(
        plan_3d.execute_in_place_with_transpose_workspace(&mut short_many, &mut short_workspace),
        Err(LocalTransposeError::ScratchTooSmall { required, actual })
            if required == required_local && actual == required_local - 1
    ));
    assert_eq!(short_many.active_view().unwrap().as_slice(), short_before);
    assert_eq!(short_workspace.send_len(), required_local - 1);

    plan_3d
        .execute_views(source_3d_array.view(), destination_3d_array.view_mut())
        .unwrap();
    assert_eq!(source_3d_array.as_slice(), source_3d_before.as_slice());
    let destination_3d_shape = destination_3d.local_shape_logical();
    for x in 0..destination_3d_shape[0] {
        for y in 0..destination_3d_shape[1] {
            for z in 0..destination_3d_shape[2] {
                assert_eq!(
                    destination_3d_array.get_local(&[], [x, y, z]),
                    Some(&value_3d(&source_3d, [x, y, z])),
                );
            }
        }
    }
    let reverse_plan_3d =
        LocalTransposePlan::new(Arc::clone(&destination_3d), Arc::clone(&source_3d)).unwrap();
    let mut reverse_3d_array =
        PencilArray::from_elem(Arc::clone(&source_3d), ExtraShape::scalar(), 0_u64).unwrap();
    reverse_plan_3d
        .execute_views(destination_3d_array.view(), reverse_3d_array.view_mut())
        .unwrap();
    for x in 0..shape_3d[0] {
        for y in 0..shape_3d[1] {
            for z in 0..shape_3d[2] {
                assert_eq!(
                    reverse_3d_array.get_local(&[], [x, y, z]),
                    Some(&value_3d(&source_3d, [x, y, z])),
                );
            }
        }
    }
    assert_eq!(reverse_3d_array.as_slice(), source_3d_before.as_slice());

    let required_3d = source_3d.local_len();
    let max_required_3d = required_3d.max(redistributed_3d.local_len());
    let suffix_sentinel = 0xfeed_u64;
    let mut in_place_3d_storage = source_3d_before.clone();
    in_place_3d_storage.resize(max_required_3d, suffix_sentinel);
    let mut in_place_3d = ManyPencilArray::from_vec(
        vec![
            Arc::clone(&source_3d),
            Arc::clone(&destination_3d),
            Arc::clone(&redistributed_3d),
        ],
        0,
        ExtraShape::scalar(),
        in_place_3d_storage,
    )
    .unwrap();
    let mut scratch_3d = Vec::with_capacity(required_3d);
    let scratch_3d_ptr = scratch_3d.as_ptr();
    let scratch_3d_capacity = scratch_3d.capacity();
    plan_3d
        .execute_in_place(&mut in_place_3d, &mut scratch_3d)
        .unwrap();
    assert_eq!(scratch_3d.as_ptr(), scratch_3d_ptr);
    assert_eq!(scratch_3d.capacity(), scratch_3d_capacity);
    assert_eq!(scratch_3d.as_slice(), source_3d_before.as_slice());
    {
        let view = in_place_3d.active_view().unwrap();
        for x in 0..destination_3d_shape[0] {
            for y in 0..destination_3d_shape[1] {
                for z in 0..destination_3d_shape[2] {
                    assert_eq!(
                        view.get_local(&[], [x, y, z]),
                        Some(&value_3d(&source_3d, [x, y, z])),
                    );
                }
            }
        }
    }
    if redistributed_3d.local_len() > required_3d {
        in_place_3d
            .overwrite_with(redistributed_3d.as_ref(), |mut view| {
                assert!(
                    view.as_slice()[required_3d..]
                        .iter()
                        .all(|&value| value == suffix_sentinel)
                );
                view.as_mut_slice().fill(suffix_sentinel);
                Ok::<_, ()>(())
            })
            .unwrap();
    }

    let empty_source = PencilArray::from_elem(
        Arc::clone(&empty_source_pencil),
        ExtraShape::scalar(),
        31_u8,
    )
    .unwrap();
    let mut empty_destination = PencilArray::from_elem(
        Arc::clone(&empty_destination_pencil),
        ExtraShape::scalar(),
        37_u8,
    )
    .unwrap();
    let empty_plan = LocalTransposePlan::new(
        Arc::clone(&empty_source_pencil),
        Arc::clone(&empty_destination_pencil),
    )
    .unwrap();
    if rank == 0 {
        empty_plan
            .execute_views(empty_source.view(), empty_destination.view_mut())
            .unwrap();
        if world_size == 4 {
            assert!(empty_source.is_empty());
            assert!(empty_destination.is_empty());
        } else {
            assert!(
                empty_destination
                    .as_slice()
                    .iter()
                    .all(|&value| value == 31)
            );
        }
    }

    let mut empty_many = ManyPencilArray::from_elem(
        vec![
            Arc::clone(&empty_source_pencil),
            Arc::clone(&empty_destination_pencil),
        ],
        0,
        ExtraShape::scalar(),
        31_u8,
    )
    .unwrap();
    let mut empty_scratch = Vec::with_capacity(empty_source_pencil.local_len());
    empty_plan
        .execute_in_place(&mut empty_many, &mut empty_scratch)
        .unwrap();
    assert!(
        empty_many
            .active_pencil()
            .unwrap()
            .same_layout(empty_destination_pencil.as_ref())
    );
    assert_eq!(
        empty_many.active_view().unwrap().len(),
        empty_destination_pencil.local_len()
    );
    assert_eq!(empty_scratch.len(), empty_destination_pencil.local_len());

    let panic_budget = Rc::new(Cell::new(2));
    let panic_source_values = (0..source_2d.local_len())
        .map(|value| PanicClone {
            value,
            remaining: Rc::clone(&panic_budget),
        })
        .collect();
    let panic_destination_values = (0..destination_2d.local_len())
        .map(|value| PanicClone {
            value,
            remaining: Rc::new(Cell::new(usize::MAX)),
        })
        .collect();
    let panic_source = PencilArray::from_vec(
        Arc::clone(&source_2d),
        ExtraShape::scalar(),
        panic_source_values,
    )
    .unwrap();
    let mut panic_destination = PencilArray::from_vec(
        Arc::clone(&destination_2d),
        ExtraShape::scalar(),
        panic_destination_values,
    )
    .unwrap();
    let panic_source_before: Vec<_> = panic_source
        .as_slice()
        .iter()
        .map(|item| item.value)
        .collect();
    let panic_plan =
        LocalTransposePlan::new(Arc::clone(&source_2d), Arc::clone(&destination_2d)).unwrap();
    let panic_result = catch_unwind(AssertUnwindSafe(|| {
        panic_plan
            .execute_views(panic_source.view(), panic_destination.view_mut())
            .unwrap();
    }));
    assert!(panic_result.is_err());
    let panic_source_after: Vec<_> = panic_source
        .as_slice()
        .iter()
        .map(|item| item.value)
        .collect();
    assert_eq!(panic_source_after, panic_source_before);

    let mut too_small_array = ManyPencilArray::from_vec(
        vec![Arc::clone(&source_2d), Arc::clone(&destination_2d)],
        0,
        extra.clone(),
        source_before.clone(),
    )
    .unwrap();
    let too_small_before = too_small_array.active_view().unwrap().as_slice().to_vec();
    let mut too_small_scratch = Vec::<u64>::new();
    let too_small_scratch_before = too_small_scratch.clone();
    let too_small_ptr = too_small_scratch.as_ptr();
    let too_small_capacity = too_small_scratch.capacity();
    assert!(too_small_capacity < required_2d);
    assert!(matches!(
        plan_2d.execute_in_place(&mut too_small_array, &mut too_small_scratch),
        Err(LocalTransposeError::ScratchTooSmall {
            required,
            actual,
        }) if required == required_2d && actual == too_small_capacity,
    ));
    assert!(
        too_small_array
            .active_pencil()
            .unwrap()
            .same_layout(source_2d.as_ref())
    );
    assert_eq!(
        too_small_array.active_view().unwrap().as_slice(),
        too_small_before.as_slice()
    );
    assert_eq!(too_small_scratch, too_small_scratch_before);
    assert_eq!(too_small_scratch.as_ptr(), too_small_ptr);
    assert_eq!(too_small_scratch.capacity(), too_small_capacity);

    let mut wrong_active_array = ManyPencilArray::from_vec(
        vec![Arc::clone(&source_2d), Arc::clone(&destination_2d)],
        1,
        extra.clone(),
        source_before.clone(),
    )
    .unwrap();
    let wrong_active_before = wrong_active_array
        .active_view()
        .unwrap()
        .as_slice()
        .to_vec();
    let mut wrong_active_scratch = Vec::with_capacity(required_2d);
    wrong_active_scratch.push(73);
    assert!(matches!(
        plan_2d.execute_in_place(&mut wrong_active_array, &mut wrong_active_scratch),
        Err(LocalTransposeError::SourceLayoutMismatch),
    ));
    assert!(
        wrong_active_array
            .active_pencil()
            .unwrap()
            .same_layout(destination_2d.as_ref())
    );
    assert_eq!(
        wrong_active_array.active_view().unwrap().as_slice(),
        wrong_active_before.as_slice()
    );
    assert_eq!(wrong_active_scratch.as_slice(), &[73]);

    let mut unregistered_destination_array = ManyPencilArray::from_vec(
        vec![Arc::clone(&source_2d)],
        0,
        extra.clone(),
        source_before.clone(),
    )
    .unwrap();
    let unregistered_before = unregistered_destination_array
        .active_view()
        .unwrap()
        .as_slice()
        .to_vec();
    let mut unregistered_scratch = Vec::with_capacity(required_2d);
    unregistered_scratch.push(74);
    assert!(matches!(
        plan_2d.execute_in_place(
            &mut unregistered_destination_array,
            &mut unregistered_scratch,
        ),
        Err(LocalTransposeError::DestinationLayoutMismatch),
    ));
    assert!(
        unregistered_destination_array
            .active_pencil()
            .unwrap()
            .same_layout(source_2d.as_ref())
    );
    assert_eq!(
        unregistered_destination_array
            .active_view()
            .unwrap()
            .as_slice(),
        unregistered_before.as_slice()
    );
    assert_eq!(unregistered_scratch.as_slice(), &[74]);

    let staging_budget = Rc::new(Cell::new(2));
    let staging_values = (0..required_2d)
        .map(|value| PanicClone {
            value,
            remaining: Rc::clone(&staging_budget),
        })
        .collect();
    let mut staging_array = ManyPencilArray::from_vec(
        vec![Arc::clone(&source_2d), Arc::clone(&destination_2d)],
        0,
        extra.clone(),
        staging_values,
    )
    .unwrap();
    let staging_before: Vec<_> = staging_array
        .active_view()
        .unwrap()
        .as_slice()
        .iter()
        .map(|item| item.value)
        .collect();
    let mut staging_scratch = Vec::with_capacity(required_2d);
    let staging_result = catch_unwind(AssertUnwindSafe(|| {
        plan_2d
            .execute_in_place(&mut staging_array, &mut staging_scratch)
            .unwrap();
    }));
    assert!(staging_result.is_err());
    assert!(
        staging_array
            .active_pencil()
            .unwrap()
            .same_layout(source_2d.as_ref())
    );
    let staging_after: Vec<_> = staging_array
        .active_view()
        .unwrap()
        .as_slice()
        .iter()
        .map(|item| item.value)
        .collect();
    assert_eq!(staging_after, staging_before);

    let clear_drop_panic = Rc::new(Cell::new(false));
    let clear_values: Vec<_> = (0..required_2d)
        .map(|value| PanicDrop {
            value,
            panic_on_drop: Rc::clone(&clear_drop_panic),
        })
        .collect();
    let mut clear_array = ManyPencilArray::from_vec(
        vec![Arc::clone(&source_2d), Arc::clone(&destination_2d)],
        0,
        extra.clone(),
        clear_values,
    )
    .unwrap();
    let clear_before: Vec<_> = clear_array
        .active_view()
        .unwrap()
        .as_slice()
        .iter()
        .map(|item| item.value)
        .collect();
    let mut clear_scratch = Vec::with_capacity(required_2d);
    clear_scratch.push(PanicDrop {
        value: usize::MAX,
        panic_on_drop: Rc::clone(&clear_drop_panic),
    });
    clear_drop_panic.set(true);
    let clear_result = catch_unwind(AssertUnwindSafe(|| {
        plan_2d
            .execute_in_place(&mut clear_array, &mut clear_scratch)
            .unwrap();
    }));
    assert!(clear_result.is_err());
    assert!(!clear_drop_panic.get());
    assert!(
        clear_array
            .active_pencil()
            .unwrap()
            .same_layout(source_2d.as_ref())
    );
    let clear_after: Vec<_> = clear_array
        .active_view()
        .unwrap()
        .as_slice()
        .iter()
        .map(|item| item.value)
        .collect();
    assert_eq!(clear_after, clear_before);

    let body_drop_panic = Rc::new(Cell::new(false));
    let body_drop_values: Vec<_> = (0..required_2d)
        .map(|value| PanicDrop {
            value,
            panic_on_drop: Rc::clone(&body_drop_panic),
        })
        .collect();
    let mut body_drop_array = ManyPencilArray::from_vec(
        vec![Arc::clone(&source_2d), Arc::clone(&destination_2d)],
        0,
        extra.clone(),
        body_drop_values,
    )
    .unwrap();
    let mut body_drop_scratch = Vec::with_capacity(required_2d);
    body_drop_panic.set(true);
    let body_drop_result = catch_unwind(AssertUnwindSafe(|| {
        plan_2d
            .execute_in_place(&mut body_drop_array, &mut body_drop_scratch)
            .unwrap();
    }));
    assert!(body_drop_result.is_err());
    assert!(!body_drop_panic.get());
    assert_eq!(
        body_drop_array.active_pencil().unwrap_err(),
        ArrayError::Poisoned
    );
    body_drop_array
        .overwrite_with(destination_2d.as_ref(), |mut view| {
            view.as_mut_slice().fill(PanicDrop {
                value: 902,
                panic_on_drop: Rc::clone(&body_drop_panic),
            });
            Ok::<_, ()>(())
        })
        .unwrap();
    assert!(
        body_drop_array
            .active_view()
            .unwrap()
            .as_slice()
            .iter()
            .all(|item| item.value == 902)
    );

    let clone_from_panic = Rc::new(Cell::new(true));
    let clone_from_values = (0..required_2d)
        .map(|value| PanicCloneFrom {
            value,
            panic: Rc::clone(&clone_from_panic),
        })
        .collect();
    let mut post_poison_array = ManyPencilArray::from_vec(
        vec![Arc::clone(&source_2d), Arc::clone(&destination_2d)],
        0,
        extra.clone(),
        clone_from_values,
    )
    .unwrap();
    let mut post_poison_scratch = Vec::with_capacity(required_2d);
    let post_poison_ptr = post_poison_scratch.as_ptr();
    let post_poison_capacity = post_poison_scratch.capacity();
    let post_poison_result = catch_unwind(AssertUnwindSafe(|| {
        plan_2d
            .execute_in_place(&mut post_poison_array, &mut post_poison_scratch)
            .unwrap();
    }));
    assert!(post_poison_result.is_err());
    assert_eq!(
        post_poison_array.active_pencil().unwrap_err(),
        ArrayError::Poisoned
    );
    assert_eq!(
        post_poison_array.active_view().unwrap_err(),
        ArrayError::Poisoned
    );
    assert_eq!(post_poison_scratch.len(), required_2d);
    let post_poison_scratch_before: Vec<_> =
        post_poison_scratch.iter().map(|item| item.value).collect();
    assert_eq!(post_poison_scratch.as_ptr(), post_poison_ptr);
    assert_eq!(post_poison_scratch.capacity(), post_poison_capacity);
    assert!(matches!(
        plan_2d.execute_in_place(&mut post_poison_array, &mut post_poison_scratch),
        Err(LocalTransposeError::Array(ArrayError::Poisoned)),
    ));
    assert_eq!(
        post_poison_scratch
            .iter()
            .map(|item| item.value)
            .collect::<Vec<_>>(),
        post_poison_scratch_before,
    );
    assert_eq!(post_poison_scratch.as_ptr(), post_poison_ptr);
    post_poison_array
        .overwrite_with(destination_2d.as_ref(), |mut view| {
            for item in view.as_mut_slice() {
                item.value = 901;
            }
            Ok::<_, ()>(())
        })
        .unwrap();
    assert!(
        post_poison_array
            .active_pencil()
            .unwrap()
            .same_layout(destination_2d.as_ref())
    );
    assert!(
        post_poison_array
            .active_view()
            .unwrap()
            .as_slice()
            .iter()
            .all(|item| item.value == 901)
    );

    // No rank other than rank zero enters this local execution.
    let root_source =
        PencilArray::from_elem(Arc::clone(&source_2d), ExtraShape::scalar(), 41_u64).unwrap();
    let mut root_destination =
        PencilArray::from_elem(Arc::clone(&destination_2d), ExtraShape::scalar(), 0_u64).unwrap();
    if rank == 0 {
        plan_2d
            .execute_views(root_source.view(), root_destination.view_mut())
            .unwrap();
        assert!(root_destination.as_slice().iter().all(|&value| value == 41));
    }

    let mut root_many = ManyPencilArray::from_elem(
        vec![Arc::clone(&source_2d), Arc::clone(&destination_2d)],
        0,
        ExtraShape::scalar(),
        41_u64,
    )
    .unwrap();
    let mut root_scratch = Vec::with_capacity(source_2d.local_len());
    if rank == 0 {
        plan_2d
            .execute_in_place(&mut root_many, &mut root_scratch)
            .unwrap();
        assert!(
            root_many
                .active_pencil()
                .unwrap()
                .same_layout(destination_2d.as_ref())
        );
        assert!(
            root_many
                .active_view()
                .unwrap()
                .as_slice()
                .iter()
                .all(|&value| value == 41)
        );
    }
}
