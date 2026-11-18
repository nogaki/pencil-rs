use std::sync::Arc;

use mpi::traits::*;

use super::{PencilArrayView, PencilArrayViewMut};
use crate::{
    ArrayError, AxisPermutation, ExtraShape, LocalTransposeError, LocalTransposePlan,
    ManyPencilArray, MpiTopology, Pencil, PointToPointTransposePlan, TransposeWorkspace,
    point_to_point_transpose::{test_trace_event, test_trace_finish, test_trace_start},
};

#[test]
fn views_validate_storage_and_expose_permuted_local_layout() {
    let universe = mpi::initialize().expect("MPI initialization failed");
    let world = universe.world();
    if world.size() != 1 {
        let topology = MpiTopology::<1>::new(&world, [world.size() as usize]).unwrap();
        check_callback_schedule(&topology);
        return;
    }

    let topology = MpiTopology::<1>::new(&world, [1]).unwrap();
    let pencil = Pencil::<2, 1>::new_permuted(
        Arc::clone(&topology),
        [2, 3],
        [0],
        AxisPermutation::new([1, 0]).unwrap(),
    )
    .unwrap();
    let extra_shape = ExtraShape::new([2]).unwrap();

    assert_eq!(
        PencilArrayView::new(&pencil, &extra_shape, &[0; 11]).unwrap_err(),
        ArrayError::StorageLengthMismatch {
            required: 12,
            actual: 11,
        },
    );
    assert_eq!(
        PencilArrayView::new(&pencil, &extra_shape, &[0; 13]).unwrap_err(),
        ArrayError::StorageLengthMismatch {
            required: 12,
            actual: 13,
        },
    );
    let mut short_storage = [0; 11];
    assert_eq!(
        PencilArrayViewMut::new(&pencil, &extra_shape, &mut short_storage).unwrap_err(),
        ArrayError::StorageLengthMismatch {
            required: 12,
            actual: 11,
        },
    );

    let storage: Vec<_> = (0..12).collect();
    let view = PencilArrayView::new(&pencil, &extra_shape, &storage).unwrap();
    assert!(std::ptr::eq(view.pencil(), pencil.as_ref()));
    assert_eq!(view.extra_shape(), &extra_shape);
    assert_eq!(view.local_spatial_shape(), [2, 3]);
    assert_eq!(view.local_spatial_memory_shape(), [3, 2]);
    assert_eq!(view.logical_shape(), [2, 2, 3]);
    assert_eq!(view.memory_shape(), [2, 3, 2]);
    assert_eq!(view.len(), 12);
    assert!(!view.is_empty());
    assert_eq!(view.as_slice(), storage.as_slice());
    assert_eq!(view.get_local(&[1], [1, 2]), Some(&11));
    assert_eq!(view.get_local(&[0], [1, 0]), Some(&1));
    assert_eq!(view.get_local(&[0], [0, 1]), Some(&2));
    assert_eq!(view.get_local(&[], [1, 2]), None);
    assert_eq!(view.get_local(&[2], [1, 2]), None);
    assert_eq!(view.get_local(&[0], [2, 0]), None);
    assert_eq!(view.get_local(&[0], [0, 3]), None);

    let mut storage: Vec<_> = (0..12).collect();
    {
        let mut view = PencilArrayViewMut::new(&pencil, &extra_shape, &mut storage).unwrap();
        assert_eq!(view.logical_shape(), [2, 2, 3]);
        assert_eq!(view.memory_shape(), [2, 3, 2]);
        assert_eq!(view.get_local(&[0], [1, 2]), Some(&5));
        assert_eq!(view.get_local(&[0], [1, 0]), Some(&1));
        assert_eq!(view.get_local(&[0], [0, 1]), Some(&2));
        *view.get_local_mut(&[0], [1, 2]).unwrap() = 99;
        view.as_mut_slice()[0] = 77;
    }
    assert_eq!(storage[5], 99);
    assert_eq!(storage[0], 77);

    let destination = pencil
        .with_permutation(AxisPermutation::new([0, 1]).unwrap())
        .unwrap();
    let transpose_plan =
        LocalTransposePlan::new(Arc::clone(&pencil), Arc::clone(&destination)).unwrap();
    let mut many = ManyPencilArray::from_vec(
        vec![Arc::clone(&pencil), Arc::clone(&destination)],
        0,
        extra_shape.clone(),
        (0..12).collect(),
    )
    .unwrap();
    let expected_logical = {
        let view = many.active_view().unwrap();
        let mut values = Vec::new();
        for extra_index in 0..2 {
            for x in 0..2 {
                for y in 0..3 {
                    values.push(*view.get_local(&[extra_index], [x, y]).unwrap());
                }
            }
        }
        values
    };
    let tail = [0xfeed_i32, 0xcafe_i32, 0xbeef_i32];
    let mut send = Vec::with_capacity(32);
    send.resize(12, 0);
    send.extend_from_slice(&tail);
    let send_ptr = send.as_ptr();
    let send_len = send.len();
    let send_capacity = send.capacity();
    let receive = vec![0x1234_i32, 0x5678_i32];
    let receive_before = receive.clone();
    let receive_ptr = receive.as_ptr();
    let receive_len = receive.len();
    let receive_capacity = receive.capacity();
    let mut workspace = TransposeWorkspace::from_vecs(send, receive);
    transpose_plan
        .execute_in_place_with_transpose_workspace(&mut many, &mut workspace)
        .unwrap();
    assert!(
        many.active_pencil()
            .unwrap()
            .same_layout(destination.as_ref())
    );
    {
        let view = many.active_view().unwrap();
        let mut index = 0;
        for extra_index in 0..2 {
            for x in 0..2 {
                for y in 0..3 {
                    assert_eq!(
                        view.get_local(&[extra_index], [x, y]),
                        Some(&expected_logical[index])
                    );
                    index += 1;
                }
            }
        }
        assert_eq!(index, expected_logical.len());
    }
    let expected_staged: Vec<_> = (0..12).collect();
    assert_eq!(&workspace.send_buffer[..12], expected_staged.as_slice());
    assert_eq!(&workspace.send_buffer[12..], &tail);
    assert_eq!(workspace.send_buffer.as_ptr(), send_ptr);
    assert_eq!(workspace.send_buffer.len(), send_len);
    assert_eq!(workspace.send_buffer.capacity(), send_capacity);
    assert_eq!(
        workspace.receive_buffer.as_slice(),
        receive_before.as_slice()
    );
    assert_eq!(workspace.receive_buffer.as_ptr(), receive_ptr);
    assert_eq!(workspace.receive_buffer.len(), receive_len);
    assert_eq!(workspace.receive_buffer.capacity(), receive_capacity);

    let mut short_many = ManyPencilArray::from_vec(
        vec![Arc::clone(&pencil), Arc::clone(&destination)],
        0,
        extra_shape,
        (0..12).collect(),
    )
    .unwrap();
    let short_values = short_many.active_view().unwrap().as_slice().to_vec();
    let mut short_send = Vec::with_capacity(12);
    short_send.resize(11, 0);
    let short_send_before = short_send.clone();
    let short_send_ptr = short_send.as_ptr();
    let short_send_capacity = short_send.capacity();
    let short_receive = vec![0x9abc_i32, 0xdef0_i32];
    let short_receive_before = short_receive.clone();
    let short_receive_ptr = short_receive.as_ptr();
    let short_receive_capacity = short_receive.capacity();
    let mut short_workspace = TransposeWorkspace::from_vecs(short_send, short_receive);
    assert!(matches!(
        transpose_plan
            .execute_in_place_with_transpose_workspace(&mut short_many, &mut short_workspace,),
        Err(LocalTransposeError::ScratchTooSmall {
            required: 12,
            actual: 11,
        })
    ));
    assert!(
        short_many
            .active_pencil()
            .unwrap()
            .same_layout(pencil.as_ref())
    );
    assert_eq!(short_many.active_view().unwrap().as_slice(), short_values);
    assert_eq!(
        short_workspace.send_buffer.as_slice(),
        short_send_before.as_slice()
    );
    assert_eq!(short_workspace.send_buffer.as_ptr(), short_send_ptr);
    assert_eq!(short_workspace.send_buffer.capacity(), short_send_capacity);
    assert_eq!(
        short_workspace.receive_buffer.as_slice(),
        short_receive_before.as_slice()
    );
    assert_eq!(short_workspace.receive_buffer.as_ptr(), short_receive_ptr);
    assert_eq!(
        short_workspace.receive_buffer.capacity(),
        short_receive_capacity
    );

    check_callback_schedule(&topology);
}

fn check_callback_schedule(topology: &Arc<MpiTopology<1>>) {
    let callback_source_pencil = Pencil::<2, 1>::new(Arc::clone(topology), [2, 3], [0]).unwrap();
    let callback_destination_pencil = Pencil::<2, 1>::new_permuted(
        Arc::clone(topology),
        [2, 3],
        [1],
        AxisPermutation::new([1, 0]).unwrap(),
    )
    .unwrap();
    let callback_plan = PointToPointTransposePlan::new(
        Arc::clone(&callback_source_pencil),
        Arc::clone(&callback_destination_pencil),
    )
    .unwrap();
    let callback_requirements = callback_plan
        .workspace_requirements(&ExtraShape::new([2]).unwrap())
        .unwrap();
    let mut callback_workspace = TransposeWorkspace::from_vecs(
        vec![0; callback_requirements.send_len],
        vec![0; callback_requirements.receive_len],
    );
    let callback_source = vec![0_i32; callback_source_pencil.local_len() * 2];
    let mut callback_destination = vec![0_i32; callback_destination_pencil.local_len() * 2];
    test_trace_start();
    callback_plan
        .execute_views_with_callback(
            PencilArrayView::new(
                &callback_source_pencil,
                &ExtraShape::new([2]).unwrap(),
                &callback_source,
            )
            .unwrap(),
            PencilArrayViewMut::new(
                &callback_destination_pencil,
                &ExtraShape::new([2]).unwrap(),
                &mut callback_destination,
            )
            .unwrap(),
            &mut callback_workspace,
            |_values| {
                test_trace_event("callback");
                Ok::<_, ()>(())
            },
        )
        .unwrap();
    assert_eq!(test_trace_finish(), ["unpack", "callback", "send_wait"]);
}
