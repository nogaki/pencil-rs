use mpi::topology::Communicator;

use super::*;

#[test]
fn identity_transition_supports_overlap() {
    assert!(C2cTransition::<2, 1>::Identity.supports_overlap());
}

#[test]
#[ignore = "MPI must be initialized in a fresh one-rank test process"]
fn transition_overlap_support_covers_every_transport_kind() {
    let universe = mpi::initialize().expect("MPI initialization failed");
    let world = universe.world();
    assert_eq!(world.size(), 1, "run this test with one MPI rank");
    let topology = MpiTopology::<1>::new(&world, [1]).unwrap();
    let source = Pencil::<2, 1>::new(Arc::clone(&topology), [4, 4], [0]).unwrap();
    let destination = Pencil::<2, 1>::new(Arc::clone(&topology), [4, 4], [1]).unwrap();

    let local_destination = source
        .with_permutation(AxisPermutation::new([1, 0]).unwrap())
        .unwrap();
    assert!(
        C2cTransition::Local(
            LocalTransposePlan::new(Arc::clone(&source), local_destination).unwrap()
        )
        .supports_overlap()
    );
    assert!(
        !build_distributed_transition(
            Arc::clone(&source),
            Arc::clone(&destination),
            TransposeMethod::AllToAllv,
        )
        .unwrap()
        .supports_overlap()
    );
    assert!(
        build_distributed_transition(source, destination, TransposeMethod::PointToPoint,)
            .unwrap()
            .supports_overlap()
    );

    #[cfg(feature = "fftw")]
    {
        let selection = AxisSelection::from_indices([1]).unwrap();
        let rust = c2r::C2rPlan::<f64, 2, 1>::from_shape_with_selection_and_layout(
            Arc::clone(&topology),
            [2, 4],
            6,
            ExtraShape::scalar(),
            selection,
            DistributedLayout::default(),
        )
        .unwrap();
        assert_eq!(rust.input_pencil().global_shape(), &[2, 4]);
        assert_eq!(rust.output_pencil().global_shape(), &[2, 6]);
        assert_eq!(rust.reduction_axis(), 1);
        assert_eq!(rust.layout(), DistributedLayout::default());
        assert_eq!(rust.backend_kind(), crate::BackendKind::RustFft);

        let check_invalid_spectrum = |plan: &c2r::C2rPlan<f64, 2, 1>| {
            let mut source = plan.allocate_input().unwrap();
            *source.get_global_mut(&[], [0, 0]).unwrap() = Complex::new(0.0, 1.0e30);
            let source_before = source.as_slice().to_vec();
            let mut destination = plan.allocate_output().unwrap();
            destination.as_mut_slice().fill(17.0);
            let destination_before = destination.as_slice().to_vec();
            let mut workspace = plan.allocate_workspace().unwrap();
            assert!(matches!(
                plan.forward(&source, &mut destination, &mut workspace),
                Err(c2r::C2rError::InvalidSpectrum)
            ));
            assert_eq!(source.as_slice(), source_before);
            assert_eq!(destination.as_slice(), destination_before);
        };
        check_invalid_spectrum(&rust);

        let native = rust.with_fftw(crate::PlanOptions::default()).unwrap();
        assert_eq!(native.real_len(), 6);
        assert_eq!(native.input_pencil().global_shape(), &[2, 4]);
        assert_eq!(native.output_pencil().global_shape(), &[2, 6]);
        assert_eq!(native.reduction_axis(), 1);
        assert_eq!(native.layout(), DistributedLayout::default());
        assert_eq!(native.backend_kind(), crate::BackendKind::Fftw);
        check_invalid_spectrum(&native);
    }
}

#[test]
fn strided_differing_in_place_lines_match_compression_and_expansion_oracles() {
    let mut compressed = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let mut source_line = [0; 2];
    let mut destination_line = [0; 1];
    let mut visited = Vec::new();
    for_each_strided_line_differing_in_place::<i32, i32, i32, FftError>(
        &mut compressed,
        &mut source_line,
        &mut destination_line,
        2,
        4,
        2,
        2,
        |value| *value,
        |slot, value| *slot = value,
        |source, destination| {
            visited.push(source.to_vec());
            destination[0] = source[0] * 10 + source[1];
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(visited, [vec![1, 3], vec![2, 4], vec![5, 7], vec![6, 8]]);
    assert_eq!(compressed, [13, 24, 57, 68, 5, 6, 7, 8]);

    let mut expanded: Vec<_> = (1..=12).collect();
    let mut source_line = [0; 2];
    let mut destination_line = [0; 3];
    let mut visited = Vec::new();
    for_each_strided_line_differing_in_place::<i32, i32, i32, FftError>(
        &mut expanded,
        &mut source_line,
        &mut destination_line,
        2,
        6,
        6,
        2,
        |value| *value,
        |slot, value| *slot = value,
        |source, destination| {
            visited.push(source.to_vec());
            destination.copy_from_slice(&[source[0], source[1], source[0] + source[1]]);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(visited, [vec![1, 3], vec![2, 4], vec![7, 9], vec![8, 10]]);
    assert_eq!(expanded, [1, 2, 3, 4, 4, 6, 7, 8, 9, 10, 16, 18]);
}

#[test]
fn strided_differing_line_size_overflow_keeps_preparation_error() {
    let mut destination = [];
    let mut source_line = [0; 2];
    let mut destination_line = [0; 1];
    let result = for_each_strided_line_differing::<i32, i32, FftError>(
        &[],
        &mut destination,
        &mut source_line,
        &mut destination_line,
        1,
        usize::MAX,
        |_, _| Ok(()),
    );
    assert!(matches!(result, Err(FftError::PreparationFailed)));
}

#[test]
fn r2c_storage_requirements_keep_contiguous_strided_and_empty_capacities() {
    assert_eq!(
        r2c::in_place_storage_requirements::<f64, _, _>([7].into_iter(), [3].into_iter(), 2, 1)
            .unwrap(),
        (14, 6, 112, 7)
    );
    assert_eq!(
        r2c::in_place_storage_requirements::<f64, _, _>([7].into_iter(), [3].into_iter(), 2, 2)
            .unwrap(),
        (14, 6, 224, 14)
    );
    assert_eq!(
        r2c::in_place_storage_requirements::<f64, _, _>([7].into_iter(), [3].into_iter(), 0, 2,)
            .unwrap(),
        (0, 0, 0, 0)
    );
    assert!(matches!(
        r2c::in_place_storage_requirements::<f64, _, _>(
            [usize::MAX].into_iter(),
            std::iter::empty(),
            2,
            1,
        ),
        Err(FftError::PreparationFailed)
    ));
}

#[test]
fn r2r_workspace_buffer_allocator_preserves_typed_lengths_and_zeroes() {
    for (send, receive, embedding, scratch, line) in [
        (2, 3, 4, 5, 6),
        (0, 0, 0, 0, 0),
        (0, 3, 0, 5, 0),
        (2, 0, 4, 0, 6),
    ] {
        let buffers =
            allocate_r2r_workspace_buffers::<f64>(send, receive, embedding, scratch, line).unwrap();
        let lengths = [
            buffers.transpose.send_buffer.len(),
            buffers.transpose.receive_buffer.len(),
            buffers.embedding_line.len(),
            buffers.fft_scratch.len(),
            buffers.line_buffer.len(),
        ];
        let capacities = [
            buffers.transpose.send_buffer.capacity(),
            buffers.transpose.receive_buffer.capacity(),
            buffers.embedding_line.capacity(),
            buffers.fft_scratch.capacity(),
            buffers.line_buffer.capacity(),
        ];
        let requested = [send, receive, embedding, scratch, line];
        assert_eq!(lengths, requested);
        for (capacity, length) in capacities.into_iter().zip(requested) {
            if length == 0 {
                assert_eq!(capacity, 0);
            } else {
                assert!(capacity >= length);
            }
        }
        assert!(
            buffers
                .transpose
                .send_buffer
                .iter()
                .all(|&value| value == 0.0)
        );
        assert!(
            buffers
                .transpose
                .receive_buffer
                .iter()
                .all(|&value| value == 0.0)
        );
        assert!(
            buffers
                .embedding_line
                .iter()
                .all(|&value| value == Complex::new(0.0, 0.0))
        );
        assert!(
            buffers
                .fft_scratch
                .iter()
                .all(|&value| value == Complex::new(0.0, 0.0))
        );
        assert!(buffers.line_buffer.iter().all(|&value| value == 0.0));

        let storage = [
            (send, buffers.transpose.send_buffer.as_ptr().cast::<()>()),
            (
                receive,
                buffers.transpose.receive_buffer.as_ptr().cast::<()>(),
            ),
            (embedding, buffers.embedding_line.as_ptr().cast::<()>()),
            (scratch, buffers.fft_scratch.as_ptr().cast::<()>()),
            (line, buffers.line_buffer.as_ptr().cast::<()>()),
        ];
        for left in 0..storage.len() {
            for right in (left + 1)..storage.len() {
                if storage[left].0 > 0 && storage[right].0 > 0 {
                    assert_ne!(storage[left].1, storage[right].1);
                }
            }
        }
    }
}
