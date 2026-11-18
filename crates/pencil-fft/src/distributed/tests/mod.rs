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
    assert!(matches!(strided_line_count(0, 2, 0), Ok(0)));
    assert!(matches!(
        strided_line_count(1, 2, 0),
        Err(FftError::PreparationFailed)
    ));
    assert!(matches!(
        strided_line_count(0, 2, usize::MAX),
        Err(FftError::PreparationFailed)
    ));
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
    let beyond_isize = isize::MAX as usize / std::mem::size_of::<f64>() + 1;
    assert!(matches!(
        r2c::in_place_storage_requirements::<f64, _, _>(
            [beyond_isize].into_iter(),
            std::iter::empty(),
            1,
            1,
        ),
        Err(FftError::PreparationFailed)
    ));
    assert!(matches!(
        r2c::validate_recast_capacity::<f64, Complex<f64>>(usize::MAX, 0),
        Err(FftError::StorageLayoutMismatch)
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

#[test]
#[ignore = "MPI must be initialized in a fresh one-rank test process"]
fn strided_c2c_wrappers_and_r2c_registry_preserve_contracts() {
    let universe = mpi::initialize().expect("MPI initialization failed");
    let world = universe.world();
    assert_eq!(world.size(), 1, "run this test with one MPI rank");
    // MPI topology is intentionally !Send/!Sync; the public Pencil API uses Arc,
    // and the equivalent pencil below checks pointer-identity deduplication.
    #[allow(clippy::arc_with_non_send_sync)]
    let topology = Arc::new(MpiTopology::<1>::new(&world, [1]).unwrap());
    let contiguous = Pencil::<2, 1>::new_default(Arc::clone(&topology), [2, 3]).unwrap();
    let strided = contiguous
        .with_permutation(AxisPermutation::new([1, 0]).unwrap())
        .unwrap();
    let axis0_memory_position = strided
        .permutation()
        .axes()
        .iter()
        .position(|axis| axis.index() == 0)
        .unwrap();
    let expected_axis0_stride: usize = strided.local_shape_memory()[..axis0_memory_position]
        .iter()
        .product();
    assert_eq!(expected_axis0_stride, 3);
    let plan = LocalC2cPlan::<f64>::new(2).unwrap();
    let scratch_len = plan.scratch_len();
    let zero = Complex::new(0.0, 0.0);
    let source = vec![zero; 5];
    let mut destination = vec![zero; 6];
    let mut short_line = [zero; 1];
    let mut scratch = vec![zero; scratch_len];

    for result in [
        execute_strided_complex_forward(
            &plan,
            &strided,
            0,
            &source,
            &mut destination,
            &mut scratch,
            &mut short_line,
        ),
        execute_strided_complex_forward_in_place(
            &plan,
            &strided,
            0,
            &mut source.clone(),
            &mut scratch,
            &mut short_line,
        ),
        execute_strided_complex_reverse(
            &plan,
            &strided,
            0,
            &source,
            &mut destination,
            &mut scratch,
            &mut short_line,
            true,
        ),
        execute_strided_complex_reverse_in_place(
            &plan,
            &strided,
            0,
            &mut source.clone(),
            &mut scratch,
            &mut short_line,
            false,
        ),
    ] {
        assert!(matches!(
            result,
            Err(FftError::WorkspaceTooSmall {
                required: 2,
                actual: 1,
                ..
            })
        ));
    }

    let mut line = [zero; 2];
    assert!(matches!(
        execute_strided_complex_forward(
            &plan,
            &strided,
            0,
            &source,
            &mut destination,
            &mut scratch,
            &mut line,
        ),
        Err(FftError::LocalC2c(LocalC2cError::NonIntegralBatch))
    ));
    let source = vec![zero; 6];
    let mut short_destination = vec![zero; 5];
    assert!(matches!(
        execute_strided_complex_forward(
            &plan,
            &strided,
            0,
            &source,
            &mut short_destination,
            &mut scratch,
            &mut line,
        ),
        Err(FftError::LocalC2c(LocalC2cError::BufferLengthMismatch))
    ));
    assert!(matches!(
        execute_strided_complex_reverse(
            &plan,
            &strided,
            0,
            &source,
            &mut short_destination,
            &mut scratch,
            &mut line,
            false,
        ),
        Err(FftError::LocalC2c(LocalC2cError::BufferLengthMismatch))
    ));

    let source = [Complex::new(1.0, 0.5), Complex::new(-0.25, 0.75)];
    let mut destination = [zero; 2];
    let mut no_line = [];
    assert!(
        execute_strided_complex_forward(
            &plan,
            &contiguous,
            0,
            &source,
            &mut destination,
            &mut scratch,
            &mut no_line,
        )
        .is_ok()
    );
    let mut data = source;
    assert!(
        execute_strided_complex_forward_in_place(
            &plan,
            &contiguous,
            0,
            &mut data,
            &mut scratch,
            &mut no_line,
        )
        .is_ok()
    );
    assert!(
        execute_strided_complex_reverse(
            &plan,
            &contiguous,
            0,
            &source,
            &mut destination,
            &mut scratch,
            &mut no_line,
            true,
        )
        .is_ok()
    );
    assert!(
        execute_strided_complex_reverse_in_place(
            &plan,
            &contiguous,
            0,
            &mut data,
            &mut scratch,
            &mut no_line,
            false,
        )
        .is_ok()
    );

    let equivalent = Pencil::<2, 1>::new_default(Arc::clone(&topology), [2, 3]).unwrap();
    let alternate = strided;
    let registered =
        registered_pencils([&contiguous, &equivalent, &alternate].into_iter()).unwrap();
    assert_eq!(registered.len(), 2);
    assert!(Arc::ptr_eq(&registered[0], &contiguous));
    assert!(Arc::ptr_eq(&registered[1], &alternate));
}
