use mpi::traits::*;
use pencil_array::{MpiTopology, TopologyError};

#[test]
fn cartesian_topology_owns_and_maps_mpi_resources() {
    let universe = mpi::initialize().expect("MPI initialization failed");
    let world = universe.world();
    let world_size = usize::try_from(world.size()).unwrap();
    let process_grid = match world_size {
        1 => [1, 1],
        4 => [2, 2],
        other => panic!("run this test with mpiexec -n 1 or -n 4, got {other}"),
    };

    if world_size == 4 {
        let first_group = world.rank() < 2;
        let local = world
            .split_by_color(mpi::topology::Color::with_value(i32::from(!first_group)))
            .unwrap();
        let remote_leader = if first_group { 2 } else { 0 };
        let mut raw_intercommunicator = std::mem::MaybeUninit::uninit();
        // SAFETY: the two local communicators are disjoint and have local
        // leader 0. Their remote leaders are valid ranks in `world`; all four
        // ranks call with the same tag and a writable output communicator.
        let status = unsafe {
            mpi::ffi::MPI_Intercomm_create(
                local.as_raw(),
                0,
                world.as_raw(),
                remote_leader,
                42,
                raw_intercommunicator.as_mut_ptr(),
            )
        };
        assert_eq!(status, 0, "MPI_Intercomm_create failed");
        // SAFETY: the successful call initialized a live intercommunicator;
        // ownership is transferred exactly once and the raw handle is unused.
        let intercommunicator = unsafe {
            mpi::topology::InterCommunicator::from_raw(raw_intercommunicator.assume_init())
        };
        assert_eq!(
            MpiTopology::<1>::new(&intercommunicator, [2]).unwrap_err(),
            TopologyError::InterCommunicatorUnsupported
        );
        assert_eq!(
            MpiTopology::<1>::auto(&intercommunicator).unwrap_err(),
            TopologyError::InterCommunicatorUnsupported
        );
        world.barrier();
    }

    // A local validation failure must not leave peers entering MPI_Cart_create.
    let invalid_on_root = if world.rank() == 0 {
        [0, world_size]
    } else {
        process_grid
    };
    assert!(MpiTopology::<2>::new(&world, invalid_on_root).is_err());
    world.barrier();

    if world_size == 4 {
        // Both grids are individually valid, but MPI requires agreement.
        let disagreeing_grid = if world.rank() == 0 { [1, 4] } else { [2, 2] };
        assert!(MpiTopology::<2>::new(&world, disagreeing_grid).is_err());
        world.barrier();

        // The preflight itself must use the same count for different const M.
        let dimension_mismatch = if world.rank() == 0 {
            MpiTopology::<1>::new(&world, [4]).map(|_| ())
        } else {
            MpiTopology::<2>::new(&world, [2, 2]).map(|_| ())
        };
        assert!(dimension_mismatch.is_err());
        world.barrier();

        let auto_dimension_mismatch = if world.rank() == 0 {
            MpiTopology::<0>::auto(&world).map(|_| ())
        } else {
            MpiTopology::<2>::auto(&world).map(|_| ())
        };
        assert!(auto_dimension_mismatch.is_err());
        world.barrier();

        let auto_valid_dimension_mismatch = if world.rank() == 0 {
            MpiTopology::<1>::auto(&world).map(|_| ())
        } else {
            MpiTopology::<2>::auto(&world).map(|_| ())
        };
        assert!(auto_valid_dimension_mismatch.is_err());
        world.barrier();

        let cartesian = world
            .create_cartesian_communicator(&[2, 2], &[false; 2], false)
            .unwrap();
        let import_dimension_mismatch = if world.rank() == 0 {
            MpiTopology::<1>::from_cartesian(&cartesian).map(|_| ())
        } else {
            MpiTopology::<2>::from_cartesian(&cartesian).map(|_| ())
        };
        assert!(import_dimension_mismatch.is_err());
        world.barrier();
    }

    assert!(matches!(
        MpiTopology::<0>::new(&world, []),
        Err(TopologyError::ZeroDimensions)
    ));
    assert!(matches!(
        MpiTopology::<0>::auto(&world),
        Err(TopologyError::ZeroDimensions)
    ));
    assert!(matches!(
        MpiTopology::<2>::new(&world, [0, world_size]),
        Err(TopologyError::ZeroExtent { axis: 0 })
    ));
    assert!(matches!(
        MpiTopology::<1>::new(&world, [world_size + 1]),
        Err(TopologyError::CommunicatorSizeMismatch { .. })
    ));

    let topology = MpiTopology::<2>::new(&world, process_grid).unwrap();
    assert_eq!(topology.process_grid(), &process_grid);
    assert_eq!(topology.size(), world_size);
    assert_eq!(
        topology.rank_at(*topology.local_coords()).unwrap(),
        topology.rank()
    );

    for i in 0..process_grid[0] {
        for j in 0..process_grid[1] {
            let rank = topology.rank_at([i, j]).unwrap();
            assert!((0..world.size()).contains(&rank));
        }
    }

    assert_eq!(
        topology.rank_at([process_grid[0], 0]),
        Err(TopologyError::CoordinateOutOfBounds {
            axis: 0,
            coordinate: process_grid[0],
            extent: process_grid[0],
        })
    );

    for (axis, expected_size) in process_grid.into_iter().enumerate() {
        assert_eq!(topology.subcommunicator_size(axis).unwrap(), expected_size);
    }
    assert_eq!(
        topology.subcommunicator_size(2),
        Err(TopologyError::AxisOutOfBounds {
            axis: 2,
            dimensions: 2
        })
    );

    let automatic = MpiTopology::<1>::auto(&world).unwrap();
    assert_eq!(automatic.process_grid(), &[world_size]);

    let duplicate = {
        let dims = process_grid.map(|extent| i32::try_from(extent).unwrap());
        let periods = [false; 2];
        let cartesian = world
            .create_cartesian_communicator(&dims, &periods, false)
            .unwrap();
        MpiTopology::<2>::from_cartesian(&cartesian).unwrap()
    };
    assert_eq!(duplicate.process_grid(), &process_grid);
    assert_eq!(duplicate.size(), world_size);
    assert_eq!(
        duplicate.rank_at(*duplicate.local_coords()).unwrap(),
        duplicate.rank()
    );
}
