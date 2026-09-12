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
        assert_eq!(
            topology.subcommunicator_size(axis).unwrap(),
            expected_size
        );
    }

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
