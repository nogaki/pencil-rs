use mpi::traits::*;
use pencil_array::MpiTopology;

#[test]
fn cartesian_topology_maps_all_coordinates() {
    let universe = mpi::initialize().expect("MPI initialization failed");
    let world = universe.world();
    assert_eq!(world.size(), 4, "run this test with mpiexec -n 4");

    let topology = MpiTopology::<2>::new(&world, [2, 2]).unwrap();
    assert_eq!(topology.process_grid(), &[2, 2]);
    assert_eq!(topology.size(), 4);

    for i in 0..2 {
        for j in 0..2 {
            let rank = topology.rank_at([i, j]).unwrap();
            assert!((0..4).contains(&rank));
        }
    }
}
