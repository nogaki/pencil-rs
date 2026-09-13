use pencil_array::MpiTopology;

fn expose_communicators(topology: &MpiTopology<2>) {
    let _ = topology.cartesian();
    let _ = topology.subcommunicator(0);
}

fn main() {}
