use std::sync::Arc;

use mpi::traits::{Communicator, CommunicatorCollectives};
use pencil_array::{ExtraShape, MpiTopology, PencilArray};
use pencil_fft::{
    AxisR2rKind, AxisTransform, C2cPlan, CollectionError, Complex, DhtPlan, DistributedLayout,
    FourierDirection, FourierDirections, MixedC2cPlan, MixedR2cPlan, R2cPlan, R2rKind, R2rPlan,
    TransposeMethod,
};

pub(super) fn indexed<E: std::fmt::Debug>(result: Result<(), CollectionError<E>>, expected: usize) {
    match result.unwrap_err() {
        CollectionError::Member { index, .. } | CollectionError::PeerPreflight { index } => {
            assert_eq!(index, expected)
        }
        other => panic!("expected indexed preflight failure, got {other:?}"),
    }
}

const SHAPE: [usize; 2] = [4, 3];
const EXTRA: [usize; 1] = [1];

fn layout() -> DistributedLayout {
    DistributedLayout {
        transpose_method: TransposeMethod::AllToAllv,
        permute_dims: false,
    }
}

fn c2c(topology: &Arc<MpiTopology<1>>, shape: [usize; 2]) -> C2cPlan<f64, 2, 1> {
    C2cPlan::from_shape_with_layout(
        Arc::clone(topology),
        shape,
        ExtraShape::new(EXTRA.to_vec()).unwrap(),
        layout(),
    )
    .unwrap()
}

fn c2c32(topology: &Arc<MpiTopology<1>>) -> C2cPlan<f32, 2, 1> {
    C2cPlan::from_shape_with_layout(
        Arc::clone(topology),
        SHAPE,
        ExtraShape::new(EXTRA.to_vec()).unwrap(),
        layout(),
    )
    .unwrap()
}

fn r2c(topology: &Arc<MpiTopology<1>>, shape: [usize; 2]) -> R2cPlan<f64, 2, 1> {
    R2cPlan::from_shape_with_layout(
        Arc::clone(topology),
        shape,
        ExtraShape::new(EXTRA.to_vec()).unwrap(),
        layout(),
    )
    .unwrap()
}

fn fill_c(a: &mut PencilArray<Complex<f64>, 2, 1>, value: f64) {
    let n = a.pencil().local_shape_logical();
    for i in 0..n[0] {
        for j in 0..n[1] {
            *a.get_local_mut(&[0], [i, j]).unwrap() =
                Complex::new(value + i as f64, -value - j as f64);
        }
    }
}

fn fill_r(a: &mut PencilArray<f64, 2, 1>, value: f64) {
    let n = a.pencil().local_shape_logical();
    for i in 0..n[0] {
        for j in 0..n[1] {
            *a.get_local_mut(&[0], [i, j]).unwrap() = value + i as f64 + j as f64;
        }
    }
}

fn c2c_oop_ip(topology: &Arc<MpiTopology<1>>) {
    let good = c2c(topology, SHAPE);
    let foreign = c2c(topology, [3, 4]);
    let rank = topology.communicator().rank();
    let size = topology.communicator().size() as usize;

    // Normal OOP collection, then the indexed last-member layout regression.
    let count = 2 + size % 2;
    let mut src = (0..count)
        .map(|i| {
            let mut a = good.allocate_input().unwrap();
            fill_c(&mut a, i as f64 + 1.0);
            a
        })
        .collect::<Vec<_>>();
    let mut dst = (0..count)
        .map(|_| good.allocate_output().unwrap())
        .collect::<Vec<_>>();
    let mut ws = good.allocate_out_of_place_workspace().unwrap();
    good.forward_many(&src, &mut dst, &mut ws).unwrap();
    let earlier = dst
        .iter()
        .map(|a| a.as_slice().to_vec())
        .collect::<Vec<_>>();
    if rank == 0 {
        src[count - 1] = foreign.allocate_input().unwrap();
    }
    indexed(good.forward_many(&src, &mut dst, &mut ws), count - 1);
    assert_eq!(
        dst.iter()
            .map(|a| a.as_slice().to_vec())
            .collect::<Vec<_>>(),
        earlier
    );

    src[count - 1] = good.allocate_input().unwrap();

    // A workspace belonging to another core is rejected before any output write.
    let mut dst = (0..count)
        .map(|_| good.allocate_output().unwrap())
        .collect::<Vec<_>>();
    dst[0].as_mut_slice().fill(Complex::new(77.0, -31.0));
    let before = dst
        .iter()
        .map(|a| a.as_slice().to_vec())
        .collect::<Vec<_>>();
    let mut wrong_ws = if rank == 0 {
        foreign.allocate_out_of_place_workspace().unwrap()
    } else {
        good.allocate_out_of_place_workspace().unwrap()
    };
    indexed(good.forward_many(&src, &mut dst, &mut wrong_ws), 0);
    assert_eq!(
        before,
        dst.iter()
            .map(|a| a.as_slice().to_vec())
            .collect::<Vec<_>>()
    );

    // IP state mismatch is indexed, leaves earlier inputs alone, and a fresh retry works.
    let mut arrays = (0..count)
        .map(|i| {
            let mut a = good.allocate_in_place().unwrap();
            a.view_mut()
                .unwrap()
                .as_mut_slice()
                .fill(Complex::new(i as f64 + 9.0, -3.0));
            a
        })
        .collect::<Vec<_>>();
    let mut ipws = good.allocate_in_place_workspace().unwrap();
    let before = arrays
        .iter()
        .map(|a| a.view().unwrap().as_slice().to_vec())
        .collect::<Vec<_>>();
    good.forward_in_place(&mut arrays[count - 1], &mut ipws)
        .unwrap();
    indexed(
        good.forward_many_in_place(&mut arrays, &mut ipws),
        count - 1,
    );
    assert_eq!(
        before[..count - 1],
        arrays[..count - 1]
            .iter()
            .map(|a| a.view().unwrap().as_slice().to_vec())
            .collect::<Vec<_>>()[..]
    );
    let mut retry = (0..count)
        .map(|i| {
            let mut a = good.allocate_in_place().unwrap();
            a.view_mut()
                .unwrap()
                .as_mut_slice()
                .fill(Complex::new(i as f64 + 19.0, -3.0));
            a
        })
        .collect::<Vec<_>>();
    good.forward_many_in_place(&mut retry, &mut ipws).unwrap();

    // Different counts, direction, and single-vs-collection are all header rejects.
    let one = vec![good.allocate_input().unwrap()];
    let mut two = vec![
        good.allocate_output().unwrap(),
        good.allocate_output().unwrap(),
    ];
    assert!(good.forward_many(&one, &mut two, &mut ws).is_err());
    let mut a = (0..2)
        .map(|_| good.allocate_input().unwrap())
        .collect::<Vec<_>>();
    let mut b = (0..2)
        .map(|_| good.allocate_output().unwrap())
        .collect::<Vec<_>>();
    if size > 1 {
        if rank % 2 == 0 {
            assert!(good.forward_many(&a, &mut b, &mut ws).is_err());
        } else {
            assert!(good.inverse_many(&b, &mut a, &mut ws).is_err());
        }
        let n = if rank == 0 { 1 } else { 2 };
        assert!(good.forward_many(&a[..n], &mut b[..n], &mut ws).is_err());
    }
    let single_src = good.allocate_input().unwrap();
    let mut single_dst = good.allocate_output().unwrap();
    let collection_src = vec![good.allocate_input().unwrap()];
    let mut collection_dst = vec![good.allocate_output().unwrap()];
    if size > 1 {
        if rank == 0 {
            assert!(good.forward(&single_src, &mut single_dst, &mut ws).is_err());
        } else {
            assert!(
                good.forward_many(&collection_src, &mut collection_dst, &mut ws)
                    .is_err()
            );
        }
    }
}

fn cross_headers(topology: &Arc<MpiTopology<1>>) {
    let comm = topology.communicator();
    if comm.size() == 1 {
        return;
    }
    let rank = comm.rank();
    let c = c2c(topology, SHAPE);
    let r = r2c(topology, SHAPE);
    let cs = vec![c.allocate_input().unwrap()];
    let mut cd = vec![c.allocate_output().unwrap()];
    let mut cw = c.allocate_out_of_place_workspace().unwrap();
    let rs = vec![r.allocate_input().unwrap()];
    let mut rd = vec![r.allocate_output().unwrap()];
    let mut rw = r.allocate_workspace().unwrap();
    if rank == 0 {
        assert!(c.forward_many(&cs, &mut cd, &mut cw).is_err());
    } else {
        assert!(r.forward_many(&rs, &mut rd, &mut rw).is_err());
    }

    let f32_plan = c2c32(topology);
    let f32s = vec![f32_plan.allocate_input().unwrap()];
    let mut f32d = vec![f32_plan.allocate_output().unwrap()];
    let mut f32w = f32_plan.allocate_out_of_place_workspace().unwrap();
    let f64s = vec![c.allocate_input().unwrap()];
    let mut f64d = vec![c.allocate_output().unwrap()];
    if rank == 0 {
        assert!(f32_plan.forward_many(&f32s, &mut f32d, &mut f32w).is_err());
    } else {
        assert!(c.forward_many(&f64s, &mut f64d, &mut cw).is_err());
    }
}

fn r2c_oop_ip(topology: &Arc<MpiTopology<1>>) {
    let plan = r2c(topology, SHAPE);
    let count = 2;
    let mut src = (0..count)
        .map(|i| {
            let mut a = plan.allocate_input().unwrap();
            fill_r(&mut a, i as f64 + 2.0);
            a
        })
        .collect::<Vec<_>>();
    let mut dst = (0..count)
        .map(|_| plan.allocate_output().unwrap())
        .collect::<Vec<_>>();
    let mut ws = plan.allocate_workspace().unwrap();
    plan.forward_many(&src, &mut dst, &mut ws).unwrap();
    let destination_before = dst
        .iter()
        .map(|a| a.as_slice().to_vec())
        .collect::<Vec<_>>();
    plan.inverse_many(&dst, &mut src, &mut ws).unwrap();
    assert_eq!(
        destination_before,
        dst.iter()
            .map(|a| a.as_slice().to_vec())
            .collect::<Vec<_>>()
    );

    let mut arrays = (0..count)
        .map(|i| {
            let mut a = plan.allocate_in_place().unwrap();
            a.real_view_mut()
                .unwrap()
                .as_mut_slice()
                .fill(i as f64 + 4.0);
            a
        })
        .collect::<Vec<_>>();
    let mut ipws = plan.allocate_in_place_workspace().unwrap();
    let before = arrays[0].real_view().unwrap().as_slice().to_vec();
    plan.forward_in_place(&mut arrays[1], &mut ipws).unwrap();
    indexed(plan.forward_many_in_place(&mut arrays, &mut ipws), 1);
    assert_eq!(before, arrays[0].real_view().unwrap().as_slice());
    plan.inverse_in_place(&mut arrays[1], &mut ipws).unwrap();
    plan.forward_many_in_place(&mut arrays, &mut ipws).unwrap();
    plan.inverse_many_in_place(&mut arrays, &mut ipws).unwrap();

    let foreign = r2c(topology, [3, 4]);
    if topology.communicator().rank() == 0 {
        dst[1] = foreign.allocate_output().unwrap();
    }
    let before = dst
        .iter()
        .map(|a| a.as_slice().to_vec())
        .collect::<Vec<_>>();
    indexed(plan.forward_many(&src, &mut dst, &mut ws), 1);
    assert_eq!(
        before,
        dst.iter()
            .map(|a| a.as_slice().to_vec())
            .collect::<Vec<_>>()
    );
    dst[1] = plan.allocate_output().unwrap();
    let before = dst
        .iter()
        .map(|a| a.as_slice().to_vec())
        .collect::<Vec<_>>();
    let mut wrong_ws = foreign.allocate_workspace().unwrap();
    indexed(plan.forward_many(&src, &mut dst, &mut wrong_ws), 0);
    assert_eq!(
        before,
        dst.iter()
            .map(|a| a.as_slice().to_vec())
            .collect::<Vec<_>>()
    );
}

fn operation_mismatch(topology: &Arc<MpiTopology<1>>) {
    if topology.communicator().size() == 1 {
        return;
    }
    let plan = c2c(topology, SHAPE);
    let rank = topology.communicator().rank();
    let source = vec![plan.allocate_input().unwrap()];
    let mut destination = vec![plan.allocate_output().unwrap()];
    destination[0].as_mut_slice().fill(Complex::new(31.0, -7.0));
    let before = destination[0].as_slice().to_vec();
    let source_before = source[0].as_slice().to_vec();
    let mut ws = plan.allocate_out_of_place_workspace().unwrap();
    let mut arrays = vec![plan.allocate_in_place().unwrap()];
    arrays[0]
        .view_mut()
        .unwrap()
        .as_mut_slice()
        .fill(Complex::new(3.0, 2.0));
    let ip_before = arrays[0].view().unwrap().as_slice().to_vec();
    let mut ipws = plan.allocate_in_place_workspace().unwrap();
    let ws_before = format!("{ws:?}");
    let ipws_before = format!("{ipws:?}");
    if rank == 0 {
        assert!(matches!(
            plan.forward_many(&source, &mut destination, &mut ws),
            Err(CollectionError::HeaderMismatch)
        ));
    } else {
        assert!(matches!(
            plan.forward_many_in_place(&mut arrays, &mut ipws),
            Err(CollectionError::HeaderMismatch)
        ));
    }
    assert_eq!(before, destination[0].as_slice());
    assert_eq!(source_before, source[0].as_slice());
    assert_eq!(ip_before, arrays[0].view().unwrap().as_slice());
    assert_eq!(ws_before, format!("{ws:?}"));
    assert_eq!(ipws_before, format!("{ipws:?}"));
    // Coordinated retries reuse both workspaces, including each rejected one.
    plan.forward_many(&source, &mut destination, &mut ws)
        .unwrap();
    plan.forward_many_in_place(&mut arrays, &mut ipws).unwrap();
}

fn descriptor_case(
    topology: &Arc<MpiTopology<1>>,
    baseline: C2cPlan<f64, 2, 1>,
    alternative: C2cPlan<f64, 2, 1>,
) {
    let plans = [baseline, alternative];
    let mut sources = plans.each_ref().map(|p| vec![p.allocate_input().unwrap()]);
    let mut destinations = plans.each_ref().map(|p| vec![p.allocate_output().unwrap()]);
    let mut workspaces = plans
        .each_ref()
        .map(|p| p.allocate_out_of_place_workspace().unwrap());
    for i in 0..2 {
        sources[i][0].as_mut_slice().fill(Complex::new(3.0, 7.0));
        destinations[i][0]
            .as_mut_slice()
            .fill(Complex::new(13.0, -2.0));
    }
    let before_s = sources.each_ref().map(|s| s[0].as_slice().to_vec());
    let before_d = destinations.each_ref().map(|d| d[0].as_slice().to_vec());
    let selected = usize::from(topology.communicator().rank() != 0);
    let ws_before = format!("{:?}", workspaces[selected]);
    assert!(matches!(
        plans[selected].forward_many(
            &sources[selected],
            &mut destinations[selected],
            &mut workspaces[selected]
        ),
        Err(CollectionError::HeaderMismatch)
    ));
    assert_eq!(ws_before, format!("{:?}", workspaces[selected]));
    for i in 0..2 {
        assert_eq!(before_s[i], sources[i][0].as_slice());
        assert_eq!(before_d[i], destinations[i][0].as_slice());
        plans[i]
            .forward_many(&sources[i], &mut destinations[i], &mut workspaces[i])
            .unwrap();
    }
}

fn descriptor_mismatch(topology: &Arc<MpiTopology<1>>) {
    if topology.communicator().size() == 1 {
        return;
    }
    let make = |shape, plan_layout, directions| {
        C2cPlan::<f64, 2, 1>::from_shape_with_layout(
            Arc::clone(topology),
            shape,
            ExtraShape::new(EXTRA.to_vec()).unwrap(),
            plan_layout,
        )
        .unwrap()
        .with_fft_directions(directions)
        .unwrap()
    };
    let forward = FourierDirections::new([FourierDirection::Forward; 2]);
    let backward = FourierDirections::new([FourierDirection::Forward, FourierDirection::Backward]);
    // Construct both sides identically on every rank; only the plan used for
    // the collective is selected by rank.
    descriptor_case(
        topology,
        make(SHAPE, layout(), forward),
        make(
            SHAPE,
            DistributedLayout {
                transpose_method: TransposeMethod::PointToPoint,
                ..layout()
            },
            forward,
        ),
    );
    descriptor_case(
        topology,
        make(SHAPE, layout(), forward),
        make(
            SHAPE,
            DistributedLayout {
                permute_dims: true,
                ..layout()
            },
            forward,
        ),
    );
    descriptor_case(
        topology,
        make(SHAPE, layout(), forward),
        make(SHAPE, layout(), backward),
    );
    descriptor_case(
        topology,
        make(SHAPE, layout(), forward),
        make([3, 4], layout(), forward),
    );
    topology.communicator().barrier();
}

macro_rules! peer_preflight_ip_family {
    ($topology:expr, $view:ident, $view_mut:ident, $good:expr, $foreign:expr) => {{
        let good = $good;
        let foreign = $foreign;
        let rank = $topology.communicator().rank();
        let mut arrays = vec![
            good.allocate_in_place().unwrap(),
            good.allocate_in_place().unwrap(),
        ];
        let old = if rank == 0 {
            std::mem::replace(&mut arrays[1], foreign.allocate_in_place().unwrap())
        } else {
            good.allocate_in_place().unwrap()
        };
        for array in &mut arrays {
            array.$view_mut().unwrap().as_mut_slice().fill(7.0.into());
        }
        let before = arrays
            .iter()
            .map(|a| a.$view().unwrap().as_slice().to_vec())
            .collect::<Vec<_>>();
        let mut ws = good.allocate_in_place_workspace().unwrap();
        let ws_before = format!("{ws:?}");
        indexed(good.forward_many_in_place(&mut arrays, &mut ws), 1);
        assert_eq!(ws_before, format!("{ws:?}"));
        assert_eq!(
            before,
            arrays
                .iter()
                .map(|a| a.$view().unwrap().as_slice().to_vec())
                .collect::<Vec<_>>()
        );
        arrays[1] = old;
        let mut bad_ws = if rank == 0 {
            foreign.allocate_in_place_workspace().unwrap()
        } else {
            good.allocate_in_place_workspace().unwrap()
        };
        let before = arrays
            .iter()
            .map(|a| a.$view().unwrap().as_slice().to_vec())
            .collect::<Vec<_>>();
        let ws_before = format!("{bad_ws:?}");
        indexed(good.forward_many_in_place(&mut arrays, &mut bad_ws), 0);
        assert_eq!(ws_before, format!("{bad_ws:?}"));
        assert_eq!(
            before,
            arrays
                .iter()
                .map(|a| a.$view().unwrap().as_slice().to_vec())
                .collect::<Vec<_>>()
        );
        good.forward_many_in_place(&mut arrays, &mut ws).unwrap();
        // Retry each rejected workspace with its owning plan, in identical order.
        let mut foreign_arrays = vec![foreign.allocate_in_place().unwrap()];
        let mut foreign_ws = foreign.allocate_in_place_workspace().unwrap();
        if rank == 0 {
            std::mem::swap(&mut bad_ws, &mut foreign_ws);
        }
        foreign
            .forward_many_in_place(&mut foreign_arrays, &mut foreign_ws)
            .unwrap();
        good.inverse_many_in_place(&mut arrays, &mut ws).unwrap();
        if rank != 0 {
            std::mem::swap(&mut bad_ws, &mut ws);
        }
        good.forward_many_in_place(&mut arrays, &mut ws).unwrap();
    }};
}

macro_rules! peer_preflight_family {
    ($topology:expr, $good:expr, $foreign:expr) => {{
        let good = $good;
        let foreign = $foreign;
        let rank = $topology.communicator().rank();
        let mut source = vec![
            good.allocate_input().unwrap(),
            good.allocate_input().unwrap(),
        ];
        for a in &mut source {
            a.as_mut_slice().fill(7.0.into());
        }
        let mut destination = vec![
            good.allocate_output().unwrap(),
            good.allocate_output().unwrap(),
        ];
        let source_before = source
            .iter()
            .map(|a| a.as_slice().to_vec())
            .collect::<Vec<_>>();
        let old = if rank == 0 {
            std::mem::replace(&mut destination[1], foreign.allocate_output().unwrap())
        } else {
            good.allocate_output().unwrap()
        };
        for a in &mut destination {
            a.as_mut_slice().fill(31.0.into());
        }
        let before = destination
            .iter()
            .map(|a| a.as_slice().to_vec())
            .collect::<Vec<_>>();
        let mut ws = good.allocate_workspace().unwrap();
        let ws_before = format!("{ws:?}");
        indexed(good.forward_many(&source, &mut destination, &mut ws), 1);
        assert_eq!(ws_before, format!("{ws:?}"));
        assert_eq!(
            before,
            destination
                .iter()
                .map(|a| a.as_slice().to_vec())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            source_before,
            source
                .iter()
                .map(|a| a.as_slice().to_vec())
                .collect::<Vec<_>>()
        );
        destination[1] = old;
        let mut bad_ws = if rank == 0 {
            foreign.allocate_workspace().unwrap()
        } else {
            good.allocate_workspace().unwrap()
        };
        let before = destination
            .iter()
            .map(|a| a.as_slice().to_vec())
            .collect::<Vec<_>>();
        let ws_before = format!("{bad_ws:?}");
        indexed(good.forward_many(&source, &mut destination, &mut bad_ws), 0);
        assert_eq!(ws_before, format!("{bad_ws:?}"));
        assert_eq!(
            before,
            destination
                .iter()
                .map(|a| a.as_slice().to_vec())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            source_before,
            source
                .iter()
                .map(|a| a.as_slice().to_vec())
                .collect::<Vec<_>>()
        );
        good.forward_many(&source, &mut destination, &mut ws)
            .unwrap();
        let foreign_source = vec![foreign.allocate_input().unwrap()];
        let mut foreign_destination = vec![foreign.allocate_output().unwrap()];
        let mut foreign_ws = foreign.allocate_workspace().unwrap();
        if rank == 0 {
            std::mem::swap(&mut bad_ws, &mut foreign_ws);
        }
        foreign
            .forward_many(&foreign_source, &mut foreign_destination, &mut foreign_ws)
            .unwrap();
        if rank != 0 {
            std::mem::swap(&mut bad_ws, &mut ws);
        }
        good.forward_many(&source, &mut destination, &mut ws)
            .unwrap();
    }};
}

fn peer_preflight_families(topology: &Arc<MpiTopology<1>>) {
    if topology.communicator().size() == 1 {
        return;
    }
    let dct = AxisTransform::R2r(AxisR2rKind::Fftw(R2rKind::DctII));
    peer_preflight_family!(
        topology,
        R2rPlan::<f64, 2, 1>::from_shape_with_layout(
            Arc::clone(topology),
            SHAPE,
            ExtraShape::new(EXTRA.to_vec()).unwrap(),
            [Some(R2rKind::DctII); 2],
            layout()
        )
        .unwrap(),
        R2rPlan::<f64, 2, 1>::from_shape_with_layout(
            Arc::clone(topology),
            [3, 4],
            ExtraShape::new(EXTRA.to_vec()).unwrap(),
            [Some(R2rKind::DctII); 2],
            layout()
        )
        .unwrap()
    );
    peer_preflight_family!(
        topology,
        DhtPlan::<f64, 2, 1>::from_shape_with_layout(
            Arc::clone(topology),
            SHAPE,
            ExtraShape::new(EXTRA.to_vec()).unwrap(),
            layout()
        )
        .unwrap(),
        DhtPlan::<f64, 2, 1>::from_shape_with_layout(
            Arc::clone(topology),
            [3, 4],
            ExtraShape::new(EXTRA.to_vec()).unwrap(),
            layout()
        )
        .unwrap()
    );
    peer_preflight_family!(
        topology,
        MixedC2cPlan::<f64, 2, 1>::from_shape_with_layout(
            Arc::clone(topology),
            SHAPE,
            ExtraShape::new(EXTRA.to_vec()).unwrap(),
            [AxisTransform::Fft, dct],
            layout()
        )
        .unwrap(),
        MixedC2cPlan::<f64, 2, 1>::from_shape_with_layout(
            Arc::clone(topology),
            [3, 4],
            ExtraShape::new(EXTRA.to_vec()).unwrap(),
            [AxisTransform::Fft, dct],
            layout()
        )
        .unwrap()
    );
    peer_preflight_family!(
        topology,
        MixedR2cPlan::<f64, 2, 1>::from_shape_with_layout(
            Arc::clone(topology),
            SHAPE,
            ExtraShape::new(EXTRA.to_vec()).unwrap(),
            [AxisTransform::Rfft, dct],
            layout()
        )
        .unwrap(),
        MixedR2cPlan::<f64, 2, 1>::from_shape_with_layout(
            Arc::clone(topology),
            [3, 4],
            ExtraShape::new(EXTRA.to_vec()).unwrap(),
            [AxisTransform::Rfft, dct],
            layout()
        )
        .unwrap()
    );

    peer_preflight_ip_family!(
        topology,
        view,
        view_mut,
        R2rPlan::<f64, 2, 1>::from_shape_with_layout(
            Arc::clone(topology),
            SHAPE,
            ExtraShape::new(EXTRA.to_vec()).unwrap(),
            [Some(R2rKind::DctII); 2],
            layout()
        )
        .unwrap(),
        R2rPlan::<f64, 2, 1>::from_shape_with_layout(
            Arc::clone(topology),
            [3, 4],
            ExtraShape::new(EXTRA.to_vec()).unwrap(),
            [Some(R2rKind::DctII); 2],
            layout()
        )
        .unwrap()
    );
    peer_preflight_ip_family!(
        topology,
        view,
        view_mut,
        DhtPlan::<f64, 2, 1>::from_shape_with_layout(
            Arc::clone(topology),
            SHAPE,
            ExtraShape::new(EXTRA.to_vec()).unwrap(),
            layout()
        )
        .unwrap(),
        DhtPlan::<f64, 2, 1>::from_shape_with_layout(
            Arc::clone(topology),
            [3, 4],
            ExtraShape::new(EXTRA.to_vec()).unwrap(),
            layout()
        )
        .unwrap()
    );
    peer_preflight_ip_family!(
        topology,
        view,
        view_mut,
        MixedC2cPlan::<f64, 2, 1>::from_shape_with_layout(
            Arc::clone(topology),
            SHAPE,
            ExtraShape::new(EXTRA.to_vec()).unwrap(),
            [AxisTransform::Fft, dct],
            layout()
        )
        .unwrap(),
        MixedC2cPlan::<f64, 2, 1>::from_shape_with_layout(
            Arc::clone(topology),
            [3, 4],
            ExtraShape::new(EXTRA.to_vec()).unwrap(),
            [AxisTransform::Fft, dct],
            layout()
        )
        .unwrap()
    );
    peer_preflight_ip_family!(
        topology,
        real_view,
        real_view_mut,
        MixedR2cPlan::<f64, 2, 1>::from_shape_with_layout(
            Arc::clone(topology),
            SHAPE,
            ExtraShape::new(EXTRA.to_vec()).unwrap(),
            [AxisTransform::Rfft, dct],
            layout()
        )
        .unwrap(),
        MixedR2cPlan::<f64, 2, 1>::from_shape_with_layout(
            Arc::clone(topology),
            [3, 4],
            ExtraShape::new(EXTRA.to_vec()).unwrap(),
            [AxisTransform::Rfft, dct],
            layout()
        )
        .unwrap()
    );
}

pub fn run(topology: &Arc<MpiTopology<1>>) {
    operation_mismatch(topology);
    descriptor_mismatch(topology);
    peer_preflight_families(topology);
    c2c_oop_ip(topology);
    r2c_oop_ip(topology);
    let plan = c2c(topology, SHAPE);
    let empty_s = Vec::new();
    let mut empty_d = Vec::new();
    let mut empty_w = plan.allocate_out_of_place_workspace().unwrap();
    assert!(
        plan.forward_many(&empty_s, &mut empty_d, &mut empty_w)
            .is_err()
    );
    cross_headers(topology);
    topology.communicator().barrier();
}
