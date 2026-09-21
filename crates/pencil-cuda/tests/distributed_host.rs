#![cfg(feature = "distributed")]

use std::sync::Arc;

use mpi::traits::*;
use pencil_array::{ExtraShape, MpiTopology, Pencil};
use pencil_cuda::distributed::{DistributedPlan, Error};
use pencil_fft::{
    AxisSelection, DistributedLayout, FourierDirection, FourierDirections, TransposeMethod,
};

#[test]
fn distributed_constructor_preflight_is_collective_without_cuda() {
    let universe = mpi::initialize().expect("MPI initialization failed");
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();
    let topology = MpiTopology::<1>::new(&world, [size as usize]).unwrap();
    let input = Pencil::<2, 1>::new(Arc::clone(&topology), [8, 8], [0]).unwrap();
    let extra = ExtraShape::scalar();
    let forward = FourierDirections::forward();
    let layout = DistributedLayout::default();
    let ordinal = if rank == 0 { usize::MAX } else { 0 };

    let call = |real: bool,
                selection: AxisSelection<2>,
                layout: DistributedLayout,
                signs: FourierDirections<2>,
                extra: ExtraShape| {
        if real {
            DistributedPlan::<f32, 2, 1>::r2c(Arc::clone(&input), extra, selection, layout, ordinal)
        } else {
            DistributedPlan::<f32, 2, 1>::c2c(
                Arc::clone(&input),
                extra,
                selection,
                layout,
                signs,
                ordinal,
            )
        }
    };

    // The constructor header is collective even when ranks select C2C/R2C differently.
    let cross_kind = call(
        rank != 0,
        AxisSelection::all(),
        layout,
        forward,
        extra.clone(),
    );
    if size > 1 {
        assert!(matches!(cross_kind, Err(Error::Descriptor)));
    } else {
        assert!(cross_kind.is_err());
    }

    for (selection, differing_layout, differing_signs, differing_extra) in [
        (
            if rank == 0 {
                AxisSelection::all()
            } else {
                AxisSelection::empty()
            },
            layout,
            forward,
            extra.clone(),
        ),
        (
            AxisSelection::all(),
            if rank == 0 {
                DistributedLayout {
                    transpose_method: TransposeMethod::AllToAllv,
                    permute_dims: true,
                }
            } else {
                DistributedLayout {
                    transpose_method: TransposeMethod::PointToPoint,
                    permute_dims: false,
                }
            },
            forward,
            extra.clone(),
        ),
        (
            AxisSelection::all(),
            layout,
            if rank == 0 {
                forward
            } else {
                FourierDirections::new([FourierDirection::Backward; 2])
            },
            extra.clone(),
        ),
        (
            AxisSelection::all(),
            layout,
            forward,
            if rank == 0 {
                ExtraShape::new([2]).unwrap()
            } else {
                ExtraShape::new([3]).unwrap()
            },
        ),
    ] {
        let result = call(
            false,
            selection,
            differing_layout,
            differing_signs,
            differing_extra,
        );
        if size > 1 {
            assert!(matches!(result, Err(Error::Descriptor)));
        } else {
            // One rank cannot create a descriptor mismatch; the invalid ordinal still must fail.
            assert!(result.is_err());
        }
    }

    // Shape, memory permutation, and precision are exact descriptor words,
    // not hashes or local ranges (which legitimately differ by rank).
    for permuted in [false, true] {
        let mut pencil = Pencil::<2, 1>::new(
            topology.clone(),
            if !permuted && rank != 0 {
                [9, 8]
            } else {
                [8, 8]
            },
            [0],
        )
        .unwrap();
        if permuted && rank != 0 {
            pencil = pencil
                .with_permutation(pencil_array::AxisPermutation::new([1, 0]).unwrap())
                .unwrap();
        }
        let result = DistributedPlan::<f32, 2, 1>::c2c(
            pencil,
            extra.clone(),
            AxisSelection::all(),
            layout,
            forward,
            ordinal,
        );
        if size > 1 {
            assert!(matches!(result, Err(Error::Descriptor)));
        } else {
            assert!(result.is_err());
        }
    }
    let typed = if rank == 0 {
        call(false, AxisSelection::all(), layout, forward, extra.clone()).map(|_| ())
    } else {
        DistributedPlan::<f64, 2, 1>::c2c(
            input.clone(),
            extra.clone(),
            AxisSelection::all(),
            layout,
            forward,
            ordinal,
        )
        .map(|_| ())
    };
    if size > 1 {
        assert!(matches!(typed, Err(Error::Descriptor)));
    } else {
        assert!(typed.is_err());
    }

    // A failed collective must leave the communicator usable for both constructors.
    let c2c_retry = call(false, AxisSelection::all(), layout, forward, extra.clone());
    assert!(matches!(c2c_retry, Err(Error::Cuda(_) | Error::Peer)));
    let r2c_retry = call(true, AxisSelection::all(), layout, forward, extra);
    assert!(matches!(r2c_retry, Err(Error::Cuda(_) | Error::Peer)));
    if rank == 0 {
        println!("CUDA_DISTRIBUTED_HOST_OK ranks={size}");
    }
}
