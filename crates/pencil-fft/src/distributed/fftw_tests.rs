use std::{cell::Cell, sync::Arc, time::Duration};

use crate::{
    AxisR2rKind, AxisTransform, BackendInitError, BackendKind, C2cPlan, DhtPlan, FftwError,
    MixedC2cPlan, MixedR2cPlan, PlanOptions, PlanningRigor, R2cPlan, R2rKind, R2rPlan,
};
use mpi::traits::*;
use pencil_array::{ExtraShape, MpiTopology};

thread_local! {
    static FACTORY_COUNT: Cell<usize> = const { Cell::new(0) };
    static FAIL_NEXT: Cell<bool> = const { Cell::new(false) };
}

pub(crate) fn before_factory() -> Result<(), FftwError> {
    FACTORY_COUNT.with(|count| count.set(count.get() + 1));
    if FAIL_NEXT.with(|fail| fail.replace(false)) {
        Err(FftwError::Load("injected consumer factory failure".into()))
    } else {
        Ok(())
    }
}

fn count() -> usize {
    FACTORY_COUNT.with(Cell::get)
}
fn reset() {
    FACTORY_COUNT.with(|count| count.set(0));
    FAIL_NEXT.with(|fail| fail.set(false));
}
fn opts(rigor: PlanningRigor, limit: Option<Duration>) -> PlanOptions {
    PlanOptions::new(rigor, limit).unwrap()
}

#[test]
fn native_operation_words_do_not_collide() {
    let mut words: std::collections::BTreeMap<u64, &str> = (121..=126)
        .map(|word| (word, "collections reservation"))
        .collect();
    for source in [
        include_str!("../distributed.rs"),
        include_str!("r2c.rs"),
        include_str!("r2r.rs"),
        include_str!("mixed.rs"),
    ] {
        for line in source
            .lines()
            .filter(|line| line.starts_with("const OPERATION_"))
        {
            let (name, value) = line.split_once(" = ").expect("literal operation word");
            let word: u64 = value.trim_end_matches(';').parse().unwrap();
            assert!(
                words.insert(word, name).is_none(),
                "operation word collision: {line}"
            );
        }
    }
    for word in 127..=132 {
        let name = words.get(&word).expect("native family reservation");
        assert!(name.contains("NATIVE"));
    }
}

#[test]
#[ignore = "requires explicit native FFTW MPI run"]
fn fftw_descriptor_and_factory_preflight_all_six() {
    let _lock = super::MPI_TEST_LOCK
        .get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap();
    let universe = mpi::initialize().expect("MPI initialization failed");
    let world = universe.world();
    let size = usize::try_from(world.size()).unwrap();
    let topology = MpiTopology::<1>::new(&world, [size]).unwrap();
    let extra = ExtraShape::scalar();
    let shape = [4 * size, 3];
    let rank = world.rank();
    let native = opts(PlanningRigor::Estimate, Some(Duration::new(2, 17)));

    // Check each counter immediately: constructing an array of results first
    // would reset counters before the assertions and hide premature planning.
    if size > 1 {
        for case in 0..6 {
            reset();
            let result = if case == 0 && rank != 0 {
                C2cPlan::<f64, 2, 1>::from_shape(Arc::clone(&topology), shape, extra.clone())
                    .map(|_| ())
                    .map_err(|e| format!("{e:?}"))
            } else if case == 1 && rank == 0 {
                C2cPlan::<f32, 2, 1>::from_shape_with_fftw(
                    Arc::clone(&topology),
                    shape,
                    extra.clone(),
                    native,
                )
                .map(|_| ())
                .map_err(|e| format!("{e:?}"))
            } else {
                let peer = match case {
                    2 => opts(PlanningRigor::Measure, Some(Duration::new(2, 17))),
                    3 => opts(PlanningRigor::Estimate, None),
                    4 => opts(PlanningRigor::Estimate, Some(Duration::new(2, 18))),
                    5 => opts(PlanningRigor::Estimate, Some(Duration::new(3, 17))),
                    _ => native,
                };
                C2cPlan::<f64, 2, 1>::from_shape_with_fftw(
                    Arc::clone(&topology),
                    shape,
                    extra.clone(),
                    if rank == 0 { native } else { peer },
                )
                .map(|_| ())
                .map_err(|e| format!("{e:?}"))
            };
            assert!(
                result.is_err(),
                "descriptor case {case} unexpectedly agreed"
            );
            assert_eq!(
                count(),
                0,
                "case {case} planned before descriptor agreement"
            );
        }
    }
    eprintln!(
        "FFTW runtime f32: {}",
        crate::runtime_version::<f32>().unwrap()
    );
    eprintln!(
        "FFTW runtime f64: {}",
        crate::runtime_version::<f64>().unwrap()
    );

    macro_rules! injected {
        ($make:expr) => {{
            reset();
            FAIL_NEXT.with(|fail| fail.set(rank == 0));
            let result = $make;
            if rank == 0 {
                assert!(
                    matches!(result, Err(BackendInitError::Native(FftwError::Load(_)))),
                    "{result:?}"
                );
            } else {
                assert!(
                    matches!(result, Err(BackendInitError::PeerPreflight)),
                    "{result:?}"
                );
            }
            assert!(count() > 0);
            reset();
            let recovered = $make.unwrap();
            assert_eq!(recovered.backend_kind(), BackendKind::Fftw);
            assert!(count() > 0);
        }};
    }
    injected!(C2cPlan::<f64, 2, 1>::from_shape_with_fftw(
        Arc::clone(&topology),
        shape,
        extra.clone(),
        native
    ));
    injected!(R2cPlan::<f64, 2, 1>::from_shape_with_fftw(
        Arc::clone(&topology),
        shape,
        extra.clone(),
        native
    ));
    injected!(R2rPlan::<f64, 2, 1>::from_shape_with_fftw(
        Arc::clone(&topology),
        shape,
        extra.clone(),
        [Some(R2rKind::DctII); 2],
        native
    ));
    injected!(DhtPlan::<f64, 2, 1>::from_shape_with_fftw(
        Arc::clone(&topology),
        shape,
        extra.clone(),
        native
    ));
    injected!(MixedC2cPlan::<f64, 2, 1>::from_shape_with_fftw(
        Arc::clone(&topology),
        shape,
        extra.clone(),
        [
            AxisTransform::R2r(AxisR2rKind::Dht),
            AxisTransform::R2r(AxisR2rKind::Fftw(R2rKind::DctII))
        ],
        native
    ));
    injected!(MixedR2cPlan::<f64, 2, 1>::from_shape_with_fftw(
        Arc::clone(&topology),
        shape,
        extra.clone(),
        [AxisTransform::R2r(AxisR2rKind::Dht), AxisTransform::Rfft],
        native
    ));
    if rank == 0 {
        println!("PASSED fftw_descriptor_and_factory_preflight_all_six ranks={size}");
    }
}
