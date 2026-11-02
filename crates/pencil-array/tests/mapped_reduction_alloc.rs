use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::mem::{align_of, size_of};
use std::sync::Arc;

use mpi::collective::{CommunicatorCollectives, SystemOperation};
use mpi::topology::Communicator;
use pencil_array::{
    CollectiveError, ExtraShape, MpiTopology, Pencil, PencilArray, max_by, max_many_by, min_by,
    min_many_by, norm_by, norm_many_by, sum_by, sum_many_by, zip_norm_by, zip_sum_by,
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum InjectionMode {
    Observe,
    Fail,
}

#[derive(Clone, Copy)]
struct Injection {
    size: usize,
    align: usize,
    mode: InjectionMode,
    matched: usize,
    failed: usize,
}

thread_local! {
    static INJECTION: Cell<Option<Injection>> = const { Cell::new(None) };
}

// The allocator never allocates, formats, or panics: it only updates the TLS
// injection state and delegates to the system allocator.
fn should_fail(size: usize, align: usize) -> bool {
    INJECTION.with(|cell| {
        let Some(mut injection) = cell.get() else {
            return false;
        };
        if injection.size != size || injection.align != align {
            return false;
        }
        injection.matched += 1;
        let fail = injection.mode == InjectionMode::Fail && injection.failed == 0;
        if fail {
            injection.failed += 1;
        }
        cell.set(Some(injection));
        fail
    })
}

struct CountingAllocator;

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if should_fail(layout.size(), layout.align()) {
            return std::ptr::null_mut();
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if should_fail(layout.size(), layout.align()) {
            return std::ptr::null_mut();
        }
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if should_fail(new_size, layout.align()) {
            return std::ptr::null_mut();
        }
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

fn with_injection<T>(
    mode: InjectionMode,
    size: usize,
    align: usize,
    operation: impl FnOnce() -> T,
) -> (T, usize, usize) {
    INJECTION.with(|cell| {
        cell.set(Some(Injection {
            size,
            align,
            mode,
            matched: 0,
            failed: 0,
        }));
    });
    let result = operation();
    let injection = INJECTION.with(Cell::get).expect("injection state");
    INJECTION.with(|cell| cell.set(None));
    (result, injection.matched, injection.failed)
}

fn check_all<C: CommunicatorCollectives>(communicator: &C, local: bool) {
    let local_word = i32::from(local);
    let mut all_valid = 0i32;
    communicator.all_reduce_into(&local_word, &mut all_valid, SystemOperation::min());
    assert_eq!(all_valid, 1, "a rank failed the collective test check");
}

// Each call reserves partials with exactly P * size_of::<i16>() bytes and i16
// alignment; Observe verifies that this request matches only the partials
// reservation.
fn exercise_integer_partial_failure(
    communicator: &mpi::topology::CartesianCommunicator,
    rank: usize,
    size: usize,
    failing_rank: usize,
    local_output: usize,
    mut run: impl FnMut(&Cell<usize>) -> Result<i16, CollectiveError>,
) {
    let bytes = size * size_of::<i16>();
    let align = align_of::<i16>();

    // Observe: exactly one allocation request matches the partials vector.
    let calls = Cell::new(0usize);
    let (result, matched, _failed) =
        with_injection(InjectionMode::Observe, bytes, align, || run(&calls));
    check_all(
        communicator,
        result == Ok(3) && calls.get() == local_output && matched == 1,
    );

    // Fail: only the designated rank fails its local partials reservation.
    let calls = Cell::new(0usize);
    let mode = if rank == failing_rank {
        InjectionMode::Fail
    } else {
        InjectionMode::Observe
    };
    let (result, _matched, failed) = with_injection(mode, bytes, align, || run(&calls));
    let local_ok = calls.get() == 0
        && if rank == failing_rank {
            result == Err(CollectiveError::AllocationFailed { elements: size }) && failed == 1
        } else {
            result == Err(CollectiveError::CollectivePreconditionFailed) && failed == 0
        };
    check_all(communicator, local_ok);

    // Retry without injection recovers the collective sequence.
    let calls = Cell::new(0usize);
    let result = run(&calls);
    check_all(communicator, result == Ok(3) && calls.get() == local_output);
}

fn check_success_values(
    communicator: &mpi::topology::CartesianCommunicator,
    size: usize,
    topology: &Arc<MpiTopology<1>>,
    n: usize,
) {
    let pencil = Pencil::<1, 1>::new(Arc::clone(topology), [size * n], [0]).unwrap();
    check_all(communicator, pencil.local_len() == n);

    let x = PencilArray::from_elem(Arc::clone(&pencil), ExtraShape::scalar(), 1.0_f64).unwrap();
    let y = PencilArray::from_elem(Arc::clone(&pencil), ExtraShape::scalar(), 1.0_f64).unwrap();
    let expected_sum = (size * n) as f64;
    let expected_norm = expected_sum.sqrt();

    let calls = Cell::new(0usize);
    let result = sum_by(&x.view(), |value| {
        calls.set(calls.get() + 1);
        *value
    });
    check_all(communicator, result == Ok(expected_sum) && calls.get() == n);

    let calls = Cell::new(0usize);
    let result = norm_by(&x.view(), |value| {
        calls.set(calls.get() + 1);
        *value
    });
    let local_ok = match result {
        Ok(value) => (value / expected_norm - 1.0).abs() < 1.0e-12,
        Err(_) => false,
    } && calls.get() == n;
    check_all(communicator, local_ok);

    let calls = Cell::new(0usize);
    let result = min_by(&x.view(), |value| {
        calls.set(calls.get() + 1);
        *value
    });
    check_all(communicator, result == Ok(Some(1.0)) && calls.get() == n);

    let calls = Cell::new(0usize);
    let result = max_by(&x.view(), |value| {
        calls.set(calls.get() + 1);
        *value
    });
    check_all(communicator, result == Ok(Some(1.0)) && calls.get() == n);

    let calls = Cell::new(0usize);
    let result = zip_sum_by(&x.view(), &y.view(), |left, right| {
        calls.set(calls.get() + 1);
        left * right
    });
    check_all(communicator, result == Ok(expected_sum) && calls.get() == n);

    let calls = Cell::new(0usize);
    let result = zip_norm_by(&x.view(), &y.view(), |left, right| {
        calls.set(calls.get() + 1);
        left * right
    });
    let local_ok = match result {
        Ok(value) => (value / expected_norm - 1.0).abs() < 1.0e-12,
        Err(_) => false,
    } && calls.get() == n;
    check_all(communicator, local_ok);

    let calls = Cell::new(0usize);
    let result = sum_many_by(communicator, &[x.view(), y.view()], |values| {
        calls.set(calls.get() + 1);
        values[0] * values[1]
    });
    check_all(communicator, result == Ok(expected_sum) && calls.get() == n);

    let calls = Cell::new(0usize);
    let result = norm_many_by(communicator, &[x.view(), y.view()], |values| {
        calls.set(calls.get() + 1);
        values[0] * values[1]
    });
    let local_ok = match result {
        Ok(value) => (value / expected_norm - 1.0).abs() < 1.0e-12,
        Err(_) => false,
    } && calls.get() == n;
    check_all(communicator, local_ok);

    let calls = Cell::new(0usize);
    let result = min_many_by(communicator, &[x.view(), y.view()], |values| {
        calls.set(calls.get() + 1);
        values[0] * values[1]
    });
    check_all(communicator, result == Ok(Some(1.0)) && calls.get() == n);

    let calls = Cell::new(0usize);
    let result = max_many_by(communicator, &[x.view(), y.view()], |values| {
        calls.set(calls.get() + 1);
        values[0] * values[1]
    });
    check_all(communicator, result == Ok(Some(1.0)) && calls.get() == n);

    let xi = PencilArray::from_elem(Arc::clone(&pencil), ExtraShape::scalar(), 1_i64).unwrap();
    let yi = PencilArray::from_elem(Arc::clone(&pencil), ExtraShape::scalar(), 1_i64).unwrap();
    let expected_integer_sum = (size * n) as i64;

    let calls = Cell::new(0usize);
    let result = sum_by(&xi.view(), |value| {
        calls.set(calls.get() + 1);
        *value
    });
    check_all(
        communicator,
        result == Ok(expected_integer_sum) && calls.get() == n,
    );

    let calls = Cell::new(0usize);
    let result = zip_sum_by(&xi.view(), &yi.view(), |left, right| {
        calls.set(calls.get() + 1);
        left * right
    });
    check_all(
        communicator,
        result == Ok(expected_integer_sum) && calls.get() == n,
    );

    let calls = Cell::new(0usize);
    let result = sum_many_by(communicator, &[xi.view(), yi.view()], |values| {
        calls.set(calls.get() + 1);
        values[0] * values[1]
    });
    check_all(
        communicator,
        result == Ok(expected_integer_sum) && calls.get() == n,
    );
}

#[test]
fn mapped_reduction_alloc() {
    let universe = mpi::initialize().expect("MPI initialization failed");
    let world = universe.world();
    let size = usize::try_from(world.size()).unwrap();
    assert!(matches!(size, 1 | 4 | 6), "run with -n 1, -n 4, or -n 6");
    let rank = usize::try_from(world.rank()).unwrap();
    let topology = MpiTopology::<1>::new(&world, [size]).unwrap();
    let communicator = topology.communicator();

    // Three global points leave communicator rank zero empty for P = 4 and 6
    // under the floor partition, while P = 1 gives its only rank three points.
    let partial_pencil = Pencil::<1, 1>::new(Arc::clone(&topology), [3], [0]).unwrap();
    check_all(
        communicator,
        if size == 1 {
            partial_pencil.local_len() == 3
        } else {
            rank != 0 || partial_pencil.local_len() == 0
        },
    );

    let x =
        PencilArray::from_elem(Arc::clone(&partial_pencil), ExtraShape::scalar(), 1_i64).unwrap();
    let y =
        PencilArray::from_elem(Arc::clone(&partial_pencil), ExtraShape::scalar(), 1_i64).unwrap();
    let failing_rank = 0;
    let local_output = x.len();

    exercise_integer_partial_failure(
        communicator,
        rank,
        size,
        failing_rank,
        local_output,
        |calls| {
            sum_by(&x.view(), |value| {
                calls.set(calls.get() + 1);
                *value as i16
            })
        },
    );
    exercise_integer_partial_failure(
        communicator,
        rank,
        size,
        failing_rank,
        local_output,
        |calls| {
            zip_sum_by(&x.view(), &y.view(), |left, _| {
                calls.set(calls.get() + 1);
                *left as i16
            })
        },
    );
    exercise_integer_partial_failure(
        communicator,
        rank,
        size,
        failing_rank,
        local_output,
        |calls| {
            sum_many_by(communicator, &[x.view(), y.view()], |values| {
                calls.set(calls.get() + 1);
                *values[0] as i16
            })
        },
    );

    check_success_values(communicator, size, &topology, 4096);

    eprintln!("EXECUTED mapped reduction alloc rank={rank} size={size}");
}
