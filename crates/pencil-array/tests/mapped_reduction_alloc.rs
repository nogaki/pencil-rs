use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;
use std::mem::{align_of, size_of};
use std::sync::Arc;

use mpi::collective::{CommunicatorCollectives, SystemOperation};
use mpi::topology::{CartesianCommunicator, Communicator};
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Stats {
    alloc: usize,
    alloc_zeroed: usize,
    realloc: usize,
    total_bytes: usize,
    max_request_bytes: usize,
}

impl Stats {
    const ZERO: Self = Self {
        alloc: 0,
        alloc_zeroed: 0,
        realloc: 0,
        total_bytes: 0,
        max_request_bytes: 0,
    };
}

#[derive(Clone, Copy)]
enum Entry {
    Alloc,
    AllocZeroed,
    Realloc,
}

thread_local! {
    static INJECTION: Cell<Option<Injection>> = const { Cell::new(None) };
    static TRACKING: Cell<bool> = const { Cell::new(false) };
    static STATS: Cell<Stats> = const { Cell::new(Stats::ZERO) };
}

// The allocator never allocates, formats, or panics: it only updates TLS and
// delegates to the system allocator.
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

fn record(entry: Entry, size: usize) {
    TRACKING.with(|tracking| {
        if !tracking.get() {
            return;
        }
        STATS.with(|stats| {
            let mut current = stats.get();
            match entry {
                Entry::Alloc => current.alloc += 1,
                Entry::AllocZeroed => current.alloc_zeroed += 1,
                Entry::Realloc => current.realloc += 1,
            }
            current.total_bytes += size;
            current.max_request_bytes = current.max_request_bytes.max(size);
            stats.set(current);
        });
    });
}

struct CountingAllocator;

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(Entry::Alloc, layout.size());
        if should_fail(layout.size(), layout.align()) {
            return std::ptr::null_mut();
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(Entry::AllocZeroed, layout.size());
        if should_fail(layout.size(), layout.align()) {
            return std::ptr::null_mut();
        }
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record(Entry::Realloc, new_size);
        if should_fail(new_size, layout.align()) {
            return std::ptr::null_mut();
        }
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

fn measure<R>(operation: impl FnOnce() -> R) -> (R, Stats) {
    TRACKING.with(|tracking| tracking.set(false));
    STATS.with(|stats| stats.set(Stats::ZERO));
    TRACKING.with(|tracking| tracking.set(true));
    let result = operation();
    TRACKING.with(|tracking| tracking.set(false));
    (result, STATS.with(Cell::get))
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

fn agree_failures(communicator: &CartesianCommunicator, failures: &[String]) {
    let local_failed = i32::from(!failures.is_empty());
    let mut any_failed = 0i32;
    communicator.all_reduce_into(&local_failed, &mut any_failed, SystemOperation::max());
    assert_eq!(
        any_failed,
        0,
        "rank {} failures:\n{}",
        communicator.rank(),
        failures.join("\n")
    );
}

// Each call reserves partials with exactly P * size_of::<i16>() bytes and i16
// alignment; Observe verifies that this request matches only the partials
// reservation.
fn exercise_integer_partial_failure(
    communicator: &CartesianCommunicator,
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

fn grows_with_input(small: Stats, large: Stats) -> bool {
    large.total_bytes > small.total_bytes || large.max_request_bytes > small.max_request_bytes
}

fn staging_stats(n: usize) -> (bool, Stats) {
    let ((reserved, len), stats) = measure(|| {
        let mut values = Vec::<f64>::new();
        let reserved = values.try_reserve_exact(n).is_ok();
        if reserved {
            for index in 0..n {
                values.push(index as f64);
            }
        }
        black_box(&values);
        (reserved, values.len())
    });
    (reserved && len == n, stats)
}

fn positive_control_failures() -> Vec<String> {
    let mut failures = Vec::new();

    let ((), stats) = measure(|| {
        let buffer = Vec::<u8>::with_capacity(64);
        black_box(&buffer);
    });
    if stats.alloc == 0 || stats.total_bytes < 64 || stats.max_request_bytes < 64 {
        failures.push(format!("alloc positive control: {stats:?}"));
    }

    let ((), stats) = measure(|| {
        let buffer = vec![0u8; 4096];
        black_box(&buffer);
    });
    if stats.alloc_zeroed == 0 || stats.total_bytes < 4096 || stats.max_request_bytes < 4096 {
        failures.push(format!("alloc_zeroed positive control: {stats:?}"));
    }

    let ((), stats) = measure(|| {
        let mut buffer = Vec::<u8>::with_capacity(1);
        buffer.reserve(4096);
        black_box(&buffer);
    });
    if stats.realloc == 0 || stats.total_bytes < 4096 || stats.max_request_bytes < 4096 {
        failures.push(format!("realloc positive control: {stats:?}"));
    }

    let (small_ok, small) = staging_stats(4096);
    let (large_ok, large) = staging_stats(65536);
    if !small_ok || !large_ok {
        failures.push(format!(
            "staging positive control reservation failed: small={small_ok} large={large_ok}"
        ));
    } else if !grows_with_input(small, large) {
        failures.push(format!(
            "staging positive control was not detected: small={small:?} large={large:?}"
        ));
    }

    failures
}

struct Inputs {
    x: PencilArray<f64, 1, 1>,
    y: PencilArray<f64, 1, 1>,
    xi: PencilArray<i64, 1, 1>,
    yi: PencilArray<i64, 1, 1>,
}

fn build_inputs(
    topology: &Arc<MpiTopology<1>>,
    size: usize,
    n: usize,
) -> (Arc<Pencil<1, 1>>, Inputs) {
    let pencil = Pencil::<1, 1>::new(Arc::clone(topology), [size * n], [0]).unwrap();
    let inputs = Inputs {
        x: PencilArray::from_elem(Arc::clone(&pencil), ExtraShape::scalar(), 1.0_f64).unwrap(),
        y: PencilArray::from_elem(Arc::clone(&pencil), ExtraShape::scalar(), 1.0_f64).unwrap(),
        xi: PencilArray::from_elem(Arc::clone(&pencil), ExtraShape::scalar(), 1_i64).unwrap(),
        yi: PencilArray::from_elem(Arc::clone(&pencil), ExtraShape::scalar(), 1_i64).unwrap(),
    };
    (pencil, inputs)
}

struct CallResult {
    name: &'static str,
    values_ok: bool,
    stats: [Stats; 2],
}

fn run_twice<R>(
    mut operation: impl FnMut(&Cell<usize>) -> R,
    mut check: impl FnMut(&R, usize) -> bool,
) -> (bool, [Stats; 2]) {
    let mut values_ok = true;
    let mut stats = [Stats::ZERO; 2];
    for entry in &mut stats {
        let calls = Cell::new(0usize);
        let (result, measured) = measure(|| operation(&calls));
        values_ok &= check(&result, calls.get());
        *entry = measured;
    }
    (values_ok, stats)
}

fn run_calls(
    communicator: &CartesianCommunicator,
    size: usize,
    inputs: &Inputs,
    n: usize,
) -> Vec<CallResult> {
    let x = &inputs.x;
    let y = &inputs.y;
    let xi = &inputs.xi;
    let yi = &inputs.yi;
    let expected_sum = (size * n) as f64;
    let expected_norm = expected_sum.sqrt();
    let expected_integer_sum = (size * n) as i64;
    let mut results = Vec::with_capacity(13);

    let (values_ok, stats) = run_twice(
        |calls| {
            sum_by(&x.view(), |value| {
                calls.set(calls.get() + 1);
                *value
            })
        },
        |result, calls| *result == Ok(expected_sum) && calls == n,
    );
    results.push(CallResult {
        name: "sum_by f64",
        values_ok,
        stats,
    });

    let (values_ok, stats) = run_twice(
        |calls| {
            norm_by(&x.view(), |value| {
                calls.set(calls.get() + 1);
                *value
            })
        },
        |result, calls| {
            (match result {
                Ok(value) => (*value / expected_norm - 1.0).abs() < 1.0e-12,
                Err(_) => false,
            }) && calls == n
        },
    );
    results.push(CallResult {
        name: "norm_by f64",
        values_ok,
        stats,
    });

    let (values_ok, stats) = run_twice(
        |calls| {
            min_by(&x.view(), |value| {
                calls.set(calls.get() + 1);
                *value
            })
        },
        |result, calls| *result == Ok(Some(1.0)) && calls == n,
    );
    results.push(CallResult {
        name: "min_by f64",
        values_ok,
        stats,
    });

    let (values_ok, stats) = run_twice(
        |calls| {
            max_by(&x.view(), |value| {
                calls.set(calls.get() + 1);
                *value
            })
        },
        |result, calls| *result == Ok(Some(1.0)) && calls == n,
    );
    results.push(CallResult {
        name: "max_by f64",
        values_ok,
        stats,
    });

    let (values_ok, stats) = run_twice(
        |calls| {
            zip_sum_by(&x.view(), &y.view(), |left, right| {
                calls.set(calls.get() + 1);
                left * right
            })
        },
        |result, calls| *result == Ok(expected_sum) && calls == n,
    );
    results.push(CallResult {
        name: "zip_sum_by f64",
        values_ok,
        stats,
    });

    let (values_ok, stats) = run_twice(
        |calls| {
            zip_norm_by(&x.view(), &y.view(), |left, right| {
                calls.set(calls.get() + 1);
                left * right
            })
        },
        |result, calls| {
            (match result {
                Ok(value) => (*value / expected_norm - 1.0).abs() < 1.0e-12,
                Err(_) => false,
            }) && calls == n
        },
    );
    results.push(CallResult {
        name: "zip_norm_by f64",
        values_ok,
        stats,
    });

    let (values_ok, stats) = run_twice(
        |calls| {
            sum_many_by(communicator, &[x.view(), y.view()], |values| {
                calls.set(calls.get() + 1);
                values[0] * values[1]
            })
        },
        |result, calls| *result == Ok(expected_sum) && calls == n,
    );
    results.push(CallResult {
        name: "sum_many_by f64",
        values_ok,
        stats,
    });

    let (values_ok, stats) = run_twice(
        |calls| {
            norm_many_by(communicator, &[x.view(), y.view()], |values| {
                calls.set(calls.get() + 1);
                values[0] * values[1]
            })
        },
        |result, calls| {
            (match result {
                Ok(value) => (*value / expected_norm - 1.0).abs() < 1.0e-12,
                Err(_) => false,
            }) && calls == n
        },
    );
    results.push(CallResult {
        name: "norm_many_by f64",
        values_ok,
        stats,
    });

    let (values_ok, stats) = run_twice(
        |calls| {
            min_many_by(communicator, &[x.view(), y.view()], |values| {
                calls.set(calls.get() + 1);
                values[0] * values[1]
            })
        },
        |result, calls| *result == Ok(Some(1.0)) && calls == n,
    );
    results.push(CallResult {
        name: "min_many_by f64",
        values_ok,
        stats,
    });

    let (values_ok, stats) = run_twice(
        |calls| {
            max_many_by(communicator, &[x.view(), y.view()], |values| {
                calls.set(calls.get() + 1);
                values[0] * values[1]
            })
        },
        |result, calls| *result == Ok(Some(1.0)) && calls == n,
    );
    results.push(CallResult {
        name: "max_many_by f64",
        values_ok,
        stats,
    });

    let (values_ok, stats) = run_twice(
        |calls| {
            sum_by(&xi.view(), |value| {
                calls.set(calls.get() + 1);
                *value
            })
        },
        |result, calls| *result == Ok(expected_integer_sum) && calls == n,
    );
    results.push(CallResult {
        name: "sum_by i64",
        values_ok,
        stats,
    });

    let (values_ok, stats) = run_twice(
        |calls| {
            zip_sum_by(&xi.view(), &yi.view(), |left, right| {
                calls.set(calls.get() + 1);
                left * right
            })
        },
        |result, calls| *result == Ok(expected_integer_sum) && calls == n,
    );
    results.push(CallResult {
        name: "zip_sum_by i64",
        values_ok,
        stats,
    });

    let (values_ok, stats) = run_twice(
        |calls| {
            sum_many_by(communicator, &[xi.view(), yi.view()], |values| {
                calls.set(calls.get() + 1);
                values[0] * values[1]
            })
        },
        |result, calls| *result == Ok(expected_integer_sum) && calls == n,
    );
    results.push(CallResult {
        name: "sum_many_by i64",
        values_ok,
        stats,
    });

    results
}

fn allocation_failures(small: &[CallResult], large: &[CallResult]) -> Vec<String> {
    let mut failures = Vec::new();
    if small.len() != large.len() {
        failures.push(format!(
            "call count mismatch: n1={} n2={}",
            small.len(),
            large.len()
        ));
    }
    for (small_call, large_call) in small.iter().zip(large) {
        if small_call.name != large_call.name {
            failures.push(format!(
                "call name mismatch: {} vs {}",
                small_call.name, large_call.name
            ));
            continue;
        }
        if !small_call.values_ok || !large_call.values_ok {
            failures.push(format!(
                "{}: value or callback count check failed (n1 ok={}, n2 ok={})",
                small_call.name, small_call.values_ok, large_call.values_ok
            ));
        }
        for (run, label) in ["first", "second"].iter().enumerate() {
            let small_stats = small_call.stats[run];
            let large_stats = large_call.stats[run];
            if grows_with_input(small_stats, large_stats) {
                failures.push(format!(
                    "{} {}: n1 total={} max={}, n2 total={} max={}",
                    small_call.name,
                    label,
                    small_stats.total_bytes,
                    small_stats.max_request_bytes,
                    large_stats.total_bytes,
                    large_stats.max_request_bytes
                ));
            }
        }
    }
    failures
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
    eprintln!("STAGE ok: integer partial allocation injection rank={rank}");

    let controls = positive_control_failures();
    agree_failures(communicator, &controls);
    eprintln!("STAGE ok: positive controls rank={rank}");

    let (pencil_small, inputs_small) = build_inputs(&topology, size, 4096);
    let (pencil_large, inputs_large) = build_inputs(&topology, size, 65536);
    check_all(communicator, pencil_small.local_len() == 4096);
    check_all(communicator, pencil_large.local_len() == 65536);
    let small = run_calls(communicator, size, &inputs_small, 4096);
    let large = run_calls(communicator, size, &inputs_large, 65536);
    let failures = allocation_failures(&small, &large);
    agree_failures(communicator, &failures);
    eprintln!("STAGE ok: mapped reduction allocation growth rank={rank}");

    eprintln!("EXECUTED mapped reduction alloc rank={rank} size={size}");
}
