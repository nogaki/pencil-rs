use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;

use pencil_fft::{Complex, FftReal, LocalC2cNdError, LocalC2cNdPlan};
use rustfft::FftPlanner;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Counts {
    alloc: usize,
    alloc_zeroed: usize,
    realloc: usize,
}

impl Counts {
    const ZERO: Self = Self {
        alloc: 0,
        alloc_zeroed: 0,
        realloc: 0,
    };
}

thread_local! {
    static TRACKING: Cell<bool> = const { Cell::new(false) };
    static COUNTS: Cell<Counts> = const { Cell::new(Counts::ZERO) };
}

struct CountingAllocator;

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[derive(Clone, Copy)]
enum Allocation {
    Alloc,
    AllocZeroed,
    Realloc,
}

fn record(kind: Allocation) {
    TRACKING.with(|tracking| {
        if tracking.get() {
            COUNTS.with(|counts| {
                let mut current = counts.get();
                match kind {
                    Allocation::Alloc => current.alloc += 1,
                    Allocation::AllocZeroed => current.alloc_zeroed += 1,
                    Allocation::Realloc => current.realloc += 1,
                }
                counts.set(current);
            });
        }
    });
}

// The TLS state and counters are initialized before measurement; each test's
// measured calls run synchronously on this same thread.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(Allocation::Alloc);
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(Allocation::AllocZeroed);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record(Allocation::Realloc);
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

fn measure<T>(operation: impl FnOnce() -> T) -> (Counts, T) {
    TRACKING.with(|tracking| tracking.set(false));
    COUNTS.with(|counts| counts.set(Counts::ZERO));
    TRACKING.with(|tracking| tracking.set(true));
    let result = operation();
    TRACKING.with(|tracking| tracking.set(false));
    let counts = COUNTS.with(Cell::get);
    (counts, result)
}

fn assert_no_allocations<T>(operation: impl FnOnce() -> T) -> T {
    let (counts, result) = measure(operation);
    assert_eq!(counts, Counts::ZERO);
    result
}

fn allocator_positive_controls() {
    let layout = Layout::from_size_align(16, 8).unwrap();
    let (counts, pointer) = measure(|| {
        let pointer = unsafe { ALLOCATOR.alloc(layout) };
        black_box(pointer);
        pointer
    });
    assert_eq!(
        counts,
        Counts {
            alloc: 1,
            ..Counts::ZERO
        }
    );
    assert!(!pointer.is_null());
    unsafe { ALLOCATOR.dealloc(pointer, layout) };

    let (counts, pointer) = measure(|| {
        let pointer = unsafe { ALLOCATOR.alloc_zeroed(layout) };
        black_box(pointer);
        pointer
    });
    assert_eq!(
        counts,
        Counts {
            alloc_zeroed: 1,
            ..Counts::ZERO
        }
    );
    assert!(!pointer.is_null());
    unsafe { ALLOCATOR.dealloc(pointer, layout) };

    let new_size = 32;
    let new_layout = Layout::from_size_align(new_size, 8).unwrap();
    let original = unsafe { System.alloc(layout) };
    assert!(!original.is_null());
    let (counts, pointer) = measure(|| {
        let pointer = unsafe { ALLOCATOR.realloc(original, layout, new_size) };
        black_box(pointer);
        pointer
    });
    assert_eq!(
        counts,
        Counts {
            realloc: 1,
            ..Counts::ZERO
        }
    );
    assert!(!pointer.is_null());
    unsafe { ALLOCATOR.dealloc(pointer, new_layout) };
}

trait TestReal: FftReal {
    fn convert(value: f64) -> Self;
}

impl TestReal for f32 {
    fn convert(value: f64) -> Self {
        value as f32
    }
}

impl TestReal for f64 {
    fn convert(value: f64) -> Self {
        value
    }
}

fn positive_scratch_length<R: FftReal>(inverse: bool) -> usize {
    let mut planner = FftPlanner::<R>::new();
    for length in (1..=32).chain([37, 97, 127, 1024]) {
        let fft = if inverse {
            planner.plan_fft_inverse(length)
        } else {
            planner.plan_fft_forward(length)
        };
        if fft.get_immutable_scratch_len() > 0 {
            return length;
        }
    }
    panic!("no candidate has positive RustFFT immutable scratch");
}

fn data<R: TestReal>(length: usize) -> Vec<Complex<R>> {
    (0..length)
        .map(|index| {
            Complex::new(
                R::convert((index as f64 + 1.0) / 7.0),
                R::convert((index as f64 - 2.0) / 11.0),
            )
        })
        .collect()
}

fn run_operation<R: FftReal>(
    plan: &mut LocalC2cNdPlan<R>,
    values: &mut [Complex<R>],
    inverse: bool,
) -> Result<(), LocalC2cNdError> {
    if inverse {
        plan.inverse(values)
    } else {
        plan.forward(values)
    }
}

fn check_first_and_repeated<R: FftReal>(
    plan: &mut LocalC2cNdPlan<R>,
    values: &mut [Complex<R>],
    inverse: bool,
) {
    let first = assert_no_allocations(|| run_operation(plan, values, inverse));
    assert_eq!(first, Ok(()));

    let repeated = assert_no_allocations(|| {
        for _ in 0..8 {
            run_operation(plan, values, inverse)?;
        }
        Ok::<(), LocalC2cNdError>(())
    });
    assert_eq!(repeated, Ok(()));
}

fn check_precision_and_direction<R: TestReal>(inverse: bool) {
    // Prime TLS state and verify that each allocator entry point is counted.
    TRACKING.with(|tracking| tracking.set(false));
    COUNTS.with(Cell::get);
    allocator_positive_controls();

    let length = positive_scratch_length::<R>(inverse);
    let mut scratch_plan = LocalC2cNdPlan::<R>::new_col_major(&[2, length, 3], &[1]).unwrap();
    let mut scratch_data = data::<R>(6 * length);
    let mut empty_plan = LocalC2cNdPlan::<R>::new_col_major(&[2, length, 3], &[]).unwrap();
    let mut empty_data = data::<R>(6 * length);
    let mut unit_plan = LocalC2cNdPlan::<R>::new_col_major(&[1], &[0]).unwrap();
    let mut unit_data = data::<R>(1);

    check_first_and_repeated(&mut scratch_plan, &mut scratch_data, inverse);
    check_first_and_repeated(&mut empty_plan, &mut empty_data, inverse);
    check_first_and_repeated(&mut unit_plan, &mut unit_data, inverse);
}

#[test]
fn f32_forward_first_and_repeated_allocate_nothing() {
    check_precision_and_direction::<f32>(false);
}

#[test]
fn f32_inverse_first_and_repeated_allocate_nothing() {
    check_precision_and_direction::<f32>(true);
}

#[test]
fn f64_forward_first_and_repeated_allocate_nothing() {
    check_precision_and_direction::<f64>(false);
}

#[test]
fn f64_inverse_first_and_repeated_allocate_nothing() {
    check_precision_and_direction::<f64>(true);
}
