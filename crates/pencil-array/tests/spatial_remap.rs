use std::{
    alloc::{GlobalAlloc, Layout, System},
    fmt::Debug,
    mem::offset_of,
    sync::Arc,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

use mpi::{
    Count,
    datatype::{Datatype, Equivalence, UncommittedDatatype, UserDatatype},
    raw::traits::AsRaw,
    traits::*,
};
use num_complex::Complex;
use pencil_array::{
    AxisPermutation, ExtraShape, MpiTopology, Pencil, PencilArrayView, PencilArrayViewMut,
    SpatialRemapError, SpatialRemapPlan, all_ranks_succeeded,
};

struct CountingAllocator;

static TRACK_ALLOCATIONS: AtomicBool = AtomicBool::new(false);
static ALLOC_CALLS: AtomicUsize = AtomicUsize::new(0);
static ALLOC_ZEROED_CALLS: AtomicUsize = AtomicUsize::new(0);
static REALLOC_CALLS: AtomicUsize = AtomicUsize::new(0);
static ALLOCATION_ORDINAL: AtomicUsize = AtomicUsize::new(0);
static FAIL_AT_ORDINAL: AtomicUsize = AtomicUsize::new(0);
static ALLOCATION_FAILURE_HIT: AtomicBool = AtomicBool::new(false);

fn count_allocation(counter: &AtomicUsize) -> bool {
    if !TRACK_ALLOCATIONS.load(Ordering::SeqCst) {
        return false;
    }
    counter.fetch_add(1, Ordering::SeqCst);
    let ordinal = ALLOCATION_ORDINAL.fetch_add(1, Ordering::SeqCst) + 1;
    if FAIL_AT_ORDINAL
        .compare_exchange(ordinal, 0, Ordering::SeqCst, Ordering::SeqCst)
        .is_ok()
    {
        ALLOCATION_FAILURE_HIT.store(true, Ordering::SeqCst);
        true
    } else {
        false
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

// SAFETY: every operation delegates to the matching System allocator method.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if count_allocation(&ALLOC_CALLS) {
            return std::ptr::null_mut();
        }
        // SAFETY: forwarded unchanged to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if count_allocation(&ALLOC_ZEROED_CALLS) {
            return std::ptr::null_mut();
        }
        // SAFETY: forwarded unchanged to the system allocator.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: forwarded unchanged to the system allocator.
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if count_allocation(&REALLOC_CALLS) {
            return std::ptr::null_mut();
        }
        // SAFETY: forwarded unchanged to the system allocator.
        unsafe { System.realloc(pointer, layout, size) }
    }
}

struct AllocationTracking;

impl Drop for AllocationTracking {
    fn drop(&mut self) {
        TRACK_ALLOCATIONS.store(false, Ordering::SeqCst);
        FAIL_AT_ORDINAL.store(0, Ordering::SeqCst);
    }
}

fn measured<T>(operation: impl FnOnce() -> T) -> (T, [usize; 3]) {
    ALLOC_CALLS.store(0, Ordering::SeqCst);
    ALLOC_ZEROED_CALLS.store(0, Ordering::SeqCst);
    REALLOC_CALLS.store(0, Ordering::SeqCst);
    ALLOCATION_ORDINAL.store(0, Ordering::SeqCst);
    FAIL_AT_ORDINAL.store(0, Ordering::SeqCst);
    ALLOCATION_FAILURE_HIT.store(false, Ordering::SeqCst);
    TRACK_ALLOCATIONS.store(true, Ordering::SeqCst);
    let tracking = AllocationTracking;
    let result = operation();
    drop(tracking);
    (
        result,
        [
            ALLOC_CALLS.load(Ordering::SeqCst),
            ALLOC_ZEROED_CALLS.load(Ordering::SeqCst),
            REALLOC_CALLS.load(Ordering::SeqCst),
        ],
    )
}

fn fail_nth_allocation<T>(
    ordinal: usize,
    fail_here: bool,
    operation: impl FnOnce() -> T,
) -> (T, usize, bool) {
    ALLOC_CALLS.store(0, Ordering::SeqCst);
    ALLOC_ZEROED_CALLS.store(0, Ordering::SeqCst);
    REALLOC_CALLS.store(0, Ordering::SeqCst);
    ALLOCATION_ORDINAL.store(0, Ordering::SeqCst);
    FAIL_AT_ORDINAL.store(if fail_here { ordinal } else { 0 }, Ordering::SeqCst);
    ALLOCATION_FAILURE_HIT.store(false, Ordering::SeqCst);
    TRACK_ALLOCATIONS.store(true, Ordering::SeqCst);
    let tracking = AllocationTracking;
    let result = operation();
    drop(tracking);
    (
        result,
        ALLOCATION_ORDINAL.load(Ordering::SeqCst),
        ALLOCATION_FAILURE_HIT.load(Ordering::SeqCst),
    )
}

static EQUIVALENCE_CALLS: AtomicUsize = AtomicUsize::new(0);
static CLONE_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUSTOM_OUT_AS_RAW_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUSTOM_OUT_CLONE_CALLS: AtomicUsize = AtomicUsize::new(0);
static CUSTOM_OUT_DROP_CALLS: AtomicUsize = AtomicUsize::new(0);

#[repr(transparent)]
#[derive(Copy, Debug, PartialEq, Eq)]
struct Probe(u64);

#[derive(Copy, Clone)]
struct ZeroSized;

// SAFETY: this intentionally mismatched datatype lets prepare verify the Rust ZST rejection.
unsafe impl Equivalence for ZeroSized {
    type Out = <u8 as Equivalence>::Out;

    fn equivalent_datatype() -> Self::Out {
        <u8 as Equivalence>::equivalent_datatype()
    }
}

#[allow(clippy::non_canonical_clone_impl)]
impl Clone for Probe {
    fn clone(&self) -> Self {
        CLONE_CALLS.fetch_add(1, Ordering::Relaxed);
        *self
    }
}

// SAFETY: Probe is transparent over u64, and its MPI datatype describes the
// complete initialized representation without exposing padding.
unsafe impl Equivalence for Probe {
    type Out = <u64 as Equivalence>::Out;

    fn equivalent_datatype() -> Self::Out {
        EQUIVALENCE_CALLS.fetch_add(1, Ordering::Relaxed);
        <u64 as Equivalence>::equivalent_datatype()
    }
}

#[repr(C)]
#[derive(Copy, Debug, PartialEq, Eq)]
struct PaddedProbe {
    tag: u8,
    value: u64,
}

#[allow(clippy::non_canonical_clone_impl)]
impl Clone for PaddedProbe {
    fn clone(&self) -> Self {
        CLONE_CALLS.fetch_add(1, Ordering::Relaxed);
        *self
    }
}

struct TrackedDatatype(UserDatatype);

impl Clone for TrackedDatatype {
    fn clone(&self) -> Self {
        CUSTOM_OUT_CLONE_CALLS.fetch_add(1, Ordering::Relaxed);
        Self(self.0.clone())
    }
}

impl Drop for TrackedDatatype {
    fn drop(&mut self) {
        CUSTOM_OUT_DROP_CALLS.fetch_add(1, Ordering::Relaxed);
    }
}

// SAFETY: the wrapper owns the committed datatype and forwards its valid handle.
unsafe impl AsRaw for TrackedDatatype {
    type Raw = <UserDatatype as AsRaw>::Raw;

    fn as_raw(&self) -> Self::Raw {
        CUSTOM_OUT_AS_RAW_CALLS.fetch_add(1, Ordering::Relaxed);
        self.0.as_raw()
    }
}

impl UncommittedDatatype for TrackedDatatype {
    type DuplicatedDatatype = UserDatatype;
}

impl Datatype for TrackedDatatype {}

// SAFETY: the typemap covers only initialized fields; its extent is the
// repr(C) size, including the internal padding between `tag` and `value`.
unsafe impl Equivalence for PaddedProbe {
    type Out = TrackedDatatype;

    fn equivalent_datatype() -> Self::Out {
        EQUIVALENCE_CALLS.fetch_add(1, Ordering::Relaxed);
        let fields = [
            <u8 as Equivalence>::equivalent_datatype(),
            <u64 as Equivalence>::equivalent_datatype(),
        ];
        TrackedDatatype(UserDatatype::structured(
            &[1, 1],
            &[
                offset_of!(PaddedProbe, tag) as mpi::Address,
                offset_of!(PaddedProbe, value) as mpi::Address,
            ],
            &fields,
        ))
    }
}

fn padded_probe_value(extra: usize, global: usize) -> PaddedProbe {
    PaddedProbe {
        tag: (extra % 251) as u8,
        value: (extra as u64) * 1_000_000 + global as u64,
    }
}

fn probe_value(extra: usize, global: usize) -> Probe {
    Probe((extra as u64) * 1_000_000 + global as u64)
}

fn i32_value(extra: usize, global: usize) -> i32 {
    (extra * 1_000 + global) as i32
}

fn f32_value(extra: usize, global: usize) -> f32 {
    (extra * 1_000 + global) as f32 / 4.0
}

fn f64_value(extra: usize, global: usize) -> f64 {
    (extra * 1_000 + global) as f64 / 8.0
}

fn complex32_value(extra: usize, global: usize) -> Complex<f32> {
    Complex::new((extra + global) as f32, (global % 11) as f32)
}

fn complex64_value(extra: usize, global: usize) -> Complex<f64> {
    Complex::new((extra + global) as f64, (global % 13) as f64)
}

fn global_linear<const N: usize>(coordinate: [usize; N], shape: [usize; N]) -> usize {
    coordinate
        .iter()
        .zip(shape)
        .fold(0usize, |linear, (&index, extent)| linear * extent + index)
}

fn local_global<const N: usize, const M: usize>(
    pencil: &Pencil<N, M>,
    spatial_linear: usize,
) -> [usize; N] {
    let memory_shape = pencil.local_shape_memory();
    let mut remainder = spatial_linear;
    let mut memory = [0usize; N];
    for axis in (0..N).rev() {
        memory[axis] = remainder % memory_shape[axis];
        remainder /= memory_shape[axis];
    }
    let mut global = [0usize; N];
    for (position, axis) in pencil.permutation().axes().iter().enumerate() {
        global[axis.index()] = pencil.local_ranges()[axis.index()].start + memory[position];
    }
    global
}

fn affine_oracle<const N: usize>(
    destination: [usize; N],
    shape: [usize; N],
    signs: [i8; N],
    offsets: [i128; N],
) -> [usize; N] {
    std::array::from_fn(|axis| {
        let extent = i128::try_from(shape[axis]).unwrap();
        let destination = i128::try_from(destination[axis]).unwrap();
        let sign = i128::from(signs[axis]);
        let normalized_offset = offsets[axis].rem_euclid(extent);
        usize::try_from((sign * destination + normalized_offset).rem_euclid(extent)).unwrap()
    })
}

fn make_values<T, const N: usize, const M: usize>(
    pencil: &Pencil<N, M>,
    extra_count: usize,
    value: impl Fn(usize, usize) -> T,
) -> Vec<T> {
    let length = pencil.local_len() * extra_count;
    let mut values = Vec::with_capacity(length);
    for extra in 0..extra_count {
        for spatial in 0..pencil.local_len() {
            let global = local_global(pencil, spatial);
            values.push(value(extra, global_linear(global, *pencil.global_shape())));
        }
    }
    values
}

fn assert_oracle<T, const N: usize, const M: usize>(
    pencil: &Pencil<N, M>,
    extra_count: usize,
    signs: [i8; N],
    offsets: [i128; N],
    value: impl Fn(usize, usize) -> T,
    actual: &[T],
) where
    T: PartialEq + Debug,
{
    for extra in 0..extra_count {
        for spatial in 0..pencil.local_len() {
            let destination = local_global(pencil, spatial);
            let source = affine_oracle(destination, *pencil.global_shape(), signs, offsets);
            let expected = value(extra, global_linear(source, *pencil.global_shape()));
            assert_eq!(actual[extra * pencil.local_len() + spatial], expected);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_case<T>(
    topology: &Arc<MpiTopology<2>>,
    source: &Arc<Pencil<3, 2>>,
    destination_decomposition: [usize; 2],
    destination_permutation: [usize; 3],
    extra: ExtraShape,
    signs: [i8; 3],
    offsets: [i128; 3],
    value: fn(usize, usize) -> T,
    measure_first_apply: bool,
) where
    T: Equivalence + Copy + PartialEq + Debug,
{
    let destination = Pencil::<3, 2>::new_permuted(
        Arc::clone(topology),
        *source.global_shape(),
        destination_decomposition,
        AxisPermutation::new(destination_permutation).unwrap(),
    )
    .unwrap();
    let source_data = make_values(source, extra.element_count(), value);
    let initial = value(999, 999);
    let mut destination_data = std::iter::repeat_with(|| initial)
        .take(destination.local_len() * extra.element_count())
        .collect::<Vec<_>>();
    let mut plan = SpatialRemapPlan::prepare(
        Arc::clone(source),
        Arc::clone(&destination),
        extra.clone(),
        signs,
        offsets,
        value(0, 0),
    )
    .unwrap();
    let source_before = source_data.to_vec();
    EQUIVALENCE_CALLS.store(0, Ordering::SeqCst);
    CLONE_CALLS.store(0, Ordering::SeqCst);
    let source_view = PencilArrayView::from_slice(source, &extra, &source_data).unwrap();
    let destination_view =
        PencilArrayViewMut::from_slice_mut(&destination, &extra, &mut destination_data).unwrap();

    if measure_first_apply {
        let (result, counts) = measured(|| plan.apply_views(source_view, destination_view));
        result.unwrap();
        assert_eq!(counts, [0, 0, 0], "first apply allocated");
        assert_eq!(CLONE_CALLS.load(Ordering::SeqCst), 0);
        assert_eq!(EQUIVALENCE_CALLS.load(Ordering::SeqCst), 0);
    } else {
        plan.apply_views(source_view, destination_view).unwrap();
    }
    assert_eq!(source_data, source_before);
    assert_oracle(
        &destination,
        extra.element_count(),
        signs,
        offsets,
        value,
        &destination_data,
    );

    if measure_first_apply {
        let mut repeated = std::iter::repeat_with(|| initial)
            .take(destination.local_len() * extra.element_count())
            .collect::<Vec<_>>();
        let repeated_source_data = make_values(source, extra.element_count(), |extra, global| {
            value(extra + 1000, global)
        });
        let source_view =
            PencilArrayView::from_slice(source, &extra, &repeated_source_data).unwrap();
        let destination_view =
            PencilArrayViewMut::from_slice_mut(&destination, &extra, &mut repeated).unwrap();
        let (result, counts) = measured(|| plan.apply_views(source_view, destination_view));
        result.unwrap();
        assert_eq!(counts, [0, 0, 0], "repeated apply allocated");
        assert_eq!(CLONE_CALLS.load(Ordering::SeqCst), 0);
        assert_eq!(EQUIVALENCE_CALLS.load(Ordering::SeqCst), 0);
        assert_oracle(
            &destination,
            extra.element_count(),
            signs,
            offsets,
            |extra, global| value(extra + 1000, global),
            &repeated,
        );
        assert_ne!(
            repeated, destination_data,
            "repeated apply reused old payload"
        );
    }
}

fn mismatch_result(result: Result<(), SpatialRemapError>, rank: i32, local: SpatialRemapError) {
    if rank == 0 {
        assert!(
            matches!(result, Err(error) if std::mem::discriminant(&error) == std::mem::discriminant(&local))
        );
    } else {
        assert!(matches!(
            result,
            Err(SpatialRemapError::CollectivePreconditionFailed)
        ));
    }
}

#[test]
fn spatial_remap_collective_contract_and_oracles() {
    let universe = mpi::initialize().expect("MPI must not already be initialized");
    let world = universe.world();
    let rank = world.rank();
    let size = usize::try_from(world.size()).unwrap();
    let grid = match size {
        1 => [1, 1],
        4 => [2, 2],
        6 => [2, 3],
        n => [1, n],
    };
    let topology = MpiTopology::<2>::new(&world, grid).unwrap();
    assert!(all_ranks_succeeded(&topology, true));
    assert!(!all_ranks_succeeded(&topology, rank != 0));

    let source = Pencil::<3, 2>::new_permuted(
        Arc::clone(&topology),
        [5, 7, 4],
        [0, 1],
        AxisPermutation::new([2, 0, 1]).unwrap(),
    )
    .unwrap();
    let extra = ExtraShape::new([2, 3]).unwrap();
    EQUIVALENCE_CALLS.store(0, Ordering::SeqCst);
    CLONE_CALLS.store(0, Ordering::SeqCst);
    let (_, positive_control) = measured(|| Box::new(7u64));
    assert!(
        positive_control.iter().sum::<usize>() > 0,
        "allocator probe failed"
    );

    for (ordinal, stage) in [
        (1, "collective descriptor"),
        (2, "owner request routes"),
        (7, "request receive displacements"),
        (16, "payload send buffer"),
    ] {
        let fault_source = Arc::clone(&source);
        let fault_destination = Arc::clone(&source);
        let fault_extra = extra.clone();
        let (result, attempts, failed) = fail_nth_allocation(ordinal, rank == 0, move || {
            SpatialRemapPlan::<Probe, 3, 2>::prepare(
                fault_source,
                fault_destination,
                fault_extra,
                [1, -1, 1],
                [1, 2, 3],
                Probe(0),
            )
        });
        assert_eq!(failed, rank == 0, "unexpected failpoint hit at {ordinal}");
        if rank == 0 {
            assert_eq!(attempts, ordinal);
            assert!(
                matches!(result, Err(SpatialRemapError::Allocation { stage: actual, .. }) if actual == stage)
            );
        } else {
            assert!(attempts >= ordinal);
            assert!(matches!(
                result,
                Err(SpatialRemapError::CollectivePreconditionFailed)
            ));
        }
    }

    let oversized_extra =
        ExtraShape::new([usize::try_from(Count::MAX).unwrap().checked_add(1).unwrap()]).unwrap();
    assert!(matches!(
        SpatialRemapPlan::<Probe, 3, 2>::prepare(
            Arc::clone(&source),
            Arc::clone(&source),
            oversized_extra,
            [1; 3],
            [0; 3],
            Probe(0),
        ),
        Err(SpatialRemapError::MpiCountOverflow)
    ));

    let mut recovered = SpatialRemapPlan::<Probe, 3, 2>::prepare(
        Arc::clone(&source),
        Arc::clone(&source),
        extra.clone(),
        [1; 3],
        [0; 3],
        Probe(0),
    )
    .unwrap();
    let recovered_source = make_values(&source, extra.element_count(), probe_value);
    let mut recovered_destination = std::iter::repeat_with(|| Probe(0))
        .take(recovered_source.len())
        .collect::<Vec<_>>();
    let source_view = PencilArrayView::from_slice(&source, &extra, &recovered_source).unwrap();
    let destination_view =
        PencilArrayViewMut::from_slice_mut(&source, &extra, &mut recovered_destination).unwrap();
    recovered
        .apply_views(source_view, destination_view)
        .unwrap();
    assert_oracle(
        &source,
        extra.element_count(),
        [1; 3],
        [0; 3],
        probe_value,
        &recovered_destination,
    );

    run_case(
        &topology,
        &source,
        [0, 1],
        [0, 1, 2],
        extra.clone(),
        [1, 1, 1],
        [0, 0, 0],
        probe_value,
        true,
    );
    run_case(
        &topology,
        &source,
        [2, 1],
        [1, 2, 0],
        extra.clone(),
        [1, -1, 1],
        [2, -4, 7],
        i32_value,
        false,
    );
    run_case(
        &topology,
        &source,
        [1, 0],
        [0, 2, 1],
        extra.clone(),
        [-1, 1, -1],
        [i128::MIN, i128::MAX, i128::MIN],
        f32_value,
        false,
    );
    run_case(
        &topology,
        &source,
        [1, 0],
        [2, 1, 0],
        extra.clone(),
        [1, -1, -1],
        [i128::MAX, -1, 3],
        f64_value,
        false,
    );
    run_case(
        &topology,
        &source,
        [2, 1],
        [1, 0, 2],
        extra.clone(),
        [-1, -1, 1],
        [i128::MIN, 1, i128::MAX],
        complex32_value,
        false,
    );
    run_case(
        &topology,
        &source,
        [1, 0],
        [2, 0, 1],
        extra.clone(),
        [1, 1, -1],
        [0, i128::MAX, i128::MIN],
        complex64_value,
        false,
    );

    // A zero-product extra shape still follows the same route and collective path.
    run_case(
        &topology,
        &source,
        [2, 1],
        [2, 1, 0],
        ExtraShape::new([2, 0]).unwrap(),
        [-1, 1, 1],
        [i128::MAX, i128::MIN, 5],
        i32_value,
        false,
    );
    // Most ranks own no spatial values for this positive global shape.
    let tiny = Pencil::<3, 2>::new_permuted(
        Arc::clone(&topology),
        [1, 1, 1],
        [0, 1],
        AxisPermutation::identity(),
    )
    .unwrap();
    run_case(
        &topology,
        &tiny,
        [2, 1],
        [1, 2, 0],
        ExtraShape::scalar(),
        [-1, 1, -1],
        [i128::MIN, i128::MAX, -1],
        i32_value,
        false,
    );

    let split_source = Pencil::<3, 2>::new_permuted(
        Arc::clone(&topology),
        [5, 7, 1],
        [0, 1],
        AxisPermutation::new([2, 0, 1]).unwrap(),
    )
    .unwrap();
    let split_destination = Pencil::<3, 2>::new_permuted(
        Arc::clone(&topology),
        [5, 7, 1],
        [2, 1],
        AxisPermutation::new([1, 2, 0]).unwrap(),
    )
    .unwrap();
    if size > 1 {
        assert_ne!(
            topology.rank_at([0, 0]).unwrap(),
            topology.rank_at([1, 0]).unwrap(),
            "source and destination have distinct nonempty owners",
        );
        if topology.local_coords() == &[0, 0] {
            assert!(split_source.local_ranges()[0].contains(&0));
            assert!(split_source.local_ranges()[1].contains(&0));
            assert!(split_source.local_len() > 0);
            assert_eq!(split_destination.local_len(), 0);
        }
        if topology.local_coords() == &[1, 0] {
            assert!(split_destination.local_ranges()[0].contains(&0));
            assert!(split_destination.local_ranges()[1].contains(&0));
            assert!(split_destination.local_ranges()[2].contains(&0));
            assert!(split_destination.local_len() > 0);
        }
    }
    run_case(
        &topology,
        &split_source,
        [2, 1],
        [1, 2, 0],
        ExtraShape::scalar(),
        [1; 3],
        [0; 3],
        i32_value,
        false,
    );

    assert!(offset_of!(PaddedProbe, value) > std::mem::size_of::<u8>());
    let padded_destination = Pencil::<3, 2>::new_permuted(
        Arc::clone(&topology),
        *source.global_shape(),
        [2, 1],
        AxisPermutation::new([1, 2, 0]).unwrap(),
    )
    .unwrap();
    let padded_wrong_destination = Pencil::<3, 2>::new_permuted(
        Arc::clone(&topology),
        *source.global_shape(),
        [2, 1],
        AxisPermutation::identity(),
    )
    .unwrap();
    let padded_extra = ExtraShape::new([2, 3]).unwrap();
    let padded_initial = padded_probe_value(999, 999);
    let mut padded_destination_data = std::iter::repeat_with(|| padded_initial)
        .take(padded_destination.local_len() * padded_extra.element_count())
        .collect::<Vec<_>>();
    let mut padded_repeated = std::iter::repeat_with(|| padded_initial)
        .take(padded_destination.local_len() * padded_extra.element_count())
        .collect::<Vec<_>>();
    let mut padded_failed_data = std::iter::repeat_with(|| padded_initial)
        .take(padded_destination.local_len() * padded_extra.element_count())
        .collect::<Vec<_>>();
    let padded_source_data = make_values(&source, padded_extra.element_count(), padded_probe_value);
    let padded_source_fields = padded_source_data
        .iter()
        .map(|value| (value.tag, value.value))
        .collect::<Vec<_>>();
    CUSTOM_OUT_AS_RAW_CALLS.store(0, Ordering::SeqCst);
    CUSTOM_OUT_CLONE_CALLS.store(0, Ordering::SeqCst);
    CUSTOM_OUT_DROP_CALLS.store(0, Ordering::SeqCst);
    CLONE_CALLS.store(0, Ordering::SeqCst);
    EQUIVALENCE_CALLS.store(0, Ordering::SeqCst);
    let mut padded_plan = SpatialRemapPlan::<PaddedProbe, 3, 2>::prepare(
        Arc::clone(&source),
        Arc::clone(&padded_destination),
        padded_extra.clone(),
        [-1, 1, -1],
        [3, -4, 7],
        padded_initial,
    )
    .unwrap();
    assert_eq!(CUSTOM_OUT_AS_RAW_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(CUSTOM_OUT_CLONE_CALLS.load(Ordering::SeqCst), 0);
    assert_eq!(CUSTOM_OUT_DROP_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(EQUIVALENCE_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(CLONE_CALLS.load(Ordering::SeqCst), 0);

    let source_view =
        PencilArrayView::from_slice(&source, &padded_extra, &padded_source_data).unwrap();
    let destination_view = PencilArrayViewMut::from_slice_mut(
        &padded_destination,
        &padded_extra,
        &mut padded_destination_data,
    )
    .unwrap();
    let (first_apply, first_apply_allocations) =
        measured(|| padded_plan.apply_views(source_view, destination_view));
    first_apply.unwrap();
    assert_eq!(first_apply_allocations, [0, 0, 0]);
    assert_eq!(CUSTOM_OUT_AS_RAW_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(CUSTOM_OUT_CLONE_CALLS.load(Ordering::SeqCst), 0);
    assert_eq!(CUSTOM_OUT_DROP_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(EQUIVALENCE_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(CLONE_CALLS.load(Ordering::SeqCst), 0);
    assert_oracle(
        &padded_destination,
        padded_extra.element_count(),
        [-1, 1, -1],
        [3, -4, 7],
        padded_probe_value,
        &padded_destination_data,
    );

    let changed_padded_source =
        make_values(&source, padded_extra.element_count(), |extra, global| {
            padded_probe_value(extra + 1000, global)
        });
    let changed_padded_source_fields = changed_padded_source
        .iter()
        .map(|value| (value.tag, value.value))
        .collect::<Vec<_>>();
    let source_view =
        PencilArrayView::from_slice(&source, &padded_extra, &changed_padded_source).unwrap();
    let destination_view = PencilArrayViewMut::from_slice_mut(
        &padded_destination,
        &padded_extra,
        &mut padded_repeated,
    )
    .unwrap();
    let (repeated_apply, repeated_apply_allocations) =
        measured(|| padded_plan.apply_views(source_view, destination_view));
    repeated_apply.unwrap();
    assert_eq!(repeated_apply_allocations, [0, 0, 0]);
    assert_ne!(padded_repeated, padded_destination_data);
    assert_oracle(
        &padded_destination,
        padded_extra.element_count(),
        [-1, 1, -1],
        [3, -4, 7],
        |extra, global| padded_probe_value(extra + 1000, global),
        &padded_repeated,
    );

    let repeated_fields_before = padded_repeated
        .iter()
        .map(|value| (value.tag, value.value))
        .collect::<Vec<_>>();
    let failed_fields_before = padded_failed_data
        .iter()
        .map(|value| (value.tag, value.value))
        .collect::<Vec<_>>();
    let source_view =
        PencilArrayView::from_slice(&source, &padded_extra, &changed_padded_source).unwrap();
    let (failed_apply, failed_apply_allocations) = measured(|| {
        let destination_view = if rank == 0 {
            PencilArrayViewMut::from_slice_mut(
                &padded_wrong_destination,
                &padded_extra,
                &mut padded_failed_data,
            )
            .unwrap()
        } else {
            PencilArrayViewMut::from_slice_mut(
                &padded_destination,
                &padded_extra,
                &mut padded_repeated,
            )
            .unwrap()
        };
        padded_plan.apply_views(source_view, destination_view)
    });
    assert_eq!(failed_apply_allocations, [0, 0, 0]);
    mismatch_result(failed_apply, rank, SpatialRemapError::LayoutMismatch);
    assert_eq!(
        padded_failed_data
            .iter()
            .map(|value| (value.tag, value.value))
            .collect::<Vec<_>>(),
        failed_fields_before,
    );
    assert_eq!(
        padded_repeated
            .iter()
            .map(|value| (value.tag, value.value))
            .collect::<Vec<_>>(),
        repeated_fields_before,
    );
    assert_eq!(
        padded_source_data
            .iter()
            .map(|value| (value.tag, value.value))
            .collect::<Vec<_>>(),
        padded_source_fields,
    );
    assert_eq!(
        changed_padded_source
            .iter()
            .map(|value| (value.tag, value.value))
            .collect::<Vec<_>>(),
        changed_padded_source_fields,
    );
    assert_eq!(CUSTOM_OUT_AS_RAW_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(CUSTOM_OUT_CLONE_CALLS.load(Ordering::SeqCst), 0);
    assert_eq!(CUSTOM_OUT_DROP_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(EQUIVALENCE_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(CLONE_CALLS.load(Ordering::SeqCst), 0);

    let source_view =
        PencilArrayView::from_slice(&source, &padded_extra, &changed_padded_source).unwrap();
    let destination_view = PencilArrayViewMut::from_slice_mut(
        &padded_destination,
        &padded_extra,
        &mut padded_repeated,
    )
    .unwrap();
    padded_plan
        .apply_views(source_view, destination_view)
        .unwrap();
    assert_oracle(
        &padded_destination,
        padded_extra.element_count(),
        [-1, 1, -1],
        [3, -4, 7],
        |extra, global| padded_probe_value(extra + 1000, global),
        &padded_repeated,
    );
    assert_eq!(CUSTOM_OUT_AS_RAW_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(CUSTOM_OUT_CLONE_CALLS.load(Ordering::SeqCst), 0);
    assert_eq!(CUSTOM_OUT_DROP_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(EQUIVALENCE_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(CLONE_CALLS.load(Ordering::SeqCst), 0);

    let mut valid_plan = SpatialRemapPlan::<Probe, 3, 2>::prepare(
        Arc::clone(&source),
        Arc::clone(&source),
        extra.clone(),
        [1, -1, 1],
        [1, 2, 3],
        Probe(0),
    )
    .unwrap();
    let source_data = make_values(&source, extra.element_count(), probe_value);

    if size > 1 {
        let sign_mismatch = SpatialRemapPlan::<Probe, 3, 2>::prepare(
            Arc::clone(&source),
            Arc::clone(&source),
            extra.clone(),
            if rank == 0 { [-1, -1, 1] } else { [1, -1, 1] },
            [1, 2, 3],
            Probe(0),
        );
        assert!(matches!(
            sign_mismatch,
            Err(SpatialRemapError::CollectiveDescriptorMismatch)
        ));

        let offset_mismatch = SpatialRemapPlan::<Probe, 3, 2>::prepare(
            Arc::clone(&source),
            Arc::clone(&source),
            extra.clone(),
            [1, -1, 1],
            [
                if rank == 0 {
                    i128::MIN
                } else {
                    i128::MIN + (5i128 << 64)
                },
                2,
                3,
            ],
            Probe(0),
        );
        assert!(matches!(
            offset_mismatch,
            Err(SpatialRemapError::CollectiveDescriptorMismatch)
        ));

        let extra_rank_mismatch = SpatialRemapPlan::<Probe, 3, 2>::prepare(
            Arc::clone(&source),
            Arc::clone(&source),
            if rank == 0 {
                ExtraShape::new([2, 3]).unwrap()
            } else {
                ExtraShape::new([6]).unwrap()
            },
            [1, -1, 1],
            [1, 2, 3],
            Probe(0),
        );
        assert!(matches!(
            extra_rank_mismatch,
            Err(SpatialRemapError::CollectiveDescriptorMismatch)
        ));

        let type_mismatch = if rank == 0 {
            SpatialRemapPlan::<i64, 3, 2>::prepare(
                Arc::clone(&source),
                Arc::clone(&source),
                extra.clone(),
                [1, 1, 1],
                [0; 3],
                0,
            )
            .map(|_| ())
        } else {
            SpatialRemapPlan::<f64, 3, 2>::prepare(
                Arc::clone(&source),
                Arc::clone(&source),
                extra.clone(),
                [1, 1, 1],
                [0; 3],
                0.0,
            )
            .map(|_| ())
        };
        assert!(matches!(
            type_mismatch,
            Err(SpatialRemapError::CollectiveDescriptorMismatch)
        ));
    }

    let zero_sized = SpatialRemapPlan::<ZeroSized, 3, 2>::prepare(
        Arc::clone(&source),
        Arc::clone(&source),
        extra.clone(),
        [1; 3],
        [0; 3],
        ZeroSized,
    );
    assert!(matches!(
        zero_sized,
        Err(SpatialRemapError::UnsupportedDatatype)
    ));

    let invalid_sign = SpatialRemapPlan::<Probe, 3, 2>::prepare(
        Arc::clone(&source),
        Arc::clone(&source),
        extra.clone(),
        [1, 0, -1],
        [0; 3],
        Probe(0),
    );
    assert!(matches!(
        invalid_sign,
        Err(SpatialRemapError::InvalidAffineSign { axis: 1, sign: 0 })
    ));

    let different_shape = Pencil::<3, 2>::new(Arc::clone(&topology), [5, 7, 5], [0, 1]).unwrap();
    let shape_error = SpatialRemapPlan::<Probe, 3, 2>::prepare(
        Arc::clone(&source),
        different_shape,
        extra.clone(),
        [1; 3],
        [0; 3],
        Probe(0),
    );
    assert!(matches!(
        shape_error,
        Err(SpatialRemapError::DifferentShape)
    ));

    let mut destination_data = vec![Probe(55); source.local_len() * extra.element_count()];
    let before_destination = destination_data.clone();
    let wrong_destination = Pencil::<3, 2>::new_permuted(
        Arc::clone(&topology),
        *source.global_shape(),
        [0, 1],
        AxisPermutation::new([1, 0, 2]).unwrap(),
    )
    .unwrap();
    let wrong_extra = ExtraShape::new([3, 2]).unwrap();
    let source_view = PencilArrayView::from_slice(&source, &extra, &source_data).unwrap();
    EQUIVALENCE_CALLS.store(0, Ordering::SeqCst);
    CLONE_CALLS.store(0, Ordering::SeqCst);
    let (layout_or_extra_error, allocation_counts) = measured(|| {
        if rank == 0 {
            let destination_view = PencilArrayViewMut::from_slice_mut(
                &wrong_destination,
                &wrong_extra,
                &mut destination_data,
            )
            .unwrap();
            valid_plan.apply_views(source_view, destination_view)
        } else {
            let destination_view =
                PencilArrayViewMut::from_slice_mut(&source, &extra, &mut destination_data).unwrap();
            valid_plan.apply_views(source_view, destination_view)
        }
    });
    assert_eq!(allocation_counts, [0, 0, 0], "failed apply allocated");
    assert_eq!(CLONE_CALLS.load(Ordering::SeqCst), 0);
    assert_eq!(EQUIVALENCE_CALLS.load(Ordering::SeqCst), 0);
    mismatch_result(
        layout_or_extra_error,
        rank,
        SpatialRemapError::DifferentExtraShape,
    );
    assert_eq!(destination_data, before_destination);

    let source_view = PencilArrayView::from_slice(&source, &extra, &source_data).unwrap();
    let layout_error = if rank == 0 {
        let destination_view =
            PencilArrayViewMut::from_slice_mut(&wrong_destination, &extra, &mut destination_data)
                .unwrap();
        valid_plan.apply_views(source_view, destination_view)
    } else {
        let destination_view =
            PencilArrayViewMut::from_slice_mut(&source, &extra, &mut destination_data).unwrap();
        valid_plan.apply_views(source_view, destination_view)
    };
    mismatch_result(layout_error, rank, SpatialRemapError::LayoutMismatch);
    assert_eq!(destination_data, before_destination);

    if size > 1 {
        let mut alternate_plan = SpatialRemapPlan::<Probe, 3, 2>::prepare(
            Arc::clone(&source),
            Arc::clone(&source),
            extra.clone(),
            [1; 3],
            [0; 3],
            Probe(0),
        )
        .unwrap();
        let before_plan_mismatch = destination_data.clone();
        let source_view = PencilArrayView::from_slice(&source, &extra, &source_data).unwrap();
        let plan_mismatch = if rank % 2 == 0 {
            let destination_view =
                PencilArrayViewMut::from_slice_mut(&source, &extra, &mut destination_data).unwrap();
            valid_plan.apply_views(source_view, destination_view)
        } else {
            let destination_view =
                PencilArrayViewMut::from_slice_mut(&source, &extra, &mut destination_data).unwrap();
            alternate_plan.apply_views(source_view, destination_view)
        };
        assert!(matches!(
            plan_mismatch,
            Err(SpatialRemapError::CollectiveDescriptorMismatch)
        ));
        assert_eq!(destination_data, before_plan_mismatch);

        // A prepare/apply disagreement is rejected by the common fixed-size header.
        let mixed_operation = if rank == 0 {
            SpatialRemapPlan::<Probe, 3, 2>::prepare(
                Arc::clone(&source),
                Arc::clone(&source),
                extra.clone(),
                [1, -1, 1],
                [1, 2, 3],
                Probe(0),
            )
            .map(|_| ())
        } else {
            let source_view = PencilArrayView::from_slice(&source, &extra, &source_data).unwrap();
            let destination_view =
                PencilArrayViewMut::from_slice_mut(&source, &extra, &mut destination_data).unwrap();
            valid_plan.apply_views(source_view, destination_view)
        };
        assert!(matches!(
            mixed_operation,
            Err(SpatialRemapError::CollectiveDescriptorMismatch)
        ));

        // Different const-generic N values still use the same fixed-size header.
        {
            let one_dimensional = MpiTopology::<1>::new(&world, [size]).unwrap();
            let pencil_1d = Pencil::<1, 1>::new(Arc::clone(&one_dimensional), [5], [0]).unwrap();
            let pencil_2d = Pencil::<2, 1>::new(Arc::clone(&one_dimensional), [3, 5], [0]).unwrap();
            let result = if rank == 0 {
                SpatialRemapPlan::<i64, 1, 1>::prepare(
                    Arc::clone(&pencil_1d),
                    Arc::clone(&pencil_1d),
                    ExtraShape::scalar(),
                    [1],
                    [0],
                    0,
                )
                .map(|_| ())
            } else {
                SpatialRemapPlan::<i64, 2, 1>::prepare(
                    Arc::clone(&pencil_2d),
                    Arc::clone(&pencil_2d),
                    ExtraShape::scalar(),
                    [1, 1],
                    [0, 0],
                    0,
                )
                .map(|_| ())
            };
            assert!(matches!(
                result,
                Err(SpatialRemapError::CollectiveDescriptorMismatch)
            ));
        }
    }

    // A failed apply is followed by an ordinary successful collective.
    let mut destination_data = vec![Probe(0); source.local_len() * extra.element_count()];
    let source_view = PencilArrayView::from_slice(&source, &extra, &source_data).unwrap();
    let destination_view =
        PencilArrayViewMut::from_slice_mut(&source, &extra, &mut destination_data).unwrap();
    valid_plan
        .apply_views(source_view, destination_view)
        .unwrap();
    assert_oracle(
        &source,
        extra.element_count(),
        [1, -1, 1],
        [1, 2, 3],
        probe_value,
        &destination_data,
    );
}
