//! MPI regression checks for the local payload byte-count preflight.
//!
//! The module owns a test-only global allocator probe.  It observes Rust
//! allocations on the test thread and can fail one exact-size request on one
//! rank; native MPI/HDF5 allocations are outside its scope.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use mpi::collective::{CommunicatorCollectives, SystemOperation};
use mpi::topology::{Communicator, SimpleCommunicator};
use pencil_array::{ExtraShape, MpiTopology, Pencil, PencilArray};

use crate::mpi_io::{
    LocalByteLimitGuard, MPI_LOCAL_BYTE_LIMIT_WHAT, agree_mpi_preparation, build_layout,
    mpi_preparation_error,
};
use crate::{
    CollectionIoError, IoError, MpiFileSession, MpiIoMode, MpiIoOptions, MpiSessionError,
    NamedIoError, RawReadOptions,
};

mod support {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/mod.rs"));
}
use support::{cleanup_owned_temp_dir, owned_temp_dir, reset_file};

const MAIN_COLUMNS: usize = 16_384;
const COLLECTION_COLUMNS: usize = 8_192;
const BOUNDARY_COLUMNS: usize = 24_576;
const BOUNDARY_COLLECTION_COLUMNS: usize = 12_288;
const EMPTY_COLUMNS: usize = 16_384;
const TEST_LIMIT: usize = 196_608;
const PROBE_THRESHOLD: usize = 131_072;
const EMPTY_LIMIT: usize = 65_536;

// The probe must not allocate, log, or panic while the allocator is active.
struct Probe {
    enabled: Cell<bool>,
    threshold: Cell<usize>,
    matching: Cell<usize>,
    largest: Cell<usize>,
    inject_size: Cell<Option<usize>>,
    injected: Cell<bool>,
    reached: Cell<bool>,
    injected_size: Cell<Option<usize>>,
}

impl Probe {
    const fn new() -> Self {
        Self {
            enabled: Cell::new(false),
            threshold: Cell::new(0),
            matching: Cell::new(0),
            largest: Cell::new(0),
            inject_size: Cell::new(None),
            injected: Cell::new(false),
            reached: Cell::new(false),
            injected_size: Cell::new(None),
        }
    }

    fn start(&self, threshold: usize, inject_size: Option<usize>) {
        self.enabled.set(true);
        self.threshold.set(threshold);
        self.matching.set(0);
        self.largest.set(0);
        self.inject_size.set(inject_size);
        self.injected.set(false);
        self.reached.set(false);
        self.injected_size.set(None);
    }

    fn observe(&self, size: usize) -> bool {
        if !self.enabled.get() {
            return false;
        }
        if size >= self.threshold.get() {
            self.matching.set(self.matching.get() + 1);
        }
        if size > self.largest.get() {
            self.largest.set(size);
        }
        if !self.injected.get() && self.inject_size.get() == Some(size) {
            self.injected.set(true);
            self.reached.set(true);
            self.injected_size.set(Some(size));
            return true;
        }
        false
    }

    fn stop(&self) -> Snapshot {
        self.enabled.set(false);
        Snapshot {
            matching: self.matching.get(),
            largest: self.largest.get(),
            reached: self.reached.get(),
            injected_size: self.injected_size.get(),
        }
    }
}

thread_local! {
    static PROBE: Probe = const { Probe::new() };
}

#[derive(Clone, Copy)]
struct Snapshot {
    matching: usize,
    largest: usize,
    reached: bool,
    injected_size: Option<usize>,
}

struct ProbeGuard {
    active: bool,
}

impl ProbeGuard {
    fn new(threshold: usize, inject_size: Option<usize>) -> Self {
        PROBE.with(|probe| probe.start(threshold, inject_size));
        Self { active: true }
    }

    fn finish(mut self) -> Snapshot {
        self.active = false;
        PROBE.with(Probe::stop)
    }
}

impl Drop for ProbeGuard {
    fn drop(&mut self) {
        if self.active {
            PROBE.with(|probe| {
                probe.stop();
            });
        }
    }
}

struct ProbeAllocator;

unsafe impl GlobalAlloc for ProbeAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let fail = PROBE
            .try_with(|probe| probe.observe(layout.size()))
            .unwrap_or(false);
        if fail {
            return std::ptr::null_mut();
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let fail = PROBE
            .try_with(|probe| probe.observe(layout.size()))
            .unwrap_or(false);
        if fail {
            return std::ptr::null_mut();
        }
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let fail = PROBE
            .try_with(|probe| probe.observe(new_size))
            .unwrap_or(false);
        if fail {
            return std::ptr::null_mut();
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: ProbeAllocator = ProbeAllocator;

struct Rig<'a> {
    world: &'a SimpleCommunicator,
    directory: &'a Path,
    topology: &'a Arc<MpiTopology<1>>,
    main: &'a PencilArray<f64, 2, 1>,
    boundary: &'a PencilArray<f64, 2, 1>,
    main_shape: [usize; 2],
    boundary_shape: [usize; 2],
    main_bytes: usize,
    boundary_bytes: usize,
}

fn is_local_limit(error: &IoError) -> bool {
    matches!(
        error,
        IoError::SizeLimit { what } if *what == MPI_LOCAL_BYTE_LIMIT_WHAT
    )
}

fn is_named_limit(error: &NamedIoError) -> bool {
    matches!(error, NamedIoError::Io(inner) if is_local_limit(inner))
}

fn is_session_limit(error: &MpiSessionError) -> bool {
    matches!(error, MpiSessionError::Named(inner) if is_named_limit(inner))
}

fn is_collection_limit(error: &CollectionIoError) -> bool {
    matches!(error, CollectionIoError::Io(inner) if is_local_limit(inner))
}

fn collective_and<C: CommunicatorCollectives>(world: &C, local_ok: bool) -> bool {
    let local = i32::from(local_ok);
    let mut all = 0;
    world.all_reduce_into(&local, &mut all, SystemOperation::min());
    all == 1
}

fn all_ranks<C: CommunicatorCollectives>(world: &C, local_ok: bool, what: &str) {
    assert!(collective_and(world, local_ok), "{what}");
}

fn all_sum<C: CommunicatorCollectives>(world: &C, value: u64) -> u64 {
    let mut total = 0;
    world.all_reduce_into(&value, &mut total, SystemOperation::sum());
    total
}

fn modes() -> [MpiIoOptions; 2] {
    [
        MpiIoOptions::default().mode(MpiIoMode::Collective),
        MpiIoOptions::default().mode(MpiIoMode::Independent),
    ]
}

fn unlimited<T>(f: impl FnOnce() -> T) -> T {
    let _guard = LocalByteLimitGuard::new(usize::MAX);
    f()
}

fn fill(array: &mut PencilArray<f64, 2, 1>, base: f64) {
    let mut view = array.view_mut();
    let ranges = view.pencil().local_ranges().clone();
    for x in 0..ranges[0].len() {
        for y in 0..ranges[1].len() {
            let value = base + ((ranges[0].start + x) * 1000 + ranges[1].start + y) as f64;
            *view.get_local_mut(&[], [x, y]).unwrap() = value;
        }
    }
}

fn array_for(
    topology: &Arc<MpiTopology<1>>,
    shape: [usize; 2],
    base: f64,
) -> PencilArray<f64, 2, 1> {
    let pencil = Pencil::<2, 1>::new(topology.clone(), shape, [0]).unwrap();
    let mut array = PencilArray::from_elem(pencil, ExtraShape::scalar(), base).unwrap();
    fill(&mut array, base);
    array
}

fn raw_global_bytes(shape: [usize; 2], base: f64) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(shape[0] * shape[1] * std::mem::size_of::<f64>());
    for x in 0..shape[0] {
        for y in 0..shape[1] {
            bytes.extend_from_slice(&(base + (x * 1000 + y) as f64).to_le_bytes());
        }
    }
    bytes
}

fn write_raw_file(rig: &Rig<'_>, path: &Path, shape: [usize; 2], base: f64) {
    reset_file(rig.world, path);
    support::root_status(rig.world, || {
        std::fs::write(path, raw_global_bytes(shape, base)).map_err(|error| error.to_string())
    });
    rig.world.barrier();
}

fn rejected<C, E, F, P>(world: &C, label: &str, operation: F, expected: P) -> Result<(), E>
where
    C: CommunicatorCollectives,
    F: FnOnce() -> Result<(), E>,
    P: Fn(&E) -> bool,
{
    let duplicates = crate::ffi::DUPLICATE_CALLS.with(Cell::get);
    let file_calls = crate::ffi::FILE_CALLS.with(Cell::get);
    let at_calls = crate::ffi::test_at_calls();
    let payload_calls = crate::ffi::test_payload_calls();
    let probe = ProbeGuard::new(PROBE_THRESHOLD, None);
    let result = operation();
    let snapshot = probe.finish();
    let exact = result.as_ref().err().is_some_and(&expected);
    all_ranks(world, exact, &format!("{label}: exact limit error"));
    all_ranks(
        world,
        snapshot.matching == 0,
        &format!("{label}: no >= threshold allocation"),
    );
    all_ranks(
        world,
        !snapshot.reached,
        &format!("{label}: no allocation injection"),
    );
    all_ranks(
        world,
        duplicates == crate::ffi::DUPLICATE_CALLS.with(Cell::get)
            && file_calls == crate::ffi::FILE_CALLS.with(Cell::get)
            && at_calls == crate::ffi::test_at_calls()
            && payload_calls == crate::ffi::test_payload_calls(),
        &format!("{label}: no native or payload call"),
    );
    result
}

pub(crate) fn run() {
    let universe = mpi::initialize().expect("MPI must initialize once");
    let world = universe.world();
    let size = usize::try_from(world.size()).unwrap();
    let topology = MpiTopology::<1>::new(&world, [size]).unwrap();
    let directory = owned_temp_dir(&world, "pencil-i39-preflight");

    let main_shape = [size + 1, MAIN_COLUMNS];
    let boundary_shape = [size, BOUNDARY_COLUMNS];
    let main = array_for(&topology, main_shape, 0.0);
    let boundary = array_for(&topology, boundary_shape, 1_000_000.0);
    let main_bytes = main.view().len() * std::mem::size_of::<f64>();
    let boundary_bytes = boundary.view().len() * std::mem::size_of::<f64>();
    assert_eq!(boundary_bytes, TEST_LIMIT, "boundary fixture bytes");
    let rig = Rig {
        world: &world,
        directory: &directory,
        topology: &topology,
        main: &main,
        boundary: &boundary,
        main_shape,
        boundary_shape,
        main_bytes,
        boundary_bytes,
    };

    agreement_checks(&world);
    metadata_allocation_failure(&rig);
    positive_controls(&rig);
    normal_checks(&rig);
    named_checks(&rig);
    raw_checks(&rig);
    chunked_checks(&rig);
    collection_checks(&rig);
    session_checks(&rig);
    empty_checks(&rig);
    hdf5_checks(&rig);

    cleanup_owned_temp_dir(&world, &directory);
}

fn agreement_checks(world: &SimpleCommunicator) {
    all_ranks(
        world,
        agree_mpi_preparation(world, None, "three-state test").is_ok(),
        "three-state all-success",
    );

    let other = if world.rank() == 0 {
        Some(IoError::AllocationFailed { requested: 1 })
    } else {
        None
    };
    let other = agree_mpi_preparation(world, other.as_ref(), "three-state other");
    all_ranks(
        world,
        matches!(
            other,
            Err(IoError::CollectivePrecondition {
                phase: "three-state other"
            })
        ),
        "three-state one other error",
    );

    let size_limit = if world.rank() == 0 {
        Some(IoError::SizeLimit {
            what: MPI_LOCAL_BYTE_LIMIT_WHAT,
        })
    } else {
        None
    };
    let size_limit = agree_mpi_preparation(world, size_limit.as_ref(), "three-state size");
    all_ranks(
        world,
        matches!(&size_limit, Err(error) if is_local_limit(error)),
        "three-state one size-limit error",
    );

    if world.size() > 1 {
        let mixed = match world.rank() {
            0 => Some(IoError::AllocationFailed { requested: 1 }),
            1 => Some(IoError::SizeLimit {
                what: MPI_LOCAL_BYTE_LIMIT_WHAT,
            }),
            _ => None,
        };
        let mixed = agree_mpi_preparation(world, mixed.as_ref(), "three-state mixed");
        all_ranks(
            world,
            matches!(&mixed, Err(error) if is_local_limit(error)),
            "three-state state 1 and 2 mix",
        );
    }

    let local = IoError::AllocationFailed { requested: 7 };
    assert!(matches!(
        mpi_preparation_error(
            Some(local.clone()),
            IoError::SizeLimit {
                what: MPI_LOCAL_BYTE_LIMIT_WHAT
            }
        ),
        IoError::SizeLimit { what } if what == MPI_LOCAL_BYTE_LIMIT_WHAT
    ));
    assert!(matches!(
        mpi_preparation_error(
            Some(local),
            IoError::CollectivePrecondition { phase: "local" }
        ),
        IoError::AllocationFailed { requested: 7 }
    ));
}

fn positive_controls(rig: &Rig<'_>) {
    let packed = {
        let probe = ProbeGuard::new(PROBE_THRESHOLD, None);
        let packed = crate::format::pack_view(&rig.main.view()).unwrap();
        let snapshot = probe.finish();
        all_ranks(
            rig.world,
            snapshot.matching > 0 && snapshot.largest >= rig.main_bytes,
            "pack_view allocation is visible to the probe",
        );
        packed
    };
    all_ranks(
        rig.world,
        packed.len() == rig.main_bytes,
        "packed main payload bytes",
    );

    let path = rig.directory.join("main-normal.pio");
    reset_file(rig.world, &path);
    let written = crate::write_mpi(&path, rig.main.view());
    all_ranks(rig.world, written.is_ok(), "normal write positive control");

    let pencil = Pencil::<2, 1>::new(rig.topology.clone(), rig.main_shape, [0]).unwrap();
    let mut destination = PencilArray::from_elem(pencil, ExtraShape::scalar(), -7.0).unwrap();
    let read = crate::read_mpi(&path, destination.view_mut());
    all_ranks(rig.world, read.is_ok(), "normal read positive control");
    all_ranks(
        rig.world,
        destination.as_slice() == rig.main.as_slice(),
        "normal read values",
    );
}

fn normal_checks(rig: &Rig<'_>) {
    let _limit = LocalByteLimitGuard::new(TEST_LIMIT);
    let main_path = rig.directory.join("main-normal.pio");

    for (index, options) in modes().iter().enumerate() {
        let path = rig
            .directory
            .join(format!("rejected-normal-write-{index}.pio"));
        reset_file(rig.world, &path);
        let result = rejected(
            rig.world,
            "normal write options",
            || crate::write_mpi_with_options(&path, rig.main.view(), options),
            is_local_limit,
        );
        all_ranks(rig.world, result.is_err(), "normal write options rejected");
        all_ranks(
            rig.world,
            !path.exists(),
            "normal write options path absent",
        );
    }
    let path = rig.directory.join("rejected-normal-write.pio");
    reset_file(rig.world, &path);
    let result = rejected(
        rig.world,
        "normal write",
        || crate::write_mpi(&path, rig.main.view()),
        is_local_limit,
    );
    all_ranks(rig.world, result.is_err(), "normal write rejected");
    all_ranks(rig.world, !path.exists(), "normal write path absent");

    let pencil = Pencil::<2, 1>::new(rig.topology.clone(), rig.main_shape, [0]).unwrap();
    let mut destination =
        PencilArray::from_elem(pencil.clone(), ExtraShape::scalar(), -7.0).unwrap();
    let before = destination.as_slice().to_vec();
    let result = rejected(
        rig.world,
        "normal read",
        || crate::read_mpi(&main_path, destination.view_mut()),
        is_local_limit,
    );
    all_ranks(rig.world, result.is_err(), "normal read rejected");
    all_ranks(
        rig.world,
        destination.as_slice() == before.as_slice(),
        "normal read destination unchanged",
    );
    for (index, options) in modes().iter().enumerate() {
        let mut destination =
            PencilArray::from_elem(pencil.clone(), ExtraShape::scalar(), -7.0).unwrap();
        let before = destination.as_slice().to_vec();
        let result = rejected(
            rig.world,
            &format!("normal read options {index}"),
            || crate::read_mpi_with_options(&main_path, destination.view_mut(), options),
            is_local_limit,
        );
        all_ranks(rig.world, result.is_err(), "normal read options rejected");
        all_ranks(
            rig.world,
            destination.as_slice() == before.as_slice(),
            "normal read options destination unchanged",
        );
    }

    let boundary_path = rig.directory.join("boundary-normal.pio");
    reset_file(rig.world, &boundary_path);
    all_ranks(
        rig.world,
        crate::write_mpi(&boundary_path, rig.boundary.view()).is_ok(),
        "normal boundary write",
    );
    let boundary_pencil =
        Pencil::<2, 1>::new(rig.topology.clone(), rig.boundary_shape, [0]).unwrap();
    let mut destination =
        PencilArray::from_elem(boundary_pencil.clone(), ExtraShape::scalar(), -7.0).unwrap();
    let boundary_read = crate::read_mpi(&boundary_path, destination.view_mut());
    all_ranks(rig.world, boundary_read.is_ok(), "normal boundary read");
    all_ranks(
        rig.world,
        destination.as_slice() == rig.boundary.as_slice(),
        "normal boundary values",
    );

    let injected_path = rig.directory.join("injected-normal-write.pio");
    reset_file(rig.world, &injected_path);
    let (result, snapshot) = {
        let probe = ProbeGuard::new(
            PROBE_THRESHOLD,
            (rig.world.rank() == 0).then_some(rig.boundary_bytes),
        );
        let result = crate::write_mpi(&injected_path, rig.boundary.view());
        (result, probe.finish())
    };
    assert_injected_allocation(rig, "normal write pack", &snapshot, rig.boundary_bytes);
    all_ranks(
        rig.world,
        match (&result, rig.world.rank() == 0) {
            (Err(IoError::AllocationFailed { requested }), true) => {
                *requested == rig.boundary_bytes
            }
            (Err(IoError::CollectivePrecondition { .. }), false) => true,
            _ => false,
        },
        "normal write pack failure variants",
    );
    all_ranks(
        rig.world,
        !injected_path.exists(),
        "normal write pack failure path absent",
    );

    let mut destination =
        PencilArray::from_elem(boundary_pencil, ExtraShape::scalar(), -7.0).unwrap();
    let before = destination.as_slice().to_vec();
    let payload_calls = crate::ffi::test_payload_calls();
    let (result, snapshot) = {
        let probe = ProbeGuard::new(
            PROBE_THRESHOLD,
            (rig.world.rank() == 0).then_some(rig.boundary_bytes),
        );
        let result = crate::read_mpi(&boundary_path, destination.view_mut());
        (result, probe.finish())
    };
    assert_injected_allocation(rig, "normal read staging", &snapshot, rig.boundary_bytes);
    all_ranks(
        rig.world,
        match (&result, rig.world.rank() == 0) {
            (Err(IoError::AllocationFailed { requested }), true) => {
                *requested == rig.boundary_bytes
            }
            (Err(IoError::CollectivePrecondition { .. }), false) => true,
            _ => false,
        },
        "normal read staging failure variants",
    );
    all_ranks(
        rig.world,
        destination.as_slice() == before.as_slice(),
        "normal read staging destination unchanged",
    );
    all_ranks(
        rig.world,
        payload_calls == crate::ffi::test_payload_calls(),
        "normal read staging has no payload call",
    );
}

fn assert_injected_allocation(rig: &Rig<'_>, label: &str, snapshot: &Snapshot, bytes: usize) {
    let local_ok = if rig.world.rank() == 0 {
        snapshot.reached && snapshot.injected_size == Some(bytes)
    } else {
        !snapshot.reached
    };
    all_ranks(rig.world, local_ok, label);
}

fn named_checks(rig: &Rig<'_>) {
    let _limit = LocalByteLimitGuard::new(TEST_LIMIT);
    let small = array_for(rig.topology, [rig.world.size() as usize, 512], 100.0);
    let base_path = rig.directory.join("named-base.pio");
    reset_file(rig.world, &base_path);
    all_ranks(
        rig.world,
        crate::write_mpi_named(&base_path, "base", small.view()).is_ok(),
        "named base write",
    );
    let base_bytes = support::root_read(rig.world, &base_path);

    let created = rig.directory.join("named-rejected-create.pio");
    reset_file(rig.world, &created);
    let result = rejected(
        rig.world,
        "named create",
        || crate::write_mpi_named(&created, "big", rig.main.view()),
        is_named_limit,
    );
    all_ranks(rig.world, result.is_err(), "named create rejected");
    all_ranks(rig.world, !created.exists(), "named create path absent");
    for (index, options) in modes().iter().enumerate() {
        let path = rig
            .directory
            .join(format!("named-rejected-create-{index}.pio"));
        reset_file(rig.world, &path);
        let result = rejected(
            rig.world,
            "named create options",
            || crate::write_mpi_named_with_options(&path, "big", rig.main.view(), options),
            is_named_limit,
        );
        all_ranks(rig.world, result.is_err(), "named create options rejected");
        all_ranks(
            rig.world,
            !path.exists(),
            "named create options path absent",
        );
    }

    let result = rejected(
        rig.world,
        "named append",
        || crate::append_mpi_named(&base_path, "big", rig.main.view()),
        is_named_limit,
    );
    all_ranks(rig.world, result.is_err(), "named append rejected");
    all_ranks(
        rig.world,
        support::root_read(rig.world, &base_path) == base_bytes,
        "named append file unchanged",
    );
    for (index, options) in modes().iter().enumerate() {
        let result = rejected(
            rig.world,
            &format!("named append options {index}"),
            || crate::append_mpi_named_with_options(&base_path, "big", rig.main.view(), options),
            is_named_limit,
        );
        all_ranks(
            rig.world,
            result.is_err(),
            &format!("named append options {index} rejected"),
        );
        all_ranks(
            rig.world,
            support::root_read(rig.world, &base_path) == base_bytes,
            "named append options file unchanged",
        );
    }
    all_ranks(
        rig.world,
        crate::append_mpi_named(&base_path, "after", small.view()).is_ok(),
        "named append reuse",
    );
    let mut after_destination = PencilArray::from_elem(
        Pencil::<2, 1>::new(rig.topology.clone(), [rig.world.size() as usize, 512], [0]).unwrap(),
        ExtraShape::scalar(),
        -7.0,
    )
    .unwrap();
    all_ranks(
        rig.world,
        crate::read_mpi_named(&base_path, "after", after_destination.view_mut()).is_ok()
            && after_destination.as_slice() == small.as_slice(),
        "named append reuse read",
    );

    let pencil = Pencil::<2, 1>::new(rig.topology.clone(), rig.main_shape, [0]).unwrap();
    let named_main_path = rig.directory.join("named-main.pio");
    reset_file(rig.world, &named_main_path);
    all_ranks(
        rig.world,
        unlimited(|| crate::write_mpi_named(&named_main_path, "main", rig.main.view())).is_ok(),
        "named main fixture write",
    );
    let mut destination =
        PencilArray::from_elem(pencil.clone(), ExtraShape::scalar(), -7.0).unwrap();
    all_ranks(
        rig.world,
        unlimited(|| crate::read_mpi_named(&named_main_path, "main", destination.view_mut()))
            .is_ok()
            && destination.as_slice() == rig.main.as_slice(),
        "named main fixture read",
    );
    let before = destination.as_slice().to_vec();
    let result = rejected(
        rig.world,
        "named read",
        || crate::read_mpi_named(&named_main_path, "main", destination.view_mut()),
        is_named_limit,
    );
    all_ranks(rig.world, result.is_err(), "named read rejected");
    all_ranks(
        rig.world,
        destination.as_slice() == before.as_slice(),
        "named read destination unchanged",
    );
    for (index, options) in modes().iter().enumerate() {
        let mut destination =
            PencilArray::from_elem(pencil.clone(), ExtraShape::scalar(), -7.0).unwrap();
        let before = destination.as_slice().to_vec();
        let result = rejected(
            rig.world,
            &format!("named read options {index}"),
            || {
                crate::read_mpi_named_with_options(
                    &named_main_path,
                    "main",
                    destination.view_mut(),
                    options,
                )
            },
            is_named_limit,
        );
        all_ranks(
            rig.world,
            result.is_err(),
            &format!("named read options {index} rejected"),
        );
        all_ranks(
            rig.world,
            destination.as_slice() == before.as_slice(),
            "named read options destination unchanged",
        );
    }

    let boundary_path = rig.directory.join("named-boundary.pio");
    reset_file(rig.world, &boundary_path);
    all_ranks(
        rig.world,
        crate::write_mpi_named(&boundary_path, "boundary", rig.boundary.view()).is_ok(),
        "named boundary write",
    );
    let boundary_pencil =
        Pencil::<2, 1>::new(rig.topology.clone(), rig.boundary_shape, [0]).unwrap();
    let mut destination =
        PencilArray::from_elem(boundary_pencil.clone(), ExtraShape::scalar(), -7.0).unwrap();
    all_ranks(
        rig.world,
        crate::read_mpi_named(&boundary_path, "boundary", destination.view_mut()).is_ok(),
        "named boundary read",
    );
    all_ranks(
        rig.world,
        destination.as_slice() == rig.boundary.as_slice(),
        "named boundary values",
    );

    let injected_path = rig.directory.join("named-injected.pio");
    reset_file(rig.world, &injected_path);
    let (result, snapshot) = {
        let probe = ProbeGuard::new(
            PROBE_THRESHOLD,
            (rig.world.rank() == 0).then_some(rig.boundary_bytes),
        );
        let result = crate::write_mpi_named(&injected_path, "injected", rig.boundary.view());
        (result, probe.finish())
    };
    assert_injected_allocation(rig, "named write pack", &snapshot, rig.boundary_bytes);
    all_ranks(
        rig.world,
        matches!(
            &result,
            Err(NamedIoError::Io(IoError::CollectivePrecondition {
                phase: "named payload packing"
            }))
        ),
        "named write pack agreement error",
    );
    all_ranks(
        rig.world,
        !injected_path.exists(),
        "named write pack path absent",
    );

    let mut destination =
        PencilArray::from_elem(boundary_pencil, ExtraShape::scalar(), -7.0).unwrap();
    let before = destination.as_slice().to_vec();
    let (result, snapshot) = {
        let probe = ProbeGuard::new(
            PROBE_THRESHOLD,
            (rig.world.rank() == 0).then_some(rig.boundary_bytes),
        );
        let result = crate::read_mpi_named(&boundary_path, "boundary", destination.view_mut());
        (result, probe.finish())
    };
    assert_injected_allocation(rig, "named read allocation", &snapshot, rig.boundary_bytes);
    all_ranks(
        rig.world,
        matches!(
            &result,
            Err(NamedIoError::Io(IoError::CollectivePrecondition {
                phase: "named allocation"
            }))
        ),
        "named read allocation agreement error",
    );
    all_ranks(
        rig.world,
        destination.as_slice() == before.as_slice(),
        "named read allocation destination unchanged",
    );
}

fn raw_checks(rig: &Rig<'_>) {
    let main_raw = rig.directory.join("main-raw.bin");
    write_raw_file(rig, &main_raw, rig.main_shape, 0.0);
    let boundary_raw = rig.directory.join("boundary-raw.bin");
    write_raw_file(rig, &boundary_raw, rig.boundary_shape, 1_000_000.0);

    let _limit = LocalByteLimitGuard::new(TEST_LIMIT);
    let pencil = Pencil::<2, 1>::new(rig.topology.clone(), rig.main_shape, [0]).unwrap();
    let mut destination =
        PencilArray::from_elem(pencil.clone(), ExtraShape::scalar(), -7.0).unwrap();
    let before = destination.as_slice().to_vec();
    let result = rejected(
        rig.world,
        "raw read",
        || crate::read_mpi_raw(&main_raw, destination.view_mut(), RawReadOptions::default()),
        is_local_limit,
    );
    all_ranks(rig.world, result.is_err(), "raw read rejected");
    all_ranks(
        rig.world,
        destination.as_slice() == before.as_slice(),
        "raw read destination unchanged",
    );
    for (index, options) in modes().iter().enumerate() {
        let mut destination =
            PencilArray::from_elem(pencil.clone(), ExtraShape::scalar(), -7.0).unwrap();
        let before = destination.as_slice().to_vec();
        let read_options = RawReadOptions::default().mode(options.mode);
        let result = rejected(
            rig.world,
            "raw read options",
            || crate::read_mpi_raw(&main_raw, destination.view_mut(), read_options),
            is_local_limit,
        );
        all_ranks(
            rig.world,
            result.is_err(),
            &format!("raw read options {index} rejected"),
        );
        all_ranks(
            rig.world,
            destination.as_slice() == before.as_slice(),
            "raw read options destination unchanged",
        );
    }

    let boundary_pencil =
        Pencil::<2, 1>::new(rig.topology.clone(), rig.boundary_shape, [0]).unwrap();
    let mut destination =
        PencilArray::from_elem(boundary_pencil.clone(), ExtraShape::scalar(), -7.0).unwrap();
    all_ranks(
        rig.world,
        crate::read_mpi_raw(
            &boundary_raw,
            destination.view_mut(),
            RawReadOptions::default(),
        )
        .is_ok(),
        "raw boundary read",
    );
    all_ranks(
        rig.world,
        destination.as_slice() == rig.boundary.as_slice(),
        "raw boundary values",
    );

    let mut destination =
        PencilArray::from_elem(boundary_pencil.clone(), ExtraShape::scalar(), -7.0).unwrap();
    let before = destination.as_slice().to_vec();
    let payload_calls = crate::ffi::test_payload_calls();
    let (result, snapshot) = {
        let probe = ProbeGuard::new(
            PROBE_THRESHOLD,
            (rig.world.rank() == 0).then_some(rig.boundary_bytes),
        );
        let result = crate::read_mpi_raw(
            &boundary_raw,
            destination.view_mut(),
            RawReadOptions::default(),
        );
        (result, probe.finish())
    };
    assert_injected_allocation(rig, "raw staging", &snapshot, rig.boundary_bytes);
    all_ranks(
        rig.world,
        match (&result, rig.world.rank() == 0) {
            (Err(IoError::AllocationFailed { requested }), true) => {
                *requested == rig.boundary_bytes
            }
            (Err(IoError::CollectivePrecondition { .. }), false) => true,
            _ => false,
        },
        "raw staging failure variants",
    );
    all_ranks(
        rig.world,
        destination.as_slice() == before.as_slice(),
        "raw staging destination unchanged",
    );
    all_ranks(
        rig.world,
        payload_calls == crate::ffi::test_payload_calls(),
        "raw staging has no payload call",
    );
}

fn chunked_checks(rig: &Rig<'_>) {
    let main_chunked = rig.directory.join("main-chunked.pio");
    reset_file(rig.world, &main_chunked);
    all_ranks(
        rig.world,
        unlimited(|| {
            crate::write_mpi_chunked(&main_chunked, rig.main.view(), &MpiIoOptions::default())
        })
        .is_ok(),
        "chunked main setup write",
    );

    let _limit = LocalByteLimitGuard::new(TEST_LIMIT);
    for (index, options) in modes().iter().enumerate() {
        let path = rig
            .directory
            .join(format!("rejected-chunked-write-{index}.pio"));
        reset_file(rig.world, &path);
        let result = rejected(
            rig.world,
            "chunked write",
            || crate::write_mpi_chunked(&path, rig.main.view(), options),
            is_local_limit,
        );
        all_ranks(rig.world, result.is_err(), "chunked write rejected");
        all_ranks(rig.world, !path.exists(), "chunked write path absent");
    }

    let pencil = Pencil::<2, 1>::new(rig.topology.clone(), rig.main_shape, [0]).unwrap();
    let mut destination =
        PencilArray::from_elem(pencil.clone(), ExtraShape::scalar(), -7.0).unwrap();
    let before = destination.as_slice().to_vec();
    let result = rejected(
        rig.world,
        "chunked read",
        || {
            crate::read_mpi_chunked(
                &main_chunked,
                destination.view_mut(),
                &MpiIoOptions::default(),
            )
        },
        is_local_limit,
    );
    all_ranks(rig.world, result.is_err(), "chunked read rejected");
    all_ranks(
        rig.world,
        destination.as_slice() == before.as_slice(),
        "chunked read destination unchanged",
    );
    for (index, options) in modes().iter().enumerate() {
        let mut destination =
            PencilArray::from_elem(pencil.clone(), ExtraShape::scalar(), -7.0).unwrap();
        let before = destination.as_slice().to_vec();
        let result = rejected(
            rig.world,
            "chunked read options",
            || crate::read_mpi_chunked(&main_chunked, destination.view_mut(), options),
            is_local_limit,
        );
        all_ranks(
            rig.world,
            result.is_err(),
            &format!("chunked read options {index} rejected"),
        );
        all_ranks(
            rig.world,
            destination.as_slice() == before.as_slice(),
            "chunked read options destination unchanged",
        );
    }

    let boundary_pencil =
        Pencil::<2, 1>::new(rig.topology.clone(), rig.boundary_shape, [0]).unwrap();
    for (index, options) in modes().iter().enumerate() {
        let path = rig.directory.join(format!("boundary-chunked-{index}.pio"));
        reset_file(rig.world, &path);
        all_ranks(
            rig.world,
            crate::write_mpi_chunked(&path, rig.boundary.view(), options).is_ok(),
            "chunked boundary write",
        );
        let mut destination =
            PencilArray::from_elem(boundary_pencil.clone(), ExtraShape::scalar(), -7.0).unwrap();
        all_ranks(
            rig.world,
            crate::read_mpi_chunked(&path, destination.view_mut(), options).is_ok(),
            "chunked boundary read",
        );
        all_ranks(
            rig.world,
            destination.as_slice() == rig.boundary.as_slice(),
            "chunked boundary values",
        );
    }

    let injected_path = rig.directory.join("injected-chunked-write.pio");
    reset_file(rig.world, &injected_path);
    let (result, snapshot) = {
        let probe = ProbeGuard::new(
            PROBE_THRESHOLD,
            (rig.world.rank() == 0).then_some(rig.boundary_bytes),
        );
        let result = crate::write_mpi_chunked(
            &injected_path,
            rig.boundary.view(),
            &MpiIoOptions::default(),
        );
        (result, probe.finish())
    };
    assert_injected_allocation(rig, "chunked write buffer", &snapshot, rig.boundary_bytes);
    all_ranks(
        rig.world,
        match (&result, rig.world.rank() == 0) {
            (Err(IoError::AllocationFailed { requested }), true) => {
                *requested == rig.boundary_bytes
            }
            (Err(IoError::CollectivePrecondition { .. }), false) => true,
            _ => false,
        },
        "chunked write buffer failure variants",
    );
    all_ranks(
        rig.world,
        !injected_path.exists(),
        "chunked write buffer path absent",
    );

    let boundary_chunked = rig.directory.join("boundary-chunked-0.pio");
    let mut destination =
        PencilArray::from_elem(boundary_pencil, ExtraShape::scalar(), -7.0).unwrap();
    let before = destination.as_slice().to_vec();
    let at_calls = crate::ffi::test_at_calls();
    let (result, snapshot) = {
        let probe = ProbeGuard::new(
            PROBE_THRESHOLD,
            (rig.world.rank() == 0).then_some(rig.boundary_bytes),
        );
        let result = crate::read_mpi_chunked(
            &boundary_chunked,
            destination.view_mut(),
            &MpiIoOptions::default(),
        );
        (result, probe.finish())
    };
    assert_injected_allocation(rig, "chunked read buffer", &snapshot, rig.boundary_bytes);
    all_ranks(
        rig.world,
        match (&result, rig.world.rank() == 0) {
            (Err(IoError::AllocationFailed { requested }), true) => {
                *requested == rig.boundary_bytes
            }
            (Err(IoError::CollectivePrecondition { .. }), false) => true,
            _ => false,
        },
        "chunked read buffer failure variants",
    );
    all_ranks(
        rig.world,
        destination.as_slice() == before.as_slice(),
        "chunked read buffer destination unchanged",
    );
    all_ranks(
        rig.world,
        {
            let after = crate::ffi::test_at_calls();
            after[0] == at_calls[0]
                && after[1] == at_calls[1]
                && after[2] == at_calls[2] + 2
                && after[3] == at_calls[3]
                && after[4] < at_calls[4] + rig.boundary_bytes
        },
        "chunked read buffer only scans metadata",
    );
}

fn collection_checks(rig: &Rig<'_>) {
    let size = rig.world.size() as usize;
    let main_shape = [size + 1, COLLECTION_COLUMNS];
    let boundary_shape = [size, BOUNDARY_COLLECTION_COLUMNS];
    let main0 = array_for(rig.topology, main_shape, 2_000_000.0);
    let main1 = array_for(rig.topology, main_shape, 3_000_000.0);
    let boundary0 = array_for(rig.topology, boundary_shape, 4_000_000.0);
    let boundary1 = array_for(rig.topology, boundary_shape, 5_000_000.0);
    let comm = rig.topology.communicator();

    let main_path = rig.directory.join("main-collection.pio");
    reset_file(rig.world, &main_path);
    all_ranks(
        rig.world,
        unlimited(|| {
            let views = [main0.view(), main1.view()];
            crate::write_mpi_collection(&main_path, comm, &views)
        })
        .is_ok(),
        "collection main setup write",
    );

    let named_main_path = rig.directory.join("main-named-collection.pio");
    reset_file(rig.world, &named_main_path);
    all_ranks(
        rig.world,
        unlimited(|| {
            let views = [main0.view(), main1.view()];
            crate::write_mpi_named_collection(
                &named_main_path,
                "main",
                comm,
                &views,
                &MpiIoOptions::default(),
            )
        })
        .is_ok(),
        "named collection main setup write",
    );
    let named_main_bytes = support::root_read(rig.world, &named_main_path);

    let _limit = LocalByteLimitGuard::new(TEST_LIMIT);
    let rejected_write = |path: PathBuf, label: &str, options: Option<&MpiIoOptions>| {
        reset_file(rig.world, &path);
        let staging_before = crate::collections::STAGING_CALLS.with(Cell::get);
        let result = rejected(
            rig.world,
            label,
            || {
                let views = [main0.view(), main1.view()];
                match options {
                    Some(options) => {
                        crate::write_mpi_collection_with_options(&path, comm, &views, options)
                    }
                    None => crate::write_mpi_collection(&path, comm, &views),
                }
            },
            is_collection_limit,
        );
        all_ranks(rig.world, result.is_err(), label);
        all_ranks(rig.world, !path.exists(), label);
        all_ranks(
            rig.world,
            crate::collections::STAGING_CALLS.with(Cell::get) == staging_before,
            "collection rejection does not stage",
        );
    };

    rejected_write(
        rig.directory.join("rejected-collection.pio"),
        "collection write",
        None,
    );
    for (index, options) in modes().iter().enumerate() {
        rejected_write(
            rig.directory
                .join(format!("rejected-collection-{index}.pio")),
            "collection write options",
            Some(options),
        );
    }

    let discarded = |path: PathBuf, label: &str, options: Option<&MpiIoOptions>| {
        let mut d0 = PencilArray::from_elem(
            Pencil::<2, 1>::new(rig.topology.clone(), main_shape, [0]).unwrap(),
            ExtraShape::scalar(),
            -7.0,
        )
        .unwrap();
        let mut d1 = PencilArray::from_elem(
            Pencil::<2, 1>::new(rig.topology.clone(), main_shape, [0]).unwrap(),
            ExtraShape::scalar(),
            -7.0,
        )
        .unwrap();
        let before0 = d0.as_slice().to_vec();
        let before1 = d1.as_slice().to_vec();
        let staging_before = crate::collections::STAGING_CALLS.with(Cell::get);
        let result = rejected(
            rig.world,
            label,
            || {
                let mut views = [d0.view_mut(), d1.view_mut()];
                match options {
                    Some(options) => {
                        crate::read_mpi_collection_with_options(&path, comm, &mut views, options)
                    }
                    None => crate::read_mpi_collection(&path, comm, &mut views),
                }
            },
            is_collection_limit,
        );
        all_ranks(rig.world, result.is_err(), label);
        all_ranks(
            rig.world,
            d0.as_slice() == before0.as_slice() && d1.as_slice() == before1.as_slice(),
            label,
        );
        all_ranks(
            rig.world,
            crate::collections::STAGING_CALLS.with(Cell::get) == staging_before,
            "collection read rejection does not stage",
        );
    };
    discarded(main_path.clone(), "collection read", None);
    for (index, options) in modes().iter().enumerate() {
        discarded(
            main_path.clone(),
            &format!("collection read options {index}"),
            Some(options),
        );
    }

    let rejected_named = |path: PathBuf, label: &str, options: &MpiIoOptions| {
        let result = rejected(
            rig.world,
            label,
            || {
                let views = [main0.view(), main1.view()];
                crate::write_mpi_named_collection(&path, "big", comm, &views, options)
            },
            is_collection_limit,
        );
        all_ranks(rig.world, result.is_err(), label);
        all_ranks(rig.world, !path.exists(), label);
    };
    for (index, options) in modes().iter().enumerate() {
        rejected_named(
            rig.directory
                .join(format!("rejected-named-collection-{index}.pio")),
            "named collection write",
            options,
        );
    }
    for (index, options) in modes().iter().enumerate() {
        let result = rejected(
            rig.world,
            "named collection append",
            || {
                let views = [main0.view(), main1.view()];
                crate::append_mpi_named_collection(&named_main_path, "big", comm, &views, options)
            },
            is_collection_limit,
        );
        all_ranks(
            rig.world,
            result.is_err(),
            &format!("named append options {index}"),
        );
        all_ranks(
            rig.world,
            support::root_read(rig.world, &named_main_path) == named_main_bytes,
            "named collection append file unchanged",
        );
    }

    for (index, options) in modes().iter().enumerate() {
        let mut d0 = PencilArray::from_elem(
            Pencil::<2, 1>::new(rig.topology.clone(), main_shape, [0]).unwrap(),
            ExtraShape::scalar(),
            -7.0,
        )
        .unwrap();
        let mut d1 = PencilArray::from_elem(
            Pencil::<2, 1>::new(rig.topology.clone(), main_shape, [0]).unwrap(),
            ExtraShape::scalar(),
            -7.0,
        )
        .unwrap();
        let before0 = d0.as_slice().to_vec();
        let before1 = d1.as_slice().to_vec();
        let result = rejected(
            rig.world,
            &format!("named collection read {index}"),
            || {
                let mut views = [d0.view_mut(), d1.view_mut()];
                crate::read_mpi_named_collection(
                    &named_main_path,
                    "main",
                    comm,
                    &mut views,
                    options,
                )
            },
            is_collection_limit,
        );
        all_ranks(
            rig.world,
            result.is_err(),
            &format!("named collection read {index} rejected"),
        );
        all_ranks(
            rig.world,
            d0.as_slice() == before0.as_slice() && d1.as_slice() == before1.as_slice(),
            "named collection read destination unchanged",
        );
    }

    for (index, options) in modes().iter().enumerate() {
        let path = rig
            .directory
            .join(format!("boundary-collection-{index}.pio"));
        reset_file(rig.world, &path);
        all_ranks(
            rig.world,
            {
                let views = [boundary0.view(), boundary1.view()];
                crate::write_mpi_collection_with_options(&path, comm, &views, options).is_ok()
            },
            "collection boundary write",
        );
        let mut d0 = PencilArray::from_elem(
            Pencil::<2, 1>::new(rig.topology.clone(), boundary_shape, [0]).unwrap(),
            ExtraShape::scalar(),
            -7.0,
        )
        .unwrap();
        let mut d1 = PencilArray::from_elem(
            Pencil::<2, 1>::new(rig.topology.clone(), boundary_shape, [0]).unwrap(),
            ExtraShape::scalar(),
            -7.0,
        )
        .unwrap();
        all_ranks(
            rig.world,
            {
                let mut views = [d0.view_mut(), d1.view_mut()];
                crate::read_mpi_collection_with_options(&path, comm, &mut views, options).is_ok()
            },
            "collection boundary read",
        );
        all_ranks(
            rig.world,
            d0.as_slice() == boundary0.as_slice() && d1.as_slice() == boundary1.as_slice(),
            "collection boundary values",
        );
    }

    for (index, options) in modes().iter().enumerate() {
        let path = rig
            .directory
            .join(format!("boundary-named-collection-{index}.pio"));
        reset_file(rig.world, &path);
        all_ranks(
            rig.world,
            {
                let views = [boundary0.view(), boundary1.view()];
                crate::write_mpi_named_collection(&path, "boundary", comm, &views, options).is_ok()
            },
            "named collection boundary write",
        );
        let mut d0 = PencilArray::from_elem(
            Pencil::<2, 1>::new(rig.topology.clone(), boundary_shape, [0]).unwrap(),
            ExtraShape::scalar(),
            -7.0,
        )
        .unwrap();
        let mut d1 = PencilArray::from_elem(
            Pencil::<2, 1>::new(rig.topology.clone(), boundary_shape, [0]).unwrap(),
            ExtraShape::scalar(),
            -7.0,
        )
        .unwrap();
        all_ranks(
            rig.world,
            {
                let mut views = [d0.view_mut(), d1.view_mut()];
                crate::read_mpi_named_collection(&path, "boundary", comm, &mut views, options)
                    .is_ok()
            },
            "named collection boundary read",
        );
        all_ranks(
            rig.world,
            d0.as_slice() == boundary0.as_slice() && d1.as_slice() == boundary1.as_slice(),
            "named collection boundary values",
        );
        all_ranks(
            rig.world,
            {
                let views = [boundary0.view(), boundary1.view()];
                crate::append_mpi_named_collection(&path, "appended", comm, &views, options).is_ok()
            },
            "named collection boundary append",
        );
        let mut d0 = PencilArray::from_elem(
            Pencil::<2, 1>::new(rig.topology.clone(), boundary_shape, [0]).unwrap(),
            ExtraShape::scalar(),
            -7.0,
        )
        .unwrap();
        let mut d1 = PencilArray::from_elem(
            Pencil::<2, 1>::new(rig.topology.clone(), boundary_shape, [0]).unwrap(),
            ExtraShape::scalar(),
            -7.0,
        )
        .unwrap();
        all_ranks(
            rig.world,
            {
                let mut views = [d0.view_mut(), d1.view_mut()];
                crate::read_mpi_named_collection(&path, "appended", comm, &mut views, options)
                    .is_ok()
            },
            "named collection boundary append read",
        );
        all_ranks(
            rig.world,
            d0.as_slice() == boundary0.as_slice() && d1.as_slice() == boundary1.as_slice(),
            "named collection boundary append values",
        );
    }

    let injected_path = rig.directory.join("injected-collection.pio");
    reset_file(rig.world, &injected_path);
    let (result, snapshot) = {
        let probe = ProbeGuard::new(
            PROBE_THRESHOLD,
            (rig.world.rank() == 0).then_some(rig.boundary_bytes),
        );
        let views = [boundary0.view(), boundary1.view()];
        let result = crate::write_mpi_collection(&injected_path, comm, &views);
        (result, probe.finish())
    };
    assert_injected_allocation(rig, "collection staging", &snapshot, rig.boundary_bytes);
    all_ranks(
        rig.world,
        match (&result, rig.world.rank() == 0) {
            (Err(CollectionIoError::Io(IoError::AllocationFailed { requested })), true) => {
                *requested == rig.boundary_bytes
            }
            (Err(CollectionIoError::Io(IoError::CollectivePrecondition { .. })), false) => true,
            _ => false,
        },
        "collection staging failure variants",
    );
    all_ranks(
        rig.world,
        !injected_path.exists(),
        "collection staging path absent",
    );

    let boundary_collection_path = rig.directory.join("boundary-collection-0.pio");
    let mut d0 = PencilArray::from_elem(
        Pencil::<2, 1>::new(rig.topology.clone(), boundary_shape, [0]).unwrap(),
        ExtraShape::scalar(),
        -7.0,
    )
    .unwrap();
    let mut d1 = PencilArray::from_elem(
        Pencil::<2, 1>::new(rig.topology.clone(), boundary_shape, [0]).unwrap(),
        ExtraShape::scalar(),
        -7.0,
    )
    .unwrap();
    let before0 = d0.as_slice().to_vec();
    let before1 = d1.as_slice().to_vec();
    let payload_calls = crate::ffi::test_payload_calls();
    let (result, snapshot) = {
        let probe = ProbeGuard::new(
            PROBE_THRESHOLD,
            (rig.world.rank() == 0).then_some(rig.boundary_bytes),
        );
        let result = {
            let mut views = [d0.view_mut(), d1.view_mut()];
            crate::read_mpi_collection(&boundary_collection_path, comm, &mut views)
        };
        (result, probe.finish())
    };
    assert_injected_allocation(
        rig,
        "collection read staging",
        &snapshot,
        rig.boundary_bytes,
    );
    all_ranks(
        rig.world,
        match (&result, rig.world.rank() == 0) {
            (Err(CollectionIoError::Io(IoError::AllocationFailed { requested })), true) => {
                *requested == rig.boundary_bytes
            }
            (Err(CollectionIoError::Io(IoError::CollectivePrecondition { .. })), false) => true,
            _ => false,
        },
        "collection read staging failure variants",
    );
    all_ranks(
        rig.world,
        d0.as_slice() == before0.as_slice() && d1.as_slice() == before1.as_slice(),
        "collection read staging destination unchanged",
    );
    all_ranks(
        rig.world,
        payload_calls == crate::ffi::test_payload_calls(),
        "collection read staging has no payload call",
    );
}

fn session_checks(rig: &Rig<'_>) {
    let _limit = LocalByteLimitGuard::new(TEST_LIMIT);
    let size = rig.world.size() as usize;
    let small = array_for(rig.topology, [size, 512], 200.0);
    let boundary_pencil =
        Pencil::<2, 1>::new(rig.topology.clone(), rig.boundary_shape, [0]).unwrap();

    for mode in [MpiIoMode::Collective, MpiIoMode::Independent] {
        let path = rig.directory.join(format!("session-{mode:?}.pio"));
        reset_file(rig.world, &path);
        let options = MpiIoOptions::default().mode(mode);
        let mut failures: Vec<&'static str> = Vec::new();
        match MpiFileSession::create(rig.topology.communicator(), &path, &options) {
            Ok(mut session) => {
                if !collective_and(rig.world, session.write_named("base", small.view()).is_ok()) {
                    failures.push("session base write");
                }

                let file_calls = crate::ffi::FILE_CALLS.with(Cell::get);
                let at_calls = crate::ffi::test_at_calls();
                let payload_calls = crate::ffi::test_payload_calls();
                let probe = ProbeGuard::new(PROBE_THRESHOLD, None);
                let error = session.write_named("big", rig.main.view());
                let snapshot = probe.finish();
                let local_ok = error.as_ref().err().is_some_and(is_session_limit)
                    && snapshot.matching == 0
                    && !snapshot.reached
                    && file_calls == crate::ffi::FILE_CALLS.with(Cell::get)
                    && at_calls == crate::ffi::test_at_calls()
                    && payload_calls == crate::ffi::test_payload_calls();
                if !collective_and(rig.world, local_ok) {
                    failures.push("session rejected write");
                }

                if !collective_and(
                    rig.world,
                    unlimited(|| session.write_named("main", rig.main.view())).is_ok(),
                ) {
                    failures.push("session main fixture write");
                }
                let mut main_destination = PencilArray::from_elem(
                    Pencil::<2, 1>::new(rig.topology.clone(), rig.main_shape, [0]).unwrap(),
                    ExtraShape::scalar(),
                    -7.0,
                )
                .unwrap();
                let main_read =
                    unlimited(|| session.read_named("main", main_destination.view_mut()));
                if !collective_and(
                    rig.world,
                    main_read.is_ok() && main_destination.as_slice() == rig.main.as_slice(),
                ) {
                    failures.push("session main fixture read");
                }
                let before = main_destination.as_slice().to_vec();
                let file_calls = crate::ffi::FILE_CALLS.with(Cell::get);
                let at_calls = crate::ffi::test_at_calls();
                let payload_calls = crate::ffi::test_payload_calls();
                let probe = ProbeGuard::new(PROBE_THRESHOLD, None);
                let error = session.read_named("main", main_destination.view_mut());
                let snapshot = probe.finish();
                let local_ok = error.as_ref().err().is_some_and(is_session_limit)
                    && snapshot.matching == 0
                    && !snapshot.reached
                    && main_destination.as_slice() == before.as_slice()
                    && file_calls == crate::ffi::FILE_CALLS.with(Cell::get)
                    && at_calls == crate::ffi::test_at_calls()
                    && payload_calls == crate::ffi::test_payload_calls();
                if !collective_and(rig.world, local_ok) {
                    failures.push("session rejected read");
                }

                if !collective_and(
                    rig.world,
                    session.write_named("boundary", rig.boundary.view()).is_ok(),
                ) {
                    failures.push("session boundary write");
                }
                let mut destination =
                    PencilArray::from_elem(boundary_pencil.clone(), ExtraShape::scalar(), -7.0)
                        .unwrap();
                let boundary_read = session.read_named("boundary", destination.view_mut());
                if !collective_and(
                    rig.world,
                    boundary_read.is_ok() && destination.as_slice() == rig.boundary.as_slice(),
                ) {
                    failures.push("session boundary read");
                }

                let (error, snapshot) = {
                    let probe = ProbeGuard::new(
                        PROBE_THRESHOLD,
                        (rig.world.rank() == 0).then_some(rig.boundary_bytes),
                    );
                    let error = session.write_named("injected", rig.boundary.view());
                    (error, probe.finish())
                };
                let injected_ok = if rig.world.rank() == 0 {
                    snapshot.reached && snapshot.injected_size == Some(rig.boundary_bytes)
                } else {
                    !snapshot.reached
                };
                if !collective_and(
                    rig.world,
                    injected_ok
                        && matches!(
                            &error,
                            Err(MpiSessionError::Named(NamedIoError::Io(
                                IoError::CollectivePrecondition { .. }
                            )))
                        ),
                ) {
                    failures.push("session write allocation failure");
                }

                let mut destination =
                    PencilArray::from_elem(boundary_pencil.clone(), ExtraShape::scalar(), -7.0)
                        .unwrap();
                let before = destination.as_slice().to_vec();
                let (error, snapshot) = {
                    let probe = ProbeGuard::new(
                        PROBE_THRESHOLD,
                        (rig.world.rank() == 0).then_some(rig.boundary_bytes),
                    );
                    let error = session.read_named("boundary", destination.view_mut());
                    (error, probe.finish())
                };
                let injected_ok = if rig.world.rank() == 0 {
                    snapshot.reached && snapshot.injected_size == Some(rig.boundary_bytes)
                } else {
                    !snapshot.reached
                };
                if !collective_and(
                    rig.world,
                    injected_ok
                        && matches!(
                            &error,
                            Err(MpiSessionError::Named(NamedIoError::Io(
                                IoError::CollectivePrecondition { .. }
                            )))
                        )
                        && destination.as_slice() == before.as_slice(),
                ) {
                    failures.push("session read allocation failure");
                }

                if !collective_and(
                    rig.world,
                    session.write_named("after", small.view()).is_ok(),
                ) {
                    failures.push("session reuse write");
                }
                let mut destination = PencilArray::from_elem(
                    Pencil::<2, 1>::new(rig.topology.clone(), [size, 512], [0]).unwrap(),
                    ExtraShape::scalar(),
                    -7.0,
                )
                .unwrap();
                let after_read = session.read_named("after", destination.view_mut());
                if !collective_and(
                    rig.world,
                    after_read.is_ok() && destination.as_slice() == small.as_slice(),
                ) {
                    failures.push("session reuse read");
                }

                if !collective_and(rig.world, session.close().is_ok()) {
                    failures.push("session close");
                }
            }
            Err(_) => failures.push("session create"),
        }
        let total = all_sum(rig.world, failures.len() as u64);
        assert_eq!(total, 0, "session {mode:?} local failures: {failures:?}");
    }
}

fn empty_checks(rig: &Rig<'_>) {
    let empty = array_for(rig.topology, [1, EMPTY_COLUMNS], 300.0);
    let local_bytes = empty.view().len() * std::mem::size_of::<f64>();
    let empty_ranks = all_sum(rig.world, u64::from(local_bytes == 0));
    if rig.world.size() > 1 {
        assert!(empty_ranks > 0, "empty fixture has an empty rank");
        assert!(
            empty_ranks < rig.world.size() as u64,
            "empty fixture has a nonempty rank"
        );
    } else {
        assert!(local_bytes > EMPTY_LIMIT);
    }

    let _limit = LocalByteLimitGuard::new(EMPTY_LIMIT);
    let path = rig.directory.join("rejected-empty.pio");
    reset_file(rig.world, &path);
    let result = rejected(
        rig.world,
        "empty rank write",
        || crate::write_mpi(&path, empty.view()),
        is_local_limit,
    );
    all_ranks(rig.world, result.is_err(), "empty rank write rejected");
    all_ranks(rig.world, !path.exists(), "empty rank write path absent");
}

#[cfg(feature = "parallel-hdf5")]
fn hdf5_checks(rig: &Rig<'_>) {
    let _limit = LocalByteLimitGuard::new(1);
    let size = rig.world.size() as usize;
    let a = array_for(rig.topology, [size, 128], 10.0);
    let b = array_for(rig.topology, [size, 128], 20.0);
    let comm = rig.topology.communicator();
    let path = rig.directory.join("collection.h5");
    reset_file(rig.world, &path);
    all_ranks(
        rig.world,
        {
            let views = [a.view(), b.view()];
            crate::write_hdf5_collection(&path, comm, &views).is_ok()
        },
        "HDF5 collection is not subject to the MPI local byte limit",
    );
    let mut da = array_for(rig.topology, [size, 128], -1.0);
    let mut db = array_for(rig.topology, [size, 128], -1.0);
    all_ranks(
        rig.world,
        {
            let mut views = [da.view_mut(), db.view_mut()];
            crate::read_hdf5_collection(&path, comm, &mut views).is_ok()
        },
        "HDF5 collection read is not subject to the MPI local byte limit",
    );
    all_ranks(
        rig.world,
        da.as_slice() == a.as_slice() && db.as_slice() == b.as_slice(),
        "HDF5 collection values",
    );

    let named_path = rig.directory.join("named-collection.h5");
    reset_file(rig.world, &named_path);
    let write_options = crate::Hdf5WriteOptions::default();
    all_ranks(
        rig.world,
        {
            let views = [a.view(), b.view()];
            crate::write_hdf5_named_collection(&named_path, "named", comm, &views, &write_options)
                .is_ok()
        },
        "named HDF5 collection is not subject to the MPI local byte limit",
    );
    let read_options = crate::Hdf5ReadOptions::default();
    let mut da = array_for(rig.topology, [size, 128], -1.0);
    let mut db = array_for(rig.topology, [size, 128], -1.0);
    all_ranks(
        rig.world,
        {
            let mut views = [da.view_mut(), db.view_mut()];
            crate::read_hdf5_named_collection(&named_path, "named", comm, &mut views, &read_options)
                .is_ok()
        },
        "named HDF5 collection read is not subject to the MPI local byte limit",
    );
}

#[cfg(not(feature = "parallel-hdf5"))]
fn hdf5_checks(_rig: &Rig<'_>) {}

#[cfg(test)]
fn metadata_allocation_failure(rig: &Rig<'_>) {
    let rank_request = (rig.main.pencil().global_shape().len() + 1) * std::mem::size_of::<i32>();
    let (result, snapshot) = {
        let probe = ProbeGuard::new(
            PROBE_THRESHOLD,
            (rig.world.rank() == 0).then_some(rank_request),
        );
        let result = build_layout(
            rig.main.pencil().global_shape(),
            &[],
            &rig.main.pencil().local_shape_logical(),
            rig.main.pencil().local_ranges(),
            rig.main.view().len(),
            std::mem::size_of::<f64>(),
        );
        (result, probe.finish())
    };
    let local_ok = if rig.world.rank() == 0 {
        snapshot.reached
            && snapshot.injected_size == Some(rank_request)
            && matches!(
                &result,
                Err(IoError::AllocationFailed { requested }) if *requested == rank_request
            )
    } else {
        !snapshot.reached && result.is_ok()
    };
    all_ranks(rig.world, local_ok, "build_layout metadata injection");
    let agreement =
        agree_mpi_preparation(rig.world, result.as_ref().err(), "build_layout metadata");
    all_ranks(
        rig.world,
        matches!(
            agreement,
            Err(IoError::CollectivePrecondition {
                phase: "build_layout metadata"
            })
        ),
        "build_layout metadata state 1",
    );
}
