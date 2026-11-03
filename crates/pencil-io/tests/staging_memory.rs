//! Whole-process Rust live-byte accounting for the public read paths.
//!
//! The measured quantity is the peak of *requested live bytes* handed to the
//! global allocator. It is neither RSS nor the allocation MPI and HDF5 perform
//! natively. A process-wide counter is only meaningful on one rank, so the test
//! refuses to run otherwise.
//!
//! The fixture is large enough that a second payload buffer is visible: E =
//! 2 * 256 * 1024 = 524,288 `Complex<f64>` elements, one payload P = 8 MiB. The
//! old helper keeps the byte staging and a typed `Vec<T>` alive together, so its
//! peak grows by about 2P. The target is one payload plus 1 MiB of slack.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use mpi::traits::*;
use num_complex::Complex;
use pencil_array::{
    AxisPermutation, ExtraShape, MpiTopology, Pencil, PencilArray, PencilArrayViewMut,
};
#[cfg(feature = "parallel-hdf5")]
use pencil_io::{
    Hdf5FileSession, Hdf5ReadOptions, Hdf5WriteOptions, read_hdf5, read_hdf5_collection,
    read_hdf5_collection_with_options, read_hdf5_named_collection, write_hdf5,
    write_hdf5_collection, write_hdf5_collection_with_options, write_hdf5_named_collection,
};
use pencil_io::{
    MpiFileSession, MpiIoOptions, RawByteOrder, RawReadOptions, read_mpi, read_mpi_chunked,
    read_mpi_collection, read_mpi_collection_with_options, read_mpi_named,
    read_mpi_named_collection, read_mpi_raw, write_mpi, write_mpi_chunked, write_mpi_collection,
    write_mpi_collection_with_options, write_mpi_named, write_mpi_named_collection,
};

mod support;
use support::{cleanup_owned_temp_dir, owned_temp_dir};

const EXTRA: usize = 2;
const SPATIAL: [usize; 2] = [256, 1024];
const ELEMENTS: usize = EXTRA * SPATIAL[0] * SPATIAL[1];
const WIDTH: usize = 16;
const PAYLOAD_BYTES: usize = ELEMENTS * WIDTH;
const LIMIT_BYTES: usize = PAYLOAD_BYTES + 1024 * 1024;

// Collection fixture: two members of `ExtraShape([1])` whose combined payload
// is exactly `PAYLOAD_BYTES`.
const MEMBERS: usize = 2;
const MEMBER_ELEMENTS: usize = ELEMENTS / MEMBERS;
type Array = PencilArray<Complex<f64>, 2, 2>;
type ViewMut<'a> = PencilArrayViewMut<'a, Complex<f64>, 2, 2>;

// ---------------------------------------------------------------------------
// Allocator accounting
// ---------------------------------------------------------------------------

static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);

struct AccountedSystem;

#[global_allocator]
static ALLOCATOR: AccountedSystem = AccountedSystem;

fn grow(size: usize) {
    let live = LIVE_BYTES.fetch_add(size, Ordering::SeqCst) + size;
    PEAK_BYTES.fetch_max(live, Ordering::SeqCst);
}

fn shrink(size: usize) {
    LIVE_BYTES.fetch_sub(size, Ordering::SeqCst);
}

// No logging, formatting or container lives here: the hooks run inside the
// allocator and must not allocate themselves.
unsafe impl GlobalAlloc for AccountedSystem {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            grow(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            grow(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        shrink(layout.size());
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let replacement = unsafe { System.realloc(pointer, layout, new_size) };
        // A failed realloc keeps the old allocation owned by the caller, so the
        // accounting must keep the old size as well.
        if !replacement.is_null() {
            if new_size >= layout.size() {
                grow(new_size - layout.size());
            } else {
                shrink(layout.size() - new_size);
            }
        }
        replacement
    }
}

struct Sample {
    path: &'static str,
    baseline: usize,
    peak: usize,
}

impl Sample {
    fn increment(&self) -> usize {
        self.peak - self.baseline
    }
}

/// Snapshots live bytes, resets the peak to that baseline, runs the read, and
/// snapshots the peak again before anything else can allocate.
fn measure<T>(path: &'static str, operation: impl FnOnce() -> T) -> (T, Sample) {
    // The test binary runs one test on one rank with `--test-threads=1`, so no
    // other Rust allocation can interleave with the measured call.
    let baseline = LIVE_BYTES.load(Ordering::SeqCst);
    PEAK_BYTES.store(baseline, Ordering::SeqCst);
    let result = operation();
    let peak = PEAK_BYTES.load(Ordering::SeqCst);
    (
        result,
        Sample {
            path,
            baseline,
            peak,
        },
    )
}

fn allocator_self_check() {
    // An allocation that stays alive across the measured windows, so the
    // measured baseline is not zero and a release is not silently absorbed.
    let retained = vec![0x51u8; 1024];
    let before = LIVE_BYTES.load(Ordering::SeqCst);
    assert!(
        before >= retained.len(),
        "live bytes must already include a retained allocation"
    );

    let layout = Layout::from_size_align(4096, 8).expect("layout");
    let pointer = unsafe { ALLOCATOR.alloc(layout) };
    assert!(!pointer.is_null(), "alloc returned null");
    black_box(pointer);
    assert_eq!(
        LIVE_BYTES.load(Ordering::SeqCst),
        before + 4096,
        "alloc must add the requested size"
    );
    assert!(
        PEAK_BYTES.load(Ordering::SeqCst) >= before + 4096,
        "alloc growth must reach the peak"
    );

    let zero_layout = Layout::from_size_align(2048, 8).expect("layout");
    let zeroed = unsafe { ALLOCATOR.alloc_zeroed(zero_layout) };
    assert!(!zeroed.is_null(), "alloc_zeroed returned null");
    black_box(zeroed);
    assert_eq!(
        LIVE_BYTES.load(Ordering::SeqCst),
        before + 4096 + 2048,
        "alloc_zeroed must add the requested size"
    );

    let grown_layout = Layout::from_size_align(8192, 8).expect("layout");
    let grown = unsafe { ALLOCATOR.realloc(pointer, layout, 8192) };
    assert!(!grown.is_null(), "growing realloc returned null");
    black_box(grown);
    assert_eq!(
        LIVE_BYTES.load(Ordering::SeqCst),
        before + 8192 + 2048,
        "growing realloc must replace the old size with the new one"
    );

    let shrunk_layout = Layout::from_size_align(512, 8).expect("layout");
    let shrunk = unsafe { ALLOCATOR.realloc(grown, grown_layout, 512) };
    assert!(!shrunk.is_null(), "shrinking realloc returned null");
    black_box(shrunk);
    assert_eq!(
        LIVE_BYTES.load(Ordering::SeqCst),
        before + 512 + 2048,
        "shrinking realloc must return the difference"
    );

    unsafe { ALLOCATOR.dealloc(shrunk, shrunk_layout) };
    assert_eq!(
        LIVE_BYTES.load(Ordering::SeqCst),
        before + 2048,
        "dealloc must subtract the released size"
    );
    unsafe { ALLOCATOR.dealloc(zeroed, zero_layout) };
    assert_eq!(
        LIVE_BYTES.load(Ordering::SeqCst),
        before,
        "the self-check must return to its own baseline"
    );

    let large_layout = Layout::from_size_align(4096, 8).expect("large layout");
    let small_layout = Layout::from_size_align(512, 8).expect("small layout");
    let (allocated, large_sample) = measure("allocator_peak", || unsafe {
        let large = ALLOCATOR.alloc(large_layout);
        if large.is_null() {
            return false;
        }
        ALLOCATOR.dealloc(large, large_layout);

        let small = ALLOCATOR.alloc(small_layout);
        if small.is_null() {
            return false;
        }
        ALLOCATOR.dealloc(small, small_layout);
        true
    });
    assert!(allocated, "allocator peak self-check allocation failed");
    assert_eq!(
        large_sample.peak,
        large_sample.baseline + large_layout.size(),
        "peak must retain the largest allocation after it is freed"
    );
    assert_eq!(
        LIVE_BYTES.load(Ordering::SeqCst),
        large_sample.baseline,
        "large and small allocations must be released"
    );

    let (allocated, small_sample) = measure("allocator_peak_reset", || unsafe {
        let small = ALLOCATOR.alloc(small_layout);
        if small.is_null() {
            return false;
        }
        ALLOCATOR.dealloc(small, small_layout);
        true
    });
    assert!(allocated, "allocator reset self-check allocation failed");
    assert_eq!(small_sample.baseline, large_sample.baseline);
    assert_eq!(
        small_sample.peak,
        small_sample.baseline + small_layout.size(),
        "the next measurement must reset the previous peak"
    );
    assert!(
        small_sample.peak < large_sample.peak,
        "the second measurement must not retain the first measurement's peak"
    );
    assert_eq!(
        LIVE_BYTES.load(Ordering::SeqCst),
        small_sample.baseline,
        "live bytes must return to baseline after both measurements"
    );
    drop(retained);
}

// ---------------------------------------------------------------------------
// Values and their independent oracle
// ---------------------------------------------------------------------------

// Real and imaginary parts draw from different tables, so a component swap or a
// lost sign cannot pass. Signed zero, subnormals, infinities and NaN payloads
// are all present; the comparison is on `to_bits()`, never on `==`.
const SPECIAL: [(u64, u64); 8] = [
    (0x0000_0000_0000_0000, 0x8000_0000_0000_0000),
    (0x8000_0000_0000_0000, 0x0000_0000_0000_0000),
    (0x0000_0000_0000_0001, 0x800f_ffff_ffff_ffff),
    (0x000f_ffff_ffff_ffff, 0x0000_0000_0000_0001),
    (0x7ff0_0000_0000_0000, 0xfff0_0000_0000_0000),
    (0xfff0_0000_0000_0000, 0x7ff0_0000_0000_0000),
    (0x7ff8_0000_0000_0001, 0x7ff8_0000_0000_00ff),
    (0xfff8_0000_0000_beef, 0x7ff8_dead_beef_0001),
];

const MANTISSA: u64 = 0x000f_ffff_ffff_ffff;

fn payload(flat: usize) -> (u64, u64) {
    if let Some(bits) = SPECIAL.get(flat) {
        return *bits;
    }
    // Both parts stay injective in `flat` (below 2^19), so two distinct logical
    // positions never share a bit pattern.
    let flat = flat as u64;
    let mixed = flat.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    (
        0x3ff0_0000_0000_0000 | (flat & MANTISSA),
        0x4000_0000_0000_0000 | (mixed & MANTISSA),
    )
}

fn value_at(e: usize, x: usize, y: usize) -> Complex<f64> {
    let flat = (e * SPATIAL[0] + x) * SPATIAL[1] + y;
    let (re, im) = payload(flat);
    Complex::new(f64::from_bits(re), f64::from_bits(im))
}

/// Reader storage offset for the `[1, 0]` permutation: the memory shape is
/// `[extra, y, x]`, computed here without any production decoder or inverse
/// permutation.
fn physical_offset(e: usize, x: usize, y: usize) -> usize {
    (e * SPATIAL[1] + y) * SPATIAL[0] + x
}

/// Reader storage offset for the identity `[0, 1]` permutation.
fn identity_offset(e: usize, x: usize, y: usize) -> usize {
    (e * SPATIAL[0] + x) * SPATIAL[1] + y
}

fn verify_values(label: &str, array: &PencilArray<Complex<f64>, 2, 2>) {
    verify_at(label, array, physical_offset);
}

fn verify_identity_values(label: &str, array: &PencilArray<Complex<f64>, 2, 2>) {
    verify_at(label, array, identity_offset);
}

fn verify_at(
    label: &str,
    array: &PencilArray<Complex<f64>, 2, 2>,
    offset: fn(usize, usize, usize) -> usize,
) {
    let storage = array.as_slice();
    assert_eq!(storage.len(), ELEMENTS, "{label}: local element count");
    for e in 0..EXTRA {
        for x in 0..SPATIAL[0] {
            for y in 0..SPATIAL[1] {
                let expected = value_at(e, x, y);
                let actual = storage[offset(e, x, y)];
                assert_eq!(
                    actual.re.to_bits(),
                    expected.re.to_bits(),
                    "{label}: real bits at e={e} x={x} y={y}"
                );
                assert_eq!(
                    actual.im.to_bits(),
                    expected.im.to_bits(),
                    "{label}: imaginary bits at e={e} x={x} y={y}"
                );
            }
        }
    }
}

/// Values of the two-member collection: the extra axis of each member is the
/// member index, so both members hold different bit patterns.
fn verify_collection_values(label: &str, arrays: &[Array; MEMBERS]) {
    for (member, array) in arrays.iter().enumerate() {
        let storage = array.as_slice();
        assert_eq!(
            storage.len(),
            MEMBER_ELEMENTS,
            "{label}: member {member} element count"
        );
        for x in 0..SPATIAL[0] {
            for y in 0..SPATIAL[1] {
                let expected = value_at(member, x, y);
                let actual = storage[physical_offset(0, x, y)];
                assert_eq!(
                    actual.re.to_bits(),
                    expected.re.to_bits(),
                    "{label}: member {member} real bits at x={x} y={y}"
                );
                assert_eq!(
                    actual.im.to_bits(),
                    expected.im.to_bits(),
                    "{label}: member {member} imaginary bits at x={x} y={y}"
                );
            }
        }
    }
}

fn member_views(arrays: &mut [Array; MEMBERS]) -> [ViewMut<'_>; MEMBERS] {
    let [first, second] = arrays;
    [first.view_mut(), second.view_mut()]
}

fn reset_collection(arrays: &mut [Array; MEMBERS]) {
    for array in arrays {
        reset_destination(array);
    }
}

fn reset_destination(destination: &mut PencilArray<Complex<f64>, 2, 2>) {
    let sentinel = Complex::new(
        f64::from_bits(0x7ff8_1234_5678_9abc),
        f64::from_bits(0xfff8_9876_5432_1abc),
    );
    destination.as_mut_slice().fill(sentinel);
}

fn write_big_endian_payload(path: &Path) {
    let mut bytes = Vec::with_capacity(PAYLOAD_BYTES);
    for e in 0..EXTRA {
        for x in 0..SPATIAL[0] {
            for y in 0..SPATIAL[1] {
                let value = value_at(e, x, y);
                bytes.extend_from_slice(&value.re.to_bits().to_be_bytes());
                bytes.extend_from_slice(&value.im.to_bits().to_be_bytes());
            }
        }
    }
    assert_eq!(bytes.len(), PAYLOAD_BYTES);
    std::fs::write(path, &bytes).expect("raw fixture");
}

// ---------------------------------------------------------------------------
// Non-self-inverse three-axis permutation (correctness, outside measurement)
// ---------------------------------------------------------------------------

fn three_axis_permutation_case(world: &mpi::topology::SimpleCommunicator, directory: &Path) {
    let topology = MpiTopology::<2>::new(world, [1, 1]).expect("3-axis topology");
    let extra = ExtraShape::new([2]).expect("extra shape");

    let writer_pencil =
        Pencil::<3, 2>::new(topology.clone(), [2, 3, 5], [0, 1]).expect("writer pencil");
    let mut source =
        PencilArray::from_elem(writer_pencil, extra.clone(), -1i32).expect("writer array");
    assert_eq!(source.as_slice().len(), 60);
    for e in 0..2 {
        for x in 0..2 {
            for y in 0..3 {
                for z in 0..5 {
                    let flat = ((e * 2 + x) * 3 + y) * 5 + z;
                    source.as_mut_slice()[flat] = oracle(e, x, y, z);
                }
            }
        }
    }

    let path = directory.join("three-axis.pio");
    write_mpi(&path, source.view()).expect("3-axis write");

    let reader_pencil = Pencil::<3, 2>::new_permuted(
        topology,
        [2, 3, 5],
        [0, 1],
        AxisPermutation::new([1, 2, 0]).expect("reader permutation"),
    )
    .expect("reader pencil");
    let mut destination =
        PencilArray::from_elem(reader_pencil, extra, -777i32).expect("reader array");
    read_mpi(&path, destination.view_mut()).expect("3-axis read");

    let storage = destination.as_slice();
    assert_eq!(storage.len(), 60, "reader physical length");
    let mut covered = vec![false; storage.len()];
    for e in 0..2 {
        for x in 0..2 {
            for y in 0..3 {
                for z in 0..5 {
                    // Reader memory shape is [extra, y, z, x].
                    let offset = ((e * 3 + y) * 5 + z) * 2 + x;
                    assert_eq!(
                        storage[offset],
                        oracle(e, x, y, z),
                        "three-axis value at e={e} x={x} y={y} z={z}"
                    );
                    assert!(
                        !covered[offset],
                        "three-axis offset {offset} must be unique"
                    );
                    covered[offset] = true;
                }
            }
        }
    }
    assert!(
        covered.iter().all(|&seen| seen),
        "every offset must be covered"
    );
}

fn oracle(e: usize, x: usize, y: usize, z: usize) -> i32 {
    1000 * e as i32 + 100 * x as i32 + 10 * y as i32 + z as i32
}

// ---------------------------------------------------------------------------
// The memory test
// ---------------------------------------------------------------------------

#[test]
fn staging_peak_is_one_payload() {
    let universe = mpi::initialize().expect("MPI must initialize once");
    let world = universe.world();
    let size = usize::try_from(world.size()).expect("rank count");
    assert_eq!(
        size, 1,
        "staging_memory measures whole-process Rust live bytes, so it requires \
         exactly one MPI rank; CI runs it with `mpiexec -n 1`"
    );

    allocator_self_check();

    let directory = owned_temp_dir(&world, "pencil-staging-memory");
    let topology = MpiTopology::<2>::new(&world, [1, 1]).expect("topology");
    let comm = topology.communicator();
    let extra = ExtraShape::new([EXTRA]).expect("extra shape");

    // Writer fixture: identity permutation, and one rank means local == global.
    let writer_pencil =
        Pencil::<2, 2>::new(topology.clone(), SPATIAL, [0, 1]).expect("writer pencil");
    let mut source = PencilArray::from_elem(writer_pencil, extra.clone(), Complex::new(-1.0, -1.0))
        .expect("writer array");
    assert_eq!(source.as_slice().len(), ELEMENTS);
    for e in 0..EXTRA {
        for x in 0..SPATIAL[0] {
            for y in 0..SPATIAL[1] {
                let flat = (e * SPATIAL[0] + x) * SPATIAL[1] + y;
                source.as_mut_slice()[flat] = value_at(e, x, y);
            }
        }
    }

    // The chunked format stores each rank's physical slice order and the reader
    // must use the identical layout, so this destination keeps `[0, 1]` while
    // the permuted one below is what the other decoders have to undo.
    let mut chunked_destination = PencilArray::from_elem(
        Pencil::<2, 2>::new(topology.clone(), SPATIAL, [0, 1]).expect("chunked reader pencil"),
        extra.clone(),
        Complex::new(-1.0, -1.0),
    )
    .expect("chunked destination");
    assert_eq!(chunked_destination.as_slice().len(), ELEMENTS);

    // Destination: the spatial axes are swapped in memory, so `[1, 0]` is what
    // the decoder has to undo.
    let reader_pencil = Pencil::<2, 2>::new_permuted(
        topology.clone(),
        SPATIAL,
        [0, 1],
        AxisPermutation::new([1, 0]).expect("reader permutation"),
    )
    .expect("reader pencil");
    let mut destination = PencilArray::from_elem(reader_pencil, extra, Complex::new(-1.0, -1.0))
        .expect("destination");
    assert_eq!(destination.as_slice().len(), ELEMENTS);

    // Every fixture, the destination and the open sessions exist before the
    // first baseline is taken.
    let mpi_path = directory.join("staging.pio");
    write_mpi(&mpi_path, source.view()).expect("MPI fixture");

    let raw_path = directory.join("staging.raw");
    write_big_endian_payload(&raw_path);

    let named_path = directory.join("staging-named.pio");
    write_mpi_named(&named_path, "large", source.view()).expect("named MPI fixture");

    let options = MpiIoOptions::default();
    let chunked_path = directory.join("staging-chunked.pio");
    write_mpi_chunked(&chunked_path, source.view(), &options).expect("chunked MPI fixture");

    let session_path = directory.join("staging-session.pio");
    let mut mpi_session =
        MpiFileSession::create(comm, &session_path, &options).expect("MPI session");
    mpi_session
        .write_named("large", source.view())
        .expect("session MPI fixture");

    #[cfg(feature = "parallel-hdf5")]
    let hdf5_path = directory.join("staging.h5");
    #[cfg(feature = "parallel-hdf5")]
    write_hdf5(&hdf5_path, source.view()).expect("HDF5 fixture");

    #[cfg(feature = "parallel-hdf5")]
    let hdf5_read_options = Hdf5ReadOptions::default();
    #[cfg(feature = "parallel-hdf5")]
    let hdf5_session_path = directory.join("staging-session.h5");
    #[cfg(feature = "parallel-hdf5")]
    let mut hdf5_session =
        Hdf5FileSession::create(&hdf5_session_path, comm, &MpiIoOptions::default())
            .expect("HDF5 session");
    #[cfg(feature = "parallel-hdf5")]
    hdf5_session
        .write("large", source.view(), &Hdf5WriteOptions::default())
        .expect("session HDF5 fixture");

    // Collection fixtures: two members whose combined payload is one P, read
    // through a permuted destination.
    let member_extra = ExtraShape::new([1]).expect("member extra shape");
    // Reuse the fixture pencils: the writer keeps `[0, 1]` and the single-view
    // destination already uses the `[1, 0]` reader permutation.
    let mut collection_sources: [Array; MEMBERS] = std::array::from_fn(|_| {
        Array::from_elem(
            Arc::clone(source.pencil()),
            member_extra.clone(),
            Complex::new(-1.0, -1.0),
        )
        .expect("collection source")
    });
    let mut collection_destinations: [Array; MEMBERS] = std::array::from_fn(|_| {
        Array::from_elem(
            Arc::clone(destination.pencil()),
            member_extra.clone(),
            Complex::new(-1.0, -1.0),
        )
        .expect("collection destination")
    });
    for (member, array) in collection_sources.iter_mut().enumerate() {
        for x in 0..SPATIAL[0] {
            for y in 0..SPATIAL[1] {
                array.as_mut_slice()[identity_offset(0, x, y)] = value_at(member, x, y);
            }
        }
    }
    assert_eq!(collection_sources[0].as_slice().len(), MEMBER_ELEMENTS);

    let collection_path = directory.join("staging-collection.pio");
    write_mpi_collection(
        &collection_path,
        comm,
        &[collection_sources[0].view(), collection_sources[1].view()],
    )
    .expect("MPI collection fixture");

    let collection_options_path = directory.join("staging-collection-options.pio");
    write_mpi_collection_with_options(
        &collection_options_path,
        comm,
        &[collection_sources[0].view(), collection_sources[1].view()],
        &options,
    )
    .expect("option MPI collection fixture");

    let collection_named_path = directory.join("staging-collection-named.pio");
    write_mpi_named_collection(
        &collection_named_path,
        "members",
        comm,
        &[collection_sources[0].view(), collection_sources[1].view()],
        &options,
    )
    .expect("named MPI collection fixture");

    #[cfg(feature = "parallel-hdf5")]
    let hdf5_write_options = Hdf5WriteOptions::default();
    #[cfg(feature = "parallel-hdf5")]
    let hdf5_collection_path = directory.join("staging-collection.h5");
    #[cfg(feature = "parallel-hdf5")]
    write_hdf5_collection(
        &hdf5_collection_path,
        comm,
        &[collection_sources[0].view(), collection_sources[1].view()],
    )
    .expect("HDF5 collection fixture");
    #[cfg(feature = "parallel-hdf5")]
    let hdf5_collection_options_path = directory.join("staging-collection-options.h5");
    #[cfg(feature = "parallel-hdf5")]
    write_hdf5_collection_with_options(
        &hdf5_collection_options_path,
        comm,
        &[collection_sources[0].view(), collection_sources[1].view()],
        &hdf5_write_options,
    )
    .expect("option HDF5 collection fixture");
    #[cfg(feature = "parallel-hdf5")]
    let hdf5_collection_named_path = directory.join("staging-collection-named.h5");
    #[cfg(feature = "parallel-hdf5")]
    write_hdf5_named_collection(
        &hdf5_collection_named_path,
        "members",
        comm,
        &[collection_sources[0].view(), collection_sources[1].view()],
        &hdf5_write_options,
    )
    .expect("named HDF5 collection fixture");

    let mut samples: Vec<Sample> = Vec::new();

    reset_destination(&mut destination);
    let (result, sample) = measure("mpi_io", || read_mpi(&mpi_path, destination.view_mut()));
    assert!(result.is_ok(), "read_mpi failed: {result:?}");
    samples.push(sample);
    verify_values("read_mpi", &destination);

    let raw_options = RawReadOptions::default().byte_order(RawByteOrder::Big);
    reset_destination(&mut destination);
    let (result, sample) = measure("mpi_raw", || {
        read_mpi_raw(&raw_path, destination.view_mut(), raw_options)
    });
    assert!(result.is_ok(), "read_mpi_raw failed: {result:?}");
    samples.push(sample);
    verify_values("read_mpi_raw", &destination);

    reset_destination(&mut destination);
    let (result, sample) = measure("mpi_named", || {
        read_mpi_named(&named_path, "large", destination.view_mut())
    });
    assert!(result.is_ok(), "read_mpi_named failed: {result:?}");
    samples.push(sample);
    verify_values("read_mpi_named", &destination);

    reset_destination(&mut destination);
    let (result, sample) = measure("mpi_session", || {
        mpi_session.read_named("large", destination.view_mut())
    });
    assert!(
        result.is_ok(),
        "MpiFileSession::read_named failed: {result:?}"
    );
    samples.push(sample);
    verify_values("MpiFileSession::read_named", &destination);

    reset_destination(&mut chunked_destination);
    let (result, sample) = measure("mpi_chunked", || {
        read_mpi_chunked(&chunked_path, chunked_destination.view_mut(), &options)
    });
    assert!(result.is_ok(), "read_mpi_chunked failed: {result:?}");
    samples.push(sample);
    verify_identity_values("read_mpi_chunked", &chunked_destination);

    reset_collection(&mut collection_destinations);
    let (result, sample) = measure("mpi_collection", || {
        let mut views = member_views(&mut collection_destinations);
        read_mpi_collection(&collection_path, comm, &mut views)
    });
    assert!(result.is_ok(), "read_mpi_collection failed: {result:?}");
    samples.push(sample);
    verify_collection_values("read_mpi_collection", &collection_destinations);

    reset_collection(&mut collection_destinations);
    let (result, sample) = measure("mpi_collection_options", || {
        let mut views = member_views(&mut collection_destinations);
        read_mpi_collection_with_options(&collection_options_path, comm, &mut views, &options)
    });
    assert!(
        result.is_ok(),
        "read_mpi_collection_with_options failed: {result:?}"
    );
    samples.push(sample);
    verify_collection_values("read_mpi_collection_with_options", &collection_destinations);

    reset_collection(&mut collection_destinations);
    let (result, sample) = measure("mpi_named_collection", || {
        let mut views = member_views(&mut collection_destinations);
        read_mpi_named_collection(
            &collection_named_path,
            "members",
            comm,
            &mut views,
            &options,
        )
    });
    assert!(
        result.is_ok(),
        "read_mpi_named_collection failed: {result:?}"
    );
    samples.push(sample);
    verify_collection_values("read_mpi_named_collection", &collection_destinations);

    #[cfg(feature = "parallel-hdf5")]
    {
        reset_destination(&mut destination);
        let (result, sample) = measure("hdf5", || read_hdf5(&hdf5_path, destination.view_mut()));
        assert!(result.is_ok(), "read_hdf5 failed: {result:?}");
        samples.push(sample);
        verify_values("read_hdf5", &destination);

        reset_destination(&mut destination);
        let (result, sample) = measure("hdf5_session", || {
            hdf5_session.read("large", destination.view_mut(), &hdf5_read_options)
        });
        assert!(result.is_ok(), "Hdf5FileSession::read failed: {result:?}");
        samples.push(sample);
        verify_values("Hdf5FileSession::read", &destination);

        reset_collection(&mut collection_destinations);
        let (result, sample) = measure("hdf5_collection", || {
            let mut views = member_views(&mut collection_destinations);
            read_hdf5_collection(&hdf5_collection_path, comm, &mut views)
        });
        assert!(result.is_ok(), "read_hdf5_collection failed: {result:?}");
        samples.push(sample);
        verify_collection_values("read_hdf5_collection", &collection_destinations);

        reset_collection(&mut collection_destinations);
        let (result, sample) = measure("hdf5_collection_options", || {
            let mut views = member_views(&mut collection_destinations);
            read_hdf5_collection_with_options(
                &hdf5_collection_options_path,
                comm,
                &mut views,
                &hdf5_read_options,
            )
        });
        assert!(
            result.is_ok(),
            "read_hdf5_collection_with_options failed: {result:?}"
        );
        samples.push(sample);
        verify_collection_values(
            "read_hdf5_collection_with_options",
            &collection_destinations,
        );

        reset_collection(&mut collection_destinations);
        let (result, sample) = measure("hdf5_named_collection", || {
            let mut views = member_views(&mut collection_destinations);
            read_hdf5_named_collection(
                &hdf5_collection_named_path,
                "members",
                comm,
                &mut views,
                &hdf5_read_options,
            )
        });
        assert!(
            result.is_ok(),
            "read_hdf5_named_collection failed: {result:?}"
        );
        samples.push(sample);
        verify_collection_values("read_hdf5_named_collection", &collection_destinations);
    }

    // Sessions close only after every measured read has returned.
    mpi_session.close().expect("MPI session close");
    #[cfg(feature = "parallel-hdf5")]
    hdf5_session.close().expect("HDF5 session close");

    three_axis_permutation_case(&world, &directory);
    cleanup_owned_temp_dir(&world, &directory);

    // Reporting, the path-set check and the limit check all run outside the
    // measured windows.
    for sample in &samples {
        eprintln!(
            "staging_memory: {:<11} baseline={:>9} peak={:>9} increment={:>9} limit={:>9}",
            sample.path,
            sample.baseline,
            sample.peak,
            sample.increment(),
            LIMIT_BYTES,
        );
    }

    let mut measured: Vec<&str> = samples.iter().map(|sample| sample.path).collect();
    measured.sort_unstable();
    let mut expected = vec![
        "mpi_chunked",
        "mpi_collection",
        "mpi_collection_options",
        "mpi_io",
        "mpi_named",
        "mpi_named_collection",
        "mpi_raw",
        "mpi_session",
    ];
    #[cfg(feature = "parallel-hdf5")]
    expected.extend([
        "hdf5",
        "hdf5_collection",
        "hdf5_collection_options",
        "hdf5_named_collection",
        "hdf5_session",
    ]);
    expected.sort_unstable();
    assert_eq!(
        measured, expected,
        "every read path of this feature set must be measured exactly once"
    );

    let oversized: Vec<String> = samples
        .iter()
        .filter(|sample| sample.increment() > LIMIT_BYTES)
        .map(|sample| format!("{}={}", sample.path, sample.increment()))
        .collect();
    assert!(
        oversized.is_empty(),
        "reads keep more than one payload alive (limit {LIMIT_BYTES} bytes): {oversized:?}"
    );
}
