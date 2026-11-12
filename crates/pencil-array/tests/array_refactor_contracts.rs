//! Behaviour and structure contracts for the array refactors #113-#118.
//!
//! The behaviour checks pin public results and error values. The structural
//! checks are deliberately small `include_str!` scans: these refactors are
//! meant to preserve behaviour, so the only observable evidence that the
//! duplication is gone is the shape of the production source.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::collections::HashSet;
use std::hint::black_box;
use std::sync::Arc;

use mpi::traits::*;
use pencil_array::{
    ArrayError, AxisPermutation, CollectiveError, ExtraShape, GeometryError, ManyPencilArray,
    MpiTopology, Pencil, PencilArray, PencilArrayView, map_reduce_many, pointwise_many,
};

// ---------------------------------------------------------------------------
// Allocation counting (same TLS pattern as tests/mapped_reduction_alloc.rs).
// ---------------------------------------------------------------------------

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

thread_local! {
    static TRACKING: Cell<bool> = const { Cell::new(false) };
    static STATS: Cell<Stats> = const { Cell::new(Stats::ZERO) };
}

fn record(alloc: bool, alloc_zeroed: bool, realloc: bool, size: usize) {
    TRACKING.with(|tracking| {
        if !tracking.get() {
            return;
        }
        STATS.with(|stats| {
            let mut current = stats.get();
            if alloc {
                current.alloc += 1;
            }
            if alloc_zeroed {
                current.alloc_zeroed += 1;
            }
            if realloc {
                current.realloc += 1;
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
        record(true, false, false, layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(false, true, false, layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record(false, false, true, new_size);
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

fn allocation_tracking_works() -> bool {
    let ((), stats) = measure(|| {
        let buffer = Vec::<u8>::with_capacity(64);
        black_box(&buffer);
    });
    stats.alloc > 0 && stats.total_bytes >= 64
}

// ---------------------------------------------------------------------------
// Small source scanners. They only remove the unit-test module and inspect
// production text, so renamed helpers do not matter.
// ---------------------------------------------------------------------------

fn production_source(source: &str) -> &str {
    match source.find("\n#[cfg(test)]\nmod tests") {
        Some(index) => &source[..index],
        None => source,
    }
}

fn occurrences(haystack: &str, needle: &str) -> usize {
    haystack.match_indices(needle).count()
}

/// Returns the first textual definition of `fn <name>` including its body.
fn function_body<'a>(source: &'a str, name: &str) -> &'a str {
    let marker = format!("fn {name}");
    let start = source
        .find(&marker)
        .unwrap_or_else(|| panic!("missing function `{name}`"));
    let open = source[start + marker.len()..]
        .find('{')
        .map(|offset| start + marker.len() + offset)
        .unwrap_or_else(|| panic!("missing body for `{name}`"));
    let bytes = source.as_bytes();
    let mut depth = 0usize;
    for (offset, byte) in bytes[open..].iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &source[start..open + offset + 1];
                }
            }
            _ => {}
        }
    }
    panic!("unterminated body for `{name}`");
}

fn function_names(source: &str) -> Vec<String> {
    let bytes = source.as_bytes();
    let mut names = Vec::new();
    let mut index = 0usize;
    while let Some(found) = source[index..].find("fn ") {
        let start = index + found + "fn ".len();
        let mut end = start;
        while end < bytes.len() && (bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_') {
            end += 1;
        }
        if end > start {
            names.push(source[start..end].to_string());
            index = end;
        } else {
            index = start;
        }
    }
    names
}

/// Counts `agree_header` calls reachable from `prepare_map_reduce_many` inside
/// one file. The map-reduce-many preflight must perform exactly one fixed
/// header agreement; a second one is the #116 duplication.
fn header_agreements_on_many_path(source: &str) -> usize {
    let names = function_names(source);
    let mut seen: HashSet<String> = HashSet::new();
    let mut pending = vec!["prepare_map_reduce_many".to_string()];
    let mut total = 0usize;
    while let Some(name) = pending.pop() {
        if !seen.insert(name.clone()) {
            continue;
        }
        let body = function_body(source, &name);
        total += occurrences(body, "agree_header(");
        for candidate in &names {
            if seen.contains(candidate) {
                continue;
            }
            if body.contains(&format!("{candidate}(")) || body.contains(&format!("{candidate}::<"))
            {
                pending.push(candidate.clone());
            }
        }
    }
    total
}

/// Counts `checked_product(...)` calls whose argument text mentions a local
/// spatial length. The local storage length must be computed in one helper.
fn local_length_product_sites(source: &str) -> usize {
    let bytes = source.as_bytes();
    let mut count = 0usize;
    let mut index = 0usize;
    while let Some(found) = source[index..].find("checked_product(") {
        let open = index + found + "checked_product".len();
        let mut depth = 0usize;
        let mut end = open;
        while end < bytes.len() {
            match bytes[end] {
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        end += 1;
                        break;
                    }
                }
                _ => {}
            }
            end += 1;
        }
        let arguments = &source[open..end.min(source.len())];
        if arguments.contains("local_len") {
            count += 1;
        }
        index = open;
    }
    count
}

/// Checks that a `broadcast_linear` call is not left inside the spatial loop.
fn broadcast_hoisted_out_of_spatial_loop(body: &str) -> bool {
    if occurrences(body, "broadcast_linear(") == 0 {
        return false;
    }
    let spatial_loop = body
        .find("for k in 0..plan.spatial")
        .or_else(|| body.find("0..plan.spatial"))
        .or_else(|| body.find("0..spatial"));
    match spatial_loop {
        Some(index) => !body[index..].contains("broadcast_linear("),
        None => false,
    }
}

// ---------------------------------------------------------------------------
// Contract test.
// ---------------------------------------------------------------------------

#[test]
fn array_refactor_contracts() {
    let universe = mpi::initialize().expect("MPI initialization failed");
    let world = universe.world();
    let size = usize::try_from(world.size()).expect("world size fits");
    let topology = MpiTopology::<1>::new(&world, [size]).expect("topology");
    let mut failures: Vec<String> = Vec::new();

    let pencil = Pencil::<2, 1>::new_permuted(
        Arc::clone(&topology),
        [2 * size, 3],
        [0],
        AxisPermutation::new([1, 0]).unwrap(),
    )
    .unwrap();
    let extra = ExtraShape::new([2]).unwrap();
    let local_x = pencil.local_ranges()[0].clone();
    // The local spatial length is always 2 * 3, so the tested buffers are
    // rank-size independent.
    assert_eq!(pencil.local_len(), 6);

    // #118: valid view access must not allocate, and lookup semantics remain.
    if !allocation_tracking_works() {
        failures.push("#118: allocation counting positive control failed".to_string());
    }
    let mut storage = [0i64; 12];
    for e in 0..2usize {
        for x in 0..2usize {
            for y in 0..3usize {
                storage[e * 6 + y * 2 + x] = (100 * e + 10 * (local_x.start + x) + y) as i64;
            }
        }
    }
    let view = PencilArrayView::from_slice(&pencil, &extra, &storage).unwrap();
    let mut lookups_ok = true;
    for e in 0..2usize {
        for x in 0..2usize {
            for y in 0..3usize {
                let expected = (100 * e + 10 * (local_x.start + x) + y) as i64;
                lookups_ok &= view.get_local(&[e], [x, y]) == Some(&expected);
                lookups_ok &= view.get_global(&[e], [local_x.start + x, y]) == Some(&expected);
            }
        }
    }
    lookups_ok &= view.get_local(&[], [0, 0]).is_none();
    lookups_ok &= view.get_local(&[0, 0], [0, 0]).is_none();
    lookups_ok &= view.get_local(&[2], [0, 0]).is_none();
    lookups_ok &= view.get_local(&[0], [2, 0]).is_none();
    lookups_ok &= view.get_local(&[0], [0, 3]).is_none();
    lookups_ok &= view.get_global(&[0], [2 * size, 0]).is_none();
    if local_x.start > 0 {
        lookups_ok &= view.get_global(&[0], [local_x.start - 1, 0]).is_none();
    }
    let ((), access_stats) = measure(|| {
        for e in 0..2usize {
            for x in 0..2usize {
                for y in 0..3usize {
                    black_box(view.get_local(&[e], [x, y]));
                    black_box(view.get_global(&[e], [local_x.start + x, y]));
                }
            }
        }
    });
    if access_stats.alloc != 0 || access_stats.alloc_zeroed != 0 || access_stats.realloc != 0 {
        failures.push(format!(
            "#118: valid get_local/get_global performed {} alloc / {} alloc_zeroed / {} realloc (expected 0)",
            access_stats.alloc, access_stats.alloc_zeroed, access_stats.realloc
        ));
    }
    if !lookups_ok {
        failures.push("#118: local/global bounds and value lookups changed".to_string());
    }

    // #115: exact storage length checks keep their variant, fields and error
    // precedence (checked overflow before the mismatch).
    let short = [0i64; 11];
    let short_error = ArrayError::StorageLengthMismatch {
        required: 12,
        actual: 11,
    };
    if PencilArrayView::from_slice(&pencil, &extra, &short).unwrap_err() != short_error {
        failures.push("#115: view storage mismatch changed".to_string());
    }
    if PencilArray::<i64, 2, 1>::from_vec(Arc::clone(&pencil), extra.clone(), vec![0i64; 11])
        .unwrap_err()
        != short_error
    {
        failures.push("#115: owned array storage mismatch changed".to_string());
    }
    let permuted = pencil
        .with_permutation(AxisPermutation::new([0, 1]).unwrap())
        .unwrap();
    if ManyPencilArray::<i64, 2, 1>::from_vec(
        vec![Arc::clone(&pencil), Arc::clone(&permuted)],
        0,
        extra.clone(),
        vec![0i64; 11],
    )
    .unwrap_err()
        != short_error
    {
        failures.push("#115: many-array storage mismatch changed".to_string());
    }
    let huge = ExtraShape::new([usize::MAX / 2 + 1]).unwrap();
    let overflow = ArrayError::Geometry(GeometryError::SizeOverflow);
    let empty: [i64; 0] = [];
    if PencilArrayView::from_slice(&pencil, &huge, &empty).unwrap_err() != overflow {
        failures.push("#115: view length overflow changed".to_string());
    }
    if PencilArray::<i64, 2, 1>::from_vec(Arc::clone(&pencil), huge.clone(), Vec::new())
        .unwrap_err()
        != overflow
    {
        failures.push("#115: owned array length overflow changed".to_string());
    }
    if ManyPencilArray::<i64, 2, 1>::from_vec(
        vec![Arc::clone(&pencil), Arc::clone(&permuted)],
        0,
        huge.clone(),
        Vec::new(),
    )
    .unwrap_err()
        != overflow
    {
        failures.push("#115: many-array length overflow changed".to_string());
    }

    let array_source = production_source(include_str!("../src/array.rs"));
    let view_source = production_source(include_str!("../src/view.rs"));
    let many_source = production_source(include_str!("../src/many.rs"));
    let local_transpose_source = production_source(include_str!("../src/local_transpose.rs"));
    let transpose_source = production_source(include_str!("../src/transpose.rs"));
    let checked_source = production_source(include_str!("../src/checked.rs"));
    let length_files = [
        array_source,
        view_source,
        many_source,
        local_transpose_source,
        transpose_source,
        checked_source,
    ];
    let length_product_sites: usize = length_files
        .iter()
        .map(|source| local_length_product_sites(source))
        .sum();
    if length_product_sites > 1 {
        failures.push(format!(
            "#115: local storage length product is still computed at {length_product_sites} sites (expected one helper)"
        ));
    }
    let mismatch_sites: usize = length_files
        .iter()
        .map(|source| occurrences(source, "StorageLengthMismatch {"))
        .sum();
    if mismatch_sites > 2 {
        failures.push(format!(
            "#115: StorageLengthMismatch is still built at {mismatch_sites} sites (expected one shared helper)"
        ));
    }

    // #113: pack and unpack must share one region traversal; only the copy
    // direction may differ. A shared walker leaves one payload odometer and
    // one advance per odometer step.
    let payload_odometers = occurrences(transpose_source, "let mut payload_offset");
    if payload_odometers > 1 {
        failures.push(format!(
            "#113: pack and unpack still carry {payload_odometers} payload odometers (expected one shared walk)"
        ));
    }
    let advance_sites = occurrences(transpose_source, "advance_region(");
    if advance_sites > 3 {
        failures.push(format!(
            "#113: region traversal still advances at {advance_sites} sites (expected one shared walk)"
        ));
    }

    // #114: the normal and callback transports must share request reservation,
    // posting, and waiting, without adding another state all-reduce.
    let point_to_point_source = include_str!("../src/point_to_point_transpose.rs");
    for needle in [
        "try_reserve(receive_slots)",
        "try_reserve(send_slots)",
        "immediate_receive_into_with_tag",
        "immediate_send_with_tag",
    ] {
        let sites = occurrences(point_to_point_source, needle);
        if sites > 1 {
            failures.push(format!(
                "#114: `{needle}` still appears at {sites} sites (expected one shared operation)"
            ));
        }
    }
    let waits = occurrences(point_to_point_source, "wait_without_status");
    if waits > 2 {
        failures.push(format!(
            "#114: request waiting is still written at {waits} sites (expected shared waits)"
        ));
    }
    let reductions = occurrences(point_to_point_source, "all_reduce_into");
    if reductions != 2 {
        failures.push(format!(
            "#114: state all-reduces changed from 2 to {reductions}; the normal path must not gain one"
        ));
    }

    // #117: pointwise_many computes each input's broadcast index once per extra
    // index, outside the spatial loop.
    let pointwise_source = production_source(include_str!("../src/pointwise.rs"));
    let many_views_body = function_body(pointwise_source, "pointwise_many_views");
    if !broadcast_hoisted_out_of_spatial_loop(many_views_body) {
        failures.push(
            "#117: pointwise_many_views still computes broadcast_linear inside (or without) the spatial loop"
                .to_string(),
        );
    }

    // #116: the map-reduce-many preflight must agree exactly one fixed header.
    let collectives_source = production_source(include_str!("../src/collectives.rs"));
    let header_agreements = header_agreements_on_many_path(collectives_source);
    if header_agreements != 1 {
        failures.push(format!(
            "#116: map_reduce_many path agrees {header_agreements} fixed headers (expected one)"
        ));
    }

    // #117 behaviour: broadcast values must be unchanged by the hoist.
    let fill = |dims: [usize; 2]| -> Vec<i64> {
        let mut values = Vec::new();
        for e0 in 0..dims[0] {
            for e1 in 0..dims[1] {
                // Memory order is [extra..., y, x], with x fastest.
                for y in 0..3usize {
                    for x in 0..2usize {
                        values.push(
                            (e0 * 1_000_000 + e1 * 10_000 + (local_x.start + x) * 100 + y) as i64,
                        );
                    }
                }
            }
        }
        values
    };
    let a = PencilArray::from_vec(
        Arc::clone(&pencil),
        ExtraShape::new([1, 3]).unwrap(),
        fill([1, 3]),
    )
    .unwrap();
    let b = PencilArray::from_vec(
        Arc::clone(&pencil),
        ExtraShape::new([2, 1]).unwrap(),
        fill([2, 1]),
    )
    .unwrap();
    let mut output = PencilArray::from_vec(
        Arc::clone(&pencil),
        ExtraShape::new([2, 3]).unwrap(),
        vec![0i64; 36],
    )
    .unwrap();
    let pointwise_result = pointwise_many(&[&a, &b], &mut output, |values| *values[0] + *values[1]);
    let mut pointwise_ok = pointwise_result.is_ok();
    for e0 in 0..2usize {
        for e1 in 0..3usize {
            for x in 0..2usize {
                for y in 0..3usize {
                    let gx = local_x.start + x;
                    let a_value = (e1 * 10_000 + gx * 100 + y) as i64;
                    let b_value = (e0 * 1_000_000 + gx * 100 + y) as i64;
                    pointwise_ok &=
                        output.get_local(&[e0, e1], [x, y]) == Some(&(a_value + b_value));
                }
            }
        }
    }
    if !pointwise_ok {
        failures.push("#117: pointwise_many broadcast values changed".to_string());
    }

    // #116 behaviour: the single-header preflight still succeeds for ordinary
    // broadcasts and still reports the empty-input precondition.
    let mapped = map_reduce_many(
        topology.communicator(),
        &[a.view(), b.view()],
        0i64,
        |values: &[&i64]| *values[0] + *values[1],
        |left: i64, right: i64| left + right,
    );
    if mapped.is_err() {
        failures.push(format!(
            "#116: map_reduce_many failed after header consolidation: {mapped:?}"
        ));
    }
    let no_inputs: [PencilArrayView<'_, i64, 2, 1>; 0] = [];
    let empty_result = map_reduce_many(
        topology.communicator(),
        &no_inputs,
        0i64,
        |_values: &[&i64]| 0i64,
        |left: i64, right: i64| left + right,
    );
    if empty_result != Err(CollectiveError::CollectivePreconditionFailed) {
        failures.push(format!(
            "#116: empty many-input precondition changed: {empty_result:?}"
        ));
    }

    assert!(
        failures.is_empty(),
        "array refactor contract failures:\n{}",
        failures.join("\n")
    );
}
