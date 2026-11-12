//! Structural contracts for the local FFT refactor (#97-#102, #119-#121).
//!
//! These requirements only remove duplication from the implementation, so
//! there is no public behavior that can observe them. The checks therefore
//! inspect the crate sources the same way the existing `include_str`
//! contracts in `src/distributed/fftw_tests.rs` do, and each test names the
//! requirement it enforces.

const BACKEND: &str = include_str!("../src/backend.rs");
const LIB: &str = include_str!("../src/lib.rs");
const R2C: &str = include_str!("../src/r2c.rs");
const R2R: &str = include_str!("../src/r2r.rs");
const DHT: &str = include_str!("../src/dht.rs");

/// Unit tests live in a `#[cfg(test)] mod tests` block at the end of a file.
/// Production code must not accidentally satisfy (or break) these contracts
/// through test-only mocks.
fn production(source: &str) -> &str {
    source
        .split("#[cfg(test)]\nmod tests {")
        .next()
        .unwrap_or(source)
}

fn normalized(source: &str) -> String {
    source.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn count(haystack: &str, needle: &str) -> usize {
    haystack.match_indices(needle).count()
}

/// Returns the brace-delimited block that starts at `open`.
fn brace_block(source: &str, open: usize) -> &str {
    let mut depth = 0usize;
    for (offset, character) in source[open..].char_indices() {
        match character {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return &source[open..=open + offset];
                }
            }
            _ => {}
        }
    }
    panic!("unbalanced braces at byte {open}");
}

/// Returns the body of the first function starting with `signature`.
fn body_of<'a>(source: &'a str, signature: &str) -> &'a str {
    let start = source
        .find(signature)
        .unwrap_or_else(|| panic!("missing function signature: {signature}"));
    let open = start + source[start..].find('{').expect("missing function body");
    brace_block(source, open)
}

/// Returns the bodies of all freely defined functions, nested ones included.
fn function_bodies(source: &str) -> Vec<&str> {
    let mut bodies = Vec::new();
    let mut offset = 0;
    while let Some(found) = source[offset..].find("fn ") {
        let start = offset + found;
        let Some(relative_open) = source[start..].find('{') else {
            break;
        };
        let open = start + relative_open;
        if source[start..open].contains(';') {
            offset = open;
            continue;
        }
        let body = brace_block(source, open);
        bodies.push(body);
        offset = open + body.len();
    }
    bodies
}

/// Returns the bodies of all `struct`/`enum` declarations in `source`.
fn type_bodies(source: &str) -> Vec<&str> {
    let mut bodies = Vec::new();
    let mut offset = 0;
    while offset < source.len() {
        let next = ["struct ", "enum "]
            .iter()
            .filter_map(|keyword| {
                source[offset..]
                    .find(keyword)
                    .map(|at| (offset + at, *keyword))
            })
            .min_by_key(|(at, _)| *at);
        let Some((start, _)) = next else {
            break;
        };
        let Some(relative_open) = source[start..].find('{') else {
            break;
        };
        let open = start + relative_open;
        let body = brace_block(source, open);
        bodies.push(body);
        offset = open + body.len();
    }
    bodies
}

/// #97: out-of-place forward takes no complex line buffer and out-of-place
/// inverse takes no real line buffer. Validating a fabricated "required"
/// length for those buffers can never fail, so the fake-length branch must be
/// removed while the real, complex, and scratch checks stay.
#[test]
fn r2c_out_of_place_validation_does_not_invent_line_buffers() {
    let source = normalized(production(R2C));
    assert!(
        !source.contains("source_line_is_real"),
        "out-of-place validation still selects a fabricated line length for a buffer it does not own"
    );
    assert!(
        !source.contains("self.complex_len, scratch_len"),
        "forward validation still passes the plan's complex length for a buffer it does not accept"
    );
    assert!(
        !source.contains("self.real_len, complex_line_len"),
        "inverse validation still passes the plan's real length for a buffer it does not accept"
    );
    for required in ["RealLineTooSmall", "ComplexLineTooSmall", "ScratchTooSmall"] {
        assert!(
            source.contains(required),
            "line or scratch validation disappeared: {required}"
        );
    }
}

/// #98: a packed in-place operation derives `(real_len, batch_count)` once and
/// casts each backing view once instead of re-deriving lengths and recasting
/// the same view per batch.
#[test]
fn r2c_in_place_lengths_are_derived_once_and_views_cast_once() {
    let source = production(R2C);
    for body in function_bodies(source) {
        let flat = normalized(body);
        assert!(
            !(flat.contains("checked_in_place_lengths(") && flat.contains(".checked_lengths()")),
            "an in-place path derives the packed lengths twice: {body}"
        );
        assert!(
            count(body, "cast_real_slice_mut(") <= 1,
            "an in-place path casts the mutable real view more than once: {body}"
        );
    }
}

/// #100: out-of-place and in-place forward share one per-line routine, and the
/// same holds for inverse, so RealFFT is invoked from shared helpers only.
#[test]
fn r2c_line_processing_is_shared_between_place_modes() {
    let calls = count(production(R2C), "process_with_scratch");
    assert!(
        calls <= 2,
        "out-of-place/in-place line processing is not shared: {calls} RealFFT call sites"
    );
}

/// #99: the backend kind and the FFTW options are held in one common type in
/// `backend.rs`; the plans delegate to it instead of storing both fields.
#[test]
fn local_plans_share_backend_kind_and_options() {
    for (name, source) in [
        ("lib.rs", LIB),
        ("r2c.rs", R2C),
        ("r2r.rs", R2R),
        ("dht.rs", DHT),
    ] {
        let flat = normalized(production(source));
        assert!(
            !flat.contains("backend_options:"),
            "{name} still stores the FFTW options instead of the shared backend type"
        );
        assert!(
            !flat.contains("backend: BackendKind") && !flat.contains("backend: super::BackendKind"),
            "{name} still stores the backend kind directly instead of the shared backend type"
        );
    }
    let shared = type_bodies(production(BACKEND)).into_iter().any(|body| {
        let flat = normalized(body);
        flat.contains("BackendKind") && flat.contains("PlanOptions")
    });
    assert!(
        shared,
        "backend.rs must define the shared backend kind + options type"
    );
}

/// #121: the r2c backend is constructed through the backend module, so the
/// test factory hook and the direct `pencil_fftw::plan_*` calls are no longer
/// duplicated in `r2c.rs`.
#[test]
fn r2c_native_planning_goes_through_backend_module() {
    let source = production(R2C);
    assert!(
        !source.contains("before_factory"),
        "r2c.rs still calls the native factory hook directly"
    );
    assert!(
        !source.contains("pencil_fftw::plan_"),
        "r2c.rs still plans native transforms directly"
    );
    assert!(
        source.contains("backend::"),
        "r2c.rs does not build native plans through the backend module"
    );
    assert!(
        production(BACKEND).contains("before_factory"),
        "backend.rs must keep the native factory hook"
    );
    let hook_calls: usize = [LIB, R2C, R2R, DHT]
        .iter()
        .map(|source| count(production(source), "before_factory"))
        .sum();
    assert_eq!(
        hook_calls, 0,
        "native plan construction outside backend.rs still calls the factory hook"
    );
}

/// #101: the Hartley transform is self-paired, so backward delegates to the
/// forward implementation (and likewise in place) instead of repeating the
/// same unnormalized call.
#[test]
fn dht_backward_delegates_to_forward() {
    let source = production(DHT);
    let backward = body_of(source, "pub fn backward(");
    assert!(
        backward.contains("self.forward("),
        "DHT backward must delegate to forward"
    );
    let backward_in_place = body_of(source, "pub fn backward_in_place(");
    assert!(
        backward_in_place.contains("self.forward_in_place("),
        "DHT backward_in_place must delegate to forward_in_place"
    );
}

/// #102: identical embedding match arms are merged: DCT-II with DCT-IV and
/// DST-II with DST-IV.
#[test]
fn r2r_embedding_merges_equivalent_kind_arms() {
    let source = production(R2R);
    let block = source
        .match_indices("match kind {")
        .map(|(start, matched)| brace_block(source, start + matched.len() - 1))
        .find(|block| block.contains("put_pair"))
        .expect("R2R embedding match over R2rKind");
    let arms: Vec<String> = block
        .lines()
        .filter_map(|line| {
            line.split_once("=>")
                .map(|(pattern, _)| normalized(pattern))
        })
        .filter(|pattern| pattern.contains("R2rKind::"))
        .collect();
    let dct = arms
        .iter()
        .find(|arm| arm.contains("R2rKind::DctII"))
        .expect("DCT-II embedding arm");
    assert!(
        dct.contains("R2rKind::DctIV"),
        "DCT-II and DCT-IV embedding arms are not merged: {dct}"
    );
    let dst = arms
        .iter()
        .find(|arm| arm.contains("R2rKind::DstII"))
        .expect("DST-II embedding arm");
    assert!(
        dst.contains("R2rKind::DstIV"),
        "DST-II and DST-IV embedding arms are not merged: {dst}"
    );
}

/// #119: R2rKernel dispatch and native output shaping are shared in `r2r.rs`,
/// so adding a backend variant does not require editing `dht.rs`.
#[test]
fn dht_does_not_branch_on_r2r_backend_variants() {
    let source = production(DHT);
    assert!(
        !source.contains("native_r2r_output"),
        "dht.rs still formats native output itself"
    );
    assert!(
        !source.contains("R2rKernel::Native(_)"),
        "dht.rs still tests the native kernel variant"
    );
    for body in function_bodies(source) {
        assert!(
            !(body.contains("R2rKernel::Rust") && body.contains("R2rKernel::Native")),
            "dht.rs still branches on R2rKernel variants: {body}"
        );
    }
    assert!(
        production(R2R).contains("R2rKernel::Native"),
        "r2r.rs must own the shared variant dispatch"
    );
}

/// #120: the public forward/backward out-of-place/in-place paths delegate to a
/// common execution helper instead of calling RustFFT themselves.
#[test]
fn local_c2c_paths_delegate_to_shared_execution() {
    let source = production(LIB);
    for signature in [
        "pub fn forward(",
        "pub fn backward(",
        "pub fn forward_in_place(",
        "pub fn backward_in_place(",
    ] {
        let body = body_of(source, signature);
        assert!(
            !body.contains("with_scratch"),
            "{signature} still calls RustFFT directly instead of a shared helper"
        );
    }
    let calls =
        count(source, "process_immutable_with_scratch") + count(source, ".process_with_scratch");
    assert!(
        calls <= 2,
        "the four local c2c execution paths are not shared: {calls} RustFFT call sites"
    );
}
