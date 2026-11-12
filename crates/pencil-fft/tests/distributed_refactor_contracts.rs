//! Source-level contracts for the distributed refactor issues #103-#108.
//!
//! These refactors are not observable through the public API: the numeric
//! behaviour, error variants, collective order, and workspace capacity are
//! already pinned by the `distributed_*` suites.  What must change is that the
//! duplicated internal implementations are replaced by one shared helper.
//! These checks therefore read the production sources with the same
//! `include_str!` convention used by `distributed/fftw_tests.rs` and assert the
//! single-owner shape directly.  They deliberately derive helper names from
//! the source instead of pinning new names.

#![cfg(feature = "distributed")]

const DISTRIBUTED_RS: &str = include_str!("../src/distributed.rs");
const R2C_RS: &str = include_str!("../src/distributed/r2c.rs");
const R2R_RS: &str = include_str!("../src/distributed/r2r.rs");
const MIXED_RS: &str = include_str!("../src/distributed/mixed.rs");
const C2R_RS: &str = include_str!("../src/distributed/c2r.rs");
const COLLECTIONS_RS: &str = include_str!("../src/distributed/collections.rs");

/// Drops the `#[cfg(test)] mod ...` tail so test fixtures cannot satisfy (or
/// trip) a production-source contract.
fn production(source: &str) -> &str {
    match source.find("\n#[cfg(test)]\nmod ") {
        Some(index) => &source[..index],
        None => source,
    }
}

fn line_count(source: &str, needle: &str) -> usize {
    source.lines().filter(|line| line.contains(needle)).count()
}

#[derive(Clone, Copy)]
struct Function<'a> {
    name: &'a str,
    body: &'a str,
}

/// Minimal top-level function scanner: a function starts on a line whose
/// trimmed text is `[pub...] fn name`, and its body ends at the matching brace.
fn functions(source: &str) -> Vec<Function<'_>> {
    let mut result = Vec::new();
    let mut offset = 0;
    for line in source.split_inclusive('\n') {
        let trimmed = line.trim_start();
        let is_function = trimmed.starts_with("fn ")
            || trimmed.starts_with("pub fn ")
            || trimmed.starts_with("pub(super) fn ")
            || trimmed.starts_with("pub(crate) fn ")
            || trimmed.starts_with("const fn ");
        if is_function {
            let start = offset + (line.len() - trimmed.len());
            let rest = &source[start..];
            let Some(keyword) = rest.find("fn ") else {
                offset += line.len();
                continue;
            };
            let after = &rest[keyword + 3..];
            let name_end = after
                .find(|character: char| !(character.is_alphanumeric() || character == '_'))
                .unwrap_or(after.len());
            let name = &after[..name_end];
            if let Some(body) = function_body(rest) {
                result.push(Function { name, body });
            }
        }
        offset += line.len();
    }
    result
}

fn function_body(rest: &str) -> Option<&str> {
    let open = rest.find('{')?;
    let mut depth = 0usize;
    for (index, character) in rest[open..].char_indices() {
        match character {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&rest[..open + index + 1]);
                }
            }
            _ => {}
        }
    }
    None
}

fn signature<'a>(source: &'a str, needle: &str) -> &'a str {
    let start = source
        .find(needle)
        .unwrap_or_else(|| panic!("missing `{needle}`"));
    let rest = &source[start..];
    let open = rest
        .find('{')
        .unwrap_or_else(|| panic!("missing body after `{needle}`"));
    &rest[..open]
}

fn function_containing<'a>(sources: &[&'a str], needle: &str) -> Vec<Function<'a>> {
    sources
        .iter()
        .flat_map(|source| functions(source))
        .filter(|function| function.body.contains(needle))
        .collect()
}

/// #108: one generic mapper each for `OverlapError` and `OverwriteError`.
#[test]
fn overlap_and_overwrite_mappings_are_single_generic_helpers() {
    let sources = [
        production(DISTRIBUTED_RS),
        production(R2C_RS),
        production(MIXED_RS),
    ];
    let total = |needle: &str| -> usize {
        sources
            .iter()
            .map(|source| line_count(source, needle))
            .sum()
    };

    assert_eq!(
        total("OverwriteError::Array"),
        1,
        "#108: OverwriteError::Array must be mapped in one generic helper, not per call site"
    );
    assert_eq!(
        total("OverwriteError::Writer"),
        1,
        "#108: OverwriteError::Writer mapping must live in the same helper"
    );
    for variant in [
        "OverlapError::Transpose",
        "OverlapError::Callback",
        "OverlapError::PeerPanicked",
        "OverlapError::PeerCallbackFailed",
        "OverlapError::CollectivePreconditionFailed",
    ] {
        assert_eq!(
            total(variant),
            1,
            "#108: every `{variant}` mapping must be preserved exactly once"
        );
    }

    let overlap = function_containing(&sources, "OverlapError::Transpose");
    assert_eq!(
        overlap.len(),
        1,
        "#108: the OverlapError mapping must be a single function, found {}",
        overlap.len()
    );
    let overlap_signature = overlap[0].body.split('{').next().unwrap();
    assert!(
        overlap_signature.contains('<'),
        "#108: the single OverlapError mapper must be generic: {overlap_signature}"
    );

    let overwrite = function_containing(&sources, "OverwriteError::Array");
    assert_eq!(
        overwrite.len(),
        1,
        "#108: the OverwriteError mapping must be a single function, found {}",
        overwrite.len()
    );
    let overwrite_signature = overwrite[0].body.split('{').next().unwrap();
    assert!(
        overwrite_signature.contains('<'),
        "#108: the single OverwriteError mapper must be generic: {overwrite_signature}"
    );
}

/// #103: mixed R2R/DHT stages reuse the single strided wrapper in `r2r.rs`
/// instead of re-implementing `MixedR2rLocal` and the line loops.
#[test]
fn mixed_reuses_the_r2r_strided_wrapper() {
    let all_sources = [
        production(DISTRIBUTED_RS),
        production(R2C_RS),
        production(R2R_RS),
        production(MIXED_RS),
        production(C2R_RS),
        production(COLLECTIONS_RS),
    ];
    let mixed = production(MIXED_RS);

    let local_plans: usize = all_sources
        .iter()
        .map(|source| line_count(source, "MixedR2rLocal"))
        .sum();
    assert_eq!(
        local_plans, 0,
        "#103: MixedR2rLocal duplicates R2rLocal and must be removed"
    );

    assert_eq!(
        line_count(mixed, "for_each_strided_line"),
        0,
        "#103: mixed must call the shared R2R/DHT strided wrapper, not drive strided lines itself"
    );
    assert!(
        mixed
            .lines()
            .any(|line| line.contains("R2rLocal") && !line.contains("MixedR2rLocal")),
        "#103: mixed must reuse r2r::R2rLocal"
    );
    assert!(
        line_count(production(R2R_RS), "for_each_strided_line") >= 1,
        "#103: the strided R2R/DHT wrapper must live in r2r.rs"
    );
}

/// #104: mixed's R2C boundary reuses the generic strided real/complex wrapper
/// in `r2c.rs` instead of keeping a second copy.
#[test]
fn mixed_reuses_the_r2c_boundary_strided_wrapper() {
    let r2c = production(R2C_RS);
    let mixed = production(MIXED_RS);

    assert_eq!(
        line_count(mixed, "for_each_strided_line_differing"),
        0,
        "#104: mixed must reuse r2c's boundary strided wrapper instead of iterating differing lines itself"
    );
    assert!(
        line_count(r2c, "for_each_strided_line_differing") >= 1,
        "#104: the generic R2C boundary strided wrapper must live in r2c.rs"
    );
}

/// #105: `build_transitions` takes the pencil columns directly, so r2r and
/// mixed no longer fabricate `TransformStage`/`StagePreparation` values with
/// `LocalTransform::Identity` only to recover `input`/`output` pencils.
#[test]
fn transitions_take_pencil_columns_without_dummy_stages() {
    let distributed = production(DISTRIBUTED_RS);
    let r2r = production(R2R_RS);
    let mixed = production(MIXED_RS);

    assert_eq!(
        line_count(r2r, "StagePreparation"),
        0,
        "#105: r2r must pass its route pencils to build_transitions, not a dummy StagePreparation"
    );
    assert_eq!(
        line_count(mixed, "StagePreparation"),
        0,
        "#105: mixed must pass its stage pencils to build_transitions, not a dummy StagePreparation"
    );
    assert_eq!(
        line_count(r2r, "local: super::LocalTransform::Identity"),
        0,
        "#105: r2r's dummy TransformStage/Identity construction must be removed"
    );
    assert_eq!(
        line_count(mixed, "local: LocalTransform::Identity"),
        0,
        "#105: mixed's dummy TransformStage/Identity constructions must be removed"
    );

    let build_transitions = signature(distributed, "fn build_transitions");
    assert!(
        !build_transitions.contains("StagePreparation"),
        "#105: build_transitions must not consume StagePreparation: {build_transitions}"
    );
    assert!(
        build_transitions.contains("Pencil"),
        "#105: build_transitions must take pencil columns directly: {build_transitions}"
    );
}

/// #106: the in-place real/complex repack (strided slot promotion, packed row
/// FFTs, and reverse slot demotion) has one kernel shared by r2c and mixed.
#[test]
fn in_place_repack_has_one_shared_kernel() {
    let r2c = production(R2C_RS);
    let mixed = production(MIXED_RS);
    let all_sources = [
        production(DISTRIBUTED_RS),
        r2c,
        mixed,
        production(R2R_RS),
        production(C2R_RS),
        production(COLLECTIONS_RS),
    ];

    // Distinctive statements of the duplicated repack bodies.  They may be
    // renamed when moved into the shared kernel, so only the duplicate in
    // mixed is required to disappear and at most one copy may remain.
    for marker in [
        "real_storage_len).rev()",
        "Complex::new(value, R::zero())",
        "real_storage[i] = value;",
        "for row in 0..rows",
        "for row in (0..rows).rev()",
    ] {
        assert_eq!(
            line_count(mixed, marker),
            0,
            "#106: mixed must call the shared in-place repack kernel instead of duplicating `{marker}`"
        );
        let total: usize = all_sources
            .iter()
            .map(|source| line_count(source, marker))
            .sum();
        assert!(
            total <= 1,
            "#106: the in-place repack `{marker}` must have one shared implementation, found {total}"
        );
    }
}

/// #107: the DC/Nyquist endpoint rule (DC always, last bin only when the
/// reduced length is even) is one predicate consumed by both the endpoint
/// check and endpoint zeroing in r2c and mixed.
#[test]
fn dc_nyquist_rule_is_one_predicate_shared_by_check_and_zeroing() {
    let r2c = production(R2C_RS);
    let mixed = production(MIXED_RS);
    let sources = [r2c, mixed];
    let targets = [
        "validate_boundary",
        "zero_accepted_boundary",
        "validate_mixed_boundary_data",
        "zero_mixed_boundary_data",
    ];

    let predicates: Vec<Function<'_>> = sources
        .iter()
        .flat_map(|source| functions(source))
        .filter(|function| {
            !targets.contains(&function.name)
                && function.body.lines().count() <= 100
                && function.body.contains("k == 0")
                && function.body.contains("plane_count")
        })
        .collect();
    assert_eq!(
        predicates.len(),
        1,
        "#107: the DC/Nyquist endpoint rule must be one predicate shared by r2c and mixed, found {}",
        predicates.len()
    );
    let predicate = predicates[0].name;

    for target in targets {
        let function = sources
            .iter()
            .flat_map(|source| functions(source))
            .find(|function| function.name == target)
            .unwrap_or_else(|| panic!("#107: missing `{target}`"));
        assert!(
            function.body.contains(&format!("{predicate}("))
                || function.body.contains(&format!("{predicate}::<")),
            "#107: `{target}` must use the shared `{predicate}` predicate"
        );
    }
}
