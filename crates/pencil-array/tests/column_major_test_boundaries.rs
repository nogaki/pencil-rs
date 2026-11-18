use std::{fs, path::Path};

const MODULES: &[(&str, &str, &str, &str)] = &[
    (
        "src/geometry.rs",
        "tests",
        "tests/geometry.rs",
        "src/tests/geometry.rs",
    ),
    ("src/view.rs", "tests", "tests/view.rs", "src/tests/view.rs"),
    (
        "src/transpose.rs",
        "tests",
        "tests/transpose.rs",
        "src/tests/transpose.rs",
    ),
    (
        "src/spatial_remap.rs",
        "tests",
        "tests/spatial_remap.rs",
        "src/tests/spatial_remap.rs",
    ),
    (
        "../pencil-fft/src/distributed.rs",
        "tests",
        "distributed/tests/layout_tests.rs",
        "../pencil-fft/src/distributed/tests/layout_tests.rs",
    ),
    (
        "../pencil-fft/src/distributed/r2c.rs",
        "endpoint_reduction_tests",
        "r2c/tests/endpoint_reduction.rs",
        "../pencil-fft/src/distributed/r2c/tests/endpoint_reduction.rs",
    ),
    (
        "../pencil-fft/src/distributed/mixed.rs",
        "reverse_overlap_tests",
        "mixed/tests/reverse_overlap.rs",
        "../pencil-fft/src/distributed/mixed/tests/reverse_overlap.rs",
    ),
];

fn inline_test_body<'a>(source: &'a str, module_name: &str) -> Option<&'a str> {
    let marker = format!("#[cfg(test)]\nmod {module_name} {{");
    let body = source.split_once(&marker)?.1.strip_prefix('\n')?;
    Some(body.rsplit_once("\n}")?.0)
}

fn normalized_test_body(body: &str) -> String {
    let compact: String = body
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    compact
        .replace(",)", ")")
        .replace(",]", "]")
        .replace(",}", "}")
}

#[test]
fn layout_sensitive_unit_modules_use_their_external_test_files() {
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));

    let mut missing_declarations = Vec::new();
    for &(source_path, module_name, module_path, external_path) in MODULES {
        let source = fs::read_to_string(crate_root.join(source_path))
            .unwrap_or_else(|error| panic!("read {source_path}: {error}"));
        let external_file = crate_root.join(external_path);
        assert!(
            external_file.is_file(),
            "{source_path}::{module_name} external test file is missing: {external_path}"
        );
        let external_body = fs::read_to_string(&external_file)
            .unwrap_or_else(|error| panic!("read {external_path}: {error}"));
        if let Some(inline_body) = inline_test_body(&source, module_name) {
            assert_eq!(
                normalized_test_body(inline_body),
                normalized_test_body(&external_body),
                "{source_path}::{module_name} external test body differs from the inline original"
            );
        }
        assert_eq!(
            Path::new(source_path).parent().unwrap().join(module_path),
            Path::new(external_path),
            "{source_path}::{module_name} path attribute does not resolve to {external_path}"
        );

        let declaration = format!("#[cfg(test)]\n#[path = \"{module_path}\"]\nmod {module_name};");
        if !source.contains(&declaration) {
            missing_declarations.push(format!(
                "{source_path}::{module_name} must be wired to {external_path} with:\n{declaration}"
            ));
        }
    }

    assert!(
        missing_declarations.is_empty(),
        "layout-sensitive test modules are not externally wired:\n{}",
        missing_declarations.join("\n")
    );
}
