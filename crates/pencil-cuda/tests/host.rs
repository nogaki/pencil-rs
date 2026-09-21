#[test]
fn default_boundary_has_no_mpi_or_cpu_fft() {
    // The optional distributed dependencies must not enter the default graph.
    let manifest = include_str!("../Cargo.toml");
    assert!(manifest.contains("default = []"));
    for line in manifest.lines().filter(|l| {
        l.starts_with("mpi =") || l.starts_with("pencil-array =") || l.starts_with("pencil-fft =")
    }) {
        assert!(line.contains("optional = true"), "{line}");
    }
    let implementation = include_str!("../src/lib.rs");
    assert!(!implementation.contains("rustfft"));
    assert!(!implementation.contains("pencil_fft"));
    assert!(!implementation.contains("NormalizationUnavailable"));
}
