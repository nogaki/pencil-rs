# Releasing the Cargo packages

The publishable components are `pencil-array`, `pencil-fftw`, `pencil-fft`, and
`pencil-io`, initially version 0.1.0. Package preparation is not publication.
Crates.io API name lookups returning 404 do not reserve names or guarantee future
availability. No Julia/GPU helper packages are part of this release.

## Prerequisites

- Review a clean, committed tree and each package's metadata, README, LICENSE,
  and NOTICE.md. Keep the four license/notice copies byte-identical to the root.
- Preserve Rust 1.85 support (edition 2024). Check with exactly Rust 1.85.0 as well
  as stable. Workspace staging commands below require a recent stable Cargo
  supporting native `--workspace` packaging/publish dry-runs (tested with 1.98.1).
  Cargo 1.85 accepts `package --workspace` but cannot stage these unpublished
  intercrate dependencies offline. Use recent stable Cargo for staging and set
  `RUSTC`/`RUSTDOC` to the installed 1.85.0 binaries to verify the archives with
  the MSRV compiler; do not substitute workspace paths for packaged dependencies.
- Native checks need MPI headers, `mpicc`, libclang, and parallel HDF5 built with
  the same MPI. Use the same MPI at runtime. FFTW checks need separately installed
  f32/f64 base and thread runtimes. Default `pencil-fft` is RustFFT/RealFFT and
  MPI-free; native FFTW remains explicit opt-in. Julia is not required.
- Use a clean checkout and separate target directories for each compiler.
  A private Cargo home can isolate registry/build settings. Do not include
  credentials or environment dumps in verification logs or package archives.

## Preparation and verification

From the workspace root, with the required native development tools available:

```sh
cargo fmt --all -- --check
cargo metadata --no-deps --format-version 1 --locked
cargo package --workspace --all-features --locked
cargo publish --dry-run --workspace --all-features --locked
```

Verify package builds with the installed exact MSRV compiler as well:

```sh
RUSTC="$(rustup which --toolchain 1.85.0 rustc)" \
RUSTDOC="$(rustup which --toolchain 1.85.0 rustdoc)" \
CARGO_TARGET_DIR=target/package-msrv \
  cargo +stable package --workspace --all-features --locked
```

This uses stable Cargo's workspace staging while compiling with Rust 1.85.0;
it is not a claim that Cargo 1.85 provides the same staging feature.

During uncommitted preparation only, append `--allow-dirty` to package and
publish dry-run commands. Repeat without it after committing the final tree.
Use Cargo's built-in workspace staging, not a custom local registry. Packaging
can use `--offline` when dependencies are cached; a publish dry-run may still
need the registry API and must not be described as successful if network access
fails. Refreshing registry data requires permitted network access.

Inspect every `.crate` archive in each configured `$CARGO_TARGET_DIR/package`
(default `target/package`; the MSRV example uses `target/package-msrv/package`):
normalized/original Cargo manifests, generated Cargo metadata/lockfile, README,
LICENSE, NOTICE.md, Rust source/tests/examples, and the FFT collection panic test
script only. Confirm `pencil-io/tests/support/mod.rs` (included by a unit test),
FFTW's README (included by rustdoc), and FFT source files used by `include_str!`
are present. No environment files, credentials, targets, logs, audit/planning
material, Julia/GPU helpers, or temporary files may ship. Cargo may generate
`.cargo_vcs_info.json`; this is package provenance metadata, not a user dotfile.

Compile all targets and run library/doctest checks on both toolchains:

```sh
for toolchain in stable 1.85.0; do
  export CARGO_TARGET_DIR="target/release-check-$toolchain"
  cargo +"$toolchain" check --workspace --all-features --all-targets --locked
  cargo +"$toolchain" test --workspace --all-features --all-targets --no-run --locked
  cargo +"$toolchain" test --workspace --all-features --lib --locked -- --test-threads=1
  cargo +"$toolchain" test --workspace --all-features --doc --locked
done
unset CARGO_TARGET_DIR
cargo +stable test -p pencil-fft --no-default-features --locked
cargo +stable test -p pencil-fftw --lib --locked -- --ignored --test-threads=1
cargo +stable test -p pencil-fft --features fftw --test fftw_local --locked -- --ignored --test-threads=1
```

Run the supported MPI matrices in [README.md](README.md#verification) and
[the workflow](.github/workflows/ci.yml). The core array geometry suites support
ranks 1/4; distributed transpose/FFT/I/O suites support 1/4/6. For example:

```sh
for n in 1 4 6; do
  timeout --kill-after=5s 240s mpiexec --oversubscribe -n "$n" \
    cargo +stable test -p pencil-array --test alltoallv_transpose --locked -- --test-threads=1
  timeout --kill-after=5s 240s mpiexec --oversubscribe -n "$n" \
    cargo +stable test -p pencil-fft --features distributed --test distributed_c2c --locked -- --test-threads=1
  timeout --kill-after=5s 240s mpiexec --oversubscribe -n "$n" \
    cargo +stable test -p pencil-io --features parallel-hdf5 --test hdf5_io --locked -- --test-threads=1
done
```

Native FFTW's ignored tests require both precisions. MPI collectives require all
ranks to participate in order, and persistent I/O sessions must be closed
collectively before MPI finalization. Package build verification alone does not
execute these runtime checks; an unavailable prerequisite is not a passing test.

## Publication gate and order

**Stop for explicit user confirmation before any actual `cargo publish`.**
Dry-runs do not upload and need no login or token request. No automatic upload,
CI credential setup, or credential inspection is part of this procedure.

Actual publication requires access to the registry's upload API, authorization
for the chosen crate names, acceptance of registry policy/metadata requirements,
and an unused version. Authentication is a separate user-controlled operation;
never request tokens in reports or logs. API/index availability must be checked
again at release time; dry-run success is not registry acceptance.

Publish `pencil-array` and `pencil-fftw` first (independent of each other). Wait
until both versions are available in the registry index before publishing
`pencil-fft`; `pencil-io` depends only on `pencil-array` and can follow it.
Workspace dry-runs stage dependencies locally; this does not prove those
versions have propagated in the real registry.

All intercrate path dependencies also specify `version = "0.1.0"`, Cargo's
compatible `^0.1.0` range (`>=0.1.0, <0.2.0`). Packaged manifests use registry
versions rather than local paths. Keep these requirements compatible when
changing package versions, regenerate/review Cargo.lock, and repeat verification.
Do not bump 0.1.0 merely for preparation; confirmed name/version conflicts need
a separate decision. Registry releases are immutable.
