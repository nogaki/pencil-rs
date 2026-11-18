# AGENTS.md

Guidance for agents working in this repository.

Before acting, read the shared tensor4all agent rules, starting from
`https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/index.md`
(offline fallback: `../tensor4all-agent-rules/rules/index.md`). Load only the rule
files the task needs: the common rules for any implementation work, the Rust rules
for this workspace, and the provenance rules before writing code that follows a
third-party reference. If neither source is reachable, continue from this file and
say in the pull request that the shared rules were unavailable.

Then read `README.md`. It is the user guide and the single statement of the
supported API surface, the feature matrix, the prerequisites, the verification
procedure and the CI jobs. Read `RELEASING.md` before any packaging or publication
work, and the `README.md` of a crate you change (each is also its docs.rs landing
page). `NOTICE.md` records the PencilArrays.jl / PencilFFTs.jl references and
their licenses; it is provenance, not a build input.

## Working tree and branches

- Create a worktree and a branch before starting a feature, and implement, test and
  verify inside it; do not develop features directly in the shared checkout:
  `git worktree add ../pencil-rs-<topic> -b <topic> origin/dev`.
- Branch from `origin/dev`. Feature and fix pull requests target `dev`, and `dev`
  reaches `main` through its own pull request. `main` carries the repository
  ruleset: pull requests, the six required checks (`local-fft`, `rust`, `msrv`,
  `serial-hdf5`, `rsmpi-rt`, `mpi-io-openmpi4`) and a branch up to date with
  `main`. `ci.yml` defines a seventh job, `reference-source-integrity`, which the
  ruleset does not require. Renaming or removing a CI job therefore also updates
  that ruleset.
- Never push to a branch whose pull request is already merged.

## Layout

`crates/pencil-array` — `Pencil` spatial layouts, `PencilArray`,
`ManyPencilArray`, borrowed views, local/Alltoallv/point-to-point transposes,
spatial remap and the collective reductions. It is intentionally independent of
RustFFT, RealFFT, FFTW and any FFT API; that independence is the dependency set
itself, so a new dependency there is a design change.

`crates/pencil-fft` — local C2C/R2C/C2R/R2R/DHT plans, MPI- and FFTW-free by
default (`default = []`). The opt-in `distributed` feature adds the plans over
`pencil-array`'s checked transports; `fftw` adds the optional native backend.
CI fails if the default tree resolves `mpi`, `pencil-array` or `pencil-fftw`.
Tests that need a feature declare it in `required-features`, so such a test target
does not build without it.

`crates/pencil-fftw` — the opt-in native FFTW adapter for the RustFFT/RealFFT
traits. It loads a separately installed FFTW runtime at execution time, vendors
no FFTW code, and an ordinary build or test does not need FFTW installed.

`crates/pencil-io` — MPI-IO by default; `hdf5` adds serial HDF5 without MPI,
`parallel-hdf5` adds collective HDF5. CI fails if the serial HDF5 tree resolves
`mpi`, `mpi-sys`, `pencil-array` or the MPI build probes.

Supporting trees: `tools/fftw-reference/` (opt-in Julia/FFTW reference oracle and
its source-integrity check), `tools/rsmpi-rt.toml` (experimental runtime-MPI
patch), `.github/workflows/ci.yml` (the gate), `.github/audit/` (the daily
scheduled design/ponytail audit; Japanese prompts, files issues through
`issues.py`). There is no `docs/` tree, no `CONTRIBUTING.md` and no separate
rules file: `README.md` plus the crate READMEs are the documentation.

## Build and test

Prerequisites: Rust 1.96 or newer, a C MPI with `mpicc`/`mpiexec`, `libclang`
(bindgen builds the MPI bindings), HDF5 1.12.1 or newer for the I/O features, and
separately installed `f32`+`f64` FFTW runtimes for the native tests. Default
`pencil-fft` and serial-only `pencil-io` need no MPI. A missing native
prerequisite means a run was not performed; it never means it passed.

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo clippy -p pencil-fft --features distributed --all-targets --locked -- -D warnings
cargo clippy -p pencil-fft --all-features --all-targets --locked -- -D warnings
cargo clippy -p pencil-io --all-features --all-targets --locked -- -D warnings
cargo test --workspace --lib --locked
cargo test -p pencil-fft --no-default-features --locked
cargo test -p pencil-fftw --locked
cargo check -p pencil-fft --features distributed --all-targets --locked
cargo test --workspace --doc --locked -- --show-output
cargo doc --workspace --no-deps --locked
cargo +1.96.0 check --workspace --all-targets --locked
```

MPI behaviour is not covered by those. Run the matrices from
[`README.md` §Verification](README.md#verification) with
`mpiexec --oversubscribe -n <ranks>`, always `--test-threads=1`: `topology`,
`pencil`, `array`, `many` and `local_transpose` at 1 and 4 ranks; `array_access`,
`collectives`, `alltoallv_transpose` and the distributed FFT suites at 1, 4 and 6;
the `pencil-io` MPI-IO and parallel-HDF5 suites also cover 2 ranks. Each
integration test binary initializes MPI once per process, so keep
one binary per `mpiexec` invocation, drop every topology and array before MPI
finalizes, and wrap runs in `timeout` so a deadlock fails instead of hanging.

Test placement: unit tests live in the module under `#[cfg(test)] mod tests`, and
integration tests in `crates/<crate>/tests/<suite>.rs`. Prefer extending an
existing suite family over adding a parallel one: `*_refactor_contracts.rs`
(structural and API contracts), `*_timing_overlap*.rs` (progress, send overlap,
timing), `local_c2c_nd_no_alloc.rs` and `mapped_reduction_alloc.rs` (allocation
counting through a global allocator), `legacy_*_compat.rs` (behaviour of the
compatibility aliases), `collections_panic.sh` (a failing member must fail stop,
not hang) and `staging_memory` (whole-process live bytes). Native FFTW tests are
`#[ignore]`d and need both precisions, run with `--ignored`.

## Verification and adoption

Scale the gate to the change; the hosted CI jobs remain authoritative.

- Prose or link only: no build. Check that every path and command quoted still
  exists and that each claim matches the code.
- Any Rust change: fmt, workspace Clippy with `-D warnings`, library tests, doc
  tests and the doc build. Add the per-feature checks of every feature you
  touched (`distributed`, the `pencil-fft` all-features set, the `pencil-io` HDF5
  variants).
- Anything that changes a collective, a transport or a distributed plan: the
  matching MPI matrices at their 1/4 and 1/4/6 ranks.
- Anything MSRV-sensitive, and every dependency change: the `1.96.0` checks.
- `pencil-fftw` or native `fftw` paths: the ignored native tests with both
  precisions, plus the host-only adapter tests that need no FFTW installed
  (`cargo test -p pencil-fftw --locked`).

Report the exact commands and their results, and state plainly what was not run
and why. A numerical claim needs the known-value or reconstruction check that
backs it. Changing a tolerance or an expected value to turn a red test green is a
decision for the user, not a fix, and a contract test is not weakened to accept
new behaviour.

## Conventions that are contracts

- Storage order is per API and documented: `pencil-array` uses logical
  `[extra..., spatial...]` with a row-major `[extra..., permuted spatial...]`
  buffer; `LocalC2cNdPlan` is column-major with axis 0 fastest. Do not add a
  conversion shim: convert at the boundary that already owns the order.
- Transform conventions: forward and backward are unnormalized, backward uses the
  positive sign, and `inverse` divides by the line length (the complex length for
  C2C, the original real length for R2C/C2R). Distributed plans keep the same
  conventions per axis.
- Errors are per-crate `thiserror` enums in the module that owns them. Ordinary
  validation errors return `Err`, preserve caller data, workspace and plan state,
  and are checked before payload communication; backend and resource panics are
  not converted into errors. MPI failures, arbitrary panics and process loss do
  not guarantee global recovery.
- Unsafe is confined: `pencil-fft` is `#![forbid(unsafe_code)]`, `pencil-fftw`
  allows it only in `mod ffi`, and the two MPI-facing crates set
  `unsafe_op_in_unsafe_fn` (`forbid` in `pencil-array`, `deny` in `pencil-io`).
- `pencil-array`, `pencil-fft` and `pencil-io` enable
  `#![warn(missing_docs)]`: a new public item gets documentation. New rustdoc
  examples must run and assert; `ignore` is not used here, and the single
  `no_run` (`crates/pencil-io/src/lib.rs`, the serial-HDF5 file example that
  needs a real HDF5 install) is the only exception. The serial-HDF5 job builds
  docs with `-D warnings`.
- Dependencies are pinned: `mpi = "=0.8.2"` in the workspace, a committed
  `Cargo.lock`, and `--locked` on the CI build and test commands — the
  `rsmpi-rt` job being the deliberate exception, because its first check
  regenerates that fork's lockfile in its own checkout. A dependency change
  updates the lockfile in the same change.
- Steady-state local FFT, view and reduction paths are tested to allocate
  nothing. A new path keeps that property or documents and tests the exception.

## Release and publication

`RELEASING.md` owns the procedure: package preparation, archive inspection, the
exact MSRV verification, the six-green-checks record per published SHA, and the
dependency-ordered publication (`pencil-array` and `pencil-fftw`, then
`pencil-fft`, then `pencil-io`). Do not duplicate it here and do not improvise a
shortcut. An actual `cargo publish` needs explicit user confirmation; dry-runs do
not. Registry releases are immutable, and the five `LICENSE`/`NOTICE.md` copies
stay byte-identical to the root ones.

## Review and pull requests

- One pull request per agreed deliverable. The body states the change, the issue
  it serves, and the exact local commands run with their results, including what
  was not run.
- Merging needs an independent review plus confirmation that all six required
  checks passed on the pull-request head; the ruleset enforces neither. Re-read
  the head and its checks before a consequential action, and do not rely on
  checks from an earlier head.
- Do not change the CI workflow contents, a job name, or the audit scripts as a
  side effect of an unrelated change.

## Docs and worklogs

`README.md` is the guide and must stay in step with the code: the feature
descriptions, the crate table, the prerequisites, the verification commands and
the CI job names. A change to the public surface, a feature, a command or a
prerequisite updates `README.md`, and the affected crate `README.md`, in the same
branch. `crates/pencil-fftw/src/lib.rs` includes its README, so there it is also
compiled documentation.

There is no `docs/` tree yet. Following the shared worklog policy, a nontrivial
multi-phase change, a non-obvious design choice or a performance experiment gets a
record at `docs/worklogs/<theme>.md` (the directory appears with the first
record), linked from its pull request; small fixes, and AI assistance alone, need
none. Keep a measurement with the command, environment and versions that produced
it instead of restating the number without them.
