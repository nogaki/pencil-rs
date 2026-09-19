# Milestone 9: local Julia/FFTW cross-validation

- Date: 2026-09-19
- Base: `e6f4862c07b8e5c95cba16171fe471a4e4f4d286`
- Status: implemented and independently reviewed; benchmarks remain out of scope.

1. Keep the Julia 1.12.6/FFTW.jl 1.10.0 project locked and instantiate only
   from a temporary copy with caller-controlled or private depot state.
2. Generate 16 temporary fixtures (8 base cases × 2 precisions) in one fixed,
   typed, counted format; record runtime provenance but derive Rust constants.
3. Build the distributed reference test once, run its ordinary self-check, and
   run the explicit ignored test through Cargo at 1, 4, and 6 ranks with both
   transpose methods. Cover all valid M=1/2 layouts: 26 case/layout
   combinations per method per rank.
4. Check raw-slice ownership/indexing independently from `get_local`, preserve
   OOP/IP source checks, and require a corrupted fixture to fail visibly.
5. Keep Julia optional for normal Rust builds, tests, docs, and CI. The tool
   README is the canonical setup, format, matrix, and command reference.

## Verification

- Exactly Rust 1.85.0 passed default and distributed workspace all-target checks
  with the locked Cargo dependencies in isolated Rustup/target directories.
  The global Rustup configuration and production/library sources were unchanged.
- Julia 1.12.6, FFTW.jl 1.10.0, provider `fftw`, reported native FFTW 3.3.11:
  16 fixtures / 26 layouts per method passed at 1/4/6 ranks under both the
  project stable toolchain and Rust 1.85.0. Both runs rejected corrupted data.
- Existing default tests (56 unit / 17 doc), distributed tests (60 unit / 22 doc),
  the new parser/offset/comparator self-check, and all 16 existing MPI runs passed.
  Total successful MPI launches including both reference matrices: 22.
- Formatting, default/distributed Clippy, docs, dependency boundaries, explicit
  missing-fixture failure, runner SIGTERM cleanup, and native timeout escalation
  were checked. Cargo.lock, existing tests, and CI were not changed.
- CI success is not a merge prerequisite this month by repository-owner approval;
  local verification and independent review are the acceptance evidence.
