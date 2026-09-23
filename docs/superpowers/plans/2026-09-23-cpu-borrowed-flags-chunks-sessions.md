# CPU parity: borrowed storage, FFTW flags, chunked MPI storage, persistent I/O

User requested candidates 1–4; Julia file-format compatibility (candidate 5) and GPU are excluded. Base main: `c3c11d0bb95034d5ce91ae8e49f8afe2e872de70`, tree `c2d5c4aa55680c04a7f32b49f74ce82ecd319509`.

## Contracts

Keep existing public signatures, closed enums, default formats, source/preflight atomicity and native cleanup/poison semantics. New collectives start with two five-u64 MIN/MAX Allreduces on the caller's original Cartesian communicator, before rank-local error returns or operation-specific communication. Array remains FFT/I/O-independent; default local FFT remains MPI/FFTW-runtime-free; unsafe remains in audited private FFI modules. No GPU, Julia-wire parser, system installs, global configuration, CI billing/protection/concurrency changes, or unrelated environment-file changes. September 2026 CI-success waiver remains; never claim failed/unstarted CI passed.

## Scoped deliverables

- [x] Public checked external-slice immutable/mutable views, sharing the existing constructor validation and lifetimes. Start with contiguous physical-order slices; do not invent a generic storage backend or arbitrary-stride framework. Validate caller storage length, including empty dimensions; no allocation/copy or ownership transfer. Demonstrate actual pointwise/collective/transpose/I/O reuse where those APIs accept views.
- [x] FFTW `WISDOM_ONLY` and `CONSERVE_MEMORY` controls with default false, public builders/getters, named native constants, actual native flags for every plan/family and exact distributed descriptors. Preserve legacy rigor/errors and mandatory alignment safety. Wisdom miss is a real planning error, with no silent fallback. Keep pinned-runtime/thread-init ordering, known-one reset and source preservation. Both configuration orders and all six families retain settings.
- [x] Explicit rank-contiguous MPI chunked write/read APIs, using a separate bounded/versioned format, not altering legacy v1/v2 defaults or confusing HDF5 chunks. Store each rank's physical-order little-endian block contiguously. Validate type, shapes, writer topology/decomposition/permutation, rank ownership/counts, offsets, exact coverage and final file size before payload reads. Same-writer-layout restriction is explicit; metadata inspection must report this storage mode honestly. Existing collective/independent payload mode and MPI hints remain usable. Ordinary errors do not modify read destinations.
- [x] Real persistent MPI and HDF5 file sessions, plus real hierarchical HDF5 groups. Native file/duplicate-communicator handles remain open across methods; never emulate persistence by reopening path APIs.

## Persistent I/O design decisions

MPI sessions manage the existing append-only named v2 container, not the new single-dataset chunk format. `create` starts an empty container; `open_read` supports committed-prefix payload reads; `open_append` requires a complete committed tail. Writes append unique names and retain commit ordering. Catalogs remain strict snapshots. Chunked file APIs are separate and explicit; no undocumented session/chunk format mixing.

Every session method first agrees on the original borrowed communicator, then checks a root-broadcast unique per-open identity, mode and closed/poisoned state before entering the file's duplicated context. Opening the same path twice must still produce distinct identities. Handle root counter exhaustion, keep the original communicator borrowed for the session lifetime, and prohibit overlapping operations on it. Rank-local independent counters are not valid collective identities.

Use explicit collective `close(&mut self)` so a preflight mismatch leaves the session retryable, rather than consuming it and triggering a destructor abort. Unclosed drop is fail-stop, not an uncoordinated collective close; after MPI finalization do not call MPI routines other than the permitted finalized query before process abort. Document explicit close before MPI finalization. A session poisoned after uncertain mutation remains collectively closeable. Read-only sessions reject writes before mutation.

HDF5 sessions use a distinct known Rust namespace with real group/dataset path components, not reinterpreted encoded legacy named keys. Bounded normalized relative paths reject NUL, empty components, `.` and `..`; parents must already exist and `create_group` creates only the final component. No recursive mkdir, overwrite, resize, or Julia compatibility. Check hard-link and object kind before opening every component. Catalog traversal rejects hard-link cycles/aliases using native object identity and bounds depth, count, component/path length and aggregate metadata. Apply existing filter/capability rules and options.

Separate per-operation native resource cleanup from persistent file/communicator cleanup. Session reads publish after operation-resource cleanup/agreement; a later file close does not roll back prior successful operations. Existing path APIs retain their stronger close-before-destination-copy contract; do not implement them atop a session read that commits early. Shared read cores may return staged values.

## Implemented API and review corrections

External views expose `from_slice` / `from_slice_mut`. FFTW options expose
`with_wisdom_only`, `wisdom_only`, `with_conserve_memory`, and `conserve_memory`;
backend descriptors are eight words with schema 5. Required flags reach every
native plan. NaN-filled independent collection outputs and five no-op mutants
check that new collection tests cannot pass by retaining old expected values.

Rank files use separate `write_mpi_chunked`, `read_mpi_chunked`, and
`read_mpi_chunked_catalog` APIs with mandatory existing MPI options. The catalog
returns one `DatasetInfo`; the API itself identifies rank-contiguous storage.
Combined spatial-plus-extra rank is checked at write and parse boundaries,
not merely each rank separately. The format does not repartition.

`MpiFileSession` uses v2 named containers with explicit `write_named`,
`read_named`, `catalog`, `flush`, and `close`; every metadata scan first restores
a zero-displacement one-byte MPI view. Invalid paths reject before duplication
or opening. A failed explicit flush remains retryable, while uncertain writes
still poison. `Hdf5FileSession` exposes real groups and datasets under
`/pencil_io_tree_v1`; transfer properties are prepared/agreed before dataset
creation and poisoning. Its native malicious-file, reopen-persistence,
cross-wired same-path handles and fail-stop tests cover the public surface.

Integration unifies duplicate finalized/communicator comparison FFI helpers
using fallible results and conservative error handling. Chunk private tests
reuse the crate's existing single MPI initialization. A combined external-buffer
I/O test exercises old MPI/HDF paths, chunk storage, persistent MPI and filtered
HDF hierarchy reads without transferring buffer ownership. Old path APIs retain
close-before-copy atomicity; session reads have the documented per-operation
commit boundary. Independent Sol reviews approved each feature, corrections
and the combined source/CI/documentation changes.

## Parent verification results (2026-09-23)

Stable Rust 1.98.1 and exact Rust 1.85.0 passed default/all-feature/all-target
checks, test compilation, unit/rustdoc and native FFTW suites. Stable fmt,
strict Clippy, warning-denying rustdoc and dependency-boundary checks passed.
The final parent MPI matrices passed 110 positive launches per compiler at the
supported 1/4/6 ranks, plus expected asymmetric panic-abort tests. Legacy array
suites restricted to 1/4 retain that restriction. The first concurrent MSRV run
hit a PMIx bootstrap failure before MPI initialization; a full sequential rerun
passed without source changes, and the initial failure log was retained.

Six complete Julia reference runs passed: RustFFT and native FFTW in both
configuration orders on both compilers, with 87 fixtures / 312 configurations
per run at ranks 1/4/6 and all 30 intentional corruptions rejected. Native
reference plans request two threads; Julia's independent oracle remains at
one. Together with the main matrices, this is 256 positive MPI launches and
180 reference corruption rejections, excluding expected fail-stop children.
Native flag-specific tests additionally cover wisdom miss/hit and all kinds,
partial C2C cleanup/reset, exact option mismatch and retry. No memory-saving,
thread-utilization or speedup guarantee is inferred.

Session lifecycle/failure tests exercise actual native handles and operation
cleanup. The HDF native-close child forces invalid-ID `H5Fclose` failure after
real cleanup; it does not claim to emulate a storage failure closing a valid
file. Older MSRV Clippy's existing grid lifetime warning is not claimed fixed;
strict Clippy evidence here is normal stable.

Evidence: `/tmp/pencil-cpu5-parent-{static,msrv}.log`,
`/tmp/pencil-cpu5-final-static.log`,
`/tmp/pencil-cpu5-parent-mpi-{stable,msrv}-summary.log`, and
`/tmp/pencil-cpu5-reference-{stable,msrv}-{rustfft,fftw-native-first,fftw-directions-first}.log`.
Source worktrees and compiler targets are isolated. Local validation is not a
claim of CI success; the existing waiver and excluded Julia/GPU scope remain.

## Verification and ownership

Astra implements isolated array, FFTW, and combined I/O worktrees; Sol independently reviews design and source; parent integrates, verifies and publishes. Separate stable/MSRV targets per source worktree, normal Rust 1.98.1 and exact 1.85.0. Use existing native Open MPI 4.1.7, parallel HDF5 1.10.7, FFTW 3.3.8 (headers 3.3.10), Julia oracle FFTW 3.3.11. No dependency installation.

Test new views with external buffers, non-Clone elements, wrong lengths, zero extents and MPI 1/4/6. Test native flags without accepting no-op implementations: empty wisdom miss, seeded wisdom-only hit after dropping plans, both precisions/all transforms, flag preservation, partial native cleanup and MPI mismatch/retry. Test physical chunk layout bytes, uneven/empty ranks, all scalar kinds, hostile/truncated metadata, exact ownership validation and staged read failures. Test multiple operations with one native file open, session cross-wiring (including identical paths), read-only/mode/name mismatches, explicit-close retry, poison/cleanup failures, group cycles/aliases/soft/external links, and strict catalogs without payload reads. Existing full MPI/native/Julia matrices and legacy enum checks remain intact.
