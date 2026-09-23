# Pencil Arrays for Rust

A row-major, MPI-distributed multidimensional array foundation inspired by
PencilArrays.jl, with a separately layered FFT implementation.

The workspace contains the `pencil-array` core crate, the local FFT
`pencil-fft` crate, the separate `pencil-io` crate, and the optional native CPU
`pencil-fftw` adapter. `pencil-array` is
intentionally independent of RustFFT, RealFFT, FFTW, and any FFT-specific API.
The local `pencil-fft` path accepts flat slices, uses RustFFT/RealFFT, and is
independent of MPI and `pencil-array`. Local C2C provides unnormalized
forward and positive-sign `backward` transforms plus normalized `inverse`. An
opt-in `pencil-fft/distributed` feature adds out-of-place, input-preserving and
single-buffer in-place distributed C2C FFTs, including raw positive-sign
`backward`, and out-of-place distributed R2C/C2R forward, normalized inverse,
and raw backward over checked Alltoallv or point-to-point transitions. Its
R2C/C2R API also provides a state-checked single-allocation in-place buffer.
Local out-of-place and packed single-allocation in-place R2C/C2R are also
available. `LocalR2rPlan` provides all eight
FFTW-compatible DCT/DST-I-IV kinds for real and complex `f32`/`f64`, with
out-of-place and in-place execution. Its `forward`/`backward` operations use
raw unnormalized FFTW conventions, while `inverse` is normalized by the
logical transform factor. Each line uses at most `8n` complex embedding values
plus queried native scratch, using the existing RustFFT backend without MPI.
`LocalDhtPlan` provides the local self-paired discrete Hartley transform for
the same four scalar types, with the same caller-owned line/scratch and
out-of-place/in-place conventions.
Distributed R2R uses the same eight kinds per logical axis, with `None` identity
stages, for real and complex `f32`/`f64` over the existing checked transports.
`DhtPlan` provides selected-axis self-paired discrete Hartley transforms without
exposing the legacy R2R axis-kind constructors.

`Pencil` describes spatial distribution. `PencilArray` owns one layout and
one local buffer. `ManyPencilArray` owns a buffer large enough for several
registered layouts and exposes only its active layout. Array shapes use
logical order `[extra..., spatial...]`; their row-major buffers use memory
order `[extra..., permuted spatial...]`. `get_global` is a noncollective,
local-only lookup: remotely owned or out-of-range coordinates return `None`.
`Pencil::local_grid` borrows one complete global coordinate slice per spatial
axis and iterates this rank's spatial coordinates in physical memory order;
extra batches repeat that spatial grid. `LocalTransposePlan` provides
process-local memory-axis permutations. `AllToAllvTransposePlan` and
`PointToPointTransposePlan` provide checked distributed redistribution through
out-of-place views and shared-storage in-place execution. The optional
`pencil-fft/distributed` feature composes these checked transitions into
out-of-place and in-place distributed C2C forward/inverse/backward transforms and
out-of-place plus single-allocation in-place R2C/C2R; this does not change
the MPI-free local FFT default.
`C2cPlan` and `R2cPlan` provide matching selection-aware shape, pencil, and
array constructors through `AxisSelection<N>`, plus the existing
`from_*_with_method` forms. `AxisSelection::all()` is the default; its
`from_indices` input is validated for duplicates/out-of-bounds values and its
order is ignored. The legacy constructors retain the Alltoallv default; R2C
output reduces the selected reduction axis to `n/2+1`.

Alltoallv and point-to-point construction and execution are collective: every
rank must use the same source communicator context, API, order, `T`, and
correct `Equivalence`. Source and destination pencils must share the same
`MpiTopology` object and global shape and differ in exactly one ordered
decomposition position. Descriptor checks catch common mismatches but do not
replace that communicator, collective-order, type, or `Equivalence` contract.

`TransposeError`, `TransposeWorkspace<T>`, and
`TransposeWorkspaceRequirements` are the canonical shared API. The older
`AllToAllvTransposeError`, `AllToAllvTransposeWorkspace<T>`, and
`AllToAllvTransposeWorkspaceRequirements` names are compatibility aliases for
the same types. `workspace_requirements` and
`TransposeWorkspace::from_vecs` are noncollective and do not call MPI;
execution checks initialized `len` (not capacity), does not resize or
reallocate workspace vectors, and checks count/length/displacement/offset
limits before payload communication. Ordinary out-of-place preflight errors
preserve the source, destination, and workspace. Ordinary in-place preflight
errors preserve the array state, active data, and workspace for both
Alltoallv and point-to-point. Point-to-point uses a fixed internal tag on the
topology-owned changed-axis context, posts all receives before sends, waits for
every request, and must not overlap unfinished transposes on that context. MPI
failures, arbitrary panics, and process loss do not guarantee global recovery;
an unfinished request scope may abort.

## Borrowing external CPU buffers

`PencilArrayView::from_slice(&pencil, &extra_shape, &buffer)` and
`PencilArrayViewMut::from_slice_mut(&pencil, &extra_shape, &mut buffer)` attach
checked pencil metadata to caller-owned storage without allocation, copying or
ownership transfer. The slice must contain exactly the full local buffer in
physical row-major `[extra..., permuted spatial...]` order. Metadata and storage
remain borrowed for the view's lifetime; mutable views retain Rust's exclusive
borrow rules. Zero-length layouts are valid, and elements need not implement
`Clone`. Existing view-based pointwise, reduction, transpose and I/O operations
accept these views. This does not introduce arbitrary-stride views or an
`ndarray`/GPU storage framework.

## Global array collectives

`pencil-array` provides collective `global_sum`, `global_min`, `global_max`,
`l2_norm`, `any`, `all`, `sum_by`, and `norm_by` operations, plus root
`gather`. Numeric reductions use the sealed `i8`-through-`u64`, `f32`/`f64`,
and complex scalar set; integer sums are checked and floating-point NaN and
infinity results follow the documented IEEE policy. Methods are available on
`PencilArray` and `PencilArrayView`; free
functions take a validated view. `ManyPencilArray` intentionally has no direct
collective methods: coordinate `active_view()` validity across ranks first,
then call the corresponding view method or free function. Gather returns
logical `[extra..., spatial...]` row-major order on the root and `None`
elsewhere; all ranks validate descriptors, counts, and root allocation before
payload communication.

## Checked pointwise operations and multi-input reductions

`pointwise2` and `pointwise2_views` write a caller-provided output; the
`pointwise2_in_place` and `pointwise2_in_place_views` forms update the left input
without cloning it. Inputs must have the same spatial pencil layout and extra
rank. An input extra extent may be one or the output extent, so extra axes can
broadcast without expanding a temporary. Scalars can be captured by the closure.
Traversal follows physical spatial storage order. These operations are local,
not MPI collectives: they never redistribute a spatial singleton. Validation
errors preserve the output and invoke no callback; a callback panic can leave
partial output.

`map_reduce2` adds two-input mapped reduction with an explicit neutral value and
an associative reducer. It folds each rank locally, gathers one scalar partial
per rank, then folds in rank order: the deliberate O(P) storage/communication
cost avoids Rust callbacks inside MPI. Callback semantics, associativity and
neutrality are caller obligations; callbacks must not panic or call MPI.
`zip_sum_by` and `zip_norm_by` retain the existing checked-integer/IEEE and scaled
norm policies. `min_by` and `max_by` reduce mapped values and return `None` for
a globally empty input. New collective descriptors agree both layouts and exact
neutral bits before invoking callbacks or exchanging typed data.

### Three or more inputs

`pointwise3` accepts three differently typed inputs; `pointwise_many` accepts an
arbitrary nonempty slice of same-typed inputs. Both have borrowed-view and
in-place counterparts, using checked singleton-extra broadcasting and reusable
O(input count × extra rank) mapping/reference storage, not per-point allocations
or full input clones. `MultiInputError` identifies invalid inputs without changing
the original `PointwiseError` variants.

`map_reduce3` and `map_reduce_many` add custom multi-input folds. The many-input
collective accepts a Cartesian communicator explicitly so even an empty or
rank-dependent input list can be rejected through the common header. Every input,
output scalar, neutral value and broadcast shape is checked before callbacks.
`sum_many_by`, `norm_many_by`, `min_many_by`, and `max_many_by` retain the checked
integer, scaled norm and IEEE policies. Reducers must be deterministic,
associative and neutral-compatible; their results must not depend on mutable
invocation count/order. Callbacks must not panic or call MPI. Empty many-input
lists are rejected; zero-element arrays remain valid.

## Collections of separate arrays

All six distributed plan families provide `forward_many`, `inverse_many`, and
`backward_many`, plus `*_many_in_place` for their existing in-place array types.
They take standard slices of separate arrays and reuse one plan-bound workspace
sequentially. This differs from one array with `ExtraShape` batch axes. Use
standard Rust iterator/`Vec` allocation rather than a separate collection owner.

A collection call collectively agrees its operation, family, count and complete
plan/member metadata, then validates every member before executing the first.
Empty collections are rejected collectively. Ordinary preflight failures leave
all arrays and the workspace unchanged; a later execution error reports the
member index through `CollectionError<E>` and does not roll back earlier members.
An arbitrary panic during member execution is fail-stop via MPI abort, not a
promise of cross-rank panic recovery or another potentially mismatched reduction.

`pencil-io` also provides `write_mpi_collection` / `read_mpi_collection` and their
HDF5 counterparts. These accept the Cartesian communicator explicitly and view
slices, and store one `[component, extra..., spatial...]` payload using the
existing single-dataset format. Every read destination is staged until native
cleanup and agreement succeed. Option-bearing collection forms preserve the
same staging guarantees. `write_mpi_named_collection`, `append_mpi_named_collection`,
`read_mpi_named_collection` and their HDF5 counterparts store one combined named
dataset and accept explicit options. Names, options and all members agree before
payload staging or mutation. These APIs do not pretend several independent
dataset writes are one collection.

## Distributed R2C/C2R FFT

`R2cPlan<R, N, M>` accepts a canonical real input pencil and a non-empty
`AxisSelection<N>`. The largest selected Rust axis `r` is the real-to-complex
axis and its extent `n` is reduced to `n/2+1`; selected axes below `r` use
complex FFTs and every unselected axis is an identity stage. The final output
uses decomposition `[1..=M]` and reversed spatial memory order by default;
`DistributedLayout::permute_dims = false` keeps identity memory order and uses
strided line kernels. The route nevertheless always contains all `N` stages
and `N-1` transitions, so
unselected axes remain real-prefix or complex-suffix transposes rather than
being skipped. `allocate_workspace` is noncollective; coordinate any local
allocation failure before the next collective. Non-last real axes add an
optional real intermediate and real transpose workspace. `forward`, normalized `inverse`, and unnormalized positive-sign `backward`
are exposed in both out-of-place and state-checked single-allocation in-place
forms. Empty R2C selections are collectively rejected before
planning. The constructors and operations require the same communicator
context, API/order, scalar type, selection, method, and layouts on every rank.
Legacy constructors use Alltoallv; point-to-point uses the fixed `0x5054` tag
and must not overlap unfinished transposes on its context. The in-place array
exposes only the state-matching real or complex `PencilArrayView`; its one
allocation is recast safely between separate real-prefix and reduced-complex
suffix registries. Forward processes packed boundary rows back-to-front and
C2R processes them front-to-back. Initial preflight errors preserve the array;
after start, an error, invalid spectrum, or panic leaves it poisoned. In-place
operation words are 25/26/27; 18..24 remain reserved for distributed R2R.

The inverse performs the transverse complex inverse stages first, then
validates each extra batch and constrained DC/Nyquist plane. It accepts a
plane when either `max(abs(imaginary)) <= 128 * min_subnormal_R * D` or its
imaginary L2 norm is at most `128 * epsilon_R * D` times its real L2 norm,
where `D = 1 + sum(ceil(log2(n_a)))` over selected axes below the real axis.
This is an explicit normwise-relative/componentwise-absolute acceptance
policy, not a formal RustFFT error bound. All constrained endpoint values
must be finite; odd lengths constrain DC only (for `n=1` the final bin is DC),
even lengths also constrain Nyquist, and interior bins have no blanket finite
policy. Nonfinite or materially non-real constrained planes return
`R2cError::InvalidSpectrum` before any real destination write; the workspace
may already have changed on that post-start error path, while source and
destination remain unchanged. `backward` uses the same relative endpoint
criterion, but its componentwise absolute threshold is the inverse threshold
times the product of the selected transverse extents, excluding the real axis
and extra dimensions. That finite-positive factor is collectively validated at
plan construction. A forward/backward pair scales by the product of all
selected spatial extents; identity axes do not contribute.

## Distributed C2C FFT

A C2C plan also accepts `AxisSelection<N>`. It always follows the canonical
full route from axis `N-1` through `0`, with one stage and transition per
axis. A selected stage performs its local FFT; an unselected stage is an
identity with no native plan or scaling. Thus an empty selection is a valid
identity transform whose selected-layout transposes still run. The output
pencil uses decomposition `[1..=M]` and the reversed permutation by default;
`DistributedLayout::permute_dims = false` keeps identity memory order and uses
strided line kernels. Inverse/raw-backward normalization includes only
selected axes. `AxisSelection::all()` preserves the legacy transform exactly.

Enable the feature in this workspace with
`cargo check -p pencil-fft --features distributed --locked`. The public
`C2cPlan` supports `N >= 2` and `1 <= M < N`, canonical identity input
pencils, exact extra shapes, reusable plan-bound
`C2cOutOfPlaceWorkspace`, and input-preserving forward, normalized inverse,
and raw positive-sign backward execution. Forward consumes the canonical input
layout and produces the selected output layout; both inverse and raw backward
consume that output layout and produce canonical input. Raw backward does not
normalize, so a forward/backward pair scales by the product of selected
spatial extents; identity axes and extra dimensions do not contribute. Construction and execution are collective
on the topology's Cartesian communicator; every rank must call matching
operations in order and select the same `TransposeMethod`. The selected method is appended to the
minimal checked C2C descriptor (after global shape, process grid, extra shape,
axis mask, value kind, precision width, and method), so a rank-local method
mismatch returns
`FftError::CollectiveDescriptorMismatch` before native FFT, output, workspace,
or in-place state changes. Each rank checks its actual source and destination
against the plan before a full-Cartesian preflight agreement, and no data is
changed before that initial agreement. Once execution starts, a resource or
later transition metadata failure may mutate the workspace; no general
allocation-free rollback is promised. Point-to-point uses the existing
changed-axis topology context, fixed internal `0x5054` tag, receive-before-send,
wait-all, and MPI failure contract; it reserves request metadata. Do not
overlap unfinished point-to-point transposes on that context. The
`distributed` API is not enabled by default.

## Distributed R2R DCT/DST

`R2rPlan<T, N, M>` accepts `[Option<R2rKind>; N]` in logical-axis order;
`None` keeps that canonical route stage as an identity. It preserves the
original global shape, uses the same `N` stages and `N - 1` checked
transitions, and produces the reversed output layout by default; an explicit
`DistributedLayout` can retain identity memory order with strided line
kernels. `forward`, normalized paired `inverse`, and raw paired `backward` are input-preserving and support
real or complex `f32`/`f64`. The in-place API uses a plan-bound
`ManyPencilArray` and the shared `Input -> Poisoned -> Output` state contract;
public `R2rState` reuses that completion-state enum. R2R descriptor agreement
includes scalar value kind, underlying precision, all
per-axis kind codes, and transport before local native planning or data writes.

Plan construction and transform calls are collective. In-place array/workspace
allocation and views are noncollective; callers must coordinate an allocation
failure before the next collective call. `R2rPlan::allocate_in_place` returns
an opaque `R2rInPlaceArray` and `allocate_in_place_workspace` returns its
plan-bound scratch. The array exposes only `state`, `view`, and `view_mut`:
`Input -> Poisoned -> Output` for forward and `Output -> Poisoned -> Input`
for normalized inverse or raw backward. Initial collective preflight errors
preserve state, data, and workspace; once execution begins, the array is
poisoned before its first write, and any later failure or panic leaves it
`Poisoned`, so callers reallocate it. A mutable output view may contain
arbitrary spectral data for either reverse operation.

## Distributed mixed-axis plans

`MixedC2cPlan<R, N, M>` and `MixedR2cPlan<R, N, M>` apply a typed
`[AxisTransform; N]` in canonical descending route order. `None` is an
identity, `Fft` is a complex FFT, `R2r(AxisR2rKind::Fftw(...))` selects a DCT
or DST, and `R2r(AxisR2rKind::Dht)` selects a Hartley transform. `MixedC2cPlan`
rejects `Rfft`; `MixedR2cPlan` requires exactly one `Rfft`, permits only
identity/R2R/DHT stages on the real prefix and identity/FFT/R2R stages on the
complex suffix, and reduces only the boundary axis. It is not an arbitrary
multi-RFFT/BRFFT graph API. Both plans provide
input-preserving out-of-place forward/inverse/raw-backward operations,
plan-bound workspaces, checked single-allocation in-place arrays, both
Alltoallv and point-to-point transitions, and the same normalized/raw pairing
as the homogeneous plans. `MixedR2cPlan` validates finite DC/Nyquist planes
collectively before C2R output and preserves the poisoned in-place state after
post-start errors or panics; a representation handoff panic may discard the
backing owner. `DistributedLayout::permute_dims = false` selects the strided
per-line kernels. The mixed descriptors use distinct collective
operation words, scalar widths, reduced shape, axis transform codes, and
transport policy, so mixed and legacy calls cannot accidentally agree.

The existing `tools/fftw-reference` Julia/FFTW checker remains the independent
numerical reference for the component FFT, R2R, DHT, and R2C kernels. Mixed
plans are validated by composing those same one-axis references in route order.
The reusable oracle is `tools/fftw-reference/mixed_reference.jl`; focused MPI
coverage is in `crates/pencil-fft/tests/distributed_mixed.rs`.

## Per-axis Fourier directions, timing, and send overlap

`FourierDirections<N>` selects `FourierDirection::Forward` (negative exponent)
or `Backward` (positive exponent) per complex FFT axis. `C2cPlan`,
`MixedC2cPlan`, and `MixedR2cPlan` expose `with_fft_directions`; it collectively
creates a fresh plan core, so allocate workspaces/in-place arrays from the
returned plan. Existing constructors keep their original signs. A configured
positive-sign forward pairs with a negative-sign reverse; only `inverse`
normalizes. Identity, R2R and RFFT axes must retain the default direction.
The existing closed `AxisTransform` and error enums remain unchanged.

All distributed CPU plan families provide `forward_with_timing`,
`inverse_with_timing`, `backward_with_timing`, and corresponding in-place
methods. `TransformTiming<N>` stores local-rank, fixed-size route-stage records:
local transform time/call count, transition time/call count, and pack/unpack and
transport wait measurements. Reports use `std::time::Instant`; there is no
implicit global timing reduction. Alltoallv records its collective wait rather
than inventing separate receive/send waits.

For point-to-point plans, the out-of-place `*_with_overlap` methods wait for
receives and unpack first, execute the next local FFT while sends can still be
outstanding, and then wait for every send. They return the additive
`FftOverlapError<E>` rather than changing legacy error enums. Callback errors
and panic status are agreed after requests are drained; this does not promise
recovery from MPI failure or process loss. In-place overlap is not exposed;
in-place profiling remains supported. Overlap on Alltoallv is rejected, and
old, profiled and overlap calls have distinct collective operation words.

## Optional native CPU FFTW backend

Enable `pencil-fft/fftw` to use explicit `new_fftw` constructors for local C2C,
R2C/C2R, R2R and DHT plans. All distributed families provide `with_fftw(options)`
and native shape constructors. Existing constructors **always select RustFFT /
RealFFT**, even when the feature is enabled; default builds need no native FFTW.
A rebuilt distributed plan has a fresh identity, so allocate its own workspaces
and in-place arrays. Fourier-direction rebuilding preserves the selected backend.

`PlanOptions` selects `PlanningRigor::{Estimate, Measure, Patient, Exhaustive}`
and an optional positive `Duration` planning budget. It is FFTW's approximate
planning limit, not a real-time deadline. Backend, precision, rigor and exact
requested duration are agreed before native planning. `BackendInitError<E>`
reports initialization failures without extending legacy error enums; a missing
native library never silently falls back to RustFFT. Plan `backend_kind()` and
option accessors expose the actual selection.

The MPI-free adapter dynamically loads Linux `libfftw3.so.3` and
`libfftw3f.so.3`, with initialized private planning buffers, separate native
in-place/out-of-place plans, and unaligned new-array execution. R2R/DHT retain
their current embedding algorithms with FFTW complex kernels; this is not a
claim of native specialized DCT/DST performance. Planning/destruction are locked
per precision, and the adapter resets its planning time limit to NO_TIMELIMIT.
Uncoordinated foreign FFTW planner/state changes are outside its guarantee.

`PlanOptions::with_threads(n)` requests a positive per-plan CPU thread count
(default one); `requested_threads()` reports that request, not observed thread
utilization or a speedup. Distributed descriptors include the exact count.
Before the first coordinated stateful FFTW call, the adapter probes and, when
available, initializes the matching pthread runtime. Base/thread libraries remain
pinned for process lifetime. A pre-initialization load/symbol failure leaves
serial planning and wisdom available but rejects requests above one; it is not
silently retried after serial use. Attempted native initialization failure blocks
further stateful calls. Successful planning resets the native count to known one,
not an invented prior foreign setting.

`PlanOptions::with_wisdom_only(true)` requests native `FFTW_WISDOM_ONLY`:
planning fails with `NullPlan` if compatible wisdom is unavailable, rather than
silently planning afresh. Train/import wisdom with the required precision,
rigor, thread count and plan forms first; C2C needs both in-place and
out-of-place native plans. `with_conserve_memory(true)` requests native
`FFTW_CONSERVE_MEMORY`, not a measured memory-reduction guarantee. Both flags
default false; `wisdom_only()` and `conserve_memory()` report the settings.
All distributed families agree both flags before native planning and preserve
them through either direction/backend configuration order. Mandatory alignment
safety and existing input-preservation contracts are unchanged; arbitrary unsafe
FFTW flags are not exposed.

`export_wisdom::<R>()`, `import_wisdom::<R>()`, and `forget_wisdom::<R>()` manage
actual precision-specific native wisdom. Use ordinary file I/O to persist the
returned string; these operations are process-local, not implicit MPI broadcasts.
Forgetting wisdom does not invalidate existing plans. The pinned runtime keeps
imported wisdom alive even with no live plans. Invalid imports have no rollback
guarantee, and wisdom portability across FFTW versions, machines, flags and
thread settings is determined by FFTW. Coordinate local errors before entering
subsequent MPI operations.

**Licensing:** this project's wrapper source remains MIT; FFTW is GPL-2.0-or-later
or separately commercially licensed. No FFTW source or binary is vendored.
Dynamic loading is **not** a licensing exemption; FFTW-enabled distributions
must address the applicable terms. See [`NOTICE.md`](NOTICE.md) and
[`crates/pencil-fftw/README.md`](crates/pencil-fftw/README.md).

Native tests are explicitly opt-in and fail if the runtime is unavailable:

```bash
cargo test -p pencil-fftw -- --ignored
cargo test -p pencil-fft --features fftw --test fftw_local -- --ignored
PENCIL_FFT_BACKEND=fftw tools/fftw-reference/check.sh
PENCIL_FFT_BACKEND=fftw PENCIL_FFT_DIRECTION_ORDER=directions-first tools/fftw-reference/check.sh
PENCIL_FFT_BACKEND=fftw PENCIL_FFT_THREADS=2 tools/fftw-reference/check.sh
```

## Local Julia/FFTW reference validation

The opt-in Milestone 9 checker generates temporary Julia 1.12.6/FFTW.jl
1.10.0 references and validates the distributed C2C, R2C/C2R, R2R, and DHT
forward/inverse/raw backward APIs at 1, 4, and 6 MPI ranks with both
transpose methods and both memory-layout policies. It covers exactly 82
format-7 fixtures and 136 base case/layout configurations per memory-layout
policy (272 with both policies), including mixed-axis C2C/R2C and real/complex
R2R and DHT `f32`/`f64`.
A separate direction-format generator adds 5 signed-axis fixtures and 40
configurations per MPI size, retaining all 82 legacy fixtures and 272 legacy
configurations. Direction-specific corruption checks also verify that ignoring
or altering Fourier signs is detected.
The external comparison is explicitly opt-in; normal Rust tests need no Julia.
See [`tools/fftw-reference/README.md`](tools/fftw-reference/README.md) and run
`tools/fftw-reference/check.sh` only when Julia, FFTW.jl, and MPI are locally
available.

## Native collective I/O

`pencil-io` is a separate crate that adds collective,
decomposition-independent persistence for `PencilArray` views without changing
`pencil-array` or `pencil-fft`. Its native MPI-IO backend writes one versioned
header and canonical little-endian row-major payload per file, and uses MPI
byte-subarray file views; readers stage and validate the complete payload before
mutating the destination. Writes use exclusive file creation and a flushed commit
marker, so incomplete and committed states are reported separately. The format
records type, width, logical shapes, writer process grid, and writer permutation;
writer layout metadata is provenance, not a read-layout requirement. Native
cleanup failures that can leave an unrecoverable handle are fail-stop paths via
MPI abort, not recoverable Rust errors.

Enable the optional native parallel HDF5 backend with
`pencil-io/parallel-hdf5`. It stores one versioned dataset at
`/pencil_io_v1/data` in the same logical order and strict scalar or `{r,i}`
compound little-endian type, with typed metadata attributes and the same
incomplete/committed protocol. This is a native self-describing representation,
not Julia PencilIO wire compatibility. The feature requires a parallel HDF5
installation discoverable by `pkg-config` or `HDF5_DIR`; it is intentionally not
enabled by default. Both APIs are
collective over the view's Cartesian communicator, and every rank must enter
calls in the same order without overlapping another operation on that
communicator. The HDF5 path explicitly creates a dataset-transfer property
list with `H5Pset_dxpl_mpio(..., H5FD_MPIO_COLLECTIVE)`; verify that the HDF5
and MPI libraries resolved by the build and runtime are the same ABI. The
lockfile's shared `mpi-sys` dependency is not ABI evidence: the pair is
validated only by inspecting the test executable's `ldd` output and recording
runtime MPI and HDF5 versions from that same environment.

### Named datasets and append

`write_mpi_named`, `append_mpi_named`, and `read_mpi_named` provide an append-only
multi-dataset container separate from the original single-dataset format.
Append adds a new name; it never overwrites an earlier dataset. Each record has
its own checked framing and commit marker. Earlier committed records remain
readable when a later tail is malformed or incomplete, while further append is
refused until that invalid tail is handled outside this API. There is no
artificial global 1 GiB limit; native per-rank count, dimension and file-offset
limits still apply.

The optional HDF5 counterparts are `write_hdf5_named`, `append_hdf5_named`, and
`read_hdf5_named`. They use `/pencil_io_named_v1` with injectively encoded UTF-8
keys, original-name attributes, and per-dataset metadata/commit markers. Append
opens the file read/write without truncation. Names are keys, not filesystem
paths; duplicate names are rejected before mutation. `NamedIoError` preserves
underlying `IoError` values, including commit uncertainty. Both backends retain
staged reads that update the destination only after successful cleanup. HDF5
is not a crash-atomic journal, and neither backend promises process-loss
recovery. The original exclusive-create v1 APIs and file representations remain
unchanged; no Julia PencilIO wire compatibility is claimed.

### Explicit raw input and I/O controls

`read_mpi_raw(path, view_mut, RawReadOptions)` reads external headerless binary
input. The view supplies type and global shape; options select a byte offset
and native/little/big byte order. File elements follow canonical logical
`[extra..., spatial...]` row-major order; each complex component is decoded
separately. Prefixes and trailing records are allowed, but the complete requested
range must fit the file and native count/offset limits. No type, shape, layout or
Julia-wire detection is implied. Existing readers never silently fall back to
this less self-describing format, and failed raw reads preserve the destination.

Additive `*_with_options` APIs for ordinary and named I/O accept `MpiIoOptions`,
`Hdf5ReadOptions`, or `Hdf5WriteOptions`. Builders configure `MpiIoMode`, validated
MPI hints, and (for HDF5 writes) native chunk dimensions in canonical logical
order. Hints are forwarded to MPI open/view and HDF5 MPIO properties; the native
implementation may ignore unsupported hints. Chunking uses a real dataset
creation property list, including support for empty extents; it does not expose
dataset resizing or change the existing metadata schema.

HDF5 writes also accept `shuffle(true)` and `deflate(level)` (0 through 9).
Filters require explicit chunks and collective payload writes; level zero still
requests Deflate. Filter availability and encode/decode capabilities are checked
on all ranks before file creation/opening. Unsupported requests fail explicitly,
without silently disabling compression. Empty local ranks still participate in
collective writes for a nonempty global dataset. These are native HDF5 filters,
not a new compression dependency or a guarantee that every dataset shrinks.

`Independent` selects actual independent **payload** transfers, not a one-rank
API: all ranks still enter metadata, validation, commit and cleanup in order.
Same-file modification across jobs or communicators requires external writer
serialization, and reads require stable file contents. No internal file lock or
concurrent-append guarantee is added. Existing entry points retain collective
payloads, null/default hints and their original failure guarantees.

### Dataset catalogs

`read_mpi_catalog`, `read_mpi_named_catalog`, and `read_hdf5_catalog` inspect the
supported formats collectively without reading their data payloads or requiring
an allocated destination array. Pass the Cartesian communicator explicitly.
`DatasetInfo` exposes optional names, `ScalarType`, stored spatial and extra
shapes, and available writer provenance. Extra dimensions are reported verbatim;
a collection component count is not inferred from an ordinary extra axis.

Catalogs are strict snapshots: malformed/incomplete metadata, a bad named-MPI
tail, or an incomplete HDF5 dataset rejects the whole inspection after cleanup.
Existing named reads may still recover a known committed MPI-prefix dataset.
Catalogs do not infer types from raw files or prove payload integrity. HDF5
inspection accepts only the supported hard-link group/dataset structures and does
not follow soft/external links. Native Unix file-path bytes remain supported;
dataset keys themselves remain bounded UTF-8. Catalog size/name/header bounds
are documented in the API.

### Rank-contiguous MPI files

`write_mpi_chunked(path, view, &options)` and
`read_mpi_chunked(path, view_mut, &options)` use a distinct versioned format,
leaving the existing MPI v1/v2 formats unchanged. Each Cartesian rank stores one
contiguous physical-order little-endian block. This is the MPI rank-block layout
option, not HDF5 dataset chunking. Collective/independent payload mode and MPI
hints use the existing `MpiIoOptions`.

Reads require the writer's exact process grid, ownership, decomposition and
permutation; repartitioning is not supported in this format. Type, shapes,
counts, offsets, payload coverage and exact file length are checked before
payload reads, and destination writes wait for native cleanup agreement.
`read_mpi_chunked_catalog(path, comm, &options)` returns one metadata-only
`DatasetInfo`; its separate API identifies this storage mode. Metadata is
bounded to 1 MiB and each rank's payload to `i32::MAX` bytes. Existing files are
never replaced. No Julia binary/JSON compatibility is implied.

### Persistent parallel file sessions

`MpiFileSession::create(comm, path, &options)`, `open_read` and `open_append`
retain a native MPI file and duplicated communicator across `write_named`,
`read_named`, `catalog` and `flush` calls. They use the existing named v2
container, interoperating with the old named path APIs—not the new chunked
format. Writes append unique committed names. Read-only sessions can read the
known committed prefix, while append and catalog operations require a complete
valid tail. MPI hints and payload mode are fixed when opening the session.

With `parallel-hdf5`, `Hdf5FileSession::create(path, comm, &options)` and matching
open methods retain the native HDF5 file. `create_group("flow")` followed by
`write("flow/velocity", view, &write_options)` creates real groups/datasets below
`/pencil_io_tree_v1`, not encoded flat names. `read` accepts `Hdf5ReadOptions`;
`catalog` inspects this hierarchy on the retained handle. Dataset filters use
the existing checked options; file hints are fixed at open. Parents must already
exist, and a group call creates only its final component. There is no overwrite,
resize, recursive group creation, or reinterpretation of legacy named keys.
Paths are normalized relative components; soft/external links, wrong object
kinds, hard-link aliases/cycles and incomplete metadata are rejected. Traversal
bounds include depth 32, 65,536 objects, 255-byte components, 4,096-byte paths and
16 MiB of accumulated tree names. Use the session catalog for this namespace;
the older path catalogs retain their original format scope.

**Call `session.close()` collectively before MPI finalization.** Close takes
`&mut self`, allowing a rejected preflight to be corrected and retried. Dropping
an open session is fail-stop, never an implicit rank-local collective close.
Every method agrees on the original borrowed communicator and a unique per-open
identity before using retained native contexts—even two handles for the same
path are distinct. Do not overlap calls on that communicator. Read-only and
ordinary preflight errors leave the session usable; uncertain mutations poison
it until close. Session reads publish after per-operation cleanup agreement;
a later close failure cannot undo earlier successful operations. Old path reads
retain their close-before-destination-copy guarantee. External serialization
of same-file writers is still the caller's responsibility.

Run the MPI-IO integration test at the required 1/4/6 rank matrix:

```bash
for n in 1 4 6; do
  timeout --foreground 120s mpiexec --oversubscribe -n "$n" \
    cargo test -p pencil-io --test mpi_io --locked -- --nocapture --test-threads=1
done
```

The integration cases cover every supported scalar family (`i8`/`u8`,
`i16`/`u16`, `i32`/`u32`, `i64`/`u64`, `f32`/`f64`, and complex `f32`/`f64`),
changed decompositions and permutations, committed-marker rejection, and
empty local ranks.

With parallel HDF5 configured, run the corresponding HDF5 matrix:

```bash
for n in 1 4 6; do
  timeout --foreground 180s mpiexec --oversubscribe -n "$n" \
    cargo test -p pencil-io --features parallel-hdf5 --test hdf5_io --locked \
      -- --nocapture --test-threads=1
done
```

The feature-enabled build checks used by CI are:

```bash
pkg-config --exists hdf5-openmpi
pkg-config --exists hdf5
cargo check -p pencil-io --all-features --all-targets --locked
cargo test -p pencil-io --no-default-features --test mpi_io --no-run --locked
cargo test -p pencil-io --features parallel-hdf5 --test hdf5_io --no-run --locked
cargo clippy -p pencil-io --all-features --all-targets --locked -- -D warnings
cargo doc -p pencil-io --all-features --no-deps --locked
timeout --foreground 120s mpiexec --oversubscribe -n 1 \
  cargo test -p pencil-io --features parallel-hdf5 --lib \
  post_cleanup_errors_preserve_destination_and_valid_commits --locked \
  -- --nocapture --test-threads=1
```

For the declared MSRV, run
`cargo +1.85.0 check --workspace --all-targets --locked` and
`cargo +1.85.0 check -p pencil-io --all-features --all-targets --locked` before the native
matrix.

After building the HDF5 test binary, inspect its resolved native ABI before
running the matrix (the exact binary name is printed by Cargo):

```bash
ldd target/debug/deps/hdf5_io-* | grep -E 'lib(hdf5|mpi|open-rte|open-pal)'
mpiexec --version
pkg-config --modversion hdf5-openmpi
```

## Prerequisites

- Rust stable, with a minimum supported Rust version of 1.85
- A C MPI implementation such as Open MPI or MPICH (for `pencil-array` and
  distributed tests; local `pencil-fft` tests do not require MPI)
- `mpicc` and `mpiexec` available on `PATH`
- libclang and its C development headers, required by bindgen while building
  the MPI bindings

## Verification

```bash
set -euo pipefail
cargo fmt --all -- --check
cargo test -p pencil-fft --no-default-features --locked
cargo tree -p pencil-fft --no-default-features --edges normal --locked | tee /tmp/pencil-fft-local-tree.txt
if grep -Eq '(^|[[:space:]])(mpi|pencil-array)([[:space:]]|$)' /tmp/pencil-fft-local-tree.txt; then exit 1; fi
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo clippy -p pencil-fft --features distributed --all-targets --locked -- -D warnings
cargo test --workspace --lib --locked
cargo check -p pencil-fft --features distributed --all-targets --locked
cargo test -p pencil-fft --features distributed --lib --locked
cargo test -p pencil-fft --features distributed --lib distributed::r2r::tests::in_place_error_panic_backend_and_short_workspace_poison_contracts --locked -- --ignored --exact --nocapture --test-threads=1
for n in 1 4 6; do
  timeout --foreground --kill-after=5s 120s mpiexec --oversubscribe -n "$n" \
    cargo test -p pencil-fft --features distributed --lib --locked \
      distributed::tests::in_place_transaction_poison_survives_error_and_panic \
      -- --exact --nocapture --test-threads=1
done

mpiexec -n 1 cargo test -p pencil-array --test topology --locked -- --nocapture --test-threads=1
mpiexec -n 4 cargo test -p pencil-array --test topology --locked -- --nocapture --test-threads=1
mpiexec -n 1 cargo test -p pencil-array --test pencil --locked -- --nocapture --test-threads=1
mpiexec -n 4 cargo test -p pencil-array --test pencil --locked -- --nocapture --test-threads=1
mpiexec -n 1 cargo test -p pencil-array --test array --locked -- --nocapture --test-threads=1
mpiexec -n 4 cargo test -p pencil-array --test array --locked -- --nocapture --test-threads=1
mpiexec -n 1 cargo test -p pencil-array --test many --locked -- --nocapture --test-threads=1
mpiexec -n 4 cargo test -p pencil-array --test many --locked -- --nocapture --test-threads=1
# The local operation is noncollective; keep a timeout to catch deadlocks.
timeout --foreground 120s mpiexec -n 1 cargo test -p pencil-array --test local_transpose --locked -- --nocapture --test-threads=1
timeout --foreground 120s mpiexec -n 4 cargo test -p pencil-array --test local_transpose --locked -- --nocapture --test-threads=1
timeout --foreground 120s mpiexec -n 1 cargo test -p pencil-array --test alltoallv_transpose --locked -- --nocapture --test-threads=1
timeout --foreground 120s mpiexec -n 4 cargo test -p pencil-array --test alltoallv_transpose --locked -- --nocapture --test-threads=1
timeout --foreground 120s mpiexec -n 6 cargo test -p pencil-array --test alltoallv_transpose --locked -- --nocapture --test-threads=1
timeout --foreground 120s mpiexec -n 1 cargo test -p pencil-array --test collectives --locked -- --nocapture --test-threads=1
timeout --foreground 120s mpiexec -n 4 cargo test -p pencil-array --test collectives --locked -- --nocapture --test-threads=1
timeout --foreground 120s mpiexec -n 6 cargo test -p pencil-array --test collectives --locked -- --nocapture --test-threads=1

cargo doc --workspace --no-deps --locked
cargo doc -p pencil-fft --features distributed --no-deps --locked
cargo test --workspace --doc --locked -- --show-output
cargo test -p pencil-fft --features distributed --doc --locked -- --show-output
# The distributed suites cover C2C, R2C/C2R, R2R, and mixed-axis plans over
# Alltoallv/P2P; distributed C2C, R2C/C2R, R2R, and mixed plans also have
# in-place APIs; local C2C
# and local R2C also have in-place APIs.
for n in 1 4 6; do
  timeout --foreground --kill-after=5s 120s mpiexec --oversubscribe -n "$n" \
    cargo test -p pencil-fft --features distributed --test distributed_c2c \
      --locked -- --nocapture --test-threads=1 || exit 1
  timeout --foreground --kill-after=5s 120s mpiexec --oversubscribe -n "$n" \
    cargo test -p pencil-fft --features distributed --test distributed_r2r \
      --locked -- --nocapture --test-threads=1 || exit 1
  timeout --foreground --kill-after=5s 120s mpiexec --oversubscribe -n "$n" \
    cargo test -p pencil-fft --features distributed --test distributed_mixed \
      --locked -- --nocapture --test-threads=1 || exit 1
done
```

Rustdoc tests cover usage examples and compile-time borrowing and visibility
restrictions. `LocalR2cPlan` preserves its source by copying one line at a time
into caller-owned initialized storage, uses caller-owned complex scratch, and
provides a packed single-allocation in-place API. It exposes the original real
length `n`, the reduced complex length `n/2+1`, and the shared native scratch
requirement; forward and backward are unscaled and inverse divides each line by
`n`. The in-place API owns one `Vec<Complex<R>>` and exposes only its
initialized real or complex prefix for the current state. Its inverse and
backward accept only strict-zero DC and, for even lengths, Nyquist imaginary
components; for odd lengths greater than one, the final-bin imaginary
component is unconstrained, while `n = 1` DC remains constrained. Ordinary
validation errors preserve data and workspace, while backend/resource panics
are not converted.

The topology constructors are collective over an MPI intracommunicator. Run
each integration-test binary with one MPI initialization per process, and drop
all topologies and arrays before MPI finalizes.

## Design and plans

- `docs/superpowers/specs/2026-09-11-pencil-arrays-rust-port-design.md`
- `docs/superpowers/plans/2026-09-11-pencil-array-core-implementation.md`
- `docs/superpowers/plans/2026-09-14-local-transpose-implementation.md`
- `docs/superpowers/plans/2026-09-14-alltoallv-transpose-implementation.md`
- `docs/superpowers/plans/2026-09-14-alltoallv-in-place-implementation.md`
- [Point-to-point transpose implementation plan](docs/superpowers/plans/2026-09-15-point-to-point-transpose-implementation.md)
- [Point-to-point in-place implementation plan](docs/superpowers/plans/2026-09-15-point-to-point-in-place-implementation.md)
- [Local C2C implementation plan](docs/superpowers/plans/2026-09-15-local-c2c-implementation.md)
- [Local R2C/C2R implementation plan](docs/superpowers/plans/2026-09-17-local-r2c-implementation.md)
- [Distributed C2C implementation plan](docs/superpowers/plans/2026-09-17-distributed-c2c-implementation.md)
- [Distributed C2C in-place implementation plan](docs/superpowers/plans/2026-09-18-distributed-c2c-in-place-implementation.md)
- [Distributed C2C point-to-point implementation plan](docs/superpowers/plans/2026-09-18-distributed-c2c-point-to-point-implementation.md)
- [Distributed R2C/C2R implementation plan](docs/superpowers/plans/2026-09-18-distributed-r2c-implementation.md)
- [Julia/FFTW cross-validation plan](docs/superpowers/plans/2026-09-19-fftw-cross-validation.md)
