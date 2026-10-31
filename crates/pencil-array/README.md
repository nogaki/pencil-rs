# pencil-array

MPI-distributed pencil arrays, Cartesian decompositions, local views, global
collectives, and Alltoallv or point-to-point transposes. This crate provides the
array layer; it does not depend on FFT or I/O crates.

Requires Rust 1.96 or newer. The default build needs a native MPI installation
(including headers and `mpicc`) and libclang for MPI bindings. Build and run
against the same MPI implementation. Julia, FFTW, HDF5, and GPUs are not required.
Experimental [rsmpi-rt support](https://github.com/nogaki/pencil-rs#experimental-runtime-mpi-rsmpi-rt)
allows building without native MPI tools and loading MPIwrapper at runtime;
it requires an application-level Cargo patch and backend feature selection.

Initialize MPI before constructing topologies. Every participating rank must
enter collective construction, transpose, and global reduction operations in
the same order with compatible descriptors. Coordinate local allocation or
validation failures before entering the next collective. Drop MPI-owned objects
before MPI finalization. See the API documentation for individual contracts.

## Affine spatial remapping

`SpatialRemapPlan<T, N, M>` maps each destination coordinate `d` to source
coordinate `s[a] = (sign[a] * d[a] + offset[a]) mod global_extent[a]`, with each
sign equal to `-1` or `1`. Offsets are compared as raw `i128` values across
ranks, then reduced modulo the positive extent only while building routes. The
source and destination pencils must share the same rank-local topology object
and global shape, but their decompositions may be unchanged, differ at one
position, or differ at several positions; memory permutations may also differ.
Every extra-shape group uses the same spatial map. Extra axes are unchanged,
and zero-product extra shapes and empty local partitions are supported.

`prepare` and every `apply_views` call are collective on the full Cartesian
communicator. Preparation agrees the fixed operation header, signs, topology
and shape, complete descriptor, owner routes, request counts, received local
indices, payload counts, initialized buffers, and duplicated datatype in that
order. `apply_views` agrees its header and immutable descriptor before checking
view extra shapes and layouts, then packs, performs one payload `Alltoallv`, and
unpacks. Checked view construction is local; callers must collectively agree a
view-construction failure before calling `apply_views`.

The reusable plan owns both pencils, the exact `ExtraShape`, complete immutable
descriptor, owner/index maps, MPI counts/displacements, checked standard
`UserDatatype`, and initialized send/receive buffers. `T` needs only
`Equivalence + Copy`; `buffer_init` initializes work buffers without requiring
`Default`. Apply, including ordinary failed preflight, performs no Rust
allocation and does not call user `Clone`, `Default`, `Equivalence`, or datatype
methods. Ordinary errors preserve source, destination, and all plan contents.
Zero-sized payloads and counts/displacements beyond MPI `Count` are rejected;
large-message chunking is not provided. The unsafe `Equivalence` implementation
must accurately describe initialized fields within each `T`; the plan checks
extent and true extent but cannot prove a custom typemap correct. Drop the plan
before MPI finalization. MPI failures, process loss, and arbitrary panics do not
guarantee recovery.

For transforms, `pencil-fft` uses MPI-free RustFFT/RealFFT by default; its
`distributed` feature adds this crate, and native FFTW is a separate opt-in.
For collective storage, use `pencil-io`.

Source license: MIT; see LICENSE and NOTICE.md for attribution and the separate
licensing obligations of the optional native FFTW runtime. No Julia runtime or
helper package is needed.
