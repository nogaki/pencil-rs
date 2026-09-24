# pencil-io

Collective MPI-IO and optional parallel HDF5 storage for `pencil-array` views,
including named datasets, collections, and persistent sessions. This is an I/O
layer, independent of FFT providers.

Requires Rust 1.85 or newer, native MPI headers and `mpicc`, and libclang for MPI
bindings. Build and run with the same MPI implementation. Default builds use
MPI-IO without HDF5. The `parallel-hdf5` feature additionally requires a parallel
(MPI-enabled) HDF5 installation with headers and libraries built against that
same MPI; a serial HDF5 installation is insufficient.

Every participating rank must enter collective open/read/write/close operations
in the same order with compatible descriptors. Coordinate local failures before
the next collective. **Call persistent `session.close()` collectively before
MPI finalization.** Dropping an open session is fail-stop, not an implicit
rank-local close. Drop other MPI-owned objects before finalization as well.
Consult the API documentation for file formats and per-operation guarantees.

No Julia, FFTW, or GPU is required. If transforms are also needed, `pencil-fft`
uses MPI-free RustFFT/RealFFT by default, with distributed and native FFTW paths
separately opt-in. Source license: MIT; see LICENSE and NOTICE.md for attribution
and the distinct optional FFTW licensing obligations.
