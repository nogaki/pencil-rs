# pencil-array

MPI-distributed pencil arrays, Cartesian decompositions, local views, global
collectives, and Alltoallv or point-to-point transposes. This crate provides the
array layer; it does not depend on FFT or I/O crates.

Requires Rust 1.85 or newer, a native MPI installation (including headers and
`mpicc`), and libclang for MPI bindings. Build and run against the same MPI
implementation. Julia, FFTW, HDF5, and GPUs are not required.

Initialize MPI before constructing topologies. Every participating rank must
enter collective construction, transpose, and global reduction operations in
the same order with compatible descriptors. Coordinate local allocation or
validation failures before entering the next collective. Drop MPI-owned objects
before MPI finalization. See the API documentation for individual contracts.

For transforms, `pencil-fft` uses MPI-free RustFFT/RealFFT by default; its
`distributed` feature adds this crate, and native FFTW is a separate opt-in.
For collective storage, use `pencil-io`.

Source license: MIT; see LICENSE and NOTICE.md for attribution and the separate
licensing obligations of the optional native FFTW runtime. No Julia runtime or
helper package is needed.
