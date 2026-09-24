# pencil-fft

Local complex, real, and real-to-real transforms, with optional MPI-distributed
pencil transforms. Requires Rust 1.85 or newer. The default RustFFT/RealFFT CPU
backend is MPI-free and needs neither native FFTW nor Julia.

- Default features: local transforms using RustFFT/RealFFT.
- `distributed`: adds MPI and `pencil-array`. Requires native MPI headers,
  `mpicc`, and libclang; use the same MPI implementation to build and run.
- `fftw`: exposes explicit native FFTW constructors through `pencil-fftw`.
  Enabling it does not switch the default provider. Explicit FFTW construction
  requires the separately installed Linux `libfftw3.so.3` (f64) or
  `libfftw3f.so.3` (f32); threaded plans also require the matching thread runtime.
  Loading/planning failures are errors, not silent RustFFT fallback.

Local forward and backward transforms are unnormalized; inverse methods apply
normalization. Consult the API documentation for buffer sizes and layout rules.
Distributed plan construction and execution are collective: all communicator
ranks must enter compatible calls in the same order. Coordinate local allocation
failures before the next collective, and drop MPI-owned objects before MPI
finalization. The `mpi_directions` example requires `distributed`.

No HDF5, GPU, or Julia helper package is required. Source license: MIT; see
LICENSE and NOTICE.md. FFTW itself is GPL v2 or later (or separately commercially
licensed); dynamic loading is not a GPL exemption. Evaluate the runtime's terms
before distributing FFTW-backed software.
