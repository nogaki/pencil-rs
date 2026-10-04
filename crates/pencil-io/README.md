# pencil-io

Collective MPI-IO for `pencil-array` views, plus optional serial or parallel
HDF5 storage. This is an I/O layer, independent of FFT providers.

Requires Rust 1.85 or newer. Native dependencies depend on the selected features:

- `mpi` (default): MPI-IO, named datasets, collections, and persistent sessions.
  Requires native MPI headers, `mpicc`, and libclang for MPI bindings. Build and
  run with the same MPI implementation; HDF5 is not needed.
- `hdf5`: serial HDF5 over flat slices. Disable default features to exclude MPI
  and `pencil-array`. An installed HDF5 library and headers are still required;
  use a serial HDF5 build to avoid native MPI dependencies as well.
- `parallel-hdf5`: enables `mpi`, `hdf5`, and HDF5 MPIO support. Requires parallel
  HDF5 headers and libraries built against the same MPI implementation;
  a serial HDF5 installation is insufficient for this feature.

## Serial HDF5 without MPI

```toml
[dependencies]
pencil-io = { version = "0.1.0", default-features = false, features = ["hdf5"] }
```

Install serial HDF5 (for example, `sudo apt-get install libhdf5-dev pkg-config`
on Debian/Ubuntu). Discovery uses `pkg-config` or `HDF5_DIR`. Cargo features are
additive: enabling `mpi` or `parallel-hdf5` elsewhere brings MPI back.

Both serial functions take `(path, global_shape: &[usize], extra_shape: &[usize],
buffer)` and return `Result<(), IoError>`; the buffer is `&[T]` for writing and
`&mut [T]` for reading, with `T: IoElement`. Supply the complete buffer in
row-major `[extra..., spatial...]` order. For example, two `[2, 3]` arrays:

```rust
use pencil_io::{IoError, read_hdf5_serial, write_hdf5_serial};

fn main() -> Result<(), IoError> {
    let global_shape = [2, 3];
    let extra_shape = [2];
    let values: Vec<f64> = (0..12).map(f64::from).collect();
    // field.h5 must not already exist: writes never overwrite a file.
    write_hdf5_serial("field.h5", &global_shape, &extra_shape, &values)?;
    let mut destination = vec![0.0; values.len()];
    read_hdf5_serial("field.h5", &global_shape, &extra_shape, &mut destination)?;
    assert_eq!(destination, values);
    Ok(())
}
```

The single versioned dataset `/pencil_io_v1/data` uses the same format as the
parallel HDF5 APIs, including metadata and the commit marker. All 12 scalar
representations are supported: `i8`/`u8`, `i16`/`u16`, `i32`/`u32`, `i64`/`u64`,
`f32`/`f64`, and `num_complex::Complex<f32>`/`Complex<f64>`. Complex values use
little-endian `{r,i}` compound types. A returned read error preserves the entire
destination. Writes exclusively create files; failed writes may leave a file
behind, with no rollback or crash-recovery promise. The serial API is limited
to this single-dataset format, not named datasets, collections, catalogs,
sessions, or raw input.

```bash
cargo test -p pencil-io --no-default-features --features hdf5 --locked
cargo clippy -p pencil-io --no-default-features --features hdf5 --all-targets --locked -- -D warnings
cargo doc -p pencil-io --no-default-features --features hdf5 --no-deps --locked
```

## Collective I/O (`mpi` / `parallel-hdf5`)

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
