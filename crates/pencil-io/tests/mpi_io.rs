use std::fmt::Debug;
use std::path::Path;

use mpi::collective::SystemOperation;
use mpi::traits::*;
use num_complex::Complex;
use pencil_array::{AxisPermutation, ExtraShape, MpiTopology, Pencil, PencilArray};
use pencil_io::{IoElement, IoError, read_mpi, write_mpi};

mod support;
use support::{cleanup_owned_temp_dir, owned_temp_dir};

fn root_status<C, F>(world: &C, operation: F)
where
    C: CommunicatorCollectives,
    F: FnOnce() -> Result<(), String>,
{
    let result = if world.rank() == 0 {
        operation()
    } else {
        Ok(())
    };
    let local_ok = i32::from(result.is_ok());
    let mut all_ok = 0;
    world.all_reduce_into(&local_ok, &mut all_ok, SystemOperation::min());
    if all_ok != 1 {
        panic!(
            "root filesystem operation failed: {}",
            result.err().unwrap_or_default()
        );
    }
}

fn root_read<C>(world: &C, path: &Path) -> Vec<u8>
where
    C: CommunicatorCollectives,
{
    let result = if world.rank() == 0 {
        std::fs::read(path).map_err(|error| error.to_string())
    } else {
        Ok(Vec::new())
    };
    let local_ok = i32::from(result.is_ok());
    let mut all_ok = 0;
    world.all_reduce_into(&local_ok, &mut all_ok, SystemOperation::min());
    if all_ok != 1 {
        panic!(
            "root file read failed: {}",
            result.err().unwrap_or_default()
        );
    }
    result.unwrap_or_default()
}

fn reset_file<C>(world: &C, path: &Path)
where
    C: CommunicatorCollectives,
{
    root_status(world, || match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.to_string()),
    });
    world.barrier();
}

fn patch_u64(bytes: &mut [u8], offset: usize, value: u64) -> Result<(), String> {
    let end = offset
        .checked_add(8)
        .ok_or_else(|| "patch offset overflow".to_owned())?;
    let slot = bytes
        .get_mut(offset..end)
        .ok_or_else(|| format!("patch offset {offset} outside file"))?;
    slot.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn prepare_case<C, F>(world: &C, baseline: &Path, case: &Path, mutate: F)
where
    C: CommunicatorCollectives,
    F: FnOnce(&mut Vec<u8>) -> Result<(), String>,
{
    root_status(world, || {
        let mut bytes = std::fs::read(baseline).map_err(|error| error.to_string())?;
        mutate(&mut bytes)?;
        std::fs::write(case, bytes).map_err(|error| error.to_string())
    });
    world.barrier();
}

fn assert_mpi_failure<C, F>(
    world: &C,
    path: &Path,
    destination: &mut PencilArray<i32, 2, 2>,
    expected: F,
) where
    C: Communicator + CommunicatorCollectives,
    F: Fn(&IoError) -> bool,
{
    let before = destination.as_slice().to_vec();
    let error = read_mpi(path, destination.view_mut()).expect_err("read must fail");
    assert!(expected(&error), "unexpected MPI error: {error:?}");
    assert_eq!(destination.as_slice(), before.as_slice());
    world.barrier();
}

fn run_mpi_scalar_case<C, T, F>(world: &C, directory: &Path, name: &str, make_value: F)
where
    C: Communicator + CommunicatorCollectives,
    T: IoElement + Debug + PartialEq,
    F: Fn(usize, usize) -> T,
{
    let size = usize::try_from(world.size()).unwrap();
    let path = directory.join(name);
    reset_file(world, &path);
    let writer_topology = MpiTopology::<2>::new(world, [size, 1]).unwrap();
    let writer_pencil = Pencil::<2, 2>::new_permuted(
        writer_topology,
        [4, 5],
        [0, 1],
        AxisPermutation::new([1, 0]).unwrap(),
    )
    .unwrap();
    let mut source =
        PencilArray::from_elem(writer_pencil, ExtraShape::scalar(), make_value(0, 0)).unwrap();
    {
        let mut view = source.view_mut();
        let ranges = view.pencil().local_ranges().clone();
        for x in 0..ranges[0].len() {
            for y in 0..ranges[1].len() {
                *view.get_local_mut(&[], [x, y]).unwrap() =
                    make_value(ranges[0].start + x, ranges[1].start + y);
            }
        }
    }
    write_mpi(&path, source.view()).unwrap();
    world.barrier();
    let reader_topology = MpiTopology::<2>::new(world, [1, size]).unwrap();
    let reader_pencil = Pencil::<2, 2>::new(reader_topology, [4, 5], [1, 0]).unwrap();
    let mut destination =
        PencilArray::from_elem(reader_pencil, ExtraShape::scalar(), make_value(0, 0)).unwrap();
    read_mpi(&path, destination.view_mut()).unwrap();
    let view = destination.view();
    let ranges = view.pencil().local_ranges().clone();
    for x in 0..ranges[0].len() {
        for y in 0..ranges[1].len() {
            assert_eq!(
                view.get_local(&[], [x, y]),
                Some(&make_value(ranges[0].start + x, ranges[1].start + y)),
            );
        }
    }
    world.barrier();
}

#[test]
fn mpi_io_preserves_logical_order_and_rejects_bad_files() {
    let universe = mpi::initialize().expect("MPI must initialize once");
    let world = universe.world();
    let size = usize::try_from(world.size()).unwrap();
    let directory = owned_temp_dir(&world, "pencil-io-test");

    let writer_topology = MpiTopology::<2>::new(&world, [size, 1]).unwrap();
    let writer_pencil = Pencil::<2, 2>::new_permuted(
        writer_topology,
        [4, 5],
        [0, 1],
        AxisPermutation::new([1, 0]).unwrap(),
    )
    .unwrap();
    let extra = ExtraShape::new([2]).unwrap();
    let mut source = PencilArray::from_elem(writer_pencil, extra.clone(), -1i32).unwrap();
    {
        let mut view = source.view_mut();
        let ranges = view.pencil().local_ranges().clone();
        for e in 0..2 {
            for x in 0..ranges[0].len() {
                for y in 0..ranges[1].len() {
                    let global_x = ranges[0].start + x;
                    let global_y = ranges[1].start + y;
                    *view.get_local_mut(&[e], [x, y]).unwrap() =
                        (e * 1000 + global_x * 100 + global_y) as i32;
                }
            }
        }
    }
    let path = directory.join("array.pio");
    write_mpi(&path, source.view()).unwrap();
    world.barrier();

    let baseline = root_read(&world, &path);
    root_status(&world, || {
        if baseline.get(0..8) != Some(b"PENCILIO") {
            return Err("MPI magic".to_owned());
        }
        let marker = baseline
            .get(24..32)
            .ok_or_else(|| "MPI commit marker bounds".to_owned())?;
        if marker != 0x434f_4d4d_4954_5445u64.to_le_bytes() {
            return Err("MPI commit marker".to_owned());
        }
        let mut expected_payload = Vec::new();
        for e in 0..2 {
            for x in 0..4 {
                for y in 0..5 {
                    let value: i32 = e * 1000 + x * 100 + y;
                    expected_payload.extend_from_slice(&value.to_le_bytes());
                }
            }
        }
        if baseline.get(152..) != Some(expected_payload.as_slice()) {
            return Err("MPI canonical payload".to_owned());
        }
        Ok(())
    });
    world.barrier();

    let before_existing = root_read(&world, &path);
    let existing_error = write_mpi(&path, source.view()).expect_err("exclusive write must fail");
    assert!(matches!(
        existing_error,
        IoError::Mpi {
            operation: "MPI_File_open",
            ..
        }
    ));
    root_status(&world, || {
        let after = std::fs::read(&path).map_err(|error| error.to_string())?;
        if after != before_existing {
            return Err("existing MPI file changed".to_owned());
        }
        Ok(())
    });
    world.barrier();

    let reader_topology = MpiTopology::<2>::new(&world, [1, size]).unwrap();
    let reader_pencil = Pencil::<2, 2>::new(reader_topology, [4, 5], [1, 0]).unwrap();
    let mut destination = PencilArray::from_elem(reader_pencil, extra, -777i32).unwrap();
    read_mpi(&path, destination.view_mut()).unwrap();
    assert_i32_values(&destination);

    let missing_path = directory.join("missing.pio");
    reset_file(&world, &missing_path);
    let before_missing = destination.as_slice().to_vec();
    let missing_error = read_mpi(&missing_path, destination.view_mut()).expect_err("missing read");
    assert!(matches!(
        missing_error,
        IoError::Mpi {
            operation: "MPI_File_open",
            ..
        }
    ));
    assert_eq!(destination.as_slice(), before_missing.as_slice());
    world.barrier();

    let baseline_path = directory.join("baseline.pio");
    root_status(&world, || {
        std::fs::write(&baseline_path, &baseline).map_err(|error| error.to_string())
    });
    world.barrier();
    let case_path = directory.join("case.pio");

    prepare_case(&world, &baseline_path, &case_path, |bytes| {
        bytes[0] = b'X';
        Ok(())
    });
    assert_mpi_failure(&world, &case_path, &mut destination, |error| {
        matches!(error, IoError::InvalidFile { reason: "magic" })
    });

    prepare_case(&world, &baseline_path, &case_path, |bytes| {
        patch_u64(bytes, 8, 2)
    });
    assert_mpi_failure(&world, &case_path, &mut destination, |error| {
        matches!(error, IoError::InvalidFile { reason: "version" })
    });

    prepare_case(&world, &baseline_path, &case_path, |bytes| {
        patch_u64(bytes, 40, 99)
    });
    assert_mpi_failure(&world, &case_path, &mut destination, |error| {
        matches!(
            error,
            IoError::InvalidFile {
                reason: "type descriptor"
            }
        )
    });

    // The first writer-grid dimension is provenance metadata and must stay a
    // positive extent even though it is not a reader-layout requirement.
    prepare_case(&world, &baseline_path, &case_path, |bytes| {
        patch_u64(bytes, 120, 0)
    });
    assert_mpi_failure(&world, &case_path, &mut destination, |error| {
        matches!(
            error,
            IoError::InvalidFile {
                reason: "header metadata"
            }
        )
    });

    // A valid header with a different logical extent is a metadata mismatch.
    prepare_case(&world, &baseline_path, &case_path, |bytes| {
        patch_u64(bytes, 96, 3)?;
        patch_u64(bytes, 72, 2 * 3 * 5 * 4)?;
        bytes.truncate(152 + 2 * 3 * 5 * 4);
        Ok(())
    });
    assert_mpi_failure(&world, &case_path, &mut destination, |error| {
        matches!(
            error,
            IoError::MetadataMismatch {
                field: "global shape"
            }
        )
    });

    prepare_case(&world, &baseline_path, &case_path, |bytes| {
        patch_u64(bytes, 24, 7)
    });
    assert_mpi_failure(&world, &case_path, &mut destination, |error| {
        matches!(
            error,
            IoError::InvalidFile {
                reason: "commit marker"
            }
        )
    });

    prepare_case(&world, &baseline_path, &case_path, |bytes| {
        patch_u64(bytes, 24, 0x494e_434f_4d50_4c45)
    });
    assert_mpi_failure(&world, &case_path, &mut destination, |error| {
        matches!(error, IoError::IncompleteFile)
    });

    prepare_case(&world, &baseline_path, &case_path, |bytes| {
        bytes.truncate(8);
        Ok(())
    });
    assert_mpi_failure(&world, &case_path, &mut destination, |error| {
        matches!(
            error,
            IoError::InvalidFile {
                reason: "short header"
            }
        )
    });

    prepare_case(&world, &baseline_path, &case_path, |bytes| {
        bytes.pop();
        Ok(())
    });
    assert_mpi_failure(&world, &case_path, &mut destination, |error| {
        matches!(
            error,
            IoError::InvalidFile {
                reason: "trailing or truncated payload"
            }
        )
    });

    prepare_case(&world, &baseline_path, &case_path, |bytes| {
        bytes.push(0);
        Ok(())
    });
    assert_mpi_failure(&world, &case_path, &mut destination, |error| {
        matches!(
            error,
            IoError::InvalidFile {
                reason: "trailing or truncated payload"
            }
        )
    });

    prepare_case(&world, &baseline_path, &case_path, |bytes| {
        patch_u64(bytes, 40, 6)
    });
    assert_mpi_failure(&world, &case_path, &mut destination, |error| {
        matches!(
            error,
            IoError::MetadataMismatch {
                field: "type or rank"
            }
        )
    });

    // The untouched baseline remains readable after every rejected case.
    read_mpi(&baseline_path, destination.view_mut()).unwrap();
    assert_i32_values(&destination);
    world.barrier();

    let zero_path = directory.join("zero-extra.pio");
    reset_file(&world, &zero_path);
    let zero_writer_topology = MpiTopology::<2>::new(&world, [size, 1]).unwrap();
    let zero_writer_pencil = Pencil::<2, 2>::new(zero_writer_topology, [4, 5], [0, 1]).unwrap();
    let zero_extra = ExtraShape::new([0]).unwrap();
    let zero_source = PencilArray::from_elem(zero_writer_pencil, zero_extra.clone(), 7i32).unwrap();
    write_mpi(&zero_path, zero_source.view()).unwrap();
    world.barrier();
    let zero_reader_topology = MpiTopology::<2>::new(&world, [1, size]).unwrap();
    let zero_reader_pencil = Pencil::<2, 2>::new(zero_reader_topology, [4, 5], [1, 0]).unwrap();
    let mut zero_destination =
        PencilArray::from_elem(zero_reader_pencil, zero_extra, -9i32).unwrap();
    read_mpi(&zero_path, zero_destination.view_mut()).unwrap();
    assert!(zero_destination.as_slice().is_empty());
    world.barrier();

    let complex_path = directory.join("complex-f64.pio");
    reset_file(&world, &complex_path);
    let complex_writer_topology = MpiTopology::<2>::new(&world, [size, 1]).unwrap();
    let complex_writer_pencil = Pencil::<2, 2>::new_permuted(
        complex_writer_topology,
        [4, 5],
        [0, 1],
        AxisPermutation::new([1, 0]).unwrap(),
    )
    .unwrap();
    let mut complex_source = PencilArray::from_elem(
        complex_writer_pencil,
        ExtraShape::scalar(),
        Complex::new(0.0, 0.0),
    )
    .unwrap();
    {
        let mut view = complex_source.view_mut();
        let ranges = view.pencil().local_ranges().clone();
        for x in 0..ranges[0].len() {
            for y in 0..ranges[1].len() {
                let global_x = ranges[0].start + x;
                let global_y = ranges[1].start + y;
                *view.get_local_mut(&[], [x, y]).unwrap() = Complex::new(
                    (global_x * 100 + global_y) as f64,
                    (global_x as f64) - global_y as f64,
                );
            }
        }
    }
    write_mpi(&complex_path, complex_source.view()).unwrap();
    world.barrier();
    let complex_reader_topology = MpiTopology::<2>::new(&world, [1, size]).unwrap();
    let complex_reader_pencil =
        Pencil::<2, 2>::new(complex_reader_topology, [4, 5], [1, 0]).unwrap();
    let mut complex_destination = PencilArray::from_elem(
        complex_reader_pencil,
        ExtraShape::scalar(),
        Complex::new(-1.0, -1.0),
    )
    .unwrap();
    read_mpi(&complex_path, complex_destination.view_mut()).unwrap();
    assert_complex_values(&complex_destination);

    macro_rules! scalar_case {
        ($name:literal, $ty:ty, $make:expr) => {
            run_mpi_scalar_case::<_, $ty, _>(&world, &directory, $name, $make);
        };
    }
    scalar_case!("i8.pio", i8, |x, y| (x * 100 + y) as i8);
    scalar_case!("u8.pio", u8, |x, y| (x * 10 + y) as u8);
    scalar_case!("i16.pio", i16, |x, y| (x * 100 + y) as i16);
    scalar_case!("u16.pio", u16, |x, y| (x * 100 + y) as u16);
    scalar_case!("u32.pio", u32, |x, y| (x * 100 + y) as u32);
    scalar_case!("i64.pio", i64, |x, y| (x * 100 + y) as i64);
    scalar_case!("u64.pio", u64, |x, y| (x * 100 + y) as u64);
    scalar_case!("f32.pio", f32, |x, y| (x * 100 + y) as f32);
    scalar_case!("f64.pio", f64, |x, y| (x * 100 + y) as f64);
    scalar_case!("complex-f32.pio", Complex<f32>, |x, y| Complex::new(
        (x * 100 + y) as f32,
        (x as f32) - (y as f32),
    ));
    scalar_case!("complex-f64.pio", Complex<f64>, |x, y| Complex::new(
        (x * 100 + y) as f64,
        (x as f64) - (y as f64),
    ));

    cleanup_owned_temp_dir(&world, &directory);
}

fn assert_i32_values(array: &PencilArray<i32, 2, 2>) {
    let view = array.view();
    let ranges = view.pencil().local_ranges().clone();
    for e in 0..2 {
        for x in 0..ranges[0].len() {
            for y in 0..ranges[1].len() {
                let global_x = ranges[0].start + x;
                let global_y = ranges[1].start + y;
                assert_eq!(
                    view.get_local(&[e], [x, y]),
                    Some(&((e * 1000 + global_x * 100 + global_y) as i32))
                );
            }
        }
    }
}

fn assert_complex_values(array: &PencilArray<Complex<f64>, 2, 2>) {
    let view = array.view();
    let ranges = view.pencil().local_ranges().clone();
    for x in 0..ranges[0].len() {
        for y in 0..ranges[1].len() {
            let global_x = ranges[0].start + x;
            let global_y = ranges[1].start + y;
            assert_eq!(
                view.get_local(&[], [x, y]),
                Some(&Complex::new(
                    (global_x * 100 + global_y) as f64,
                    (global_x as f64) - (global_y as f64),
                )),
            );
        }
    }
}
