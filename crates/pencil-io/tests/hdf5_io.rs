#![cfg(feature = "parallel-hdf5")]

use std::fmt::Debug;
use std::path::Path;

use hdf5_metno::{File, H5Type};
use mpi::collective::SystemOperation;
use mpi::traits::*;
use num_complex::Complex;
use pencil_array::{AxisPermutation, ExtraShape, MpiTopology, Pencil, PencilArray};
use pencil_io::{IoElement, IoError, read_hdf5, write_hdf5};

mod support;
use support::{cleanup_owned_temp_dir, owned_temp_dir};

const COMMIT_MARKER: u64 = 0x434f_4d4d_4954_5445;
const INCOMPLETE_MARKER: u64 = 0x494e_434f_4d50_4c45;

#[repr(C)]
#[derive(Clone, Copy, H5Type)]
struct WrongCompound {
    r: f32,
    j: f32,
}

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
            "root operation failed: {}",
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

fn h5<T>(result: hdf5_metno::Result<T>) -> Result<T, String> {
    result.map_err(|error| error.to_string())
}

fn replace_u64_attr(
    dataset: &hdf5_metno::Dataset,
    name: &str,
    values: &[u64],
) -> Result<(), String> {
    let _ = dataset.delete_attr(name);
    if values.len() == 1 {
        let attr = h5(dataset.new_attr::<u64>().create(name))?;
        h5(attr.as_writer().write_scalar(&values[0]))
    } else {
        let attr = h5(dataset.new_attr::<u64>().shape(values.len()).create(name))?;
        h5(attr.as_writer().write_raw(values))
    }
}

fn write_protocol_attrs(
    dataset: &hdf5_metno::Dataset,
    size: usize,
    type_code: u64,
    width: u64,
    extra: &[u64],
    global: &[u64],
) -> Result<(), String> {
    replace_u64_attr(dataset, "pencil_io_version", &[1])?;
    replace_u64_attr(dataset, "pencil_io_commit", &[COMMIT_MARKER])?;
    replace_u64_attr(dataset, "pencil_io_n", &[2])?;
    replace_u64_attr(dataset, "pencil_io_type", &[type_code])?;
    replace_u64_attr(dataset, "pencil_io_width", &[width])?;
    replace_u64_attr(dataset, "pencil_io_extra_rank", &[extra.len() as u64])?;
    if !extra.is_empty() {
        replace_u64_attr(dataset, "pencil_io_extra_shape", extra)?;
    }
    replace_u64_attr(dataset, "pencil_io_global_shape", global)?;
    replace_u64_attr(dataset, "pencil_io_writer_grid", &[size as u64, 1])?;
    replace_u64_attr(dataset, "pencil_io_writer_permutation", &[1, 0])
}

#[allow(clippy::too_many_arguments)]
fn recreate_dataset(
    path: &Path,
    size: usize,
    shape: &[usize],
    kind: DatasetKind,
    type_code: u64,
    width: u64,
    extra: &[u64],
    global: &[u64],
) -> Result<(), String> {
    let file = h5(File::open_rw(path))?;
    let group = h5(file.group("/pencil_io_v1"))?;
    h5(group.unlink("data"))?;
    let dataset = match kind {
        DatasetKind::I32 => h5(group.new_dataset::<i32>().shape(shape).create("data"))?,
        DatasetKind::F64 => h5(group.new_dataset::<f64>().shape(shape).create("data"))?,
        DatasetKind::WrongCompound => h5(group
            .new_dataset::<WrongCompound>()
            .shape(shape)
            .create("data"))?,
    };
    write_protocol_attrs(&dataset, size, type_code, width, extra, global)?;
    h5(file.flush())?;
    h5(file.close())
}

#[derive(Clone, Copy)]
enum DatasetKind {
    I32,
    F64,
    WrongCompound,
}

fn prepare_case<C, F>(world: &C, baseline: &Path, case: &Path, mutate: F)
where
    C: CommunicatorCollectives,
    F: FnOnce(&Path) -> Result<(), String>,
{
    root_status(world, || {
        std::fs::copy(baseline, case).map_err(|error| error.to_string())?;
        mutate(case)
    });
    world.barrier();
}

fn assert_hdf5_failure<C, F>(
    world: &C,
    path: &Path,
    destination: &mut PencilArray<i32, 2, 2>,
    expected: F,
) where
    C: Communicator + CommunicatorCollectives,
    F: Fn(&IoError) -> bool,
{
    let before = destination.as_slice().to_vec();
    let error = read_hdf5(path, destination.view_mut()).expect_err("HDF5 read must fail");
    assert!(expected(&error), "unexpected HDF5 error: {error:?}");
    assert_eq!(destination.as_slice(), before.as_slice());
    world.barrier();
}

fn run_hdf5_scalar_case<C, T, F>(world: &C, directory: &Path, name: &str, make_value: F)
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
    write_hdf5(&path, source.view()).unwrap();
    world.barrier();
    let reader_topology = MpiTopology::<2>::new(world, [1, size]).unwrap();
    let reader_pencil = Pencil::<2, 2>::new(reader_topology, [4, 5], [1, 0]).unwrap();
    let mut destination =
        PencilArray::from_elem(reader_pencil, ExtraShape::scalar(), make_value(0, 0)).unwrap();
    read_hdf5(&path, destination.view_mut()).unwrap();
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
fn parallel_hdf5_preserves_logical_order_and_rejects_bad_files() {
    let universe = mpi::initialize().expect("MPI must initialize once");
    let world = universe.world();
    let size = usize::try_from(world.size()).unwrap();
    let directory = owned_temp_dir(&world, "pencil-io-hdf5-test");

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
    let path = directory.join("array.h5");
    write_hdf5(&path, source.view()).unwrap();
    world.barrier();

    root_status(&world, || {
        let file = h5(File::open(&path))?;
        let values = h5(file.group("/pencil_io_v1"))
            .and_then(|group| h5(group.dataset("data")))
            .and_then(|dataset| h5(dataset.read_raw::<i32>()))?;
        let expected: Vec<_> = (0..2)
            .flat_map(|e| (0..4).flat_map(move |x| (0..5).map(move |y| e * 1000 + x * 100 + y)))
            .collect();
        if values != expected {
            return Err("HDF5 logical payload".to_owned());
        }
        h5(file.close())
    });
    world.barrier();

    let before_existing = root_read(&world, &path);
    let existing_error = write_hdf5(&path, source.view()).expect_err("exclusive HDF5 write");
    assert!(matches!(
        existing_error,
        IoError::Native {
            operation: "H5Fcreate",
            ..
        }
    ));
    root_status(&world, || {
        let after = std::fs::read(&path).map_err(|error| error.to_string())?;
        if after != before_existing {
            return Err("existing HDF5 file changed".to_owned());
        }
        Ok(())
    });
    world.barrier();

    let reader_topology = MpiTopology::<2>::new(&world, [1, size]).unwrap();
    let reader_pencil = Pencil::<2, 2>::new(reader_topology, [4, 5], [1, 0]).unwrap();
    let mut destination = PencilArray::from_elem(reader_pencil, extra, -777i32).unwrap();
    read_hdf5(&path, destination.view_mut()).unwrap();
    assert_i32_values(&destination);

    let missing_path = directory.join("missing.h5");
    reset_file(&world, &missing_path);
    let before_missing = destination.as_slice().to_vec();
    let missing_error =
        read_hdf5(&missing_path, destination.view_mut()).expect_err("missing HDF5 read");
    assert!(matches!(
        missing_error,
        IoError::Native {
            operation: "H5Fopen",
            ..
        }
    ));
    assert_eq!(destination.as_slice(), before_missing.as_slice());
    world.barrier();

    let baseline_path = directory.join("baseline.h5");
    root_status(&world, || {
        std::fs::copy(&path, &baseline_path)
            .map(|_| ())
            .map_err(|error| error.to_string())
    });
    world.barrier();
    let case_path = directory.join("case.h5");

    prepare_case(&world, &baseline_path, &case_path, |case| {
        let file = h5(File::open_rw(case))?;
        let dataset =
            h5(file.group("/pencil_io_v1")).and_then(|group| h5(group.dataset("data")))?;
        h5(dataset.delete_attr("pencil_io_width"))?;
        h5(file.flush())?;
        h5(file.close())
    });
    assert_hdf5_failure(&world, &case_path, &mut destination, |error| {
        matches!(
            error,
            IoError::MetadataMismatch {
                field: "missing HDF5 attribute"
            }
        )
    });

    prepare_case(&world, &baseline_path, &case_path, |case| {
        let file = h5(File::open_rw(case))?;
        let dataset =
            h5(file.group("/pencil_io_v1")).and_then(|group| h5(group.dataset("data")))?;
        h5(dataset.delete_attr("pencil_io_type"))?;
        let attr = h5(dataset.new_attr::<u32>().create("pencil_io_type"))?;
        h5(attr.as_writer().write_scalar(&5u32))?;
        h5(file.flush())?;
        h5(file.close())
    });
    assert_hdf5_failure(&world, &case_path, &mut destination, |error| {
        matches!(
            error,
            IoError::MetadataMismatch {
                field: "HDF5 attribute type or shape"
            }
        )
    });

    prepare_case(&world, &baseline_path, &case_path, |case| {
        let file = h5(File::open_rw(case))?;
        let dataset =
            h5(file.group("/pencil_io_v1")).and_then(|group| h5(group.dataset("data")))?;
        replace_u64_attr(&dataset, "pencil_io_width", &[4, 4])?;
        h5(file.flush())?;
        h5(file.close())
    });
    assert_hdf5_failure(&world, &case_path, &mut destination, |error| {
        matches!(
            error,
            IoError::MetadataMismatch {
                field: "HDF5 attribute type or shape"
            }
        )
    });

    prepare_case(&world, &baseline_path, &case_path, |case| {
        let file = h5(File::open_rw(case))?;
        let dataset =
            h5(file.group("/pencil_io_v1")).and_then(|group| h5(group.dataset("data")))?;
        replace_u64_attr(&dataset, "pencil_io_version", &[2])?;
        h5(file.flush())?;
        h5(file.close())
    });
    assert_hdf5_failure(&world, &case_path, &mut destination, |error| {
        matches!(
            error,
            IoError::MetadataMismatch {
                field: "version, rank, or type"
            }
        )
    });

    prepare_case(&world, &baseline_path, &case_path, |case| {
        let file = h5(File::open_rw(case))?;
        let dataset =
            h5(file.group("/pencil_io_v1")).and_then(|group| h5(group.dataset("data")))?;
        replace_u64_attr(&dataset, "pencil_io_commit", &[7])?;
        h5(file.flush())?;
        h5(file.close())
    });
    assert_hdf5_failure(&world, &case_path, &mut destination, |error| {
        matches!(
            error,
            IoError::InvalidFile {
                reason: "HDF5 commit marker"
            }
        )
    });

    prepare_case(&world, &baseline_path, &case_path, |case| {
        let file = h5(File::open_rw(case))?;
        let dataset =
            h5(file.group("/pencil_io_v1")).and_then(|group| h5(group.dataset("data")))?;
        replace_u64_attr(&dataset, "pencil_io_commit", &[INCOMPLETE_MARKER])?;
        h5(file.flush())?;
        h5(file.close())
    });
    assert_hdf5_failure(&world, &case_path, &mut destination, |error| {
        matches!(error, IoError::IncompleteFile)
    });

    prepare_case(&world, &baseline_path, &case_path, |case| {
        let file = h5(File::open_rw(case))?;
        let dataset =
            h5(file.group("/pencil_io_v1")).and_then(|group| h5(group.dataset("data")))?;
        replace_u64_attr(&dataset, "pencil_io_global_shape", &[4, 4])?;
        h5(file.flush())?;
        h5(file.close())
    });
    assert_hdf5_failure(&world, &case_path, &mut destination, |error| {
        matches!(
            error,
            IoError::MetadataMismatch {
                field: "shape or writer provenance"
            }
        )
    });

    prepare_case(&world, &baseline_path, &case_path, |case| {
        recreate_dataset(
            case,
            size,
            &[2, 4, 5],
            DatasetKind::F64,
            5,
            4,
            &[2],
            &[4, 5],
        )
    });
    assert_hdf5_failure(&world, &case_path, &mut destination, |error| {
        matches!(
            error,
            IoError::MetadataMismatch {
                field: "dataset datatype"
            }
        )
    });

    prepare_case(&world, &baseline_path, &case_path, |case| {
        recreate_dataset(
            case,
            size,
            &[2, 4, 4],
            DatasetKind::I32,
            5,
            4,
            &[2],
            &[4, 5],
        )
    });
    assert_hdf5_failure(&world, &case_path, &mut destination, |error| {
        matches!(
            error,
            IoError::MetadataMismatch {
                field: "dataset dimensions"
            }
        )
    });

    read_hdf5(&baseline_path, destination.view_mut()).unwrap();
    assert_i32_values(&destination);
    world.barrier();

    let complex_path = directory.join("complex-f32.h5");
    reset_file(&world, &complex_path);
    let complex_writer_topology = MpiTopology::<2>::new(&world, [size, 1]).unwrap();
    let complex_writer_pencil = Pencil::<2, 2>::new_permuted(
        complex_writer_topology,
        [4, 5],
        [0, 1],
        AxisPermutation::new([1, 0]).unwrap(),
    )
    .unwrap();
    let complex_source = PencilArray::from_elem(
        complex_writer_pencil,
        ExtraShape::scalar(),
        Complex::new(1.0f32, 2.0),
    )
    .unwrap();
    write_hdf5(&complex_path, complex_source.view()).unwrap();
    world.barrier();
    let complex_baseline = directory.join("complex-baseline.h5");
    root_status(&world, || {
        std::fs::copy(&complex_path, &complex_baseline)
            .map(|_| ())
            .map_err(|error| error.to_string())
    });
    world.barrier();
    let complex_case = directory.join("complex-case.h5");
    prepare_case(&world, &complex_baseline, &complex_case, |case| {
        recreate_dataset(
            case,
            size,
            &[4, 5],
            DatasetKind::WrongCompound,
            11,
            8,
            &[],
            &[4, 5],
        )
    });
    let complex_reader_topology = MpiTopology::<2>::new(&world, [1, size]).unwrap();
    let complex_reader_pencil =
        Pencil::<2, 2>::new(complex_reader_topology, [4, 5], [1, 0]).unwrap();
    let mut complex_destination = PencilArray::from_elem(
        complex_reader_pencil,
        ExtraShape::scalar(),
        Complex::new(-1.0f32, -1.0),
    )
    .unwrap();
    let before_complex = complex_destination.as_slice().to_vec();
    let complex_error = read_hdf5(&complex_case, complex_destination.view_mut())
        .expect_err("wrong HDF5 compound fields");
    assert!(matches!(
        complex_error,
        IoError::MetadataMismatch {
            field: "dataset datatype"
        }
    ));
    assert_eq!(complex_destination.as_slice(), before_complex.as_slice());
    world.barrier();
    read_hdf5(&complex_baseline, complex_destination.view_mut()).unwrap();
    world.barrier();

    let zero_path = directory.join("zero-extra.h5");
    reset_file(&world, &zero_path);
    let zero_writer_topology = MpiTopology::<2>::new(&world, [size, 1]).unwrap();
    let zero_writer_pencil = Pencil::<2, 2>::new(zero_writer_topology, [4, 5], [0, 1]).unwrap();
    let zero_extra = ExtraShape::new([0]).unwrap();
    let zero_source = PencilArray::from_elem(zero_writer_pencil, zero_extra.clone(), 7i32).unwrap();
    write_hdf5(&zero_path, zero_source.view()).unwrap();
    world.barrier();
    let zero_reader_topology = MpiTopology::<2>::new(&world, [1, size]).unwrap();
    let zero_reader_pencil = Pencil::<2, 2>::new(zero_reader_topology, [4, 5], [1, 0]).unwrap();
    let mut zero_destination =
        PencilArray::from_elem(zero_reader_pencil, zero_extra, -9i32).unwrap();
    read_hdf5(&zero_path, zero_destination.view_mut()).unwrap();
    assert!(zero_destination.as_slice().is_empty());
    world.barrier();

    macro_rules! scalar_case {
        ($name:literal, $ty:ty, $make:expr) => {
            run_hdf5_scalar_case::<_, $ty, _>(&world, &directory, $name, $make);
        };
    }
    scalar_case!("i8.h5", i8, |x, y| (x * 100 + y) as i8);
    scalar_case!("u8.h5", u8, |x, y| (x * 10 + y) as u8);
    scalar_case!("i16.h5", i16, |x, y| (x * 100 + y) as i16);
    scalar_case!("u16.h5", u16, |x, y| (x * 100 + y) as u16);
    scalar_case!("u32.h5", u32, |x, y| (x * 100 + y) as u32);
    scalar_case!("i64.h5", i64, |x, y| (x * 100 + y) as i64);
    scalar_case!("u64.h5", u64, |x, y| (x * 100 + y) as u64);
    scalar_case!("f32.h5", f32, |x, y| (x * 100 + y) as f32);
    scalar_case!("f64.h5", f64, |x, y| (x * 100 + y) as f64);
    scalar_case!("complex-f32.h5", Complex<f32>, |x, y| Complex::new(
        (x * 100 + y) as f32,
        (x as f32) - (y as f32),
    ));
    scalar_case!("complex-f64.h5", Complex<f64>, |x, y| Complex::new(
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
