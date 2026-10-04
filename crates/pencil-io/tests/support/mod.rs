use std::fmt::Debug;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use mpi::collective::{CommunicatorCollectives, Root, SystemOperation};
use mpi::traits::Communicator;
use pencil_array::{
    AxisPermutation, ExtraShape, MpiTopology, Pencil, PencilArray, PencilArrayView,
    PencilArrayViewMut,
};

// Each test binary uses only the helpers it needs.
#[allow(dead_code)]
pub(crate) fn root_status<C, F>(world: &C, operation: F)
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

#[allow(dead_code)]
pub(crate) fn root_read<C>(world: &C, path: &Path) -> Vec<u8>
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

#[allow(dead_code)]
pub(crate) fn reset_file<C>(world: &C, path: &Path)
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

#[allow(dead_code)]
pub(crate) fn run_scalar_case<C, T, F>(
    world: &C,
    directory: &Path,
    name: &str,
    make_value: F,
    write: impl FnOnce(&Path, PencilArrayView<'_, T, 2, 2>),
    read: impl FnOnce(&Path, PencilArrayViewMut<'_, T, 2, 2>),
) where
    C: Communicator + CommunicatorCollectives,
    T: Clone + Debug + PartialEq,
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
    write(&path, source.view());
    world.barrier();
    let reader_topology = MpiTopology::<2>::new(world, [1, size]).unwrap();
    let reader_pencil = Pencil::<2, 2>::new(reader_topology, [4, 5], [1, 0]).unwrap();
    let mut destination =
        PencilArray::from_elem(reader_pencil, ExtraShape::scalar(), make_value(0, 0)).unwrap();
    read(&path, destination.view_mut());
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

pub(crate) fn owned_temp_dir<C>(world: &C, prefix: &str) -> PathBuf
where
    C: Communicator + CommunicatorCollectives,
{
    let setup = if world.rank() == 0 {
        create_unique_dir(prefix)
    } else {
        Ok(String::new())
    };
    let mut status = i32::from(setup.is_ok());
    world.process_at_rank(0).broadcast_into(&mut status);
    let mut all_ok = 0;
    world.all_reduce_into(&status, &mut all_ok, SystemOperation::min());
    if all_ok != 1 {
        panic!(
            "owned test directory setup failed: {}",
            setup.err().unwrap_or_default()
        );
    }

    let mut path = if world.rank() == 0 {
        setup
            .expect("directory setup agreement established")
            .into_bytes()
    } else {
        Vec::new()
    };
    let mut length = i32::try_from(path.len()).expect("temporary path fits MPI count");
    world.process_at_rank(0).broadcast_into(&mut length);
    path.resize(
        usize::try_from(length).expect("temporary path length is non-negative"),
        0,
    );
    world.process_at_rank(0).broadcast_into(&mut path[..]);
    PathBuf::from(String::from_utf8(path).expect("temporary path is UTF-8"))
}

pub(crate) fn cleanup_owned_temp_dir<C>(world: &C, path: &Path)
where
    C: Communicator + CommunicatorCollectives,
{
    world.barrier();
    let cleanup = if world.rank() == 0 {
        std::fs::remove_dir_all(path).map_err(|error| error.to_string())
    } else {
        Ok(())
    };
    let mut status = i32::from(cleanup.is_ok());
    world.process_at_rank(0).broadcast_into(&mut status);
    let mut all_ok = 0;
    world.all_reduce_into(&status, &mut all_ok, SystemOperation::min());
    if all_ok != 1 {
        panic!(
            "owned test directory cleanup failed: {}",
            cleanup.err().unwrap_or_default()
        );
    }
    world.barrier();
}

fn create_unique_dir(prefix: &str) -> Result<String, String> {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| error.to_string())?
        .as_nanos();
    let pid = std::process::id();
    let mut attempt = 0u64;
    loop {
        let path = std::env::temp_dir().join(format!("{prefix}-{nanos}-{pid}-{attempt}"));
        match std::fs::create_dir(&path) {
            Ok(()) => {
                return path.to_str().map(str::to_owned).ok_or_else(|| {
                    let _ = std::fs::remove_dir_all(&path);
                    "temporary path is not valid UTF-8".to_owned()
                });
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                attempt = attempt.saturating_add(1);
            }
            Err(error) => return Err(error.to_string()),
        }
    }
}
