use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use mpi::collective::{CommunicatorCollectives, Root, SystemOperation};
use mpi::traits::Communicator;

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
