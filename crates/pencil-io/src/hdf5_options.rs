use crate::format::IoElement;
use crate::hdf5_io::{self, Hdf5Settings};
use crate::{IoError, MpiIoMode};
use pencil_array::{PencilArrayView, PencilArrayViewMut};
use std::path::Path;

/// Parallel HDF5 read controls. All ranks participate even in independent mode.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Hdf5ReadOptions {
    pub(crate) mode: MpiIoMode,
    pub(crate) hints: Vec<(String, String)>,
}
/// Parallel HDF5 write controls.
///
/// Writers to the same file from different jobs or communicators must be
/// externally serialized. Writes are not rolled back and do not overwrite or
/// resize an existing dataset.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Hdf5WriteOptions {
    pub(crate) mode: MpiIoMode,
    pub(crate) hints: Vec<(String, String)>,
    pub(crate) chunks: Option<Vec<usize>>,
    pub(crate) shuffle: bool,
    pub(crate) deflate: Option<u8>,
}

macro_rules! hint_method {
    () => {
        /// Add a native MPI hint; duplicates and invalid strings fail collectively.
        pub fn hint(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
            self.hints.push((key.into(), value.into()));
            self.hints.sort_unstable_by(|a, b| a.0.cmp(&b.0));
            self
        }
    };
}
impl Hdf5ReadOptions {
    /// Select native payload mode.
    pub fn mode(mut self, mode: MpiIoMode) -> Self {
        self.mode = mode;
        self
    }
    hint_method!();
    pub(crate) fn settings(&self) -> Hdf5Settings<'_> {
        Hdf5Settings {
            collective: self.mode == MpiIoMode::Collective,
            chunks: None,
            hints: &self.hints,
            explicit: true,
            shuffle: false,
            deflate: None,
        }
    }
}
fn agree_hdf5_controls(
    comm: &mpi::topology::CartesianCommunicator,
    settings: Hdf5Settings<'_>,
    rank: usize,
    width: usize,
) -> Result<(), IoError> {
    let Hdf5Settings {
        collective,
        hints,
        chunks,
        shuffle,
        deflate,
        ..
    } = settings;
    let mode = if collective {
        MpiIoMode::Collective
    } else {
        MpiIoMode::Independent
    };
    let filtered = shuffle || deflate.is_some();
    let valid = chunks.is_none_or(|chunks| {
        let bytes = chunks
            .iter()
            .try_fold(width as u64, |n, &x| n.checked_mul(x as u64));
        chunks.len() == rank
            && chunks.iter().all(|&x| x > 0 && x <= u32::MAX as usize)
            && bytes.is_some_and(|n| n < (1u64 << 32))
    }) && (!filtered || chunks.is_some())
        && deflate.is_none_or(|level| level <= 9)
        && !(filtered && mode == MpiIoMode::Independent && chunks.is_some());
    crate::mpi_io::agree_phase(comm, valid, "HDF5 filter/chunk validation")?;
    let mut extra = vec![0u64; chunks.map_or(4, |c| c.len() + 4)];
    extra[0] = 2;
    extra[1] = u64::from(shuffle);
    extra[2] = deflate.map_or(0, |level| u64::from(level) + 1);
    if let Some(chunks) = chunks {
        extra[3] = chunks.len() as u64;
        for (dst, &value) in extra[4..].iter_mut().zip(chunks) {
            *dst = value as u64;
        }
    }
    crate::options::agree_controls(comm, mode, hints, &extra)
}

pub(crate) fn agree_collection_write_options(
    comm: &mpi::topology::CartesianCommunicator,
    options: &Hdf5WriteOptions,
    rank: usize,
    width: usize,
) -> Result<(), IoError> {
    agree_hdf5_controls(comm, options.settings(), rank, width)
}

pub(crate) fn agree_collection_read_options(
    comm: &mpi::topology::CartesianCommunicator,
    options: &Hdf5ReadOptions,
    rank: usize,
    width: usize,
) -> Result<(), IoError> {
    agree_hdf5_controls(comm, options.settings(), rank, width)
}

impl Hdf5WriteOptions {
    /// Select native payload mode.
    pub fn mode(mut self, mode: MpiIoMode) -> Self {
        self.mode = mode;
        self
    }
    /// Set native HDF5 chunk extents in canonical logical order.
    /// Chunks may exceed the current extent, including an empty extent.
    pub fn chunks(mut self, chunks: impl Into<Vec<usize>>) -> Self {
        self.chunks = Some(chunks.into());
        self
    }
    /// Enable or disable the HDF5 shuffle filter.
    pub fn shuffle(mut self, enabled: bool) -> Self {
        self.shuffle = enabled;
        self
    }
    /// Set the HDF5 deflate compression level.
    pub fn deflate(mut self, level: u8) -> Self {
        self.deflate = Some(level);
        self
    }
    hint_method!();
    pub(crate) fn settings(&self) -> Hdf5Settings<'_> {
        Hdf5Settings {
            collective: self.mode == MpiIoMode::Collective,
            chunks: self.chunks.as_deref(),
            hints: &self.hints,
            explicit: true,
            shuffle: self.shuffle,
            deflate: self.deflate,
        }
    }
}
/// Read using explicit native HDF5 controls.
///
/// The file is not overwritten or resized.
pub fn read_hdf5_with_options<P, T, const N: usize, const M: usize>(
    path: P,
    view: PencilArrayViewMut<'_, T, N, M>,
    options: &Hdf5ReadOptions,
) -> Result<(), IoError>
where
    P: AsRef<Path>,
    T: IoElement,
{
    hdf5_io::read_hdf5_inner_options(path, view, options.settings())
}
/// Create a dataset using explicit native HDF5 controls.
///
/// Writers to the same file from different jobs or communicators must be
/// externally serialized. A failed write is not rolled back, and this
/// operation does not overwrite or resize an existing dataset.
pub fn write_hdf5_with_options<P, T, const N: usize, const M: usize>(
    path: P,
    view: PencilArrayView<'_, T, N, M>,
    options: &Hdf5WriteOptions,
) -> Result<(), IoError>
where
    P: AsRef<Path>,
    T: IoElement,
{
    hdf5_io::write_hdf5_inner_options(path, view, options.settings())
}
