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
/// Parallel HDF5 write controls. Writers to the same file must be externally
/// serialized across all jobs and communicators; append is not a transaction.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Hdf5WriteOptions {
    pub(crate) mode: MpiIoMode,
    pub(crate) hints: Vec<(String, String)>,
    pub(crate) chunks: Option<Vec<usize>>,
}
impl Hdf5ReadOptions {
    /// Select native payload mode.
    pub fn mode(mut self, mode: MpiIoMode) -> Self {
        self.mode = mode;
        self
    }
    /// Add a native MPI hint; duplicates and invalid strings fail collectively.
    pub fn hint(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.hints.push((key.into(), value.into()));
        self.hints.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        self
    }
    pub(crate) fn settings(&self) -> Hdf5Settings<'_> {
        Hdf5Settings {
            collective: self.mode == MpiIoMode::Collective,
            chunks: None,
            hints: &self.hints,
            explicit: true,
        }
    }
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
    /// Add a native MPI hint; duplicates and invalid strings fail collectively.
    pub fn hint(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.hints.push((key.into(), value.into()));
        self.hints.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        self
    }
    pub(crate) fn settings(&self) -> Hdf5Settings<'_> {
        Hdf5Settings {
            collective: self.mode == MpiIoMode::Collective,
            chunks: self.chunks.as_deref(),
            hints: &self.hints,
            explicit: true,
        }
    }
}
/// Read using explicit native HDF5 controls.
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
