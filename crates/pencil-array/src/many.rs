use std::sync::Arc;

use thiserror::Error;

use crate::{
    ArrayError, ExtraShape, Pencil, PencilArrayView, PencilArrayViewMut, checked::checked_product,
};

#[derive(Debug, Error)]
/// An error from a transactional [`ManyPencilArray::overwrite_with`] operation.
pub enum OverwriteError<E> {
    /// The target layout or array state was invalid.
    #[error(transparent)]
    Array(#[from] ArrayError),

    /// The writer returned its own error after the array had been poisoned.
    #[error("overwrite closure failed")]
    Writer(E),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LayoutState {
    Valid(usize),
    Poisoned,
}

#[derive(Debug)]
/// One local buffer registered for several compatible pencil layouts.
///
/// Only the active layout is observable. A layout changes only after a full,
/// successful [`Self::overwrite_with`] operation.
pub struct ManyPencilArray<T, const N: usize, const M: usize> {
    pencils: Box<[Arc<Pencil<N, M>>]>,
    extra_shape: ExtraShape,
    storage: Vec<T>,
    state: LayoutState,
}

impl<T, const N: usize, const M: usize> ManyPencilArray<T, N, M> {
    /// Takes ownership of a buffer sized for the largest registered local layout.
    ///
    /// The registry must be nonempty, contain distinct layouts with the same
    /// topology and global shape, and contain `active`.
    pub fn from_vec(
        pencils: impl Into<Box<[Arc<Pencil<N, M>>]>>,
        active: usize,
        extra_shape: ExtraShape,
        storage: Vec<T>,
    ) -> Result<Self, ArrayError> {
        let pencils = pencils.into();
        let required = validate_registry(&pencils, active, &extra_shape)?;
        if storage.len() != required {
            return Err(ArrayError::StorageLengthMismatch {
                required,
                actual: storage.len(),
            });
        }

        Ok(Self {
            pencils,
            extra_shape,
            storage,
            state: LayoutState::Valid(active),
        })
    }

    /// Allocates the largest registered local layout and fills it with `value`.
    pub fn from_elem(
        pencils: impl Into<Box<[Arc<Pencil<N, M>>]>>,
        active: usize,
        extra_shape: ExtraShape,
        value: T,
    ) -> Result<Self, ArrayError>
    where
        T: Clone,
    {
        let pencils = pencils.into();
        let required = validate_registry(&pencils, active, &extra_shape)?;
        let mut storage = Vec::new();
        storage
            .try_reserve_exact(required)
            .map_err(|_| ArrayError::AllocationFailed { required })?;
        storage.resize(required, value);

        Ok(Self {
            pencils,
            extra_shape,
            storage,
            state: LayoutState::Valid(active),
        })
    }

    /// Returns every registered pencil in registration order.
    pub fn pencils(&self) -> &[Arc<Pencil<N, M>>] {
        &self.pencils
    }

    /// Returns the active pencil, or [`ArrayError::Poisoned`] after an incomplete write.
    pub fn active_pencil(&self) -> Result<&Pencil<N, M>, ArrayError> {
        Ok(self.pencils[self.active_index()?].as_ref())
    }

    /// Returns the extra shape shared by every registered layout.
    pub fn extra_shape(&self) -> &ExtraShape {
        &self.extra_shape
    }

    /// Returns a read-only view of the active layout's buffer prefix.
    pub fn active_view(&self) -> Result<PencilArrayView<'_, T, N, M>, ArrayError> {
        let index = self.active_index()?;
        let used_len = self.layout_len(index)?;
        PencilArrayView::new(
            self.pencils[index].as_ref(),
            &self.extra_shape,
            &self.storage[..used_len],
        )
    }

    /// Returns an exclusive view of the active layout's buffer prefix.
    pub fn active_view_mut(&mut self) -> Result<PencilArrayViewMut<'_, T, N, M>, ArrayError> {
        let index = self.active_index()?;
        let used_len = self.layout_len(index)?;
        PencilArrayViewMut::new(
            self.pencils[index].as_ref(),
            &self.extra_shape,
            &mut self.storage[..used_len],
        )
    }

    /// Replaces the complete target view and makes its registered layout active.
    ///
    /// `target` must match a registered layout. Before `write` runs, the array is
    /// marked poisoned. The writer must assign meaningful values to **every**
    /// element of the supplied target view before returning `Ok`. If it returns
    /// `Err` or unwinds, active access remains poisoned because the buffer may be
    /// only partly written. A later complete, successful overwrite is the recovery
    /// path, including when the array was already poisoned.
    ///
    /// This operation only controls ownership and validity of the shared buffer;
    /// it performs no data redistribution or transpose by itself.
    pub fn overwrite_with<F, E>(
        &mut self,
        target: &Pencil<N, M>,
        write: F,
    ) -> Result<(), OverwriteError<E>>
    where
        F: FnOnce(PencilArrayViewMut<'_, T, N, M>) -> Result<(), E>,
    {
        let target_index = self
            .find_layout(target)
            .ok_or(ArrayError::IncompatiblePencils)?;
        let mut guard = LayoutWriteGuard::new(self);
        write(guard.view_mut(target_index)?).map_err(OverwriteError::Writer)?;
        guard.commit(target_index)?;
        Ok(())
    }

    pub(crate) fn active_index(&self) -> Result<usize, ArrayError> {
        match self.state {
            LayoutState::Valid(index) if index < self.pencils.len() => Ok(index),
            LayoutState::Valid(index) => Err(ArrayError::InvalidActiveLayout {
                index,
                layout_count: self.pencils.len(),
            }),
            LayoutState::Poisoned => Err(ArrayError::Poisoned),
        }
    }

    pub(crate) fn find_layout(&self, pencil: &Pencil<N, M>) -> Option<usize> {
        self.pencils
            .iter()
            .position(|registered| registered.same_layout(pencil))
    }

    #[allow(dead_code)]
    pub(crate) fn begin_in_place_write(
        &mut self,
    ) -> Result<LayoutWriteGuard<'_, T, N, M>, ArrayError> {
        self.active_index()?;
        Ok(LayoutWriteGuard::new(self))
    }

    fn layout_len(&self, index: usize) -> Result<usize, ArrayError> {
        let pencil = self
            .pencils
            .get(index)
            .ok_or(ArrayError::InvalidActiveLayout {
                index,
                layout_count: self.pencils.len(),
            })?;
        Ok(checked_product(&[
            pencil.local_len(),
            self.extra_shape.element_count(),
        ])?)
    }
}

#[derive(Debug)]
pub(crate) struct LayoutWriteGuard<'a, T, const N: usize, const M: usize> {
    array: &'a mut ManyPencilArray<T, N, M>,
}

impl<'a, T, const N: usize, const M: usize> LayoutWriteGuard<'a, T, N, M> {
    fn new(array: &'a mut ManyPencilArray<T, N, M>) -> Self {
        array.state = LayoutState::Poisoned;
        Self { array }
    }

    #[allow(dead_code)]
    pub(crate) fn storage_mut(&mut self) -> &mut [T] {
        &mut self.array.storage
    }

    pub(crate) fn commit(self, index: usize) -> Result<(), ArrayError> {
        if index >= self.array.pencils.len() {
            return Err(ArrayError::InvalidActiveLayout {
                index,
                layout_count: self.array.pencils.len(),
            });
        }
        self.array.state = LayoutState::Valid(index);
        Ok(())
    }

    fn view_mut(&mut self, index: usize) -> Result<PencilArrayViewMut<'_, T, N, M>, ArrayError> {
        let used_len = self.array.layout_len(index)?;
        PencilArrayViewMut::new(
            self.array.pencils[index].as_ref(),
            &self.array.extra_shape,
            &mut self.array.storage[..used_len],
        )
    }
}

fn validate_registry<const N: usize, const M: usize>(
    pencils: &[Arc<Pencil<N, M>>],
    active: usize,
    extra_shape: &ExtraShape,
) -> Result<usize, ArrayError> {
    let Some(first) = pencils.first() else {
        return Err(ArrayError::IncompatiblePencils);
    };
    if active >= pencils.len() {
        return Err(ArrayError::InvalidActiveLayout {
            index: active,
            layout_count: pencils.len(),
        });
    }

    if pencils
        .iter()
        .skip(1)
        .any(|pencil| !first.same_topology(pencil) || first.global_shape() != pencil.global_shape())
    {
        return Err(ArrayError::IncompatiblePencils);
    }
    for (index, pencil) in pencils.iter().enumerate() {
        if pencils[..index]
            .iter()
            .any(|registered| registered.same_layout(pencil))
        {
            return Err(ArrayError::IncompatiblePencils);
        }
    }

    let max_local_len = pencils
        .iter()
        .map(|pencil| pencil.local_len())
        .max()
        .expect("non-empty registry was checked");
    Ok(checked_product(&[
        max_local_len,
        extra_shape.element_count(),
    ])?)
}
