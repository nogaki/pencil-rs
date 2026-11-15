use hdf5_metno_sys::h5i::hid_t;

pub(super) trait GlobalId {
    /// # Safety
    /// For `usize`, its value must be a non-null, aligned pointer to an initialized
    /// `hid_t` that remains readable for this call.
    unsafe fn resolve(&self) -> hid_t;
}

impl GlobalId for hid_t {
    unsafe fn resolve(&self) -> hid_t {
        *self
    }
}

impl GlobalId for usize {
    unsafe fn resolve(&self) -> hid_t {
        // SAFETY: upheld by this method's caller.
        unsafe { *(*self as *const hid_t) }
    }
}

/// Resolves an HDF5 global ID using the binding's declared ABI type.
///
/// This does not initialize HDF5; initialize the library before resolving an actual global.
///
/// # Safety
/// For `usize`, its value must be a non-null, aligned pointer to an initialized
/// `hid_t` that remains readable for this call.
pub(super) unsafe fn resolve<T: GlobalId>(binding: &T) -> hid_t {
    // SAFETY: upheld by this function's caller.
    unsafe { binding.resolve() }
}
