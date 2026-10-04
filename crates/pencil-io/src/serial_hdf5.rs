//! Single-process access to the existing HDF5 v1 dataset format.

use std::{ffi::CStr, path::Path};

use hdf5_metno::{Attribute, Dataset, Dataspace, Datatype, File};
use hdf5_metno_sys::{h5a, h5d, h5p::H5P_DEFAULT, h5s::H5S_MAX_RANK, h5t};

use crate::{
    COMMIT_MARKER, FORMAT_VERSION, INCOMPLETE_MARKER, IoElement, IoError, MAX_PROTOCOL_RANK,
    format::element_count,
};

const DATASET: &str = "/pencil_io_v1/data";
const VERSION: &str = "pencil_io_version";
const COMMIT: &str = "pencil_io_commit";
const N: &str = "pencil_io_n";
const TYPE: &str = "pencil_io_type";
const WIDTH: &str = "pencil_io_width";
const EXTRA_RANK: &str = "pencil_io_extra_rank";
const EXTRA: &str = "pencil_io_extra_shape";
const GLOBAL: &str = "pencil_io_global_shape";
const GRID: &str = "pencil_io_writer_grid";
const PERM: &str = "pencil_io_writer_permutation";

/// Creates a serial HDF5 file from a full, canonical row-major slice.
///
/// `values` is ordered as `[extra_shape..., global_shape...]`. The global shape
/// must have at least one axis; zero extents are allowed. The file uses the same
/// `/pencil_io_v1/data` schema as [`crate`]'s parallel HDF5 backend, with a
/// single-process writer grid and identity axis permutation. No MPI is used.
///
/// The path must not exist. Arguments and packing are validated before file
/// creation. A subsequent failure may leave a partial file: no rollback or
/// crash-atomicity is promised. A commit/flush/close failure can leave a valid
/// commit and is not a safe-retry guarantee. Same-file access must be externally
/// serialized. Requires the `hdf5` feature and an installed HDF5 library.
pub fn write_hdf5_serial<P: AsRef<Path>, T: IoElement>(
    path: P,
    global_shape: &[usize],
    extra_shape: &[usize],
    values: &[T],
) -> Result<(), IoError> {
    let (shape, bytes) = layout::<T>(global_shape, extra_shape, values.len())?;
    let mut packed = buffer(bytes)?;
    for &value in values {
        value.encode_le(&mut packed);
    }
    hdf5_metno::sync::sync(|| {
        let datatype = datatype::<T>()?;
        let space = native(Dataspace::try_new(shape), "HDF5 serial dataspace")?;
        let file = native(File::create_excl(path), "HDF5 serial create")?;
        let result = (|| {
            let group = native(file.create_group("pencil_io_v1"), "HDF5 serial group")?;
            // SAFETY: live group/type/space, static NUL-terminated name, default
            // property lists. The new dataset handle is immediately RAII-owned.
            let dataset: Dataset = native(
                unsafe {
                    hdf5_metno::from_id(h5d::H5Dcreate2(
                        group.id(),
                        c"data".as_ptr(),
                        datatype.id(),
                        space.id(),
                        H5P_DEFAULT,
                        H5P_DEFAULT,
                        H5P_DEFAULT,
                    ))
                },
                "HDF5 serial dataset create",
            )?;
            let global: Vec<_> = global_shape.iter().map(|&n| n as u64).collect();
            let extra: Vec<_> = extra_shape.iter().map(|&n| n as u64).collect();
            let permutation: Vec<_> = (0..global_shape.len() as u64).collect();
            for (name, values) in [
                (c"pencil_io_version", &[FORMAT_VERSION][..]),
                (c"pencil_io_commit", &[INCOMPLETE_MARKER][..]),
                (c"pencil_io_n", &[global_shape.len() as u64][..]),
                (c"pencil_io_type", &[T::CODE][..]),
                (c"pencil_io_width", &[T::WIDTH as u64][..]),
                (c"pencil_io_extra_rank", &[extra_shape.len() as u64][..]),
                (c"pencil_io_global_shape", global.as_slice()),
                (c"pencil_io_writer_grid", &[1][..]),
                (c"pencil_io_writer_permutation", permutation.as_slice()),
            ] {
                write_attribute(&dataset, name, values)?;
            }
            if !extra.is_empty() {
                write_attribute(&dataset, c"pencil_io_extra_shape", &extra)?;
            }
            native(file.flush(), "HDF5 serial initial metadata flush")?;
            if !packed.is_empty() {
                // SAFETY: both dataspaces have the checked shape, and packed
                // contains exactly that many canonical values of this datatype.
                let code = unsafe {
                    h5d::H5Dwrite(
                        dataset.id(),
                        datatype.id(),
                        space.id(),
                        space.id(),
                        H5P_DEFAULT,
                        packed.as_ptr().cast(),
                    )
                };
                if code < 0 {
                    return Err(IoError::WriteIncomplete {
                        stage: "HDF5 serial payload",
                    });
                }
            }
            file.flush().map_err(|_| IoError::WriteIncomplete {
                stage: "HDF5 serial payload flush",
            })?;
            let commit = native(dataset.attr(COMMIT), "HDF5 serial commit attribute")?;
            commit
                .write_raw(&[COMMIT_MARKER])
                .map_err(|_| IoError::CommitUncertain {
                    stage: "HDF5 serial commit marker",
                })?;
            file.flush().map_err(|_| IoError::CommitUncertain {
                stage: "HDF5 serial commit flush",
            })
        })();
        // All dataset/group/attribute handles have left the closure before close.
        let closed = file.close().map_err(|_| IoError::CommitUncertain {
            stage: "HDF5 serial file close",
        });
        result.and(closed)
    })
}

/// Reads a serial or parallel HDF5 v1 dataset into a full canonical slice.
///
/// The expected shapes and scalar type must match exactly; no type conversion
/// or shape inference is performed. `destination` uses row-major
/// `[extra_shape..., global_shape...]` order, independently of the writer's
/// process grid or memory permutation. No MPI is used.
///
/// Reads stage the payload and explicitly close the file before publishing it.
/// A returned error leaves the entire destination unchanged. Same-file access
/// must be externally serialized. Requires the `hdf5` feature.
pub fn read_hdf5_serial<P: AsRef<Path>, T: IoElement>(
    path: P,
    global_shape: &[usize],
    extra_shape: &[usize],
    destination: &mut [T],
) -> Result<(), IoError> {
    let (shape, bytes) = layout::<T>(global_shape, extra_shape, destination.len())?;
    let mut packed = buffer(bytes)?;
    packed.resize(bytes, 0);
    hdf5_metno::sync::sync(|| {
        let file = native(File::open(path), "HDF5 serial open")?;
        let result = (|| {
            let dataset = native(file.dataset(DATASET), "HDF5 serial dataset open")?;
            validate_metadata::<T>(&dataset, global_shape, extra_shape)?;
            let datatype = datatype::<T>()?;
            if native(dataset.dtype(), "HDF5 serial dataset type")? != datatype {
                return Err(IoError::MetadataMismatch {
                    field: "dataset datatype",
                });
            }
            let space = native(dataset.space(), "HDF5 serial dataset space")?;
            if native_shape(&space)? != shape {
                return Err(IoError::MetadataMismatch {
                    field: "dataset dimensions",
                });
            }
            if !packed.is_empty() {
                // SAFETY: the validated space contains exactly packed.len() /
                // T::WIDTH values. The memory datatype is canonical, fixed-size
                // and identical to the file type, including complex field layout.
                let code = unsafe {
                    h5d::H5Dread(
                        dataset.id(),
                        datatype.id(),
                        space.id(),
                        space.id(),
                        H5P_DEFAULT,
                        packed.as_mut_ptr().cast(),
                    )
                };
                if code < 0 {
                    return Err(IoError::Native {
                        operation: "HDF5 serial payload read",
                        code: i64::from(code),
                    });
                }
            }
            Ok(())
        })();
        let closed = native(file.close(), "HDF5 serial file close");
        result.and(closed)
    })?;
    // IoElement is sealed: decoding exact-width chunks cannot fail or allocate.
    for (value, bytes) in destination.iter_mut().zip(packed.chunks_exact(T::WIDTH)) {
        *value = T::decode_le(bytes);
    }
    Ok(())
}

fn layout<T: IoElement>(
    global: &[usize],
    extra: &[usize],
    len: usize,
) -> Result<(Vec<usize>, usize), IoError> {
    if global.is_empty()
        || global
            .len()
            .checked_add(extra.len())
            .is_none_or(|rank| rank > H5S_MAX_RANK as usize)
    {
        return Err(IoError::InvalidInput("HDF5 logical rank must be 1..=32"));
    }
    let elements = element_count(global)?
        .checked_mul(element_count(extra)?)
        .ok_or(IoError::SizeLimit {
            what: "global element count",
        })?;
    if elements != len {
        return Err(IoError::InvalidInput("slice length does not match shape"));
    }
    let bytes = len
        .checked_mul(T::WIDTH)
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or(IoError::SizeLimit {
            what: "serial payload bytes",
        })?;
    Ok((extra.iter().chain(global).copied().collect(), bytes))
}

fn buffer(bytes: usize) -> Result<Vec<u8>, IoError> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(bytes)
        .map_err(|_| IoError::AllocationFailed { requested: bytes })?;
    Ok(result)
}

fn native<T>(result: hdf5_metno::Result<T>, operation: &'static str) -> Result<T, IoError> {
    result.map_err(|_| IoError::Native {
        operation,
        code: -1,
    })
}

// These helpers are private and always run under hdf5_metno's reentrant lock,
// including when the installed serial HDF5 library was built without thread safety.
fn datatype<T: IoElement>() -> Result<Datatype, IoError> {
    use hdf5_metno::globals::*;
    let scalar = match T::CODE {
        1 => *H5T_STD_I8LE,
        2 => *H5T_STD_U8LE,
        3 => *H5T_STD_I16LE,
        4 => *H5T_STD_U16LE,
        5 => *H5T_STD_I32LE,
        6 => *H5T_STD_U32LE,
        7 => *H5T_STD_I64LE,
        8 => *H5T_STD_U64LE,
        9 | 11 => *H5T_IEEE_F32LE,
        10 | 12 => *H5T_IEEE_F64LE,
        _ => unreachable!("IoElement is sealed"),
    };
    // SAFETY: predefined scalar IDs are initialized by globals; both constructors
    // return a new owned datatype, immediately transferred to an RAII handle.
    let datatype: Datatype = native(
        unsafe {
            hdf5_metno::from_id(if T::CODE <= 10 {
                h5t::H5Tcopy(scalar)
            } else {
                h5t::H5Tcreate(h5t::H5T_class_t::H5T_COMPOUND, T::WIDTH)
            })
        },
        "HDF5 serial datatype create",
    )?;
    if T::CODE > 10 {
        for (name, offset) in [(c"r", 0), (c"i", T::WIDTH / 2)] {
            // SAFETY: live compound/scalar IDs; the two fields exactly fill the
            // compound, with static NUL-terminated names and no padding.
            if unsafe { h5t::H5Tinsert(datatype.id(), name.as_ptr(), offset, scalar) } < 0 {
                return Err(IoError::Native {
                    operation: "HDF5 serial complex datatype",
                    code: -1,
                });
            }
        }
    }
    Ok(datatype)
}

fn write_attribute(dataset: &Dataset, name: &CStr, values: &[u64]) -> Result<(), IoError> {
    let datatype = datatype::<u64>()?;
    let space = native(
        Dataspace::try_new([values.len()]),
        "HDF5 serial attribute space",
    )?;
    // SAFETY: live dataset/type/space, validated static attribute name, default
    // property lists. from_id takes ownership of the newly created attribute.
    let attribute: Attribute = native(
        unsafe {
            hdf5_metno::from_id(h5a::H5Acreate2(
                dataset.id(),
                name.as_ptr(),
                datatype.id(),
                space.id(),
                H5P_DEFAULT,
                H5P_DEFAULT,
            ))
        },
        "HDF5 serial attribute create",
    )?;
    native(attribute.write_raw(values), "HDF5 serial attribute write")
}

// hdf5-metno's shape() narrows hsize_t with `as usize`. File-controlled extents
// must use checked conversion, especially on 32-bit hosts, before allocating
// or passing a native dataspace to H5Dread/H5Aread.
fn native_shape(space: &Dataspace) -> Result<Vec<usize>, IoError> {
    // SAFETY: live dataspace. Query rank before providing a fixed-size output;
    // the same HDF5 lock remains held throughout both calls.
    let rank = unsafe { hdf5_metno_sys::h5s::H5Sget_simple_extent_ndims(space.id()) };
    if !(0..=H5S_MAX_RANK as i32).contains(&rank) {
        return Err(IoError::InvalidFile {
            reason: "HDF5 native rank",
        });
    }
    let mut dims = [0; H5S_MAX_RANK as usize];
    // SAFETY: dims has room for every validated axis; maxdims is optional.
    if unsafe {
        hdf5_metno_sys::h5s::H5Sget_simple_extent_dims(
            space.id(),
            dims.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    } != rank
    {
        return Err(IoError::InvalidFile {
            reason: "HDF5 native dimensions",
        });
    }
    dims[..rank as usize]
        .iter()
        .map(|&dim| {
            usize::try_from(dim).map_err(|_| IoError::SizeLimit {
                what: "HDF5 native dimension",
            })
        })
        .collect()
}

fn read_attribute(
    dataset: &Dataset,
    name: &str,
    expected_len: Option<usize>,
) -> Result<Vec<u64>, IoError> {
    let malformed = || IoError::MetadataMismatch {
        field: "HDF5 attribute type or shape",
    };
    let attribute = dataset.attr(name).map_err(|_| IoError::MetadataMismatch {
        field: "missing HDF5 attribute",
    })?;
    let space = native(attribute.space(), "HDF5 serial attribute space")?;
    let shape = native_shape(&space)?;
    let len = match shape.as_slice() {
        [] if space.is_scalar() => 1,
        [len] => *len,
        _ => return Err(malformed()),
    };
    if len > MAX_PROTOCOL_RANK
        || expected_len.is_some_and(|expected| len != expected)
        || native(attribute.dtype(), "HDF5 serial attribute type")? != datatype::<u64>()?
    {
        return Err(malformed());
    }
    native(attribute.read_raw::<u64>(), "HDF5 serial attribute read")
}

fn validate_metadata<T: IoElement>(
    dataset: &Dataset,
    global: &[usize],
    extra: &[usize],
) -> Result<(), IoError> {
    let scalar = |name| read_attribute(dataset, name, Some(1)).map(|values| values[0]);
    if scalar(VERSION)? != FORMAT_VERSION
        || scalar(N)? != global.len() as u64
        || scalar(TYPE)? != T::CODE
        || scalar(WIDTH)? != T::WIDTH as u64
        || scalar(EXTRA_RANK)? != extra.len() as u64
    {
        return Err(IoError::MetadataMismatch {
            field: "version, rank, or type",
        });
    }
    match scalar(COMMIT)? {
        COMMIT_MARKER => {}
        INCOMPLETE_MARKER => return Err(IoError::IncompleteFile),
        _ => {
            return Err(IoError::InvalidFile {
                reason: "HDF5 commit marker",
            });
        }
    }
    if extra.is_empty() {
        // SAFETY: live dataset and static NUL-terminated attribute name.
        match unsafe { h5a::H5Aexists(dataset.id(), c"pencil_io_extra_shape".as_ptr()) } {
            0 => {}
            code if code < 0 => {
                return Err(IoError::Native {
                    operation: "HDF5 serial attribute exists",
                    code: i64::from(code),
                });
            }
            _ => {
                return Err(IoError::MetadataMismatch {
                    field: "extra shape",
                });
            }
        }
    } else if !read_attribute(dataset, EXTRA, Some(extra.len()))?
        .iter()
        .copied()
        .eq(extra.iter().map(|&n| n as u64))
    {
        return Err(IoError::MetadataMismatch {
            field: "extra shape",
        });
    }
    let stored_global = read_attribute(dataset, GLOBAL, Some(global.len()))?;
    let grid = read_attribute(dataset, GRID, None)?;
    let perm = read_attribute(dataset, PERM, Some(global.len()))?;
    if !stored_global
        .iter()
        .copied()
        .eq(global.iter().map(|&n| n as u64))
        || grid.is_empty()
        || grid.contains(&0)
        || grid
            .iter()
            .try_fold(1u64, |n, &x| n.checked_mul(x))
            .is_none()
        || perm
            .iter()
            .enumerate()
            .any(|(i, &x)| x >= global.len() as u64 || perm[..i].contains(&x))
    {
        return Err(IoError::MetadataMismatch {
            field: "shape or writer provenance",
        });
    }
    Ok(())
}

#[cfg(test)]
#[test]
fn native_dimensions_do_not_truncate() {
    hdf5_metno::sync::sync(|| {
        for dim in [1u64 << 32, (1u64 << 32) + 1] {
            // SAFETY: one valid u64 extent; the new dataspace is RAII-owned.
            // This creates metadata only, never a multi-gigabyte payload.
            let space: Dataspace = unsafe {
                hdf5_metno::from_id(hdf5_metno_sys::h5s::H5Screate_simple(
                    1,
                    &dim,
                    std::ptr::null(),
                ))
            }
            .unwrap();
            match usize::try_from(dim) {
                Ok(expected) => assert_eq!(native_shape(&space).unwrap(), [expected]),
                Err(_) => assert!(matches!(
                    native_shape(&space),
                    Err(IoError::SizeLimit { .. })
                )),
            }
        }
    });
}
