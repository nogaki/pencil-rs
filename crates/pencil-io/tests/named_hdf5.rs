#![cfg(feature = "parallel-hdf5")]

use std::str::FromStr;

use hdf5_metno::{File, types::FixedUnicode};
use mpi::collective::{CommunicatorCollectives, SystemOperation};
use mpi::traits::Communicator;
use pencil_array::{AxisPermutation, ExtraShape, MpiTopology, Pencil, PencilArray};
use pencil_io::{NamedIoError, append_hdf5_named, read_hdf5_named, write_hdf5_named};

mod support;
use support::{cleanup_owned_temp_dir, owned_temp_dir};

fn check<C: CommunicatorCollectives>(world: &C, name: &str, ok: bool) {
    let local = i32::from(ok);
    let mut all = 0;
    world.all_reduce_into(&local, &mut all, SystemOperation::min());
    assert_eq!(all, 1, "{name}");
}

fn make_a<C: Communicator>(
    world: &C,
    grid: [usize; 2],
    shape: [usize; 2],
) -> PencilArray<i32, 2, 2> {
    let p = Pencil::<2, 2>::new(MpiTopology::new(world, grid).unwrap(), shape, [0, 1]).unwrap();
    let mut a = PencilArray::from_elem(p, ExtraShape::new([2]).unwrap(), -1).unwrap();
    let ranges = a.view().pencil().local_ranges().clone();
    for e in 0..2 {
        for x in 0..ranges[0].len() {
            for y in 0..ranges[1].len() {
                *a.view_mut().get_local_mut(&[e], [x, y]).unwrap() =
                    (e * 10_000 + (ranges[0].start + x) * 100 + ranges[1].start + y) as i32;
            }
        }
    }
    a
}

fn assert_a(a: &PencilArray<i32, 2, 2>) {
    let v = a.view();
    let r = v.pencil().local_ranges().clone();
    for e in 0..2 {
        for x in 0..r[0].len() {
            for y in 0..r[1].len() {
                assert_eq!(
                    v.get_local(&[e], [x, y]),
                    Some(&((e * 10_000 + (r[0].start + x) * 100 + r[1].start + y) as i32))
                );
            }
        }
    }
}

fn root_status<C, F>(world: &C, f: F) -> Result<(), String>
where
    C: CommunicatorCollectives,
    F: FnOnce() -> Result<(), String>,
{
    let result = if world.rank() == 0 { f() } else { Ok(()) };
    let ok = i32::from(result.is_ok());
    let mut all = 0;
    world.all_reduce_into(&ok, &mut all, SystemOperation::min());
    assert_eq!(all, 1, "root HDF5 oracle: {result:?}");
    result
}

#[test]
fn named_hdf5_matrix() {
    let universe = mpi::initialize().expect("MPI initialize once");
    let world = universe.world();
    let size = world.size() as usize;
    let dir = owned_temp_dir(&world, "pencil-io-named-hdf5");
    let path = dir.join("named.h5");

    let a = make_a(&world, [size, 1], [4, 5]);
    check(
        &world,
        "write A",
        write_hdf5_named(&path, "A", a.view()).is_ok(),
    );
    for bad in ["", "bad\0name", &"x".repeat(1025)] {
        check(
            &world,
            "invalid name",
            matches!(
                append_hdf5_named(&path, bad, a.view()),
                Err(NamedIoError::InvalidName)
            ),
        );
    }
    let mut missing =
        PencilArray::from_elem(a.pencil().clone(), ExtraShape::new([2]).unwrap(), -77).unwrap();
    let old = missing.as_slice().to_vec();
    check(
        &world,
        "missing unchanged",
        matches!(
            read_hdf5_named(&path, "missing", missing.view_mut()),
            Err(NamedIoError::NotFound)
        ) && missing.as_slice() == old,
    );

    let type_array =
        PencilArray::from_elem(a.pencil().clone(), ExtraShape::new([2]).unwrap(), 1i64).unwrap();
    let layout_array = PencilArray::from_elem(
        Pencil::<2, 2>::new(a.pencil().topology().clone(), [4, 6], [0, 1]).unwrap(),
        ExtraShape::new([2]).unwrap(),
        1i32,
    )
    .unwrap();
    if size > 1 {
        let mismatch = if world.rank() == 0 {
            pencil_io::write_hdf5(&path, a.view()).map_err(NamedIoError::Io)
        } else {
            write_hdf5_named(&path, "A", a.view())
        };
        check(&world, "old/new mismatch", mismatch.is_err());
        let name = if world.rank() == 0 { "same" } else { "diff" };
        check(
            &world,
            "name mismatch",
            append_hdf5_named(&path, name, a.view()).is_err(),
        );
        let type_result = if world.rank() == 0 {
            write_hdf5_named(&path, "type-mismatch", a.view())
        } else {
            write_hdf5_named(&path, "type-mismatch", type_array.view())
        };
        check(&world, "per-rank type mismatch", type_result.is_err());
        let layout_result = if world.rank() == 0 {
            write_hdf5_named(&path, "layout-mismatch", a.view())
        } else {
            write_hdf5_named(&path, "layout-mismatch", layout_array.view())
        };
        check(&world, "per-rank layout mismatch", layout_result.is_err());
        let op = if world.rank() == 0 {
            append_hdf5_named(&path, "op-mismatch", a.view())
        } else {
            read_hdf5_named(&path, "op-mismatch", missing.view_mut())
        };
        check(&world, "append/read operation mismatch", op.is_err());
    }

    let b_p =
        Pencil::<2, 2>::new(MpiTopology::new(&world, [size, 1]).unwrap(), [3, 4], [0, 1]).unwrap();
    let b = PencilArray::from_elem(b_p, ExtraShape::scalar(), 3.25f64).unwrap();
    check(
        &world,
        "append B distinct dtype and shape",
        append_hdf5_named(&path, "B", b.view()).is_ok(),
    );
    root_status(&world, || {
        let file = File::open(&path).map_err(|e| e.to_string())?;
        let group = file
            .group("/pencil_io_named_v1")
            .map_err(|e| e.to_string())?;
        let ds = group.dataset("42").map_err(|e| e.to_string())?;
        if ds.shape() != [3, 4]
            || ds.read_raw::<f64>().map_err(|e| e.to_string())? != vec![3.25; 12]
        {
            return Err("B payload/shape".into());
        }
        if ds
            .attr("pencil_io_type")
            .map_err(|e| e.to_string())?
            .read_scalar::<u64>()
            .map_err(|e| e.to_string())?
            != 10
        {
            return Err("B dtype metadata".into());
        }
        if ds
            .attr("pencil_io_width")
            .map_err(|e| e.to_string())?
            .read_scalar::<u64>()
            .map_err(|e| e.to_string())?
            != 8
        {
            return Err("B width metadata".into());
        }
        let a_ds = group.dataset("41").map_err(|e| e.to_string())?;
        for dataset in [&a_ds, &ds] {
            assert_eq!(
                dataset
                    .attr("pencil_io_commit")
                    .map_err(|e| e.to_string())?
                    .read_scalar::<u64>()
                    .map_err(|e| e.to_string())?,
                0x434f_4d4d_4954_5445
            );
            assert_eq!(
                dataset
                    .attr("pencil_io_version")
                    .map_err(|e| e.to_string())?
                    .read_scalar::<u64>()
                    .map_err(|e| e.to_string())?,
                1
            );
        }
        let expected_a: Vec<i32> = (0..2)
            .flat_map(|e| (0..4).flat_map(move |x| (0..5).map(move |y| e * 10_000 + x * 100 + y)))
            .collect();
        if a_ds.shape() != [2, 4, 5]
            || a_ds.read_raw::<i32>().map_err(|e| e.to_string())? != expected_a
        {
            return Err("A payload after B append".into());
        }
        if a_ds
            .attr("pencil_io_original_name")
            .map_err(|e| e.to_string())?
            .read_scalar::<FixedUnicode<2>>()
            .map_err(|e| e.to_string())?
            .as_str()
            != "A"
        {
            return Err("A name metadata".into());
        }
        file.close().map_err(|e| e.to_string())
    })
    .unwrap();

    let reader_p = Pencil::<2, 2>::new_permuted(
        MpiTopology::new(&world, [1, size]).unwrap(),
        [4, 5],
        [1, 0],
        AxisPermutation::new([1, 0]).unwrap(),
    )
    .unwrap();
    let mut ar = PencilArray::from_elem(reader_p, ExtraShape::new([2]).unwrap(), -1).unwrap();
    check(
        &world,
        "read A changed grid and AxisPermutation",
        read_hdf5_named(&path, "A", ar.view_mut()).is_ok(),
    );
    assert_a(&ar);

    let brp = Pencil::<2, 2>::new_permuted(
        MpiTopology::new(&world, [1, size]).unwrap(),
        [3, 4],
        [1, 0],
        AxisPermutation::new([1, 0]).unwrap(),
    )
    .unwrap();
    let mut br = PencilArray::from_elem(brp, ExtraShape::scalar(), -9f64).unwrap();
    read_hdf5_named(&path, "B", br.view_mut()).unwrap();
    assert!(br.as_slice().iter().all(|&v| v == 3.25));
    append_hdf5_named(&path, "α/温度", a.view()).unwrap();
    read_hdf5_named(&path, "α/温度", ar.view_mut()).unwrap();
    assert_a(&ar);
    let before_duplicate = std::fs::read(&path).unwrap();
    assert!(matches!(
        append_hdf5_named(&path, "B", b.view()),
        Err(NamedIoError::DuplicateName)
    ));
    world.barrier();
    assert_eq!(std::fs::read(&path).unwrap(), before_duplicate);
    root_status(&world, || {
        let file = File::open_rw(&path).map_err(|e| e.to_string())?;
        file.dataset("/pencil_io_named_v1/42")
            .map_err(|e| e.to_string())?
            .attr("pencil_io_commit")
            .map_err(|e| e.to_string())?
            .write_scalar(&0u64)
            .map_err(|e| e.to_string())?;
        file.close().map_err(|e| e.to_string())
    })
    .unwrap();
    let before_b = br.as_slice().to_vec();
    assert!(read_hdf5_named(&path, "B", br.view_mut()).is_err());
    assert_eq!(br.as_slice(), before_b);
    read_hdf5_named(&path, "A", ar.view_mut()).unwrap();
    assert_a(&ar);

    // Independently constructed malformed name attributes must never size an unsafe read.
    for vector in [false, true] {
        let malformed_path = dir.join(format!("bad-name-{vector}.h5"));
        root_status(&world, || {
            let file = File::create(&malformed_path).map_err(|e| e.to_string())?;
            let group = file
                .create_group("pencil_io_named_v1")
                .map_err(|e| e.to_string())?;
            let ds = group
                .new_dataset::<i32>()
                .shape([2, 4, 5])
                .create("41")
                .map_err(|e| e.to_string())?;
            if vector {
                ds.new_attr::<FixedUnicode<2>>()
                    .shape([2])
                    .create("pencil_io_original_name")
                    .map_err(|e| e.to_string())?
                    .write_raw(&[
                        FixedUnicode::<2>::from_str("A").unwrap(),
                        FixedUnicode::<2>::from_str("A").unwrap(),
                    ])
                    .map_err(|e| e.to_string())?;
            } else {
                ds.new_attr::<hdf5_metno::types::VarLenUnicode>()
                    .shape(())
                    .create("pencil_io_original_name")
                    .map_err(|e| e.to_string())?
                    .write_scalar(&hdf5_metno::types::VarLenUnicode::from_str("A").unwrap())
                    .map_err(|e| e.to_string())?;
            }
            file.close().map_err(|e| e.to_string())
        })
        .unwrap();
        let before = ar.as_slice().to_vec();
        assert!(read_hdf5_named(&malformed_path, "A", ar.view_mut()).is_err());
        assert_eq!(ar.as_slice(), before);
    }

    let tamper = root_status(&world, || {
        let file = File::open_rw(&path).map_err(|e| e.to_string())?;
        let ds = file
            .group("/pencil_io_named_v1")
            .map_err(|e| e.to_string())?
            .dataset("41")
            .map_err(|e| e.to_string())?;
        let attr = ds
            .attr("pencil_io_original_name")
            .map_err(|e| e.to_string())?;
        attr.as_writer()
            .write_scalar(&FixedUnicode::<2>::from_str("X").map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())
    });
    tamper.unwrap();
    let before = ar.as_slice().to_vec();
    check(
        &world,
        "tampered original name fails unchanged",
        read_hdf5_named(&path, "A", ar.view_mut()).is_err() && ar.as_slice() == before,
    );

    let z = PencilArray::from_elem(
        Pencil::<2, 2>::new(MpiTopology::new(&world, [1, size]).unwrap(), [2, 2], [1, 0]).unwrap(),
        ExtraShape::new([0]).unwrap(),
        0i32,
    )
    .unwrap();
    check(
        &world,
        "empty extras and ranks",
        write_hdf5_named(dir.join("empty.h5"), "zero", z.view()).is_ok(),
    );
    let mut zr = PencilArray::from_elem(
        Pencil::<2, 2>::new(MpiTopology::new(&world, [size, 1]).unwrap(), [2, 2], [0, 1]).unwrap(),
        ExtraShape::new([0]).unwrap(),
        -1i32,
    )
    .unwrap();
    read_hdf5_named(dir.join("empty.h5"), "zero", zr.view_mut()).unwrap();
    assert!(append_hdf5_named(dir.join("missing.h5"), "A", a.view()).is_err());
    let v1 = dir.join("v1.h5");
    pencil_io::write_hdf5(&v1, a.view()).unwrap();
    let before_v1 = std::fs::read(&v1).unwrap();
    assert!(append_hdf5_named(&v1, "A", a.view()).is_err());
    assert_eq!(std::fs::read(&v1).unwrap(), before_v1);
    world.barrier();
    cleanup_owned_temp_dir(&world, &dir);
    world.barrier();
    if world.rank() == 0 {
        eprintln!("\nNAMED_HDF5_PASSED ranks={}", world.size());
    }
    world.barrier();
}
