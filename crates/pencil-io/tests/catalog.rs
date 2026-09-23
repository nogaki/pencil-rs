use std::path::{Path, PathBuf};

use mpi::collective::CommunicatorCollectives;
use mpi::traits::Communicator;
use num_complex::{Complex32, Complex64};
use pencil_array::{ExtraShape, MpiTopology, Pencil, PencilArray};
use pencil_io::{ScalarType, append_mpi_named, write_mpi, write_mpi_collection, write_mpi_named};
use pencil_io::{read_mpi_catalog, read_mpi_named, read_mpi_named_catalog};

mod support;
use support::{cleanup_owned_temp_dir, owned_temp_dir};

fn barrier_file<C: CommunicatorCollectives>(c: &C, path: &Path, bytes: &[u8]) {
    if c.rank() == 0 {
        std::fs::write(path, bytes).unwrap();
    }
    c.barrier();
}

fn array_f32<C: Communicator>(
    c: &C,
    shape: [usize; 2],
    extra: [usize; 1],
) -> PencilArray<f32, 2, 2> {
    let p = Pencil::<2, 2>::new(
        MpiTopology::new(c, [c.size() as usize, 1]).unwrap(),
        shape,
        [0, 1],
    )
    .unwrap();
    PencilArray::from_elem(p, ExtraShape::new(extra).unwrap(), 1.25).unwrap()
}
fn array_f64<C: Communicator>(c: &C, shape: [usize; 2]) -> PencilArray<f64, 2, 2> {
    let p = Pencil::<2, 2>::new(
        MpiTopology::new(c, [c.size() as usize, 1]).unwrap(),
        shape,
        [0, 1],
    )
    .unwrap();
    PencilArray::from_elem(p, ExtraShape::scalar(), 2.5).unwrap()
}
fn u64_at(b: &[u8], p: usize) -> u64 {
    u64::from_le_bytes(b[p..p + 8].try_into().unwrap())
}

// Keep these fixtures independent of the array writers: catalog decoding must
// be tested against the on-disk codes, not only against Rust type dispatch.
fn native_fixture(code: u64, width: usize) -> Vec<u8> {
    let n = 2usize;
    let er = 2usize;
    let grid = [1u64];
    let global = [4u64, 5];
    let extra = [3u64, 2];
    let perm = [0u64, 1];
    let header_len = 96 + (n + er + grid.len() + n) * 8;
    let payload = global.iter().chain(extra.iter()).product::<u64>() as usize * width;
    let mut h = vec![0u8; 96];
    h[..8].copy_from_slice(b"PENCILIO");
    for (p, x) in [
        (8, 1),
        (16, header_len as u64),
        (24, 0x434f_4d4d_4954_5445u64),
        (32, n as u64),
        (40, code),
        (48, width as u64),
        (56, er as u64),
        (64, header_len as u64),
        (72, payload as u64),
        (80, grid.len() as u64),
        (88, 0),
    ] {
        h[p..p + 8].copy_from_slice(&x.to_le_bytes());
    }
    for x in global.into_iter().chain(extra).chain(grid).chain(perm) {
        h.extend_from_slice(&x.to_le_bytes());
    }
    h.extend(vec![0; payload]);
    h
}

fn named_fixture(code: u64, width: usize) -> Vec<u8> {
    let name = b"scalar";
    let mut meta = Vec::new();
    for x in [2u64, 2, 4, 5, 3, 2] {
        meta.extend_from_slice(&x.to_le_bytes());
    }
    let payload = 4 * 5 * 3 * 2 * width;
    let total = 72 + name.len() + meta.len() + payload + 8;
    let mut out = Vec::new();
    out.extend_from_slice(b"PIONAM02");
    for x in [2u64, 32, 0] {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out.extend_from_slice(b"PIOREC02");
    for x in [
        2,
        name.len() as u64,
        meta.len() as u64,
        payload as u64,
        total as u64,
        code,
        width as u64,
        0,
    ] {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out.extend_from_slice(name);
    out.extend_from_slice(&meta);
    out.extend(vec![0; payload]);
    out.extend_from_slice(&0x434f_4d4d_4954_5445u64.to_le_bytes());
    out
}

#[test]
fn catalog_integration() {
    let universe = mpi::initialize().expect("MPI must initialize once");
    let world = universe.world();
    let topo = MpiTopology::<2>::new(&world, [world.size() as usize, 1]).unwrap();
    let comm = topo.communicator(); // one explicit Cartesian communicator for every catalog call
    let dir = owned_temp_dir(&world, "pencil-io-catalog");
    let p: PathBuf = dir.join("old.pio");

    // v1, f32, and the extra extent are reported verbatim (not flattened).
    let a = array_f32(&world, [4, 5], [3]);
    write_mpi(&p, a.view()).unwrap();
    let v = read_mpi_catalog(&p, comm).unwrap();
    assert_eq!(v.len(), 1);
    assert_eq!(v[0].scalar_type(), ScalarType::F32);
    assert_eq!(v[0].global_shape(), &[4, 5]);
    assert_eq!(v[0].extra_shape(), &[3]);
    assert_eq!(v[0].provenance().len(), 32);

    // Exercise every stable scalar code and width with bytes written directly.
    let scalars = [
        (ScalarType::I8, 1, 1),
        (ScalarType::U8, 2, 1),
        (ScalarType::I16, 3, 2),
        (ScalarType::U16, 4, 2),
        (ScalarType::I32, 5, 4),
        (ScalarType::U32, 6, 4),
        (ScalarType::I64, 7, 8),
        (ScalarType::U64, 8, 8),
        (ScalarType::F32, 9, 4),
        (ScalarType::F64, 10, 8),
        (ScalarType::ComplexF32, 11, 8),
        (ScalarType::ComplexF64, 12, 16),
    ];
    for (st, code, width) in scalars {
        assert_eq!(st.code(), code);
        assert_eq!(st.width(), width);
        let vp = dir.join(format!("scalar-{}.pio", code));
        barrier_file(comm, &vp, &native_fixture(code, width));
        let got = read_mpi_catalog(&vp, comm).unwrap();
        assert_eq!(got[0].scalar_type(), st);
        assert_eq!(got[0].scalar_type().code(), code);
        assert_eq!(got[0].scalar_type().width(), width);
        assert_eq!(got[0].global_shape(), &[4, 5]);
        assert_eq!(got[0].extra_shape(), &[3, 2]);
        let np = dir.join(format!("scalar-{}.named", code));
        barrier_file(comm, &np, &named_fixture(code, width));
        let named = read_mpi_named_catalog(&np, comm).unwrap();
        assert_eq!(named[0].scalar_type(), st);
        assert_eq!(named[0].scalar_type().code(), code);
        assert_eq!(named[0].scalar_type().width(), width);
        assert_eq!(named[0].global_shape(), &[4, 5]);
        assert_eq!(named[0].extra_shape(), &[3, 2]);
    }
    for (label, code, width) in [("code", 99, 4), ("width", 9, 8)] {
        let vp = dir.join(format!("bad-scalar-{label}.pio"));
        barrier_file(comm, &vp, &native_fixture(code, width));
        assert!(read_mpi_catalog(&vp, comm).is_err());
        let np = dir.join(format!("bad-scalar-{label}.named"));
        barrier_file(comm, &np, &named_fixture(code, width));
        assert!(read_mpi_named_catalog(&np, comm).is_err());
    }

    // Collections keep the collection extra shape; the member count is not inferred.
    let collection_topo = MpiTopology::new(&world, [world.size() as usize, 1]).unwrap();
    let collection_pencil = Pencil::<2, 2>::new(collection_topo.clone(), [4, 5], [0, 1]).unwrap();
    let members: Vec<_> = (0..2)
        .map(|i| {
            PencilArray::from_elem(collection_pencil.clone(), ExtraShape::new([3]).unwrap(), i)
                .unwrap()
        })
        .collect();
    let collection = dir.join("collection.pio");
    let views: Vec<_> = members.iter().map(|x| x.view()).collect();
    write_mpi_collection(
        &collection,
        collection_pencil.topology().communicator(),
        &views,
    )
    .unwrap();
    let cv = read_mpi_catalog(&collection, collection_pencil.topology().communicator()).unwrap();
    assert_eq!(cv.len(), 1);
    assert_eq!(cv[0].scalar_type(), ScalarType::I32);
    assert_eq!(cv[0].global_shape(), &[4, 5]);
    assert_eq!(cv[0].extra_shape(), &[2, 3]);

    // Complex scalar widths and codes are catalogued without flattening.
    let cp = Pencil::<2, 2>::new(
        MpiTopology::new(&world, [world.size() as usize, 1]).unwrap(),
        [2, 2],
        [0, 1],
    )
    .unwrap();
    let c32 =
        PencilArray::from_elem(cp.clone(), ExtraShape::scalar(), Complex32::new(1.0, 2.0)).unwrap();
    let c64 =
        PencilArray::from_elem(cp.clone(), ExtraShape::scalar(), Complex64::new(3.0, 4.0)).unwrap();
    let c32_path = dir.join("complex32.pio");
    let c64_path = dir.join("complex64.pio");
    write_mpi(&c32_path, c32.view()).unwrap();
    write_mpi(&c64_path, c64.view()).unwrap();
    assert_eq!(
        read_mpi_catalog(&c32_path, cp.topology().communicator()).unwrap()[0].scalar_type(),
        ScalarType::ComplexF32
    );
    assert_eq!(
        read_mpi_catalog(&c64_path, cp.topology().communicator()).unwrap()[0].scalar_type(),
        ScalarType::ComplexF64
    );
    assert_eq!(
        read_mpi_catalog(&c32_path, cp.topology().communicator()).unwrap()[0].extra_shape(),
        &[]
    );

    // Named keys are UTF-8 and may contain slashes; multiple scalar types work.
    let named = dir.join("named.pio");
    write_mpi_named(&named, "日本/速度", a.view()).unwrap();
    let b = array_f64(&world, [2, 3]);
    append_mpi_named(&named, "pressure/f64", b.view()).unwrap();
    let nv = read_mpi_named_catalog(&named, comm).unwrap();
    assert_eq!(
        nv.iter().map(|x| x.name()).collect::<Vec<_>>(),
        vec![Some("日本/速度"), Some("pressure/f64")]
    );
    assert_eq!(nv[0].scalar_type(), ScalarType::F32);
    assert_eq!(nv[1].scalar_type(), ScalarType::F64);

    let long_name = format!("{}/{}é", "a".repeat(510), "b".repeat(511));
    assert_eq!(long_name.len(), 1024);
    let long_path = dir.join("long-name.pio");
    write_mpi_named(&long_path, &long_name, a.view()).unwrap();
    assert_eq!(
        read_mpi_named_catalog(&long_path, comm).unwrap()[0].name(),
        Some(long_name.as_str())
    );

    // A committed v1 prefix followed by garbage is rejected by the strict catalog.
    let mut bad = std::fs::read(&p).unwrap();
    bad.extend_from_slice(b"uncommitted tail");
    let bad_path = dir.join("bad-tail.pio");
    barrier_file(comm, &bad_path, &bad);
    assert!(read_mpi_catalog(&bad_path, comm).is_err());

    // Every v1 fixed-header offset is validated, including reserved bytes.
    for (offset, value, label) in [
        (32, 0, "rank"),
        (40, 999, "type"),
        (64, 0, "payload"),
        (88, 1, "reserved"),
    ] {
        let mut bytes = std::fs::read(&p).unwrap();
        bytes[offset..offset + 8].copy_from_slice(&(value as u64).to_le_bytes());
        let path = dir.join(format!("bad-{label}.pio"));
        barrier_file(comm, &path, &bytes);
        assert!(
            read_mpi_catalog(&path, comm).is_err(),
            "accepted bad {label}"
        );
    }

    // An incomplete or malformed final named record must not be silently dropped.
    for (suffix, tail) in [("incomplete", vec![0u8; 1]), ("header-tail", vec![0u8; 72])] {
        let path = dir.join(format!("named-{suffix}.pio"));
        let mut bytes = std::fs::read(&named).unwrap();
        if suffix == "header-tail" {
            bytes.extend_from_slice(&tail);
            let n = bytes.len();
            bytes[n - 72..n - 64].copy_from_slice(b"bad-head");
        } else {
            bytes.extend_from_slice(&tail);
        }
        barrier_file(comm, &path, &bytes);
        assert!(read_mpi_named_catalog(&path, comm).is_err());
    }

    // Named data recovery intentionally returns its committed prefix.
    let recover = dir.join("recover.pio");
    write_mpi_named(&recover, "ok", a.view()).unwrap();
    if world.rank() == 0 {
        std::fs::OpenOptions::new()
            .append(true)
            .open(&recover)
            .unwrap()
            .write_all(b"broken")
            .unwrap();
    }
    comm.barrier();
    let mut out = array_f32(&world, [4, 5], [3]);
    assert!(read_mpi_named_catalog(&recover, comm).is_err());
    assert!(read_mpi_named(&recover, "ok", out.view_mut()).is_ok());

    // Wrong format and raw bytes never pass a versioned catalog parser.
    let raw = dir.join("raw");
    barrier_file(comm, &raw, &[0u8; 64]);
    assert!(read_mpi_catalog(&raw, comm).is_err());
    assert!(read_mpi_named_catalog(&p, comm).is_err());

    // The operation descriptor is checked before any file header/data access.
    if world.size() > 1 {
        let mismatch = if world.rank() == 0 {
            read_mpi_catalog(&p, comm).is_err()
        } else {
            read_mpi_named_catalog(&p, comm).is_err()
        };
        assert!(mismatch);
    }

    // Metadata-only sparse file: a valid header and payload extent need not contain data bytes.
    let sparse = dir.join("sparse.pio");
    let mut h = std::fs::read(&p).unwrap();
    let payload = u64_at(&h, 72) as usize;
    h.truncate(152);
    if world.rank() == 0 {
        std::fs::write(&sparse, &h).unwrap();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&sparse)
            .unwrap()
            .set_len((152 + payload) as u64)
            .unwrap();
    }
    comm.barrier();
    assert_eq!(
        read_mpi_catalog(&sparse, comm).unwrap()[0].global_shape(),
        &[4, 5]
    );

    #[cfg(feature = "parallel-hdf5")]
    hdf_cases(&world, comm, &dir, &a);

    if world.rank() == 0 {
        println!("CATALOG_OK");
    }
    comm.barrier();
    cleanup_owned_temp_dir(&world, &dir);
}

#[cfg(feature = "parallel-hdf5")]
fn write_hdf_scalar<T: pencil_io::IoElement + Clone>(
    path: &Path,
    a: &PencilArray<f32, 2, 2>,
    value: T,
) {
    let x = PencilArray::from_elem(a.pencil().clone(), ExtraShape::new([3, 2]).unwrap(), value)
        .unwrap();
    pencil_io::write_hdf5(path, x.view()).unwrap();
}

#[cfg(feature = "parallel-hdf5")]
fn write_hdf_scalar_named<T: pencil_io::IoElement + Clone>(
    path: &Path,
    a: &PencilArray<f32, 2, 2>,
    value: T,
) {
    let x = PencilArray::from_elem(a.pencil().clone(), ExtraShape::new([3, 2]).unwrap(), value)
        .unwrap();
    pencil_io::write_hdf5_named(path, "data", x.view()).unwrap();
}

#[cfg(feature = "parallel-hdf5")]
fn hdf_cases<C: Communicator>(
    _world: &C,
    comm: &mpi::topology::CartesianCommunicator,
    dir: &Path,
    a: &PencilArray<f32, 2, 2>,
) {
    use pencil_io::{append_hdf5_named, read_hdf5_catalog, write_hdf5, write_hdf5_named};
    // An unrelated HDF5 file has no recognized catalog namespace.
    let unknown = dir.join("unknown.h5");
    if comm.rank() == 0 {
        let f = hdf5_metno::File::create(&unknown).unwrap();
        f.create_group("other").unwrap();
        f.close().unwrap();
    }
    comm.barrier();
    assert!(read_hdf5_catalog(&unknown, comm).is_err());

    // Links are not catalog objects: neither namespace roots nor leaf datasets
    // may escape the file/object graph inspected by the catalog reader.
    let leaf_target = dir.join("leaf-target.h5");
    write_hdf5(&leaf_target, a.view()).unwrap();
    let named_target = dir.join("named-leaf-target.h5");
    write_hdf5_named(&named_target, "data", a.view()).unwrap();
    comm.barrier();
    for (label, external) in [("soft", false), ("external", true)] {
        let bad = dir.join(format!("linked-{label}.h5"));
        // Keep a valid writer-generated named dataset as the target of the
        // legacy link; rejection must be because the link is followed, not
        // because its target has no metadata.
        write_hdf5_named(&bad, "data", a.view()).unwrap();
        if comm.rank() == 0 {
            let f = hdf5_metno::File::open_rw(&bad).unwrap();
            f.create_group("pencil_io_v1").unwrap();
            if external {
                f.link_external(
                    leaf_target.to_str().unwrap(),
                    "/pencil_io_v1/data",
                    "pencil_io_v1/data",
                )
                .unwrap();
            } else {
                f.link_soft("/pencil_io_named_v1/64617461", "pencil_io_v1/data")
                    .unwrap();
            }
            f.close().unwrap();
        }
        comm.barrier();
        assert!(read_hdf5_catalog(&bad, comm).is_err());
    }

    // Every link kind is rejected at both the legacy and named namespace roots
    // and leaves. A valid namespace elsewhere must not mask a bad one.
    for named_ns in [false, true] {
        let ns = if named_ns {
            "pencil_io_named_v1"
        } else {
            "pencil_io_v1"
        };
        for kind in [
            "soft-leaf",
            "external-leaf",
            "hard-group-leaf",
            "soft-root",
            "external-root",
            "hard-dataset-root",
        ] {
            let bad = dir.join(format!(
                "malicious-{}-{}.h5",
                if named_ns { "named" } else { "legacy" },
                kind
            ));
            // Populate the opposite namespace through the real writer, so
            // every link target is a valid dataset/group with both attrs.
            if named_ns {
                write_hdf5(&bad, a.view()).unwrap();
            } else {
                write_hdf5_named(&bad, "data", a.view()).unwrap();
            }
            let target_dataset = if named_ns {
                "/pencil_io_v1/data"
            } else {
                "/pencil_io_named_v1/64617461"
            };
            let target_group = if named_ns {
                "/pencil_io_v1"
            } else {
                "/pencil_io_named_v1"
            };
            let target_file = if named_ns {
                &leaf_target
            } else {
                &named_target
            };
            if comm.rank() == 0 {
                let f = hdf5_metno::File::open_rw(&bad).unwrap();
                match kind {
                    "soft-leaf" => {
                        f.create_group(ns).unwrap();
                        f.link_soft(target_dataset, &format!("{ns}/data")).unwrap();
                    }
                    "external-leaf" => {
                        f.create_group(ns).unwrap();
                        f.link_external(
                            target_file.to_str().unwrap(),
                            target_dataset,
                            &format!("{ns}/data"),
                        )
                        .unwrap();
                    }
                    "hard-group-leaf" => {
                        f.create_group(ns).unwrap();
                        f.link_hard(target_group, &format!("{ns}/data")).unwrap();
                    }
                    "soft-root" => f.link_soft(target_group, ns).unwrap(),
                    "external-root" => f
                        .link_external(target_file.to_str().unwrap(), target_group, ns)
                        .unwrap(),
                    "hard-dataset-root" => f.link_hard(target_dataset, ns).unwrap(),
                    _ => unreachable!(),
                }
                f.close().unwrap();
            }
            comm.barrier();
            assert!(read_hdf5_catalog(&bad, comm).is_err());
        }
    }

    // A valid other namespace cannot hide a malicious root; valid files still recover.
    for (named_valid, bad_ns) in [(true, "pencil_io_v1"), (false, "pencil_io_named_v1")] {
        let bad = dir.join(format!(
            "malicious-with-valid-{}.h5",
            if named_valid { "named" } else { "legacy" }
        ));
        if named_valid {
            write_hdf5_named(&bad, "ok", a.view()).unwrap();
        } else {
            write_hdf5(&bad, a.view()).unwrap();
        }
        if comm.rank() == 0 {
            let f = hdf5_metno::File::open_rw(&bad).unwrap();
            f.link_soft("/missing", bad_ns).unwrap();
            f.close().unwrap();
        }
        comm.barrier();
        assert!(read_hdf5_catalog(&bad, comm).is_err());
    }
    // A structurally valid HDF5 container without Pencil metadata is rejected
    // during metadata inspection, before any dataset payload read.
    let missing = dir.join("missing-metadata.h5");
    if comm.rank() == 0 {
        let f = hdf5_metno::File::create(&missing).unwrap();
        f.create_group("pencil_io_v1")
            .unwrap()
            .new_dataset::<f32>()
            .shape([1])
            .create("0")
            .unwrap();
        f.close().unwrap();
    }
    comm.barrier();
    assert!(read_hdf5_catalog(&missing, comm).is_err());

    let old = dir.join("old.h5");
    write_hdf5(&old, a.view()).unwrap();
    let x = read_hdf5_catalog(&old, comm).unwrap();
    assert_eq!(x[0].scalar_type(), ScalarType::F32);
    assert_eq!(x[0].global_shape(), &[4, 5]);
    assert_eq!(x[0].extra_shape(), &[3]);

    // HDF5 uses the same twelve catalog codes, including packed complex types.
    macro_rules! hdf_scalar {
        ($code:expr, $value:expr) => {{
            let path = dir.join(format!("scalar-{}.h5", $code.code()));
            write_hdf_scalar(&path, a, $value);
            let named_path = dir.join(format!("scalar-named-{}.h5", $code.code()));
            write_hdf_scalar_named(&named_path, a, $value);
            let got = read_hdf5_catalog(&path, comm).unwrap();
            let named_got = read_hdf5_catalog(&named_path, comm).unwrap();
            assert_eq!(got[0].scalar_type(), $code);
            assert_eq!(got[0].scalar_type().code(), $code.code());
            assert_eq!(got[0].scalar_type().width(), $code.width());
            assert_eq!(got[0].extra_shape(), &[3, 2]);
            assert_eq!(named_got[0].scalar_type(), $code);
            assert_eq!(named_got[0].name(), Some("data"));
            if comm.rank() == 0 {
                for (file, dataset) in [
                    (&path, "/pencil_io_v1/data"),
                    (&named_path, "/pencil_io_named_v1/64617461"),
                ] {
                    let f = hdf5_metno::File::open(file).unwrap();
                    let d = f.dataset(dataset).unwrap();
                    assert_eq!(
                        d.attr("pencil_io_type")
                            .unwrap()
                            .read_scalar::<u64>()
                            .unwrap(),
                        $code.code()
                    );
                    assert_eq!(
                        d.attr("pencil_io_width")
                            .unwrap()
                            .read_scalar::<u64>()
                            .unwrap(),
                        $code.width() as u64
                    );
                    f.close().unwrap();
                }
            }
            comm.barrier();
        }};
    }
    hdf_scalar!(ScalarType::I8, -1i8);
    hdf_scalar!(ScalarType::U8, 1u8);
    hdf_scalar!(ScalarType::I16, -1i16);
    hdf_scalar!(ScalarType::U16, 1u16);
    hdf_scalar!(ScalarType::I32, -1i32);
    hdf_scalar!(ScalarType::U32, 1u32);
    hdf_scalar!(ScalarType::I64, -1i64);
    hdf_scalar!(ScalarType::U64, 1u64);
    hdf_scalar!(ScalarType::F32, 1.0f32);
    hdf_scalar!(ScalarType::F64, 1.0f64);
    hdf_scalar!(ScalarType::ComplexF32, Complex32::new(1.0, 2.0));
    hdf_scalar!(ScalarType::ComplexF64, Complex64::new(1.0, 2.0));

    for (label, attr, value) in [
        ("code", "pencil_io_type", 99u64),
        ("width", "pencil_io_width", 8u64),
    ] {
        let bad = dir.join(format!("bad-{label}.h5"));
        write_hdf5(&bad, a.view()).unwrap();
        if comm.rank() == 0 {
            let f = hdf5_metno::File::open_rw(&bad).unwrap();
            let d = f.group("/pencil_io_v1").unwrap().dataset("data").unwrap();
            d.attr(attr)
                .unwrap()
                .as_writer()
                .write_raw(&[value])
                .unwrap();
            f.close().unwrap();
        }
        comm.barrier();
        assert!(read_hdf5_catalog(&bad, comm).is_err());
    }

    let named = dir.join("named.h5");
    write_hdf5_named(&named, "hdf/日本", a.view()).unwrap();
    let y = read_hdf5_catalog(&named, comm).unwrap();
    assert_eq!(
        y.iter().filter_map(|d| d.name()).collect::<Vec<_>>(),
        vec!["hdf/日本"]
    );
    // Recovery: a valid legacy and named catalog remains readable after the hostile fixtures.
    assert_eq!(
        read_hdf5_catalog(&old, comm).unwrap()[0].scalar_type(),
        ScalarType::F32
    );
    assert_eq!(read_hdf5_catalog(&named, comm).unwrap().len(), 1);

    // A valid first named dataset does not make a corrupted second one invisible.
    let corrupt = dir.join("named-corrupt.h5");
    write_hdf5_named(&corrupt, "first", a.view()).unwrap();
    append_hdf5_named(&corrupt, "second", a.view()).unwrap();
    if comm.rank() == 0 {
        let f = hdf5_metno::File::open_rw(&corrupt).unwrap();
        let g = f.group("/pencil_io_named_v1").unwrap();
        let mut names = g.member_names().unwrap();
        names.sort();
        let d = g.dataset(names.last().unwrap()).unwrap();
        d.attr("pencil_io_commit")
            .unwrap()
            .as_writer()
            .write_raw(&[0u64])
            .unwrap();
        f.close().unwrap();
    }
    comm.barrier();
    assert!(read_hdf5_catalog(&corrupt, comm).is_err());
}

use std::io::Write;
