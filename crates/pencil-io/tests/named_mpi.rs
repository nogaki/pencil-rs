use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use mpi::collective::SystemOperation;
use mpi::traits::*;
use pencil_array::{AxisPermutation, ExtraShape, MpiTopology, Pencil, PencilArray};
use pencil_io::{NamedIoError, append_mpi_named, read_mpi_named, write_mpi, write_mpi_named};

mod support;
use support::{cleanup_owned_temp_dir, owned_temp_dir};

fn root_bytes<C: CommunicatorCollectives>(world: &C, path: &Path) -> Vec<u8> {
    let mut n = if world.rank() == 0 {
        std::fs::metadata(path).unwrap().len()
    } else {
        0
    };
    world.process_at_rank(0).broadcast_into(&mut n);
    let mut b = vec![0; n as usize];
    if world.rank() == 0 {
        b = std::fs::read(path).unwrap();
    }
    world.process_at_rank(0).broadcast_into(&mut b);
    b
}
fn reset<C: CommunicatorCollectives>(world: &C, path: &Path) {
    if world.rank() == 0 {
        let _ = std::fs::remove_file(path);
    }
    world.barrier();
}
fn check<C: CommunicatorCollectives>(world: &C, failures: &mut Vec<String>, name: &str, ok: bool) {
    let x = i32::from(ok);
    let mut all = 0;
    world.all_reduce_into(&x, &mut all, SystemOperation::min());
    if all == 0 {
        failures.push(name.into());
    }
}

fn make_i32<C: Communicator>(
    world: &C,
    grid: [usize; 2],
    shape: [usize; 2],
    extra: [usize; 1],
) -> PencilArray<i32, 2, 2> {
    let p = Pencil::<2, 2>::new(MpiTopology::new(world, grid).unwrap(), shape, [0, 1]).unwrap();
    let mut a = PencilArray::from_elem(p, ExtraShape::new(extra).unwrap(), -1).unwrap();
    let mut v = a.view_mut();
    let r = v.pencil().local_ranges().clone();
    for e in 0..extra[0] {
        for x in 0..r[0].len() {
            for y in 0..r[1].len() {
                *v.get_local_mut(&[e], [x, y]).unwrap() =
                    (e * 10000 + (r[0].start + x) * 100 + r[1].start + y) as i32;
            }
        }
    }
    a
}
fn assert_i32(a: &PencilArray<i32, 2, 2>, extra: usize) {
    let v = a.view();
    let r = v.pencil().local_ranges().clone();
    for e in 0..extra {
        for x in 0..r[0].len() {
            for y in 0..r[1].len() {
                assert_eq!(
                    v.get_local(&[e], [x, y]),
                    Some(&((e * 10000 + (r[0].start + x) * 100 + r[1].start + y) as i32))
                );
            }
        }
    }
}

// Independent v2 parser: no production format helpers.
fn raw_oracle(bytes: &[u8]) {
    fn word(b: &[u8], p: usize) -> usize {
        u64::from_le_bytes(b[p..p + 8].try_into().unwrap()) as usize
    }
    assert_eq!(&bytes[..8], b"PIONAM02");
    assert_eq!(word(bytes, 8), 2);
    assert_eq!(word(bytes, 16), 32);
    let mut p = 32;
    for (key, code, width, dims, extra) in
        [("A", 5, 4, [4, 5], vec![2]), ("B", 10, 8, [3, 4], vec![])]
    {
        assert_eq!(&bytes[p..p + 8], b"PIOREC02");
        assert_eq!(word(bytes, p + 8), 2);
        let nl = word(bytes, p + 16);
        let ml = word(bytes, p + 24);
        let payload = word(bytes, p + 32);
        let total = word(bytes, p + 40);
        assert_eq!(word(bytes, p + 48), code);
        assert_eq!(word(bytes, p + 56), width);
        assert_eq!(word(bytes, p + 64), 0);
        assert_eq!(&bytes[p + 72..p + 72 + nl], key.as_bytes());
        let meta = p + 72 + nl;
        assert_eq!(word(bytes, meta), 2);
        assert_eq!(word(bytes, meta + 8), extra.len());
        assert_eq!(ml, 16 + 8 * (2 + extra.len()));
        for (i, &d) in dims.iter().chain(extra.iter()).enumerate() {
            assert_eq!(word(bytes, meta + 16 + 8 * i), d);
        }
        let data = meta + ml;
        assert_eq!(
            payload,
            dims.iter().product::<usize>() * extra.iter().product::<usize>() * width
        );
        assert_eq!(total, 72 + nl + ml + payload + 8);
        assert_eq!(
            &bytes[p + total - 8..p + total],
            &0x434f_4d4d_4954_5445u64.to_le_bytes()
        );
        if key == "A" {
            let expected: Vec<i32> = (0..2)
                .flat_map(|e| {
                    (0..4).flat_map(move |x| (0..5).map(move |y| e * 10000 + x * 100 + y))
                })
                .collect();
            let actual: Vec<i32> = bytes[data..data + payload]
                .chunks_exact(4)
                .map(|v| i32::from_le_bytes(v.try_into().unwrap()))
                .collect();
            assert_eq!(actual, expected);
        } else {
            assert!(
                bytes[data..data + payload]
                    .chunks_exact(8)
                    .all(|v| f64::from_le_bytes(v.try_into().unwrap()) == 3.25)
            );
        }
        p += total;
    }
    assert_eq!(p, bytes.len());
}

// Independent sparse v2 fixture: >1 GiB payload, but only bounded header and
// selected payload bytes are ever allocated or read, and none are broadcast.
const LARGE_PAYLOAD: u64 = (1 << 30) + 8;
const LARGE_DATA: u64 = 32 + 72 + 5 + 24;
const LARGE_END: u64 = LARGE_DATA + LARGE_PAYLOAD + 8;

fn sparse_fixture(path: &Path) -> std::io::Result<()> {
    let mut file = std::fs::File::create_new(path)?;
    file.write_all(b"PIONAM02")?;
    for word in [2u64, 32, 0] {
        file.write_all(&word.to_le_bytes())?;
    }
    file.write_all(b"PIOREC02")?;
    for word in [2, 5, 24, LARGE_PAYLOAD, LARGE_END - 32, 1, 1, 0] {
        file.write_all(&word.to_le_bytes())?;
    }
    file.write_all(b"large")?;
    for word in [1u64, 0, LARGE_PAYLOAD] {
        file.write_all(&word.to_le_bytes())?;
    }
    for offset in [LARGE_DATA, LARGE_DATA + LARGE_PAYLOAD / 2, LARGE_END - 16] {
        file.seek(SeekFrom::Start(offset))?;
        file.write_all(b"sentinel")?;
    }
    file.write_all(&0x434f_4d4d_4954_5445u64.to_le_bytes())?;
    file.sync_all()
}

fn sparse_snapshot(path: &Path) -> std::io::Result<Vec<u8>> {
    let mut file = std::fs::File::open(path)?;
    let mut bytes = vec![0; LARGE_DATA as usize + 32];
    file.read_exact(&mut bytes[..LARGE_DATA as usize])?;
    for (i, offset) in [
        LARGE_DATA,
        LARGE_DATA + LARGE_PAYLOAD / 2,
        LARGE_END - 16,
        LARGE_END - 8,
    ]
    .into_iter()
    .enumerate()
    {
        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(
            &mut bytes[LARGE_DATA as usize + i * 8..LARGE_DATA as usize + (i + 1) * 8],
        )?;
    }
    Ok(bytes)
}

#[test]
fn named_mpi_matrix() {
    eprintln!("named_mpi:start");
    let universe = mpi::initialize().expect("MPI initialize once");
    let world = universe.world();
    let size = world.size() as usize;
    let dir = owned_temp_dir(&world, "pencil-named-mpi");
    let path = dir.join("named.pio");
    let mut failures = Vec::new();

    let a = make_i32(&world, [size, 1], [4, 5], [2]);
    reset(&world, &path);
    let slash = write_mpi_named(&path, "utf8/α", a.view());
    check(
        &world,
        &mut failures,
        "UTF8 slash name should be accepted",
        slash.is_ok(),
    );
    let mut unicode = make_i32(&world, [size, 1], [4, 5], [2]);
    unicode.as_mut_slice().fill(-5);
    read_mpi_named(&path, "utf8/α", unicode.view_mut()).unwrap();
    assert_i32(&unicode, 2);
    reset(&world, &path);
    check(
        &world,
        &mut failures,
        "write A",
        write_mpi_named(&path, "A", a.view()).is_ok(),
    );
    let baseline = root_bytes(&world, &path);

    let reader_p = Pencil::<2, 2>::new_permuted(
        MpiTopology::new(&world, [1, size]).unwrap(),
        [4, 5],
        [1, 0],
        AxisPermutation::new([1, 0]).unwrap(),
    )
    .unwrap();
    let mut ar = PencilArray::from_elem(reader_p, ExtraShape::new([2]).unwrap(), -77).unwrap();
    let read_a = read_mpi_named(&path, "A", ar.view_mut());
    check(&world, &mut failures, "read A changed grid", read_a.is_ok());
    if read_a.is_ok() {
        assert_i32(&ar, 2);
    }

    let bpath = &path;
    let bp =
        Pencil::<2, 2>::new(MpiTopology::new(&world, [size, 1]).unwrap(), [3, 4], [0, 1]).unwrap();
    let b = PencilArray::from_elem(bp, ExtraShape::scalar(), 3.25f64).unwrap();
    check(
        &world,
        &mut failures,
        "append B distinct dtype/global shape",
        append_mpi_named(bpath, "B", b.view()).is_ok(),
    );
    let raw = root_bytes(&world, bpath);
    check(
        &world,
        &mut failures,
        "raw v2 oracle",
        raw.get(0..8) == Some(b"PIONAM02")
            && raw.get(8..16) == Some(&2u64.to_le_bytes())
            && raw.get(32..40) == Some(b"PIOREC02")
            && raw.get(40..48) == Some(&2u64.to_le_bytes()),
    );
    assert_eq!(&raw[..baseline.len()], baseline.as_slice());
    raw_oracle(&raw);
    let brp = Pencil::<2, 2>::new_permuted(
        MpiTopology::new(&world, [1, size]).unwrap(),
        [3, 4],
        [1, 0],
        AxisPermutation::new([1, 0]).unwrap(),
    )
    .unwrap();
    let mut br = PencilArray::from_elem(brp, ExtraShape::scalar(), -2f64).unwrap();
    read_mpi_named(&path, "B", br.view_mut()).unwrap();
    assert!(br.as_slice().iter().all(|&v| v == 3.25));
    read_mpi_named(&path, "A", ar.view_mut()).unwrap();
    assert_i32(&ar, 2);
    let before = raw.clone();
    let duplicate = append_mpi_named(&path, "A", a.view());
    check(
        &world,
        &mut failures,
        "duplicate rejected",
        matches!(duplicate, Err(NamedIoError::DuplicateName))
            && root_bytes(&world, &path) == before,
    );

    let missing = dir.join("missing.pio");
    reset(&world, &missing);
    let old = ar.as_slice().to_vec();
    let e = read_mpi_named(&missing, "A", ar.view_mut());
    check(
        &world,
        &mut failures,
        "missing and target unchanged",
        matches!(e, Err(NamedIoError::NotFound | NamedIoError::Io(_)))
            && ar.as_slice() == old.as_slice(),
    );

    let mismatch_before = root_bytes(&world, &path);
    let op = if world.rank() == 0 {
        write_mpi(&path, a.view()).map_err(NamedIoError::Io)
    } else {
        append_mpi_named(&path, "X", a.view())
    };
    check(
        &world,
        &mut failures,
        "old/new operation rank mismatch",
        op.is_err() && root_bytes(&world, &path) == mismatch_before,
    );
    world.barrier();
    let name = if world.rank() == 0 {
        "name-a"
    } else {
        "name-b"
    };
    let mismatch = if size == 1 {
        None
    } else {
        Some(append_mpi_named(&path, name, a.view()))
    };
    check(
        &world,
        &mut failures,
        "name mismatch recovery",
        size == 1 || (mismatch.unwrap().is_err() && root_bytes(&world, &path) == mismatch_before),
    );
    world.barrier();
    let type_array =
        PencilArray::from_elem(a.pencil().clone(), ExtraShape::new([2]).unwrap(), 1i64).unwrap();
    let layout_array = PencilArray::from_elem(
        Pencil::<2, 2>::new(a.pencil().topology().clone(), [4, 6], [0, 1]).unwrap(),
        ExtraShape::new([2]).unwrap(),
        1i32,
    )
    .unwrap();
    if size > 1 {
        let typ = if world.rank() == 0 {
            append_mpi_named(&path, "type", a.view())
        } else {
            append_mpi_named(&path, "type", type_array.view())
        };
        assert!(typ.is_err());
        let layout = if world.rank() == 0 {
            append_mpi_named(&path, "shape", a.view())
        } else {
            append_mpi_named(&path, "shape", layout_array.view())
        };
        assert!(layout.is_err());
        let key = if world.rank() == 0 { "" } else { "valid" };
        assert!(append_mpi_named(&path, key, a.view()).is_err());
        let other_path = dir.join("other.pio");
        let chosen_path = if world.rank() == 0 {
            &path
        } else {
            &other_path
        };
        assert!(append_mpi_named(chosen_path, "path", a.view()).is_err());
        assert_eq!(root_bytes(&world, &path), mismatch_before);
    }
    check(
        &world,
        &mut failures,
        "append recovery",
        append_mpi_named(&path, "recovery", a.view()).is_ok(),
    );

    // Empty extras/ranks are valid and remain readable.
    let zpath = dir.join("zero.pio");
    reset(&world, &zpath);
    let z = make_i32(&world, [size, 1], [2, 2], [0]);
    let zw = write_mpi_named(&zpath, "zero", z.view());
    check(&world, &mut failures, "empty extras/ranks", zw.is_ok());
    let mut zr = make_i32(&world, [1, size], [2, 2], [0]);
    read_mpi_named(&zpath, "zero", zr.view_mut()).unwrap();
    for name in ["", "bad\0key", &"x".repeat(1025)] {
        let unchanged = root_bytes(&world, &path);
        assert!(matches!(
            append_mpi_named(&path, name, a.view()),
            Err(NamedIoError::InvalidName)
        ));
        assert_eq!(root_bytes(&world, &path), unchanged);
        let before = ar.as_slice().to_vec();
        assert!(read_mpi_named(&path, name, ar.view_mut()).is_err());
        assert_eq!(ar.as_slice(), before);
    }
    let old = ar.as_slice().to_vec();
    assert!(matches!(
        read_mpi_named(&path, "absent", ar.view_mut()),
        Err(NamedIoError::NotFound)
    ));
    assert_eq!(ar.as_slice(), old);
    assert!(read_mpi_named(&path, "B", ar.view_mut()).is_err());
    assert_eq!(ar.as_slice(), old);

    // A malformed/incomplete tail is rejected without changing the file; the first record remains readable.
    if world.rank() == 0 {
        std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"tail")
            .unwrap();
    }
    world.barrier();
    let malformed = root_bytes(&world, &path);
    let tail = append_mpi_named(&path, "tail", a.view());
    check(
        &world,
        &mut failures,
        "malformed tail append rejected unchanged",
        tail.is_err() && root_bytes(&world, &path) == malformed,
    );
    let mut again = PencilArray::from_elem(
        Pencil::<2, 2>::new(MpiTopology::new(&world, [1, size]).unwrap(), [4, 5], [1, 0]).unwrap(),
        ExtraShape::new([2]).unwrap(),
        0,
    )
    .unwrap();
    let good = read_mpi_named(&path, "A", again.view_mut());
    check(
        &world,
        &mut failures,
        "read A with incomplete tail",
        good.is_ok(),
    );

    assert_i32(&again, 2);
    let bstart = baseline.len();
    let mut bad_tails = Vec::new();
    for (offset, value) in [
        (8, 99),
        (16, u64::MAX),
        (24, u64::MAX),
        (32, 1),
        (40, u64::MAX),
        (48, 99),
        (56, 1),
        (64, 1),
        (73, u64::MAX),
    ] {
        let mut corrupt = raw.clone();
        corrupt[bstart + offset..bstart + offset + 8].copy_from_slice(&value.to_le_bytes());
        bad_tails.push(corrupt);
    }
    for cut in [bstart + 1, bstart + 72, raw.len() - 8, raw.len() - 1] {
        bad_tails.push(raw[..cut].to_vec());
    }
    for corrupt in bad_tails {
        if world.rank() == 0 {
            std::fs::write(&path, &corrupt).unwrap();
        }
        world.barrier();
        read_mpi_named(&path, "A", again.view_mut()).unwrap();
        assert_i32(&again, 2);
        assert!(matches!(
            append_mpi_named(&path, "new", a.view()),
            Err(NamedIoError::InvalidTail)
        ));
        assert_eq!(root_bytes(&world, &path), corrupt);
        let before = again.as_slice().to_vec();
        assert!(read_mpi_named(&path, "B", again.view_mut()).is_err());
        assert_eq!(again.as_slice(), before);
    }
    // A plausible record header whose end cannot fit MPI_Offset is not a
    // committed record, even in a sparse file larger than the former cap.
    if world.rank() == 0 {
        let mut overflow = raw.clone();
        overflow[bstart + 40..bstart + 48].copy_from_slice(&(i64::MAX as u64).to_le_bytes());
        std::fs::write(&path, &overflow).unwrap();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(1024 * 1024 * 1024 + 1)
            .unwrap();
    }
    world.barrier();
    read_mpi_named(&path, "A", again.view_mut()).unwrap();
    assert_i32(&again, 2);
    assert!(matches!(
        append_mpi_named(&path, "oversized-tail", a.view()),
        Err(NamedIoError::InvalidTail)
    ));
    if world.rank() == 0 {
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            1024 * 1024 * 1024 + 1
        );
    }
    world.barrier();

    let large_path = dir.join("sparse-valid.pio");
    let fixture = if world.rank() == 0 {
        sparse_fixture(&large_path).and_then(|()| sparse_snapshot(&large_path))
    } else {
        Ok(Vec::new())
    };
    let mut fixture_ok = i32::from(fixture.is_ok());
    world.process_at_rank(0).broadcast_into(&mut fixture_ok);
    if fixture_ok == 1 {
        let before = fixture.unwrap();
        let appended = append_mpi_named(&large_path, "small", a.view());
        check(
            &world,
            &mut failures,
            "append beyond 1 GiB",
            appended.is_ok(),
        );
        again.as_mut_slice().fill(-99);
        let read = read_mpi_named(&large_path, "small", again.view_mut());
        check(&world, &mut failures, "read beyond 1 GiB", read.is_ok());
        if read.is_ok() {
            assert_i32(&again, 2);
        }
        check(
            &world,
            &mut failures,
            "large record remains indexed",
            matches!(
                append_mpi_named(&large_path, "large", a.view()),
                Err(NamedIoError::DuplicateName)
            ),
        );
        let unchanged = world.rank() != 0
            || (|| -> std::io::Result<bool> {
                Ok(sparse_snapshot(&large_path)? == before
                    && std::fs::metadata(&large_path)?.len()
                        == LARGE_END + (baseline.len() - 32 + 4) as u64)
            })()
            .unwrap_or(false);
        check(
            &world,
            &mut failures,
            "large metadata and selected bytes unchanged; exact appended size",
            unchanged,
        );
    } else {
        eprintln!("sparse fixture creation failed: {fixture:?}");
        failures.push("sparse fixture creation failed collectively".into());
    }
    cleanup_owned_temp_dir(&world, &dir);
    if !failures.is_empty() {
        panic!("named MPI failures: {failures:?}");
    }
    world.barrier();
    if world.rank() == 0 {
        eprintln!("\nNAMED_MPI_PASSED ranks={}", world.size());
    }
    world.barrier();
}
