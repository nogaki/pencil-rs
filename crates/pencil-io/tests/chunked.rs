//! One MPI initialization, exercised under 1, 4 and 6 ranks.
use mpi::traits::*;
use pencil_array::{
    AxisPermutation, ExtraShape, MpiTopology, Pencil, PencilArray, partition_range,
};
use pencil_io::{
    IoElement, MpiIoMode, MpiIoOptions, read_mpi_chunked, read_mpi_chunked_catalog,
    write_mpi_chunked,
};
use std::{path::Path, sync::Arc};
mod support;
use support::{cleanup_owned_temp_dir, owned_temp_dir};
fn replace(c: &impl CommunicatorCollectives, p: &Path, b: &[u8]) {
    if c.rank() == 0 {
        std::fs::write(p, b).unwrap();
    }
    c.barrier();
}
fn u(b: &[u8], i: usize) -> usize {
    u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap()) as usize
}
fn array<T: IoElement>(
    t: Arc<MpiTopology<2>>,
    extra: usize,
    v: T,
    perm: [usize; 2],
) -> PencilArray<T, 2, 2> {
    let p = Pencil::new_permuted(t, [3, 5], [0, 1], AxisPermutation::new(perm).unwrap()).unwrap();
    PencilArray::from_elem(p, ExtraShape::new([extra]).unwrap(), v).unwrap()
}
fn scalar<T: IoElement + PartialEq + std::fmt::Debug>(
    world: &mpi::topology::SimpleCommunicator,
    dir: &Path,
    value: impl Fn(usize) -> T,
) {
    let t = MpiTopology::new(world, [world.size() as usize, 1]).unwrap();
    let c = t.communicator();
    for extra in [2, 0] {
        for perm in [[1, 0], [0, 1]] {
            for mode in [MpiIoMode::Collective, MpiIoMode::Independent] {
                let mut src = array(t.clone(), extra, value(0), perm);
                // Values indexed logically make permuted physical order independently visible.
                let r = src.pencil().local_ranges().clone();
                for e in 0..extra {
                    for x in 0..r[0].len() {
                        for y in 0..5 {
                            *src.view_mut().get_local_mut(&[e], [x, y]).unwrap() =
                                value(e * 15 + (r[0].start + x) * 5 + y);
                        }
                    }
                }
                let p = dir.join(format!("{}-{extra}-{perm:?}-{mode:?}", T::CODE));
                let opts = MpiIoOptions::default()
                    .mode(mode)
                    .hint("cb_buffer_size", "1048576");
                write_mpi_chunked(&p, src.view(), &opts).unwrap();
                let mut dst = array(t.clone(), extra, value(99), perm);
                read_mpi_chunked(&p, dst.view_mut(), &opts).unwrap();
                assert_eq!(src.as_slice(), dst.as_slice());
                let info = read_mpi_chunked_catalog(&p, c, &opts).unwrap();
                assert_eq!(info.name(), None);
                assert_eq!(info.global_shape(), [3, 5]);
                assert_eq!(info.extra_shape(), [extra as u64]);
                assert_eq!(info.scalar_type().code(), T::CODE);
                let provenance: Vec<_> = [world.size() as u64, 1, perm[0] as u64, perm[1] as u64]
                    .into_iter()
                    .flat_map(u64::to_le_bytes)
                    .collect();
                assert_eq!(info.provenance(), provenance);
                if c.rank() == 0 {
                    let b = std::fs::read(&p).unwrap();
                    assert_eq!(&b[..8], b"PNCHUNK3");
                    let mut expected = Vec::new();
                    for rank in 0..c.size() {
                        let coords = c.rank_to_coordinates(rank);
                        let r = partition_range(3, c.size() as usize, coords[0] as usize).unwrap();
                        for e in 0..extra {
                            if perm == [1, 0] {
                                for y in 0..5 {
                                    for x in r.clone() {
                                        value(e * 15 + x * 5 + y).encode_le(&mut expected);
                                    }
                                }
                            } else {
                                for x in r.clone() {
                                    for y in 0..5 {
                                        value(e * 15 + x * 5 + y).encode_le(&mut expected);
                                    }
                                }
                            }
                        }
                    }
                    assert_eq!(&b[u(&b, 2)..], expected);
                    assert_eq!(u(&b, 3), b.len());
                }
                c.barrier();
            }
        }
    }
}
fn total_rank_bound(world: &mpi::topology::SimpleCommunicator, dir: &Path) {
    // Each rank count is legal independently, but 2 + 1023 exceeds 1024.
    let t = MpiTopology::new(world, [world.size() as usize, 1]).unwrap();
    let c = t.communicator();
    let pencil = Pencil::new(t.clone(), [3, 5], [0, 1]).unwrap();
    let mut dst = PencilArray::from_elem(
        pencil.clone(),
        ExtraShape::new(vec![1; 1023]).unwrap(),
        99u64,
    )
    .unwrap();
    let opts = MpiIoOptions::default();
    let absent = dir.join("rank-bound-absent");
    let existing = dir.join("rank-bound-existing");
    replace(c, &existing, b"unchanged");
    for path in [&absent, &existing] {
        assert!(write_mpi_chunked(path, dst.view(), &opts).is_err());
        assert!(read_mpi_chunked(path, dst.view_mut(), &opts).is_err());
    }
    // A single invalid rank must also reject collectively, after common entry.
    if world.size() > 1 {
        let valid = array(t.clone(), 1, 42u64, [0, 1]);
        let src = if c.rank() == 0 {
            dst.view()
        } else {
            valid.view()
        };
        assert!(write_mpi_chunked(&absent, src, &opts).is_err());
        let result = if c.rank() == 0 {
            write_mpi_chunked(&absent, dst.view(), &opts)
        } else {
            read_mpi_chunked_catalog(&absent, c, &opts).map(|_| ())
        };
        assert!(matches!(
            result,
            Err(pencil_io::IoError::CollectiveDescriptorMismatch)
        ));
    }
    assert!(!absent.exists());
    assert_eq!(std::fs::read(&existing).unwrap(), b"unchanged");
    assert!(dst.as_slice().iter().all(|&x| x == 99));

    // The boundary itself is legal. Add one unit extra axis to its otherwise
    // canonical file, shifting the header/end and every block offset together.
    let valid =
        PencilArray::from_elem(pencil, ExtraShape::new(vec![1; 1022]).unwrap(), 42u64).unwrap();
    let path = dir.join("rank-bound-header");
    write_mpi_chunked(&path, valid.view(), &opts).unwrap();
    assert!(read_mpi_chunked_catalog(&path, c, &opts).is_ok());
    if c.rank() == 0 {
        let mut b = std::fs::read(&path).unwrap();
        let header_len = u(&b, 2) + 8;
        let end = b.len() + 8;
        b.splice((12 + 2 + 1022) * 8..(12 + 2 + 1022) * 8, 1u64.to_le_bytes());
        for (i, value) in [(2, header_len), (3, end), (7, 1023)] {
            b[i * 8..i * 8 + 8].copy_from_slice(&(value as u64).to_le_bytes());
        }
        let records = 12 + 2 * 2 + 1023 + 2 * 2;
        for rank in 0..c.size() as usize {
            let i = records + rank * 8 + 7;
            let offset = u(&b, i) + 8;
            b[i * 8..i * 8 + 8].copy_from_slice(&(offset as u64).to_le_bytes());
        }
        std::fs::write(&path, b).unwrap();
    }
    c.barrier();
    let before = std::fs::read(&path).unwrap();
    let mut valid_dst = valid;
    assert!(read_mpi_chunked(&path, valid_dst.view_mut(), &opts).is_err());
    assert!(valid_dst.as_slice().iter().all(|&x| x == 42));
    assert!(read_mpi_chunked(&path, dst.view_mut(), &opts).is_err());
    assert!(dst.as_slice().iter().all(|&x| x == 99));
    assert!(read_mpi_chunked_catalog(&path, c, &opts).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), before);
}
#[test]
fn chunked() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let dir = owned_temp_dir(&world, "chunked");
    total_rank_bound(&world, &dir);
    macro_rules! real { ($($t:ty),*)=> { $(scalar(&world,&dir,|i|i as $t);)* }; }
    real!(i8, u8, i16, u16, i32, u32, i64, u64, f32, f64);
    scalar(&world, &dir, |i| {
        num_complex::Complex::new(i as f32 + 0.25, -(i as f32) - 0.5)
    });
    scalar(&world, &dir, |i| {
        num_complex::Complex::new(i as f64 + 0.25, -(i as f64) - 0.5)
    });
    let t = MpiTopology::new(&world, [world.size() as usize, 1]).unwrap();
    let c = t.communicator();
    let p = dir.join("corruption");
    let opts = MpiIoOptions::default();
    let src = array(t.clone(), 2, 42u64, [1, 0]);
    write_mpi_chunked(&p, src.view(), &opts).unwrap();
    // Broadcast a pristine image, then corrupt every metadata word, including
    // every rank's coordinates, ranges, local count and byte offset.
    let mut b = if c.rank() == 0 {
        std::fs::read(&p).unwrap()
    } else {
        Vec::new()
    };
    let mut len = b.len() as u64;
    c.process_at_rank(0).broadcast_into(&mut len);
    b.resize(len as usize, 0);
    c.process_at_rank(0).broadcast_into(&mut b);
    let mut dst = array(t.clone(), 2, 99u64, [1, 0]);
    for i in 0..u(&b, 2) / 8 {
        for value in [u64::MAX, 0] {
            if value == u(&b, i) as u64 {
                continue;
            }
            let mut bad = b.clone();
            bad[i * 8..i * 8 + 8].copy_from_slice(&value.to_le_bytes());
            replace(c, &p, &bad);
            assert!(
                read_mpi_chunked(&p, dst.view_mut(), &opts).is_err(),
                "word {i}"
            );
            assert!(dst.as_slice().iter().all(|&x| x == 99));
            assert!(
                read_mpi_chunked_catalog(&p, c, &opts).is_err(),
                "catalog word {i}"
            );
        }
    }
    // Offset overlap, header intrusion, gap, and out-of-bounds.
    let first_offset = 12 + 2 * 2 + 1 + 2 * 2 + 2 + 2 * 2 + 1;
    for off in [8, u(&b, 2) - 1, u(&b, 2) + 1, b.len() + 1] {
        let mut bad = b.clone();
        bad[first_offset * 8..first_offset * 8 + 8].copy_from_slice(&(off as u64).to_le_bytes());
        replace(c, &p, &bad);
        assert!(read_mpi_chunked(&p, dst.view_mut(), &opts).is_err());
        assert!(dst.as_slice().iter().all(|&x| x == 99));
    }
    for cut in [0, 7, 95, u(&b, 2) - 1, b.len() - 1] {
        replace(c, &p, &b[..cut]);
        assert!(read_mpi_chunked(&p, dst.view_mut(), &opts).is_err());
        assert!(dst.as_slice().iter().all(|&x| x == 99));
    }
    let mut trailing = b.clone();
    trailing.push(0);
    replace(c, &p, &trailing);
    assert!(read_mpi_chunked(&p, dst.view_mut(), &opts).is_err());
    replace(c, &p, &b);
    let mut wrong = array(t.clone(), 2, 99i64, [1, 0]);
    assert!(read_mpi_chunked(&p, wrong.view_mut(), &opts).is_err());
    assert!(wrong.as_slice().iter().all(|&x| x == 99));
    let mut wrong = array(t.clone(), 2, 99u64, [0, 1]);
    assert!(read_mpi_chunked(&p, wrong.view_mut(), &opts).is_err());
    assert!(wrong.as_slice().iter().all(|&x| x == 99));
    if world.size() > 1 {
        let other = MpiTopology::new(&world, [1, world.size() as usize]).unwrap();
        let same =
            Pencil::new_permuted(other, [3, 5], [1, 0], AxisPermutation::new([1, 0]).unwrap())
                .unwrap();
        assert_eq!(same.local_ranges(), src.pencil().local_ranges());
        let mut wrong = PencilArray::from_elem(same, ExtraShape::new([2]).unwrap(), 99u64).unwrap();
        assert!(read_mpi_chunked(&p, wrong.view_mut(), &opts).is_err());
        assert!(wrong.as_slice().iter().all(|&x| x == 99));
        let one_axis = MpiTopology::new(&world, [world.size() as usize]).unwrap();
        assert!(read_mpi_chunked_catalog(&p, one_axis.communicator(), &opts).is_err());
        let solo = world
            .split_by_color(mpi::topology::Color::with_value(world.rank()))
            .unwrap();
        let solo_topology = MpiTopology::new(&solo, [1, 1]).unwrap();
        assert!(read_mpi_chunked_catalog(&p, solo_topology.communicator(), &opts).is_err());
        let divergent = if c.rank() == 0 {
            MpiIoOptions::default()
        } else {
            MpiIoOptions::default().mode(MpiIoMode::Independent)
        };
        assert!(read_mpi_chunked(&p, dst.view_mut(), &divergent).is_err());
        let divergent = if c.rank() == 0 {
            MpiIoOptions::default()
        } else {
            MpiIoOptions::default().hint("cb_buffer_size", "1048576")
        };
        assert!(read_mpi_chunked(&p, dst.view_mut(), &divergent).is_err());
        // Legacy/new calls must agree on the original communicator before any
        // new-format branch or communicator duplication.
        if c.rank() == 0 {
            assert!(pencil_io::read_mpi(&p, dst.view_mut()).is_err());
        } else {
            assert!(read_mpi_chunked(&p, dst.view_mut(), &opts).is_err());
        }
        // Different operation and arity must stop at the common entry protocol.
        if c.rank() == 0 {
            assert!(read_mpi_chunked_catalog(&p, c, &opts).is_err());
        } else {
            assert!(read_mpi_chunked(&p, dst.view_mut(), &opts).is_err());
        }
        let p1 = Pencil::new(t.clone(), [3, 5, 1], [0, 1]).unwrap();
        let mut a1 = PencilArray::from_elem(p1, ExtraShape::new([2]).unwrap(), 99u64).unwrap();
        if world.rank() == 0 {
            assert!(read_mpi_chunked(&p, a1.view_mut(), &opts).is_err());
        } else {
            assert!(read_mpi_chunked(&p, dst.view_mut(), &opts).is_err());
        }
    }
    assert!(pencil_io::read_mpi(&p, dst.view_mut()).is_err());
    let invalid_options = MpiIoOptions::default().hint("bad\0key", "value");
    let invalid_path = dir.join("must-not-exist");
    assert!(write_mpi_chunked(&invalid_path, src.view(), &invalid_options).is_err());
    assert!(!invalid_path.exists());
    assert!(src.as_slice().iter().all(|&x| x == 42));
    assert!(dst.as_slice().iter().all(|&x| x == 99));
    read_mpi_chunked(&p, dst.view_mut(), &opts).unwrap();
    assert_eq!(dst.as_slice(), src.as_slice());
    cleanup_owned_temp_dir(&world, &dir);
}
