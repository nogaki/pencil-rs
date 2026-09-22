use mpi::traits::*;
use pencil_array::{AxisPermutation, ExtraShape, MpiTopology, Pencil, PencilArray};
use pencil_io::{read_mpi_collection, write_mpi_collection};
mod support;

#[test]
fn collections_are_single_payloads_and_atomic_reads() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let dir = support::owned_temp_dir(&world, "io-collections");
    let topology = MpiTopology::<2>::new(&world, [world.size() as usize, 1]).unwrap();
    let pencil = Pencil::<2, 2>::new(topology.clone(), [3, 5], [0, 1]).unwrap();
    let reader = Pencil::<2, 2>::new_permuted(
        topology.clone(),
        [3, 5],
        [0, 1],
        AxisPermutation::new([1, 0]).unwrap(),
    )
    .unwrap();
    let comm = topology.communicator();
    for count in [2, 3] {
        let mut source: Vec<_> = (0..count)
            .map(|_| {
                PencilArray::from_elem(pencil.clone(), ExtraShape::new([2]).unwrap(), 0i32).unwrap()
            })
            .collect();
        for (c, a) in source.iter_mut().enumerate() {
            let mut v = a.view_mut();
            let r = v.pencil().local_ranges().clone();
            for e in 0..2 {
                for x in 0..r[0].len() {
                    for y in 0..r[1].len() {
                        *v.get_local_mut(&[e], [x, y]).unwrap() =
                            (c * 1000 + e * 100 + (x + r[0].start) * 5 + y + r[1].start) as i32;
                    }
                }
            }
        }
        let preserved: Vec<_> = source.iter().map(|a| a.as_slice().to_vec()).collect();
        let views: Vec<_> = source.iter().map(|a| a.view()).collect();
        let mut dest: Vec<_> = (0..count)
            .map(|_| {
                PencilArray::from_elem(reader.clone(), ExtraShape::new([2]).unwrap(), -9i32)
                    .unwrap()
            })
            .collect();
        let path = dir.join(format!("collection-{count}.pio"));
        write_mpi_collection(&path, comm, &views).unwrap();
        let oracle: Vec<i32> = (0..count)
            .flat_map(|c| {
                (0..2).flat_map(move |e| (0..15).map(move |i| (c * 1000 + e * 100 + i) as i32))
            })
            .collect();
        world.barrier();
        if world.rank() == 0 {
            let bytes = std::fs::read(&path).unwrap();
            let offset = u64::from_le_bytes(bytes[64..72].try_into().unwrap()) as usize;
            let actual: Vec<_> = bytes[offset..]
                .chunks_exact(4)
                .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            assert_eq!(actual, oracle);
        }
        world.barrier();
        read_mpi_collection(
            &path,
            comm,
            &mut dest.iter_mut().map(|a| a.view_mut()).collect::<Vec<_>>(),
        )
        .unwrap();
        for (c, a) in dest.iter().enumerate() {
            let v = a.view();
            let r = v.pencil().local_ranges();
            for e in 0..2 {
                for x in 0..r[0].len() {
                    for y in 0..r[1].len() {
                        assert_eq!(
                            v.get_local(&[e], [x, y]),
                            Some(
                                &((c * 1000 + e * 100 + (x + r[0].start) * 5 + y + r[1].start)
                                    as i32)
                            )
                        );
                    }
                }
            }
        }
        if world.size() > 1 {
            let n = if world.rank() == 0 { count - 1 } else { count };
            assert!(write_mpi_collection(dir.join("count-bad.pio"), comm, &views[..n]).is_err());
            let mismatch = if world.rank() == 0 {
                pencil_io::read_mpi(&path, dest[0].view_mut()).is_err()
            } else {
                read_mpi_collection(
                    &path,
                    comm,
                    &mut dest.iter_mut().map(|a| a.view_mut()).collect::<Vec<_>>(),
                )
                .is_err()
            };
            assert!(mismatch);
        }
        let bad = PencilArray::from_elem(
            pencil.clone(),
            ExtraShape::new([if world.rank() == 0 { 3 } else { 2 }]).unwrap(),
            0i32,
        )
        .unwrap();
        assert!(
            write_mpi_collection(
                dir.join("member-bad.pio"),
                comm,
                &[source[0].view(), bad.view()]
            )
            .is_err()
        );
        assert!(write_mpi_collection::<_, i32, 2, 2>(&path, comm, &[]).is_err());
        let before: Vec<_> = dest.iter().map(|a| a.as_slice().to_vec()).collect();
        let mut bad_destination = PencilArray::from_elem(
            reader.clone(),
            ExtraShape::new([if world.rank() == 0 { 3 } else { 2 }]).unwrap(),
            -777i32,
        )
        .unwrap();
        let bad_before = bad_destination.as_slice().to_vec();
        assert!(
            read_mpi_collection(
                &path,
                comm,
                &mut [dest[0].view_mut(), bad_destination.view_mut()]
            )
            .is_err()
        );
        assert_eq!(bad_destination.as_slice(), bad_before);
        assert!(
            read_mpi_collection(
                dir.join("missing"),
                comm,
                &mut dest.iter_mut().map(|a| a.view_mut()).collect::<Vec<_>>()
            )
            .is_err()
        );
        for (a, b) in dest.iter().zip(&before) {
            assert_eq!(a.as_slice(), b);
        }
        for (a, b) in source.iter().zip(&preserved) {
            assert_eq!(a.as_slice(), b);
        }
        #[cfg(feature = "parallel-hdf5")]
        {
            let hpath = dir.join(format!("collection-{count}.h5"));
            pencil_io::write_hdf5_collection(&hpath, comm, &views).unwrap();
            world.barrier();
            if world.rank() == 0 {
                let f = hdf5_metno::File::open(&hpath).unwrap();
                let d = f.dataset("/pencil_io_v1/data").unwrap();
                assert_eq!(d.shape(), vec![count, 2, 3, 5]);
                assert_eq!(d.read_raw::<i32>().unwrap(), oracle);
            }
            world.barrier();
            for a in &mut dest {
                a.as_mut_slice().fill(-1);
            }
            pencil_io::read_hdf5_collection(
                &hpath,
                comm,
                &mut dest.iter_mut().map(|a| a.view_mut()).collect::<Vec<_>>(),
            )
            .unwrap();
            for (a, b) in dest.iter().zip(&before) {
                assert_eq!(a.as_slice(), b);
            }
            assert!(
                pencil_io::read_hdf5_collection(
                    dir.join("missing.h5"),
                    comm,
                    &mut dest.iter_mut().map(|a| a.view_mut()).collect::<Vec<_>>()
                )
                .is_err()
            );
            for (a, b) in dest.iter().zip(&before) {
                assert_eq!(a.as_slice(), b);
            }
        }
    }
    // A malformed existing payload must preserve every member, not only the first.
    let mut dest: Vec<_> = (0..3)
        .map(|_| {
            PencilArray::from_elem(reader.clone(), ExtraShape::new([2]).unwrap(), -123i32).unwrap()
        })
        .collect();
    let corrupt = dir.join("collection-3.pio");
    if world.rank() == 0 {
        std::fs::OpenOptions::new()
            .write(true)
            .open(&corrupt)
            .unwrap()
            .set_len(100)
            .unwrap();
    }
    world.barrier();
    assert!(
        read_mpi_collection(
            &corrupt,
            comm,
            &mut dest.iter_mut().map(|a| a.view_mut()).collect::<Vec<_>>()
        )
        .is_err()
    );
    assert!(dest.iter().all(|a| a.as_slice().iter().all(|&x| x == -123)));
    support::cleanup_owned_temp_dir(&world, &dir);
}
