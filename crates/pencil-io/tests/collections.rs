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
    for count in [1, 2, 3] {
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
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| i32::from_le_bytes(*b))
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
        #[cfg(feature = "parallel-hdf5")]
        let file_values: Vec<_> = dest.iter().map(|a| a.as_slice().to_vec()).collect();
        for a in &mut dest {
            a.as_mut_slice().fill(-321);
        }
        let sentinel: Vec<_> = dest.iter().map(|a| a.as_slice().to_vec()).collect();
        let mut bad_destination = PencilArray::from_elem(
            reader.clone(),
            ExtraShape::new([if world.rank() == 0 { 3 } else { 2 }]).unwrap(),
            -777i32,
        )
        .unwrap();
        let bad_before = bad_destination.as_slice().to_vec();
        // Replace the last member with a differently shaped one on rank 0, so a
        // later member is rejected before the first destination is touched.
        let mut bad_views: Vec<_> = dest.iter_mut().map(|a| a.view_mut()).collect();
        bad_views.pop();
        bad_views.push(bad_destination.view_mut());
        assert!(read_mpi_collection(&path, comm, &mut bad_views).is_err());
        assert_eq!(bad_destination.as_slice(), bad_before);
        for (a, b) in dest.iter().zip(&sentinel) {
            assert_eq!(a.as_slice(), b);
        }
        assert!(
            read_mpi_collection(
                dir.join("missing"),
                comm,
                &mut dest.iter_mut().map(|a| a.view_mut()).collect::<Vec<_>>()
            )
            .is_err()
        );
        for (a, b) in dest.iter().zip(&sentinel) {
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
            for (a, b) in dest.iter().zip(&file_values) {
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
            for (a, b) in dest.iter().zip(&file_values) {
                assert_eq!(a.as_slice(), b);
            }
        }
    }

    // Multiple extra axes and a non-self-inverse three-axis permutation.
    {
        const COUNT: usize = 3;
        let extra = ExtraShape::new([2, 3]).unwrap();
        let writer_pencil = Pencil::<3, 2>::new(topology.clone(), [2, 3, 5], [0, 1]).unwrap();
        let permuted_reader = Pencil::<3, 2>::new_permuted(
            topology.clone(),
            [2, 3, 5],
            [0, 1],
            AxisPermutation::new([1, 2, 0]).unwrap(),
        )
        .unwrap();
        let mut source: Vec<_> = (0..COUNT)
            .map(|_| PencilArray::from_elem(writer_pencil.clone(), extra.clone(), 0i32).unwrap())
            .collect();
        fill_permuted_collection(&mut source);
        let mut dest: Vec<_> = (0..COUNT)
            .map(|_| {
                PencilArray::from_elem(permuted_reader.clone(), extra.clone(), -11i32).unwrap()
            })
            .collect();
        let path = dir.join("permuted-collection.pio");
        write_mpi_collection(
            &path,
            comm,
            &source.iter().map(|a| a.view()).collect::<Vec<_>>(),
        )
        .unwrap();
        read_mpi_collection(
            &path,
            comm,
            &mut dest.iter_mut().map(|a| a.view_mut()).collect::<Vec<_>>(),
        )
        .unwrap();
        assert_permuted_collection(&dest);
        #[cfg(feature = "parallel-hdf5")]
        {
            let hpath = dir.join("permuted-collection.h5");
            pencil_io::write_hdf5_collection(
                &hpath,
                comm,
                &source.iter().map(|a| a.view()).collect::<Vec<_>>(),
            )
            .unwrap();
            for a in &mut dest {
                a.as_mut_slice().fill(-12);
            }
            pencil_io::read_hdf5_collection(
                &hpath,
                comm,
                &mut dest.iter_mut().map(|a| a.view_mut()).collect::<Vec<_>>(),
            )
            .unwrap();
            assert_permuted_collection(&dest);
        }
    }

    // A zero extra extent produces an empty collection payload on every path.
    {
        let extra = ExtraShape::new([0]).unwrap();
        let count = 2;
        let source: Vec<_> = (0..count)
            .map(|_| PencilArray::from_elem(pencil.clone(), extra.clone(), 7i32).unwrap())
            .collect();
        let mut dest: Vec<_> = (0..count)
            .map(|_| PencilArray::from_elem(reader.clone(), extra.clone(), -9i32).unwrap())
            .collect();
        let path = dir.join("zero-collection.pio");
        write_mpi_collection(
            &path,
            comm,
            &source.iter().map(|a| a.view()).collect::<Vec<_>>(),
        )
        .unwrap();
        read_mpi_collection(
            &path,
            comm,
            &mut dest.iter_mut().map(|a| a.view_mut()).collect::<Vec<_>>(),
        )
        .unwrap();
        assert!(dest.iter().all(|a| a.is_empty()));
        #[cfg(feature = "parallel-hdf5")]
        {
            let hpath = dir.join("zero-collection.h5");
            pencil_io::write_hdf5_collection(
                &hpath,
                comm,
                &source.iter().map(|a| a.view()).collect::<Vec<_>>(),
            )
            .unwrap();
            pencil_io::read_hdf5_collection(
                &hpath,
                comm,
                &mut dest.iter_mut().map(|a| a.view_mut()).collect::<Vec<_>>(),
            )
            .unwrap();
            assert!(dest.iter().all(|a| a.is_empty()));
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

fn permuted_collection_value(
    component: usize,
    e: usize,
    f: usize,
    x: usize,
    y: usize,
    z: usize,
) -> i32 {
    (component * 1_000_000 + e * 100_000 + f * 10_000 + x * 100 + y * 10 + z) as i32
}

fn fill_permuted_collection(source: &mut [PencilArray<i32, 3, 2>]) {
    for (component, array) in source.iter_mut().enumerate() {
        let mut view = array.view_mut();
        let ranges = view.pencil().local_ranges().clone();
        for e in 0..2 {
            for f in 0..3 {
                for x in 0..ranges[0].len() {
                    for y in 0..ranges[1].len() {
                        for z in 0..ranges[2].len() {
                            *view.get_local_mut(&[e, f], [x, y, z]).unwrap() =
                                permuted_collection_value(
                                    component,
                                    e,
                                    f,
                                    ranges[0].start + x,
                                    ranges[1].start + y,
                                    ranges[2].start + z,
                                );
                        }
                    }
                }
            }
        }
    }
}

fn assert_permuted_collection(dest: &[PencilArray<i32, 3, 2>]) {
    for (component, array) in dest.iter().enumerate() {
        let view = array.view();
        let ranges = view.pencil().local_ranges().clone();
        for e in 0..2 {
            for f in 0..3 {
                for x in 0..ranges[0].len() {
                    for y in 0..ranges[1].len() {
                        for z in 0..ranges[2].len() {
                            assert_eq!(
                                view.get_local(&[e, f], [x, y, z]),
                                Some(&permuted_collection_value(
                                    component,
                                    e,
                                    f,
                                    ranges[0].start + x,
                                    ranges[1].start + y,
                                    ranges[2].start + z,
                                ))
                            );
                        }
                    }
                }
            }
        }
    }
}
