use mpi::traits::*;
use pencil_array::{
    AxisPermutation, ExtraShape, MpiTopology, Pencil, PencilArrayView, PencilArrayViewMut,
};
use pencil_io::{
    MpiFileSession, MpiIoOptions, read_mpi, read_mpi_chunked, write_mpi, write_mpi_chunked,
};

mod support;
use support::{cleanup_owned_temp_dir, owned_temp_dir};

#[test]
fn external_buffers_across_io_backends() {
    let universe = mpi::initialize().expect("one MPI initialization");
    let world = universe.world();
    let topology = MpiTopology::<1>::new(&world, [world.size() as usize]).unwrap();
    let pencil = Pencil::<2, 1>::new_permuted(
        topology.clone(),
        [world.size() as usize + 1, 3],
        [0],
        AxisPermutation::new([1, 0]).unwrap(),
    )
    .unwrap();
    let extra = ExtraShape::new([2]).unwrap();
    let shape = pencil.local_ranges();
    let len = 2 * shape[0].len() * shape[1].len();
    let mut source = vec![0i64; len];
    {
        let mut view = PencilArrayViewMut::from_slice_mut(&pencil, &extra, &mut source).unwrap();
        for e in 0..2 {
            for x in 0..shape[0].len() {
                for y in 0..shape[1].len() {
                    *view.get_local_mut(&[e], [x, y]).unwrap() =
                        (100 * e + 10 * (shape[0].start + x) + shape[1].start + y) as i64;
                }
            }
        }
    }
    let source_before = source.clone();
    let source_ptr = source.as_ptr();
    let mut destination = vec![-999i64; len];
    let destination_ptr = destination.as_ptr();
    let directory = owned_temp_dir(&world, "external-views-io");
    let options = MpiIoOptions::default();
    for chunked in [false, true] {
        let path = directory.join(if chunked {
            "chunked.pio"
        } else {
            "canonical.pio"
        });
        let view = PencilArrayView::from_slice(&pencil, &extra, &source).unwrap();
        if chunked {
            write_mpi_chunked(&path, view, &options).unwrap();
        } else {
            write_mpi(&path, view).unwrap();
        }
        destination.fill(-999);
        let view = PencilArrayViewMut::from_slice_mut(&pencil, &extra, &mut destination).unwrap();
        if chunked {
            read_mpi_chunked(&path, view, &options).unwrap();
        } else {
            read_mpi(&path, view).unwrap();
        }
        assert_eq!(destination, source);
    }
    let path = directory.join("session.pio");
    let mut session = MpiFileSession::create(topology.communicator(), &path, &options).unwrap();
    session
        .write_named(
            "borrowed",
            PencilArrayView::from_slice(&pencil, &extra, &source).unwrap(),
        )
        .unwrap();
    destination.fill(-999);
    session
        .read_named(
            "borrowed",
            PencilArrayViewMut::from_slice_mut(&pencil, &extra, &mut destination).unwrap(),
        )
        .unwrap();
    assert_eq!(destination, source);
    session.close().unwrap();

    #[cfg(feature = "parallel-hdf5")]
    {
        use pencil_io::{
            Hdf5FileSession, Hdf5ReadOptions, Hdf5WriteOptions, read_hdf5, write_hdf5,
        };
        let path = directory.join("canonical.h5");
        write_hdf5(
            &path,
            PencilArrayView::from_slice(&pencil, &extra, &source).unwrap(),
        )
        .unwrap();
        destination.fill(-999);
        read_hdf5(
            &path,
            PencilArrayViewMut::from_slice_mut(&pencil, &extra, &mut destination).unwrap(),
        )
        .unwrap();
        assert_eq!(destination, source);
        let path = directory.join("session.h5");
        let mut session =
            Hdf5FileSession::create(&path, topology.communicator(), &options).unwrap();
        session.create_group("external").unwrap();
        let filters = Hdf5WriteOptions::default()
            .chunks(vec![1, 1, 2])
            .shuffle(true)
            .deflate(1);
        session
            .write(
                "external/buffer",
                PencilArrayView::from_slice(&pencil, &extra, &source).unwrap(),
                &filters,
            )
            .unwrap();
        session.close().unwrap();
        let mut session =
            Hdf5FileSession::open_read(&path, topology.communicator(), &options).unwrap();
        destination.fill(-999);
        session
            .read(
                "external/buffer",
                PencilArrayViewMut::from_slice_mut(&pencil, &extra, &mut destination).unwrap(),
                &Hdf5ReadOptions::default(),
            )
            .unwrap();
        assert_eq!(destination, source);
        session.close().unwrap();
    }
    assert_eq!(source, source_before);
    assert_eq!(source.as_ptr(), source_ptr);
    assert_eq!(destination.as_ptr(), destination_ptr);
    cleanup_owned_temp_dir(&world, &directory);
}
