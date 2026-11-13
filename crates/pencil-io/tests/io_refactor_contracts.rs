//! Small source-level contracts and behavioral regressions for the I/O
//! refactors (#109-#112).
//!
//! The source contracts compile without MPI or HDF5. They only pin duplication
//! that a featureless integration test cannot observe (extra collective
//! agreement, copies of the record-commit ordering, per-backend metadata
//! rules). Mismatch rejection and destination preservation stay covered by the
//! existing MPI/HDF5 suites.
//!
//! Error variants are behavioral, not source-level: the same MPI-initialized
//! test below also pins the #110 conversion boundary of the shared named
//! payload helper. A failed session payload read must stay
//! `MpiSessionError::Io(..)`; re-wrapping it as
//! `MpiSessionError::Named(NamedIoError::Io(..))` changes the public API.
//!
//! The HDF5 regression below runs under `parallel-hdf5` and reproduces the two
//! #111 metadata defects with real files: (1) a shape whose final product is
//! zero must be accepted even when an earlier partial product would overflow,
//! identically in the serial, parallel, catalog and session readers, while a
//! nonzero overflow stays rejected; (2) a `H5S_NULL` u64 attribute must be
//! rejected as malformed instead of being mistaken for a scalar.

/// Collective hand-off primitives a caller has already executed.
/// `agree_named_options` is excluded (it decides internally).
const AGREEMENT_TOKENS: [&str; 6] = [
    "descriptor_agreement(",
    "agree_decomposition(",
    "agree_options(",
    "agree_name(",
    "name_bytes(",
    "named_link(",
];

/// Words that an existing hand-off parameter uses to say "already agreed".
const AGREED_WORDS: [&str; 4] = ["agreed", "already", "skip", "fresh"];

/// Top-level `fn` items as `(name, source)`, sliced to the next column-zero `}`.
fn top_level_fns(src: &str) -> Vec<(String, String)> {
    let lines: Vec<&str> = src.lines().collect();
    let mut out = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let Some(at) = line.find("fn ") else { continue };
        if !matches!(&line[..at], "" | "pub " | "pub(crate) " | "pub(super) ") {
            continue;
        }
        let name: String = line[at + 3..]
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        let close = (i..lines.len())
            .find(|&j| lines[j] == "}")
            .unwrap_or(lines.len() - 1);
        out.push((name, lines[i..=close].join("\n")));
    }
    out
}

fn item<'a>(fns: &'a [(String, String)], name: &str) -> &'a str {
    let found = fns
        .iter()
        .find(|(candidate, _)| candidate == name)
        .unwrap_or_else(|| panic!("missing top-level fn `{name}`"));
    &found.1
}

#[test]
fn session_agrees_fixed_options_once_and_keeps_state_checks() {
    const SESSION: &str = include_str!("../src/named_mpi/session.rs");

    // #112: creation agrees the fixed options/hints; operations reuse them.
    assert_eq!(
        SESSION.matches("agree_options(").count(),
        1,
        "session operations must not re-agree the fixed options/hints"
    );
    let agreement = SESSION.find("agree_options(").expect("one agreement");
    assert!(
        SESSION.find("fn open_session").unwrap() < agreement
            && agreement < SESSION.find("Ok(Self {").unwrap(),
        "the single options agreement must be the session constructor"
    );
    assert!(
        !SESSION.contains("&self.options"),
        "session operations must not pass the retained options into an agreement call"
    );

    // The per-operation session state and decomposition-axis checks stay.
    assert!(
        [
            "self.id",
            "self.poisoned",
            "self.readonly",
            "decomposition()"
        ]
        .iter()
        .all(|retained| SESSION.contains(retained)),
        "per-operation session state/axis checks must be retained"
    );
}

#[test]
fn collection_and_named_layers_do_not_reagree_pre_agreed_inputs() {
    const MPI: &str = include_str!("../src/mpi_io.rs");
    const NAMED: &str = include_str!("../src/named_mpi.rs");
    const HDF5: &str = include_str!("../src/hdf5_io.rs");
    // Inner layers reached through the collection/named hand-off. The caller
    // already agreed descriptor, decomposition, options and name, so an
    // execution here must be conditional on that hand-off state.
    let targets = [
        ("mpi_io.rs", "write_mpi_inner"),
        ("mpi_io.rs", "read_mpi_inner"),
        ("named_mpi.rs", "write_named"),
        ("named_mpi.rs", "read_named_inner"),
        ("hdf5_io.rs", "write_hdf5_inner"),
        ("hdf5_io.rs", "read_hdf5_inner"),
        ("hdf5_io.rs", "write_hdf5_named_with_options_inner"),
        ("hdf5_io.rs", "append_hdf5_named_with_options_inner"),
        ("hdf5_io.rs", "read_named_with_options"),
    ];

    let mut diagnostics = Vec::new();
    for (file, name) in targets {
        let src = match file {
            "mpi_io.rs" => MPI,
            "named_mpi.rs" => NAMED,
            _ => HDF5,
        };
        let fns = top_level_fns(src);
        let lines: Vec<&str> = item(&fns, name).lines().collect();
        for (i, line) in lines.iter().enumerate() {
            if !AGREEMENT_TOKENS.iter().any(|token| line.contains(token)) {
                continue;
            }
            let guard = lines[i.saturating_sub(3)..=i].join(" ");
            let agreed = guard.contains("if ") && AGREED_WORDS.iter().any(|w| guard.contains(w));
            if !agreed {
                diagnostics.push(format!("{file}:{name}: {}", line.trim()));
            }
        }
    }
    assert!(
        diagnostics.is_empty(),
        "collection/named hand-off re-executes agreements the caller already made (#109):\n{}",
        diagnostics.join("\n")
    );
}

#[test]
fn single_named_and_session_share_payload_and_commit_procedure() {
    const MPI: &str = include_str!("../src/mpi_io.rs");
    const NAMED: &str = include_str!("../src/named_mpi.rs");
    const SESSION: &str = include_str!("../src/named_mpi/session.rs");
    let all = format!("{MPI}\n{NAMED}\n{SESSION}");

    // #110: one payload transfer and one commit marker serve the single,
    // named and session records.
    for token in [
        "COMMIT_MARKER.to_le_bytes()",
        "ffi::file_write_independent(",
        "ffi::file_write_all(",
        "ffi::file_read_independent(",
        "ffi::file_read_all(",
    ] {
        let count = all.matches(token).count();
        assert_eq!(
            count, 1,
            "`{token}` must live in one shared payload/record helper, found {count}"
        );
    }

    // The shared record writer keeps the publish order: payload, payload flush,
    // commit marker, marker flush.
    let record = [MPI, NAMED, SESSION]
        .into_iter()
        .flat_map(top_level_fns)
        .find(|(_, body)| {
            body.contains("COMMIT_MARKER.to_le_bytes()") && body.contains("ffi::file_write_all(")
        })
        .map(|(_, body)| body)
        .expect("one shared record writer ordering payload and commit marker");
    let at = |token: &str| {
        record
            .find(token)
            .unwrap_or_else(|| panic!("missing {token}"))
    };
    let payload = at("ffi::file_write_all(");
    let marker = at("COMMIT_MARKER.to_le_bytes()");
    let payload_flush = payload + record[payload..].find("file_sync(").expect("payload flush");
    let marker_flush = record[marker..].find("file_sync(").expect("marker flush");
    assert!(
        payload_flush < marker && marker_flush > 0,
        "record publish order must stay payload -> flush -> commit marker -> flush"
    );
}

#[test]
fn hdf5_readers_use_one_metadata_acceptance_predicate() {
    const SCHEMA: &str = include_str!("../src/hdf5_schema.rs");
    const SERIAL: &str = include_str!("../src/serial_hdf5.rs");
    const PARALLEL: &str = include_str!("../src/hdf5_io.rs");
    const SESSION: &str = include_str!("../src/hdf5_io/session/tree.rs");

    // #111: one schema predicate carries the commit, shape and provenance
    // acceptance rules instead of per-reader copies.
    let predicate = top_level_fns(SCHEMA)
        .into_iter()
        .find(|(_, body)| {
            body.contains("commit_state(")
                && body.contains("checked_mul")
                && [
                    "valid_writer_grid(",
                    "valid_permutation(",
                    "valid_writer_provenance(",
                ]
                .iter()
                .any(|token| body.contains(token))
        })
        .map(|(name, _)| name)
        .expect(
            "hdf5_schema must expose one metadata acceptance predicate (commit + shape + provenance)",
        );

    // The serial, parallel (including the HDF5 catalog) and session readers all
    // apply that same predicate, so a corrupt attribute is accepted/rejected
    // identically on each path.
    for (file, src) in [
        ("serial_hdf5.rs", SERIAL),
        ("hdf5_io.rs", PARALLEL),
        ("hdf5_io/session/tree.rs", SESSION),
    ] {
        assert!(
            src.contains(&format!("{predicate}(")),
            "{file} must use the shared `{predicate}` metadata predicate"
        );
        assert!(
            !src.contains("fn valid_writer_provenance("),
            "{file}: writer-provenance acceptance must be defined once, in hdf5_schema"
        );
    }
}

#[cfg(feature = "parallel-hdf5")]
mod hdf5_metadata_regressions {
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    use hdf5_metno::{Dataset, Extents, File};
    use mpi::collective::{CommunicatorCollectives, SystemOperation};
    use mpi::traits::*;
    use pencil_array::{ExtraShape, MpiTopology, Pencil, PencilArray};
    use pencil_io::{
        Hdf5FileSession, IoError, MpiFileSession, MpiIoOptions, MpiSessionError, ScalarType,
        read_hdf5, read_hdf5_catalog, read_hdf5_serial,
    };

    const COMMIT_MARKER: u64 = 0x434f_4d4d_4954_5445;

    /// Runs a root-only fixture operation; every rank panics together when it
    /// fails, matching the existing HDF5 test fixtures.
    fn root_status<C, F>(world: &C, operation: F)
    where
        C: Communicator + CommunicatorCollectives,
        F: FnOnce() -> Result<(), String>,
    {
        let result = if world.rank() == 0 {
            operation()
        } else {
            Ok(())
        };
        let local_ok = i32::from(result.is_ok());
        let mut all_ok = 0;
        world.all_reduce_into(&local_ok, &mut all_ok, SystemOperation::min());
        if all_ok != 1 {
            panic!(
                "root fixture operation failed: {}",
                result.err().unwrap_or_default()
            );
        }
    }

    fn owned_temp_dir<C>(world: &C, prefix: &str) -> PathBuf
    where
        C: Communicator + CommunicatorCollectives,
    {
        let setup = if world.rank() == 0 {
            (|| -> Result<String, String> {
                let nanos = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map_err(|error| error.to_string())?
                    .as_nanos();
                let path =
                    std::env::temp_dir().join(format!("{prefix}-{nanos}-{}", std::process::id()));
                std::fs::create_dir(&path)
                    .map(|()| path.to_string_lossy().into_owned())
                    .map_err(|error| error.to_string())
            })()
        } else {
            Ok(String::new())
        };
        let mut status = i32::from(setup.is_ok());
        world.process_at_rank(0).broadcast_into(&mut status);
        let mut all_ok = 0;
        world.all_reduce_into(&status, &mut all_ok, SystemOperation::min());
        if all_ok != 1 {
            panic!(
                "owned test directory setup failed: {}",
                setup.err().unwrap_or_default()
            );
        }
        let mut path = if world.rank() == 0 {
            setup.expect("directory setup agreement").into_bytes()
        } else {
            Vec::new()
        };
        let mut length = i32::try_from(path.len()).expect("temporary path fits MPI count");
        world.process_at_rank(0).broadcast_into(&mut length);
        path.resize(
            usize::try_from(length).expect("non-negative path length"),
            0,
        );
        world.process_at_rank(0).broadcast_into(&mut path[..]);
        PathBuf::from(String::from_utf8(path).expect("temporary path is UTF-8"))
    }

    fn cleanup_owned_temp_dir<C>(world: &C, path: &Path)
    where
        C: Communicator + CommunicatorCollectives,
    {
        world.barrier();
        let cleanup = if world.rank() == 0 {
            std::fs::remove_dir_all(path).map_err(|error| error.to_string())
        } else {
            Ok(())
        };
        let mut status = i32::from(cleanup.is_ok());
        world.process_at_rank(0).broadcast_into(&mut status);
        let mut all_ok = 0;
        world.all_reduce_into(&status, &mut all_ok, SystemOperation::min());
        if all_ok != 1 {
            panic!(
                "owned test directory cleanup failed: {}",
                cleanup.err().unwrap_or_default()
            );
        }
        world.barrier();
    }

    fn h5<T>(result: hdf5_metno::Result<T>) -> Result<T, String> {
        result.map_err(|error| error.to_string())
    }

    fn write_u64_attr(dataset: &Dataset, name: &str, values: &[u64]) -> Result<(), String> {
        let _ = dataset.delete_attr(name);
        if values.len() == 1 {
            let attribute = h5(dataset.new_attr::<u64>().create(name))?;
            h5(attribute.as_writer().write_scalar(&values[0]))
        } else {
            let attribute = h5(dataset.new_attr::<u64>().shape(values.len()).create(name))?;
            h5(attribute.as_writer().write_raw(values))
        }
    }

    /// Creates a chunked f64 dataset through the native HDF5 API. Chunked
    /// storage is metadata-only until a chunk is written, and avoiding the
    /// high-level builder keeps `Extents::size` from multiplying a deliberately
    /// overflowing metadata shape.
    fn create_chunked_f64_dataset(
        group: &hdf5_metno::Group,
        name: &str,
        shape: &[usize],
    ) -> Result<Dataset, String> {
        use hdf5_metno_sys::{h5d, h5p, h5s};

        let dims: Vec<u64> = shape.iter().map(|&extent| extent as u64).collect();
        let chunks: Vec<u64> = vec![1; dims.len()];
        hdf5_metno::sync::sync(|| {
            let name = std::ffi::CString::new(name).map_err(|error| error.to_string())?;
            // SAFETY: dims/chunks are rank-sized arrays; every handle returned
            // by HDF5 is closed below, and the dataset is RAII-owned after
            // from_id.
            unsafe {
                let space =
                    h5s::H5Screate_simple(dims.len() as i32, dims.as_ptr(), std::ptr::null());
                if space < 0 {
                    return Err("H5Screate_simple".to_owned());
                }
                let dcpl = h5p::H5Pcreate(*hdf5_metno::globals::H5P_DATASET_CREATE);
                if dcpl < 0 {
                    let _ = h5s::H5Sclose(space);
                    return Err("H5Pcreate".to_owned());
                }
                let chunk_code = h5p::H5Pset_chunk(dcpl, dims.len() as i32, chunks.as_ptr());
                let did = h5d::H5Dcreate2(
                    group.id(),
                    name.as_ptr(),
                    *hdf5_metno::globals::H5T_IEEE_F64LE,
                    space,
                    h5p::H5P_DEFAULT,
                    dcpl,
                    h5p::H5P_DEFAULT,
                );
                let _ = h5p::H5Pclose(dcpl);
                let _ = h5s::H5Sclose(space);
                if chunk_code < 0 || did < 0 {
                    return Err(format!("H5Dcreate2 chunk={chunk_code} id={did}"));
                }
                hdf5_metno::from_id(did).map_err(|error| error.to_string())
            }
        })
    }

    /// Writes one v1 HDF5 dataset with explicit metadata. Shapes are metadata
    /// only: the dataset uses chunked storage and is never written.
    fn write_fixture(
        path: &Path,
        group_name: &str,
        dataset_name: &str,
        extra: &[u64],
        global: &[u64],
        grid: &[u64],
    ) -> Result<(), String> {
        let file = h5(File::create(path))?;
        let group = h5(file.create_group(group_name))?;
        let shape: Vec<usize> = extra
            .iter()
            .chain(global)
            .map(|&extent| usize::try_from(extent).map_err(|error| error.to_string()))
            .collect::<Result<_, _>>()?;
        let dataset = create_chunked_f64_dataset(&group, dataset_name, &shape)?;
        let rank = global.len() as u64;
        let permutation: Vec<u64> = (0..rank).collect();
        write_u64_attr(&dataset, "pencil_io_version", &[1])?;
        write_u64_attr(&dataset, "pencil_io_commit", &[COMMIT_MARKER])?;
        write_u64_attr(&dataset, "pencil_io_n", &[rank])?;
        write_u64_attr(&dataset, "pencil_io_type", &[ScalarType::F64.code()])?;
        write_u64_attr(
            &dataset,
            "pencil_io_width",
            &[ScalarType::F64.width() as u64],
        )?;
        write_u64_attr(&dataset, "pencil_io_extra_rank", &[extra.len() as u64])?;
        if !extra.is_empty() {
            write_u64_attr(&dataset, "pencil_io_extra_shape", extra)?;
        }
        write_u64_attr(&dataset, "pencil_io_global_shape", global)?;
        write_u64_attr(&dataset, "pencil_io_writer_grid", grid)?;
        write_u64_attr(&dataset, "pencil_io_writer_permutation", &permutation)?;
        h5(file.flush())?;
        h5(file.close())
    }

    /// Replaces one u64 attribute with an `H5S_NULL` dataspace of the same
    /// type. Reading it must be rejected as malformed, not treated as a scalar
    /// zero.
    fn replace_with_null_dataspace(
        path: &Path,
        group_name: &str,
        dataset_name: &str,
        attribute: &str,
    ) -> Result<(), String> {
        let file = h5(File::open_rw(path))?;
        let group = h5(file.group(group_name))?;
        let dataset = h5(group.dataset(dataset_name))?;
        let _ = dataset.delete_attr(attribute);
        let null = h5(dataset
            .new_attr::<u64>()
            .shape(Extents::null())
            .create(attribute))?;
        drop(null);
        h5(file.flush())?;
        h5(file.close())
    }

    /// Opens a session tree, catalogs it, and always closes it collectively
    /// before returning. Dropping an open session aborts by design.
    fn session_catalog(
        path: &Path,
        comm: &mpi::topology::CartesianCommunicator,
    ) -> Result<Vec<pencil_io::DatasetInfo>, String> {
        let mut session = Hdf5FileSession::open_read(path, comm, &MpiIoOptions::default())
            .map_err(|error| error.to_string())?;
        let catalog = session.catalog().map_err(|error| error.to_string());
        let closed = session.close().map_err(|error| error.to_string());
        let catalog = catalog?;
        closed?;
        Ok(catalog)
    }

    /// Runs a rank-zero operation and broadcasts its success flag so every
    /// rank reaches the same assertion and the same cleanup barrier.
    fn root_ok<C, F>(world: &C, operation: F) -> bool
    where
        C: Communicator + CommunicatorCollectives,
        F: FnOnce() -> Result<(), String>,
    {
        let mut ok = i32::from(if world.rank() == 0 {
            operation().is_ok()
        } else {
            true
        });
        world.process_at_rank(0).broadcast_into(&mut ok);
        ok == 1
    }

    fn expect_accept<T, E: std::fmt::Debug>(
        failures: &mut Vec<String>,
        label: &str,
        result: Result<T, E>,
    ) {
        if let Err(error) = result {
            failures.push(format!("{label}: expected acceptance, got {error:?}"));
        }
    }

    fn expect_reject<T: std::fmt::Debug, E>(
        failures: &mut Vec<String>,
        label: &str,
        result: Result<T, E>,
    ) {
        if let Ok(value) = result {
            failures.push(format!("{label}: expected rejection, got {value:?}"));
        }
    }

    /// #110: the shared payload/record helper must keep the traditional public
    /// error variant. `MpiFileSession::read_named` reports a failed payload
    /// transfer as `MpiSessionError::Io(..)`; wrapping the same failure in
    /// `MpiSessionError::Named(NamedIoError::Io(..))` is a public API change.
    ///
    /// A `cb_buffer_size` hint no allocator can satisfy makes the native
    /// collective payload read fail on every rank, so the conversion is
    /// observed through the public read API instead of a source string or a
    /// test-only hook.
    fn session_read_failure_keeps_the_io_variant<C>(
        world: &C,
        directory: &Path,
        failures: &mut Vec<String>,
    ) where
        C: Communicator + CommunicatorCollectives,
    {
        let size = usize::try_from(world.size()).expect("world size fits usize");
        let topology = MpiTopology::<1>::new(world, [size]).expect("1-D process grid");
        let comm = topology.communicator();
        let pencil = || Pencil::<1, 1>::new(topology.clone(), [4], [0]).expect("1-D pencil");
        let path = directory.join("session-payload-variant.pio");

        let source =
            PencilArray::from_elem(pencil(), ExtraShape::scalar(), 11i32).expect("source array");
        let mut session = match MpiFileSession::create(comm, &path, &MpiIoOptions::default()) {
            Ok(session) => session,
            Err(error) => {
                failures.push(format!("session create: {error:?}"));
                return;
            }
        };
        let written = session.write_named("data", source.view());
        let closed = session.close();
        if let Err(error) = written {
            failures.push(format!("session write: {error:?}"));
        }
        if let Err(error) = closed {
            failures.push(format!("session close after write: {error:?}"));
            return;
        }

        let unallocatable = MpiIoOptions::default().hint("cb_buffer_size", "1000000000000000000");
        let mut reader = match MpiFileSession::open_read(comm, &path, &unallocatable) {
            Ok(session) => session,
            Err(error) => {
                failures.push(format!("session open for read: {error:?}"));
                return;
            }
        };
        let mut target = PencilArray::from_elem(pencil(), ExtraShape::scalar(), -1i32)
            .expect("destination array");
        let read = reader.read_named("data", target.view_mut());
        let closed = reader.close();
        match read {
            Err(MpiSessionError::Io(IoError::CollectivePrecondition {
                phase: "session payload read",
            })) => {}
            Err(error) => failures.push(format!(
                "failed session payload read must keep the MpiSessionError::Io variant: {error:?}"
            )),
            Ok(()) => failures.push(
                "the unallocatable collective buffer did not fail the session payload read"
                    .to_owned(),
            ),
        }
        if let Err(error) = closed {
            failures.push(format!("session close after read: {error:?}"));
        }
        if target.as_slice().iter().any(|&value| value != -1) {
            failures.push("failed session read modified its destination".to_owned());
        }
    }

    #[test]
    fn hdf5_empty_shape_and_null_dataspace_acceptance_is_shared() {
        let universe = mpi::initialize().expect("one MPI initialization per process");
        let world = universe.world();
        let size = usize::try_from(world.size()).expect("world size fits usize");
        let grid = [size as u64];
        let directory = owned_temp_dir(&world, "pencil-io-metadata-regressions");
        let mut failures = Vec::new();

        // Runs first: the #110 error-variant check shares this process's single
        // MPI initialization, and collecting the failure keeps the HDF5
        // metadata regressions below executing.
        session_read_failure_keeps_the_io_variant(&world, &directory, &mut failures);

        let topology = MpiTopology::<1>::new(&world, [size]).expect("1-D process grid");
        let comm = topology.communicator();

        // #111 defect 1: global [2^32] and extra [2^32, 0]. The final element
        // count is zero, but the combined global/extra product overflows u64
        // before the trailing zero is reached. The dataset itself has zero
        // elements, so only metadata is exercised.
        let empty_path = directory.join("empty-after-overflow.h5");
        let empty_tree = directory.join("empty-after-overflow-tree.h5");
        root_status(&world, || {
            write_fixture(
                &empty_path,
                "pencil_io_v1",
                "data",
                &[1u64 << 32, 0],
                &[1u64 << 32],
                &grid,
            )?;
            write_fixture(
                &empty_tree,
                "pencil_io_tree_v1",
                "x",
                &[1u64 << 32, 0],
                &[1u64 << 32],
                &grid,
            )
        });
        world.barrier();
        let mut serial_destination: [f64; 0] = [];
        let serial_ok = root_ok(&world, || {
            read_hdf5_serial(
                &empty_path,
                &[1usize << 32],
                &[1usize << 32, 0],
                &mut serial_destination,
            )
            .map_err(|error| {
                eprintln!("serial empty-shape read error: {error:?}");
                error.to_string()
            })
        });
        if !serial_ok {
            failures.push(
                "serial read: trailing zero after overflowing extents was rejected".to_owned(),
            );
        }
        let pencil =
            Pencil::<1, 1>::new(topology.clone(), [1usize << 32], [0]).expect("1-D pencil");
        let mut parallel_destination = PencilArray::from_elem(
            pencil.clone(),
            ExtraShape::new([1usize << 32, 0]).unwrap(),
            0.0f64,
        )
        .expect("zero-element destination");
        expect_accept(
            &mut failures,
            "parallel read: trailing zero after overflowing extents",
            read_hdf5(&empty_path, parallel_destination.view_mut()),
        );
        expect_accept(
            &mut failures,
            "catalog: trailing zero after overflowing extents",
            read_hdf5_catalog(&empty_path, comm),
        );
        expect_accept(
            &mut failures,
            "session catalog: trailing zero after overflowing extents",
            session_catalog(&empty_tree, comm),
        );

        // The session catalog historically started the byte product from the
        // scalar width instead of one. With extra shape [2^30, 0] and global
        // [2^32] the catalog accepts (the zero is reached before the width)
        // but the session rejects (8 * 2^32 * 2^30 overflows before the zero).
        // All four readers must accept the same empty data.
        let width_path = directory.join("empty-width-first.h5");
        let width_tree = directory.join("empty-width-first-tree.h5");
        root_status(&world, || {
            write_fixture(
                &width_path,
                "pencil_io_v1",
                "data",
                &[1u64 << 30, 0],
                &[1u64 << 32],
                &grid,
            )?;
            write_fixture(
                &width_tree,
                "pencil_io_tree_v1",
                "x",
                &[1u64 << 30, 0],
                &[1u64 << 32],
                &grid,
            )
        });
        world.barrier();
        let mut serial_destination: [f64; 0] = [];
        let serial_ok = root_ok(&world, || {
            read_hdf5_serial(
                &width_path,
                &[1usize << 32],
                &[1usize << 30, 0],
                &mut serial_destination,
            )
            .map_err(|error| {
                eprintln!("serial width-first read error: {error:?}");
                error.to_string()
            })
        });
        if !serial_ok {
            failures.push("serial read: width-first empty shape was rejected".to_owned());
        }
        let mut parallel_destination =
            PencilArray::from_elem(pencil, ExtraShape::new([1usize << 30, 0]).unwrap(), 0.0f64)
                .expect("zero-element destination");
        expect_accept(
            &mut failures,
            "parallel read: width-first empty shape",
            read_hdf5(&width_path, parallel_destination.view_mut()),
        );
        expect_accept(
            &mut failures,
            "catalog: width-first empty shape",
            read_hdf5_catalog(&width_path, comm),
        );
        expect_accept(
            &mut failures,
            "session catalog: width-first empty shape",
            session_catalog(&width_tree, comm),
        );

        // The same acceptance rule must still reject a nonzero product that
        // overflows u64. Catalog and session read metadata only; their stored
        // dataset shape matches the overflowing metadata exactly.
        let overflow_path = directory.join("nonzero-overflow.h5");
        let overflow_tree = directory.join("nonzero-overflow-tree.h5");
        root_status(&world, || {
            write_fixture(
                &overflow_path,
                "pencil_io_v1",
                "data",
                &[],
                &[1u64 << 32, 1u64 << 32],
                &[1, 1],
            )?;
            write_fixture(
                &overflow_tree,
                "pencil_io_tree_v1",
                "x",
                &[],
                &[1u64 << 32, 1u64 << 32],
                &[1, 1],
            )
        });
        world.barrier();
        expect_reject(
            &mut failures,
            "catalog: nonzero product overflow",
            read_hdf5_catalog(&overflow_path, comm),
        );
        expect_reject(
            &mut failures,
            "session catalog: nonzero product overflow",
            session_catalog(&overflow_tree, comm),
        );

        // #111 defect 2: the extra-rank attribute is a scalar zero on disk.
        // Replacing it with an H5S_NULL dataspace of the same u64 type must be
        // rejected by all four readers; only a scalar dataspace carries the
        // one-element value.
        let null_path = directory.join("null-extra-rank.h5");
        let null_tree = directory.join("null-extra-rank-tree.h5");
        root_status(&world, || {
            write_fixture(&null_path, "pencil_io_v1", "data", &[], &[4], &grid)?;
            replace_with_null_dataspace(
                &null_path,
                "pencil_io_v1",
                "data",
                "pencil_io_extra_rank",
            )?;
            write_fixture(&null_tree, "pencil_io_tree_v1", "x", &[], &[4], &grid)?;
            replace_with_null_dataspace(
                &null_tree,
                "pencil_io_tree_v1",
                "x",
                "pencil_io_extra_rank",
            )
        });
        world.barrier();
        let mut serial_destination = [-1.0f64; 4];
        let serial_rejected = root_ok(&world, || {
            match read_hdf5_serial(&null_path, &[4], &[], &mut serial_destination) {
                Ok(()) => Err("H5S_NULL scalar attribute was accepted".to_owned()),
                Err(error) => {
                    eprintln!("serial null-dataspace read rejected: {error:?}");
                    Ok(())
                }
            }
        });
        if !serial_rejected {
            failures.push("serial read: H5S_NULL scalar attribute was accepted".to_owned());
        }
        let pencil = Pencil::<1, 1>::new(topology.clone(), [4], [0]).expect("1-D pencil");
        let mut parallel_destination =
            PencilArray::from_elem(pencil, ExtraShape::scalar(), -1.0f64)
                .expect("four-element destination");
        expect_reject(
            &mut failures,
            "parallel read: H5S_NULL scalar attribute",
            read_hdf5(&null_path, parallel_destination.view_mut()),
        );
        expect_reject(
            &mut failures,
            "catalog: H5S_NULL scalar attribute",
            read_hdf5_catalog(&null_path, comm),
        );
        expect_reject(
            &mut failures,
            "session catalog: H5S_NULL scalar attribute",
            session_catalog(&null_tree, comm),
        );

        cleanup_owned_temp_dir(&world, &directory);
        assert!(
            failures.is_empty(),
            "#111 HDF5 metadata acceptance regressions:\n{}",
            failures.join("\n")
        );
    }
}
