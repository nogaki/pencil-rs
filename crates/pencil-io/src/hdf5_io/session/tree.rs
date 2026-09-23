use super::*;
use crate::catalog::{DatasetInfo, ScalarType};
use std::collections::BTreeSet;

impl Hdf5FileSession<'_> {
    /// Inspect all dataset metadata on the retained file, without payload reads.
    /// Unknown objects, incomplete leaves, aliases and cycles reject the snapshot.
    pub fn catalog(&mut self) -> Result<Vec<DatasetInfo>, Hdf5SessionError> {
        self.entry(0x707, "")?;
        self.inspect_tree()
    }
    pub(super) fn inspect_tree(&self) -> Result<Vec<DatasetInfo>, Hdf5SessionError> {
        let root = self.root()?;
        let result = (|| {
            let mut identities = BTreeSet::new();
            let identity = identity(self.comm, self.file.expect("open"), cstr(TREE))?;
            identities.insert(identity);
            let mut out = Vec::new();
            let mut bytes = 0;
            self.walk(root, "", &mut identities, &mut bytes, &mut out)?;
            Ok(out)
        })();
        let cleanup = close_group(self.comm, root);
        match (result, cleanup) {
            (Err(e), _) => Err(e),
            (_, Err(e)) => Err(e.into()),
            (Ok(out), Ok(())) => Ok(out),
        }
    }
    fn walk(
        &self,
        group: native::Hid,
        prefix: &str,
        seen: &mut BTreeSet<(u64, u64)>,
        bytes: &mut usize,
        out: &mut Vec<DatasetInfo>,
    ) -> Result<(), Hdf5SessionError> {
        let count = native::group_link_count(group);
        agree_phase(self.comm, count.is_ok(), "session catalog link count")?;
        let count = count.map_err(|c| hdf5_error("H5Gget_info", c))?;
        agree_bytes(self.comm, &count.to_le_bytes())?;
        agree_phase(
            self.comm,
            count <= (MAX_OBJECTS - seen.len()) as u64,
            "session catalog object bound",
        )?;
        for index in 0..count {
            let mut buf = [0u8; MAX_COMPONENT + 1];
            let len = native::link_name_by_idx(group, index, &mut buf);
            agree_phase(
                self.comm,
                matches!(len, Ok(n) if n > 0 && n <= MAX_COMPONENT),
                "session catalog component bound",
            )?;
            let len = len.map_err(|c| hdf5_error("H5Lget_name_by_idx", c))?;
            agree_bytes(self.comm, &buf[..len])?;
            let component = std::str::from_utf8(&buf[..len]);
            agree_phase(self.comm, component.is_ok(), "session catalog UTF-8")?;
            let component = component.expect("agreed UTF-8");
            let path = if prefix.is_empty() {
                component.to_owned()
            } else {
                format!("{prefix}/{component}")
            };
            agree_phase(self.comm, valid_path(&path), "session catalog path bound")?;
            *bytes += path.len();
            agree_phase(
                self.comm,
                *bytes <= MAX_TREE_BYTES,
                "session catalog aggregate bound",
            )?;
            let name = CString::new(component).expect("validated component");
            let is_group =
                native::link_is_hard_and_type(group, &name, native::LinkObjectType::Group);
            agree_phase(self.comm, is_group.is_ok(), "session catalog group kind")?;
            let is_group = is_group.map_err(|c| hdf5_error("session object kind", c))?;
            agree_bytes(self.comm, &[u8::from(is_group)])?;
            if !is_group {
                check_kind(self.comm, group, &name, native::LinkObjectType::Dataset)?;
            }
            let object_id = identity(self.comm, group, &name)?;
            agree_phase(
                self.comm,
                seen.len() < MAX_OBJECTS && seen.insert(object_id),
                "session catalog alias or cycle",
            )?;
            if is_group {
                let child = collective_handle_phase(
                    self.comm,
                    native::group_open(group, &name),
                    "session catalog group open",
                )?;
                let result = self.walk(child, &path, seen, bytes, out);
                let cleanup = close_group(self.comm, child);
                result?;
                cleanup?;
            } else {
                let dataset = collective_handle_phase(
                    self.comm,
                    native::dataset_open(group, &name),
                    "session catalog dataset open",
                )?;
                let mut resources = Hdf5Resources {
                    dataset: Some(dataset),
                    ..Hdf5Resources::default()
                };
                let result = leaf(
                    self.comm,
                    self.duplicate.expect("open"),
                    dataset,
                    path,
                    &mut resources,
                );
                let cleanup = data::finish_session(self.comm, resources);
                if let Some(e) = cleanup {
                    return Err(e.into());
                }
                let info = result?;
                *bytes +=
                    (info.global_shape.len() + info.extra_shape.len()) * 8 + info.provenance.len();
                agree_phase(
                    self.comm,
                    *bytes <= MAX_TREE_BYTES,
                    "session catalog metadata bound",
                )?;
                let allocation = out.try_reserve(1);
                agree_phase(self.comm, allocation.is_ok(), "session catalog allocation")?;
                out.push(info);
            }
        }
        Ok(())
    }
}
fn identity(
    comm: &CartesianCommunicator,
    group: native::Hid,
    name: &CStr,
) -> Result<(u64, u64), Hdf5SessionError> {
    let result = native::hard_object_identity(group, name);
    if let Err(agreement) = agree_phase(comm, result.is_ok(), "session hard object identity") {
        return Err(result
            .err()
            .map(|c| hdf5_error("session object identity", c))
            .unwrap_or(agreement)
            .into());
    }
    let identity = result.map_err(|c| hdf5_error("session object identity", c))?;
    // Native file numbers are process-local; only use them in the local visited
    // set. Rank agreement on alias detection is performed by the caller.
    Ok(identity)
}
fn leaf(
    comm: &CartesianCommunicator,
    duplicate: ffi::MPI_Comm,
    dataset: native::Hid,
    path: String,
    resources: &mut Hdf5Resources,
) -> Result<DatasetInfo, Hdf5SessionError> {
    macro_rules! attr {
        ($name:expr, $length:expr, $max:expr) => {{
            let value = read_attr_phase(
                comm,
                duplicate,
                dataset,
                $name,
                $length,
                $max,
                "session catalog attribute",
            )?;
            let mut bytes = [0u8; MAX_PROTOCOL_RANK * 8];
            for (slot, value) in bytes.chunks_exact_mut(8).zip(&value) {
                slot.copy_from_slice(&value.to_le_bytes());
            }
            agree_bytes(comm, &bytes[..value.len() * 8])?;
            value
        }};
    }
    let version = attr!(ATTR_VERSION, Some(1), 1);
    let commit = attr!(ATTR_COMMIT, Some(1), 1);
    let n = attr!(ATTR_N, Some(1), 1)[0];
    let typ = attr!(ATTR_TYPE, Some(1), 1)[0];
    let width = attr!(ATTR_WIDTH, Some(1), 1)[0];
    let er = attr!(ATTR_EXTRA_RANK, Some(1), 1)[0];
    let scalar = ScalarType::decode(typ, width);
    agree_phase(
        comm,
        version == [FORMAT_VERSION]
            && commit == [COMMIT_MARKER]
            && n > 0
            && n <= MAX_PROTOCOL_RANK as u64
            && er <= MAX_PROTOCOL_RANK as u64
            && n + er <= MAX_PROTOCOL_RANK as u64
            && scalar.is_some(),
        "session catalog scalar metadata",
    )?;
    let scalar = scalar.expect("agreed scalar");
    let n = n as usize;
    let er = er as usize;
    let extra = if er == 0 {
        let exists = attribute_exists_phase(comm, dataset, ATTR_EXTRA, "session extra metadata")?;
        agree_phase(comm, !exists, "session absent extra shape")?;
        Vec::new()
    } else {
        attr!(ATTR_EXTRA, Some(er), er)
    };
    let global = attr!(ATTR_GLOBAL, Some(n), n);
    let grid = attr!(ATTR_GRID, None, MAX_PROTOCOL_RANK);
    let perm = attr!(ATTR_PERM, Some(n), n);
    let size = global
        .iter()
        .chain(&extra)
        .try_fold(width, |n, &x| n.checked_mul(x));
    let ranks = grid.iter().try_fold(1u64, |n, &x| n.checked_mul(x));
    agree_phase(
        comm,
        size.is_some()
            && !grid.is_empty()
            && grid.iter().all(|&x| x > 0)
            && ranks.is_some()
            && is_permutation(&perm, n),
        "session catalog shape and provenance",
    )?;
    let (space, error) = local_handle_phase(
        comm,
        native::dataset_space(dataset),
        "session catalog dataspace",
    );
    resources.file_space = space;
    if let Some(error) = error {
        return Err(error.into());
    }
    let actual = native::space_shape(space.expect("agreed space"));
    agree_phase(comm, actual.is_ok(), "session catalog shape query")?;
    let actual = actual.map_err(|c| hdf5_error("session catalog shape", c))?;
    agree_phase(
        comm,
        actual
            .iter()
            .copied()
            .eq(extra.iter().chain(&global).copied()),
        "session catalog shape",
    )?;
    let (datatype, error) = local_handle_phase(
        comm,
        native::dataset_type(dataset),
        "session catalog datatype",
    );
    resources.datatype = datatype;
    if let Some(error) = error {
        return Err(error.into());
    }
    let matches = native::type_matches(
        datatype.expect("agreed datatype"),
        scalar.code(),
        scalar.width(),
        duplicate,
    );
    agree_phase(
        comm,
        matches!(matches, Ok(true)),
        "session catalog datatype match",
    )?;
    let mut provenance = Vec::new();
    agree_phase(
        comm,
        provenance
            .try_reserve_exact((grid.len() + perm.len()) * 8)
            .is_ok(),
        "session provenance allocation",
    )?;
    for value in grid.iter().chain(&perm) {
        provenance.extend_from_slice(&value.to_le_bytes());
    }
    Ok(DatasetInfo {
        name: Some(path),
        scalar_type: scalar,
        global_shape: global,
        extra_shape: extra,
        provenance,
    })
}
