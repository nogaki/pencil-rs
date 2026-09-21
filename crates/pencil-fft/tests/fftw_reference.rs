use std::{
    env, fs,
    ops::Range,
    path::{Path, PathBuf},
    sync::Arc,
};

use mpi::{
    collective::{CommunicatorCollectives, SystemOperation},
    datatype::Equivalence,
    traits::*,
};
use pencil_array::{ExtraShape, MpiTopology, Pencil, SpatialAxis};
use pencil_fft::{
    AxisR2rKind, AxisSelection, AxisTransform, C2cPlan, Complex, DhtPlan, DistributedLayout,
    FftReal, FourierDirection, FourierDirections, MixedC2cPlan, MixedR2cPlan, R2cPlan, R2rKind,
    R2rPlan, R2rScalar, R2rState, TransposeMethod,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    C2c,
    R2c,
    R2r,
    Dht,
    MixedC2c,
    MixedR2c,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ElementKind {
    Real,
    Complex,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum R2rAxisKind {
    None,
    Fft,
    Rfft,
    DctI,
    DctII,
    DctIII,
    DctIV,
    DstI,
    DstII,
    DstIII,
    DstIV,
    Dht,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Precision {
    F32,
    F64,
}

struct DirectionFixture {
    case: String,
    shape: Vec<usize>,
    transforms: Vec<R2rAxisKind>,
    directions: Vec<FourierDirection>,
    input: Vec<Complex<f64>>,
    inverse_input: Vec<Complex<f64>>,
    forward: Vec<Complex<f64>>,
    inverse: Vec<Complex<f64>>,
    backward: Vec<Complex<f64>>,
}

struct Fixture {
    case: String,
    kind: Kind,
    element_kind: ElementKind,
    precision: Precision,
    shape: Vec<usize>,
    original_n: Option<usize>,
    extra: Vec<usize>,
    axis_kinds: Vec<R2rAxisKind>,
    selection: Vec<usize>,
    input: Vec<Complex<f64>>,
    inverse_input: Vec<Complex<f64>>,
    forward: Vec<Complex<f64>>,
    inverse: Vec<Complex<f64>>,
    backward: Vec<Complex<f64>>,
}

fn next<'a>(lines: &[&'a str], cursor: &mut usize, field: &str) -> Result<&'a str, String> {
    let line = lines
        .get(*cursor)
        .copied()
        .ok_or_else(|| format!("missing {field}"))?;
    *cursor += 1;
    Ok(line)
}

fn field<'a>(line: &'a str, name: &str) -> Result<&'a str, String> {
    let words: Vec<_> = line.split_whitespace().collect();
    (words.len() == 2 && words[0] == name)
        .then(|| words[1])
        .ok_or_else(|| format!("expected {name} VALUE, got {line:?}"))
}

fn usize_list(words: &[&str], name: &str) -> Result<Vec<usize>, String> {
    words
        .iter()
        .map(|word| {
            word.parse()
                .map_err(|error| format!("{name}: invalid usize {word:?}: {error}"))
        })
        .collect()
}

fn product(values: &[usize], name: &str) -> Result<usize, String> {
    values.iter().try_fold(1usize, |total, &value| {
        total
            .checked_mul(value)
            .ok_or_else(|| format!("{name}: product overflow"))
    })
}

fn shape(line: &str, name: &str) -> Result<Vec<usize>, String> {
    let words: Vec<_> = line.split_whitespace().collect();
    if words.first().copied() != Some(name) {
        return Err(format!("expected {name} ..., got {line:?}"));
    }
    let shape = usize_list(&words[1..], name)?;
    if shape.contains(&0) {
        return Err(format!("{name}: zero extent is not allowed"));
    }
    Ok(shape)
}

fn finite(word: &str, name: &str) -> Result<f64, String> {
    let value = word
        .parse::<f64>()
        .map_err(|error| format!("{name}: invalid number {word:?}: {error}"))?;
    value
        .is_finite()
        .then_some(value)
        .ok_or_else(|| format!("{name}: non-finite number {word:?}"))
}

fn runtime_value<'a>(word: &'a str, name: &str) -> Result<&'a str, String> {
    word.strip_prefix(&format!("{name}="))
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("runtime: expected {name}=VALUE"))
}

fn section(
    lines: &[&str],
    cursor: &mut usize,
    name: &str,
    kind: &str,
    expected: usize,
) -> Result<Vec<Complex<f64>>, String> {
    let header = next(lines, cursor, name)?;
    let words: Vec<_> = header.split_whitespace().collect();
    if words.len() != 4 || words[0] != "section" || words[1] != name || words[2] != kind {
        return Err(format!(
            "expected section {name} {kind} COUNT, got {header:?}"
        ));
    }
    let count: usize = words[3]
        .parse()
        .map_err(|error| format!("section count: invalid usize: {error}"))?;
    if count != expected {
        return Err(format!(
            "section {name}: count {count} does not match expected {expected}"
        ));
    }
    if count >= lines.len().saturating_sub(*cursor) {
        return Err(format!("section {name}: not enough value/end lines"));
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| format!("section {name}: count allocation failed"))?;
    for index in 0..count {
        let line = next(lines, cursor, &format!("{name} value"))?;
        let words: Vec<_> = line.split_whitespace().collect();
        let value = match (kind, words.as_slice()) {
            ("real", [real]) => Complex::new(finite(real, name)?, 0.0),
            ("complex", [real, imaginary]) => {
                Complex::new(finite(real, name)?, finite(imaginary, name)?)
            }
            ("real", _) => {
                return Err(format!(
                    "section {name}: real value {index} must have one number"
                ));
            }
            ("complex", _) => {
                return Err(format!(
                    "section {name}: complex value {index} must have two numbers"
                ));
            }
            _ => return Err(format!("invalid section kind {kind:?}")),
        };
        values.push(value);
    }
    if next(lines, cursor, &format!("{name} end"))? != "end" {
        return Err(format!("section {name}: expected end"));
    }
    Ok(values)
}

fn selected_axes(line: &str, dimensions: usize) -> Result<Vec<usize>, String> {
    let words: Vec<_> = line.split_whitespace().collect();
    if words.first().copied() != Some("selected_axes") {
        return Err(format!("expected selected_axes ..., got {line:?}"));
    }
    let axes = usize_list(&words[1..], "selected_axes")?;
    if axes.iter().any(|&axis| axis >= dimensions) {
        return Err("selected_axes: axis is out of bounds".into());
    }
    if axes.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err("selected_axes: axes must be strictly increasing".into());
    }
    Ok(axes)
}

fn parse_r2r_axis_kind(word: &str) -> Result<R2rAxisKind, String> {
    match word {
        "none" => Ok(R2rAxisKind::None),
        "fft" => Ok(R2rAxisKind::Fft),
        "rfft" => Ok(R2rAxisKind::Rfft),
        "dcti" => Ok(R2rAxisKind::DctI),
        "dctii" => Ok(R2rAxisKind::DctII),
        "dctiii" => Ok(R2rAxisKind::DctIII),
        "dctiv" => Ok(R2rAxisKind::DctIV),
        "dsti" => Ok(R2rAxisKind::DstI),
        "dstii" => Ok(R2rAxisKind::DstII),
        "dstiii" => Ok(R2rAxisKind::DstIII),
        "dstiv" => Ok(R2rAxisKind::DstIV),
        "dht" => Ok(R2rAxisKind::Dht),
        other => Err(format!("axis_kinds: invalid kind {other:?}")),
    }
}

fn axis_kinds(line: &str, dimensions: usize) -> Result<Vec<R2rAxisKind>, String> {
    let words: Vec<_> = line.split_whitespace().collect();
    if words.first().copied() != Some("axis_kinds") {
        return Err(format!("expected axis_kinds ..., got {line:?}"));
    }
    if words.len() != dimensions + 1 {
        return Err(format!(
            "axis_kinds: expected {dimensions} values, got {}",
            words.len().saturating_sub(1)
        ));
    }
    words[1..]
        .iter()
        .map(|word| parse_r2r_axis_kind(word))
        .collect()
}

fn reduced(
    kind: Kind,
    shape: &[usize],
    selection: &[usize],
    axis_kinds: &[R2rAxisKind],
) -> Vec<usize> {
    let mut result = shape.to_vec();
    if matches!(kind, Kind::R2c | Kind::MixedR2c) {
        let axis = if kind == Kind::MixedR2c {
            axis_kinds
                .iter()
                .position(|kind| *kind == R2rAxisKind::Rfft)
                .expect("mixed R2C reference has an RFFT axis")
        } else {
            *selection
                .iter()
                .max()
                .expect("R2C reference selection is nonempty")
        };
        result[axis] = result[axis] / 2 + 1;
    }
    result
}

fn section_kind(element_kind: ElementKind) -> &'static str {
    match element_kind {
        ElementKind::Real => "real",
        ElementKind::Complex => "complex",
    }
}

fn parse_direction_fixture(text: &str) -> Result<DirectionFixture, String> {
    let lines: Vec<_> = text.lines().map(str::trim).collect();
    let mut cursor = 0;
    if next(&lines, &mut cursor, "version")? != "PENCIL_FFTW_DIRECTION_REFERENCE 1" {
        return Err("invalid direction reference version header".into());
    }
    let runtime: Vec<_> = next(&lines, &mut cursor, "runtime")?
        .split_whitespace()
        .collect();
    if runtime.len() != 5
        || runtime[0] != "runtime"
        || runtime_value(runtime[1], "julia")? != "1.12.6"
        || runtime_value(runtime[2], "fftw_jl")? != "1.10.0"
        || runtime_value(runtime[3], "native")?.is_empty()
        || runtime_value(runtime[4], "provider")? != "fftw"
    {
        return Err("reference runtime metadata is not the pinned FFTW setup".into());
    }
    let case = field(next(&lines, &mut cursor, "case")?, "case")?.to_owned();
    let shape = shape(next(&lines, &mut cursor, "shape")?, "shape")?;
    if !(2..=4).contains(&shape.len()) {
        return Err("shape: expected 2 to 4 dimensions".into());
    }
    let transform_words: Vec<_> = next(&lines, &mut cursor, "transforms")?
        .split_whitespace()
        .collect();
    if transform_words.first().copied() != Some("transforms")
        || transform_words.len() != shape.len() + 1
    {
        return Err("transforms: expected one value per axis".into());
    }
    let transforms = transform_words[1..]
        .iter()
        .map(|word| parse_r2r_axis_kind(word))
        .collect::<Result<Vec<_>, _>>()?;
    if transforms.iter().any(|kind| {
        !matches!(
            kind,
            R2rAxisKind::Fft
                | R2rAxisKind::Rfft
                | R2rAxisKind::None
                | R2rAxisKind::DctI
                | R2rAxisKind::DctII
                | R2rAxisKind::DctIII
                | R2rAxisKind::DctIV
                | R2rAxisKind::DstI
                | R2rAxisKind::DstII
                | R2rAxisKind::DstIII
                | R2rAxisKind::DstIV
                | R2rAxisKind::Dht
        )
    }) {
        return Err("transforms: invalid kind".into());
    }
    let direction_words: Vec<_> = next(&lines, &mut cursor, "directions")?
        .split_whitespace()
        .collect();
    if direction_words.first().copied() != Some("directions")
        || direction_words.len() != shape.len() + 1
    {
        return Err("directions: expected one value per axis".into());
    }
    let directions = direction_words[1..]
        .iter()
        .map(|word| match *word {
            "forward" => Ok(FourierDirection::Forward),
            "backward" => Ok(FourierDirection::Backward),
            other => Err(format!("directions: invalid value {other:?}")),
        })
        .collect::<Result<Vec<_>, _>>()?;
    if transforms
        .iter()
        .zip(&directions)
        .any(|(kind, sign)| *kind != R2rAxisKind::Fft && *sign != FourierDirection::Forward)
    {
        return Err("non-FFT backward sign".into());
    }
    let mut output_shape = shape.clone();
    let boundaries: Vec<_> = transforms
        .iter()
        .enumerate()
        .filter(|(_, kind)| **kind == R2rAxisKind::Rfft)
        .map(|(axis, _)| axis)
        .collect();
    if boundaries.len() > 1 {
        return Err("multiple RFFT axes".into());
    }
    if let Some(&axis) = boundaries.first() {
        if transforms[..axis]
            .iter()
            .any(|k| !matches!(k, R2rAxisKind::Fft | R2rAxisKind::None))
            || transforms[axis + 1..].contains(&R2rAxisKind::Fft)
        {
            return Err("invalid RFFT graph".into());
        }
        output_shape[axis] = shape[axis] / 2 + 1;
    }
    let output_count = product(&output_shape, "output shape")?;
    let count = product(&shape, "shape")?;
    let input = section(&lines, &mut cursor, "input", "complex", count)?;
    let inverse_input = section(
        &lines,
        &mut cursor,
        "inverse_input",
        "complex",
        output_count,
    )?;
    let forward = section(
        &lines,
        &mut cursor,
        "forward_expected",
        "complex",
        output_count,
    )?;
    let inverse = section(&lines, &mut cursor, "inverse_expected", "complex", count)?;
    let backward = section(&lines, &mut cursor, "backward_expected", "complex", count)?;
    if !boundaries.is_empty()
        && input
            .iter()
            .chain(&inverse)
            .chain(&backward)
            .any(|value| value.im != 0.0)
    {
        return Err("RFFT real sections must have zero imaginary parts".into());
    }
    if cursor != lines.len() {
        return Err(format!(
            "trailing direction reference tokens starting at {:?}",
            lines[cursor]
        ));
    }
    Ok(DirectionFixture {
        case,
        shape,
        transforms,
        directions,
        input,
        inverse_input,
        forward,
        inverse,
        backward,
    })
}

fn parse_fixture(text: &str) -> Result<Fixture, String> {
    let mut lines: Vec<_> = text.lines().map(str::trim).collect();
    while lines.last().copied() == Some("") {
        lines.pop();
    }
    let mut cursor = 0;
    if next(&lines, &mut cursor, "version")? != "PENCIL_FFTW_REFERENCE 7" {
        return Err("invalid reference version header".into());
    }

    let runtime: Vec<_> = next(&lines, &mut cursor, "runtime")?
        .split_whitespace()
        .collect();
    if runtime.len() != 5 || runtime[0] != "runtime" {
        return Err("malformed runtime line".into());
    }
    let julia = runtime_value(runtime[1], "julia")?;
    let fftw_jl = runtime_value(runtime[2], "fftw_jl")?;
    let native = runtime_value(runtime[3], "native")?;
    let provider = runtime_value(runtime[4], "provider")?;
    if julia != "1.12.6" || fftw_jl != "1.10.0" || native.is_empty() || provider != "fftw" {
        return Err("reference runtime metadata is not the pinned FFTW setup".into());
    }

    let case = field(next(&lines, &mut cursor, "case")?, "case")?.to_owned();
    let kind = match field(next(&lines, &mut cursor, "kind")?, "kind")? {
        "c2c" => Kind::C2c,
        "r2c" => Kind::R2c,
        "r2r" => Kind::R2r,
        "dht" => Kind::Dht,
        "mixed_c2c" => Kind::MixedC2c,
        "mixed_r2c" => Kind::MixedR2c,
        other => return Err(format!("invalid kind {other:?}")),
    };
    let element_kind = match field(next(&lines, &mut cursor, "element_kind")?, "element_kind")? {
        "real" => ElementKind::Real,
        "complex" => ElementKind::Complex,
        other => return Err(format!("invalid element_kind {other:?}")),
    };
    let precision = match field(next(&lines, &mut cursor, "precision")?, "precision")? {
        "f32" => Precision::F32,
        "f64" => Precision::F64,
        other => return Err(format!("invalid precision {other:?}")),
    };
    let original_shape = shape(
        next(&lines, &mut cursor, "original_shape")?,
        "original_shape",
    )?;
    if !(2..=4).contains(&original_shape.len()) {
        return Err("original_shape: expected 2 to 4 dimensions".into());
    }
    let extra = shape(next(&lines, &mut cursor, "extra_shape")?, "extra_shape")?;
    let axis_kinds = axis_kinds(
        next(&lines, &mut cursor, "axis_kinds")?,
        original_shape.len(),
    )?;
    let selection = selected_axes(
        next(&lines, &mut cursor, "selected_axes")?,
        original_shape.len(),
    )?;
    let original_n = if kind == Kind::MixedR2c {
        Some(
            field(next(&lines, &mut cursor, "original_n")?, "original_n")?
                .parse()
                .map_err(|error| format!("original_n: invalid usize: {error}"))?,
        )
    } else {
        None
    };
    let derived_selection = axis_kinds
        .iter()
        .enumerate()
        .filter_map(|(axis, kind)| (*kind != R2rAxisKind::None).then_some(axis))
        .collect::<Vec<_>>();
    if matches!(kind, Kind::R2c | Kind::MixedR2c) && selection.is_empty() {
        return Err("R2C selected_axes must be nonempty".into());
    }
    if matches!(kind, Kind::MixedC2c | Kind::MixedR2c) {
        let rfft_count = axis_kinds
            .iter()
            .filter(|kind| **kind == R2rAxisKind::Rfft)
            .count();
        if kind == Kind::MixedC2c && rfft_count != 0 {
            return Err("mixed C2C cannot contain rfft".into());
        }
        if kind == Kind::MixedR2c && rfft_count != 1 {
            return Err("mixed R2C requires exactly one rfft".into());
        }
        if kind == Kind::MixedR2c {
            let boundary = axis_kinds
                .iter()
                .position(|kind| *kind == R2rAxisKind::Rfft)
                .expect("rfft count was checked");
            for (axis, kind) in axis_kinds.iter().enumerate() {
                if axis > boundary && matches!(kind, R2rAxisKind::Fft | R2rAxisKind::Rfft) {
                    return Err("mixed R2C: real prefix contains a complex axis".into());
                }
            }
            if original_n != Some(original_shape[boundary]) {
                return Err("original_n does not match the RFFT axis extent".into());
            }
        }
        if selection != derived_selection {
            return Err("selected_axes: does not match mixed axis_kinds".into());
        }
    } else {
        if matches!(kind, Kind::C2c | Kind::R2c)
            && axis_kinds.iter().any(|kind| *kind != R2rAxisKind::None)
        {
            return Err("axis_kinds: FFT/RFFT/R2R kinds require a mixed or R2R fixture".into());
        }
        if kind == Kind::R2r
            && axis_kinds.iter().any(|kind| {
                matches!(
                    kind,
                    R2rAxisKind::Fft | R2rAxisKind::Rfft | R2rAxisKind::Dht
                )
            })
        {
            return Err("axis_kinds: R2R fixtures require DCT/DST kinds or none".into());
        }
        if !matches!(kind, Kind::R2r | Kind::Dht)
            && axis_kinds.iter().any(|kind| *kind != R2rAxisKind::None)
        {
            return Err("axis_kinds: non-identity kinds require kind r2r or dht".into());
        }
        if kind == Kind::R2r && axis_kinds.contains(&R2rAxisKind::Dht) {
            return Err("axis_kinds: dht requires kind dht".into());
        }
        if kind == Kind::Dht
            && axis_kinds
                .iter()
                .any(|kind| *kind != R2rAxisKind::None && *kind != R2rAxisKind::Dht)
        {
            return Err("axis_kinds: DHT fixtures require dht or none".into());
        }
        if matches!(kind, Kind::R2r | Kind::Dht) && selection != derived_selection {
            return Err("selected_axes: does not match non-identity axis_kinds".into());
        }
    }
    let expected_element_kind = match kind {
        Kind::C2c | Kind::MixedC2c => ElementKind::Complex,
        Kind::R2c | Kind::MixedR2c => ElementKind::Real,
        Kind::R2r | Kind::Dht => element_kind,
    };
    if element_kind != expected_element_kind {
        return Err("element_kind is inconsistent with fixture kind".into());
    }
    let mut input_shape = extra.clone();
    input_shape.extend(&original_shape);
    let mut output_shape = extra.clone();
    output_shape.extend(reduced(kind, &original_shape, &selection, &axis_kinds));
    let input_count = product(&input_shape, "input shape")?;
    let output_count = product(&output_shape, "output shape")?;
    let input_kind = section_kind(element_kind);
    let transformed_kind = if matches!(kind, Kind::R2c | Kind::MixedR2c) {
        "complex"
    } else {
        input_kind
    };
    let input = section(&lines, &mut cursor, "input", input_kind, input_count)?;
    let inverse_input = section(
        &lines,
        &mut cursor,
        "inverse_input",
        transformed_kind,
        output_count,
    )?;
    let forward = section(
        &lines,
        &mut cursor,
        "forward_expected",
        transformed_kind,
        output_count,
    )?;
    let inverse = section(
        &lines,
        &mut cursor,
        "inverse_expected",
        input_kind,
        input_count,
    )?;
    let backward_kind = if matches!(kind, Kind::C2c | Kind::MixedC2c) {
        "complex"
    } else {
        input_kind
    };
    let backward = section(
        &lines,
        &mut cursor,
        "backward_expected",
        backward_kind,
        input_count,
    )?;
    if cursor != lines.len() {
        return Err(format!(
            "trailing fixture tokens starting at {:?}",
            lines[cursor]
        ));
    }
    Ok(Fixture {
        case,
        kind,
        element_kind,
        precision,
        shape: original_shape,
        original_n,
        extra,
        axis_kinds,
        selection,
        input,
        inverse_input,
        forward,
        inverse,
        backward,
    })
}

fn joined(shape: &[usize]) -> String {
    shape
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join("x")
}

fn all_axes(dimensions: usize) -> Vec<usize> {
    (0..dimensions).collect()
}

fn axis_kind_name(kind: R2rAxisKind) -> &'static str {
    match kind {
        R2rAxisKind::None => "none",
        R2rAxisKind::Fft => "fft",
        R2rAxisKind::Rfft => "rfft",
        R2rAxisKind::DctI => "dcti",
        R2rAxisKind::DctII => "dctii",
        R2rAxisKind::DctIII => "dctiii",
        R2rAxisKind::DctIV => "dctiv",
        R2rAxisKind::DstI => "dsti",
        R2rAxisKind::DstII => "dstii",
        R2rAxisKind::DstIII => "dstiii",
        R2rAxisKind::DstIV => "dstiv",
        R2rAxisKind::Dht => "dht",
    }
}

fn expected_case(
    kind: Kind,
    precision: Precision,
    shape: &[usize],
    extra: &[usize],
    axis_kinds: &[R2rAxisKind],
    selection: &[usize],
) -> String {
    let mut name = format!(
        "{}_{}d_{}",
        match kind {
            Kind::C2c => "c2c",
            Kind::R2c => "r2c",
            Kind::R2r => "r2r",
            Kind::Dht => "dht",
            Kind::MixedC2c => "mixed_c2c",
            Kind::MixedR2c => "mixed_r2c",
        },
        shape.len(),
        joined(shape)
    );
    if !extra.is_empty()
        && matches!(
            kind,
            Kind::R2c | Kind::R2r | Kind::Dht | Kind::MixedC2c | Kind::MixedR2c
        )
    {
        name.push_str("_extra");
        name.push_str(&joined(extra));
    }
    if matches!(
        kind,
        Kind::R2r | Kind::Dht | Kind::MixedC2c | Kind::MixedR2c
    ) {
        name.push('_');
        name.push_str(
            &axis_kinds
                .iter()
                .map(|kind| axis_kind_name(*kind))
                .collect::<Vec<_>>()
                .join("-"),
        );
    } else if selection != all_axes(shape.len()).as_slice() {
        name.push_str("_sel");
        if selection.is_empty() {
            name.push_str("none");
        } else {
            name.push_str(
                &selection
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join("-"),
            );
        }
    }
    name.push('_');
    name.push_str(if precision == Precision::F32 {
        "f32"
    } else {
        "f64"
    });
    name
}

type CaseSpec = (Kind, Vec<usize>, Vec<usize>, Vec<usize>);
type MixedCaseSpec = (Kind, Vec<usize>, Vec<usize>, Vec<usize>, Vec<R2rAxisKind>);

fn base_cases() -> Vec<CaseSpec> {
    vec![
        (Kind::C2c, vec![3, 4], vec![], vec![0, 1]),
        (Kind::C2c, vec![3, 2, 5], vec![2, 3], vec![0, 1, 2]),
        (Kind::C2c, vec![2, 1, 3, 4], vec![2], vec![0, 1, 2, 3]),
        (Kind::R2c, vec![3, 1], vec![], vec![0, 1]),
        (Kind::R2c, vec![3, 2], vec![], vec![0, 1]),
        (Kind::R2c, vec![3, 1, 4], vec![2, 3], vec![0, 1, 2]),
        (Kind::R2c, vec![3, 1, 5], vec![2, 3], vec![0, 1, 2]),
        (Kind::R2c, vec![2, 1, 3, 3], vec![2], vec![0, 1, 2, 3]),
    ]
}

fn partial_cases() -> Vec<CaseSpec> {
    vec![
        (Kind::C2c, vec![2, 3, 2, 3], vec![], vec![]),
        (Kind::C2c, vec![2, 3, 2, 3], vec![], vec![0, 3]),
        (Kind::R2c, vec![2, 3, 2, 3], vec![], vec![0]),
        (Kind::R2c, vec![2, 3, 2, 3], vec![], vec![0, 2]),
        (Kind::R2c, vec![2, 3, 2, 3], vec![], vec![0, 3]),
        (Kind::R2c, vec![2, 3, 4], vec![2], vec![0, 2]),
    ]
}

fn mixed_cases() -> Vec<MixedCaseSpec> {
    vec![
        (
            Kind::MixedC2c,
            vec![3, 2, 4],
            vec![2],
            vec![0, 1, 2],
            vec![R2rAxisKind::Fft, R2rAxisKind::DctII, R2rAxisKind::Dht],
        ),
        (
            Kind::MixedC2c,
            vec![2, 3, 2, 3],
            vec![],
            vec![1, 2, 3],
            vec![
                R2rAxisKind::None,
                R2rAxisKind::Fft,
                R2rAxisKind::DctIV,
                R2rAxisKind::Dht,
            ],
        ),
        (
            Kind::MixedC2c,
            vec![2, 3, 2, 3],
            vec![],
            vec![0, 1, 2, 3],
            vec![
                R2rAxisKind::DctI,
                R2rAxisKind::DstII,
                R2rAxisKind::DctIII,
                R2rAxisKind::DstIV,
            ],
        ),
        (
            Kind::MixedC2c,
            vec![2, 3, 2, 3],
            vec![],
            vec![0, 1, 2, 3],
            vec![
                R2rAxisKind::DctIV,
                R2rAxisKind::DstI,
                R2rAxisKind::DstIII,
                R2rAxisKind::DctII,
            ],
        ),
        (
            Kind::MixedR2c,
            vec![4, 3, 5],
            vec![],
            vec![0, 1, 2],
            vec![R2rAxisKind::Rfft, R2rAxisKind::DctII, R2rAxisKind::Dht],
        ),
        (
            Kind::MixedR2c,
            vec![3, 4, 5],
            vec![2],
            vec![0, 1, 2],
            vec![R2rAxisKind::Fft, R2rAxisKind::Rfft, R2rAxisKind::Dht],
        ),
        (
            Kind::MixedR2c,
            vec![3, 4],
            vec![],
            vec![0, 1],
            vec![R2rAxisKind::Rfft, R2rAxisKind::Dht],
        ),
    ]
}

fn dht_cases() -> Vec<(Vec<usize>, Vec<usize>, Vec<R2rAxisKind>)> {
    vec![
        (vec![3, 4], vec![], vec![R2rAxisKind::Dht, R2rAxisKind::Dht]),
        (
            vec![3, 2, 4],
            vec![2, 3],
            vec![R2rAxisKind::Dht, R2rAxisKind::None, R2rAxisKind::Dht],
        ),
        (
            vec![2, 3, 2, 3],
            vec![],
            vec![
                R2rAxisKind::Dht,
                R2rAxisKind::None,
                R2rAxisKind::None,
                R2rAxisKind::Dht,
            ],
        ),
        (
            vec![2, 3, 2, 3],
            vec![],
            vec![
                R2rAxisKind::None,
                R2rAxisKind::None,
                R2rAxisKind::None,
                R2rAxisKind::None,
            ],
        ),
    ]
}

fn r2r_cases() -> Vec<(Vec<usize>, Vec<usize>, Vec<R2rAxisKind>)> {
    vec![
        (
            vec![3, 4],
            vec![],
            vec![R2rAxisKind::DctI, R2rAxisKind::DctII],
        ),
        (
            vec![3, 4],
            vec![],
            vec![R2rAxisKind::DctIII, R2rAxisKind::DctIV],
        ),
        (
            vec![3, 4],
            vec![],
            vec![R2rAxisKind::DstI, R2rAxisKind::DstII],
        ),
        (
            vec![3, 4],
            vec![],
            vec![R2rAxisKind::DstIII, R2rAxisKind::DstIV],
        ),
        (
            vec![3, 2, 4],
            vec![2, 3],
            vec![R2rAxisKind::DctII, R2rAxisKind::None, R2rAxisKind::DstII],
        ),
        (
            vec![2, 1, 3, 3],
            vec![2],
            vec![
                R2rAxisKind::None,
                R2rAxisKind::None,
                R2rAxisKind::None,
                R2rAxisKind::None,
            ],
        ),
    ]
}

fn direction_fixtures(directory: &Path) -> Vec<DirectionFixture> {
    assert!(
        directory.is_dir(),
        "direction reference directory does not exist: {directory:?}"
    );
    let mut paths: Vec<_> = fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    paths.sort();
    assert_eq!(paths.len(), 5, "unexpected direction reference count");
    let expected = [
        "c2c_2d_3x4_forward-backward",
        "c2c_3d_3x2x4_backward-forward-forward",
        "mixed_c2c_3d_3x2x4_backward-forward-forward",
        "mixed_r2c_even",
        "mixed_r2c_odd",
    ];
    for (path, name) in paths.iter().zip(expected) {
        assert_eq!(
            path.file_stem().and_then(|s| s.to_str()),
            Some(name),
            "direction fixture matrix"
        );
    }

    paths
        .into_iter()
        .map(|path| {
            assert_eq!(
                path.extension().and_then(|value| value.to_str()),
                Some("txt")
            );
            let fixture = parse_direction_fixture(&fs::read_to_string(&path).unwrap())
                .unwrap_or_else(|error| panic!("invalid direction fixture {path:?}: {error}"));
            assert_eq!(
                path.file_stem().and_then(|s| s.to_str()),
                Some(fixture.case.as_str())
            );
            assert_eq!(
                fixture.input.len(),
                product(&fixture.shape, "direction shape").unwrap()
            );
            assert_eq!(fixture.directions.len(), fixture.shape.len());
            match fixture.directions.len() {
                2 => {
                    let _: FourierDirections<2> =
                        FourierDirections::new(fixture.directions.clone().try_into().unwrap());
                }
                3 => {
                    let _: FourierDirections<3> =
                        FourierDirections::new(fixture.directions.clone().try_into().unwrap());
                }
                4 => {
                    let _: FourierDirections<4> =
                        FourierDirections::new(fixture.directions.clone().try_into().unwrap());
                }
                _ => unreachable!(),
            }
            fixture
        })
        .collect()
}

fn fixtures(directory: &Path) -> Vec<Fixture> {
    assert!(
        directory.is_dir(),
        "fixture directory does not exist: {directory:?}"
    );
    let mut paths: Vec<PathBuf> = fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<_, _>>()
        .unwrap();
    paths.sort();
    let expected_files = 2 * (base_cases().len() + partial_cases().len())
        + 4 * r2r_cases().len()
        + 2 * mixed_cases().len()
        + 4 * dht_cases().len();
    assert_eq!(
        paths.len(),
        expected_files,
        "fixture matrix has an unexpected file count"
    );
    assert!(
        paths.iter().all(|path| path.is_file()
            && path.extension().and_then(|value| value.to_str()) == Some("txt")),
        "fixture matrix has a non-txt or non-file entry"
    );
    let result = paths
        .into_iter()
        .map(|path| {
            let text = fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("cannot read {path:?}: {error}"));
            let fixture = parse_fixture(&text)
                .unwrap_or_else(|error| panic!("invalid fixture {path:?}: {error}"));
            assert_eq!(
                fixture.case,
                expected_case(
                    fixture.kind,
                    fixture.precision,
                    &fixture.shape,
                    &fixture.extra,
                    &fixture.axis_kinds,
                    &fixture.selection,
                )
            );
            fixture
        })
        .collect::<Vec<_>>();
    let expected_cases = base_cases()
        .into_iter()
        .chain(partial_cases())
        .collect::<Vec<_>>();
    for (kind, shape, extra, selection) in expected_cases {
        for precision in [Precision::F32, Precision::F64] {
            assert_eq!(
                result
                    .iter()
                    .filter(|fixture| fixture.kind == kind
                        && fixture.precision == precision
                        && fixture.shape == shape
                        && fixture.extra == extra
                        && fixture.selection == selection)
                    .count(),
                1,
                "missing or duplicate fixture"
            );
        }
    }
    for (shape, extra, axis_kinds) in r2r_cases() {
        for precision in [Precision::F32, Precision::F64] {
            for element_kind in [ElementKind::Real, ElementKind::Complex] {
                assert_eq!(
                    result
                        .iter()
                        .filter(|fixture| fixture.kind == Kind::R2r
                            && fixture.element_kind == element_kind
                            && fixture.precision == precision
                            && fixture.shape == shape
                            && fixture.extra == extra
                            && fixture.axis_kinds == axis_kinds)
                        .count(),
                    1,
                    "missing or duplicate R2R fixture"
                );
            }
        }
    }
    for (kind, shape, extra, selection, axis_kinds) in mixed_cases() {
        for precision in [Precision::F32, Precision::F64] {
            let matches = result
                .iter()
                .filter(|fixture| {
                    fixture.kind == kind
                        && fixture.element_kind
                            == if kind == Kind::MixedC2c {
                                ElementKind::Complex
                            } else {
                                ElementKind::Real
                            }
                        && fixture.precision == precision
                        && fixture.shape == shape
                        && fixture.extra == extra
                        && fixture.selection == selection
                        && fixture.axis_kinds == axis_kinds
                })
                .collect::<Vec<_>>();
            assert_eq!(matches.len(), 1, "missing or duplicate mixed fixture");
            let fixture = matches[0];
            assert_eq!(
                fixture.original_n,
                (kind == Kind::MixedR2c).then(|| {
                    let axis = axis_kinds
                        .iter()
                        .position(|kind| *kind == R2rAxisKind::Rfft)
                        .unwrap();
                    shape[axis]
                })
            );
            assert_eq!(
                fixture.input.len(),
                product(
                    &fixture
                        .extra
                        .iter()
                        .chain(fixture.shape.iter())
                        .copied()
                        .collect::<Vec<_>>(),
                    "mixed input",
                )
                .unwrap()
            );
        }
    }
    for (shape, extra, axis_kinds) in dht_cases() {
        for precision in [Precision::F32, Precision::F64] {
            for element_kind in [ElementKind::Real, ElementKind::Complex] {
                assert_eq!(
                    result
                        .iter()
                        .filter(|fixture| fixture.kind == Kind::Dht
                            && fixture.element_kind == element_kind
                            && fixture.precision == precision
                            && fixture.shape == shape
                            && fixture.extra == extra
                            && fixture.axis_kinds == axis_kinds)
                        .count(),
                    1,
                    "missing or duplicate DHT fixture"
                );
            }
        }
    }
    result
}

fn tolerance(precision: Precision) -> (f64, f64) {
    if precision == Precision::F32 {
        (2e-4, 2e-5)
    } else {
        (2e-10, 2e-12)
    }
}

fn unravel(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    let mut result = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        result[axis] = linear % shape[axis];
        linear /= shape[axis];
    }
    result
}

fn row_offset(shape: &[usize], indices: &[usize]) -> Result<usize, String> {
    if shape.len() != indices.len() {
        return Err("row-major rank mismatch".into());
    }
    shape
        .iter()
        .zip(indices)
        .try_fold(0usize, |offset, (&extent, &index)| {
            if index >= extent {
                return Err(format!(
                    "row-major index {index} is outside extent {extent}"
                ));
            }
            offset
                .checked_mul(extent)
                .and_then(|value| value.checked_add(index))
                .ok_or_else(|| "row-major offset overflow".into())
        })
}

fn offsets<const N: usize>(
    extra: &[usize],
    global: [usize; N],
    ranges: &[Range<usize>; N],
    permutation: [usize; N],
) -> Vec<usize> {
    let local_shape: [usize; N] = std::array::from_fn(|axis| ranges[permutation[axis]].len());
    let total = product(extra, "extra shape")
        .unwrap()
        .checked_mul(product(&local_shape, "local shape").unwrap())
        .unwrap();
    let mut physical_shape = extra.to_vec();
    physical_shape.extend(local_shape);
    let mut logical_shape = extra.to_vec();
    logical_shape.extend(global);
    let mut result = Vec::with_capacity(total);
    for linear in 0..total {
        let physical = unravel(linear, &physical_shape);
        let mut index = physical[..extra.len()].to_vec();
        let mut spatial = [0; N];
        for (memory_axis, &logical_axis) in permutation.iter().enumerate() {
            spatial[logical_axis] =
                ranges[logical_axis].start + physical[extra.len() + memory_axis];
        }
        index.extend(spatial);
        result.push(row_offset(&logical_shape, &index).unwrap());
    }
    result
}

#[derive(Debug)]
struct Snapshot {
    offsets: Vec<usize>,
    global_count: usize,
}

fn identity<const N: usize>() -> [usize; N] {
    std::array::from_fn(|axis| axis)
}
fn reverse<const N: usize>() -> [usize; N] {
    std::array::from_fn(|axis| N - axis - 1)
}
fn input_decomp<const M: usize>() -> [usize; M] {
    std::array::from_fn(|axis| axis)
}
fn output_decomp<const M: usize>() -> [usize; M] {
    std::array::from_fn(|axis| axis + 1)
}

struct Layout<'a, const N: usize> {
    input: [usize; N],
    output: [usize; N],
    extra: &'a [usize],
    permute_dims: bool,
}

fn snapshot<const N: usize, const M: usize>(
    pencil: &Pencil<N, M>,
    extra: &ExtraShape,
    layout: &Layout<N>,
    output: bool,
    length: usize,
    label: &str,
) -> Snapshot {
    let (shape, permutation, decomposition) = if output {
        (
            layout.output,
            if layout.permute_dims {
                reverse()
            } else {
                identity()
            },
            output_decomp(),
        )
    } else {
        (layout.input, identity(), input_decomp())
    };
    assert_eq!(*pencil.global_shape(), shape, "{label}: global shape");
    assert_eq!(
        pencil.permutation().axes().map(SpatialAxis::index),
        permutation,
        "{label}: permutation"
    );
    assert_eq!(
        pencil.decomposition().map(SpatialAxis::index),
        decomposition,
        "{label}: decomposition"
    );
    assert_eq!(extra.dimensions(), layout.extra, "{label}: extra shape");
    let offsets = offsets(layout.extra, shape, pencil.local_ranges(), permutation);
    assert_eq!(offsets.len(), length, "{label}: raw storage length");
    let mut full = layout.extra.to_vec();
    full.extend(shape);
    Snapshot {
        offsets,
        global_count: product(&full, "fixture global shape").unwrap(),
    }
}

fn ownership<const N: usize, const M: usize>(
    pencil: &Pencil<N, M>,
    snapshot: &Snapshot,
    label: &str,
) {
    let mut local = vec![0i32; snapshot.global_count];
    for &offset in &snapshot.offsets {
        assert!(offset < snapshot.global_count);
        local[offset] += 1;
    }
    let mut global = vec![0i32; snapshot.global_count];
    pencil.topology().communicator().all_reduce_into(
        &local[..],
        &mut global[..],
        SystemOperation::sum(),
    );
    assert!(
        global.iter().all(|&count| count == 1),
        "{label}: ownership counts {global:?}"
    );
}

trait Real: FftReal + Equivalence {
    fn from_f64(value: f64) -> Self;
    fn to_f64(value: Self) -> f64;
}
impl Real for f32 {
    fn from_f64(value: f64) -> Self {
        value as f32
    }
    fn to_f64(value: Self) -> f64 {
        value as f64
    }
}
impl Real for f64 {
    fn from_f64(value: f64) -> Self {
        value
    }
    fn to_f64(value: Self) -> f64 {
        value
    }
}

trait R2rValue: R2rScalar + Equivalence + std::fmt::Debug + PartialEq {
    fn from_complex(value: Complex<f64>) -> Self;
    fn to_complex(value: Self) -> Complex<f64>;
}

impl R2rValue for f32 {
    fn from_complex(value: Complex<f64>) -> Self {
        value.re as f32
    }

    fn to_complex(value: Self) -> Complex<f64> {
        Complex::new(value as f64, 0.0)
    }
}

impl R2rValue for f64 {
    fn from_complex(value: Complex<f64>) -> Self {
        value.re
    }

    fn to_complex(value: Self) -> Complex<f64> {
        Complex::new(value, 0.0)
    }
}

impl R2rValue for Complex<f32> {
    fn from_complex(value: Complex<f64>) -> Self {
        Complex::new(value.re as f32, value.im as f32)
    }

    fn to_complex(value: Self) -> Complex<f64> {
        Complex::new(value.re as f64, value.im as f64)
    }
}

impl R2rValue for Complex<f64> {
    fn from_complex(value: Complex<f64>) -> Self {
        value
    }

    fn to_complex(value: Self) -> Complex<f64> {
        value
    }
}

fn fill_r2r<T: R2rValue>(storage: &mut [T], snapshot: &Snapshot, values: &[Complex<f64>]) {
    assert_eq!(storage.len(), snapshot.offsets.len());
    for (slot, &offset) in storage.iter_mut().zip(&snapshot.offsets) {
        *slot = <T as R2rValue>::from_complex(values[offset]);
    }
}

fn fill_complex<R: Real>(storage: &mut [Complex<R>], snapshot: &Snapshot, values: &[Complex<f64>]) {
    assert_eq!(storage.len(), snapshot.offsets.len());
    for (slot, &offset) in storage.iter_mut().zip(&snapshot.offsets) {
        let value = values[offset];
        *slot = Complex::new(
            <R as Real>::from_f64(value.re),
            <R as Real>::from_f64(value.im),
        );
    }
}
fn fill_real<R: Real>(storage: &mut [R], snapshot: &Snapshot, values: &[Complex<f64>]) {
    assert_eq!(storage.len(), snapshot.offsets.len());
    for (slot, &offset) in storage.iter_mut().zip(&snapshot.offsets) {
        let value = values[offset];
        assert_eq!(value.im, 0.0);
        *slot = <R as Real>::from_f64(value.re);
    }
}

fn compare(actual: f64, expected: f64, abs: f64, relative: f64, label: &str) -> Result<(), String> {
    if !actual.is_finite() || !expected.is_finite() {
        return Err(format!(
            "{label}: non-finite comparison value actual={actual:?} expected={expected:?}"
        ));
    }
    let bound = abs + relative * actual.abs().max(expected.abs());
    if (actual - expected).abs() > bound {
        return Err(format!(
            "{label}: actual={actual:.17e} expected={expected:.17e} error={:.3e} bound={bound:.3e}",
            (actual - expected).abs()
        ));
    }
    Ok(())
}

fn check_complex<R: Real>(
    storage: &[Complex<R>],
    snapshot: &Snapshot,
    expected: &[Complex<f64>],
    precision: Precision,
    label: &str,
) {
    assert_eq!(storage.len(), snapshot.offsets.len());
    let (abs, relative) = tolerance(precision);
    for (value, &offset) in storage.iter().zip(&snapshot.offsets) {
        let expected = expected[offset];
        compare(
            R::to_f64(value.re),
            expected.re,
            abs,
            relative,
            &format!("{label} offset={offset} real"),
        )
        .unwrap();
        compare(
            R::to_f64(value.im),
            expected.im,
            abs,
            relative,
            &format!("{label} offset={offset} imaginary"),
        )
        .unwrap();
    }
}
fn check_real<R: Real>(
    storage: &[R],
    snapshot: &Snapshot,
    expected: &[Complex<f64>],
    precision: Precision,
    label: &str,
) {
    assert_eq!(storage.len(), snapshot.offsets.len());
    let (abs, relative) = tolerance(precision);
    for (value, &offset) in storage.iter().zip(&snapshot.offsets) {
        compare(
            R::to_f64(*value),
            expected[offset].re,
            abs,
            relative,
            &format!("{label} offset={offset}"),
        )
        .unwrap();
    }
}

fn check_r2r<T: R2rValue>(
    storage: &[T],
    snapshot: &Snapshot,
    expected: &[Complex<f64>],
    precision: Precision,
    label: &str,
) {
    assert_eq!(storage.len(), snapshot.offsets.len());
    let (abs, relative) = tolerance(precision);
    for (value, &offset) in storage.iter().zip(&snapshot.offsets) {
        let actual = <T as R2rValue>::to_complex(*value);
        compare(
            actual.re,
            expected[offset].re,
            abs,
            relative,
            &format!("{label} offset={offset} real"),
        )
        .unwrap();
        compare(
            actual.im,
            expected[offset].im,
            abs,
            relative,
            &format!("{label} offset={offset} imaginary"),
        )
        .unwrap();
    }
}

macro_rules! snap {
    ($layout:expr, $array:expr, $output:expr, $label:expr) => {
        snapshot(
            $array.pencil(),
            $array.extra_shape(),
            &$layout,
            $output,
            $array.len(),
            $label,
        )
    };
}
macro_rules! check_c {
    ($array:expr, $snapshot:expr, $expected:expr, $fixture:expr, $rank:expr, $method:expr, $label:expr) => {
        check_complex(
            $array,
            $snapshot,
            $expected,
            $fixture.precision,
            &format!("rank {} {} {:?} {}", $rank, $fixture.case, $method, $label),
        )
    };
}
macro_rules! check_r {
    ($array:expr, $snapshot:expr, $expected:expr, $fixture:expr, $rank:expr, $method:expr, $label:expr) => {
        check_real(
            $array,
            $snapshot,
            $expected,
            $fixture.precision,
            &format!("rank {} {} {:?} {}", $rank, $fixture.case, $method, $label),
        )
    };
}
macro_rules! check_r2r {
    ($array:expr, $snapshot:expr, $expected:expr, $fixture:expr, $rank:expr, $method:expr, $label:expr) => {
        check_r2r(
            $array,
            $snapshot,
            $expected,
            $fixture.precision,
            &format!("rank {} {} {:?} {}", $rank, $fixture.case, $method, $label),
        )
    };
}

fn extra(fixture: &Fixture) -> ExtraShape {
    ExtraShape::new(fixture.extra.clone().into_boxed_slice()).unwrap()
}

fn axis_selection<const N: usize>(fixture: &Fixture) -> AxisSelection<N> {
    AxisSelection::from_indices(fixture.selection.iter().copied()).unwrap()
}

fn r2r_kind(kind: R2rAxisKind) -> Option<R2rKind> {
    match kind {
        R2rAxisKind::None | R2rAxisKind::Fft | R2rAxisKind::Rfft => None,
        R2rAxisKind::DctI => Some(R2rKind::DctI),
        R2rAxisKind::DctII => Some(R2rKind::DctII),
        R2rAxisKind::DctIII => Some(R2rKind::DctIII),
        R2rAxisKind::DctIV => Some(R2rKind::DctIV),
        R2rAxisKind::DstI => Some(R2rKind::DstI),
        R2rAxisKind::DstII => Some(R2rKind::DstII),
        R2rAxisKind::DstIII => Some(R2rKind::DstIII),
        R2rAxisKind::DstIV => Some(R2rKind::DstIV),
        R2rAxisKind::Dht => None,
    }
}

fn r2r_kinds<const N: usize>(fixture: &Fixture) -> [Option<R2rKind>; N] {
    fixture
        .axis_kinds
        .iter()
        .copied()
        .map(r2r_kind)
        .collect::<Vec<_>>()
        .try_into()
        .expect("fixture axis kind rank was validated")
}

fn mixed_transform(kind: R2rAxisKind) -> AxisTransform {
    match kind {
        R2rAxisKind::None => AxisTransform::None,
        R2rAxisKind::Fft => AxisTransform::Fft,
        R2rAxisKind::Rfft => AxisTransform::Rfft,
        R2rAxisKind::DctI => AxisTransform::R2r(AxisR2rKind::Fftw(R2rKind::DctI)),
        R2rAxisKind::DctII => AxisTransform::R2r(AxisR2rKind::Fftw(R2rKind::DctII)),
        R2rAxisKind::DctIII => AxisTransform::R2r(AxisR2rKind::Fftw(R2rKind::DctIII)),
        R2rAxisKind::DctIV => AxisTransform::R2r(AxisR2rKind::Fftw(R2rKind::DctIV)),
        R2rAxisKind::DstI => AxisTransform::R2r(AxisR2rKind::Fftw(R2rKind::DstI)),
        R2rAxisKind::DstII => AxisTransform::R2r(AxisR2rKind::Fftw(R2rKind::DstII)),
        R2rAxisKind::DstIII => AxisTransform::R2r(AxisR2rKind::Fftw(R2rKind::DstIII)),
        R2rAxisKind::DstIV => AxisTransform::R2r(AxisR2rKind::Fftw(R2rKind::DstIV)),
        R2rAxisKind::Dht => AxisTransform::R2r(AxisR2rKind::Dht),
    }
}

fn mixed_transforms<const N: usize>(fixture: &Fixture) -> [AxisTransform; N] {
    fixture
        .axis_kinds
        .iter()
        .copied()
        .map(mixed_transform)
        .collect::<Vec<_>>()
        .try_into()
        .expect("mixed fixture axis rank was validated")
}

fn c2c_case<R: Real, const N: usize, const M: usize>(
    fixture: &Fixture,
    topology: &Arc<MpiTopology<M>>,
    method: TransposeMethod,
    permute_dims: bool,
    rank: i32,
) where
    Complex<R>: Equivalence,
{
    let shape: [usize; N] = fixture.shape.as_slice().try_into().unwrap();
    let extra = extra(fixture);
    let layout = Layout {
        input: shape,
        output: shape,
        extra: &fixture.extra,
        permute_dims,
    };
    let selection = axis_selection::<N>(fixture);
    let plan = C2cPlan::from_shape_with_selection_and_layout(
        Arc::clone(topology),
        shape,
        extra.clone(),
        selection,
        DistributedLayout {
            transpose_method: method,
            permute_dims,
        },
    )
    .unwrap();
    let mut source = plan.allocate_input().unwrap();
    let source_snapshot = snap!(layout, source, false, "C2C source");
    ownership(source.pencil(), &source_snapshot, "C2C source");
    fill_complex(source.as_mut_slice(), &source_snapshot, &fixture.input);
    let source_before = source.as_slice().to_vec();
    let mut output = plan.allocate_output().unwrap();
    let output_snapshot = snap!(layout, output, true, "C2C output");
    ownership(output.pencil(), &output_snapshot, "C2C output");
    let mut workspace = plan.allocate_out_of_place_workspace().unwrap();
    plan.forward(&source, &mut output, &mut workspace).unwrap();
    assert_eq!(source.as_slice(), source_before.as_slice());
    check_c!(
        output.as_slice(),
        &output_snapshot,
        &fixture.forward,
        fixture,
        rank,
        method,
        "C2C forward"
    );
    let mut inverse_source = plan.allocate_output().unwrap();
    let inverse_snapshot = snap!(layout, inverse_source, true, "C2C inverse source");
    fill_complex(
        inverse_source.as_mut_slice(),
        &inverse_snapshot,
        &fixture.inverse_input,
    );
    let inverse_before = inverse_source.as_slice().to_vec();
    let mut recovered = plan.allocate_input().unwrap();
    let recovered_snapshot = snap!(layout, recovered, false, "C2C recovered");
    plan.inverse(&inverse_source, &mut recovered, &mut workspace)
        .unwrap();
    assert_eq!(inverse_source.as_slice(), inverse_before.as_slice());
    check_c!(
        recovered.as_slice(),
        &recovered_snapshot,
        &fixture.inverse,
        fixture,
        rank,
        method,
        "OOP inverse"
    );
    let mut backward = plan.allocate_input().unwrap();
    let backward_snapshot = snap!(layout, backward, false, "C2C backward");
    ownership(backward.pencil(), &backward_snapshot, "C2C backward");
    plan.backward(&inverse_source, &mut backward, &mut workspace)
        .unwrap();
    assert_eq!(inverse_source.as_slice(), inverse_before.as_slice());
    check_c!(
        backward.as_slice(),
        &backward_snapshot,
        &fixture.backward,
        fixture,
        rank,
        method,
        "C2C backward"
    );
    let mut inplace = plan.allocate_in_place().unwrap();
    {
        let mut view = inplace.view_mut().unwrap();
        let snapshot = snap!(layout, view, false, "C2C in-place input");
        fill_complex(view.as_mut_slice(), &snapshot, &fixture.input);
    }
    let mut inplace_workspace = plan.allocate_in_place_workspace().unwrap();
    plan.forward_in_place(&mut inplace, &mut inplace_workspace)
        .unwrap();
    {
        let view = inplace.view().unwrap();
        let snapshot = snap!(layout, view, true, "C2C in-place output");
        check_c!(
            view.as_slice(),
            &snapshot,
            &fixture.forward,
            fixture,
            rank,
            method,
            "in-place forward"
        );
    }
    {
        let mut view = inplace.view_mut().unwrap();
        let snapshot = snap!(layout, view, true, "C2C in-place inverse input");
        fill_complex(view.as_mut_slice(), &snapshot, &fixture.inverse_input);
    }
    plan.inverse_in_place(&mut inplace, &mut inplace_workspace)
        .unwrap();
    let view = inplace.view().unwrap();
    let snapshot = snap!(layout, view, false, "C2C in-place recovered");
    check_c!(
        view.as_slice(),
        &snapshot,
        &fixture.inverse,
        fixture,
        rank,
        method,
        "in-place inverse"
    );
    plan.forward_in_place(&mut inplace, &mut inplace_workspace)
        .unwrap();
    {
        let mut view = inplace.view_mut().unwrap();
        let snapshot = snap!(layout, view, true, "C2C in-place backward input");
        fill_complex(view.as_mut_slice(), &snapshot, &fixture.inverse_input);
    }
    plan.backward_in_place(&mut inplace, &mut inplace_workspace)
        .unwrap();
    let view = inplace.view().unwrap();
    let snapshot = snap!(layout, view, false, "C2C in-place backward");
    check_c!(
        view.as_slice(),
        &snapshot,
        &fixture.backward,
        fixture,
        rank,
        method,
        "in-place backward"
    );
}

fn c2c_methods<R: Real, const N: usize, const M: usize>(
    fixture: &Fixture,
    topology: &Arc<MpiTopology<M>>,
    permute_dims: bool,
    rank: i32,
) where
    Complex<R>: Equivalence,
{
    for method in [TransposeMethod::AllToAllv, TransposeMethod::PointToPoint] {
        c2c_case::<R, N, M>(fixture, topology, method, permute_dims, rank);
    }
}

fn mixed_c2c_case<R: Real, const N: usize, const M: usize>(
    fixture: &Fixture,
    topology: &Arc<MpiTopology<M>>,
    method: TransposeMethod,
    permute_dims: bool,
    rank: i32,
) where
    Complex<R>: Equivalence,
{
    let shape: [usize; N] = fixture.shape.as_slice().try_into().unwrap();
    let extra = extra(fixture);
    let transforms = mixed_transforms::<N>(fixture);
    let layout = Layout {
        input: shape,
        output: shape,
        extra: &fixture.extra,
        permute_dims,
    };
    let plan = MixedC2cPlan::<R, N, M>::from_shape_with_layout(
        Arc::clone(topology),
        shape,
        extra.clone(),
        transforms,
        DistributedLayout {
            transpose_method: method,
            permute_dims,
        },
    )
    .unwrap();
    assert_eq!(plan.transforms(), transforms);
    let mut source = plan.allocate_input().unwrap();
    let source_snapshot = snap!(layout, source, false, "mixed C2C source");
    ownership(source.pencil(), &source_snapshot, "mixed C2C source");
    fill_complex(source.as_mut_slice(), &source_snapshot, &fixture.input);
    let source_before = source.as_slice().to_vec();
    let mut output = plan.allocate_output().unwrap();
    let output_snapshot = snap!(layout, output, true, "mixed C2C output");
    ownership(output.pencil(), &output_snapshot, "mixed C2C output");
    let mut workspace = plan.allocate_workspace().unwrap();
    plan.forward(&source, &mut output, &mut workspace).unwrap();
    assert_eq!(source.as_slice(), source_before.as_slice());
    check_c!(
        output.as_slice(),
        &output_snapshot,
        &fixture.forward,
        fixture,
        rank,
        method,
        "mixed C2C forward"
    );

    let mut inverse_source = plan.allocate_output().unwrap();
    let inverse_snapshot = snap!(layout, inverse_source, true, "mixed C2C inverse source");
    fill_complex(
        inverse_source.as_mut_slice(),
        &inverse_snapshot,
        &fixture.inverse_input,
    );
    let inverse_before = inverse_source.as_slice().to_vec();
    let mut recovered = plan.allocate_input().unwrap();
    let recovered_snapshot = snap!(layout, recovered, false, "mixed C2C recovered");
    plan.inverse(&inverse_source, &mut recovered, &mut workspace)
        .unwrap();
    assert_eq!(inverse_source.as_slice(), inverse_before.as_slice());
    check_c!(
        recovered.as_slice(),
        &recovered_snapshot,
        &fixture.inverse,
        fixture,
        rank,
        method,
        "mixed C2C inverse"
    );

    let mut backward = plan.allocate_input().unwrap();
    let backward_snapshot = snap!(layout, backward, false, "mixed C2C backward");
    plan.backward(&inverse_source, &mut backward, &mut workspace)
        .unwrap();
    assert_eq!(inverse_source.as_slice(), inverse_before.as_slice());
    check_c!(
        backward.as_slice(),
        &backward_snapshot,
        &fixture.backward,
        fixture,
        rank,
        method,
        "mixed C2C backward"
    );

    let mut inplace = plan.allocate_in_place().unwrap();
    {
        let mut view = inplace.view_mut().unwrap();
        let snapshot = snap!(layout, view, false, "mixed C2C in-place input");
        fill_complex(view.as_mut_slice(), &snapshot, &fixture.input);
    }
    let pointer = inplace.view().unwrap().as_slice().as_ptr();
    let mut inplace_workspace = plan.allocate_in_place_workspace().unwrap();
    plan.forward_in_place(&mut inplace, &mut inplace_workspace)
        .unwrap();
    {
        let view = inplace.view().unwrap();
        assert_eq!(view.as_slice().as_ptr(), pointer);
        let snapshot = snap!(layout, view, true, "mixed C2C in-place output");
        check_c!(
            view.as_slice(),
            &snapshot,
            &fixture.forward,
            fixture,
            rank,
            method,
            "mixed C2C in-place forward"
        );
    }
    {
        let mut view = inplace.view_mut().unwrap();
        let snapshot = snap!(layout, view, true, "mixed C2C in-place inverse source");
        fill_complex(view.as_mut_slice(), &snapshot, &fixture.inverse_input);
    }
    plan.inverse_in_place(&mut inplace, &mut inplace_workspace)
        .unwrap();
    {
        let view = inplace.view().unwrap();
        assert_eq!(view.as_slice().as_ptr(), pointer);
        let snapshot = snap!(layout, view, false, "mixed C2C in-place inverse");
        check_c!(
            view.as_slice(),
            &snapshot,
            &fixture.inverse,
            fixture,
            rank,
            method,
            "mixed C2C in-place inverse"
        );
    }
    plan.forward_in_place(&mut inplace, &mut inplace_workspace)
        .unwrap();
    {
        let mut view = inplace.view_mut().unwrap();
        let snapshot = snap!(layout, view, true, "mixed C2C in-place backward source");
        fill_complex(view.as_mut_slice(), &snapshot, &fixture.inverse_input);
    }
    plan.backward_in_place(&mut inplace, &mut inplace_workspace)
        .unwrap();
    let view = inplace.view().unwrap();
    assert_eq!(view.as_slice().as_ptr(), pointer);
    let snapshot = snap!(layout, view, false, "mixed C2C in-place backward");
    check_c!(
        view.as_slice(),
        &snapshot,
        &fixture.backward,
        fixture,
        rank,
        method,
        "mixed C2C in-place backward"
    );
}

fn mixed_c2c_methods<R: Real, const N: usize, const M: usize>(
    fixture: &Fixture,
    topology: &Arc<MpiTopology<M>>,
    permute_dims: bool,
    rank: i32,
) where
    Complex<R>: Equivalence,
{
    for method in [TransposeMethod::AllToAllv, TransposeMethod::PointToPoint] {
        mixed_c2c_case::<R, N, M>(fixture, topology, method, permute_dims, rank);
    }
}

fn r2c_case<R: Real, const N: usize, const M: usize>(
    fixture: &Fixture,
    topology: &Arc<MpiTopology<M>>,
    method: TransposeMethod,
    permute_dims: bool,
    rank: i32,
) where
    Complex<R>: Equivalence,
{
    let input_shape: [usize; N] = fixture.shape.as_slice().try_into().unwrap();
    let selection = axis_selection::<N>(fixture);
    let output_shape: [usize; N] = reduced(
        Kind::R2c,
        &fixture.shape,
        &fixture.selection,
        &fixture.axis_kinds,
    )
    .as_slice()
    .try_into()
    .unwrap();
    let extra = extra(fixture);
    let layout = Layout {
        input: input_shape,
        output: output_shape,
        extra: &fixture.extra,
        permute_dims,
    };
    let plan = R2cPlan::from_shape_with_selection_and_layout(
        Arc::clone(topology),
        input_shape,
        extra.clone(),
        selection,
        DistributedLayout {
            transpose_method: method,
            permute_dims,
        },
    )
    .unwrap();
    let mut source = plan.allocate_input().unwrap();
    let source_snapshot = snap!(layout, source, false, "R2C source");
    ownership(source.pencil(), &source_snapshot, "R2C source");
    fill_real(source.as_mut_slice(), &source_snapshot, &fixture.input);
    let source_before = source.as_slice().to_vec();
    let mut output = plan.allocate_output().unwrap();
    let output_snapshot = snap!(layout, output, true, "R2C output");
    ownership(output.pencil(), &output_snapshot, "R2C output");
    let mut workspace = plan.allocate_workspace().unwrap();
    plan.forward(&source, &mut output, &mut workspace).unwrap();
    assert_eq!(source.as_slice(), source_before.as_slice());
    check_c!(
        output.as_slice(),
        &output_snapshot,
        &fixture.forward,
        fixture,
        rank,
        method,
        "R2C forward"
    );
    let mut inverse_source = plan.allocate_output().unwrap();
    let inverse_snapshot = snap!(layout, inverse_source, true, "C2R inverse source");
    fill_complex(
        inverse_source.as_mut_slice(),
        &inverse_snapshot,
        &fixture.inverse_input,
    );
    let inverse_before = inverse_source.as_slice().to_vec();
    let mut recovered = plan.allocate_input().unwrap();
    let recovered_snapshot = snap!(layout, recovered, false, "C2R recovered");
    plan.inverse(&inverse_source, &mut recovered, &mut workspace)
        .unwrap();
    assert_eq!(inverse_source.as_slice(), inverse_before.as_slice());
    check_r!(
        recovered.as_slice(),
        &recovered_snapshot,
        &fixture.inverse,
        fixture,
        rank,
        method,
        "C2R inverse"
    );
    let mut backward = plan.allocate_input().unwrap();
    let backward_snapshot = snap!(layout, backward, false, "C2R backward");
    plan.backward(&inverse_source, &mut backward, &mut workspace)
        .unwrap();
    assert_eq!(inverse_source.as_slice(), inverse_before.as_slice());
    check_r!(
        backward.as_slice(),
        &backward_snapshot,
        &fixture.backward,
        fixture,
        rank,
        method,
        "R2C backward"
    );

    let mut inplace = plan.allocate_in_place().unwrap();
    {
        let mut view = inplace.real_view_mut().unwrap();
        let snapshot = snap!(layout, view, false, "R2C in-place input");
        fill_real(view.as_mut_slice(), &snapshot, &fixture.input);
    }
    let mut inplace_workspace = plan.allocate_in_place_workspace().unwrap();
    plan.forward_in_place(&mut inplace, &mut inplace_workspace)
        .unwrap();
    {
        let view = inplace.complex_view().unwrap();
        let snapshot = snap!(layout, view, true, "R2C in-place output");
        check_c!(
            view.as_slice(),
            &snapshot,
            &fixture.forward,
            fixture,
            rank,
            method,
            "R2C in-place forward"
        );
    }
    {
        let mut view = inplace.complex_view_mut().unwrap();
        let snapshot = snap!(layout, view, true, "C2R in-place inverse input");
        fill_complex(view.as_mut_slice(), &snapshot, &fixture.inverse_input);
    }
    plan.inverse_in_place(&mut inplace, &mut inplace_workspace)
        .unwrap();
    {
        let view = inplace.real_view().unwrap();
        let snapshot = snap!(layout, view, false, "C2R in-place recovered");
        check_r!(
            view.as_slice(),
            &snapshot,
            &fixture.inverse,
            fixture,
            rank,
            method,
            "C2R in-place inverse"
        );
    }
    plan.forward_in_place(&mut inplace, &mut inplace_workspace)
        .unwrap();
    {
        let mut view = inplace.complex_view_mut().unwrap();
        let snapshot = snap!(layout, view, true, "R2C in-place backward input");
        fill_complex(view.as_mut_slice(), &snapshot, &fixture.inverse_input);
    }
    plan.backward_in_place(&mut inplace, &mut inplace_workspace)
        .unwrap();
    let view = inplace.real_view().unwrap();
    let snapshot = snap!(layout, view, false, "R2C in-place backward");
    check_r!(
        view.as_slice(),
        &snapshot,
        &fixture.backward,
        fixture,
        rank,
        method,
        "R2C in-place backward"
    );
}

fn r2c_methods<R: Real, const N: usize, const M: usize>(
    fixture: &Fixture,
    topology: &Arc<MpiTopology<M>>,
    permute_dims: bool,
    rank: i32,
) where
    Complex<R>: Equivalence,
{
    for method in [TransposeMethod::AllToAllv, TransposeMethod::PointToPoint] {
        r2c_case::<R, N, M>(fixture, topology, method, permute_dims, rank);
    }
}

fn mixed_r2c_case<R: Real, const N: usize, const M: usize>(
    fixture: &Fixture,
    topology: &Arc<MpiTopology<M>>,
    method: TransposeMethod,
    permute_dims: bool,
    rank: i32,
) where
    Complex<R>: Equivalence,
{
    let input_shape: [usize; N] = fixture.shape.as_slice().try_into().unwrap();
    let output_shape: [usize; N] = reduced(
        Kind::MixedR2c,
        &fixture.shape,
        &fixture.selection,
        &fixture.axis_kinds,
    )
    .as_slice()
    .try_into()
    .unwrap();
    let extra = extra(fixture);
    let transforms = mixed_transforms::<N>(fixture);
    let layout = Layout {
        input: input_shape,
        output: output_shape,
        extra: &fixture.extra,
        permute_dims,
    };
    let plan = MixedR2cPlan::<R, N, M>::from_shape_with_layout(
        Arc::clone(topology),
        input_shape,
        extra.clone(),
        transforms,
        DistributedLayout {
            transpose_method: method,
            permute_dims,
        },
    )
    .unwrap();
    assert_eq!(plan.transforms(), transforms);
    let reduction_axis = fixture
        .axis_kinds
        .iter()
        .position(|kind| *kind == R2rAxisKind::Rfft)
        .unwrap();
    assert_eq!(plan.reduction_axis(), reduction_axis);
    assert_eq!(plan.original_n(), input_shape[reduction_axis]);

    let mut source = plan.allocate_input().unwrap();
    let source_snapshot = snap!(layout, source, false, "mixed R2C source");
    ownership(source.pencil(), &source_snapshot, "mixed R2C source");
    fill_real(source.as_mut_slice(), &source_snapshot, &fixture.input);
    let source_before = source.as_slice().to_vec();
    let mut output = plan.allocate_output().unwrap();
    let output_snapshot = snap!(layout, output, true, "mixed R2C output");
    ownership(output.pencil(), &output_snapshot, "mixed R2C output");
    let mut workspace = plan.allocate_workspace().unwrap();
    plan.forward(&source, &mut output, &mut workspace).unwrap();
    assert_eq!(source.as_slice(), source_before.as_slice());
    check_c!(
        output.as_slice(),
        &output_snapshot,
        &fixture.forward,
        fixture,
        rank,
        method,
        "mixed R2C forward"
    );

    let mut inverse_source = plan.allocate_output().unwrap();
    let inverse_snapshot = snap!(layout, inverse_source, true, "mixed C2R inverse source");
    fill_complex(
        inverse_source.as_mut_slice(),
        &inverse_snapshot,
        &fixture.inverse_input,
    );
    let inverse_before = inverse_source.as_slice().to_vec();
    let mut recovered = plan.allocate_input().unwrap();
    let recovered_snapshot = snap!(layout, recovered, false, "mixed C2R recovered");
    plan.inverse(&inverse_source, &mut recovered, &mut workspace)
        .unwrap_or_else(|error| {
            panic!(
                "{} {:?} mixed C2R inverse failed: {error:?}",
                fixture.case, method,
            )
        });
    assert_eq!(inverse_source.as_slice(), inverse_before.as_slice());
    check_r!(
        recovered.as_slice(),
        &recovered_snapshot,
        &fixture.inverse,
        fixture,
        rank,
        method,
        "mixed C2R inverse"
    );

    let mut backward = plan.allocate_input().unwrap();
    let backward_snapshot = snap!(layout, backward, false, "mixed C2R backward");
    plan.backward(&inverse_source, &mut backward, &mut workspace)
        .unwrap();
    assert_eq!(inverse_source.as_slice(), inverse_before.as_slice());
    check_r!(
        backward.as_slice(),
        &backward_snapshot,
        &fixture.backward,
        fixture,
        rank,
        method,
        "mixed R2C backward"
    );

    let mut inplace = plan.allocate_in_place().unwrap();
    {
        let mut view = inplace.real_view_mut().unwrap();
        let snapshot = snap!(layout, view, false, "mixed R2C in-place input");
        fill_real(view.as_mut_slice(), &snapshot, &fixture.input);
    }
    let pointer = inplace.real_view().unwrap().as_slice().as_ptr();
    let mut inplace_workspace = plan.allocate_in_place_workspace().unwrap();
    plan.forward_in_place(&mut inplace, &mut inplace_workspace)
        .unwrap();
    {
        let view = inplace.complex_view().unwrap();
        let snapshot = snap!(layout, view, true, "mixed R2C in-place output");
        check_c!(
            view.as_slice(),
            &snapshot,
            &fixture.forward,
            fixture,
            rank,
            method,
            "mixed R2C in-place forward"
        );
    }
    {
        let mut view = inplace.complex_view_mut().unwrap();
        let snapshot = snap!(layout, view, true, "mixed C2R in-place inverse source");
        fill_complex(view.as_mut_slice(), &snapshot, &fixture.inverse_input);
    }
    plan.inverse_in_place(&mut inplace, &mut inplace_workspace)
        .unwrap();
    {
        let view = inplace.real_view().unwrap();
        assert_eq!(view.as_slice().as_ptr(), pointer);
        let snapshot = snap!(layout, view, false, "mixed C2R in-place inverse");
        check_r!(
            view.as_slice(),
            &snapshot,
            &fixture.inverse,
            fixture,
            rank,
            method,
            "mixed C2R in-place inverse"
        );
    }
    plan.forward_in_place(&mut inplace, &mut inplace_workspace)
        .unwrap();
    {
        let mut view = inplace.complex_view_mut().unwrap();
        let snapshot = snap!(layout, view, true, "mixed C2R in-place backward source");
        fill_complex(view.as_mut_slice(), &snapshot, &fixture.inverse_input);
    }
    plan.backward_in_place(&mut inplace, &mut inplace_workspace)
        .unwrap();
    let view = inplace.real_view().unwrap();
    assert_eq!(view.as_slice().as_ptr(), pointer);
    let snapshot = snap!(layout, view, false, "mixed R2C in-place backward");
    check_r!(
        view.as_slice(),
        &snapshot,
        &fixture.backward,
        fixture,
        rank,
        method,
        "mixed R2C in-place backward"
    );
}

fn mixed_r2c_methods<R: Real, const N: usize, const M: usize>(
    fixture: &Fixture,
    topology: &Arc<MpiTopology<M>>,
    permute_dims: bool,
    rank: i32,
) where
    Complex<R>: Equivalence,
{
    for method in [TransposeMethod::AllToAllv, TransposeMethod::PointToPoint] {
        mixed_r2c_case::<R, N, M>(fixture, topology, method, permute_dims, rank);
    }
}

#[derive(Debug, PartialEq)]
struct R2rSnapshots<T>(Vec<T>, Vec<T>, Vec<T>, Vec<T>, Vec<T>, Vec<T>);

fn r2r_case<T: R2rValue, const N: usize, const M: usize>(
    fixture: &Fixture,
    topology: &Arc<MpiTopology<M>>,
    method: TransposeMethod,
    permute_dims: bool,
    rank: i32,
) -> R2rSnapshots<T> {
    let shape: [usize; N] = fixture.shape.as_slice().try_into().unwrap();
    let extra = extra(fixture);
    let layout = Layout {
        input: shape,
        output: shape,
        extra: &fixture.extra,
        permute_dims,
    };
    let kinds = r2r_kinds::<N>(fixture);
    let plan = R2rPlan::<T, N, M>::from_shape_with_layout(
        Arc::clone(topology),
        shape,
        extra.clone(),
        kinds,
        DistributedLayout {
            transpose_method: method,
            permute_dims,
        },
    )
    .unwrap();
    assert_eq!(plan.kinds(), kinds);
    let mut source = plan.allocate_input().unwrap();
    let source_snapshot = snap!(layout, source, false, "R2R source");
    if source_snapshot.global_count != 0 {
        ownership(source.pencil(), &source_snapshot, "R2R source");
    }
    fill_r2r(source.as_mut_slice(), &source_snapshot, &fixture.input);
    let source_before = source.as_slice().to_vec();
    let mut output = plan.allocate_output().unwrap();
    let output_snapshot = snap!(layout, output, true, "R2R output");
    if output_snapshot.global_count != 0 {
        ownership(output.pencil(), &output_snapshot, "R2R output");
    }
    let mut workspace = plan.allocate_workspace().unwrap();
    plan.forward(&source, &mut output, &mut workspace).unwrap();
    assert_eq!(source.as_slice(), source_before.as_slice());
    check_r2r!(
        output.as_slice(),
        &output_snapshot,
        &fixture.forward,
        fixture,
        rank,
        method,
        "R2R forward"
    );
    let forward_snapshot = output.as_slice().to_vec();

    let mut inverse_source = plan.allocate_output().unwrap();
    let inverse_source_snapshot = snap!(layout, inverse_source, true, "R2R inverse source");
    fill_r2r(
        inverse_source.as_mut_slice(),
        &inverse_source_snapshot,
        &fixture.inverse_input,
    );
    let inverse_source_before = inverse_source.as_slice().to_vec();
    let mut inverse = plan.allocate_input().unwrap();
    let inverse_snapshot = snap!(layout, inverse, false, "R2R inverse");
    plan.inverse(&inverse_source, &mut inverse, &mut workspace)
        .unwrap();
    assert_eq!(inverse_source.as_slice(), inverse_source_before.as_slice());
    check_r2r!(
        inverse.as_slice(),
        &inverse_snapshot,
        &fixture.inverse,
        fixture,
        rank,
        method,
        "OOP inverse"
    );
    let inverse_result = inverse.as_slice().to_vec();

    let mut backward = plan.allocate_input().unwrap();
    let backward_snapshot = snap!(layout, backward, false, "R2R backward");
    plan.backward(&inverse_source, &mut backward, &mut workspace)
        .unwrap();
    assert_eq!(inverse_source.as_slice(), inverse_source_before.as_slice());
    check_r2r!(
        backward.as_slice(),
        &backward_snapshot,
        &fixture.backward,
        fixture,
        rank,
        method,
        "R2R backward"
    );
    let backward_result = backward.as_slice().to_vec();

    let mut inplace = plan.allocate_in_place().unwrap();
    {
        let mut view = inplace.view_mut().unwrap();
        let snapshot = snap!(layout, view, false, "R2R in-place input");
        fill_r2r(view.as_mut_slice(), &snapshot, &fixture.input);
    }
    let pointer = inplace.view().unwrap().as_slice().as_ptr();
    let mut inplace_workspace = plan.allocate_in_place_workspace().unwrap();
    plan.forward_in_place(&mut inplace, &mut inplace_workspace)
        .unwrap();
    assert_eq!(inplace.state(), R2rState::Output);
    assert_eq!(inplace.view().unwrap().as_slice().as_ptr(), pointer);
    {
        let view = inplace.view().unwrap();
        let snapshot = snap!(layout, view, true, "R2R in-place forward");
        check_r2r!(
            view.as_slice(),
            &snapshot,
            &fixture.forward,
            fixture,
            rank,
            method,
            "in-place forward"
        );
    }
    let inplace_forward = inplace.view().unwrap().as_slice().to_vec();

    {
        let mut view = inplace.view_mut().unwrap();
        let snapshot = snap!(layout, view, true, "R2R in-place inverse source");
        fill_r2r(view.as_mut_slice(), &snapshot, &fixture.inverse_input);
    }
    plan.inverse_in_place(&mut inplace, &mut inplace_workspace)
        .unwrap();
    assert_eq!(inplace.state(), R2rState::Input);
    assert_eq!(inplace.view().unwrap().as_slice().as_ptr(), pointer);
    {
        let view = inplace.view().unwrap();
        let snapshot = snap!(layout, view, false, "R2R in-place inverse");
        check_r2r!(
            view.as_slice(),
            &snapshot,
            &fixture.inverse,
            fixture,
            rank,
            method,
            "in-place inverse"
        );
    }
    let inplace_inverse = inplace.view().unwrap().as_slice().to_vec();

    plan.forward_in_place(&mut inplace, &mut inplace_workspace)
        .unwrap();
    {
        let mut view = inplace.view_mut().unwrap();
        let snapshot = snap!(layout, view, true, "R2R in-place backward source");
        fill_r2r(view.as_mut_slice(), &snapshot, &fixture.inverse_input);
    }
    plan.backward_in_place(&mut inplace, &mut inplace_workspace)
        .unwrap();
    assert_eq!(inplace.state(), R2rState::Input);
    assert_eq!(inplace.view().unwrap().as_slice().as_ptr(), pointer);
    {
        let view = inplace.view().unwrap();
        let snapshot = snap!(layout, view, false, "R2R in-place backward");
        check_r2r!(
            view.as_slice(),
            &snapshot,
            &fixture.backward,
            fixture,
            rank,
            method,
            "in-place backward"
        );
    }
    let inplace_backward = inplace.view().unwrap().as_slice().to_vec();

    // Reuse the same workspaces after all three directions.
    plan.forward(&source, &mut output, &mut workspace).unwrap();
    plan.inverse(&inverse_source, &mut inverse, &mut workspace)
        .unwrap();
    plan.forward_in_place(&mut inplace, &mut inplace_workspace)
        .unwrap();
    plan.inverse_in_place(&mut inplace, &mut inplace_workspace)
        .unwrap();

    R2rSnapshots(
        forward_snapshot,
        inverse_result,
        backward_result,
        inplace_forward,
        inplace_inverse,
        inplace_backward,
    )
}

fn r2r_methods<T: R2rValue + Equivalence, const N: usize, const M: usize>(
    fixture: &Fixture,
    topology: &Arc<MpiTopology<M>>,
    permute_dims: bool,
    rank: i32,
) {
    let alltoallv = r2r_case::<T, N, M>(
        fixture,
        topology,
        TransposeMethod::AllToAllv,
        permute_dims,
        rank,
    );
    let point_to_point = r2r_case::<T, N, M>(
        fixture,
        topology,
        TransposeMethod::PointToPoint,
        permute_dims,
        rank,
    );
    assert_eq!(alltoallv, point_to_point, "R2R transport parity");
}

fn dht_case<T: R2rValue, const N: usize, const M: usize>(
    fixture: &Fixture,
    topology: &Arc<MpiTopology<M>>,
    method: TransposeMethod,
    permute_dims: bool,
    rank: i32,
) -> R2rSnapshots<T> {
    let shape: [usize; N] = fixture.shape.as_slice().try_into().unwrap();
    let extra = extra(fixture);
    let layout = Layout {
        input: shape,
        output: shape,
        extra: &fixture.extra,
        permute_dims,
    };
    let selection = axis_selection::<N>(fixture);
    let plan = DhtPlan::<T, N, M>::from_shape_with_selection_and_layout(
        Arc::clone(topology),
        shape,
        extra.clone(),
        selection,
        DistributedLayout {
            transpose_method: method,
            permute_dims,
        },
    )
    .unwrap();
    assert_eq!(plan.selection(), selection);
    let mut source = plan.allocate_input().unwrap();
    let source_snapshot = snap!(layout, source, false, "DHT source");
    ownership(source.pencil(), &source_snapshot, "DHT source");
    fill_r2r(source.as_mut_slice(), &source_snapshot, &fixture.input);
    let source_before = source.as_slice().to_vec();
    let mut output = plan.allocate_output().unwrap();
    let output_snapshot = snap!(layout, output, true, "DHT output");
    ownership(output.pencil(), &output_snapshot, "DHT output");
    let mut workspace = plan.allocate_workspace().unwrap();
    plan.forward(&source, &mut output, &mut workspace).unwrap();
    assert_eq!(source.as_slice(), source_before.as_slice());
    check_r2r!(
        output.as_slice(),
        &output_snapshot,
        &fixture.forward,
        fixture,
        rank,
        method,
        "DHT forward"
    );
    let forward_result = output.as_slice().to_vec();

    let mut inverse_source = plan.allocate_output().unwrap();
    let inverse_source_snapshot = snap!(layout, inverse_source, true, "DHT inverse source");
    fill_r2r(
        inverse_source.as_mut_slice(),
        &inverse_source_snapshot,
        &fixture.inverse_input,
    );
    let inverse_before = inverse_source.as_slice().to_vec();
    let mut inverse = plan.allocate_input().unwrap();
    let inverse_snapshot = snap!(layout, inverse, false, "DHT inverse");
    plan.inverse(&inverse_source, &mut inverse, &mut workspace)
        .unwrap();
    assert_eq!(inverse_source.as_slice(), inverse_before.as_slice());
    check_r2r!(
        inverse.as_slice(),
        &inverse_snapshot,
        &fixture.inverse,
        fixture,
        rank,
        method,
        "DHT inverse"
    );
    let inverse_result = inverse.as_slice().to_vec();
    let mut backward = plan.allocate_input().unwrap();
    let backward_snapshot = snap!(layout, backward, false, "DHT backward");
    plan.backward(&inverse_source, &mut backward, &mut workspace)
        .unwrap();
    assert_eq!(inverse_source.as_slice(), inverse_before.as_slice());
    check_r2r!(
        backward.as_slice(),
        &backward_snapshot,
        &fixture.backward,
        fixture,
        rank,
        method,
        "DHT backward"
    );
    let backward_result = backward.as_slice().to_vec();

    let mut inplace = plan.allocate_in_place().unwrap();
    {
        let mut view = inplace.view_mut().unwrap();
        let snapshot = snap!(layout, view, false, "DHT in-place input");
        fill_r2r(view.as_mut_slice(), &snapshot, &fixture.input);
    }
    let pointer = inplace.view().unwrap().as_slice().as_ptr();
    let mut inplace_workspace = plan.allocate_in_place_workspace().unwrap();
    plan.forward_in_place(&mut inplace, &mut inplace_workspace)
        .unwrap();
    assert_eq!(inplace.state(), R2rState::Output);
    assert_eq!(inplace.view().unwrap().as_slice().as_ptr(), pointer);
    let view = inplace.view().unwrap();
    let snapshot = snap!(layout, view, true, "DHT in-place forward");
    check_r2r!(
        view.as_slice(),
        &snapshot,
        &fixture.forward,
        fixture,
        rank,
        method,
        "DHT in-place forward"
    );
    {
        let mut view = inplace.view_mut().unwrap();
        let snapshot = snap!(layout, view, true, "DHT in-place inverse source");
        fill_r2r(view.as_mut_slice(), &snapshot, &fixture.inverse_input);
    }
    plan.inverse_in_place(&mut inplace, &mut inplace_workspace)
        .unwrap();
    assert_eq!(inplace.state(), R2rState::Input);
    {
        let view = inplace.view().unwrap();
        assert_eq!(view.as_slice().as_ptr(), pointer);
        assert!(view.pencil().same_layout(plan.input_pencil().as_ref()));
        let snapshot = snap!(layout, view, false, "DHT in-place inverse");
        check_r2r!(
            view.as_slice(),
            &snapshot,
            &fixture.inverse,
            fixture,
            rank,
            method,
            "DHT in-place inverse"
        );
    }
    plan.forward_in_place(&mut inplace, &mut inplace_workspace)
        .unwrap();
    {
        let mut view = inplace.view_mut().unwrap();
        let snapshot = snap!(layout, view, true, "DHT in-place backward source");
        fill_r2r(view.as_mut_slice(), &snapshot, &fixture.inverse_input);
    }
    plan.backward_in_place(&mut inplace, &mut inplace_workspace)
        .unwrap();
    assert_eq!(inplace.state(), R2rState::Input);
    {
        let view = inplace.view().unwrap();
        assert_eq!(view.as_slice().as_ptr(), pointer);
        assert!(view.pencil().same_layout(plan.input_pencil().as_ref()));
        let snapshot = snap!(layout, view, false, "DHT in-place backward");
        check_r2r!(
            view.as_slice(),
            &snapshot,
            &fixture.backward,
            fixture,
            rank,
            method,
            "DHT in-place backward"
        );
    }

    // Reuse the same allocation and workspace through every direction again.
    for cycle in 0..2 {
        {
            let mut view = inplace.view_mut().unwrap();
            let snapshot = snap!(layout, view, false, "DHT repeated input");
            fill_r2r(view.as_mut_slice(), &snapshot, &fixture.input);
        }
        plan.forward_in_place(&mut inplace, &mut inplace_workspace)
            .unwrap();
        assert_eq!(inplace.state(), R2rState::Output);
        {
            let view = inplace.view().unwrap();
            assert_eq!(view.as_slice().as_ptr(), pointer);
            assert!(view.pencil().same_layout(plan.output_pencil().as_ref()));
            let snapshot = snap!(layout, view, true, "DHT repeated forward");
            check_r2r!(
                view.as_slice(),
                &snapshot,
                &fixture.forward,
                fixture,
                rank,
                method,
                &format!("DHT repeated forward {cycle}")
            );
        }
        {
            let mut view = inplace.view_mut().unwrap();
            let snapshot = snap!(layout, view, true, "DHT repeated inverse source");
            fill_r2r(view.as_mut_slice(), &snapshot, &fixture.inverse_input);
        }
        plan.inverse_in_place(&mut inplace, &mut inplace_workspace)
            .unwrap();
        assert_eq!(inplace.state(), R2rState::Input);
        {
            let view = inplace.view().unwrap();
            assert_eq!(view.as_slice().as_ptr(), pointer);
            assert!(view.pencil().same_layout(plan.input_pencil().as_ref()));
            let snapshot = snap!(layout, view, false, "DHT repeated inverse");
            check_r2r!(
                view.as_slice(),
                &snapshot,
                &fixture.inverse,
                fixture,
                rank,
                method,
                &format!("DHT repeated inverse {cycle}")
            );
        }
        plan.forward_in_place(&mut inplace, &mut inplace_workspace)
            .unwrap();
        {
            let mut view = inplace.view_mut().unwrap();
            let snapshot = snap!(layout, view, true, "DHT repeated backward source");
            fill_r2r(view.as_mut_slice(), &snapshot, &fixture.inverse_input);
        }
        plan.backward_in_place(&mut inplace, &mut inplace_workspace)
            .unwrap();
        assert_eq!(inplace.state(), R2rState::Input);
        {
            let view = inplace.view().unwrap();
            assert_eq!(view.as_slice().as_ptr(), pointer);
            assert!(view.pencil().same_layout(plan.input_pencil().as_ref()));
            let snapshot = snap!(layout, view, false, "DHT repeated backward");
            check_r2r!(
                view.as_slice(),
                &snapshot,
                &fixture.backward,
                fixture,
                rank,
                method,
                &format!("DHT repeated backward {cycle}")
            );
        }
    }

    R2rSnapshots(
        forward_result,
        inverse_result,
        backward_result,
        Vec::new(),
        Vec::new(),
        Vec::new(),
    )
}

fn dht_methods<T: R2rValue + Equivalence, const N: usize, const M: usize>(
    fixture: &Fixture,
    topology: &Arc<MpiTopology<M>>,
    permute_dims: bool,
    rank: i32,
) {
    let alltoallv = dht_case::<T, N, M>(
        fixture,
        topology,
        TransposeMethod::AllToAllv,
        permute_dims,
        rank,
    );
    let point_to_point = dht_case::<T, N, M>(
        fixture,
        topology,
        TransposeMethod::PointToPoint,
        permute_dims,
        rank,
    );
    assert_eq!(alltoallv.0, point_to_point.0, "DHT transport parity");
    assert_eq!(alltoallv.1, point_to_point.1, "DHT transport parity");
    assert_eq!(alltoallv.2, point_to_point.2, "DHT transport parity");
}

fn run_fixture(
    fixture: &Fixture,
    one: &Arc<MpiTopology<1>>,
    two: &Arc<MpiTopology<2>>,
    permute_dims: bool,
    rank: i32,
) {
    macro_rules! dispatch {
        ($run:ident, $real:ty, $n:expr) => {
            for m in 1..=std::cmp::min(2, $n - 1) {
                match m {
                    1 => $run::<$real, $n, 1>(fixture, one, permute_dims, rank),
                    2 => $run::<$real, $n, 2>(fixture, two, permute_dims, rank),
                    _ => unreachable!(),
                }
            }
        };
    }
    match (
        fixture.kind,
        fixture.element_kind,
        fixture.precision,
        fixture.shape.len(),
    ) {
        (Kind::C2c, _, Precision::F32, 2) => dispatch!(c2c_methods, f32, 2),
        (Kind::C2c, _, Precision::F64, 2) => dispatch!(c2c_methods, f64, 2),
        (Kind::C2c, _, Precision::F32, 3) => dispatch!(c2c_methods, f32, 3),
        (Kind::C2c, _, Precision::F64, 3) => dispatch!(c2c_methods, f64, 3),
        (Kind::C2c, _, Precision::F32, 4) => dispatch!(c2c_methods, f32, 4),
        (Kind::C2c, _, Precision::F64, 4) => dispatch!(c2c_methods, f64, 4),
        (Kind::R2c, _, Precision::F32, 2) => dispatch!(r2c_methods, f32, 2),
        (Kind::R2c, _, Precision::F64, 2) => dispatch!(r2c_methods, f64, 2),
        (Kind::R2c, _, Precision::F32, 3) => dispatch!(r2c_methods, f32, 3),
        (Kind::R2c, _, Precision::F64, 3) => dispatch!(r2c_methods, f64, 3),
        (Kind::R2c, _, Precision::F32, 4) => dispatch!(r2c_methods, f32, 4),
        (Kind::R2c, _, Precision::F64, 4) => dispatch!(r2c_methods, f64, 4),
        (Kind::MixedC2c, _, Precision::F32, 2) => dispatch!(mixed_c2c_methods, f32, 2),
        (Kind::MixedC2c, _, Precision::F64, 2) => dispatch!(mixed_c2c_methods, f64, 2),
        (Kind::MixedC2c, _, Precision::F32, 3) => dispatch!(mixed_c2c_methods, f32, 3),
        (Kind::MixedC2c, _, Precision::F64, 3) => dispatch!(mixed_c2c_methods, f64, 3),
        (Kind::MixedC2c, _, Precision::F32, 4) => dispatch!(mixed_c2c_methods, f32, 4),
        (Kind::MixedC2c, _, Precision::F64, 4) => dispatch!(mixed_c2c_methods, f64, 4),
        (Kind::MixedR2c, _, Precision::F32, 2) => dispatch!(mixed_r2c_methods, f32, 2),
        (Kind::MixedR2c, _, Precision::F64, 2) => dispatch!(mixed_r2c_methods, f64, 2),
        (Kind::MixedR2c, _, Precision::F32, 3) => dispatch!(mixed_r2c_methods, f32, 3),
        (Kind::MixedR2c, _, Precision::F64, 3) => dispatch!(mixed_r2c_methods, f64, 3),
        (Kind::MixedR2c, _, Precision::F32, 4) => dispatch!(mixed_r2c_methods, f32, 4),
        (Kind::MixedR2c, _, Precision::F64, 4) => dispatch!(mixed_r2c_methods, f64, 4),
        (Kind::R2r, ElementKind::Real, Precision::F32, 2) => dispatch!(r2r_methods, f32, 2),
        (Kind::R2r, ElementKind::Real, Precision::F64, 2) => dispatch!(r2r_methods, f64, 2),
        (Kind::R2r, ElementKind::Real, Precision::F32, 3) => dispatch!(r2r_methods, f32, 3),
        (Kind::R2r, ElementKind::Real, Precision::F64, 3) => dispatch!(r2r_methods, f64, 3),
        (Kind::R2r, ElementKind::Real, Precision::F32, 4) => dispatch!(r2r_methods, f32, 4),
        (Kind::R2r, ElementKind::Real, Precision::F64, 4) => dispatch!(r2r_methods, f64, 4),
        (Kind::R2r, ElementKind::Complex, Precision::F32, 2) => {
            dispatch!(r2r_methods, Complex<f32>, 2)
        }
        (Kind::R2r, ElementKind::Complex, Precision::F64, 2) => {
            dispatch!(r2r_methods, Complex<f64>, 2)
        }
        (Kind::R2r, ElementKind::Complex, Precision::F32, 3) => {
            dispatch!(r2r_methods, Complex<f32>, 3)
        }
        (Kind::R2r, ElementKind::Complex, Precision::F64, 3) => {
            dispatch!(r2r_methods, Complex<f64>, 3)
        }
        (Kind::R2r, ElementKind::Complex, Precision::F32, 4) => {
            dispatch!(r2r_methods, Complex<f32>, 4)
        }
        (Kind::R2r, ElementKind::Complex, Precision::F64, 4) => {
            dispatch!(r2r_methods, Complex<f64>, 4)
        }
        (Kind::Dht, ElementKind::Real, Precision::F32, 2) => dispatch!(dht_methods, f32, 2),
        (Kind::Dht, ElementKind::Real, Precision::F64, 2) => dispatch!(dht_methods, f64, 2),
        (Kind::Dht, ElementKind::Real, Precision::F32, 3) => dispatch!(dht_methods, f32, 3),
        (Kind::Dht, ElementKind::Real, Precision::F64, 3) => dispatch!(dht_methods, f64, 3),
        (Kind::Dht, ElementKind::Real, Precision::F32, 4) => dispatch!(dht_methods, f32, 4),
        (Kind::Dht, ElementKind::Real, Precision::F64, 4) => dispatch!(dht_methods, f64, 4),
        (Kind::Dht, ElementKind::Complex, Precision::F32, 2) => {
            dispatch!(dht_methods, Complex<f32>, 2)
        }
        (Kind::Dht, ElementKind::Complex, Precision::F64, 2) => {
            dispatch!(dht_methods, Complex<f64>, 2)
        }
        (Kind::Dht, ElementKind::Complex, Precision::F32, 3) => {
            dispatch!(dht_methods, Complex<f32>, 3)
        }
        (Kind::Dht, ElementKind::Complex, Precision::F64, 3) => {
            dispatch!(dht_methods, Complex<f64>, 3)
        }
        (Kind::Dht, ElementKind::Complex, Precision::F32, 4) => {
            dispatch!(dht_methods, Complex<f32>, 4)
        }
        (Kind::Dht, ElementKind::Complex, Precision::F64, 4) => {
            dispatch!(dht_methods, Complex<f64>, 4)
        }
        _ => panic!("unsupported fixture"),
    }
}

#[test]
fn parser_and_offset_self_check() {
    let values = "1 0\n2 0\n3 0\n4 0\n5 0\n6 0\n";
    let section = |name: &str| format!("section {name} complex 6\n{values}end\n");
    let backward_section = section("backward_expected");
    let valid = format!(
        "PENCIL_FFTW_REFERENCE 7\nruntime julia=1.12.6 fftw_jl=1.10.0 native=3.3.12 provider=fftw\ncase c2c_2d_2x3_f64\nkind c2c\nelement_kind complex\nprecision f64\noriginal_shape 2 3\nextra_shape\naxis_kinds none none\nselected_axes 0 1\n{}{}{}{}{}",
        section("input"),
        section("inverse_input"),
        section("forward_expected"),
        section("inverse_expected"),
        backward_section
    );
    assert_eq!(parse_fixture(&valid).unwrap().input.len(), 6);
    assert_eq!(parse_fixture(&valid).unwrap().backward.len(), 6);
    assert!(
        parse_fixture(&valid.replacen("PENCIL_FFTW_REFERENCE 7", "PENCIL_FFTW_REFERENCE 6", 1,))
            .is_err()
    );
    assert!(parse_fixture(&valid.replacen("selected_axes 0 1", "selected_axes 1 0", 1)).is_err());
    assert!(parse_fixture(&valid.replacen("selected_axes 0 1", "selected_axes 0 0", 1)).is_err());
    assert!(parse_fixture(&valid.replace("axis_kinds none none", "axis_kinds fft none")).is_err());
    assert!(parse_fixture(&valid.replace("axis_kinds none none", "axis_kinds rfft none")).is_err());

    let r2c_section = |name: &str, kind: &str, count: usize, values: &str| {
        format!("section {name} {kind} {count}\n{values}end\n")
    };
    let r2c = format!(
        "PENCIL_FFTW_REFERENCE 7\nruntime julia=1.12.6 fftw_jl=1.10.0 native=3.3.12 provider=fftw\ncase r2c_2d_2x3_f64\nkind r2c\nelement_kind real\nprecision f64\noriginal_shape 2 3\nextra_shape\naxis_kinds none none\nselected_axes 0 1\n{}{}{}{}{}",
        r2c_section("input", "real", 6, "1\n2\n3\n4\n5\n6\n"),
        r2c_section("inverse_input", "complex", 4, "1 0\n2 0\n3 0\n4 0\n"),
        r2c_section("forward_expected", "complex", 4, "1 0\n2 0\n3 0\n4 0\n"),
        r2c_section("inverse_expected", "real", 6, "1\n2\n3\n4\n5\n6\n"),
        r2c_section("backward_expected", "real", 6, "1\n2\n3\n4\n5\n6\n"),
    );
    let parsed_r2c = parse_fixture(&r2c).unwrap();
    assert_eq!(parsed_r2c.kind, Kind::R2c);
    assert!(parse_fixture(&r2c.replacen("selected_axes 0 1", "selected_axes", 1)).is_err());
    assert_eq!(parsed_r2c.backward.len(), 6);
    let r2c_without_backward = r2c.replacen(
        &r2c_section("backward_expected", "real", 6, "1\n2\n3\n4\n5\n6\n"),
        "",
        1,
    );
    assert!(parse_fixture(&r2c_without_backward).is_err());
    assert!(parse_fixture(&format!("{r2c}{backward_section}")).is_err());
    assert!(
        parse_fixture(&r2c.replacen(
            "section backward_expected real 6",
            "section backward_expected real 7",
            1,
        ))
        .is_err()
    );

    let mixed_c2c = format!(
        "PENCIL_FFTW_REFERENCE 7\nruntime julia=1.12.6 fftw_jl=1.10.0 native=3.3.12 provider=fftw\ncase mixed_c2c_2d_2x3_fft-dctii_f64\nkind mixed_c2c\nelement_kind complex\nprecision f64\noriginal_shape 2 3\nextra_shape\naxis_kinds fft dctii\nselected_axes 0 1\n{}{}{}{}{}",
        section("input"),
        section("inverse_input"),
        section("forward_expected"),
        section("inverse_expected"),
        backward_section,
    );
    let parsed_mixed_c2c = parse_fixture(&mixed_c2c).unwrap();
    assert_eq!(parsed_mixed_c2c.kind, Kind::MixedC2c);
    assert_eq!(parsed_mixed_c2c.element_kind, ElementKind::Complex);
    assert_eq!(parsed_mixed_c2c.original_n, None);
    assert!(
        parse_fixture(&mixed_c2c.replace("axis_kinds fft dctii", "axis_kinds rfft dctii")).is_err()
    );

    let mixed_r2c = format!(
        "PENCIL_FFTW_REFERENCE 7\nruntime julia=1.12.6 fftw_jl=1.10.0 native=3.3.12 provider=fftw\ncase mixed_r2c_2d_3x2_rfft-dht_f64\nkind mixed_r2c\nelement_kind real\nprecision f64\noriginal_shape 3 2\nextra_shape\naxis_kinds rfft dht\nselected_axes 0 1\noriginal_n 3\n{}{}{}{}{}",
        r2c_section("input", "real", 6, "1\n2\n3\n4\n5\n6\n"),
        r2c_section("inverse_input", "complex", 4, "1 0\n2 0\n3 0\n4 0\n"),
        r2c_section("forward_expected", "complex", 4, "1 0\n2 0\n3 0\n4 0\n"),
        r2c_section("inverse_expected", "real", 6, "1\n2\n3\n4\n5\n6\n"),
        r2c_section("backward_expected", "real", 6, "1\n2\n3\n4\n5\n6\n"),
    );
    let parsed_mixed_r2c = parse_fixture(&mixed_r2c).unwrap();
    assert_eq!(parsed_mixed_r2c.kind, Kind::MixedR2c);
    assert_eq!(parsed_mixed_r2c.original_n, Some(3));
    assert!(
        parse_fixture(&mixed_r2c.replace("axis_kinds rfft dht", "axis_kinds rfft fft")).is_err()
    );
    assert!(
        parse_fixture(&mixed_r2c.replace("axis_kinds rfft dht", "axis_kinds none dht")).is_err()
    );
    assert!(
        parse_fixture(&mixed_r2c.replace("axis_kinds rfft dht", "axis_kinds rfft rfft")).is_err()
    );
    assert!(
        parse_fixture(&mixed_r2c.replace("element_kind real", "element_kind complex")).is_err()
    );
    assert!(
        parse_fixture(&mixed_r2c.replace("axis_kinds rfft dht", "axis_kinds rfft none")).is_err()
    );
    assert!(parse_fixture(&mixed_r2c.replace("original_n 3", "original_n 4")).is_err());
    assert!(parse_fixture(&mixed_r2c.replace("original_n 3\n", "")).is_err());
    assert!(
        parse_fixture(&mixed_r2c.replace("original_n 3\n", "original_n 3\noriginal_n 3\n"))
            .is_err()
    );
    assert!(
        parse_fixture(&mixed_r2c.replace(
            "section forward_expected complex 4",
            "section forward_expected real 4"
        ))
        .is_err()
    );
    assert!(
        parse_fixture(&mixed_r2c.replace(
            "section backward_expected real 6",
            "section backward_expected real 5"
        ))
        .is_err()
    );
    assert!(
        parse_fixture(&r2c.replacen(
            "section backward_expected real 6\n1",
            "section backward_expected real 6\nNaN",
            1,
        ))
        .is_err()
    );

    let r2r_real = format!(
        "PENCIL_FFTW_REFERENCE 7\nruntime julia=1.12.6 fftw_jl=1.10.0 native=3.3.12 provider=fftw\ncase r2r_2d_2x3_dctii-none_f64\nkind r2r\nelement_kind real\nprecision f64\noriginal_shape 2 3\nextra_shape\naxis_kinds dctii none\nselected_axes 0\n{}{}{}{}{}",
        r2c_section("input", "real", 6, "1\n2\n3\n4\n5\n6\n"),
        r2c_section("inverse_input", "real", 6, "7\n8\n9\n10\n11\n12\n"),
        r2c_section("forward_expected", "real", 6, "13\n14\n15\n16\n17\n18\n"),
        r2c_section("inverse_expected", "real", 6, "19\n20\n21\n22\n23\n24\n"),
        r2c_section("backward_expected", "real", 6, "25\n26\n27\n28\n29\n30\n"),
    );
    let parsed_r2r_real = parse_fixture(&r2r_real).unwrap();
    assert_eq!(parsed_r2r_real.kind, Kind::R2r);
    assert_eq!(parsed_r2r_real.element_kind, ElementKind::Real);
    assert_eq!(
        parsed_r2r_real.axis_kinds,
        vec![R2rAxisKind::DctII, R2rAxisKind::None]
    );
    assert_eq!(parsed_r2r_real.selection, vec![0]);
    assert_eq!(parsed_r2r_real.forward.len(), 6);
    assert!(
        parse_fixture(&r2r_real.replace("axis_kinds dctii none", "axis_kinds fft none")).is_err()
    );
    assert!(
        parse_fixture(&r2r_real.replace("axis_kinds dctii none", "axis_kinds rfft none")).is_err()
    );
    let dht = r2r_real
        .replace("r2r_2d_2x3_dctii-none_f64", "dht_2d_2x3_dht-none_f64")
        .replace("kind r2r", "kind dht")
        .replace("axis_kinds dctii none", "axis_kinds dht none");
    let parsed_dht = parse_fixture(&dht).unwrap();
    assert_eq!(parsed_dht.kind, Kind::Dht);
    assert_eq!(parsed_dht.axis_kinds[0], R2rAxisKind::Dht);
    assert!(
        parse_fixture(&r2r_real.replace("axis_kinds dctii none", "axis_kinds dht none")).is_err()
    );
    let dht_with_dct_axis = dht.replace("axis_kinds dht none", "axis_kinds dctii none");
    assert!(parse_fixture(&dht_with_dct_axis).is_err());

    let complex_values = "1 0\n2 1\n3 2\n4 3\n5 4\n6 5\n";
    let r2r_complex = format!(
        "PENCIL_FFTW_REFERENCE 7\nruntime julia=1.12.6 fftw_jl=1.10.0 native=3.3.12 provider=fftw\ncase r2r_2d_2x3_none-dstiv_f64\nkind r2r\nelement_kind complex\nprecision f64\noriginal_shape 2 3\nextra_shape\naxis_kinds none dstiv\nselected_axes 1\n{}{}{}{}{}",
        r2c_section("input", "complex", 6, complex_values),
        r2c_section("inverse_input", "complex", 6, complex_values),
        r2c_section("forward_expected", "complex", 6, complex_values),
        r2c_section("inverse_expected", "complex", 6, complex_values),
        r2c_section("backward_expected", "complex", 6, complex_values),
    );
    let parsed_r2r_complex = parse_fixture(&r2r_complex).unwrap();
    assert_eq!(parsed_r2r_complex.kind, Kind::R2r);
    assert_eq!(parsed_r2r_complex.element_kind, ElementKind::Complex);
    assert_eq!(
        parsed_r2r_complex.axis_kinds,
        vec![R2rAxisKind::None, R2rAxisKind::DstIV]
    );
    assert_eq!(parsed_r2r_complex.selection, vec![1]);
    assert_eq!(parsed_r2r_complex.inverse_input.len(), 6);

    assert!(parse_fixture(&r2r_real.replacen("axis_kinds dctii none\n", "", 1)).is_err());
    assert!(
        parse_fixture(&r2r_real.replacen("axis_kinds dctii none", "axis_kinds dctii", 1)).is_err()
    );
    assert!(
        parse_fixture(&r2r_real.replacen("axis_kinds dctii none", "axis_kinds dctii none none", 1))
            .is_err()
    );
    assert!(
        parse_fixture(&r2r_real.replacen("axis_kinds dctii none", "axis_kinds unknown none", 1))
            .is_err()
    );
    assert!(parse_fixture(&r2r_real.replacen("selected_axes 0", "selected_axes 1", 1)).is_err());
    assert!(
        parse_fixture(&r2r_real.replacen("element_kind real", "element_kind scalar", 1)).is_err()
    );
    assert!(
        parse_fixture(&valid.replacen("element_kind complex", "element_kind real", 1)).is_err()
    );
    assert!(parse_fixture(&r2c.replacen("element_kind real", "element_kind complex", 1)).is_err());
    assert!(
        parse_fixture(&r2r_real.replacen("section input real 6", "section input complex 6", 1,))
            .is_err()
    );
    assert!(
        parse_fixture(&r2r_complex.replacen("section input complex 6", "section input real 6", 1,))
            .is_err()
    );
    assert!(
        parse_fixture(&r2r_real.replacen(
            "section forward_expected real 6",
            "section forward_expected real 5",
            1,
        ))
        .is_err()
    );
    let r2r_real_without_backward = r2r_real.replacen(
        &r2c_section("backward_expected", "real", 6, "25\n26\n27\n28\n29\n30\n"),
        "",
        1,
    );
    assert!(parse_fixture(&r2r_real_without_backward).is_err());
    assert!(
        parse_fixture(&r2r_real.replacen(
            "section forward_expected real 6\n13",
            "section forward_expected real 6\nNaN",
            1,
        ))
        .is_err()
    );
    assert!(
        parse_fixture(&r2r_complex.replacen(
            "section forward_expected complex 6\n1 0",
            "section forward_expected complex 6\n1 NaN",
            1,
        ))
        .is_err()
    );
    assert!(parse_fixture(&format!("{r2r_complex}unexpected\n")).is_err());

    for malformed in ["kind", "", "kind c2c trailing"] {
        assert!(field(malformed, "kind").is_err());
    }
    assert!(parse_fixture(&valid.replacen("kind c2c\n", "kind\n", 1)).is_err());
    assert_eq!(
        offsets(&[2], [2, 3], &[0..2, 0..3], [1, 0]),
        vec![0, 3, 1, 4, 2, 5, 6, 9, 7, 10, 8, 11]
    );
    assert!(parse_fixture(&valid.replacen("complex 6", "complex 7", 1)).is_err());
    assert!(parse_fixture(&valid.replacen("1 0", "NaN 0", 1)).is_err());
    assert!(parse_fixture(&valid.replacen("kind c2c\n", "kind c2c\nkind c2c\n", 1)).is_err());
    assert!(parse_fixture(&valid.replacen(&backward_section, "", 1)).is_err());
    assert!(
        parse_fixture(&valid.replacen(
            "section backward_expected complex 6",
            "section backward_expected real 6",
            1,
        ))
        .is_err()
    );
    assert!(
        parse_fixture(&valid.replacen(
            "section backward_expected complex 6",
            "section backward_expected complex 7",
            1,
        ))
        .is_err()
    );
    assert!(
        parse_fixture(&valid.replacen(
            "section backward_expected complex 6\n1 0",
            "section backward_expected complex 6\nNaN 0",
            1,
        ))
        .is_err()
    );
    assert!(parse_fixture(&format!("{valid}unexpected\n")).is_err());
    assert!(compare(1.0, 1.0, 1e-6, 1e-6, "equal").is_ok());
    assert!(compare(2.0, 1.0, 1e-6, 1e-6, "mismatch").is_err());
    assert!(compare(f64::NAN, 1.0, 1e-6, 1e-6, "nonfinite").is_err());
}

fn run_direction_c2c<R: Real, const N: usize, const M: usize>(
    topology: Arc<MpiTopology<M>>,
    fixture: &DirectionFixture,
    method: TransposeMethod,
    permute_dims: bool,
) where
    Complex<R>: Equivalence,
{
    let shape: [usize; N] = fixture.shape.clone().try_into().unwrap();
    let directions = FourierDirections::new(fixture.directions.clone().try_into().unwrap());
    let plan = C2cPlan::<R, N, M>::from_shape_with_layout(
        Arc::clone(&topology),
        shape,
        ExtraShape::scalar(),
        DistributedLayout {
            transpose_method: method,
            permute_dims,
        },
    )
    .unwrap()
    .with_fft_directions(directions)
    .unwrap();
    let layout = Layout {
        input: shape,
        output: shape,
        extra: &[],
        permute_dims,
    };
    let mut source = plan.allocate_input().unwrap();
    let input_snap = snapshot(
        plan.input_pencil(),
        &ExtraShape::scalar(),
        &layout,
        false,
        source.as_slice().len(),
        "direction input",
    );
    fill_complex(source.as_mut_slice(), &input_snap, &fixture.input);
    let mut output = plan.allocate_output().unwrap();
    let output_snap = snapshot(
        plan.output_pencil(),
        &ExtraShape::scalar(),
        &layout,
        true,
        output.as_slice().len(),
        "direction output",
    );
    let mut workspace = plan.allocate_out_of_place_workspace().unwrap();
    plan.forward(&source, &mut output, &mut workspace).unwrap();
    for (v, &offset) in output.as_slice().iter().zip(&output_snap.offsets) {
        let e = fixture.forward[offset];
        compare(R::to_f64(v.re), e.re, 3e-5, 3e-5, "direction forward real").unwrap();
        compare(R::to_f64(v.im), e.im, 3e-5, 3e-5, "direction forward imag").unwrap();
    }
    let mut spectrum = plan.allocate_output().unwrap();
    fill_complex(
        spectrum.as_mut_slice(),
        &output_snap,
        &fixture.inverse_input,
    );
    let mut inverse = plan.allocate_input().unwrap();
    plan.inverse(&spectrum, &mut inverse, &mut workspace)
        .unwrap();
    let mut backward = plan.allocate_input().unwrap();
    plan.backward(&spectrum, &mut backward, &mut workspace)
        .unwrap();
    for ((a, b), &offset) in inverse
        .as_slice()
        .iter()
        .zip(backward.as_slice())
        .zip(&input_snap.offsets)
    {
        let e = fixture.inverse[offset];
        compare(R::to_f64(a.re), e.re, 3e-5, 3e-5, "direction inverse real").unwrap();
        compare(R::to_f64(a.im), e.im, 3e-5, 3e-5, "direction inverse imag").unwrap();
        let e = fixture.backward[offset];
        compare(R::to_f64(b.re), e.re, 3e-5, 3e-5, "direction backward real").unwrap();
        compare(R::to_f64(b.im), e.im, 3e-5, 3e-5, "direction backward imag").unwrap();
    }
}

fn run_direction_mixed<R: Real, const N: usize, const M: usize>(
    topology: Arc<MpiTopology<M>>,
    fixture: &DirectionFixture,
    method: TransposeMethod,
    permute_dims: bool,
) where
    Complex<R>: Equivalence,
{
    let shape: [usize; N] = fixture.shape.clone().try_into().unwrap();
    let directions = FourierDirections::new(fixture.directions.clone().try_into().unwrap());
    let transforms: [AxisTransform; N] = fixture
        .transforms
        .iter()
        .map(|kind| match kind {
            R2rAxisKind::Fft => AxisTransform::Fft,
            R2rAxisKind::DctII => AxisTransform::R2r(AxisR2rKind::Fftw(R2rKind::DctII)),
            R2rAxisKind::Dht => AxisTransform::R2r(AxisR2rKind::Dht),
            _ => panic!("unsupported direction transform"),
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let plan = MixedC2cPlan::<R, N, M>::from_shape_with_layout(
        Arc::clone(&topology),
        shape,
        ExtraShape::scalar(),
        transforms,
        DistributedLayout {
            transpose_method: method,
            permute_dims,
        },
    )
    .unwrap()
    .with_fft_directions(directions)
    .unwrap();
    let layout = Layout {
        input: shape,
        output: shape,
        extra: &[],
        permute_dims,
    };
    let mut source = plan.allocate_input().unwrap();
    let input_snap = snapshot(
        plan.input_pencil(),
        &ExtraShape::scalar(),
        &layout,
        false,
        source.as_slice().len(),
        "direction input",
    );
    fill_complex(source.as_mut_slice(), &input_snap, &fixture.input);
    let mut output = plan.allocate_output().unwrap();
    let output_snap = snapshot(
        plan.output_pencil(),
        &ExtraShape::scalar(),
        &layout,
        true,
        output.as_slice().len(),
        "direction output",
    );
    let mut workspace = plan.allocate_out_of_place_workspace().unwrap();
    plan.forward(&source, &mut output, &mut workspace).unwrap();
    for (v, &offset) in output.as_slice().iter().zip(&output_snap.offsets) {
        let e = fixture.forward[offset];
        compare(R::to_f64(v.re), e.re, 3e-5, 3e-5, "direction forward real").unwrap();
        compare(R::to_f64(v.im), e.im, 3e-5, 3e-5, "direction forward imag").unwrap();
    }
    let mut spectrum = plan.allocate_output().unwrap();
    fill_complex(
        spectrum.as_mut_slice(),
        &output_snap,
        &fixture.inverse_input,
    );
    let mut inverse = plan.allocate_input().unwrap();
    plan.inverse(&spectrum, &mut inverse, &mut workspace)
        .unwrap();
    let mut backward = plan.allocate_input().unwrap();
    plan.backward(&spectrum, &mut backward, &mut workspace)
        .unwrap();
    for ((a, b), &offset) in inverse
        .as_slice()
        .iter()
        .zip(backward.as_slice())
        .zip(&input_snap.offsets)
    {
        let e = fixture.inverse[offset];
        compare(R::to_f64(a.re), e.re, 3e-5, 3e-5, "direction inverse real").unwrap();
        compare(R::to_f64(a.im), e.im, 3e-5, 3e-5, "direction inverse imag").unwrap();
        let e = fixture.backward[offset];
        compare(R::to_f64(b.re), e.re, 3e-5, 3e-5, "direction backward real").unwrap();
        compare(R::to_f64(b.im), e.im, 3e-5, 3e-5, "direction backward imag").unwrap();
    }
}

fn run_direction_real<R: Real, const N: usize, const M: usize>(
    topology: Arc<MpiTopology<M>>,
    fixture: &DirectionFixture,
    method: TransposeMethod,
    permute_dims: bool,
) where
    Complex<R>: Equivalence,
{
    let shape: [usize; N] = fixture.shape.clone().try_into().unwrap();
    let directions = FourierDirections::new(fixture.directions.clone().try_into().unwrap());
    let transforms: [AxisTransform; N] = fixture
        .transforms
        .iter()
        .map(|kind| match kind {
            R2rAxisKind::Rfft => AxisTransform::Rfft,
            R2rAxisKind::None => AxisTransform::None,
            R2rAxisKind::Fft => AxisTransform::Fft,
            R2rAxisKind::DctII => AxisTransform::R2r(AxisR2rKind::Fftw(R2rKind::DctII)),
            R2rAxisKind::Dht => AxisTransform::R2r(AxisR2rKind::Dht),
            _ => panic!("unsupported direction transform"),
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let plan = MixedR2cPlan::<R, N, M>::from_shape_with_layout(
        Arc::clone(&topology),
        shape,
        ExtraShape::scalar(),
        transforms,
        DistributedLayout {
            transpose_method: method,
            permute_dims,
        },
    )
    .unwrap()
    .with_fft_directions(directions)
    .unwrap();
    let layout = Layout {
        input: shape,
        output: *plan.output_pencil().global_shape(),
        extra: &[],
        permute_dims,
    };
    let mut source = plan.allocate_input().unwrap();
    let input_snap = snapshot(
        plan.input_pencil(),
        &ExtraShape::scalar(),
        &layout,
        false,
        source.as_slice().len(),
        "direction input",
    );
    fill_real(source.as_mut_slice(), &input_snap, &fixture.input);
    let mut output = plan.allocate_output().unwrap();
    let output_snap = snapshot(
        plan.output_pencil(),
        &ExtraShape::scalar(),
        &layout,
        true,
        output.as_slice().len(),
        "direction output",
    );
    let mut workspace = plan.allocate_workspace().unwrap();
    plan.forward(&source, &mut output, &mut workspace).unwrap();
    for (v, &offset) in output.as_slice().iter().zip(&output_snap.offsets) {
        let e = fixture.forward[offset];
        compare(R::to_f64(v.re), e.re, 3e-5, 3e-5, "direction forward real").unwrap();
        compare(R::to_f64(v.im), e.im, 3e-5, 3e-5, "direction forward imag").unwrap();
    }
    let mut spectrum = plan.allocate_output().unwrap();
    fill_complex(
        spectrum.as_mut_slice(),
        &output_snap,
        &fixture.inverse_input,
    );
    let mut inverse = plan.allocate_input().unwrap();
    plan.inverse(&spectrum, &mut inverse, &mut workspace)
        .unwrap();
    let mut backward = plan.allocate_input().unwrap();
    plan.backward(&spectrum, &mut backward, &mut workspace)
        .unwrap();
    for ((a, b), &offset) in inverse
        .as_slice()
        .iter()
        .zip(backward.as_slice())
        .zip(&input_snap.offsets)
    {
        let e = fixture.inverse[offset];
        compare(R::to_f64(*a), e.re, 3e-5, 3e-5, "direction inverse real").unwrap();
        compare(0.0, e.im, 3e-5, 3e-5, "direction inverse imag").unwrap();
        let e = fixture.backward[offset];
        compare(R::to_f64(*b), e.re, 3e-5, 3e-5, "direction backward real").unwrap();
        compare(0.0, e.im, 3e-5, 3e-5, "direction backward imag").unwrap();
    }
}

#[test]
#[ignore = "opt-in local Julia/FFTW cross-validation; run tools/fftw-reference/check.sh"]
fn fftw_direction_reference_parser() {
    let directory = env::var_os("PENCIL_FFTW_DIRECTION_FIXTURES")
        .map(PathBuf::from)
        .expect("PENCIL_FFTW_DIRECTION_FIXTURES is required for the opted-in direction test");
    let fixtures = direction_fixtures(&directory);
    let universe = mpi::initialize().expect("MPI initialization failed");
    let world = universe.world();
    let size = usize::try_from(world.size()).unwrap();
    let topology = MpiTopology::<1>::new(&world, [size]).unwrap();
    for fixture in &fixtures {
        assert!(!fixture.case.is_empty());
        for method in [TransposeMethod::AllToAllv, TransposeMethod::PointToPoint] {
            for permute_dims in [false, true] {
                match fixture.shape.len() {
                    2 => {
                        if fixture.transforms.contains(&R2rAxisKind::Rfft) {
                            run_direction_real::<f32, 2, 1>(
                                Arc::clone(&topology),
                                fixture,
                                method,
                                permute_dims,
                            );
                            run_direction_real::<f64, 2, 1>(
                                Arc::clone(&topology),
                                fixture,
                                method,
                                permute_dims,
                            );
                        } else if fixture.transforms.iter().all(|k| *k == R2rAxisKind::Fft) {
                            run_direction_c2c::<f32, 2, 1>(
                                Arc::clone(&topology),
                                fixture,
                                method,
                                permute_dims,
                            );
                            run_direction_c2c::<f64, 2, 1>(
                                Arc::clone(&topology),
                                fixture,
                                method,
                                permute_dims,
                            );
                        } else {
                            run_direction_mixed::<f32, 2, 1>(
                                Arc::clone(&topology),
                                fixture,
                                method,
                                permute_dims,
                            );
                            run_direction_mixed::<f64, 2, 1>(
                                Arc::clone(&topology),
                                fixture,
                                method,
                                permute_dims,
                            );
                        }
                    }
                    3 => {
                        if fixture.transforms.contains(&R2rAxisKind::Rfft) {
                            run_direction_real::<f32, 3, 1>(
                                Arc::clone(&topology),
                                fixture,
                                method,
                                permute_dims,
                            );
                            run_direction_real::<f64, 3, 1>(
                                Arc::clone(&topology),
                                fixture,
                                method,
                                permute_dims,
                            );
                        } else if fixture.transforms.iter().all(|k| *k == R2rAxisKind::Fft) {
                            run_direction_c2c::<f32, 3, 1>(
                                Arc::clone(&topology),
                                fixture,
                                method,
                                permute_dims,
                            );
                            run_direction_c2c::<f64, 3, 1>(
                                Arc::clone(&topology),
                                fixture,
                                method,
                                permute_dims,
                            );
                        } else {
                            run_direction_mixed::<f32, 3, 1>(
                                Arc::clone(&topology),
                                fixture,
                                method,
                                permute_dims,
                            );
                            run_direction_mixed::<f64, 3, 1>(
                                Arc::clone(&topology),
                                fixture,
                                method,
                                permute_dims,
                            );
                        }
                    }
                    _ => panic!("unsupported direction rank"),
                }
            }
        }
    }
    if world.rank() == 0 {
        println!("PENCIL_FFTW_DIRECTION_REFERENCE_RAN fixtures=5 configurations=40");
    }
}

#[test]
#[ignore = "opt-in local Julia/FFTW cross-validation; run tools/fftw-reference/check.sh"]
fn fftw_reference_matrix() {
    let directory = env::var_os("PENCIL_FFTW_FIXTURES")
        .map(PathBuf::from)
        .expect("PENCIL_FFTW_FIXTURES is required for the opted-in reference test");
    let fixtures = fixtures(&directory);
    let universe = mpi::initialize().expect("MPI initialization failed");
    let world = universe.world();
    let size = usize::try_from(world.size()).unwrap();
    assert!(matches!(size, 1 | 4 | 6));
    let one = MpiTopology::<1>::new(&world, [size]).unwrap();
    let grid = match size {
        1 => [1, 1],
        4 => [2, 2],
        6 => [2, 3],
        _ => unreachable!(),
    };
    let two = MpiTopology::<2>::new(&world, grid).unwrap();
    assert_eq!(one.process_grid(), &[size]);
    assert_eq!(two.process_grid(), &grid);
    let layouts: usize = fixtures
        .iter()
        .map(|fixture| std::cmp::min(2, fixture.shape.len() - 1))
        .sum();
    let layout_count = |shape: &[usize]| std::cmp::min(2, shape.len() - 1);
    let expected_layouts = base_cases()
        .into_iter()
        .chain(partial_cases())
        .map(|(_, shape, _, _)| layout_count(&shape))
        .sum::<usize>()
        * 2
        + r2r_cases()
            .into_iter()
            .map(|(shape, _, _)| layout_count(&shape))
            .sum::<usize>()
            * 4
        + mixed_cases()
            .into_iter()
            .map(|(_, shape, _, _, _)| layout_count(&shape))
            .sum::<usize>()
            * 2
        + dht_cases()
            .into_iter()
            .map(|(shape, _, _)| layout_count(&shape))
            .sum::<usize>()
            * 4;
    assert_eq!(layouts, expected_layouts);
    let layout_policies = 2;
    let total_layouts = layouts * layout_policies;
    println!(
        "PENCIL_FFTW_REFERENCE_MATRIX_STARTED fixtures={} layouts={layouts} layout_policies={layout_policies} total_layouts={total_layouts}",
        fixtures.len()
    );
    for permute_dims in [true, false] {
        for fixture in &fixtures {
            run_fixture(fixture, &one, &two, permute_dims, world.rank());
        }
    }
    println!(
        "PENCIL_FFTW_REFERENCE_MATRIX_RAN fixtures={} layouts={layouts} layout_policies={layout_policies} total_layouts={total_layouts}",
        fixtures.len()
    );
}
