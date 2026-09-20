#!/usr/bin/env bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT"
PROJECT="$ROOT/tools/fftw-reference"
GENERATOR="$PROJECT/generate.jl"
JULIA=${JULIA:-julia}
MPIEXEC=${MPIEXEC:-mpiexec}

JULIA_BIN=$(command -v "$JULIA" 2>/dev/null) || {
    printf 'missing Julia executable: %s\n' "$JULIA" >&2
    exit 1
}
command -v "$MPIEXEC" >/dev/null 2>&1 || {
    printf 'missing MPI launcher: %s\n' "$MPIEXEC" >&2
    exit 1
}
command -v timeout >/dev/null 2>&1 || {
    printf 'missing GNU timeout command\n' >&2
    exit 1
}
[[ -x "$JULIA_BIN" ]] || {
    printf 'Julia is not executable: %s\n' "$JULIA_BIN" >&2
    exit 1
}

WORK=$(mktemp -d "${TMPDIR:-/tmp}/pencil-fftw-reference.XXXXXX")
FIXTURES="$WORK/fixtures"
JULIA_PROJECT="$WORK/project"
mkdir -p "$FIXTURES" "$JULIA_PROJECT"
if [[ -z "${JULIA_DEPOT_PATH:-}" ]]; then
    export JULIA_DEPOT_PATH="$WORK/depot"
fi

cleanup() {
    rm -rf "$WORK"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

export JULIA_NUM_THREADS=${JULIA_NUM_THREADS:-1}
export JULIA_NUM_PRECOMPILE_TASKS=${JULIA_NUM_PRECOMPILE_TASKS:-1}
export JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_LOAD_PATH="@:@stdlib"

cp -- "$PROJECT/Project.toml" "$PROJECT/Manifest.toml" "$JULIA_PROJECT/"

"$JULIA_BIN" --startup-file=no --history-file=no \
    -e 'VERSION == v"1.12.6" || error("Julia 1.12.6 is required, got ", VERSION)'
"$JULIA_BIN" --startup-file=no --history-file=no --project="$JULIA_PROJECT" \
    -e 'using Pkg; Pkg.instantiate()'
if ! cmp -s "$JULIA_PROJECT/Manifest.toml" "$PROJECT/Manifest.toml"; then
    printf 'Pkg.resolve changed the checked-in Julia manifest\n' >&2
    exit 1
fi
"$JULIA_BIN" --startup-file=no --history-file=no --project="$JULIA_PROJECT" \
    -e 'using FFTW; String(FFTW.fftw_provider) == "fftw" || error("FFTW provider is not fftw"); Base.pkgversion(FFTW) == v"1.10.0" || error("FFTW.jl is not 1.10.0")'
"$JULIA_BIN" --startup-file=no --history-file=no --project="$JULIA_PROJECT" \
    "$GENERATOR" "$FIXTURES"

shopt -s nullglob
fixture_entries=("$FIXTURES"/*)
fixture_files=("$FIXTURES"/*.txt)
expected_fixture_count=68
[[ ${#fixture_entries[@]} -eq "$expected_fixture_count" && ${#fixture_files[@]} -eq "$expected_fixture_count" ]] || {
    printf 'expected exactly %s fixture files, found %s entries and %s txt files\n' \
        "$expected_fixture_count" "${#fixture_entries[@]}" "${#fixture_files[@]}" >&2
    exit 1
}
for fixture in "${fixture_files[@]}"; do
    [[ -s "$fixture" ]] || {
        printf 'empty fixture: %s\n' "$fixture" >&2
        exit 1
    }
    grep -Fxq 'PENCIL_FFTW_REFERENCE 6' "$fixture" || {
        printf 'fixture is missing mandatory format-6 metadata: %s\n' "$fixture" >&2
        exit 1
    }
    grep -Eq '^element_kind (real|complex)$' "$fixture" || {
        printf 'fixture is missing mandatory element_kind metadata: %s\n' "$fixture" >&2
        exit 1
    }
    grep -Eq '^axis_kinds( |$)' "$fixture" || {
        printf 'fixture is missing mandatory axis_kinds metadata: %s\n' "$fixture" >&2
        exit 1
    }
    grep -Eq '^selected_axes( |$)' "$fixture" || {
        printf 'fixture is missing mandatory selected_axes metadata: %s\n' "$fixture" >&2
        exit 1
    }
done

CARGO_TEST_ARGS=(
    --manifest-path "$ROOT/Cargo.toml"
    --package pencil-fft
    --features distributed
    --test fftw_reference
    --locked
)
cargo test "${CARGO_TEST_ARGS[@]}" --no-run
cargo test "${CARGO_TEST_ARGS[@]}" -- --test-threads=1

mpi_flags=()
if [[ "${PENCIL_FFTW_NO_OVERSUBSCRIBE:-0}" != 1 ]]; then
    mpi_version=$("$MPIEXEC" --version 2>&1 || true)
    if grep -Eq 'Open MPI|OpenRTE' <<<"$mpi_version"; then
        mpi_flags+=(--oversubscribe)
    fi
fi

run_reference() {
    local ranks=$1
    local fixture_directory=$2
    local log_file=$3
    if ! timeout --kill-after=5s 120s "$MPIEXEC" "${mpi_flags[@]}" -n "$ranks" \
        env PENCIL_FFTW_FIXTURES="$fixture_directory" \
        cargo test "${CARGO_TEST_ARGS[@]}" -- \
            --ignored --nocapture --test-threads=1 >"$log_file" 2>&1; then
        cat "$log_file" >&2
        return 1
    fi
    cat "$log_file"
    if ! grep -Fq 'PENCIL_FFTW_REFERENCE_MATRIX_RAN' "$log_file"; then
        printf 'MPI run did not execute the explicit ignored reference test\n' >&2
        return 1
    fi
}

for ranks in 1 4 6; do
    printf '\n== Julia/FFTW reference: %s MPI ranks ==\n' "$ranks"
    run_reference "$ranks" "$FIXTURES" "$WORK/reference-$ranks.log"
done

CORRUPT_C2C_FORWARD="$WORK/corrupt-c2c-forward-fixtures"
CORRUPT_C2C_BACKWARD="$WORK/corrupt-c2c-backward-fixtures"
CORRUPT_R2C_FORWARD="$WORK/corrupt-r2c-forward-fixtures"
CORRUPT_R2C_BACKWARD="$WORK/corrupt-r2c-backward-fixtures"
CORRUPT_R2R_FORWARD="$WORK/corrupt-r2r-forward-fixtures"
CORRUPT_R2R_BACKWARD="$WORK/corrupt-r2r-backward-fixtures"
CORRUPT_DHT_FORWARD="$WORK/corrupt-dht-forward-fixtures"
CORRUPT_DHT_BACKWARD="$WORK/corrupt-dht-backward-fixtures"
corrupt_directories=(
    "$CORRUPT_C2C_FORWARD"
    "$CORRUPT_C2C_BACKWARD"
    "$CORRUPT_R2C_FORWARD"
    "$CORRUPT_R2C_BACKWARD"
    "$CORRUPT_R2R_FORWARD"
    "$CORRUPT_R2R_BACKWARD"
    "$CORRUPT_DHT_FORWARD"
    "$CORRUPT_DHT_BACKWARD"
)
mkdir -p "${corrupt_directories[@]}"
for corrupt_directory in "${corrupt_directories[@]}"; do
    cp -- "${fixture_files[@]}" "$corrupt_directory/"
    corrupt_files=("$corrupt_directory"/*.txt)
    [[ ${#corrupt_files[@]} -eq "$expected_fixture_count" ]] || {
        printf 'corrupted fixture copy is incomplete\n' >&2
        exit 1
    }
done
corrupt_c2c_forward_file=
for candidate in "$CORRUPT_C2C_FORWARD"/*.txt; do
    if grep -Fxq 'kind c2c' "$candidate" \
        && grep -Fq 'section forward_expected complex ' "$candidate"; then
        corrupt_c2c_forward_file=$candidate
        break
    fi
done
corrupt_c2c_backward_file=
for candidate in "$CORRUPT_C2C_BACKWARD"/*.txt; do
    if grep -Fxq 'kind c2c' "$candidate" \
        && grep -Fq 'section backward_expected complex ' "$candidate"; then
        corrupt_c2c_backward_file=$candidate
        break
    fi
done
corrupt_r2c_forward_file=
for candidate in "$CORRUPT_R2C_FORWARD"/*.txt; do
    if grep -Fxq 'kind r2c' "$candidate" \
        && grep -Fq 'section forward_expected complex ' "$candidate"; then
        corrupt_r2c_forward_file=$candidate
        break
    fi
done
corrupt_r2c_backward_file=
for candidate in "$CORRUPT_R2C_BACKWARD"/*.txt; do
    if grep -Fxq 'kind r2c' "$candidate" \
        && grep -Fq 'section backward_expected real ' "$candidate"; then
        corrupt_r2c_backward_file=$candidate
        break
    fi
done
corrupt_r2r_forward_file=
for candidate in "$CORRUPT_R2R_FORWARD"/*.txt; do
    if grep -Fxq 'kind r2r' "$candidate" \
        && grep -Fq 'section forward_expected real ' "$candidate"; then
        corrupt_r2r_forward_file=$candidate
        break
    fi
done
corrupt_r2r_backward_file=
for candidate in "$CORRUPT_R2R_BACKWARD"/*.txt; do
    if grep -Fxq 'kind r2r' "$candidate" \
        && grep -Fq 'section backward_expected real ' "$candidate"; then
        corrupt_r2r_backward_file=$candidate
        break
    fi
done
corrupt_dht_forward_file=
for candidate in "$CORRUPT_DHT_FORWARD"/*.txt; do
    if grep -Fxq 'kind dht' "$candidate" \
        && grep -Fq 'section forward_expected real ' "$candidate"; then
        corrupt_dht_forward_file=$candidate
        break
    fi
done
corrupt_dht_backward_file=
for candidate in "$CORRUPT_DHT_BACKWARD"/*.txt; do
    if grep -Fxq 'kind dht' "$candidate" \
        && grep -Fq 'section backward_expected real ' "$candidate"; then
        corrupt_dht_backward_file=$candidate
        break
    fi
done
[[ -n "$corrupt_c2c_forward_file" \
    && -n "$corrupt_c2c_backward_file" \
    && -n "$corrupt_r2c_forward_file" \
    && -n "$corrupt_r2c_backward_file" \
    && -n "$corrupt_r2r_forward_file" \
    && -n "$corrupt_r2r_backward_file" \
    && -n "$corrupt_dht_forward_file" \
    && -n "$corrupt_dht_backward_file" ]] || {
    printf 'required C2C, R2C, R2R, and DHT corruption fixtures were not generated\n' >&2
    exit 1
}
"$JULIA_BIN" --startup-file=no --history-file=no --project="$JULIA_PROJECT" -e '
function corrupt(path, section_name)
    lines = readlines(path)
    changed = false
    marker = "section " * section_name * " "
    for index in eachindex(lines)
        if startswith(lines[index], marker)
            index < length(lines) || error(section_name, " section has no value")
            tokens = split(lines[index + 1])
            isempty(tokens) && error(section_name, " section has an empty value")
            tokens[1] = string(parse(Float64, tokens[1]) + 1.0)
            lines[index + 1] = join(tokens, " ")
            changed = true
            break
        end
    end
    changed || error(section_name, " section not found")
    open(path, "w") do io
        write(io, join(lines, "\n"), "\n")
    end
end
corrupt(ARGS[1], ARGS[2])
corrupt(ARGS[3], ARGS[4])
corrupt(ARGS[5], ARGS[6])
corrupt(ARGS[7], ARGS[8])
corrupt(ARGS[9], ARGS[10])
corrupt(ARGS[11], ARGS[12])
corrupt(ARGS[13], ARGS[14])
corrupt(ARGS[15], ARGS[16])
' \
    "$corrupt_c2c_forward_file" forward_expected \
    "$corrupt_c2c_backward_file" backward_expected \
    "$corrupt_r2c_forward_file" forward_expected \
    "$corrupt_r2c_backward_file" backward_expected \
    "$corrupt_r2r_forward_file" forward_expected \
    "$corrupt_r2r_backward_file" backward_expected \
    "$corrupt_dht_forward_file" forward_expected \
    "$corrupt_dht_backward_file" backward_expected

check_corruption_log() {
    local log_file=$1
    local context=${2:-}
    if ! grep -Fq 'PENCIL_FFTW_REFERENCE_MATRIX_STARTED' "$log_file" \
        || ! grep -Fq 'actual=' "$log_file" \
        || ! grep -Fq 'expected=' "$log_file" \
        || ! grep -Fq 'bound=' "$log_file"; then
        cat "$log_file" >&2
        printf 'corruption check failed without the expected comparison marker\n' >&2
        return 1
    fi
    if [[ -n "$context" ]] && ! grep -Fq "$context" "$log_file"; then
        cat "$log_file" >&2
        printf 'corruption check failed without %s comparison context\n' "$context" >&2
        return 1
    fi
}

printf '\n== corrupted C2C forward_expected rejection ==\n'
C2C_FORWARD_CORRUPT_LOG="$WORK/corrupted-c2c-forward.log"
if run_reference 1 "$CORRUPT_C2C_FORWARD" "$C2C_FORWARD_CORRUPT_LOG"; then
    printf 'checker accepted a deliberately corrupted C2C forward_expected value\n' >&2
    exit 1
fi
check_corruption_log "$C2C_FORWARD_CORRUPT_LOG" 'C2C forward'
printf 'corrupted C2C forward_expected rejected as intended\n'

printf '\n== corrupted C2C backward_expected rejection ==\n'
C2C_BACKWARD_CORRUPT_LOG="$WORK/corrupted-c2c-backward.log"
if run_reference 1 "$CORRUPT_C2C_BACKWARD" "$C2C_BACKWARD_CORRUPT_LOG"; then
    printf 'checker accepted a deliberately corrupted C2C backward_expected value\n' >&2
    exit 1
fi
check_corruption_log "$C2C_BACKWARD_CORRUPT_LOG" 'C2C backward'
printf 'corrupted C2C backward_expected rejected as intended\n'

printf '\n== corrupted R2C forward_expected rejection ==\n'
R2C_FORWARD_CORRUPT_LOG="$WORK/corrupted-r2c-forward.log"
if run_reference 1 "$CORRUPT_R2C_FORWARD" "$R2C_FORWARD_CORRUPT_LOG"; then
    printf 'checker accepted a deliberately corrupted R2C forward_expected value\n' >&2
    exit 1
fi
check_corruption_log "$R2C_FORWARD_CORRUPT_LOG" 'R2C forward'
printf 'corrupted R2C forward_expected rejected as intended\n'

printf '\n== corrupted R2C backward_expected rejection ==\n'
R2C_BACKWARD_CORRUPT_LOG="$WORK/corrupted-r2c-backward.log"
if run_reference 1 "$CORRUPT_R2C_BACKWARD" "$R2C_BACKWARD_CORRUPT_LOG"; then
    printf 'checker accepted a deliberately corrupted R2C backward_expected value\n' >&2
    exit 1
fi
check_corruption_log "$R2C_BACKWARD_CORRUPT_LOG" 'R2C backward'
printf 'corrupted R2C backward_expected rejected as intended\n'

printf '\n== corrupted R2R forward_expected rejection ==\n'
R2R_FORWARD_CORRUPT_LOG="$WORK/corrupted-r2r-forward.log"
if run_reference 1 "$CORRUPT_R2R_FORWARD" "$R2R_FORWARD_CORRUPT_LOG"; then
    printf 'checker accepted a deliberately corrupted R2R forward_expected value\n' >&2
    exit 1
fi
check_corruption_log "$R2R_FORWARD_CORRUPT_LOG" 'R2R forward'
printf 'corrupted R2R forward_expected rejected as intended\n'

printf '\n== corrupted R2R backward_expected rejection ==\n'
R2R_BACKWARD_CORRUPT_LOG="$WORK/corrupted-r2r-backward.log"
if run_reference 1 "$CORRUPT_R2R_BACKWARD" "$R2R_BACKWARD_CORRUPT_LOG"; then
    printf 'checker accepted a deliberately corrupted R2R backward_expected value\n' >&2
    exit 1
fi
check_corruption_log "$R2R_BACKWARD_CORRUPT_LOG" 'R2R backward'
printf 'corrupted R2R backward_expected rejected as intended\n'

printf '\n== corrupted DHT forward_expected rejection ==\n'
DHT_FORWARD_CORRUPT_LOG="$WORK/corrupted-dht-forward.log"
if run_reference 1 "$CORRUPT_DHT_FORWARD" "$DHT_FORWARD_CORRUPT_LOG"; then
    printf 'checker accepted a deliberately corrupted DHT forward_expected value\n' >&2
    exit 1
fi
check_corruption_log "$DHT_FORWARD_CORRUPT_LOG" 'DHT forward'
printf 'corrupted DHT forward_expected rejected as intended\n'

printf '\n== corrupted DHT backward_expected rejection ==\n'
DHT_BACKWARD_CORRUPT_LOG="$WORK/corrupted-dht-backward.log"
if run_reference 1 "$CORRUPT_DHT_BACKWARD" "$DHT_BACKWARD_CORRUPT_LOG"; then
    printf 'checker accepted a deliberately corrupted DHT backward_expected value\n' >&2
    exit 1
fi
check_corruption_log "$DHT_BACKWARD_CORRUPT_LOG" 'DHT backward'
printf 'corrupted DHT backward_expected rejected as intended\n'
