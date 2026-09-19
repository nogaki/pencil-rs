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
expected_fixture_count=28
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
    grep -Fxq 'PENCIL_FFTW_REFERENCE 4' "$fixture" || {
        printf 'fixture is missing mandatory format-4 metadata: %s\n' "$fixture" >&2
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

CORRUPT_FORWARD="$WORK/corrupt-forward-fixtures"
CORRUPT_BACKWARD="$WORK/corrupt-backward-fixtures"
mkdir -p "$CORRUPT_FORWARD" "$CORRUPT_BACKWARD"
cp -- "${fixture_files[@]}" "$CORRUPT_FORWARD/"
cp -- "${fixture_files[@]}" "$CORRUPT_BACKWARD/"
for corrupt_directory in "$CORRUPT_FORWARD" "$CORRUPT_BACKWARD"; do
    corrupt_files=("$corrupt_directory"/*.txt)
    [[ ${#corrupt_files[@]} -eq "$expected_fixture_count" ]] || {
        printf 'corrupted fixture copy is incomplete\n' >&2
        exit 1
    }
done
corrupt_backward_file=
for candidate in "$CORRUPT_BACKWARD"/*.txt; do
    if grep -Fxq 'kind r2c' "$candidate" \
        && grep -Fq 'section backward_expected real ' "$candidate"; then
        corrupt_backward_file=$candidate
        break
    fi
done
corrupt_forward_file="$CORRUPT_FORWARD/$(basename "$corrupt_backward_file")"
[[ -n "$corrupt_backward_file" && -f "$corrupt_forward_file" ]] || {
    printf 'no R2C fixture with real backward_expected was generated\n' >&2
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
' "$corrupt_forward_file" forward_expected "$corrupt_backward_file" backward_expected

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

printf '\n== corrupted forward_expected rejection ==\n'
FORWARD_CORRUPT_LOG="$WORK/corrupted-forward.log"
if run_reference 1 "$CORRUPT_FORWARD" "$FORWARD_CORRUPT_LOG"; then
    printf 'checker accepted a deliberately corrupted forward_expected value\n' >&2
    exit 1
fi
check_corruption_log "$FORWARD_CORRUPT_LOG"
printf 'corrupted forward_expected rejected as intended\n'

printf '\n== corrupted backward_expected rejection ==\n'
BACKWARD_CORRUPT_LOG="$WORK/corrupted-backward.log"
if run_reference 1 "$CORRUPT_BACKWARD" "$BACKWARD_CORRUPT_LOG"; then
    printf 'checker accepted a deliberately corrupted backward_expected value\n' >&2
    exit 1
fi
check_corruption_log "$BACKWARD_CORRUPT_LOG" 'R2C backward'
printf 'corrupted backward_expected rejected as intended\n'
