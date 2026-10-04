#!/usr/bin/env bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT"
PROJECT="$ROOT/tools/fftw-reference"
GENERATOR="$PROJECT/generate.jl"
JULIA=${JULIA:-julia}
MPIEXEC=${MPIEXEC:-mpiexec}

# Validate selectors before invoking Julia, Cargo, or the MPI launcher.
case "${PENCIL_FFT_BACKEND-rustfft}" in
    rustfft) features=distributed ;;
    fftw) features=distributed,fftw ;;
    *) printf 'invalid PENCIL_FFT_BACKEND: %s\n' "$PENCIL_FFT_BACKEND" >&2; exit 1 ;;
esac
case "${PENCIL_FFT_DIRECTION_ORDER-native-first}" in
    native-first|directions-first) ;;
    *) printf 'invalid PENCIL_FFT_DIRECTION_ORDER: %s\n' "$PENCIL_FFT_DIRECTION_ORDER" >&2; exit 1 ;;
esac

threads=${PENCIL_FFT_THREADS-1}
# Bound the length before arithmetic so even huge inputs cannot overflow Bash.
if [[ ! $threads =~ ^[1-9][0-9]*$ ]] || (( ${#threads} > 10 )) || (( threads > 2147483647 )); then
    printf 'invalid PENCIL_FFT_THREADS: %s\n' "$threads" >&2
    exit 1
fi
if [[ ${PENCIL_FFT_BACKEND-rustfft} == rustfft && $threads != 1 ]]; then
    printf 'PENCIL_FFT_THREADS > 1 requires PENCIL_FFT_BACKEND=fftw\n' >&2
    exit 1
fi

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
DIRECTION_FIXTURES="$WORK/direction-fixtures"
JULIA_PROJECT="$WORK/project"
mkdir -p "$FIXTURES" "$DIRECTION_FIXTURES" "$JULIA_PROJECT"
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
"$JULIA_BIN" --compiled-modules=no --startup-file=no --history-file=no --project="$JULIA_PROJECT" \
    -L "$PROJECT/verify_sources.jl" \
    -e 'verify_sources(; preflight=true); Pkg.instantiate(; allow_build=false, allow_autoprecomp=false); verify_sources()'
if ! cmp -s "$JULIA_PROJECT/Manifest.toml" "$PROJECT/Manifest.toml"; then
    printf 'Pkg.resolve changed the checked-in Julia manifest\n' >&2
    exit 1
fi
"$JULIA_BIN" --compiled-modules=no --startup-file=no --history-file=no --project="$JULIA_PROJECT" \
    -L "$PROJECT/verify_sources.jl" -e 'verify_sources(); using FFTW; String(FFTW.fftw_provider) == "fftw" || error("FFTW provider is not fftw"); Base.pkgversion(FFTW) == v"1.10.0" || error("FFTW.jl is not 1.10.0")'
"$JULIA_BIN" --compiled-modules=no --startup-file=no --history-file=no --project="$JULIA_PROJECT" \
    -L "$PROJECT/verify_sources.jl" -e 'verify_sources(); include(popfirst!(ARGS))' "$GENERATOR" "$FIXTURES"
"$JULIA_BIN" --compiled-modules=no --startup-file=no --history-file=no --project="$JULIA_PROJECT" \
    -L "$PROJECT/verify_sources.jl" -e 'verify_sources(); include(popfirst!(ARGS))' "$PROJECT/directions_reference.jl" "$DIRECTION_FIXTURES"
direction_files=("$DIRECTION_FIXTURES"/*.txt)
[[ ${#direction_files[@]} -eq 5 ]] || {
    printf 'expected exactly 5 direction reference files, found %s\n' "${#direction_files[@]}" >&2
    exit 1
}

shopt -s nullglob
fixture_entries=("$FIXTURES"/*)
fixture_files=("$FIXTURES"/*.txt)
expected_fixture_count=82
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
    grep -Fxq 'PENCIL_FFTW_REFERENCE 7' "$fixture" || {
        printf 'fixture is missing mandatory format-7 metadata: %s\n' "$fixture" >&2
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
    if grep -Fxq 'kind mixed_r2c' "$fixture"; then
        grep -Eq '^original_n [0-9]+$' "$fixture" || {
            printf 'mixed R2C fixture is missing original_n metadata: %s\n' "$fixture" >&2
            exit 1
        }
    fi
done

CARGO_TEST_ARGS=(
    --manifest-path "$ROOT/Cargo.toml"
    --package pencil-fft
    --features "$features"
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

run_direction_reference() {
    local ranks=$1 directory=$2 log_file=$3
    direction_status=0
    timeout --kill-after=5s 120s "$MPIEXEC" "${mpi_flags[@]}" -n "$ranks" env PENCIL_FFTW_DIRECTION_FIXTURES="$directory" \
        cargo test "${CARGO_TEST_ARGS[@]}" -- --ignored fftw_direction_reference --nocapture --test-threads=1 >"$log_file" 2>&1 || direction_status=$?
    if ((direction_status != 0)); then
        cat "$log_file" >&2
        return "$direction_status"
    fi
    [[ $(grep -Fc 'PENCIL_FFTW_DIRECTION_REFERENCE_RAN fixtures=5 configurations=40' "$log_file") -eq 1 ]]
}

# Accept only an executed Rust test rejecting the intended fixture, never a failed runner.
check_direction_rejection() {
    local status=$1 log=$2 case_name=$3 reason=$4
    [[ $status == 101 || $status == 1 ]] || return 1
    grep -Fq 'PENCIL_FFTW_DIRECTION_REFERENCE_STARTED' "$log" || return 1
    grep -Eq '^test result: FAILED\. 0 passed; 1 failed;' "$log" || return 1
    grep -Fxq '    fftw_direction_reference_parser' "$log" || return 1
    if [[ -n $case_name ]]; then
        [[ $(grep -F 'PENCIL_FFTW_DIRECTION_REFERENCE_STARTED case=' "$log" | tail -n1) == "PENCIL_FFTW_DIRECTION_REFERENCE_STARTED case=$case_name" ]] || return 1
        grep -E "${reason}: actual=.*expected=.*bound=" "$log" >/dev/null || return 1
    else
        grep -Fq "$reason" "$log" || return 1
    fi
}

require_direction_rejection() {
    if ! check_direction_rejection "$direction_status" "$1" "$2" "$3"; then
        cat "$1" >&2
        printf 'direction rejection lacks intended test/reason evidence (status=%s)\n' "$direction_status" >&2
        exit 1
    fi
}

# Small executable guard regression: even plausible logs cannot hide runner failures.
guard_log="$WORK/direction-guard.log"
printf '%s\n' 'PENCIL_FFTW_DIRECTION_REFERENCE_STARTED' \
    'PENCIL_FFTW_DIRECTION_REFERENCE_STARTED case=mixed_r2c_even' \
    'direction forward real: actual=1 expected=2 bound=0.001' \
    '    fftw_direction_reference_parser' \
    'test result: FAILED. 0 passed; 1 failed;' >"$guard_log"
for status in 101 1; do
    check_direction_rejection "$status" "$guard_log" mixed_r2c_even 'direction forward real'
done
for status in 0 124 137 2 127; do
    if check_direction_rejection "$status" "$guard_log" mixed_r2c_even 'direction forward real'; then
        printf 'guard accepted invalid status %s\n' "$status" >&2; exit 1
    fi
done
if check_direction_rejection 101 "$guard_log" c2c_2d_3x4_forward-backward 'direction forward real' \
    || check_direction_rejection 101 "$guard_log" mixed_r2c_even 'direction backward real'; then
    printf 'guard accepted wrong comparison context\n' >&2; exit 1
fi
grep -v 'PENCIL_FFTW_DIRECTION_REFERENCE_STARTED' "$guard_log" >"$guard_log.no-start"
if check_direction_rejection 101 "$guard_log.no-start" '' 'actual='; then
    printf 'guard accepted missing execution marker\n' >&2; exit 1
fi
printf '%s\n' 'error: could not compile pencil-fft' >"$guard_log"
if check_direction_rejection 101 "$guard_log" '' 'could not compile'; then
    printf 'guard accepted build failure\n' >&2; exit 1
fi
printf 'direction rejection guard checks passed\n'

run_reference() {
    local ranks=$1
    local fixture_directory=$2
    local log_file=$3
    reference_status=0
    timeout --kill-after=5s 120s "$MPIEXEC" "${mpi_flags[@]}" -n "$ranks" \
        env -u PENCIL_FFTW_DIRECTION_FIXTURES PENCIL_FFTW_FIXTURES="$fixture_directory" \
        cargo test "${CARGO_TEST_ARGS[@]}" -- \
            --ignored fftw_reference_matrix --nocapture --test-threads=1 >"$log_file" 2>&1 || reference_status=$?
    if ((reference_status != 0)); then
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
    run_direction_reference "$ranks" "$DIRECTION_FIXTURES" "$WORK/direction-positive-$ranks.log"
done
printf 'direction references accepted at MPI ranks 1, 4, and 6\n'
# Direction-matrix omissions and MixedR2C outputs must fail, not merely parse.
for corruption in missing duplicate r2c-forward r2c-inverse r2c-backward r2c-sign r2c-real-imaginary; do
    directory="$WORK/direction-$corruption"
    mkdir "$directory"
    cp -- "${direction_files[@]}" "$directory/"
    case "$corruption" in
        missing) rm "$directory/mixed_r2c_odd.txt" ;;
        duplicate)
            if cmp -s "$directory/mixed_r2c_even.txt" "$directory/mixed_r2c_odd.txt"; then
                printf 'duplicate corruption would not change fixture\n' >&2; exit 1
            fi
            cp "$directory/mixed_r2c_even.txt" "$directory/mixed_r2c_odd.txt" ;;
        *) "$JULIA_BIN" --startup-file=no --history-file=no --project="$JULIA_PROJECT" -e '
            path, kind = ARGS
            lines = readlines(path)
            original = copy(lines)
            if kind == "r2c-sign"
                i = only(findall(line -> startswith(line, "directions "), lines))
                count(==("backward"), split(lines[i])) == 1 || error("expected one backward sign")
                lines[i] = replace(lines[i], "backward" => "forward")
            elseif kind == "r2c-real-imaginary"
                i = only(findall(line -> startswith(line, "section input "), lines))
                values = split(lines[i+1]); parse(Float64, values[2]) == 0 || error("input already imaginary")
                values[2] = "1.0"
                lines[i+1] = join(values, " ")
            else
                section = replace(kind, "r2c-" => "") * "_expected"
                i = only(findall(line -> startswith(line, "section " * section * " "), lines))
                values = split(lines[i+1]); old = parse(Float64, values[1])
                values[1] = string(old + 100)
                parse(Float64, values[1]) != old || error("unchanged value")
                lines[i+1] = join(values, " ")
            end
            count(lines .!= original) == 1 || error("corruption must change exactly one line")
            write(path, join(lines, "\n") * "\n")
        ' "$directory/mixed_r2c_even.txt" "$corruption" ;;
    esac
    if run_direction_reference 1 "$directory" "$WORK/$corruption.log"; then
        printf 'checker accepted direction corruption: %s\n' "$corruption" >&2
        exit 1
    fi
    case "$corruption" in
        missing) require_direction_rejection "$WORK/$corruption.log" '' 'unexpected direction reference count' ;;
        duplicate)
            require_direction_rejection "$WORK/$corruption.log" '' 'left: Some("mixed_r2c_odd")'
            require_direction_rejection "$WORK/$corruption.log" '' 'right: Some("mixed_r2c_even")' ;;
        r2c-real-imaginary) require_direction_rejection "$WORK/$corruption.log" '' 'RFFT real sections must have zero imaginary parts' ;;
        r2c-sign) require_direction_rejection "$WORK/$corruption.log" mixed_r2c_even 'direction forward (real|imag)' ;;
        *) require_direction_rejection "$WORK/$corruption.log" mixed_r2c_even "direction ${corruption#r2c-} real" ;;
    esac
    printf 'direction corruption rejected: %s\n' "$corruption"
done
DIRECTION_CORRUPT="$WORK/direction-corrupt"
mkdir -p "$DIRECTION_CORRUPT"
cp -- "$DIRECTION_FIXTURES"/*.txt "$DIRECTION_CORRUPT/"
first_direction="$DIRECTION_CORRUPT/c2c_2d_3x4_forward-backward.txt"
"$JULIA_BIN" --startup-file=no --history-file=no --project="$JULIA_PROJECT" -e '
path = ARGS[1]; lines = readlines(path);
i = only(findall(line -> startswith(line, "section forward_expected "), lines))
old = lines[i + 1]
words = split(old); words[1] = string(parse(Float64, words[1]) + 1.0); lines[i + 1] = join(words, " ")
parse(Float64, split(old)[1]) != parse(Float64, words[1]) || error("unchanged value")
open(path, "w") do io; write(io, join(lines, "\n"), "\n"); end
' "$first_direction"
if run_direction_reference 1 "$DIRECTION_CORRUPT" "$WORK/direction-corrupt.log"; then
    printf 'checker accepted a corrupted direction reference\n' >&2
    exit 1
fi
require_direction_rejection "$WORK/direction-corrupt.log" c2c_2d_3x4_forward-backward 'direction forward real'
printf 'corrupted direction forward_expected rejected as intended\n'
DIRECTION_BACKWARD_CORRUPT="$WORK/direction-backward-corrupt"
mkdir -p "$DIRECTION_BACKWARD_CORRUPT"
cp -- "$DIRECTION_FIXTURES"/*.txt "$DIRECTION_BACKWARD_CORRUPT/"
first_direction_backward="$DIRECTION_BACKWARD_CORRUPT/c2c_2d_3x4_forward-backward.txt"
"$JULIA_BIN" --startup-file=no --history-file=no --project="$JULIA_PROJECT" -e '
path = ARGS[1]; lines = readlines(path);
i = only(findall(line -> startswith(line, "section backward_expected "), lines))
old = lines[i + 1]
words = split(old); words[1] = string(parse(Float64, words[1]) + 1.0); lines[i + 1] = join(words, " ")
parse(Float64, split(old)[1]) != parse(Float64, words[1]) || error("unchanged value")
open(path, "w") do io; write(io, join(lines, "\n"), "\n"); end
' "$first_direction_backward"
if run_direction_reference 1 "$DIRECTION_BACKWARD_CORRUPT" "$WORK/direction-backward-corrupt.log"; then
    printf 'checker accepted corrupted direction backward_expected\n' >&2
    exit 1
fi
require_direction_rejection "$WORK/direction-backward-corrupt.log" c2c_2d_3x4_forward-backward 'direction backward real'
printf 'corrupted direction backward_expected rejected as intended\n'
DIRECTION_SIGN_CORRUPT="$WORK/direction-sign-corrupt"
mkdir -p "$DIRECTION_SIGN_CORRUPT"
cp -- "$DIRECTION_FIXTURES"/*.txt "$DIRECTION_SIGN_CORRUPT/"
first_direction_sign="$DIRECTION_SIGN_CORRUPT/c2c_2d_3x4_forward-backward.txt"
"$JULIA_BIN" --startup-file=no --history-file=no --project="$JULIA_PROJECT" -e '
path = ARGS[1]; lines = readlines(path);
i = only(findall(line -> startswith(line, "directions "), lines))
words = split(lines[i]); words[2] == "forward" || error("unexpected original sign")
words[2] = "sideways"; lines[i] = join(words, " ")
open(path, "w") do io; write(io, join(lines, "\n"), "\n"); end
' "$first_direction_sign"
if run_direction_reference 1 "$DIRECTION_SIGN_CORRUPT" "$WORK/direction-sign-corrupt.log"; then
    printf 'checker accepted corrupted direction metadata\n' >&2
    exit 1
fi
require_direction_rejection "$WORK/direction-sign-corrupt.log" '' 'directions: invalid value "sideways"'
printf 'corrupted direction metadata rejected as intended\n'

for ranks in 1 4 6; do
    printf '\n== Julia/FFTW reference: %s MPI ranks ==\n' "$ranks"
    run_reference "$ranks" "$FIXTURES" "$WORK/reference-$ranks.log"
done

check_corruption_log() {
    local log_file=$1
    local context=${2:-}
    if [[ $reference_status != 101 && $reference_status != 1 ]] \
        || ! grep -Eq '^test result: FAILED\. 0 passed; 1 failed;' "$log_file" \
        || ! grep -Fxq '    fftw_reference_matrix' "$log_file" \
        || ! grep -Fq 'PENCIL_FFTW_REFERENCE_MATRIX_STARTED' "$log_file" \
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

# name:kind:section:optional section type:comparison context
corruption_specs=(
    'c2c-forward:c2c:forward_expected:complex:C2C forward'
    'c2c-backward:c2c:backward_expected:complex:C2C backward'
    'r2c-forward:r2c:forward_expected:complex:R2C forward'
    'r2c-backward:r2c:backward_expected:real:R2C backward'
    'r2r-forward:r2r:forward_expected:real:R2R forward'
    'r2r-backward:r2r:backward_expected:real:R2R backward'
    'dht-forward:dht:forward_expected:real:DHT forward'
    'dht-backward:dht:backward_expected:real:DHT backward'
    'mixed-c2c-forward:mixed_c2c:forward_expected::mixed C2C forward'
    'mixed-c2c-inverse:mixed_c2c:inverse_expected::mixed C2C inverse'
    'mixed-c2c-backward:mixed_c2c:backward_expected::mixed C2C backward'
    'mixed-c2c-forward-2:mixed_c2c:forward_expected::mixed C2C forward'
    'mixed-c2c-inverse-2:mixed_c2c:inverse_expected::mixed C2C inverse'
    'mixed-c2c-backward-2:mixed_c2c:backward_expected::mixed C2C backward'
    'mixed-r2c-forward:mixed_r2c:forward_expected::mixed R2C forward'
    'mixed-r2c-inverse:mixed_r2c:inverse_expected::mixed C2R inverse'
    'mixed-r2c-backward:mixed_r2c:backward_expected::mixed R2C backward'
    'mixed-r2c-forward-2:mixed_r2c:forward_expected::mixed R2C forward'
    'mixed-r2c-inverse-2:mixed_r2c:inverse_expected::mixed C2R inverse'
    'mixed-r2c-backward-2:mixed_r2c:backward_expected::mixed R2C backward'
)
for spec in "${corruption_specs[@]}"; do
    IFS=: read -r name kind section section_type context <<<"$spec"
    directory="$WORK/corrupt-$name-fixtures"
    mkdir -p "$directory"
    cp -- "${fixture_files[@]}" "$directory/"
    corrupt_files=("$directory"/*.txt)
    [[ ${#corrupt_files[@]} -eq "$expected_fixture_count" ]] || {
        printf 'corrupted fixture copy is incomplete: %s\n' "$name" >&2
        exit 1
    }
    candidate=
    candidate_skip=0
    [[ $name == *-2 ]] && candidate_skip=1
    for fixture in "$directory"/*.txt; do
        if grep -Fxq "kind $kind" "$fixture" \
            && grep -Fq "section $section ${section_type:+$section_type }" "$fixture"; then
            if ((candidate_skip > 0)); then
                ((candidate_skip--))
                continue
            fi
            candidate=$fixture
            break
        fi
    done
    [[ -n "$candidate" ]] || {
        printf 'missing corruption candidate for %s\n' "$name" >&2
        exit 1
    }
    "$JULIA_BIN" --startup-file=no --history-file=no --project="$JULIA_PROJECT" -e '
lines = readlines(ARGS[1])
marker = "section " * ARGS[2] * " "
for index in eachindex(lines)
    if startswith(lines[index], marker)
        index < length(lines) || error("section has no value")
        tokens = split(lines[index + 1])
        isempty(tokens) && error("section has an empty value")
        old = parse(Float64, tokens[1])
        tokens[1] = string(old + 1.0)
        parse(Float64, tokens[1]) != old || error("unchanged value")
        lines[index + 1] = join(tokens, " ")
        open(ARGS[1], "w") do io
            write(io, join(lines, "\n"), "\n")
        end
        exit()
    end
end
error("section not found")
' "$candidate" "$section"
    log_file="$WORK/corrupted-$name.log"
    printf '\n== corrupted %s rejection ==\n' "$context"
    if run_reference 1 "$directory" "$log_file"; then
        printf 'checker accepted a deliberately corrupted %s value\n' "$context" >&2
        exit 1
    fi
    check_corruption_log "$log_file" "$context"
    printf 'corrupted %s rejected as intended\n' "$context"
done
