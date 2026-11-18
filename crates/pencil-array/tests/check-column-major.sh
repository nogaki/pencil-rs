#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
log_dir=target/cm-array-test-logs
mkdir -p "$log_dir"
index=0
failures=0
initial_red=0
failure_labels=()

record() {
    printf '%s\n' "$*" >>"$status_file"
}

next_log() {
    index=$((index + 1))
    local safe_label=${1//[^a-zA-Z0-9_.-]/_}
    printf -v REPLY '%s/%03d-%s.log' "$run_log_dir" "$index" "$safe_label"
}

executed_tests() {
    awk '
        /test result:/ {
            passed = 0
            failed = 0
            for (i = 1; i <= NF; i++) {
                if ($i == "passed;") passed = $(i - 1)
                if ($i == "failed;") failed = $(i - 1)
            }
            total = passed + failed
        }
        END { print total + 0 }
    ' "$1"
}

failure_excerpt() {
    local logfile=$1
    if ! grep -a -n -A2 -B1 -E 'CM_LAYOUT_CONTRACT|error:|test result: FAILED|panicked at|MPI_ABORT|timed out' "$logfile" | head -n 14; then
        tail -n 8 "$logfile"
    fi
}

run_test() {
    local label=$1
    shift
    next_log "$label"
    local logfile=$REPLY
    record "COMMAND label=$label: $(printf '%q ' "$@")"
    "$@" >"$logfile" 2>&1
    local status=$?
    local count
    count=$(executed_tests "$logfile")
    local summary
    summary=$(grep -a 'test result:' "$logfile" | tail -n 1 || true)
    record "RESULT label=$label exit=$status executed_tests=$count summary=${summary:-missing} log=$logfile"
    if ((status != 0 || count == 0)); then
        failures=$((failures + 1))
        failure_labels+=("$label(exit=$status,tests=$count)")
        printf 'FAIL %s exit=%s executed_tests=%s log=%s\n' "$label" "$status" "$count" "$logfile"
        if ((failures <= 3)); then
            failure_excerpt "$logfile"
        fi
        return 1
    fi
    printf 'PASS %s executed_tests=%s\n' "$label" "$count"
    return 0
}

valid_test_count() {
    awk '
        /test result:/ {
            summaries++
            if ($0 !~ /^test result: ok\./) invalid = 1
            passed = 0
            failed = 0
            for (i = 1; i <= NF; i++) {
                if ($i == "passed;") passed = $(i - 1)
                if ($i == "failed;") failed = $(i - 1)
            }
            current = passed + failed
            if (current == 0) invalid = 1
            total += current
        }
        END {
            if (summaries == 0 || invalid || total == 0) exit 1
            print total
        }
    ' "$1"
}

rank_stdout_file() {
    find "$1" -maxdepth 1 -type f -name "rank.$2.stdout" -print -quit
}

rank_stdout_count() {
    find "$1" -maxdepth 1 -type f -name 'rank.*.stdout' -print | awk 'END { print NR + 0 }'
}

validate_rank_logs() {
    local launcher_status=$1
    local ranks=$2
    local ranked_dir=$3
    local file_count rank rank_log rank_count total=0
    local -a rank_summaries=()
    RANK_EXECUTED_TESTS=0
    RANK_SUMMARY=
    RANK_LOG_FAILURE=

    if ((launcher_status != 0)); then
        RANK_LOG_FAILURE="launcher exit=$launcher_status"
        return 1
    fi
    if [[ ! $ranks =~ ^[1-9][0-9]*$ ]]; then
        RANK_LOG_FAILURE="invalid requested rank count=$ranks"
        return 1
    fi
    file_count=$(rank_stdout_count "$ranked_dir")
    if ((file_count != ranks)); then
        RANK_LOG_FAILURE="stdout rank files=$file_count expected=$ranks"
        return 1
    fi
    for ((rank = 0; rank < ranks; rank++)); do
        rank_log=$(rank_stdout_file "$ranked_dir" "$rank")
        if [[ -z $rank_log ]]; then
            RANK_LOG_FAILURE="missing stdout for rank=$rank"
            return 1
        fi
        rank_count=$(find "$ranked_dir" -maxdepth 1 -type f -name "rank.$rank.stdout" -print | awk 'END { print NR + 0 }')
        if ((rank_count != 1)); then
            RANK_LOG_FAILURE="stdout files for rank=$rank count=$rank_count"
            return 1
        fi
        if ! rank_count=$(valid_test_count "$rank_log"); then
            RANK_LOG_FAILURE="rank=$rank has no successful nonzero test summary log=$rank_log"
            return 1
        fi
        total=$((total + rank_count))
        rank_summaries+=("$rank:$rank_count")
    done
    RANK_EXECUTED_TESTS=$total
    RANK_SUMMARY=$(IFS=,; printf '%s' "${rank_summaries[*]}")
}

check_rank_log_validator() {
    local fixture_dir failed=0
    fixture_dir=$(mktemp -d "$log_dir/rank-validator-check.XXXXXX") || return 1
    printf 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s\n' >"$fixture_dir/rank.0.stdout"
    printf 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s\n' >"$fixture_dir/rank.1.stdout"

    if validate_rank_logs 0 2 "$fixture_dir"; then
        printf 'PASS rank-log validator accepts successful nonzero rank summaries\n'
    else
        printf 'FAIL rank-log validator rejected valid rank summaries: %s\n' "$RANK_LOG_FAILURE"
        failed=1
    fi

    printf 'test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s\n' >"$fixture_dir/rank.1.stdout"
    if validate_rank_logs 0 2 "$fixture_dir"; then
        printf 'FAIL rank-log validator accepted zero-test rank\n'
        failed=1
    else
        printf 'PASS rank-log validator rejects zero-test rank\n'
    fi

    rm "$fixture_dir/rank.1.stdout"
    if validate_rank_logs 0 2 "$fixture_dir"; then
        printf 'FAIL rank-log validator accepted missing rank\n'
        failed=1
    else
        printf 'PASS rank-log validator rejects missing rank\n'
    fi

    printf 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s\n' >"$fixture_dir/rank.1.stdout"
    if validate_rank_logs 1 2 "$fixture_dir"; then
        printf 'FAIL rank-log validator accepted failed launcher\n'
        failed=1
    else
        printf 'PASS rank-log validator rejects failed launcher\n'
    fi
    rm -rf "$fixture_dir"
    return "$failed"
}

run_mpi_test() {
    local label=$1
    local ranks=$2
    shift 2
    next_log "$label"
    local logfile=$REPLY
    local safe_label=${label//[^a-zA-Z0-9_.-]/_}
    local ranked_dir
    ranked_dir=$(mktemp -d "$run_log_dir/${index}-${safe_label}.ranks.XXXXXX")
    local output_prefix="$ranked_dir/openmpi"
    local -a command=(timeout --kill-after=5s 240s mpiexec --oversubscribe \
        --output-filename "$output_prefix" -n "$ranks" bash -c \
        'ranked_dir=$1; shift; exec >"$ranked_dir/rank.${OMPI_COMM_WORLD_RANK}.stdout" 2>&1; exec "$@"' \
        bash "$ranked_dir" "$@")
    record "COMMAND label=$label: $(printf '%q ' "${command[@]}")"
    "${command[@]}" >"$logfile" 2>&1
    local status=$?
    local count=0 summary=missing rank_result=invalid
    if validate_rank_logs "$status" "$ranks" "$ranked_dir"; then
        count=$RANK_EXECUTED_TESTS
        summary=$RANK_SUMMARY
        rank_result=valid
    else
        rank_result=$RANK_LOG_FAILURE
    fi
    record "RESULT label=$label exit=$status executed_tests=$count rank_check=$rank_result rank_summaries=${summary:-missing} log=$logfile rank_logs=$ranked_dir"
    if ((status != 0)) || [[ $rank_result != valid ]] || ((count == 0)); then
        failures=$((failures + 1))
        failure_labels+=("$label(exit=$status,tests=$count,rank_check=$rank_result)")
        printf 'FAIL %s exit=%s executed_tests=%s rank_check=%s log=%s rank_logs=%s\n' \
            "$label" "$status" "$count" "$rank_result" "$logfile" "$ranked_dir"
        if ((failures <= 3)); then
            failure_excerpt "$logfile"
        fi
        return 1
    fi
    printf 'PASS %s executed_tests=%s rank_summaries=%s\n' "$label" "$count" "$summary"
    return 0
}

run_simple() {
    local label=$1
    shift
    next_log "$label"
    local logfile=$REPLY
    record "COMMAND label=$label: $(printf '%q ' "$@")"
    "$@" >"$logfile" 2>&1
    local status=$?
    record "RESULT label=$label exit=$status log=$logfile"
    if ((status != 0)); then
        failures=$((failures + 1))
        failure_labels+=("$label(exit=$status)")
        printf 'FAIL %s exit=%s log=%s\n' "$label" "$status" "$logfile"
        if ((failures <= 3)); then
            failure_excerpt "$logfile"
        fi
        return 1
    fi
    printf 'PASS %s\n' "$label"
    return 0
}

if [[ ${1:-} == --check-rank-log-validator ]]; then
    if (($# != 1)); then
        printf 'usage: %s [--check-rank-log-validator]\n' "$0" >&2
        exit 2
    fi
    check_rank_log_validator
    exit $?
fi
if (($# != 0)); then
    printf 'usage: %s [--check-rank-log-validator]\n' "$0" >&2
    exit 2
fi
run_log_dir=$(mktemp -d "$log_dir/run.XXXXXX") || exit 1
status_file="$run_log_dir/status.txt"
printf 'START role=test-writer scope=crates/pencil-array goal=column-major-contract\n' >"$status_file"

first_label=column_major_layout_contract_n1
next_log "$first_label"
first_log=$REPLY
first_safe_label=${first_label//[^a-zA-Z0-9_.-]/_}
first_ranked_dir=$(mktemp -d "$run_log_dir/${index}-${first_safe_label}.ranks.XXXXXX")
first_output_prefix="$first_ranked_dir/openmpi"
first_command=(timeout --kill-after=5s 240s mpiexec --oversubscribe \
    --output-filename "$first_output_prefix" -n 1 bash -c \
    'ranked_dir=$1; shift; exec >"$ranked_dir/rank.${OMPI_COMM_WORLD_RANK}.stdout" 2>&1; exec "$@"' \
    bash "$first_ranked_dir" cargo test -p pencil-array --test column_major_layout \
    --locked --offline column_major_layout_contract -- --exact --nocapture --test-threads=1)
record "COMMAND label=$first_label: $(printf '%q ' "${first_command[@]}")"
"${first_command[@]}" >"$first_log" 2>&1
first_status=$?
first_rank_log=$(rank_stdout_file "$first_ranked_dir" 0)
first_rank_files=$(rank_stdout_count "$first_ranked_dir")
first_count=0
first_summary=missing
if [[ -n $first_rank_log ]]; then
    first_count=$(executed_tests "$first_rank_log")
    first_summary=$(grep -a 'test result:' "$first_rank_log" | tail -n 1 || true)
fi
record "RESULT label=$first_label exit=$first_status executed_tests=$first_count rank_files=$first_rank_files summary=${first_summary:-missing} log=$first_log rank_log=${first_rank_log:-missing}"
if ((first_status == 0)) && validate_rank_logs "$first_status" 1 "$first_ranked_dir"; then
    first_count=$RANK_EXECUTED_TESTS
    first_summary=$RANK_SUMMARY
    printf 'PASS %s executed_tests=%s rank_summaries=%s\n' "$first_label" "$first_count" "$first_summary"
elif ((first_status != 0 && first_rank_files == 1 && first_count > 0)) && \
        grep -a -q 'CM_LAYOUT_CONTRACT' "$first_rank_log" && grep -a -q 'test result: FAILED' "$first_rank_log"; then
    initial_red=1
    failures=$((failures + 1))
    failure_labels+=("$first_label(CM_LAYOUT_CONTRACT,exit=$first_status,tests=$first_count)")
    printf 'EXPECTED RED %s exit=%s executed_tests=%s summary=%s rank_log=%s\n' \
        "$first_label" "$first_status" "$first_count" "$first_summary" "$first_rank_log"
    failure_excerpt "$first_rank_log"
else
    failures=$((failures + 1))
    failure_labels+=("$first_label(invalid-result,exit=$first_status,tests=$first_count)")
    printf 'INVALID RESULT %s exit=%s executed_tests=%s summary=%s log=%s rank_log=%s\n' \
        "$first_label" "$first_status" "$first_count" "${first_summary:-missing}" "$first_log" "${first_rank_log:-missing}"
    failure_excerpt "$first_log"
fi

for ranks in 4 6; do
    run_mpi_test "column_major_layout_contract_n${ranks}" "$ranks" \
        cargo test -p pencil-array --test column_major_layout --locked --offline \
        column_major_layout_contract -- --exact --nocapture --test-threads=1 || true
done

run_test array_lib_nonignored cargo test -p pencil-array --lib --locked --offline -- --test-threads=1 || true
run_test array_test_boundaries cargo test -p pencil-array --test column_major_test_boundaries \
    --locked --offline -- --nocapture --test-threads=1 || true

for suite in topology pencil array many array_refactor_contracts; do
    for ranks in 1 4; do
        run_mpi_test "${suite}_n${ranks}" "$ranks" \
            cargo test -p pencil-array --test "$suite" --locked --offline -- \
            --nocapture --test-threads=1 || true
    done
done

for suite in array_access collectives alltoallv_transpose spatial_remap; do
    for ranks in 1 4 6; do
        run_mpi_test "${suite}_n${ranks}" "$ranks" \
            cargo test -p pencil-array --test "$suite" --locked --offline -- \
            --nocapture --test-threads=1 || true
    done
done

for ranks in 1 4; do
    run_mpi_test "local_transpose_n${ranks}" "$ranks" \
        cargo test -p pencil-array --test local_transpose --locked --offline -- \
        --nocapture --test-threads=1 || true
done
# local_transpose only accepts 1 or 4 ranks; its existing CI matrix omits 6.

for suite in array_ops timing_overlap multi_input external_views mapped_reduction_alloc; do
    for ranks in 1 4 6; do
        run_mpi_test "${suite}_n${ranks}" "$ranks" \
            cargo test -p pencil-array --test "$suite" --locked --offline -- \
            --nocapture --test-threads=1 || true
    done
done

for ranks in 1 4 6; do
    run_mpi_test "spatial_remap_preflight_ignored_n${ranks}" "$ranks" \
        cargo test -p pencil-array --lib --locked --offline \
        spatial_remap::tests::mpi_preflight_failure_preserves_all_plan_state -- \
        --exact --ignored --nocapture --test-threads=1 || true
done

run_simple bash_syntax bash -n crates/pencil-array/tests/check-column-major.sh || true
run_simple workspace_format cargo fmt --all -- --check || true

record "SUMMARY matrix_jobs=$index initial_cm_layout_red=$initial_red matrix_failures=$failures"
if ((${#failure_labels[@]})); then
    printf 'FAILED JOBS (%s): %s\n' "${#failure_labels[@]}" "${failure_labels[*]}"
fi
if ((failures != 0)); then
    if ((initial_red != 0)); then
        printf 'RED confirmed at CM_LAYOUT_CONTRACT; %s matrix/validation jobs failed.\n' "$failures"
    else
        printf 'FAIL: %s matrix/validation jobs failed.\n' "$failures"
    fi
    exit 1
fi
printf 'All array column-major contract and matrix checks passed.\n'
exit 0
