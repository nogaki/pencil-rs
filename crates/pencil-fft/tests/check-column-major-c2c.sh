#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
log_root=target/cm-c2c-test-logs
mkdir -p "$log_root" target
run_log_dir=$(mktemp -d "$log_root/run.XXXXXX") || exit 1
status_file=target/cm-c2c-preflight-fix-test.txt
index=0
failures=0

record() {
    printf '%s\n' "$*" >>"$status_file"
    printf '%s\n' "$*" >>"$run_log_dir/status.txt"
}

next_log() {
    index=$((index + 1))
    local safe=${1//[^a-zA-Z0-9_.-]/_}
    printf -v REPLY '%s/%03d-%s.log' "$run_log_dir" "$index" "$safe"
}

executed_tests() {
    awk '
        /test result:/ {
            passed = failed = 0
            for (i = 1; i <= NF; i++) {
                if ($i == "passed;") passed = $(i - 1)
                if ($i == "failed;") failed = $(i - 1)
            }
            total = passed + failed
        }
        END { print total + 0 }
    ' "$1"
}

valid_test_count() {
    awk '
        /test result:/ {
            summaries++
            if ($0 !~ /^test result: ok\./) invalid = 1
            passed = failed = 0
            for (i = 1; i <= NF; i++) {
                if ($i == "passed;") passed = $(i - 1)
                if ($i == "failed;") failed = $(i - 1)
            }
            count = passed + failed
            if (count == 0) invalid = 1
            total += count
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

validate_c2c_preflight_markers() {
    local ranked_dir=$1 ranks=$2 rank rank_log case_name marker count
    RANK_LOG_FAILURE=
    for ((rank = 0; rank < ranks; rank++)); do
        rank_log=$(rank_stdout_file "$ranked_dir" "$rank")
        for case_name in empty_collection precision_header_mismatch; do
            marker="C2C_PREFLIGHT_CASE=$case_name rank=$rank size=$ranks"
            count=$(grep -Fxc "$marker" "$rank_log" || true)
            if [[ $count != 1 ]]; then
                RANK_LOG_FAILURE="rank=$rank marker count=$count expected=1 marker=$marker"
                return 1
            fi
        done
    done
}

validate_rank_logs() {
    local launcher_status=$1 ranks=$2 ranked_dir=$3 rank rank_log rank_count total=0
    RANK_EXECUTED_TESTS=0
    RANK_LOG_FAILURE=
    if ((launcher_status != 0)); then
        RANK_LOG_FAILURE="launcher exit=$launcher_status"
        return 1
    fi
    if [[ ! $ranks =~ ^[1-9][0-9]*$ ]]; then
        RANK_LOG_FAILURE="invalid requested rank count=$ranks"
        return 1
    fi
    if [[ $(rank_stdout_count "$ranked_dir") -ne $ranks ]]; then
        RANK_LOG_FAILURE="stdout rank files=$(rank_stdout_count "$ranked_dir") expected=$ranks"
        return 1
    fi
    for ((rank = 0; rank < ranks; rank++)); do
        rank_log=$(rank_stdout_file "$ranked_dir" "$rank")
        if [[ -z $rank_log ]]; then
            RANK_LOG_FAILURE="missing stdout for rank=$rank"
            return 1
        fi
        rank_count=$(find "$ranked_dir" -maxdepth 1 -type f -name "rank.$rank.stdout" -print | awk 'END { print NR + 0 }')
        if ((rank_count != 1)) || ! rank_count=$(valid_test_count "$rank_log"); then
            RANK_LOG_FAILURE="rank=$rank has duplicate, failed, or zero-test stdout=$rank_log"
            return 1
        fi
        total=$((total + rank_count))
    done
    RANK_EXECUTED_TESTS=$total
}

check_rank_log_validator() {
    local fixture failed=0 rank case_name marker
    fixture=$(mktemp -d "$run_log_dir/rank-check.XXXXXX") || return 1
    printf 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s\n' >"$fixture/rank.0.stdout"
    printf 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s\n' >"$fixture/rank.1.stdout"
    validate_rank_logs 0 2 "$fixture" || failed=1

    for rank in 0 1; do
        printf 'C2C_PREFLIGHT_CASE=empty_collection rank=%s size=2\n' "$rank" >>"$fixture/rank.$rank.stdout"
        printf 'C2C_PREFLIGHT_CASE=precision_header_mismatch rank=%s size=2\n' "$rank" >>"$fixture/rank.$rank.stdout"
    done
    if validate_c2c_preflight_markers "$fixture" 2; then
        printf 'PASS C2C preflight marker validator accepts both cases on every rank\n'
    else
        printf 'FAIL C2C preflight marker validator rejected complete fixture: %s\n' "$RANK_LOG_FAILURE"
        failed=1
    fi
    for case_name in empty_collection precision_header_mismatch; do
        marker="C2C_PREFLIGHT_CASE=$case_name rank=1 size=2"
        cp "$fixture/rank.1.stdout" "$fixture/rank.1.saved"
        grep -Fv "$marker" "$fixture/rank.1.stdout" >"$fixture/rank.1.tmp" || true
        mv "$fixture/rank.1.tmp" "$fixture/rank.1.stdout"
        if validate_c2c_preflight_markers "$fixture" 2; then
            printf 'FAIL C2C preflight marker validator accepted missing %s\n' "$case_name"
            failed=1
        else
            printf 'PASS C2C preflight marker validator rejects missing %s\n' "$case_name"
        fi
        mv "$fixture/rank.1.saved" "$fixture/rank.1.stdout"
    done

    printf 'test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s\n' >"$fixture/rank.1.stdout"
    if validate_rank_logs 0 2 "$fixture"; then
        printf 'FAIL rank validator accepted zero tests\n'
        failed=1
    else
        printf 'PASS rank validator rejects zero tests\n'
    fi
    rm "$fixture/rank.1.stdout"
    if validate_rank_logs 0 2 "$fixture"; then
        printf 'FAIL rank validator accepted missing rank\n'
        failed=1
    else
        printf 'PASS rank validator rejects missing rank\n'
    fi
    printf 'test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s\n' >"$fixture/rank.1.stdout"
    if validate_rank_logs 0 2 "$fixture"; then
        printf 'FAIL rank validator accepted failed rank\n'
        failed=1
    else
        printf 'PASS rank validator rejects failed rank\n'
    fi
    if validate_rank_logs 1 2 "$fixture"; then
        printf 'FAIL rank validator accepted nonzero launcher exit\n'
        failed=1
    else
        printf 'PASS rank validator rejects nonzero launcher exit\n'
    fi
    rm -rf "$fixture"
    return "$failed"
}

run_single() {
    local label=$1
    shift
    next_log "$label"
    local logfile=$REPLY status count
    record "COMMAND label=$label: $(printf '%q ' "$@")"
    "$@" >"$logfile" 2>&1
    status=$?
    count=$(executed_tests "$logfile")
    record "RESULT label=$label exit=$status executed_tests=$count log=$logfile"
    if ((status != 0 || count == 0)); then
        failures=$((failures + 1))
        printf 'FAIL %s exit=%s executed_tests=%s log=%s\n' "$label" "$status" "$count" "$logfile"
        grep -a -n -A2 -B1 -E 'CM_FFT_C2C_CONTRACT|error:|test result: FAILED|panicked at|timed out' "$logfile" | head -n 14 || tail -n 8 "$logfile"
        return 1
    fi
    printf 'PASS %s executed_tests=%s\n' "$label" "$count"
}

run_mpi() {
    local label=$1 ranks=$2
    shift 2
    next_log "$label"
    local logfile=$REPLY safe=${label//[^a-zA-Z0-9_.-]/_} ranked_dir output_prefix status count
    ranked_dir=$(mktemp -d "$run_log_dir/${index}-${safe}.ranks.XXXXXX") || return 1
    output_prefix="$ranked_dir/openmpi"
    local -a command=(timeout --kill-after=5s 240s mpiexec --oversubscribe \
        --output-filename "$output_prefix" -n "$ranks" bash -c \
        'ranked_dir=$1; shift; exec >"$ranked_dir/rank.${OMPI_COMM_WORLD_RANK}.stdout" 2>&1; exec "$@"' \
        bash "$ranked_dir" "$@")
    record "COMMAND label=$label: $(printf '%q ' "${command[@]}")"
    "${command[@]}" >"$logfile" 2>&1
    status=$?
    if validate_rank_logs "$status" "$ranks" "$ranked_dir"; then
        count=$RANK_EXECUTED_TESTS
        if [[ $label == collections_c2c_n4 || $label == collections_c2c_n6 ]]; then
            if ! validate_c2c_preflight_markers "$ranked_dir" "$ranks"; then
                count=0
                record "RANK_REJECT label=$label reason=$RANK_LOG_FAILURE"
            else
                record "PREFLIGHT_MARKERS label=$label ranks=$ranks cases=empty_collection,precision_header_mismatch"
            fi
        fi
    else
        count=0
        record "RANK_REJECT label=$label reason=$RANK_LOG_FAILURE"
    fi
    record "RESULT label=$label exit=$status executed_tests=$count log=$logfile rank_logs=$ranked_dir"
    if ((status != 0 || count == 0)); then
        failures=$((failures + 1))
        printf 'FAIL %s exit=%s executed_tests=%s log=%s ranks=%s\n' "$label" "$status" "$count" "$logfile" "$ranked_dir"
        grep -a -n -A2 -B1 -E 'CM_FFT_C2C_CONTRACT|error:|test result: FAILED|panicked at|MPI_ABORT|timed out' "$logfile" | head -n 14 || true
        if ((count == 0)); then
            find "$ranked_dir" -maxdepth 1 -type f -name 'rank.*.stdout' -print -exec tail -n 6 {} \;
        fi
        return 1
    fi
    printf 'PASS %s executed_tests=%s\n' "$label" "$count"
}

run_contract() {
    local ranks=$1
    run_mpi "column_major_c2c_n${ranks}" "$ranks" \
        cargo test -p pencil-fft --all-features --test column_major_c2c --locked --offline \
        column_major_c2c_contract -- --exact --nocapture --test-threads=1
}

record 'START role=test-writer scope=C2C-test-files-only goal=column-major-c2c-contract test-only-followup=1/3 formal-red=pending'
record 'REFERENCE candidate=25060401a194d0d595fec022143ecd0eb485d514 source-worktree=/tmp/pi-super-20820215-7618-43df-9745-70aa292d2b6d/worktrees/2e378ea3-a711-451d-b614-878385fc2400 Sol=NOT_CLEAN read-only test-files-driver-reused=true'
record 'CHANGE_REASON timing=partial-selection-initial+later-transpose-share-slot; regression checks actual call/slot counts plus source aggregation for every timing phase'
record 'NEGATIVE_CALLGRAPH selection=distributed_c2c_contract_one_mpi_binary->negative_c2c_selection_mask_cases; collection=collections_c2c_one_mpi_binary->collections_preflight::run_c2c_preflight->{operation_mismatch,descriptor_mismatch,peer_preflight_c2c,c2c_oop_ip,empty_c2c_collection,precision_header_mismatch}; callback=lib:distributed::tests::c2c_in_place_transaction_poison_survives_error_and_panic'
record 'PREFLIGHT_CALLGRAPH full=collections_one_mpi_binary->run->{run_c2c_preflight,peer_preflight_families,r2c_oop_ip,cross_family_headers}; c2c-only=collections_c2c_one_mpi_binary->run_c2c->run_c2c_preflight; real/mixed-only cases stay full-entry'
record 'PREFLIGHT_CASES empty_collection=CollectionError::Empty OOP+IP, successful canonical C2C control, workspace unchanged, arrays empty; precision_header_mismatch=f32-rank0/f64-peers n4,n6, both precision controls succeed, exact CollectionError::HeaderMismatch, both inputs/outputs/workspaces unchanged; marker=per-rank stdout after assertions; driver rejects absent/duplicate marker on every n4/n6 rank'
record 'PREFLIGHT_AUDIT operation_mismatch=shared-called n1,4,6, exact HeaderMismatch exercised n4,6, OOP/IP data+workspaces unchanged; descriptor_mismatch=shared-called n1,4,6, method/permutation/direction/shape HeaderMismatch exercised n4,6, arrays+workspaces unchanged; peer_preflight_c2c=shared-called n1,4,6, n1 returns, n4,6 exact OutputLayoutMismatch/IncompatiblePencils/WorkspaceMismatch, data/state/workspaces unchanged and retry; c2c_oop_ip=shared n1,4,6 indexed member/workspace/state/count/direction/single-vs-many checks; empty_c2c_collection=shared n1,4,6 typed Empty OOP+IP/workspace unchanged; precision_header_mismatch=shared-called n1,4,6, n1 returns, n4,6 exact HeaderMismatch and both precision inputs/outputs/workspaces unchanged; peer_preflight_families=full-only n1,4,6 (peer cases n4,6), real/mixed typed indexed checks; r2c_oop_ip=full-only n1,4,6 real indexed data/workspace checks; cross_family_headers=full-only n1,4,6 (mismatch n4,6), exact C2C-vs-R2C HeaderMismatch+arrays/workspaces unchanged'
record 'ROUTE_CALLGRAPH ignored-lib-test=distributed::tests::column_major_routes_match_production_stage_geometry_and_transitions; exact --ignored runs at n4,n6'
record 'TIMING_CALLGRAPH exact-integration-filter=distributed_c2c_partial_route_timing_one_mpi_binary; ranks=n4,n6; methods=alltoallv,p2p; layouts=permute,strided; forward/inverse oop+ip'
record 'SOURCE_MAP contract=test:column_major_c2c_contract; distributed-C2C=test:distributed_c2c_contract_one_mpi_binary; timing-C2C=test:distributed_c2c_timing_overlap_one_mpi_binary; partial-timing=test:distributed_c2c_partial_route_timing_one_mpi_binary; collection-C2C=test:collections_c2c_one_mpi_binary; callback-peer=lib:distributed::tests::c2c_in_place_transaction_poison_survives_error_and_panic; strided/registry=lib:distributed::regression_tests::strided_c2c_wrappers_and_r2c_registry_preserve_contracts; route=lib:distributed::tests::column_major_routes_match_production_stage_geometry_and_transitions; timing-aggregation-source-contract=test:distributed_refactor_contracts::repeated_c2c_transitions_accumulate_every_timing_field; local-C2C-allocation=test:local_c2c_nd_no_alloc; source-contracts=distributed_refactor_contracts.rs,local_refactor_contracts.rs'
record 'CHANGED_SCOPE crates/pencil-fft/tests/{column_major_c2c.rs,distributed_c2c.rs,collections.rs,collections_preflight/mod.rs,timing_overlap.rs,distributed_refactor_contracts.rs,check-column-major-c2c.sh}; crates/pencil-fft/src/distributed/tests/{layout_tests.rs,mod.rs} (tests only)'
record 'REMAINING production C2C source not yet copied by implementer; expected standalone red halts before MPI matrix; new preflight runtime not claimed green; other-family IO remains out of scope'
record 'ENV inherited BINDGEN_EXTRA_CLANG_ARGS=-isystem /usr/lib/gcc/x86_64-linux-gnu/15/include CARGO_NET_OFFLINE=true HDF5_DIR=/usr/lib/x86_64-linux-gnu/hdf5/openmpi LIBCLANG_PATH=/usr/lib/llvm-21/lib MPICC=mpicc RUST_BACKTRACE=1 RUST_TEST_THREADS=1'
record 'COMMAND cargo_fmt_check: cargo fmt --all -- --check'
if ! cargo fmt --all -- --check; then
    record 'RESULT fmt_check exit=nonzero'
    exit 1
fi
record 'RESULT fmt_check exit=0'
if ! bash -n crates/pencil-fft/tests/check-column-major-c2c.sh; then
    record 'RESULT bash_syntax exit=nonzero'
    exit 1
fi
record 'RESULT bash_syntax exit=0'
record 'COMMAND rank_validator_selfcheck: check_rank_log_validator (includes missing/duplicate/zero/failed rank, nonzero launcher, and each missing preflight marker)'
if ! check_rank_log_validator; then
    record 'RESULT rank_validator_selfcheck exit=nonzero'
    exit 1
fi
record 'RESULT rank_validator_selfcheck exit=0 zero_test=reject missing_rank=reject failed_rank=reject nonzero_launcher=reject'

# This first rank is the formal red gate. The assertion itself must identify
# the default column-major identity/trailing-decomposition constructor contract.
next_log column_major_c2c_initial_n1
initial_log=$REPLY
initial_rank_dir=$(mktemp -d "$run_log_dir/${index}-initial-n1.ranks.XXXXXX") || exit 1
initial_prefix="$initial_rank_dir/openmpi"
initial_command=(timeout --kill-after=5s 240s mpiexec --oversubscribe \
    --output-filename "$initial_prefix" -n 1 bash -c \
    'ranked_dir=$1; shift; exec >"$ranked_dir/rank.${OMPI_COMM_WORLD_RANK}.stdout" 2>&1; exec "$@"' \
    bash "$initial_rank_dir" cargo test -p pencil-fft --all-features --test column_major_c2c \
    --locked --offline column_major_c2c_contract -- --exact --nocapture --test-threads=1)
record "COMMAND label=column_major_c2c_initial_n1: $(printf '%q ' "${initial_command[@]}")"
"${initial_command[@]}" >"$initial_log" 2>&1
initial_status=$?
initial_rank_log=$(rank_stdout_file "$initial_rank_dir" 0)
initial_rank_files=$(rank_stdout_count "$initial_rank_dir")
initial_rank_zero_files=$(find "$initial_rank_dir" -maxdepth 1 -type f -name 'rank.0.stdout' -print | awk 'END { print NR + 0 }')
initial_count=0
if [[ -n $initial_rank_log ]]; then
    initial_count=$(executed_tests "$initial_rank_log")
fi
record "RESULT label=column_major_c2c_initial_n1 exit=$initial_status executed_tests=$initial_count log=$initial_log rank_log=${initial_rank_log:-missing}"
if ((initial_status != 0)) && ((initial_count > 0)) && ((initial_rank_files == 1)) && \
        ((initial_rank_zero_files == 1)) && [[ -n $initial_rank_log ]] && \
        grep -a -q 'CM_FFT_C2C_CONTRACT: C2C constructor must accept identity memory order with default trailing decomposition' "$initial_rank_log" && \
        grep -a -q 'test result: FAILED' "$initial_rank_log"; then
    record "EXPECTED_RED assertion=CM_FFT_C2C_CONTRACT_constructor_default_identity_trailing_decomposition exit=$initial_status rank_log=$initial_rank_log"
    printf 'EXPECTED RED: constructor assertion CM_FFT_C2C_CONTRACT; exit=%s log=%s\n' "$initial_status" "$initial_rank_log"
    grep -a -n -A3 -B1 'CM_FFT_C2C_CONTRACT: C2C constructor must accept' "$initial_rank_log" | head -n 12
    record "END exit=$initial_status changes=column_major_c2c.rs,distributed_c2c.rs,collections.rs,collections_preflight/mod.rs,timing_overlap.rs,distributed_refactor_contracts.rs,src/distributed/tests/layout_tests.rs,src/distributed/tests/mod.rs,check-column-major-c2c.sh formal_red=CM_FFT_C2C_CONTRACT_constructor_invalid_input_layout remaining=preflight-runtime-not-run-before-source-integration"
    exit 1
fi
if ((initial_status != 0 || initial_count == 0)); then
    record "INVALID_RED exit=$initial_status executed_tests=$initial_count rank_files=$initial_rank_files rank_zero_files=$initial_rank_zero_files rank_log=${initial_rank_log:-missing}"
    printf 'INVALID RED: initial contract did not fail at the required assertion; exit=%s tests=%s log=%s\n' "$initial_status" "$initial_count" "$initial_log"
    tail -n 30 "$initial_log"
    exit 1
fi
record 'RESULT label=column_major_c2c_initial_n1 exit=0 executed_tests=1'
printf 'PASS column_major_c2c_initial_n1 executed_tests=1\n'

for ranks in 4 6; do
    run_contract "$ranks" || true
done

for suite in distributed_c2c timing_overlap collections; do
    case "$suite" in
        distributed_c2c) test_name=distributed_c2c_contract_one_mpi_binary ;;
        timing_overlap) test_name=distributed_c2c_timing_overlap_one_mpi_binary ;;
        collections) test_name=collections_c2c_one_mpi_binary ;;
    esac
    for ranks in 1 4 6; do
        run_mpi "${suite}_c2c_n${ranks}" "$ranks" \
            cargo test -p pencil-fft --all-features --test "$suite" --locked --offline \
            "$test_name" -- --exact --nocapture --test-threads=1 || true
    done
done

run_mpi c2c_in_place_transaction_unit_n1 1 \
    cargo test -p pencil-fft --all-features --lib --locked --offline \
    distributed::tests::c2c_in_place_transaction_poison_survives_error_and_panic \
    -- --exact --ignored --nocapture --test-threads=1 || true
run_mpi c2c_in_place_transaction_peer_n4 4 \
    cargo test -p pencil-fft --all-features --lib --locked --offline \
    distributed::tests::c2c_in_place_transaction_poison_survives_error_and_panic \
    -- --exact --ignored --nocapture --test-threads=1 || true
run_mpi c2c_strided_registry_unit_n1 1 \
    cargo test -p pencil-fft --all-features --lib --locked --offline \
    distributed::regression_tests::strided_c2c_wrappers_and_r2c_registry_preserve_contracts \
    -- --exact --ignored --nocapture --test-threads=1 || true
for ranks in 4 6; do
    run_mpi "c2c_production_route_n${ranks}" "$ranks" \
        cargo test -p pencil-fft --all-features --lib --locked --offline \
        distributed::tests::column_major_routes_match_production_stage_geometry_and_transitions \
        -- --exact --ignored --nocapture --test-threads=1 || true
    run_mpi "c2c_partial_timing_n${ranks}" "$ranks" \
        cargo test -p pencil-fft --all-features --test timing_overlap --locked --offline \
        distributed_c2c_partial_route_timing_one_mpi_binary \
        -- --exact --nocapture --test-threads=1 || true
done
run_single c2c_timing_aggregation_source_contract \
    cargo test -p pencil-fft --all-features --test distributed_refactor_contracts --locked --offline \
    repeated_c2c_transitions_accumulate_every_timing_field -- --exact --nocapture --test-threads=1 || true
run_single c2c_overlap_source_contract \
    cargo test -p pencil-fft --all-features --test distributed_refactor_contracts --locked --offline \
    overlap_and_overwrite_mappings_are_single_generic_helpers -- --exact --nocapture --test-threads=1 || true
run_single c2c_route_source_contract \
    cargo test -p pencil-fft --all-features --test distributed_refactor_contracts --locked --offline \
    transitions_take_pencil_columns_without_dummy_stages -- --exact --nocapture --test-threads=1 || true
run_single local_c2c_source_contracts \
    cargo test -p pencil-fft --all-features --test local_refactor_contracts --locked --offline -- --test-threads=1 || true
run_single local_c2c_nd_allocation_contracts \
    cargo test -p pencil-fft --all-features --test local_c2c_nd_no_alloc --locked --offline -- --test-threads=1 || true
run_single fftw_local_nonignored \
    cargo test -p pencil-fft --all-features --test fftw_local --locked --offline -- --test-threads=1 || true
run_single fftw_local_ignored_f64 \
    cargo test -p pencil-fft --all-features --test fftw_local --locked --offline \
    native_f64_all_local -- --exact --ignored --nocapture --test-threads=1 || true
run_single fftw_local_ignored_f32 \
    cargo test -p pencil-fft --all-features --test fftw_local --locked --offline \
    native_f32_all_local -- --exact --ignored --nocapture --test-threads=1 || true

record "SUMMARY jobs=$index failures=$failures"
record "END exit=$((failures != 0)) changes=column_major_c2c.rs,distributed_c2c.rs,collections.rs,collections_preflight/mod.rs,timing_overlap.rs,distributed_refactor_contracts.rs,src/distributed/tests/layout_tests.rs,src/distributed/tests/mod.rs,check-column-major-c2c.sh remaining=runtime-matrix-failures-if-any formal-red-in-this-worktree=true"
if ((failures != 0)); then
    printf 'FAIL: %s C2C test/driver jobs failed; logs=%s\n' "$failures" "$run_log_dir"
    exit 1
fi
printf 'All column-major C2C contract and matrix checks passed; logs=%s\n' "$run_log_dir"
