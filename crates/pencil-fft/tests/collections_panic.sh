#!/usr/bin/env bash
# Fresh MPI jobs only: never continue ordinary tests after MPI_Abort.
# Usage: bash crates/pencil-fft/tests/collections_panic.sh /absolute/path/to/pencil_fft-unit-binary [log-dir]
set -euo pipefail
binary=${1:?supply the distributed pencil-fft unit-test executable}
log_dir=${2:-/tmp/pencil-collections-panic-logs}
[[ -x "$binary" ]] || { echo "missing test executable: $binary" >&2; exit 1; }
mkdir -p "$log_dir"
for ranks in 1 4 6; do
    log="$log_dir/panic-$ranks.log"
    status=0
    PENCIL_COLLECTION_PANIC_SUBPROCESS=1 timeout --kill-after=5s 60s \
        "${MPIEXEC:-mpiexec}" --oversubscribe -n "$ranks" "$binary" \
        --exact distributed::tests::in_place_transaction_poison_survives_error_and_panic \
        --nocapture --test-threads=1 >"$log" 2>&1 || status=$?
    # OpenMPI propagates the explicit error code. Reject timeouts, test-filter
    # mistakes, launch/build failures, and unrelated nonzero exits.
    if [[ $status -ne 86 ]] ||
        ! grep -q 'COLLECTION_RANK_ASYMMETRIC_PANIC' "$log" ||
        ! grep -q 'COLLECTION_MEMBER_PANIC_ABORT index=0' "$log" ||
        ! grep -q 'MPI_ABORT' "$log"; then
        echo "FAIL ranks=$ranks status=$status log=$log" >&2
        exit 1
    fi
    echo "PASS ranks=$ranks expected_abort_status=$status log=$log"
done
