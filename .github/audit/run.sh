#!/usr/bin/env bash
# Run the scheduled audit with one model.
#
# Inputs (environment):
#   AUDIT_MODEL       model for this run
#   AUDIT_PROMPT      prompt file
#   AUDIT_EXISTING    existing audit issues (JSON)
#   AUDIT_REPORT      report output path
#   AUDIT_MODEL_FILE  written with the model that produced the report
#
# Exit codes:
#   0  pi succeeded; AUDIT_MODEL is written to AUDIT_MODEL_FILE
#   1  pi failed for any other reason (pi's status is not passed through)
#   2  a required input is missing
#   3  pi failed on usage limits or capacity; the caller may retry with another model
#
# The report is echoed to stdout (the job log shows it through tee).
set -euo pipefail

die() {
  echo "run.sh: $*" >&2
  exit 2
}

for name in AUDIT_MODEL AUDIT_PROMPT AUDIT_EXISTING AUDIT_REPORT AUDIT_MODEL_FILE; do
  [[ -n ${!name:-} ]] || die "$name is required"
done

tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT

run_model() {
  local model=$1 status=0
  # read・grep・find・ls だけを渡す（bash・edit・write は使えない）。
  # 呼び出し元が if の条件かどうかによらず stderr をコピーしてから戻す。
  pi -p --no-session --no-context-files --tools read,grep,find,ls \
    --model "$model" --thinking high \
    @"$AUDIT_PROMPT" \
    @"$AUDIT_EXISTING" \
    "添付の指示に従って監査し、最後に指定の JSON ブロックを出力してください。" \
    2>"$tmp" | tee "$AUDIT_REPORT" || status=$?
  cat "$tmp" >&2
  return "$status"
}

status=0
run_model "$AUDIT_MODEL" || status=$?
if (( status == 0 )); then
  printf '%s\n' "$AUDIT_MODEL" >"$AUDIT_MODEL_FILE"
  exit 0
fi
if grep -Eiq '429|rate.?limit|usage.?limit|quota|capacity|overloaded' "$tmp"; then
  exit 3
fi
# pi 自身の終了状態は渡さない（2・3 は上記の予約コードと衝突するため）。
exit 1
