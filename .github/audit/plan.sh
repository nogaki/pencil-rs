#!/usr/bin/env bash
# Decide today's audit range, focus and audit kinds.
#
# Inputs (environment):
#   AUDIT_DAY     days since 1970-01-01 UTC. Defaults to the current day.
#   AUDIT_RANGE   auto (default) | array | fft-local | fft-distributed | io
#   AUDIT_KIND    both (default) | design | ponytail
#
# Output (one line each): range=... focus=... scope=... audits=...
# Written to $GITHUB_OUTPUT when set, otherwise to stdout.
set -euo pipefail

die() {
  echo "plan.sh: $*" >&2
  exit 2
}

day=${AUDIT_DAY:-$(( $(date -u +%s) / 86400 ))}
range=${AUDIT_RANGE:-auto}
kind=${AUDIT_KIND:-both}

[[ $day =~ ^[0-9]+$ ]] || die "AUDIT_DAY must be a non-negative integer, got '$day'"
# 先頭ゼロ付きの値（08 など）を 8 進数と解釈させない。
day=$((10#$day))

ranges=(array fft-local fft-distributed io)
case "$range" in
  auto) range=${ranges[$((day % 4))]} ;;
  array | fft-local | fft-distributed | io) ;;
  *) die "AUDIT_RANGE must be auto, array, fft-local, fft-distributed or io, got '$range'" ;;
esac

foci=(DRY KISS SOLID)
focus=${foci[$(((day / 4) % 3))]}

case "$range" in
  array) scope='crates/pencil-array/src' ;;
  fft-local) scope='crates/pencil-fft/src（distributed.rs と distributed/ を除く）, crates/pencil-fftw/src' ;;
  fft-distributed) scope='crates/pencil-fft/src/distributed.rs, crates/pencil-fft/src/distributed/' ;;
  io) scope='crates/pencil-io/src' ;;
esac

case "$kind" in
  both) audits='["design","ponytail"]' ;;
  design) audits='["design"]' ;;
  ponytail) audits='["ponytail"]' ;;
  *) die "AUDIT_KIND must be both, design or ponytail, got '$kind'" ;;
esac

emit() {
  if [[ -n ${GITHUB_OUTPUT:-} ]]; then
    printf '%s\n' "$1" >>"$GITHUB_OUTPUT"
  else
    printf '%s\n' "$1"
  fi
}

emit "range=$range"
emit "focus=$focus"
emit "scope=$scope"
emit "audits=$audits"
