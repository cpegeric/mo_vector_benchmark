#!/usr/bin/env bash
# gen_88m.sh — split an .fbin (e.g. 88M vectors) into N CSV shards in parallel
# by launching N gen.py processes, each handling a non-overlapping row range.
#
# Usage:
#   ./gen_88m.sh -i <input.fbin> -o <output_prefix> [-n <shards>]
# Example:
#   ./gen_88m.sh -i data/wiki_all_88M/base.88M.fbin -o data/wiki_all_88M/wiki_88M_part
#   -> writes wiki_88M_part0.csv ... wiki_88M_part9.csv
#
# NOTE on id uniqueness:
#   gen.py writes each row's `id` column starting from 1 in every process, so
#   ids will collide across shards. If the target table uses `id` as a PK, you
#   must either (a) change col-1 to \N and let MatrixOne AUTO_INCREMENT assign
#   ids on LOAD, or (b) patch gen.py so global_i starts at skip_rows + 1.

set -euo pipefail

SHARDS=10
INPUT=""
PREFIX=""

usage() {
  echo "Usage: $0 -i <input.fbin> -o <output_prefix> [-n <shards>]" >&2
  exit 2
}

while getopts "i:o:n:h" opt; do
  case "$opt" in
    i) INPUT=$OPTARG ;;
    o) PREFIX=$OPTARG ;;
    n) SHARDS=$OPTARG ;;
    h|*) usage ;;
  esac
done

[[ -z "$INPUT" || -z "$PREFIX" ]] && usage
[[ ! -f "$INPUT" ]] && { echo "input not found: $INPUT" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN_PY="$SCRIPT_DIR/gen.py"
[[ ! -f "$GEN_PY" ]] && { echo "gen.py not found at $GEN_PY" >&2; exit 1; }

# Read total row count from .fbin header (first 4 bytes, little-endian uint32).
N=$(python3 -c "import struct,sys; print(struct.unpack('<II', open(sys.argv[1],'rb').read(8))[0])" "$INPUT")
echo "fbin: $INPUT  rows=$N  shards=$SHARDS  prefix=$PREFIX" >&2

PER=$(( (N + SHARDS - 1) / SHARDS ))

OUT_DIR="$(dirname "$PREFIX")"
[[ -n "$OUT_DIR" && "$OUT_DIR" != "." ]] && mkdir -p "$OUT_DIR"

pids=()
shards_started=0
for ((i=0; i<SHARDS; i++)); do
  SKIP=$(( i * PER ))
  (( SKIP >= N )) && break
  REMAIN=$(( N - SKIP ))
  MAX=$(( REMAIN < PER ? REMAIN : PER ))
  OUT="${PREFIX}${i}.csv"
  LOG="${OUT}.log"
  echo "[shard $i] skip=$SKIP max=$MAX -> $OUT (log: $LOG)" >&2
  python3 "$GEN_PY" \
    --fbin "$INPUT" \
    -o "$OUT" \
    --skip-rows "$SKIP" \
    --max-rows "$MAX" \
    --seed $((42 + i)) \
    >"$LOG" 2>&1 &
  pids+=($!)
  shards_started=$((shards_started + 1))
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
    echo "shard pid=$pid failed; check ${PREFIX}*.csv.log" >&2
  fi
done

(( fail )) && exit 1

echo "Done. Generated $shards_started CSV(s):" >&2
for ((i=0; i<shards_started; i++)); do
  OUT="${PREFIX}${i}.csv"
  [[ -f "$OUT" ]] && ls -lh "$OUT" >&2
done
