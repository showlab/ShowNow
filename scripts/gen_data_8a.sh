#!/bin/bash
# 8-action ViZDoom fork generation (CPU only; sim/gen_vizdoom_cf8.py), several worker processes.
#   scripts/gen_data_8a.sh <out dir> <first index> <workers> <forks per worker> [extra generator args]
# hold-key forks (every branch holds its key for 3 s):  ... [--maps MAP01,...]
# key-switch forks (keys switch every 0.25-1.5 s):      ... --mode switch
set -u
cd "$(dirname "$0")/../sim"
OUT=$1; S0=$2; NW=$3; C=$4; shift 4
mkdir -p "$OUT"
for p in $(seq 0 $((NW - 1))); do
  start=$(( S0 + p * C ))
  python gen_vizdoom_cf8.py --out "$OUT" --start $start --count $C --crf 23 --threads 2 --skip-existing "$@" \
      > "$OUT/log_$(printf %05d $start).txt" 2>&1 &
done
wait
echo "generated $S0 .. $((S0 + NW * C - 1))"
