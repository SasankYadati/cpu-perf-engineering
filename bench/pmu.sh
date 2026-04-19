#!/usr/bin/env bash
# pmu.sh — collect PMU counters via xctrace and dump to XML.
#
# One-time setup: open Instruments.app → File → New → "CPU Counters" template
# → click the Counters track → in the bottom pane add the events you care about
# (cycles, INST_ALL, L1D_CACHE_MISS_LD, INST_BRANCH_MISPRED, FED_IC_MISS_DEM,
#  plus a SIMD/FP event if available). Save. The xctrace CLI inherits this
# saved configuration.
#
# Usage: ./bench/pmu.sh ./path/to/bin [args...]
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <binary> [args...]" >&2
  exit 1
fi

OUT="results/$(basename "$1")_$(date +%Y%m%d_%H%M%S).trace"
mkdir -p results

xctrace record \
  --template 'CPU Counters' \
  --output "$OUT" \
  --launch -- "$@"

echo "[pmu.sh] trace written to $OUT"
echo "[pmu.sh] open with: open '$OUT'   (Instruments GUI)"
echo "[pmu.sh] or export: xctrace export --input '$OUT' --toc"
