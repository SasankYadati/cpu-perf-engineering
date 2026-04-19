#!/usr/bin/env bash
# run.sh — wrapper for stable benchmark runs on macOS / Apple Silicon.
# Reminders, then high-priority exec. Run as: ./bench/run.sh ./path/to/bin [args...]
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <binary> [args...]" >&2
  exit 1
fi

cat <<'EOF' >&2
[run.sh] Pre-flight reminders:
  1. Laptop plugged in?
  2. Low Power Mode disabled? (System Settings → Battery)
  3. Slack / browser / other heavy apps closed?
  4. First time this session — did you run verify_pcore.sh in another terminal?
EOF

# nice -n -20 needs sudo to actually take effect; without it macOS clamps it.
# We attempt without sudo (still slightly helpful) and don't fail if denied.
exec nice -n -20 "$@"
