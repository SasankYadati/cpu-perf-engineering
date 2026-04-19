#!/usr/bin/env bash
# verify_pcore.sh — confirm a benchmark thread is running on a P-core at max freq.
# Run this in another terminal WHILE a benchmark is running. Look for:
#   - P-cluster active residency near 100% on one core
#   - P-cluster frequency near 4500 MHz
#   - E-cluster mostly idle
# Ctrl-C when satisfied.
exec sudo powermetrics --samplers cpu_power -i 200
