# Project 0A: CPU Performance Engineering on M4 Pro

## Goal
Optimize two kernels from naive to within 50% of hardware peak, single P-core,
single thread, FP32. The transferable skill is the measure-hypothesize-optimize-verify
loop, not the specific optimizations.

## Machine baseline (measured in step 0)
- Peak FP32 compute: **132.77 GFLOPS** (single P-core)
- Peak single-core DRAM bandwidth: **131.86 GB/s** (STREAM-Triad)
- Roofline knee: ~1.01 flops/byte

## Gold bars
- SGEMM: **66.4 GFLOPS** (50% of compute peak)
- Prefix sum: **~66 GB/s** (50% of bandwidth peak)

## Kernels

### Kernel 1: SGEMM (compute-bound)
Square NxN FP32 matrix multiply. Primary target N=1024, also test 256, 2048, 4096.
Arithmetic intensity = N/6 — solidly compute-bound at any interesting size.

Optimization ladder:

| # | Step | Targets | Expected step speedup |
|---|------|---------|------|
| 0 | Naive `ijk` triple loop, `-O3` | baseline | — |
| 1 | Loop reorder -> `ikj` | unit-stride B access, autovec-friendly | 3-6x |
| 2 | Clean autovectorization (`__restrict`, alignment, verify with `-Rpass=loop-vectorize`) | NEON FMA issued | 2-3x |
| 3 | Register tile 8x8 with NEON intrinsics (`vfmaq_laneq_f32`) | fill all 4 FMA pipes, hide 3-cycle latency | 2-4x |
| 4 | L1 cache blocking (start: mc=64, kc=256, nc=256). Constraint: working set < ~100 KB in 128 KiB L1D | keep panels resident | 1.3-1.8x |
| 5 | Pack B into contiguous kc x nr panel (and A into mc x kc). Classic BLIS packA/packB | eliminate stride/TLB misses in micro-kernel | 1.2-1.5x |
| 6 | BLIS 5-loop outer structure (nc outer ~256-512, mc inner ~64-128). Reuse packed B across A panels | L2 reuse | 1.1-1.3x |
| 7 | SW prefetch (`__builtin_prefetch` next A row + next B panel, ~2 iterations ahead) | hide L2->L1 latency | 1.05-1.2x |
| 8 | (stretch) Micro-kernel asm polish: unroll kc, interleave loads with FMAs, eliminate spills | fill 4 FMA pipes every cycle | 1.1-1.3x |

Gold (~66 GFLOPS) expected around step 5-6.

Sanity check at the end: compare against Apple Accelerate's `cblas_sgemm`.
Note: Accelerate dispatches to AMX/SME (~1.5-2 TFLOPS), not NEON, so being
~10x slower than Accelerate is expected and correct.

### Kernel 2: Prefix sum (memory-bound)
FP32, 1D, sizes 1M / 16M / 256M elements. The 256M case is the pure-bandwidth
case where Gold matters most. Arithmetic intensity = 0.125 flops/byte.

Optimization ladder:

| # | Step | Targets |
|---|------|---------|
| 0 | Naive `out[i] = out[i-1] + in[i]` | baseline; serial dependency chain |
| 1 | Two-pass blocked scan: chunk into blocks, scan each locally, second pass adds block offsets | enables ILP and SIMD |
| 2 | NEON SIMD inner block scan (Hillis-Steele or independent lanes with fixup) | vector throughput |
| 3 | SW prefetch next block during current one | hide DRAM latency |
| 4 | Non-temporal stores (`stnp`) for output if array >> L2 | preserve L2 for input |
| 5 | Tune block size empirically; check IPC with PMU | last-mile tuning |

Gold (~66 GB/s) = `(2 * N * 4 bytes) / runtime`.

## Schedule (milestone-based, sequential)

| Block | Work | Hours |
|---|---|---|
| A | Step 0: microbenchmarks + harness | 7 |
| B | PMU pipeline + plotting + roofline | 3 |
| | **Checkpoint 1**: harness works, roofline drawn | |
| C | 5 perf-ninja warmups | 5 |
| D | SGEMM steps 0-3 (naive -> register tile) | 8 |
| | **Checkpoint 2**: SGEMM at ~30-50 GF, can read PMU output | |
| E | SGEMM steps 4-6 (blocking, packing, BLIS outer) | 8 |
| | **Checkpoint 3**: SGEMM Gold (~66 GF) | |
| F | SGEMM steps 7-8 (prefetch, asm polish) - optional | 4 |
| G | Prefix sum steps 0-5 | 10 |
| | **Checkpoint 4**: Prefix sum Gold | |
| H | Writeup polish, top-level README | 5 |

Total: ~50 hours.

## Compile flags
```
clang++ -O3 -std=c++17 -mcpu=apple-m1 -ffast-math -fno-math-errno -Wall -Wextra
```
(`apple-m1` is the accepted proxy; clang has no `apple-m4` target yet.)

Naive baseline uses the same `-O3` flags — not `-O0`. The question is
"can I beat what the compiler does on its own," not "can I beat unoptimized code."

## Writeup discipline
- Each kernel gets a markdown report following the template:
  setup -> naive baseline -> roofline analysis -> optimizations (each with
  hypothesis written BEFORE measuring, change, measurement, verdict) ->
  final results -> sanity check -> what I'd do differently
- Hypotheses are written before profiling, recorded unedited regardless of outcome
- "I was wrong" entries are expected and valuable

## Key references
- Apple Silicon CPU Optimization Guide v4.0 (local PDF, Registered Developer Use Only)
  - Read before SGEMM step 3: sections 3.2, 3.4, 3.5.1, 3.5.4, 3.6.1.1
  - Read before SGEMM step 4: sections 5.6.3, 5.6.5, 5.6.6, 5.6.12
  - Read before prefix sum: sections 5.6.8 (NT accesses), 5.6.10-11
  - PMU events: section 8.1.2
- Algorithmica HPC — Memory Hierarchy chapter (before SGEMM step 4)
- Siboehm — "Fast MMM on CPU" (after SGEMM steps 0-2)
- Salykova — "Beating NumPy's matrix multiplication" (after SGEMM steps 0-2)
- ARM intrinsics search: https://developer.arm.com/architectures/instruction-sets/intrinsics/
- BLIS kernels guide: https://github.com/flame/blis/blob/master/docs/KernelsHowTo.md
- flame/how-to-optimize-gemm: https://github.com/flame/how-to-optimize-gemm

## Warmups (5 from perf-ninja, before SGEMM)
1. `labs/memory_bound/loop_interchange_1` — ikj vs ijk
2. `labs/memory_bound/loop_tiling_1` — cache blocking
3. `labs/memory_bound/data_packing` — AoS -> SoA
4. `labs/memory_bound/sw_prefetching` — __builtin_prefetch
5. `labs/core_bound/vectorization_1` — removing autovec blockers
