# SGEMM — single-precision matrix multiply

## Context
Square N×N row-major FP32 matmul, single P-core, single thread.
- Compute roof: 132.77 GFLOPS (from `00_machine.md`)
- Arithmetic intensity: N/6 flops/byte → solidly compute-bound at any interesting N
- **Gold bar: 66.4 GFLOPS (50% of peak)**
- Sanity check at the end: compare against Apple Accelerate's `cblas_sgemm`
  (which dispatches to AMX/SME, ~1.5–2 TFLOPS, so expect ~10× gap — that's
  the architecture, not your code)

## The ladder
The numbers below are for N=1024, where cache effects are prominent.

| # | Step                                | Status | GFLOPS | % of peak |
|---|-------------------------------------|--------|--------|-----------|
| 0 | Naive ijk                           | done   | 2.44   | 1.8%      |
| 1 | Loop reorder → ikj                  | | | |
| 2 | Clean autovectorization             | | | |
| 3 | Register tile 8×8 (NEON intrinsics) | | | |
| 4 | L1 cache blocking                   | | | |
| 5 | Pack B (and A) panels               | | | |
| 6 | BLIS 5-loop outer structure         | | | |
| 7 | Software prefetch                   | | | |
| 8 | (stretch) Micro-kernel asm polish   | | | |

---

## Step 0: Naive ijk

**Hypothesis.** I expect far below the peak of 132.77 GFLOPS due to multiple reasons: 1) we are moving way more than program minimum data from memory, 2) we are not leveraging linear access for matrix B thrashing the cache for each access when B is large. I expect below 5 GFLOPS. 

**Code.**
```cpp
void naive(const float* A, const float* B, float* C, int N) {

  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      for (int k=0; k<N; k++) {
        C[i * N + j] += A[i * N + k] * B[k * N + j];
      }
    }
  }
}
```

**Result.**
| N    | GFLOPS | % of peak |
|------|--------|-----------|
| 256  | 3.04   | 2.3%      |
| 512  | 2.84   | 2.1%      |
| 1024 | 2.44   | 1.8%      |

**Asm inspection.**
```bash
clang++ -O3 -mcpu=apple-m1 -std=c++17 -ffast-math -fno-math-errno -I bench \
    -S kernels/sgemm/naive.cpp -o - \
    | awk '/_ZN5sgemm5naive/,/\.cfi_endproc/' \
    | grep -E "fmadd|fmla|fmul|fadd|ldr|str" \
    | head -25
```
> There are four things happening in this command: 1) Compiling to assembly, 2) Isolating the function using range expression, 3) Keeping only the interesting instructions and 4) Capping the output.

Here's what we got from the assembly, with annotations:
```
  ldr    s0, [x12, x11, lsl #2]   ; (1) load one float from C[i*N+j]
  ldr    s1, [x16], #4             ; (2) load one float from A[i*N+k], post-increment
  ldr    s2, [x15]                 ; (3) load one float from B[k*N+j]
  fmadd  s0, s2, s1, s0            ; (4) s0 = s2 * s1 + s0   (C += A*B)
  str    s0, [x12, x11, lsl #2]    ; (5) store s0 back to C[i*N+j]
```

The key things to take away from here:
1. `ldr s0 ...` means loading a single float so there's no auto vectorization happening. 
2. The fact that B's load uses a plain [x15] tells us the compiler is keeping a pre-computed pointer to B's current element in x15 and updating it elsewhere by adding 4*N bytes on each iteration. When N=1024, this stride is 4096 bytes between consecutive B loads, which is 64 cache lines apart (64B sized cache line), so every load is a new cache line.
3. In each iteration we are loading C[i,j] and writing to it at the end. Compiler cannot prove A and B don't alias with C and therefore can't accumulate C[i,j] result in registers.
4. The useful work per iteration is about 20% (1 `fmadd` and 4 `ldr`,`str`). In the case of peak_fma the useful work ratio was 89%. So the naive sgemm is doing ~4.5x less useful work per instruction. Combined with the fact that peak_fma used vector instructions (4 floats) and sgemm uses scalar instructions (single float) which produces only 1/4 of the flops of a peak_fma's instruction giving 18x less flops per instruction.

**PMU counters.** We will defer measuring these since GFLOPS vs N and the asm analysis are sufficient evidence here.

**Verdict.** We confirmed the hypothesis and identified three bottlenecks:
1. Cache thrashing on B
2. No SIMD
3. C round-trip load and store per iteration

We will target bottleneck 1 by reordering loops so that B's access is linear without disturbing C and A's linear access. As a consequence, the inner loop becomes a vector operation which should unlock autovectorization from the compiler.

---

## Step 1: Loop reorder → ikj

*(to be added)*

---

## Final results

*(filled in after the ladder is done)*

## Sanity check vs Apple Accelerate

*(filled in at the end)*

## What I'd do differently

*(filled in after the ladder is done)*
