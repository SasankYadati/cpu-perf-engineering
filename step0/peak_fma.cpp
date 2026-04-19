// step0/peak_fma.cpp — measure your P-core's actual FP32 FMA throughput.
//
// Idea: a tight loop of N independent FMA chains so the FMA pipes are saturated
// and there are no data hazards. With 4 NEON FMA pipes × 4 lanes × 2 flops/FMA
// you should see ~32 flops/cycle ≈ 144 GFLOPS @ 4.5 GHz on M4 P-core.
//
// Build:
//   clang++ -O3 -mcpu=apple-m1 -std=c++17 -I bench step0/peak_fma.cpp -o step0/peak_fma
// Run:
//   ./bench/run.sh ./step0/peak_fma

#include <arm_neon.h>
#include <cstdio>
#include "../bench/bench.hpp"

// Number of independent accumulator vectors. Must be ≥12 to hide the
// 3-cycle FMA latency across 4 pipes. We pick 16 (one per architectural
// vector register pair we can comfortably afford).
constexpr int CHAINS = 16;
constexpr long long ITERS = 200'000'000LL;  // tune so each call ~0.5–1.0 s

static void peak_fma_kernel() {
  float32x4_t acc[CHAINS];
  for (int i = 0; i < CHAINS; ++i) acc[i] = vdupq_n_f32(1.0f);
  const float32x4_t a = vdupq_n_f32(1.0000001f);
  const float32x4_t b = vdupq_n_f32(0.9999999f);

  // TODO(you): write a loop that runs ITERS iterations and inside each
  // iteration issues one vfmaq_f32(acc[i], a, b) for every i in [0, CHAINS).
  // Unroll by hand or trust the compiler — check the asm with:
  //     clang++ -O3 -mcpu=apple-m1 -S step0/peak_fma.cpp -o -
  // and confirm you see ~16 fmla instructions per loop body, with no spills.
  //
  // Then prevent the compiler from deleting the work: after the loop, sum
  // acc[0..CHAINS) into a scalar and write it through a volatile sink.
  //
  // Hint:  for (long long k = 0; k < ITERS; ++k) {
  //          acc[0] = vfmaq_f32(acc[0], a, b);
  //          acc[1] = vfmaq_f32(acc[1], a, b);
  //          ... (all CHAINS) ...
  //        }
  for (long long k = 0; k < ITERS; k++) {
    for (int i = 0; i < CHAINS; i++) {
      acc[i] = vfmaq_f32(acc[i], a, b);
    }
  }

  // Sink to keep the optimizer honest:
  float32x4_t s = acc[0];
  for (int i = 1; i < CHAINS; ++i) s = vaddq_f32(s, acc[i]);
  // Force the compiler to keep `s` live without actually doing I/O.
  asm volatile("" : : "w"(s));
}

int main() {
  bench::ScopedQoS qos;
  bench::Harness h("results/step0_peak_fma.csv");

  // flops per call: ITERS * CHAINS lanes(4) * 2 (FMA = mul+add)
  const double flops = double(ITERS) * CHAINS * 4 * 2;
  h.run("peak_fma", flops, peak_fma_kernel);

  std::printf("\nIf this is well below ~120 GFLOPS, check:\n"
              "  - laptop plugged in, Low Power Mode off\n"
              "  - QoS bias actually landed on a P-core (verify_pcore.sh)\n"
              "  - the inner loop emitted fmla, not generic fmul/fadd\n");
  return 0;
}
