// kernels/sgemm/main.cpp — driver for the SGEMM ladder.
//
// For each registered variant, at each problem size:
//   1. zero C
//   2. run the variant once (verify against naive reference)
//   3. time it via bench::Harness (5 timed iterations after 1 warmup)
//   4. report GFLOPS and % of peak
//
// Add a new step: implement it in a new .cpp file, declare it in sgemm.hpp,
// add one entry to the kVariants table below, and add the .cpp to the Makefile.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

#include "../../bench/bench.hpp"
#include "sgemm.hpp"

namespace {

using SgemmFn = void (*)(const float*, const float*, float*, int);

struct Variant {
  const char* name;
  SgemmFn fn;
};

// Add new ladder steps here.
const Variant kVariants[] = {
  {"naive", sgemm::naive},
};

// Problem sizes to sweep. Start small while naive is slow; expand once we
// have a faster baseline.
const int kSizes[] = {256, 512, 1024};

constexpr double PEAK_GFLOPS = 132.77;  // from step 0 peak_fma

void fill_random(float* p, size_t n, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < n; ++i) p[i] = dist(rng);
}

}  // namespace

int main() {
  bench::ScopedQoS qos;
  bench::Harness h("results/sgemm.csv");

  for (int N : kSizes) {
    const size_t bytes = size_t(N) * N * sizeof(float);
    float* A     = (float*)bench::aligned_alloc_128(bytes);
    float* B     = (float*)bench::aligned_alloc_128(bytes);
    float* C     = (float*)bench::aligned_alloc_128(bytes);
    float* C_ref = (float*)bench::aligned_alloc_128(bytes);

    fill_random(A, size_t(N) * N, /*seed=*/42);
    fill_random(B, size_t(N) * N, /*seed=*/43);

    // Compute reference using naive (also serves as the naive timing).
    std::memset(C_ref, 0, bytes);
    sgemm::naive(A, B, C_ref, N);

    std::printf("\n=== N=%d  (peak: %.1f GFLOPS) ===\n", N, PEAK_GFLOPS);

    for (const Variant& v : kVariants) {
      // Correctness check (skip for naive — it IS the reference).
      if (v.fn != sgemm::naive) {
        std::memset(C, 0, bytes);
        v.fn(A, B, C, N);
        if (!sgemm::verify(C, C_ref, N)) {
          std::printf("  %s: SKIPPED (failed verification)\n", v.name);
          continue;
        }
      }

      // Time it. Each call zeroes C inside the lambda.
      char label[64];
      std::snprintf(label, sizeof(label), "%s_N%d", v.name, N);
      auto fn = [&]{
        std::memset(C, 0, bytes);
        v.fn(A, B, C, N);
      };
      auto result = h.run(label, sgemm::flops(N), fn, /*repeats=*/5, /*warmup=*/1);
      std::printf("  %-12s  %6.2f GFLOPS  (%.1f%% of peak)\n",
                  v.name, result.rate_min, 100.0 * result.rate_min / PEAK_GFLOPS);
    }

    std::free(A); std::free(B); std::free(C); std::free(C_ref);
  }
  return 0;
}
