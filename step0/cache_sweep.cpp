// step0/cache_sweep.cpp — sum a buffer at sizes 16 KB → 256 MB.
// You should see distinct GB/s plateaus for L1 (192 KB), L2 (16 MB shared),
// and DRAM. The cliffs are your empirical confirmation of the M4 Pro cache
// hierarchy and they will reappear inside SGEMM size sweeps.
//
// Build:
//   clang++ -O3 -mcpu=apple-m1 -ffast-math -std=c++17 -I bench step0/cache_sweep.cpp -o step0/cache_sweep

#include <cstdio>
#include "../bench/bench.hpp"

static float sum_buffer(const float* __restrict p, size_t n) {
  // Multiple accumulators so the reduction itself isn't latency-bound.
  float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
  for (size_t i = 0; i < n; i += 4) {
    a0 += p[i + 0];
    a1 += p[i + 1];
    a2 += p[i + 2];
    a3 += p[i + 3];
  }
  return (a0 + a1) + (a2 + a3);
}

int main() {
  bench::ScopedQoS qos;
  bench::Harness h("results/step0_cache_sweep.csv");

  // Sizes (bytes): 16K, 64K, 192K (≈L1), 1M, 4M, 16M (≈L2), 64M, 256M
  const size_t sizes[] = {
    16 * 1024,   64 * 1024,   192 * 1024, 1 * 1024 * 1024,
    4 * 1024 * 1024, 16 * 1024 * 1024, 64 * 1024 * 1024, 256 * 1024 * 1024,
  };

  for (size_t bytes : sizes) {
    size_t n = bytes / sizeof(float);
    float* buf = (float*)bench::aligned_alloc_128(bytes);
    for (size_t i = 0; i < n; ++i) buf[i] = 1.0f;

    // Run multiple sweeps per timed call so each call is long enough to time.
    int sweeps = std::max(1, int((1 << 26) / int(n)));  // ~64M elems / call
    char name[64];
    std::snprintf(name, sizeof(name), "sum_%zuKB", bytes / 1024);
    h.run_bw(name, double(bytes) * sweeps, [=]{
      volatile float s = 0;
      for (int k = 0; k < sweeps; ++k) s += sum_buffer(buf, n);
      (void)s;
    });
    std::free(buf);
  }
  return 0;
}
