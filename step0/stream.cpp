// step0/stream.cpp — single-core STREAM-Triad: a[i] = b[i] + s * c[i].
// Measures the realistic single-P-core sustained DRAM bandwidth ceiling,
// which is the denominator for "% of peak" on memory-bound kernels like
// prefix sum. Expect roughly 70–110 GB/s on M4 Pro single-core.
//
// Build:
//   clang++ -O3 -mcpu=apple-m1 -ffast-math -std=c++17 -I bench step0/stream.cpp -o step0/stream
// Run:
//   ./bench/run.sh ./step0/stream

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "../bench/bench.hpp"

// 256 MiB per array → 768 MiB total. Far larger than the 16 MB L2.
constexpr size_t N = 64 * 1024 * 1024;  // 64M floats = 256 MB

static float* A = nullptr;
static float* B = nullptr;
static float* C = nullptr;

static void triad() {
  const float s = 3.0f;
  // TODO(you): write the triad loop.
  // Use __restrict on the pointers (declare locals) and let clang autovectorize.
  // Verify with: clang++ ... -Rpass=loop-vectorize -c step0/stream.cpp
  // You should see "vectorized loop" emitted for this function.
  float* __restrict a = A;
  const float* __restrict b = B;
  const float* __restrict c = C;
  for (size_t i = 0; i < N; ++i) {
    a[i] = b[i] + s * c[i];
  }
  (void)s;
}

int main() {
  bench::ScopedQoS qos;
  bench::Harness h("results/step0_stream.csv");

  A = (float*)bench::aligned_alloc_128(N * sizeof(float));
  B = (float*)bench::aligned_alloc_128(N * sizeof(float));
  C = (float*)bench::aligned_alloc_128(N * sizeof(float));
  for (size_t i = 0; i < N; ++i) { A[i] = 0; B[i] = 1.0f; C[i] = 2.0f; }

  // Bytes moved per triad call: read B (N*4) + read C (N*4) + write A (N*4)
  // = 3 * N * 4 bytes. Note: write-allocate may cause an extra read of A on
  // some architectures, making the "true" traffic 4*N*4. We report the
  // STREAM-standard 3*N*4 figure for comparability.
  const double bytes = 3.0 * N * sizeof(float);
  h.run_bw("stream_triad", bytes, triad);

  std::printf("\nThis is your single-core memory roofline. Plug it into\n"
              "bench/plot.py PEAK_BW_GBS so prefix-sum %% of peak is honest.\n");
  return 0;
}
