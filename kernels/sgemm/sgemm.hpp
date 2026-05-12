// sgemm.hpp — kernel signatures and shared utilities for SGEMM variants.
//
// Convention: row-major storage. A is M×K, B is K×N, C is M×N.
// For our project, M = N = K (square matrices), so the API takes a single N.
//
// Semantics: C = A * B. Caller is responsible for zeroing C before each call.
//
// Each variant is a separate function declared here and implemented in its own
// .cpp file. The main.cpp harness runs all of them and compares against naive.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdio>

namespace sgemm {

// ---- variant signatures ---------------------------------------------------
// All variants follow this signature. Add a new declaration here whenever you
// implement a new step in the SGEMM ladder.

void naive(const float* A, const float* B, float* C, int N);

// ---- verification ---------------------------------------------------------
// Compare two N×N matrices element-wise. SGEMM accumulates O(N) products per
// output element, so FP32 rounding can give a max absolute error in the
// ~1e-3 range at N=1024 even when the algorithm is correct. We check both
// absolute and relative tolerance.
inline bool verify(const float* C, const float* C_ref, int N,
                   float atol = 1e-2f, float rtol = 1e-3f) {
  float max_abs = 0, max_rel = 0;
  int bad_i = -1, bad_j = -1;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      float a = C[i * N + j];
      float r = C_ref[i * N + j];
      float diff = std::fabs(a - r);
      float rel = diff / (std::fabs(r) + 1e-9f);
      if (diff > max_abs) { max_abs = diff; bad_i = i; bad_j = j; }
      if (rel > max_rel) max_rel = rel;
    }
  }
  bool ok = (max_abs <= atol) || (max_rel <= rtol);
  if (!ok) {
    std::printf("  VERIFY FAIL: max_abs=%.4e max_rel=%.4e at (%d,%d): "
                "got %.6f, ref %.6f\n",
                max_abs, max_rel, bad_i, bad_j,
                C[bad_i * N + bad_j], C_ref[bad_i * N + bad_j]);
  }
  return ok;
}

// ---- flop count -----------------------------------------------------------
// 2 flops (mul + add) per inner-loop iteration, N^3 iterations total.
inline double flops(int N) {
  return 2.0 * double(N) * double(N) * double(N);
}

}  // namespace sgemm
