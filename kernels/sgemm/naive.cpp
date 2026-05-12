// kernels/sgemm/naive.cpp — Step 0 of the SGEMM ladder.
//
// The simplest possible correct matrix multiply: three nested loops in ijk
// order, accumulating into C[i*N+j]. Same -O3 flags as everything else —
// this is the "what does the compiler do on its own" baseline, not -O0.
//
// Expected: 1–5 GFLOPS at N=1024. Far below the 132.77 GFLOPS peak.
// The reason is the column-strided access into B (B[k*N+j] with k varying)
// which thrashes L1D — but don't take my word for it, measure and verify.

#include "sgemm.hpp"

namespace sgemm {

void naive(const float* A, const float* B, float* C, int N) {
  // TODO(you): write the standard ijk triple loop.
  //
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      for (int k=0; k<N; k++) {
        C[i * N + j] += A[i * N + k] * B[k * N + j];
      }
    }
  }
}

}  // namespace sgemm
