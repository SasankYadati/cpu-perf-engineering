// bench.hpp — minimal microbenchmark harness for Apple Silicon (M-series).
// Header-only C++17. No dependencies beyond libc / libpthread / Mach.
//
// Usage:
//   #include "bench.hpp"
//   int main() {
//     bench::ScopedQoS qos;            // bias this thread to P-cores
//     bench::Harness h("results.csv"); // append-only CSV
//     h.run("sgemm_naive", /*flops=*/2.0*N*N*N, [&]{ sgemm_naive(A,B,C,N); });
//   }
//
// Reports min / median / p99 over `repeats` timed iterations after `warmup`.
// Writes one CSV row per call to run().

#pragma once

#include <mach/mach_time.h>
#include <pthread/qos.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

namespace bench {

// ---------- timing ----------------------------------------------------------
// On Apple Silicon, mach_absolute_time() returns ticks of the system timebase.
// The tick is NOT 1 ns. mach_timebase_info gives numer/denom; on M1..M4 the
// ratio is 125/3 → one tick ≈ 41.67 ns. We cache it on first use.

inline double ticks_to_ns(uint64_t ticks) {
  static mach_timebase_info_data_t tb = []{
    mach_timebase_info_data_t t;
    mach_timebase_info(&t);
    return t;
  }();
  // TODO(you): convert `ticks` to nanoseconds using tb.numer / tb.denom.
  // Watch for integer overflow on long runs — cast to double early, or do
  // (ticks / tb.denom) * tb.numer + (ticks % tb.denom) * tb.numer / tb.denom.
  // Return as double ns.
  return (ticks / tb.denom) * tb.numer + (ticks % tb.denom) * tb.numer / tb.denom;
}

inline uint64_t now_ticks() { return mach_absolute_time(); }

// ---------- QoS / P-core biasing -------------------------------------------
// macOS does not allow hard CPU affinity without a private entitlement.
// QOS_CLASS_USER_INTERACTIVE biases this thread onto a P-core in practice.
// Verify once per session with: sudo powermetrics --samplers cpu_power -i 200
// (P-cluster active, E-cluster idle while the bench is running).

struct ScopedQoS {
  ScopedQoS() {
    // TODO(you): call pthread_set_qos_class_self_np with
    // QOS_CLASS_USER_INTERACTIVE and relative priority 0. Print a warning
    // (not an error) if it returns nonzero.
    int rc = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    if (rc != 0) {
      std::fprintf(stderr,
                   "[bench] warning: pthread_set_qos_class_self_np failed (%d); "
                   "thread may run on an E-core. Verify with verify_pcore.sh.\n",
                   rc);
    }
  }
};

// ---------- aligned alloc ---------------------------------------------------
// Apple Silicon cache line size is 128 B (hw.cachelinesize). Align to it.
inline void* aligned_alloc_128(size_t bytes) {
  void* p = nullptr;
  if (posix_memalign(&p, 128, bytes) != 0) return nullptr;
  return p;
}

// ---------- harness ---------------------------------------------------------
struct Result {
  std::string name;
  double work_units;   // flops or bytes — caller decides
  std::string unit;    // "GFLOPS" or "GB/s"
  double min_ns, median_ns, p99_ns;
  double rate_min;     // computed using min time (best case)
  size_t repeats;
};

class Harness {
 public:
  explicit Harness(const std::string& csv_path)
      : csv_(csv_path, std::ios::app) {
    // Write header if file is empty.
    csv_.seekp(0, std::ios::end);
    if (csv_.tellp() == 0) {
      csv_ << "name,unit,work_units,repeats,min_ns,median_ns,p99_ns,rate_min\n";
    }
  }

  // GFLOPS variant
  template <typename Fn>
  Result run(const std::string& name, double flops, Fn&& fn,
             size_t repeats = 11, size_t warmup = 3) {
    return run_impl(name, flops, "GFLOPS", 1e-9, std::forward<Fn>(fn),
                    repeats, warmup);
  }

  // GB/s variant
  template <typename Fn>
  Result run_bw(const std::string& name, double bytes, Fn&& fn,
                size_t repeats = 11, size_t warmup = 3) {
    return run_impl(name, bytes, "GB/s", 1e-9, std::forward<Fn>(fn),
                    repeats, warmup);
  }

 private:
  template <typename Fn>
  Result run_impl(const std::string& name, double work, const char* unit,
                  double scale, Fn&& fn, size_t repeats, size_t warmup) {
    for (size_t i = 0; i < warmup; ++i) fn();

    std::vector<double> times_ns;
    times_ns.reserve(repeats);
    for (size_t i = 0; i < repeats; ++i) {
      uint64_t t0 = now_ticks();
      fn();
      uint64_t t1 = now_ticks();
      times_ns.push_back(ticks_to_ns(t1 - t0));
    }
    std::sort(times_ns.begin(), times_ns.end());
    double tmin = times_ns.front();
    double tmed = times_ns[times_ns.size() / 2];
    double tp99 = times_ns[(times_ns.size() * 99) / 100];

    // rate = work / time. work is flops or bytes; ns→s via 1e-9; G via 1e-9.
    double rate = (work / tmin) * 1e9 * scale;  // = work / tmin_seconds * 1e-9

    Result r{name, work, unit, tmin, tmed, tp99, rate, repeats};
    csv_ << name << "," << unit << "," << work << "," << repeats << ","
         << tmin << "," << tmed << "," << tp99 << "," << rate << "\n";
    csv_.flush();

    std::printf("%-32s  %8.2f %s   (min %8.1f ns, med %8.1f, p99 %8.1f)\n",
                name.c_str(), rate, unit, tmin, tmed, tp99);
    return r;
  }

  std::ofstream csv_;
};

}  // namespace bench
