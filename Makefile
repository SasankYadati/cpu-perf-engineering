CXX      ?= clang++
CXXFLAGS ?= -O3 -std=c++17 -mcpu=apple-m1 -ffast-math -fno-math-errno -Wall -Wextra -I bench

STEP0 := step0/peak_fma step0/stream step0/cache_sweep

SGEMM_SRCS := kernels/sgemm/main.cpp kernels/sgemm/naive.cpp
SGEMM_BIN  := kernels/sgemm/sgemm

.PHONY: all step0 sgemm clean
all: step0 sgemm
step0: $(STEP0)
sgemm: $(SGEMM_BIN)

step0/%: step0/%.cpp bench/bench.hpp
	$(CXX) $(CXXFLAGS) $< -o $@

$(SGEMM_BIN): $(SGEMM_SRCS) kernels/sgemm/sgemm.hpp bench/bench.hpp
	$(CXX) $(CXXFLAGS) $(SGEMM_SRCS) -o $(SGEMM_BIN)

clean:
	rm -f $(STEP0) $(SGEMM_BIN)
