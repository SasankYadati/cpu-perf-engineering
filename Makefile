CXX      ?= clang++
CXXFLAGS ?= -O3 -std=c++17 -mcpu=apple-m1 -ffast-math -fno-math-errno -Wall -Wextra -I bench

STEP0 := step0/peak_fma step0/stream step0/cache_sweep

.PHONY: all step0 clean
all: step0
step0: $(STEP0)

step0/%: step0/%.cpp bench/bench.hpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(STEP0)
