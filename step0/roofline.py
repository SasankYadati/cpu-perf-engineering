#!/usr/bin/env python3
"""Draw the baseline roofline plot from step0 measurements."""
import numpy as np
import matplotlib.pyplot as plt

PEAK_GFLOPS = 132.77
PEAK_BW_GBS = 131.86

ai = np.logspace(-2, 3, 500)
roof = np.minimum(PEAK_GFLOPS, PEAK_BW_GBS * ai)
knee = PEAK_GFLOPS / PEAK_BW_GBS

fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(ai, roof, "k-", lw=2.5)

ax.axhline(PEAK_GFLOPS, color="C0", ls="--", lw=1, label=f"compute ceiling: {PEAK_GFLOPS} GFLOPS")
ax.axvline(knee, color="gray", ls=":", lw=1, label=f"knee: {knee:.2f} flops/byte")

ax.fill_between(ai, roof, alpha=0.07, color="k")

ax.set_xlabel("Arithmetic intensity (flops / byte)")
ax.set_ylabel("Attainable GFLOPS")
ax.set_title("Single P-core roofline — M4 Pro")
ax.set_xlim(0.01, 1000)
ax.set_ylim(0.1, 300)
ax.legend(loc="lower right")
ax.grid(True, which="both", ls=":", alpha=0.3)
fig.tight_layout()
fig.savefig("writeups/step0_roofline.png", dpi=150)
print("wrote writeups/step0_roofline.png")
