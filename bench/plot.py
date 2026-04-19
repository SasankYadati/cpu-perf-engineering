#!/usr/bin/env python3
"""plot.py — turn results.csv into the three plots we care about.

Usage:
    python bench/plot.py results/sgemm.csv writeups/sgemm/

Produces in the output dir:
    speedup.png   — bar chart, each kernel's rate vs the first row (naive).
    rate_vs_n.png — if names look like 'foo_N1024', GFLOPS as a function of N.
    roofline.png  — log-log roofline plot. Edit PEAK_GFLOPS / PEAK_BW_GBS
                    after you measure them in step 0.
"""
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Fill these in from your step0 measurements.
PEAK_GFLOPS = 132.77      # single P-core FP32 ceiling (measured)
PEAK_BW_GBS = 122.01      # single-core sustained DRAM bandwidth (measured)


def speedup_chart(df: pd.DataFrame, out: Path) -> None:
    base = df["rate_min"].iloc[0]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df["name"], df["rate_min"] / base)
    ax.set_ylabel(f"speedup over {df['name'].iloc[0]}")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out / "speedup.png", dpi=140)


def rate_vs_n(df: pd.DataFrame, out: Path) -> None:
    pat = re.compile(r"_N(\d+)$")
    df = df.assign(N=df["name"].map(lambda s: int(m.group(1)) if (m := pat.search(s)) else None))
    df = df.dropna(subset=["N"])
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for kernel, sub in df.groupby(df["name"].str.replace(pat, "", regex=True)):
        ax.plot(sub["N"], sub["rate_min"], marker="o", label=kernel)
    ax.set_xlabel("N")
    ax.set_ylabel(df["unit"].iloc[0])
    ax.set_xscale("log", base=2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "rate_vs_n.png", dpi=140)


def roofline(df: pd.DataFrame, out: Path) -> None:
    # Caller must add an 'arith_intensity' column (flops/byte) for this to plot.
    if "arith_intensity" not in df.columns:
        return
    import numpy as np
    ai = np.logspace(-2, 3, 200)
    roof = np.minimum(PEAK_GFLOPS, PEAK_BW_GBS * ai)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(ai, roof, "k-", lw=2, label="roofline")
    ax.scatter(df["arith_intensity"], df["rate_min"], c="C1", s=60, zorder=5)
    for _, r in df.iterrows():
        ax.annotate(r["name"], (r["arith_intensity"], r["rate_min"]), fontsize=8)
    ax.set_xlabel("arithmetic intensity (flops/byte)")
    ax.set_ylabel("GFLOPS")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "roofline.png", dpi=140)


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    csv, outdir = Path(sys.argv[1]), Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv)
    speedup_chart(df, outdir)
    rate_vs_n(df, outdir)
    roofline(df, outdir)
    print(f"wrote plots to {outdir}/")


if __name__ == "__main__":
    main()
