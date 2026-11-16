"""Box plot of GlobalMPPI per-cycle computation time (KSOS rollout, KSOS solver, MPPI) for one run.

The controller appends one line per cycle to test/log/globalmppi_timing.txt for every run, so the
run's lines are found by matching its per-seed cycle counts (from the logged robot_pos_history).

    python examples/plot_timing.py                      # latest run in logs/pushT_globalmppi_globalmppi
    python examples/plot_timing.py --run-dir logs/pushT_globalmppi_globalmppi/20261116_103632
"""
import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np

PARTS = {"total_ksos_rollout_time": "KSOS rollout", "total_ksos_solver_time": "KSOS solver", "total_mppi_time": "MPPI"}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3


def run_lines(timing_file, run_dir):
    """The timing lines of `run_dir`: the last block of seed segments whose lengths match its histories."""
    histories = sorted(glob.glob(f"{run_dir}/*robot_pos_history_seed*.npy"), key=os.path.getmtime)
    lengths = [len(np.load(f)) for f in histories]
    lines = open(timing_file).read().splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith("step=0,")] + [len(lines)]
    segments = np.diff(starts)
    for k in range(len(segments) - len(lengths), -1, -1):
        if list(segments[k:k + len(lengths)]) == lengths:
            return lines[starts[k]:starts[k + len(lengths)]], len(lengths)
    raise SystemExit(f"no block of {lengths} cycles in {timing_file}; the run may predate the timing log")


def main(args):
    run_dir = args.run_dir or max(glob.glob("logs/pushT_globalmppi_globalmppi/*/"), key=os.path.getmtime)
    lines, n_seeds = run_lines(args.timing_file, run_dir)
    ms = {label: 1000 * np.array([float(re.search(rf"{key}=([\d.e-]+)", line).group(1)) for line in lines])
          for key, label in PARTS.items()}

    fig, ax = plt.subplots(figsize=(6, 4))
    box = ax.boxplot(list(ms.values()), widths=0.5, patch_artist=True, showfliers=True,
                     medianprops=dict(color="#0b0b0a", linewidth=2),
                     flierprops=dict(marker="o", markersize=3, alpha=0.4, markeredgewidth=0))
    for patch, flier, color in zip(box["boxes"], box["fliers"], COLORS):
        patch.set(facecolor=color, edgecolor=color, alpha=0.85)
        flier.set_markerfacecolor(color)
    for i, values in enumerate(ms.values(), start=1):
        ax.annotate(f"{np.median(values):.1f} ms", (i + 0.28, np.median(values)), va="center", fontsize=9, color="#3a3a37")
    ax.set_xticks(range(1, len(ms) + 1), ms.keys())
    ax.set_ylabel("Average computation time (ms)")
    ax.set_title(f"Computation time ({len(lines)} time steps, {n_seeds} seeds)", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()

    out = args.output or os.path.join(run_dir, "timing_boxplot.png")
    fig.savefig(out, dpi=200)
    print(f"{run_dir}: {len(lines)} cycles over {n_seeds} seeds")
    for label, values in ms.items():
        print(f"  {label:<13} median {np.median(values):6.2f} ms, p90 {np.percentile(values, 90):6.2f} ms, max {values.max():6.2f} ms")
    print(f"saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default=None, help="logs/<task>_<alg>/<timestamp> (default: the latest run)")
    p.add_argument("--timing-file", default="test/log/globalmppi_timing.txt")
    p.add_argument("--output", default=None, help="PNG path (default: <run-dir>/timing_boxplot.png)")
    main(p.parse_args())
