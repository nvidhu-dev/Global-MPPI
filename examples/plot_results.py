"""Task-success and control-quality statistics for one or more PushT runs.

    python examples/plot_results.py --run "NF+L2O LSE=logs/pushT_globalmppi_globalmppi/20261116_103632" \
                                    --run "NF+L2O plain=logs/pushT_globalmppi_globalmppi/20261116_110055"

Reads robot_pos_history (block x, y, yaw, pusher x, y per time step) and best_cost_history of every seed.
An episode stops when the block reaches the run's goal tolerance (--stop-tol, 5 mm by default); after
that the block is taken to stay where it ended, so curves over time hold the final value.
Writes one PNG per plot and summary.csv to --out.
"""
import argparse
import csv
import glob
import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # categorical slots, fixed order
PUSHER_OFFSET = np.array([0.0, 0.1])  # pusher qpos is relative to its body position


def load_run(run_dir):
    seeds = {}
    for f in glob.glob(f"{run_dir}/*robot_pos_history_seed*.npy"):
        seed = int(re.search(r"seed(\d+)\.npy$", f).group(1))
        cost = f.replace("robot_pos_history", "best_cost_history")
        seeds[seed] = (np.load(f), np.load(cost).ravel() if os.path.exists(cost) else None)
    if not seeds:
        raise SystemExit(f"no robot_pos_history files in {run_dir}")
    return dict(sorted(seeds.items()))


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def time_to_goal(q, pos_tol, angle_tol, dt):
    ok = (np.hypot(q[:, 0], q[:, 1]) < pos_tol) & (np.abs(wrap(q[:, 2])) < angle_tol)
    return dt * np.argmax(ok) if ok.any() else np.nan


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def hold_last(x, steps):
    """Pad a per-step series to `steps` with its last value (the episode stopped at the goal)."""
    return np.concatenate([x[:steps], np.full(max(steps - len(x), 0), x[-1])])


def style(ax, ylabel, xlabel=None):
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig, out, name):
    fig.tight_layout()
    path = os.path.join(out, name)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"saved {path}")


def main(args):
    runs = {}
    for spec in args.run:
        name, _, path = spec.partition("=")
        runs[name if path else os.path.basename(os.path.normpath(name))] = load_run(path or name)
    out = args.out or os.path.join("logs", "results_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    dt, steps = args.dt, round(args.duration / args.dt) + 1
    t = dt * np.arange(steps)
    names, colors = list(runs), dict(zip(runs, COLORS))

    stats = {}
    for name, seeds in runs.items():
        qs = [q for q, _ in seeds.values()]
        ttg = np.array([time_to_goal(q, args.stop_tol, args.angle_tol, dt) for q in qs])
        dist = np.array([hold_last(np.hypot(q[:, 0], q[:, 1]), steps) for q in qs])
        path = np.array([np.sum(np.linalg.norm(np.diff(q[:, 3:5], axis=0), axis=1)) for q in qs])
        costs = [c for _, c in seeds.values() if c is not None and len(c) and c[0] > 0]
        stats[name] = dict(seeds=list(seeds), ttg=ttg, dist=dist, path=path,
                           final_pos=100 * np.array([np.hypot(*q[-1, :2]) for q in qs]),
                           final_yaw=np.degrees(np.abs([wrap(q[-1, 2]) for q in qs])),
                           cost=np.array([hold_last(c / c[0], steps) for c in costs]) if costs else None,
                           tols=[np.mean([np.isfinite(time_to_goal(q, tol, args.angle_tol, dt)) for q in qs])
                                 for tol in args.tols])

    # 1. Success rate with 95% Wilson interval
    fig, ax = plt.subplots(figsize=(max(3.5, 1.6 + 1.3 * len(names)), 4))
    for i, name in enumerate(names):
        s = stats[name]
        k, n = int(np.isfinite(s["ttg"]).sum()), len(s["ttg"])
        lo, hi = wilson(k, n)
        ax.bar(i, k / n, width=0.6, color=colors[name])
        ax.errorbar(i, k / n, yerr=[[k / n - lo], [hi - k / n]], color="#3a3a37", capsize=6, linewidth=1.2)
        ax.annotate(f"{k}/{n}", (i, hi + 0.03), ha="center", fontsize=9, color="#3a3a37")
    ax.set_xticks(range(len(names)), names)
    ax.set_ylim(0, 1.12)
    ax.set_title(f"Success rate ({1000 * args.stop_tol:g} mm, {np.degrees(args.angle_tol):.1f}°)", fontsize=11)
    style(ax, "Success rate (95% Wilson interval)")
    save(fig, out, "success_rate.png")

    # 2. Time to goal, successful seeds only
    fig, ax = plt.subplots(figsize=(1.6 + 1.3 * len(names), 4))
    data = [stats[n]["ttg"][np.isfinite(stats[n]["ttg"])] for n in names]
    box = ax.boxplot([d if len(d) else [np.nan] for d in data], widths=0.5, patch_artist=True,
                     medianprops=dict(color="#0b0b0a", linewidth=2), flierprops=dict(marker="o", markersize=3))
    for patch, name in zip(box["boxes"], names):
        patch.set(facecolor=colors[name], edgecolor=colors[name], alpha=0.85)
    for i, d in enumerate(data, start=1):
        ax.scatter(np.full(len(d), i) + np.random.default_rng(0).uniform(-0.12, 0.12, len(d)), d, s=10, color="#3a3a37", zorder=3)
    ax.set_xticks(range(1, len(names) + 1), names)
    ax.set_ylim(bottom=0)
    ax.set_title("Time to reach the goal (successful seeds)", fontsize=11)
    style(ax, "Time to goal (s)")
    save(fig, out, "time_to_goal.png")

    # 3. Fraction of seeds solved by time t
    fig, ax = plt.subplots(figsize=(6, 4))
    for name in names:
        ttg = stats[name]["ttg"]
        ax.step(t, [np.mean(ttg <= tt) for tt in t], where="post", color=colors[name], linewidth=2, label=name)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(0, 1.02)
    ax.set_title("Fraction of seeds at the goal", fontsize=11)
    ax.legend(frameon=False, loc="lower right")
    style(ax, "Fraction solved", "Time (s)")
    save(fig, out, "fraction_solved.png")

    # 4. Success against position tolerance
    fig, ax = plt.subplots(figsize=(6, 4))
    for name in names:
        ax.plot(1000 * np.array(args.tols), stats[name]["tols"], marker="o", markersize=6, color=colors[name], linewidth=2, label=name)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Success rate against position tolerance (angle < {np.degrees(args.angle_tol):.1f}°)", fontsize=11)
    ax.legend(frameon=False, loc="lower right")
    style(ax, "Success rate", "Position tolerance (mm)")
    save(fig, out, "success_vs_tolerance.png")

    # 5. Final position and angle error
    fig, ax = plt.subplots(figsize=(6, 4))
    for name in names:
        ax.scatter(stats[name]["final_pos"], stats[name]["final_yaw"], s=40, color=colors[name], edgecolor="white", linewidth=1, label=name)
    ax.axvline(100 * args.stop_tol, color="#8a8984", linestyle="--", linewidth=1)
    ax.axhline(np.degrees(args.angle_tol), color="#8a8984", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1.15 * max(100 * args.stop_tol, *(stats[n]["final_pos"].max() for n in names)))
    ax.set_ylim(0, 1.15 * max(np.degrees(args.angle_tol), *(stats[n]["final_yaw"].max() for n in names)))
    ax.set_title("Final error per seed (dashed: goal tolerance)", fontsize=11)
    ax.legend(frameon=False)
    style(ax, "Final angle error (deg)", "Final position error (cm)")
    save(fig, out, "final_error.png")

    # 6. Distance to goal over time: median and 25-75% band
    fig, ax = plt.subplots(figsize=(6, 4))
    for name in names:
        d = 100 * stats[name]["dist"]
        ax.fill_between(t, np.percentile(d, 25, axis=0), np.percentile(d, 75, axis=0), color=colors[name], alpha=0.2, linewidth=0)
        ax.plot(t, np.median(d, axis=0), color=colors[name], linewidth=2, label=name)
    ax.axhline(100 * args.stop_tol, color="#8a8984", linestyle="--", linewidth=1)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(bottom=0)
    ax.set_title("Block distance to goal (median, 25-75% band)", fontsize=11)
    ax.legend(frameon=False)
    style(ax, "Distance to goal (cm)", "Time (s)")
    save(fig, out, "distance_over_time.png")

    # 7. Pusher path length
    fig, ax = plt.subplots(figsize=(1.6 + 1.3 * len(names), 4))
    box = ax.boxplot([100 * stats[n]["path"] for n in names], widths=0.5, patch_artist=True,
                     medianprops=dict(color="#0b0b0a", linewidth=2), flierprops=dict(marker="o", markersize=3))
    for patch, name in zip(box["boxes"], names):
        patch.set(facecolor=colors[name], edgecolor=colors[name], alpha=0.85)
    ax.set_xticks(range(1, len(names) + 1), names)
    ax.set_ylim(bottom=0)
    ax.set_title("Pusher path length per episode", fontsize=11)
    style(ax, "Path length (cm)")
    save(fig, out, "pusher_path.png")

    # 8. Best cost over time, normalised by each seed's first cost
    if all(stats[n]["cost"] is not None for n in names):
        fig, ax = plt.subplots(figsize=(6, 4))
        for name in names:
            c = stats[name]["cost"]
            ax.fill_between(t, np.percentile(c, 25, axis=0), np.percentile(c, 75, axis=0), color=colors[name], alpha=0.2, linewidth=0)
            ax.plot(t, np.median(c, axis=0), color=colors[name], linewidth=2, label=name)
        ax.set_xlim(0, t[-1])
        ax.set_ylim(bottom=0)
        ax.set_title("Best cost, relative to the first step (median, 25-75% band)", fontsize=11)
        ax.legend(frameon=False)
        style(ax, "Best cost / initial best cost", "Time (s)")
        save(fig, out, "cost_over_time.png")

    # Summary table
    rows = []
    for name in names:
        s = stats[name]
        k, n = int(np.isfinite(s["ttg"]).sum()), len(s["ttg"])
        lo, hi = wilson(k, n)
        ok = s["ttg"][np.isfinite(s["ttg"])]
        rows.append({"method": name, "seeds": n, "success": f"{k}/{n}", "success_ci95": f"[{lo:.2f}, {hi:.2f}]",
                     "time_to_goal_median_s": f"{np.median(ok):.2f}" if len(ok) else "",
                     "final_pos_err_median_cm": f"{np.median(s['final_pos']):.2f}",
                     "final_yaw_err_median_deg": f"{np.median(s['final_yaw']):.2f}",
                     "pusher_path_median_cm": f"{100 * np.median(s['path']):.1f}",
                     **{f"success_{1000 * tol:g}mm": f"{v:.2f}" for tol, v in zip(args.tols, s["tols"])}})
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print("  " + ", ".join(f"{k} {v}" for k, v in row.items()))
    seed_sets = {name: tuple(stats[name]["seeds"]) for name in names}
    if len(set(seed_sets.values())) > 1:
        print("note: runs use different seeds, so the comparison is not paired:", seed_sets)
    print(f"saved {os.path.join(out, 'summary.csv')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", required=True, help='"name=run_dir" (repeat for each method)')
    p.add_argument("--out", default=None, help="output folder (default: logs/results_<timestamp>)")
    p.add_argument("--stop-tol", type=float, default=0.005, help="goal position tolerance the runs stopped at (m)")
    p.add_argument("--angle-tol", type=float, default=0.05, help="goal angle tolerance (rad)")
    p.add_argument("--tols", type=float, nargs="+", default=[0.005, 0.0075, 0.01, 0.015, 0.02],
                   help="position tolerances for the success curve (m); values below --stop-tol are not measurable")
    p.add_argument("--dt", type=float, default=0.02, help="time per logged step (s)")
    p.add_argument("--duration", type=float, default=10.0, help="episode length (s)")
    main(p.parse_args())
