"""
Visualise / analyse a planner benchmark run.

Reads a `results/benchmark/benchmark_*/episodes.csv` file and produces:

  ┌────────────────────────────────────────────────────────────────┐
  │  STDOUT                                                         │
  │  ──────                                                         │
  │  • Per-planner aggregate table (success%, collisions, SPL, …)   │
  │  • Per-episode breakdown with an explicit failure REASON for    │
  │    every non-success run (collision / timeout / error / stuck   │
  │    short of goal / planner crash).                              │
  │  • Per-world comparison: which planners solved each world.      │
  │                                                                 │
  │  PLOTS                                                          │
  │  ─────                                                          │
  │  Saved to <run_dir>/plots/                                      │
  │  • outcome_by_planner.png   stacked bar success/timeout/coll    │
  │  • spl_navmetric.png        boxplot of SPL & NavMetric          │
  │  • time_vs_success.png      time-to-goal scatter (success only) │
  │  • per_world_heatmap.png    planner × world success matrix      │
  │  • failure_reasons.png      stacked bar of failure causes       │
  └────────────────────────────────────────────────────────────────┘

Usage
─────
    # Visualise the latest run
    python3 -m benchmark.visualize_results

    # Visualise a specific run
    python3 -m benchmark.visualize_results --run results/benchmark/benchmark_20260502_154711

    # Skip plotting (text-only)
    python3 -m benchmark.visualize_results --no-plots

    # Print only the per-episode breakdown
    python3 -m benchmark.visualize_results --episodes-only
"""

import argparse
import os
import sys
from collections import defaultdict
from typing import List, Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from configs.config import NavConfig


# ── Failure reasoning ──────────────────────────────────────────────────────────

def classify_run(row: pd.Series) -> str:
    """
    Return a one-line human readable reason for the row's outcome.

    Success rows return "OK".  Failures get a specific cause:
      • collision        — robot hit an obstacle
      • timeout-near     — timed out but within 2 m of goal (planner oscillating)
      • timeout-stuck    — timed out and barely moved (<1 m of path)
      • timeout-far      — timed out, made progress but still far from goal
      • error/crash      — Python exception in run_episode
      • unknown          — outcome string we don't recognise
    """
    outcome = str(row.get("outcome", "")).lower()
    final_dist = float(row.get("final_dist_to_goal", 999.0))
    path_len   = float(row.get("path_length", 0.0))
    collisions = int(row.get("collision_count", 0))

    if outcome == "success":
        return "OK"
    if outcome == "collision":
        return "collision (%d hit%s; final dist %.2f m)" % (
            collisions, "" if collisions == 1 else "s", final_dist
        )
    if outcome == "error":
        return "error/crash (no metrics collected — see log file)"
    if outcome == "timeout":
        if final_dist <= NavConfig.GOAL_TOLERANCE * 2:
            return "timeout-near goal (%.2f m away; oscillating)" % final_dist
        if path_len < 1.0:
            return "timeout-stuck (only %.2f m travelled)" % path_len
        return "timeout-far (%.2f m from goal after %.0f m of travel)" % (
            final_dist, path_len
        )
    return "unknown outcome '%s'" % outcome


# ── Run-dir resolution ─────────────────────────────────────────────────────────

def find_latest_run(base: str) -> Optional[str]:
    if not os.path.isdir(base):
        return None
    runs = [d for d in os.listdir(base) if d.startswith("benchmark_")]
    if not runs:
        return None
    runs.sort()
    return os.path.join(base, runs[-1])


def load_episodes(run_dir: str) -> pd.DataFrame:
    csv_path = os.path.join(run_dir, "episodes.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError("No episodes.csv in %s" % run_dir)
    df = pd.read_csv(csv_path)
    df["reason"] = df.apply(classify_run, axis=1)
    df["success"] = df["success"].astype(bool)
    return df


# ── Text reports ───────────────────────────────────────────────────────────────

def print_aggregate_table(df: pd.DataFrame) -> None:
    SEP = "─" * 95
    print("\n" + SEP)
    print(f"{'PLANNER AGGREGATE':^95}")
    print(SEP)
    hdr = (
        f"{'Planner':<14} {'N':>4} {'Success':>8} {'Coll/ep':>8} "
        f"{'Timeout':>8} {'Error':>6} {'SPL':>7} {'NavMetric':>10} "
        f"{'PathDev':>9} {'MinClr':>8}"
    )
    print(hdr)
    print(SEP)
    for planner, sub in sorted(df.groupby("planner")):
        n = len(sub)
        succ_rate = sub["success"].mean()
        coll      = sub["collision_count"].mean()
        # Outcome breakdown
        outcomes = sub["outcome"].value_counts(normalize=True)
        timeout_rate = outcomes.get("timeout", 0.0)
        error_rate   = outcomes.get("error", 0.0)
        # Successful-only averages where it makes sense
        succ_only = sub[sub["success"]]
        spl = sub["spl"].mean()
        nav = sub["nav_metric"].mean()
        dev = sub["mean_path_deviation"].mean()
        valid_clr = sub.loc[sub["min_clearance"] > 0, "min_clearance"]
        clr = valid_clr.mean() if len(valid_clr) else 0.0
        print(
            f"{planner:<14} {n:>4} {succ_rate:>8.1%} {coll:>8.2f} "
            f"{timeout_rate:>8.1%} {error_rate:>6.1%} {spl:>7.3f} {nav:>10.4f} "
            f"{dev:>9.3f} {clr:>8.3f}"
        )
    print(SEP)


def print_episodes_breakdown(df: pd.DataFrame) -> None:
    """One line per episode with explicit success/failure reason."""
    SEP = "─" * 95
    print("\n" + SEP)
    print(f"{'PER-EPISODE BREAKDOWN':^95}")
    print(SEP)
    hdr = (
        f"{'Planner':<14} {'World':>5} {'Result':>7} "
        f"{'Time(s)':>8} {'Dist(m)':>8}  Reason"
    )
    print(hdr)
    print(SEP)
    for _, row in df.sort_values(["world_idx", "planner"]).iterrows():
        flag = "✓ PASS" if row["success"] else " FAIL "
        print(
            f"{row['planner']:<14} {int(row['world_idx']):>5} {flag:>7} "
            f"{row['time_elapsed']:>8.1f} {row['final_dist_to_goal']:>8.2f}  "
            f"{row['reason']}"
        )
    print(SEP)


def print_per_world_comparison(df: pd.DataFrame) -> None:
    """Which worlds each planner solved (and which were unsolvable for all)."""
    SEP = "─" * 95
    planners = sorted(df["planner"].unique())
    worlds   = sorted(df["world_idx"].unique())

    print("\n" + SEP)
    print(f"{'PER-WORLD RESULTS  (✓=solved  ✗=failed  ·=not run)':^95}")
    print(SEP)
    print("World " + "  ".join(f"{p:<14}" for p in planners))
    print(SEP)
    unsolved_by_all = []
    for w in worlds:
        row = df[df["world_idx"] == w]
        cells = []
        any_solved = False
        for p in planners:
            r = row[row["planner"] == p]
            if r.empty:
                cells.append("·")
            elif bool(r.iloc[0]["success"]):
                cells.append("✓")
                any_solved = True
            else:
                cells.append("✗")
        print(f"{int(w):>5}  " + "  ".join(f"{c:<14}" for c in cells))
        if not any_solved:
            unsolved_by_all.append(int(w))
    print(SEP)
    if unsolved_by_all:
        print("Worlds NO planner solved: %s" % unsolved_by_all)
    print()


# ── Plots ──────────────────────────────────────────────────────────────────────

def _ensure_plot_backend():
    import matplotlib
    matplotlib.use("Agg")
    return matplotlib


def plot_outcomes_by_planner(df: pd.DataFrame, out_path: str) -> None:
    _ensure_plot_backend()
    import matplotlib.pyplot as plt
    planners = sorted(df["planner"].unique())
    counts = pd.crosstab(df["planner"], df["outcome"]).reindex(planners).fillna(0)
    order = [c for c in ["success", "timeout", "collision", "error"] if c in counts.columns]
    counts = counts[order]
    colors = {"success": "#4caf50", "timeout": "#ffb300",
              "collision": "#e53935", "error": "#7e57c2"}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(len(planners))
    for outcome in order:
        vals = counts[outcome].values
        ax.bar(planners, vals, bottom=bottom, label=outcome,
               color=colors.get(outcome, "gray"))
        bottom += vals
    ax.set_ylabel("Episodes")
    ax.set_title("Episode outcomes by planner")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_spl_navmetric(df: pd.DataFrame, out_path: str) -> None:
    _ensure_plot_backend()
    import matplotlib.pyplot as plt
    planners = sorted(df["planner"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, title in zip(axes, ("spl", "nav_metric"), ("SPL", "NavMetric")):
        data = [df.loc[df["planner"] == p, col].values for p in planners]
        ax.boxplot(data, labels=planners, showmeans=True)
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.suptitle("Path quality distribution (higher is better)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_time_vs_success(df: pd.DataFrame, out_path: str) -> None:
    _ensure_plot_backend()
    import matplotlib.pyplot as plt
    planners = sorted(df["planner"].unique())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for p in planners:
        sub = df[df["planner"] == p]
        succ = sub[sub["success"]]
        fail = sub[~sub["success"]]
        ax.scatter(succ["world_idx"], succ["time_elapsed"], s=35,
                   label=f"{p} (✓)", marker="o")
        ax.scatter(fail["world_idx"], fail["time_elapsed"], s=35,
                   label=f"{p} (✗)", marker="x", alpha=0.6)
    ax.set_xlabel("World index")
    ax.set_ylabel("Time elapsed (s)")
    ax.set_title("Time per episode (✓ success, ✗ failure)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_per_world_heatmap(df: pd.DataFrame, out_path: str) -> None:
    _ensure_plot_backend()
    import matplotlib.pyplot as plt
    planners = sorted(df["planner"].unique())
    worlds   = sorted(df["world_idx"].unique())
    # Build matrix: rows=planner, cols=world, vals=success(1)/fail(0)/nan(no-run)
    mat = np.full((len(planners), len(worlds)), np.nan)
    for i, p in enumerate(planners):
        for j, w in enumerate(worlds):
            r = df[(df["planner"] == p) & (df["world_idx"] == w)]
            if not r.empty:
                mat[i, j] = float(bool(r.iloc[0]["success"]))

    fig, ax = plt.subplots(figsize=(max(8, 0.22 * len(worlds)), 1.2 + 0.6 * len(planners)))
    cmap = plt.cm.RdYlGn
    cmap.set_bad(color="#cccccc")
    img = ax.imshow(np.ma.masked_invalid(mat), aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_yticks(range(len(planners)))
    ax.set_yticklabels(planners)
    ax.set_xticks(range(len(worlds)))
    ax.set_xticklabels(worlds, rotation=70, fontsize=7)
    ax.set_xlabel("World index")
    ax.set_title("Per-world result  (green=success, red=fail, grey=not run)")
    fig.colorbar(img, ax=ax, ticks=[0, 1], shrink=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_failure_reasons(df: pd.DataFrame, out_path: str) -> None:
    _ensure_plot_backend()
    import matplotlib.pyplot as plt
    # Group reasons into coarse categories for the plot
    def category(reason: str) -> str:
        if reason == "OK":
            return "success"
        if reason.startswith("collision"):
            return "collision"
        if reason.startswith("timeout-near"):
            return "timeout (near goal)"
        if reason.startswith("timeout-stuck"):
            return "timeout (stuck)"
        if reason.startswith("timeout-far"):
            return "timeout (far)"
        if reason.startswith("error"):
            return "error/crash"
        return "other"

    df = df.copy()
    df["category"] = df["reason"].map(category)
    planners = sorted(df["planner"].unique())
    cats = ["success", "collision", "timeout (near goal)", "timeout (stuck)",
            "timeout (far)", "error/crash", "other"]
    counts = pd.crosstab(df["planner"], df["category"]).reindex(planners).fillna(0)
    for c in cats:
        if c not in counts.columns:
            counts[c] = 0
    counts = counts[cats]
    colors = ["#4caf50", "#e53935", "#ffd54f", "#ff8a65",
              "#fb8c00", "#7e57c2", "#90a4ae"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bottom = np.zeros(len(planners))
    for c, col in zip(cats, colors):
        vals = counts[c].values
        if vals.sum() == 0:
            continue
        ax.bar(planners, vals, bottom=bottom, label=c, color=col)
        bottom += vals
    ax.set_ylabel("Episodes")
    ax.set_title("Outcome breakdown — what went wrong, when it failed")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def make_all_plots(df: pd.DataFrame, run_dir: str) -> str:
    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_outcomes_by_planner(df, os.path.join(plot_dir, "outcome_by_planner.png"))
    plot_spl_navmetric(df,      os.path.join(plot_dir, "spl_navmetric.png"))
    plot_time_vs_success(df,    os.path.join(plot_dir, "time_vs_success.png"))
    plot_per_world_heatmap(df,  os.path.join(plot_dir, "per_world_heatmap.png"))
    plot_failure_reasons(df,    os.path.join(plot_dir, "failure_reasons.png"))
    return plot_dir


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualise & analyse a benchmark run",
    )
    parser.add_argument(
        "--run", metavar="DIR",
        help="Path to a benchmark_TIMESTAMP run directory (default: latest)",
    )
    parser.add_argument(
        "--base", default=os.path.join(_PROJECT_ROOT, "results", "benchmark"),
        help="Base results directory used when --run is omitted",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip generating PNGs (text output only)",
    )
    parser.add_argument(
        "--episodes-only", action="store_true",
        help="Only print the per-episode breakdown",
    )
    parser.add_argument(
        "--export-failures", default=None, metavar="PATH",
        help="Write a CSV of all failing episodes (with reason column) to PATH",
    )
    args = parser.parse_args()

    run_dir = args.run or find_latest_run(args.base)
    if run_dir is None:
        sys.exit("No benchmark run directory found in %s" % args.base)
    if not os.path.isdir(run_dir):
        sys.exit("Not a directory: %s" % run_dir)

    print("Run: %s" % run_dir)
    df = load_episodes(run_dir)
    print("Episodes: %d   Planners: %s   Worlds: %d" % (
        len(df), sorted(df["planner"].unique()), df["world_idx"].nunique(),
    ))

    if args.episodes_only:
        print_episodes_breakdown(df)
        return

    print_aggregate_table(df)
    print_episodes_breakdown(df)
    if df["world_idx"].nunique() > 1 or df["planner"].nunique() > 1:
        print_per_world_comparison(df)

    if args.export_failures:
        failures = df[~df["success"]]
        failures.to_csv(args.export_failures, index=False)
        print("Failures CSV → %s  (%d rows)" % (args.export_failures, len(failures)))

    if not args.no_plots:
        try:
            plot_dir = make_all_plots(df, run_dir)
            print("Plots → %s" % plot_dir)
        except Exception as e:
            print("[WARN] plotting failed: %s" % e)


if __name__ == "__main__":
    main()
