"""
Metrics dataclass and CSV persistence for the planner benchmark.
"""

import csv
import dataclasses
import os
from typing import Optional

import numpy as np


@dataclasses.dataclass
class EpisodeResult:
    planner:              str
    world_idx:            int
    outcome:              str    # 'success' | 'timeout'
    success:              bool
    collision_count:      int    # total collision events (never terminates episode)
    path_length:          float  # metres actually travelled
    opt_path_length:      float  # metres of reference A* path
    time_elapsed:         float  # seconds of sim-time
    spl:                  float  # success * opt / max(actual, opt)
    nav_metric:           float  # BARN standard score
    mean_path_deviation:  float  # mean perp distance from reference path (m)
    max_path_deviation:   float  # max  perp distance from reference path (m)
    min_clearance:        float  # minimum lidar reading seen across episode (m)
    final_dist_to_goal:   float  # distance from goal at episode end (m)


def write_result(result: EpisodeResult, csv_path: str) -> None:
    """Append one result row to csv_path, creating headers on first write."""
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dataclasses.asdict(result).keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(dataclasses.asdict(result))


# ── Geometry helpers (also used by BenchmarkEnv) ──────────────────────────────

def point_to_path_dist(pt: np.ndarray, path: np.ndarray) -> float:
    """Minimum perpendicular distance from pt (2,) to the polyline path (N, 2)."""
    if path is None or len(path) == 0:
        return 0.0
    if len(path) == 1:
        return float(np.linalg.norm(pt - path[0]))
    min_d = float("inf")
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        ab = b - a
        ab2 = float(np.dot(ab, ab))
        if ab2 < 1e-12:
            d = float(np.linalg.norm(pt - a))
        else:
            t = float(np.clip(np.dot(pt - a, ab) / ab2, 0.0, 1.0))
            d = float(np.linalg.norm(pt - a - t * ab))
        if d < min_d:
            min_d = d
    return min_d


def path_length(path: np.ndarray) -> float:
    """Total length of a polyline given as (N, 2) array."""
    if path is None or len(path) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def print_summary(results) -> None:
    """Print a per-planner aggregated table to stdout."""
    from collections import defaultdict
    by_planner = defaultdict(list)
    for r in results:
        by_planner[r.planner].append(r)

    SEP = "─" * 80
    print("\n" + SEP)
    print(f"{'BENCHMARK SUMMARY':^80}")
    print(SEP)
    hdr = (
        f"{'Planner':<16} {'N':>4} {'Success':>8} "
        f"{'Collisions':>11} {'SPL':>7} {'NavMetric':>10} "
        f"{'PathDev(m)':>11} {'MinClr(m)':>10}"
    )
    print(hdr)
    print(SEP)
    for planner in sorted(by_planner):
        res = by_planner[planner]
        n   = len(res)
        succ = sum(r.success for r in res) / n
        coll = sum(r.collision_count for r in res) / n
        spl  = sum(r.spl for r in res) / n
        nav  = sum(r.nav_metric for r in res) / n
        dev  = sum(r.mean_path_deviation for r in res) / n
        valid_clr = [r.min_clearance for r in res if r.min_clearance > 0]
        clr  = sum(valid_clr) / len(valid_clr) if valid_clr else 0.0
        print(
            f"{planner:<16} {n:>4} {succ:>8.1%} "
            f"{coll:>11.2f} {spl:>7.4f} {nav:>10.4f} "
            f"{dev:>11.4f} {clr:>10.4f}"
        )
    print(SEP)
