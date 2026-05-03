"""
Planner benchmark runner.

Runs MPPI, DWA, and A*+Local on a fixed set of BARN worlds and saves detailed
per-episode metrics.  Results are written after every episode so a crash only
loses the episode in progress.

Usage examples
──────────────
    # Standard 50-world eval set, all three planners, jackal robot
    python3 -m benchmark.run_benchmark --robot jackal

    # Same 50 random worlds for every planner (reproducible via --seed)
    python3 -m benchmark.run_benchmark --robot jackal --worlds random --seed 42

    # Only MPPI and DWA, moborobo
    python3 -m benchmark.run_benchmark --robot moborobo --planners mppi dwa

    # Explicit world list
    python3 -m benchmark.run_benchmark --world-list 0 6 12 18 24 30

Requirements
────────────
    source /opt/ros/noetic/setup.bash
    source ~/barn_lab/jackal_ws/devel/setup.bash   # (+ moborobo_ws for moborobo)
"""

import argparse
import datetime
import os
import signal
import sys
import traceback

import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── Robot must be set BEFORE importing any controller or config that reads
#    SimConfig.ROBOT at class-definition time (MPPIConfig, DWAConfig, etc.)
from configs.config import SimConfig, WorldConfig
_early_parser = argparse.ArgumentParser(add_help=False)
_early_parser.add_argument("--robot", default=SimConfig.ROBOT)
_early_args, _ = _early_parser.parse_known_args()
SimConfig.ROBOT = _early_args.robot

from benchmark.benchmark_env import BenchmarkEnv
from benchmark.metrics import EpisodeResult, print_summary, write_result
from rl.controllers.mppi import MPPIController
from rl.controllers.dwa import DWAController
from rl.controllers.astar_local import AStarLocalController
from rl.controllers.online_planner import OnlineGlobalPlanner


# ── Planner factory ────────────────────────────────────────────────────────────

def make_controller(planner_name: str, goal_x: float, goal_y: float, world_idx: int):
    if planner_name == "mppi":
        return OnlineGlobalPlanner(
            MPPIController(goal_x=goal_x, goal_y=goal_y), goal_x, goal_y
        )
    if planner_name == "dwa":
        return OnlineGlobalPlanner(
            DWAController(goal_x=goal_x, goal_y=goal_y), goal_x, goal_y
        )
    if planner_name == "astar_local":
        return AStarLocalController(goal_x=goal_x, goal_y=goal_y)
    raise ValueError("Unknown planner: %s" % planner_name)


# ── World list helpers ─────────────────────────────────────────────────────────

def build_world_list(args) -> list:
    if args.world_list:
        return sorted(args.world_list)
    if args.worlds == "eval":
        return list(WorldConfig.eval_worlds())
    # random: sample --count worlds without replacement from the available range
    rng = np.random.default_rng(args.seed)
    return sorted(rng.choice(WorldConfig.world_count(), size=args.count, replace=False).tolist())


# ── Output directory ───────────────────────────────────────────────────────────

def make_out_dir(base: str) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d  = os.path.join(base, "benchmark_%s" % ts)
    os.makedirs(d, exist_ok=True)
    return d


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark MPPI / DWA / A*+Local on BARN worlds"
    )
    parser.add_argument(
        "--robot", default=SimConfig.ROBOT, choices=["jackal", "moborobo"],
        help="Robot platform (default: %(default)s)",
    )
    parser.add_argument(
        "--planners", nargs="+",
        default=["mppi", "dwa", "astar_local"],
        choices=["mppi", "dwa", "astar_local"],
        help="Planners to benchmark (default: all three)",
    )
    parser.add_argument(
        "--worlds", default="eval", choices=["eval", "random"],
        help="'eval' uses WorldConfig.EVAL_WORLDS (50 standard worlds); "
             "'random' samples --count worlds with --seed (default: eval)",
    )
    parser.add_argument(
        "--count", type=int, default=50,
        help="Number of random worlds when --worlds random (default: 50)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for world selection (default: 42)",
    )
    parser.add_argument(
        "--world-list", type=int, nargs="+", metavar="IDX",
        help="Explicit world indices (overrides --worlds)",
    )
    parser.add_argument(
        "--timeout", type=float, default=120.0,
        help="Episode timeout in seconds of sim-time (default: 120)",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output directory (default: results/benchmark/benchmark_TIMESTAMP/)",
    )
    args = parser.parse_args()

    SimConfig.ROBOT = args.robot

    # ── Output directory ────────────────────────────────────────────────────────
    if args.out is None:
        base_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "results", "benchmark"
        )
    else:
        base_dir = args.out
    out_dir = make_out_dir(base_dir)
    csv_path = os.path.join(out_dir, "episodes.csv")
    print("Results → %s" % out_dir)

    # ── World list ──────────────────────────────────────────────────────────────
    world_list = build_world_list(args)
    worlds_file = os.path.join(out_dir, "worlds.txt")
    with open(worlds_file, "w") as f:
        f.write("# world indices used in this benchmark run\n")
        f.write(" ".join(str(w) for w in world_list) + "\n")
    print("Worlds  : %d worlds — %s" % (len(world_list), world_list[:8]),
          "..." if len(world_list) > 8 else "")
    print("Planners: %s" % ", ".join(args.planners))
    print("Timeout : %.0f s / episode" % args.timeout)
    print("Robot   : %s" % args.robot)
    print()

    # ── Signal handler for clean shutdown ───────────────────────────────────────
    _stop    = [False]
    _env_ref = [None]
    _sigcount = [0]

    def _shutdown(sig, frame):
        _sigcount[0] += 1
        if _sigcount[0] == 1:
            print("\n[benchmark] Ctrl+C — finishing current episode then stopping.")
            print("            Press Ctrl+C again to force-quit immediately.")
            _stop[0] = True
            if _env_ref[0] is not None:
                _env_ref[0].close()
        else:
            print("\n[benchmark] Force quit.")
            sys.exit(1)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Benchmark loop (world-first: reuse Gazebo for all planners on same world)
    env = BenchmarkEnv(robot=args.robot, gui=False)
    _env_ref[0] = env

    goal_x = float(WorldConfig.STATIC_INIT_POSITION[0] + WorldConfig.STATIC_GOAL_OFFSET[0])
    goal_y = float(WorldConfig.STATIC_INIT_POSITION[1] + WorldConfig.STATIC_GOAL_OFFSET[1])

    all_results  = []
    n_worlds     = len(world_list)
    n_planners   = len(args.planners)
    world_num    = 0

    try:
        for world_idx in world_list:
            if _stop[0]:
                break

            world_num += 1
            print("═" * 70)
            print("World %d  [%d / %d worlds]" % (world_idx, world_num, n_worlds))
            print("═" * 70)

            for p_num, planner_name in enumerate(args.planners, 1):
                if _stop[0]:
                    break

                print("  Planner %d/%d: %s" % (p_num, n_planners, planner_name))

                try:
                    controller = make_controller(planner_name, goal_x, goal_y, world_idx)
                    result = env.run_episode(
                        controller, planner_name, world_idx, timeout=args.timeout
                    )
                except Exception:
                    traceback.print_exc()
                    print("[WARN] Episode failed — recording as timeout with zero metrics.")
                    result = EpisodeResult(
                        planner=planner_name, world_idx=world_idx, outcome="error",
                        success=False, collision_count=0,
                        path_length=0.0, opt_path_length=0.0, time_elapsed=0.0,
                        spl=0.0, nav_metric=0.0,
                        mean_path_deviation=0.0, max_path_deviation=0.0,
                        min_clearance=0.0, final_dist_to_goal=999.0,
                    )

                # ── Save immediately (crash safety) ─────────────────────────────
                write_result(result, csv_path)
                all_results.append(result)

                # ── One-line summary ─────────────────────────────────────────────
                flag = "✓" if result.success else ("✗" if result.outcome == "timeout" else "⚡")
                print(
                    "  %s %-12s  outcome=%-8s  col=%d  t=%.1fs  "
                    "SPL=%.3f  navMetric=%.3f  dev=%.2fm" % (
                        flag, planner_name, result.outcome, result.collision_count,
                        result.time_elapsed, result.spl,
                        result.nav_metric, result.mean_path_deviation,
                    )
                )

    finally:
        env.close()

    # ── Summary table ───────────────────────────────────────────────────────────
    if all_results:
        print_summary(all_results)
        summary_path = os.path.join(out_dir, "summary.txt")
        _write_summary_file(all_results, summary_path, args)
        print("Full CSV  → %s" % csv_path)
        print("Summary   → %s" % summary_path)


def _write_summary_file(results, path: str, args) -> None:
    from collections import defaultdict
    by_planner = defaultdict(list)
    for r in results:
        by_planner[r.planner].append(r)

    with open(path, "w") as f:
        def w(line=""): f.write(line + "\n")
        w("BARN Planner Benchmark — Summary")
        w("Robot   : %s" % args.robot)
        w("Timeout : %.0f s" % args.timeout)
        w("Worlds  : %d" % len(set(r.world_idx for r in results)))
        w()
        header = (
            f"{'Planner':<16} {'N':>4} {'Success%':>9} "
            f"{'Collisions':>11} {'SPL':>7} {'NavMetric':>10} "
            f"{'PathDev(m)':>11} {'MinClr(m)':>10}"
        )
        w(header)
        w("─" * 80)
        for planner in sorted(by_planner):
            res = by_planner[planner]
            n = len(res)
            succ = 100.0 * sum(r.success for r in res) / n
            coll = sum(r.collision_count for r in res) / n
            spl  = sum(r.spl for r in res) / n
            nav  = sum(r.nav_metric for r in res) / n
            dev  = sum(r.mean_path_deviation for r in res) / n
            valid_clr = [r.min_clearance for r in res if r.min_clearance > 0]
            clr  = sum(valid_clr) / len(valid_clr) if valid_clr else 0.0
            w(
                f"{planner:<16} {n:>4} {succ:>9.1f} "
                f"{coll:>11.2f} {spl:>7.4f} {nav:>10.4f} "
                f"{dev:>11.4f} {clr:>10.4f}"
            )


if __name__ == "__main__":
    main()
