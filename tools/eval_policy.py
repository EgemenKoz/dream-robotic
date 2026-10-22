#!/usr/bin/env python3
"""
Evaluate a trained SAC policy on BARN worlds, in the same multi-robot env it
was trained in (same observations, rewards, collision check), with learning
off and deterministic actions.

Every robot runs one episode per world from the start pose. With a
deterministic policy the robots are near-copies (only simulator noise
differs), so read the per-world result as "solved / not solved" rather than
as four independent trials.

    ~/barn_lab/nav_challenge/bin/python -u tools/eval_policy.py \
        --model results/rl/runs/multi_v6e/model_final.zip

Results go to results/rl/eval/<run>_<checkpoint>/results.csv (one row per
robot per world) and summary.txt. Re-running the same command skips worlds
already in results.csv, so an interrupted evaluation picks up where it
stopped. Do not run it while a training run is using Gazebo.
"""

import argparse
import collections
import csv
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from stable_baselines3 import SAC                     # noqa: E402

import rl.models.critic  # noqa: E402,F401  (LayerNormSACPolicy, needed to load)
from rl.curriculum import WorldCycler, difficulty_tiers   # noqa: E402
from rl.env.multi_barn_env import MultiRobotBarnVecEnv    # noqa: E402

FIELDS = ["world_idx", "tier", "robot", "outcome", "steps",
          "episode_reward", "dist_to_goal"]
TIERS = ("easy", "medium", "hard")


def parse_worlds(spec, all_worlds):
    if spec == "all":
        return list(all_worlds)
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def default_out_dir(model_path):
    run = os.path.basename(os.path.dirname(os.path.abspath(model_path)))
    ckpt = os.path.splitext(os.path.basename(model_path))[0].replace("model_", "")
    return os.path.join(REPO, "results", "rl", "eval", "%s_%s" % (run, ckpt))


def run_world(env, model, world, deterministic):
    """One episode per robot on `world`; returns the first result of each."""
    env._cycler = WorldCycler([world], episodes_per_world=10 ** 9, shuffle=False)
    env._launch_world(world)
    obs = env._stack(env._reset_robots(range(env.num_envs)))
    first = [None] * env.num_envs
    while any(r is None for r in first):
        actions, _ = model.predict(obs, deterministic=deterministic)
        obs, _, dones, infos = env.step(actions)
        for i, done in enumerate(dones):
            if done and first[i] is None:
                first[i] = infos[i]
        # Robots that finished keep driving a new episode until the rest are
        # done; those extra episodes are not counted.
    return first


def summarize(rows):
    lines = []

    def block(name, sub):
        n = len(sub)
        if not n:
            return
        c = collections.Counter(r["outcome"] for r in sub)
        worlds = {r["world_idx"] for r in sub}
        solved = sum(
            1 for w in worlds
            if all(r["outcome"] == "success" for r in sub if r["world_idx"] == w))
        ok_steps = [int(r["steps"]) for r in sub if r["outcome"] == "success"]
        mean_t = (sum(ok_steps) / len(ok_steps) * 0.1) if ok_steps else float("nan")
        lines.append(
            "%-7s worlds=%3d  episodes=%3d  success=%5.1f%%  collision=%5.1f%%  "
            "stuck=%5.1f%%  timeout=%5.1f%%  all-robots-solved=%3d/%-3d  "
            "mean success time=%.1fs"
            % (name, len(worlds), n, 100 * c["success"] / n, 100 * c["collision"] / n,
               100 * c["stuck"] / n, 100 * c["timeout"] / n, solved, len(worlds), mean_t))

    for t in TIERS:
        block(t, [r for r in rows if r["tier"] == t])
    block("total", rows)

    by_world = collections.defaultdict(list)
    for r in rows:
        by_world[int(r["world_idx"])].append(r["outcome"])
    failed = sorted((w, o) for w, o in by_world.items()
                    if o.count("success") < len(o))
    lines.append("")
    lines.append("Worlds not solved by every robot (%d):" % len(failed))
    for w, o in failed:
        tier = next(r["tier"] for r in rows if int(r["world_idx"]) == w)
        lines.append("  world %3d  %-6s  %s" % (w, tier, " ".join(o)))
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="SB3 .zip checkpoint")
    p.add_argument("--robot", default="moborobo")
    p.add_argument("--worlds", default="all",
                   help='"all", or a list like "0,5,10-20"')
    p.add_argument("--n-robots", type=int, default=4)
    p.add_argument("--stochastic", action="store_true",
                   help="sample actions instead of using the policy mean")
    p.add_argument("--out", default=None, help="output directory")
    args = p.parse_args()

    tiers = difficulty_tiers(args.robot)
    tier_of = {w: t for t, ws in tiers.items() for w in ws}
    worlds = parse_worlds(args.worlds, sorted(tier_of))

    out_dir = args.out or default_out_dir(args.model)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "results.csv")
    rows = []
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
    done = {int(r["world_idx"]) for r in rows}
    todo = [w for w in worlds if w not in done]
    print("Model   : %s" % args.model)
    print("Output  : %s" % out_dir)
    print("Worlds  : %d to run (%d already done), %d robots, %s actions"
          % (len(todo), len(done & set(worlds)), args.n_robots,
             "stochastic" if args.stochastic else "deterministic"), flush=True)

    model = SAC.load(args.model, device="auto")
    env = MultiRobotBarnVecEnv(
        n_robots=args.n_robots, worlds=[todo[0] if todo else worlds[0]],
        episodes_per_world=10 ** 9, robot=args.robot, log_dir="", seed=0,
    )
    t_start = time.time()
    try:
        for k, w in enumerate(todo):
            t0 = time.time()
            results = run_world(env, model, w, deterministic=not args.stochastic)
            new = [{
                "world_idx": w, "tier": tier_of.get(w, "?"), "robot": "r%d" % i,
                "outcome": info["outcome"], "steps": info["step"],
                "episode_reward": "%.2f" % info["episode_reward"],
                "dist_to_goal": "%.3f" % info["dist_to_goal"],
            } for i, info in enumerate(results)]
            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=FIELDS)
                if write_header:
                    wr.writeheader()
                wr.writerows(new)
            rows.extend({k2: str(v) for k2, v in r.items()} for r in new)
            n_ok = sum(r["outcome"] == "success" for r in new)
            elapsed = time.time() - t_start
            eta = elapsed / (k + 1) * (len(todo) - k - 1)
            print("[%3d/%d] world %3d %-6s %d/%d success  (%s)  %.0fs  eta %.0f min"
                  % (k + 1, len(todo), w, tier_of.get(w, "?"), n_ok, len(new),
                     " ".join(r["outcome"] for r in new), time.time() - t0, eta / 60),
                  flush=True)
    finally:
        env.close()

    rows = [r for r in rows if int(r["world_idx"]) in set(worlds)]
    if rows:
        text = summarize(rows)
        with open(os.path.join(out_dir, "summary.txt"), "w") as f:
            f.write("model: %s\n\n%s\n" % (args.model, text))
        print("\n" + text)


if __name__ == "__main__":
    main()
