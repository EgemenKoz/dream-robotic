"""
Checks for multi-robot training (rl/env/multi_barn_env.py).

    # Is the light lidar profile equivalent to the stock one at the /scan level?
    python3 tools/multi_probe.py compare --worlds 0 30

    # Can Gazebo feed N robots a 10 Hz scan each, and what does a step cost?
    python3 tools/multi_probe.py capacity --n-robots 4

    # Do the robots really ignore each other, and do collisions still register?
    python3 tools/multi_probe.py ghosts

Only one Gazebo stack can run on this machine at a time, so this refuses to
start while a training script is alive.
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from configs.config import SimConfig, WorldConfig

SimConfig.ROBOT = "moborobo"

import numpy as np
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, SetModelState

from configs.rl_config import RLEnvConfig
from rl.env.multi_barn_env import MultiRobotBarnVecEnv

STOCK_PROFILE = {
    "lidar_v_samples": 900, "lidar_v_min": -2.7, "lidar_v_max": 2.7,
    "lidar_max_range": 150.0,
}


def _training_running() -> bool:
    out = subprocess.run(["pgrep", "-f", r"train_(sac|ppo)\.py"],
                         stdout=subprocess.PIPE).stdout.decode().split()
    return any(int(p) != os.getpid() for p in out)


def _place(model: str, x: float, y: float, yaw: float):
    """Move a model in the plane, keeping its height so it does not have to
    be pushed back out of the ground."""
    z = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)(model, "world").pose.position.z
    state = ModelState()
    state.model_name = model
    state.pose.position.x, state.pose.position.y, state.pose.position.z = x, y, z
    state.pose.orientation.z, state.pose.orientation.w = np.sin(yaw / 2), np.cos(yaw / 2)
    state.reference_frame = "world"
    rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)(state)


def _scan_at(venv, pose, n_scans: int = 5) -> np.ndarray:
    """Per-ray minimum over a few scans taken with robot 0 standing at `pose`."""
    env = venv.envs[0]
    env._gazebo_sim.pause()
    _place(env._model_name, *pose)
    scans = _fresh_scans(env, n_scans)
    out = np.full(scans.shape[1], np.nan, dtype=np.float32)
    seen = np.isfinite(scans).any(axis=0)
    out[seen] = np.nanmin(scans[:, seen], axis=0)
    return out


def _fresh_scans(env, n: int) -> np.ndarray:
    """(n, rays) consecutive scans of one robot, all taken from now on.
    Runs the simulation while it collects them and leaves it paused."""
    sim = env._gazebo_sim
    sim.pause()
    stamp = rospy.get_time()
    env._pub_cmd(0.0, 0.0)
    sim.unpause()
    scans = []
    for _ in range(n):
        env._wait_for_scan(after_stamp=stamp)
        with env._scan_lock:
            scans.append(env._latest_scan.copy())
            stamp = env._latest_scan_stamp
    sim.pause()
    return np.stack(scans)


def _poses_along_path(venv, n: int):
    """Poses spread along the reference path, heading along it."""
    path = venv.envs[0]._ref_path
    idx = np.linspace(0, len(path) - 2, n).astype(int)
    return [(path[i][0], path[i][1],
             float(np.arctan2(path[i + 1][1] - path[i][1], path[i + 1][0] - path[i][0])))
            for i in idx]


def _sectors(scan: np.ndarray) -> np.ndarray:
    """Clip and min-pool raw scans the way the observation does."""
    clip = RLEnvConfig.SCAN_CLIP_RANGE
    clipped = np.where(np.isfinite(scan), np.minimum(scan, clip), clip)
    edges = np.linspace(0, scan.shape[1], RLEnvConfig.NUM_SCAN_RAYS + 1).astype(int)
    return np.minimum.reduceat(clipped, edges[:-1], axis=1)


def _report(label: str, ref: np.ndarray, other: np.ndarray) -> float:
    """Print how `other` differs from `ref`; returns the near-field p95."""
    a_s, b_s = _sectors(ref), _sectors(other)
    d_sec = np.abs(a_s - b_s)
    near = a_s < 2.0
    missed = near & (b_s > a_s + 0.3)
    phantom = (b_s < 2.0) & (a_s > b_s + 0.3)
    print("  %-15s returns %.1f%% vs %.1f%% | sector |diff| median %.3f p95 %.3f max %.3f"
          % (label, 100 * np.isfinite(ref).mean(), 100 * np.isfinite(other).mean(),
             np.median(d_sec), np.percentile(d_sec, 95), d_sec.max()))
    p95 = float(np.percentile(d_sec[near], 95)) if near.any() else 0.0
    print("  %-15s under 2 m: p95 %.3f m over %d sectors | missed by >0.3 m: %d | "
          "phantom by >0.3 m: %d" % ("", p95, int(near.sum()), int(missed.sum()),
                                      int(phantom.sum())))
    return p95


def cmd_compare(args):
    """Stock vs light lidar: same poses, compare /scan sector by sector.

    The stock profile is recorded twice. Two launches never agree exactly (a
    ray on the edge of a cylinder lands on it or on the one behind), so the
    stock-vs-stock line is the floor the light profile is judged against.
    """
    light = dict(RLEnvConfig.MULTI_LIDAR_PROFILE)
    if args.v_samples:
        light["lidar_v_samples"] = args.v_samples
    passes = (("stock", STOCK_PROFILE), ("stock again", STOCK_PROFILE), ("light", light))
    # One env for the whole comparison: the ROS master has to outlive the
    # Gazebo restarts, or this process's subscriptions are lost with it.
    venv = MultiRobotBarnVecEnv(1, args.worlds, episodes_per_world=10 ** 9,
                                log_dir=args.log_dir, lidar_profile=STOCK_PROFILE)
    all_scans = []
    try:
        for world in args.worlds:
            scans = {}
            for name, profile in passes:
                venv._lidar_profile = dict(profile)
                venv._launch_world(world)
                venv._reset_robots([0])
                poses = _poses_along_path(venv, args.poses)
                scans[name] = np.stack([_scan_at(venv, p) for p in poses])
            all_scans.append((world, scans))
            venv._cycler.advance()
    finally:
        venv.close()

    if args.save:
        np.savez(args.save, **{"w%d_%s" % (w, n.replace(" ", "_")): v
                               for w, sc in all_scans for n, v in sc.items()})
    print("\nlight = %s" % light)
    worst = floor = 0.0
    for world, scans in all_scans:
        print("\nworld %d, %d poses" % (world, len(scans["stock"])))
        floor = max(floor, _report("stock again", scans["stock"], scans["stock again"]))
        worst = max(worst, _report("light", scans["stock"], scans["light"]))
    print("\nNear-field p95 sector difference: light %.3f m, stock-vs-stock floor %.3f m"
          % (worst, floor))


def cmd_capacity(args):
    """Scan rate each robot gets, and what one vectorised step costs."""
    profile = STOCK_PROFILE if args.profile == "stock" else None
    venv = MultiRobotBarnVecEnv(args.n_robots, [args.world], log_dir=args.log_dir,
                                lidar_profile=profile)
    try:
        venv.reset()
        sim = venv.envs[0]._gazebo_sim

        def free_run(wall=6.0):
            sim.unpause()
            time.sleep(0.5)
            s0, w0 = rospy.get_time(), time.time()
            c0 = [_count(e) for e in venv.envs]
            time.sleep(wall)
            s1, w1 = rospy.get_time(), time.time()
            c1 = [_count(e) for e in venv.envs]
            sim.pause()
            ds = max(s1 - s0, 1e-9)
            return ds / (w1 - w0), [(b - a) / ds for a, b in zip(c0, c1)]

        counts = {e._ns: 0 for e in venv.envs}

        def _count(e):
            return counts[e._ns]

        from sensor_msgs.msg import LaserScan
        for e in venv.envs:
            rospy.Subscriber(e._scan_topic, LaserScan,
                             lambda m, ns=e._ns: counts.__setitem__(ns, counts[ns] + 1),
                             queue_size=1)

        rtf, hz = free_run()
        print("\n%d robots, %s lidar" % (args.n_robots, args.profile))
        print("  free-run            : RTF %.2f | scan Hz (sim) per robot: %s"
              % (rtf, " ".join("%.1f" % h for h in hz)))

        venv.reset()
        fwd = np.tile(np.array([0.5 * venv.action_space.high[0], 0.0], np.float32),
                      (args.n_robots, 1))
        walls, ages, sims, stale = [], [], [], []
        for _ in range(args.steps):
            w0, s0 = time.time(), rospy.get_time()
            before = [e._latest_scan_stamp for e in venv.envs]
            venv.step(fwd)
            walls.append(time.time() - w0)
            now = rospy.get_time()
            sims.append(now - s0)
            ages.append(max(now - e._latest_scan_stamp for e in venv.envs))
            stale.append(np.mean([e._latest_scan_stamp == t for e, t in zip(venv.envs, before)]))
        print("  vec step            : wall %.3f s -> %.1f transitions/s (env only, no SAC update)"
              % (np.mean(walls[5:]), args.n_robots / np.mean(walls[5:])))
        print("  sim time per step   : %.3f s mean (min %.3f, max %.3f) | steps without a new scan: %.1f%%"
              % (np.mean(sims[5:]), np.min(sims[5:]), np.max(sims[5:]), 100 * np.mean(stale[5:])))
        print("  oldest scan at obs  : %.3f s mean, %.3f s max" % (np.mean(ages[5:]), np.max(ages[5:])))

        subprocess.call(["gz", "physics", "-u", "0"])
        rtf, hz = free_run()
        print("  unthrottled         : RTF %.2f | scan Hz (sim) per robot: %s"
              % (rtf, " ".join("%.1f" % h for h in hz)))
    finally:
        venv.close()


def cmd_ghosts(args):
    """Two robots on the same spot must not disturb each other; a robot driven
    into a cylinder must register a collision while the other does not."""
    venv = MultiRobotBarnVecEnv(2, [args.world], log_dir=args.log_dir)
    try:
        venv.reset()
        e0, e1 = venv.envs
        ix, iy, iyaw = WorldConfig.STATIC_INIT_POSITION

        def pose(e):
            p = e._gazebo_sim.get_model_state().pose.position
            return np.array([p.x, p.y, p.z])

        # 1. Spawned on the same pose: if they collided they would have been
        #    thrown apart by now.
        p0, p1 = pose(e0), pose(e1)
        print("\n1. Both at the start pose after reset:")
        print("   r0 %s   r1 %s   apart %.3f m" % (np.round(p0, 3), np.round(p1, 3),
                                                   np.linalg.norm(p0[:2] - p1[:2])))
        ok1 = np.linalg.norm(p0[:2] - p1[:2]) < 0.05 and abs(p0[2] - p1[2]) < 0.02

        # 2. Drive r0 forward through r1, which stays put.
        act = np.array([[venv.action_space.high[0], 0.0], [0.0, 0.0]], np.float32)
        for _ in range(40):
            _, _, dones, infos = venv.step(act)
            if dones.any():
                break
        p0b, p1b = pose(e0), pose(e1)
        print("2. r0 driven 40 steps at full speed, r1 commanded to stand:")
        print("   r0 moved %.3f m, r1 moved %.3f m, outcomes %s"
              % (np.linalg.norm(p0b[:2] - p0[:2]), np.linalg.norm(p1b[:2] - p1[:2]),
                 [i["outcome"] for i in infos]))
        ok2 = np.linalg.norm(p0b[:2] - p0[:2]) > 0.5 and np.linalg.norm(p1b[:2] - p1[:2]) < 0.03

        # 3. Lidar: r1's own view from the start pose, first with r0 standing
        #    on the same spot (inside r1's blind ring, so not in view whatever
        #    it is made of), then with r0 parked 1 m ahead. A ray on the edge
        #    of a cylinder flickers between it and the one behind, so compare
        #    the median with r0 ahead against the nearest reading without it.
        venv.reset()
        before = _fresh_scans(e1, 8)
        e0._gazebo_sim.pause()
        _place(e0._model_name, ix, iy + 1.0, iyaw)         # r0 now 1 m ahead of r1
        after = _fresh_scans(e1, 8)
        n = before.shape[1]
        ahead = np.arange(n)[np.abs(np.arange(n) - n // 2) < n // 12]   # +-30 deg
        b = np.min(np.where(np.isfinite(before[:, ahead]), before[:, ahead], 10.0), axis=0)
        s = np.median(np.where(np.isfinite(after[:, ahead]), after[:, ahead], 10.0), axis=0)
        closer = int(np.sum(s < b - 0.3))
        print("3. r0 parked 1 m in front of r1: rays in r1's forward 60 deg that got")
        print("   shorter than without it by >0.3 m: %d of %d (nearest return ahead %.2f m, was %.2f m)"
              % (closer, len(ahead), float(np.min(s)), float(np.min(b))))
        ok3 = closer == 0

        # 4. Collision: put r0 against the nearest cylinder.
        cyl = venv._cylinders
        nearest = cyl[np.argmin(np.hypot(cyl[:, 0] - ix, cyl[:, 1] - iy))]
        hit = venv._collides(nearest[0] - 0.40, nearest[1], 0.0)
        clear = venv._collides(ix, iy, iyaw)
        print("4. Geometric collision: nose on a cylinder -> %s, start pose -> %s  (%d cylinders)"
              % (hit, clear, len(cyl)))
        ok4 = hit and not clear

        print("\n%s" % ("ALL GHOST CHECKS PASSED" if all((ok1, ok2, ok3, ok4))
                        else "FAILED: %s" % [n for n, ok in zip(
                            ("same-pose", "drive-through", "lidar", "collision"),
                            (ok1, ok2, ok3, ok4)) if not ok]))
    finally:
        venv.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-dir", default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("compare", help="stock vs light lidar at the /scan level")
    p.add_argument("--worlds", type=int, nargs="+", default=[0])
    p.add_argument("--poses", type=int, default=8)
    p.add_argument("--v-samples", type=int, default=None,
                   help="override the light profile's vertical sample count")
    p.add_argument("--save", default=None, metavar="PATH",
                   help="save the raw scans to an .npz file")
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("capacity", help="scan rate and step cost with N robots")
    p.add_argument("--n-robots", type=int, default=4)
    p.add_argument("--world", type=int, default=0)
    p.add_argument("--profile", choices=["light", "stock"], default="light")
    p.add_argument("--steps", type=int, default=60)
    p.set_defaults(fn=cmd_capacity)

    p = sub.add_parser("ghosts", help="robots ignore each other, obstacles do not")
    p.add_argument("--world", type=int, default=0)
    p.set_defaults(fn=cmd_ghosts)

    args = parser.parse_args()
    if _training_running():
        sys.exit("A training script is running; this would disturb its Gazebo. Aborting.")
    args.log_dir = args.log_dir or tempfile.mkdtemp(prefix="multi_probe_")
    args.fn(args)


if __name__ == "__main__":
    main()
