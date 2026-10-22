"""
Timing probe for the BARN Gazebo simulation.

Launches one headless Gazebo through BarnEnv and measures what limits training
throughput: real-time factor, lidar scan rate, how stale the scan is when the
policy observes it, and where the wall time of one env.step() goes.

    python3 tools/sim_probe.py                      # default settings
    python3 tools/sim_probe.py --offload off        # compare without NVIDIA offload
    python3 tools/sim_probe.py --save-scan a.npy    # dump the reset-pose scan

Only one Gazebo stack can run on this machine at a time: BarnEnv kills every
gzserver it finds when it starts, so this refuses to run while a training
script is alive.
"""

import argparse
import os
import subprocess
import sys
import time

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from configs.config import GazeboConfig, SimConfig

_parser = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
_parser.add_argument("--robot", default="moborobo", choices=["jackal", "moborobo"])
_parser.add_argument("--world", type=int, default=0)
_parser.add_argument("--offload", default="auto", choices=["auto", "on", "off"],
                     help="NVIDIA render offload for Gazebo (default: auto)")
_parser.add_argument("--steps", type=int, default=80,
                     help="env.step() calls used for the per-step timing")
_parser.add_argument("--free-run", type=float, default=6.0,
                     help="seconds of wall time per free-running measurement")
_parser.add_argument("--save-scan", default=None, metavar="PATH",
                     help="save the raw /scan ranges at the reset pose to a .npy file")
_parser.add_argument("--log-dir", default=None,
                     help="where Gazebo logs go (default: a temp directory)")
_args = _parser.parse_args()

SimConfig.ROBOT = _args.robot
if _args.offload != "auto":
    GazeboConfig.NVIDIA_OFFLOAD = (_args.offload == "on")

import numpy as np
import rospy
from sensor_msgs.msg import LaserScan

from rl.env.barn_env import BarnEnv


def _training_running() -> bool:
    out = subprocess.run(["pgrep", "-f", r"train_(sac|ppo)\.py"],
                         stdout=subprocess.PIPE).stdout.decode().split()
    return any(int(p) != os.getpid() for p in out)


def _gazebo_on_nvidia() -> str:
    """Does nvidia-smi list a gzserver process?"""
    try:
        table = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, timeout=10).stdout.decode()
    except (OSError, subprocess.TimeoutExpired):
        return "nvidia-smi unavailable"
    rows = [ln for ln in table.splitlines() if "gzserver" in ln]
    if not rows:
        return "NO - gzserver is not on the NVIDIA GPU"
    return "yes (%s)" % rows[0].split()[-2]


def main():
    if _training_running():
        sys.exit("A training script is running; starting a probe would kill its Gazebo. Aborting.")

    import tempfile
    log_dir = _args.log_dir or tempfile.mkdtemp(prefix="sim_probe_")
    env = BarnEnv(robot=_args.robot, world_idx=_args.world, managed=True, log_dir=log_dir)
    seen = {"scan_stamp": None, "n_scan": 0, "n_rays": 0, "cloud_bytes": [], "n_cloud": 0}

    def scan_cb(msg):
        seen["scan_stamp"] = msg.header.stamp.to_sec()
        seen["n_scan"] += 1
        seen["n_rays"] = len(msg.ranges)

    def cloud_cb(msg):
        seen["cloud_bytes"].append(len(msg._buff))
        seen["n_cloud"] += 1

    try:
        env.reset()
        rospy.Subscriber(env._scan_topic, LaserScan, scan_cb, queue_size=1)
        if _args.robot == "moborobo":
            rospy.Subscriber("/rs_points", rospy.AnyMsg, cloud_cb, queue_size=1,
                             buff_size=2 ** 26)
        sim = rospy.get_time
        gz = env._gazebo_sim

        print("\nGazebo on NVIDIA GPU    : %s" % _gazebo_on_nvidia())

        if _args.save_scan:
            ranges, ang_min, ang_inc, _ = env.get_scan_info()
            with env._scan_lock:
                raw = env._latest_scan.copy()
            np.save(_args.save_scan, raw)
            print("Saved reset-pose scan   : %s (%d rays, %d with a return)"
                  % (_args.save_scan, len(raw), int(np.isfinite(raw).sum())))

        def free_run():
            gz.unpause()
            time.sleep(0.5)
            s0, w0, n0, c0 = sim(), time.time(), seen["n_scan"], seen["n_cloud"]
            time.sleep(_args.free_run)
            s1, w1, n1, c1 = sim(), time.time(), seen["n_scan"], seen["n_cloud"]
            gz.pause()
            ds, dw = max(s1 - s0, 1e-9), w1 - w0
            return ds / dw, (n1 - n0) / ds, (n1 - n0) / dw, (c1 - c0) / dw

        r = free_run()
        print("Free-run, default       : RTF %.2f | scan %.1f Hz sim, %.1f Hz wall | cloud %.1f Hz wall" % r)
        if seen["cloud_bytes"]:
            print("  point cloud message   : %.1f MB | scan rays: %d"
                  % (np.mean(seen["cloud_bytes"]) / 1e6, seen["n_rays"]))

        env.reset()
        rows = []
        cmd = np.array([0.5 * env.action_space.high[0], 0.0], dtype=np.float32)
        for _ in range(_args.steps):
            s0, w0 = sim(), time.time()
            _, _, term, trunc, info = env.step(cmd)
            w1, s1 = time.time(), sim()
            rows.append((w1 - w0, s1 - s0, s1 - (seen["scan_stamp"] or s1),
                         info["dist_to_goal"]))
            if term or trunc:
                break
        a = np.array(rows[5:]) if len(rows) > 10 else np.array(rows)
        print("env.step()              : wall %.3f s | sim advanced %.3f s (min %.3f, max %.3f)"
              % (a[:, 0].mean(), a[:, 1].mean(), a[:, 1].min(), a[:, 1].max()))
        print("  scan age at observation: %.3f s mean, %.3f s max" % (a[:, 2].mean(), a[:, 2].max()))
        print("  travelled             : %.4f m/step at %.2f m/s commanded (%s)"
              % ((a[0, 3] - a[-1, 3]) / max(len(a) - 1, 1), cmd[0], info["outcome"]))

        lat = {}
        for name, fn in (("unpause", gz.unpause), ("pause", gz.pause),
                         ("get_model_state", gz.get_model_state)):
            w = time.time()
            for _ in range(20):
                fn()
            lat[name] = (time.time() - w) / 20
        print("  service latency       : " + " | ".join("%s %.1f ms" % (k, v * 1e3)
                                                        for k, v in lat.items()))

        # How fast the simulation could go if it were not throttled to real
        # time. The scan rate in wall-clock Hz is the lidar pipeline's ceiling.
        subprocess.call(["gz", "physics", "-u", "0"])
        r = free_run()
        print("Free-run, unthrottled   : RTF %.2f | scan %.1f Hz sim, %.1f Hz wall | cloud %.1f Hz wall" % r)
        print("  -> lidar pipeline ceiling: %.1f scans/s; robots it can feed at 10 Hz: %d"
              % (r[2], int(r[2] // 10)))
    finally:
        env.close()


if __name__ == "__main__":
    main()
