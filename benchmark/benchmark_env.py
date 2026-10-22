"""
BenchmarkEnv — headless Gazebo runner for planner comparison.

Key differences from BarnEnv (the RL gym env):
  - Not a gymnasium.Env; exposes run_episode() instead of step().
  - Collisions are counted but never terminate the episode.
  - World-first ordering: Gazebo is relaunched only when world_idx changes;
    multiple planners on the same world reuse the same Gazebo process (robot
    is simply teleported back to start between runs).
  - Results are returned as EpisodeResult dataclasses (see benchmark/metrics.py).
"""

import math
import os
import signal
import subprocess
import sys
import threading
import time
from os.path import join
from typing import Optional, Tuple

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

_PROJECT_ROOT = os.path.abspath(join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from configs.config import (
    GazeboConfig, MoboroboConfig, NavConfig, Paths,
    RobotConfig, SimConfig, WorldConfig, apply_env,
)
from gazebo.gazebo_simulation import GazeboSimulation
from benchmark.metrics import EpisodeResult, path_length, point_to_path_dist, write_result

# ── Robot-specific constants ───────────────────────────────────────────────────

_SCAN_TOPIC = {
    "jackal":   "front/scan",
    "moborobo": "/scan",
}
_MODEL_NAME = {
    "jackal":   "jackal",
    "moborobo": MoboroboConfig.MODEL_NAME,
}
_SCAN_MAX_RANGE = {
    "jackal":   10.0,
    "moborobo": MoboroboConfig.LIDAR_MAX_RANGE,
}

_CTRL_DT        = 0.1   # seconds between control updates
_SCAN_TIMEOUT   = 15.0  # seconds to wait for first scan after world launch
_RESET_RETRIES  = 10


class BenchmarkEnv:
    """
    Manages Gazebo and provides run_episode() for any controller that implements
    compute_action(x, y, theta, scan_ranges, angle_min, angle_inc, max_range).
    """

    def __init__(self, robot: str = "jackal", gui: bool = False):
        if robot not in ("jackal", "moborobo"):
            raise ValueError("robot must be 'jackal' or 'moborobo'")

        self.robot  = robot
        self.gui    = gui

        self._scan_topic    = _SCAN_TOPIC[robot]
        self._model_name    = _MODEL_NAME[robot]
        self._scan_max_r    = _SCAN_MAX_RANGE[robot]

        ix, iy, _ = WorldConfig.STATIC_INIT_POSITION
        self._init_pos = np.array([ix, iy], dtype=np.float64)
        gx = ix + WorldConfig.STATIC_GOAL_OFFSET[0]
        gy = iy + WorldConfig.STATIC_GOAL_OFFSET[1]
        self._goal_pos = np.array([gx, gy], dtype=np.float64)

        self._current_world:  Optional[int] = None
        self._roscore_proc:   Optional[subprocess.Popen] = None
        self._gazebo_proc:    Optional[subprocess.Popen] = None
        self._nav_proc:       Optional[subprocess.Popen] = None
        self._gazebo_sim:     Optional[GazeboSimulation]  = None
        self._cmd_pub         = None
        self._ros_ready       = False

        self._scan:          Optional[np.ndarray] = None
        # Same rays as _scan but with NaN preserved where the sensor gave no
        # return — see _scan_cb.
        self._scan_raw:      Optional[np.ndarray] = None
        self._scan_min:      float = -math.pi
        self._scan_inc:      float = 0.0087
        self._scan_lock      = threading.Lock()

        # Start a persistent roscore now. roslaunch will reuse it for every
        # world, so rospy only ever connects to one master for the entire run.
        self._start_roscore()

    # ── roscore lifecycle ──────────────────────────────────────────────────────

    def _start_roscore(self):
        """Start a persistent roscore that outlives individual Gazebo restarts."""
        import socket
        log_dir = os.path.join(Paths.RESULTS_DIR, "benchmark", "logs")
        os.makedirs(log_dir, exist_ok=True)
        rc_log = open(os.path.join(log_dir, "roscore.log"), "w")

        print("    [roscore] Starting roscore...", end="", flush=True)
        self._roscore_proc = subprocess.Popen(
            ["roscore"],
            stdout=rc_log, stderr=rc_log,
        )

        # Poll until port 11311 accepts connections (max 30s).
        deadline = time.time() + 30.0
        while time.time() < deadline:
            try:
                s = socket.create_connection(("localhost", 11311), timeout=1.0)
                s.close()
                break
            except (socket.error, socket.timeout):
                print(".", end="", flush=True)
                time.sleep(0.5)
        else:
            raise TimeoutError("roscore did not start within 30s")
        print(" OK")

        # Connect rospy once — it stays bound to this roscore forever.
        self._connect_ros()

    # ── Main public interface ──────────────────────────────────────────────────

    def run_episode(
        self,
        controller,
        planner_name: str,
        world_idx: int,
        timeout: float = 120.0,
    ) -> EpisodeResult:
        """
        Run one benchmark episode and return an EpisodeResult.

        Gazebo is relaunched only if world_idx has changed since the last call.
        The robot is always teleported back to the start before the episode begins.

        Collisions are recorded in the result but do NOT end the episode.
        """
        if world_idx != self._current_world:
            self._restart_gazebo(world_idx)

        # Unpause here so physics is live during reset and scan-wait.
        # (After each episode _run_loop calls pause(); without this the next
        # planner's _wait_for_scan() would hang forever.)
        self._gazebo_sim.unpause()

        print("    [setup] Resetting robot to start position...")
        self._reset_robot()
        print("    [setup] Waiting for first scan...")
        with self._scan_lock:
            self._scan = None
            self._scan_raw = None
        self._wait_for_scan()
        self._gazebo_sim.get_hard_collision()  # drain any collision from spawn/settle
        print("    [setup] Ready — starting navigation.")

        ref_path    = self._load_ref_path(world_idx)
        opt_len     = path_length(ref_path)

        # Give every controller the final goal once; they manage subgoals internally.
        controller.set_goal(float(self._goal_pos[0]), float(self._goal_pos[1]))

        return self._run_loop(
            controller, planner_name, world_idx, ref_path, opt_len, timeout
        )

    def close(self):
        self._kill_procs()
        if self._roscore_proc is not None:
            try:
                self._roscore_proc.send_signal(signal.SIGINT)
                self._roscore_proc.wait(timeout=5)
            except Exception:
                self._roscore_proc.terminate()
            self._roscore_proc = None

    # ── Episode loop ───────────────────────────────────────────────────────────

    def _run_loop(
        self,
        controller,
        planner_name: str,
        world_idx: int,
        ref_path: Optional[np.ndarray],
        opt_len: float,
        timeout: float,
    ) -> EpisodeResult:
        controller.reset()

        t_start   = rospy.get_time()
        prev_pos  = self._init_pos.copy()

        accu_path  = 0.0
        collisions = 0
        min_clr    = float("inf")
        deviations = []
        outcome    = "timeout"

        while True:
            t_iter = time.time()

            x, y, theta = self._get_pose()
            pos      = np.array([x, y], dtype=np.float64)
            elapsed  = rospy.get_time() - t_start

            # ── Termination checks ──────────────────────────────────────────────
            if self._gazebo_sim.get_hard_collision():
                collisions += 1
                outcome = "collision"
                break

            dist_goal = float(np.linalg.norm(pos - self._goal_pos))
            if dist_goal <= NavConfig.GOAL_TOLERANCE:
                outcome = "success"
                break
            if elapsed >= timeout:
                outcome = "timeout"
                break

            # ── Metric accumulators ─────────────────────────────────────────────
            accu_path += float(np.linalg.norm(pos - prev_pos))
            prev_pos   = pos.copy()

            with self._scan_lock:
                scan = self._scan.copy() if self._scan is not None else None
                scan_raw = (
                    self._scan_raw.copy() if self._scan_raw is not None else None
                )

            if scan_raw is not None:
                valid = np.isfinite(scan_raw) & (scan_raw > 0.01)
                if valid.any():
                    min_clr = min(min_clr, float(scan_raw[valid].min()))

            if ref_path is not None:
                deviations.append(point_to_path_dist(pos, ref_path))

            # ── Control ─────────────────────────────────────────────────────────
            if scan is None:
                scan = np.full(360, self._scan_max_r, dtype=np.float32)

            # Controllers that maintain their own near-field memory (the RL
            # policy) need the NaN-preserving scan; everything else gets the
            # sanitised one it has always been given.
            if getattr(controller, "wants_raw_scan", False) and scan_raw is not None:
                scan_in = scan_raw
            else:
                scan_in = scan

            action = controller.compute_action(
                x, y, theta,
                scan_in, self._scan_min, self._scan_inc, self._scan_max_r,
            )
            self._pub_cmd(float(action[0]), float(action[1]))

            # Live progress line — write directly to stdout to avoid buffering
            # issues from ROS WARN messages that would break \r overwriting.
            sys.stdout.write(
                "\033[2K\r  [%-12s / world %3d] t=%6.1fs  dist=%.2fm  "
                "col=%d  dev=%.2fm  clr=%.2fm" % (
                    planner_name, world_idx, elapsed, dist_goal,
                    collisions,
                    float(np.mean(deviations)) if deviations else 0.0,
                    min_clr if min_clr < float("inf") else 0.0,
                )
            )
            sys.stdout.flush()

            # Sleep for remainder of control period
            sleep_t = max(0.0, _CTRL_DT - (time.time() - t_iter))
            if sleep_t > 0:
                time.sleep(sleep_t)

        print()  # newline after \r progress

        self._pub_cmd(0.0, 0.0)
        self._gazebo_sim.pause()

        # ── Final state ─────────────────────────────────────────────────────────
        xf, yf, _ = self._get_pose()
        final_dist = float(np.linalg.norm(np.array([xf, yf]) - self._goal_pos))
        elapsed    = rospy.get_time() - t_start
        success    = outcome == "success"

        spl = (
            success * opt_len / max(accu_path, opt_len)
            if opt_len > 0 else 0.0
        )
        opt_time = opt_len / NavConfig.NOMINAL_SPEED if opt_len > 0 else 1.0
        nav_metric = float(
            success * opt_time
            / float(np.clip(
                elapsed,
                NavConfig.METRIC_TIME_LO * opt_time,
                NavConfig.METRIC_TIME_HI * opt_time,
            ))
        ) if success else 0.0

        return EpisodeResult(
            planner             = planner_name,
            world_idx           = world_idx,
            outcome             = outcome,
            success             = success,
            collision_count     = collisions,
            path_length         = accu_path,
            opt_path_length     = opt_len,
            time_elapsed        = elapsed,
            spl                 = spl,
            nav_metric          = nav_metric,
            mean_path_deviation = float(np.mean(deviations)) if deviations else 0.0,
            max_path_deviation  = float(np.max(deviations))  if deviations else 0.0,
            min_clearance       = min_clr if min_clr < float("inf") else 0.0,
            final_dist_to_goal  = final_dist,
        )

    # ── Reference path (used only for OT scoring, not fed to controllers) ────────

    def _load_ref_path(self, world_idx: int) -> Optional[np.ndarray]:
        pf = WorldConfig.path_file(world_idx)
        if not os.path.exists(pf):
            return None
        grid_pts = np.load(pf)
        pts = [self._init_pos.copy()]
        for gp in grid_pts:
            pts.append(np.array(WorldConfig.grid_to_gazebo(int(gp[0]), int(gp[1])), dtype=np.float64))
        pts.append(self._goal_pos.copy())
        return np.array(pts, dtype=np.float64)

    # ── Gazebo process management ──────────────────────────────────────────────

    def _restart_gazebo(self, world_idx: int):
        print("    [setup] Killing previous Gazebo processes...")
        self._kill_procs()
        time.sleep(2.0)

        SimConfig.ROBOT = self.robot
        apply_env(robot=self.robot)
        self._patch_ros_paths()

        world_file = WorldConfig.world_file(world_idx)
        if not os.path.exists(world_file):
            raise FileNotFoundError("World file not found: %s" % world_file)

        # Log roslaunch output to a file so crashes are diagnosable.
        log_dir = os.path.join(Paths.RESULTS_DIR, "benchmark", "logs")
        os.makedirs(log_dir, exist_ok=True)
        gz_log  = open(os.path.join(log_dir, "gazebo_world%d.log"  % world_idx), "w")
        nav_log = open(os.path.join(log_dir, "navstack_world%d.log" % world_idx), "w")

        print("    [setup] Launching Gazebo (world %d)..." % world_idx)
        if self.robot == "jackal":
            import rospkg
            base = rospkg.RosPack().get_path("jackal_helper")
            self._gazebo_proc = subprocess.Popen(
                [
                    "roslaunch",
                    join(base, "launch", "gazebo_launch.launch"),
                    "world_name:=" + world_file,
                    "gui:=false",
                ],
                stdout=gz_log, stderr=gz_log,
            )
        else:  # moborobo
            self._gazebo_proc = subprocess.Popen(
                [
                    "roslaunch",
                    MoboroboConfig.gazebo_launch_file(),
                    "world_name:=" + world_file,
                    "gui:=false",
                    "headless:=true",
                    "use_collision_publisher:=true",
                ],
                stdout=gz_log, stderr=gz_log,
            )
            print("    [setup] Launching nav stack (pointcloud_to_laserscan)...")
            self._nav_proc = subprocess.Popen(
                ["roslaunch", MoboroboConfig.nav_launch_file()],
                stdout=nav_log, stderr=nav_log,
            )

        # Poll for Gazebo services; abort early if the subprocess died.
        print("    [setup] Waiting for Gazebo services...", end="", flush=True)
        _GAZEBO_SVC_TIMEOUT = 90.0
        t0 = time.time()
        while time.time() - t0 < _GAZEBO_SVC_TIMEOUT:
            # Check if the Gazebo process crashed.
            if self._gazebo_proc.poll() is not None:
                print(" CRASHED")
                gz_log.flush()
                raise RuntimeError(
                    "Gazebo process exited unexpectedly (rc=%d). "
                    "See %s for details." % (self._gazebo_proc.returncode, gz_log.name)
                )
            try:
                rospy.wait_for_service("/gazebo/set_model_state", timeout=2.0)
                rospy.wait_for_service("/gazebo/get_model_state", timeout=2.0)
                break
            except rospy.ROSException:
                print(".", end="", flush=True)
        else:
            print(" TIMEOUT")
            raise TimeoutError(
                "Gazebo services did not come up within %.0fs. "
                "See %s for details." % (_GAZEBO_SVC_TIMEOUT, gz_log.name)
            )
        print(" OK (%.1fs)" % (time.time() - t0))
        self._init_gazebo_sim()
        self._current_world = world_idx

    def _kill_procs(self):
        # Gracefully stop our tracked subprocesses first.
        for proc in (self._nav_proc, self._gazebo_proc):
            if proc is None:
                continue
            try:
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait()
            except OSError:
                pass

        # Kill stale gzserver/gzclient but NOT rosmaster — roscore is persistent.
        for pattern in ("gzserver", "gzclient"):
            subprocess.call(
                ["pkill", "-SIGKILL", "-f", pattern],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        self._gazebo_proc = None
        self._nav_proc    = None
        self._ros_ready   = False
        self._gazebo_sim  = None

    def _patch_ros_paths(self):
        jackal_src   = join(Paths.JACKAL_WS, "src")
        jackal_devel = join(Paths.JACKAL_WS, "devel")
        if os.path.isdir(jackal_src):
            pkg = os.environ.get("ROS_PACKAGE_PATH", "")
            if jackal_src not in pkg:
                os.environ["ROS_PACKAGE_PATH"] = jackal_src + (":" + pkg if pkg else "")
        if os.path.isdir(jackal_devel):
            cm = os.environ.get("CMAKE_PREFIX_PATH", "")
            if jackal_devel not in cm:
                os.environ["CMAKE_PREFIX_PATH"] = jackal_devel + (":" + cm if cm else "")
        if self.robot == "moborobo" and os.path.isdir(Paths.ROBOTS_DIR):
            pkg = os.environ.get("ROS_PACKAGE_PATH", "")
            if Paths.ROBOTS_DIR not in pkg:
                os.environ["ROS_PACKAGE_PATH"] = Paths.ROBOTS_DIR + (":" + pkg if pkg else "")

    # ── ROS wiring ─────────────────────────────────────────────────────────────

    def _connect_ros(self):
        """Called once at startup — registers the ROS node, publisher, subscriber."""
        try:
            rospy.init_node("barn_benchmark", anonymous=True, disable_signals=True)
        except rospy.exceptions.ROSException:
            pass

        self._cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        rospy.Subscriber(
            self._scan_topic, LaserScan, self._scan_cb,
            queue_size=1, buff_size=2 ** 20,
        )
        self._ros_ready = True

    def _init_gazebo_sim(self):
        """Called after each Gazebo restart — creates a fresh GazeboSimulation
        whose service proxies point to the newly started gzserver."""
        self._gazebo_sim = GazeboSimulation(
            init_position=list(WorldConfig.STATIC_INIT_POSITION),
            model_name=self._model_name,
        )

    def _scan_cb(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float32)
        # A non-finite ray means "no return" — either nothing out to range_max,
        # or something closer than range_min (0.45 m) that was discarded.
        # `_scan_raw` keeps NaN so a controller with a near-field memory can
        # tell those apart; `_scan` collapses both to max range because the
        # classical planners expect finite ranges.
        invalid = (
            ~np.isfinite(ranges)
            | (ranges < msg.range_min)
            | (ranges > msg.range_max)
        )
        raw = ranges.copy()
        raw[invalid] = np.nan
        ranges[invalid] = msg.range_max
        with self._scan_lock:
            self._scan     = ranges
            self._scan_raw = raw
            self._scan_min = float(msg.angle_min)
            self._scan_inc = float(msg.angle_increment)

    def _wait_for_scan(self):
        deadline = time.time() + _SCAN_TIMEOUT
        while time.time() < deadline:
            with self._scan_lock:
                if self._scan is not None:
                    return
            time.sleep(0.05)
        raise TimeoutError(
            "No scan on '%s' within %.0fs — is Gazebo running?" % (
                self._scan_topic, _SCAN_TIMEOUT
            )
        )

    # ── Pose / command helpers ──────────────────────────────────────────────────

    def _get_pose(self) -> Tuple[float, float, float]:
        state = self._gazebo_sim.get_model_state()
        pos = state.pose.position
        ori = state.pose.orientation
        return pos.x, pos.y, _quat_to_yaw(ori.x, ori.y, ori.z, ori.w)

    def _get_pos(self) -> Tuple[float, float]:
        state = self._gazebo_sim.get_model_state()
        return state.pose.position.x, state.pose.position.y

    def _reset_robot(self):
        for _ in range(_RESET_RETRIES):
            self._gazebo_sim.reset()
            time.sleep(GazeboConfig.RESET_WAIT)
            x, y = self._get_pos()
            if math.hypot(x - self._init_pos[0], y - self._init_pos[1]) <= GazeboConfig.POSITION_TOL:
                break
        self._gazebo_sim.get_hard_collision()  # drain stale flag
        self._pub_cmd(0.0, 0.0)

    def _pub_cmd(self, linear: float, angular: float):
        if self._cmd_pub is None:
            return
        msg = Twist()
        msg.linear.x  = linear
        msg.angular.z = angular
        self._cmd_pub.publish(msg)


# ── Pure math helpers ──────────────────────────────────────────────────────────

def _quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
