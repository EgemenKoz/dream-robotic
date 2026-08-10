"""
Gymnasium environment wrapping the BARN Gazebo simulation.

Direct velocity control (bypasses move_base).

Two-part Dict observation (structure follows the multi-robot DRL
collision-avoidance literature — Fan & Long et al. ICRA'18 / IJRR'20):
  obs["scan"]  : (SCAN_STACK, N) — last K LiDAR frames, each min-pooled into
                 N sectors and normalised by SCAN_CLIP_RANGE → [0, 1].
  obs["state"] : (5,) — [dist_to_goal_norm (0-1),
                         sin(angle_to_goal), cos(angle_to_goal),
                         last_linear_norm (0-1), last_angular_norm (-1,1)]
  act  = [linear_x, angular_z]

Quick start — Gazebo already running externally:
    env = BarnEnv(robot="moborobo", world_idx=30, managed=False)
    obs, info = env.reset()
    obs, r, terminated, truncated, info = env.step(env.action_space.sample())
    env.close()

Quick start — env manages Gazebo (launches/kills per world change):
    env = BarnEnv(robot="moborobo", world_idx=None, managed=True)
    obs, info = env.reset()
"""

import math
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from os.path import join
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

# ── project imports ────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.abspath(join(os.path.dirname(__file__), "../.."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from configs.config import (
    GazeboConfig, MoboroboConfig, NavConfig, Paths,
    RobotConfig, SimConfig, WorldConfig, apply_env,
)
from configs.rl_config import RLEnvConfig, RLLogConfig

# GazeboSimulation lives in the local gazebo/ package inside this project.
from gazebo.gazebo_simulation import GazeboSimulation

# ── constants ──────────────────────────────────────────────────────────────────

_SCAN_WAIT_TIMEOUT     = 30.0  # seconds to wait for a scan message after reset
                                # (moborobo cold-start: gzserver + RoboSense plugin +
                                # pointcloud_to_laserscan can take 10–20 s the first time)
_GAZEBO_SVC_TIMEOUT    = 90.0  # seconds to wait for Gazebo services after roslaunch
_ROSCORE_START_TIMEOUT = 30.0  # seconds to wait for roscore port 11311 to accept

_SCAN_TOPIC = {
    "jackal":   "front/scan",
    "moborobo": "/scan",      # provided by pointcloud_to_laserscan in barn_move_base.launch
}
_MODEL_NAME = {
    "jackal":   "jackal",
    "moborobo": MoboroboConfig.MODEL_NAME,
}
_MAX_LINEAR = {
    "jackal":   RobotConfig.MAX_LINEAR_VEL,
    "moborobo": MoboroboConfig.MAX_LINEAR_VEL,
}
_MAX_ANGULAR = {
    "jackal":   RobotConfig.MAX_ANGULAR_VEL,
    "moborobo": MoboroboConfig.MAX_ANGULAR_VEL,
}
_SCAN_MAX_RANGE = {
    "jackal":   10.0,
    "moborobo": MoboroboConfig.LIDAR_MAX_RANGE,
}


class BarnEnv(gym.Env):
    """
    Gymnasium environment for BARN navigation RL training.

    Parameters
    ----------
    robot : str
        "jackal" or "moborobo".
    world_idx : int | None
        Fixed world index 0-299, or None to randomise each episode.
        When None and managed=True the env restarts Gazebo on every world change.
    num_scan_rays : int
        Laser scan sectors per frame (raw scan is min-pooled to this count).
        Moborobo produces ~722 rays after pointcloud_to_laserscan; Jackal ~1080.
    scan_stack : int
        Number of consecutive scan frames stacked in obs["scan"].
    scan_clip_range : float
        Ranges are clipped to min(scan_clip_range, robot max range) metres
        before normalising, to keep resolution near obstacles.
    step_duration : float
        Seconds to let physics run per step (sim-time at ~1× real-time factor).
    max_episode_steps : int
        Hard truncation limit per episode.
    gui : bool
        Open Gazebo GUI (only used when managed=True).
    managed : bool
        True  → env spawns/kills roslaunch processes.
        False → Gazebo must be running before calling reset().
    log_dir : str | None
        Directory for the per-run episodes.csv.
        None  → results/rl/ (default).
        ""    → logging disabled.
    reward_collision : float
        Terminal reward on collision (negative).
    reward_goal : float
        Terminal reward on reaching goal (positive).
    reward_step : float
        Per-step living penalty (negative, small).
    reward_shaping : bool
        If True, apply progress / heading / proximity shaping each step.
    reward_progress_scale : float
        Multiplier on (prev_dist − curr_dist) shaping.
    reward_prox_scale : float
        Scale of the near-obstacle penalty (see PROX_THRESHOLD).
    prox_threshold : float
        Normalised-scan distance under which the obstacle-proximity penalty
        becomes active (linear ramp toward 0).
    reward_heading_scale : float
        Scale of the |angle_to_goal|/π heading penalty.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        robot: str = "moborobo",
        world_idx: Optional[int] = None,
        num_scan_rays: int = RLEnvConfig.NUM_SCAN_RAYS,
        scan_stack: int = RLEnvConfig.SCAN_STACK,
        scan_clip_range: float = RLEnvConfig.SCAN_CLIP_RANGE,
        step_duration: float = RLEnvConfig.STEP_DURATION,
        # None → resolved per-robot via RLEnvConfig.max_episode_steps(robot).
        # Pass an int to override (e.g. for short unit tests).
        max_episode_steps: Optional[int] = None,
        gui: bool = False,
        managed: bool = True,
        log_dir: Optional[str] = None,
        reward_collision: float = RLEnvConfig.REWARD_COLLISION,
        reward_goal: float = RLEnvConfig.REWARD_GOAL,
        reward_step: float = RLEnvConfig.REWARD_STEP,
        reward_shaping: bool = RLEnvConfig.REWARD_SHAPING,
        # None → resolved per-robot via RLEnvConfig.reward_progress_scale(robot).
        reward_progress_scale: Optional[float] = None,
        reward_prox_scale: float = RLEnvConfig.REWARD_PROX_SCALE,
        prox_threshold: float = RLEnvConfig.PROX_THRESHOLD,
        reward_heading_scale: float = RLEnvConfig.REWARD_HEADING_SCALE,
        path_follow_shaping: bool = RLEnvConfig.PATH_FOLLOW_SHAPING,
        reward_path_scale: float = RLEnvConfig.REWARD_PATH_SCALE,
        path_max_dev: float = RLEnvConfig.PATH_MAX_DEV,
    ):
        super().__init__()

        if robot not in ("jackal", "moborobo"):
            raise ValueError("robot must be 'jackal' or 'moborobo', got '%s'" % robot)
        max_world = WorldConfig.world_count() - 1
        if world_idx is not None and not (0 <= world_idx <= max_world):
            raise ValueError("world_idx must be 0-%d or None, got %r" % (max_world, world_idx))

        self.robot = robot
        self._fixed_world_idx = world_idx
        self.num_scan_rays = num_scan_rays
        self.scan_stack = scan_stack
        self.step_duration = step_duration
        self.max_episode_steps = (
            max_episode_steps
            if max_episode_steps is not None
            else RLEnvConfig.max_episode_steps(robot)
        )
        self.gui = gui
        self.managed = managed

        self._r_collision = reward_collision
        self._r_goal = reward_goal
        self._r_step = reward_step
        self._r_shaping = reward_shaping
        self._r_progress_scale = (
            reward_progress_scale
            if reward_progress_scale is not None
            else RLEnvConfig.reward_progress_scale(robot)
        )
        self._r_prox_scale = reward_prox_scale
        self._prox_threshold = prox_threshold
        self._r_heading_scale = reward_heading_scale
        self._path_follow_shaping = path_follow_shaping
        self._r_path_scale = reward_path_scale
        self._path_max_dev = path_max_dev
        # Loaded lazily on world change; None disables the path penalty for
        # this episode (e.g. when path_N.npy is missing).
        self._ref_path: Optional[np.ndarray] = None

        self._scan_topic = _SCAN_TOPIC[robot]
        self._model_name = _MODEL_NAME[robot]
        self._max_linear = _MAX_LINEAR[robot]
        self._max_angular = _MAX_ANGULAR[robot]
        self._scan_max_range = _SCAN_MAX_RANGE[robot]
        # Normalisation range for the scan observation: clip to keep
        # resolution near obstacles (BARN gaps are < 1 m wide).
        self._scan_norm_range = min(float(scan_clip_range), self._scan_max_range)

        # Fixed start / goal coordinates (same across all 300 static BARN worlds)
        ix, iy, _iyaw = WorldConfig.STATIC_INIT_POSITION
        gx = ix + WorldConfig.STATIC_GOAL_OFFSET[0]
        gy = iy + WorldConfig.STATIC_GOAL_OFFSET[1]
        self._init_pos: Tuple[float, float] = (ix, iy)
        self._goal_pos: Tuple[float, float] = (gx, gy)
        self._max_dist = float(math.sqrt((gx - ix) ** 2 + (gy - iy) ** 2))

        # ── Gymnasium spaces ───────────────────────────────────────────────────
        # Two-part Dict observation:
        #   "scan"  : (scan_stack, N) — last K min-pooled scan frames, [0, 1].
        #             Row 0 is the oldest frame, row K-1 the newest.
        #   "state" : (5,) — [dist_norm       [0, 1],
        #                     sin(angle_to_goal_robot_frame)  [-1, 1],
        #                     cos(angle_to_goal_robot_frame)  [-1, 1],
        #                     last linear vel / max_linear    [0, 1],
        #                     last angular vel / max_angular  [-1, 1]]
        # sin/cos avoids the discontinuity at ±π that a raw normalised angle has.
        N = num_scan_rays
        state_low  = np.array([0.0, -1.0, -1.0, 0.0, -1.0], dtype=np.float32)
        state_high = np.array([1.0,  1.0,  1.0, 1.0,  1.0], dtype=np.float32)
        self.observation_space = spaces.Dict({
            "scan":  spaces.Box(0.0, 1.0, shape=(scan_stack, N), dtype=np.float32),
            "state": spaces.Box(state_low, state_high, dtype=np.float32),
        })

        self.action_space = spaces.Box(
            low=np.array([0.0, -self._max_angular], dtype=np.float32),
            high=np.array([self._max_linear, self._max_angular], dtype=np.float32),
            dtype=np.float32,
        )

        # ── Logging ────────────────────────────────────────────────────────────
        if log_dir is None:
            self._log_dir: Optional[str] = RLLogConfig.LOG_DIR
        elif log_dir == "":
            self._log_dir = None
        else:
            self._log_dir = log_dir
        if self._log_dir:
            os.makedirs(self._log_dir, exist_ok=True)

        # ── Episode state ──────────────────────────────────────────────────────
        self._episode: int = 0
        self._step_count: int = 0
        self._episode_reward: float = 0.0
        self._prev_dist: float = self._max_dist
        self._last_action: np.ndarray = np.zeros(2, dtype=np.float32)
        self._current_world_idx: Optional[int] = None
        # Rolling scan-frame history for obs["scan"]; cleared on reset and
        # re-seeded with K copies of the first frame of the episode.
        self._scan_history: deque = deque(maxlen=scan_stack)

        # ── ROS / Gazebo handles ───────────────────────────────────────────────
        self._roscore_proc: Optional[subprocess.Popen] = None
        self._gazebo_proc: Optional[subprocess.Popen] = None
        self._nav_proc: Optional[subprocess.Popen] = None
        self._gazebo_sim: Optional[GazeboSimulation] = None
        self._cmd_vel_pub: Optional[rospy.Publisher] = None

        self._latest_scan: Optional[np.ndarray] = None
        self._scan_angle_min: float = -math.pi
        self._scan_angle_increment: float = 0.0087   # default from barn_move_base.launch
        self._scan_lock = threading.Lock()

        self._ros_ready = False
        self._closed = False
        # Tracked separately from self._gazebo_proc (roslaunch wrapper): when
        # gzserver itself segfaults mid-init the wrapper may still be alive
        # for several seconds, during which any rospy ServiceProxy.call() to
        # /gazebo/* would hang forever.  We use this PID to fail fast instead.
        self._gzserver_pid: Optional[int] = None

    # ──────────────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ──────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self._episode += 1
        self._step_count = 0
        self._episode_reward = 0.0
        self._last_action = np.zeros(2, dtype=np.float32)
        self._scan_history.clear()

        # Choose world
        if self._fixed_world_idx is not None:
            target_world = self._fixed_world_idx
        else:
            target_world = int(self.np_random.integers(0, WorldConfig.world_count()))

        world_changed = (target_world != self._current_world_idx)

        if self.managed:
            if self._gazebo_proc is None or world_changed:
                self._restart_gazebo(target_world)
        else:
            if not self._ros_ready:
                self._connect_ros()

        if world_changed or self._ref_path is None:
            self._ref_path = self._load_ref_path(target_world)
        self._current_world_idx = target_world

        # Fail fast if gzserver died between _restart_gazebo and now —
        # otherwise the unpause() service call below blocks forever.
        self._check_gzserver_alive()

        # Unpause physics so sensors tick: every step() ends with pause(), so
        # without this _wait_for_scan() would block forever on the 2nd reset.
        self._gazebo_sim.unpause()

        # Teleport robot back to start, wait until it arrives
        retries = 0
        while retries < 10:
            self._gazebo_sim.reset()
            time.sleep(GazeboConfig.RESET_WAIT)
            state = self._gazebo_sim.get_model_state()
            pos = state.pose.position
            if _dist(self._init_pos, (pos.x, pos.y)) <= GazeboConfig.POSITION_TOL:
                break
            retries += 1

        # Drain stale collision flag and stop the robot
        self._gazebo_sim.get_hard_collision()
        self._pub_cmd(0.0, 0.0)

        # Invalidate cached scan so we get a fresh reading
        with self._scan_lock:
            self._latest_scan = None
        if self._episode == 1:
            print("    [reset] Waiting for first LaserScan on %s..." % self._scan_topic,
                  flush=True)
        self._wait_for_scan()

        obs = self._build_obs()
        state = self._gazebo_sim.get_model_state()
        self._prev_dist = _dist(self._goal_pos, (state.pose.position.x, state.pose.position.y))

        # NOTE: the key is "episode_idx" (not "episode") to avoid clashing with
        # SB3's Monitor wrapper, which writes info["episode"] = {"r","l","t"} on
        # episode termination — see TrainProgressCallback in train_sac.py.
        return obs, {"world_idx": self._current_world_idx, "episode_idx": self._episode}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.clip(action, self.action_space.low, self.action_space.high)
        lin = float(action[0])
        ang = float(action[1])
        self._last_action[0] = lin
        self._last_action[1] = ang

        # Run physics for one timestep
        self._gazebo_sim.unpause()
        self._pub_cmd(lin, ang)
        time.sleep(self.step_duration)
        self._gazebo_sim.pause()

        # Observe
        obs = self._build_obs()
        state = self._gazebo_sim.get_model_state()
        pos = state.pose.position
        ori = state.pose.orientation
        curr_dist = _dist(self._goal_pos, (pos.x, pos.y))
        collided = self._gazebo_sim.get_hard_collision()

        # Reward — track each component so it can be surfaced in info / debug.
        r_step = float(self._r_step)
        r_progress = 0.0
        r_prox = 0.0
        r_heading = 0.0
        r_path = 0.0
        path_dev = 0.0
        if self._r_shaping:
            # progress: positive when moving toward goal
            r_progress = self._r_progress_scale * (self._prev_dist - curr_dist)

            # near-obstacle penalty (linear ramp from threshold to 0)
            scan_norm = obs["scan"][-1]   # newest frame
            min_scan = float(np.min(scan_norm)) if scan_norm.size else 1.0
            if min_scan < self._prox_threshold:
                r_prox = -self._r_prox_scale * (
                    (self._prox_threshold - min_scan) / self._prox_threshold
                )

            # heading penalty: scaled |angle_to_goal|/pi
            yaw = _quat_to_yaw(ori.x, ori.y, ori.z, ori.w)
            dx = self._goal_pos[0] - pos.x
            dy = self._goal_pos[1] - pos.y
            angle_robot = _wrap_angle(math.atan2(dy, dx) - yaw)
            r_heading = -self._r_heading_scale * (abs(angle_robot) / math.pi)

            # path-following: perpendicular deviation from A* reference polyline.
            # Path is reward-only — never appears in the observation.
            if self._path_follow_shaping and self._ref_path is not None:
                path_dev = _point_to_path_dist(
                    np.array([pos.x, pos.y], dtype=np.float64),
                    self._ref_path,
                )
                ratio = min(path_dev, self._path_max_dev) / self._path_max_dev
                r_path = -self._r_path_scale * ratio

        reward = r_step + r_progress + r_prox + r_heading + r_path
        self._prev_dist = curr_dist

        terminated = truncated = False
        if collided:
            reward += self._r_collision
            terminated = True
            outcome = "collision"
        elif curr_dist <= NavConfig.GOAL_TOLERANCE:
            reward += self._r_goal
            terminated = True
            outcome = "success"
        else:
            self._step_count += 1
            if self._step_count >= self.max_episode_steps:
                truncated = True
                outcome = "timeout"
            else:
                outcome = "running"

        self._episode_reward += reward

        info: Dict[str, Any] = {
            "world_idx": self._current_world_idx,
            "episode_idx": self._episode,
            "step": self._step_count,
            "dist_to_goal": curr_dist,
            "collided": collided,
            "outcome": outcome,
            "episode_reward": self._episode_reward,
            "reward_components": {
                "step": r_step,
                "progress": r_progress,
                "prox": r_prox,
                "heading": r_heading,
                "path": r_path,
                "path_dev_m": path_dev,
            },
        }

        if terminated or truncated:
            self._pub_cmd(0.0, 0.0)
            self._log_episode(outcome, curr_dist)

        return obs, reward, terminated, truncated, info

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._pub_cmd(0.0, 0.0)
        except Exception:
            pass
        if self.managed:
            self._kill_processes()
            if self._roscore_proc is not None:
                try:
                    self._roscore_proc.send_signal(signal.SIGINT)
                    self._roscore_proc.wait(timeout=5)
                except Exception:
                    try:
                        self._roscore_proc.terminate()
                    except Exception:
                        pass
                self._roscore_proc = None

    # ──────────────────────────────────────────────────────────────────────────
    # Process management
    # ──────────────────────────────────────────────────────────────────────────

    def _restart_gazebo(self, world_idx: int):
        # Start (or keep) a persistent roscore: roslaunch will reuse it, so
        # `rospy.init_node` only blocks on the very first call.
        self._ensure_roscore()

        print("    [setup] Killing previous Gazebo processes...", flush=True)
        self._kill_processes()
        time.sleep(1.0)

        SimConfig.ROBOT = self.robot
        apply_env(robot=self.robot)
        self._patch_ros_paths()

        world_file = WorldConfig.world_file(world_idx)
        if not os.path.exists(world_file):
            raise FileNotFoundError("World file not found: %s" % world_file)

        log_dir = self._log_dir or os.path.join(_PROJECT_ROOT, "results", "rl", "logs")
        os.makedirs(log_dir, exist_ok=True)
        gz_log_path  = os.path.join(log_dir, "gazebo_world%d.log" % world_idx)
        nav_log_path = os.path.join(log_dir, "navstack_world%d.log" % world_idx)
        gz_log  = open(gz_log_path,  "w")
        nav_log = open(nav_log_path, "w")

        print("    [setup] Launching Gazebo (world %d)... log=%s"
              % (world_idx, gz_log_path), flush=True)
        if self.robot == "jackal":
            import rospkg
            base = rospkg.RosPack().get_path("jackal_helper")
            self._gazebo_proc = subprocess.Popen(
                [
                    "roslaunch",
                    join(base, "launch", "gazebo_launch.launch"),
                    "world_name:=" + world_file,
                    "gui:=" + ("true" if self.gui else "false"),
                ],
                stdout=gz_log, stderr=gz_log,
            )
        else:
            self._gazebo_proc = subprocess.Popen(
                [
                    "roslaunch",
                    MoboroboConfig.gazebo_launch_file(),
                    "world_name:=" + world_file,
                    "gui:=" + ("true" if self.gui else "false"),
                    "headless:=" + ("false" if self.gui else "true"),
                    "use_collision_publisher:=true",
                ],
                stdout=gz_log, stderr=gz_log,
            )
            # barn_move_base.launch provides pointcloud_to_laserscan → /scan.
            # move_base idles because we never send it a goal.
            self._nav_proc = subprocess.Popen(
                ["roslaunch", MoboroboConfig.nav_launch_file()],
                stdout=nav_log, stderr=nav_log,
            )

        # ── Poll for Gazebo services instead of fixed sleep — surfaces a
        # crashed gzserver fast (and avoids hangs when 5s isn't enough). ───────
        print("    [setup] Waiting for Gazebo services...", end="", flush=True)
        t0 = time.time()
        while time.time() - t0 < _GAZEBO_SVC_TIMEOUT:
            if self._gazebo_proc.poll() is not None:
                print(" CRASHED")
                raise RuntimeError(
                    "Gazebo process exited unexpectedly (rc=%d). See %s for details."
                    % (self._gazebo_proc.returncode, gz_log_path)
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
                "Gazebo services did not come up within %.0fs. See %s for details."
                % (_GAZEBO_SVC_TIMEOUT, gz_log_path)
            )
        print(" OK (%.1fs)" % (time.time() - t0), flush=True)

        # Find the actual gzserver PID (not the roslaunch wrapper).  Used by
        # _check_gzserver_alive() to fail fast on a crashed gzserver instead
        # of hanging in a Gazebo ROS service call.
        self._gzserver_pid = _find_gzserver_pid()
        if self._gzserver_pid is None:
            raise RuntimeError(
                "gzserver process not found after services came up. "
                "See %s for the crash." % gz_log_path
            )

        # Connect ROS (lazy — only really does work on the first reset).
        self._connect_ros()
        # The proxies created by GazeboSimulation point at the freshly started
        # gzserver — must be recreated on every world change.
        self._gazebo_sim = GazeboSimulation(
            init_position=list(WorldConfig.STATIC_INIT_POSITION),
            model_name=self._model_name,
        )

    def _ensure_roscore(self):
        """Lazy-start a persistent roscore that outlives Gazebo restarts."""
        if self._roscore_proc is not None and self._roscore_proc.poll() is None:
            return  # already running
        import socket
        log_dir = self._log_dir or os.path.join(_PROJECT_ROOT, "results", "rl", "logs")
        os.makedirs(log_dir, exist_ok=True)
        rc_log = open(os.path.join(log_dir, "roscore.log"), "w")
        print("    [roscore] Starting roscore...", end="", flush=True)
        self._roscore_proc = subprocess.Popen(
            ["roscore"], stdout=rc_log, stderr=rc_log,
        )
        deadline = time.time() + _ROSCORE_START_TIMEOUT
        while time.time() < deadline:
            try:
                s = socket.create_connection(("localhost", 11311), timeout=1.0)
                s.close()
                break
            except (socket.error, socket.timeout):
                print(".", end="", flush=True)
                time.sleep(0.5)
        else:
            raise TimeoutError("roscore did not start within %.0fs"
                               % _ROSCORE_START_TIMEOUT)
        print(" OK", flush=True)

    def _kill_processes(self):
        # Gracefully stop tracked subprocesses (NOT the roscore — keep that).
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
        # Pkill any orphan processes — survives roslaunch crashes and dead
        # python sessions from previous runs.  Roscore is intentionally left
        # alive (see _ensure_roscore); everything else gets reaped.
        for pattern in ("gzserver", "gzclient",
                        "pointcloud_to_laserscan", "move_base",
                        "roslaunch.*barn_"):
            subprocess.call(
                ["pkill", "-SIGKILL", "-f", pattern],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        self._gazebo_proc = None
        self._nav_proc = None
        self._gazebo_sim = None

    def _patch_ros_paths(self):
        """Mirror the path-patching logic from run_barn_custom.py."""
        jackal_src = join(Paths.JACKAL_WS, "src")
        jackal_devel = join(Paths.JACKAL_WS, "devel")
        if os.path.isdir(jackal_src):
            pkg = os.environ.get("ROS_PACKAGE_PATH", "")
            if jackal_src not in pkg:
                os.environ["ROS_PACKAGE_PATH"] = jackal_src + (":" + pkg if pkg else "")
        if os.path.isdir(jackal_devel):
            cmake = os.environ.get("CMAKE_PREFIX_PATH", "")
            if jackal_devel not in cmake:
                os.environ["CMAKE_PREFIX_PATH"] = jackal_devel + (":" + cmake if cmake else "")
        if self.robot == "moborobo" and os.path.isdir(Paths.ROBOTS_DIR):
            pkg = os.environ.get("ROS_PACKAGE_PATH", "")
            if Paths.ROBOTS_DIR not in pkg:
                os.environ["ROS_PACKAGE_PATH"] = Paths.ROBOTS_DIR + (":" + pkg if pkg else "")

    # ──────────────────────────────────────────────────────────────────────────
    # ROS wiring
    # ──────────────────────────────────────────────────────────────────────────

    def _connect_ros(self):
        """Initialise ROS node, publishers, subscribers.
        Idempotent — safe to call after every Gazebo restart."""
        if not self._ros_ready:
            try:
                rospy.init_node("barn_rl_env", anonymous=True, disable_signals=True)
            except rospy.exceptions.ROSException:
                pass  # already initialised in this process

            self._cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
            rospy.Subscriber(
                self._scan_topic, LaserScan, self._scan_cb, queue_size=1,
                buff_size=2 ** 20,
            )
            self._ros_ready = True

    def _scan_cb(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float32)
        # Replace NaN / Inf (use_inf=true in the launch means inf = max range)
        ranges = np.where(np.isfinite(ranges), ranges, msg.range_max)
        ranges = np.clip(ranges, msg.range_min, msg.range_max)
        with self._scan_lock:
            self._latest_scan = ranges
            self._scan_angle_min = float(msg.angle_min)
            self._scan_angle_increment = float(msg.angle_increment)

    def _wait_for_scan(self):
        deadline = time.time() + _SCAN_WAIT_TIMEOUT
        last_health_check = 0.0
        while time.time() < deadline:
            with self._scan_lock:
                if self._latest_scan is not None:
                    return
            # Cheap process-liveness check every 1 s so a crashed gzserver
            # surfaces immediately instead of blocking the whole timeout.
            if time.time() - last_health_check > 1.0:
                self._check_gzserver_alive()
                last_health_check = time.time()
            time.sleep(0.05)
        raise TimeoutError(
            "No LaserScan received on '%s' within %.1f s. "
            "Is Gazebo running and the sensor converter up?" % (
                self._scan_topic, _SCAN_WAIT_TIMEOUT
            )
        )

    def _check_gzserver_alive(self):
        """Raise RuntimeError if the tracked gzserver PID is no longer running.

        Catches the case where gzserver segfaults during plugin init: the
        roslaunch wrapper stays alive for several seconds while killing
        sibling nodes, and any /gazebo/* ServiceProxy.call() would otherwise
        hang forever waiting for a reply from a dead process.
        """
        pid = self._gzserver_pid
        if pid is None:
            return
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, OSError):
            raise RuntimeError(
                "gzserver (pid=%d) is no longer alive. Likely segfaulted "
                "during init — check the Gazebo log in the run dir." % pid
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Observation building
    # ──────────────────────────────────────────────────────────────────────────

    def _get_scan_normalised(self) -> np.ndarray:
        with self._scan_lock:
            raw = self._latest_scan
        if raw is None:
            return np.ones(self.num_scan_rays, dtype=np.float32)

        # Min-pool into num_scan_rays sectors: unlike linear interpolation,
        # a thin obstacle hit by a single raw ray can never disappear.
        pooled = _min_pool(raw, self.num_scan_rays)
        return np.clip(pooled / self._scan_norm_range, 0.0, 1.0).astype(np.float32)

    def _build_obs(self) -> Dict[str, np.ndarray]:
        scan = self._get_scan_normalised()

        # Maintain the frame stack: seed with K copies on the first frame of
        # an episode (history is cleared in reset()), append afterwards.
        if not self._scan_history:
            for _ in range(self.scan_stack):
                self._scan_history.append(scan)
        else:
            self._scan_history.append(scan)
        scan_stack = np.stack(self._scan_history).astype(np.float32)

        state = self._gazebo_sim.get_model_state()
        pos = state.pose.position
        ori = state.pose.orientation

        yaw = _quat_to_yaw(ori.x, ori.y, ori.z, ori.w)
        dx = self._goal_pos[0] - pos.x
        dy = self._goal_pos[1] - pos.y
        dist = math.sqrt(dx * dx + dy * dy)
        dist_norm = float(np.clip(dist / self._max_dist, 0.0, 1.0))

        angle_robot = _wrap_angle(math.atan2(dy, dx) - yaw)
        sin_a = math.sin(angle_robot)
        cos_a = math.cos(angle_robot)

        last_lin_norm = float(np.clip(self._last_action[0] / self._max_linear, 0.0, 1.0))
        last_ang_norm = float(np.clip(self._last_action[1] / self._max_angular, -1.0, 1.0))

        return {
            "scan": scan_stack,
            "state": np.array(
                [dist_norm, sin_a, cos_a, last_lin_norm, last_ang_norm],
                dtype=np.float32,
            ),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────
    # Public helpers for MPPI / classical controllers
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def goal_pos(self) -> Tuple[float, float]:
        """World-frame goal position (x, y)."""
        return self._goal_pos

    def get_scan_info(self) -> Tuple[Optional[np.ndarray], float, float, float]:
        """
        Return raw scan data needed by MPPI.

        Returns
        -------
        ranges : np.ndarray | None
            Range values in metres (None if no scan received yet).
        angle_min : float
            Start angle of the scan (radians).
        angle_increment : float
            Angular step between consecutive rays (radians).
        max_range : float
            Maximum valid range of the sensor (metres).
        """
        with self._scan_lock:
            ranges = self._latest_scan.copy() if self._latest_scan is not None else None
        return ranges, self._scan_angle_min, self._scan_angle_increment, self._scan_max_range

    def get_pose(self) -> Tuple[float, float, float]:
        """
        Return the current robot pose from Gazebo ground truth.

        Returns
        -------
        x, y : float   — world-frame position (metres)
        theta : float  — yaw angle (radians)
        """
        state = self._gazebo_sim.get_model_state()
        pos = state.pose.position
        ori = state.pose.orientation
        return pos.x, pos.y, _quat_to_yaw(ori.x, ori.y, ori.z, ori.w)

    # ──────────────────────────────────────────────────────────────────────────

    def _pub_cmd(self, linear: float, angular: float):
        if self._cmd_vel_pub is None:
            return
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self._cmd_vel_pub.publish(msg)

    def _load_ref_path(self, world_idx: int) -> Optional[np.ndarray]:
        """Load the A* reference polyline for path-following shaping.

        Mirrors benchmark/benchmark_env.py: grid coordinates → Gazebo frame,
        with the static start prepended and the goal appended. Returns None
        when no path file exists for this world (penalty is then disabled).
        """
        pf = WorldConfig.path_file(world_idx)
        if not os.path.exists(pf):
            return None
        try:
            grid_pts = np.load(pf)
        except Exception:
            return None
        pts = [np.array(self._init_pos, dtype=np.float64)]
        for gp in grid_pts:
            pts.append(np.array(
                WorldConfig.grid_to_gazebo(int(gp[0]), int(gp[1])),
                dtype=np.float64,
            ))
        pts.append(np.array(self._goal_pos, dtype=np.float64))
        return np.array(pts, dtype=np.float64)

    def _log_episode(self, outcome: str, final_dist: float):
        if not self._log_dir:
            return
        log_file = join(self._log_dir, "episodes.csv")
        write_header = not os.path.exists(log_file)
        with open(log_file, "a") as f:
            if write_header:
                f.write("episode,world_idx,outcome,steps,total_reward,final_dist_to_goal\n")
            f.write("%d,%d,%s,%d,%.4f,%.4f\n" % (
                self._episode,
                self._current_world_idx,
                outcome,
                self._step_count,
                self._episode_reward,
                final_dist,
            ))


# ── Pure math helpers (also importable for unit tests) ────────────────────────

def _min_pool(ranges: np.ndarray, n: int) -> np.ndarray:
    """Downsample `ranges` (M,) to (n,) by taking the minimum per sector.

    When M < n (rare — coarse sensor), fall back to linear interpolation
    since there is nothing to pool.
    """
    ranges = np.asarray(ranges, dtype=np.float32)
    m = len(ranges)
    if m == n:
        return ranges.copy()
    if m < n:
        src_idx = np.linspace(0, m - 1, n)
        return np.interp(src_idx, np.arange(m), ranges).astype(np.float32)
    # M > n: sector edges are strictly increasing because the step m/n > 1.
    edges = np.linspace(0, m, n + 1).astype(np.int64)
    return np.minimum.reduceat(ranges, edges[:-1]).astype(np.float32)


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _find_gzserver_pid() -> Optional[int]:
    """Return the PID of the currently running gzserver binary, or None.

    There are usually two PIDs after `roslaunch gazebo_ros`: a /bin/sh
    wrapper and the real gzserver process; we prefer the latter.
    """
    # Match by process *name* (comm), not full command line, so we don't
    # false-match shell invocations whose argv happens to contain "gzserver".
    try:
        out = subprocess.check_output(
            ["pgrep", "-x", "gzserver"],
            stderr=subprocess.DEVNULL,
        ).decode()
    except subprocess.CalledProcessError:
        return None
    pids = [int(p) for p in out.split() if p.strip().isdigit()]
    if not pids:
        return None
    # When multiple matches exist, prefer the highest PID (started last).
    return max(pids)


def _point_to_path_dist(pt: np.ndarray, path: np.ndarray) -> float:
    """Minimum perpendicular distance from pt (2,) to polyline path (N, 2).

    Duplicated from benchmark/metrics.py to avoid a benchmark→rl import cycle.
    """
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


def _wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle
