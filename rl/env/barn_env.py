"""
Gymnasium environment wrapping the BARN Gazebo simulation.

Direct velocity control (bypasses move_base).
  obs  = [scan_ranges (norm 0-1, N rays), dist_to_goal (0-1), angle_to_goal (-1,1)]
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

_SCAN_WAIT_TIMEOUT = 10.0  # seconds to wait for a scan message after reset

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
        Laser scan bins in the observation (scan is resampled to this count).
        Moborobo produces ~722 rays after pointcloud_to_laserscan; Jackal ~1080.
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
        If True, add (prev_dist − curr_dist) shaping each step so the agent
        gets a dense signal for moving toward the goal.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        robot: str = "moborobo",
        world_idx: Optional[int] = None,
        num_scan_rays: int = RLEnvConfig.NUM_SCAN_RAYS,
        step_duration: float = RLEnvConfig.STEP_DURATION,
        max_episode_steps: int = RLEnvConfig.MAX_EPISODE_STEPS,
        gui: bool = False,
        managed: bool = True,
        log_dir: Optional[str] = None,
        reward_collision: float = RLEnvConfig.REWARD_COLLISION,
        reward_goal: float = RLEnvConfig.REWARD_GOAL,
        reward_step: float = RLEnvConfig.REWARD_STEP,
        reward_shaping: bool = RLEnvConfig.REWARD_SHAPING,
    ):
        super().__init__()

        if robot not in ("jackal", "moborobo"):
            raise ValueError("robot must be 'jackal' or 'moborobo', got '%s'" % robot)
        if world_idx is not None and not (0 <= world_idx <= 299):
            raise ValueError("world_idx must be 0-299 or None, got %r" % world_idx)

        self.robot = robot
        self._fixed_world_idx = world_idx
        self.num_scan_rays = num_scan_rays
        self.step_duration = step_duration
        self.max_episode_steps = max_episode_steps
        self.gui = gui
        self.managed = managed

        self._r_collision = reward_collision
        self._r_goal = reward_goal
        self._r_step = reward_step
        self._r_shaping = reward_shaping

        self._scan_topic = _SCAN_TOPIC[robot]
        self._model_name = _MODEL_NAME[robot]
        self._max_linear = _MAX_LINEAR[robot]
        self._max_angular = _MAX_ANGULAR[robot]
        self._scan_max_range = _SCAN_MAX_RANGE[robot]

        # Fixed start / goal coordinates (same across all 300 static BARN worlds)
        ix, iy, _iyaw = WorldConfig.STATIC_INIT_POSITION
        gx = ix + WorldConfig.STATIC_GOAL_OFFSET[0]
        gy = iy + WorldConfig.STATIC_GOAL_OFFSET[1]
        self._init_pos: Tuple[float, float] = (ix, iy)
        self._goal_pos: Tuple[float, float] = (gx, gy)
        self._max_dist = float(math.sqrt((gx - ix) ** 2 + (gy - iy) ** 2))

        # ── Gymnasium spaces ───────────────────────────────────────────────────
        # Observation layout:
        #   [0 : num_scan_rays]   → normalised scan ranges  [0, 1]
        #   [num_scan_rays]       → normalised dist to goal [0, 1]
        #   [num_scan_rays + 1]   → normalised angle to goal [-1, 1]
        obs_dim = num_scan_rays + 2
        obs_low = np.zeros(obs_dim, dtype=np.float32)
        obs_high = np.ones(obs_dim, dtype=np.float32)
        obs_low[-1] = -1.0   # angle_to_goal lower bound
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

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
        self._current_world_idx: Optional[int] = None

        # ── ROS / Gazebo handles ───────────────────────────────────────────────
        self._gazebo_proc: Optional[subprocess.Popen] = None
        self._nav_proc: Optional[subprocess.Popen] = None
        self._gazebo_sim: Optional[GazeboSimulation] = None
        self._cmd_vel_pub: Optional[rospy.Publisher] = None

        self._latest_scan: Optional[np.ndarray] = None
        self._scan_lock = threading.Lock()

        self._ros_ready = False
        self._closed = False

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

        # Choose world
        if self._fixed_world_idx is not None:
            target_world = self._fixed_world_idx
        else:
            target_world = int(self.np_random.integers(0, WorldConfig.STATIC_WORLD_COUNT))

        world_changed = (target_world != self._current_world_idx)

        if self.managed:
            if self._gazebo_proc is None or world_changed:
                self._restart_gazebo(target_world)
        else:
            if not self._ros_ready:
                self._connect_ros()

        self._current_world_idx = target_world

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
        self._wait_for_scan()

        obs = self._build_obs()
        state = self._gazebo_sim.get_model_state()
        self._prev_dist = _dist(self._goal_pos, (state.pose.position.x, state.pose.position.y))

        return obs, {"world_idx": self._current_world_idx, "episode": self._episode}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.clip(action, self.action_space.low, self.action_space.high)
        lin = float(action[0])
        ang = float(action[1])

        # Run physics for one timestep
        self._gazebo_sim.unpause()
        self._pub_cmd(lin, ang)
        time.sleep(self.step_duration)
        self._gazebo_sim.pause()

        # Observe
        obs = self._build_obs()
        state = self._gazebo_sim.get_model_state()
        pos = state.pose.position
        curr_dist = _dist(self._goal_pos, (pos.x, pos.y))
        collided = self._gazebo_sim.get_hard_collision()

        # Reward
        reward = float(self._r_step)
        if self._r_shaping:
            reward += self._prev_dist - curr_dist   # positive when moving toward goal
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
            "episode": self._episode,
            "step": self._step_count,
            "dist_to_goal": curr_dist,
            "collided": collided,
            "outcome": outcome,
            "episode_reward": self._episode_reward,
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

    # ──────────────────────────────────────────────────────────────────────────
    # Process management
    # ──────────────────────────────────────────────────────────────────────────

    def _restart_gazebo(self, world_idx: int):
        self._kill_processes()
        time.sleep(1.0)

        SimConfig.ROBOT = self.robot
        apply_env(robot=self.robot)
        self._patch_ros_paths()

        world_file = WorldConfig.world_file(world_idx)
        if not os.path.exists(world_file):
            raise FileNotFoundError("World file not found: %s" % world_file)

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
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
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
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # barn_move_base.launch provides pointcloud_to_laserscan → /scan.
            # move_base idles because we never send it a goal.
            self._nav_proc = subprocess.Popen(
                ["roslaunch", MoboroboConfig.nav_launch_file()],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        time.sleep(GazeboConfig.STARTUP_WAIT)
        self._connect_ros()

    def _kill_processes(self):
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
        self._gazebo_proc = None
        self._nav_proc = None
        self._ros_ready = False
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
        """Initialise ROS node, publishers, subscribers, and GazeboSimulation."""
        try:
            rospy.init_node("barn_rl_env", anonymous=True, disable_signals=True)
        except rospy.exceptions.ROSException:
            pass  # already initialised in this process

        self._cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        rospy.Subscriber(
            self._scan_topic, LaserScan, self._scan_cb, queue_size=1,
            buff_size=2 ** 20,
        )

        self._gazebo_sim = GazeboSimulation(
            init_position=list(WorldConfig.STATIC_INIT_POSITION),
            model_name=self._model_name,
        )
        self._ros_ready = True

    def _scan_cb(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float32)
        # Replace NaN / Inf (use_inf=true in the launch means inf = max range)
        ranges = np.where(np.isfinite(ranges), ranges, msg.range_max)
        ranges = np.clip(ranges, msg.range_min, msg.range_max)
        with self._scan_lock:
            self._latest_scan = ranges

    def _wait_for_scan(self):
        deadline = time.time() + _SCAN_WAIT_TIMEOUT
        while time.time() < deadline:
            with self._scan_lock:
                if self._latest_scan is not None:
                    return
            time.sleep(0.05)
        raise TimeoutError(
            "No LaserScan received on '%s' within %.1f s. "
            "Is Gazebo running and the sensor converter up?" % (
                self._scan_topic, _SCAN_WAIT_TIMEOUT
            )
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Observation building
    # ──────────────────────────────────────────────────────────────────────────

    def _get_scan_normalised(self) -> np.ndarray:
        with self._scan_lock:
            raw = self._latest_scan
        if raw is None:
            return np.ones(self.num_scan_rays, dtype=np.float32)

        # Resample to target resolution with linear interpolation
        if len(raw) != self.num_scan_rays:
            src_idx = np.linspace(0, len(raw) - 1, self.num_scan_rays)
            raw = np.interp(src_idx, np.arange(len(raw)), raw).astype(np.float32)

        return np.clip(raw / self._scan_max_range, 0.0, 1.0).astype(np.float32)

    def _build_obs(self) -> np.ndarray:
        scan = self._get_scan_normalised()

        state = self._gazebo_sim.get_model_state()
        pos = state.pose.position
        ori = state.pose.orientation

        yaw = _quat_to_yaw(ori.x, ori.y, ori.z, ori.w)
        dx = self._goal_pos[0] - pos.x
        dy = self._goal_pos[1] - pos.y
        dist = math.sqrt(dx * dx + dy * dy)
        dist_norm = float(np.clip(dist / self._max_dist, 0.0, 1.0))

        angle_world = math.atan2(dy, dx)
        angle_robot = _wrap_angle(angle_world - yaw)
        angle_norm = float(np.clip(angle_robot / math.pi, -1.0, 1.0))

        return np.concatenate([scan, [dist_norm, angle_norm]]).astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _pub_cmd(self, linear: float, angular: float):
        if self._cmd_vel_pub is None:
            return
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self._cmd_vel_pub.publish(msg)

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

def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle
