"""
RL-policy controller — drop-in replacement for the classical planners in the
benchmark.

Loads a Stable Baselines 3 model saved by `train_sac.py` or `train_ppo.py`
and exposes the same `reset / set_goal / compute_action` interface as
MPPIController / DWAController.

Three things are auto-detected from the checkpoint file so you never need to
specify them manually:

  * Algorithm: SAC or PPO — detected by peeking inside the zip before loading.
  * Obs format Dict (current) vs Box (legacy): detected from the model's
    observation space after loading.
  * Scan stack depth and sector count: read from the loaded obs space shape.

Usage
─────
    from rl.controllers.rl_sac_controller import SACController
    ctrl = SACController(goal_x=gx, goal_y=gy, model_path="results/rl/runs/.../model_final")
    action = ctrl.compute_action(x, y, theta, ranges, ang_min, ang_inc, max_r)

In the benchmark
────────────────
    python3 -m benchmark.run_benchmark --planners rl_sac \\
        --rl-model results/rl/runs/RUN_ID/model_final
"""

import math
import os
import zipfile
from collections import deque
from typing import Optional

import numpy as np

try:
    from stable_baselines3 import SAC, PPO
    import gymnasium as gym
    _HAS_SB3 = True
except ImportError as _e:   # pragma: no cover
    _HAS_SB3 = False
    _IMPORT_ERROR = _e

from configs.rl_config import RLEnvConfig
from configs.config import MoboroboConfig, RobotConfig, SimConfig, WorldConfig


_SCAN_MAX_RANGE = {
    "jackal":   10.0,
    "moborobo": MoboroboConfig.LIDAR_MAX_RANGE,
}
_MAX_LINEAR = {
    "jackal":   RobotConfig.MAX_LINEAR_VEL,
    "moborobo": MoboroboConfig.MAX_LINEAR_VEL,
}
_MAX_ANGULAR = {
    "jackal":   RobotConfig.MAX_ANGULAR_VEL,
    "moborobo": MoboroboConfig.MAX_ANGULAR_VEL,
}


def _detect_algo(zip_path: str):
    """Return the SB3 algorithm class (SAC or PPO) stored in the checkpoint.

    SB3 zip files contain a `data` entry pickled with cloudpickle.  The data
    dict always has a `n_steps` key for PPO (on-policy rollout length) but not
    for SAC, so we use that as the discriminator rather than parsing the class
    name (which would require cloudpickle).
    """
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            # PPO checkpoints always contain "n_steps" in their raw bytes
            # (the key is short ASCII, survives pickle encoding).
            if "data" in names:
                raw = zf.read("data")
                if b"n_steps" in raw and b"tau" not in raw:
                    return PPO
    except Exception:
        pass
    return SAC


def _resolve_model_path(path: Optional[str]) -> str:
    """Accept paths with or without the trailing `.zip`. Try `model_final` as a
    fallback when `path` is a run directory."""
    if path is None:
        raise FileNotFoundError(
            "SACController requires a model_path (set via --rl-model in the benchmark)."
        )
    if os.path.isdir(path):
        for cand in ("model_final.zip", "model_final"):
            p = os.path.join(path, cand)
            if os.path.exists(p):
                return p
        # fall back to the most recent checkpoint in the dir
        ckpts = sorted(
            f for f in os.listdir(path)
            if f.startswith("model_") and f.endswith(".zip")
        )
        if ckpts:
            return os.path.join(path, ckpts[-1])
        raise FileNotFoundError("No SAC checkpoint in %s" % path)
    if not path.endswith(".zip") and not os.path.exists(path):
        if os.path.exists(path + ".zip"):
            return path + ".zip"
    return path


class SACController:
    """
    Wraps a trained SAC policy as a planner with the controller interface.

    The policy is deterministic at inference time (no exploration noise).

    Parameters
    ----------
    goal_x, goal_y : float
        World-frame goal position. Used to compute the goal-relative features
        in the observation.
    model_path : str
        Path to the SB3 SAC zip (or a run directory — `model_final.zip` is
        picked automatically).
    robot : str
        "jackal" or "moborobo" — chooses the LiDAR max range used for
        normalisation.  Defaults to SimConfig.ROBOT.
    deterministic : bool
        If True (default), use the policy mean; if False, sample.
    """

    def __init__(
        self,
        goal_x: float,
        goal_y: float,
        model_path: Optional[str] = None,
        robot: Optional[str] = None,
        deterministic: bool = True,
    ):
        if not _HAS_SB3:
            raise ImportError(
                "SACController requires stable-baselines3. Install with:\n"
                "    pip install stable-baselines3\n"
                "Original error: %r" % _IMPORT_ERROR
            )

        if model_path is None:
            model_path = os.environ.get("RL_SAC_MODEL")
        path = _resolve_model_path(model_path)
        algo_cls = _detect_algo(path)
        print("[SACController] detected algorithm: %s" % algo_cls.__name__)
        self._model = algo_cls.load(path, device="auto")

        self._robot = robot or SimConfig.ROBOT
        if self._robot not in _SCAN_MAX_RANGE:
            raise ValueError("Unknown robot %r" % self._robot)
        self._scan_max_range = _SCAN_MAX_RANGE[self._robot]
        self._max_linear = _MAX_LINEAR[self._robot]
        self._max_angular = _MAX_ANGULAR[self._robot]

        # ── Detect obs format from the checkpoint itself ──────────────────────
        obs_space = self._model.observation_space
        self._dict_obs = isinstance(obs_space, gym.spaces.Dict)
        self._scan_mem = None
        if self._dict_obs:
            # Shapes come from the model, so a checkpoint trained with
            # non-default stack/sector counts still works.
            rows, self._num_scan_rays = obs_space["scan"].shape
            # Scan memory appends one channel (the "directly measured this
            # step" mask), so a memory-trained checkpoint has SCAN_STACK + 1
            # rows.  Infer it from the shape rather than trusting the current
            # config, which may have moved on since the checkpoint was saved.
            self._use_memory = rows == RLEnvConfig.SCAN_STACK + 1
            self._scan_stack = rows - 1 if self._use_memory else rows
            self._scan_norm_range = min(
                RLEnvConfig.SCAN_CLIP_RANGE, self._scan_max_range
            )
            self._scan_history: deque = deque(maxlen=self._scan_stack)
            if self._use_memory:
                from rl.env.scan_memory import ScanMemory
                self._scan_mem = ScanMemory(
                    num_sectors=self._num_scan_rays,
                    blind_radius=RLEnvConfig.SCAN_BLIND_RADIUS,
                    mem_radius=RLEnvConfig.SCAN_MEM_RADIUS,
                    voxel=RLEnvConfig.SCAN_MEM_VOXEL,
                    max_age_steps=RLEnvConfig.SCAN_MEM_MAX_AGE,
                )
                print("[SACController] near-field scan memory ON "
                      "(%d sectors, blind radius %.2f m)"
                      % (self._num_scan_rays, RLEnvConfig.SCAN_BLIND_RADIUS))
        else:
            self._use_memory = False
            # Legacy flat obs: [scan(N), dist, sin, cos, last_lin, last_ang]
            self._num_scan_rays = obs_space.shape[0] - 5
            self._scan_norm_range = self._scan_max_range
            self._scan_history = deque(maxlen=1)

        self._goal = np.array([goal_x, goal_y], dtype=np.float64)
        self._deterministic = bool(deterministic)

        # Used to normalise the goal-distance feature, same as BarnEnv:
        # max possible distance is the start→goal straight line.
        ix, iy, _ = WorldConfig.STATIC_INIT_POSITION
        gx_default = ix + WorldConfig.STATIC_GOAL_OFFSET[0]
        gy_default = iy + WorldConfig.STATIC_GOAL_OFFSET[1]
        self._max_dist = float(math.hypot(gx_default - ix, gy_default - iy))

        # The policy network is stateless, but the observation embeds the
        # previous action — track it across compute_action() calls.
        self._last_action = np.zeros(2, dtype=np.float32)

    # ── API ───────────────────────────────────────────────────────────────────

    # Tells BenchmarkEnv to hand over the NaN-preserving scan: this controller
    # needs to know which rays gave no return so ScanMemory can fill the
    # near-field blind ring instead of reading it as open space.
    @property
    def wants_raw_scan(self) -> bool:
        return self._scan_mem is not None

    def reset(self):
        self._last_action[:] = 0.0
        self._scan_history.clear()
        if self._scan_mem is not None:
            self._scan_mem.reset()

    def set_goal(self, goal_x: float, goal_y: float):
        self._goal[:] = [goal_x, goal_y]

    def compute_action(
        self,
        robot_x: float,
        robot_y: float,
        robot_theta: float,
        scan_ranges,
        scan_angle_min: float,
        scan_angle_increment: float,
        scan_max_range: float,
    ) -> np.ndarray:
        obs = self._build_obs(
            robot_x, robot_y, robot_theta,
            scan_ranges, scan_angle_min, scan_angle_increment,
        )
        action, _ = self._model.predict(obs, deterministic=self._deterministic)
        action = np.asarray(action, dtype=np.float32)
        self._last_action[:] = action
        return action

    # ── Observation reconstruction (mirrors BarnEnv._build_obs) ───────────────

    def _scan_frame(self, scan_ranges, x, y, theta, angle_min, angle_increment):
        """One normalised scan frame at the training-time sector count.

        Returns (frame, fresh) where `fresh` is None unless scan memory is on.
        """
        scan = np.asarray(scan_ranges, dtype=np.float32)
        if scan.size == 0:
            scan = np.full(self._num_scan_rays, self._scan_max_range, dtype=np.float32)

        if self._scan_mem is not None:
            # Identical fusion to BarnEnv: NaN rays are filled from the
            # egocentric point store, so a sector inside the blind ring
            # reports its remembered range rather than max range.
            fused, fresh = self._scan_mem.update(
                x, y, theta, scan, angle_min, angle_increment,
                self._scan_max_range,
            )
            frame = np.clip(fused / self._scan_norm_range, 0.0, 1.0)
            return frame.astype(np.float32), fresh.astype(np.float32)

        # No memory: NaN means "no return", which without a store is
        # indistinguishable from open space.
        scan = np.where(np.isfinite(scan), scan, self._scan_max_range)
        if self._dict_obs:
            from rl.env.barn_env import _min_pool
            scan = _min_pool(scan, self._num_scan_rays)
        elif scan.size != self._num_scan_rays:
            # legacy checkpoints were trained on linear-interp resampling
            src_idx = np.linspace(0, scan.size - 1, self._num_scan_rays)
            scan = np.interp(src_idx, np.arange(scan.size), scan).astype(np.float32)

        # Normalise using the policy's training-time range (NOT the
        # ROS-provided max_range, which may differ between scan sources).
        return np.clip(scan / self._scan_norm_range, 0.0, 1.0).astype(np.float32), None

    def _build_obs(self, robot_x, robot_y, robot_theta, scan_ranges,
                   angle_min=-math.pi, angle_increment=0.0087):
        frame, fresh = self._scan_frame(
            scan_ranges, robot_x, robot_y, robot_theta, angle_min, angle_increment
        )

        dx = float(self._goal[0]) - robot_x
        dy = float(self._goal[1]) - robot_y
        dist = math.hypot(dx, dy)
        dist_norm = float(np.clip(dist / self._max_dist, 0.0, 1.0))

        angle_robot = _wrap_angle(math.atan2(dy, dx) - robot_theta)
        sin_a = math.sin(angle_robot)
        cos_a = math.cos(angle_robot)

        # [-1, 1]: reverse-capable policies emit negative linear velocity;
        # forward-only checkpoints never do, so this is identical for them.
        last_lin_norm = float(np.clip(self._last_action[0] / self._max_linear, -1.0, 1.0))
        last_ang_norm = float(np.clip(self._last_action[1] / self._max_angular, -1.0, 1.0))

        state = [dist_norm, sin_a, cos_a, last_lin_norm, last_ang_norm]

        if not self._dict_obs:
            return np.concatenate([frame, state]).astype(np.float32)

        # Maintain the frame stack exactly like BarnEnv: seed with K copies
        # on the first frame after reset(), append afterwards.
        if not self._scan_history:
            for _ in range(self._scan_stack):
                self._scan_history.append(frame)
        else:
            self._scan_history.append(frame)

        rows = list(self._scan_history)
        if fresh is not None:
            rows.append(fresh)

        return {
            "scan": np.stack(rows).astype(np.float32),
            "state": np.array(state, dtype=np.float32),
        }


def _wrap_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a
