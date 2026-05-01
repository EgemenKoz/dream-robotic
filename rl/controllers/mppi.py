"""
Model Predictive Path Integral (MPPI) controller for 2D differential-drive
navigation in the BARN simulation.

Pure numpy — no ROS, no PyTorch, no gymnasium dependency.
Designed to be called once per control step (same rate as BarnEnv.step_duration).

Algorithm overview
──────────────────
  1. Sample K perturbations ε ~ N(0, Σ) around the nominal control sequence U.
  2. Form perturbed sequences V = U + ε and rollout K trajectories using the
     unicycle dynamics model.
  3. Evaluate cost S_k for each trajectory:
       S_k = W_OBSTACLE × Σ_t obstacle_cost(x_t^k)
           + W_GOAL     × ||x_T^k − goal||
           + W_GOAL_STEP× mean_t ||x_t^k − goal||
           + LAMBDA     × Σ_t (u_t · Σ^{-1} · ε_{k,t})   ← MPPI regulariser
  4. Compute softmax weights:  w_k = exp(−(S_k − min S) / LAMBDA) / Z
  5. Update nominal sequence:  U ← Σ_k w_k · V_k
  6. Apply U[0], shift U left (receding horizon).

Usage with BarnEnv
──────────────────
    from rl import BarnEnv
    from rl.controllers import MPPIController
    from configs.config import WorldConfig

    env = BarnEnv(robot="moborobo", world_idx=30, managed=False)
    gx, gy = env.goal_pos       # exposes goal position
    ctrl = MPPIController(goal_x=gx, goal_y=gy)

    obs, info = env.reset()
    ctrl.reset()

    for _ in range(500):
        ranges, angle_min, angle_inc, max_r = env.get_scan_info()
        x, y, theta = env.get_pose()
        action = ctrl.compute_action(x, y, theta, ranges, angle_min, angle_inc, max_r)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
            ctrl.reset()

    env.close()

Usage standalone (no gym env, raw ROS data)
───────────────────────────────────────────
    import rospy, numpy as np
    from sensor_msgs.msg import LaserScan
    from gazebo.gazebo_simulation import GazeboSimulation
    from rl.controllers import MPPIController

    sim = GazeboSimulation(init_position=[-2.25, 3, 1.57])
    ctrl = MPPIController(goal_x=-2.25, goal_y=13.0)

    def step():
        scan = rospy.wait_for_message("/scan", LaserScan)
        state = sim.get_model_state()
        x, y = state.pose.position.x, state.pose.position.y
        from rl.env.barn_env import _quat_to_yaw
        ori = state.pose.orientation
        theta = _quat_to_yaw(ori.x, ori.y, ori.z, ori.w)
        return ctrl.compute_action(
            x, y, theta,
            scan.ranges, scan.angle_min, scan.angle_increment, scan.range_max,
        )
"""

import math
from typing import Optional, Tuple, Type

import numpy as np


class MPPIController:
    """
    MPPI controller for unicycle/differential-drive robots.

    Parameters
    ----------
    goal_x, goal_y : float
        World-frame goal position.
    config : class, optional
        Configuration class with MPPI parameters (defaults to MPPIConfig).
        Pass a subclass or modified class to override individual parameters.
    seed : int, optional
        Random seed for reproducible sampling.
    """

    def __init__(
        self,
        goal_x: float,
        goal_y: float,
        config=None,
        seed: Optional[int] = None,
    ):
        if config is None:
            from configs.mppi_config import MPPIConfig
            config = MPPIConfig

        self.cfg = config
        self._goal = np.array([goal_x, goal_y], dtype=np.float64)
        self._rng = np.random.default_rng(seed)

        # Nominal control sequence — initialised to zeros (stationary)
        self._nom_v = np.zeros(config.T, dtype=np.float64)
        self._nom_w = np.zeros(config.T, dtype=np.float64)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self):
        """Reset nominal control sequence to zero.  Call at every episode start."""
        self._nom_v[:] = 0.0
        self._nom_w[:] = 0.0

    def set_goal(self, goal_x: float, goal_y: float):
        """Update goal mid-episode (e.g. for multi-waypoint navigation)."""
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
        """
        Run one MPPI iteration and return the optimal [linear_x, angular_z] action.

        Parameters
        ----------
        robot_x, robot_y, robot_theta : float
            Current robot pose in the world frame.
        scan_ranges : array-like
            Raw LaserScan range values (metres).
        scan_angle_min : float
            Start angle of the scan (radians).
        scan_angle_increment : float
            Angle step between consecutive scan rays (radians).
        scan_max_range : float
            Maximum valid scan range (metres).

        Returns
        -------
        np.ndarray, shape (2,), dtype float32
            [linear_x (m/s), angular_z (rad/s)] clipped to robot limits.
        """
        cfg = self.cfg
        K, T = cfg.K, cfg.T

        # ── Build obstacle cloud in world frame ───────────────────────────────
        obstacles = self._scan_to_obstacles(
            scan_ranges, scan_angle_min, scan_angle_increment,
            scan_max_range, robot_x, robot_y, robot_theta,
        )

        # ── Sample perturbations ──────────────────────────────────────────────
        eps_v = self._rng.normal(0.0, cfg.SIGMA_V, (K, T))   # (K, T)
        eps_w = self._rng.normal(0.0, cfg.SIGMA_W, (K, T))

        # Perturbed controls — forward velocity is non-negative
        V = np.clip(self._nom_v[np.newaxis] + eps_v, 0.0, cfg.MAX_V)   # (K, T)
        W = np.clip(self._nom_w[np.newaxis] + eps_w, -cfg.MAX_W, cfg.MAX_W)

        # ── Rollout K trajectories ────────────────────────────────────────────
        positions, final_x, final_y = self._rollout(robot_x, robot_y, robot_theta, V, W)

        # ── Compute trajectory costs ──────────────────────────────────────────
        S = np.zeros(K, dtype=np.float64)

        if len(obstacles) > 0:
            S += self._obstacle_cost(positions, obstacles) * cfg.W_OBSTACLE

        # Terminal goal cost
        terminal_dist = np.sqrt((final_x - self._goal[0]) ** 2 + (final_y - self._goal[1]) ** 2)
        S += terminal_dist * cfg.W_GOAL

        # Per-step goal cost (dense: encourages progress throughout horizon)
        step_dist = np.sqrt(
            (positions[:, :, 0] - self._goal[0]) ** 2 +
            (positions[:, :, 1] - self._goal[1]) ** 2
        ).mean(axis=1)   # (K,)
        S += step_dist * cfg.W_GOAL_STEP

        # MPPI control regularisation term: λ · u^T Σ^{-1} ε
        reg = (
            (self._nom_v[np.newaxis] / (cfg.SIGMA_V ** 2)) * eps_v +
            (self._nom_w[np.newaxis] / (cfg.SIGMA_W ** 2)) * eps_w
        ).sum(axis=1)   # (K,)
        S += cfg.LAMBDA * reg

        # ── Softmax weight update ─────────────────────────────────────────────
        beta = S.min()
        w = np.exp(-(S - beta) / cfg.LAMBDA)
        w /= w.sum() + 1e-10

        # ── Update nominal sequence ───────────────────────────────────────────
        self._nom_v = np.clip(
            (w[:, np.newaxis] * V).sum(axis=0), 0.0, cfg.MAX_V
        )
        self._nom_w = np.clip(
            (w[:, np.newaxis] * W).sum(axis=0), -cfg.MAX_W, cfg.MAX_W
        )

        # ── Extract first action and shift sequence ───────────────────────────
        action = np.array([self._nom_v[0], self._nom_w[0]], dtype=np.float32)
        self._nom_v = np.roll(self._nom_v, -1)
        self._nom_w = np.roll(self._nom_w, -1)

        return action

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _rollout(
        self,
        x0: float,
        y0: float,
        theta0: float,
        V: np.ndarray,
        W: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorised unicycle rollout for K trajectories.

        Parameters
        ----------
        V, W : np.ndarray of shape (K, T)
            Linear and angular velocity sequences.

        Returns
        -------
        positions : np.ndarray, shape (K, T, 2)
            (x, y) at each step for each trajectory.
        final_x, final_y : np.ndarray, shape (K,)
            Terminal positions.
        """
        K, T = V.shape
        dt = self.cfg.DT

        x = np.full(K, x0)
        y = np.full(K, y0)
        theta = np.full(K, theta0)
        positions = np.empty((K, T, 2), dtype=np.float64)

        for t in range(T):
            x = x + V[:, t] * np.cos(theta) * dt
            y = y + V[:, t] * np.sin(theta) * dt
            theta = theta + W[:, t] * dt
            positions[:, t, 0] = x
            positions[:, t, 1] = y

        return positions, x, y

    def _obstacle_cost(
        self,
        positions: np.ndarray,
        obstacles: np.ndarray,
    ) -> np.ndarray:
        """
        Exponential obstacle cost summed over the planning horizon.

        Parameters
        ----------
        positions : np.ndarray, shape (K, T, 2)
        obstacles : np.ndarray, shape (N, 2)

        Returns
        -------
        np.ndarray, shape (K,) — cost per trajectory.
        """
        cfg = self.cfg
        # (K, T, N) pairwise distances via broadcasting
        diff = (
            positions[:, :, np.newaxis, :]          # (K, T, 1, 2)
            - obstacles[np.newaxis, np.newaxis, :, :]  # (1, 1, N, 2)
        )
        dists = np.sqrt((diff ** 2).sum(axis=-1))    # (K, T, N)
        min_dists = dists.min(axis=-1)               # (K, T)

        clearance = np.maximum(min_dists - cfg.ROBOT_RADIUS, 0.0)
        cost_per_step = np.exp(-clearance / cfg.OBS_SIGMA)  # (K, T)
        return cost_per_step.sum(axis=1)             # (K,)

    def _scan_to_obstacles(
        self,
        ranges,
        angle_min: float,
        angle_increment: float,
        max_range: float,
        robot_x: float,
        robot_y: float,
        robot_theta: float,
    ) -> np.ndarray:
        """
        Convert LaserScan to a world-frame obstacle point cloud.

        Returns np.ndarray of shape (N, 2), or shape (0, 2) if scan is empty.
        """
        cfg = self.cfg
        ranges = np.asarray(ranges, dtype=np.float64)
        n = len(ranges)

        # Subsample scan rays
        idx = np.arange(0, n, cfg.SCAN_SUBSAMPLE)
        r = ranges[idx]
        a = angle_min + idx * angle_increment

        # Discard invalid rays
        valid = np.isfinite(r) & (r > 0.01) & (r < max_range * 0.99)
        if not valid.any():
            return np.zeros((0, 2), dtype=np.float64)

        r = r[valid]
        a = a[valid]

        # Obstacle positions in robot frame
        ox_r = r * np.cos(a)
        oy_r = r * np.sin(a)

        # Rotate to world frame
        cos_t = math.cos(robot_theta)
        sin_t = math.sin(robot_theta)
        ox_w = robot_x + cos_t * ox_r - sin_t * oy_r
        oy_w = robot_y + sin_t * ox_r + cos_t * oy_r

        return np.stack([ox_w, oy_w], axis=1)
