"""
DWA (Dynamic Window Approach) controller for 2D differential-drive navigation.

Pure numpy — no ROS, no gymnasium dependency. Same call signature as MPPIController
so it drops in directly to the benchmark loop.

Algorithm
─────────
1. Compute the dynamic window: feasible (v, ω) pairs reachable within one DT given
   current velocity and acceleration limits.
2. Sample N_V × N_W combinations in the window.
3. Simulate each trajectory forward SIM_TIME seconds using unicycle dynamics.
4. Score each trajectory:
       score = W_HEADING   × heading_score   (alignment with goal at trajectory end)
             + W_CLEARANCE × clearance_score (min dist to obstacles, capped)
             + W_VELOCITY  × velocity_score  (prefer faster forward motion)
   Trajectories that pass within ROBOT_RADIUS of any obstacle are rejected (−∞).
5. Return [v, ω] of the best trajectory. If every trajectory collides, rotate in place.
"""

import math
from typing import Optional

import numpy as np


class DWAController:
    """
    DWA controller for unicycle/differential-drive robots.

    Parameters
    ----------
    goal_x, goal_y : float
        World-frame goal position.
    config : class, optional
        Configuration class (defaults to DWAConfig).
    """

    def __init__(self, goal_x: float, goal_y: float, config=None):
        if config is None:
            from configs.dwa_config import DWAConfig
            config = DWAConfig
        self.cfg = config
        self._goal = np.array([goal_x, goal_y], dtype=np.float64)
        self._curr_v = 0.0
        self._curr_w = 0.0

    # ── Public API ─────────────────────────────────────────────────────────────

    def reset(self):
        self._curr_v = 0.0
        self._curr_w = 0.0

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
        """
        Run one DWA iteration and return [linear_x, angular_z].

        Parameters match MPPIController.compute_action exactly.
        """
        cfg = self.cfg

        obstacles = self._scan_to_obstacles(
            scan_ranges, scan_angle_min, scan_angle_increment,
            scan_max_range, robot_x, robot_y, robot_theta,
        )

        # ── Dynamic window ──────────────────────────────────────────────────────
        v_lo = max(0.0, self._curr_v - cfg.MAX_ACC_V * cfg.DT)
        v_hi = min(cfg.MAX_V, self._curr_v + cfg.MAX_ACC_V * cfg.DT)
        w_lo = max(-cfg.MAX_W, self._curr_w - cfg.MAX_ACC_W * cfg.DT)
        w_hi = min(cfg.MAX_W, self._curr_w + cfg.MAX_ACC_W * cfg.DT)

        vs = np.linspace(v_lo, v_hi, cfg.N_V)
        ws = np.linspace(w_lo, w_hi, cfg.N_W)
        VV, WW = np.meshgrid(vs, ws)
        V = VV.ravel()   # (K,)
        W = WW.ravel()   # (K,)
        K = len(V)

        # ── Vectorised rollout ──────────────────────────────────────────────────
        n_steps = max(1, int(round(cfg.SIM_TIME / cfg.DT)))
        x = np.full(K, robot_x)
        y = np.full(K, robot_y)
        theta = np.full(K, robot_theta)
        positions = np.empty((K, n_steps, 2), dtype=np.float64)

        for t in range(n_steps):
            x = x + V * np.cos(theta) * cfg.DT
            y = y + V * np.sin(theta) * cfg.DT
            theta = theta + W * cfg.DT
            positions[:, t, 0] = x
            positions[:, t, 1] = y

        final_theta = theta   # (K,)

        # ── Heading score ───────────────────────────────────────────────────────
        dx = self._goal[0] - x
        dy = self._goal[1] - y
        goal_bearing = np.arctan2(dy, dx)
        heading_err = np.abs(
            np.arctan2(
                np.sin(goal_bearing - final_theta),
                np.cos(goal_bearing - final_theta),
            )
        )
        heading_score = (np.pi - heading_err) / np.pi   # (K,) in [0, 1]

        # ── Velocity score ──────────────────────────────────────────────────────
        vel_score = V / max(cfg.MAX_V, 1e-9)   # (K,) in [0, 1]

        # ── Clearance score (and collision filter) ──────────────────────────────
        if len(obstacles) > 0:
            diff = (
                positions[:, :, np.newaxis, :]             # (K, T, 1, 2)
                - obstacles[np.newaxis, np.newaxis, :, :]  # (1, 1, N, 2)
            )
            dists = np.sqrt((diff ** 2).sum(-1))           # (K, T, N)
            min_dists = dists.min(axis=(1, 2))             # (K,)
            colliding = min_dists < cfg.ROBOT_RADIUS
            clearance = np.clip(min_dists - cfg.ROBOT_RADIUS, 0.0, cfg.CLEARANCE_MAX)
            clear_score = clearance / max(cfg.CLEARANCE_MAX, 1e-9)
        else:
            colliding = np.zeros(K, dtype=bool)
            clear_score = np.ones(K, dtype=np.float64)

        # ── Combined score ──────────────────────────────────────────────────────
        scores = (
            cfg.W_HEADING   * heading_score
            + cfg.W_CLEARANCE * clear_score
            + cfg.W_VELOCITY  * vel_score
        )
        scores[colliding] = -np.inf

        if np.all(np.isinf(scores)):
            # Every trajectory collides — rotate in place to escape
            best_v = 0.0
            best_w = cfg.MAX_W * 0.5
        else:
            best_idx = int(np.argmax(scores))
            best_v = float(V[best_idx])
            best_w = float(W[best_idx])

        self._curr_v = best_v
        self._curr_w = best_w
        return np.array([best_v, best_w], dtype=np.float32)

    # ── Internal helpers ────────────────────────────────────────────────────────

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
        """LaserScan → world-frame obstacle point cloud (same logic as MPPIController)."""
        cfg = self.cfg
        ranges = np.asarray(ranges, dtype=np.float64)
        n = len(ranges)
        idx = np.arange(0, n, cfg.SCAN_SUBSAMPLE)
        r = ranges[idx]
        a = angle_min + idx * angle_increment
        valid = np.isfinite(r) & (r > 0.01) & (r < max_range * 0.99)
        if not valid.any():
            return np.zeros((0, 2), dtype=np.float64)
        r, a = r[valid], a[valid]
        cos_t = math.cos(robot_theta)
        sin_t = math.sin(robot_theta)
        ox_r = r * np.cos(a)
        oy_r = r * np.sin(a)
        ox_w = robot_x + cos_t * ox_r - sin_t * oy_r
        oy_w = robot_y + sin_t * ox_r + cos_t * oy_r
        return np.stack([ox_w, oy_w], axis=1)
