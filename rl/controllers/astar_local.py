"""
A* global planner + pure-pursuit local controller (online mapping version).

Global plan
───────────
Builds an occupancy grid incrementally from LiDAR scans (OccupancyGrid from
online_planner.py) and reruns A* every REPLAN_STEPS control steps.  No static
c-space or path files are loaded — the controller uses only pose + goal + LiDAR,
matching the BARN challenge evaluation conditions.

Local execution
───────────────
Pure-pursuit: steers toward the point LOOKAHEAD_DIST ahead on the current path.
A frontal-sector scan check slows the robot when obstacles are unexpectedly close.
"""

import math
from typing import Optional

import numpy as np

from configs.config import NavConfig, WorldConfig
from rl.controllers.online_planner import OccupancyGrid, _astar


class AStarLocalController:
    """
    Online A* global planner + pure-pursuit local controller.

    Parameters
    ----------
    goal_x, goal_y : float
        World-frame goal position.
    config : class, optional
        Configuration class (defaults to AStarLocalConfig).
    """

    REPLAN_STEPS  = 30    # replan every N control steps
    MIN_SCANS     = 3     # minimum new-obstacle events before first replan

    def __init__(self, goal_x: float, goal_y: float, config=None):
        if config is None:
            from configs.astar_local_config import AStarLocalConfig
            config = AStarLocalConfig
        self.cfg = config

        self._goal  = np.array([goal_x, goal_y], dtype=np.float64)
        init = WorldConfig.STATIC_INIT_POSITION
        self._start = np.array([init[0], init[1]], dtype=np.float64)

        self._grid      = OccupancyGrid()
        self._waypoints = np.array([self._start, self._goal], dtype=np.float64)
        self._wpt_idx   = 0
        self._step      = 0
        self._obs_count = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def reset(self):
        self._grid.reset()
        self._waypoints = np.array([self._start, self._goal], dtype=np.float64)
        self._wpt_idx   = 0
        self._step      = 0
        self._obs_count = 0

    def set_goal(self, goal_x: float, goal_y: float):
        self._goal[:] = [goal_x, goal_y]
        self._waypoints[-1] = self._goal.copy()

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
        self._step += 1

        # Update occupancy grid.
        self._obs_count += self._grid.update(
            robot_x, robot_y, robot_theta,
            scan_ranges, scan_angle_min, scan_angle_increment, scan_max_range,
        )

        # Replan when enough obstacles have been observed and on schedule.
        if (self._step == 1 and self._obs_count >= self.MIN_SCANS) or (
            self._step % self.REPLAN_STEPS == 0 and self._obs_count >= self.MIN_SCANS
        ):
            self._replan(robot_x, robot_y)

        pos = np.array([robot_x, robot_y], dtype=np.float64)

        # Advance waypoint index.
        while self._wpt_idx < len(self._waypoints) - 1:
            if np.linalg.norm(pos - self._waypoints[self._wpt_idx]) < self.cfg.WAYPOINT_REACH:
                self._wpt_idx += 1
            else:
                break

        target = self._lookahead_point(pos)

        dx  = target[0] - robot_x
        dy  = target[1] - robot_y
        err = _wrap(math.atan2(dy, dx) - robot_theta)

        cfg = self.cfg
        v   = cfg.MAX_V * max(0.0, 1.0 - abs(err) / math.pi * cfg.DECEL)
        v   = max(cfg.MIN_V, v)
        w   = float(np.clip(cfg.K_HEADING * err, -cfg.MAX_W, cfg.MAX_W))

        safety = self._frontal_safety(
            scan_ranges, scan_angle_min, scan_angle_increment
        )
        v *= safety

        return np.array([v, w], dtype=np.float32)

    # ── Planning ────────────────────────────────────────────────────────────────

    def _replan(self, robot_x: float, robot_y: float):
        path = self._grid.plan((robot_x, robot_y), tuple(self._goal))
        if path is not None and len(path) > 0:
            self._waypoints = path
            self._wpt_idx   = 0

    # ── Pure pursuit helpers ────────────────────────────────────────────────────

    def _lookahead_point(self, pos: np.ndarray) -> np.ndarray:
        remaining = self.cfg.LOOKAHEAD_DIST
        idx = self._wpt_idx
        while idx < len(self._waypoints) - 1:
            seg     = self._waypoints[idx + 1] - self._waypoints[idx]
            seg_len = float(np.linalg.norm(seg))
            if seg_len < 1e-9:
                idx += 1
                continue
            if remaining <= seg_len:
                return self._waypoints[idx] + (remaining / seg_len) * seg
            remaining -= seg_len
            idx += 1
        return self._waypoints[-1].copy()

    def _frontal_safety(
        self, ranges, angle_min: float, angle_inc: float
    ) -> float:
        cfg    = self.cfg
        ranges = np.asarray(ranges, dtype=np.float64)
        angles = angle_min + np.arange(len(ranges)) * angle_inc
        front  = np.abs(angles) < cfg.FRONT_SECTOR
        valid  = front & np.isfinite(ranges) & (ranges > 0.01)
        if not valid.any():
            return 1.0
        min_r = float(ranges[valid].min())
        if min_r < cfg.OBSTACLE_STOP_DIST:
            return 0.0
        if min_r < cfg.OBSTACLE_SLOW_DIST:
            return (min_r - cfg.OBSTACLE_STOP_DIST) / (
                cfg.OBSTACLE_SLOW_DIST - cfg.OBSTACLE_STOP_DIST
            )
        return 1.0


def _wrap(angle: float) -> float:
    while angle >  math.pi: angle -= 2 * math.pi
    while angle < -math.pi: angle += 2 * math.pi
    return angle
