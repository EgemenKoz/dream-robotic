"""
Elastic Band controller for 2D differential-drive navigation.

Implements the Elastic Band method (Quinlan & Khatib, ICRA 1993): a global path
is represented as a chain of overlapping circular "bubbles" of free space and
deformed online under two forces until it reaches equilibrium —

  * internal contraction — neighbouring bubbles pull on each other, shortening
    and straightening the band (like tension in an elastic).
  * external repulsion    — bubbles inside the clearance threshold are pushed
    away from the nearest obstacle.

A pure-pursuit tracker then follows the smoothed band.

Architecture
────────────
This is a self-contained planner+controller, a sibling to OnlineGlobalPlanner:

  * Reuses OccupancyGrid + A* (from online_planner) to obtain / refresh the
    *global* path that seeds the band.
  * Uses the *current* LiDAR scan (world-frame obstacle cloud) for bubble
    clearance and the repulsion force, so the band reacts to local geometry
    between replans.

Same call signature as MPPIController / DWAController, so it drops straight into
the benchmark loop:

    ctrl = ElasticBandController(goal_x=gx, goal_y=gy)
    action = ctrl.compute_action(x, y, theta, ranges, ang_min, ang_inc, max_r)
"""

import math
from typing import List, Optional

import numpy as np

from rl.controllers.online_planner import OccupancyGrid


class ElasticBandController:
    """
    Elastic Band planner + pure-pursuit tracker for unicycle robots.

    Parameters
    ----------
    goal_x, goal_y : float
        World-frame goal position.
    config : class, optional
        Configuration class (defaults to ElasticBandConfig).
    """

    def __init__(self, goal_x: float, goal_y: float, config=None):
        if config is None:
            from configs.elastic_band_config import ElasticBandConfig
            config = ElasticBandConfig
        self.cfg = config

        self._goal = np.array([goal_x, goal_y], dtype=np.float64)
        self._grid = OccupancyGrid()

        # Band state: parallel arrays of bubble centres (M, 2) and radii (M,).
        self._band: Optional[np.ndarray] = None
        self._radii: Optional[np.ndarray] = None

        # Fresh obstacle cloud (world frame) rebuilt from each scan.
        self._obstacles = np.zeros((0, 2), dtype=np.float64)

        self._step = 0
        self._obs_count = 0

    # ── Public API (mirrors local controller interface) ────────────────────────

    def reset(self):
        self._grid.reset()
        self._band = None
        self._radii = None
        self._obstacles = np.zeros((0, 2), dtype=np.float64)
        self._step = 0
        self._obs_count = 0

    def set_goal(self, goal_x: float, goal_y: float):
        self._goal[:] = [goal_x, goal_y]
        self._band = None  # force a rebuild on the next step

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
        cfg = self.cfg
        self._step += 1

        # Integrate scan into the grid (for global A*) and refresh the local cloud.
        self._obs_count += self._grid.update(
            robot_x, robot_y, robot_theta,
            scan_ranges, scan_angle_min, scan_angle_increment, scan_max_range,
        )
        self._obstacles = self._scan_to_obstacles(
            scan_ranges, scan_angle_min, scan_angle_increment,
            scan_max_range, robot_x, robot_y, robot_theta,
        )

        # (Re)seed the band from a fresh A* path on schedule.
        need_plan = (
            (self._band is None and self._obs_count >= cfg.MIN_SCANS)
            or (self._step % cfg.REPLAN_STEPS == 0 and self._obs_count >= cfg.MIN_SCANS)
        )
        if need_plan:
            self._seed_band(robot_x, robot_y)

        # Anchor the band start at the robot and prune passed bubbles.
        if self._band is not None and len(self._band) >= 2:
            self._band[0] = [robot_x, robot_y]
            self._prune_passed(robot_x, robot_y)
            for _ in range(cfg.DEFORM_ITERS):
                self._deform()
            self._maintain()

        # Track the band with pure pursuit (fall back to the goal if no band).
        target = self._lookahead_target(robot_x, robot_y)
        return self._pursue(robot_x, robot_y, robot_theta, target)

    # ── Band construction ───────────────────────────────────────────────────────

    def _seed_band(self, robot_x: float, robot_y: float):
        path = self._grid.plan((robot_x, robot_y), tuple(self._goal))
        if path is None or len(path) == 0:
            return
        # Ensure the band starts at the robot and ends at the goal.
        pts = [np.array([robot_x, robot_y], dtype=np.float64)]
        for p in path:
            pts.append(np.asarray(p, dtype=np.float64))
        pts.append(self._goal.copy())
        band = np.array(pts, dtype=np.float64)
        self._band = band
        self._radii = self._clearance(band)

    # ── Deformation ───────────────────────────────────────────────────────────

    def _deform(self):
        """One pass of internal + external forces over the interior bubbles."""
        cfg = self.cfg
        band = self._band
        m = len(band)
        if m < 3:
            self._radii = self._clearance(band)
            return

        radii = self._clearance(band)
        new_band = band.copy()

        for i in range(1, m - 1):
            prev, cur, nxt = band[i - 1], band[i], band[i + 1]

            # Internal contraction toward both neighbours (unit vectors).
            d1 = prev - cur
            d2 = nxt - cur
            n1 = float(np.linalg.norm(d1))
            n2 = float(np.linalg.norm(d2))
            if n1 < 1e-9 or n2 < 1e-9:
                continue
            f_int = cfg.K_INTERNAL * (d1 / n1 + d2 / n2)

            # External repulsion when inside the clearance threshold.
            rho = radii[i]
            f_ext = np.zeros(2, dtype=np.float64)
            if rho < cfg.RHO0:
                grad = self._clearance_grad(cur)   # points toward more open space
                f_ext = cfg.K_EXTERNAL * (cfg.RHO0 - rho) * grad

            disp = cfg.STEP_ALPHA * (f_int + f_ext)
            # Clamp per-iteration motion so a bubble can't jump across an obstacle.
            d_norm = float(np.linalg.norm(disp))
            if d_norm > cfg.MAX_BUBBLE_MOVE:
                disp *= cfg.MAX_BUBBLE_MOVE / d_norm
            new_band[i] = cur + disp

        self._band = new_band
        self._radii = self._clearance(new_band)

    # ── Band maintenance: insert / remove bubbles to keep them overlapping ──────

    def _maintain(self):
        cfg = self.cfg
        band = list(self._band)
        radii = list(self._clearance(self._band))

        # Insert bubbles where adjacent ones no longer overlap enough.
        out_pts: List[np.ndarray] = [band[0]]
        for i in range(1, len(band)):
            gap = float(np.linalg.norm(band[i] - band[i - 1]))
            overlap_need = cfg.OVERLAP_INSERT * (radii[i] + radii[i - 1])
            if gap > overlap_need and len(out_pts) < cfg.MAX_BUBBLES:
                out_pts.append(0.5 * (band[i - 1] + band[i]))   # midpoint bubble
            out_pts.append(band[i])

        pts = np.array(out_pts, dtype=np.float64)
        rad = self._clearance(pts)

        # Remove interior bubbles that sit well inside their neighbours' overlap.
        keep = [0]
        for i in range(1, len(pts) - 1):
            gap = float(np.linalg.norm(pts[i + 1] - pts[keep[-1]]))
            overlap_have = cfg.OVERLAP_REMOVE * (rad[i + 1] + rad[keep[-1]])
            if gap < overlap_have:
                continue  # redundant — skip this bubble
            keep.append(i)
        keep.append(len(pts) - 1)

        self._band = pts[keep]
        self._radii = self._clearance(self._band)

    def _prune_passed(self, robot_x: float, robot_y: float):
        """Drop leading bubbles the robot has already passed (keep ≥ 2)."""
        cfg = self.cfg
        pos = np.array([robot_x, robot_y], dtype=np.float64)
        band = self._band
        drop = 0
        # band[0] is the robot anchor; consider band[1:] for pruning.
        while drop + 2 < len(band) and \
                float(np.linalg.norm(band[drop + 1] - pos)) < cfg.WP_REACH:
            drop += 1
        if drop > 0:
            self._band = band[drop:].copy()
            self._band[0] = pos

    # ── Clearance from the obstacle cloud ───────────────────────────────────────

    def _clearance(self, pts: np.ndarray) -> np.ndarray:
        """Distance from each point to the nearest obstacle, clamped to [min,max] R."""
        cfg = self.cfg
        pts = np.atleast_2d(pts)
        if self._obstacles.shape[0] == 0:
            return np.full(len(pts), cfg.MAX_BUBBLE_R, dtype=np.float64)
        # (P, 1, 2) - (1, N, 2) → (P, N)
        diff = pts[:, None, :] - self._obstacles[None, :, :]
        d = np.sqrt((diff ** 2).sum(-1)).min(axis=1)
        return np.clip(d, cfg.MIN_BUBBLE_R, cfg.MAX_BUBBLE_R)

    def _clearance_grad(self, p: np.ndarray) -> np.ndarray:
        """Unit-ish gradient of clearance at p (points away from obstacles)."""
        eps = 0.10
        px1 = self._clearance(p + [eps, 0.0])[0]
        px0 = self._clearance(p - [eps, 0.0])[0]
        py1 = self._clearance(p + [0.0, eps])[0]
        py0 = self._clearance(p - [0.0, eps])[0]
        g = np.array([(px1 - px0), (py1 - py0)], dtype=np.float64) / (2.0 * eps)
        n = float(np.linalg.norm(g))
        if n < 1e-9:
            # Inside an obstacle / flat field — push directly away from nearest point.
            nearest = self._obstacles[np.argmin(
                ((self._obstacles - p) ** 2).sum(1)
            )] if self._obstacles.shape[0] else p
            away = p - nearest
            na = float(np.linalg.norm(away))
            return away / na if na > 1e-9 else np.zeros(2)
        return g / n

    # ── Pure-pursuit tracking ───────────────────────────────────────────────────

    def _lookahead_target(self, robot_x: float, robot_y: float) -> np.ndarray:
        cfg = self.cfg
        if self._band is None or len(self._band) < 2:
            return self._goal.copy()
        pos = np.array([robot_x, robot_y], dtype=np.float64)
        remaining = cfg.LOOKAHEAD
        prev = pos
        for i in range(1, len(self._band)):
            seg = self._band[i] - prev
            seg_len = float(np.linalg.norm(seg))
            if seg_len < 1e-9:
                continue
            if remaining <= seg_len:
                return prev + (remaining / seg_len) * seg
            remaining -= seg_len
            prev = self._band[i]
        return self._band[-1].copy()

    def _pursue(
        self, robot_x: float, robot_y: float, robot_theta: float, target: np.ndarray
    ) -> np.ndarray:
        cfg = self.cfg
        dx = float(target[0]) - robot_x
        dy = float(target[1]) - robot_y
        bearing = math.atan2(dy, dx)
        err = math.atan2(math.sin(bearing - robot_theta), math.cos(bearing - robot_theta))

        # Angular: proportional to heading error.
        w = float(np.clip(cfg.K_HEADING * err, -cfg.MAX_W, cfg.MAX_W))

        # Linear: brake for sharp turns and for tight clearance at the robot.
        turn_factor = max(0.0, 1.0 - cfg.TURN_SLOWDOWN * abs(err) / math.pi)
        clr = float(self._clearance(np.array([[robot_x, robot_y]]))[0])
        clr_factor = float(np.clip(clr / cfg.CLEAR_SLOWDOWN_R, 0.2, 1.0))
        v = cfg.MAX_V * turn_factor * clr_factor

        return np.array([v, w], dtype=np.float32)

    # ── LaserScan → world-frame obstacle cloud (same logic as DWA / MPPI) ───────

    def _scan_to_obstacles(
        self, ranges, angle_min, angle_increment, max_range,
        robot_x, robot_y, robot_theta,
    ) -> np.ndarray:
        cfg = self.cfg
        ranges = np.asarray(ranges, dtype=np.float64)
        n = len(ranges)
        if n == 0:
            return np.zeros((0, 2), dtype=np.float64)
        idx = np.arange(0, n, cfg.SCAN_SUBSAMPLE)
        r = ranges[idx]
        a = angle_min + idx * angle_increment
        valid = np.isfinite(r) & (r > 0.01) & (r < max_range * 0.99)
        if not valid.any():
            return np.zeros((0, 2), dtype=np.float64)
        r, a = r[valid], a[valid]
        cos_t, sin_t = math.cos(robot_theta), math.sin(robot_theta)
        ox_r = r * np.cos(a)
        oy_r = r * np.sin(a)
        ox_w = robot_x + cos_t * ox_r - sin_t * oy_r
        oy_w = robot_y + sin_t * ox_r + cos_t * oy_r
        return np.stack([ox_w, oy_w], axis=1)
