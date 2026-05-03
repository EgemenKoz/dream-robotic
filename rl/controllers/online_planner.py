"""
Online occupancy grid + A* global planner.

OccupancyGrid  — 2D grid updated incrementally from LiDAR scans.
OnlineGlobalPlanner — wraps any local controller (MPPI / DWA) with online
                      global planning so it navigates using only pose + goal +
                      LiDAR, matching the BARN challenge conditions.

Coordinate system: Gazebo world frame (metres).
The grid covers the full BARN navigation corridor with generous margins:
  X ∈ [-6, 6],  Y ∈ [1, 15]  at 0.15 m/cell  → ~80×93 cells
"""

import math
from heapq import heappush, heappop
from typing import List, Optional, Tuple

import numpy as np


# ── Occupancy grid ─────────────────────────────────────────────────────────────

class OccupancyGrid:
    """
    Incremental 2D occupancy grid built from LiDAR hit points.

    Cells are binary: free (False) or occupied/inflated (True).
    Hit-count thresholding prevents single spurious readings from blocking paths.
    """

    RESOLUTION   = 0.15   # metres per cell
    X_MIN, X_MAX = -6.0, 6.0
    Y_MIN, Y_MAX =  1.0, 15.0
    INFLATE_CELLS = 1     # inflation radius in cells (0.15 m from detected surface ≈ 0.225 m
                          # from cylinder centre — enough for moborobo inscribed radius 0.235 m).
                          # 2 cells closed most BARN corridors (only 2-4 cells wide).
    HIT_THRESH    = 2     # LiDAR hits required before a cell is marked occupied

    def __init__(self):
        self._nx = int(round((self.X_MAX - self.X_MIN) / self.RESOLUTION)) + 1
        self._ny = int(round((self.Y_MAX - self.Y_MIN) / self.RESOLUTION)) + 1
        self._hits     = np.zeros((self._nx, self._ny), dtype=np.int16)
        self._obstacle = np.zeros((self._nx, self._ny), dtype=bool)

    def reset(self):
        self._hits[:]     = 0
        self._obstacle[:] = False

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        return (
            int((x - self.X_MIN) / self.RESOLUTION),
            int((y - self.Y_MIN) / self.RESOLUTION),
        )

    def grid_to_world(self, ix: int, iy: int) -> Tuple[float, float]:
        return (
            self.X_MIN + ix * self.RESOLUTION,
            self.Y_MIN + iy * self.RESOLUTION,
        )

    def in_bounds(self, ix: int, iy: int) -> bool:
        return 0 <= ix < self._nx and 0 <= iy < self._ny

    def update(
        self,
        robot_x: float,
        robot_y: float,
        robot_theta: float,
        scan_ranges,
        angle_min: float,
        angle_inc: float,
        max_range: float,
        subsample: int = 4,
    ) -> int:
        """
        Integrate one LiDAR scan into the grid.
        Returns the number of cells that newly crossed the obstacle threshold.
        """
        ranges = np.asarray(scan_ranges, dtype=np.float64)
        n = len(ranges)
        new_obstacles = 0

        for i in range(0, n, subsample):
            r = ranges[i]
            if not np.isfinite(r) or r <= 0.01 or r >= max_range * 0.99:
                continue
            angle_w = robot_theta + angle_min + i * angle_inc
            ex = robot_x + r * math.cos(angle_w)
            ey = robot_y + r * math.sin(angle_w)
            ix, iy = self.world_to_grid(ex, ey)
            if not self.in_bounds(ix, iy):
                continue
            prev = self._hits[ix, iy]
            self._hits[ix, iy] = min(prev + 1, 32767)
            if prev < self.HIT_THRESH <= self._hits[ix, iy]:
                # Newly confirmed obstacle — circular inflation.
                r_inf = self.INFLATE_CELLS
                for dx in range(-r_inf, r_inf + 1):
                    for dy in range(-r_inf, r_inf + 1):
                        if dx * dx + dy * dy <= r_inf * r_inf:
                            nx_, ny_ = ix + dx, iy + dy
                            if self.in_bounds(nx_, ny_):
                                self._obstacle[nx_, ny_] = True
                new_obstacles += 1

        return new_obstacles

    def plan(
        self,
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
    ) -> Optional[np.ndarray]:
        """
        A* from start to goal on the current occupancy grid.
        Returns (N, 2) world-frame waypoints (downsampled) or None if no path.
        """
        sx, sy = self.world_to_grid(*start_xy)
        gx, gy = self.world_to_grid(*goal_xy)
        sx = int(np.clip(sx, 0, self._nx - 1))
        sy = int(np.clip(sy, 0, self._ny - 1))
        gx = int(np.clip(gx, 0, self._nx - 1))
        gy = int(np.clip(gy, 0, self._ny - 1))

        path = _astar(self._obstacle, (sx, sy), (gx, gy))
        if path is None:
            return None

        # Downsample: keep every 3rd cell, always include the last.
        idx = list(range(0, len(path), 3))
        if idx[-1] != len(path) - 1:
            idx.append(len(path) - 1)
        waypoints = np.array(
            [self.grid_to_world(path[i][0], path[i][1]) for i in idx],
            dtype=np.float64,
        )
        return waypoints


# ── Global planner wrapper ──────────────────────────────────────────────────────

class OnlineGlobalPlanner:
    """
    Wraps a local controller (MPPI / DWA) with an online A* global planner.

    The wrapper:
      - Updates an OccupancyGrid from every LiDAR scan.
      - Replans with A* every REPLAN_STEPS control steps.
      - Computes a lookahead subgoal along the current path and passes it to
        the inner local controller via set_goal().
      - Falls back to the final goal when no path has been computed yet.

    Exposes the same compute_action / reset / set_goal interface as the inner
    controller, so it is a drop-in replacement anywhere a local controller is used.
    """

    REPLAN_STEPS  = 20    # replan every N control steps
    WP_REACH      = 0.5   # metres — advance past waypoint when within this distance
    LOOKAHEAD     = 1.5   # metres — subgoal lookahead along planned path
    MIN_SCANS     = 3     # minimum new-obstacle events before first replan

    def __init__(self, local_controller, goal_x: float, goal_y: float):
        self._ctrl = local_controller
        self._goal = np.array([goal_x, goal_y], dtype=np.float64)
        self._grid = OccupancyGrid()
        self._waypoints: Optional[np.ndarray] = None
        self._wpt_idx   = 0
        self._step      = 0
        self._obs_count = 0   # cumulative new-obstacle events seen so far

    # ── Public API (mirrors local controller interface) ────────────────────────

    def reset(self):
        self._ctrl.reset()
        self._grid.reset()
        self._waypoints = None
        self._wpt_idx   = 0
        self._step      = 0
        self._obs_count = 0

    def set_goal(self, goal_x: float, goal_y: float):
        self._goal[:] = [goal_x, goal_y]
        self._waypoints = None  # force replan on next step

    def compute_action(
        self,
        robot_x: float,
        robot_y: float,
        robot_theta: float,
        scan_ranges,
        angle_min: float,
        angle_inc: float,
        max_range: float,
    ) -> np.ndarray:
        self._step += 1
        self._obs_count += self._grid.update(
            robot_x, robot_y, robot_theta,
            scan_ranges, angle_min, angle_inc, max_range,
        )

        # Replan once enough obstacles have been observed and on the schedule.
        if (self._waypoints is None and self._obs_count >= self.MIN_SCANS) or (
            self._step % self.REPLAN_STEPS == 0 and self._obs_count >= self.MIN_SCANS
        ):
            self._replan(robot_x, robot_y)

        subgoal = self._get_subgoal(robot_x, robot_y)
        self._ctrl.set_goal(float(subgoal[0]), float(subgoal[1]))

        return self._ctrl.compute_action(
            robot_x, robot_y, robot_theta,
            scan_ranges, angle_min, angle_inc, max_range,
        )

    # ── Internal ───────────────────────────────────────────────────────────────

    def _replan(self, robot_x: float, robot_y: float):
        path = self._grid.plan((robot_x, robot_y), tuple(self._goal))
        if path is not None and len(path) > 0:
            self._waypoints = path
            self._wpt_idx   = 0

    def _get_subgoal(self, robot_x: float, robot_y: float) -> np.ndarray:
        if self._waypoints is None or len(self._waypoints) == 0:
            return self._goal.copy()

        pos = np.array([robot_x, robot_y], dtype=np.float64)

        # Advance past already-reached waypoints.
        while self._wpt_idx < len(self._waypoints) - 1:
            if np.linalg.norm(pos - self._waypoints[self._wpt_idx]) < self.WP_REACH:
                self._wpt_idx += 1
            else:
                break

        # Walk LOOKAHEAD metres ahead along the remaining path.
        remaining = self.LOOKAHEAD
        prev = pos
        for i in range(self._wpt_idx, len(self._waypoints)):
            seg     = self._waypoints[i] - prev
            seg_len = float(np.linalg.norm(seg))
            if seg_len < 1e-9:
                continue
            if remaining <= seg_len:
                return prev + (remaining / seg_len) * seg
            remaining -= seg_len
            prev = self._waypoints[i]

        return self._waypoints[-1].copy()


# ── A* on boolean obstacle grid ────────────────────────────────────────────────

def _astar(
    obstacle: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> Optional[List[Tuple[int, int]]]:
    """
    A* on a 2D boolean grid (True = occupied).
    Returns list of (ix, iy) cells from start to goal, or None if no path.
    """
    h, w = obstacle.shape

    def heur(r: int, c: int) -> float:
        return math.hypot(r - goal[0], c - goal[1])

    open_set: List = []
    heappush(open_set, (heur(*start), 0.0, start))
    g_cost = {start: 0.0}
    parent: dict = {start: None}
    closed: set = set()

    while open_set:
        _, g, cur = heappop(open_set)
        if cur in closed:
            continue
        closed.add(cur)

        if cur == goal:
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return path

        r, c = cur
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if obstacle[nr, nc]:
                continue
            nb = (nr, nc)
            if nb in closed:
                continue
            new_g = g + math.hypot(dr, dc)
            if new_g < g_cost.get(nb, float("inf")):
                g_cost[nb] = new_g
                parent[nb] = cur
                heappush(open_set, (new_g + heur(nr, nc), new_g, nb))

    return None
