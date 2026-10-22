"""
Egocentric obstacle memory for a LiDAR with a near-field blind ring.

Why this exists
───────────────
The RS-16 cannot report anything closer than 0.40 m (`RS-16.urdf.xacro`), and
`pointcloud_to_laserscan` clips a further 5 cm (`range_min: 0.45`).  With
`use_inf: true` those rays come out as `inf`, and the naive sanitisation
"non-finite → range_max" turns an obstacle about to be hit into a reading of
"10 m, maximally free" — the observation is not merely missing the danger, it
reports the opposite.

The geometry makes that fatal for moborobo:

    front bumper                 0.387 m from base_link
    circumscribed (turn) radius  0.4528 m
    sensor blind radius          0.45 m

so the blind ring is the same size as the footprint's own rotational sweep,
and there are ~6 cm — about three control steps at 0.2 m/s — between the last
sighting of an obstacle and contact with it.

But the blind zone is only *instantaneously* unobservable.  An obstacle now at
0.40 m was measured at 0.50 m a moment ago, and the robot knows how far it has
moved since.  This class keeps the returns in a world-frame point store and
re-projects them into the current robot frame each step, so a sector that has
gone blind reports the remembered range instead of "free".

What it is not
──────────────
Not a SLAM map: the store is pruned to `mem_radius` around the robot and is
reset every episode.  It is a short-range memory whose only job is to fill in
the blind ring.

Fusion rule
───────────
Per sector, the output is the minimum of what is measured now and what is
remembered.  Remembered points are dropped when a current ray sees straight
*through* where they sit (raycast clearing) — except inside the blind ring,
which is never cleared by measurement, because no measurement there is
possible.  That single exception is what makes the memory survive exactly
where it is needed.

Usage
─────
    mem = ScanMemory(num_sectors=90, blind_radius=0.45)
    mem.reset()
    fused_m, fresh = mem.update(x, y, theta, ranges, angle_min, angle_inc,
                                range_max)
    # fused_m : (N,) per-sector range in metres, range_max where nothing known
    # fresh   : (N,) 1.0 where the sector was directly measured this step

`ranges` must carry NaN (not range_max) where the sensor returned nothing —
see `BarnEnv._scan_cb`.
"""

import math
from typing import Optional, Tuple

import numpy as np

# Voxel key packing: coordinates are quantised to `voxel` metres and folded
# into one int64 so de-duplication is a 1-D np.unique (much faster than
# axis=0 on an (N, 2) array).  +-4096 voxels at 5 cm covers +-204 m, far
# beyond mem_radius.
_KEY_OFFSET = 4096
_KEY_STRIDE = 8192


class ScanMemory:
    """Short-range egocentric obstacle memory that survives the sensor blind ring.

    Parameters
    ----------
    num_sectors : int
        Output sector count (matches ``RLEnvConfig.NUM_SCAN_RAYS``).
    fov_min, fov_max : float
        Angular extent covered by the sectors, in the robot frame (radians).
    blind_radius : float
        Sensor near-field cutoff in metres.  Remembered points closer than
        ``blind_radius + blind_guard`` are never removed by raycast clearing.
    mem_radius : float
        Points further than this from the robot are dropped.
    voxel : float
        De-duplication grid size in metres.
    max_age_steps : int
        Points not re-confirmed within this many steps are dropped.  Acts as a
        safety valve against phantoms that parked inside the blind ring and can
        never be cleared by measurement.
    clear_margin : float
        A remembered point is contradicted only when the current reading in its
        sector is at least this much further away, which absorbs sector
        quantisation and pose noise.
    blind_guard : float
        Extra protection band above ``blind_radius`` (metres).
    max_points : int
        Hard cap on the store; the nearest points are kept.
    """

    def __init__(
        self,
        num_sectors: int,
        fov_min: float = -math.pi,
        fov_max: float = math.pi,
        blind_radius: float = 0.45,
        mem_radius: float = 4.0,
        voxel: float = 0.05,
        max_age_steps: int = 150,
        clear_margin: float = 0.15,
        blind_guard: float = 0.05,
        max_points: int = 8000,
    ):
        if num_sectors <= 0:
            raise ValueError("num_sectors must be positive, got %r" % num_sectors)
        if fov_max <= fov_min:
            raise ValueError("fov_max must exceed fov_min")

        self.num_sectors = int(num_sectors)
        self.fov_min = float(fov_min)
        self.fov_max = float(fov_max)
        self.blind_radius = float(blind_radius)
        self.mem_radius = float(mem_radius)
        self.voxel = float(voxel)
        self.max_age_steps = int(max_age_steps)
        self.clear_margin = float(clear_margin)
        self.blind_guard = float(blind_guard)
        self.max_points = int(max_points)

        self._span = self.fov_max - self.fov_min
        self.reset()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self):
        """Drop all remembered geometry. Call once per episode."""
        self._pts = np.zeros((0, 2), dtype=np.float64)
        self._seen = np.zeros((0,), dtype=np.int64)
        self._step = 0

    @property
    def num_points(self) -> int:
        return int(len(self._pts))

    def snapshot(self) -> Tuple[np.ndarray, np.ndarray]:
        """Copy of the store for visualisation: world-frame (M, 2) points and
        each point's age in steps since it was last measured."""
        return self._pts.copy(), (self._step - self._seen).astype(np.float32)

    def update(
        self,
        x: float,
        y: float,
        theta: float,
        ranges: Optional[np.ndarray],
        angle_min: float,
        angle_increment: float,
        range_max: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fold one scan into the memory and return the fused sector ranges.

        Parameters
        ----------
        x, y, theta : float
            Current robot pose in the world frame.
        ranges : np.ndarray | None
            Raw ray ranges in metres with **NaN where the sensor returned
            nothing**.  None is treated as an all-NaN scan, so the fused output
            comes purely from memory.
        angle_min, angle_increment : float
            Scan geometry, radians.
        range_max : float
            Value used for sectors where nothing is known.

        Returns
        -------
        fused : (num_sectors,) float32, metres
        fresh : (num_sectors,) float32, 1.0 where directly measured this step
        """
        self._step += 1
        n = self.num_sectors

        # ── 1. Direct measurement, min-pooled into sectors ────────────────────
        meas = np.full(n, np.inf, dtype=np.float64)
        pts_new = np.zeros((0, 2), dtype=np.float64)

        if ranges is not None and len(ranges) > 0:
            r = np.asarray(ranges, dtype=np.float64)
            bearing = angle_min + np.arange(len(r), dtype=np.float64) * angle_increment
            sec = self._sector_of(_wrap(bearing))
            ok = np.isfinite(r) & (r > 0.0) & (sec >= 0)
            if ok.any():
                np.minimum.at(meas, sec[ok], r[ok])
                # World-frame points for the store.
                a = theta + bearing[ok]
                pts_new = np.stack(
                    [x + r[ok] * np.cos(a), y + r[ok] * np.sin(a)], axis=1
                )

        fresh = np.isfinite(meas)

        # ── 2. Raycast clearing against the *existing* store ──────────────────
        # Done before inserting this frame's points so we never clear what we
        # just measured.
        if len(self._pts):
            rng, sec_p = self._project(self._pts, x, y, theta)
            in_fov = sec_p >= 0
            seen_through = np.zeros(len(rng), dtype=bool)
            if in_fov.any():
                m = meas[sec_p[in_fov]]
                # Contradicted: we can currently see past where this point sits.
                # Never applies inside the blind ring — no measurement there is
                # possible, so absence of a return is not evidence of absence.
                seen_through[in_fov] = (
                    np.isfinite(m)
                    & (rng[in_fov] < m - self.clear_margin)
                    & (rng[in_fov] > self.blind_radius + self.blind_guard)
                )
            keep = ~seen_through
            # ── 3. Prune by radius and age in the same pass ───────────────────
            keep &= rng <= self.mem_radius
            keep &= (self._step - self._seen) <= self.max_age_steps
            self._pts = self._pts[keep]
            self._seen = self._seen[keep]

        # ── 4. Insert this frame's returns, de-duplicated ─────────────────────
        if len(pts_new):
            self._pts = np.concatenate([self._pts, pts_new], axis=0)
            self._seen = np.concatenate(
                [self._seen, np.full(len(pts_new), self._step, dtype=np.int64)]
            )
            self._dedup()
            self._cap(x, y)

        # ── 5. Project the whole store into sectors and fuse ──────────────────
        rem = np.full(n, np.inf, dtype=np.float64)
        if len(self._pts):
            rng, sec_p = self._project(self._pts, x, y, theta)
            ok = sec_p >= 0
            if ok.any():
                np.minimum.at(rem, sec_p[ok], rng[ok])

        fused = np.minimum(meas, rem)
        fused = np.where(np.isfinite(fused), fused, float(range_max))

        return fused.astype(np.float32), fresh.astype(np.float32)

    def arc_min(self, fused: np.ndarray, half_angle: float,
                centre: float = 0.0) -> float:
        """Minimum of `fused` over sectors within ±half_angle of `centre`.

        `centre` is a robot-frame bearing: 0 for straight ahead, pi for
        straight behind (used while reversing).  The proximity penalty must
        ignore obstacles the robot is moving away from: once cleared, a
        cylinder sits behind at a range that would otherwise keep the penalty
        on forever.  Restricting to an arc around the direction of travel keeps
        the sides (where the 0.4528 m turning sweep matters) while dropping
        the opposite side.
        """
        centres = self.sector_centres()
        mask = np.abs(_wrap(centres - centre)) <= half_angle
        if not mask.any():
            return float(np.min(fused)) if len(fused) else float("inf")
        return float(np.min(np.asarray(fused)[mask]))

    def sector_centres(self) -> np.ndarray:
        """Bearing of each sector centre in the robot frame (radians)."""
        step = self._span / self.num_sectors
        return self.fov_min + (np.arange(self.num_sectors) + 0.5) * step

    # ──────────────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────────────

    def _sector_of(self, ang: np.ndarray) -> np.ndarray:
        """Sector index for each bearing; -1 when outside the FOV."""
        idx = np.floor((ang - self.fov_min) / self._span * self.num_sectors)
        idx = idx.astype(np.int64)
        outside = (idx < 0) | (idx >= self.num_sectors)
        idx[outside] = -1
        return idx

    def _project(
        self, pts: np.ndarray, x: float, y: float, theta: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """World points → (range, sector index) in the current robot frame."""
        dx = pts[:, 0] - x
        dy = pts[:, 1] - y
        rng = np.hypot(dx, dy)
        bearing = _wrap(np.arctan2(dy, dx) - theta)
        return rng, self._sector_of(bearing)

    def _dedup(self):
        """Collapse points sharing a voxel, keeping the most recent sighting."""
        keys = np.round(self._pts / self.voxel).astype(np.int64)
        packed = (keys[:, 0] + _KEY_OFFSET) * _KEY_STRIDE + (keys[:, 1] + _KEY_OFFSET)
        # np.unique returns the first occurrence of each key, so order
        # newest-first to keep the freshest timestamp per voxel.
        order = np.argsort(-self._seen, kind="stable")
        packed = packed[order]
        _, first = np.unique(packed, return_index=True)
        sel = order[first]
        self._pts = self._pts[sel]
        self._seen = self._seen[sel]

    def _cap(self, x: float, y: float):
        """Enforce max_points by keeping the nearest points to the robot."""
        if len(self._pts) <= self.max_points:
            return
        d = np.hypot(self._pts[:, 0] - x, self._pts[:, 1] - y)
        keep = np.argpartition(d, self.max_points)[: self.max_points]
        self._pts = self._pts[keep]
        self._seen = self._seen[keep]


def _wrap(a):
    """Wrap angle(s) to [-pi, pi)."""
    return (np.asarray(a, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi
