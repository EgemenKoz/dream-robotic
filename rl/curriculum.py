"""
Difficulty-based curriculum for BARN RL training.

Ranks every world in a robot's dataset by a combined difficulty score derived
from the 5 BARN metrics (jackal-map-creation / "Benchmarking Metric Ground
Navigation"), splits them into easy / medium / hard tiers, and exposes a
cumulative curriculum schedule.

Metrics file layout (metrics_<idx>.npy, one row of 5 values):
    [0] distance to closest obstacle   — higher = easier
    [1] average visibility             — higher = easier
    [2] dispersion                     — higher = harder
    [3] characteristic dimension       — higher = easier
    [4] tortuosity (arc / chord)       — higher = harder

The combined difficulty in [0, 1] min-max normalises each metric across the
dataset, flips the "higher = easier" ones, and averages.

Usage
─────
    from rl.curriculum import difficulty_tiers, cumulative_phases

    tiers = difficulty_tiers("moborobo")
    # → {"easy": [...], "medium": [...], "hard": [...]}

    phases = cumulative_phases("moborobo", total_steps=200_000)
    # → [(0, [easy]), (66666, [easy+med]), (133333, [easy+med+hard])]
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from configs.config import WorldConfig, SimConfig


# Metric difficulty signs: +1 means "higher value → harder", -1 means easier.
_METRIC_SIGNS = np.array([
    -1.0,   # distance to closest obstacle  (more clearance = easier)
    -1.0,   # average visibility            (see further    = easier)
    +1.0,   # dispersion                    (more clutter   = harder)
    -1.0,   # characteristic dimension      (more open      = easier)
    +1.0,   # tortuosity                    (more winding   = harder)
])


def _world_count(robot: str) -> int:
    prev = SimConfig.ROBOT
    SimConfig.ROBOT = robot
    try:
        return WorldConfig.world_count()
    finally:
        SimConfig.ROBOT = prev


def _metrics_path(robot: str, idx: int) -> str:
    prev = SimConfig.ROBOT
    SimConfig.ROBOT = robot
    try:
        return WorldConfig.metrics_file(idx)
    finally:
        SimConfig.ROBOT = prev


def load_difficulty_scores(robot: Optional[str] = None) -> Dict[int, float]:
    """Return {world_idx: difficulty in [0, 1]} for every world with a metrics file.

    Worlds whose metrics file is missing are skipped (not in the returned dict).
    """
    robot = robot or SimConfig.ROBOT
    n = _world_count(robot)

    idxs: List[int] = []
    rows: List[np.ndarray] = []
    for i in range(n):
        f = _metrics_path(robot, i)
        if os.path.exists(f):
            rows.append(np.load(f))
            idxs.append(i)
    if not rows:
        raise FileNotFoundError(
            "No metrics files found for robot '%s'. Expected metrics_<idx>.npy "
            "under the dataset's metrics_files/." % robot
        )

    M = np.array(rows, dtype=np.float64)            # (W, 5)
    # min-max normalise each column across the dataset
    lo = M.min(axis=0)
    hi = M.max(axis=0)
    span = np.where((hi - lo) > 1e-9, hi - lo, 1.0)
    norm = (M - lo) / span                          # (W, 5) in [0, 1]
    # flip so every column is "higher = harder", then average
    orient = np.where(_METRIC_SIGNS > 0, norm, 1.0 - norm)
    diff = orient.mean(axis=1)                       # (W,)

    return {idx: float(d) for idx, d in zip(idxs, diff)}


def difficulty_tiers(
    robot: Optional[str] = None,
    fractions: Tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
) -> Dict[str, List[int]]:
    """Split worlds into easy / medium / hard by ascending difficulty.

    Parameters
    ----------
    robot : str | None
        Robot dataset to use (default: SimConfig.ROBOT).
    fractions : (easy, medium, hard)
        Relative tier sizes; normalised internally. Default = even thirds.
    """
    scores = load_difficulty_scores(robot)
    order = sorted(scores, key=lambda w: scores[w])   # easy → hard
    n = len(order)

    f = np.array(fractions, dtype=np.float64)
    f = f / f.sum()
    n_easy = int(round(f[0] * n))
    n_med = int(round(f[1] * n))
    # remainder to hard so the three always sum to n
    n_easy = min(n_easy, n)
    n_med = min(n_med, n - n_easy)

    return {
        "easy":   sorted(order[:n_easy]),
        "medium": sorted(order[n_easy:n_easy + n_med]),
        "hard":   sorted(order[n_easy + n_med:]),
    }


def cumulative_phases(
    robot: Optional[str] = None,
    total_steps: int = 200_000,
    fractions: Tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
) -> List[Tuple[int, List[int]]]:
    """Cumulative curriculum: easy → easy+medium → easy+medium+hard.

    Returns a list of (start_step, world_list) sorted by start_step. The trainer
    activates the world_list of the latest phase whose start_step ≤ num_timesteps.
    Training time is split into three equal step ranges.
    """
    tiers = difficulty_tiers(robot, fractions)
    easy = tiers["easy"]
    med = easy + tiers["medium"]
    allw = med + tiers["hard"]

    third = total_steps // 3
    return [
        (0,           sorted(easy)),
        (third,       sorted(med)),
        (2 * third,   sorted(allw)),
    ]


def phase_for_step(phases: List[Tuple[int, List[int]]], step: int) -> List[int]:
    """World list of the latest phase whose start_step ≤ step."""
    active = phases[0][1]
    for start, worlds in phases:
        if step >= start:
            active = worlds
        else:
            break
    return active


class WorldCycler:
    """Visits a world pool `episodes_per_world` episodes at a time.

    The pool is walked in a shuffled order and reshuffled after every full
    pass, so each world comes up once per pass but the sequence never repeats.
    Walking it in index order (as before) meant a 214k-step run only ever saw
    16 of its 51 starting worlds, and a curriculum phase change merely kept
    walking the low indices of the bigger pool.

    Used by MultiRobotBarnVecEnv, where the count is over the episodes of
    every robot sharing the world, and by train_sac.WorldCyclingEnv.
    """

    def __init__(self, worlds: List[int], episodes_per_world: int = 5,
                 seed: Optional[int] = None, shuffle: bool = True):
        if not worlds:
            raise ValueError("worlds must not be empty")
        self._episodes_per_world = int(episodes_per_world)
        self._shuffle = shuffle
        self._rng = np.random.default_rng(seed)
        self._worlds = list(worlds)
        self._order = self._new_order(avoid_first=None)
        self._ptr = 0
        self._current = self._order[0]
        self._done_here = 0

    @property
    def current(self) -> int:
        return self._current

    def episode_finished(self) -> bool:
        """Count one finished episode. True when the quota for the current
        world is used up and `advance()` should be called."""
        self._done_here += 1
        return self._done_here >= self._episodes_per_world

    def advance(self) -> int:
        """Move to the next world and return it."""
        self._ptr += 1
        if self._ptr >= len(self._order):
            self._order = self._new_order(avoid_first=self._current)
            self._ptr = 0
        self._current = self._order[self._ptr]
        self._done_here = 0
        return self._current

    def set_world_list(self, worlds: List[int]):
        """Swap the pool (curriculum phase change). The current world keeps
        running until its quota is used up; the next `advance()` starts a
        fresh pass over the new pool."""
        if worlds and list(worlds) != self._worlds:
            self._worlds = list(worlds)
            self._order = self._new_order(avoid_first=self._current)
            self._ptr = -1

    def _new_order(self, avoid_first: Optional[int]) -> List[int]:
        if not self._shuffle:
            order = list(self._worlds)
            if avoid_first in order:
                k = order.index(avoid_first) + 1   # carry on after the current one
                order = order[k:] + order[:k]
            return order
        order = [int(w) for w in self._rng.permutation(self._worlds)]
        if len(order) > 1 and order[0] == avoid_first:
            order[0], order[-1] = order[-1], order[0]
        return order
