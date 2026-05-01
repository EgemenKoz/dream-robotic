"""
Training logger for BARN RL runs.

Creates a timestamped run directory under results/rl/runs/{run_id}/:
  run_config.json   — hyperparameters and env config snapshot
  episodes.csv      — one row per episode (appended in real-time)
  training.log      — human-readable progress with rolling statistics

Usage:
    logger = RLLogger(run_id="ppo_moborobo_v1", config={"algo": "PPO", "lr": 3e-4})

    # after every episode ends:
    logger.log_episode(
        episode=1, world_idx=30, outcome="success",
        steps=250, total_reward=87.3, final_dist=0.4,
    )

    # periodically (e.g. every 1000 env steps):
    logger.log_progress(timestep=1000)

    logger.close()   # writes run summary and flushes files
"""

import csv
import datetime
import json
import os
import time
from collections import deque
from os.path import join
from typing import Any, Dict, Optional

import numpy as np


class RLLogger:
    """
    Per-run logger that tracks episode-level metrics and emits rolling stats.

    Parameters
    ----------
    run_id : str
        Human-readable identifier, e.g. "ppo_moborobo_world30_v1".
    config : dict | None
        Arbitrary hyperparameter / config dict stored in run_config.json.
    results_dir : str | None
        Parent directory for runs. Defaults to results/rl/runs/.
    window : int
        Number of most-recent episodes used for rolling statistics.
    """

    def __init__(
        self,
        run_id: str,
        config: Optional[Dict[str, Any]] = None,
        results_dir: Optional[str] = None,
        window: int = None,
    ):
        from configs.rl_config import RLLogConfig
        if results_dir is None:
            results_dir = RLLogConfig.RUNS_DIR
        if window is None:
            window = RLLogConfig.ROLLING_WINDOW

        self.run_id = run_id
        self.run_dir = join(results_dir, run_id)
        os.makedirs(self.run_dir, exist_ok=True)

        self._window = window
        self._recent_rewards: deque = deque(maxlen=window)
        self._recent_steps: deque = deque(maxlen=window)
        self._outcome_counts: Dict[str, int] = {
            "success": 0, "collision": 0, "timeout": 0,
        }
        self._total_episodes = 0
        self._start_wall = time.time()

        # ── run_config.json ────────────────────────────────────────────────────
        cfg_path = join(self.run_dir, "run_config.json")
        with open(cfg_path, "w") as f:
            json.dump(
                {
                    "run_id": run_id,
                    "started_at": datetime.datetime.now().isoformat(),
                    **(config or {}),
                },
                f,
                indent=2,
            )

        # ── episodes.csv ───────────────────────────────────────────────────────
        self._csv_path = join(self.run_dir, "episodes.csv")
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "episode", "world_idx", "outcome", "steps",
            "total_reward", "final_dist_to_goal", "wall_time_s",
        ])
        self._csv_file.flush()

        # ── training.log ───────────────────────────────────────────────────────
        self._log_path = join(self.run_dir, "training.log")
        self._log_file = open(self._log_path, "w")
        self._write_log_header(config)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def log_episode(
        self,
        episode: int,
        world_idx: int,
        outcome: str,
        steps: int,
        total_reward: float,
        final_dist: float,
    ):
        """Record one completed episode. Call this at every episode end."""
        wall = time.time() - self._start_wall

        self._csv_writer.writerow([
            episode, world_idx, outcome, steps,
            "%.4f" % total_reward, "%.4f" % final_dist, "%.1f" % wall,
        ])
        self._csv_file.flush()

        self._recent_rewards.append(total_reward)
        self._recent_steps.append(steps)
        self._outcome_counts[outcome] = self._outcome_counts.get(outcome, 0) + 1
        self._total_episodes += 1

    def log_progress(self, timestep: int):
        """Print and log a rolling-stats line. Call every N timesteps."""
        if not self._recent_rewards:
            return

        n = self._total_episodes
        win = min(n, self._window)
        mean_r = float(np.mean(self._recent_rewards))
        std_r = float(np.std(self._recent_rewards))
        mean_s = float(np.mean(self._recent_steps))

        total_done = max(sum(self._outcome_counts.values()), 1)
        sr = self._outcome_counts.get("success", 0) / total_done * 100
        cr = self._outcome_counts.get("collision", 0) / total_done * 100
        tr = self._outcome_counts.get("timeout", 0) / total_done * 100

        elapsed = datetime.timedelta(seconds=int(time.time() - self._start_wall))

        line = (
            "[%s] step=%-8d ep=%-6d | "
            "reward(last %d): %.2f ± %.2f | "
            "avg_steps: %.1f | "
            "SR %.1f%%  CR %.1f%%  TO %.1f%%"
        ) % (elapsed, timestep, n, win, mean_r, std_r, mean_s, sr, cr, tr)

        self._log_file.write(line + "\n")
        self._log_file.flush()
        print(line)

    def close(self):
        """Write the final summary, flush and close all files."""
        elapsed = datetime.timedelta(seconds=int(time.time() - self._start_wall))
        total_done = max(sum(self._outcome_counts.values()), 1)

        SEP = "=" * 72
        THIN = "-" * 72
        lines = [
            "",
            SEP,
            "RUN SUMMARY  —  %s" % self.run_id,
            SEP,
            "%-25s %s" % ("Finished at:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            "%-25s %s" % ("Wall time:", elapsed),
            "%-25s %d" % ("Total episodes:", self._total_episodes),
            THIN,
            "%-25s %.2f%%" % ("Success rate:", self._outcome_counts.get("success", 0) / total_done * 100),
            "%-25s %.2f%%" % ("Collision rate:", self._outcome_counts.get("collision", 0) / total_done * 100),
            "%-25s %.2f%%" % ("Timeout rate:", self._outcome_counts.get("timeout", 0) / total_done * 100),
        ]
        if self._recent_rewards:
            lines += [
                THIN,
                "%-25s %.4f" % (
                    "Mean reward (last %d):" % self._window,
                    float(np.mean(self._recent_rewards)),
                ),
                "%-25s %.4f" % (
                    "Std  reward (last %d):" % self._window,
                    float(np.std(self._recent_rewards)),
                ),
            ]
        lines += [SEP, ""]

        for line in lines:
            self._log_file.write(line + "\n")
            print(line)

        self._log_file.close()
        self._csv_file.close()

    # ──────────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────────

    def _write_log_header(self, config: Optional[Dict[str, Any]]):
        SEP = "=" * 72
        lines = [
            SEP,
            "BARN RL Training Log",
            "%-20s %s" % ("Run ID:", self.run_id),
            "%-20s %s" % ("Started:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            SEP,
        ]
        if config:
            lines.append("Config:")
            for k, v in config.items():
                lines.append("  %-22s %s" % (str(k) + ":", v))
            lines.append(SEP)
        lines.append("")
        for line in lines:
            self._log_file.write(line + "\n")
        self._log_file.flush()
