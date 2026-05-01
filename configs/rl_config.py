"""
RL configuration for the BARN Gym integration.

Import in any script:
    from configs.rl_config import RLEnvConfig, RLLogConfig, RLEvalConfig

These classes are the single source of truth for all RL-related defaults.
Every parameter can be overridden per-instance in BarnEnv / RLLogger
constructors, but changing the value here changes the default everywhere.
"""

from os.path import join
from configs.config import Paths


# ── Environment ────────────────────────────────────────────────────────────────

class RLEnvConfig:
    """Parameters that define the Gymnasium environment."""

    # ── Observation ──────────────────────────────────────────────────────────
    # Number of laser scan rays in the observation vector.
    # The raw scan (moborobo: ~722 rays, jackal: ~1080) is resampled to this count.
    # Lower = faster training; higher = more spatial resolution.
    NUM_SCAN_RAYS: int = 360

    # ── Timing ───────────────────────────────────────────────────────────────
    # Seconds of simulated time per step (Gazebo runs at ~1× real-time).
    # 0.1 s → ~10 Hz policy frequency.  Increase to let the robot travel further
    # per step (coarser control); decrease for finer control but slower training.
    STEP_DURATION: float = 0.1

    # Hard truncation limit.  Episode ends with truncated=True after this many steps.
    # 500 steps × 0.1 s/step = 50 s — well under the 100 s BARN timeout.
    MAX_EPISODE_STEPS: int = 500

    # ── Rewards ──────────────────────────────────────────────────────────────
    # Terminal reward on collision.  Should be strongly negative.
    REWARD_COLLISION: float = -100.0

    # Terminal reward on reaching the goal.  Should be strongly positive.
    REWARD_GOAL: float = 100.0

    # Per-step living penalty.  Small negative value discourages loitering.
    REWARD_STEP: float = -0.01

    # If True, each step also receives (prev_dist_to_goal − curr_dist_to_goal).
    # This dense shaping signal usually speeds up early learning significantly.
    REWARD_SHAPING: bool = True


# ── Logging ────────────────────────────────────────────────────────────────────

class RLLogConfig:
    """Parameters that control where and how training metrics are saved."""

    # Root directory for all RL outputs.  Individual run directories are created
    # under RUNS_DIR/{run_id}/.
    # Set to "" in BarnEnv(log_dir="") to disable episode CSV logging entirely.
    LOG_DIR: str = join(Paths.RESULTS_DIR, "rl")
    RUNS_DIR: str = join(Paths.RESULTS_DIR, "rl", "runs")

    # Number of most-recent episodes used for rolling statistics in log_progress().
    ROLLING_WINDOW: int = 100

    # How often (in env steps) to call logger.log_progress() in a training loop.
    LOG_EVERY_STEPS: int = 1_000


# ── Evaluation ────────────────────────────────────────────────────────────────

class RLEvalConfig:
    """Which worlds to use for evaluation and how to run it."""

    # Standard 50-world BARN evaluation set (every 6th index, same as run_barn_custom.py).
    EVAL_WORLDS: list = list(range(0, 300, 6))   # [0, 6, 12, ..., 294]

    # Smaller 10-world set for quick mid-training checks.
    QUICK_EVAL_WORLDS: list = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270]

    # Number of episodes per world during evaluation (average over multiple runs
    # for noisy environments; 1 is fine for deterministic BARN worlds).
    EPISODES_PER_WORLD: int = 1

    # Max steps per evaluation episode (can be higher than training to allow
    # the policy to fully demonstrate its capability).
    MAX_EVAL_STEPS: int = 600
