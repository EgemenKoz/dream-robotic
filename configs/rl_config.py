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
    # The observation is a two-part Dict (structure from the multi-robot DRL
    # collision-avoidance literature — Fan & Long et al. ICRA'18/IJRR'20):
    #   "scan"  : (SCAN_STACK, NUM_SCAN_RAYS)  stacked min-pooled LiDAR frames
    #   "state" : (5,)  [dist_norm, sin(angle), cos(angle), last_lin, last_ang]
    #
    # Number of laser scan sectors in each frame.  The raw scan (moborobo:
    # ~722 rays, jackal: ~1080) is MIN-POOLED into this many sectors — min,
    # not interpolation, so a thin obstacle hit by a single raw ray can never
    # vanish from the observation.  90 sectors ≈ 4°/sector.
    NUM_SCAN_RAYS: int = 90

    # Number of consecutive scan frames stacked in the "scan" part.
    # 3 frames @ 10 Hz gives the CNN 0.3 s of history — enough to infer
    # approach speed toward obstacles (the policy is otherwise memoryless).
    SCAN_STACK: int = 3

    # Ranges are clipped to this many metres before normalising (scan_norm =
    # clip(range, 0, SCAN_CLIP_RANGE) / SCAN_CLIP_RANGE).  BARN corridors are
    # < 1 m wide; normalising by the raw 10 m sensor range squashes all
    # decision-relevant geometry into [0, 0.1].  5 m keeps resolution where
    # obstacles actually matter.  Values beyond the robot's own max range are
    # clipped by min(SCAN_CLIP_RANGE, robot max range) in BarnEnv.
    SCAN_CLIP_RANGE: float = 5.0

    # ── Timing ───────────────────────────────────────────────────────────────
    # Seconds of simulated time per step (Gazebo runs at ~1× real-time).
    # 0.1 s → ~10 Hz policy frequency.  Increase to let the robot travel further
    # per step (coarser control); decrease for finer control but slower training.
    STEP_DURATION: float = 0.1

    # Hard truncation limit.  Per-robot, because moborobo's max speed (0.2 m/s)
    # is 2.5× slower than jackal's (0.5 m/s).  At 0.1 s/step:
    #   jackal:   500 × 0.1 s = 50 s  → 25 m at max speed (plenty of slack)
    #   moborobo: 1000 × 0.1 s = 100 s → 20 m at max speed (matches the
    #             standard BARN 100 s time budget; the 10 m straight-line
    #             course needs ~75 s at realistic 0.13 m/s average).
    # Use RLEnvConfig.max_episode_steps(robot) to resolve the right value.
    MAX_EPISODE_STEPS_JACKAL:   int = 500
    MAX_EPISODE_STEPS_MOBOROBO: int = 1000
    # Backward-compat alias kept for tests / scripts that didn't pass a robot.
    MAX_EPISODE_STEPS: int = MAX_EPISODE_STEPS_JACKAL

    @staticmethod
    def max_episode_steps(robot: str) -> int:
        if robot == "moborobo":
            return RLEnvConfig.MAX_EPISODE_STEPS_MOBOROBO
        return RLEnvConfig.MAX_EPISODE_STEPS_JACKAL

    # ── Rewards ──────────────────────────────────────────────────────────────
    # Terminal reward on collision.
    REWARD_COLLISION: float = -50.0

    # Terminal reward on reaching the goal.
    REWARD_GOAL: float = 100.0

    # Per-step living penalty.  Small negative value discourages loitering.
    REWARD_STEP: float = -0.01

    # Dense progress shaping multiplier on (prev_dist - curr_dist).
    # Raw progress per step is small (≤0.05 m for jackal, ≤0.02 m for moborobo),
    # so the scale must be robot-relative: a single global value gives moborobo
    # 2.5× weaker progress reward than jackal for identical motion, which lets
    # the dense penalties (path/prox/heading) drown the "go to goal" signal.
    # The scales below equalise the max per-step progress reward across robots:
    #   jackal:   5.0  × (0.5 m/s × 0.1 s) = +0.25 / step at full speed
    #   moborobo: 12.5 × (0.2 m/s × 0.1 s) = +0.25 / step at full speed
    # Use RLEnvConfig.reward_progress_scale(robot) to resolve the right value.
    REWARD_PROGRESS_SCALE_JACKAL:   float = 5.0
    REWARD_PROGRESS_SCALE_MOBOROBO: float = 12.5
    # Backward-compat alias kept for tests / scripts that didn't pass a robot.
    REWARD_PROGRESS_SCALE: float = REWARD_PROGRESS_SCALE_JACKAL

    @staticmethod
    def reward_progress_scale(robot: str) -> float:
        if robot == "moborobo":
            return RLEnvConfig.REWARD_PROGRESS_SCALE_MOBOROBO
        return RLEnvConfig.REWARD_PROGRESS_SCALE_JACKAL

    # Penalty for being close to obstacles.  Applied as
    #   -REWARD_PROX_SCALE * max(0, PROX_THRESHOLD - min_scan_norm) / PROX_THRESHOLD
    # where min_scan_norm is the closest normalised scan range in [0, 1].
    REWARD_PROX_SCALE: float = 0.05
    PROX_THRESHOLD: float = 0.20   # of SCAN_CLIP_RANGE → 1 m at the 5 m default.
    # Note: scans are normalised by SCAN_CLIP_RANGE (5 m), not the raw sensor
    # max range, so 0.20 ramps in below 1 m — tight enough that the penalty
    # actually differentiates inside BARN corridors instead of firing always.

    # Penalty for heading misalignment with the goal vector.
    # Applied as -REWARD_HEADING_SCALE * |angle_to_goal| / pi.
    REWARD_HEADING_SCALE: float = 0.02

    # ── Path-following shaping (does NOT appear in the observation) ──────────
    # When True, BarnEnv loads the precomputed A* reference path for the active
    # world and adds a per-step penalty proportional to perpendicular deviation
    # from that polyline. The path is reward-only: the policy still has to
    # infer the corridor from LiDAR.
    PATH_FOLLOW_SHAPING: bool = True
    # Per-step penalty: -REWARD_PATH_SCALE * min(d_perp, PATH_MAX_DEV) / PATH_MAX_DEV
    # Kept well below the max per-step progress reward (+0.25) so path-following
    # is a nudge toward the A* corridor, not a force that cancels progress.
    REWARD_PATH_SCALE: float = 0.03
    PATH_MAX_DEV: float = 1.5   # metres — deviation past this is clipped

    # If True, apply REWARD_PROGRESS_SCALE * (prev_dist - curr_dist) every step.
    # Kept as a kill-switch for ablations / sparse-reward experiments.
    REWARD_SHAPING: bool = True


# ── Model / network architecture ──────────────────────────────────────────────

class RLModelConfig:
    """Feature-extractor and policy-network architecture.

    The policy is two-branch (see rl/models/feature_extractor.py):
        scan  (SCAN_STACK, NUM_SCAN_RAYS) → 1D-CNN → SCAN_FEATURES
        state (5,)                        → FC     → STATE_FEATURES
        concat → net_arch MLP heads (actor + twin critics)
    """

    # Output width of the CNN branch that encodes the stacked scan.
    SCAN_FEATURES: int = 128

    # Output width of the FC branch that encodes the goal/velocity state.
    STATE_FEATURES: int = 32

    # 1D-conv layers over the scan stack: (out_channels, kernel, stride).
    # Two stride-2 convs shrink 90 sectors → 23 positions; receptive field
    # ≈ 9 sectors ≈ 36°, enough to see an entire BARN gap in one unit.
    CONV_LAYERS = [(32, 5, 2), (32, 3, 2)]

    # Hidden layers of the actor/critic heads after feature concat.
    NET_ARCH = [256, 256]


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
