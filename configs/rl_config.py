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
    #             last_lin is in [-REVERSE_SPEED_RATIO, 1] (negative = reversing).
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

    # ── Near-field memory (see rl/env/scan_memory.py) ────────────────────────
    # The RS-16 reports nothing closer than 0.40 m and pointcloud_to_laserscan
    # clips to 0.45 m, so an obstacle about to be hit produces inf — which the
    # old sanitisation turned into "10 m, free".  The robot's own circumscribed
    # (turning) radius is 0.4528 m, i.e. it was blind across exactly the area
    # it sweeps, with ~6 cm between last sighting and contact.
    #
    # With memory on, returns are kept in a world-frame point store and
    # re-projected each step, so a sector that goes blind reports the
    # remembered range.  Adds one channel to obs["scan"] marking which sectors
    # were directly measured this step vs. recalled.
    SCAN_MEMORY: bool = True

    # Sensor near-field cutoff, metres.  Must match the *effective* floor of
    # the scan pipeline: max(LiDAR plugin <min>, pointcloud_to_laserscan
    # range_min) = max(0.40, 0.45).  Remembered points inside this radius are
    # never dropped by raycast clearing.
    SCAN_BLIND_RADIUS: float = 0.45

    # Points beyond this distance from the robot are forgotten (metres).
    SCAN_MEM_RADIUS: float = 4.0

    # De-duplication grid for the point store (metres).
    SCAN_MEM_VOXEL: float = 0.05

    # Points not re-confirmed within this many steps are dropped.  Safety valve
    # against phantoms parked inside the blind ring, which measurement can
    # never clear.  150 steps @ 10 Hz = 15 s.
    SCAN_MEM_MAX_AGE: int = 150

    # ── Lidar load for multi-robot training ──────────────────────────────────
    # RS-16 vertical sampling used by MultiRobotBarnVecEnv (passed to
    # barn_rl_robot.launch).  The stock description casts 900 x 900 rays and
    # ships a 9.4 MB point cloud per scan; that pipeline tops out at ~16
    # scans/s on this machine — one robot's worth — and the limit is CPU-side
    # point handling, not rendering (moving Gazebo to the NVIDIA GPU left it
    # unchanged).  Load scales with ray count, so several robots need fewer rays.
    #
    # Only the downward half matters: the lidar sits ~0.79 m above the ground
    # and the cylinders top out at 0.5 m, so every ray above horizontal hits
    # nothing.  It must still look steeply down — a cylinder 0.45 m away is
    # only visible below -33 deg — which is why this is not the real RS-16's
    # +-15 deg.  The single-robot env and the benchmark keep the stock profile.
    MULTI_LIDAR_PROFILE = {
        "lidar_v_samples": 64,
        "lidar_v_min": -0.785,      # rad (-45 deg)
        "lidar_v_max": 0.0,
        "lidar_max_range": 10.0,    # m — pointcloud_to_laserscan's range_max
    }

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

    # ── Obstacle proximity ───────────────────────────────────────────────────
    # Expressed in METRES (it used to be a fraction of SCAN_CLIP_RANGE, which
    # made it impossible to reason about against the 0.45 m sensor floor).
    #
    #   t      = clip((PROX_RAMP_START_M - d) / (PROX_RAMP_START_M - PROX_FLOOR_M), 0, 1)
    #   r_prox = -REWARD_PROX_SCALE * t^2 * (0.5 + 0.5 * |lin| / max_linear)
    #
    # Three deliberate changes from the flat linear version:
    #   * quadratic, so the gradient sharpens as the blind ring approaches
    #     instead of being a constant nudge;
    #   * velocity-weighted, because what kills is *speed* across the last
    #     6 cm, not position alone;
    #   * scaled 3x, because at 0.05 the penalty peaked at -0.0275/step against
    #     a +0.25/step progress reward — a 9:1 ratio the policy correctly
    #     ignored.  At 0.15 the floor value is -0.15, comparable to progress.
    REWARD_PROX_SCALE: float = 0.15
    PROX_RAMP_START_M: float = 0.90   # penalty begins here
    PROX_FLOOR_M: float = 0.50        # penalty saturates here (just above blind ring)

    # Half-angle of the arc used for the proximity penalty (radians), centred
    # on the direction of travel: straight ahead when driving forward, straight
    # behind when reversing.  Obstacles the robot is moving away from would
    # otherwise keep the penalty on forever.  120 deg keeps the sides — where
    # the 0.4528 m turning sweep matters — and drops only the opposite side.
    PROX_ARC: float = 2.0944          # 120 degrees

    # ── Reverse driving ──────────────────────────────────────────────────────
    # The linear action range is [-REVERSE_SPEED_RATIO * max_linear,
    # max_linear].  Reversing lets the policy back out of a dead end or a
    # corner it has nosed into instead of having to spin in place inside the
    # blind ring.  Kept slower than forward because jackal's UST-10 only sees
    # 270 deg — its rear is covered by the scan memory alone.
    #
    # The ratio also sets where the *untrained* policy sits: SAC's tanh output
    # of 0 maps to the middle of the range.  At 1.0 the range is symmetric, the
    # middle is 0 m/s, and a noisy policy dithers in place — measured over 300
    # uniform-random steps: -0.08 m of progress at 1.0 against +1.62 m at 0.3.
    # That zero-mean jitter is what the `reverse_v1` run collapsed into, so the
    # range must stay forward-biased.
    REVERSE_SPEED_RATIO: float = 0.3

    # Per-step penalty while reversing:
    #   r_reverse = -REWARD_REVERSE_SCALE * |lin| / max_linear   (lin < 0 only)
    # Small on purpose — it only needs to stop the policy from driving the
    # whole course backwards; backing out of a trap must stay cheap.
    REWARD_REVERSE_SCALE: float = 0.05

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
    #
    # These were 0.03 / 1.5 m, which made the term noise: at the deviation the
    # policy actually ran (0.34-0.54 m mean, measured over the benchmark) the
    # penalty was -0.008/step against +0.25/step of progress — 30:1.  The
    # policy duly ignored the corridor and took a route 10-18% SHORTER than
    # A* (9.24-9.89 m actual vs 11.05-11.62 m reference), cutting corners until
    # min clearance sat on the 0.45 m sensor floor in 92-96% of episodes.
    #
    # Every one of the 67 trained worlds has >=0.52 m of bottleneck clearance
    # and the A* reference itself never enters the blind ring (min 0.525 m), so
    # the safe corridor was there the whole time — the reward simply never
    # asked for it.
    #
    # On the clip: 0.5 m was too tight.  Measured mean deviation is 0.449 m and
    # the max runs to 1.2 m, so the term spent much of the episode pinned at
    # its ceiling — and a clipped penalty has ZERO gradient, so it stopped
    # telling the policy which way to steer and became a flat -0.15/step tax on
    # existing.  That inflated the cost of every non-success episode and helped
    # push the policy into freezing.  1.0 m keeps a live gradient across the
    # whole operating range while still costing -0.067/step at the mean.
    REWARD_PATH_SCALE: float = 0.15
    PATH_MAX_DEV: float = 1.0   # metres — deviation past this is clipped
    # The dataset paths are 8-connected A* on a 0.15 m grid, so they run in
    # staircases and hug the edges of the inflated obstacle map, with detours
    # no driver would take.  When True the path is shortened by string pulling
    # (rl/env/geometry.py:shortcut_path) before it is used — never into a
    # segment tighter than the stretch of original path it replaces, so the
    # corridor the penalty asks for stays at least as wide as before.
    PATH_SMOOTHING: bool = True

    # If True, apply REWARD_PROGRESS_SCALE * (prev_dist - curr_dist) every step.
    # Kept as a kill-switch for ablations / sparse-reward experiments.
    REWARD_SHAPING: bool = True

    # ── Stuck termination ────────────────────────────────────────────────────
    # End the episode as a genuine terminal when the robot has not gone
    # anywhere: its position is within STUCK_MIN_DISPLACEMENT_M of where it was
    # STUCK_WINDOW_STEPS ago.  Measured as net displacement, not progress
    # toward the goal, so a lateral detour or backing out of a trap never
    # trips it — only standing still or dithering does.
    #
    # Two jobs.  It stops a stalled policy from burning the full step budget
    # (89% of `reverse_v1`'s 240k steps went into 1000-step timeouts that never
    # left the start area), and it gives the critic a real terminal to anchor
    # on: a timeout is bootstrapped, so a buffer holding only timeouts leaves
    # the Q level undetermined and free to drift.
    #
    # As costly as a collision.  It was -25 (half a collision) so that
    # stopping when boxed in would beat ramming the obstacle, but the policy
    # turned that into a cheap way out: in `multi_v3` (200k steps) 148 of 430
    # episodes ended stuck against 124 collisions, nearly all of them dithering
    # forward and back (54% of steps reversing) ~0.6 m in front of the first
    # obstacles instead of trying to get round them.  Giving up must not pay
    # better than trying.
    STUCK_TERMINATION: bool = True
    STUCK_WINDOW_STEPS: int = 200          # 20 s at 10 Hz
    STUCK_MIN_DISPLACEMENT_M: float = 0.3  # i.e. under 0.015 m/s net
    REWARD_STUCK: float = -50.0

    # ── Terminal potential (OFF — kept for ablation only) ────────────────────
    # When True, a terminal transition is paid +progress_scale * dist so the
    # progress shaping becomes potential-based with Phi(terminal) = 0 (Ng et
    # al. 1999) and sums to the same constant for every trajectory.
    #
    # That is exactly why it is off.  Making the shaping unable to rank
    # outcomes removes progress as an incentive: crashing at 8 m and at 2 m
    # score the same, a collision transition is paid +12.5 * dist - 50 (+40 at
    # 7 m), and the only reason left to move is a goal 500+ steps away.  That
    # in turn needs gamma -> 1, and at gamma=0.999 the critic has almost no
    # contraction: in the `reverse_v1` run it drifted to Q = +716 against a
    # physical maximum of 225, went flat across actions, and the policy
    # decayed into noise.
    #
    # With it off, progress is a real reward (+0.25/step at full speed), which
    # is the signal the Jul/Aug runs learned from (47-63% success) and which
    # works with gamma=0.99.
    TERMINAL_POTENTIAL_ZERO: bool = False


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

    # LayerNorm after every hidden layer of the twin critics
    # (rl/models/critic.py).  Added after `multi_v4`, whose stock critic
    # diverged to Q ≈ +7000 on a buffer whose rewards never exceeded +0.35.
    CRITIC_LAYER_NORM: bool = True


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
