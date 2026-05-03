"""
MPPI (Model Predictive Path Integral) controller configuration.

Import in any script:
    from configs.mppi_config import MPPIConfig

All parameters can be overridden per-instance by passing a modified class or
a dict. See rl/controllers/mppi.py for usage.
"""

from configs.config import MoboroboConfig, RobotConfig, SimConfig


class MPPIConfig:
    """
    Hyperparameters for the MPPI controller.

    MPPI works by sampling K random perturbations around a nominal control
    sequence, rolling out each trajectory using the dynamics model, computing
    the cost of each trajectory, then computing a weighted average update
    (softmax weights via temperature LAMBDA).

    ── Tuning guide ──────────────────────────────────────────────────────────
    K / T        More samples / longer horizon = better plans but slower.
                 Start: K=500, T=20. Raise K if the robot freezes near obstacles.
    LAMBDA       Lower = sharper (exploit best trajectory). Higher = softer mix.
                 Typical range 0.5 – 5.0.
    SIGMA_V/W    Noise std for velocity/angular velocity sampling.
                 Should be ~30-50% of the max velocity range.
    W_OBSTACLE   Raise if the robot keeps hitting walls.
    W_GOAL       Raise if the robot ignores the goal.
    OBS_SIGMA    Steepness of obstacle cost (metres). Lower = sharper penalty.
    ROBOT_RADIUS Safety clearance added on top of the actual robot footprint.
    """

    # ── Sampling ──────────────────────────────────────────────────────────────
    K: int = 900     # number of sampled trajectories per control update
    T: int = 40      # planning horizon (steps); total lookahead = T × DT seconds
                     # at MAX_V=0.2 m/s: 40×0.1s×0.2 = 0.8 m — enough to see around corners
    OBSTACLE_STRIDE: int = 3  # check obstacle cost only every Nth timestep in rollout
                               # reduces (K,T,N,2) array 3× without losing collision signal

    # Timestep must match BarnEnv.step_duration / RLEnvConfig.STEP_DURATION
    DT: float = 0.1   # seconds per MPPI rollout step

    # Temperature — controls how aggressively the best trajectories are favoured.
    # Lower → greedier (exploit); Higher → softer (more exploration).
    LAMBDA: float = 1.0   # sharper — good forward trajectories dominate over zero-velocity ones

    # ── Control noise ─────────────────────────────────────────────────────────
    SIGMA_V: float = 0.12   # linear velocity noise std (m/s)
    SIGMA_W: float = 0.4    # angular velocity noise std (rad/s)
                            # 0.8 caused most trajectories to curve into walls in narrow corridors
                            # (lateral deviation ~0.64m >> 0.08m safe band); 0.4 keeps samples viable

    # ── Velocity limits (copied from robot config by default) ─────────────────
    # Change ROBOT = "jackal" or "moborobo" to load the correct limits,
    # or override MAX_V / MAX_W directly.
    _robot = SimConfig.ROBOT
    MAX_V: float = MoboroboConfig.MAX_LINEAR_VEL if _robot == "moborobo" else RobotConfig.MAX_LINEAR_VEL
    MAX_W: float = MoboroboConfig.MAX_ANGULAR_VEL if _robot == "moborobo" else RobotConfig.MAX_ANGULAR_VEL

    # ── Cost weights ──────────────────────────────────────────────────────────
    # W_OBSTACLE scales only the soft proximity term inside _obstacle_cost().
    # Hard collision rejection is handled separately via COLLISION_PENALTY.
    W_OBSTACLE: float = 3.0     # soft proximity weight
    W_GOAL: float = 4.0         # terminal distance-to-goal
    W_GOAL_STEP: float = 1.0    # per-step distance-to-goal (dense reward)
    W_SPEED: float = 25.0        # penalise mean trajectory speed below MAX_V
                                 # cost = W_SPEED * (MAX_V - mean_v); range [0, W_SPEED*MAX_V=1.6]

    MIN_V: float = 0.0    # no reverse — backward trajectories beat forward ones near walls
                          # because they always have lower obstacle cost, pulling weighted
                          # average negative. robot should rotate to find opening instead.

    # ── Obstacle cost ─────────────────────────────────────────────────────────
    OBS_SIGMA: float = 0.30       # metres — steepness of soft proximity falloff
    ROBOT_RADIUS: float = 0.26    # metres — actual moborobo inscribed radius is 0.235m;
                                  # 0.40 was too conservative and invalidated all corridor trajectories
    COLLISION_PENALTY: float = 40.0  # further reduced — narrow corridors make all fwd trajectories
                                     # clip the old radius; penalty must not overwhelm goal cost

    # ── Scan subsampling ──────────────────────────────────────────────────────
    # Use every Nth ray when building the obstacle cloud from the LaserScan.
    # SCAN_SUBSAMPLE=4 on a 720-ray scan → 180 obstacle points (fast & sufficient).
    SCAN_SUBSAMPLE: int = 4
