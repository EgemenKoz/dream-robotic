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
    K: int = 500      # number of sampled trajectories per control update
    T: int = 20       # planning horizon (steps); total lookahead = T × DT seconds

    # Timestep must match BarnEnv.step_duration / RLEnvConfig.STEP_DURATION
    DT: float = 0.1   # seconds per MPPI rollout step

    # Temperature — controls how aggressively the best trajectories are favoured.
    # Lower → greedier (exploit); Higher → softer (more exploration).
    LAMBDA: float = 1.0

    # ── Control noise ─────────────────────────────────────────────────────────
    SIGMA_V: float = 0.12   # linear velocity noise std (m/s)
    SIGMA_W: float = 0.5    # angular velocity noise std (rad/s)

    # ── Velocity limits (copied from robot config by default) ─────────────────
    # Change ROBOT = "jackal" or "moborobo" to load the correct limits,
    # or override MAX_V / MAX_W directly.
    _robot = SimConfig.ROBOT
    MAX_V: float = MoboroboConfig.MAX_LINEAR_VEL if _robot == "moborobo" else RobotConfig.MAX_LINEAR_VEL
    MAX_W: float = MoboroboConfig.MAX_ANGULAR_VEL if _robot == "moborobo" else RobotConfig.MAX_ANGULAR_VEL

    # ── Cost weights ──────────────────────────────────────────────────────────
    W_OBSTACLE: float = 80.0    # weight on the obstacle cost term
    W_GOAL: float = 2.0         # weight on terminal distance-to-goal cost
    W_GOAL_STEP: float = 0.5    # weight on per-step distance-to-goal (dense)

    # ── Obstacle cost ─────────────────────────────────────────────────────────
    # Obstacle cost = exp(−clearance / OBS_SIGMA) summed over the horizon.
    # clearance = max(0, dist_to_nearest_obstacle − ROBOT_RADIUS)
    OBS_SIGMA: float = 0.4    # metres — steepness of cost falloff from obstacles
    ROBOT_RADIUS: float = 0.4 # metres — conservative safety margin (moborobo is ~0.47 m wide)

    # ── Scan subsampling ──────────────────────────────────────────────────────
    # Use every Nth ray when building the obstacle cloud from the LaserScan.
    # SCAN_SUBSAMPLE=4 on a 720-ray scan → 180 obstacle points (fast & sufficient).
    SCAN_SUBSAMPLE: int = 4
