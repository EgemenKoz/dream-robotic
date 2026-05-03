from configs.config import MoboroboConfig, RobotConfig, SimConfig


class AStarLocalConfig:
    """Hyperparameters for the A*+pure-pursuit local controller."""

    # ── Pure pursuit ───────────────────────────────────────────────────────────
    LOOKAHEAD_DIST: float = 0.8   # m — lookahead distance along the path
    WAYPOINT_REACH: float = 0.35  # m — distance to consider a waypoint reached

    # Proportional gain: angular_velocity = K_HEADING * heading_error
    K_HEADING: float = 2.5

    # Speed reduction factor when heading error is large.
    # v = MAX_V * max(0, 1 - |err|/π * DECEL)
    DECEL: float = 1.8

    _robot = SimConfig.ROBOT
    MIN_V:  float = 0.05
    MAX_V: float = (
        MoboroboConfig.MAX_LINEAR_VEL if _robot == "moborobo" else RobotConfig.MAX_LINEAR_VEL
    )
    MAX_W: float = (
        MoboroboConfig.MAX_ANGULAR_VEL if _robot == "moborobo" else RobotConfig.MAX_ANGULAR_VEL
    )

    # ── Frontal obstacle safety ────────────────────────────────────────────────
    FRONT_SECTOR: float = 0.524   # rad (≈ 30°) half-angle of the frontal safety sector
    OBSTACLE_STOP_DIST: float = 0.30  # m — stop if obstacle is closer than this
    OBSTACLE_SLOW_DIST: float = 0.80  # m — begin slowing below this distance
