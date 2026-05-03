from configs.config import MoboroboConfig, RobotConfig, SimConfig


class DWAConfig:
    """Hyperparameters for the pure-Python DWA (Dynamic Window Approach) controller."""

    DT: float = 0.1        # seconds between control updates
    SIM_TIME: float = 2.0  # trajectory simulation horizon (seconds)

    N_V: int = 10   # linear velocity samples per dynamic window
    N_W: int = 21   # angular velocity samples (odd → includes 0)

    _robot = SimConfig.ROBOT
    MAX_V: float = (
        MoboroboConfig.MAX_LINEAR_VEL if _robot == "moborobo" else RobotConfig.MAX_LINEAR_VEL
    )
    MAX_W: float = (
        MoboroboConfig.MAX_ANGULAR_VEL if _robot == "moborobo" else RobotConfig.MAX_ANGULAR_VEL
    )

    MAX_ACC_V: float = 0.5   # m/s²  — max linear acceleration
    MAX_ACC_W: float = 2.0   # rad/s² — max angular acceleration

    W_HEADING:   float = 1.0   # reward aligning final heading with goal bearing
    W_CLEARANCE: float = 1.5   # reward trajectories that stay clear of obstacles
    W_VELOCITY:  float = 0.3   # reward higher forward speed

    ROBOT_RADIUS:  float = 0.4   # m — safety margin; trajectories inside are rejected
    CLEARANCE_MAX: float = 2.0   # m — cap on clearance to avoid open-space bias

    SCAN_SUBSAMPLE: int = 4   # use every Nth lidar ray for obstacle cloud
