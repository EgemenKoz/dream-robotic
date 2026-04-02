"""
Central configuration for BARN simulation runs.

Import this in any script:
    from config import RobotConfig, WorldConfig, GazeboConfig, NavConfig
"""

import os
from os.path import expanduser, join


# ── Paths ──────────────────────────────────────────────────────────────────────

class Paths:
    BARN_DATASET     = expanduser("~/Desktop/BARN_dataset")
    WORLD_FILES      = join(BARN_DATASET, "world_files")
    PATH_FILES       = join(BARN_DATASET, "path_files")
    MAP_FILES        = join(BARN_DATASET, "map_files")
    GRID_FILES       = join(BARN_DATASET, "grid_files")
    CSPACE_FILES     = join(BARN_DATASET, "cspace_files")
    METRICS_FILES    = join(BARN_DATASET, "metrics_files")

    JACKAL_WS        = "/home/egemen/barn_lab/jackal_ws"
    BARN_CHALLENGE   = join(JACKAL_WS, "src/the-barn-challenge")
    JACKAL_HELPER    = join(BARN_CHALLENGE, "jackal_helper")

    RESULTS_DIR      = expanduser("~/Desktop/dream/results")


# ── Jackal Robot ───────────────────────────────────────────────────────────────

class RobotConfig:
    # Laser sensor
    LASER_ENABLED    = True
    LASER_MODEL      = "ust10"             # Hokuyo UST-10 (270 deg FoV, 0.05–10 m)
    LASER_OFFSET     = "-0.065 0 0.01"    # x y z offset from chassis center

    # Physical limits (enforced by move_base params, not hardware)
    MAX_LINEAR_VEL   = 0.5    # m/s
    MAX_ANGULAR_VEL  = 1.57   # rad/s

    # Footprint (used by costmap) — matches Jackal body dimensions
    FOOTPRINT = [
        [-0.21, -0.165],
        [-0.21,  0.165],
        [ 0.21,  0.165],
        [ 0.21, -0.165],
    ]


# ── World / Environment ────────────────────────────────────────────────────────

class WorldConfig:
    # Static BARN worlds: indices 0–299
    STATIC_WORLD_COUNT  = 300
    STATIC_WORLD_RANGE  = range(0, 300)

    # Standard evaluation set (50 worlds, every 6th index)
    EVAL_WORLDS = list(range(0, 300, 6))   # [0, 6, 12, ..., 294]

    # Starting pose for static worlds (x, y, yaw in radians)
    STATIC_INIT_POSITION = [-2.25, 3, 1.57]

    # Goal position relative to STATIC_INIT_POSITION
    STATIC_GOAL_OFFSET   = [0, 10]

    # Coordinate transform constants (grid → Gazebo world frame)
    OBSTACLE_RADIUS      = 0.075   # meters, cylinder radius used in world gen
    GRID_SIZE            = 30      # 30x30 grid cells

    @staticmethod
    def world_file(world_idx: int) -> str:
        return join(Paths.WORLD_FILES, "world_%d.world" % world_idx)

    @staticmethod
    def path_file(world_idx: int) -> str:
        return join(Paths.PATH_FILES, "path_%d.npy" % world_idx)

    @staticmethod
    def map_yaml(world_idx: int) -> str:
        return join(Paths.MAP_FILES, "yaml_%d.yaml" % world_idx)

    @staticmethod
    def grid_file(world_idx: int) -> str:
        return join(Paths.GRID_FILES, "grid_%d.npy" % world_idx)

    @staticmethod
    def cspace_file(world_idx: int) -> str:
        return join(Paths.CSPACE_FILES, "cspace_%d.npy" % world_idx)


# ── Gazebo Simulation ──────────────────────────────────────────────────────────

class GazeboConfig:
    GUI              = True           # Set True to open gzclient viewer
    STARTUP_WAIT     = 5               # Seconds to wait after launching Gazebo
    RESET_WAIT       = 1               # Seconds between reset retries
    POSITION_TOL     = 0.1            # Meters — acceptable distance from init pose

    # Physics
    USE_SIM_TIME     = True            # ROS uses Gazebo clock (always True in sim)


# ── Navigation ────────────────────────────────────────────────────────────────

class NavConfig:
    # Planner choice: "dwa" or "eband"
    LOCAL_PLANNER    = "dwa"

    # Timeout for a single navigation episode
    TIMEOUT          = 100.0          # seconds (sim time)

    # Goal reached threshold
    GOAL_TOLERANCE   = 1.0            # meters

    # Navigation metric: score = success * optimal_time / clip(actual, lo, hi)
    METRIC_TIME_LO   = 2.0            # multiplier on optimal_time for clip lower bound
    METRIC_TIME_HI   = 8.0            # multiplier on optimal_time for clip upper bound
    NOMINAL_SPEED    = 0.5            # m/s — used to compute optimal_time from path length

    # Costmap
    OBSTACLE_RANGE   = 2.5            # meters
    RAYTRACE_RANGE   = 3.0            # meters
    INFLATION_RADIUS = 0.3            # meters

    # DWA local planner
    DWA_SIM_TIME     = 2.0            # seconds forward simulation
    DWA_XY_TOLERANCE = 0.25          # meters
    DWA_YAW_TOLERANCE= 0.157         # radians (~9 deg)

    @staticmethod
    def launch_file() -> str:
        base = Paths.JACKAL_HELPER
        if NavConfig.LOCAL_PLANNER == "eband":
            return join(base, "launch", "move_base_eband.launch")
        return join(base, "launch", "move_base_DWA.launch")


# ── Apply env vars for Jackal laser (call before roslaunch) ───────────────────

def apply_env():
    """Set all required environment variables. Call once before launching Gazebo."""
    os.environ["JACKAL_LASER"]        = "1" if RobotConfig.LASER_ENABLED else "0"
    os.environ["JACKAL_LASER_MODEL"]  = RobotConfig.LASER_MODEL
    os.environ["JACKAL_LASER_OFFSET"] = RobotConfig.LASER_OFFSET
    os.environ["GAZEBO_PLUGIN_PATH"]  = join(Paths.JACKAL_HELPER, "plugins")
