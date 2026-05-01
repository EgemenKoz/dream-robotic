"""
Central configuration for BARN simulation runs.

Import this in any script:
    from config import SimConfig, RobotConfig, MoboroboConfig, WorldConfig, GazeboConfig, NavConfig

Set SimConfig.ROBOT = "jackal" or "moborobo" to choose the platform.
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
    METRICS_FILES      = join(BARN_DATASET, "metrics_files")
    NORM_METRICS_FILES = join(BARN_DATASET, "norm_metrics_files")

    MOBOROBO_DATASET     = expanduser("~/Desktop/moborobo_dataset")
    MOBOROBO_WORLD_FILES      = join(MOBOROBO_DATASET, "world_files")
    MOBOROBO_PATH_FILES       = join(MOBOROBO_DATASET, "path_files")
    MOBOROBO_MAP_FILES        = join(MOBOROBO_DATASET, "map_files")
    MOBOROBO_GRID_FILES       = join(MOBOROBO_DATASET, "grid_files")
    MOBOROBO_CSPACE_FILES     = join(MOBOROBO_DATASET, "cspace_files")
    MOBOROBO_METRICS_FILES    = join(MOBOROBO_DATASET, "metrics_files")
    MOBOROBO_NORM_METRICS_FILES = join(MOBOROBO_DATASET, "norm_metrics_files")

    JACKAL_WS        = "/home/egemen/barn_lab/jackal_ws"
    BARN_CHALLENGE   = join(JACKAL_WS, "src/the-barn-challenge")
    JACKAL_HELPER    = join(BARN_CHALLENGE, "jackal_helper")

    ROBOTS_DIR       = expanduser("~/Desktop/dream/robots")
    MOBOROBO_PKG     = join(ROBOTS_DIR, "moborobo")

    RESULTS_DIR      = expanduser("~/Desktop/dream/results")


# ── Robot Selection ────────────────────────────────────────────────────────────

class SimConfig:
    """Top-level simulation switch. Change ROBOT to select the platform.

    "jackal"   — Clearpath Jackal UGV with Hokuyo UST-10 2D LiDAR (BARN default)
    "moborobo" — Moborobo robot with RoboSense RS-16 3D LiDAR
    """
    ROBOT = "jackal"   # "jackal" | "moborobo"


# ── Jackal Robot ───────────────────────────────────────────────────────────────

class RobotConfig:
    """Configuration for the Clearpath Jackal UGV (BARN challenge default)."""
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


# ── Moborobo Robot ─────────────────────────────────────────────────────────────

class MoboroboConfig:
    """Configuration for the Moborobo robot with RoboSense RS-16 3D LiDAR.

    The RS-16 is a 360-degree 3D LiDAR. In the BARN simulation it is converted
    to a 2D LaserScan (published on /scan) by the pointcloud_to_laserscan node
    inside barn_move_base.launch.
    """

    # RoboSense RS-16 3D LiDAR topics
    LIDAR_POINTCLOUD_TOPIC = "/rs_points"   # raw point cloud from RS-16 Gazebo plugin
    LIDAR_SCAN_TOPIC       = "/scan"        # 2D LaserScan after pointcloud_to_laserscan
    LIDAR_FRAME            = "robosense"

    # pointcloud_to_laserscan filter settings (must match barn_move_base.launch)
    LIDAR_MIN_RANGE   = 0.45   # m
    LIDAR_MAX_RANGE   = 10.0   # m
    LIDAR_HEIGHT_MIN  = 0.0    # m — keep returns between these heights (ground slice)
    LIDAR_HEIGHT_MAX  = 1.0    # m

    # Physical limits (from trajectory_planner.yaml)
    MAX_LINEAR_VEL   = 0.2    # m/s
    MAX_ANGULAR_VEL  = 1.0    # rad/s

    # Footprint (from costmap_common_params.yaml — larger than Jackal)
    FOOTPRINT = [
        [ 0.387,  0.235],
        [ 0.387, -0.235],
        [-0.387, -0.235],
        [-0.387,  0.235],
    ]

    # Gazebo model name (must match -model arg in barn_gazebo.launch)
    MODEL_NAME = "moborobo"

    @staticmethod
    def gazebo_launch_file() -> str:
        """Launch file that spawns Moborobo in a BARN world."""
        return join(Paths.MOBOROBO_PKG, "launch", "barn_gazebo.launch")

    @staticmethod
    def nav_launch_file() -> str:
        """Launch file for Moborobo navigation stack (move_base + sensor converter)."""
        return join(Paths.MOBOROBO_PKG, "launch", "barn_move_base.launch")


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
    def _is_moborobo():
        return SimConfig.ROBOT == "moborobo"

    @staticmethod
    def world_file(world_idx: int) -> str:
        base = Paths.MOBOROBO_WORLD_FILES if WorldConfig._is_moborobo() else Paths.WORLD_FILES
        return join(base, "world_%d.world" % world_idx)

    @staticmethod
    def path_file(world_idx: int) -> str:
        base = Paths.MOBOROBO_PATH_FILES if WorldConfig._is_moborobo() else Paths.PATH_FILES
        return join(base, "path_%d.npy" % world_idx)

    @staticmethod
    def map_yaml(world_idx: int) -> str:
        base = Paths.MOBOROBO_MAP_FILES if WorldConfig._is_moborobo() else Paths.MAP_FILES
        return join(base, "yaml_%d.yaml" % world_idx)

    @staticmethod
    def grid_file(world_idx: int) -> str:
        base = Paths.MOBOROBO_GRID_FILES if WorldConfig._is_moborobo() else Paths.GRID_FILES
        return join(base, "grid_%d.npy" % world_idx)

    @staticmethod
    def cspace_file(world_idx: int) -> str:
        base = Paths.MOBOROBO_CSPACE_FILES if WorldConfig._is_moborobo() else Paths.CSPACE_FILES
        return join(base, "cspace_%d.npy" % world_idx)

    @staticmethod
    def metrics_file(world_idx: int) -> str:
        base = Paths.MOBOROBO_METRICS_FILES if WorldConfig._is_moborobo() else Paths.METRICS_FILES
        return join(base, "metrics_%d.npy" % world_idx)

    @staticmethod
    def norm_metrics_file(world_idx: int) -> str:
        base = Paths.MOBOROBO_NORM_METRICS_FILES if WorldConfig._is_moborobo() else Paths.NORM_METRICS_FILES
        return join(base, "norm_metrics_%d.npy" % world_idx)


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
    NOMINAL_SPEED    = 2.0            # m/s — Jackal max speed, used by BARN challenge to compute OT

    # Costmap
    OBSTACLE_RANGE   = 2.5            # meters
    RAYTRACE_RANGE   = 3.0            # meters
    INFLATION_RADIUS = 0.3            # meters

    # DWA local planner
    DWA_SIM_TIME     = 2.0            # seconds forward simulation
    DWA_XY_TOLERANCE = 0.25          # meters
    DWA_YAW_TOLERANCE= 0.157         # radians (~9 deg)

    @staticmethod
    def launch_file(robot: str = None) -> str:
        """Return the navigation launch file for the given robot.

        Args:
            robot: "jackal" or "moborobo". Defaults to SimConfig.ROBOT.
        """
        if robot is None:
            robot = SimConfig.ROBOT
        if robot == "moborobo":
            return MoboroboConfig.nav_launch_file()
        # jackal — pick DWA or EBand
        base = Paths.JACKAL_HELPER
        if NavConfig.LOCAL_PLANNER == "eband":
            return join(base, "launch", "move_base_eband.launch")
        return join(base, "launch", "move_base_DWA.launch")


# ── Apply env vars before roslaunch (call once before spawning Gazebo) ─────────

def apply_env(robot: str = None):
    """Set required environment variables for the chosen robot.

    Args:
        robot: "jackal" or "moborobo". Defaults to SimConfig.ROBOT.
    """
    if robot is None:
        robot = SimConfig.ROBOT

    if robot == "jackal":
        os.environ["JACKAL_LASER"]        = "1" if RobotConfig.LASER_ENABLED else "0"
        os.environ["JACKAL_LASER_MODEL"]  = RobotConfig.LASER_MODEL
        os.environ["JACKAL_LASER_OFFSET"] = RobotConfig.LASER_OFFSET
        os.environ["GAZEBO_PLUGIN_PATH"]  = join(Paths.JACKAL_HELPER, "plugins")
    elif robot == "moborobo":
        # The RS-16 plugin is in robots/robosense_simulator/robosense_gazebo_plugins
        # After building, the .so lands in the catkin devel/lib directory.
        # GAZEBO_PLUGIN_PATH must include that lib dir so Gazebo can find it.
        # Adjust this path to wherever catkin built the plugin.
        robosense_plugin_dir = expanduser("~/barn_lab/moborobo_ws/devel/lib")
        existing = os.environ.get("GAZEBO_PLUGIN_PATH", "")
        os.environ["GAZEBO_PLUGIN_PATH"] = (
            robosense_plugin_dir + ":" + existing if existing else robosense_plugin_dir
        )
