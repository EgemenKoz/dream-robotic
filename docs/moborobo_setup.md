# Moborobo Robot — Setup & Integration Guide

This document covers the Moborobo robot files that have been integrated into the DREAM project, what sensor it uses, and how the robot is wired into the BARN simulation pipeline.

---

## Table of Contents

1. [What Was Moved and Why](#what-was-moved-and-why)
2. [File Layout](#file-layout)
3. [Moborobo Robot Specs](#moborobo-robot-specs)
4. [RoboSense RS-16 3D LiDAR](#robosense-rs-16-3d-lidar)
5. [Sensor Pipeline: 3D → 2D LaserScan](#sensor-pipeline-3d--2d-laserscan)
6. [Moborobo vs Jackal — Key Differences](#moborobo-vs-jackal--key-differences)
7. [Catkin Workspace Setup](#catkin-workspace-setup)
8. [Launch Files Explained](#launch-files-explained)
9. [Config Class Reference](#config-class-reference)

---

## What Was Moved and Why

The original `Moborobo-Project/` is a separate git repository containing both a simulation package (`simulation/src/moborobo`) and a real-robot package (`on_robot/`). Only the simulation-side files are needed for the BARN challenge.

Two ROS packages were copied into `robots/`:

| Source (Moborobo-Project)                          | Destination (dream project)         | Why |
|----------------------------------------------------|-------------------------------------|-----|
| `simulation/src/moborobo/`                         | `robots/moborobo/`                  | Robot URDF, meshes, costmap configs, launch files |
| `simulation/src/robosense_simulator/`              | `robots/robosense_simulator/`       | RS-16 Gazebo GPU ray plugin + URDF description |

Two new launch files were added to `robots/moborobo/launch/`:
- `barn_gazebo.launch` — spawns Moborobo in a BARN world file
- `barn_move_base.launch` — runs the RS-16 sensor converter + move_base navigation

---

## File Layout

```
dream/
  robots/
    moborobo/
      CMakeLists.txt
      package.xml
      description/
        robot.udrf.xacro       ← top-level robot file (includes all below)
        robot_core.xacro       ← base_link, wheels, casters, inertia
        controller.xacro       ← diff_drive Gazebo plugin (left/right wheel joints)
        camera.xacro           ← RGB camera (640x480, 10 Hz)
      urdf/
        RS-16.urdf.xacro       ← RoboSense RS-16 mount + Gazebo sensor plugin
      meshes/
        moborobo_short.stl     ← body mesh
        robosense_16.dae       ← RS-16 sensor visual mesh
      config/
        costmap_common_params.yaml    ← footprint, obstacle/inflation layers
        local_costmap_params.yaml     ← rolling window 3x3 m
        global_costmap_params.yaml    ← 20x20 m map frame costmap
        trajectory_planner.yaml       ← TrajectoryPlannerROS velocity limits
        map.rviz                      ← RViz config for viewing
      launch/
        navigation.launch             ← original moborobo launch (own world, gmapping)
        barn_gazebo.launch            ← NEW: spawn moborobo in BARN world
        barn_move_base.launch         ← NEW: RS-16→scan + move_base for BARN
      world/
        work.world                    ← moborobo's own test world
        work/model.config
        work/model.sdf
      maps/
        room.yaml
        room.pgm
    robosense_simulator/
      robosense_description/
        urdf/RS-16.urdf.xacro         ← standalone RS-16 URDF (reference)
        meshes/robosense_16.dae
        launch/example.launch
      robosense_gazebo_plugins/
        src/GazeboRosRoboSenseLaser.cpp   ← GPU ray plugin source (must be built)
        include/robosense_gazebo_plugins/GazeboRosRoboSenseLaser.h
        CMakeLists.txt
        package.xml
```

---

## Moborobo Robot Specs

| Property | Value |
|---|---|
| Drive type | Differential drive (2 powered wheels + 4 caster wheels) |
| Body size (collision box) | 0.70 × 0.47 × 0.30 m |
| Body mass | 20 kg |
| Drive wheel radius | 0.10 m |
| Drive wheel separation | 0.380 m |
| Max linear velocity | 0.2 m/s (trajectory planner limit) |
| Max angular velocity | 1.0 rad/s |
| Footprint (costmap) | 0.387 × 0.235 m half-extents (rect) |
| Gazebo model name | `moborobo` |
| Controller | `libgazebo_ros_diff_drive.so` |
| cmd_vel topic | `/cmd_vel` |
| odom topic | `/odom` |

**Footprint vertices (used in costmap):**
```
[ 0.387,  0.235]
[ 0.387, -0.235]
[-0.387, -0.235]
[-0.387,  0.235]
```

The Moborobo is physically larger than the Jackal (0.42 × 0.31 m). The wider footprint means the costmap inflation radius is more critical in narrow BARN corridors.

---

## RoboSense RS-16 3D LiDAR

The Moborobo carries a **RoboSense RS-16** 16-channel 3D LiDAR mounted on top of the chassis at z = 0.53 m above base_link.

| Property | Value |
|---|---|
| Model | RoboSense RS-16 |
| Channels | 16 vertical |
| Horizontal FoV | 360° (-π to +π) |
| Vertical FoV | ±2.7 rad (~±155°) |
| Horizontal samples | 900 per rotation |
| Min range | 0.40 m |
| Max range | 150.0 m |
| Gazebo plugin | `libgazebo_ros_robosense_gpu_laser.so` |
| Raw point cloud topic | `/rs_points` |
| Message type | `sensor_msgs/PointCloud2` |
| TF frame | `robosense` |
| Update rate | 1 Hz (simulation — increase in RS-16.urdf.xacro if needed) |

The plugin is in `robots/robosense_simulator/robosense_gazebo_plugins/` and **must be compiled** before use (see Catkin Workspace Setup below).

---

## Sensor Pipeline: 3D → 2D LaserScan

The BARN navigation stack (move_base) requires a `/scan` topic of type `sensor_msgs/LaserScan`. The RS-16 produces a 3D point cloud, so `pointcloud_to_laserscan` bridges the gap.

```
RS-16 Gazebo plugin
      ↓
  /rs_points  (sensor_msgs/PointCloud2, 360° 3D)
      ↓
  pointcloud_to_laserscan_node   (launched in barn_move_base.launch)
      ↓
  /scan  (sensor_msgs/LaserScan, 360° 2D slice)
      ↓
  move_base obstacle layer  (reads /scan for local costmap)
```

**Slice parameters** (configured in `barn_move_base.launch`):

| Parameter | Value | Meaning |
|---|---|---|
| `min_height` | 0.0 m | Ignore returns below ground |
| `max_height` | 1.0 m | Keep returns up to 1 m (obstacle height) |
| `range_min` | 0.45 m | Ignore very close returns (sensor blind zone) |
| `range_max` | 10.0 m | Match costmap obstacle range |
| `angle_min/max` | ±π | Full 360° |
| `angle_increment` | 0.0087 rad | ~0.5° resolution |

---

## Moborobo vs Jackal — Key Differences

| Aspect | Jackal | Moborobo |
|---|---|---|
| Sensor | Hokuyo UST-10 (2D, 270°) | RoboSense RS-16 (3D, 360°) |
| Sensor topic | `/scan` | `/rs_points` → `/scan` via converter |
| Max speed | 0.5 m/s | 0.2 m/s |
| Footprint | 0.42 × 0.33 m | 0.774 × 0.47 m |
| Planner | DWA or EBand | TrajectoryPlannerROS |
| Gazebo model name | `jackal` | `moborobo` |
| Plugin build required | No (pre-built in jackal_ws) | Yes (robosense_gazebo_plugins) |

---

## Catkin Workspace Setup

The moborobo packages need a catkin workspace so ROS can find them via `$(find moborobo)`.

### Step 0 — Install missing ROS packages (once)

```bash
sudo apt install ros-noetic-pointcloud-to-laserscan \
                 ros-noetic-robot-state-publisher \
                 ros-noetic-move-base \
                 ros-noetic-gmapping
```

`pointcloud_to_laserscan` is the most critical one — without it, the RS-16 3D cloud can't be converted to a 2D `/scan` for move_base.

### Step 1 — Create and build the workspace (once)

```bash
mkdir -p ~/barn_lab/moborobo_ws/src
cd ~/barn_lab/moborobo_ws/src

# Symlink the packages from this repo (changes to robots/ are reflected automatically)
ln -s ~/Desktop/dream/robots/moborobo         moborobo
ln -s ~/Desktop/dream/robots/robosense_simulator/robosense_description    robosense_description
ln -s ~/Desktop/dream/robots/robosense_simulator/robosense_gazebo_plugins robosense_gazebo_plugins

cd ~/barn_lab/moborobo_ws
source /opt/ros/noetic/setup.bash
catkin_make
```

### Step 2 — Full source sequence for every new terminal

The virtualenv **must** be activated before sourcing ROS workspaces, otherwise Python imports (`rospkg`, `numpy`, etc.) won't resolve.

```bash
export PATH="$HOME/barn_lab/nav_challenge/bin:$PATH"   # activate virtualenv
source /opt/ros/noetic/setup.bash
source ~/barn_lab/jackal_ws/devel/setup.bash           # collision publisher + BARN tools
source ~/barn_lab/moborobo_ws/devel/setup.bash         # moborobo + RS-16 plugin
cd ~/Desktop/dream
```

The compiled RS-16 plugin (`libgazebo_ros_robosense_gpu_laser.so`) lands in `~/barn_lab/moborobo_ws/devel/lib/`. `apply_env(robot="moborobo")` in `configs/config.py` adds this to `GAZEBO_PLUGIN_PATH` automatically.

---

## Launch Files Explained

### `barn_gazebo.launch`

Spawns Moborobo in a given BARN `.world` file. Called by `run_barn_custom.py` when `--robot moborobo`.

| Argument | Default | Description |
|---|---|---|
| `world_name` | required | Absolute path to `.world` file |
| `gui` | false | Open Gazebo GUI (gzclient) |
| `headless` | true | Run without display (set false when gui=true) |
| `init_x/y/z` | -2.25 / 3.0 / 0.2 | Robot spawn position |
| `init_yaw` | 1.5708 | Robot spawn yaw (rad) — faces +Y (BARN direction) |

Starts: Gazebo server, moborobo spawn, robot_state_publisher, collision_publisher_node.

### `barn_move_base.launch`

Navigation stack for BARN. Called by `run_barn_custom.py` after Gazebo is ready.

Starts: `pointcloud_to_laserscan` (RS-16 → /scan), `move_base` with moborobo params.

### `navigation.launch` (original)

The original moborobo launch from the Moborobo-Project. Loads moborobo's own `work.world`, runs gmapping for SLAM, and starts move_base. Use this for standalone moborobo testing outside of BARN.

---

## Config Class Reference

All moborobo settings live in `configs/config.py` under `MoboroboConfig`:

```python
from configs.config import SimConfig, MoboroboConfig

# Switch the active robot
SimConfig.ROBOT = "moborobo"

# Access moborobo properties
MoboroboConfig.LIDAR_SCAN_TOPIC       # "/scan"
MoboroboConfig.LIDAR_POINTCLOUD_TOPIC # "/rs_points"
MoboroboConfig.MAX_LINEAR_VEL         # 0.2 m/s
MoboroboConfig.FOOTPRINT              # [[0.387, 0.235], ...]
MoboroboConfig.MODEL_NAME             # "moborobo"
MoboroboConfig.gazebo_launch_file()   # path to barn_gazebo.launch
MoboroboConfig.nav_launch_file()      # path to barn_move_base.launch
```
