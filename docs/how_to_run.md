# How to Run the BARN Simulation

Covers setup, running, config options, and troubleshooting for both the **Jackal** and **Moborobo** robots.

---

## Table of Contents

1. [One-Time Setup](#one-time-setup)
2. [Every Terminal Session](#every-terminal-session)
3. [Quick Start](#quick-start)
4. [Command-Line Arguments](#command-line-arguments)
5. [Config Reference](#config-reference)
6. [Switching Between Robots](#switching-between-robots)
7. [Running Multiple Worlds](#running-multiple-worlds)
8. [Reading Results](#reading-results)
9. [Troubleshooting](#troubleshooting)

---

## One-Time Setup

Run these once on a fresh machine. Skip anything already done.

### Install missing ROS packages

```bash
sudo apt install ros-noetic-pointcloud-to-laserscan
```

Required for Moborobo. Without it the RS-16 point cloud won't be converted to `/scan` and the robot navigates blind (no obstacle avoidance).

### Build the Moborobo catkin workspace

```bash
# Source the base workspaces first — the chain gets baked into moborobo_ws at build time
source /opt/ros/noetic/setup.bash
source ~/barn_lab/jackal_ws/devel/setup.bash

mkdir -p ~/barn_lab/moborobo_ws/src
cd ~/barn_lab/moborobo_ws/src
ln -s ~/Desktop/dream/robots/moborobo                                     moborobo
ln -s ~/Desktop/dream/robots/robosense_simulator/robosense_description    robosense_description
ln -s ~/Desktop/dream/robots/robosense_simulator/robosense_gazebo_plugins robosense_gazebo_plugins

cd ~/barn_lab/moborobo_ws
catkin_make
```

> **Important:** Always source `jackal_ws` *before* running `catkin_make`. If you build without it, the workspace chain won't include `jackal_helper` and the collision publisher won't be found at runtime.

---

## Every Terminal Session

Do this in every new terminal before running the simulation. Order matters.

```bash
# 1. Activate the Python virtualenv
#    (provides rospkg, numpy, defusedxml, netifaces)
export PATH="$HOME/barn_lab/nav_challenge/bin:$PATH"

# 2. Source ROS and workspaces
source /opt/ros/noetic/setup.bash
source ~/barn_lab/jackal_ws/devel/setup.bash
source ~/barn_lab/moborobo_ws/devel/setup.bash   # Moborobo only — skip for Jackal-only runs

# 3. Go to the project
cd ~/Desktop/dream
```

> **Why the venv?** The runner script imports `rospkg`, `numpy`, etc. These are installed inside `~/barn_lab/nav_challenge/`, not in the system Python. Skipping this causes `ModuleNotFoundError` on startup.

> **Why source jackal_ws even for Moborobo?** The `collision_publisher_node` binary lives in `jackal_helper`. Without it, collisions won't end the episode. The script warns and continues if it's missing, but collision detection will be inactive.

---

## Quick Start

```bash
# Jackal, world 0, headless
python3 run_barn_custom.py --world_idx 0

# Jackal, world 0, with Gazebo GUI
python3 run_barn_custom.py --world_idx 0 --gui

# Moborobo, world 0, headless
python3 run_barn_custom.py --world_idx 0 --robot moborobo

# Moborobo, world 42, with GUI
python3 run_barn_custom.py --world_idx 42 --robot moborobo --gui

# Save results to a specific file
python3 run_barn_custom.py --world_idx 0 --robot moborobo --out my_results.txt
```

Stop a run with **Ctrl+C**. The shutdown handler sends SIGINT to roslaunch nodes for a clean exit (avoids the GPU plugin segfault that SIGTERM causes).

---

## Command-Line Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--world_idx` | int | `0` | BARN world index (0–299) |
| `--robot` | str | `"jackal"` | Platform: `jackal` or `moborobo` |
| `--gui` | flag | off | Open Gazebo GUI (slower but visible) |
| `--out` | str | `"results.txt"` | File to append result line to |

---

## Config Reference

All settings live in `configs/config.py`.

```python
from configs.config import (
    SimConfig, RobotConfig, MoboroboConfig,
    WorldConfig, GazeboConfig, NavConfig, Paths,
)
```

### SimConfig — Robot Selection

```python
SimConfig.ROBOT = "jackal"   # "jackal" | "moborobo"
```

Default robot when `--robot` is not passed. The CLI `--robot` flag overrides this at runtime.

---

### RobotConfig — Jackal

| Setting | Default | Description |
|---|---|---|
| `LASER_ENABLED` | `True` | Enable Hokuyo UST-10 LiDAR |
| `LASER_MODEL` | `"ust10"` | Passed via `JACKAL_LASER_MODEL` env var |
| `LASER_OFFSET` | `"-0.065 0 0.01"` | Sensor mount offset (x y z) from chassis |
| `MAX_LINEAR_VEL` | `0.5` m/s | Max forward speed |
| `MAX_ANGULAR_VEL` | `1.57` rad/s | Max turn rate |
| `FOOTPRINT` | `[[-0.21,-0.165], ...]` | Costmap footprint |

---

### MoboroboConfig — Moborobo

| Setting | Default | Description |
|---|---|---|
| `LIDAR_POINTCLOUD_TOPIC` | `"/rs_points"` | Raw RS-16 3D point cloud |
| `LIDAR_SCAN_TOPIC` | `"/scan"` | 2D scan after `pointcloud_to_laserscan` |
| `LIDAR_FRAME` | `"robosense"` | TF frame of the RS-16 |
| `LIDAR_MIN_RANGE` | `0.45` m | Near clip for 2D slice |
| `LIDAR_MAX_RANGE` | `10.0` m | Far clip for 2D slice |
| `LIDAR_HEIGHT_MIN` | `0.0` m | Height filter floor (keep ground-level slice) |
| `LIDAR_HEIGHT_MAX` | `1.0` m | Height filter ceiling |
| `MAX_LINEAR_VEL` | `0.2` m/s | Max forward speed |
| `MAX_ANGULAR_VEL` | `1.0` rad/s | Max turn rate |
| `FOOTPRINT` | `[[0.387, 0.235], ...]` | Costmap footprint (larger than Jackal) |
| `MODEL_NAME` | `"moborobo"` | Gazebo model name |

---

### WorldConfig — BARN Dataset

| Setting | Default | Description |
|---|---|---|
| `STATIC_INIT_POSITION` | `[-2.25, 3, 1.57]` | Start pose [x, y, yaw] for all worlds |
| `STATIC_GOAL_OFFSET` | `[0, 10]` | Goal = start + [dx, dy] |
| `EVAL_WORLDS` | `range(0, 300, 6)` | Standard 50-world evaluation set |
| `OBSTACLE_RADIUS` | `0.075` m | Cylinder radius in world files |

---

### GazeboConfig — Simulator

| Setting | Default | Description |
|---|---|---|
| `STARTUP_WAIT` | `5` s | Wait after `roslaunch` before connecting |
| `RESET_WAIT` | `1` s | Wait between reset retries |
| `POSITION_TOL` | `0.1` m | Tolerance for confirming robot is at init pose |

Increase `STARTUP_WAIT` if Gazebo is slow to start on your machine.

---

### NavConfig — Navigation

| Setting | Default | Description |
|---|---|---|
| `LOCAL_PLANNER` | `"dwa"` | Jackal only: `"dwa"` or `"eband"` |
| `TIMEOUT` | `100.0` s | Max episode length (sim time) |
| `GOAL_TOLERANCE` | `1.0` m | Distance from goal to count as success |
| `NOMINAL_SPEED` | `0.5` m/s | Used to compute optimal path time |

---

### Paths — File Locations

| Attribute | Path |
|---|---|
| `BARN_DATASET` | `~/Desktop/BARN_dataset` |
| `JACKAL_WS` | `~/barn_lab/jackal_ws` |
| `ROBOTS_DIR` | `~/Desktop/dream/robots` |
| `MOBOROBO_PKG` | `~/Desktop/dream/robots/moborobo` |
| `RESULTS_DIR` | `~/Desktop/dream/results` |

---

## Switching Between Robots

```bash
# Via CLI (recommended)
python3 run_barn_custom.py --world_idx 0 --robot jackal
python3 run_barn_custom.py --world_idx 0 --robot moborobo

# Or set the permanent default in configs/config.py
SimConfig.ROBOT = "moborobo"
```

---

## Running Multiple Worlds

```bash
# Jackal — standard 50-world eval set
for i in $(seq 0 6 294); do
    python3 run_barn_custom.py --world_idx $i --out jackal_eval.txt
done

# Moborobo — same eval set
for i in $(seq 0 6 294); do
    python3 run_barn_custom.py --world_idx $i --robot moborobo --out moborobo_eval.txt
done
```

---

## Reading Results

Each run appends one line to the output file:

```
world_idx  succeeded  collided  timed_out  time_elapsed  nav_metric
```

| Column | Description |
|---|---|
| `world_idx` | BARN world index |
| `succeeded` | 1 = reached goal before timeout |
| `collided` | 1 = hit an obstacle |
| `timed_out` | 1 = 100 s elapsed without success |
| `time_elapsed` | Simulated seconds from start to end |
| `nav_metric` | BARN score 0–1 (higher = better; 0 = failure) |

**Score formula:**
```
nav_metric = success * optimal_time / clip(actual_time, 2*optimal_time, 8*optimal_time)
```

---

## Troubleshooting

**`Cannot locate node of type [collision_publisher_node] in package [jackal_helper]`**
The package is found but roslaunch can't find the compiled binary. This happens when `CMAKE_PREFIX_PATH` doesn't include `jackal_ws/devel` — roslaunch uses this to locate binaries in `devel/lib/`. The runner now sets both `ROS_PACKAGE_PATH` and `CMAKE_PREFIX_PATH` for jackal_ws automatically. If you still see this, make sure `jackal_ws` was built:
```bash
source /opt/ros/noetic/setup.bash
cd ~/barn_lab/jackal_ws && catkin_make
```

**`cannot launch node of type [pointcloud_to_laserscan/...]`**
Package not installed:
```bash
sudo apt install ros-noetic-pointcloud-to-laserscan
```

**Robot moves but ignores obstacles**
`/scan` is empty — `pointcloud_to_laserscan` isn't running. Install it (above), then rerun.

**`$(find moborobo)` not found / moborobo package missing**
Source the moborobo workspace and confirm the build succeeded:
```bash
source ~/barn_lab/moborobo_ws/devel/setup.bash
rospack find moborobo   # should print the package path
```
If `rospack find` fails, rebuild:
```bash
source /opt/ros/noetic/setup.bash
source ~/barn_lab/jackal_ws/devel/setup.bash
cd ~/barn_lab/moborobo_ws && catkin_make
```

**Gazebo hangs at startup**
Increase `GazeboConfig.STARTUP_WAIT` in `configs/config.py` from `5` to `10` or more.

**Segfault when pressing Ctrl+C**
This was caused by SIGTERM killing the Gazebo GPU plugin mid-frame. The shutdown handler now sends SIGINT first for clean teardown. If you still see it occasionally, it's a known Gazebo/GPU plugin issue and doesn't affect results.

**`Timed out waiting for transform from base_link to map`**
The costmaps are now set to `global_frame: odom` so this should not appear. If it does, check that `moborobo_ws` was rebuilt after the config changes:
```bash
cd ~/barn_lab/moborobo_ws && catkin_make
```

**Robot doesn't reset to start position**
The model name in Gazebo must match. Jackal uses `"jackal"`, Moborobo uses `"moborobo"`. This is handled automatically by the `--robot` flag.
