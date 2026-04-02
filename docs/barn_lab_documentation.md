# BARN Lab — Full Project Documentation

## Overview

`barn_lab` is a ROS Noetic workspace housing the **ICRA BARN Navigation Challenge** platform. It provides a complete simulation pipeline to benchmark autonomous mobile robot navigation algorithms using the Clearpath Jackal robot in Gazebo. The project supports 300 static obstacle environments (BARN dataset) and 60 dynamic environments (DynaBARN dataset).

**Location:** `/home/egemen/barn_lab/`

---

## Directory Structure

```
/home/egemen/barn_lab/
├── jackal_ws/                          # ROS Catkin workspace (main workspace)
│   ├── src/                            # All ROS packages live here
│   │   ├── the-barn-challenge/         # Core challenge package
│   │   ├── jackal/                     # Clearpath Jackal robot packages
│   │   ├── jackal_simulator/           # Gazebo simulation support
│   │   ├── jackal_desktop/             # RViz visualization tools
│   │   └── eband_local_planner/        # Elastic Band local planner plugin
│   ├── build/                          # CMake build artifacts (auto-generated)
│   └── devel/                          # Compiled binaries, libraries, setup scripts
└── nav_challenge/                      # Python virtual environment (isolated deps)
```

---

## ROS Packages In Detail

### 1. `the-barn-challenge` — Core Challenge Package

**Path:** `jackal_ws/src/the-barn-challenge/`

This is the primary package. Everything related to running, evaluating, and extending the navigation benchmark lives here.

#### 1.1 Entry Point: `run.py`

The main script that orchestrates a full simulation run.

**What it does:**
1. Parses CLI arguments (`--world_idx`, `--gui`, `--out`, `--record_bag`, `--laser_cost_factor`, etc.)
2. Sets environment variables for robot laser configuration
3. Launches Gazebo with the selected world file via `roslaunch`
4. Spawns Jackal at its initial pose
5. Starts `move_base` navigation stack
6. Sends a navigation goal to move_base via `actionlib`
7. Monitors robot state (position, collision, time) in a loop
8. Records outcome (success / collision / timeout) and computes navigation metric
9. Appends results to the output file

**Initial Poses:**
- Static worlds (0–299): `x=-2.25, y=3, yaw=1.57 rad` (pointing along the corridor)
- DynaBARN worlds (300–359): `x=11, y=0, yaw=3.14 rad`

**Navigation Goal:**
- Static: `x=0, y=0` (origin)
- DynaBARN: `x=-11, y=0`

**Success Condition:** Robot reaches within 1.0 m of goal within 100 seconds.

**Timeout:** 100 seconds of simulated time.

**Key CLI Flags:**
| Flag | Default | Description |
|---|---|---|
| `--world_idx` | 0 | World index to run (0–299 static, 300–359 DynaBARN) |
| `--gui` | False | Launch Gazebo with GUI (`gzclient`) |
| `--out` | `result.txt` | Output file for results |
| `--record_bag` | False | Record a ROS bag file |
| `--laser_cost_factor` | 0.5 | Costmap laser inflation factor |

**User Extension Point:** Lines 89–109 in `run.py` are marked as the section to replace with a custom navigation algorithm. The default launches `move_base_DWA.launch`.

---

#### 1.2 `gazebo_simulation.py` — Simulation Interface Class

Provides a clean Python API to interact with Gazebo during a run.

**Class:** `GazeboSimulation`

| Method | Description |
|---|---|
| `reset()` | Resets Gazebo world to initial state |
| `pause()` | Pauses Gazebo physics |
| `unpause()` | Resumes Gazebo physics |
| `get_model_state()` | Returns robot pose `(x, y, z, angle_rad)` using quaternion-to-euler conversion |
| `get_laser_scan()` | Returns latest `/scan` LaserScan message |
| `get_hard_collision()` | Returns `True` if robot has collided with an obstacle (not ground) |

**ROS Topics Subscribed:**
- `/collision` — Published by `collision_publisher_node`; boolean collision flag
- `/ground_truth/state` — Ground truth odometry from Gazebo

**ROS Services Used:**
- `/gazebo/reset_world`
- `/gazebo/pause_physics`
- `/gazebo/unpause_physics`
- `/gazebo/get_model_state`

---

#### 1.3 `report_test.py` — Evaluation Script

Reads a results file and reports aggregate metrics over the standard 50-world evaluation set.

**Evaluation worlds:** Indices `0, 6, 12, 18, ..., 294` (every 6th world from 0–299)

**Metrics reported:**
| Metric | Description |
|---|---|
| Avg Time | Mean navigation time across successful runs |
| Avg Metric | Mean navigation metric score |
| Avg Success | Success rate (%) |
| Avg Collision | Collision rate (%) |
| Avg Timeout | Timeout rate (%) |

**Navigation Metric Formula:**
```
nav_metric = success * (optimal_time / 2) / clip(actual_time, 2*optimal_time, 8*optimal_time)
```
- `optimal_time` = optimal path length / 0.5 (Jackal's nominal speed)
- Score is 0 on failure, and approaches 1.0 for near-optimal navigation

---

#### 1.4 `jackal_helper` — C++ Plugin and Launch Configurations

**Path:** `the-barn-challenge/jackal_helper/`

##### C++ Source: `collision_publisher.cpp`

A Gazebo model plugin that publishes collision events to ROS.

- Subscribes to Gazebo's internal contact manager
- Filters out ground plane contacts (only reports obstacle hits)
- Monitors robot altitude to detect if the robot has flipped or become airborne
- Publishes `True/False` to `/collision` topic at each physics step

##### Launch Files

**`launch/gazebo_launch.launch`**
- Starts `gazebo_ros empty_world` with the selected `.world` SDF file
- Spawns the Jackal URDF at `(0, 0, 1.0)` — note: `run.py` uses a service call to relocate it after spawn
- Loads the `collision_publisher_node` plugin
- Respects `gui:=true/false` argument for headless vs. GUI mode
- Sets laser model environment variables before spawning

**`launch/move_base_DWA.launch`**
- Starts `move_base` node
- Global planner: `NavfnROS` (A* / Dijkstra)
- Local planner: `TrajectoryPlannerROS` (DWA — Dynamic Window Approach)
- Loads costmap and planner configs from `configs/params/`

**`launch/move_base_eband.launch`**
- Same as above but uses `eband_local_planner/EBandPlannerROS` as local planner
- Useful for comparing planners

##### Configuration Files (`configs/params/`)

**`move_base_params.yaml`**
```yaml
controller_frequency: 20   # Hz
planner_frequency: 20      # Hz
recovery_behaviors: [conservative_reset, rotate_recovery, aggressive_reset]
```

**`costmap_common_params.yaml`**
```yaml
obstacle_range: 2.5        # meters
raytrace_range: 3.0        # meters
footprint: [[-0.21,-0.165],[-0.21,0.165],[0.21,0.165],[0.21,-0.165]]  # Jackal footprint
inflation_radius: 0.3      # meters
```

**`base_local_planner_params.yaml`** (DWA)
```yaml
max_vel_x: 0.5             # m/s
max_vel_theta: 1.57        # rad/s (~90 deg/s)
min_vel_x: 0.0
sim_time: 2.0              # seconds ahead to simulate
xy_goal_tolerance: 0.25    # meters
yaw_goal_tolerance: 0.157  # radians (~9 deg)
```

---

#### 1.5 World Files

**Path:** `the-barn-challenge/jackal_helper/worlds/`

```
jackal_helper/worlds/
├── BARN/
│   ├── world_0.world ... world_299.world    # 300 static environments
│   └── path_files/
│       ├── path_0.npy ... path_299.npy      # Pre-computed optimal paths (NumPy arrays)
└── DynaBARN/
    └── world_0.world ... world_59.world     # 60 dynamic environments
```

> **Note:** A richer version of the dataset also lives at `~/Desktop/BARN_dataset/` and includes additional files (cspace grids, occupancy maps, per-world metrics). The world and path files there are equivalent and can be used interchangeably.

**World Format:** Gazebo SDF 1.6
- Ground plane + directional lighting
- Cylindrical obstacles of varying density and arrangement
- ODE physics engine
- Each world is procedurally generated with a unique obstacle configuration

**Path Files:** Each `.npy` file contains a 2D NumPy array of `(x, y)` waypoints representing the pre-computed optimal path through the environment. Used to calculate `optimal_time` for the navigation metric.

---

#### 1.6 Plugins Directory

**Path:** `the-barn-challenge/plugins/`

Contains 360 precompiled Gazebo shared libraries (`easylibobs_0.so` through `easylibobs_359.so`).

These are binary obstacle set plugins loaded dynamically by the world files. Each `.so` file corresponds to one world's obstacle configuration. The `GAZEBO_PLUGIN_PATH` environment variable is set to this directory before launching Gazebo.

---

### 2. `jackal` — Robot Description Packages

**Path:** `jackal_ws/src/jackal/`

Official Clearpath Robotics packages for the Jackal UGV.

#### 2.1 `jackal_description` — URDF/Xacro Robot Model

**Main file:** `urdf/jackal.urdf.xacro`

**Physical Specifications:**
| Parameter | Value |
|---|---|
| Total mass | ~20 kg |
| Chassis dimensions | 0.420 m × 0.310 m × 0.184 m |
| Wheelbase | 0.262 m |
| Track width | 0.37559 m |
| Wheel radius | 0.098 m |
| Wheel width | 0.040 m |
| Drive | 4-wheel differential |

**Sensor Accessories (controlled via env vars):**

| Environment Variable | Sensor Added |
|---|---|
| `JACKAL_LASER=1` | Front-facing 2D lidar |
| `JACKAL_LASER_MODEL=ust10` | Hokuyo UST-10 (270° FoV, 0.05–10 m range) |
| `JACKAL_LASER_OFFSET` | Mounting offset `x y z` |
| `JACKAL_FRONT_ACCESSORY_FENDER` | Front fender accessory mount |

**Joint structure:**
- `base_link` — chassis root
- 4× wheel links with continuous joints: `front_left_wheel`, `front_right_wheel`, `rear_left_wheel`, `rear_right_wheel`
- Laser link mounted on chassis via fixed joint
- `imu_link` — IMU sensor frame

**Transmissions:** Each wheel joint has a `SimpleTransmission` with hardware interface `EffortJointInterface`, allowing Gazebo's `ros_control` to drive the wheels.

#### 2.2 `jackal_control`

Provides `ros_control` configuration and teleop:
- `config/control.yaml` — PID gains and joint controller settings
- `launch/control.launch` — Starts joint state publisher and wheel controllers
- `launch/teleop.launch` — Keyboard/joystick teleoperation

#### 2.3 `jackal_msgs`

Custom message definitions:
- `msg/Drive.msg` — Wheel speed commands
- `msg/Feedback.msg` — Battery, motor, system feedback

#### 2.4 `jackal_navigation`

Navigation-specific configurations independent of the BARN challenge:
- Default costmap parameters
- Navigation launch files for real-robot deployment

---

### 3. `jackal_simulator` — Gazebo Simulation Support

**Path:** `jackal_ws/src/jackal_simulator/`

Provides the Gazebo integration layer for Jackal.

- **`jackal_gazebo/launch/spawn_jackal.launch`** — Spawns Jackal into an already-running Gazebo instance
- **`jackal_gazebo/worlds/`** — Example Gazebo world files (empty world, outdoor terrain)
- **`jackal_gazebo/media/`** — Textures and 3D models for environments

**Key dependencies:**
- `gazebo_ros_control` — Links `ros_control` to Gazebo actuators
- `hector_gazebo_plugins` — IMU and GPS simulation plugins

---

### 4. `jackal_desktop` — Visualization Tools

**Path:** `jackal_ws/src/jackal_desktop/`

Desktop tools for monitoring and visualization.

- **`jackal_viz/launch/view_model.launch`** — Opens RViz with URDF model (useful for checking URDF changes offline)
- **`jackal_viz/launch/view_robot.launch`** — Opens RViz with live robot topics (laser, odometry, TF)
- **`jackal_viz/launch/view_diagnostics.launch`** — Opens RQT diagnostics panel

---

### 5. `eband_local_planner` — Elastic Band Planner

**Path:** `jackal_ws/src/eband_local_planner/`

A `move_base` plugin implementing the **Elastic Band** local planning algorithm.

**Algorithm summary:** Takes the global plan path and treats it as an elastic band with internal tension and external repulsion forces from obstacles. Iteratively deforms the path to be smooth and collision-free, then commands velocities to follow it.

**Key source files:**
| File | Purpose |
|---|---|
| `eband_local_planner.cpp` | Core elastic band algorithm (bubble generation, force propagation) |
| `eband_local_planner_ros.cpp` | ROS plugin wrapper, implements `nav_core::BaseLocalPlanner` |
| `eband_trajectory_controller.cpp` | Converts elastic band to velocity commands |
| `eband_visualization.cpp` | Publishes elastic band markers to RViz |
| `conversions_and_types.cpp` | SE2 geometry utilities |

**Plugin registration:** Registered as `eband_local_planner/EBandPlannerROS` in `move_base` plugin system via `elp_plugin.xml`.

---

## Build System

**ROS Distribution:** Noetic (Ubuntu 20.04)

**Build tool:** `catkin_make`

### Build Steps

```bash
# Source ROS base
source /opt/ros/noetic/setup.bash

# Navigate to workspace
cd /home/egemen/barn_lab/jackal_ws

# Build all packages
catkin_make

# Source workspace overlay
source devel/setup.bash
```

### Environment Setup (required before every session)

```bash
source /opt/ros/noetic/setup.bash
source /home/egemen/barn_lab/jackal_ws/devel/setup.bash
```

Or use the helper script in `/home/egemen/Desktop/dream/run_barn.sh`.

---

## Python Virtual Environment

**Path:** `/home/egemen/barn_lab/nav_challenge/`

Isolated Python 3 environment for running the challenge scripts.

**Activate:**
```bash
source /home/egemen/barn_lab/nav_challenge/bin/activate
```

**Installed packages:**
- `numpy` — Path file loading, trajectory computation
- `rospkg` — ROS package path resolution from Python
- `defusedxml` — Safe XML parsing
- `netifaces` — Network interface enumeration (used by ROS)

---

## ROS Graph Overview

When a simulation run is active, the following nodes are running:

```
/gazebo                     ← Gazebo simulator node
/collision_publisher_node   ← Jackal helper plugin
/move_base                  ← Navigation stack
/robot_state_publisher      ← Publishes TF from URDF joints
/joint_state_publisher      ← Publishes wheel joint states
```

**Key Topics:**
| Topic | Type | Description |
|---|---|---|
| `/scan` | `sensor_msgs/LaserScan` | Hokuyo UST-10 laser readings |
| `/collision` | `std_msgs/Bool` | Hard collision flag |
| `/ground_truth/state` | `nav_msgs/Odometry` | Ground truth robot pose from Gazebo |
| `/odom` | `nav_msgs/Odometry` | Odometry (remapped from `odometry/filtered`) |
| `/move_base/goal` | `move_base_msgs/MoveBaseActionGoal` | Navigation goal |
| `/move_base/status` | `actionlib_msgs/GoalStatusArray` | Navigation status |
| `/cmd_vel` | `geometry_msgs/Twist` | Velocity commands to robot |
| `/tf` | `tf2_msgs/TFMessage` | Transform tree |

**TF Tree:**
```
map → odom → base_link → laser → ...
                       → imu_link
                       → front_left_wheel
                       → front_right_wheel
                       → rear_left_wheel
                       → rear_right_wheel
```

---

## Results File Format

Running `run.py` appends one line per episode to the output file:

```
world_idx  succeeded  collided  timed_out  time_elapsed  nav_metric
0          1          0         0          29.461        0.8234
1          0          1         0          14.220        0.0000
2          0          0         1          100.000       0.0000
```

Fields:
- `world_idx` — World number used
- `succeeded` — 1 if goal reached within 1m and 100s
- `collided` — 1 if hard obstacle contact detected
- `timed_out` — 1 if 100s elapsed without success
- `time_elapsed` — Simulated time in seconds
- `nav_metric` — Score 0.0–1.0 (0 on failure)

---

## Related Files in `/home/egemen/Desktop/dream/`

| File | Purpose |
|---|---|
| `barn_sim_commands.md` | Quick-reference build and run commands |
| `run_barn.sh` | Shell script to source environment and run world 0 |
