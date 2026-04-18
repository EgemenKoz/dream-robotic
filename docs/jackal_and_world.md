# Jackal Robot & BARN World Documentation

A reference document covering the Jackal UGV robot configuration, the BARN dataset world files, all associated file types, and the constraints/behavior of the simulation environment.

---

## Table of Contents

1. [Jackal Robot Overview](#jackal-robot-overview)
2. [Sensors](#sensors)
3. [Actuation & Velocity Limits](#actuation--velocity-limits)
4. [ROS Topics & TF Tree](#ros-topics--tf-tree)
5. [BARN Dataset Overview](#barn-dataset-overview)
6. [World Files (.world)](#world-files-world)
7. [C-Space Files (cspace_*.npy)](#c-space-files-cspace_npy)
8. [Grid Files (grid_*.npy)](#grid-files-grid_npy)
9. [Map Files (yaml + pgm)](#map-files-yaml--pgm)
10. [Path Files (path_*.npy)](#path-files-path_npy)
11. [Metrics Files](#metrics-files)
12. [Simulation Constraints & Episode Termination](#simulation-constraints--episode-termination)
13. [Navigation Metric Formula](#navigation-metric-formula)
14. [Navigation Stack Configuration](#navigation-stack-configuration)

---

## Jackal Robot Overview

The robot used in all BARN experiments is the **Clearpath Robotics Jackal UGV** (Unmanned Ground Vehicle).

| Property | Value |
|---|---|
| Drive type | 4-wheel differential drive |
| Mass | ~20 kg |
| Chassis length | 0.420 m |
| Chassis width | 0.310 m |
| Chassis height | 0.184 m |
| Wheelbase | 0.262 m |
| Track width | 0.376 m |
| Wheel radius | 0.098 m |
| Wheel width | 0.040 m |

**Footprint used in costmap planning:**
```
[-0.21, -0.165], [-0.21, 0.165], [0.21, 0.165], [0.21, -0.165]
```

**Starting pose in all static BARN worlds:**
- x = -2.25 m, y = 3.0 m, yaw = 1.57 rad (facing forward along the corridor)

**Goal pose:**
- x = -2.25 m, y = 13.0 m (10 m straight ahead from start)
- Success radius: **1.0 m** from goal center

---

## Sensors

### Hokuyo UST-10 2D LiDAR

The primary (and only) exteroceptive sensor in the BARN setup.

| Property | Value |
|---|---|
| Model | Hokuyo UST-10 |
| Field of view | 270 degrees |
| Min range | 0.05 m |
| Max range | 10 m |
| Angular resolution | ~0.25 degrees |
| Mount offset (x, y, z) | -0.065 m, 0, 0.01 m from chassis center |
| ROS topic | `/scan` |
| Message type | `sensor_msgs/LaserScan` |

Enabled via environment variables set before launching the simulation:
```bash
JACKAL_LASER=1
JACKAL_LASER_MODEL=ust10
JACKAL_LASER_OFFSET="-0.065 0 0.01"
```

### IMU (Inertial Measurement Unit)

Built into the robot. Used for odometry and localization filtering. Simulated via the `hector_gazebo_plugins` package. There is no dedicated ROS topic exposed to navigation — it feeds into the odometry filter internally.

---

## Actuation & Velocity Limits

The robot is driven by sending `geometry_msgs/Twist` messages to `/cmd_vel`.

| Property | Hardware Limit | BARN Config Limit |
|---|---|---|
| Max linear velocity (x) | 2.0 m/s | **0.5 m/s** |
| Max angular velocity (yaw) | ~3.14 rad/s | **1.57 rad/s** |
| Min linear velocity | 0.0 m/s | 0.0 m/s |
| Controller frequency | — | 20 Hz |

The BARN limits are enforced at the move_base level (DWA/EBand local planner parameters), not at the hardware abstraction layer.

---

## ROS Topics & TF Tree

**Key topics:**

| Topic | Type | Description |
|---|---|---|
| `/scan` | `sensor_msgs/LaserScan` | LiDAR readings from Hokuyo UST-10 |
| `/cmd_vel` | `geometry_msgs/Twist` | Velocity commands to the robot |
| `/odom` | `nav_msgs/Odometry` | Filtered odometry |
| `/ground_truth/state` | `nav_msgs/Odometry` | Gazebo ground truth pose |
| `/collision` | `std_msgs/Bool` | True when robot contacts an obstacle |
| `/tf` | — | Transform tree |

**TF tree:**
```
map
 └── odom
      └── base_link
           ├── laser_link
           ├── imu_link
           ├── front_left_wheel
           ├── front_right_wheel
           ├── rear_left_wheel
           └── rear_right_wheel
```

---

## BARN Dataset Overview

The BARN (**B**enchmark for **A**utonomous **R**obot **N**avigation) dataset contains **300 static obstacle environments** designed to evaluate autonomous navigation algorithms.

| Component | Count | Format | Approx. Size |
|---|---|---|---|
| World files | 300 | Gazebo SDF 1.6 (`.world`) | 161 MB |
| C-space grids | 300 | NumPy (`.npy`) | 2.4 MB |
| Occupancy grids | 300 | NumPy (`.npy`) | 2.4 MB |
| Optimal path files | 300 | NumPy (`.npy`) | 1.2 MB |
| Map YAML configs | 300 | YAML | — |
| Map images | 300 | PGM | 2.4 MB |
| Metrics | 300 | NumPy (`.npy`) | 1.2 MB |
| Normalized metrics | 300 | NumPy (`.npy`) | 1.2 MB |
| **Total** | **2,400 files** | mixed | **~172 MB** |

Worlds are indexed 0–299. The standard evaluation subset uses every 6th world: `[0, 6, 12, 18, ..., 294]` (50 worlds).

**Dataset location on this machine:** `/home/egemen/Desktop/BARN_dataset/`

---

## World Files (.world)

**Directory:** `BARN_dataset/world_files/`
**Format:** Gazebo SDF 1.6 (XML)
**Files:** `world_0.world` — `world_299.world`

Each world file is a complete Gazebo simulation scene. It contains:

### Scene elements

- **Lighting:** One directional sun light source
- **Ground plane:** 100 m × 100 m, high-friction surface
- **Obstacles:** 418 static cylinders placed at varied positions unique to each world

### Obstacle specification

| Property | Value |
|---|---|
| Shape | Cylinder |
| Radius | 0.075 m |
| Height | 1.0 m |
| Static | Yes (does not move) |
| Color | Blue (RGB: 0.152, 0.379, 0.720) |

Example obstacle definition:
```xml
<model name='unit_cylinder_0'>
  <static>1</static>
  <pose frame=''>-0.075000 0.075000 0.000000 0 0 0</pose>
  <link name='link'>
    <collision>
      <geometry>
        <cylinder>
          <radius>0.075000</radius>
          <length>1</length>
        </cylinder>
      </geometry>
    </collision>
    <visual>
      <geometry>
        <cylinder>
          <radius>0.075000</radius>
          <length>1</length>
        </cylinder>
      </geometry>
    </visual>
  </link>
</model>
```

### Physics engine settings

| Setting | Value |
|---|---|
| Engine | ODE (Open Dynamics Engine) |
| Max step size | 0.001 s |
| Real-time factor | 1.0 |
| Real-time update rate | 1000 Hz |

---

## C-Space Files (cspace_*.npy)

**Directory:** `BARN_dataset/cspace_files/`
**Files:** `cspace_0.npy` — `cspace_299.npy`
**Format:** NumPy array, shape `(30, 30)`, dtype `int64`
**Values:** `0` = free, `1` = occupied

### What is C-Space?

C-space (Configuration Space) is a representation of the environment that accounts for the physical size of the robot. Instead of showing where obstacles are as raw points, the C-space inflates each obstacle by the robot's footprint radius. Any cell marked `1` means the **center** of the robot cannot be placed there without causing a collision.

This is different from the raw grid (which shows where obstacles actually are) — the C-space is what path planners actually use to guarantee collision-free paths for a robot with a non-zero footprint.

**Example (world_0):**
- Grid size: 30 × 30 = 900 cells
- Occupied cells: 464 (51.6%)
- Free cells: 436 (48.4%)

The relatively high occupation rate (~50%) compared to the grid files (~12.6%) is expected: inflating each small cylinder by the robot footprint covers significantly more area.

---

## Grid Files (grid_*.npy)

**Directory:** `BARN_dataset/grid_files/`
**Files:** `grid_0.npy` — `grid_299.npy`
**Format:** NumPy array, shape `(30, 30)`, dtype same as cspace
**Values:** `0` = free, `1` = obstacle present

These are the raw cellular-automaton-generated occupancy grids used to **generate** each world. Each cell represents a physical tile in the environment. An occupied cell means there is a cylinder placed in that tile.

**Example (world_0):**
- Grid size: 30 × 30 = 900 cells
- Filled cells: 113 (12.6%)
- Free cells: 787 (87.4%)

**Coordinate mapping:**
```
gazebo_x = col * (2 * RADIUS) + x_offset
gazebo_y = row * (2 * RADIUS) + y_offset
# where RADIUS = 0.075 m
```

---

## Map Files (yaml + pgm)

**Directory:** `BARN_dataset/map_files/`
**Files:**
- `yaml_0.yaml` — `yaml_299.yaml` (ROS map configs)
- `map_pgm_0.pgm` — `map_pgm_299.pgm` (grayscale map images)

These files are in the format used by the ROS `map_server` package, which serves a 2D occupancy grid to the navigation stack.

**YAML config example:**
```yaml
image: map_pgm_0.pgm
resolution: 0.15        # meters per pixel
origin: [-4.5, 0.0, 0]  # bottom-left corner of map in world coordinates
occupied_thresh: 0.50   # pixels darker than this = occupied
free_thresh: 0.50       # pixels lighter than this = free
negate: 0
```

**PGM files:** Portable Graymap image files where each pixel encodes occupancy. White = free, black = occupied. These are what you'd visualize in RViz as the static map layer.

The `map_server` node reads the YAML, loads the PGM, and publishes the `/map` topic used by the global planner.

---

## Path Files (path_*.npy)

**Directory:** `BARN_dataset/path_files/`
**Files:** `path_0.npy` — `path_299.npy`
**Format:** NumPy array, shape `(N, 2)`, dtype float
**Values:** Each row is `[row, col]` in grid coordinates

These are pre-computed **optimal paths** through each world, calculated offline (likely using A* or Dijkstra on the C-space grid). The path is a sequence of grid waypoints from the start cell to the goal cell.

**Example (world_0):** 43 waypoints

**Primary use:** Computing the `optimal_time` baseline for the navigation metric:
```python
optimal_time = optimal_path_length_in_meters / 0.5  # nominal speed = 0.5 m/s
```

---

## Metrics Files

### metrics_*.npy

**Directory:** `BARN_dataset/metrics_files/`
**Format:** NumPy array, shape `(5,)`, dtype `float64`

Five scalar measurements characterizing the difficulty and geometry of each world:

| Index | Metric | Description |
|---|---|---|
| 0 | Distance to Closest Obstacle | Minimum clearance from robot path to any obstacle |
| 1 | Average Visibility | How far the robot can see on average (open-ness) |
| 2 | Dispersion | How spread out the obstacles are |
| 3 | Characteristic Dimension | Typical scale of obstacle clusters |
| 4 | Tortuosity | How winding the optimal path is (1.0 = straight line) |

**Example (world_0):** `[3.676, 6.846, 3.442, 5.109, 1.638]`

### norm_metrics_*.npy

**Directory:** `BARN_dataset/norm_metrics_files/`
**Format:** Same structure, z-score normalized (subtract mean, divide by std)

Used for comparing worlds on equal footing and for difficulty stratification in experiments.

**Example (world_0):** `[1.409, 0.461, -1.013, 0.398, 2.992]`

---

## Simulation Constraints & Episode Termination

Each simulation run is one **episode**. An episode has the following hard constraints:

### Time Limit

- **Maximum episode duration: 100 seconds** (simulated time, not wall clock)
- If the robot has not reached the goal within 100 s, the episode terminates as **timed out**
- Result logged: `timed_out=True`, navigation metric = 0

### Collision Constraint

- Collision is detected by a custom Gazebo C++ plugin (`collision_publisher_node`)
- The plugin monitors the ODE contact manager each physics step
- Ground plane contacts are filtered out — only obstacle contacts count
- Also detects if the robot **flips** or becomes airborne
- Published to `/collision` topic as `std_msgs/Bool`
- On collision: episode is flagged `collided=True`, navigation metric = 0

There is **no automatic episode reset on collision** — the runner script (`run_barn_custom.py`) reads the `/collision` topic and decides to terminate the episode.

### Success Condition

- Robot center reaches within **1.0 m** of the goal position
- Must happen before the 100 s time limit
- Result: `succeeded=True`, navigation metric computed from actual vs optimal time

### Summary

| Outcome | Trigger | Navigation Metric |
|---|---|---|
| Success | Within 1.0 m of goal, time < 100 s | > 0 (see formula below) |
| Collision | Contact with any obstacle | 0 |
| Timeout | 100 s elapsed, goal not reached | 0 |

---

## Navigation Metric Formula

The BARN challenge uses a single scalar metric to score each run:

```
nav_metric = success * (optimal_time / 2) / clip(actual_time, 2*optimal_time, 8*optimal_time)
```

Where:
- `success` = 1 if goal reached in time, else 0
- `optimal_time` = optimal path length in meters / 0.5 m/s
- `actual_time` = simulated seconds elapsed until goal reached
- `clip(val, low, high)` = clamp value to range [low, high]

**Intuition:** A perfect run (actual_time = optimal_time) scores ~0.5. A slower run scores less. The clip prevents the metric from exploding for very fast or very slow completions. Failure always scores 0.

**Results log format:**
```
world_idx  succeeded  collided  timed_out  time_elapsed  nav_metric
```

---

## Navigation Stack Configuration

The default navigation stack is ROS `move_base`:

| Setting | Value |
|---|---|
| Global planner | NavfnROS (A* / Dijkstra) |
| Local planner | DWA (default) or EBand |
| Obstacle detection range | 2.5 m |
| Raytrace range | 3.0 m |
| Inflation radius | 0.3 m |
| Planner frequency | 20 Hz |
| Controller frequency | 20 Hz |
| XY goal tolerance | 0.25 m |
| Yaw goal tolerance | 0.157 rad (~9 deg) |
| DWA sim time | 2.0 s |

The global planner uses the static map from `map_server`. The local planner uses the live `/scan` data to avoid obstacles dynamically within the global plan.
