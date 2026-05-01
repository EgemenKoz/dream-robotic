# Programmer's Guide — BARN Challenge & Jackal Robot

## What This Guide Covers

This guide explains how to work with the BARN navigation challenge simulation end-to-end: how to run simulations, what world files exist, how the Jackal robot is configured, how navigation is set up, and how to plug in your own navigation algorithm.

---

## Quick Start

### 1. Set Up Environment

Every terminal session needs these sourced:

```bash
source /opt/ros/noetic/setup.bash
source /home/egemen/barn_lab/jackal_ws/devel/setup.bash
```

Or run the helper script from the `dream` project:

```bash
bash /home/egemen/Desktop/dream/run_barn.sh
```

### 2. Run a Single World (Headless)

```bash
cd /home/egemen/Desktop/dream
python3 run_barn_custom.py --world_idx 0
```

### 3. Run a Single World with GUI

```bash
python3 run_barn_custom.py --world_idx 0 --gui
```

This opens both `gzserver` (physics) and `gzclient` (3D viewer). Useful for watching the robot navigate.

### 4. Run and Save Results

```bash
python3 run_barn_custom.py --world_idx 42 --out my_results.txt
```

### 5. Evaluate Over the Standard 50 Worlds

After running all 50 evaluation worlds:

```bash
python3 /home/egemen/barn_lab/jackal_ws/src/the-barn-challenge/report_test.py --out my_results.txt
```

Outputs success rate, collision rate, average navigation metric, etc.

---

## World Files

### Static BARN Worlds (Indices 0–299)

**Location:** `jackal_helper/worlds/BARN/world_0.world` through `world_299.world`
(or equivalently `~/Desktop/BARN_dataset/world_files/world_0.world` — same files)

These are procedurally generated indoor-like environments packed with cylindrical obstacles of varying density. Each world is an SDF file (Gazebo format) defining:
- Ground plane
- Scene lighting
- An array of cylindrical obstacles (different positions per world)
- Physics settings (ODE engine)

The robot starts at `(-2.25, 3)` facing toward the origin and must navigate to `(0, 0)`.

**Optimal paths** for each static world are stored in:
```
worlds/BARN/path_files/path_0.npy ... path_299.npy
```
Each `.npy` file is a NumPy array of `(x, y)` waypoints. Load them with:
```python
import numpy as np
path = np.load("worlds/BARN/path_files/path_0.npy")
# path.shape = (N, 2), each row is (x, y)
```

### Dynamic DynaBARN Worlds (Indices 300–359)

**Location:** `worlds/DynaBARN/world_0.world` through `world_59.world`

These environments contain moving obstacles. World index in `run.py` maps to DynaBARN as:
- `world_idx 300` → `DynaBARN/world_0.world`
- `world_idx 359` → `DynaBARN/world_59.world`

The robot starts at `(11, 0)` and must reach `(-11, 0)`.

### Selecting a World

```bash
# Static world 50
python3 run.py --world_idx 50

# DynaBARN world (world_idx - 300 = DynaBARN index)
python3 run.py --world_idx 305   # uses DynaBARN/world_5.world
```

---

## The Jackal Robot

### Physical Configuration

The Jackal is a compact 4-wheeled ground vehicle made by Clearpath Robotics.

| Property | Value |
|---|---|
| Drive type | 4-wheel differential |
| Max speed | ~2 m/s (limited to 0.5 m/s in BARN config) |
| Footprint | 0.42 m × 0.31 m |
| Wheel radius | 0.098 m |
| Weight | ~20 kg |

### Laser Sensor (UST-10)

In the BARN simulation, the Jackal is equipped with a **Hokuyo UST-10** 2D laser scanner.

| Property | Value |
|---|---|
| Model | Hokuyo UST-10 |
| Field of view | 270° |
| Range | 0.05 m – 10 m |
| Resolution | ~0.25° |
| Mounting offset | -0.065 m (x), 0 (y), 0.01 m (z) from chassis center |
| ROS topic | `/scan` (type: `sensor_msgs/LaserScan`) |

These are set by environment variables before launching:
```python
os.environ["JACKAL_LASER"] = "1"
os.environ["JACKAL_LASER_MODEL"] = "ust10"
os.environ["JACKAL_LASER_OFFSET"] = "-0.065 0 0.01"
```

### Reading Laser Data in Python

```python
import rospy
from sensor_msgs.msg import LaserScan

def laser_callback(msg):
    ranges = msg.ranges          # list of distances (floats), NaN/Inf if out of range
    angle_min = msg.angle_min    # start angle in radians
    angle_max = msg.angle_max    # end angle in radians
    angle_increment = msg.angle_increment  # radians between measurements

rospy.Subscriber("/scan", LaserScan, laser_callback)
```

### Getting Robot Pose

Ground truth pose from Gazebo (no noise):
```python
from gazebo_simulation import GazeboSimulation
sim = GazeboSimulation(init_position=[-2.25, 3, 1.57])
x, y, z, angle = sim.get_model_state()
```

Or subscribe directly:
```python
from nav_msgs.msg import Odometry
rospy.Subscriber("/ground_truth/state", Odometry, callback)
```

### Sending Velocity Commands

```python
from geometry_msgs.msg import Twist
pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

cmd = Twist()
cmd.linear.x = 0.3   # m/s forward
cmd.angular.z = 0.5  # rad/s counter-clockwise
pub.publish(cmd)
```

---

## Navigation Stack

The default navigation uses **ROS move_base** with the following setup:

### Component Overview

```
LaserScan (/scan)
    │
    ▼
Costmap 2D (global + local)
    │
    ├─► Global Planner (NavfnROS / A*)  ──► Global path
    │
    └─► Local Planner (DWA or EBand)   ──► /cmd_vel
```

### Global Planner: NavfnROS

Uses A* (or Dijkstra) on the global costmap to compute a collision-free path from the robot's current position to the goal. Called when a new goal is set or when the robot gets stuck.

### Local Planner Options

Two local planners are available via separate launch files:

#### Option A: DWA (Default)

**Launch:** `move_base_DWA.launch`

Dynamic Window Approach — samples velocity commands, simulates trajectories forward in time, and picks the best one based on clearance, goal progress, and smoothness.

Key params (`base_local_planner_params.yaml`):
```yaml
max_vel_x: 0.5        # m/s
max_vel_theta: 1.57   # rad/s
sim_time: 2.0         # seconds to simulate forward
xy_goal_tolerance: 0.25
```

To use: set `LOCAL_PLANNER = "dwa"` in `configs/config.py` (this is the default).

#### Option B: Elastic Band (EBand)

**Launch:** `move_base_eband.launch`

Deforms the global path into a smooth, collision-avoiding trajectory by treating path waypoints as an elastic band with obstacle repulsion. Often smoother than DWA in cluttered spaces.

To use: set `LOCAL_PLANNER = "eband"` in `configs/config.py`.

### Costmap Configuration

Both global and local costmaps use these shared parameters:

```yaml
obstacle_range: 2.5       # Add obstacles within 2.5 m to costmap
raytrace_range: 3.0       # Clear free space within 3.0 m
inflation_radius: 0.3     # Inflate obstacles by 0.3 m (safety buffer)
footprint:                # Rectangle matching Jackal body
  [[-0.21, -0.165], [-0.21, 0.165], [0.21, 0.165], [0.21, -0.165]]
```

### Setting a Navigation Goal Programmatically

```python
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Quaternion
import tf

client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
client.wait_for_server()

goal = MoveBaseGoal()
goal.target_pose.header.frame_id = "odom"
goal.target_pose.pose.position.x = 0.0
goal.target_pose.pose.position.y = 0.0
goal.target_pose.pose.orientation = Quaternion(
    *tf.transformations.quaternion_from_euler(0, 0, 0)
)
client.send_goal(goal)
client.wait_for_result()
```

---

## Implementing Your Own Navigation Algorithm

The BARN challenge is designed to be extended. Here's how:

### Where to Add Your Code

In `run_barn_custom.py`, look for **Section 1** (marked with the comment block). It currently launches whichever planner is set in `configs/config.py`. Replace the `nav_launch` line or add your own node launch there.

### Minimal Custom Navigator Template

```python
# my_navigator.py
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import tf

class MyNavigator:
    def __init__(self):
        rospy.init_node("my_navigator")
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        rospy.Subscriber("/scan", LaserScan, self.laser_cb)
        rospy.Subscriber("/ground_truth/state", Odometry, self.odom_cb)
        self.ranges = []
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

    def laser_cb(self, msg):
        self.ranges = list(msg.ranges)

    def odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, self.yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])

    def run(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            cmd = Twist()
            # --- Your logic here ---
            cmd.linear.x = 0.3
            cmd.angular.z = 0.0
            # -----------------------
            self.cmd_pub.publish(cmd)
            rate.sleep()

if __name__ == "__main__":
    nav = MyNavigator()
    nav.run()
```

### Using `GazeboSimulation` in Your Code

```python
import sys
sys.path.insert(0, "/home/egemen/barn_lab/jackal_ws/src/the-barn-challenge")
from gazebo_simulation import GazeboSimulation
from configs.config import WorldConfig

sim = GazeboSimulation(init_position=WorldConfig.STATIC_INIT_POSITION)

# Get robot state
x, y, z, angle = sim.get_model_state()

# Get laser scan
scan = sim.get_laser_scan()

# Check collision
if sim.get_hard_collision():
    print("Robot collided!")

# Pause/unpause physics
sim.pause()
sim.unpause()

# Reset world
sim.reset()
```

---

## Collision Detection

The `collision_publisher_node` (C++ Gazebo plugin) monitors contacts in physics simulation and publishes to `/collision`.

```python
# Check for collision via GazeboSimulation
collided = sim.get_hard_collision()  # Returns True/False

# Or subscribe directly
from std_msgs.msg import Bool
rospy.Subscriber("/collision", Bool, lambda msg: print(msg.data))
```

**What counts as a collision:** Any contact between the robot body and an obstacle cylinder. Ground contact is filtered out. If the robot flips or becomes airborne, that also triggers the collision flag.

---

## Visualization

### Launch RViz (Live Robot)

```bash
roslaunch jackal_viz view_robot.launch
```

Shows: laser scan, TF tree, odometry, costmap (if move_base is running), global and local plans.

### Launch RViz (URDF Only / Offline)

```bash
roslaunch jackal_viz view_model.launch
```

Shows the Jackal URDF model without requiring a running simulation.

### RQT Diagnostics

```bash
roslaunch jackal_viz view_diagnostics.launch
```

---

## Running Multiple Worlds in a Loop (Batch Evaluation)

```bash
cd /home/egemen/Desktop/dream
for i in $(seq 0 6 294); do
    python3 run_barn_custom.py --world_idx $i --out results.txt
done
python3 /home/egemen/barn_lab/jackal_ws/src/the-barn-challenge/report_test.py --out results.txt
```

This runs all 50 evaluation worlds and generates aggregate metrics.

---

## Recording ROS Bags

```bash
cd /home/egemen/Desktop/dream
python3 run_barn_custom.py --world_idx 0 --record_bag
```

A bag file is created in the current directory. Replay with:

```bash
rosbag play <bagfile>.bag
```

---

## Common Issues and Fixes

| Problem | Likely Cause | Fix |
|---|---|---|
| Gazebo hangs at startup | Previous Gazebo instance still running | `killall gzserver gzclient` |
| `GAZEBO_PLUGIN_PATH` error | Env not set | Run `source devel/setup.bash` first |
| Robot immediately collides | Spawn position inside obstacle | Try a different `world_idx` |
| `move_base` not found | Workspace not built | Run `catkin_make` in `jackal_ws/` |
| Laser topic empty | `JACKAL_LASER` not set | Set env vars before launching |
| Very slow simulation | GUI consuming resources | Run headless (no `--gui`) |

---

## Key File Reference

**This repo (`~/Desktop/dream/`):**

| File | Purpose |
|---|---|
| `run_barn_custom.py` | Main entry point — run simulations |
| `configs/config.py` | All simulation, robot, and nav settings |
| `configs/rl_config.py` | RL environment, logging, and eval defaults |
| `configs/mppi_config.py` | MPPI controller hyperparameters |
| `rl/env/barn_env.py` | Gymnasium environment wrapping Gazebo |
| `rl/controllers/mppi.py` | MPPI optimal controller (pure numpy) |
| `rl/logging/rl_logger.py` | Per-run CSV + human-readable training logger |
| `rl/tests/test_barn_env.py` | Unit and Gazebo integration tests |
| `docs/rl_guide.md` | Full RL package documentation |
| `docs/barn_lab_documentation.md` | Full barn_lab reference documentation |
| `docs/programmers_guide.md` | This guide |

**barn_lab workspace (`~/barn_lab/jackal_ws/src/`):**

| File | Purpose |
|---|---|
| `the-barn-challenge/gazebo_simulation.py` | Python API for Gazebo interaction |
| `the-barn-challenge/report_test.py` | Evaluate results over 50 worlds |
| `jackal_helper/launch/gazebo_launch.launch` | Launches Gazebo + Jackal |
| `jackal_helper/launch/move_base_DWA.launch` | Launches move_base with DWA |
| `jackal_helper/launch/move_base_eband.launch` | Launches move_base with EBand |
| `jackal_helper/configs/params/*.yaml` | Navigation tuning parameters |
| `jackal_helper/src/collision_publisher.cpp` | Gazebo collision detection plugin |
| `jackal/jackal_description/urdf/jackal.urdf.xacro` | Jackal robot URDF model |
| `jackal_helper/plugins/easylibobs_*.so` | Precompiled obstacle set Gazebo plugins |

**BARN Dataset (`~/Desktop/BARN_dataset/`):**

| File | Purpose |
|---|---|
| `world_files/world_*.world` | 300 static Gazebo environments |
| `path_files/path_*.npy` | Pre-computed optimal paths |
| `grid_files/grid_*.npy` | 30×30 occupancy grids |
| `cspace_files/cspace_*.npy` | Jackal C-space occupancy grids |
| `map_files/yaml_*.yaml` + `map_pgm_*.pgm` | ROS map_server compatible maps |
| `metrics_files/metrics_*.npy` | Per-world metrics (visibility, tortuosity, etc.) |
