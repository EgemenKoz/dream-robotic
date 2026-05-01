# RL Package Guide — BARN Gym Integration

Covers setup, configuration, environment API, logging, and running RL training in the BARN Gazebo simulation.

---

## Table of Contents

1. [Overview](#overview)
2. [Folder Structure](#folder-structure)
3. [Setup & Dependencies](#setup--dependencies)
4. [Configuration Reference](#configuration-reference)
5. [BarnEnv — Environment API](#barnenv--environment-api)
6. [RLLogger — Training Logger](#rllogger--training-logger)
7. [Running Training](#running-training)
8. [MPPI Controller](#mppi-controller)
9. [Running the Tests](#running-the-tests)
10. [Output Files](#output-files)
11. [Troubleshooting](#troubleshooting)

---

## Overview

The `rl/` package wraps the BARN Gazebo simulation as a [Gymnasium](https://gymnasium.farama.org/) environment (`gym.Env`). The RL policy controls the robot directly via `cmd_vel` (velocity commands) — it does **not** use move_base. This is standard end-to-end RL: scan observation → policy → velocity action.

```
Observation: [scan (N rays, 0–1), dist_to_goal (0–1), angle_to_goal (−1–1)]
Action:      [linear_x (m/s), angular_z (rad/s)]
```

The environment is compatible with any RL library that consumes a gymnasium interface: **stable-baselines3**, RLlib, CleanRL, Tianshou, etc.

---

## Folder Structure

```
rl/
├── __init__.py              exports BarnEnv
├── env/
│   ├── __init__.py
│   └── barn_env.py          BarnEnv(gym.Env) — main environment
├── logging/
│   ├── __init__.py
│   └── rl_logger.py         RLLogger — per-run CSV + human-readable log
└── tests/
    ├── conftest.py           registers pytest marks
    ├── __init__.py
    └── test_barn_env.py      49 unit tests + 7 Gazebo integration tests

configs/
└── rl_config.py             all RL defaults (edit this to tune the env)

results/rl/
├── episodes.csv             quick episode log from BarnEnv
└── runs/
    └── {run_id}/
        ├── run_config.json  hyperparameter snapshot
        ├── episodes.csv     full per-episode CSV (real-time flushed)
        └── training.log     human-readable progress + final summary
```

---

## Setup & Dependencies

### Python packages

```bash
pip3 install gymnasium numpy
```

### ROS setup (same as for `run_barn_custom.py`)

```bash
source /opt/ros/noetic/setup.bash
source ~/barn_lab/jackal_ws/devel/setup.bash
source ~/barn_lab/moborobo_ws/devel/setup.bash   # moborobo only
```

### Moborobo: pointcloud_to_laserscan

The moborobo RS-16 LiDAR publishes a 3D point cloud (`/rs_points`). The environment subscribes to the 2D `/scan` topic produced by `pointcloud_to_laserscan`. This node is launched automatically in **managed mode**. In **unmanaged mode** you must start it yourself:

```bash
roslaunch moborobo barn_move_base.launch
```

move_base will run but idle — it never receives a goal, so it doesn't conflict with the RL policy's `cmd_vel` commands.

---

## Configuration Reference

All RL defaults live in `configs/rl_config.py`. Import them like the other config classes:

```python
from configs.rl_config import RLEnvConfig, RLLogConfig, RLEvalConfig
```

### RLEnvConfig — Environment

| Setting | Default | Description |
|---|---|---|
| `NUM_SCAN_RAYS` | `360` | Laser scan bins in observation. Raw scan is resampled to this count. |
| `STEP_DURATION` | `0.1` s | Physics time per step (≈10 Hz policy). |
| `MAX_EPISODE_STEPS` | `500` | Hard truncation limit per episode (500 × 0.1 s = 50 s). |
| `REWARD_COLLISION` | `−100.0` | Terminal reward on collision. |
| `REWARD_GOAL` | `+100.0` | Terminal reward on reaching goal. |
| `REWARD_STEP` | `−0.01` | Per-step living penalty. |
| `REWARD_SHAPING` | `True` | Add `(prev_dist − curr_dist)` dense shaping each step. |

### RLLogConfig — Logging

| Setting | Default | Description |
|---|---|---|
| `LOG_DIR` | `results/rl/` | Root for BarnEnv's per-episode CSV. |
| `RUNS_DIR` | `results/rl/runs/` | RLLogger writes one subdirectory per run here. |
| `ROLLING_WINDOW` | `100` | Episodes used for rolling stats in `log_progress()`. |
| `LOG_EVERY_STEPS` | `1000` | Suggested call frequency for `log_progress()` in training loops. |

### RLEvalConfig — Evaluation

| Setting | Default | Description |
|---|---|---|
| `EVAL_WORLDS` | `[0, 6, 12, ..., 294]` | Standard 50-world evaluation set. |
| `QUICK_EVAL_WORLDS` | 10 worlds | Fast mid-training check. |
| `EPISODES_PER_WORLD` | `1` | Runs per world during eval. |
| `MAX_EVAL_STEPS` | `600` | Step limit for eval episodes. |

### Changing a default

Edit `configs/rl_config.py` — the change takes effect everywhere automatically:

```python
# configs/rl_config.py
class RLEnvConfig:
    NUM_SCAN_RAYS = 180          # halve observation size
    STEP_DURATION = 0.2          # 5 Hz policy
    REWARD_SHAPING = False       # sparse reward only
```

### Overriding per-instance

Constructor kwargs override the config for that instance only:

```python
env = BarnEnv(num_scan_rays=180, reward_shaping=False)
```

---

## BarnEnv — Environment API

### Constructor

```python
from rl import BarnEnv

env = BarnEnv(
    robot="moborobo",          # "jackal" | "moborobo"
    world_idx=30,              # 0-299 (fixed) or None (random each episode)
    num_scan_rays=360,         # defaults from RLEnvConfig
    step_duration=0.1,
    max_episode_steps=500,
    gui=False,                 # True opens Gazebo GUI (managed mode only)
    managed=True,              # True = env manages roslaunch processes
    log_dir=None,              # None = results/rl/, "" = disabled
    reward_collision=-100.0,
    reward_goal=100.0,
    reward_step=-0.01,
    reward_shaping=True,
)
```

**managed=True** (default):
The env spawns `barn_gazebo.launch` (and for moborobo: `barn_move_base.launch`) when first reset, and restarts Gazebo when the world changes. Slow between world switches (~5 s) but fully self-contained.

**managed=False**:
Gazebo must be running externally before `reset()` is called. Faster — no Gazebo restart needed between episodes on the same world. Recommended for development.

### Observation space

Shape: `(num_scan_rays + 2,)`, dtype `float32`.

| Index | Feature | Range | Description |
|---|---|---|---|
| `0 : num_scan_rays` | scan ranges | `[0, 1]` | Each ray normalised by `max_range` |
| `num_scan_rays` | dist to goal | `[0, 1]` | Euclidean distance / max possible distance |
| `num_scan_rays + 1` | angle to goal | `[−1, 1]` | Robot-frame angle / π |

### Action space

Shape: `(2,)`, dtype `float32`.

| Index | Command | Range |
|---|---|---|
| `0` | `linear_x` | `[0, MAX_LINEAR_VEL]` m/s (forward only) |
| `1` | `angular_z` | `[−MAX_ANGULAR_VEL, MAX_ANGULAR_VEL]` rad/s |

Velocity limits come from `RobotConfig` (jackal) or `MoboroboConfig` (moborobo) in `configs/config.py`.

### Reward

Each step:
```
reward = REWARD_STEP
       + (prev_dist − curr_dist)   ← if REWARD_SHAPING=True

on collision:   reward += REWARD_COLLISION,  terminated=True
on goal:        reward += REWARD_GOAL,       terminated=True
on max steps:   truncated=True
```

### Step info dict

`step()` returns an info dict containing:

| Key | Description |
|---|---|
| `world_idx` | Current world index |
| `episode` | Episode counter |
| `step` | Step counter within episode |
| `dist_to_goal` | Current Euclidean distance to goal (metres) |
| `collided` | Whether this step ended with a collision |
| `outcome` | `"running"` / `"success"` / `"collision"` / `"timeout"` |
| `episode_reward` | Cumulative reward so far this episode |

### Basic usage

```python
from rl import BarnEnv

# Fixed world, unmanaged (Gazebo already running)
env = BarnEnv(robot="moborobo", world_idx=30, managed=False)

obs, info = env.reset()
for _ in range(500):
    action = env.action_space.sample()   # replace with your policy
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### World randomisation

```python
# Random world each episode — managed=True restarts Gazebo when world changes
env = BarnEnv(robot="moborobo", world_idx=None, managed=True)
```

---

## RLLogger — Training Logger

### Constructor

```python
from rl.logging import RLLogger

logger = RLLogger(
    run_id="ppo_moborobo_v1",   # must be unique per run
    config={                     # saved to run_config.json
        "algo": "PPO",
        "lr": 3e-4,
        "robot": "moborobo",
        "world_idx": 30,
    },
    results_dir=None,            # None = RLLogConfig.RUNS_DIR
    window=100,                  # None = RLLogConfig.ROLLING_WINDOW
)
```

### Methods

**`log_episode(episode, world_idx, outcome, steps, total_reward, final_dist)`**

Call at every episode end. Appends one row to `episodes.csv` immediately.

```python
logger.log_episode(
    episode=1,
    world_idx=info["world_idx"],
    outcome=info["outcome"],
    steps=info["step"],
    total_reward=info["episode_reward"],
    final_dist=info["dist_to_goal"],
)
```

**`log_progress(timestep)`**

Prints and writes a rolling-stats line to `training.log`. Call every `RLLogConfig.LOG_EVERY_STEPS` steps.

```python
if total_steps % RLLogConfig.LOG_EVERY_STEPS == 0:
    logger.log_progress(timestep=total_steps)
```

Sample output:
```
[0:05:23] step=10000    ep=200    | reward(last 100): 47.32 ± 28.41 | avg_steps: 312.5 | SR 34.0%  CR 52.0%  TO 14.0%
```

**`logger.close()`**

Flushes files, prints and writes the final run summary. Always call this at the end.

---

## Running Training

### Minimal training loop (any algorithm)

```python
from rl import BarnEnv
from rl.logging import RLLogger
from configs.rl_config import RLLogConfig

env = BarnEnv(robot="moborobo", world_idx=30, managed=False)
logger = RLLogger("my_run_v1", config={"algo": "random_baseline"})

obs, info = env.reset()
episode_reward = 0.0
total_steps = 0

for _ in range(100_000):
    action = env.action_space.sample()    # ← replace with policy.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward
    total_steps += 1

    if terminated or truncated:
        logger.log_episode(
            episode=info["episode"],
            world_idx=info["world_idx"],
            outcome=info["outcome"],
            steps=info["step"],
            total_reward=info["episode_reward"],
            final_dist=info["dist_to_goal"],
        )
        obs, info = env.reset()
        episode_reward = 0.0

    if total_steps % RLLogConfig.LOG_EVERY_STEPS == 0:
        logger.log_progress(timestep=total_steps)

logger.close()
env.close()
```

### With stable-baselines3

```bash
pip3 install stable-baselines3
```

```python
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from rl import BarnEnv
from rl.logging import RLLogger
from configs.rl_config import RLLogConfig

class LoggerCallback(BaseCallback):
    """Bridge between SB3 episode stats and RLLogger."""

    def __init__(self, logger: RLLogger):
        super().__init__()
        self._rl_logger = logger

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if info.get("outcome") not in (None, "running"):
                self._rl_logger.log_episode(
                    episode=info["episode"],
                    world_idx=info["world_idx"],
                    outcome=info["outcome"],
                    steps=info["step"],
                    total_reward=info["episode_reward"],
                    final_dist=info["dist_to_goal"],
                )
        if self.num_timesteps % RLLogConfig.LOG_EVERY_STEPS == 0:
            self._rl_logger.log_progress(self.num_timesteps)
        return True


env = BarnEnv(robot="moborobo", world_idx=30, managed=False)
logger = RLLogger("ppo_moborobo_world30", config={
    "algo": "PPO", "lr": 3e-4, "robot": "moborobo", "world_idx": 30,
})

model = PPO("MlpPolicy", env, learning_rate=3e-4, verbose=0)
model.learn(total_timesteps=500_000, callback=LoggerCallback(logger))
model.save("results/rl/ppo_moborobo_world30")

logger.close()
env.close()
```

---

## MPPI Controller

MPPI (Model Predictive Path Integral) is a sampling-based optimal control method. Instead of learning a policy, it samples K random perturbations around a nominal control sequence every timestep, rolls out the dynamics model to evaluate each trajectory's cost, then takes a soft-weighted average to improve the nominal sequence.

**Why use MPPI here?**
- Good baseline to compare RL against — no training required
- Works with the existing `BarnEnv` infrastructure (world loading, collision detection, logging)
- Tunable via `configs/mppi_config.py` with no code changes
- Pure numpy — no additional dependencies

### How it works in ROS / Gazebo

MPPI does **not** use move_base. It runs as a Python controller that:
1. Reads the current robot pose from Gazebo ground truth (`get_pose()`)
2. Reads the current laser scan (`get_scan_info()`)
3. Samples K velocity sequences, rolls them out using a unicycle dynamics model
4. Picks the optimal action and publishes it to `/cmd_vel` via `env.step(action)`

The dynamics model is a simple unicycle:
```
x_{t+1} = x_t + v_t · cos(θ_t) · dt
y_{t+1} = y_t + v_t · sin(θ_t) · dt
θ_{t+1} = θ_t + ω_t · dt
```

Cost terms:
| Term | Config key | Description |
|---|---|---|
| Obstacle cost | `W_OBSTACLE` | `exp(−clearance / OBS_SIGMA)` summed over horizon |
| Terminal goal | `W_GOAL` | Euclidean distance to goal at end of horizon |
| Per-step goal | `W_GOAL_STEP` | Mean distance to goal across all steps (dense) |
| Regularisation | `LAMBDA` | MPPI control cost: λ · u^T Σ⁻¹ ε |

### MPPI configuration (`configs/mppi_config.py`)

| Setting | Default | Description |
|---|---|---|
| `K` | `500` | Number of sampled trajectories per step |
| `T` | `20` | Planning horizon (steps); lookahead = T × DT seconds |
| `DT` | `0.1` s | Should match `RLEnvConfig.STEP_DURATION` |
| `LAMBDA` | `1.0` | Temperature — lower = greedier, higher = softer |
| `SIGMA_V` | `0.12` m/s | Linear velocity noise std |
| `SIGMA_W` | `0.5` rad/s | Angular velocity noise std |
| `MAX_V` | from robot config | Linear velocity ceiling |
| `MAX_W` | from robot config | Angular velocity ceiling |
| `W_OBSTACLE` | `80.0` | Obstacle cost weight |
| `W_GOAL` | `2.0` | Terminal goal cost weight |
| `W_GOAL_STEP` | `0.5` | Per-step goal cost weight |
| `OBS_SIGMA` | `0.4` m | Obstacle cost decay distance |
| `ROBOT_RADIUS` | `0.4` m | Safety clearance (conservative) |
| `SCAN_SUBSAMPLE` | `4` | Use every Nth ray (720-ray scan → 180 obstacle pts) |

### Usage with BarnEnv

```python
from rl import BarnEnv
from rl.controllers import MPPIController
from rl.logging import RLLogger
from configs.rl_config import RLLogConfig

env = BarnEnv(robot="moborobo", world_idx=30, managed=False)
ctrl = MPPIController(goal_x=env.goal_pos[0], goal_y=env.goal_pos[1])
logger = RLLogger("mppi_world30", config={"algo": "MPPI", "K": 500, "T": 20})

obs, info = env.reset()
ctrl.reset()

for step in range(10_000):
    # Get raw state for MPPI (bypasses the normalised gym observation)
    ranges, angle_min, angle_inc, max_r = env.get_scan_info()
    x, y, theta = env.get_pose()

    if ranges is not None:
        action = ctrl.compute_action(x, y, theta, ranges, angle_min, angle_inc, max_r)
    else:
        action = env.action_space.low.copy()   # wait for first scan

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        logger.log_episode(
            episode=info["episode"],
            world_idx=info["world_idx"],
            outcome=info["outcome"],
            steps=info["step"],
            total_reward=info["episode_reward"],
            final_dist=info["dist_to_goal"],
        )
        obs, info = env.reset()
        ctrl.reset()

    if step % RLLogConfig.LOG_EVERY_STEPS == 0:
        logger.log_progress(timestep=step)

logger.close()
env.close()
```

### Standalone usage (without BarnEnv)

MPPI can also run as a ROS node independently, without the gym wrapper — useful for quick navigation tests.

```python
import rospy, time, numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from gazebo.gazebo_simulation import GazeboSimulation
from rl.controllers import MPPIController
from rl.env.barn_env import _quat_to_yaw
from configs.config import WorldConfig

rospy.init_node("mppi_node", anonymous=True)
sim = GazeboSimulation(init_position=WorldConfig.STATIC_INIT_POSITION)
gx = WorldConfig.STATIC_INIT_POSITION[0] + WorldConfig.STATIC_GOAL_OFFSET[0]
gy = WorldConfig.STATIC_INIT_POSITION[1] + WorldConfig.STATIC_GOAL_OFFSET[1]
ctrl = MPPIController(goal_x=gx, goal_y=gy)
pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

rate = rospy.Rate(10)   # 10 Hz, matches DT=0.1
while not rospy.is_shutdown():
    scan = rospy.wait_for_message("/scan", LaserScan, timeout=5.0)
    state = sim.get_model_state()
    pos = state.pose.position
    ori = state.pose.orientation
    theta = _quat_to_yaw(ori.x, ori.y, ori.z, ori.w)

    action = ctrl.compute_action(
        pos.x, pos.y, theta,
        scan.ranges, scan.angle_min, scan.angle_increment, scan.range_max,
    )

    msg = Twist()
    msg.linear.x, msg.angular.z = float(action[0]), float(action[1])
    pub.publish(msg)
    rate.sleep()
```

### Tuning tips

- **Robot freezes near obstacles** → lower `OBS_SIGMA` (sharper cliff) or raise `K`
- **Robot ignores obstacles and collides** → raise `W_OBSTACLE`
- **Robot does not move toward goal** → raise `W_GOAL` or `W_GOAL_STEP`
- **Trajectories are jerky** → raise `LAMBDA` or lower `SIGMA_V`/`SIGMA_W`
- **Too slow** → lower `K` (300 is usually fine) or raise `SCAN_SUBSAMPLE`

---

## Running the Tests

Unit tests (no Gazebo required):

```bash
cd ~/Desktop/dream
python3.8 -m pytest rl/tests/ -m "not gazebo" -v
```

Gazebo integration tests (Gazebo must be running on world 30):

```bash
# Terminal 1 — start Gazebo
roslaunch moborobo barn_gazebo.launch world_name:=$(python3.8 -c \
  "from configs.config import WorldConfig; print(WorldConfig.world_file(30))")
roslaunch moborobo barn_move_base.launch

# Terminal 2 — run integration tests
cd ~/Desktop/dream
python3.8 -m pytest rl/tests/ -m gazebo -v
```

---

## Output Files

### `results/rl/episodes.csv` (from BarnEnv)

Appended every episode. One row per episode.

```
episode,world_idx,outcome,steps,total_reward,final_dist_to_goal
1,30,collision,47,-101.23,6.82
2,30,timeout,500,−0.12,3.41
3,30,success,312,87.54,0.31
```

### `results/rl/runs/{run_id}/episodes.csv` (from RLLogger)

Same columns plus `wall_time_s`. More detailed than the BarnEnv CSV.

### `results/rl/runs/{run_id}/run_config.json`

Snapshot of the config dict passed to `RLLogger(config=...)`, plus `run_id` and `started_at`.

### `results/rl/runs/{run_id}/training.log`

Human-readable log:
```
========================================================================
BARN RL Training Log
Run ID  : ppo_moborobo_v1
Started : 2026-04-26 14:30:00
========================================================================
Config:
  algo:                  PPO
  lr:                    0.0003
...

[0:05:23] step=10000    ep=200    | reward(last 100): 47.32 ± 28.41 | ...
[0:10:41] step=20000    ep=398    | reward(last 100): 61.18 ± 24.09 | ...

========================================================================
RUN SUMMARY  —  ppo_moborobo_v1
...
Success rate:             38.20%
Collision rate:           49.60%
Timeout rate:             12.20%
========================================================================
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `TimeoutError: No LaserScan on '/scan'` | Sensor converter not running | Start `barn_move_base.launch`; or use `managed=True` |
| `ModuleNotFoundError: No module named 'gymnasium'` | Not installed | `pip3 install gymnasium` |
| `ModuleNotFoundError: No module named 'rospy'` | ROS not sourced | `source /opt/ros/noetic/setup.bash` |
| `FileNotFoundError: World file not found` | Wrong dataset path | Check `Paths.MOBOROBO_WORLD_FILES` in `configs/config.py` |
| Observation scan is all-ones | No `/scan` message yet | Call `reset()` and wait; scan arrives within 1–2 seconds |
| Robot doesn't move after `step()` | Physics paused | `gazebo_sim.unpause()` is called in `step()` — check Gazebo is up |
| Gazebo restart takes too long | `GazeboConfig.STARTUP_WAIT` too low | Increase it in `configs/config.py` |
| move_base competing with RL policy | move_base has an active goal | Don't send move_base goals; use `managed=True` which never sends them |
