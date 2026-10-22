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
8. [RL Training (SAC)](#rl-training-sac)
9. [Benchmarking Planners](#benchmarking-planners)
10. [Reading Results](#reading-results)
11. [Troubleshooting](#troubleshooting)

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

source "$HOME/barn_lab/nav_challenge/bin/activate"
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

## RL Training (SAC)

Trains a Soft Actor-Critic policy on the BARN worlds via `train_sac.py`.
See [`docs/rl_structure.md`](rl_structure.md) for the full observation / reward / architecture description.

### Before you start

1. Sanity-check the obs and reward scaling (no training, ~30 s):
   ```bash
   python3 train_sac.py --robot moborobo --debug-obs
   ```
2. Verify the Gazebo/ROS pipeline (resets + a few random steps):
   ```bash
   python3 train_sac.py --robot moborobo --smoke-test
   ```

### Start a fresh run

```bash
# Moborobo, standard 50-world eval set, 200k steps
python3 train_sac.py --robot moborobo --total-steps 200000

# Smaller/faster: 10 worlds, 100k steps
python3 train_sac.py --robot moborobo \
    --worlds 0 30 60 90 120 150 180 210 240 270 \
    --total-steps 100000
```

Each run creates `results/rl/runs/{timestamp}/` containing:

| File | Contents |
|---|---|
| `episodes.csv` | one row per episode (outcome, steps, reward, final dist) |
| `model_<N>_steps.zip` | policy checkpoint every `--save-every` steps (default 20k) |
| `model_<N>_steps_replay_buffer*.pkl` | replay buffer saved alongside each checkpoint |
| `model_final.zip` | policy saved when the run exits (**no** replay buffer) |
| `tb/` | TensorBoard logs |

### Curriculum learning (easy → hard)

`--curriculum` trains on progressively harder worlds. Worlds are ranked by a
combined difficulty score from the 5 BARN metrics (clearance, visibility,
dispersion, characteristic dimension, tortuosity — see `rl/curriculum.py`) and
split into even-thirds tiers. `--total-steps` is divided into three equal phases:

| Phase | Steps | World pool |
|---|---|---|
| 1 | 0 – ⅓ | easy (51 worlds) |
| 2 | ⅓ – ⅔ | easy + medium (102) |
| 3 | ⅔ – end | all (153) |

The pool stays *cumulative* so the policy keeps revisiting mastered worlds and
doesn't forget them. Tiers are derived from the dataset's `metrics_files/`.

```bash
python3 train_sac.py --robot moborobo --curriculum --total-steps 200000
```

Inspect the tier assignment without training:
```bash
python3 -c "from rl.curriculum import difficulty_tiers; t=difficulty_tiers('moborobo'); \
print({k: v for k,v in t.items()})"
```

`--curriculum` overrides `--worlds`. The phase boundaries print at startup and
each advance is logged (`[curriculum] step N: advancing to M worlds`).

### Several robots in one world (faster data collection)

`--n-robots N` puts N robots into the same Gazebo world and steps them
together (`rl/env/multi_barn_env.py`). They are ghosts to each other: each has
its own collision bit, so they drive through one another but still hit the
obstacles, and none has rendered geometry, so no lidar sees another robot.
Every robot therefore faces the ordinary single-robot task, and one physics
step yields N transitions.

```bash
python3 train_sac.py --robot moborobo --curriculum --total-steps 200000 --n-robots 4
```

What differs from the single-robot path (Moborobo only):

- **Lidar**: a lighter vertical profile (`RLEnvConfig.MULTI_LIDAR_PROFILE`,
  64 samples over [−45°, 0°] instead of 900 over the full sphere). At the
  `/scan` level it matches the stock profile to a median of 8 mm per sector.
- **Collisions** are tested geometrically (chassis box vs. cylinder centres,
  `rl/env/geometry.py`), not through the `/collision` topic.
- **No `move_base`**, no camera, no collision publisher.
- `--episodes-per-world` is per robot: the world changes after
  `N × episodes-per-world` episodes; episodes still running are cut as
  truncations (`outcome=world_change`, not written to `episodes.csv`).
- `--utd` sets the gradient steps per collected transition (default 1.0, i.e.
  N updates per vectorised step). Lower it (e.g. `--utd 0.5`) to trade sample
  efficiency for wall-clock speed.
- `episodes.csv` gets an extra `robot` column.
- It refuses to start while another Gazebo server is running, instead of
  killing it.

Pick N with the capacity probe: each robot must still get its 10 Hz scan and
the simulation must keep up with real time. On the development machine
(RTX 2060) N=4 holds both; N=6 does not.

```bash
python3 tools/multi_probe.py capacity --n-robots 4   # scan rate, step cost, stale scans
python3 tools/multi_probe.py ghosts                  # robots ignore each other, obstacles do not
python3 tools/multi_probe.py compare --worlds 0 100  # light vs stock lidar, with a stock-vs-stock floor
python3 tools/sim_probe.py                           # single-robot timing breakdown
```

### Resume a previous run

`--resume` continues training from a checkpoint. `--total-steps` is then **additional** steps (e.g. `--total-steps 100000` after a 40k checkpoint trains up to 160k total).

```bash
# Resume from the 40k checkpoint (keeps the warm replay buffer — preferred)
python3 train_sac.py --robot moborobo \
    --resume results/rl/runs/20260519_171857/model_40000_steps \
    --total-steps 160000

# Resume from model_final (no replay buffer — SAC refills it over ~5k noisy steps)
python3 train_sac.py --robot moborobo \
    --resume results/rl/runs/20260519_171857/model_final \
    --total-steps 160000
```

Notes:
- Pass the path **without** `.zip` — the script appends it.
- **Prefer a `model_<N>_steps` checkpoint over `model_final`**: only the numbered
  checkpoints have a saved replay buffer next to them. The script auto-loads the
  buffer if it finds `model_replay_buffer_<N>_steps.pkl` in the same directory;
  otherwise it warns and starts with an empty buffer.
- The resumed run writes to a **new** `results/rl/runs/{timestamp}/` directory.
  TensorBoard still shows continuous curves if you point `--logdir` at the
  parent `results/rl/runs`.

### Monitor training

```bash
# Live episode log
tail -f results/rl/runs/{RUN_ID}/episodes.csv

# TensorBoard (separate terminal)
tensorboard --logdir results/rl/runs
# → http://localhost:6006   (watch rollout/ep_rew_mean, train/ent_coef, train/critic_loss, train/q_mean)
```

Every progress line in the log ends with `q_mean`, the critic's value of the
current policy on buffer states. It should stay within roughly
[−50, +170]; past `2 × REWARD_GOAL` the line adds
`[WARN] ... the critic is diverging` — stop the run and investigate.

### Key hyperparameters

| Flag | Default | Meaning |
|---|---|---|
| `--total-steps` | 200000 | Env steps (additional steps when `--resume`) |
| `--episodes-per-world` | 5 | Episodes before moving to the next world (worlds are visited in a shuffled order, reshuffled every pass) |
| `--curriculum-steps` | whole run | Span the three curriculum phases are laid over; keep it fixed (e.g. 300000) when extending a run with `--resume` so the phase changes stay at 100k/200k |
| `--seed` | random | Seeds SAC and the world order; printed at startup (`Seed    : N`) so a run can be repeated |
| `--save-every` | 20000 | Checkpoint interval (steps) |
| `--lr` | 3e-4 | Learning rate |
| `--batch-size` | 256 | SAC batch size |
| `--buffer-size` | 300000 | Replay buffer capacity |
| `--ent-coef` | auto | SAC entropy coefficient |

### Deploy a trained policy in the benchmark

The trained SAC policy is one of the planners in `benchmark/run_benchmark.py`
(see [Benchmarking Planners](#benchmarking-planners)):

```bash
python3 -m benchmark.run_benchmark --planners rl_sac \
    --rl-model results/rl/runs/{RUN_ID}/model_final
```

### If a run hangs or crashes

A crashed run can leave orphan ROS/Gazebo processes that block the next run.
Clean them up before retrying:

```bash
pkill -SIGKILL -f 'gzserver|gzclient'
pkill -SIGINT  -f 'roslaunch|move_base|pointcloud_to_laserscan'
sleep 2
pkill -SIGKILL -f 'roslaunch|move_base|pointcloud_to_laserscan|rosmaster|roscore'
```

The env now also reaps these automatically on the next launch, and bails out
with a clear `RuntimeError` if `gzserver` segfaults during init (instead of
hanging forever in a Gazebo service call).

---

## Benchmarking Planners

`benchmark/run_benchmark.py` compares one or more planners on a shared set of
worlds (Gazebo is launched once per world and reused across planners, so all
planners see identical conditions). Available `--planners`:

| Name | Description |
|---|---|
| `mppi` | Model-Predictive Path Integral (+ online A* global planner) |
| `mppi_jax` | MPPI, JAX backend (needs `pip install 'jax[cpu]'`) |
| `dwa` | Dynamic Window Approach (+ online A* global planner) |
| `astar_local` | A* with a local follower |
| `elastic_band` | **Elastic Band** — an A*-seeded chain of free-space "bubbles" deformed each step by internal contraction + external obstacle-repulsion forces, then tracked with pure pursuit |
| `rl_sac` | Trained SAC policy (requires `--rl-model`) |

```bash
# Default: mppi + dwa + astar_local on the 50-world eval set (jackal)
python3 -m benchmark.run_benchmark

# Elastic Band on its own, moborobo, standard eval worlds
python3 -m benchmark.run_benchmark --robot moborobo --planners elastic_band

# Elastic Band on specific worlds (e.g. an easy, a medium, and a hard one)
python3 -m benchmark.run_benchmark --robot moborobo --planners elastic_band \
    --world-list 0 24 42

# Head-to-head: classical planners + Elastic Band + the trained RL policy
python3 -m benchmark.run_benchmark --robot moborobo \
    --planners dwa mppi elastic_band rl_sac \
    --rl-model results/rl/runs/{RUN_ID}/model_final
```

Each planner's per-world results are written under `results/benchmark/`. See
[Reading Results](#reading-results) for the score columns.

### Elastic Band tuning

All knobs live in `configs/elastic_band_config.py`:

| Parameter | Meaning |
|---|---|
| `K_INTERNAL` | Contraction-force gain (higher → straighter / tauter band) |
| `K_EXTERNAL` | Repulsion-force gain (higher → pushes harder off obstacles) |
| `RHO0` | Clearance (m) under which repulsion activates |
| `DEFORM_ITERS` | Force-update iterations per control step |
| `LOOKAHEAD` | Pure-pursuit carrot distance along the band |
| `REPLAN_STEPS` | Rebuild the band from a fresh A* path every N steps |

If the band cuts corners into walls, raise `K_EXTERNAL` / `RHO0`. If it's
jittery, lower `K_EXTERNAL` or `STEP_ALPHA`.

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
