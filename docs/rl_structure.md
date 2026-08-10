# RL structure — BARN SAC

How the RL stack fits together end-to-end: env, observation, action, reward, training, and deployment.

## Layout

```
rl/
├── env/
│   └── barn_env.py              # BarnEnv — Gymnasium wrapper around Gazebo
├── models/
│   └── feature_extractor.py     # ScanStateExtractor — two-branch CNN+FC encoder
├── controllers/
│   └── rl_sac_controller.py     # SACController — loads a trained policy for the benchmark
├── logging/
│   └── rl_logger.py             # per-run logger (run_config.json, episodes.csv, training.log)
└── tests/test_barn_env.py       # unit + (optional) Gazebo integration tests

train_sac.py                     # entry point: SB3 SAC + world-cycling wrapper
configs/rl_config.py             # single source of truth for env / model / log / eval defaults
```

## Episode flow

1. `train_sac.py` builds `WorldCyclingEnv` (extends `BarnEnv`) over a list of worlds.
2. On the first `reset()` the env spawns `roscore` (persistent), `gzserver`, and
   for moborobo also `barn_move_base.launch` (provides `pointcloud_to_laserscan → /scan`).
3. Subsequent resets teleport the robot back to the static start pose and drain
   the collision flag. Gazebo is only relaunched when the world index changes.
4. Each `step()` unpauses Gazebo, publishes `/cmd_vel`, sleeps `STEP_DURATION`
   (default 0.1 s → 10 Hz policy), pauses Gazebo, then reads scan + ground-truth pose.
5. Episode ends on `get_hard_collision()`, `dist_to_goal ≤ GOAL_TOLERANCE` (1.0 m),
   or `MAX_EPISODE_STEPS` (500). Outcome (`success` / `collision` / `timeout`)
   is logged to `results/rl/runs/{run_id}/episodes.csv`.

## Observation (two-part `Dict`)

Single source of truth: `BarnEnv._build_obs()` in `rl/env/barn_env.py`.
The SAC deployment controller in `rl/controllers/rl_sac_controller.py`
reconstructs the exact same dict at inference time (and still supports
legacy flat-obs checkpoints, auto-detected from the model file).

The two-part split follows the LiDAR DRL-navigation literature (Fan & Long
et al., ICRA'18 / IJRR'20 "Distributed non-communicating multi-robot
collision avoidance"; Tai et al., IROS'17 mapless navigation; DRL-VO,
T-RO'23): a *perception* part processed by a 1D-CNN and a low-dimensional
*state* part processed by a small FC layer, fused before the policy heads.

### `obs["scan"]` — shape `(SCAN_STACK, NUM_SCAN_RAYS)` = `(3, 90)`

- Each frame: the raw scan (~722/1080 rays) **min-pooled** into 90 sectors.
  Min — not linear interpolation — so a thin obstacle hit by a single raw
  ray can never disappear from the observation.
- Normalised by `SCAN_CLIP_RANGE` (5 m), not the 10 m sensor max: BARN gaps
  are < 1 m wide, so a 10 m scale squashes all decision-relevant geometry
  into `[0, 0.1]`. Clipping keeps resolution where obstacles matter.
- 3 consecutive frames stacked (row 0 = oldest). At 10 Hz this gives the
  CNN 0.3 s of history — enough to infer approach speed toward obstacles.
  On reset the stack is seeded with 3 copies of the first frame.

### `obs["state"]` — shape `(5,)`

| Index | Feature              | Range     | Notes                                       |
| ----- | -------------------- | --------- | ------------------------------------------- |
| `[0]` | `dist / max_dist`    | `[0, 1]`  | `max_dist` = start→goal straight line       |
| `[1]` | `sin(angle_to_goal)` | `[-1, 1]` | In robot frame                              |
| `[2]` | `cos(angle_to_goal)` | `[-1, 1]` | sin/cos avoids the ±π discontinuity         |
| `[3]` | `last_lin / max_lin` | `[0, 1]`  | Previous step's commanded linear velocity   |
| `[4]` | `last_ang / max_ang` | `[-1, 1]` | Previous step's commanded angular velocity  |

Why `last_action` in obs: the policy runs at 10 Hz on a robot with inertia.
Without it the network has no idea whether it just commanded full throttle
or a hard turn, and tends to produce chattery, collision-prone control.

## Policy network (`rl/models/feature_extractor.py`)

```
obs["scan"] (3, 90) ─► Conv1d(3→32, k5, s2) ─► Conv1d(32→32, k3, s2) ─► FC 128 ─►┐
                                                                                 ├─► concat (160) ─► [256, 256] actor / twin-critic heads
obs["state"] (5,)   ─► FC 32 ────────────────────────────────────────────────────┘
```

- The conv treats the 3 stacked frames as input *channels*, so each kernel
  sees how a sector evolves over time (like frame-stacked Atari CNNs).
- Receptive field after two convs ≈ 9 sectors ≈ 36° — one BARN gap fits in
  a single feature unit.
- A flat-MLP baseline must *learn* that ray *i* and *i+1* are neighbours;
  the conv bakes this locality in, which is the main sample-efficiency win.
- All sizes configurable in `RLModelConfig` (`configs/rl_config.py`).

## Action (dim 2, `Box`)

| Index | Feature       | Range                                   |
| ----- | ------------- | --------------------------------------- |
| `[0]` | `linear_x`    | `[0, MAX_LINEAR]` — forward-only        |
| `[1]` | `angular_z`   | `[-MAX_ANGULAR, +MAX_ANGULAR]`          |

Per-robot limits (`configs/config.py`):

| Robot     | `MAX_LINEAR` | `MAX_ANGULAR` |
| --------- | ------------ | ------------- |
| moborobo  | 0.2 m/s      | 1.0 rad/s     |
| jackal    | 0.5 m/s      | 1.57 rad/s    |

`linear_x` is clamped at 0 (no reverse) to match the deployed planners and
avoid pathological "back out of every corner" policies.

## Reward

Per-step formula (from `BarnEnv.step()`):

```
r  = REWARD_STEP                              # -0.01     time penalty
   + REWARD_PROGRESS_SCALE * (prev_d - d)     # × 5.0     dense shaping toward goal
   - REWARD_PROX_SCALE * prox_term            # × 0.05    near-obstacle penalty
   - REWARD_HEADING_SCALE * |angle|/π         # × 0.02    facing-the-goal penalty
   - REWARD_PATH_SCALE  * path_term           # × 0.10    A* path-deviation penalty

prox_term = max(0, PROX_THRESHOLD - min_scan_norm) / PROX_THRESHOLD
          # min_scan_norm in [0, 1]; PROX_THRESHOLD = 0.20 (~2 m on a 10-m LiDAR)

path_term = min(d_perp, PATH_MAX_DEV) / PATH_MAX_DEV
          # d_perp = perpendicular distance from (x, y) to the A* polyline (m)
          # PATH_MAX_DEV = 1.5 m; deviation past this is clipped
```

Each component is also exposed in `info["reward_components"]` for debugging.

Terminal additions:

| Outcome   | Bonus    | Episode flag |
| --------- | -------- | ------------ |
| Goal      | `+100`   | `terminated` |
| Collision | `-50`    | `terminated` |
| Timeout   | none     | `truncated`  |

### Why each term

- **Progress (`× 5.0`)**: raw `prev_d - d` per step is ≤ 0.05 m (jackal) or
  ≤ 0.02 m (moborobo). Without scaling the signal is below the step penalty
  and the agent learns to sit still. ×5 lifts it above the noise floor.
- **Proximity**: a linear penalty that activates inside 2 m gives a smooth
  gradient toward "stay clear of walls", instead of relying solely on the
  terminal -50.
- **Heading**: small penalty for not facing the goal makes the agent prefer
  forward motion along the goal vector rather than circling.
- **Collision reduced to -50**: balanced against the progress + proximity
  shaping so the agent doesn't learn "stop moving = safest".
- **Path-deviation (`× 0.10`)**: the A* reference path from `path_N.npy` is
  used as a *reward-only* supervisor. Solves BARN's "which side of this
  obstacle?" ambiguity, which is invisible from a single-frame LiDAR. The
  path is never added to the observation — the policy still has to navigate
  locally using LiDAR; the shaping just pulls it toward the human-curated
  corridor on average. Disable via `RLEnvConfig.PATH_FOLLOW_SHAPING=False`.

### Reward magnitude budget (per episode, rough caps)

- Progress total: bounded by `REWARD_PROGRESS_SCALE * max_dist` ≈ +50.
- Step penalty: `-0.01 × 500` = -5.
- Proximity: ≤ `0.05 × 500` = -25 if always near walls.
- Heading: ≤ `0.02 × 500` = -10.
- Goal bonus: +100.  Collision: -50.

So a successful episode earns roughly +100 to +140, a fast collision earns
~-50, and a timeout that didn't move earns ~-15. The signs are clean.

## Training (`train_sac.py`)

- **Episode budget** (`MAX_EPISODE_STEPS`, resolved per-robot in `RLEnvConfig.max_episode_steps`):
  - jackal: **500 steps = 50 s** (max ~25 m at 0.5 m/s — plenty)
  - moborobo: **1000 steps = 100 s** (matches the standard BARN time budget;
    a realistic 0.13 m/s average needs ~75 s to clear the 10 m course)
- **Algorithm**: SB3 SAC, `MultiInputPolicy` with the two-branch
  `ScanStateExtractor` (CNN scan + FC state) and `[256, 256]` heads.
- **Optimizer**: Adam, `lr=3e-4`, `batch_size=256`, `buffer_size=100_000`,
  `learning_starts=1_000`, `gamma=0.99`, `tau=0.005`, entropy coef auto.
- **World cycling**: `WorldCyclingEnv` (subclass of `BarnEnv`) fixes a world
  for N episodes (default 5) then advances. Restarting Gazebo costs ~10 s, so
  this keeps wall-clock dominated by env stepping, not setup.
- **Checkpointing**: SB3 `CheckpointCallback` every `--save-every` steps
  (default 20k), plus `model_final.zip` on exit.
- **Logging**: `TrainProgressCallback` prints a one-line rolling summary
  every `--log-every` steps. `BarnEnv` itself appends one row per episode to
  `episodes.csv` (outcome, steps, total_reward, final_dist_to_goal).

### Wall-clock budget

Stepping is gated by `time.sleep(STEP_DURATION)` to let Gazebo advance physics
in real time. At 10 Hz, 200 000 steps ≈ 5.5 h floor + reset / Gazebo restart
overhead. Plan training runs accordingly.

### Smoke test

```bash
python3 train_sac.py --robot moborobo --smoke-test
```

Launches Gazebo, resets, runs five random steps, exits — use it to verify
the ROS/Gazebo pipeline before committing to a long run.

### Obs / reward debug dump

```bash
python3 train_sac.py --robot moborobo --debug-obs
```

Drives a deterministic action script (forward, left, right, stop) and prints,
each step:

- scan slice `min / mean / max`
- `dist_norm`, `sin(angle)`, `cos(angle)`, `last_lin`, `last_ang`
- per-component reward breakdown from `info["reward_components"]`:
  `step`, `progress`, `prox`, `heading`, `path`, plus the raw `path_dev_m`

Use this to verify obs scaling and reward magnitudes look correct before
committing to a multi-hour training run.

### TensorBoard

`train_sac.py` writes SB3's standard scalars to `results/rl/runs/{run_id}/tb/`.
View them with:

```bash
tensorboard --logdir results/rl/runs
```

Useful curves: `rollout/ep_rew_mean`, `train/entropy_coef`,
`train/critic_loss`, `train/actor_loss`. Watch for entropy collapsing to 0
(policy stopped exploring) or critic loss exploding (reward scale mismatch).

### Typical commands

```bash
# Full moborobo training over the 50-world eval set
python3 train_sac.py --robot moborobo --total-steps 200000

# Quick 10-world subset
python3 train_sac.py --robot moborobo \
    --worlds 0 30 60 90 120 150 180 210 240 270 \
    --total-steps 100000

# Resume from a checkpoint
python3 train_sac.py --robot moborobo \
    --resume results/rl/runs/RUN_ID/model_100000
```

## Deployment (`SACController`)

Loaded by the benchmark with `--planners rl_sac --rl-model PATH`. Mirrors the
training observation exactly:

- Auto-detects the checkpoint's obs format: Dict (current two-part) vs Box
  (legacy flat) — old checkpoints keep working.
- Min-pools the live scan into the training-time sector count and keeps a
  rolling `SCAN_STACK`-deep frame history across `compute_action()` calls
  (cleared by `reset()`).
- Normalises by the training-time clip range (not the ROS `range_max`,
  which may differ between scan sources).
- Tracks `last_action` between calls so the policy sees the same obs it was
  trained on.
- Predicts deterministically (no exploration noise) by default.

## Tunable knobs (in `configs/rl_config.py`)

| Parameter                     | Default | What it does                                  |
| ----------------------------- | ------- | --------------------------------------------- |
| `NUM_SCAN_RAYS`               | 90      | Scan sectors per frame (min-pooled from raw). |
| `SCAN_STACK`                  | 3       | Consecutive frames stacked in `obs["scan"]`.  |
| `SCAN_CLIP_RANGE`             | 5.0     | Metres — scan clip + normalisation range.     |
| `STEP_DURATION`               | 0.1 s   | Sim-time per env step (10 Hz policy).         |
| `MAX_EPISODE_STEPS`           | 500     | Truncation limit.                             |
| `REWARD_GOAL`                 | +100    | Terminal bonus on reaching goal.              |
| `REWARD_COLLISION`            | -50     | Terminal penalty on collision.                |
| `REWARD_STEP`                 | -0.01   | Per-step time penalty.                        |
| `REWARD_PROGRESS_SCALE`       | 5.0     | Multiplier on `prev_d - curr_d`.              |
| `REWARD_PROX_SCALE`           | 0.05    | Near-obstacle penalty scale.                  |
| `PROX_THRESHOLD`              | 0.20    | Normalised scan distance under which it kicks in. |
| `REWARD_HEADING_SCALE`        | 0.02    | Heading-misalignment penalty scale.           |
| `PATH_FOLLOW_SHAPING`         | `True`  | Enable A* path-deviation penalty.             |
| `REWARD_PATH_SCALE`           | 0.10    | Scale of the path-deviation penalty.          |
| `PATH_MAX_DEV`                | 1.5     | Metres past which the deviation is clipped.   |
| `REWARD_SHAPING`              | `True`  | Master kill-switch for all shaping terms.     |

Set `REWARD_SHAPING=False` for a sparse-reward ablation (goal/collision/step only).

## How the robot knows its own pose

`BarnEnv` reads `pose.position` and `pose.orientation` from
`GazeboSimulation.get_model_state()` — Gazebo's `/gazebo/get_model_state`
service, i.e. **ground-truth simulator pose**. Not odometry, not AMCL.

- ✅ Standard BARN convention; matches the classical baselines (`mppi`,
  `dwa`), so train ≡ test.
- ⚠️ Not realistic for a physical robot. `SACController.compute_action()`
  takes pose as a *function argument*, so the deployment caller chooses
  the pose source — swap in `/odom` or AMCL when porting to hardware.
- To harden against real-world noise during training, add Gaussian noise
  to `pos` in `_build_obs` (a form of domain randomisation). Not enabled
  by default.
