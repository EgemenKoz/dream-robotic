# RL structure — BARN SAC

How the RL stack fits together end-to-end: env, observation, action, reward, training, and deployment.

## Layout

```
rl/
├── env/
│   ├── barn_env.py              # BarnEnv — Gymnasium wrapper around Gazebo
│   ├── multi_barn_env.py        # MultiRobotBarnVecEnv — N ghost robots in one world (--n-robots)
│   └── geometry.py              # box-vs-cylinder collision test used by the multi-robot env
├── curriculum.py                # difficulty tiers, curriculum phases, world cycling
├── models/
│   ├── feature_extractor.py     # ScanStateExtractor — two-branch CNN+FC encoder
│   └── critic.py                # LayerNormSACPolicy — twin critics with LayerNorm
├── controllers/
│   └── rl_sac_controller.py     # SACController — loads a trained policy for the benchmark
├── logging/
│   └── rl_logger.py             # per-run logger (run_config.json, episodes.csv, training.log)
└── tests/test_barn_env.py       # unit + (optional) Gazebo integration tests

train_sac.py                     # entry point: SB3 SAC + world-cycling wrapper
tools/sim_probe.py               # single-robot timing probe (RTF, scan rate, step cost)
tools/multi_probe.py             # multi-robot checks: ghosts, lidar equivalence, capacity
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
5. Episode ends on `get_hard_collision()`,
   `dist_to_goal ≤ GOAL_TOLERANCE` (1.0 m), or `MAX_EPISODE_STEPS`. Outcome
   (`success` / `collision` / `timeout`) is logged to
   `results/rl/runs/{run_id}/episodes.csv`.

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

### `obs["scan"]` — shape `(SCAN_STACK + 1, NUM_SCAN_RAYS)` = `(4, 90)`

- Each frame: the raw scan (~722/1080 rays) **min-pooled** into 90 sectors.
  Min — not linear interpolation — so a thin obstacle hit by a single raw
  ray can never disappear from the observation.
- Normalised by `SCAN_CLIP_RANGE` (5 m), not the 10 m sensor max: BARN gaps
  are < 1 m wide, so a 10 m scale squashes all decision-relevant geometry
  into `[0, 0.1]`. Clipping keeps resolution where obstacles matter.
- 3 consecutive frames stacked (row 0 = oldest). At 10 Hz this gives the
  CNN 0.3 s of history — enough to infer approach speed toward obstacles.
  On reset the stack is seeded with 3 copies of the first frame.
- **Row 3 is the "directly measured this step" mask** added by the near-field
  memory (`RLEnvConfig.SCAN_MEMORY`, see below). `Conv1d` treats rows as
  channels, so it drops into the existing CNN unchanged.

#### Near-field memory (`rl/env/scan_memory.py`)

The RS-16 reports nothing closer than 0.40 m and `pointcloud_to_laserscan`
clips to 0.45 m; with `use_inf: true` those rays arrive as `inf`. Sanitising
them to `range_max` — which is what the code used to do — turned an obstacle
6 cm from the bumper into a reading of *10 m, maximally free*: the observation
did not merely miss the danger, it reported the opposite. The geometry makes
that fatal:

| quantity                    | value      |
| --------------------------- | ---------- |
| front bumper from base_link | 0.387 m    |
| circumscribed (turn) radius | 0.4528 m   |
| sensor blind radius         | 0.45 m     |

The blind ring is the same size as the footprint's own rotational sweep, and
there are ~6 cm — three control steps at 0.2 m/s — between last sighting and
contact.

The zone is only *instantaneously* unobservable, though: an obstacle now at
0.40 m was measured at 0.50 m a moment ago. `ScanMemory` keeps returns in a
world-frame point store (voxel-deduplicated, pruned by radius and age) and
re-projects them into the current robot frame each step, so a blind sector
reports its remembered range. Points are dropped when a live ray sees straight
*through* them — except inside the blind ring, where absence of a return is
not evidence of absence. That single exception is what makes the memory
survive where it is needed. Cost is ~0.3 ms/step against a 100 ms budget.

Set `RLEnvConfig.SCAN_MEMORY = False` for the old behaviour; the extra channel
disappears and the shape returns to `(3, 90)`.

### `obs["state"]` — shape `(5,)`

| Index | Feature              | Range     | Notes                                       |
| ----- | -------------------- | --------- | ------------------------------------------- |
| `[0]` | `dist / max_dist`    | `[0, 1]`  | `max_dist` = start→goal straight line       |
| `[1]` | `sin(angle_to_goal)` | `[-1, 1]` | In robot frame                              |
| `[2]` | `cos(angle_to_goal)` | `[-1, 1]` | sin/cos avoids the ±π discontinuity         |
| `[3]` | `last_lin / max_lin` | `[-1, 1]` | Previous step's commanded linear velocity (<0 = reverse) |
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
- With `RLModelConfig.CRITIC_LAYER_NORM` (default on) the twin critics put a
  LayerNorm after each hidden layer (`rl/models/critic.py`). Without it
  `multi_v4` diverged to Q ≈ +7000 on rewards that never exceeded +0.35.

## Action (dim 2, `Box`)

| Index | Feature       | Range                                   |
| ----- | ------------- | --------------------------------------- |
| `[0]` | `linear_x`    | `[-REVERSE_SPEED_RATIO·MAX_LINEAR, MAX_LINEAR]` — reverse allowed (0.3× speed) |
| `[1]` | `angular_z`   | `[-MAX_ANGULAR, +MAX_ANGULAR]`          |

Per-robot limits (`configs/config.py`):

| Robot     | `MAX_LINEAR` | `MAX_ANGULAR` |
| --------- | ------------ | ------------- |
| moborobo  | 0.2 m/s      | 1.0 rad/s     |
| jackal    | 0.5 m/s      | 1.57 rad/s    |

`linear_x` may go negative so the policy can back out of a dead end. Reverse
is capped at half speed and carries a small per-step cost (`REWARD_REVERSE_SCALE`)
so it is used to escape traps rather than to drive the course backwards. Set
`reverse_speed_ratio=0` on `BarnEnv` for the old forward-only behaviour.
Checkpoints trained forward-only are **not** resumable on the new action space.

## Reward

Per-step formula (from `BarnEnv.step()`):

```
r  = REWARD_STEP                              # -0.01     time penalty
   + REWARD_PROGRESS_SCALE * (prev_d - d)     # × 12.5    dense shaping toward goal
   - REWARD_PROX_SCALE * prox_term            # × 0.15    near-obstacle penalty
   - REWARD_HEADING_SCALE * |angle|/π         # × 0.02    facing-the-goal penalty
   - REWARD_PATH_SCALE  * path_term           # × 0.15    A* path-deviation penalty
   - REWARD_REVERSE_SCALE * |lin|/max_linear  # × 0.05    only while lin < 0

prox_term = t² × (0.5 + 0.5 × |lin|/max_linear)
          # t = clip((PROX_RAMP_START_M - d_m) / (PROX_RAMP_START_M - PROX_FLOOR_M), 0, 1)
          # d_m = clearance in METRES over the travel-direction arc (rear arc
          #       when reversing), fused with the scan memory
          # ramp 0.90 m → floor 0.50 m; quadratic and speed-weighted

path_term = min(d_perp, PATH_MAX_DEV) / PATH_MAX_DEV
          # d_perp = perpendicular distance from (x, y) to the A* polyline (m)
          # PATH_MAX_DEV = 0.5 m; deviation past this is clipped
```

Each component is also exposed in `info["reward_components"]` for debugging,
alongside `info["min_range_m"]` and `info["fresh_frac"]`.

Terminal additions:

| Outcome     | Bonus  | Episode flag | Fires when                          |
| ----------- | ------ | ------------ | ----------------------------------- |
| Goal        | `+100` | `terminated` | `dist ≤ GOAL_TOLERANCE`             |
| Collision   | `-50`  | `terminated` | contact reported on `/collision`    |
| Timeout     | none   | `truncated`  | `MAX_EPISODE_STEPS`                 |

| Stuck       | `-50`  | `terminated` | within `STUCK_MIN_DISPLACEMENT_M` (0.3 m) of the position `STUCK_WINDOW_STEPS` (200) ago |

Progress is a real reward: an episode keeps whatever it banked before it ended.
`TERMINAL_POTENTIAL_ZERO` (paying `+REWARD_PROGRESS_SCALE * d` on `terminated`
so the shaping sums to a constant) still exists for ablation but is **off** —
see the note under *Design notes* below. The paragraph that follows describes
that ablation mode.

> **Sign discipline.** The correction *adds* the leftover potential. Subtracting
> it is off by `2 × scale × d` — 150 points at 6 m — and makes failing early far
> worse than failing late. That variant shipped once and taught the policy to
> stop dead mid-course rather than risk a terminal: timeouts went 2.5% → 17%,
> success fell to 29%, and the robot would drive a few metres and freeze for the
> rest of the episode. `test_total_shaping_is_independent_of_where_the_episode_ends`
> guards the invariant.

### Why each term

- **Progress (`× 12.5` moborobo)**: raw `prev_d - d` per step is ≤ 0.05 m
  (jackal) or ≤ 0.02 m (moborobo). Without scaling the signal is below the
  step penalty and the agent learns to sit still.
- **Proximity, in metres and quadratic**: it used to be a flat linear ramp on
  the *normalised* scan, peaking at -0.0275/step against +0.25/step of
  progress. The policy correctly ignored a 9:1 ratio and drove until min
  clearance sat on the 0.45 m sensor floor in 92-96% of episodes. Now the
  penalty is expressed against the blind ring it is supposed to defend,
  sharpens as that ring approaches, and scales with speed — because what
  kills is crossing the last 6 cm quickly, not standing near a wall.
- **Heading**: small penalty for not facing the goal makes the agent prefer
  forward motion along the goal vector rather than circling.
- **Reverse (`× 0.05`)**: the linear action goes down to
  `-REVERSE_SPEED_RATIO × max_linear` (0.3× speed) so the policy can back out
  of a dead end instead of spinning inside the blind ring. The range is kept
  forward-biased on purpose: tanh output 0 maps to its middle, and a symmetric
  range makes an untrained policy dither in place. A small
  `-0.05 × |lin| / max_linear` per-step cost keeps it from driving the course
  backwards. While reversing, the proximity penalty looks at the **rear** arc
  (`PROX_ARC` centred on π) instead of the front. Jackal's UST-10 only covers
  270°, so its rear is seen through the scan memory alone.
- **Terminal potential is off; `--gamma 0.99`** (changed 2026-10-13). The
  potential-based variant (`TERMINAL_POTENTIAL_ZERO=True`) made the shaping
  contribute the same `+125` to every episode, which removes progress as an
  incentive: a collision transition is paid `+12.5 × d − 50` (+40 at 7 m) and
  the only reason left to move is a goal 500+ steps away. That forced
  `γ=0.999`, where the target `r + 0.999·Q` barely contracts. In the
  `reverse_v1` run (240k steps) the critic drifted to **Q = +716** against a
  physical maximum of 225, went flat across actions (full-forward 713.2, stop
  712.8, full-reverse 712.4), and the policy decayed into zero-mean jitter —
  89% of the steps were spent in timeouts that never left the start area. The
  Jul/Aug runs with real progress reward and `γ=0.99` reached 47–63% success.
- **Stuck termination**: standing still or dithering ends the episode as a
  real terminal (`-50`, as costly as a collision), so a stalled policy cannot burn the 1000-step budget
  and the critic always has terminals to anchor on. It is measured as net
  displacement, so a sideways detour or backing out of a trap never trips it.
- **Path-deviation (`× 0.15`, clipped at 1.0 m)**: the A* reference path from
  `path_N.npy` is a *reward-only* supervisor — never added to the observation.
  At the old `0.03 / 1.5 m` it was noise: -0.008/step at the deviation the
  policy actually ran, against +0.25/step of progress. The policy duly took a
  route 10-18% *shorter* than A* by cutting corners. Every one of the 67
  trained worlds has ≥0.52 m of bottleneck clearance and the A* reference
  itself never enters the blind ring (min 0.525 m), so the safe corridor was
  always there — the reward simply never asked for it.
  The clip must stay above the operating range: at `PATH_MAX_DEV = 0.5 m` the
  term sat pinned at its ceiling (measured mean deviation 0.449 m, max 1.2 m),
  and a clipped penalty has **zero gradient** — it stops saying which way to
  steer and becomes a flat tax on existing.
  Disable via `RLEnvConfig.PATH_FOLLOW_SHAPING=False`.

### Reward magnitude budget (per episode, rough caps)

- Progress total: `REWARD_PROGRESS_SCALE × max_dist` = +125 (moborobo).
- Step penalty: `-0.01 × 500` = -5.
- Proximity: ≤ `0.15 × 500` = -75 if pinned at the floor the whole way.
- Heading: ≤ `0.02 × 500` = -10.
- Goal +100 · Collision -50 · terminal potential `+12.5 × d`.

Successful episode ≈ **+200**; a crash 4 m in ≈ **-85**; a timeout that made
partial progress ≈ **+50**. The ordering success ≫ timeout ≫ collision is what
makes cautious
progress the optimal policy rather than a coin-flip charge.

Note the tuning constraint: the tightest trained world has a 0.52 m
bottleneck, where the prox penalty is -0.135 against +0.25 of progress. Net
per-step reward there stays **positive**, so threading the narrowest gap is
still profitable — raising `REWARD_PROX_SCALE` past the progress cap would
make the necessary passages unprofitable and freeze the robot.

## Training (`train_sac.py`)

- **Episode budget** (`MAX_EPISODE_STEPS`, resolved per-robot in `RLEnvConfig.max_episode_steps`):
  - jackal: **500 steps = 50 s** (max ~25 m at 0.5 m/s — plenty)
  - moborobo: **1000 steps = 100 s** (matches the standard BARN time budget;
    a realistic 0.13 m/s average needs ~75 s to clear the 10 m course)
- **Algorithm**: SB3 SAC, `MultiInputPolicy` with the two-branch
  `ScanStateExtractor` (CNN scan + FC state) and `[256, 256]` heads.
- **Optimizer**: Adam, `lr=3e-4`, `batch_size=256`, `buffer_size=300_000`,
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
| `SCAN_MEMORY`                 | `True`  | Fill the blind ring from memory; +1 channel.  |
| `SCAN_BLIND_RADIUS`           | 0.45    | Metres — effective sensor near-field floor.   |
| `SCAN_MEM_RADIUS`             | 4.0     | Metres — forget points beyond this.           |
| `SCAN_MEM_VOXEL`              | 0.05    | Metres — point-store dedup grid.              |
| `SCAN_MEM_MAX_AGE`            | 150     | Steps before an unconfirmed point is dropped. |
| `STEP_DURATION`               | 0.1 s   | Sim-time per env step (10 Hz policy).         |
| `MAX_EPISODE_STEPS`           | 500     | Truncation limit.                             |
| `REWARD_GOAL`                 | +100    | Terminal bonus on reaching goal.              |
| `REWARD_COLLISION`            | -50     | Terminal penalty on collision.                |
| `REWARD_STEP`                 | -0.01   | Per-step time penalty.                        |
| `REWARD_PROGRESS_SCALE`       | 12.5    | Multiplier on `prev_d - curr_d` (moborobo).   |
| `REWARD_PROX_SCALE`           | 0.15    | Near-obstacle penalty scale.                  |
| `PROX_RAMP_START_M`           | 0.90    | Metres — where the prox penalty begins.       |
| `PROX_FLOOR_M`                | 0.50    | Metres — where it saturates.                  |
| `PROX_ARC`                    | 2.0944  | Radians — half-angle of the travel-direction arc. |
| `REVERSE_SPEED_RATIO`         | 0.3     | Reverse speed limit as a fraction of max.     |
| `REWARD_REVERSE_SCALE`        | 0.05    | Per-step penalty scale while reversing.       |
| `REWARD_HEADING_SCALE`        | 0.02    | Heading-misalignment penalty scale.           |
| `PATH_FOLLOW_SHAPING`         | `True`  | Enable A* path-deviation penalty.             |
| `REWARD_PATH_SCALE`           | 0.15    | Scale of the path-deviation penalty.          |
| `PATH_MAX_DEV`                | 1.0     | Metres past which the deviation is clipped.   |
| `TERMINAL_POTENTIAL_ZERO`     | `False` | Ablation: pay `+scale·d` on `terminated`.     |
| `STUCK_TERMINATION`           | `True`  | Terminate when the robot goes nowhere.        |
| `STUCK_WINDOW_STEPS`          | 200     | Steps the displacement is measured over.      |
| `STUCK_MIN_DISPLACEMENT_M`    | 0.3     | Metres — less than this in the window = stuck. |
| `REWARD_STUCK`                | -50     | Terminal reward on a stuck termination.       |
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
