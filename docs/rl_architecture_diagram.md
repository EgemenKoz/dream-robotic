# RL Infrastructure — Architecture Diagram

BARN navigation with **SAC** (Soft Actor-Critic, model-free off-policy RL).
Continuous state → continuous action. **LiDAR is the primary sensor.** Pose comes
from Gazebo ground-truth (not odometry/AMCL during training).

## At a glance

| Aspect | Choice |
| --- | --- |
| Algorithm | SAC (SB3), `MultiInputPolicy` + two-branch `ScanStateExtractor`, heads `[256, 256]` |
| Paradigm | Model-free, off-policy, continuous control |
| State-based? | Two-part Dict: LiDAR stack → 1D-CNN branch, goal/vel state → FC branch |
| LiDAR used? | **Yes** — 3 stacked frames of 90 min-pooled sectors, clipped at 5 m |
| Observation | `{"scan": (3, 90), "state": (5,)}` (scan stack + goal-relative + last action) |
| Action | 2 continuous: `linear_x ∈ [0, MAX_LIN]`, `angular_z ∈ [±MAX_ANG]` |
| Pose source | Gazebo `/gazebo/get_model_state` (ground truth) |
| Reward | Sparse terminal (+100 / -50) + dense shaping (progress/prox/heading/A*-path) |
| Sim | Gazebo @ 10 Hz policy rate (`STEP_DURATION = 0.1 s`) |

## System architecture

```mermaid
flowchart TB
    subgraph TRAIN["Training — train_sac.py"]
        SAC["SB3 SAC agent\nMultiInputPolicy: CNN scan + FC state → [256,256]\nactor + twin critics + replay buffer"]
        WCE["WorldCyclingEnv\n(fix world N episodes, then cycle)"]
        CURR["CurriculumCallback\neasy → +medium → +hard\n(rl/curriculum.py, BARN metrics)"]
        CKPT["CheckpointCallback\nmodel_*.zip + replay buffer"]
        PROG["TrainProgressCallback\nrolling reward/len"]
    end

    subgraph ENV["BarnEnv (Gymnasium) — rl/env/barn_env.py"]
        OBS["_build_obs()\nassemble Dict obs {scan (3,90), state (5,)}"]
        STEP["step(): unpause → /cmd_vel → sleep 0.1s → pause"]
        RWD["reward = step + progress + prox + heading + A*path\n+ terminal goal/collision"]
        RESET["reset(): teleport to start, drain collision"]
    end

    subgraph ROS["ROS / Gazebo (managed subprocesses)"]
        GZ["gzserver\nBARN static world 0–299"]
        SCAN["LiDAR → /scan\n(moborobo: pointcloud_to_laserscan)"]
        POSE["/gazebo/get_model_state\nGROUND-TRUTH pose"]
        CMD["/cmd_vel (Twist)"]
    end

    SAC -->|action a_t| WCE
    WCE --> STEP
    STEP --> CMD --> GZ
    GZ --> SCAN
    GZ --> POSE
    SCAN -->|ranges| OBS
    POSE -->|x,y,yaw| OBS
    OBS -->|obs s_t+1| WCE
    STEP --> RWD -->|reward r_t| WCE
    WCE -->|s,a,r,s'| SAC
    CURR -.swaps world pool.-> WCE
    CKPT -.-> SAC
    PROG -.-> SAC
    RESET --> GZ

    classDef sensor fill:#cfe8ff,stroke:#2b6cb0;
    class SCAN,POSE sensor;
```

## Observation + network — what the policy actually sees

LSTM DENENECEK CEVRESEL HIZ VS EKLE


```mermaid
flowchart LR
    subgraph S["Observation s_t (Dict, all float32)"]
        direction TB
        L["obs['scan']  (3, 90)\n3 stacked frames @ 10 Hz (0.3 s history)\nraw ~722/1080 rays → MIN-pooled to 90 sectors\nclip(range, 0, 5 m) / 5 m → [0,1]"]
        ST["obs['state']  (5,)\n[dist/max_dist, sin(angle), cos(angle),\n last_lin/MAX_LIN, last_ang/MAX_ANG]"]
    end
    subgraph NET["ScanStateExtractor → SAC heads"]
        direction TB
        CNN["Conv1d 3→32 (k5, s2)\nConv1d 32→32 (k3, s2)\n→ FC 128"]
        FC["FC 32"]
        CAT["concat (160)\n→ [256, 256] actor / twin critics"]
    end
    LIDAR(["LiDAR /scan"]) --> L
    GT(["Ground-truth pose + goal"]) --> ST
    PREV(["Previous action"]) --> ST
    L --> CNN --> CAT
    ST --> FC --> CAT

    classDef sensor fill:#cfe8ff,stroke:#2b6cb0;
    class LIDAR sensor;
```

Notes that the diagram encodes:
- **Two-part structure** (Fan & Long et al. ICRA'18; DRL-VO T-RO'23): perception and
  goal/velocity state are encoded by separate branches and fused before the heads.
  The conv treats the 3 stacked frames as channels, so kernels see per-sector change
  over time (approach speed) — the policy is otherwise memoryless.
- **Min-pooling, not interpolation:** a thin obstacle hit by one raw ray can never
  vanish from the downsampled scan.
- **5 m clip normalisation:** BARN gaps are < 1 m; normalising by the raw 10 m sensor
  range would squash all decision-relevant geometry into [0, 0.1].
- **The A\* reference path is reward-only** — it shapes the reward (path-deviation penalty)
  but is deliberately **never** in the observation, so the policy must still infer the
  corridor from LiDAR alone.

## Train ≡ deploy contract

```mermaid
flowchart LR
    M["model_final.zip\n(trained SAC policy)"]
    subgraph DEP["SACController — rl/controllers/rl_sac_controller.py"]
        RB["_build_obs() rebuilds the EXACT Dict obs\n(same min-pool + clip + frame stack as BarnEnv;\nlegacy flat checkpoints auto-detected)"]
        PRED["model.predict(obs, deterministic=True)"]
    end
    BENCH["benchmark/run_benchmark.py\n--planners rl_sac --rl-model PATH"]
    M --> DEP
    BENCH --> DEP
    LIVE(["live LiDAR + pose (caller-supplied)"]) --> RB --> PRED -->|linear, angular| BENCH
```

`SACController` reproduces `BarnEnv._build_obs()` bit-for-bit so the deployed policy
sees the same input distribution it trained on. At deploy time the **pose source is a
function argument** — the benchmark passes ground-truth, but a real robot would inject
`/odom` or AMCL there instead.
