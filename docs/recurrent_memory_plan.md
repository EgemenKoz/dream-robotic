# Adding recurrent memory (LSTM) to the BARN RL policy

A research plan for giving the navigation policy temporal memory, written to sit
alongside `decision_transformer_eval.md`. It covers what "memory" actually means
here, why the current policy is memoryless, the concrete implementation path on
this codebase (Stable-Baselines3), and an academic-style experimental design.

## TL;DR

- The current policy is **deliberately memoryless**: it maps the last 3 stacked
  LiDAR frames + goal state → action, with no hidden state carried forward. An
  LSTM adds memory **within a single episode** (across the ~100–1000 steps of one
  run), *not* across episodes.
- "The robot loses stuff between episodes" is expected and correct behaviour —
  episodes are meant to be independent (Markov resets). What is preserved
  *between* episodes is the **network weights** (the learned skill), not any
  per-run memory. If the real complaint is "it forgets easy worlds after training
  on hard ones", that is **catastrophic forgetting**, a *different* problem that
  an LSTM does **not** fix (see §2).
- Recommended first step: **Recurrent PPO** (LSTM) from `sb3-contrib`, reusing the
  existing `ScanStateExtractor` CNN as the per-step encoder, with the scan frame
  stack reduced to 1 (the LSTM subsumes frame-stacking). Recurrent SAC is a much
  larger research effort and is proposed only as a stretch goal.
- Expected payoff is **modest but real**: BARN with LiDAR + goal is close to
  Markov, so recurrence mainly helps with occlusion (obstacles that leave the
  field of view), dead-end backtracking, and implicit velocity estimation.

---

## 0. First, the "2k vs. much more" question

There is a terminology mix-up worth clearing up because it changes how you read
every training log.

| Term | What it means here | Typical value |
| --- | --- | --- |
| **step / timestep** | one 0.1 s control tick (one `env.step()`, one policy action) | the training budget is counted in these |
| **episode** | one attempt at a world, from `reset()` to goal / collision / timeout | 162–410 steps each in practice |
| **`--total-steps`** | the training budget, in *steps* not episodes | **SAC default = 200 000**, **PPO default = 500 000** |

The "one was 2k, the other much more" is almost certainly the two **default
training budgets** baked into the two scripts:

- `train_sac.py`  → `--total-steps 200000` (SAC replays each transition many
  times from its buffer, so it is sample-efficient and needs fewer steps).
- `train_ppo.py`  → `--total-steps 500000` (PPO is on-policy — it throws each
  rollout away after a few gradient epochs, so it needs 3–5× more environment
  steps to reach the same performance; this is documented in the `train_ppo.py`
  header).

So it is not "2 000 vs. a lot" — it is **200 000 vs. 500 000 steps**, and the
gap exists because of *on-policy (PPO) vs. off-policy (SAC)* sample efficiency,
not because one run is a toy. For reference, one 200 k-step SAC run takes
~5.5 h of wall-clock because Gazebo runs at ~10 Hz real-time (the bottleneck is
data collection, not gradient steps).

What is actually on disk in `results/rl/runs/`:

- `20260521_211755` — SAC, trained out to **520 000 steps** (checkpoints up to
  `model_520000_steps.zip`), 1 387 episodes, **≈35 % success**.
- `20260712_005418` — SAC, ~180–200 k steps.
- `ppo_20260713_235029` — PPO run, ~180 k steps so far.

Number of *episodes* in a run (390, 780, 1387…) is just a by-product of how many
steps each episode happened to consume — it is not the training budget.

---

## 1. What "memory" means, and why the current policy has none

### 1.1 The policy is memoryless by construction

The observation (see `rl/env/barn_env.py` and `configs/rl_config.py`) is:

```
obs["scan"]  : (SCAN_STACK=3, NUM_SCAN_RAYS=90)   last 3 min-pooled LiDAR frames
obs["state"] : (5,)  [dist_norm, sin(θ), cos(θ), last_lin, last_ang]
```

The 3-frame stack is a **fixed, tiny memory window** — 0.3 s at 10 Hz — chosen
so the CNN can infer *approach speed* toward an obstacle (exactly the Atari
frame-stack trick). Beyond those 3 frames the policy remembers nothing: feed it
the same current stack and it produces the same action distribution regardless
of what happened 5 seconds ago. The config comment even says so: *"the policy is
otherwise memoryless."*

### 1.2 Two very different "memories" — do not conflate them

1. **Within-episode memory** (what an LSTM gives you). A hidden state `h_t`
   carried from step to step *inside one run*. Reset to zeros at every
   `env.reset()`. This lets the robot remember "I already tried the left branch
   and it was a dead end" for the duration of *that* episode.

2. **Across-episode / across-run memory** = the **network weights**. This is the
   learned skill, updated by gradient descent, and it *is* preserved between
   episodes and between training sessions (that is what checkpoints save). The
   robot does not "lose" this between episodes — if it did, training could never
   converge.

**An LSTM only addresses (1).** If your concern is (2) — the policy getting worse
at easy worlds after you train it on hard ones during the curriculum — that is
*catastrophic forgetting*, and the fixes are continual-learning tricks
(rehearsal / mixed sampling, EWC, larger nets), **not** recurrence. Good news:
the curriculum here is already **cumulative** (easy → easy+medium → all worlds;
see `rl/curriculum.py`), so easy worlds stay in the training pool and forgetting
is already partly mitigated. If forgetting is the real target, say so and this
plan changes substantially.

The rest of this document assumes the goal is **within-episode memory (1)** — a
recurrent policy — which is the standard meaning of "add an LSTM to the RL
agent".

---

## 2. Is BARN even a POMDP? (the honest motivation)

Recurrence only helps if the task is partially observed — if the current
observation is a *sufficient statistic* for the optimal action, an LSTM is dead
weight. BARN with LiDAR + goal is **close to Markov**, but not fully:

| Partial-observability source | Present in BARN? | Does memory help? |
| --- | --- | --- |
| Obstacle leaves the LiDAR field of view (behind/beside robot) | Yes — 2D scan, limited arc, occlusion | Yes — remember it is there while turning |
| Dead ends / pockets requiring backtracking | Some hard worlds | Yes — remember explored branches |
| Velocity / acceleration not in a single frame | Partially handled by 3-frame stack | LSTM subsumes the stack, cleaner |
| Sensor noise / dropouts | Mild in sim | Yes — temporal smoothing |
| Goal direction unknown | **No** — `state` gives dist + bearing every step | No |

So the expected win is **real but bounded**. This is worth stating up front in
any paper: the interesting scientific question is not "does LSTM help" (it
sometimes marginally does) but **"how much of BARN's residual difficulty is
partial observability vs. exploration/optimisation?"** — and the frame-stack vs.
LSTM vs. no-history ablation (below) answers exactly that.

---

## 3. Implementation path on this codebase

### 3.1 Recommended: Recurrent PPO from `sb3-contrib` (low risk)

Stable-Baselines3 core (installed here: **2.4.1**) has **no** recurrent policy.
Its sister package **`sb3-contrib`** ships `RecurrentPPO` with an LSTM and — key
for us — a `MultiInputLstmPolicy` that accepts Dict observations. This is the
smallest, most defensible change.

```bash
pip install sb3-contrib   # not currently installed in this env
```

Design:

- **Keep** `ScanStateExtractor` (the two-branch CNN+FC) as the per-step feature
  extractor. It outputs a 160-d vector (128 scan + 32 state).
- **Insert an LSTM** after the extractor, before the actor/critic heads.
  `RecurrentPPO` does this for you: extractor → LSTM → heads.
- **Reduce `SCAN_STACK` 3 → 1.** The LSTM now provides temporal memory, so
  stacking is redundant; keeping both is a valid ablation but start with 1 to
  isolate the LSTM's contribution. (This changes the CNN input channels from 3
  to 1 — handled automatically by `ScanStateExtractor`, which reads
  `k = observation_space["scan"].shape[0]`.)
- **Hidden size** 128–256; **1 LSTM layer** to start. Shared LSTM between actor
  and critic (`shared_lstm=True, enable_critic_lstm=False`) is cheaper and
  usually fine at this scale; the alternative (separate critic LSTM) is a tuning
  knob, not a day-one decision.

New file `train_recurrent_ppo.py` (fork of `train_ppo.py`), changed lines only:

```python
from sb3_contrib import RecurrentPPO

policy_kwargs = dict(
    features_extractor_class=ScanStateExtractor,
    net_arch=RLModelConfig.NET_ARCH,
    lstm_hidden_size=256,
    n_lstm_layers=1,
    shared_lstm=True,
    enable_critic_lstm=False,
)

model = RecurrentPPO(
    "MultiInputLstmPolicy", env,
    n_steps=args.n_steps, batch_size=args.batch_size, ...
    policy_kwargs=policy_kwargs,
)
```

Everything else — `WorldCyclingEnv`, the curriculum callback, checkpointing,
`episodes.csv` logging — is unchanged. **Evaluation/inference must carry the
hidden state**: at deploy time you keep the returned `(lstm_states, episode_start)`
and pass them into the next `model.predict(...)`, resetting `episode_start=True`
on each `env.reset()`. The benchmark harness (`benchmark/run_benchmark.py`) and
any RL controller wrapper (`rl/controllers/rl_sac_controller.py`) need this small
change to score the recurrent policy correctly.

**Config knobs to add** (`configs/rl_config.py`, new `RLRecurrentConfig` class):
`LSTM_HIDDEN_SIZE`, `N_LSTM_LAYERS`, `SHARED_LSTM`, so recurrence is tunable from
one place like the rest of the RL config.

### 3.2 Stretch goal: Recurrent SAC (high risk, more novel)

SAC is the project's main workhorse (200 k-step budget, off-policy, sample
efficient), but **neither SB3 nor sb3-contrib ships a recurrent SAC**. Doing it
right is an R2D2-style effort:

- Store **sequences** (not single transitions) in the replay buffer, with the
  LSTM hidden state at sequence start.
- **Burn-in**: replay a prefix (e.g. 10–20 steps) to warm the hidden state
  before computing the loss on the remainder (Kapturowski et al., R2D2, ICLR'19),
  because stored hidden states go stale as the network updates.
- Handle episode boundaries inside sequences (zero the state on `done`).

This is a genuine implementation project (custom buffer, custom policy, careful
masking) and a stronger paper contribution, but it should **not** be attempted
before the RecurrentPPO baseline shows recurrence helps at all. If it does not
help PPO, it will not justify the SAC engineering.

### 3.3 What NOT to do

- Don't hand-roll an LSTM into the current SAC `MultiInputPolicy` without the
  sequence buffer + burn-in — feeding a per-transition replay buffer through an
  LSTM with zeroed hidden state each time trains the recurrence on garbage.
- Don't drop the CNN. Scan locality still matters per frame; the LSTM is memory
  *over* frames, not a replacement for spatial structure.

---

## 4. Experimental design (academic framing)

### 4.1 Research question & hypotheses

- **RQ:** How much of BARN navigation difficulty is due to partial observability,
  and does an explicit recurrent memory close that gap relative to fixed
  frame-stacking?
- **H1:** A recurrent policy improves success rate on the *hard* tier (occlusion,
  dead ends) more than on the *easy* tier.
- **H2:** With an LSTM, `SCAN_STACK=1` matches or beats `SCAN_STACK=3` (the LSTM
  subsumes the stack), i.e. recurrence and stacking are substitutes, not
  complements.
- **H0 (null we must be able to accept):** On this obs space, recurrence gives no
  significant gain — BARN is effectively Markov given LiDAR + goal.

### 4.2 Conditions / ablation grid

Hold reward shaping, curriculum, world set, and total-steps fixed; vary only the
memory mechanism:

| Cond. | Memory | `SCAN_STACK` | Notes |
| --- | --- | --- | --- |
| A (baseline) | none | 1 | true memoryless reference |
| B (current) | frame stack | 3 | the shipping policy |
| C | LSTM | 1 | the proposed method |
| D | LSTM + stack | 3 | do they compound? |
| (E, stretch) | Recurrent SAC | 1 | §3.2 |

Run each on the same **50-world eval set** (`RLEvalConfig.EVAL_WORLDS`) with
≥3 seeds; report mean ± std. Same `--total-steps` for a fair *sample* budget
(note PPO ≠ SAC step-for-step, so keep the algorithm fixed within a comparison —
compare A–D under PPO, and treat SAC separately).

### 4.3 Metrics

Beyond success rate, log/derive from `episodes.csv` and the benchmark harness:

- **Success rate** per difficulty tier (easy/medium/hard from `rl/curriculum.py`).
- **Collision vs. timeout breakdown** (memory should cut *timeouts* in dead-end
  worlds specifically — the tell-tale sign recurrence is doing its job).
- **SPL** (Success weighted by Path Length) or time-to-goal on solved worlds.
- **Sample efficiency**: success-rate-vs-steps curves (from TensorBoard).
- **A memory-probe test**: a hand-built or hand-picked dead-end world where the
  optimal move requires remembering an already-explored branch. Condition C
  should beat B here even if aggregate numbers are close — this is the cleanest
  qualitative evidence that memory is being used.

### 4.4 Compute budget & risks

- One PPO condition ≈ 500 k steps ≈ ~14 h wall-clock (Gazebo-bound). 4 conditions
  × 3 seeds = 12 runs ≈ 1–2 GPU-weeks serial; parallelise across machines/worlds
  where possible (Gazebo is the bottleneck, GPU is idle most of the time — the
  box here has CUDA available but training is I/O/sim-bound).
- **Risk:** recurrence is finicky to train (BPTT, exploding/vanishing gradients,
  sensitivity to `n_steps` sequence length). Mitigate with grad clipping (SB3
  default 0.5), start with hidden size 128, and sanity-check on the 10-world
  `QUICK_EVAL_WORLDS` before committing to full runs.
- **Risk:** the honest outcome may be H0 (no gain). That is still a publishable
  negative result *if* framed as "quantifying partial observability in BARN" with
  the ablation grid above — which is why A–D matters more than just shipping C.

---

## 5. Recommended order of work

1. `pip install sb3-contrib`; write `train_recurrent_ppo.py` (fork of
   `train_ppo.py`); add `RLRecurrentConfig` to `configs/rl_config.py`.
2. Make evaluation/inference stateful (carry `lstm_states` across steps, reset on
   episode start) in the benchmark harness and the RL controller wrapper.
3. Smoke-test on `QUICK_EVAL_WORLDS` (10 worlds) — verify the hidden state resets
   correctly and the policy learns at all.
4. Run the A–D ablation on the 50-world set, ≥3 seeds.
5. Build the dead-end memory-probe world; run the qualitative test.
6. Only if C clearly beats B: invest in Recurrent SAC (§3.2).

---

## 6. Verdict

| Question | Answer |
| --- | --- |
| Does an LSTM make the robot "remember between episodes"? | No — within an episode only. Between-episode memory is the weights, already preserved. |
| Cheapest correct implementation? | `sb3-contrib` `RecurrentPPO` + `MultiInputLstmPolicy`, reuse `ScanStateExtractor`, `SCAN_STACK=1`. |
| Will it clearly beat the current frame-stack policy? | Uncertain — BARN is near-Markov; expect modest, tier-dependent gains. Ablation A–D is the real contribution. |
| Recurrent SAC? | Higher ceiling and more novel, but a real R2D2-style build — do it only after RecurrentPPO validates the idea. |
| Should we do it now? | Yes as a scoped study, provided the goal is within-episode memory. If the goal is "stop forgetting easy worlds", that's catastrophic forgetting — a different plan. |
