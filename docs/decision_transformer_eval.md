# Could we use a Decision Transformer for BARN navigation?

Evaluation of replacing (or complementing) SAC with a Decision Transformer
(DT — Chen et al., NeurIPS 2021) on this project's setup: BARN static worlds,
Gazebo at 10 Hz, LiDAR + goal observation, continuous `(linear, angular)` action.

## TL;DR

**Feasible, but not as a replacement for online SAC — and not first.** The one
genuinely attractive fit is *offline learning from our own classical planners*:
we already have MPPI / DWA / elastic-band controllers that beat the current RL
policy, and the benchmark harness can log their trajectories across all 300
worlds for free. That is exactly the "learn from mixed-quality offline data"
regime DT was designed for. But DT will at best approach the quality of the
data it is fed, so it competes with plain behaviour cloning of MPPI rather
than with a well-tuned SAC. Recommended order: (1) retrain SAC on the new
two-part structure, (2) if we want an offline track, log expert datasets from
the benchmark anyway (cheap, parallelisable), (3) only then try DT vs. BC on
that dataset.

## What DT would look like here

- **Token sequence**: `(R̂_t, s_t, a_t)` triplets; state = the same two-part
  obs (scan stack through the ScanStateExtractor CNN → embedding, state vector
  → embedding), action = 2-d continuous head, return-to-go conditioning.
- **Context window**: K = 20–30 steps (2–3 s at 10 Hz) is the standard DT
  size and fine here — obstacle avoidance is a local decision; the goal
  direction is already in the state, so long-horizon context is not needed.
- **Inference cost**: a 3-layer, 128-d transformer over K=20 context runs in
  well under 10 ms on CPU — no problem at a 10 Hz control rate.
- **Data source**: `benchmark/run_benchmark.py` episodes with MPPI / DWA /
  elastic-band across 300 worlds, plus failed runs (DT benefits from seeing
  bad trajectories with low return-to-go labels).

## Arguments for

1. **We own a free expert.** MPPI already navigates BARN well. Offline data
   collection runs in parallel across machines/worlds, unlike online SAC
   which is serialised behind real-time Gazebo (~5.5 h per 200 k steps).
2. **No exploration problem.** The 28 % SAC result is partly an exploration /
   reward-shaping problem; DT sidesteps both — no reward shaping needed at
   all, only episode returns (success/collision/time suffice).
3. **Return-conditioning gives a "quality dial".** Condition on high
   return-to-go at deploy time to ask for expert-level behaviour, even though
   the dataset mixes good and bad runs.
4. **Multi-planner distillation.** One DT can be trained on trajectories from
   *all* our planners at once and, in principle, stitch together whichever
   local behaviour was best per situation.

## Arguments against

1. **Ceiling = dataset quality.** DT does not improve beyond the data
   (limited "stitching" in practice — see Emmons et al. '22, RvS). If MPPI
   solves a world, DT imitates it; if no planner solves a world, DT won't
   either. A trained-to-convergence online SAC has a higher ceiling.
2. **Data volume.** DT papers use ~1 M transitions. 300 worlds × ~300 steps
   × 5 planners ≈ 450 k transitions — workable but on the thin side; we'd
   want repeated runs with noise injection for diversity.
3. **Return-to-go is awkward with dense shaped rewards.** Our current shaped
   reward makes RTG world-difficulty-dependent and hard to pick at deploy
   time. Fix: train DT on a *sparse* return (success bonus − time − collision),
   which we can relabel offline for free.
4. **Distribution shift at deploy time.** Compounding errors once the robot
   leaves the expert state distribution — classic imitation problem; BARN's
   tight corridors punish it hard. (DAgger-style relabelling with MPPI is
   possible since the expert is queryable — but that's extra machinery.)
5. **Effort vs. payoff.** No SB3 support — new training stack (e.g.
   HuggingFace `DecisionTransformerModel` or a custom GPT), new replay/dataset
   tooling, new controller. Meanwhile the structural fixes to SAC (this
   change) attack the actual observed failure mode.

## Verdict

| Question | Answer |
| --- | --- |
| Technically possible with this env/obs/action? | Yes — obs/action map cleanly to DT tokens; 10 Hz inference is easy. |
| Best use of it here | Offline distillation of MPPI/DWA demonstrations, sparse-return conditioned. |
| Will it beat MPPI? | Almost certainly not — it learns *from* MPPI. |
| Will it beat a well-trained SAC? | Unlikely if SAC is trained to convergence on the new structure. |
| Should we do it now? | Not yet. Retrain SAC on the two-part structure first; start logging benchmark trajectories in the meantime (zero-cost enabler, useful for BC/DT/offline-RL later). |

A middle path worth noting: **use the demonstrations without a transformer.**
Seeding the SAC replay buffer with MPPI transitions, or adding a BC loss term
(as in TD3+BC / SAC-fD), captures most of the "learn from the classical
planners" benefit with ~50 lines of change instead of a new training stack.
