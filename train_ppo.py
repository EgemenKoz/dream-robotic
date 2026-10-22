"""
PPO training script for BARN navigation.

Trains a Proximal Policy Optimisation policy on the BARN Gazebo simulation
using Stable Baselines 3 and the BarnEnv Gymnasium wrapper.

Usage examples
──────────────
    # Train moborobo on the standard 50-world eval set, 500k steps
    python3 train_ppo.py --robot moborobo

    # Train jackal on a 10-world subset, save every 20k
    python3 train_ppo.py --robot jackal --worlds 0 6 12 18 24 30 36 42 48 54 \\
                         --total-steps 500000 --save-every 20000

    # Continue from a checkpoint
    python3 train_ppo.py --robot moborobo --resume results/rl/runs/RUN_ID/model_100000

    # Quick smoke test (verify Gazebo / ROS wiring)
    python3 train_ppo.py --robot moborobo --smoke-test

PPO vs SAC
──────────
PPO is on-policy: it collects a rollout of --n-steps environment steps, runs
several gradient epochs over that batch, then discards it.  SAC replays each
transition many times from a buffer.

Consequence for Gazebo training: data collection is the bottleneck (~10 Hz
real-time), so PPO needs 3–5× more total steps than SAC to reach the same
performance.  Compensate with --total-steps 500000 or more.

What PPO offers in return: simpler tuning (no replay buffer, no target-network
soft-update τ), more stable gradients under reward-shaping noise, and no
cold-start period before learning begins (SAC's --learning-starts).
"""

import argparse
import datetime
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── Set robot BEFORE any config import that reads SimConfig.ROBOT ──────────────
from configs.config import SimConfig, WorldConfig
_early = argparse.ArgumentParser(add_help=False)
_early.add_argument("--robot", default="moborobo")
_early_args, _ = _early.parse_known_args()
SimConfig.ROBOT = _early_args.robot

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from rl.env.barn_env import BarnEnv
from rl.models.feature_extractor import ScanStateExtractor
from configs.rl_config import RLEnvConfig, RLLogConfig, RLModelConfig


# ── World-cycling wrapper (identical to train_sac.py) ─────────────────────────

class WorldCyclingEnv(BarnEnv):
    """
    BarnEnv that cycles through a list of worlds N episodes at a time.

    Gazebo is only restarted when the world index actually changes, so most
    resets are cheap (teleport only).
    """

    def __init__(self, worlds: list, episodes_per_world: int = 5, **kwargs):
        self._world_list = worlds
        self._episodes_per_world = episodes_per_world
        self._world_ptr = 0
        self._ep_on_world = 0
        super().__init__(world_idx=worlds[0], **kwargs)

    def reset(self, **kwargs):
        if self._ep_on_world >= self._episodes_per_world:
            self._world_ptr = (self._world_ptr + 1) % len(self._world_list)
            self._fixed_world_idx = self._world_list[self._world_ptr]
            self._ep_on_world = 0
        self._ep_on_world += 1
        return super().reset(**kwargs)

    def set_world_list(self, worlds: list):
        if worlds and worlds != self._world_list:
            self._world_list = list(worlds)
            self._world_ptr = self._world_ptr % len(self._world_list)


# ── Progress callback ──────────────────────────────────────────────────────────

class TrainProgressCallback(BaseCallback):
    """Prints a one-line training summary every --log-every steps."""

    def __init__(self, log_every: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self._log_every = log_every
        self._ep_rewards = []
        self._ep_lengths = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if isinstance(ep, dict) and "r" in ep:
                self._ep_rewards.append(ep["r"])
                self._ep_lengths.append(ep["l"])

        if self.num_timesteps % self._log_every == 0 and self._ep_rewards:
            n = len(self._ep_rewards)
            mean_r = float(np.mean(self._ep_rewards[-100:]))
            mean_l = float(np.mean(self._ep_lengths[-100:]))
            print(
                "[%7d steps] episodes=%d  mean_reward(100)=%+7.2f  mean_len(100)=%.0f"
                % (self.num_timesteps, n, mean_r, mean_l)
            )
        return True


# ── Curriculum callback ────────────────────────────────────────────────────────

class CurriculumCallback(BaseCallback):
    """Swaps the env's world pool at cumulative-difficulty phase boundaries."""

    def __init__(self, phases, verbose: int = 1):
        super().__init__(verbose)
        self._phases = phases
        self._active_start = None

    def _current_worlds(self):
        from rl.curriculum import phase_for_step
        return phase_for_step(self._phases, self.num_timesteps)

    def _phase_start(self):
        start = self._phases[0][0]
        for s, _ in self._phases:
            if self.num_timesteps >= s:
                start = s
            else:
                break
        return start

    def _unwrap_env(self):
        env = self.training_env.envs[0]
        while hasattr(env, "env") and not hasattr(env, "set_world_list"):
            env = env.env
        return env

    def _on_training_start(self) -> None:
        self._active_start = self._phase_start()
        env = self._unwrap_env()
        worlds = self._current_worlds()
        if hasattr(env, "set_world_list"):
            env.set_world_list(worlds)
        print("[curriculum] phase @ step %d → %d worlds"
              % (self._active_start, len(worlds)), flush=True)

    def _on_step(self) -> bool:
        start = self._phase_start()
        if start != self._active_start:
            self._active_start = start
            env = self._unwrap_env()
            worlds = self._current_worlds()
            if hasattr(env, "set_world_list"):
                env.set_world_list(worlds)
            print("\n[curriculum] step %d: advancing to %d worlds\n"
                  % (self.num_timesteps, len(worlds)), flush=True)
        return True


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train PPO on BARN worlds")
    parser.add_argument(
        "--robot", default="moborobo", choices=["jackal", "moborobo"],
    )
    parser.add_argument(
        "--worlds", type=int, nargs="+", default=None, metavar="IDX",
        help="World indices to train on (default: robot's standard 50-world eval set)",
    )
    parser.add_argument(
        "--total-steps", type=int, default=500_000,
        help="Total environment steps (default: 500000 — PPO needs more than SAC)",
    )
    parser.add_argument(
        "--episodes-per-world", type=int, default=5,
        help="Episodes before cycling to next world (default: 5)",
    )
    parser.add_argument(
        "--save-every", type=int, default=20_000,
        help="Save a checkpoint every N steps (default: 20000)",
    )
    parser.add_argument(
        "--log-every", type=int, default=2_000,
        help="Print progress line every N steps (default: 2000)",
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="Open the Gazebo GUI (gzclient) so training can be watched. "
             "Rendering costs CPU/GPU, so headless (default) trains faster.",
    )
    parser.add_argument(
        "--resume", default=None, metavar="PATH",
        help="Resume training from a checkpoint (path without .zip)",
    )
    parser.add_argument(
        "--run-id", default=None,
        help="Run identifier for the output directory (default: timestamp)",
    )
    parser.add_argument(
        "--curriculum", action="store_true",
        help="Difficulty curriculum: easy → easy+medium → all worlds. "
             "Splits --total-steps into three equal phases. Overrides --worlds.",
    )
    # PPO hyperparameters
    parser.add_argument(
        "--n-steps", type=int, default=2048,
        help="Rollout length per update (default: 2048). Must be divisible by "
             "--batch-size. Increase for more on-policy diversity per update.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Minibatch size for gradient updates (default: 64). "
             "Must divide --n-steps.",
    )
    parser.add_argument("--n-epochs",   type=int,   default=10,
                        help="Gradient epochs over each rollout (default: 10)")
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument(
        "--gamma", type=float, default=0.99,
        help="Discount factor (default: 0.99). The progress reward is dense "
             "and real, so the goal need not be within the discount horizon; "
             "see train_sac.py for the full note.",
    )
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="GAE λ — trades off bias vs variance in advantage "
                             "estimates (default: 0.95)")
    parser.add_argument("--clip-range", type=float, default=0.2,
                        help="PPO clip ratio ε (default: 0.2)")
    parser.add_argument("--vf-coef",    type=float, default=0.5,
                        help="Value function loss coefficient (default: 0.5)")
    parser.add_argument("--ent-coef",   type=float, default=0.01,
                        help="Entropy bonus coefficient — encourages exploration "
                             "(default: 0.01). Raise if the policy collapses early.")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Reset env, run five steps, exit — use to verify Gazebo/ROS wiring.",
    )
    args = parser.parse_args()

    if args.n_steps % args.batch_size != 0:
        parser.error(
            "--n-steps (%d) must be divisible by --batch-size (%d)"
            % (args.n_steps, args.batch_size)
        )

    SimConfig.ROBOT = args.robot

    # ── World list ──────────────────────────────────────────────────────────────
    curriculum_phases = None
    if args.curriculum:
        from rl.curriculum import cumulative_phases, difficulty_tiers
        curriculum_phases = cumulative_phases(args.robot, total_steps=args.total_steps)
        tiers = difficulty_tiers(args.robot)
        world_list = curriculum_phases[0][1]
        print("Curriculum ON  tiers: easy=%d medium=%d hard=%d" % (
            len(tiers["easy"]), len(tiers["medium"]), len(tiers["hard"])))
        for start, worlds in curriculum_phases:
            print("  phase @ step %7d : %d worlds" % (start, len(worlds)))
    elif args.worlds is not None:
        world_list = sorted(args.worlds)
    else:
        world_list = WorldConfig.eval_worlds()
    print("Starting worlds (%d): %s%s" % (
        len(world_list), world_list[:8], " ..." if len(world_list) > 8 else ""
    ))

    # ── Output directory ────────────────────────────────────────────────────────
    run_id = args.run_id or ("ppo_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir = os.path.join(RLLogConfig.RUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    print("Run dir : %s" % run_dir)

    # ── Environment ─────────────────────────────────────────────────────────────
    print("Creating env (will launch Gazebo on first reset)...", flush=True)
    env = WorldCyclingEnv(
        worlds=world_list,
        episodes_per_world=args.episodes_per_world,
        robot=args.robot,
        managed=True,
        gui=args.gui,
        log_dir=run_dir,
    )

    if args.smoke_test:
        print("Smoke test: env.reset() ...", flush=True)
        obs, info = env.reset()
        print("Smoke test: reset OK   scan=%s state=%s  info=%s"
              % (obs["scan"].shape, obs["state"].shape, info), flush=True)
        for i in range(5):
            action = env.action_space.sample()
            obs, r, term, trunc, info = env.step(action)
            print("  step %d  reward=%+.3f  outcome=%s  dist=%.2f"
                  % (i + 1, r, info.get("outcome"), info.get("dist_to_goal", -1)))
            if term or trunc:
                break
        env.close()
        print("Smoke test PASSED. Exiting.")
        return

    env = Monitor(env)

    # ── Policy ──────────────────────────────────────────────────────────────────
    policy_kwargs = dict(
        features_extractor_class=ScanStateExtractor,
        net_arch=RLModelConfig.NET_ARCH,
    )

    tb_dir = os.path.join(run_dir, "tb")
    os.makedirs(tb_dir, exist_ok=True)

    if args.resume:
        print("Resuming from %s" % args.resume)
        model = PPO.load(
            args.resume,
            env=env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            tensorboard_log=tb_dir,
        )
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            policy_kwargs=policy_kwargs,
            verbose=0,
            tensorboard_log=tb_dir,
        )

    print(
        "Policy  : MultiInputPolicy (CNN scan + FC state)  net=%s  lr=%g  "
        "n_steps=%d  batch=%d  epochs=%d  clip=%.2f  ent=%.3f"
        % (policy_kwargs["net_arch"], args.lr, args.n_steps,
           args.batch_size, args.n_epochs, args.clip_range, args.ent_coef)
    )
    print("Steps   : %d  (save every %d)" % (args.total_steps, args.save_every))
    print()

    # ── Callbacks ───────────────────────────────────────────────────────────────
    # PPO has no replay buffer, so CheckpointCallback only saves the policy zip.
    checkpoint_cb = CheckpointCallback(
        save_freq=args.save_every,
        save_path=run_dir,
        name_prefix="model",
        verbose=1,
    )
    progress_cb = TrainProgressCallback(log_every=args.log_every)

    callbacks = [checkpoint_cb, progress_cb]
    if curriculum_phases is not None:
        callbacks.append(CurriculumCallback(curriculum_phases))

    # ── Train ───────────────────────────────────────────────────────────────────
    try:
        model.learn(
            total_timesteps=args.total_steps,
            callback=callbacks,
            reset_num_timesteps=(args.resume is None),
            progress_bar=False,
        )
    finally:
        final_path = os.path.join(run_dir, "model_final")
        model.save(final_path)
        print("\nFinal model saved → %s.zip" % final_path)
        env.close()


if __name__ == "__main__":
    main()
