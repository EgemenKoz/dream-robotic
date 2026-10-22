"""
JAX-accelerated MPPI controller — drop-in replacement for `MPPIController`.

Same public interface (`reset` / `set_goal` / `compute_action`) and same
hyperparameter source (`configs.mppi_config.MPPIConfig`) as the numpy
implementation in `rl/controllers/mppi.py`.  Differences:

  • All sampling, rollouts, cost evaluation, and softmax weighting are
    expressed as JAX ops and JIT-compiled once.
  • The hot loop runs on whatever device JAX picks (GPU if available, else
    CPU vectorised through XLA — still markedly faster than numpy at
    K=900 / T=40 due to fused kernels).
  • Random-number state is a JAX PRNGKey, advanced explicitly each call.

Install JAX (one of the following):
    # CPU only
    pip install -U "jax[cpu]"
    # CUDA 11 GPU
    pip install -U "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    # CUDA 12 GPU
    pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

Quick start
───────────
    from rl.controllers.mppi_jax import MPPIControllerJAX
    ctrl = MPPIControllerJAX(goal_x=gx, goal_y=gy)
    action = ctrl.compute_action(x, y, theta, ranges, ang_min, ang_inc, max_r)

Drop-in inside the benchmark
────────────────────────────
    # benchmark/run_benchmark.py — make_controller(...)
    if planner_name == "mppi_jax":
        return OnlineGlobalPlanner(
            MPPIControllerJAX(goal_x=goal_x, goal_y=goal_y), goal_x, goal_y
        )
"""

import math
from functools import partial
from typing import Optional, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    _HAS_JAX = True
except ImportError as _e:   # pragma: no cover
    _HAS_JAX = False
    _IMPORT_ERROR = _e


# ── Pure JAX core: rollout + cost + softmax update ────────────────────────────
# These functions are JIT-compiled once and reused every step.  Static shapes
# (K, T, N_OBS) are baked in via static_argnames so XLA can fuse the kernels.

def _mppi_step_jax(
    nom_v, nom_w,
    x0, y0, theta0,
    obstacles, n_obs,
    eps_v, eps_w,
    goal,
    dt, max_v, max_w, min_v,
    sigma_v, sigma_w, lam,
    robot_radius, collision_pen, obs_sigma,
    w_obstacle, w_goal, w_goal_step, w_speed,
):
    """
    Single MPPI iteration in JAX.  All arguments are jnp arrays / scalars.

    Returns updated (nom_v, nom_w) of shape (T,) each.
    """
    # Clip perturbed controls
    V = jnp.clip(nom_v[None, :] + eps_v, min_v, max_v)              # (K, T)
    W = jnp.clip(nom_w[None, :] + eps_w, -max_w, max_w)             # (K, T)

    # ── Rollout via lax.scan (unrolls along the T axis) ──────────────────────
    def body(carry, t):
        x, y, th = carry
        v_t = V[:, t]
        w_t = W[:, t]
        x  = x  + v_t * jnp.cos(th) * dt
        y  = y  + v_t * jnp.sin(th) * dt
        th = th + w_t * dt
        return (x, y, th), (x, y)

    K = V.shape[0]
    T = V.shape[1]
    x_init = jnp.full((K,), x0)
    y_init = jnp.full((K,), y0)
    t_init = jnp.full((K,), theta0)
    _, (xs, ys) = jax.lax.scan(body, (x_init, y_init, t_init), jnp.arange(T))
    xs = xs.T          # (K, T)
    ys = ys.T

    final_x = xs[:, -1]
    final_y = ys[:, -1]

    # ── Obstacle cost ────────────────────────────────────────────────────────
    # obstacles is (N_OBS_MAX, 2); n_obs is the number actually valid.
    # We mask invalid rows by setting their distance to a large constant.
    pos_xy = jnp.stack([xs, ys], axis=-1)                  # (K, T, 2)
    diff   = pos_xy[:, :, None, :] - obstacles[None, None, :, :]
    dists  = jnp.sqrt((diff ** 2).sum(axis=-1))            # (K, T, N_OBS_MAX)

    # mask: 1 for valid, 0 for padded
    valid_mask = (jnp.arange(obstacles.shape[0]) < n_obs).astype(dists.dtype)
    dists = dists + (1.0 - valid_mask)[None, None, :] * 1e6
    min_dists = dists.min(axis=-1)                         # (K, T)

    in_collision = (min_dists < robot_radius).any(axis=1).astype(jnp.float32)
    hard_cost = in_collision * collision_pen

    clearance = jnp.maximum(min_dists - robot_radius, 0.0)
    soft_cost = jnp.exp(-clearance / obs_sigma).sum(axis=1)

    # Only enable obstacle cost when there is at least one valid obstacle.
    has_obs = (n_obs > 0).astype(jnp.float32)
    S = has_obs * (hard_cost + soft_cost * w_obstacle)

    # ── Goal costs ──────────────────────────────────────────────────────────
    terminal_dist = jnp.sqrt((final_x - goal[0]) ** 2 + (final_y - goal[1]) ** 2)
    S = S + terminal_dist * w_goal

    step_dist = jnp.sqrt(
        (xs - goal[0]) ** 2 + (ys - goal[1]) ** 2
    ).mean(axis=1)
    S = S + step_dist * w_goal_step

    # Speed cost
    S = S + w_speed * (max_v - V.mean(axis=1))

    # MPPI regularisation
    reg = (
        (nom_v[None, :] / (sigma_v ** 2)) * eps_v
        + (nom_w[None, :] / (sigma_w ** 2)) * eps_w
    ).sum(axis=1)
    S = S + lam * reg

    # ── Softmax weights & update ────────────────────────────────────────────
    beta = S.min()
    w = jnp.exp(-(S - beta) / lam)
    w = w / (w.sum() + 1e-10)

    new_nom_v = jnp.clip((w[:, None] * V).sum(axis=0), 0.0, max_v)
    new_nom_w = jnp.clip((w[:, None] * W).sum(axis=0), -max_w, max_w)
    return new_nom_v, new_nom_w


# ── Controller class ──────────────────────────────────────────────────────────

class MPPIControllerJAX:
    """
    JAX-accelerated MPPI.  Public API matches `rl.controllers.MPPIController`.

    Parameters
    ----------
    goal_x, goal_y : float
        World-frame goal position.
    config : class, optional
        Configuration class (defaults to `configs.mppi_config.MPPIConfig`).
    seed : int, optional
        Seed for the JAX PRNG.
    max_obstacles : int, optional
        Capacity for the obstacle point cloud passed to JIT.  Must be ≥ the
        number of valid scan rays after subsampling.  Defaults to 256.
    """

    def __init__(
        self,
        goal_x: float,
        goal_y: float,
        config=None,
        seed: int = 0,
        max_obstacles: int = 256,
    ):
        if not _HAS_JAX:
            raise ImportError(
                "MPPIControllerJAX requires JAX. Install with:\n"
                "    pip install -U 'jax[cpu]'\n"
                "  or for GPU:\n"
                "    pip install -U 'jax[cuda12_pip]' "
                "-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html\n"
                "Original error: %r" % _IMPORT_ERROR
            )

        if config is None:
            from configs.mppi_config import MPPIConfig
            config = MPPIConfig

        self.cfg = config
        self._goal = np.array([goal_x, goal_y], dtype=np.float32)
        self._max_obstacles = int(max_obstacles)

        # Bake float dtype to f32 — fast on every backend
        self._dtype = jnp.float32

        # Nominal control sequence
        self._nom_v = jnp.zeros(config.T, dtype=self._dtype)
        self._nom_w = jnp.zeros(config.T, dtype=self._dtype)

        # JAX PRNG state
        self._key = jax.random.PRNGKey(int(seed))

        # JIT once — static shapes keep the compiled cache hot.
        self._jit_step = jax.jit(_mppi_step_jax)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self):
        # Warm-start at half max speed (matches numpy version)
        v0 = float(self.cfg.MAX_V) * 0.5
        self._nom_v = jnp.full((self.cfg.T,), v0, dtype=self._dtype)
        self._nom_w = jnp.zeros((self.cfg.T,), dtype=self._dtype)

    def set_goal(self, goal_x: float, goal_y: float):
        self._goal[:] = [goal_x, goal_y]

    def compute_action(
        self,
        robot_x: float,
        robot_y: float,
        robot_theta: float,
        scan_ranges,
        scan_angle_min: float,
        scan_angle_increment: float,
        scan_max_range: float,
    ) -> np.ndarray:
        cfg = self.cfg
        K, T = cfg.K, cfg.T

        # ── Obstacles (numpy → padded JAX array of fixed length) ─────────────
        obs_np = self._scan_to_obstacles(
            scan_ranges, scan_angle_min, scan_angle_increment,
            scan_max_range, robot_x, robot_y, robot_theta,
        )
        n_obs = obs_np.shape[0]
        if n_obs > self._max_obstacles:
            # Random subsample — keeps the JIT cache hot (fixed shape).
            idx = np.random.choice(n_obs, self._max_obstacles, replace=False)
            obs_np = obs_np[idx]
            n_obs = self._max_obstacles
        if n_obs < self._max_obstacles:
            pad = np.zeros((self._max_obstacles - n_obs, 2), dtype=np.float32)
            obs_np = np.concatenate([obs_np.astype(np.float32), pad], axis=0)
        obstacles = jnp.asarray(obs_np, dtype=self._dtype)

        # ── Sample perturbations on device ───────────────────────────────────
        self._key, ka, kb = jax.random.split(self._key, 3)
        eps_v = jax.random.normal(ka, (K, T), dtype=self._dtype) * cfg.SIGMA_V
        eps_w = jax.random.normal(kb, (K, T), dtype=self._dtype) * cfg.SIGMA_W

        # ── Run JIT step ─────────────────────────────────────────────────────
        min_v = float(getattr(cfg, "MIN_V", 0.0))
        w_speed = float(getattr(cfg, "W_SPEED", 0.0))

        new_v, new_w = self._jit_step(
            self._nom_v, self._nom_w,
            jnp.float32(robot_x), jnp.float32(robot_y), jnp.float32(robot_theta),
            obstacles, jnp.int32(n_obs),
            eps_v, eps_w,
            jnp.asarray(self._goal, dtype=self._dtype),
            jnp.float32(cfg.DT),
            jnp.float32(cfg.MAX_V), jnp.float32(cfg.MAX_W), jnp.float32(min_v),
            jnp.float32(cfg.SIGMA_V), jnp.float32(cfg.SIGMA_W), jnp.float32(cfg.LAMBDA),
            jnp.float32(cfg.ROBOT_RADIUS), jnp.float32(cfg.COLLISION_PENALTY),
            jnp.float32(cfg.OBS_SIGMA),
            jnp.float32(cfg.W_OBSTACLE), jnp.float32(cfg.W_GOAL),
            jnp.float32(cfg.W_GOAL_STEP), jnp.float32(w_speed),
        )

        # First action, then receding-horizon shift
        action = np.array([float(new_v[0]), float(new_w[0])], dtype=np.float32)

        # Roll left + warm-start the last slot at the previous final value
        self._nom_v = jnp.concatenate([new_v[1:], new_v[-1:]])
        self._nom_w = jnp.concatenate([new_w[1:], new_w[-1:]])
        return action

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _scan_to_obstacles(
        self,
        ranges,
        angle_min: float,
        angle_increment: float,
        max_range: float,
        robot_x: float,
        robot_y: float,
        robot_theta: float,
    ) -> np.ndarray:
        cfg = self.cfg
        ranges = np.asarray(ranges, dtype=np.float32)
        n = len(ranges)
        idx = np.arange(0, n, cfg.SCAN_SUBSAMPLE)
        r = ranges[idx]
        a = angle_min + idx * angle_increment

        valid = np.isfinite(r) & (r > 0.01) & (r < max_range * 0.99)
        if not valid.any():
            return np.zeros((0, 2), dtype=np.float32)

        r = r[valid]
        a = a[valid]
        ox_r = r * np.cos(a)
        oy_r = r * np.sin(a)
        cos_t = math.cos(robot_theta)
        sin_t = math.sin(robot_theta)
        ox_w = robot_x + cos_t * ox_r - sin_t * oy_r
        oy_w = robot_y + sin_t * ox_r + cos_t * oy_r
        return np.stack([ox_w, oy_w], axis=1).astype(np.float32)
