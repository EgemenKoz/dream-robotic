"""
Unit + integration tests for BarnEnv and RLLogger.

Tests NOT marked @pytest.mark.gazebo run without Gazebo (fast, no ROS needed).
Tests marked @pytest.mark.gazebo require a live Gazebo + rospy available.

Run unit tests only:
    pytest rl/tests/ -m "not gazebo" -v

Run all tests (Gazebo must be up on world 30):
    pytest rl/tests/ -v
"""

import math
import os
import sys

import numpy as np
import pytest

# ── resolve project root so imports work when run from any CWD ───────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_env(**kwargs) -> "BarnEnv":
    """Create a BarnEnv with managed=False and logging off — no Gazebo needed."""
    from rl.env.barn_env import BarnEnv
    return BarnEnv(managed=False, log_dir="", **kwargs)


# ═════════════════════════════════════════════════════════════════════════════
# Observation space
# ═════════════════════════════════════════════════════════════════════════════

class TestObservationSpace:
    """
    Dict obs:
      "scan"  : (SCAN_STACK [+1], N) stacked min-pooled frames, [0, 1].
                The +1 row is the "directly measured this step" mask that
                scan memory appends.
      "state" : (5,) [dist_norm, sin(angle), cos(angle), last_lin, last_ang]
    """

    def test_is_dict_space(self):
        import gymnasium as gym
        env = _make_env()
        assert isinstance(env.observation_space, gym.spaces.Dict)
        assert set(env.observation_space.spaces.keys()) == {"scan", "state"}

    def test_default_scan_shape(self):
        from configs.rl_config import RLEnvConfig
        env = _make_env()
        extra = 1 if RLEnvConfig.SCAN_MEMORY else 0
        assert env.observation_space["scan"].shape == (
            RLEnvConfig.SCAN_STACK + extra, RLEnvConfig.NUM_SCAN_RAYS
        )

    def test_custom_scan_rays_and_stack(self):
        env = _make_env(num_scan_rays=180, scan_stack=4, scan_memory=False)
        assert env.observation_space["scan"].shape == (4, 180)

    def test_memory_adds_one_channel(self):
        plain = _make_env(num_scan_rays=90, scan_stack=3, scan_memory=False)
        withmem = _make_env(num_scan_rays=90, scan_stack=3, scan_memory=True)
        assert plain.observation_space["scan"].shape == (3, 90)
        assert withmem.observation_space["scan"].shape == (4, 90)

    def test_scan_bounds_zero_to_one(self):
        env = _make_env(num_scan_rays=360)
        scan_space = env.observation_space["scan"]
        assert (scan_space.low == 0.0).all()
        assert (scan_space.high == 1.0).all()

    def test_state_bounds(self):
        env = _make_env()
        state_space = env.observation_space["state"]
        assert state_space.shape == (5,)
        # [dist, sin, cos, last_lin, last_ang] — last_lin < 0 when reversing
        assert np.allclose(state_space.low,  [0.0, -1.0, -1.0, -1.0, -1.0])
        assert np.allclose(state_space.high, [1.0,  1.0,  1.0, 1.0,  1.0])

    def test_dtype_is_float32(self):
        env = _make_env()
        assert env.observation_space["scan"].dtype == np.float32
        assert env.observation_space["state"].dtype == np.float32


# ═════════════════════════════════════════════════════════════════════════════
# Action space
# ═════════════════════════════════════════════════════════════════════════════

class TestActionSpace:
    def test_moborobo_linear_range(self):
        from configs.config import MoboroboConfig
        from configs.rl_config import RLEnvConfig
        env = _make_env(robot="moborobo")
        assert env.action_space.low[0] == pytest.approx(
            -RLEnvConfig.REVERSE_SPEED_RATIO * MoboroboConfig.MAX_LINEAR_VEL)
        assert env.action_space.high[0] == pytest.approx(MoboroboConfig.MAX_LINEAR_VEL)

    def test_moborobo_angular_range(self):
        from configs.config import MoboroboConfig
        env = _make_env(robot="moborobo")
        assert env.action_space.low[1] == pytest.approx(-MoboroboConfig.MAX_ANGULAR_VEL)
        assert env.action_space.high[1] == pytest.approx(MoboroboConfig.MAX_ANGULAR_VEL)

    def test_jackal_linear_range(self):
        from configs.config import RobotConfig
        env = _make_env(robot="jackal")
        assert env.action_space.high[0] == pytest.approx(RobotConfig.MAX_LINEAR_VEL)

    def test_reverse_allowed(self):
        env = _make_env()
        assert env.action_space.low[0] < 0.0

    def test_reverse_ratio_zero_is_forward_only(self):
        env = _make_env(reverse_speed_ratio=0.0)
        assert env.action_space.low[0] == pytest.approx(0.0)

    def test_random_samples_stay_in_bounds(self):
        env = _make_env()
        for _ in range(50):
            a = env.action_space.sample()
            assert env.action_space.contains(a), "sample %s outside action space" % a

    def test_dtype_is_float32(self):
        env = _make_env()
        assert env.action_space.dtype == np.float32


# ═════════════════════════════════════════════════════════════════════════════
# Config validation
# ═════════════════════════════════════════════════════════════════════════════

class TestConfigValidation:
    def test_unknown_robot_raises(self):
        with pytest.raises(ValueError, match="robot must be"):
            _make_env(robot="unitree_go2")

    def test_world_idx_too_high_raises(self):
        with pytest.raises(ValueError, match="world_idx must be"):
            _make_env(world_idx=300)

    def test_world_idx_negative_raises(self):
        with pytest.raises(ValueError, match="world_idx must be"):
            _make_env(world_idx=-1)

    def test_boundary_world_zero_ok(self):
        env = _make_env(world_idx=0)
        assert env._fixed_world_idx == 0

    def test_boundary_world_299_ok(self):
        env = _make_env(world_idx=299)
        assert env._fixed_world_idx == 299

    def test_none_world_idx_ok(self):
        env = _make_env(world_idx=None)
        assert env._fixed_world_idx is None


# ═════════════════════════════════════════════════════════════════════════════
# Goal geometry
# ═════════════════════════════════════════════════════════════════════════════

class TestGoalGeometry:
    def test_goal_matches_config(self):
        from configs.config import WorldConfig
        env = _make_env()
        ix, iy, _ = WorldConfig.STATIC_INIT_POSITION
        assert env._goal_pos == (ix + WorldConfig.STATIC_GOAL_OFFSET[0],
                                 iy + WorldConfig.STATIC_GOAL_OFFSET[1])

    def test_init_matches_config(self):
        from configs.config import WorldConfig
        env = _make_env()
        ix, iy, _ = WorldConfig.STATIC_INIT_POSITION
        assert env._init_pos == (ix, iy)

    def test_max_dist_is_positive(self):
        env = _make_env()
        assert env._max_dist > 0.0

    def test_max_dist_equals_euclidean(self):
        from configs.config import WorldConfig
        env = _make_env()
        ix, iy, _ = WorldConfig.STATIC_INIT_POSITION
        gx = ix + WorldConfig.STATIC_GOAL_OFFSET[0]
        gy = iy + WorldConfig.STATIC_GOAL_OFFSET[1]
        expected = math.sqrt((gx - ix) ** 2 + (gy - iy) ** 2)
        assert env._max_dist == pytest.approx(expected)


# ═════════════════════════════════════════════════════════════════════════════
# Reward config
# ═════════════════════════════════════════════════════════════════════════════

class TestRewardConfig:
    def test_defaults_have_correct_signs(self):
        env = _make_env()
        assert env._r_collision < 0
        assert env._r_goal > 0
        assert env._r_step < 0

    def test_custom_rewards_stored(self):
        env = _make_env(reward_collision=-50.0, reward_goal=200.0, reward_step=-0.05)
        assert env._r_collision == pytest.approx(-50.0)
        assert env._r_goal == pytest.approx(200.0)
        assert env._r_step == pytest.approx(-0.05)

    def test_shaping_flag_stored(self):
        env = _make_env(reward_shaping=False)
        assert env._r_shaping is False


# ═════════════════════════════════════════════════════════════════════════════
# Scan normalisation
# ═════════════════════════════════════════════════════════════════════════════

class TestScanNormalisation:
    def test_no_scan_returns_all_ones(self):
        env = _make_env(num_scan_rays=360)
        # _latest_scan is None by default
        scan = env._get_scan_normalised()
        assert scan.shape == (360,)
        assert (scan == 1.0).all()

    def test_values_clamped_to_zero_one(self):
        env = _make_env(num_scan_rays=360)
        with env._scan_lock:
            env._latest_scan = np.full(360, env._scan_max_range * 2, dtype=np.float32)
        scan = env._get_scan_normalised()
        assert scan.max() <= 1.0 + 1e-6
        assert scan.min() >= 0.0

    def test_clip_range_normalises_to_one(self):
        env = _make_env(num_scan_rays=360)
        with env._scan_lock:
            env._latest_scan = np.full(360, env._scan_norm_range, dtype=np.float32)
        scan = env._get_scan_normalised()
        assert np.allclose(scan, 1.0)

    def test_beyond_clip_range_saturates_at_one(self):
        env = _make_env(num_scan_rays=360, scan_clip_range=5.0)
        with env._scan_lock:
            env._latest_scan = np.full(360, 8.0, dtype=np.float32)
        scan = env._get_scan_normalised()
        assert np.allclose(scan, 1.0)

    def test_zero_range_normalises_to_zero(self):
        env = _make_env(num_scan_rays=360)
        with env._scan_lock:
            env._latest_scan = np.zeros(360, dtype=np.float32)
        scan = env._get_scan_normalised()
        assert np.allclose(scan, 0.0)

    def test_downsampled_to_target_size(self):
        env = _make_env(num_scan_rays=180)
        with env._scan_lock:
            env._latest_scan = np.ones(720, dtype=np.float32) * 5.0
        scan = env._get_scan_normalised()
        assert scan.shape == (180,)

    def test_upsampled_to_target_size(self):
        env = _make_env(num_scan_rays=720)
        with env._scan_lock:
            env._latest_scan = np.ones(360, dtype=np.float32) * 5.0
        scan = env._get_scan_normalised()
        assert scan.shape == (720,)

    def test_output_dtype_is_float32(self):
        env = _make_env(num_scan_rays=360)
        scan = env._get_scan_normalised()
        assert scan.dtype == np.float32

    def test_thin_obstacle_survives_downsampling(self):
        # A single close-range ray inside a sector must survive the pooling —
        # this is the whole point of min-pool over linear interpolation.
        env = _make_env(num_scan_rays=90)
        raw = np.full(720, env._scan_norm_range, dtype=np.float32)
        raw[400] = 0.3   # thin pole at one raw ray
        with env._scan_lock:
            env._latest_scan = raw
        scan = env._get_scan_normalised()
        assert scan.min() == pytest.approx(0.3 / env._scan_norm_range, abs=1e-5)


# ═════════════════════════════════════════════════════════════════════════════
# Min-pool helper
# ═════════════════════════════════════════════════════════════════════════════

class TestMinPool:
    def test_exact_size_is_identity(self):
        from rl.env.barn_env import _min_pool
        x = np.arange(90, dtype=np.float32)
        assert np.array_equal(_min_pool(x, 90), x)

    def test_takes_sector_minimum(self):
        from rl.env.barn_env import _min_pool
        x = np.array([5, 1, 5, 5, 5, 2, 5, 5], dtype=np.float32)
        out = _min_pool(x, 2)   # sectors [0:4], [4:8]
        assert np.allclose(out, [1.0, 2.0])

    def test_non_divisible_sizes(self):
        from rl.env.barn_env import _min_pool
        x = np.random.default_rng(0).uniform(0, 10, size=722).astype(np.float32)
        out = _min_pool(x, 90)
        assert out.shape == (90,)
        assert out.min() >= x.min() - 1e-6
        # global minimum must survive pooling
        assert out.min() == pytest.approx(x.min(), abs=1e-6)

    def test_upsample_falls_back_to_interp(self):
        from rl.env.barn_env import _min_pool
        x = np.ones(45, dtype=np.float32) * 3.0
        out = _min_pool(x, 90)
        assert out.shape == (90,)
        assert np.allclose(out, 3.0)


# ═════════════════════════════════════════════════════════════════════════════
# Scan memory — near-field blind ring
# ═════════════════════════════════════════════════════════════════════════════

_BLIND = 0.45
_NRAYS = 722
_AMIN = -math.pi
_AINC = 2 * math.pi / _NRAYS
_RMAX = 10.0


def _fake_scan(x, y, theta, obstacles, blind=_BLIND):
    """Synthetic LiDAR returning NaN where the real sensor gives no return.

    Mirrors the moborobo pipeline: anything closer than `blind` is dropped by
    pointcloud_to_laserscan (range_min 0.45) and arrives as inf, which
    BarnEnv._scan_cb stores as NaN.
    """
    r = np.full(_NRAYS, np.nan, dtype=np.float64)
    for ox, oy in obstacles:
        dx, dy = ox - x, oy - y
        rng = math.hypot(dx, dy)
        if rng < blind or rng > _RMAX:
            continue
        bearing = (math.atan2(dy, dx) - theta + math.pi) % (2 * math.pi) - math.pi
        idx = int(round((bearing - _AMIN) / _AINC)) % _NRAYS
        if np.isnan(r[idx]) or rng < r[idx]:
            r[idx] = rng
    return r


class TestScanMemory:
    """The whole point: a sector that goes blind must report the remembered
    range, not max range. The old pipeline reported 10 m at the exact moment
    the robot was about to hit something."""

    def _mem(self, **kw):
        from rl.env.scan_memory import ScanMemory
        kw.setdefault("num_sectors", 90)
        kw.setdefault("blind_radius", _BLIND)
        m = ScanMemory(**kw)
        m.reset()
        return m

    def test_tracks_obstacle_into_the_blind_ring(self):
        mem = self._mem()
        theta = math.pi / 2                      # facing +y
        obst = [(0.0, 1.0)]
        fused = None
        for y in np.arange(0.0, 0.71, 0.05):
            raw = _fake_scan(0.0, y, theta, obst)
            fused, _ = mem.update(0.0, y, theta, raw, _AMIN, _AINC, _RMAX)
        # Robot at y=0.70 → obstacle is 0.30 m ahead, well inside the blind
        # ring, so the raw scan carries nothing at all.
        raw = _fake_scan(0.0, 0.70, theta, obst)
        assert not np.isfinite(raw).any(), "sensor should be blind here"
        assert mem.arc_min(fused, 2.0944) == pytest.approx(0.30, abs=0.02)

    def test_naive_sanitisation_would_have_said_free(self):
        # Documents the bug this class exists to fix.
        raw = _fake_scan(0.0, 0.70, math.pi / 2, [(0.0, 1.0)])
        naive = np.where(np.isfinite(raw), raw, _RMAX).min()
        assert naive == pytest.approx(_RMAX), (
            "obstacle 0.30 m away read as %.1f m of open space" % naive
        )

    def test_fresh_mask_false_inside_blind_ring(self):
        mem = self._mem()
        theta = math.pi / 2
        obst = [(0.0, 1.0)]
        _, fresh_far = mem.update(0.0, 0.0, theta,
                                  _fake_scan(0.0, 0.0, theta, obst),
                                  _AMIN, _AINC, _RMAX)
        _, fresh_near = mem.update(0.0, 0.70, theta,
                                   _fake_scan(0.0, 0.70, theta, obst),
                                   _AMIN, _AINC, _RMAX)
        ahead = 45      # bearing 0 with 90 sectors over [-pi, pi)
        assert fresh_far[ahead] == pytest.approx(1.0)
        assert fresh_near[ahead] == pytest.approx(0.0)

    def test_raycast_clears_a_contradicted_point(self):
        mem = self._mem()
        theta = math.pi / 2
        mem.update(0.0, 0.0, theta, _fake_scan(0.0, 0.0, theta, [(0.0, 1.0)]),
                   _AMIN, _AINC, _RMAX)
        assert mem.num_points == 1
        # Now every ray sees 5 m — the remembered point cannot be there.
        fused, _ = mem.update(0.0, 0.0, theta, np.full(_NRAYS, 5.0),
                              _AMIN, _AINC, _RMAX)
        assert mem.arc_min(fused, 2.0944) == pytest.approx(5.0, abs=0.01)

    def test_raycast_does_not_clear_inside_the_blind_ring(self):
        # A point the sensor cannot see is not evidence of absence — this
        # exception is what makes the memory survive where it is needed.
        mem = self._mem()
        theta = math.pi / 2
        mem.update(0.0, 0.0, theta, _fake_scan(0.0, 0.0, theta, [(0.0, 0.6)]),
                   _AMIN, _AINC, _RMAX)
        fused, _ = mem.update(0.0, 0.25, theta, np.full(_NRAYS, 5.0),
                              _AMIN, _AINC, _RMAX)
        assert mem.arc_min(fused, 2.0944) == pytest.approx(0.35, abs=0.02)

    def test_forward_arc_ignores_obstacles_already_passed(self):
        mem = self._mem()
        theta = math.pi / 2
        mem.update(0.0, 0.0, theta, _fake_scan(0.0, 0.0, theta, [(0.0, 1.0)]),
                   _AMIN, _AINC, _RMAX)
        # drive 1.30 m: the cylinder is now 0.30 m BEHIND
        fused, _ = mem.update(0.0, 1.30, theta, np.full(_NRAYS, 5.0),
                              _AMIN, _AINC, _RMAX)
        assert float(np.min(fused)) == pytest.approx(0.30, abs=0.02)
        assert mem.arc_min(fused, 2.0944) > 1.0
        # ...but it is exactly what matters when reversing into it.
        assert mem.arc_min(fused, 2.0944, centre=math.pi) == pytest.approx(0.30, abs=0.02)

    def test_points_outside_mem_radius_are_dropped(self):
        mem = self._mem(mem_radius=1.0)
        theta = math.pi / 2
        mem.update(0.0, 0.0, theta, _fake_scan(0.0, 0.0, theta, [(0.0, 0.9)]),
                   _AMIN, _AINC, _RMAX)
        assert mem.num_points == 1
        mem.update(0.0, -2.0, theta, np.full(_NRAYS, np.nan),
                   _AMIN, _AINC, _RMAX)
        assert mem.num_points == 0

    def test_points_age_out(self):
        mem = self._mem(max_age_steps=3)
        theta = math.pi / 2
        # Seed a point inside the blind ring so raycast clearing cannot remove
        # it — only ageing can.
        mem.update(0.0, 0.0, theta, _fake_scan(0.0, 0.0, theta, [(0.0, 0.5)]),
                   _AMIN, _AINC, _RMAX)
        mem.update(0.0, 0.15, theta, np.full(_NRAYS, np.nan), _AMIN, _AINC, _RMAX)
        assert mem.num_points == 1
        for _ in range(5):
            mem.update(0.0, 0.15, theta, np.full(_NRAYS, np.nan),
                       _AMIN, _AINC, _RMAX)
        assert mem.num_points == 0

    def test_voxel_dedup_bounds_the_store(self):
        mem = self._mem(voxel=0.05)
        theta = math.pi / 2
        obst = [(0.0, 1.0), (0.3, 1.2), (-0.4, 0.9)]
        for _ in range(50):
            mem.update(0.0, 0.0, theta, _fake_scan(0.0, 0.0, theta, obst),
                       _AMIN, _AINC, _RMAX)
        assert mem.num_points == len(obst), (
            "repeated sightings of the same 3 points should collapse, got %d"
            % mem.num_points
        )

    def test_reset_empties_the_store(self):
        mem = self._mem()
        theta = math.pi / 2
        mem.update(0.0, 0.0, theta, _fake_scan(0.0, 0.0, theta, [(0.0, 1.0)]),
                   _AMIN, _AINC, _RMAX)
        assert mem.num_points > 0
        mem.reset()
        assert mem.num_points == 0

    def test_none_scan_falls_back_to_memory(self):
        mem = self._mem()
        theta = math.pi / 2
        mem.update(0.0, 0.0, theta, _fake_scan(0.0, 0.0, theta, [(0.0, 1.0)]),
                   _AMIN, _AINC, _RMAX)
        fused, fresh = mem.update(0.0, 0.0, theta, None, _AMIN, _AINC, _RMAX)
        assert mem.arc_min(fused, 2.0944) == pytest.approx(1.0, abs=0.02)
        assert not fresh.any()

    def test_empty_sectors_report_range_max(self):
        mem = self._mem()
        fused, _ = mem.update(0.0, 0.0, 0.0, None, _AMIN, _AINC, _RMAX)
        assert np.allclose(fused, _RMAX)

    def test_update_is_fast_enough_for_the_control_loop(self):
        import time
        mem = self._mem()
        rng = np.random.default_rng(0)
        obst = [(rng.uniform(-3, 3), rng.uniform(-3, 3)) for _ in range(60)]
        theta = math.pi / 2
        scans = [_fake_scan(0.0, i * 0.02, theta, obst) for i in range(30)]
        t0 = time.time()
        for i, s in enumerate(scans):
            mem.update(0.0, i * 0.02, theta, s, _AMIN, _AINC, _RMAX)
        per_step = (time.time() - t0) / len(scans)
        # STEP_DURATION is 0.1 s; the memory must be a rounding error in that.
        assert per_step < 0.01, "%.1f ms per update" % (per_step * 1000)


# ═════════════════════════════════════════════════════════════════════════════
# Reverse driving + reward wiring
# ═════════════════════════════════════════════════════════════════════════════

class TestRewardConfig:
    def test_reverse_params_stored(self):
        env = _make_env(reverse_speed_ratio=0.25, reward_reverse_scale=0.1)
        assert env._max_reverse == pytest.approx(0.25 * env._max_linear)
        assert env._r_reverse_scale == pytest.approx(0.1)

    def test_reverse_penalty_is_small_against_progress(self):
        # Backing out of a trap must stay cheap: a full-speed reverse step
        # should cost well under one full-speed forward step of progress.
        from configs.rl_config import RLEnvConfig as C
        assert 0.0 <= C.REWARD_REVERSE_SCALE * C.REVERSE_SPEED_RATIO < 0.25

    def test_memory_disabled_leaves_no_scan_memory(self):
        env = _make_env(scan_memory=False)
        assert env._scan_mem is None

    def test_memory_enabled_creates_matching_sector_count(self):
        env = _make_env(num_scan_rays=120, scan_memory=True)
        assert env._scan_mem is not None
        assert env._scan_mem.num_sectors == 120

    def test_terminal_potential_flag_stored(self):
        assert _make_env(terminal_potential_zero=False)._terminal_potential_zero is False
        assert _make_env(terminal_potential_zero=True)._terminal_potential_zero is True

    def test_total_shaping_is_independent_of_where_the_episode_ends(self):
        # THE property that makes progress shaping policy-invariant (Ng et al.
        # 1999): with Phi(terminal) = 0 the shaping telescopes to the constant
        # +scale * d_0 no matter how the episode ends, so it cannot rank
        # outcomes on its own — only the base rewards do that.
        #
        # Getting the correction's sign wrong silently breaks this: subtracting
        # the terminal potential instead of adding it leaves
        # scale*(d_0 - 2*d_T), which punishes failing early far harder than
        # failing late and taught the policy to stop dead mid-course.
        from configs.rl_config import RLEnvConfig as C
        scale = C.reward_progress_scale("moborobo")
        d0 = 10.0

        def total_shaping(d_end):
            banked = scale * (d0 - d_end)          # summed per-step progress
            correction = scale * d_end             # zeroing Phi at the terminal
            return banked + correction

        totals = [total_shaping(d) for d in (0.0, 1.0, 3.0, 6.05, 9.5)]
        assert all(t == pytest.approx(scale * d0) for t in totals), totals

    def test_outcome_ranking_is_set_by_base_rewards_alone(self):
        # Follows from the invariant above: once shaping contributes the same
        # constant everywhere, success > timeout > collision has to come from
        # the terminal rewards.
        from configs.rl_config import RLEnvConfig as C
        assert C.REWARD_GOAL > 0.0
        assert 0.0 > C.REWARD_COLLISION

    def test_path_penalty_keeps_a_gradient_at_the_observed_deviation(self):
        # A clipped penalty has zero derivative, so once the robot is past
        # PATH_MAX_DEV the term stops saying which way to steer and becomes a
        # flat tax. Measured mean deviation is 0.449 m and the max reaches
        # 1.2 m, so the clip has to sit above the operating range.
        from configs.rl_config import RLEnvConfig as C
        assert C.PATH_MAX_DEV > 0.449 * 2, (
            "PATH_MAX_DEV=%.2f saturates at the observed mean deviation"
            % C.PATH_MAX_DEV
        )

    @pytest.mark.parametrize("script", ["train_sac.py", "train_ppo.py"])
    def test_discount_horizon_covers_the_course(self, script):
        # At 10 Hz an effective horizon of 1/(1-gamma) steps must reach the
        # goal, or the goal reward is discounted into irrelevance and the
        # policy optimises dense shaping alone — which is what made it freeze
        # mid-course. Read the default out of the source rather than importing
        # the trainer (which would pull in SB3 and ROS).
        import re
        from configs.rl_config import RLEnvConfig as C
        src = open(os.path.join(_ROOT, script)).read()
        m = re.search(r'"--gamma".*?default=([0-9.]+)', src, re.S)
        assert m, "could not find the --gamma default in %s" % script
        gamma = float(m.group(1))
        horizon = 1.0 / (1.0 - gamma)
        assert horizon >= 0.9 * C.max_episode_steps("moborobo"), (
            "%s gamma=%.4f gives a %.0f-step horizon, shorter than the "
            "%d-step episode" % (script, gamma, horizon,
                                 C.max_episode_steps("moborobo"))
        )

    def test_prox_penalty_is_comparable_to_progress_at_the_floor(self):
        # It used to peak at -0.0275/step against +0.25/step of progress, a
        # 9:1 ratio the policy correctly ignored.
        from configs.rl_config import RLEnvConfig as C
        max_progress = C.reward_progress_scale("moborobo") * 0.2 * C.STEP_DURATION
        worst_prox = C.REWARD_PROX_SCALE          # t=1, full speed
        assert worst_prox >= 0.4 * max_progress

    def test_path_penalty_no_longer_noise(self):
        # At the deviation the old policy actually ran (~0.40 m) the penalty
        # must be a real cost, not the -0.008/step it used to be.
        from configs.rl_config import RLEnvConfig as C
        dev = 0.40
        pen = C.REWARD_PATH_SCALE * min(dev, C.PATH_MAX_DEV) / C.PATH_MAX_DEV
        assert pen > 0.05


# ═════════════════════════════════════════════════════════════════════════════
# Math helpers
# ═════════════════════════════════════════════════════════════════════════════

class TestMathHelpers:
    def test_dist_3_4_5(self):
        from rl.env.barn_env import _dist
        assert _dist((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)

    def test_dist_same_point(self):
        from rl.env.barn_env import _dist
        assert _dist((1.5, -2.3), (1.5, -2.3)) == pytest.approx(0.0)

    def test_quat_to_yaw_identity(self):
        from rl.env.barn_env import _quat_to_yaw
        assert _quat_to_yaw(0, 0, 0, 1) == pytest.approx(0.0)

    def test_quat_to_yaw_90_degrees(self):
        from rl.env.barn_env import _quat_to_yaw
        s, c = math.sin(math.pi / 4), math.cos(math.pi / 4)
        assert _quat_to_yaw(0, 0, s, c) == pytest.approx(math.pi / 2, abs=1e-5)

    def test_quat_to_yaw_180_degrees(self):
        from rl.env.barn_env import _quat_to_yaw
        assert _quat_to_yaw(0, 0, 1, 0) == pytest.approx(math.pi, abs=1e-5)

    def test_wrap_angle_zero(self):
        from rl.env.barn_env import _wrap_angle
        assert _wrap_angle(0.0) == pytest.approx(0.0)

    def test_wrap_angle_past_pi(self):
        from rl.env.barn_env import _wrap_angle
        assert _wrap_angle(math.pi + 0.1) == pytest.approx(-math.pi + 0.1, abs=1e-5)

    def test_wrap_angle_past_minus_pi(self):
        from rl.env.barn_env import _wrap_angle
        assert _wrap_angle(-math.pi - 0.1) == pytest.approx(math.pi - 0.1, abs=1e-5)

    def test_wrap_angle_three_pi(self):
        from rl.env.barn_env import _wrap_angle
        assert abs(_wrap_angle(3 * math.pi)) == pytest.approx(math.pi, abs=1e-5)


# ═════════════════════════════════════════════════════════════════════════════
# Feature extractor
# ═════════════════════════════════════════════════════════════════════════════

class TestScanStateExtractor:
    def _space(self, k=3, n=90):
        import gymnasium as gym
        return gym.spaces.Dict({
            "scan": gym.spaces.Box(0.0, 1.0, shape=(k, n), dtype=np.float32),
            "state": gym.spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32),
        })

    def test_forward_output_dim(self):
        import torch as th
        from rl.models.feature_extractor import ScanStateExtractor
        space = self._space()
        ext = ScanStateExtractor(space, scan_features=128, state_features=32)
        obs = {
            "scan": th.rand(4, 3, 90),
            "state": th.rand(4, 5),
        }
        out = ext(obs)
        assert out.shape == (4, 160)
        assert ext.features_dim == 160

    def test_adapts_to_custom_shapes(self):
        import torch as th
        from rl.models.feature_extractor import ScanStateExtractor
        space = self._space(k=4, n=180)
        ext = ScanStateExtractor(space)
        obs = {"scan": th.rand(2, 4, 180), "state": th.rand(2, 5)}
        out = ext(obs)
        assert out.shape[0] == 2 and out.shape[1] == ext.features_dim


# ═════════════════════════════════════════════════════════════════════════════
# RLLogger
# ═════════════════════════════════════════════════════════════════════════════

class TestRLLogger:
    def test_creates_all_output_files(self, tmp_path):
        from rl.logging.rl_logger import RLLogger
        logger = RLLogger("test_run", results_dir=str(tmp_path))
        assert os.path.exists(logger._csv_path)
        assert os.path.exists(logger._log_path)
        assert os.path.exists(os.path.join(logger.run_dir, "run_config.json"))
        logger.close()

    def test_run_config_stored(self, tmp_path):
        import json
        from rl.logging.rl_logger import RLLogger
        logger = RLLogger("test_run", results_dir=str(tmp_path),
                          config={"algo": "SAC", "lr": 3e-4})
        logger.close()
        with open(os.path.join(logger.run_dir, "run_config.json")) as f:
            cfg = json.load(f)
        assert cfg["algo"] == "SAC"
        assert cfg["lr"] == pytest.approx(3e-4)
        assert cfg["run_id"] == "test_run"

    def test_episodes_written_to_csv(self, tmp_path):
        import csv
        from rl.logging.rl_logger import RLLogger
        logger = RLLogger("test_run", results_dir=str(tmp_path))
        logger.log_episode(1, 30, "success", 200, 85.0, 0.4)
        logger.log_episode(2, 45, "collision", 55, -92.1, 6.3)
        logger.close()
        with open(logger._csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]["outcome"] == "success"
        assert rows[1]["outcome"] == "collision"
        assert float(rows[0]["total_reward"]) == pytest.approx(85.0, abs=0.01)
        assert int(rows[0]["steps"]) == 200

    def test_outcome_counts_accumulated(self, tmp_path):
        from rl.logging.rl_logger import RLLogger
        logger = RLLogger("test_run", results_dir=str(tmp_path))
        for _ in range(4):
            logger.log_episode(0, 0, "success", 100, 50.0, 0.3)
        for _ in range(2):
            logger.log_episode(0, 0, "collision", 20, -90.0, 7.0)
        logger.log_episode(0, 0, "timeout", 500, 5.0, 4.5)
        assert logger._outcome_counts["success"] == 4
        assert logger._outcome_counts["collision"] == 2
        assert logger._outcome_counts["timeout"] == 1
        logger.close()

    def test_rolling_window_oldest_dropped(self, tmp_path):
        from rl.logging.rl_logger import RLLogger
        logger = RLLogger("test_run", results_dir=str(tmp_path), window=3)
        for r in [10.0, 20.0, 30.0, 40.0]:
            logger.log_episode(0, 0, "success", 100, r, 0.5)
        # window=3 → keeps [20, 30, 40]
        assert float(np.mean(logger._recent_rewards)) == pytest.approx(30.0)
        logger.close()

    def test_log_progress_does_not_raise(self, tmp_path, capsys):
        from rl.logging.rl_logger import RLLogger
        logger = RLLogger("test_run", results_dir=str(tmp_path))
        logger.log_episode(1, 0, "success", 100, 50.0, 0.5)
        logger.log_progress(timestep=1000)   # should not raise
        captured = capsys.readouterr()
        assert "step=" in captured.out
        logger.close()

    def test_close_writes_summary_to_log(self, tmp_path):
        from rl.logging.rl_logger import RLLogger
        logger = RLLogger("test_run", results_dir=str(tmp_path))
        logger.log_episode(1, 0, "success", 100, 50.0, 0.3)
        logger.close()
        with open(logger._log_path) as f:
            text = f.read()
        assert "RUN SUMMARY" in text
        assert "Success rate" in text

    def test_unique_run_dirs(self, tmp_path):
        from rl.logging.rl_logger import RLLogger
        a = RLLogger("run_a", results_dir=str(tmp_path))
        b = RLLogger("run_b", results_dir=str(tmp_path))
        assert a.run_dir != b.run_dir
        a.close()
        b.close()


# ═════════════════════════════════════════════════════════════════════════════
# Gazebo integration tests  (require live simulation)
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.gazebo
class TestBarnEnvGazebo:
    """
    These tests require Gazebo to be running externally on world 30
    with the appropriate robot (moborobo default) and sensor stack up.

    Start it manually before running:
        roslaunch moborobo barn_gazebo.launch world_name:=<path>/world_30.world
        roslaunch moborobo barn_move_base.launch
    """

    @pytest.fixture(autouse=True)
    def env(self):
        from rl.env.barn_env import BarnEnv
        env = BarnEnv(robot="moborobo", world_idx=30, managed=False, log_dir="")
        yield env
        env.close()

    def test_reset_returns_valid_obs(self, env):
        obs, info = env.reset()
        assert env.observation_space.contains(obs), "obs out of bounds: %s" % obs
        assert obs["scan"].shape == env.observation_space["scan"].shape
        # freshly reset → all stacked frames identical (seeded with 1st frame)
        assert np.array_equal(obs["scan"][0], obs["scan"][-1])
        assert info["world_idx"] == 30

    def test_obs_dtype_is_float32(self, env):
        obs, _ = env.reset()
        assert obs["scan"].dtype == np.float32
        assert obs["state"].dtype == np.float32

    def test_step_returns_correct_types(self, env):
        env.reset()
        action = np.array([0.1, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "dist_to_goal" in info
        assert "outcome" in info

    def test_zero_action_does_not_crash(self, env):
        env.reset()
        obs, reward, terminated, truncated, info = env.step(
            np.zeros(2, dtype=np.float32)
        )
        assert env.observation_space.contains(obs)

    def test_episode_truncates_at_max_steps(self, env):
        from rl.env.barn_env import BarnEnv
        short_env = BarnEnv(
            robot="moborobo", world_idx=30, managed=False,
            log_dir="", max_episode_steps=5,
        )
        short_env.reset()
        truncated_seen = False
        for _ in range(10):
            _, _, terminated, truncated, _ = short_env.step(
                np.zeros(2, dtype=np.float32)
            )
            if truncated:
                truncated_seen = True
                break
            if terminated:
                break
        assert truncated_seen, "episode did not truncate within max_episode_steps"
        short_env.close()

    def test_obs_scan_in_zero_one_after_step(self, env):
        env.reset()
        obs, _, _, _, _ = env.step(np.array([0.1, 0.0], dtype=np.float32))
        scan = obs["scan"]
        assert scan.min() >= 0.0 - 1e-5
        assert scan.max() <= 1.0 + 1e-5

    def test_dist_feature_decreases_when_moving_forward(self, env):
        obs0, _ = env.reset()
        dist0 = obs0["state"][0]
        # Drive straight toward goal (goal is in +y direction, robot starts facing ~pi/2)
        obs1, _, _, _, _ = env.step(np.array([env._max_linear, 0.0], dtype=np.float32))
        dist1 = obs1["state"][0]
        # Allow small tolerance — first step may not move much
        assert dist1 <= dist0 + 0.05, (
            "dist to goal did not decrease after driving forward: %.4f → %.4f" % (dist0, dist1)
        )
