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
    def test_default_shape_is_scan_plus_two(self):
        env = _make_env(num_scan_rays=360)
        assert env.observation_space.shape == (362,)

    def test_custom_scan_rays(self):
        env = _make_env(num_scan_rays=180)
        assert env.observation_space.shape == (182,)

    def test_scan_portion_bounds_zero_to_one(self):
        env = _make_env(num_scan_rays=360)
        assert (env.observation_space.low[:360] == 0.0).all()
        assert (env.observation_space.high[:360] == 1.0).all()

    def test_dist_feature_bounds(self):
        env = _make_env()
        n = env.num_scan_rays
        assert env.observation_space.low[n] == 0.0
        assert env.observation_space.high[n] == 1.0

    def test_angle_feature_bounds(self):
        env = _make_env()
        n = env.num_scan_rays
        assert env.observation_space.low[n + 1] == -1.0
        assert env.observation_space.high[n + 1] == 1.0

    def test_dtype_is_float32(self):
        env = _make_env()
        assert env.observation_space.dtype == np.float32


# ═════════════════════════════════════════════════════════════════════════════
# Action space
# ═════════════════════════════════════════════════════════════════════════════

class TestActionSpace:
    def test_moborobo_linear_range(self):
        from configs.config import MoboroboConfig
        env = _make_env(robot="moborobo")
        assert env.action_space.low[0] == pytest.approx(0.0)
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

    def test_forward_only_lower_bound(self):
        # Linear velocity lower bound is 0 (robot cannot reverse)
        env = _make_env()
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

    def test_max_range_normalises_to_one(self):
        env = _make_env(num_scan_rays=360)
        with env._scan_lock:
            env._latest_scan = np.full(360, env._scan_max_range, dtype=np.float32)
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
        assert obs.shape == env.observation_space.shape
        assert env.observation_space.contains(obs), "obs out of bounds: %s" % obs
        assert info["world_idx"] == 30

    def test_obs_dtype_is_float32(self, env):
        obs, _ = env.reset()
        assert obs.dtype == np.float32

    def test_step_returns_correct_types(self, env):
        env.reset()
        action = np.array([0.1, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
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
        assert obs.shape == env.observation_space.shape

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
        scan = obs[:env.num_scan_rays]
        assert scan.min() >= 0.0 - 1e-5
        assert scan.max() <= 1.0 + 1e-5

    def test_dist_feature_decreases_when_moving_forward(self, env):
        obs0, _ = env.reset()
        dist0 = obs0[env.num_scan_rays]
        # Drive straight toward goal (goal is in +y direction, robot starts facing ~pi/2)
        obs1, _, _, _, _ = env.step(np.array([env._max_linear, 0.0], dtype=np.float32))
        dist1 = obs1[env.num_scan_rays]
        # Allow small tolerance — first step may not move much
        assert dist1 <= dist0 + 0.05, (
            "dist to goal did not decrease after driving forward: %.4f → %.4f" % (dist0, dist1)
        )
