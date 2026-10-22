"""
Several robots training in one Gazebo world, as an SB3 VecEnv.

The robots are ghosts to each other: each has its own collision bit, so they
drive through one another while still hitting the obstacles, and none has
rendered geometry, so no lidar sees another robot (see barn_rl_robot.launch).
Every robot therefore experiences exactly the single-robot task, and one
Gazebo instance produces N transitions per physics step.

Each robot is a BarnEnv in slave mode (`managed=False`, its own namespace), so
observation, reward and termination are the very same code as the single-robot
env. This class owns what is shared: the Gazebo processes, the physics clock
and the world schedule.

    venv = MultiRobotBarnVecEnv(n_robots=4, worlds=[0, 6, 8], robot="moborobo")
    obs = venv.reset()
    obs, rewards, dones, infos = venv.step(actions)   # actions: (4, 2)

One step:
    1. every robot records its action
    2. unpause -> publish all commands -> step_duration -> pause   (once)
    3. every robot observes, scores, terminates
    4. robots whose episode ended are teleported to the start while paused,
       then the simulation runs briefly with everyone stopped until each of
       them has a scan taken from the start pose

Collisions are tested geometrically (rl/env/geometry.py) against the cylinder
positions read from Gazebo when the world comes up.
"""

import os
import signal
import socket
import subprocess
import time
from collections import OrderedDict
from copy import deepcopy
from os.path import join
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import rospy
from gazebo_msgs.msg import ModelStates
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from configs.config import (
    GazeboConfig, MoboroboConfig, Paths, SimConfig, WorldConfig, apply_env,
)
from configs.rl_config import RLEnvConfig, RLLogConfig
from gazebo.gazebo_simulation import GazeboSimulation
from rl.curriculum import WorldCycler
from rl.env.barn_env import (
    BarnEnv, _GAZEBO_SVC_TIMEOUT, _ROSCORE_START_TIMEOUT, _dist,
    _gazebo_process_env, _nvidia_offload_wanted,
)
from rl.env.geometry import box_hits_circles

_MAX_ROBOTS = 16            # collide_bitmask is 16 bits wide
_SPAWN_TIMEOUT = 60.0       # s for all robots to appear in Gazebo
_LAUNCH_ATTEMPTS = 3        # tries to bring a world up before giving up
_FRESH_SCAN_TIMEOUT = 5.0   # s of wall time to get a post-teleport scan
_COLLISION_MARGIN = 0.01    # m — report contact just before physics makes it
_CYLINDER_PREFIX = "unit_cylinder"


class MultiRobotBarnVecEnv(VecEnv):
    """N ghost robots in one BARN world.

    Parameters
    ----------
    n_robots : int
        Robots sharing the world (1-16).
    worlds : list[int]
        World indices to cycle through.
    episodes_per_world : int
        Episodes *per robot* before the world changes; the world is switched
        once n_robots * episodes_per_world episodes have finished on it.
    seed : int | None
        Seeds the shuffled world order (see rl.curriculum.WorldCycler).
    robot : str
        Only "moborobo" has the namespaced description this needs.
    gui : bool
        Open the Gazebo GUI. The robots themselves stay invisible in it.
    lidar_profile : dict | None
        Overrides for the RS-16 vertical sampling, passed to
        barn_rl_robot.launch (keys: lidar_v_samples, lidar_v_min, lidar_v_max,
        lidar_max_range). None uses RLEnvConfig.MULTI_LIDAR_PROFILE.
    log_dir : str | None
        Directory for episodes.csv and the Gazebo logs ("" disables the CSV).
    **env_kwargs
        Forwarded to every BarnEnv (rewards, scan settings, ...).
    """

    def __init__(
        self,
        n_robots: int,
        worlds: Sequence[int],
        episodes_per_world: int = 5,
        robot: str = "moborobo",
        gui: bool = False,
        log_dir: Optional[str] = None,
        lidar_profile: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        **env_kwargs,
    ):
        if robot != "moborobo":
            raise ValueError("multi-robot training is only set up for moborobo")
        if not 1 <= n_robots <= _MAX_ROBOTS:
            raise ValueError("n_robots must be 1-%d, got %r" % (_MAX_ROBOTS, n_robots))

        self.robot = robot
        self.gui = gui
        self._cycler = WorldCycler(list(worlds), n_robots * episodes_per_world, seed=seed)
        self._lidar_profile = dict(
            RLEnvConfig.MULTI_LIDAR_PROFILE if lidar_profile is None else lidar_profile)

        if log_dir is None:
            self._log_dir: Optional[str] = RLLogConfig.LOG_DIR
        else:
            self._log_dir = log_dir or None
        self._proc_log_dir = self._log_dir or join(Paths.RESULTS_DIR, "rl", "logs")
        os.makedirs(self._proc_log_dir, exist_ok=True)

        half_len, half_wid = (0.5 * d for d in MoboroboConfig.COLLISION_BOX)
        self._half_len, self._half_wid = half_len, half_wid
        self._cylinders = np.zeros((0, 2), dtype=np.float64)

        self.envs: List[BarnEnv] = [
            BarnEnv(
                robot=robot,
                world_idx=self._cycler.current,
                managed=False,
                log_dir="",                      # this class writes the CSV
                ns="r%d" % i,
                model_name="%s_%d" % (MoboroboConfig.MODEL_NAME, i),
                collision_fn=self._collides,
                **env_kwargs,
            )
            for i in range(n_robots)
        ]
        super().__init__(n_robots, self.envs[0].observation_space,
                         self.envs[0].action_space)

        self._actions: Optional[np.ndarray] = None
        self._episode = 0
        self._roscore_proc: Optional[subprocess.Popen] = None
        self._procs: List[subprocess.Popen] = []
        self._nvidia_offload = _nvidia_offload_wanted()
        self._world_up = False
        self._closed = False

    # ──────────────────────────────────────────────────────────────────────────
    # VecEnv API
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self):
        if not self._world_up:
            self._launch_world(self._cycler.current)
        obs = self._reset_robots(range(self.num_envs))
        self._reset_seeds()
        self._reset_options()
        return self._stack(obs)

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions)

    def step_wait(self):
        sim = self.envs[0]._gazebo_sim
        for env, action in zip(self.envs, self._actions):
            env._apply_action(action)

        sim.unpause()
        for env in self.envs:
            env._publish_action()
        time.sleep(self.envs[0].step_duration)
        sim.pause()

        obs: List[Any] = [None] * self.num_envs
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]
        world_used_up = False

        for i, env in enumerate(self.envs):
            obs[i], rewards[i], terminated, truncated, infos[i] = env._finish_step()
            dones[i] = terminated or truncated
            infos[i]["TimeLimit.truncated"] = truncated and not terminated
            if dones[i]:
                infos[i]["terminal_observation"] = obs[i]
                self._log_episode(env, infos[i])
                world_used_up |= self._cycler.episode_finished()

        if world_used_up:
            # Everyone leaves this world. Episodes still in progress are cut
            # short as truncations, so the critic bootstraps from them rather
            # than reading the cut as a terminal state.
            for i, env in enumerate(self.envs):
                if not dones[i]:
                    dones[i] = True
                    infos[i]["TimeLimit.truncated"] = True
                    infos[i]["terminal_observation"] = obs[i]
                    infos[i]["outcome"] = "world_change"
            self._launch_world(self._cycler.advance())
            obs = self._reset_robots(range(self.num_envs))
        elif dones.any():
            done_idx = [i for i in range(self.num_envs) if dones[i]]
            for i, o in zip(done_idx, self._reset_robots(done_idx)):
                obs[i] = o

        return self._stack(obs), rewards, dones, deepcopy(infos)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for env in self.envs:
            try:
                env._pub_cmd(0.0, 0.0)
            except Exception:
                pass
        self._stop_world()
        _stop_trees([self._roscore_proc])
        self._roscore_proc = None

    def get_attr(self, attr_name: str, indices=None) -> List[Any]:
        return [getattr(self.envs[i], attr_name) for i in self._indices(indices)]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        for i in self._indices(indices):
            setattr(self.envs[i], attr_name, value)

    def env_method(self, method_name: str, *args, indices=None, **kwargs) -> List[Any]:
        # The world schedule is shared, so it is answered here once per robot.
        if method_name == "set_world_list":
            self.set_world_list(*args, **kwargs)
            return [None for _ in self._indices(indices)]
        return [getattr(self.envs[i], method_name)(*args, **kwargs)
                for i in self._indices(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None) -> List[bool]:
        return [False for _ in self._indices(indices)]

    def set_world_list(self, worlds: Sequence[int]) -> None:
        """Swap the world pool (curriculum phase change)."""
        self._cycler.set_world_list(list(worlds))

    def _indices(self, indices) -> List[int]:
        if indices is None:
            return list(range(self.num_envs))
        if isinstance(indices, int):
            return [indices]
        return list(indices)

    def _stack(self, obs_list: List[Dict[str, np.ndarray]]) -> "OrderedDict[str, np.ndarray]":
        return OrderedDict(
            (key, np.stack([o[key] for o in obs_list]))
            for key in self.observation_space.spaces
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Episode boundaries
    # ──────────────────────────────────────────────────────────────────────────

    def _reset_robots(self, indices) -> List[Dict[str, np.ndarray]]:
        """Send the given robots back to the start and return their first
        observations. Physics is left paused."""
        indices = list(indices)
        sim = self.envs[0]._gazebo_sim
        sim.pause()

        for i in indices:
            env = self.envs[i]
            env._reset_episode_state()
            env._set_world(self._cycler.current)
            self._teleport_to_start(env)
        t_teleport = rospy.get_time()

        # A scan taken before the teleport shows the old surroundings. Run the
        # simulation with every robot stopped until each reset robot has one
        # from its new pose; the others simply wait in place.
        for env in self.envs:
            env._pub_cmd(0.0, 0.0)
        sim.unpause()
        deadline = time.time() + _FRESH_SCAN_TIMEOUT
        while time.time() < deadline:
            if all(self.envs[i]._has_scan(after_stamp=t_teleport) for i in indices):
                break
            time.sleep(0.005)
        else:
            sim.pause()
            late = [self.envs[i]._ns for i in indices
                    if not self.envs[i]._has_scan(after_stamp=t_teleport)]
            raise TimeoutError(
                "No fresh LaserScan from %s within %.0f s of a reset. Check the "
                "Gazebo and robot logs in %s."
                % (", ".join(late), _FRESH_SCAN_TIMEOUT, self._proc_log_dir))
        sim.pause()

        out = []
        for i in indices:
            obs, self.reset_infos[i] = self.envs[i]._first_obs()
            out.append(obs)
        return out

    def _teleport_to_start(self, env: BarnEnv):
        """Place a robot on the start pose. Works while physics is paused."""
        for _ in range(5):
            env._gazebo_sim.reset()
            pos = env._gazebo_sim.get_model_state().pose.position
            if _dist(env._init_pos, (pos.x, pos.y)) <= GazeboConfig.POSITION_TOL:
                return
        raise RuntimeError("Could not move %s to the start pose." % env._model_name)

    def _collides(self, x: float, y: float, yaw: float) -> bool:
        return box_hits_circles(
            x, y, yaw, self._cylinders, self._half_len, self._half_wid,
            WorldConfig.OBSTACLE_RADIUS, margin=_COLLISION_MARGIN,
        )

    def _log_episode(self, env: BarnEnv, info: Dict[str, Any]):
        """One row per finished episode, same columns as BarnEnv's CSV with
        the robot appended."""
        self._episode += 1
        if not self._log_dir:
            return
        log_file = join(self._log_dir, "episodes.csv")
        write_header = not os.path.exists(log_file)
        with open(log_file, "a") as f:
            if write_header:
                f.write("episode,world_idx,outcome,steps,total_reward,"
                        "final_dist_to_goal,robot\n")
            f.write("%d,%d,%s,%d,%.4f,%.4f,%s\n" % (
                self._episode, info["world_idx"], info["outcome"], info["step"],
                info["episode_reward"], info["dist_to_goal"], env._ns,
            ))

    # ──────────────────────────────────────────────────────────────────────────
    # Process management
    # ──────────────────────────────────────────────────────────────────────────

    def _launch_world(self, world_idx: int):
        """(Re)start Gazebo on `world_idx` with all robots in it, retrying
        when the world or a robot does not come up.

        A launch occasionally loses a node: in `multi_v5` r0's
        pointcloud_to_laserscan exited cleanly 0.7 s after starting, on the
        11th world change of a run (v4 made 106 without it), and the whole run
        died on the timeout. The next launch of the same world is unaffected,
        so a failed one is torn down and tried again.
        """
        for attempt in range(1, _LAUNCH_ATTEMPTS + 1):
            try:
                self._launch_world_once(world_idx)
                return
            except TimeoutError as exc:
                if attempt == _LAUNCH_ATTEMPTS:
                    raise
                print("\n    [setup] %s\n    [setup] Relaunching world %d (attempt %d/%d)"
                      % (exc, world_idx, attempt + 1, _LAUNCH_ATTEMPTS), flush=True)

    def _launch_world_once(self, world_idx: int):
        self._ensure_roscore()
        self._stop_world()
        # Whatever is still running now is not ours. Two Gazebo servers would
        # fight over the same master port, and killing by name is how the
        # single-robot env takes down unrelated runs, so stop and say so.
        deadline = time.time() + 10.0
        stray = _live_gzservers()
        while stray and time.time() < deadline:
            time.sleep(0.5)
            stray = _live_gzservers()
        if stray:
            raise RuntimeError(
                "Another Gazebo server is running (pid %s). Stop it before "
                "starting multi-robot training." % ", ".join(map(str, stray)))

        SimConfig.ROBOT = self.robot
        apply_env(robot=self.robot)
        if os.path.isdir(Paths.ROBOTS_DIR):
            pkg = os.environ.get("ROS_PACKAGE_PATH", "")
            if Paths.ROBOTS_DIR not in pkg:
                os.environ["ROS_PACKAGE_PATH"] = Paths.ROBOTS_DIR + (":" + pkg if pkg else "")

        world_file = WorldConfig.world_file(world_idx)
        if not os.path.exists(world_file):
            raise FileNotFoundError("World file not found: %s" % world_file)

        launch_dir = join(Paths.MOBOROBO_PKG, "launch")
        gz_log_path = join(self._proc_log_dir, "gazebo_world%d.log" % world_idx)
        print("    [setup] Launching Gazebo (world %d, %d robots)%s... log=%s"
              % (world_idx, self.num_envs,
                 " on NVIDIA" if self._nvidia_offload else "", gz_log_path), flush=True)
        self._procs.append(self._popen(
            ["roslaunch", join(launch_dir, "barn_rl_world.launch"),
             "world_name:=" + world_file,
             "gui:=" + ("true" if self.gui else "false"),
             "headless:=" + ("false" if self.gui else "true")],
            gz_log_path, env=_gazebo_process_env(self._nvidia_offload),
        ))
        self._wait_for_services(gz_log_path)

        profile = ["%s:=%s" % kv for kv in self._lidar_profile.items()]
        for i, env in enumerate(self.envs):
            self._procs.append(self._popen(
                ["roslaunch", join(launch_dir, "barn_rl_robot.launch"),
                 "ns:=" + env._ns,
                 "model:=" + env._model_name,
                 "collide_bitmask:=%d" % (1 << i),
                 # Never rendered, GUI or not: a visible robot is an obstacle
                 # in every other robot's lidar.
                 "visual:=false"] + profile,
                join(self._proc_log_dir, "robot_%s_world%d.log" % (env._ns, world_idx)),
            ))

        for env in self.envs:
            env._connect_ros()
            env._gazebo_sim = GazeboSimulation(
                init_position=list(WorldConfig.STATIC_INIT_POSITION),
                model_name=env._model_name,
            )
            with env._scan_lock:
                env._latest_scan = None
        self._wait_for_robots()
        self._cylinders = self._read_cylinders()
        self._world_up = True

    def _stop_world(self):
        _stop_trees(self._procs)
        self._procs = []
        self._world_up = False
        for env in self.envs:
            env._gazebo_sim = None

    def _popen(self, cmd: List[str], log_path: str, env=None) -> subprocess.Popen:
        # Own session, so a Ctrl-C in the terminal reaches only this script
        # and the shutdown order stays ours (see _stop_trees).
        return subprocess.Popen(cmd, stdout=open(log_path, "w"),
                                stderr=subprocess.STDOUT, env=env,
                                start_new_session=True)

    def _ensure_roscore(self):
        if self._roscore_proc is not None and self._roscore_proc.poll() is None:
            return
        if _port_open(11311):
            return      # someone else's master; use it, never stop it
        print("    [roscore] Starting roscore...", end="", flush=True)
        self._roscore_proc = self._popen(
            ["roscore"], join(self._proc_log_dir, "roscore.log"))
        deadline = time.time() + _ROSCORE_START_TIMEOUT
        while time.time() < deadline:
            if _port_open(11311):
                print(" OK", flush=True)
                return
            time.sleep(0.5)
        raise TimeoutError("roscore did not start within %.0fs" % _ROSCORE_START_TIMEOUT)

    def _wait_for_services(self, gz_log_path: str):
        print("    [setup] Waiting for Gazebo services...", end="", flush=True)
        t0 = time.time()
        while time.time() - t0 < _GAZEBO_SVC_TIMEOUT:
            if self._procs[0].poll() is not None:
                raise RuntimeError("Gazebo exited unexpectedly (rc=%d). See %s."
                                   % (self._procs[0].returncode, gz_log_path))
            try:
                rospy.wait_for_service("/gazebo/spawn_urdf_model", timeout=2.0)
                rospy.wait_for_service("/gazebo/set_model_state", timeout=2.0)
                rospy.wait_for_service("/gazebo/get_model_state", timeout=2.0)
                print(" OK (%.1fs)" % (time.time() - t0), flush=True)
                return
            except rospy.ROSException:
                print(".", end="", flush=True)
        raise TimeoutError("Gazebo services did not come up within %.0fs. See %s."
                           % (_GAZEBO_SVC_TIMEOUT, gz_log_path))

    def _wait_for_robots(self):
        """Block until every robot exists in Gazebo and has produced a scan."""
        print("    [setup] Waiting for %d robots..." % self.num_envs, end="", flush=True)
        t0 = time.time()
        pending = list(self.envs)
        while pending and time.time() - t0 < _SPAWN_TIMEOUT:
            still = []
            for env in pending:
                state = env._gazebo_sim.get_model_state()
                if not (state is not None and state.success and env._has_scan()):
                    still.append(env)
            pending = still
            if pending:
                print(".", end="", flush=True)
                time.sleep(0.5)
        if pending:
            raise TimeoutError(
                "Robots %s did not come up within %.0fs (spawned and scanning). "
                "See robot_*.log in %s."
                % (", ".join(e._ns for e in pending), _SPAWN_TIMEOUT, self._proc_log_dir))
        print(" OK (%.1fs)" % (time.time() - t0), flush=True)

    def _read_cylinders(self) -> np.ndarray:
        """(N, 2) centres of the world's cylinders, as Gazebo has them."""
        msg = rospy.wait_for_message("/gazebo/model_states", ModelStates, timeout=10.0)
        centres = [(p.position.x, p.position.y)
                   for name, p in zip(msg.name, msg.pose)
                   if name.startswith(_CYLINDER_PREFIX)]
        if not centres:
            raise RuntimeError("No '%s*' models found in the world." % _CYLINDER_PREFIX)
        return np.array(centres, dtype=np.float64)


def _port_open(port: int) -> bool:
    try:
        socket.create_connection(("localhost", port), timeout=1.0).close()
        return True
    except (socket.error, socket.timeout):
        return False


def _descendants(pid: int) -> List[int]:
    """Every live process below `pid`, read from /proc."""
    children: Dict[int, List[int]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open("/proc/%s/stat" % entry) as f:
                # "pid (comm) state ppid ..." — comm may itself contain ")".
                ppid = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        children.setdefault(ppid, []).append(int(entry))
    found, stack = [], [pid]
    while stack:
        for child in children.get(stack.pop(), []):
            found.append(child)
            stack.append(child)
    return found


def _alive(pid: int) -> bool:
    """Running, as opposed to gone or a zombie waiting to be reaped."""
    try:
        with open("/proc/%d/stat" % pid) as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


def _live_gzservers() -> List[int]:
    out = subprocess.run(["pgrep", "-x", "gzserver"], stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL).stdout.decode().split()
    return [int(p) for p in out if p.isdigit() and _alive(int(p))]


def _stop_trees(procs: Sequence[Optional[subprocess.Popen]], grace: float = 6.0):
    """SIGINT the given roslaunch processes, then SIGKILL whatever is left of
    their process trees after `grace` seconds.

    roslaunch puts every node in a session of its own, so neither the process
    group nor killing roslaunch itself reaches a node that ignores the SIGINT
    (gzserver does, now and then). The trees are therefore recorded up front,
    while the parent links still exist.
    """
    procs = [p for p in procs if p is not None]
    pids: List[int] = []
    for proc in procs:
        pids += [proc.pid] + _descendants(proc.pid)
    for proc in procs:
        try:
            os.kill(proc.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
    deadline = time.time() + grace
    while time.time() < deadline:
        for proc in procs:
            proc.poll()                 # reap, or it stays "alive" as a zombie
        if not any(_alive(pid) for pid in pids):
            break
        time.sleep(0.1)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for proc in procs:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    # SIGKILL is not instant: a gzserver releasing the GPU can take a few
    # seconds to disappear, and the next launch must not overlap it.
    deadline = time.time() + 15.0
    while time.time() < deadline and any(_alive(pid) for pid in pids):
        time.sleep(0.1)
