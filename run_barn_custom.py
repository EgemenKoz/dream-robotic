"""
Custom BARN simulation runner using ~/Desktop/BARN_dataset world files.

Usage:
    python3 run_barn_custom.py --world_idx 0
    python3 run_barn_custom.py --world_idx 0 --gui
    python3 run_barn_custom.py --world_idx 0 --robot moborobo
    python3 run_barn_custom.py --world_idx 0 --out results.txt

Requirements:
    source /opt/ros/noetic/setup.bash
    source /home/egemen/barn_lab/jackal_ws/devel/setup.bash

For Moborobo, also source the moborobo catkin workspace:
    source ~/barn_lab/moborobo_ws/devel/setup.bash

World files are loaded from ~/Desktop/BARN_dataset/world_files/
Path files are loaded from ~/Desktop/BARN_dataset/path_files/
"""

import time
import datetime
import argparse
import subprocess
import os
import sys
import signal
from os.path import join

import numpy as np
import rospy
import rospkg

from configs.config import (
    SimConfig, Paths, WorldConfig, GazeboConfig, NavConfig,
    RobotConfig, MoboroboConfig, apply_env,
)

# gazebo_simulation.py lives in the-barn-challenge; add it to path
sys.path.insert(0, Paths.BARN_CHALLENGE)
from gazebo.gazebo_simulation import GazeboSimulation

INIT_POSITION = WorldConfig.STATIC_INIT_POSITION
GOAL_POSITION = WorldConfig.STATIC_GOAL_OFFSET


def compute_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def path_coord_to_gazebo_coord(x, y):
    """Convert BARN dataset grid coordinates to Gazebo world coordinates."""
    RADIUS = 0.075
    r_shift = -RADIUS - (30 * RADIUS * 2)
    c_shift = RADIUS + 5
    gazebo_x = x * (RADIUS * 2) + r_shift
    gazebo_y = y * (RADIUS * 2) + c_shift
    return (gazebo_x, gazebo_y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BARN navigation challenge with custom dataset")
    parser.add_argument("--world_idx", type=int, default=0, help="World index 0-299")
    parser.add_argument("--gui", action="store_true", help="Launch Gazebo with GUI")
    parser.add_argument("--out", type=str, default=None,
                        help="Results summary file (default: results/results_{robot}.txt)")
    parser.add_argument(
        "--robot", type=str, default=SimConfig.ROBOT,
        choices=["jackal", "moborobo"],
        help="Robot platform to simulate (default: %(default)s)",
    )
    args = parser.parse_args()
    SimConfig.ROBOT = args.robot   # propagate CLI choice into config

    if not (0 <= args.world_idx <= 299):
        raise ValueError("world_idx must be 0-299 (only static BARN worlds supported here)")

    # Ensure child processes are always terminated on Ctrl+C or SIGTERM.
    # _stop is set by the signal handler so the main loop can exit cleanly
    # and still write logs before shutting down.
    gazebo_process = None
    nav_stack_process = None
    _stop = [False]   # list so the nested function can mutate it without nonlocal

    def _kill_procs():
        for proc in [nav_stack_process, gazebo_process]:
            if proc is None:
                continue
            try:
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait()
            except OSError:
                pass

    def _shutdown(sig, frame):
        print("\nShutdown requested — finishing log then exiting...")
        _stop[0] = True   # signals the monitoring loop to break

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    world_file = WorldConfig.world_file(args.world_idx)
    if not os.path.exists(world_file):
        raise FileNotFoundError("World file not found: %s" % world_file)

    ##########################################################################################
    ## 0. Launch Gazebo Simulation
    ##########################################################################################

    apply_env(robot=args.robot)

    # Ensure jackal_ws is visible regardless of workspace chaining order.
    # Sourcing moborobo_ws last can drop jackal_ws from both ROS_PACKAGE_PATH
    # (used by rospkg to locate packages) and CMAKE_PREFIX_PATH (used by roslaunch
    # to locate compiled node binaries in devel/lib/).  Both must be set.
    jackal_src   = join(Paths.JACKAL_WS, "src")
    jackal_devel = join(Paths.JACKAL_WS, "devel")
    if os.path.isdir(jackal_src):
        pkg_path = os.environ.get("ROS_PACKAGE_PATH", "")
        if jackal_src not in pkg_path:
            os.environ["ROS_PACKAGE_PATH"] = jackal_src + (":" + pkg_path if pkg_path else "")
    if os.path.isdir(jackal_devel):
        cmake_path = os.environ.get("CMAKE_PREFIX_PATH", "")
        if jackal_devel not in cmake_path:
            os.environ["CMAKE_PREFIX_PATH"] = jackal_devel + (":" + cmake_path if cmake_path else "")

    print(">>>>>>>>>>>>>>>>>> Loading world_%d from BARN_dataset [robot=%s] <<<<<<<<<<<<<<<<<<" % (
        args.world_idx, args.robot))

    rospack = rospkg.RosPack()

    if args.robot == "jackal":
        base_path = rospack.get_path("jackal_helper")
        os.environ["GAZEBO_PLUGIN_PATH"] = join(base_path, "plugins")
        launch_file = join(base_path, "launch", "gazebo_launch.launch")
        gazebo_process = subprocess.Popen([
            "roslaunch",
            launch_file,
            "world_name:=" + world_file,
            "gui:=" + ("true" if args.gui else "false"),
        ])
    else:  # moborobo
        # Detect whether jackal_ws is sourced (collision publisher lives there)
        try:
            rospack.get_path("jackal_helper")
            has_jackal_helper = True
        except rospkg.ResourceNotFound:
            has_jackal_helper = False
            print("[WARN] jackal_helper not found — collision detection disabled.")
            print("       Source jackal_ws before running to enable it.")

        launch_file = MoboroboConfig.gazebo_launch_file()
        gazebo_process = subprocess.Popen([
            "roslaunch",
            launch_file,
            "world_name:=" + world_file,
            "gui:=" + ("true" if args.gui else "false"),
            "headless:=" + ("false" if args.gui else "true"),
            "use_collision_publisher:=" + ("true" if has_jackal_helper else "false"),
        ])
    time.sleep(GazeboConfig.STARTUP_WAIT)

    rospy.init_node("barn_custom_runner", anonymous=True)
    rospy.set_param("/use_sim_time", GazeboConfig.USE_SIM_TIME)

    model_name = MoboroboConfig.MODEL_NAME if args.robot == "moborobo" else "jackal"
    gazebo_sim = GazeboSimulation(init_position=INIT_POSITION, model_name=model_name)

    init_coor = (INIT_POSITION[0], INIT_POSITION[1])
    goal_coor = (INIT_POSITION[0] + GOAL_POSITION[0], INIT_POSITION[1] + GOAL_POSITION[1])

    # Wait until robot is in position and not colliding
    pos = gazebo_sim.get_model_state().pose.position
    curr_coor = (pos.x, pos.y)
    collided = True
    while compute_distance(init_coor, curr_coor) > GazeboConfig.POSITION_TOL or collided:
        gazebo_sim.reset()
        pos = gazebo_sim.get_model_state().pose.position
        curr_coor = (pos.x, pos.y)
        collided = gazebo_sim.get_hard_collision()
        time.sleep(GazeboConfig.RESET_WAIT)

    ##########################################################################################
    ## 1. Launch your navigation stack
    ## ── Customize this block to swap in your own planner ────────────────────────────────
    ##########################################################################################

    nav_launch = NavConfig.launch_file(robot=args.robot)  # switch robot/planner in config.py

    nav_stack_process = subprocess.Popen(["roslaunch", nav_launch])

    import actionlib
    from geometry_msgs.msg import Quaternion
    from move_base_msgs.msg import MoveBaseGoal, MoveBaseAction

    nav_as = actionlib.SimpleActionClient("/move_base", MoveBaseAction)
    mb_goal = MoveBaseGoal()
    mb_goal.target_pose.header.frame_id = "odom"
    mb_goal.target_pose.pose.position.x = GOAL_POSITION[0]
    mb_goal.target_pose.pose.position.y = GOAL_POSITION[1]
    mb_goal.target_pose.pose.position.z = 0
    mb_goal.target_pose.pose.orientation = Quaternion(0, 0, 0, 1)

    nav_as.wait_for_server()
    nav_as.send_goal(mb_goal)

    ##########################################################################################
    ## 2. Monitor navigation
    ##########################################################################################

    curr_time = rospy.get_time()
    pos = gazebo_sim.get_model_state().pose.position
    curr_coor = (pos.x, pos.y)

    # Wait for robot to start moving
    while compute_distance(init_coor, curr_coor) < 0.1 and not _stop[0]:
        curr_time = rospy.get_time()
        pos = gazebo_sim.get_model_state().pose.position
        curr_coor = (pos.x, pos.y)
        time.sleep(0.01)

    start_time = curr_time
    collided = False

    while (compute_distance(goal_coor, curr_coor) > NavConfig.GOAL_TOLERANCE
           and not collided
           and curr_time - start_time < NavConfig.TIMEOUT
           and not _stop[0]):
        curr_time = rospy.get_time()
        pos = gazebo_sim.get_model_state().pose.position
        curr_coor = (pos.x, pos.y)
        print("Time: %.2f (s), x: %.2f (m), y: %.2f (m)" % (curr_time - start_time, *curr_coor), end="\r")
        collided = gazebo_sim.get_hard_collision()
        while rospy.get_time() - curr_time < 0.1 and not _stop[0]:
            time.sleep(0.01)

    ##########################################################################################
    ## 3. Compute metrics and log results
    ##########################################################################################

    print("\n>>>>>>>>>>>>>>>>>> Test finished! <<<<<<<<<<<<<<<<<<")
    success = False
    if _stop[0] and not collided and compute_distance(goal_coor, curr_coor) > NavConfig.GOAL_TOLERANCE:
        status = "interrupted"
    elif collided:
        status = "collided"
    elif curr_time - start_time >= NavConfig.TIMEOUT:
        status = "timeout"
    else:
        status = "succeeded"
        success = True
    print("Navigation %s with time %.4f (s)" % (status, curr_time - start_time))

    # Load optimal path from BARN_dataset path_files
    path_file = WorldConfig.path_file(args.world_idx)
    path_array = np.load(path_file)
    path_array = [path_coord_to_gazebo_coord(*p) for p in path_array]
    path_array = np.insert(path_array, 0, (INIT_POSITION[0], INIT_POSITION[1]), axis=0)
    path_array = np.insert(path_array, len(path_array), (goal_coor[0], goal_coor[1]), axis=0)
    path_length = sum(compute_distance(p1, p2) for p1, p2 in zip(path_array[:-1], path_array[1:]))

    optimal_time = path_length / NavConfig.NOMINAL_SPEED
    actual_time  = curr_time - start_time
    nav_metric   = int(success) * optimal_time / np.clip(
        actual_time,
        NavConfig.METRIC_TIME_LO * optimal_time,
        NavConfig.METRIC_TIME_HI * optimal_time,
    )
    print("Navigation metric: %.4f" % nav_metric)

    elapsed = curr_time - start_time
    timed_out = elapsed >= NavConfig.TIMEOUT

    # ── Results directory ──────────────────────────────────────────────────────
    results_dir = Paths.RESULTS_DIR
    logs_dir = join(results_dir, "logs")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # ── Summary file (one line per run, appended) ──────────────────────────────
    if args.out is None:
        out_file = join(results_dir, "results_%s.txt" % args.robot)
    elif os.path.isabs(args.out):
        out_file = args.out
    else:
        out_file = join(results_dir, args.out)

    write_header = not os.path.exists(out_file)
    with open(out_file, "a") as f:
        if write_header:
            f.write("# world_idx  succeeded  collided  timed_out  time_elapsed(s)  nav_metric\n")
        f.write("%d %d %d %d %.4f %.4f\n" % (
            args.world_idx, int(success), int(collided),
            int(timed_out), elapsed, nav_metric,
        ))

    # ── Load BARN world difficulty metrics ────────────────────────────────────
    METRIC_NAMES = [
        "Distance to Closest Obstacle",
        "Average Visibility",
        "Dispersion",
        "Characteristic Dimension",
        "Tortuosity",
    ]
    world_metrics = None
    metrics_file_path = WorldConfig.metrics_file(args.world_idx)
    if os.path.exists(metrics_file_path):
        world_metrics = np.load(metrics_file_path)

    norm_metrics = None
    norm_metrics_file_path = WorldConfig.norm_metrics_file(args.world_idx)
    if os.path.exists(norm_metrics_file_path):
        norm_metrics = np.load(norm_metrics_file_path)

    # ── Per-run detailed log ───────────────────────────────────────────────────
    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = join(logs_dir, "world_%03d_%s_%s.log" % (args.world_idx, args.robot, run_ts))

    SEP  = "=" * 72
    THIN = "-" * 72

    with open(log_file, "w") as f:
        def w(line=""):
            f.write(line + "\n")

        w(SEP)
        w("BARN Navigation Run Log")
        w(SEP)
        w("Timestamp      : %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        w("World Index    : %d" % args.world_idx)
        w("Robot          : %s" % args.robot)
        w("World File     : %s" % world_file)
        w()
        w(THIN)
        w("NAVIGATION OUTCOME")
        w(THIN)
        w("Status         : %s" % status.upper())
        w("Reached Goal   : %s" % ("YES" if success else "NO"))
        w("Collided       : %s" % ("YES" if collided else "NO"))
        w("Timed Out      : %s" % ("YES" if timed_out else "NO"))
        w()
        w(THIN)
        w("TIMING")
        w(THIN)
        w("Time Elapsed   : %.4f s" % elapsed)
        w("Timeout Limit  : %.2f s" % NavConfig.TIMEOUT)
        w()
        w(THIN)
        w("PATH & SCORE METRICS")
        w(THIN)
        w("Path Length    : %.4f m" % path_length)
        w("Optimal Time   : %.4f s" % optimal_time)
        w("Actual Time    : %.4f s" % elapsed)
        w("Nav Metric     : %.6f" % nav_metric)
        w("  (formula: success * optimal_time / clip(actual, %.1f*opt, %.1f*opt))" % (
            NavConfig.METRIC_TIME_LO, NavConfig.METRIC_TIME_HI))
        if world_metrics is not None:
            w()
            w(THIN)
            w("WORLD DIFFICULTY METRICS")
            w(THIN)
            for i, name in enumerate(METRIC_NAMES):
                norm_str = ""
                if norm_metrics is not None and i < len(norm_metrics):
                    norm_str = "  (norm: %+.4f)" % norm_metrics[i]
                w("  %-35s %.4f%s" % (name + ":", world_metrics[i], norm_str))
        w()
        w(THIN)
        w("POSITIONS")
        w(THIN)
        w("Start          : (%.4f, %.4f)" % (init_coor[0], init_coor[1]))
        w("Goal           : (%.4f, %.4f)" % (goal_coor[0], goal_coor[1]))
        w("Final          : (%.4f, %.4f)" % (curr_coor[0], curr_coor[1]))
        w("Dist to Goal   : %.4f m" % compute_distance(goal_coor, curr_coor))
        w(SEP)

    print("Summary  -> %s" % out_file)
    print("Run log  -> %s" % log_file)

    _kill_procs()
