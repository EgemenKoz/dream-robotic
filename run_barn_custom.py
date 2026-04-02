"""
Custom BARN simulation runner using ~/Desktop/BARN_dataset world files.

Usage:
    python3 run_barn_custom.py --world_idx 0
    python3 run_barn_custom.py --world_idx 0 --gui
    python3 run_barn_custom.py --world_idx 0 --out results.txt

Requirements:
    source /opt/ros/noetic/setup.bash
    source /home/egemen/barn_lab/jackal_ws/devel/setup.bash

World files are loaded from ~/Desktop/BARN_dataset/world_files/
Path files are loaded from ~/Desktop/BARN_dataset/path_files/
"""

import time
import argparse
import subprocess
import os
import sys
import signal
from os.path import join

import numpy as np
import rospy
import rospkg

from configs.config import Paths, WorldConfig, GazeboConfig, NavConfig, apply_env

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
    parser.add_argument("--out", type=str, default="results.txt", help="Output file for results")
    args = parser.parse_args()

    if not (0 <= args.world_idx <= 299):
        raise ValueError("world_idx must be 0-299 (only static BARN worlds supported here)")

    # Ensure child processes are always terminated on Ctrl+C or SIGTERM
    gazebo_process = None
    nav_stack_process = None

    def _shutdown(sig, frame):
        print("\nShutting down...")
        if nav_stack_process:
            nav_stack_process.terminate()
            nav_stack_process.wait()
        if gazebo_process:
            gazebo_process.terminate()
            gazebo_process.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    world_file = WorldConfig.world_file(args.world_idx)
    if not os.path.exists(world_file):
        raise FileNotFoundError("World file not found: %s" % world_file)

    ##########################################################################################
    ## 0. Launch Gazebo Simulation
    ##########################################################################################

    apply_env()

    rospack = rospkg.RosPack()
    base_path = rospack.get_path("jackal_helper")
    os.environ["GAZEBO_PLUGIN_PATH"] = join(base_path, "plugins")

    launch_file = join(base_path, "launch", "gazebo_launch.launch")

    print(">>>>>>>>>>>>>>>>>> Loading world_%d from BARN_dataset <<<<<<<<<<<<<<<<<<" % args.world_idx)
    gazebo_process = subprocess.Popen([
        "roslaunch",
        launch_file,
        "world_name:=" + world_file,
        "gui:=" + ("true" if args.gui else "false"),
    ])
    time.sleep(GazeboConfig.STARTUP_WAIT)

    rospy.init_node("barn_custom_runner", anonymous=True)
    rospy.set_param("/use_sim_time", GazeboConfig.USE_SIM_TIME)

    gazebo_sim = GazeboSimulation(init_position=INIT_POSITION)

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

    nav_launch = NavConfig.launch_file()  # switch planner in config.py → NavConfig.LOCAL_PLANNER

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
    while compute_distance(init_coor, curr_coor) < 0.1:
        curr_time = rospy.get_time()
        pos = gazebo_sim.get_model_state().pose.position
        curr_coor = (pos.x, pos.y)
        time.sleep(0.01)

    start_time = curr_time
    collided = False

    while compute_distance(goal_coor, curr_coor) > NavConfig.GOAL_TOLERANCE and not collided and curr_time - start_time < NavConfig.TIMEOUT:
        curr_time = rospy.get_time()
        pos = gazebo_sim.get_model_state().pose.position
        curr_coor = (pos.x, pos.y)
        print("Time: %.2f (s), x: %.2f (m), y: %.2f (m)" % (curr_time - start_time, *curr_coor), end="\r")
        collided = gazebo_sim.get_hard_collision()
        while rospy.get_time() - curr_time < 0.1:
            time.sleep(0.01)

    ##########################################################################################
    ## 3. Compute metrics and log results
    ##########################################################################################

    print("\n>>>>>>>>>>>>>>>>>> Test finished! <<<<<<<<<<<<<<<<<<")
    success = False
    if collided:
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

    with open(args.out, "a") as f:
        f.write("%d %d %d %d %.4f %.4f\n" % (
            args.world_idx, success, collided,
            int(curr_time - start_time >= 100),
            curr_time - start_time, nav_metric
        ))

    gazebo_process.terminate()
    gazebo_process.wait()
    nav_stack_process.terminate()
    nav_stack_process.wait()
