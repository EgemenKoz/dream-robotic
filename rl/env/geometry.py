"""
Planar collision test between the robot's chassis box and the BARN cylinders.

Every obstacle in a BARN world is a vertical cylinder of radius 0.075 m, and
the robot's Gazebo collision shape is a box, so "is the robot touching
anything" reduces to box-versus-circle distances in the ground plane.

MultiRobotBarnVecEnv uses this instead of the /collision topic: it needs no
per-robot contact subscriber (the stock one costs ~1.4 CPU cores each), it has
no message latency, and it cannot be confused by two robots sharing a world.
"""

import math

import numpy as np


def box_circle_clearance(
    x: float,
    y: float,
    yaw: float,
    centres: np.ndarray,
    half_length: float,
    half_width: float,
    radius: float,
) -> float:
    """Smallest gap in metres between an oriented box and a set of circles.

    The box is centred on (x, y), `half_length` along its heading `yaw` and
    `half_width` across it.  `centres` is (N, 2) circle centres in the world
    frame, all of radius `radius`.  Negative when the box overlaps a circle;
    +inf when there are no circles.
    """
    centres = np.asarray(centres, dtype=np.float64).reshape(-1, 2)
    if len(centres) == 0:
        return float("inf")
    dx = centres[:, 0] - x
    dy = centres[:, 1] - y
    c, s = math.cos(yaw), math.sin(yaw)
    # Circle centres in the box frame.
    lx = c * dx + s * dy
    ly = -s * dx + c * dy
    # Distance from each centre to the box: zero inside it, so a centre that
    # has ended up within the box reads as -radius.
    ox = np.maximum(np.abs(lx) - half_length, 0.0)
    oy = np.maximum(np.abs(ly) - half_width, 0.0)
    return float(np.min(np.hypot(ox, oy)) - radius)


def box_hits_circles(
    x: float,
    y: float,
    yaw: float,
    centres: np.ndarray,
    half_length: float,
    half_width: float,
    radius: float,
    margin: float = 0.0,
) -> bool:
    """True when the box is within `margin` metres of any circle."""
    return box_circle_clearance(
        x, y, yaw, centres, half_length, half_width, radius) <= margin


def world_cylinders(world_file: str) -> np.ndarray:
    """(N, 2) centres of the `unit_cylinder*` models in a BARN .world file."""
    import xml.etree.ElementTree as ET

    world = ET.parse(world_file).getroot().find("world")
    centres = []
    for model in world.findall("model"):
        if model.get("name", "").startswith("unit_cylinder"):
            pose = model.find("pose").text.split()
            centres.append((float(pose[0]), float(pose[1])))
    return np.array(centres, dtype=np.float64).reshape(-1, 2)


def segment_clearance(a, b, centres: np.ndarray, radius: float) -> float:
    """Smallest gap in metres between segment a-b and a set of circles."""
    centres = np.asarray(centres, dtype=np.float64).reshape(-1, 2)
    if len(centres) == 0:
        return float("inf")
    a = np.asarray(a, dtype=np.float64)
    ab = np.asarray(b, dtype=np.float64) - a
    denom = float(ab @ ab)
    t = np.zeros(len(centres)) if denom == 0.0 else np.clip((centres - a) @ ab / denom, 0.0, 1.0)
    nearest = a + t[:, None] * ab
    return float(np.min(np.hypot(*(centres - nearest).T)) - radius)


def shortcut_path(path: np.ndarray, centres: np.ndarray, radius: float) -> np.ndarray:
    """Remove the staircase steps and needless detours of a grid path without
    bringing it any closer to an obstacle.

    Greedy string pulling: from each kept point, jump to the farthest later
    point whose straight segment keeps at least the clearance the original
    path had over the stretch it replaces.  A shortcut through open space
    passes; one that cuts a corner tighter than the original does not.
    """
    path = np.asarray(path, dtype=np.float64)
    if len(path) <= 2:
        return path.copy()
    seg_clr = np.array([segment_clearance(path[k], path[k + 1], centres, radius)
                        for k in range(len(path) - 1)])
    keep = [0]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            need = seg_clr[i:j].min() - 1e-9
            if segment_clearance(path[i], path[j], centres, radius) >= need:
                break
            j -= 1
        keep.append(j)
        i = j
    return path[keep]
