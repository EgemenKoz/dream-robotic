from configs.config import MoboroboConfig, RobotConfig, SimConfig


class ElasticBandConfig:
    """Hyperparameters for the Elastic Band controller.

    The Elastic Band (Quinlan & Khatib, 1993) represents a collision-free path
    as a chain of overlapping "bubbles" of free space, then deforms it under two
    forces until equilibrium:
      * internal contraction — pulls the band taut (shortens the path)
      * external repulsion    — pushes bubbles away from nearby obstacles
    A pure-pursuit tracker then follows the deformed band.
    """

    DT: float = 0.1   # seconds between control updates

    # ── Robot limits ────────────────────────────────────────────────────────
    _robot = SimConfig.ROBOT
    MAX_V: float = (
        MoboroboConfig.MAX_LINEAR_VEL if _robot == "moborobo" else RobotConfig.MAX_LINEAR_VEL
    )
    MAX_W: float = (
        MoboroboConfig.MAX_ANGULAR_VEL if _robot == "moborobo" else RobotConfig.MAX_ANGULAR_VEL
    )

    # ── Band deformation ──────────────────────────────────────────────────────
    DEFORM_ITERS: int = 8       # force-update iterations per control step
    K_INTERNAL:   float = 0.5   # contraction-force gain
    K_EXTERNAL:   float = 0.4   # repulsion-force gain
    STEP_ALPHA:   float = 0.3   # bubble displacement per iteration (× total force)
    RHO0:         float = 0.6   # m — repulsion active only when clearance < RHO0
    MAX_BUBBLE_MOVE: float = 0.3  # m — clamp on per-iteration bubble displacement

    # ── Band maintenance ──────────────────────────────────────────────────────
    # Adjacent bubbles must overlap; insert a bubble when the gap grows past
    # OVERLAP_INSERT × (r_i + r_j); drop a bubble when it sits well inside the
    # overlap of its neighbours (OVERLAP_REMOVE × ...).
    OVERLAP_INSERT: float = 0.8
    OVERLAP_REMOVE: float = 0.3
    MIN_BUBBLE_R:   float = 0.10  # m — clearance floor (avoids zero-radius bubbles)
    MAX_BUBBLE_R:   float = 1.50  # m — clearance cap (avoids open-space drift)
    MAX_BUBBLES:    int   = 60    # hard cap on band length

    # ── Re-planning (A* on the occupancy grid) ────────────────────────────────
    REPLAN_STEPS: int = 30    # rebuild the band from a fresh A* path every N steps
    MIN_SCANS:    int = 3     # new-obstacle events before the first plan

    # ── Pure-pursuit tracker ────────────────────────────────────────────────
    LOOKAHEAD:    float = 0.8   # m — carrot distance ahead along the band
    WP_REACH:     float = 0.4   # m — prune band points within this of the robot
    K_HEADING:    float = 1.5   # angular gain (ω = K_HEADING × heading error)
    # Slow down for sharp turns and tight clearance:
    TURN_SLOWDOWN: float = 1.2  # higher → brakes harder when misaligned
    CLEAR_SLOWDOWN_R: float = 0.4  # m — scale speed down when clearance below this

    # ── Obstacle cloud ────────────────────────────────────────────────────────
    SCAN_SUBSAMPLE: int = 4   # use every Nth lidar ray when building the cloud
