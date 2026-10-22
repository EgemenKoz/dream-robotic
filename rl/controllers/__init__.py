from rl.controllers.mppi import MPPIController
from rl.controllers.dwa import DWAController
from rl.controllers.astar_local import AStarLocalController
from rl.controllers.online_planner import OnlineGlobalPlanner

# MPPI-JAX is optional — only available when JAX is installed.
try:
    from rl.controllers.mppi_jax import MPPIControllerJAX
    _HAS_JAX_MPPI = True
except ImportError:
    MPPIControllerJAX = None  # type: ignore[assignment]
    _HAS_JAX_MPPI = False

# SACController is optional — requires stable-baselines3.
try:
    from rl.controllers.rl_sac_controller import SACController
    _HAS_SAC_CTRL = True
except ImportError:
    SACController = None  # type: ignore[assignment]
    _HAS_SAC_CTRL = False

__all__ = [
    "MPPIController", "DWAController", "AStarLocalController",
    "OnlineGlobalPlanner", "MPPIControllerJAX", "SACController",
]
