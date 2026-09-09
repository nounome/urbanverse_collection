"""Configurable multi-vehicle traffic managers."""

from .continuous_manager import Scene10ContinuousVehicleManager
from .multivehicle_manager import Scene10MultiVehicleManager

# Scene-independent public names.  Keep the historical Scene10 names as
# compatibility aliases because old run metadata and scripts import them.
ContinuousVehicleManager = Scene10ContinuousVehicleManager
MultiVehicleManager = Scene10MultiVehicleManager

__all__ = [
    "ContinuousVehicleManager",
    "MultiVehicleManager",
    "Scene10ContinuousVehicleManager",
    "Scene10MultiVehicleManager",
]
