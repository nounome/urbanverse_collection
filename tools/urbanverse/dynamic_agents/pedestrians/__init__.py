"""Animated pedestrian support for UrbanVerse dynamic-agent scenes."""

from .official_people import (
    OfficialPeopleManager,
    author_official_people,
    author_people_payload,
    configure_preauthored_people,
    load_people_config,
)
from .official_runtime import (
    OfficialCharacterSpec,
    OfficialPeopleRuntime,
    enable_official_people_runtime,
    goto_command,
    simulation_app_launch_config,
)
from .walkable_regions import WalkableRegionConfig, WalkableRegions
from .roaming import (
    OfficialPeopleRoamingManager,
    RoamingAssignment,
    align_initial_targets_to_waypoint_loops,
    assignments_to_specs,
    build_roaming_waypoint_loops,
    expand_waypoint_loops_with_grid_paths,
    plan_roaming_assignments,
)
from .navmesh_regions import (
    author_approved_navmesh_surface,
    finalize_approved_navmesh_surface,
)
from .project_roaming import ProjectPeopleRoamingManager, build_project_people_payload

__all__ = [
    "OfficialPeopleManager",
    "author_official_people",
    "author_people_payload",
    "configure_preauthored_people",
    "load_people_config",
    "OfficialCharacterSpec",
    "OfficialPeopleRuntime",
    "enable_official_people_runtime",
    "goto_command",
    "simulation_app_launch_config",
    "WalkableRegionConfig",
    "WalkableRegions",
    "OfficialPeopleRoamingManager",
    "RoamingAssignment",
    "align_initial_targets_to_waypoint_loops",
    "assignments_to_specs",
    "build_roaming_waypoint_loops",
    "expand_waypoint_loops_with_grid_paths",
    "plan_roaming_assignments",
    "author_approved_navmesh_surface",
    "finalize_approved_navmesh_surface",
    "ProjectPeopleRoamingManager",
    "build_project_people_payload",
]
