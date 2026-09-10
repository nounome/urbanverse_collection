"""Animated pedestrian support for UrbanVerse dynamic-agent scenes."""

from .official_people import (
    OfficialPeopleManager,
    author_official_people,
    author_people_payload,
    configure_preauthored_people,
    load_people_config,
)
from .walkable_regions import WalkableRegionConfig, WalkableRegions
from .roaming import (
    RoamingAssignment,
    plan_roaming_assignments,
)
from .project_roaming import ProjectPeopleRoamingManager, build_project_people_payload

__all__ = [
    "OfficialPeopleManager",
    "author_official_people",
    "author_people_payload",
    "configure_preauthored_people",
    "load_people_config",
    "WalkableRegionConfig",
    "WalkableRegions",
    "RoamingAssignment",
    "plan_roaming_assignments",
    "ProjectPeopleRoamingManager",
    "build_project_people_payload",
]
