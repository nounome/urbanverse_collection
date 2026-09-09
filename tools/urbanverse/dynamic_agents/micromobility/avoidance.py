"""Read-only road/obstacle/agent interfaces for micromobility admission and braking."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from .calibration import AssetCalibration
from .motion import MicromobilityState, footprint_corners


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    # Collinear lines are not necessarily overlapping *segments*. The old
    # orientation-only test stopped bikes tens of metres apart on one lane.
    if np.any(np.maximum(np.minimum(a, b), np.minimum(c, d)) >
              np.minimum(np.maximum(a, b), np.maximum(c, d)) + 1.e-10):
        return False
    def cross(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        first, second = q - p, r - p
        return float(first[0] * second[1] - first[1] * second[0])

    values = (cross(a, b, c), cross(a, b, d), cross(c, d, a), cross(c, d, b))
    return values[0] * values[1] <= 0.0 and values[2] * values[3] <= 0.0


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    x, y = map(float, point)
    inside = False
    for first, second in zip(polygon, np.roll(polygon, -1, axis=0)):
        x1, y1 = map(float, first)
        x2, y2 = map(float, second)
        if (y1 > y) != (y2 > y):
            crossing = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing:
                inside = not inside
    return inside


def polygons_intersect(first: np.ndarray, second: np.ndarray) -> bool:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if any(point_in_polygon(point, second) for point in first):
        return True
    if any(point_in_polygon(point, first) for point in second):
        return True
    return any(
        _segments_intersect(a, b, c, d)
        for a, b in zip(first, np.roll(first, -1, axis=0))
        for c, d in zip(second, np.roll(second, -1, axis=0))
    )


@dataclass(frozen=True)
class PolygonRoadSurface:
    polygons_xy: tuple[np.ndarray, ...]

    def contains_footprint(self, corners_xy: np.ndarray) -> bool:
        return any(all(point_in_polygon(point, polygon) for point in corners_xy) for polygon in self.polygons_xy)


@dataclass(frozen=True)
class StaticObstacle:
    obstacle_id: str
    footprint_xy: np.ndarray


@dataclass(frozen=True)
class AgentObservation:
    agent_id: str
    footprint_xy: np.ndarray
    velocity_xy: np.ndarray


@dataclass(frozen=True)
class AvoidanceContext:
    road: PolygonRoadSurface
    static_obstacles: tuple[StaticObstacle, ...] = ()
    other_agents: tuple[AgentObservation, ...] = ()


def collision_free(
    state: MicromobilityState,
    calibration: AssetCalibration,
    context: AvoidanceContext,
    margin_m: float = 0.0,
) -> bool:
    footprint = footprint_corners(state, calibration, margin_m)
    if not context.road.contains_footprint(footprint):
        return False
    bodies: Iterable[np.ndarray] = (
        [obstacle.footprint_xy for obstacle in context.static_obstacles]
        + [agent.footprint_xy for agent in context.other_agents]
    )
    return not any(polygons_intersect(footprint, body) for body in bodies)


def longitudinal_speed_cap(
    state: MicromobilityState,
    calibration: AssetCalibration,
    context: AvoidanceContext,
    *,
    requested_speed_mps: float,
    horizon_s: float = 3.0,
    sample_dt_s: float = 0.25,
    braking_mps2: float = 2.0,
    safety_margin_m: float = 0.2,
) -> float:
    """Return a longitudinal cap only; steering and XY remain non-holonomic."""
    requested = max(0.0, float(requested_speed_mps))
    if requested == 0.0:
        return 0.0
    forward = np.asarray([math.cos(state.yaw_rad), math.sin(state.yaw_rad)], dtype=np.float64)
    for time_s in np.arange(sample_dt_s, horizon_s + 1.0e-9, sample_dt_s):
        probe = MicromobilityState(
            x_m=state.x_m + float(forward[0]) * requested * float(time_s),
            y_m=state.y_m + float(forward[1]) * requested * float(time_s),
            yaw_rad=state.yaw_rad,
            speed_mps=requested,
        )
        if not collision_free(probe, calibration, context, margin_m=safety_margin_m):
            clearance = max(0.0, requested * float(time_s) - calibration.length_m * 0.5)
            return min(requested, math.sqrt(max(0.0, 2.0 * braking_mps2 * clearance)))
    return requested
