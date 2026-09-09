#!/usr/bin/env python3
"""Pure-Python high-level route helpers for UrbanVerse Go2 capture.

The low-level Isaac Lab locomotion policy remains an existing, frozen executor.
This module only turns a safe waypoint route into velocity commands and detects
when the executor is no longer making useful progress.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Any, Iterable

import numpy as np


CONTROLLER_PROFILES = {
    "flat": {
        "task": "Isaac-Velocity-Flat-Unitree-Go2-v0",
        "checkpoint_url": (
            "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/"
            "Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/"
            "Isaac-Velocity-Flat-Unitree-Go2-v0/checkpoint.pt"
        ),
        "terrain_observation": "none",
    },
    "rough": {
        "task": "Isaac-Velocity-Rough-Unitree-Go2-v0",
        "checkpoint_url": (
            "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/"
            "Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/"
            "Isaac-Velocity-Rough-Unitree-Go2-v0/checkpoint.pt"
        ),
        "terrain_observation": "1.6 m x 1.0 m yaw-aligned height scan at 0.1 m resolution",
    },
}


@dataclass
class TrajectoryCarrier:
    """Prescribe a smooth, repeatable out-and-back pose on one admitted road.

    This is intentionally a navigation-data carrier, not a dynamics model.  It
    keeps the articulated Go2 and its camera rig in the simulation while the
    base pose follows a known route.  The frozen locomotion policy may animate
    the joints, but contacts and gait quality are not valid evidence in this
    mode.
    """

    start_world_xy: list[float]
    end_world_xy: list[float]
    speed: float = 0.30
    settle_duration_s: float = 1.0
    endpoint_hold_duration_s: float = 0.6
    turn_duration_s: float = 2.0
    turn_sign: float = 1.0
    loop_count: int = 0
    route_complete: bool = False
    stop_at_final_waypoint: bool = False

    def __post_init__(self) -> None:
        self.start = np.asarray(self.start_world_xy, dtype=np.float64)
        self.end = np.asarray(self.end_world_xy, dtype=np.float64)
        delta = self.end - self.start
        self.distance = float(np.linalg.norm(delta))
        if self.distance <= 0.5:
            raise ValueError("trajectory carrier road segment must be longer than 0.5")
        if self.speed <= 0.0 or self.turn_duration_s <= 0.0:
            raise ValueError("trajectory carrier speed and turn duration must be positive")
        self.direction = delta / self.distance
        self.outbound_yaw = math.atan2(float(delta[1]), float(delta[0]))
        self.inbound_yaw = wrap_angle(self.outbound_yaw + float(self.turn_sign) * math.pi)
        self.travel_duration_s = self.distance / float(self.speed)
        self.cycle_duration_s = (
            2.0 * self.travel_duration_s
            + 2.0 * float(self.endpoint_hold_duration_s)
            + 2.0 * float(self.turn_duration_s)
        )

    @property
    def waypoints_world_xy(self) -> list[list[float]]:
        return [self.start.tolist(), self.end.tolist(), self.start.tolist()]

    def sample(self, timestamp_s: float) -> dict[str, Any]:
        """Return a commanded pose and body-frame velocity at ``timestamp_s``."""
        timestamp_s = max(0.0, float(timestamp_s))
        if timestamp_s < self.settle_duration_s:
            return self._row(
                self.start,
                self.outbound_yaw,
                0.0,
                0.0,
                "initial_settle",
                waypoint_index=1,
                target=self.end,
            )

        elapsed = timestamp_s - float(self.settle_duration_s)
        cycle_index = int(elapsed // self.cycle_duration_s)
        phase = elapsed - cycle_index * self.cycle_duration_s
        self.loop_count = cycle_index
        self.route_complete = cycle_index >= 1
        travel = self.travel_duration_s
        hold = float(self.endpoint_hold_duration_s)
        turn = float(self.turn_duration_s)

        if phase < travel:
            fraction = phase / travel
            position = self.start + self.direction * (fraction * self.distance)
            return self._row(position, self.outbound_yaw, self.speed, 0.0, "outbound", 1, self.end)
        phase -= travel
        if phase < hold:
            return self._row(self.end, self.outbound_yaw, 0.0, 0.0, "far_hold", 1, self.end)
        phase -= hold
        if phase < turn:
            fraction = phase / turn
            yaw_rate = float(self.turn_sign) * math.pi / turn
            yaw = wrap_angle(self.outbound_yaw + fraction * float(self.turn_sign) * math.pi)
            return self._row(self.end, yaw, 0.0, yaw_rate, "far_turn", 0, self.start)
        phase -= turn
        if phase < travel:
            fraction = phase / travel
            position = self.end - self.direction * (fraction * self.distance)
            return self._row(position, self.inbound_yaw, self.speed, 0.0, "inbound", 0, self.start)
        phase -= travel
        if phase < hold:
            return self._row(self.start, self.inbound_yaw, 0.0, 0.0, "near_hold", 0, self.start)
        phase -= hold
        fraction = min(1.0, phase / turn)
        yaw_rate = float(self.turn_sign) * math.pi / turn
        yaw = wrap_angle(self.inbound_yaw + fraction * float(self.turn_sign) * math.pi)
        return self._row(self.start, yaw, 0.0, yaw_rate, "near_turn", 1, self.end)

    def _row(
        self,
        position: np.ndarray,
        yaw: float,
        forward_speed: float,
        yaw_rate: float,
        label: str,
        waypoint_index: int,
        target: np.ndarray,
    ) -> dict[str, Any]:
        world_velocity = np.asarray(
            [forward_speed * math.cos(yaw), forward_speed * math.sin(yaw)], dtype=np.float64
        )
        return {
            "label": f"carrier_{self.loop_count:02d}/{label}",
            "linear_x": float(forward_speed),
            "linear_y": 0.0,
            "angular_z": float(yaw_rate),
            "waypoint_index": int(waypoint_index),
            "waypoint_world_xy": target.tolist(),
            "distance_to_waypoint": float(np.linalg.norm(target - position)),
            "heading_error_rad": 0.0,
            "target_position_world_xy": position.tolist(),
            "target_yaw_rad_world": float(yaw),
            "target_linear_velocity_world_xy": world_velocity.tolist(),
        }


def wrap_angle(angle: float) -> float:
    """Wrap radians to [-pi, pi)."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def resolve_scene_navigation(config: dict[str, Any], scene: str) -> dict[str, Any]:
    """Merge the global navigation defaults with a scene-specific override."""
    result = dict(config.get("navigation_defaults", {}))
    for row in config.get("scenes", []):
        if row.get("slug") == scene:
            scene_navigation = row.get("navigation", {})
            # Deep-merge the global_route_planner sub-dict so a scene block inherits
            # the defaults while overriding only the fields it authors.
            defaults_grp = result.get("global_route_planner")
            scene_grp = scene_navigation.get("global_route_planner")
            if isinstance(defaults_grp, dict) and isinstance(scene_grp, dict):
                merged_grp = dict(defaults_grp)
                merged_grp.update(scene_grp)
                scene_navigation = dict(scene_navigation)
                scene_navigation["global_route_planner"] = merged_grp
            result.update(scene_navigation)
            break
    if not result:
        raise KeyError(f"no navigation configuration for {scene}")
    return result


def transform_body_waypoints(
    origin_xy: Iterable[float], yaw_rad: float, body_waypoints: Iterable[Iterable[float]]
) -> list[list[float]]:
    """Transform XY waypoints expressed in the initial body frame into world XY."""
    origin = np.asarray(list(origin_xy), dtype=np.float64)
    rotation = np.asarray(
        [[math.cos(yaw_rad), -math.sin(yaw_rad)], [math.sin(yaw_rad), math.cos(yaw_rad)]],
        dtype=np.float64,
    )
    return [(origin + rotation @ np.asarray(list(point), dtype=np.float64)).tolist() for point in body_waypoints]


def rounded_right_angle_waypoints(
    start_xy: Iterable[float],
    corner_xy: Iterable[float],
    end_xy: Iterable[float],
    radius: float = 1.5,
    sample_count: int = 12,
) -> list[list[float]]:
    """Replace an admitted L corner with a tangent quadratic walking arc.

    The start/corner/end remain the geometric route definition.  The returned
    execution waypoints enter and leave the corner along the two admitted legs,
    avoiding one abrupt 90-degree velocity-command change.
    """
    start = np.asarray(list(start_xy), dtype=np.float64)
    corner = np.asarray(list(corner_xy), dtype=np.float64)
    end = np.asarray(list(end_xy), dtype=np.float64)
    first = corner - start
    second = end - corner
    first_length = float(np.linalg.norm(first))
    second_length = float(np.linalg.norm(second))
    if first_length <= 0.5 or second_length <= 0.5:
        raise ValueError("rounded right-angle legs must each be longer than 0.5")
    first_direction = first / first_length
    second_direction = second / second_length
    if abs(float(np.dot(first_direction, second_direction))) > 0.15:
        raise ValueError("rounded route requires approximately perpendicular legs")
    radius = min(float(radius), first_length * 0.45, second_length * 0.45)
    if radius <= 0.2:
        raise ValueError("rounded right-angle radius must exceed 0.2")
    sample_count = int(sample_count)
    if sample_count < 4:
        raise ValueError("rounded right-angle route needs at least four arc samples")
    entry = corner - first_direction * radius
    exit_point = corner + second_direction * radius
    rows = [start.tolist(), entry.tolist()]
    for index in range(1, sample_count + 1):
        t = float(index) / float(sample_count)
        point = (1.0 - t) ** 2 * entry + 2.0 * (1.0 - t) * t * corner + t**2 * exit_point
        rows.append(point.tolist())
    rows.append(end.tolist())
    return rows


def resample_polyline(points: Iterable[Iterable[float]], spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Resample a polyline to uniform arc-length spacing.

    Returns ``(path, cumulative_arc_length)`` where ``path[0] == points[0]`` and
    ``path[-1] == points[-1]``.  This is the dense reference used by the pure
    pursuit tracker so curvature can be estimated without sparse waypoints.
    """
    source = np.asarray(list(points), dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 2 or len(source) < 2:
        raise ValueError("polyline must be an Nx2 array with at least two points")
    if spacing <= 0.0:
        raise ValueError("resample spacing must be positive")
    segment_lengths = np.linalg.norm(np.diff(source, axis=0), axis=1)
    total = float(segment_lengths.sum())
    if total <= 0.0:
        raise ValueError("polyline must have positive length")
    arc = np.linspace(0.0, total, max(2, int(math.ceil(total / spacing)) + 1))
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    rows = []
    for value in arc:
        index = int(np.searchsorted(cumulative, value, side="right") - 1)
        index = min(max(index, 0), len(source) - 2)
        segment = segment_lengths[index]
        fraction = (value - cumulative[index]) / segment if segment > 1e-9 else 0.0
        rows.append(source[index] + fraction * (source[index + 1] - source[index]))
    return np.asarray(rows, dtype=np.float64), arc


def nearest_polyline_point(point: Iterable[float], path: np.ndarray, cumulative: np.ndarray) -> tuple[int, float]:
    """Closest point on ``path`` to ``point``, returning ``(segment_index, arc)``."""
    query = np.asarray(list(point), dtype=np.float64)
    starts = path[:-1]
    directions = path[1:] - path[:-1]
    denominator = np.maximum(np.sum(directions * directions, axis=1), 1e-12)
    fraction = np.sum((query - starts) * directions, axis=1) / denominator
    fraction = np.clip(fraction, 0.0, 1.0)
    closest = starts + fraction[:, None] * directions
    distances = np.sum((closest - query) ** 2, axis=1)
    index = int(np.argmin(distances))
    arc = cumulative[index] + fraction[index] * (cumulative[index + 1] - cumulative[index])
    return index, float(arc)


def arc_point_and_tangent(path: np.ndarray, cumulative: np.ndarray, arc: float) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(point, unit_tangent)`` on the polyline at arc length ``arc``."""
    index = int(np.searchsorted(cumulative, arc, side="right") - 1)
    index = min(max(index, 0), len(path) - 2)
    segment = cumulative[index + 1] - cumulative[index]
    if segment > 1e-9:
        fraction = (arc - cumulative[index]) / segment
        tangent = (path[index + 1] - path[index]) / segment
    else:
        fraction = 0.0
        tangent = np.array([1.0, 0.0], dtype=np.float64)
    return path[index] + fraction * (path[index + 1] - path[index]), tangent


@dataclass
class WaypointFollower:
    """Feedback velocity controller for a closed waypoint loop.

    It is deliberately high level: the output is [vx, vy, yaw-rate] for an
    existing locomotion policy, never joint commands.
    """

    waypoints_world_xy: list[list[float]]
    max_forward_speed: float = 0.35
    max_yaw_rate: float = 0.55
    max_tracking_yaw_rate: float = 0.15
    minimum_tracking_speed: float = 0.08
    turn_forward_speed: float = 0.0
    heading_gain: float = 1.25
    turn_in_place_threshold_rad: float = 0.18
    waypoint_tolerance: float = 0.30
    settle_duration_s: float = 1.0
    loop_stop_duration_s: float = 0.8
    stop_at_final_waypoint: bool = False
    waypoint_index: int = 1
    loop_count: int = 0
    hold_until_s: float = 0.0
    route_complete: bool = False

    def __post_init__(self) -> None:
        if len(self.waypoints_world_xy) < 2:
            raise ValueError("a waypoint route needs at least two points")
        self.hold_until_s = float(self.settle_duration_s)

    def command(self, position_xy: Iterable[float], yaw_rad: float, timestamp_s: float) -> dict[str, Any]:
        position = np.asarray(list(position_xy), dtype=np.float64)
        if self.route_complete:
            return self._row(0.0, 0.0, "route_complete_stop", position, yaw_rad)
        if timestamp_s < self.hold_until_s:
            label = "initial_settle" if self.loop_count == 0 and self.waypoint_index == 1 else "loop_stop"
            return self._row(0.0, 0.0, label, position, yaw_rad)

        target = np.asarray(self.waypoints_world_xy[self.waypoint_index], dtype=np.float64)
        distance = float(np.linalg.norm(target - position))
        if distance <= self.waypoint_tolerance:
            if self.stop_at_final_waypoint and self.waypoint_index == len(self.waypoints_world_xy) - 1:
                self.route_complete = True
                return self._row(0.0, 0.0, "route_complete_stop", position, yaw_rad)
            self.waypoint_index += 1
            if self.waypoint_index >= len(self.waypoints_world_xy):
                self.waypoint_index = 0
            if self.waypoint_index == 0:
                self.loop_count += 1
                self.hold_until_s = timestamp_s + self.loop_stop_duration_s
                return self._row(0.0, 0.0, "loop_stop", position, yaw_rad)
            target = np.asarray(self.waypoints_world_xy[self.waypoint_index], dtype=np.float64)
            distance = float(np.linalg.norm(target - position))

        desired_yaw = math.atan2(float(target[1] - position[1]), float(target[0] - position[0]))
        error = wrap_angle(desired_yaw - yaw_rad)
        if abs(error) >= self.turn_in_place_threshold_rad:
            yaw_rate = float(np.clip(self.heading_gain * error, -self.max_yaw_rate, self.max_yaw_rate))
            speed = min(self.max_forward_speed, max(0.0, self.turn_forward_speed))
            label = "forward_arc_realign" if speed > 0.0 else "pause_and_realign"
        else:
            yaw_rate = float(
                np.clip(self.heading_gain * error, -self.max_tracking_yaw_rate, self.max_tracking_yaw_rate)
            )
            alignment = max(0.0, math.cos(error))
            speed = min(self.max_forward_speed, max(self.minimum_tracking_speed, distance * 0.7)) * alignment
            label = "track_waypoint"
        return self._row(speed, yaw_rate, label, position, yaw_rad)

    def _row(self, speed: float, yaw_rate: float, label: str, position: np.ndarray, yaw_rad: float) -> dict[str, Any]:
        target = self.waypoints_world_xy[self.waypoint_index]
        return {
            "label": f"loop_{self.loop_count:02d}/{label}",
            "linear_x": float(speed),
            "linear_y": 0.0,
            "angular_z": float(yaw_rate),
            "waypoint_index": int(self.waypoint_index),
            "waypoint_world_xy": list(target),
            "distance_to_waypoint": float(np.linalg.norm(np.asarray(target) - position)),
            "heading_error_rad": wrap_angle(math.atan2(target[1] - position[1], target[0] - position[0]) - yaw_rad),
        }


@dataclass
class PurePursuitController:
    """Curvature lookahead tracker emitting velocity commands for the frozen gait.

    The delivered global route is resampled into a dense polyline and the robot
    chases a point ``lookahead_distance`` ahead of its projection, never the
    nearest waypoint.  The body-frame angle to that point becomes a signed
    curvature and the output command is

        vx       = clip(f(|curvature|), minimum_tracking_speed, max_forward_speed)
        yaw_rate = clip(curvature * vx, +/- max_tracking_yaw_rate)

    ``f`` lowers speed as the demanded curvature rises but keeps a forward floor
    so the policy never receives a mostly-rotation command with near-zero
    forward velocity (the combination that froze the gait on previous runs).
    Both channels are rate-limited so a straight-to-hard-turn snap is replaced
    by a smooth ramp, keeping the command inside the policy's familiar walking
    distribution.
    """

    waypoints_world_xy: list[list[float]]
    max_forward_speed: float = 0.34
    max_tracking_yaw_rate: float = 0.15
    minimum_tracking_speed: float = 0.28
    lookahead_distance: float = 1.2
    curvature_speed_gain: float = 1.0
    vx_rate_limit: float = 1.5
    yaw_rate_limit: float = 1.0
    goal_tolerance: float = 0.25
    deceleration_distance: float = 0.50
    settle_duration_s: float = 1.0
    stop_at_final_waypoint: bool = True
    path_resample_spacing_m: float = 0.05

    route_complete: bool = False
    hold_until_s: float = 0.0
    waypoint_index: int = 0
    loop_count: int = 0

    _path: np.ndarray = field(default=None, repr=False)
    _cumulative: np.ndarray = field(default=None, repr=False)
    _orig_cumulative: np.ndarray = field(default=None, repr=False)
    _nearest_index: int = field(default=0, repr=False)
    _s_progress: float = field(default=0.0, repr=False)
    _last_vx: float = field(default=0.0, repr=False)
    _last_yaw_rate: float = field(default=0.0, repr=False)
    _last_timestamp_s: float | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if len(self.waypoints_world_xy) < 2:
            raise ValueError("a tracking route needs at least two points")
        self._path, self._cumulative = resample_polyline(
            self.waypoints_world_xy, self.path_resample_spacing_m
        )
        # Original waypoints' own cumulative arc length (un-resampled) for progress reporting.
        originals = np.asarray(self.waypoints_world_xy, dtype=np.float64)
        lengths = np.concatenate([[0.0], np.linalg.norm(np.diff(originals, axis=0), axis=1)])
        self._orig_cumulative = np.cumsum(lengths)
        self.hold_until_s = float(self.settle_duration_s)

    def resume_after_recovery(self,arc_m:float,timestamp_s:float,settle_s:float=1.) -> None:
        """Explicit recovery only; normal tracking never writes robot poses."""
        self._s_progress=float(np.clip(arc_m,0.,self._cumulative[-1]))
        self._nearest_index=min(len(self._path)-1,int(np.searchsorted(self._cumulative,self._s_progress)))
        self.route_complete=False
        self._last_vx=self._last_yaw_rate=0.
        self._last_timestamp_s=float(timestamp_s)
        self.hold_until_s=float(timestamp_s)+float(settle_s)

    def command(self, position_xy: Iterable[float], yaw_rad: float, timestamp_s: float) -> dict[str, Any]:
        position = np.asarray(list(position_xy), dtype=np.float64)
        if self.route_complete:
            self._last_vx, self._last_yaw_rate = 0.0, 0.0
            return self._row(0.0, 0.0, "route_complete_stop", position, yaw_rad)
        if timestamp_s < self.hold_until_s:
            label = "initial_settle" if self.loop_count == 0 and self.waypoint_index == 0 else "loop_stop"
            self._last_vx, self._last_yaw_rate = 0.0, 0.0
            self._last_timestamp_s = timestamp_s
            return self._row(0.0, 0.0, label, position, yaw_rad)

        dt = 0.02 if self._last_timestamp_s is None else timestamp_s - self._last_timestamp_s
        self._last_timestamp_s = timestamp_s
        dt = min(max(dt, 1e-4), 1.0)

        # Project onto the resampled path inside a forward-biased window so the
        # progress arc never jumps backward across a corner.
        window_lo = max(0, self._nearest_index - int(0.5 / max(self.path_resample_spacing_m, 1e-3)))
        window_hi = min(len(self._path) - 1, self._nearest_index + int(10.0 / max(self.path_resample_spacing_m, 1e-3)))
        segment_arcs = self._cumulative[window_lo:window_hi + 1]
        window_path = self._path[window_lo:window_hi + 1]
        local_index, s_progress = nearest_polyline_point(position, window_path, segment_arcs)
        self._nearest_index = window_lo + local_index
        self._s_progress = max(s_progress, self._s_progress - 0.05)

        s_goal = float(self._cumulative[-1])
        remaining = max(0.0, s_goal - self._s_progress)
        distance_to_goal = float(np.linalg.norm(position - self._path[-1]))

        # Lookahead target point (clamped to the end of the route).
        s_target = min(self._s_progress + self.lookahead_distance, s_goal)
        target, _tangent = arc_point_and_tangent(self._path, self._cumulative, s_target)

        dx, dy = float(target[0] - position[0]), float(target[1] - position[1])
        x_body = math.cos(yaw_rad) * dx + math.sin(yaw_rad) * dy
        y_body = -math.sin(yaw_rad) * dx + math.cos(yaw_rad) * dy
        distance_to_target = math.hypot(dx, dy)
        alpha = math.atan2(y_body, x_body) if distance_to_target > 1e-6 else 0.0

        lookahead = max(self.lookahead_distance, 0.05)
        curvature = (2.0 * math.sin(alpha) / lookahead) if distance_to_target > 1e-6 else 0.0

        speed_desired = self.max_forward_speed / (1.0 + self.curvature_speed_gain * abs(curvature))
        if remaining < self.deceleration_distance:
            speed_desired = min(
                speed_desired,
                self.max_forward_speed * max(0.0, remaining) / max(self.deceleration_distance, 1e-3),
            )
        speed_desired = float(np.clip(speed_desired, self.minimum_tracking_speed, self.max_forward_speed))
        yaw_desired = float(np.clip(curvature * speed_desired, -self.max_tracking_yaw_rate, self.max_tracking_yaw_rate))

        vx = float(
            np.clip(
                self._last_vx + float(np.clip(speed_desired - self._last_vx, -self.vx_rate_limit * dt, self.vx_rate_limit * dt)),
                0.0,
                self.max_forward_speed,
            )
        )
        yaw_rate = float(
            np.clip(
                self._last_yaw_rate
                + float(np.clip(yaw_desired - self._last_yaw_rate, -self.yaw_rate_limit * dt, self.yaw_rate_limit * dt)),
                -self.max_tracking_yaw_rate,
                self.max_tracking_yaw_rate,
            )
        )
        self._last_vx, self._last_yaw_rate = vx, yaw_rate

        if self.stop_at_final_waypoint and (
            distance_to_goal <= self.goal_tolerance or remaining <= self.goal_tolerance
        ):
            self.route_complete = True
            self._last_vx, self._last_yaw_rate = 0.0, 0.0
            return self._row(0.0, 0.0, "route_complete_stop", position, yaw_rad)

        self.waypoint_index = int(np.searchsorted(self._orig_cumulative, self._s_progress, side="right") - 1)
        self.waypoint_index = min(max(self.waypoint_index, 0), len(self.waypoints_world_xy) - 1)
        return self._row(vx, yaw_rate, "pure_pursuit", position, yaw_rad, target=target, curvature=curvature, alpha=alpha, remaining=remaining)

    def _row(
        self,
        speed: float,
        yaw_rate: float,
        label: str,
        position: np.ndarray,
        yaw_rad: float,
        target: np.ndarray | None = None,
        curvature: float | None = None,
        alpha: float | None = None,
        remaining: float | None = None,
    ) -> dict[str, Any]:
        reported_target = self._path[-1] if target is None else target
        row = {
            "label": f"loop_{self.loop_count:02d}/{label}",
            "linear_x": float(speed),
            "linear_y": 0.0,
            "angular_z": float(yaw_rate),
            "waypoint_index": int(self.waypoint_index),
            "waypoint_world_xy": reported_target.tolist(),
            "distance_to_waypoint": float(np.linalg.norm(reported_target - position)),
            "heading_error_rad": float(alpha) if alpha is not None else 0.0,
        }
        if target is not None:
            row["tracking_debug"] = {
                "lookahead_world_xy": target.tolist(),
                "curvature_rad_per_m": float(curvature),
                "alpha_rad": float(alpha),
                "arc_progress_m": float(self._s_progress),
                "remaining_m": float(remaining),
                "desired_vx": float(self._last_vx),
            }
        return row


@dataclass
class SafeShuttleFollower:
    """Traverse prevalidated road spokes using forward locomotion.

    Each leg records an explicit body heading.  At an endpoint the controller
    first turns the Go2 toward the next target, then walks forward through the
    same admitted corridor.  Optional secondary spokes add only a prevalidated
    corridor at the common origin.  This is a conservative data-capture route,
    not autonomous planning.
    """

    legs: list[dict[str, Any]]
    max_forward_speed: float = 0.35
    max_reverse_speed: float = 0.24
    max_yaw_rate: float = 0.45
    heading_gain: float = 1.25
    heading_tolerance_rad: float = 0.12
    waypoint_tolerance: float = 0.30
    settle_duration_s: float = 1.0
    endpoint_stop_duration_s: float = 0.8
    leg_index: int = 0
    loop_count: int = 0
    hold_until_s: float = 0.0
    route_complete: bool = False
    stop_at_final_waypoint: bool = False

    def __post_init__(self) -> None:
        if len(self.legs) < 2:
            raise ValueError("a safe shuttle needs at least an outbound and return leg")
        required = {"start_world_xy", "target_world_xy", "body_yaw_rad", "reverse", "label"}
        for leg in self.legs:
            missing = required - set(leg)
            if missing:
                raise ValueError(f"safe shuttle leg is missing {sorted(missing)}")
        self.hold_until_s = float(self.settle_duration_s)

    @property
    def waypoints_world_xy(self) -> list[list[float]]:
        return [list(self.legs[0]["start_world_xy"])] + [list(leg["target_world_xy"]) for leg in self.legs]

    def command(self, position_xy: Iterable[float], yaw_rad: float, timestamp_s: float) -> dict[str, Any]:
        position = np.asarray(list(position_xy), dtype=np.float64)
        leg = self.legs[self.leg_index]
        if timestamp_s < self.hold_until_s:
            label = "initial_settle" if self.loop_count == 0 and self.leg_index == 0 else "endpoint_stop"
            return self._row(0.0, 0.0, label, position, yaw_rad, leg)

        target = np.asarray(leg["target_world_xy"], dtype=np.float64)
        distance = float(np.linalg.norm(target - position))
        if distance <= self.waypoint_tolerance:
            self.leg_index += 1
            if self.leg_index >= len(self.legs):
                self.leg_index = 0
                self.loop_count += 1
                self.route_complete = True
            self.hold_until_s = float(timestamp_s) + self.endpoint_stop_duration_s
            leg = self.legs[self.leg_index]
            return self._row(0.0, 0.0, "endpoint_stop", position, yaw_rad, leg)

        desired_yaw = self._desired_body_yaw(position, leg)
        heading_error = wrap_angle(desired_yaw - float(yaw_rad))
        yaw_rate = float(np.clip(self.heading_gain * heading_error, -self.max_yaw_rate, self.max_yaw_rate))
        if abs(heading_error) > self.heading_tolerance_rad:
            return self._row(0.0, yaw_rate, "align_for_leg", position, yaw_rad, leg)
        speed_limit = self.max_reverse_speed if bool(leg["reverse"]) else self.max_forward_speed
        speed = min(float(speed_limit), max(0.08, distance * 0.7))
        if bool(leg["reverse"]):
            speed *= -1.0
        return self._row(speed, yaw_rate, str(leg["label"]), position, yaw_rad, leg)

    def _row(
        self,
        speed: float,
        yaw_rate: float,
        label: str,
        position: np.ndarray,
        yaw_rad: float,
        leg: dict[str, Any],
    ) -> dict[str, Any]:
        target = np.asarray(leg["target_world_xy"], dtype=np.float64)
        desired_yaw = self._desired_body_yaw(position, leg)
        return {
            "label": f"shuttle_{self.loop_count:02d}/{label}",
            "linear_x": float(speed),
            "linear_y": 0.0,
            "angular_z": float(yaw_rate),
            "waypoint_index": int(self.leg_index + 1),
            "waypoint_world_xy": target.tolist(),
            "distance_to_waypoint": float(np.linalg.norm(target - position)),
            "heading_error_rad": wrap_angle(desired_yaw - float(yaw_rad)),
            "desired_body_yaw_rad": desired_yaw,
            "reverse": bool(leg["reverse"]),
            "leg_label": str(leg["label"]),
        }

    @staticmethod
    def _desired_body_yaw(position: np.ndarray, leg: dict[str, Any]) -> float:
        target = np.asarray(leg["target_world_xy"], dtype=np.float64)
        delta = target - position
        if float(np.linalg.norm(delta)) < 1.0e-9:
            return float(leg["body_yaw_rad"])
        travel_yaw = math.atan2(float(delta[1]), float(delta[0]))
        return wrap_angle(travel_yaw - math.pi if bool(leg["reverse"]) else travel_yaw)


@dataclass
class StuckMonitor:
    """Detect sustained commanded motion without pose progress in two stages.

    The rolling window raises a suspicion.  Aborting requires that suspicion to
    remain continuous for ``confirmation_s`` so a slow turn can recover without
    being mislabeled as a terminal freeze.
    """

    window_s: float = 3.0
    command_speed_threshold: float = 0.15
    minimum_progress: float = 0.08
    command_yaw_rate_threshold: float = 0.20
    minimum_yaw_progress: float = 0.15
    confirmation_s: float = 20.0
    _rows: deque[tuple[float, np.ndarray, float, float | None, float]] = field(default_factory=deque)
    _suspected_since_s: float | None = None

    def reset(self):
        self._rows.clear();self._suspected_since_s=None

    def update(
        self,
        timestamp_s: float,
        position_xy: Iterable[float],
        commanded_speed: float,
        yaw_rad: float | None = None,
        commanded_yaw_rate: float = 0.0,
    ) -> dict[str, Any]:
        position = np.asarray(list(position_xy), dtype=np.float64)
        self._rows.append(
            (
                float(timestamp_s),
                position,
                float(commanded_speed),
                float(yaw_rad) if yaw_rad is not None else None,
                float(commanded_yaw_rate),
            )
        )
        while self._rows and timestamp_s - self._rows[0][0] > self.window_s:
            self._rows.popleft()
        window_duration = float(timestamp_s - self._rows[0][0]) if self._rows else 0.0
        progress = float(np.linalg.norm(position - self._rows[0][1])) if self._rows else 0.0
        commanded_translation_fraction = (
            float(np.mean([abs(row[2]) >= self.command_speed_threshold for row in self._rows])) if self._rows else 0.0
        )
        yaw_progress = (
            abs(wrap_angle(float(yaw_rad) - float(self._rows[0][3])))
            if self._rows and yaw_rad is not None and self._rows[0][3] is not None
            else 0.0
        )
        commanded_rotation_fraction = (
            float(np.mean([abs(row[4]) >= self.command_yaw_rate_threshold for row in self._rows]))
            if self._rows and yaw_rad is not None
            else 0.0
        )
        commanded_motion_fraction = (
            float(
                np.mean(
                    [
                        abs(row[2]) >= self.command_speed_threshold
                        or abs(row[4]) >= self.command_yaw_rate_threshold
                        for row in self._rows
                    ]
                )
            )
            if self._rows
            else 0.0
        )
        full_window = window_duration >= self.window_s * 0.95
        translation_applicable = bool(full_window and commanded_translation_fraction >= 0.8)
        rotation_applicable = bool(full_window and commanded_rotation_fraction >= 0.8)
        mixed_motion_applicable = bool(
            full_window
            and commanded_motion_fraction >= 0.8
            and not translation_applicable
            and not rotation_applicable
        )
        translation_stuck = bool(translation_applicable and progress < self.minimum_progress)
        rotation_stuck = bool(rotation_applicable and yaw_progress < self.minimum_yaw_progress)
        # A walking arc deliberately combines translation and rotation.  It is
        # making useful pose progress if either component advances; requiring
        # both thresholds independently would misclassify a tight but healthy
        # turn.  Single-axis commands keep their original strict checks.
        if translation_applicable and rotation_applicable:
            candidate_stuck = translation_stuck and rotation_stuck
            decision_mode = "combined_translation_and_rotation"
        elif translation_applicable:
            candidate_stuck = translation_stuck
            decision_mode = "translation_only"
        elif rotation_applicable:
            candidate_stuck = rotation_stuck
            decision_mode = "rotation_only"
        elif mixed_motion_applicable:
            # A controller may alternate walking and turning within the same
            # window.  Previously neither channel reached the 80% duty-cycle
            # gate, so a completely static robot escaped detection forever.
            candidate_stuck = bool(
                progress < self.minimum_progress and yaw_progress < self.minimum_yaw_progress
            )
            decision_mode = "mixed_translation_rotation"
        else:
            candidate_stuck = False
            decision_mode = "not_applicable"
        if candidate_stuck:
            if self._suspected_since_s is None:
                self._suspected_since_s = float(timestamp_s)
        else:
            self._suspected_since_s = None
        confirmation_elapsed = (
            max(0.0, float(timestamp_s) - self._suspected_since_s)
            if self._suspected_since_s is not None
            else 0.0
        )
        stuck = bool(candidate_stuck and confirmation_elapsed >= self.confirmation_s)
        return {
            "stuck": stuck,
            "suspected_stuck": candidate_stuck,
            "suspected_since_s": self._suspected_since_s,
            "confirmation_elapsed_s": confirmation_elapsed,
            "confirmation_required_s": float(self.confirmation_s),
            "translation_stuck": translation_stuck,
            "rotation_stuck": rotation_stuck,
            "translation_applicable": translation_applicable,
            "rotation_applicable": rotation_applicable,
            "mixed_motion_applicable": mixed_motion_applicable,
            "decision_mode": decision_mode,
            "window_duration_s": window_duration,
            "window_xy_progress": progress,
            "window_yaw_progress_rad": yaw_progress,
            "commanded_translation_fraction": commanded_translation_fraction,
            "commanded_rotation_fraction": commanded_rotation_fraction,
            "commanded_motion_fraction": commanded_motion_fraction,
            "thresholds": {
                "window_s": self.window_s,
                "command_speed": self.command_speed_threshold,
                "minimum_progress": self.minimum_progress,
                "command_yaw_rate": self.command_yaw_rate_threshold,
                "minimum_yaw_progress_rad": self.minimum_yaw_progress,
                "confirmation_s": self.confirmation_s,
            },
        }


@dataclass
class SustainedCollisionMonitor:
    """Detect a continuous non-foot collision while tolerating single-step noise."""

    duration_s: float = 0.5
    _started_at_s: float | None = None

    def reset(self):
        self._started_at_s=None

    def update(self, timestamp_s: float, collision: bool) -> dict[str, Any]:
        timestamp_s = float(timestamp_s)
        if collision:
            if self._started_at_s is None:
                self._started_at_s = timestamp_s
            continuous_duration = max(0.0, timestamp_s - self._started_at_s)
        else:
            self._started_at_s = None
            continuous_duration = 0.0
        return {
            "sustained": bool(collision and continuous_duration >= self.duration_s),
            "continuous_duration_s": continuous_duration,
            "threshold_s": float(self.duration_s),
        }


@dataclass
class PostGoalStopMonitor:
    """Request capture shutdown after retaining a short post-goal hold."""

    hold_s: float = 3.0
    reached_at_s: float | None = None

    def update(self, timestamp_s: float, route_complete: bool) -> dict[str, Any]:
        if route_complete and self.reached_at_s is None:
            self.reached_at_s = float(timestamp_s)
        held_s = (
            max(0.0, float(timestamp_s) - self.reached_at_s)
            if self.reached_at_s is not None
            else 0.0
        )
        return {
            "applicable": self.reached_at_s is not None,
            "goal_reached_at_s": self.reached_at_s,
            "held_s": held_s,
            "hold_s": float(self.hold_s),
            "stop_due": bool(self.reached_at_s is not None and held_s >= self.hold_s),
        }


def classify_contacts(body_names: Iterable[str], force_norms: Iterable[float], threshold: float = 5.0) -> dict[str, Any]:
    """Separate expected foot support from non-foot collision contacts."""
    pairs = [(str(name), float(force)) for name, force in zip(body_names, force_norms)]
    non_foot = [(name, force) for name, force in pairs if "foot" not in name.lower() and force > threshold]
    feet = [(name, force) for name, force in pairs if "foot" in name.lower() and force > threshold]
    return {
        "non_foot_collision": bool(non_foot),
        "non_foot_contacts": [{"body": name, "force_norm": force} for name, force in non_foot],
        "supporting_feet": [{"body": name, "force_norm": force} for name, force in feet],
        "force_threshold": float(threshold),
    }


def _self_test() -> None:
    assert math.isclose(wrap_angle(3.0 * math.pi), -math.pi)
    points = transform_body_waypoints([1.0, 2.0], math.pi / 2.0, [[0.0, 0.0], [2.0, 0.0]])
    assert np.allclose(points, [[1.0, 2.0], [1.0, 4.0]])
    rounded = rounded_right_angle_waypoints([0.0, 0.0], [0.0, 6.0], [8.0, 6.0])
    assert len(rounded) == 15 and np.allclose(rounded[0], [0.0, 0.0])
    assert np.allclose(rounded[-1], [8.0, 6.0]) and np.allclose(rounded[1], [0.0, 4.5])
    follower = WaypointFollower(points, settle_duration_s=0.0)
    row = follower.command([1.0, 2.0], math.pi / 2.0, 0.0)
    assert row["linear_x"] > 0.0 and abs(row["angular_z"]) < 1e-6
    realign = follower.command([1.0, 2.0], math.pi / 2.0 + 0.3, 0.1)
    assert realign["linear_x"] == 0.0 and realign["label"].endswith("pause_and_realign")
    arc_follower = WaypointFollower(points, settle_duration_s=0.0, turn_forward_speed=0.1)
    arc = arc_follower.command([1.0, 2.0], math.pi / 2.0 + 0.3, 0.1)
    assert arc["linear_x"] == 0.1 and arc["label"].endswith("forward_arc_realign")
    minimum_speed_follower = WaypointFollower(
        [[0.0, 0.0], [0.15, 0.0]],
        settle_duration_s=0.0,
        minimum_tracking_speed=0.2,
        waypoint_tolerance=0.01,
    )
    assert minimum_speed_follower.command([0.0, 0.0], 0.0, 0.0)["linear_x"] == 0.2
    one_way = WaypointFollower([[0.0, 0.0], [0.1, 0.0]], settle_duration_s=0.0, stop_at_final_waypoint=True)
    stopped = one_way.command([0.1, 0.0], 0.0, 0.0)
    assert one_way.route_complete and stopped["label"].endswith("route_complete_stop")

    # --- PurePursuitController ---
    resampled_path, resampled_arc = resample_polyline([[0.0, 0.0], [3.0, 0.0]], 0.25)
    assert len(resampled_path) == 13 and np.allclose(resampled_path[0], [0.0, 0.0])
    assert np.allclose(resampled_path[-1], [3.0, 0.0])
    index, arc = nearest_polyline_point([0.2, 0.4], resampled_path, resampled_arc)
    assert 0.15 < arc < 0.35
    tangent_point, tangent = arc_point_and_tangent(resampled_path, resampled_arc, 1.5)
    assert np.allclose(tangent_point, [1.5, 0.0]) and np.allclose(tangent, [1.0, 0.0])

    straight = PurePursuitController([[0.0, 0.0], [3.0, 0.0]], settle_duration_s=0.0)
    straight_row = straight.command([0.0, 0.0], 0.0, 0.0)
    assert straight_row["linear_x"] > 0.0 and abs(straight_row["angular_z"]) < 1e-6
    assert straight_row["label"].endswith("pure_pursuit")
    # The vx rate limit ramps from standstill toward the cruising speed.
    for tick in range(1, 30):
        straight_row = straight.command([0.05 * tick, 0.0], 0.0, 0.02 * tick)
    assert straight_row["linear_x"] > 0.3
    # Forward progress must advance arc monotonically.
    arc_progress = straight_row["tracking_debug"]["arc_progress_m"]
    assert 0.1 < arc_progress < 1.5

    # A large commanded curvature must never drive vx below the forward floor.
    turning = PurePursuitController(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0]],
        settle_duration_s=0.0,
        minimum_tracking_speed=0.22,
        vx_rate_limit=10.0,
        yaw_rate_limit=10.0,
        goal_tolerance=0.4,
    )
    corner_row = None
    for tick in range(6):
        corner_row = turning.command([0.9, 0.0], 0.0, 0.02 * tick)
    assert corner_row["linear_x"] >= 0.22 and corner_row["linear_x"] <= turning.max_forward_speed
    assert abs(corner_row["angular_z"]) > 0.0 and "tracking_debug" in corner_row

    # yaw-rate ramp is bounded by the rate limit on a straight-to-turn transition.
    ramp = PurePursuitController([[0.0, 0.0], [3.0, 0.0]], settle_duration_s=0.0, yaw_rate_limit=1.0)
    ramp.command([0.0, 0.0], 0.0, 0.0)
    ramp_row = ramp.command([0.1, 0.0], 0.3, 0.02)
    assert abs(ramp_row["angular_z"]) <= 1.0 * 0.02 + 1e-9

    # Reaching the final waypoint completes the route and stops.
    finisher = PurePursuitController([[0.0, 0.0], [0.5, 0.0]], settle_duration_s=0.0, goal_tolerance=0.2)
    final_row = finisher.command([0.45, 0.0], 0.0, 0.0)
    assert finisher.route_complete and final_row["linear_x"] == 0.0
    assert final_row["label"].endswith("route_complete_stop")
    shuttle = SafeShuttleFollower(
        [
            {"start_world_xy": [0.0, 0.0], "target_world_xy": [1.0, 0.0], "body_yaw_rad": 0.0,
             "reverse": False, "label": "forward"},
            {"start_world_xy": [1.0, 0.0], "target_world_xy": [0.0, 0.0], "body_yaw_rad": math.pi,
             "reverse": False, "label": "turnaround_return"},
        ],
        settle_duration_s=0.0,
        endpoint_stop_duration_s=0.0,
        waypoint_tolerance=0.1,
    )
    assert shuttle.command([0.0, 0.0], 0.0, 0.0)["linear_x"] > 0.0
    shuttle.command([1.0, 0.0], 0.0, 1.0)
    turnaround = shuttle.command([1.0, 0.0], 0.0, 1.1)
    assert turnaround["linear_x"] == 0.0 and abs(turnaround["angular_z"]) > 0.0
    assert shuttle.command([1.0, 0.0], math.pi, 1.2)["linear_x"] > 0.0
    shuttle.command([0.0, 0.0], math.pi, 2.0)
    assert shuttle.route_complete and shuttle.loop_count == 1
    monitor = StuckMonitor(window_s=1.0, minimum_progress=0.05, confirmation_s=0.0)
    assert not monitor.update(0.0, [0.0, 0.0], 0.3)["stuck"]
    assert monitor.update(1.0, [0.01, 0.0], 0.3)["stuck"]
    angular_monitor = StuckMonitor(window_s=1.0, minimum_yaw_progress=0.1, confirmation_s=0.0)
    assert not angular_monitor.update(0.0, [0.0, 0.0], 0.0, 0.0, 0.5)["stuck"]
    angular_state = angular_monitor.update(1.0, [0.0, 0.0], 0.0, 0.01, 0.5)
    assert angular_state["stuck"] and angular_state["rotation_stuck"]
    arc_monitor = StuckMonitor(
        window_s=1.0, minimum_progress=0.05, minimum_yaw_progress=0.1, confirmation_s=0.0
    )
    assert not arc_monitor.update(0.0, [0.0, 0.0], 0.2, 0.0, 0.5)["stuck"]
    arc_state = arc_monitor.update(1.0, [0.01, 0.0], 0.2, 0.2, 0.5)
    assert not arc_state["stuck"] and arc_state["translation_stuck"]
    assert arc_state["decision_mode"] == "combined_translation_and_rotation"
    stalled_arc = StuckMonitor(
        window_s=1.0, minimum_progress=0.05, minimum_yaw_progress=0.1, confirmation_s=0.0
    )
    stalled_arc.update(0.0, [0.0, 0.0], 0.2, 0.0, 0.5)
    assert stalled_arc.update(1.0, [0.01, 0.0], 0.2, 0.01, 0.5)["stuck"]
    confirmed = StuckMonitor(
        window_s=10.0,
        minimum_progress=0.08,
        minimum_yaw_progress=0.1,
        command_yaw_rate_threshold=0.1,
        confirmation_s=20.0,
    )
    for tick in range(1501):
        state = confirmed.update(tick * 0.02, [0.0, 0.0], 0.24, 0.0, 0.18)
    assert state["suspected_stuck"] and state["stuck"]
    recovering = StuckMonitor(window_s=10.0, confirmation_s=20.0)
    for tick in range(1001):
        timestamp = tick * 0.02
        position = [0.0, 0.0] if timestamp < 19.0 else [0.2, 0.0]
        recovered_state = recovering.update(timestamp, position, 0.3)
    assert not recovered_state["stuck"] and recovered_state["suspected_since_s"] is None
    mixed = StuckMonitor(
        window_s=10.0,
        minimum_progress=0.08,
        minimum_yaw_progress=0.1,
        command_yaw_rate_threshold=0.1,
        confirmation_s=0.0,
    )
    for tick in range(501):
        timestamp = tick * 0.02
        walking = tick % 100 < 50
        mixed_state = mixed.update(
            timestamp,
            [0.0, 0.0],
            0.24 if walking else 0.0,
            0.0,
            0.0 if walking else 0.45,
        )
    assert mixed_state["stuck"] and mixed_state["mixed_motion_applicable"]
    assert mixed_state["decision_mode"] == "mixed_translation_rotation"
    collision_monitor = SustainedCollisionMonitor(duration_s=0.5)
    assert not collision_monitor.update(0.0, True)["sustained"]
    assert not collision_monitor.update(0.3, False)["sustained"]
    assert not collision_monitor.update(1.0, True)["sustained"]
    assert collision_monitor.update(1.5, True)["sustained"]
    post_goal = PostGoalStopMonitor(hold_s=3.0)
    assert not post_goal.update(10.0, False)["stop_due"]
    assert not post_goal.update(20.0, True)["stop_due"]
    assert not post_goal.update(22.9, True)["stop_due"]
    assert post_goal.update(23.0, True)["stop_due"]
    contacts = classify_contacts(["FL_foot", "base"], [40.0, 10.0])
    assert contacts["non_foot_collision"] and len(contacts["supporting_feet"]) == 1


if __name__ == "__main__":
    _self_test()
    print("go2_navigation_control self-test passed")
