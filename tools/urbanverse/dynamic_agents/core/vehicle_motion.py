#!/usr/bin/env python3
"""Automotive-constrained planning and motion for UrbanVerse traffic agents.

The planner is a small forward-only hybrid A*: unlike the older 2-D grid A*,
its state contains vehicle heading and its motion primitives obey a minimum
turning radius.  Runtime motion uses a kinematic bicycle controller.  ORCA is
deliberately kept outside this module and may only supply a longitudinal speed
cap; it never directly changes XY position or body yaw.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Callable

import numpy as np


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def cumulative_lengths(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


@dataclass(frozen=True)
class AutomotiveRoute:
    xy: np.ndarray
    yaw: np.ndarray
    arc_m: np.ndarray
    minimum_turning_radius_m: float
    closed: bool = False

    @property
    def length_m(self) -> float:
        return float(self.arc_m[-1])

    def full_laps_since(self, start_arc_m: float, current_arc_m: float) -> int:
        """Count full travel from a staggered start, not crossings of arc zero."""
        return int(max(0., current_arc_m-start_arc_m)//self.length_m)


def hybrid_astar_route(
    start_xy: np.ndarray,
    start_yaw: float,
    goal_xy: np.ndarray,
    goal_yaw: float,
    is_pose_clear: Callable[[np.ndarray, float], bool],
    xy_to_cell: Callable[[np.ndarray], tuple[int, int]],
    minimum_turning_radius_m: float,
    heading_bins: int = 24,
    primitive_length_m: float = 1.5,
    integration_step_m: float = 0.25,
    goal_tolerance_m: float = 2.0,
    goal_heading_tolerance_deg: float = 30.0,
    maximum_expansions: int = 300_000,
) -> AutomotiveRoute:
    """Plan a forward-only route with bounded curvature.

    The returned route includes every integration sample, not only search
    nodes.  Consequently the renderer/controller never sees the old 45-degree
    grid corners.
    """
    start_xy = np.asarray(start_xy, dtype=np.float64)
    goal_xy = np.asarray(goal_xy, dtype=np.float64)
    if not is_pose_clear(start_xy, start_yaw):
        raise RuntimeError(f"hybrid A* start pose is obstructed: {start_xy.tolist()}")

    def key(position: np.ndarray, yaw: float) -> tuple[int, int, int]:
        j, i = xy_to_cell(position)
        heading = int(round((yaw % (2.0 * math.pi)) / (2.0 * math.pi) * heading_bins)) % heading_bins
        return j, i, heading

    start_key = key(start_xy, start_yaw)
    poses = {start_key: (start_xy.copy(), float(start_yaw))}
    costs = {start_key: 0.0}
    parents: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    parent_segments: dict[tuple[int, int, int], list[tuple[np.ndarray, float]]] = {}
    queue = [(float(np.linalg.norm(goal_xy - start_xy)), 0.0, start_key)]
    curvature_choices = np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0]) / minimum_turning_radius_m
    substeps = max(2, int(math.ceil(primitive_length_m / integration_step_m)))
    ds = primitive_length_m / substeps
    terminal_key = None

    for _ in range(maximum_expansions):
        if not queue:
            break
        _, current_cost, current_key = heapq.heappop(queue)
        if current_cost > costs.get(current_key, math.inf) + 1e-9:
            continue
        position, yaw = poses[current_key]
        if (
            float(np.linalg.norm(goal_xy - position)) <= goal_tolerance_m
            and abs(wrap_angle(yaw - goal_yaw)) <= math.radians(goal_heading_tolerance_deg)
        ):
            terminal_key = current_key
            break

        for curvature in curvature_choices:
            candidate_position = position.copy()
            candidate_yaw = yaw
            segment: list[tuple[np.ndarray, float]] = []
            valid = True
            for _substep in range(substeps):
                midpoint_yaw = candidate_yaw + 0.5 * float(curvature) * ds
                candidate_position = candidate_position + np.asarray(
                    [math.cos(midpoint_yaw), math.sin(midpoint_yaw)], dtype=np.float64
                ) * ds
                candidate_yaw = wrap_angle(candidate_yaw + float(curvature) * ds)
                if not is_pose_clear(candidate_position, candidate_yaw):
                    valid = False
                    break
                segment.append((candidate_position.copy(), candidate_yaw))
            if not valid:
                continue
            candidate_key = key(candidate_position, candidate_yaw)
            candidate_cost = current_cost + primitive_length_m + 0.15 * abs(float(curvature)) * primitive_length_m
            if candidate_cost >= costs.get(candidate_key, math.inf) - 1e-9:
                continue
            costs[candidate_key] = candidate_cost
            poses[candidate_key] = (candidate_position.copy(), candidate_yaw)
            parents[candidate_key] = current_key
            parent_segments[candidate_key] = segment
            heuristic = float(np.linalg.norm(goal_xy - candidate_position))
            heuristic += 0.5 * minimum_turning_radius_m * abs(wrap_angle(goal_yaw - candidate_yaw))
            heapq.heappush(queue, (candidate_cost + heuristic, candidate_cost, candidate_key))

    if terminal_key is None:
        raise RuntimeError(
            f"hybrid A* exhausted {maximum_expansions} expansions: "
            f"{start_xy.tolist()} -> {goal_xy.tolist()}"
        )

    segments: list[list[tuple[np.ndarray, float]]] = []
    cursor = terminal_key
    while cursor != start_key:
        segments.append(parent_segments[cursor])
        cursor = parents[cursor]
    samples = [(start_xy.copy(), float(start_yaw))]
    for segment in reversed(segments):
        samples.extend(segment)
    xy = np.asarray([sample[0] for sample in samples], dtype=np.float64)
    yaw = np.unwrap(np.asarray([sample[1] for sample in samples], dtype=np.float64))
    return AutomotiveRoute(xy=xy, yaw=yaw, arc_m=cumulative_lengths(xy), minimum_turning_radius_m=minimum_turning_radius_m)


def route_pose_at_arc(route: AutomotiveRoute, arc_m: float) -> tuple[np.ndarray, float]:
    arc_m = float(arc_m % route.length_m if route.closed else np.clip(arc_m, 0.0, route.length_m))
    index = int(np.searchsorted(route.arc_m, arc_m, side="right") - 1)
    index = min(index, len(route.xy) - 2)
    length = max(float(route.arc_m[index + 1] - route.arc_m[index]), 1e-9)
    alpha = (arc_m - float(route.arc_m[index])) / length
    position = route.xy[index] * (1.0 - alpha) + route.xy[index + 1] * alpha
    yaw = float(route.yaw[index] * (1.0 - alpha) + route.yaw[index + 1] * alpha)
    return position, wrap_angle(yaw)


def nearest_forward_arc(route: AutomotiveRoute, position: np.ndarray, previous_arc_m: float) -> float:
    """Project onto a local forward route window without jumping backward."""
    if route.closed:
        # Keep progress unbounded; wrap only position lookup. Search adjacent
        # laps so the seam is an ordinary continuation, never a root reset.
        lap=int(math.floor(previous_arc_m/route.length_m))
        arcs=np.concatenate([route.arc_m[:-1]+i*route.length_m for i in (lap-1,lap,lap+1)])
        xy=np.tile(route.xy[:-1],(3,1))
        keep=(arcs>=previous_arc_m-1.)&(arcs<=previous_arc_m+min(14.,route.length_m*.45))
        if not keep.any():return float(previous_arc_m)
        choices=arcs[keep]
        nearest=int(np.argmin(np.linalg.norm(xy[keep]-np.asarray(position),axis=1)))
        return max(float(previous_arc_m),float(choices[nearest]))
    start = max(0, int(np.searchsorted(route.arc_m, max(0.0, previous_arc_m - 1.0))) - 1)
    stop = min(len(route.xy), int(np.searchsorted(route.arc_m, previous_arc_m + 14.0)) + 2)
    local = route.xy[start:stop]
    index = start + int(np.argmin(np.linalg.norm(local - np.asarray(position), axis=1)))
    return max(float(previous_arc_m), float(route.arc_m[index]))


def apply_junction_reservation(active, speed_caps, owner_id, approach_m=32.0, exit_m=18.0, dt_s=0.0):
    """Serialize conflicting routes through a junction without starvation."""
    changed = False
    active_by_id = {agent["id"]: agent for agent in active}
    if owner_id is not None:
        owner = active_by_id.get(owner_id)
        if owner is None or owner["route_arc_m"] > owner["junction_arc_m"] + exit_m:
            owner_id = None
            changed = True
    candidates = [
        agent for agent in active
        if agent["junction_arc_m"] - approach_m <= agent["route_arc_m"] <= agent["junction_arc_m"] + exit_m
    ]
    for agent in candidates:
        if agent["id"] != owner_id:
            agent["junction_wait_s"] = float(agent.get("junction_wait_s", 0.0)) + float(dt_s)
    if owner_id is None and candidates:
        winner = max(candidates, key=lambda item: (
            float(item.get("junction_wait_s", 0.0)),
            item["route_arc_m"] - item["junction_arc_m"],
            -item["id"],
        ))
        owner_id = winner["id"]
        winner["junction_wait_s"] = 0.0
        changed = True
    for agent in candidates:
        speed_caps[agent["id"]] = agent["desired_speed"] if agent["id"] == owner_id else 0.0
    return owner_id, changed


def apply_junction_group_reservation(
    active,
    speed_caps,
    owner_group,
    approach_m=32.0,
    exit_m=18.0,
    stop_line_m=10.0,
    dt_s=0.0,
):
    """Allow a compatible lane group to use the junction concurrently.

    ``junction_group`` is assigned by the scene-specific route manager.  Cars
    in the same west-east stream share one group, cars on the two parallel
    south-north lanes share another, and the turning stream is exclusive.  A
    group reservation retains deterministic starvation protection without the
    old, unrealistic one-car-at-a-time junction bottleneck.
    """
    changed = False
    candidates = [
        agent
        for agent in active
        if agent["junction_arc_m"] - approach_m
        <= agent["route_arc_m"]
        <= agent["junction_arc_m"] + exit_m
    ]
    candidate_groups = {int(agent["junction_group"]) for agent in candidates}
    if owner_group is not None and int(owner_group) not in candidate_groups:
        owner_group = None
        changed = True
    for agent in candidates:
        if int(agent["junction_group"]) != owner_group:
            agent["junction_wait_s"] = float(agent.get("junction_wait_s", 0.0)) + float(dt_s)
    if owner_group is None and candidates:
        winner = max(
            candidates,
            key=lambda item: (
                float(item.get("junction_wait_s", 0.0)),
                item["route_arc_m"] - item["junction_arc_m"],
                -item["id"],
            ),
        )
        owner_group = int(winner["junction_group"])
        changed = True
    for agent in candidates:
        if int(agent["junction_group"]) == owner_group:
            speed_caps[agent["id"]] = min(
                float(agent["desired_speed"]),
                float(speed_caps.get(agent["id"], agent["desired_speed"])),
            )
            agent["junction_wait_s"] = 0.0
        elif agent["route_arc_m"] >= agent["junction_arc_m"] - stop_line_m:
            speed_caps[agent["id"]] = 0.0
    return owner_group, changed


def apply_junction_signal_control(
    active,
    speed_caps,
    green_group,
    clearance_phase=False,
    approach_m=32.0,
    exit_m=18.0,
    stop_line_m=10.0,
):
    """Apply a green/clearance traffic-signal phase without blocking approaches."""
    for agent in active:
        distance_to_junction = float(agent["junction_arc_m"] - agent["route_arc_m"])
        if not -exit_m <= distance_to_junction <= approach_m:
            continue
        before_stop_line = distance_to_junction > stop_line_m
        owns_green = int(agent["junction_group"]) == int(green_group)
        if before_stop_line:
            if owns_green and clearance_phase:
                speed_caps[agent["id"]] = 0.0
            continue
        if not owns_green:
            speed_caps[agent["id"]] = 0.0


def bicycle_step(
    state: dict,
    route: AutomotiveRoute,
    dt: float,
    desired_speed_mps: float,
    speed_cap_mps: float,
    wheelbase_m: float,
    maximum_steering_rad: float,
    maximum_steering_rate_radps: float = 0.32,
    maximum_acceleration_mps2: float = 1.1,
    maximum_braking_mps2: float = 2.4,
    lookahead_m: float = 4.0,
    use_target_chord: bool = False,
    steer_before_drive: bool = False,
) -> dict:
    """Advance one non-holonomic vehicle using pure pursuit + bicycle motion."""
    position = np.asarray(state["position"], dtype=np.float64)
    yaw = float(state["heading"])
    speed = float(state.get("speed", 0.0))
    steering = float(state.get("steering", 0.0))
    route_arc = nearest_forward_arc(route, position, float(state.get("route_arc_m", 0.0)))
    target, _ = route_pose_at_arc(route, route_arc + lookahead_m)
    target_angle = math.atan2(float(target[1] - position[1]), float(target[0] - position[0]))
    alpha = wrap_angle(target_angle - yaw)
    pursuit_distance=max(float(np.linalg.norm(target-position)),.05) if use_target_chord else lookahead_m
    desired_steering = math.atan2(2.0 * wheelbase_m * math.sin(alpha), pursuit_distance)
    desired_steering = float(np.clip(desired_steering, -maximum_steering_rad, maximum_steering_rad))
    steering += float(np.clip(desired_steering - steering, -maximum_steering_rate_radps * dt, maximum_steering_rate_radps * dt))

    local_arc=route_arc%route.length_m if route.closed else route_arc
    route_index = min(len(route.yaw) - 2, int(np.searchsorted(route.arc_m, local_arc, side="right") - 1))
    local_curvature = abs(wrap_angle(float(route.yaw[route_index + 1] - route.yaw[route_index]))) / max(
        float(route.arc_m[route_index + 1] - route.arc_m[route_index]), 1e-6
    )
    curvature_speed = math.sqrt(1.35 / max(local_curvature, 1e-5))
    target_speed = min(float(desired_speed_mps), float(speed_cap_mps), curvature_speed)
    if steer_before_drive and speed<1e-4 and abs(desired_steering-steering)>.002:
        # Rate-limited steering at rest, no root rotation/position write. A car
        # born on a bend must not accelerate with straight wheels first.
        target_speed=0.
    delta_speed = target_speed - speed
    limit = maximum_acceleration_mps2 * dt if delta_speed >= 0.0 else maximum_braking_mps2 * dt
    speed = max(0.0, speed + float(np.clip(delta_speed, -limit, limit)))

    midpoint_yaw = yaw + 0.5 * speed / max(wheelbase_m, 1e-6) * math.tan(steering) * dt
    position = position + np.asarray([math.cos(midpoint_yaw), math.sin(midpoint_yaw)]) * speed * dt
    yaw = wrap_angle(yaw + speed / max(wheelbase_m, 1e-6) * math.tan(steering) * dt)
    return {
        **state,
        "position": position,
        "heading": yaw,
        "speed": speed,
        "steering": steering,
        "route_arc_m": route_arc,
        "velocity": np.asarray([math.cos(yaw), math.sin(yaw)]) * speed,
    }
