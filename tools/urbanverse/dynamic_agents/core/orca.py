#!/usr/bin/env python3
"""Pure-numpy planning and reciprocal avoidance for the UrbanVerse demo.

This is intentionally a small, inspectable prototype.  It follows the paper's
high-level recipe (occupancy grid -> start/goal pairs -> collision-free paths ->
online reciprocal avoidance), but it is not the authors' GPU ORCA code.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class AgentKind:
    name: str
    radius: float
    preferred_speed: float
    color: tuple[int, int, int]


KINDS = {
    "pedestrian": AgentKind("pedestrian", 0.34, 1.15, (250, 94, 92)),
    "car": AgentKind("car", 0.78, 2.15, (55, 156, 245)),
    "wheelchair": AgentKind("wheelchair", 0.48, 0.90, (174, 93, 238)),
    "scooter": AgentKind("scooter", 0.42, 1.55, (255, 177, 52)),
}


def largest_component(valid: np.ndarray) -> np.ndarray:
    """Return a mask for the largest 8-connected True component."""
    valid = np.asarray(valid, dtype=bool)
    seen = np.zeros_like(valid)
    best: list[tuple[int, int]] = []
    ny, nx = valid.shape
    for j in range(ny):
        for i in range(nx):
            if not valid[j, i] or seen[j, i]:
                continue
            stack = [(j, i)]
            seen[j, i] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cj, ci = stack.pop()
                cells.append((cj, ci))
                for dj in (-1, 0, 1):
                    for di in (-1, 0, 1):
                        if not (di or dj):
                            continue
                        nj, ni = cj + dj, ci + di
                        if 0 <= nj < ny and 0 <= ni < nx and valid[nj, ni] and not seen[nj, ni]:
                            seen[nj, ni] = True
                            stack.append((nj, ni))
            if len(cells) > len(best):
                best = cells
    result = np.zeros_like(valid)
    for j, i in best:
        result[j, i] = True
    return result


def inflate_free_space(valid: np.ndarray, radius_cells: int) -> np.ndarray:
    """Conservatively erode free space without requiring scipy."""
    valid = np.asarray(valid, dtype=bool)
    if radius_cells <= 0:
        return valid.copy()
    padded = np.pad(valid, radius_cells, constant_values=False)
    result = np.ones_like(valid)
    for dj in range(-radius_cells, radius_cells + 1):
        for di in range(-radius_cells, radius_cells + 1):
            if di * di + dj * dj > radius_cells * radius_cells:
                continue
            result &= padded[
                radius_cells + dj : radius_cells + dj + valid.shape[0],
                radius_cells + di : radius_cells + di + valid.shape[1],
            ]
    return result


def astar(valid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    """Eight-neighbour A* with diagonal corner-cut prevention."""
    valid = np.asarray(valid, dtype=bool)
    if not valid[start] or not valid[goal]:
        return []
    ny, nx = valid.shape
    frontier: list[tuple[float, float, tuple[int, int]]] = [(0.0, 0.0, start)]
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    cost = {start: 0.0}
    while frontier:
        _, current_cost, current = heapq.heappop(frontier)
        if current == goal:
            path = [current]
            while path[-1] != start:
                path.append(parent[path[-1]])
            return list(reversed(path))
        if current_cost > cost.get(current, math.inf) + 1e-9:
            continue
        cj, ci = current
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                if not (di or dj):
                    continue
                nj, ni = cj + dj, ci + di
                if not (0 <= nj < ny and 0 <= ni < nx and valid[nj, ni]):
                    continue
                if di and dj and (not valid[cj, ni] or not valid[nj, ci]):
                    continue
                step = math.sqrt(2.0) if di and dj else 1.0
                candidate = current_cost + step
                node = (nj, ni)
                if candidate + 1e-9 >= cost.get(node, math.inf):
                    continue
                cost[node] = candidate
                parent[node] = current
                heuristic = math.hypot(goal[0] - nj, goal[1] - ni)
                heapq.heappush(frontier, (candidate + heuristic, candidate, node))
    return []


def _nearest_cell(mask: np.ndarray, target_j: float, target_i: float) -> tuple[int, int]:
    cells = np.argwhere(mask)
    if not len(cells):
        raise RuntimeError("no valid cells")
    index = int(np.argmin((cells[:, 0] - target_j) ** 2 + (cells[:, 1] - target_i) ** 2))
    return int(cells[index, 0]), int(cells[index, 1])


def make_crossing_routes(
    valid: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    agent_count: int,
) -> list[np.ndarray]:
    """Create opposing and crossing A* routes through the largest component."""
    component = largest_component(valid)
    cells = np.argwhere(component)
    if len(cells) < 32:
        raise RuntimeError(f"largest traversable component is too small: {len(cells)} cells")
    j0, i0 = cells.min(axis=0)
    j1, i1 = cells.max(axis=0)
    cj, ci = cells.mean(axis=0)
    # Four bidirectional axes; small offsets avoid identical start positions.
    templates = [
        ((cj - 0.18 * (j1 - j0), i0), (cj + 0.18 * (j1 - j0), i1)),
        ((cj + 0.18 * (j1 - j0), i1), (cj - 0.18 * (j1 - j0), i0)),
        ((j0, ci - 0.20 * (i1 - i0)), (j1, ci + 0.20 * (i1 - i0))),
        ((j1, ci + 0.20 * (i1 - i0)), (j0, ci - 0.20 * (i1 - i0))),
        ((j0, i0 + 0.18 * (i1 - i0)), (j1, i1 - 0.18 * (i1 - i0))),
        ((j1, i1 - 0.18 * (i1 - i0)), (j0, i0 + 0.18 * (i1 - i0))),
        ((j0, i1 - 0.18 * (i1 - i0)), (j1, i0 + 0.18 * (i1 - i0))),
        ((j1, i0 + 0.18 * (i1 - i0)), (j0, i1 - 0.18 * (i1 - i0))),
    ]
    routes: list[np.ndarray] = []
    for index in range(agent_count):
        start_target, goal_target = templates[index % len(templates)]
        # Deterministic lane offset for additional agents.
        offset = ((index // len(templates)) % 3 - 1) * 2.0
        start = _nearest_cell(component, start_target[0] + offset, start_target[1] - offset)
        goal = _nearest_cell(component, goal_target[0] - offset, goal_target[1] + offset)
        cells_path = astar(component, start, goal)
        if len(cells_path) < 4:
            raise RuntimeError(f"A* failed for crossing route {index}: {start}->{goal}")
        points = np.asarray([[float(xs[i]), float(ys[j])] for j, i in cells_path], dtype=np.float64)
        # Remove grid stair-stepping while preserving collision-free endpoints.
        keep = [0]
        for k in range(1, len(points) - 1):
            a = points[k] - points[keep[-1]]
            b = points[k + 1] - points[k]
            cross = abs(a[0] * b[1] - a[1] * b[0])
            if cross > 1e-7:
                keep.append(k)
        keep.append(len(points) - 1)
        routes.append(points[keep])
    return routes


def make_laned_routes(
    valid: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    kind_names: list[str],
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Plan class-separated, direction-specific routes on a horizontal road.

    Cars use the two central lanes. Vulnerable road users use the two outer
    bands. Every route is still admitted by A* on the measured free-space map.
    """
    component = largest_component(valid)
    cells = np.argwhere(component)
    if len(cells) < 32:
        raise RuntimeError(f"largest traversable component is too small: {len(cells)} cells")
    j0, i0 = cells.min(axis=0)
    j1, i1 = cells.max(axis=0)
    lane_fractions = {
        "sidewalk_south": 0.14,
        "vehicle_eastbound": 0.40,
        "vehicle_westbound": 0.60,
        "sidewalk_north": 0.86,
    }
    counters = {name: 0 for name in lane_fractions}
    routes: list[np.ndarray] = []
    assignments: list[dict[str, Any]] = []
    for index, kind_name in enumerate(kind_names):
        if kind_name == "car":
            lane_id = "vehicle_eastbound" if counters["vehicle_eastbound"] <= counters["vehicle_westbound"] else "vehicle_westbound"
        else:
            lane_id = "sidewalk_south" if counters["sidewalk_south"] <= counters["sidewalk_north"] else "sidewalk_north"
        lane_index = counters[lane_id]
        counters[lane_id] += 1
        eastbound = lane_id in ("vehicle_eastbound", "sidewalk_south")
        target_j = j0 + lane_fractions[lane_id] * (j1 - j0)
        # Multiple agents in one lane get a small deterministic longitudinal
        # stagger while retaining opposite endpoints and the same traffic flow.
        inset = min(1 + lane_index * 4, max(1, (i1 - i0) // 4))
        start_i = i0 + inset if eastbound else i1 - inset
        goal_i = i1 - 1 if eastbound else i0 + 1
        start = _nearest_cell(component, target_j, start_i)
        goal = _nearest_cell(component, target_j, goal_i)
        cells_path = astar(component, start, goal)
        if len(cells_path) < 4:
            raise RuntimeError(f"A* failed for lane {lane_id}: {start}->{goal}")
        points = np.asarray([[float(xs[i]), float(ys[j])] for j, i in cells_path], dtype=np.float64)
        keep = [0]
        for k in range(1, len(points) - 1):
            a = points[k] - points[keep[-1]]
            b = points[k + 1] - points[k]
            if abs(a[0] * b[1] - a[1] * b[0]) > 1e-7:
                keep.append(k)
        keep.append(len(points) - 1)
        routes.append(points[keep])
        assignments.append(
            {
                "agent_index": index,
                "kind": kind_name,
                "traffic_class": "vehicle" if kind_name == "car" else "vulnerable_road_user",
                "lane_id": lane_id,
                "direction": "eastbound" if eastbound else "westbound",
            }
        )
    return routes, assignments


def initialize_agents(routes: list[np.ndarray], kind_names: list[str]) -> list[dict[str, Any]]:
    agents = []
    for index, (route, kind_name) in enumerate(zip(routes, kind_names)):
        kind = KINDS[kind_name]
        direction = route[1] - route[0]
        agents.append(
            {
                "id": index,
                "kind": kind_name,
                "radius": kind.radius,
                "preferred_speed": kind.preferred_speed,
                "position": route[0].copy(),
                "velocity": np.zeros(2, dtype=np.float64),
                "heading": math.atan2(direction[1], direction[0]),
                "route": route.copy(),
                "waypoint": 1,
                "direction": 1,
                "goal_reversals": 0,
                "trajectory": [route[0].copy()],
            }
        )
    return agents


def _det(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _normalise(value: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(value))
    if length < 1e-12:
        return np.asarray([1.0, 0.0], dtype=np.float64)
    return value / length


def _linear_program_1(
    lines: list[tuple[np.ndarray, np.ndarray]],
    line_no: int,
    radius: float,
    optimum: np.ndarray,
    direction_optimum: bool,
) -> tuple[bool, np.ndarray]:
    point, direction = lines[line_no]
    dot = float(point @ direction)
    discriminant = dot * dot + radius * radius - float(point @ point)
    if discriminant < 0.0:
        return False, np.zeros(2, dtype=np.float64)
    root = math.sqrt(discriminant)
    left = -dot - root
    right = -dot + root
    for other_point, other_direction in lines[:line_no]:
        denominator = _det(direction, other_direction)
        numerator = _det(other_direction, point - other_point)
        if abs(denominator) <= 1e-12:
            if numerator < 0.0:
                return False, np.zeros(2, dtype=np.float64)
            continue
        t = numerator / denominator
        if denominator >= 0.0:
            right = min(right, t)
        else:
            left = max(left, t)
        if left > right:
            return False, np.zeros(2, dtype=np.float64)
    if direction_optimum:
        result = point + (right if float(optimum @ direction) > 0.0 else left) * direction
    else:
        t = float(direction @ (optimum - point))
        result = point + float(np.clip(t, left, right)) * direction
    return True, result


def _linear_program_2(
    lines: list[tuple[np.ndarray, np.ndarray]],
    radius: float,
    optimum: np.ndarray,
    direction_optimum: bool = False,
) -> tuple[int, np.ndarray]:
    if direction_optimum:
        result = _normalise(optimum) * radius
    elif float(optimum @ optimum) > radius * radius:
        result = _normalise(optimum) * radius
    else:
        result = optimum.copy()
    for line_no, (point, direction) in enumerate(lines):
        if _det(direction, point - result) <= 0.0:
            continue
        previous = result.copy()
        feasible, result = _linear_program_1(lines, line_no, radius, optimum, direction_optimum)
        if not feasible:
            return line_no, previous
    return len(lines), result


def _linear_program_3(
    lines: list[tuple[np.ndarray, np.ndarray]],
    begin_line: int,
    radius: float,
    result: np.ndarray,
) -> np.ndarray:
    distance = 0.0
    for line_no in range(begin_line, len(lines)):
        point, direction = lines[line_no]
        violation = _det(direction, point - result)
        if violation <= distance:
            continue
        projected: list[tuple[np.ndarray, np.ndarray]] = []
        for other_no in range(line_no):
            other_point, other_direction = lines[other_no]
            determinant = _det(direction, other_direction)
            if abs(determinant) <= 1e-12:
                if float(direction @ other_direction) > 0.0:
                    continue
                projected_point = 0.5 * (point + other_point)
            else:
                projected_point = point + (
                    _det(other_direction, point - other_point) / determinant
                ) * direction
            projected_direction = _normalise(other_direction - direction)
            projected.append((projected_point, projected_direction))
        previous = result.copy()
        failed, candidate = _linear_program_2(
            projected,
            radius,
            np.asarray([-direction[1], direction[0]], dtype=np.float64),
            direction_optimum=True,
        )
        result = previous if failed < len(projected) else candidate
        distance = _det(direction, point - result)
    return result


def _orca_lines(
    agent: dict[str, Any],
    neighbours: list[dict[str, Any]],
    dt: float,
    time_horizon: float,
    neighbor_distance: float,
    safety_margin: float,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    lines: list[tuple[np.ndarray, np.ndarray]] = []
    conflicts = 0
    position = np.asarray(agent["position"], dtype=np.float64)
    velocity = np.asarray(agent["velocity"], dtype=np.float64)
    for other in neighbours:
        relative_position = np.asarray(other["position"], dtype=np.float64) - position
        distance_sq = float(relative_position @ relative_position)
        if distance_sq > neighbor_distance * neighbor_distance:
            continue
        relative_velocity = velocity - np.asarray(other["velocity"], dtype=np.float64)
        combined_radius = float(agent["radius"] + other["radius"] + safety_margin)
        combined_sq = combined_radius * combined_radius
        if distance_sq > combined_sq:
            inverse_horizon = 1.0 / time_horizon
            w = relative_velocity - inverse_horizon * relative_position
            w_sq = float(w @ w)
            dot = float(w @ relative_position)
            if dot < 0.0 and dot * dot > combined_sq * w_sq:
                unit_w = _normalise(w)
                direction = np.asarray([unit_w[1], -unit_w[0]], dtype=np.float64)
                u = (combined_radius * inverse_horizon - math.sqrt(max(w_sq, 0.0))) * unit_w
            else:
                leg = math.sqrt(max(distance_sq - combined_sq, 0.0))
                if _det(relative_position, w) > 0.0:
                    direction = np.asarray(
                        [
                            relative_position[0] * leg - relative_position[1] * combined_radius,
                            relative_position[0] * combined_radius + relative_position[1] * leg,
                        ],
                        dtype=np.float64,
                    ) / max(distance_sq, 1e-12)
                else:
                    direction = -np.asarray(
                        [
                            relative_position[0] * leg + relative_position[1] * combined_radius,
                            -relative_position[0] * combined_radius + relative_position[1] * leg,
                        ],
                        dtype=np.float64,
                    ) / max(distance_sq, 1e-12)
                u = float(relative_velocity @ direction) * direction - relative_velocity
            conflicts += int(float(np.linalg.norm(u)) > 1e-7)
        else:
            inverse_step = 1.0 / max(dt, 1e-6)
            w = relative_velocity - inverse_step * relative_position
            unit_w = _normalise(w)
            direction = np.asarray([unit_w[1], -unit_w[0]], dtype=np.float64)
            u = (combined_radius * inverse_step - float(np.linalg.norm(w))) * unit_w
            conflicts += 1
        # Dynamic agents split the correction. A prescribed robot does not
        # participate in reciprocity, so the dynamic agent takes the full u.
        responsibility = 1.0 if bool(other.get("nonreciprocal", False)) else 0.5
        lines.append((velocity + responsibility * u, direction))
    return lines, conflicts


def _preferred_velocity(agent: dict[str, Any]) -> np.ndarray:
    route = agent["route"]
    position = agent["position"]
    waypoint = int(agent["waypoint"])
    direction = int(agent["direction"])
    target = route[waypoint]
    delta = target - position
    while np.linalg.norm(delta) < 0.45:
        next_waypoint = waypoint + direction
        if next_waypoint < 0 or next_waypoint >= len(route):
            direction *= -1
            agent["direction"] = direction
            agent["goal_reversals"] += 1
            next_waypoint = waypoint + direction
        waypoint = next_waypoint
        agent["waypoint"] = waypoint
        target = route[waypoint]
        delta = target - position
    distance = max(float(np.linalg.norm(delta)), 1e-9)
    return delta / distance * float(agent["preferred_speed"])


def step_orca(
    agents: list[dict[str, Any]],
    dt: float,
    external_agents: list[dict[str, Any]] | None = None,
    time_horizon: float = 2.5,
    neighbor_distance: float = 6.0,
    safety_margin: float = 0.12,
    pair_filter: Callable[[dict[str, Any], dict[str, Any]], bool] | None = None,
) -> dict[str, float]:
    """Advance disc agents with the standard ORCA half-plane construction.

    The solver is a deterministic CPU implementation of the same disc-ORCA
    formulation used by RVO2. External agents may be prescribed and marked
    ``nonreciprocal``; in that case simulated agents take full responsibility.
    """
    external_agents = external_agents or []
    preferred = [_preferred_velocity(agent) for agent in agents]
    corrected: list[np.ndarray] = []
    constraints = 0
    predicted_conflicts = 0
    infeasible_initial_programs = 0
    candidate_evaluations = 0

    # ORCA only needs agents in the configured centre-distance neighbourhood.
    # The old implementation rebuilt an all-to-all list for every agent and
    # let ``_orca_lines`` discard distant entries.  A uniform grid preserves
    # the exact deterministic neighbour order while avoiding that O(N^2)
    # candidate construction for larger traffic populations.
    cell_size = max(float(neighbor_distance), 1.0e-6)

    def cell(position: np.ndarray) -> tuple[int, int]:
        point = np.asarray(position, dtype=np.float64)
        return math.floor(float(point[0]) / cell_size), math.floor(float(point[1]) / cell_size)

    buckets: dict[tuple[int, int], list[int]] = {}
    for other_index, other in enumerate(agents):
        buckets.setdefault(cell(other["position"]), []).append(other_index)
    external_buckets: dict[tuple[int, int], list[int]] = {}
    for other_index, other in enumerate(external_agents):
        external_buckets.setdefault(cell(other["position"]), []).append(other_index)

    for index, (agent, optimum) in enumerate(zip(agents, preferred)):
        centre = cell(agent["position"])
        nearby_indices: set[int] = set()
        nearby_external_indices: set[int] = set()
        for offset_x in (-1, 0, 1):
            for offset_y in (-1, 0, 1):
                key = centre[0] + offset_x, centre[1] + offset_y
                nearby_indices.update(buckets.get(key, ()))
                nearby_external_indices.update(external_buckets.get(key, ()))
        neighbours = [
            agents[other_index]
            for other_index in sorted(nearby_indices)
            if other_index != index
            and (pair_filter is None or pair_filter(agent, agents[other_index]))
        ]
        neighbours.extend(external_agents[other_index] for other_index in sorted(nearby_external_indices))
        candidate_evaluations += len(neighbours)
        lines, conflicts = _orca_lines(
            agent,
            neighbours,
            dt,
            time_horizon,
            neighbor_distance,
            safety_margin,
        )
        constraints += len(lines)
        predicted_conflicts += conflicts
        failed_line, velocity = _linear_program_2(lines, float(agent["preferred_speed"]), optimum)
        if failed_line < len(lines):
            infeasible_initial_programs += 1
            velocity = _linear_program_3(lines, failed_line, float(agent["preferred_speed"]), velocity)
        corrected.append(velocity)

    for agent, velocity, preferred_velocity in zip(agents, corrected, preferred):
        # Road vehicles cannot instantly rotate or translate sideways like a
        # holonomic disc.  Callers may lock them to the local route tangent;
        # ORCA then controls their longitudinal speed (including yielding and
        # stopping) without turning the body into roadside obstacles.
        if bool(agent.get("route_locked_velocity", False)):
            preferred_direction = _normalise(preferred_velocity)
            forward_speed = max(0.0, float(velocity @ preferred_direction))
            velocity = preferred_direction * min(forward_speed, float(agent["preferred_speed"]))
        agent["velocity"] = velocity
        agent["position"] = np.asarray(agent["position"], dtype=np.float64) + velocity * dt

    minimum_pair_clearance = math.inf
    minimum_external_clearance = math.inf
    physical_overlaps = 0
    for i, agent in enumerate(agents):
        for other in agents[i + 1 :]:
            clearance = float(np.linalg.norm(other["position"] - agent["position"])) - float(
                agent["radius"] + other["radius"]
            )
            minimum_pair_clearance = min(minimum_pair_clearance, clearance)
            physical_overlaps += int(clearance < -1e-6)
        for other in external_agents:
            clearance = float(np.linalg.norm(np.asarray(other["position"]) - agent["position"])) - float(
                agent["radius"] + other["radius"]
            )
            minimum_external_clearance = min(minimum_external_clearance, clearance)
            physical_overlaps += int(clearance < -1e-6)

    for agent in agents:
        velocity = agent["velocity"]
        if float(np.linalg.norm(velocity)) > 0.08:
            target_heading = math.atan2(float(velocity[1]), float(velocity[0]))
            error = (target_heading - float(agent["heading"]) + math.pi) % (2.0 * math.pi) - math.pi
            agent["heading"] += float(np.clip(error, -2.8 * dt, 2.8 * dt))
        agent["trajectory"].append(agent["position"].copy())
    return {
        "minimum_pair_clearance": float(minimum_pair_clearance if math.isfinite(minimum_pair_clearance) else 0.0),
        "minimum_external_clearance": float(
            minimum_external_clearance if math.isfinite(minimum_external_clearance) else 0.0
        ),
        "predicted_conflicts": float(predicted_conflicts),
        "orca_constraint_count": float(constraints),
        "infeasible_initial_programs": float(infeasible_initial_programs),
        "physical_overlaps": float(physical_overlaps),
        "overlap_projections": 0.0,
        "candidate_evaluations": float(candidate_evaluations),
    }


def step_reciprocal_avoidance(
    agents: list[dict[str, Any]],
    dt: float,
    time_horizon: float = 2.2,
    neighbor_distance: float = 5.0,
    safety_margin: float = 0.14,
) -> dict[str, float]:
    """Advance disc agents with deterministic RVO-style reciprocal corrections.

    This predicts pairwise closest approach and splits the velocity correction
    between both agents.  Exact ORCA would construct half-plane constraints and
    solve a linear program; the demo deliberately labels this approximation.
    """
    preferred = [_preferred_velocity(agent) for agent in agents]
    predicted_conflicts = 0
    # Synchronous velocity-obstacle sampling.  Each agent assumes neighbours
    # keep their current/preferred velocity for one horizon and selects the
    # closest safe alternative. A tiny right-pass preference breaks symmetric
    # head-on deadlocks while the synchronous update remains reciprocal.
    corrected: list[np.ndarray] = []
    angle_options = [0, -15, 15, -30, 30, -45, 45, -70, 70, -100, 100, 180]
    speed_scales = [1.0, 0.78, 0.55, 0.30, 0.0]
    for i, (agent, pref) in enumerate(zip(agents, preferred)):
        base_angle = math.atan2(float(pref[1]), float(pref[0]))
        candidates = []
        for angle_deg in angle_options:
            for scale in speed_scales:
                angle = base_angle + math.radians(angle_deg)
                velocity = np.asarray([math.cos(angle), math.sin(angle)]) * float(agent["preferred_speed"]) * scale
                candidates.append((angle_deg, scale, velocity))
        best_score = math.inf
        best_velocity = np.zeros(2)
        for angle_deg, scale, velocity in candidates:
            score = float(np.sum((velocity - pref) ** 2))
            score += 0.015 * (abs(angle_deg) / 15.0) ** 2 + 0.05 * (1.0 - scale) ** 2
            if angle_deg > 0:  # prefer passing on the agent's local right
                score += 0.018
            for j, other in enumerate(agents):
                if i == j:
                    continue
                delta = other["position"] - agent["position"]
                distance = float(np.linalg.norm(delta))
                if distance > neighbor_distance:
                    continue
                other_velocity = other["velocity"] if np.linalg.norm(other["velocity"]) > 0.05 else preferred[j]
                relative_velocity = other_velocity - velocity
                rv2 = float(relative_velocity @ relative_velocity)
                tau = 0.0 if rv2 < 1e-9 else float(np.clip(-(delta @ relative_velocity) / rv2, 0.0, time_horizon))
                future_distance = float(np.linalg.norm(delta + relative_velocity * tau))
                combined = float(agent["radius"] + other["radius"] + safety_margin)
                if future_distance < combined:
                    if angle_deg == 0 and scale == 1.0:
                        predicted_conflicts += 1
                    penetration = combined - future_distance
                    score += 90.0 * penetration * penetration + 2.0 * (time_horizon - tau) / time_horizon
                physical = float(agent["radius"] + other["radius"])
                if distance < physical + 0.08:
                    score += 300.0 * (physical + 0.08 - distance) ** 2
            if score < best_score:
                best_score = score
                best_velocity = velocity
        corrected.append(best_velocity)

    for agent, velocity in zip(agents, corrected):
        agent["velocity"] = velocity
        agent["position"] = agent["position"] + velocity * dt

    # Positional projection is only a numerical safety net; log every use.
    overlap_projections = 0
    for _ in range(4):
        changed = False
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                delta = agents[j]["position"] - agents[i]["position"]
                distance = float(np.linalg.norm(delta))
                required = float(agents[i]["radius"] + agents[j]["radius"])
                if distance + 1e-9 >= required:
                    continue
                overlap_projections += 1
                changed = True
                if distance < 1e-9:
                    direction = np.asarray([1.0, 0.0]) if (i + j) % 2 == 0 else np.asarray([0.0, 1.0])
                else:
                    direction = delta / distance
                push = direction * (0.5 * (required - distance + 1e-4))
                agents[i]["position"] -= push
                agents[j]["position"] += push
        if not changed:
            break

    min_clearance = math.inf
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            distance = float(np.linalg.norm(agents[j]["position"] - agents[i]["position"]))
            min_clearance = min(min_clearance, distance - float(agents[i]["radius"] + agents[j]["radius"]))

    for agent in agents:
        velocity = agent["velocity"]
        if float(np.linalg.norm(velocity)) > 0.08:
            target_heading = math.atan2(float(velocity[1]), float(velocity[0]))
            error = (target_heading - float(agent["heading"]) + math.pi) % (2.0 * math.pi) - math.pi
            agent["heading"] += float(np.clip(error, -2.8 * dt, 2.8 * dt))
        agent["trajectory"].append(agent["position"].copy())
    return {
        "minimum_pair_clearance": float(min_clearance if math.isfinite(min_clearance) else 0.0),
        "predicted_conflicts": float(predicted_conflicts),
        "overlap_projections": float(overlap_projections),
    }
