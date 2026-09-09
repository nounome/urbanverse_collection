"""Scene-independent automotive route loading and ORCA speed capping."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..vehicles.traffic_geometry import footprint, sat_intersects
from ..vehicles.route_planning import STATIC_SAFETY_MARGIN_M

from ..core.orca import step_orca
from ..core.vehicle_motion import AutomotiveRoute, cumulative_lengths


def resample_polyline(points: np.ndarray, spacing_m: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("an automotive route requires at least two XY control points")
    arc = cumulative_lengths(points)
    if arc[-1] <= 0.0:
        raise ValueError("automotive route length must be positive")
    samples = np.linspace(0.0, arc[-1], max(2, int(math.ceil(arc[-1] / spacing_m)) + 1))
    return np.column_stack(
        (np.interp(samples, arc, points[:, 0]), np.interp(samples, arc, points[:, 1]))
    )


def expand_control_primitives(row: dict[str, Any]) -> np.ndarray:
    """Expand compact line/arc route primitives into a smooth control polyline."""
    if "start_xy" not in row or not row.get("control_primitives"):
        raise ValueError("primitive automotive routes require start_xy and control_primitives")
    points = [np.asarray(row["start_xy"], dtype=np.float64)]
    for primitive in row["control_primitives"]:
        kind = str(primitive["kind"])
        if kind == "line":
            generated = [np.asarray(primitive["end_xy"], dtype=np.float64)]
        elif kind == "arc":
            center = np.asarray(primitive["center_xy"], dtype=np.float64)
            radius = float(primitive["radius_m"])
            count = int(primitive.get("sample_count", 33))
            if radius <= 0.0 or count < 2:
                raise ValueError("arc route primitives require positive radius and sample_count >= 2")
            angles = np.radians(
                np.linspace(
                    float(primitive["start_angle_deg"]),
                    float(primitive["end_angle_deg"]),
                    count,
                )
            )
            generated = [
                center + radius * np.asarray([math.cos(angle), math.sin(angle)])
                for angle in angles
            ]
        else:
            raise ValueError(f"unsupported automotive route primitive: {kind}")
        for point in generated:
            if point.shape != (2,):
                raise ValueError("automotive route primitive points must be XY pairs")
            if np.linalg.norm(point - points[-1]) > 1.0e-6:
                points.append(point)
    if len(points) < 2:
        raise ValueError("automotive route primitives produced fewer than two points")
    return np.asarray(points, dtype=np.float64)


def compact_route_control_points(row: dict[str, Any]) -> np.ndarray:
    """Return the authored control polyline for any supported route schema."""
    if "xy" in row:
        return np.asarray(row["xy"], dtype=np.float64)
    if "control_primitives" in row:
        return expand_control_primitives(row)
    if "control_points_xy" in row:
        return np.asarray(row["control_points_xy"], dtype=np.float64)
    raise ValueError("automotive routes require xy, control_points_xy, or control_primitives")


def load_automotive_routes(
    path: Path,
    road: Any,
    static: list[tuple[str, np.ndarray]],
    specs: list[dict[str, Any]],
) -> tuple[list[AutomotiveRoute], list[dict[str, Any]]]:
    """Load compact or dense routes and repeat full-body admission checks."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = document.get("routes", [])
    if not rows:
        raise RuntimeError("automotive route config contains no routes")
    routes: list[AutomotiveRoute] = []
    metadata: list[dict[str, Any]] = []
    # Admission uses the widest assigned body for each base route cohort.
    for index, row in enumerate(rows):
        spacing = float(row.get("sample_spacing_m", 0.25))
        control_points = compact_route_control_points(row)
        xy = control_points if "xy" in row else resample_polyline(control_points, spacing)
        if "yaw_deg" in row:
            yaw = np.radians(np.asarray(row["yaw_deg"], dtype=np.float64))
        else:
            delta = np.gradient(xy, axis=0)
            yaw = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
        route = AutomotiveRoute(
            xy=xy,
            yaw=yaw,
            arc_m=cumulative_lengths(xy),
            minimum_turning_radius_m=float(row.get("minimum_turning_radius_m", 5.5)),
            closed=bool(row.get("closed", False)),
        )
        if route.closed:
            if np.linalg.norm(route.xy[0]-route.xy[-1])>1.e-5:
                raise ValueError('closed route must contain its exact closing point')
            heading_error=(route.yaw[-1]-route.yaw[0]+np.pi)%(2*np.pi)-np.pi
            if abs(heading_error)>np.radians(2.):
                raise ValueError('closed route heading seam exceeds 2 degrees')
        assigned = [spec for spec_index, spec in enumerate(specs) if spec_index % len(rows) == index]
        if not assigned:
            assigned = [specs[index % len(specs)]]
        for spec in assigned:
            for sample_index, (position, heading) in enumerate(zip(route.xy, route.yaw)):
                body = footprint(position, float(heading), float(spec["length"]), float(spec["width"]))
                if hasattr(road,'pose_clear') and not road.pose_clear(position,float(heading),float(spec['length']),float(spec['width'])):
                    raise RuntimeError(f'mesh body admission failed at route {index} sample {sample_index}')
                if not all(road.contains(corner) for corner in body):
                    raise RuntimeError(
                        f"route {row.get('route_name', index)} left configured road at sample {sample_index}"
                    )
                if any(
                    sat_intersects(body, obstacle, margin=STATIC_SAFETY_MARGIN_M)
                    for _, obstacle in static
                ):
                    raise RuntimeError(
                        f"route {row.get('route_name', index)} hit a static vehicle at sample {sample_index}"
                    )
        routes.append(route)
        metadata.append(
            {
                **{
                    key: value
                    for key, value in row.items()
                    if key not in (
                        "xy",
                        "yaw_deg",
                        "start_xy",
                        "control_points_xy",
                        "control_primitives",
                    )
                },
                "id": index,
                "route_name": str(row.get("route_name", f"route_{index:02d}")),
                "start_xy": route.xy[0].tolist(),
                "goal_xy": route.xy[-1].tolist(),
                "start_yaw_deg": math.degrees(float(route.yaw[0])),
                "goal_yaw_deg": math.degrees(float(route.yaw[-1])),
                "length_m": route.length_m,
                "cache_validation": "full-OBB configured-road/static pass",
            }
        )
    return routes, metadata


def orca_longitudinal_caps(active: list[dict[str, Any]], dt: float):
    """Use disc ORCA for longitudinal caps, never for vehicle steering."""
    if not active:
        return {}, {"orca_constraint_count": 0.0, "predicted_conflicts": 0.0, "candidate_evaluations": 0.0}
    shadows = []
    for agent in active:
        forward = np.asarray([math.cos(agent["heading"]), math.sin(agent["heading"])])
        shadows.append(
            {
                "id": agent["id"],
                "kind": "car",
                "position": agent["position"].copy(),
                "velocity": forward * float(agent["speed"]),
                "heading": float(agent["heading"]),
                "radius": float(agent["orca_radius"]),
                "length": float(agent["length"]),
                "width": float(agent["width"]),
                "preferred_speed": float(agent["desired_speed"]),
                "route": np.asarray([agent["position"], agent["position"] + forward * 12.0]),
                "route_locked_velocity": True,
                "waypoint": 1,
                "direction": 1,
                "goal_reversals": 0,
                "trajectory": [agent["position"].copy()],
            }
        )

    def relevant_pair(first: dict[str, Any], second: dict[str, Any]) -> bool:
        """Exclude only parallel lanes whose physical widths cannot overlap.

        The enclosing discs are intentionally conservative for junctions and
        same-lane following, but they overlap across Scene09's adjacent
        opposing lanes and can stop both streams forever.  Parallel bodies
        with a measured lateral gap retain the exact downstream OBB guard and
        do not need a disc-ORCA constraint.
        """

        first_forward = np.asarray(
            [math.cos(first["heading"]), math.sin(first["heading"])],
            dtype=np.float64,
        )
        second_forward = np.asarray(
            [math.cos(second["heading"]), math.sin(second["heading"])],
            dtype=np.float64,
        )
        if abs(float(np.dot(first_forward, second_forward))) < math.cos(
            math.radians(10.0)
        ):
            return True
        relative = np.asarray(second["position"]) - np.asarray(first["position"])
        lateral = abs(
            float(first_forward[0] * relative[1] - first_forward[1] * relative[0])
        )
        required = 0.5 * (float(first["width"]) + float(second["width"])) + 0.35
        return lateral <= required

    metrics = step_orca(
        shadows,
        dt,
        time_horizon=4.0,
        neighbor_distance=16.0,
        safety_margin=0.25,
        pair_filter=relevant_pair,
    )
    caps = {}
    for agent, shadow in zip(active, shadows):
        forward = np.asarray([math.cos(agent["heading"]), math.sin(agent["heading"])])
        caps[agent["id"]] = max(0.0, float(np.dot(shadow["velocity"], forward)))
    return caps, metrics
