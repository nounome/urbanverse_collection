"""Build Go2 planning exclusions from the exact vehicle and People routes.

The combined task currently has no Go2 local dynamic-obstacle avoidance.  A
candidate Go2 reference route must therefore avoid the complete swept corridor
of every configured vehicle and pedestrian route by default.  These helpers
translate the already-approved agent configs into one canonical constraint
format; callers must not hand-copy a second set of route coordinates.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..traffic.routes import compact_route_control_points, resample_polyline


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _vehicle_envelope(catalog_path: Path) -> tuple[float, float, int]:
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    sizes = []
    for row in payload.get("records", []):
        if not bool((row.get("eligibility") or {}).get("traffic_ready", False)):
            continue
        length_width = (row.get("footprint") or {}).get("length_width_m")
        if not isinstance(length_width, list) or len(length_width) != 2:
            continue
        length, width = map(float, length_width)
        if length > 0.0 and width > 0.0:
            sizes.append((length, width))
    if not sizes:
        raise ValueError(f"vehicle catalog has no traffic-ready footprints: {catalog_path}")
    return max(length for length, _ in sizes), max(width for _, width in sizes), len(sizes)


def _automotive_constraints(
    routes_path: Path,
    *,
    go2_radius_m: float,
    clearance_m: float,
    maximum_vehicle_length_m: float,
    maximum_vehicle_width_m: float,
) -> list[dict[str, Any]]:
    payload = json.loads(routes_path.read_text(encoding="utf-8"))
    rows = payload.get("routes", [])
    if not rows:
        raise ValueError(f"automotive route config contains no routes: {routes_path}")
    corridor_radius = maximum_vehicle_width_m * 0.5 + go2_radius_m + clearance_m
    endpoint_radius = (
        math.hypot(maximum_vehicle_length_m, maximum_vehicle_width_m) * 0.5
        + go2_radius_m
        + clearance_m
    )
    constraints = []
    for index, row in enumerate(rows):
        controls = compact_route_control_points(row)
        spacing = min(0.25, float(row.get("sample_spacing_m", 0.25)))
        points = resample_polyline(controls, spacing)
        constraints.append(
            {
                "constraint_id": f"vehicle:{row.get('route_name', f'route_{index:02d}')}",
                "agent_kind": "vehicle",
                "policy": "hard_exclusion",
                "points_xy": points.tolist(),
                "corridor_radius_m": corridor_radius,
                "endpoint_radius_m": endpoint_radius,
                "reason": (
                    "Go2 has no online vehicle avoidance; exclude the maximum audited vehicle "
                    "swept width, spawn/despawn envelope, Go2 body radius and safety margin."
                ),
            }
        )
    return constraints


def _pedestrian_route_rows(config_path: Path) -> list[tuple[str, np.ndarray, list[list[float]]]]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    schema = int(payload.get("schema_version", 0))
    if schema == 2:
        inventory_path = Path(payload["route_inventory"])
        inventory_path = (
            inventory_path if inventory_path.is_absolute() else config_path.parent / inventory_path
        ).resolve()
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        result = []
        for index, row in enumerate(inventory.get("routes", [])):
            points = np.asarray(row["points_xy"], dtype=np.float64)
            result.append((str(row.get("id", f"route_{index:02d}")), points, []))
        return result
    if schema != 1:
        raise ValueError(f"unsupported pedestrian config schema: {schema}")
    result = []
    for index, row in enumerate(payload.get("pedestrians", [])):
        start = row.get("route_start_xyz", row["spawn_xyz"])
        points = np.asarray([start, *row["route_points_xyz"]], dtype=np.float64)[:, :2]
        spawn = np.asarray(row["spawn_xyz"], dtype=np.float64)[:2]
        result.append(
            (
                str(row.get("route_id", row.get("name", f"pedestrian_{index:02d}"))),
                points,
                [spawn.tolist()],
            )
        )
    return result


def _pedestrian_constraints(
    config_path: Path,
    *,
    go2_radius_m: float,
    pedestrian_radius_m: float,
    clearance_m: float,
) -> list[dict[str, Any]]:
    corridor_radius = pedestrian_radius_m + go2_radius_m + clearance_m
    constraints = []
    seen: set[tuple[tuple[float, float], ...]] = set()
    for route_id, points, spawn_points in _pedestrian_route_rows(config_path):
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError(f"invalid pedestrian route for Go2 constraint: {route_id}")
        key = tuple((round(float(x), 6), round(float(y), 6)) for x, y in points)
        if key in seen:
            continue
        seen.add(key)
        constraints.append(
            {
                "constraint_id": f"pedestrian:{route_id}",
                "agent_kind": "pedestrian",
                "policy": "hard_exclusion",
                "points_xy": points.tolist(),
                "spawn_points_xy": spawn_points,
                "corridor_radius_m": corridor_radius,
                "endpoint_radius_m": corridor_radius,
                "reason": (
                    "Default combined runs do not rely on Go2 avoiding pedestrians; exclude the "
                    "People body radius, Go2 body radius and safety margin."
                ),
            }
        )
    return constraints


def build_joint_route_constraints(
    *,
    automotive_routes_path: Path,
    vehicle_catalog_path: Path,
    pedestrian_config_path: Path | None,
    go2_radius_m: float = 0.40,
    vehicle_clearance_m: float = 0.50,
    pedestrian_radius_m: float = 0.35,
    pedestrian_clearance_m: float = 0.25,
) -> dict[str, Any]:
    """Return hard exclusions derived from the exact combined-run configs."""

    automotive_routes_path = Path(automotive_routes_path).resolve()
    vehicle_catalog_path = Path(vehicle_catalog_path).resolve()
    pedestrian_config_path = (
        Path(pedestrian_config_path).resolve() if pedestrian_config_path is not None else None
    )
    maximum_length, maximum_width, model_count = _vehicle_envelope(vehicle_catalog_path)
    constraints = _automotive_constraints(
        automotive_routes_path,
        go2_radius_m=go2_radius_m,
        clearance_m=vehicle_clearance_m,
        maximum_vehicle_length_m=maximum_length,
        maximum_vehicle_width_m=maximum_width,
    )
    if pedestrian_config_path is not None:
        constraints.extend(
            _pedestrian_constraints(
                pedestrian_config_path,
                go2_radius_m=go2_radius_m,
                pedestrian_radius_m=pedestrian_radius_m,
                clearance_m=pedestrian_clearance_m,
            )
        )
    counts = {
        kind: sum(row["agent_kind"] == kind for row in constraints)
        for kind in ("vehicle", "pedestrian")
    }
    return {
        "schema_version": 1,
        "default_policy": "hard_exclusion",
        "go2_online_dynamic_avoidance": False,
        "constraints": constraints,
        "counts": counts,
        "parameters_m": {
            "go2_radius": go2_radius_m,
            "vehicle_clearance": vehicle_clearance_m,
            "pedestrian_radius": pedestrian_radius_m,
            "pedestrian_clearance": pedestrian_clearance_m,
            "maximum_vehicle_length": maximum_length,
            "maximum_vehicle_width": maximum_width,
            "traffic_ready_vehicle_model_count": model_count,
        },
        "sources": {
            "automotive_routes": str(automotive_routes_path),
            "automotive_routes_sha256": _sha256(automotive_routes_path),
            "vehicle_catalog": str(vehicle_catalog_path),
            "vehicle_catalog_sha256": _sha256(vehicle_catalog_path),
            "pedestrian_config": str(pedestrian_config_path) if pedestrian_config_path else None,
            "pedestrian_config_sha256": _sha256(pedestrian_config_path) if pedestrian_config_path else None,
        },
    }


def rasterize_joint_route_constraints(
    xs: np.ndarray,
    ys: np.ndarray,
    constraints: list[dict[str, Any]],
    meters_per_unit_effective: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Rasterize hard route corridors into the planner grid."""

    xx, yy = np.meshgrid(np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64))
    mask = np.zeros(xx.shape, dtype=bool)
    rows = []
    mpu = float(meters_per_unit_effective)
    for constraint in constraints:
        if constraint.get("policy", "hard_exclusion") != "hard_exclusion":
            continue
        points = np.asarray(constraint["points_xy"], dtype=np.float64)
        radius = float(constraint["corridor_radius_m"]) / mpu
        endpoint_radius = float(constraint.get("endpoint_radius_m", radius * mpu)) / mpu
        local = np.zeros_like(mask)
        for left, right in zip(points[:-1], points[1:]):
            delta = right - left
            length_sq = float(np.dot(delta, delta))
            if length_sq <= 1.0e-12:
                continue
            t = np.clip(((xx - left[0]) * delta[0] + (yy - left[1]) * delta[1]) / length_sq, 0.0, 1.0)
            dx = xx - (left[0] + t * delta[0])
            dy = yy - (left[1] + t * delta[1])
            local |= dx * dx + dy * dy <= radius * radius
        endpoint_points = [points[0], points[-1]] + [
            np.asarray(point, dtype=np.float64)
            for point in constraint.get("spawn_points_xy", [])
        ]
        for point in endpoint_points:
            local |= (xx - point[0]) ** 2 + (yy - point[1]) ** 2 <= endpoint_radius**2
        mask |= local
        rows.append(
            {
                "constraint_id": constraint["constraint_id"],
                "agent_kind": constraint["agent_kind"],
                "policy": "hard_exclusion",
                "excluded_cell_count": int(np.count_nonzero(local)),
                "corridor_radius_m": float(constraint["corridor_radius_m"]),
                "endpoint_radius_m": float(constraint.get("endpoint_radius_m", radius * mpu)),
            }
        )
    return mask, rows
