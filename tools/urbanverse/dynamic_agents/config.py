"""Validated, path-aware configuration for reusable UrbanVerse traffic scenes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _resolve(base: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _validate_static_rows(rows, label: str) -> None:
    for row in rows:
        corners = np.asarray(row.get("corners_xy"), dtype=np.float64)
        if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
            raise ValueError(f"{label} body must be a finite four-corner OBB: {row.get('path')}")


@dataclass(frozen=True)
class TrafficSceneConfig:
    """Scene-specific data; algorithms remain in the shared package."""

    path: Path
    scene_id: str
    source_usd: Path
    source_tar: Path | None
    vehicle_catalog: Path
    automotive_routes: Path
    go2_reference_route: Path
    pedestrian_config: Path | None
    road_polygons_xy: tuple[np.ndarray, ...]
    road_footprint_inventory: Path | None
    fallback_ground_z_m: float
    fallback_ground_provenance: str
    static_vehicle_bodies: tuple[dict[str, Any], ...]
    static_obstacle_inventory: Path | None
    overview: dict[str, Any]
    traffic: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> "TrafficSceneConfig":
        path = Path(path).resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError(f"unsupported traffic scene schema: {payload.get('schema_version')}")
        base = path.parent
        required = (
            "scene_id",
            "source_usd",
            "vehicle_catalog",
            "automotive_routes",
            "go2_reference_route",
            "road_surface",
            "traffic",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"traffic scene config is missing: {', '.join(missing)}")
        road = payload["road_surface"]
        road_footprint_inventory = _resolve(base, road.get("footprint_inventory"))
        if road_footprint_inventory is not None:
            if not road_footprint_inventory.is_file():
                raise FileNotFoundError(
                    "road_surface.footprint_inventory does not exist: "
                    f"{road_footprint_inventory}"
                )
            footprint_payload = json.loads(
                road_footprint_inventory.read_text(encoding="utf-8")
            )
            polygon_rows = footprint_payload.get("polygons_xy")
            if not isinstance(polygon_rows, list):
                raise ValueError("road footprint inventory must contain a polygons_xy list")
        else:
            polygon_rows = road.get("polygons_xy")
            if not isinstance(polygon_rows, list):
                raise ValueError(
                    "road_surface requires polygons_xy or footprint_inventory"
                )
        polygons = tuple(np.asarray(row, dtype=np.float64) for row in polygon_rows)
        no_lane = (payload['traffic'].get('enabled') is False and
                   payload['traffic'].get('disabled_reason') == 'not_applicable_no_lane')
        if (not polygons and not no_lane) or any(poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3 for poly in polygons):
            raise ValueError("road_surface.polygons_xy must contain XY polygons with >=3 vertices")
        source_usd = _resolve(base, payload["source_usd"])
        vehicle_catalog = _resolve(base, payload["vehicle_catalog"])
        automotive_routes = _resolve(base, payload["automotive_routes"])
        go2_reference_route = _resolve(base, payload["go2_reference_route"])
        pedestrian_config = _resolve(base, payload.get("pedestrian_config"))
        assert source_usd is not None and vehicle_catalog is not None
        assert automotive_routes is not None and go2_reference_route is not None
        for label, candidate in (
            ("source_usd", source_usd),
            ("vehicle_catalog", vehicle_catalog),
            ("automotive_routes", automotive_routes),
            ("go2_reference_route", go2_reference_route),
        ):
            if not candidate.is_file():
                raise FileNotFoundError(f"{label} does not exist: {candidate}")
        if pedestrian_config is not None and not pedestrian_config.is_file():
            raise FileNotFoundError(f"pedestrian_config does not exist: {pedestrian_config}")
        static_rows = tuple(payload.get("static_vehicle_bodies", ()))
        _validate_static_rows(static_rows, "inline static")
        static_obstacle_inventory = _resolve(base, payload.get("static_obstacle_inventory"))
        if static_obstacle_inventory is not None and not static_obstacle_inventory.is_file():
            raise FileNotFoundError(
                f"static_obstacle_inventory does not exist: {static_obstacle_inventory}"
            )
        if static_obstacle_inventory is not None:
            inventory = json.loads(static_obstacle_inventory.read_text(encoding="utf-8"))
            inventory_rows = inventory.get("bodies", inventory) if isinstance(inventory, dict) else inventory
            if not isinstance(inventory_rows, list):
                raise ValueError("static_obstacle_inventory must contain a bodies list")
            _validate_static_rows(inventory_rows, "inventory static")
        traffic = dict(payload["traffic"])
        if no_lane:
            if polygons or int(traffic.get('vehicle_count', -1)) != 0:
                raise ValueError('No-Lane mode requires empty Lane polygons and zero cars')
        elif int(traffic.get("vehicle_count", 0)) < 1:
            raise ValueError("traffic.vehicle_count must be positive")
        if int(traffic.get("junction_group_count", 1)) < 1:
            raise ValueError("traffic.junction_group_count must be positive")
        return cls(
            path=path,
            scene_id=str(payload["scene_id"]),
            source_usd=source_usd,
            source_tar=_resolve(base, payload.get("source_tar")),
            vehicle_catalog=vehicle_catalog,
            automotive_routes=automotive_routes,
            go2_reference_route=go2_reference_route,
            pedestrian_config=pedestrian_config,
            road_polygons_xy=polygons,
            road_footprint_inventory=road_footprint_inventory,
            fallback_ground_z_m=float(road["fallback_ground_z_m"]),
            fallback_ground_provenance=str(road["fallback_ground_provenance"]),
            static_vehicle_bodies=static_rows,
            static_obstacle_inventory=static_obstacle_inventory,
            overview=dict(payload.get("overview", {})),
            traffic=traffic,
        )


class PolygonRoadFootprint:
    """Fast point containment for configured, visually audited road polygons."""

    spatial_cell_size_m = 5.0

    def __init__(self, polygons: tuple[np.ndarray, ...], fallback_ground_z_m: float):
        self.polygons = tuple(np.asarray(poly, dtype=np.float64) for poly in polygons)
        self.minimum = np.asarray([poly.min(axis=0) for poly in self.polygons])
        self.maximum = np.asarray([poly.max(axis=0) for poly in self.polygons])
        self.fallback_ground_z_m = float(fallback_ground_z_m)
        self.query_count = 0
        self.candidate_count = 0

    @staticmethod
    def _inside(point: np.ndarray, polygon: np.ndarray, tolerance: float) -> bool:
        # Boundary-inclusive ray casting. The distance tolerance is handled by
        # an inexpensive segment projection before the parity test.
        for left, right in zip(polygon, np.roll(polygon, -1, axis=0)):
            edge = right - left
            t = float(np.clip(np.dot(point - left, edge) / max(np.dot(edge, edge), 1.0e-12), 0.0, 1.0))
            if np.linalg.norm(point - (left + t * edge)) <= tolerance:
                return True
        inside = False
        x, y = map(float, point)
        for left, right in zip(polygon, np.roll(polygon, -1, axis=0)):
            x1, y1 = map(float, left)
            x2, y2 = map(float, right)
            if (y1 > y) != (y2 > y):
                crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
                if x <= crossing_x:
                    inside = not inside
        return inside

    def contains(self, point: np.ndarray, tolerance: float = 0.03) -> bool:
        point = np.asarray(point, dtype=np.float64)
        candidates = np.nonzero(
            np.all(point >= self.minimum - tolerance, axis=1)
            & np.all(point <= self.maximum + tolerance, axis=1)
        )[0]
        self.query_count += 1
        self.candidate_count += int(len(candidates))
        return any(self._inside(point, self.polygons[index], tolerance) for index in candidates)

    def height(self, point: np.ndarray, tolerance: float = 0.03) -> float:
        if not self.contains(point, tolerance):
            raise RuntimeError(f"no configured road surface at {np.asarray(point).tolist()}")
        return self.fallback_ground_z_m
