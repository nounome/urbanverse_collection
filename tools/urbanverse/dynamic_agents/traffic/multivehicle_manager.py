#!/usr/bin/env python3
"""Reusable UrbanVerse kinematic traffic manager for Go2 integration runs.

The manager deliberately reuses the calibrated vehicle bodies, per-vehicle
occupancy grids, swept-capsule A* routes and CPU ORCA implementation exercised
by ``vehicles/route_planning.py``.  Vehicle collision evidence
is geometric OBB evidence because the authored scene vehicles are visual
shells, not wheel/articulation models.
"""

from __future__ import annotations

import json
import math
import copy
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..vehicles.traffic_geometry import (
    LaneFootprint,
    footprint,
    polygon_clearance,
    sat_intersects,
)
from ..vehicles.route_planning import (
    MIXED_VEHICLE_IDS,
    STATIC_SAFETY_MARGIN_M,
    NativeVehicle,
    build_clearance_fields,
    build_grid,
    physical_road_height,
    plan_routes,
    static_vehicles,
)
from ..core.orca import step_orca
from ..core.vehicle_pose import transfer_front_heading_from_corresponding_points
from ..config import PolygonRoadFootprint, TrafficSceneConfig


# The original white convertible (0e249...) is geometry-valid but renders only
# through its shadow in the combined Isaac Lab RTX camera.  Use the independently
# front-validated blue convertible with nearly identical dimensions for the
# integrated Go2 traffic route; keep the original standalone evidence intact.
INTEGRATED_MIXED_VEHICLE_IDS = list(MIXED_VEHICLE_IDS)
INTEGRATED_MIXED_VEHICLE_IDS[1] = "69e5de3f68804fbc900557b7740b680c"

# A larger appearance pool selected from the 32 assets whose body centre,
# signed front direction, support height and footprint passed the Scene 10 RTX
# vehicle audit.  Very wide trucks are intentionally excluded until routes for
# their larger swept envelopes are planned separately.
DIVERSE_TRAFFIC_VEHICLE_IDS = (
    "955edc733c6d44fabc0ad7c246a15896",  # car
    "69e5de3f68804fbc900557b7740b680c",  # convertible
    "1f614e803fc24946b83bdc2dd9926d62",  # coupe
    "c640e4e7c68545e09a9348494c2c13a1",  # pickup
    "09ebc68dcb634807a9dace7404ce5e66",  # SUV
    "f56e1c4d6ace45bf8945136ded809095",  # compact car
    "37d8a4c7d5ec4d1e858dbd912a46eb89",  # convertible
    "26542d66454b4d6d99dd3aa7ae27bf20",  # coupe
    "8a1c5378fe0b4fbaa89052b049ed8929",  # compact electric car
    "63515c5e20914d6496d1470ddc366a78",  # compact pickup
    "31a894c9f1f544fd931b6bb5a0762c7b",  # SUV
    "a7ff90a4675542579170e1efb20e32da",  # compact SUV
    "403d199e0940423188f9425581342a67",  # SUV
    "3be9ac82a47e4639b7d0698e1697feb9",  # coupe
    "9826aff31f7c421b9a7de6bb78d5d4a7",  # electric car
)


class Scene10MultiVehicleManager:
    """Advance five calibrated native vehicles and collect safety metrics."""

    def __init__(
        self,
        stage: Any,
        scene_query: Any,
        registry_path: Path,
        audit_inventory_path: Path | None,
        *,
        dt: float,
        validated_routes_path: Path | None = None,
        validated_static_bodies_path: Path | None = None,
        arrival_radius_m: float = 0.55,
        stall_timeout_s: float = 40.0,
        go2_radius_m: float = 0.45,
        calibrated_flat_road_z_m: float = 0.8949999809265137,
        keep_vehicle_prims_active: bool = False,
        opposing_showcase_speed_mps: float = 3.12,
        opposing_showcase_start_route_index: int = 0,
        visibility_diagnostic_convoy: bool = False,
        fabric_xform_views: list[Any] | None = None,
        use_source_vehicle_prims: bool = False,
        preauthored_vehicle_prim_paths: Sequence[str] | None = None,
        traffic_vehicle_count: int = len(INTEGRATED_MIXED_VEHICLE_IDS),
        traffic_asset_ids: Sequence[str] | None = None,
        scene_config_path: Path | None = None,
    ) -> None:
        self.stage = stage
        self.scene_query = scene_query
        self.dt = float(dt)
        self.arrival_radius_m = float(arrival_radius_m)
        self.stall_timeout_s = float(stall_timeout_s)
        self.go2_radius_m = float(go2_radius_m)
        self.scene_config = (
            TrafficSceneConfig.load(scene_config_path) if scene_config_path is not None else None
        )
        self.scene_id = self.scene_config.scene_id if self.scene_config else "scene_10_cbd_cross_intersection_diverse_obstacles"
        self.calibrated_flat_road_z_m = float(
            self.scene_config.fallback_ground_z_m
            if self.scene_config is not None
            else calibrated_flat_road_z_m
        )
        self.ground_fallback_provenance = (
            self.scene_config.fallback_ground_provenance
            if self.scene_config is not None
            else "prior standalone Scene 10 per-frame PhysX sweep over the same five complete routes"
        )
        self.keep_vehicle_prims_active = bool(keep_vehicle_prims_active)
        self.opposing_showcase_speed_mps = float(opposing_showcase_speed_mps)
        self.opposing_showcase_start_route_index = int(opposing_showcase_start_route_index)
        self.visibility_diagnostic_convoy = bool(visibility_diagnostic_convoy)
        self.fabric_xform_views = list(fabric_xform_views or [])
        self.use_source_vehicle_prims = bool(use_source_vehicle_prims)
        self.preauthored_vehicle_prim_paths = tuple(
            str(path) for path in (preauthored_vehicle_prim_paths or ())
        )
        self.traffic_vehicle_count = int(traffic_vehicle_count)
        self.traffic_asset_ids = tuple(traffic_asset_ids or INTEGRATED_MIXED_VEHICLE_IDS)
        if self.traffic_vehicle_count < 1:
            raise ValueError("traffic vehicle count must be positive")
        if not self.traffic_asset_ids:
            raise ValueError("traffic asset pool must not be empty")
        if self.preauthored_vehicle_prim_paths and len(
            self.preauthored_vehicle_prim_paths
        ) != self.traffic_vehicle_count:
            raise ValueError(
                "preauthored vehicle prim path count must match traffic vehicle count: "
                f"{len(self.preauthored_vehicle_prim_paths)} != {self.traffic_vehicle_count}"
            )
        if self.fabric_xform_views and len(self.fabric_xform_views) != len(INTEGRATED_MIXED_VEHICLE_IDS):
            raise ValueError(
                "fabric xform view count must match integrated traffic vehicles: "
                f"{len(self.fabric_xform_views)} != {len(INTEGRATED_MIXED_VEHICLE_IDS)}"
            )
        self.fabric_pose_sync_count = 0
        self.ground_raycast_hit_count = 0
        self.ground_calibrated_fallback_count = 0
        self.ground_raycast_miss_count = 0
        # Some portable scenes contain overhead architecture or vegetation above
        # the admitted road polygon.  A closest-hit downward ray then reports the
        # occluder instead of the road and can lift a kinematic vehicle into the
        # air.  Those scenes opt into their audited flat-road height; Scene10 and
        # other existing configs keep runtime grounding enabled by default.
        self.ground_raycast_enabled = bool(
            self.scene_config.traffic.get("runtime_ground_raycast", True)
            if self.scene_config is not None
            else True
        )
        self.ground_query_cache: dict[int, dict[str, Any]] = {}
        self.ground_query_interval_s = 0.50
        self.ground_query_distance_m = 1.00
        self.ground_raycast_reuse_count = 0
        self.visibility_enforcement_count = 0
        self.visibility_readbacks: dict[str, str] = {}

        registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
        portable_direct_payload_catalog = (
            registry.get("catalog_kind") == "portable_direct_payload_vehicles"
        )
        portable_source_scene_usd = registry.get("source_scene_usd")
        portable_frame = registry.get("converted_vehicle_frame", {})
        selected_asset_ids = [
            self.traffic_asset_ids[index % len(self.traffic_asset_ids)]
            for index in range(self.traffic_vehicle_count)
        ]
        source_records = [
            next(row for row in registry["records"] if row["asset_id"] == asset_id)
            for asset_id in selected_asset_ids
        ]
        self.records = []
        for record_index, source_record in enumerate(source_records):
            record = copy.deepcopy(source_record)
            source_path = record["scene_prim_path"]
            if portable_direct_payload_catalog:
                # Asset IDs are reused across UrbanVerse scenes, but the target
                # scene's same-named prim can have a missing/broken local
                # payload or a different authored transform.  A portable
                # catalog must therefore always compose its calibrated source
                # GLB, even when a coincident target-scene prim exists.
                portable_path = (
                    self.preauthored_vehicle_prim_paths[record_index]
                    if self.preauthored_vehicle_prim_paths
                    else f"/World/ground/terrain/PortableTrafficVehicle_{record_index:02d}"
                )
                if stage.GetPrimAtPath(portable_path).IsValid():
                    record["source_scene_prim_path"] = source_path
                    record["scene_prim_path"] = portable_path
                    record["front_direction"]["runtime_proxy_heading_deg"] = float(
                        portable_frame.get("front_heading_deg", 0.0)
                    )
                else:
                    if self.preauthored_vehicle_prim_paths:
                        raise RuntimeError(
                            "explicit preauthored portable vehicle prim is absent from the "
                            f"runtime stage: {portable_path}"
                        )
                    record["external_payload_proxy"] = True
                    if portable_source_scene_usd:
                        record["portable_source_scene_usd"] = str(
                            Path(portable_source_scene_usd).resolve()
                        )
            elif not stage.GetPrimAtPath(source_path).IsValid():
                mounted_path = "/World/ground/terrain" + source_path[len("/World") :]
                if stage.GetPrimAtPath(mounted_path).IsValid():
                    record["source_scene_prim_path"] = source_path
                    record["scene_prim_path"] = mounted_path
                elif record["front_direction"].get("runtime_proxy_heading_deg") is not None:
                    # A portable catalog may reference a calibrated GLB that is
                    # not instantiated by the target scene. The direct-payload
                    # proxy branch below composes it without requiring a hidden
                    # copy of Scene 10.
                    record["external_payload_proxy"] = True
                else:
                    raise RuntimeError(
                        "vehicle is absent from the target scene and lacks a portable "
                        f"runtime heading calibration: {record['asset_id']}"
                    )
            self.records.append(record)
        for record in self.records:
            if not record["eligibility"]["traffic_ready"]:
                raise RuntimeError(f"vehicle is not traffic ready: {record['asset_id']}")
            if record["front_direction"]["status"] == "pending_visual_validation":
                raise RuntimeError(f"vehicle front is not validated: {record['asset_id']}")

        self.lane = (
            PolygonRoadFootprint(
                self.scene_config.road_polygons_xy,
                self.scene_config.fallback_ground_z_m,
            )
            if self.scene_config is not None
            else LaneFootprint(stage)
        )
        moving_paths = {
            record.get("source_scene_prim_path", record["scene_prim_path"])
            for record in self.records
        } | {record["scene_prim_path"] for record in self.records}
        if self.scene_config is not None:
            static_rows = list(self.scene_config.static_vehicle_bodies)
            if self.scene_config.static_obstacle_inventory is not None:
                inventory = json.loads(
                    self.scene_config.static_obstacle_inventory.read_text(encoding="utf-8")
                )
                inventory_rows = inventory.get("bodies", inventory)
                if not isinstance(inventory_rows, list):
                    raise ValueError("static obstacle inventory must contain a bodies list")
                static_rows.extend(inventory_rows)
            self.static = [
                (row["path"], np.asarray(row["corners_xy"], dtype=np.float64))
                for row in static_rows
            ]
        elif validated_static_bodies_path is not None:
            self.static = [
                (row["path"], np.asarray(row["corners_xy"], dtype=np.float64))
                for row in json.loads(Path(validated_static_bodies_path).read_text(encoding="utf-8"))
                if row["path"] not in moving_paths
            ]
        else:
            if audit_inventory_path is None:
                raise ValueError("audit_inventory_path is required without a portable scene config")
            self.static = [
                item for item in static_vehicles(stage, Path(audit_inventory_path))
                if item[0] not in moving_paths
            ]
        self.static_bounds = [
            (
                path,
                polygon,
                np.asarray(polygon, dtype=np.float64).min(axis=0),
                np.asarray(polygon, dtype=np.float64).max(axis=0),
            )
            for path, polygon in self.static
        ]
        if self.scene_config is not None and self.scene_config.traffic.get('mesh_navigation'):
            from ..navigation.mesh_runtime import from_scene
            self.lane=from_scene(self.scene_config)
            # Mesh mask is the authoritative static source for this workflow;
            # old rectangular inventory is retained only by legacy configs.
            self.static=[]
            self.static_bounds=[]
        self.specs = []
        for record in self.records:
            length, width = map(float, record["footprint"]["length_width_m"])
            self.specs.append(
                {
                    "asset_id": record["asset_id"],
                    "category": record["category"],
                    "length": length,
                    "width": width,
                    "radius": 0.5 * math.hypot(length, width) + 0.18,
                }
            )
        if self.scene_config is not None:
            route_payload = json.loads(self.scene_config.automotive_routes.read_text(encoding="utf-8"))
            # The continuous manager replaces these compatibility routes with
            # fully resampled AutomotiveRoute objects immediately after base
            # construction.  Accept both the historical dense ``xy`` schema
            # and the portable compact control-point schema here so generic
            # scene configs can pass through the shared base initialization.
            from .routes import compact_route_control_points

            base_routes = [
                compact_route_control_points(row) for row in route_payload["routes"]
            ]
            if any(route.ndim != 2 or route.shape[1] != 2 or len(route) < 2 for route in base_routes):
                raise ValueError(
                    "portable automotive routes require xy, control_points_xy, or control_primitives"
                )
            self.routes = [
                base_routes[index % len(base_routes)].copy()
                for index in range(self.traffic_vehicle_count)
            ]
            self.grids = []
        elif validated_routes_path is not None:
            route_payload = json.loads(Path(validated_routes_path).read_text(encoding="utf-8"))
            base_routes = [
                np.asarray(route, dtype=np.float64) for route in route_payload["routes"]
            ]
            self.routes = [
                base_routes[index % len(base_routes)].copy()
                for index in range(self.traffic_vehicle_count)
            ]
            self.grids = []
        else:
            xs, ys, road_clearance, obstacle_clearance = build_clearance_fields(self.lane, self.static)
            self.grids = [
                build_grid(road_clearance, obstacle_clearance, spec["width"] / 2.0)
                for spec in self.specs
            ]
            self.routes = plan_routes(self.grids, xs, ys, self.specs)
        if self.scene_config is not None:
            route_payload = json.loads(self.scene_config.automotive_routes.read_text(encoding="utf-8"))
            base_route_names = tuple(str(row["route_name"]) for row in route_payload["routes"])
        else:
            base_route_names = (
                "west_to_east",
                "east_to_west",
                "south_to_north",
                "north_to_south",
                "west_to_north_turn",
            )
        self.route_names = tuple(
            f"{base_route_names[index % len(base_route_names)]}_flow_{index // len(base_route_names):02d}"
            for index in range(self.traffic_vehicle_count)
        )
        junction_center = np.asarray(
            self.scene_config.traffic.get("junction_xy", [-625.0, 490.0])
            if self.scene_config is not None
            else [-625.0, 490.0],
            dtype=np.float64,
        )
        # The standalone traffic showcase used 95 s for the final west-to-north
        # turn.  In the Go2 integration that timing brings the turning SUV close
        # to the non-avoiding roadside robot.  Advancing it to 70 s retains a
        # ten-second headway behind the west-to-east car while clearing the
        # robot's downstream corridor before the robot reaches it.
        desired_merge_times = tuple(
            self.scene_config.traffic.get("desired_merge_times_s", [60.0, 14.0, 22.0, 32.0, 70.0])
            if self.scene_config is not None
            else (60.0, 14.0, 22.0, 32.0, 70.0)
        )
        self.agents: list[dict[str, Any]] = []
        for index, (route, spec, route_name) in enumerate(zip(self.routes, self.specs, self.route_names)):
            # Vehicle 1 is the visually distinctive opposing convertible. Its
            # speed remains configurable so a capture can tune the encounter
            # while preserving the previously validated 3.12 m/s default.
            route_slot = index % len(base_route_names)
            cohort = index // len(base_route_names)
            speed = (
                self.opposing_showcase_speed_mps
                if route_slot == 1
                else 3.2 - 0.08 * route_slot
            )
            merge_index = int(np.argmin(np.linalg.norm(route - junction_center, axis=1)))
            distance_to_merge = float(np.linalg.norm(np.diff(route[: merge_index + 1], axis=0), axis=1).sum())
            start_time = max(
                0.0,
                desired_merge_times[route_slot % len(desired_merge_times)] - distance_to_merge / speed,
            ) + 4.0 * cohort
            start_route_index = self.opposing_showcase_start_route_index if route_slot == 1 else 0
            if not 0 <= start_route_index < len(route) - 1:
                raise ValueError(
                    "opposing showcase start route index must leave at least one waypoint: "
                    f"index={start_route_index}, route_points={len(route)}"
                )
            start_position = route[start_route_index].copy()
            direction = route[start_route_index + 1] - start_position
            self.agents.append(
                {
                    "id": index,
                    "kind": "car",
                    "route_name": route_name,
                    "asset_id": spec["asset_id"],
                    "category": spec["category"],
                    "length": spec["length"],
                    "width": spec["width"],
                    "radius": spec["radius"],
                    "preferred_speed": speed,
                    "position": start_position,
                    "velocity": np.zeros(2),
                    "heading": math.atan2(direction[1], direction[0]),
                    "route": route,
                    "route_locked_velocity": True,
                    "waypoint": start_route_index + 1,
                    "direction": 1,
                    "goal_reversals": 0,
                    "trajectory": [start_position.copy()],
                    "status": "scheduled",
                    "start_time_s": start_time,
                    "last_progress_time_s": start_time,
                    "best_goal_distance_m": float(np.linalg.norm(route[-1] - start_position)),
                    "terminal_time_s": None,
                }
            )
        if self.visibility_diagnostic_convoy:
            # Put every selected model on the already validated east-to-west
            # route with safe longitudinal spacing.  This is a short rendering
            # regression only: it makes all five payloads cross the same camera
            # frustum without changing the normal traffic schedule.
            diagnostic_route = self.routes[1]
            start_indices = (100, 108, 116, 124, 132)
            for agent, start_route_index in zip(self.agents, start_indices):
                start_position = diagnostic_route[start_route_index].copy()
                direction = diagnostic_route[start_route_index + 1] - start_position
                agent.update(
                    {
                        "route_name": "visibility_diagnostic_east_to_west_convoy",
                        "route": diagnostic_route,
                        "position": start_position,
                        "velocity": np.zeros(2),
                        "heading": math.atan2(direction[1], direction[0]),
                        "waypoint": start_route_index + 1,
                        "trajectory": [start_position.copy()],
                        "status": "scheduled",
                        "start_time_s": 0.0,
                        "last_progress_time_s": 0.0,
                        "best_goal_distance_m": float(
                            np.linalg.norm(diagnostic_route[-1] - start_position)
                        ),
                        "terminal_time_s": None,
                    }
                )
        self.runtime_visual_clones: list[dict[str, str]] = []
        if (
            not self.use_source_vehicle_prims
            and (
                self.keep_vehicle_prims_active
                or any(record.get("external_payload_proxy") for record in self.records)
            )
        ):
            self.records = [
                self._create_payload_visual_proxy(record, index)
                for index, record in enumerate(self.records)
            ]
        self.vehicles = [NativeVehicle(stage, record) for record in self.records]
        self.vehicle_active = [self.keep_vehicle_prims_active for _ in self.vehicles]
        self.last_visual_status = ["scheduled" for _ in self.vehicles]
        self.parking_positions = [
            np.asarray([-900.0 - 14.0 * index, 400.0], dtype=np.float64)
            for index in range(len(self.vehicles))
        ]
        for index, vehicle in enumerate(self.vehicles):
            if self.keep_vehicle_prims_active:
                self._enforce_vehicle_visible(vehicle)
                # Hydra can lose a custom xform op when a referenced GLB prim is
                # deactivated and later reactivated.  Formal rendering keeps the
                # prim alive and parks non-road agents well outside the scene.
                vehicle.set_pose(
                    self.parking_positions[index],
                    float(self.agents[index]["heading"]),
                    self.calibrated_flat_road_z_m,
                )
                self._sync_vehicle_fabric_pose(index, vehicle)
            else:
                vehicle.prim.SetActive(False)

        self.minimum_dynamic_clearance_m = math.inf
        self.minimum_static_clearance_m = math.inf
        self.minimum_go2_clearance_m = math.inf
        self.dynamic_overlap_events = 0
        self.static_overlap_events = 0
        self.go2_vehicle_overlap_events = 0
        self.boundary_or_static_rejections = 0
        self.orca_constraint_count = 0
        self.orca_predicted_conflict_count = 0
        self.third_person_pass_current_s = 0.0
        self.third_person_pass_max_s = 0.0
        self.third_person_pass_current_ids: list[int] = []
        self.third_person_pass_vehicle_ids: set[int] = set()
        self.third_person_longitudinal_limits_m = (-4.0, 18.0)
        self.third_person_lateral_limit_m = 7.0
        self.third_person_range_limit_m = 18.0
        self.states: list[dict[str, Any]] = []

    @property
    def all_terminal(self) -> bool:
        return all(agent["status"] in ("arrived", "failed") for agent in self.agents)

    @property
    def all_arrived(self) -> bool:
        return all(agent["status"] == "arrived" for agent in self.agents)

    def _is_agent_active_on_road(self, agent: dict[str, Any]) -> bool:
        """Return whether an agent should participate in rendering/safety checks."""
        return agent["status"] in ("moving", "arrived", "failed")

    def update(self, timestamp_s: float, go2_xy: np.ndarray, go2_yaw_rad: float = 0.0) -> dict[str, Any]:
        for agent in self.agents:
            if agent["status"] == "scheduled" and timestamp_s + 1.0e-9 >= agent["start_time_s"]:
                agent["status"] = "moving"
                agent["last_progress_time_s"] = timestamp_s

        active = [agent for agent in self.agents if agent["status"] == "moving"]
        terminal_agents = [
            agent for agent in self.agents if agent["status"] in ("arrived", "failed")
        ]
        terminal_polys = [
            footprint(
                agent["position"],
                agent["heading"],
                agent["length"],
                agent["width"],
            )
            for agent in terminal_agents
        ]
        before = [
            {
                "position": agent["position"].copy(),
                "velocity": agent["velocity"].copy(),
                "heading": float(agent["heading"]),
                "waypoint": int(agent["waypoint"]),
                "direction": int(agent["direction"]),
                "goal_reversals": int(agent["goal_reversals"]),
                "trajectory_length": len(agent["trajectory"]),
            }
            for agent in active
        ]
        if active:
            diagnostics = step_orca(
                active,
                self.dt,
                external_agents=[
                    {
                        "position": agent["position"],
                        "velocity": np.zeros(2, dtype=np.float64),
                        "radius": agent["radius"],
                        "nonreciprocal": True,
                    }
                    for agent in terminal_agents
                ],
                time_horizon=4.0,
                neighbor_distance=14.0,
                safety_margin=0.3,
            )
            self.orca_constraint_count += int(diagnostics["orca_constraint_count"])
            self.orca_predicted_conflict_count += int(diagnostics["predicted_conflicts"])
            for agent, old in zip(active, before):
                poly = footprint(agent["position"], agent["heading"], agent["length"], agent["width"])
                invalid = not all(self.lane.contains(corner) for corner in poly) or any(
                    sat_intersects(poly, obstacle, margin=STATIC_SAFETY_MARGIN_M)
                    for _, obstacle in self.static
                ) or any(
                    sat_intersects(poly, obstacle, margin=STATIC_SAFETY_MARGIN_M)
                    for obstacle in terminal_polys
                )
                if invalid:
                    agent["position"] = old["position"]
                    agent["velocity"] = np.zeros(2)
                    agent["heading"] = old["heading"]
                    agent["waypoint"] = old["waypoint"]
                    agent["direction"] = old["direction"]
                    agent["goal_reversals"] = old["goal_reversals"]
                    del agent["trajectory"][old["trajectory_length"] :]
                    agent["trajectory"].append(agent["position"].copy())
                    self.boundary_or_static_rejections += 1

        for agent in self.agents:
            if agent["status"] != "moving":
                agent["velocity"] = np.zeros(2)
                continue
            goal_distance = float(np.linalg.norm(agent["route"][-1] - agent["position"]))
            if goal_distance <= self.arrival_radius_m or agent["goal_reversals"] > 0:
                agent["position"] = agent["route"][-1].copy()
                agent["velocity"] = np.zeros(2)
                agent["status"] = "arrived"
                agent["terminal_time_s"] = timestamp_s
            elif goal_distance < agent["best_goal_distance_m"] - 0.20:
                agent["best_goal_distance_m"] = goal_distance
                agent["last_progress_time_s"] = timestamp_s
            elif timestamp_s - agent["start_time_s"] >= 120.0 or timestamp_s - agent["last_progress_time_s"] >= self.stall_timeout_s:
                agent["velocity"] = np.zeros(2)
                agent["status"] = "failed"
                agent["terminal_time_s"] = timestamp_s

        return self._publish_frame(timestamp_s, go2_xy, go2_yaw_rad)

    def _publish_frame(
        self,
        timestamp_s: float,
        go2_xy: np.ndarray,
        go2_yaw_rad: float,
    ) -> dict[str, Any]:
        """Apply vehicle poses and collect common OBB/Go2/camera-near-field evidence."""
        frame_agents = []
        dynamic_polys = []
        go2_xy = np.asarray(go2_xy, dtype=np.float64)
        go2_forward = np.asarray([math.cos(go2_yaw_rad), math.sin(go2_yaw_rad)], dtype=np.float64)
        go2_left = np.asarray([-go2_forward[1], go2_forward[0]], dtype=np.float64)
        pass_candidates: list[int] = []
        for vehicle_index, (agent, vehicle) in enumerate(zip(self.agents, self.vehicles)):
            # Once released, terminal vehicles remain at their final pose as
            # visible stationary obstacles. Parking or hiding them at ARRIVED
            # made a complete car disappear abruptly while it was still in the
            # Go2 chase camera, which looked like another transparency failure.
            active_on_road = self._is_agent_active_on_road(agent)
            if active_on_road:
                cached_ground = self.ground_query_cache.get(int(agent["id"]))
                query_ground = cached_ground is None or (
                    timestamp_s - float(cached_ground["timestamp_s"])
                    >= self.ground_query_interval_s
                    or float(
                        np.linalg.norm(
                            np.asarray(agent["position"], dtype=np.float64)
                            - np.asarray(cached_ground["position"], dtype=np.float64)
                        )
                    )
                    >= self.ground_query_distance_m
                )
                if query_ground:
                    try:
                        if not self.ground_raycast_enabled:
                            raise RuntimeError("vehicle road raycast capability disabled after repeated misses")
                        ground_z, ground_path = physical_road_height(self.scene_query, agent["position"])
                        ground_source = "cached periodic downward PhysX raycast"
                        self.ground_raycast_hit_count += 1
                    except RuntimeError:
                        # Each scene config owns its audited fallback height;
                        # use it only after the runtime PhysX capability probe.
                        ground_z = self.calibrated_flat_road_z_m
                        ground_path = None
                        ground_source = (
                            f"calibrated {self.scene_id} road fallback after PhysX ray miss"
                        )
                        self.ground_calibrated_fallback_count += 1
                        if self.ground_raycast_enabled:
                            self.ground_raycast_miss_count += 1
                            if self.ground_raycast_miss_count >= 8 and self.ground_raycast_hit_count == 0:
                                self.ground_raycast_enabled = False
                    self.ground_query_cache[int(agent["id"])] = {
                        "timestamp_s": float(timestamp_s),
                        "position": np.asarray(agent["position"], dtype=np.float64).copy(),
                        "ground_z": float(ground_z),
                        "ground_path": ground_path,
                    }
                else:
                    ground_z = float(cached_ground["ground_z"])
                    ground_path = cached_ground["ground_path"]
                    ground_source = "reused periodic road-height sample"
                    self.ground_raycast_reuse_count += 1
                center, poly, _ = vehicle.set_pose(agent["position"], float(agent["heading"]), ground_z)
                self._sync_vehicle_fabric_pose(vehicle_index, vehicle)
                if not self.vehicle_active[vehicle_index]:
                    vehicle.prim.SetActive(True)
                    self._enforce_vehicle_visible(vehicle)
                    self.vehicle_active[vehicle_index] = True
                dynamic_polys.append((agent, poly))
                point_clearance = self._point_polygon_clearance(go2_xy, poly)
                go2_clearance = point_clearance - self.go2_radius_m
                self.minimum_go2_clearance_m = min(self.minimum_go2_clearance_m, go2_clearance)
                self.go2_vehicle_overlap_events += int(go2_clearance < 0.0)
                relative = agent["position"] - go2_xy
                longitudinal = float(np.dot(relative, go2_forward))
                lateral = float(np.dot(relative, go2_left))
                if agent["status"] == "moving" and (
                    self.third_person_longitudinal_limits_m[0]
                    <= longitudinal
                    <= self.third_person_longitudinal_limits_m[1]
                    and abs(lateral) <= self.third_person_lateral_limit_m
                    and float(np.linalg.norm(relative)) <= self.third_person_range_limit_m
                ):
                    pass_candidates.append(int(agent["id"]))
            else:
                if (
                    self.keep_vehicle_prims_active
                    and self.last_visual_status[vehicle_index] != agent["status"]
                ):
                    vehicle.set_pose(
                        self.parking_positions[vehicle_index],
                        float(agent["heading"]),
                        self.calibrated_flat_road_z_m,
                    )
                    self._sync_vehicle_fabric_pose(vehicle_index, vehicle)
                elif self.vehicle_active[vehicle_index]:
                    vehicle.prim.SetActive(False)
                    self.vehicle_active[vehicle_index] = False
                center = np.asarray([agent["position"][0], agent["position"][1], math.nan])
                ground_z, ground_path, ground_source = math.nan, None, None
            self.last_visual_status[vehicle_index] = agent["status"]
            frame_agents.append(
                {
                    "id": agent["id"],
                    "route_name": agent["route_name"],
                    "asset_id": agent["asset_id"],
                    "category": agent["category"],
                    "preferred_speed_mps": agent["preferred_speed"],
                    "start_time_s": agent["start_time_s"],
                    "status": agent["status"],
                    "active_on_road": active_on_road,
                    "center_xyz": center.tolist(),
                    "velocity_xy": agent["velocity"].tolist(),
                    "heading_deg": math.degrees(agent["heading"]),
                    "ground_z": ground_z,
                    "ground_collision": ground_path,
                    "ground_source": ground_source,
                }
            )
        if pass_candidates:
            self.third_person_pass_current_ids = pass_candidates
            self.third_person_pass_current_s += self.dt
            self.third_person_pass_max_s = max(
                self.third_person_pass_max_s, self.third_person_pass_current_s
            )
            self.third_person_pass_vehicle_ids.update(pass_candidates)
        else:
            self.third_person_pass_current_ids = []
            self.third_person_pass_current_s = 0.0
        for index, (_, poly) in enumerate(dynamic_polys):
            for _, other in dynamic_polys[index + 1 :]:
                self.minimum_dynamic_clearance_m = min(
                    self.minimum_dynamic_clearance_m, polygon_clearance(poly, other)
                )
                self.dynamic_overlap_events += int(sat_intersects(poly, other))
            body_min = poly.min(axis=0)
            body_max = poly.max(axis=0)
            for _, obstacle, obstacle_min, obstacle_max in self.static_bounds:
                gap = np.maximum(
                    np.maximum(obstacle_min - body_max, body_min - obstacle_max),
                    0.0,
                )
                aabb_clearance = float(np.linalg.norm(gap))
                if aabb_clearance <= self.minimum_static_clearance_m:
                    self.minimum_static_clearance_m = min(
                        self.minimum_static_clearance_m,
                        polygon_clearance(poly, obstacle),
                    )
                if aabb_clearance <= 0.0:
                    self.static_overlap_events += int(sat_intersects(poly, obstacle))
        state = {"timestamp_s": float(timestamp_s), "agents": frame_agents}
        self.states.append(state)
        return state

    @staticmethod
    def _visibility_token(imageable: Any) -> str:
        return str(imageable.ComputeVisibility())

    @staticmethod
    def _rotation_matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
        """Return a proper scalar-first quaternion from a possibly scaled matrix."""
        rotation_with_scale = np.asarray(matrix[:3, :3], dtype=np.float64)
        left, _, right = np.linalg.svd(rotation_with_scale)
        rotation = left @ right
        if np.linalg.det(rotation) < 0.0:
            left[:, -1] *= -1.0
            rotation = left @ right
        trace = float(np.trace(rotation))
        if trace > 0.0:
            scale = math.sqrt(trace + 1.0) * 2.0
            quaternion = np.asarray(
                [
                    0.25 * scale,
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                ]
            )
        else:
            diagonal = np.diag(rotation)
            axis = int(np.argmax(diagonal))
            if axis == 0:
                scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
                quaternion = np.asarray(
                    [
                        (rotation[2, 1] - rotation[1, 2]) / scale,
                        0.25 * scale,
                        (rotation[0, 1] + rotation[1, 0]) / scale,
                        (rotation[0, 2] + rotation[2, 0]) / scale,
                    ]
                )
            elif axis == 1:
                scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
                quaternion = np.asarray(
                    [
                        (rotation[0, 2] - rotation[2, 0]) / scale,
                        (rotation[0, 1] + rotation[1, 0]) / scale,
                        0.25 * scale,
                        (rotation[1, 2] + rotation[2, 1]) / scale,
                    ]
                )
            else:
                scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
                quaternion = np.asarray(
                    [
                        (rotation[1, 0] - rotation[0, 1]) / scale,
                        (rotation[0, 2] + rotation[2, 0]) / scale,
                        (rotation[1, 2] + rotation[2, 1]) / scale,
                        0.25 * scale,
                    ]
                )
        return quaternion / np.linalg.norm(quaternion)

    def _sync_vehicle_fabric_pose(self, vehicle_index: int, vehicle: Any) -> None:
        """Mirror the authored USD pose into the Fabric transform read by Isaac Lab cameras."""
        if not self.fabric_xform_views:
            return
        import torch

        world_matrix = np.asarray(vehicle.last_world_matrix, dtype=np.float64)
        view = self.fabric_xform_views[vehicle_index]
        position = torch.as_tensor(
            world_matrix[:3, 3][None, :], dtype=torch.float32, device=view._device
        )
        orientation = torch.as_tensor(
            self._rotation_matrix_to_quaternion_wxyz(world_matrix)[None, :],
            dtype=torch.float32,
            device=view._device,
        )
        view.set_world_poses(
            positions=position,
            orientations=orientation,
            usd=False,
        )
        self.fabric_pose_sync_count += 1

    def _create_runtime_visual_clone(self, record: dict[str, Any], index: int) -> dict[str, Any]:
        """Reference a source vehicle outside Isaac Lab's Fabric-managed terrain tree."""
        from pxr import UsdGeom

        source_path = str(record["scene_prim_path"])
        clone_path = f"/World/DynamicTraffic/Vehicle_{index:02d}"
        UsdGeom.Xform.Define(self.stage, "/World/DynamicTraffic")
        clone = self.stage.GetPrimAtPath(clone_path)
        if clone.IsValid():
            creation_phase = "preauthored before simulation start"
        else:
            clone = UsdGeom.Xform.Define(self.stage, clone_path).GetPrim()
            clone.GetReferences().AddInternalReference(source_path)
            creation_phase = "runtime fallback after simulation start"
        if not clone.IsValid():
            raise RuntimeError(f"failed to create dynamic vehicle visual clone: {clone_path}")
        cloned_record = copy.deepcopy(record)
        cloned_record["runtime_visual_source_prim_path"] = source_path
        cloned_record["scene_prim_path"] = clone_path
        self.runtime_visual_clones.append(
            {"source": source_path, "clone": clone_path, "creation_phase": creation_phase}
        )
        return cloned_record

    def _create_payload_visual_proxy(self, record: dict[str, Any], index: int) -> dict[str, Any]:
        """Create the same direct-payload visual proxy proven by the phase-one Go2 run."""
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics

        source_path = str(record["scene_prim_path"])
        source = self.stage.GetPrimAtPath(source_path)
        external_payload = bool(record.get("external_payload_proxy"))
        if external_payload:
            source = None
            resolved_payload_path = str(Path(record["source_asset"]).resolve())
            portable_source_scene_usd = record.get("portable_source_scene_usd")
            source_payload = None
        else:
            source.Load()
        source_payload = None
        source_payload_layer = None
        if not external_payload:
            for prim_spec in source.GetPrimStack():
                items = list(prim_spec.payloadList.GetAddedOrExplicitItems())
                if items:
                    source_payload = items[0]
                    source_payload_layer = prim_spec.layer
                    break
            if source_payload is None or source_payload_layer is None:
                raise RuntimeError(f"dynamic vehicle payload could not be resolved: {source_path}")
            resolved_payload_path = Sdf.ComputeAssetPathRelativeToLayer(
                source_payload_layer, source_payload.assetPath
            )
        proxy_path = f"/UrbanVerseEvaluation/TrafficVehicleVisual_{index:02d}"
        UsdGeom.Xform.Define(self.stage, "/UrbanVerseEvaluation")
        proxy = self.stage.DefinePrim(proxy_path, "Xform")
        if external_payload and portable_source_scene_usd:
            proxy.GetReferences().AddReference(
                str(portable_source_scene_usd), source_path
            )
        elif external_payload:
            proxy.GetPayloads().AddPayload(resolved_payload_path)
        else:
            proxy.GetPayloads().AddPayload(
                resolved_payload_path,
                source_payload.primPath,
                source_payload.layerOffset,
            )
        proxy.Load()
        if source is not None:
            UsdGeom.Imageable(source).MakeInvisible()
        UsdGeom.Imageable(proxy).MakeVisible()
        for child in Usd.PrimRange(proxy):
            if child.HasAPI(UsdPhysics.RigidBodyAPI):
                child.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if child.HasAPI(UsdPhysics.CollisionAPI):
                child.RemoveAPI(UsdPhysics.CollisionAPI)
        proxy_xform = UsdGeom.Xformable(proxy)
        proxy_xform.ClearXformOpOrder()
        if source is not None:
            source_world = UsdGeom.Xformable(source).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            proxy_xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, "payloadInitial").Set(source_world)
        cloned_record = copy.deepcopy(record)
        proxy_front = (
            float(record["front_direction"]["heading_deg"])
            if external_payload and portable_source_scene_usd
            else float(record["front_direction"]["runtime_proxy_heading_deg"])
            if external_payload
            else self._transfer_proxy_front_heading(source, proxy, record)
        )
        cloned_record["front_direction"]["source_registry_heading_deg"] = float(
            record["front_direction"]["heading_deg"]
        )
        cloned_record["front_direction"]["runtime_proxy_heading_deg"] = float(proxy_front)
        cloned_record["front_direction"]["runtime_transfer_method"] = (
            "corresponding source/proxy mesh front-and-rear vertex centroids"
        )
        cloned_record["runtime_visual_source_prim_path"] = source_path
        cloned_record["scene_prim_path"] = proxy_path
        self.runtime_visual_clones.append(
            {
                "source": source_path if source is not None else None,
                "proxy": proxy_path,
                "payload": resolved_payload_path,
                "portable_source_scene_usd": portable_source_scene_usd,
                "creation_phase": (
                    "portable source-prim reference after simulation start"
                    if external_payload and portable_source_scene_usd
                    else "portable direct payload proxy after simulation start"
                    if external_payload
                    else "direct payload visual proxy after simulation start"
                ),
                "source_front_heading_deg": float(record["front_direction"]["heading_deg"]),
                "proxy_front_heading_deg": float(proxy_front),
                "proxy_heading_correction_deg": float(
                    (proxy_front - float(record["front_direction"]["heading_deg"]) + 180.0)
                    % 360.0
                    - 180.0
                ),
            }
        )
        return cloned_record

    @staticmethod
    def _mesh_world_points_by_relative_path(root: Any) -> dict[str, np.ndarray]:
        """Return world-space mesh vertices keyed below a payload root."""
        from pxr import Gf, Usd, UsdGeom

        root_path = str(root.GetPath())
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        result: dict[str, np.ndarray] = {}
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            points = UsdGeom.Mesh(prim).GetPointsAttr().Get(Usd.TimeCode.Default())
            if not points:
                continue
            matrix = cache.GetLocalToWorldTransform(prim)
            result[str(prim.GetPath())[len(root_path) :]] = np.asarray(
                [
                    matrix.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
                    for point in points
                ],
                dtype=np.float64,
            )
        return result

    def _transfer_proxy_front_heading(
        self, source: Any, proxy: Any, record: dict[str, Any]
    ) -> float:
        """Measure the proxy's signed front axis using matching payload vertices."""
        source_meshes = self._mesh_world_points_by_relative_path(source)
        proxy_meshes = self._mesh_world_points_by_relative_path(proxy)
        common = sorted(set(source_meshes) & set(proxy_meshes))
        if not common:
            # Direct GLB payloads normally preserve descendant paths.  Retain a
            # deterministic single-mesh fallback for converters that rename the
            # payload root while preserving mesh topology.
            if len(source_meshes) == len(proxy_meshes) == 1:
                source_rows = list(source_meshes.values())
                proxy_rows = list(proxy_meshes.values())
            else:
                raise RuntimeError(
                    f"no corresponding source/proxy vehicle meshes: {source.GetPath()} -> {proxy.GetPath()}"
                )
        else:
            source_rows = [source_meshes[path] for path in common]
            proxy_rows = [proxy_meshes[path] for path in common]
        matching = [
            (source_points, proxy_points)
            for source_points, proxy_points in zip(source_rows, proxy_rows)
            if source_points.shape == proxy_points.shape
        ]
        if not matching:
            raise RuntimeError(
                f"source/proxy mesh topology mismatch: {source.GetPath()} -> {proxy.GetPath()}"
            )
        source_xy = np.concatenate([row[0][:, :2] for row in matching], axis=0)
        proxy_xy = np.concatenate([row[1][:, :2] for row in matching], axis=0)
        heading = transfer_front_heading_from_corresponding_points(
            source_xy,
            proxy_xy,
            math.radians(float(record["front_direction"]["heading_deg"])),
        )
        return math.degrees(heading)

    def refresh_visual_transforms(self) -> None:
        """Restore payload-proxy poses after PhysX and immediately before rendering."""
        for vehicle_index, vehicle in enumerate(self.vehicles):
            if self.vehicle_active[vehicle_index]:
                vehicle.refresh_visual_transform()

    def measure_visual_geometry_alignment(self, *, detailed_mesh_audit: bool = False) -> list[dict[str, Any]]:
        """Measure proxy bounds without rescanning every high-poly vertex by default.

        The calibrated catalog already stores each asset's OBB and signed front
        direction.  Reconstructing those values from every converted render mesh
        at the end of every run made a two-second portability probe spend minutes
        in CPU geometry traversal.  A shared USD bbox cache is sufficient for the
        routine visibility/placement contract; the full vertex audit remains
        available explicitly for asset qualification work.
        """
        from ..vehicles.asset_audit import minimum_rectangle
        from pxr import Usd, UsdGeom

        measurements = []
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
            useExtentsHint=True,
        )
        for vehicle_index, (agent, vehicle) in enumerate(zip(self.agents, self.vehicles)):
            descendants = list(Usd.PrimRange(vehicle.prim))
            bounds = bbox_cache.ComputeWorldBound(vehicle.prim).ComputeAlignedRange()
            center = np.asarray(bounds.GetMidpoint(), dtype=np.float64)
            size = np.asarray(bounds.GetSize(), dtype=np.float64)
            on_road = self._is_agent_active_on_road(agent)
            expected_xy = np.asarray(
                agent["position"] if on_road else self.parking_positions[vehicle_index],
                dtype=np.float64,
            )
            mesh_count = sum(item.IsA(UsdGeom.Mesh) for item in descendants)
            mesh_points = (
                self._mesh_world_points_by_relative_path(vehicle.prim)
                if detailed_mesh_audit
                else {}
            )
            rendered_obb = None
            rendered_axis_error_deg = None
            if mesh_points:
                projected = np.concatenate([points[:, :2] for points in mesh_points.values()], axis=0)
                rendered_obb = minimum_rectangle(projected)
                rendered_axis_heading = math.radians(float(rendered_obb["heading_deg"]))
                planner_heading = float(agent["heading"])
                rendered_axis_error_deg = math.degrees(
                    abs(
                        (rendered_axis_heading - planner_heading + math.pi / 2.0)
                        % math.pi
                        - math.pi / 2.0
                    )
                )
            bbox_valid = bool(
                mesh_count > 0
                and np.all(np.isfinite(center))
                and np.all(np.isfinite(size))
                and np.all(size > 0.0)
            )
            measurements.append(
                {
                    "id": int(agent["id"]),
                    "asset_id": str(agent["asset_id"]),
                    "category": str(agent["category"]),
                    "status": str(agent["status"]),
                    "prim_path": str(vehicle.prim.GetPath()),
                    "descendant_count": len(descendants),
                    "mesh_count": mesh_count,
                    "bbox_valid": bbox_valid,
                    "expected_location": "route" if on_road else "off-map parking",
                    "bbox_center_xyz": center.tolist() if bbox_valid else None,
                    "bbox_size_xyz": size.tolist() if bbox_valid else None,
                    "planner_center_xy": np.asarray(agent["position"], dtype=np.float64).tolist(),
                    "expected_visual_center_xy": expected_xy.tolist(),
                    "bbox_to_expected_visual_center_xy_error_m": (
                        float(np.linalg.norm(center[:2] - expected_xy)) if bbox_valid else None
                    ),
                    "rendered_obb_length_width_m": (
                        rendered_obb["length_width"] if rendered_obb is not None else None
                    ),
                    "rendered_unsigned_axis_heading_deg": (
                        rendered_obb["heading_deg"] if rendered_obb is not None else None
                    ),
                    "planner_heading_deg": math.degrees(float(agent["heading"])),
                    "rendered_axis_to_planner_error_deg": rendered_axis_error_deg,
                    "runtime_proxy_front_heading_deg": float(vehicle.initial_heading),
                    "detailed_mesh_audit": bool(detailed_mesh_audit),
                }
            )
        return measurements

    def _enforce_vehicle_visible(self, vehicle: Any) -> None:
        """Clear inherited/source visibility overrides before traffic use."""
        from pxr import UsdGeom

        imageable = UsdGeom.Imageable(vehicle.prim)
        imageable.MakeVisible()
        readback = self._visibility_token(imageable)
        if readback != str(UsdGeom.Tokens.inherited):
            raise RuntimeError(
                f"dynamic vehicle visibility readback failed at {vehicle.prim.GetPath()}: {readback}"
            )
        path = str(vehicle.prim.GetPath())
        self.visibility_enforcement_count += 1
        self.visibility_readbacks[path] = readback

    @staticmethod
    def _point_polygon_clearance(point: np.ndarray, polygon: np.ndarray) -> float:
        edge_cross = []
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
            edge = b - a
            relative = point - a
            edge_cross.append(float(edge[0] * relative[1] - edge[1] * relative[0]))
        if all(value >= -1.0e-9 for value in edge_cross) or all(
            value <= 1.0e-9 for value in edge_cross
        ):
            return 0.0
        distances = []
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
            ab = b - a
            t = float(np.clip(np.dot(point - a, ab) / max(float(np.dot(ab, ab)), 1.0e-12), 0.0, 1.0))
            distances.append(float(np.linalg.norm(point - (a + t * ab))))
        return min(distances)

    @staticmethod
    def _finite_or_none(value: float) -> float | None:
        return float(value) if math.isfinite(value) else None

    def summary(self) -> dict[str, Any]:
        return {
            "all_vehicles_terminal": self.all_terminal,
            "all_vehicles_arrived": self.all_arrived,
            "vehicle_outcomes": [
                {
                    "id": agent["id"],
                    "route_name": agent["route_name"],
                    "asset_id": agent["asset_id"],
                    "category": agent["category"],
                    "status": agent["status"],
                    "start_xy": agent["route"][0].tolist(),
                    "goal_xy": agent["route"][-1].tolist(),
                    "terminal_time_s": agent["terminal_time_s"],
                }
                for agent in self.agents
            ],
            "integrated_vehicle_asset_ids": list(self.traffic_asset_ids),
            "unique_integrated_vehicle_asset_ids": sorted(
                {agent["asset_id"] for agent in self.agents}
            ),
            "excluded_camera_visibility_asset_ids": [MIXED_VEHICLE_IDS[1]],
            "static_vehicle_count": len(self.static),
            "minimum_dynamic_obb_clearance_m": self._finite_or_none(self.minimum_dynamic_clearance_m),
            "minimum_static_obb_clearance_m": self._finite_or_none(self.minimum_static_clearance_m),
            "minimum_go2_vehicle_clearance_m": self._finite_or_none(self.minimum_go2_clearance_m),
            "go2_radius_m": self.go2_radius_m,
            "dynamic_obb_overlap_events": self.dynamic_overlap_events,
            "static_obb_overlap_events": self.static_overlap_events,
            "go2_vehicle_overlap_events": self.go2_vehicle_overlap_events,
            "third_person_dynamic_pass_max_duration_s": self.third_person_pass_max_s,
            "third_person_dynamic_pass_vehicle_ids": sorted(self.third_person_pass_vehicle_ids),
            "third_person_dynamic_pass_contract": (
                "moving vehicle centre within chase-camera near field: "
                f"longitudinal {list(self.third_person_longitudinal_limits_m)} m, "
                f"|lateral| <= {self.third_person_lateral_limit_m} m, "
                f"range <= {self.third_person_range_limit_m} m"
            ),
            "boundary_or_static_step_rejections": self.boundary_or_static_rejections,
            "orca_constraint_count": self.orca_constraint_count,
            "orca_predicted_conflict_count": self.orca_predicted_conflict_count,
            "route_lengths_m": [
                float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum()) for route in self.routes
            ],
            "opposing_showcase_start_route_index": self.opposing_showcase_start_route_index,
            "visibility_diagnostic_convoy": self.visibility_diagnostic_convoy,
            "fabric_pose_sync": {
                "enabled": bool(self.fabric_xform_views),
                "write_count": self.fabric_pose_sync_count,
                "method": "isaacsim.core.prims.XFormPrim.set_world_poses(usd=False)",
            },
            "ground_height": {
                "raycast_hit_count": self.ground_raycast_hit_count,
                "raycast_reuse_count": self.ground_raycast_reuse_count,
                "query_interval_s": self.ground_query_interval_s,
                "query_distance_m": self.ground_query_distance_m,
                "calibrated_fallback_count": self.ground_calibrated_fallback_count,
                "raycast_miss_count_before_disable": self.ground_raycast_miss_count,
                "raycast_disabled_after_capability_probe": not self.ground_raycast_enabled,
                "fallback_z_m": self.calibrated_flat_road_z_m,
                "fallback_provenance": self.ground_fallback_provenance,
            },
            "visibility": {
                "method": "UsdGeom.Imageable.MakeVisible with ComputeVisibility readback",
                "enforcement_count": self.visibility_enforcement_count,
                "readbacks": self.visibility_readbacks,
                "runtime_visual_clones": self.runtime_visual_clones,
                "visual_prim_mode": (
                    "existing source scene vehicles registered with Isaac Lab XFormPrim"
                    if self.use_source_vehicle_prims
                    else "direct source-payload proxies refreshed immediately before camera capture"
                ),
            },
            "visual_geometry_alignment": self.measure_visual_geometry_alignment(),
            "method": "calibrated whole-body kinematics + per-vehicle occupancy/A* + CPU disc ORCA + scene-configured road grounding",
            "collision_evidence": "2D oriented-body polygons against dynamic and static vehicle polygons; authored vehicle shells have no wheel dynamics",
        }
